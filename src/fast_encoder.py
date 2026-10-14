"""
Fast ColBERT query encoder (stage 0a), a drop-in for Searcher.encode on one query or a batch.

Stock encoding takes ~5 ms per query on an RTX 6000 Ada, about 8x RAGRT's retrieval. Its GPU work
is tiny (BERT-base on 32 tokens); the time is Python and launch overhead: HuggingFace's module
tree, autocast, ColBERT's batching helpers, ~150 small kernels launched one by one.

This module runs the same network as a flat function over fp16 weights (embeddings, 12 x
[self-attention with SDPA, LayerNorm, GELU MLP, LayerNorm], ColBERT's linear 768 -> 128, L2
normalize) with fixed shapes [B, 32], and captures it in a CUDA graph per batch size. Tokenization
is ColBERT's own QueryTokenizer, so the token ids, the [Q] marker and the [MASK] padding are
identical; the attention mask is ColBERT's (padding [MASK] tokens are not attended to unless the
checkpoint's config says so). Outputs match the stock encoder to fp16 precision (tests/
test_fast_encoder.py checks the flat forward against HuggingFace's BertModel; bench_encode.py
checks the real checkpoint and the effect on retrieval).
"""
import math
import torch
import torch.nn.functional as F


def bert_and_linear(checkpoint):
    """(HF BertModel, ColBERT linear) of a ColBERT Checkpoint, across ColBERT versions."""
    bert = getattr(checkpoint, "bert", None)
    if bert is None:
        m = checkpoint.model
        bert = getattr(m, "bert", None) or getattr(m, "LM")
    return bert, checkpoint.linear


def flat_weights(bert, linear, dtype=torch.float16, device="cuda"):
    sd = {k: v.detach().to(device=device, dtype=dtype) for k, v in bert.state_dict().items()}
    cfg = bert.config
    W = {"emb_word": sd["embeddings.word_embeddings.weight"],
         "emb_pos": sd["embeddings.position_embeddings.weight"],
         "emb_type": sd["embeddings.token_type_embeddings.weight"][0],
         "emb_ln": (sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"]),
         "heads": cfg.num_attention_heads, "eps": cfg.layer_norm_eps,
         "act": getattr(cfg, "hidden_act", "gelu"),
         "linear": linear.weight.detach().to(device=device, dtype=dtype), "layers": []}
    for i in range(cfg.num_hidden_layers):
        p = f"encoder.layer.{i}."
        g = lambda n: sd[p + n]
        W["layers"].append({
            "qkv_w": torch.cat([g("attention.self.query.weight"), g("attention.self.key.weight"),
                                g("attention.self.value.weight")], 0).contiguous(),
            "qkv_b": torch.cat([g("attention.self.query.bias"), g("attention.self.key.bias"),
                                g("attention.self.value.bias")], 0).contiguous(),
            "o_w": g("attention.output.dense.weight"), "o_b": g("attention.output.dense.bias"),
            "ln1": (g("attention.output.LayerNorm.weight"), g("attention.output.LayerNorm.bias")),
            "i_w": g("intermediate.dense.weight"), "i_b": g("intermediate.dense.bias"),
            "f_w": g("output.dense.weight"), "f_b": g("output.dense.bias"),
            "ln2": (g("output.LayerNorm.weight"), g("output.LayerNorm.bias")),
        })
    assert W["act"] in ("gelu", "gelu_new", "gelu_python"), f"unsupported activation {W['act']}"
    return W


def flat_forward(W, ids, mask):
    """ids, mask [B, L] (int64, on the weights' device) -> unit ColBERT query vectors [B, L, dim]."""
    B, Lq = ids.shape
    H = W["heads"]
    x = W["emb_word"][ids] + W["emb_pos"][:Lq][None] + W["emb_type"]
    D = x.shape[-1]
    x = F.layer_norm(x, (D,), *W["emb_ln"], eps=W["eps"])
    # additive key mask, as HuggingFace builds it: 0 for attended keys, the dtype's minimum otherwise
    bias = (1.0 - mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
    act = (lambda t: F.gelu(t, approximate="tanh")) if W["act"] == "gelu_new" else F.gelu
    for lw in W["layers"]:
        qkv = F.linear(x, lw["qkv_w"], lw["qkv_b"]).view(B, Lq, 3, H, D // H).permute(2, 0, 3, 1, 4)
        ctx = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=bias)
        ctx = ctx.transpose(1, 2).reshape(B, Lq, D)
        x = F.layer_norm(x + F.linear(ctx, lw["o_w"], lw["o_b"]), (D,), *lw["ln1"], eps=W["eps"])
        h = act(F.linear(x, lw["i_w"], lw["i_b"]))
        x = F.layer_norm(x + F.linear(h, lw["f_w"], lw["f_b"]), (D,), *lw["ln2"], eps=W["eps"])
    Q = F.linear(x, W["linear"])
    return F.normalize(Q.float(), p=2, dim=-1).to(x.dtype)


class FastQueryEncoder:
    """encode(text) -> ([32, 128] fp16, ntok); encode_batch(texts) -> ([B, 32, 128] fp16, ntoks)."""

    def __init__(self, checkpoint, device="cuda", use_graphs=True):
        self.tok = checkpoint.query_tokenizer
        bert, linear = bert_and_linear(checkpoint)
        self.W = flat_weights(bert, linear, device=device)
        self.device, self.use_graphs = device, use_graphs
        self.graphs = {}

    def tokenize(self, texts):
        ids, mask = self.tok.tensorize(list(texts))
        return ids, mask

    def _graph(self, B, Lq):
        key = (B, Lq)
        if key not in self.graphs:
            ids = torch.zeros(B, Lq, dtype=torch.int64, device=self.device)
            mask = torch.ones(B, Lq, dtype=torch.int64, device=self.device)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):
                    flat_forward(self.W, ids, mask)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = flat_forward(self.W, ids, mask)
            self.graphs[key] = (g, ids, mask, out)
        return self.graphs[key]

    @torch.no_grad()
    def encode_batch(self, texts):
        ids, mask = self.tokenize(texts)
        ntoks = mask.sum(1).tolist()
        if not self.use_graphs:
            return flat_forward(self.W, ids.to(self.device), mask.to(self.device)), ntoks
        g, gi, gm, out = self._graph(*ids.shape)
        gi.copy_(ids, non_blocking=True)
        gm.copy_(mask, non_blocking=True)
        g.replay()
        return out.clone(), ntoks

    def encode(self, text):
        Q, ntoks = self.encode_batch([text])
        return Q[0].contiguous(), int(ntoks[0])
