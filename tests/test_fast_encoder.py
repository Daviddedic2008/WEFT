"""
CPU test: fast_encoder.flat_forward == HuggingFace BertModel + ColBERT's linear + L2 normalize,
on a small random BERT, with padding positions masked out of attention.
Run: python3 tests/test_fast_encoder.py   (needs transformers)
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fast_encoder import flat_weights, flat_forward, bert_and_linear


def main():
    from transformers import BertConfig, BertModel
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=200, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                     intermediate_size=128, max_position_embeddings=64)
    bert = BertModel(cfg).eval()
    linear = torch.nn.Linear(64, 16, bias=False)

    class FakeCheckpoint:            # ColBERT exposes .bert and .linear
        pass
    cp = FakeCheckpoint(); cp.bert, cp.linear = bert, linear
    b, l = bert_and_linear(cp)
    assert b is bert and l is linear

    B, Lq = 3, 32
    ids = torch.randint(5, 200, (B, Lq))
    mask = torch.ones(B, Lq, dtype=torch.int64)
    mask[0, 10:] = 0; mask[1, 4:] = 0                       # [MASK] padding not attended to
    with torch.no_grad():
        ref = bert(ids, attention_mask=mask)[0]
        ref = torch.nn.functional.normalize(linear(ref), dim=-1)
        W = flat_weights(bert, linear, dtype=torch.float32, device="cpu")
        got = flat_forward(W, ids, mask)
    err = (got - ref).abs().max().item()
    print(f"  flat forward vs HuggingFace BertModel + linear: max |err| {err:.2e} over {B}x{Lq} tokens")
    assert err < 1e-5, err
    print("FAST ENCODER TEST PASSED")


if __name__ == "__main__":
    main()
