"""
MUVERA fixed-dimensional encodings (Dhulipala et al., "MUVERA: Multi-Vector Retrieval via
Fixed Dimensional Encodings", NeurIPS 2024), as a baseline for RAGRT.

For each of `reps` repetitions:
  * SimHash partition: k_sim Gaussian directions; a token's bucket is its sign pattern
    (2^k_sim buckets).
  * Inner projection: a random +-1 / sqrt(d_proj) matrix [d_proj, 128] applied to every token
    (d_proj = 128 and identity=True skips it).
  * Document block: per bucket, the MEAN of its projected tokens. Empty buckets are filled
    with the projected token whose sign pattern is closest in Hamming distance
    (the paper's fill_empty_partitions).
  * Query block: per bucket, the SUM of its projected tokens (empty buckets stay 0).
FDE = concatenation of the reps blocks: reps * 2^k_sim * d_proj dims. <query FDE, doc FDE>
approximates Chamfer (MaxSim) similarity. No final projection.

doc_fdes() is vectorized over a chunk of passages (scatter on GPU); reference_doc_fde() /
reference_query_fde() are slow per-passage versions used by the tests.
"""
import json, os
import numpy as np
import torch


def make_params(dim=128, reps=20, k_sim=4, d_proj=8, seed=0, identity=False):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(reps, k_sim, dim, generator=g)
    if identity:
        assert d_proj == dim, "identity projection needs d_proj == dim"
        P = torch.eye(dim).expand(reps, dim, dim).clone()
    else:
        P = (torch.randint(0, 2, (reps, d_proj, dim), generator=g).float() * 2 - 1) / d_proj ** 0.5
    cfg = {"dim": dim, "reps": reps, "k_sim": k_sim, "d_proj": d_proj, "seed": seed, "identity": identity,
           "fde_dim": reps * (2 ** k_sim) * d_proj}
    return cfg, {"G": G, "P": P}


def save_params(d, cfg, params):
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(os.path.join(d, "fde_config.json"), "w"), indent=1)
    torch.save({k: v.cpu() for k, v in params.items()}, os.path.join(d, "fde_params.pt"))


def load_params(d):
    cfg = json.load(open(os.path.join(d, "fde_config.json")))
    params = torch.load(os.path.join(d, "fde_params.pt"))
    return cfg, params


def _buckets(X, G):
    """X [T, dim], G [k_sim, dim] -> int64 bucket ids [T] (bit i = sign of <g_i, x>)."""
    bits = (X @ G.T > 0).to(torch.int64)
    w = (2 ** torch.arange(G.shape[0], device=X.device)).to(torch.int64)
    return (bits * w).sum(1)


def _popcount(x):
    c = torch.zeros_like(x)
    while bool((x > 0).any()):
        c += x & 1
        x = x >> 1
    return c


@torch.no_grad()
def query_fde(Q, params, cfg):
    """Q [n, dim] float -> [fde_dim] float (sum per bucket)."""
    G, P = params["G"].to(Q.device, Q.dtype), params["P"].to(Q.device, Q.dtype)
    B, dp = 2 ** cfg["k_sim"], cfg["d_proj"]
    out = []
    for r in range(cfg["reps"]):
        b = _buckets(Q, G[r])
        Y = Q @ P[r].T
        blk = torch.zeros(B, dp, device=Q.device, dtype=Q.dtype).index_add_(0, b, Y)
        out.append(blk.reshape(-1))
    return torch.cat(out)


@torch.no_grad()
def doc_fdes(X, doc_of_token, n_docs, params, cfg):
    """
    X [T, dim] float tokens of n_docs consecutive passages, doc_of_token [T] int64 in
    [0, n_docs) (non-decreasing). Returns [n_docs, fde_dim] float.
    Every passage must have at least one token.
    """
    dev = X.device
    G, P = params["G"].to(dev, X.dtype), params["P"].to(dev, X.dtype)
    B, dp = 2 ** cfg["k_sim"], cfg["d_proj"]
    T = X.shape[0]
    tok = torch.arange(T, device=dev, dtype=torch.int64)
    allb = torch.arange(B, device=dev, dtype=torch.int64)
    ham_tab = _popcount(allb[:, None] ^ allb[None, :])            # [B, B]
    out = torch.empty(n_docs, cfg["reps"], B, dp, device=dev, dtype=X.dtype)
    for r in range(cfg["reps"]):
        b = _buckets(X, G[r])
        Y = X @ P[r].T                                              # [T, dp]
        slot = doc_of_token * B + b
        s = torch.zeros(n_docs * B, dp, device=dev, dtype=X.dtype).index_add_(0, slot, Y)
        cnt = torch.zeros(n_docs * B, device=dev, dtype=X.dtype).index_add_(0, slot, torch.ones_like(slot, dtype=X.dtype))
        blk = s / cnt.clamp(min=1)[:, None]
        empty = cnt == 0
        if bool(empty.any()):
            # For every (doc, bucket): the token of that doc with the smallest Hamming distance
            # between its bucket and this bucket (ties -> lowest token index).
            key = ham_tab[b] * T + tok[:, None]                     # [T, B]; key // T = distance
            best = torch.full((n_docs * B,), torch.iinfo(torch.int64).max, device=dev, dtype=torch.int64)
            best.scatter_reduce_(0, (doc_of_token[:, None] * B + allb[None, :]).reshape(-1),
                                 key.reshape(-1), reduce="amin")
            fill_tok = best % T
            blk[empty] = Y[fill_tok[empty]]
        out[:, r] = blk.view(n_docs, B, dp)
    return out.reshape(n_docs, -1)


# --------------------------------------------------------------------- slow references (tests)
def reference_query_fde(Q, params, cfg):
    Q = np.asarray(Q, dtype=np.float64)
    G, P = params["G"].double().numpy(), params["P"].double().numpy()
    B, dp = 2 ** cfg["k_sim"], cfg["d_proj"]
    out = []
    for r in range(cfg["reps"]):
        blk = np.zeros((B, dp))
        for q in Q:
            b = sum(int(np.dot(G[r][i], q) > 0) << i for i in range(cfg["k_sim"]))
            blk[b] += P[r] @ q
        out.append(blk.reshape(-1))
    return np.concatenate(out)


def reference_doc_fde(D, params, cfg):
    D = np.asarray(D, dtype=np.float64)
    G, P = params["G"].double().numpy(), params["P"].double().numpy()
    B, dp, k = 2 ** cfg["k_sim"], cfg["d_proj"], cfg["k_sim"]
    out = []
    for r in range(cfg["reps"]):
        bs = [sum(int(np.dot(G[r][i], x) > 0) << i for i in range(k)) for x in D]
        Y = [P[r] @ x for x in D]
        blk = np.zeros((B, dp))
        for bucket in range(B):
            members = [j for j, b in enumerate(bs) if b == bucket]
            if members:
                blk[bucket] = np.mean([Y[j] for j in members], axis=0)
            else:
                dist = [bin(b ^ bucket).count("1") for b in bs]
                blk[bucket] = Y[int(np.argmin(dist))]
        out.append(blk.reshape(-1))
    return np.concatenate(out)
