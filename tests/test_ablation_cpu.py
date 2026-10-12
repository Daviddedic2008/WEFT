"""
CPU tests for the ablation baselines: the plain-IVF CSR (build_plain_ivf.py) and MUVERA FDEs
(muvera_fde.py). Run: python3 tests/test_ablation_cpu.py
"""
import os, sys, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ragrt_index_lib as L
from build_plain_ivf import plain_ivf_csr
import muvera_fde as M
from weft_prep import native_prep, imputed_passage_scores, residual_lut, residual_dot_reference


def test_plain_ivf(max_len, seed=0):
    rng = np.random.default_rng(seed)
    N, NC = 400, 30
    doclens = rng.integers(1, 40, N)
    pids = np.repeat(np.arange(N), doclens)
    codes = rng.integers(0, NC, len(pids))
    codes[codes == 7] = 3                                   # centroid 7 empty
    csr = plain_ivf_csr(codes, pids, NC, max_len=max_len)
    rp, eids, lens, P = csr["row_ptrs"], csr["col_eids"], csr["lengths"], csr["pids_sorted"]
    assert rp.shape == (L.NUM_SUBSPACES, NC + 1)
    assert (rp[1:] == len(lens)).all(), "subspaces 1-31 must be empty"
    assert lens.min() >= 1 and lens.max() <= max_len and lens.sum() == len(P)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    bs = L.block_sums(lens.astype(np.uint16))
    assert np.array_equal(L.offsets_from_block_sums(bs, lens, np.arange(len(lens))), starts)
    for c in range(NC):
        lo, hi = rp[0, c], rp[0, c + 1]
        assert np.array_equal(eids[lo:hi], np.arange(hi - lo)), "chunk eids must be 0..k-1 (sorted for binary search)"
        got = np.concatenate([P[starts[i]:starts[i] + lens[i]] for i in range(lo, hi)]) if hi > lo else np.array([], int)
        want = np.unique(pids[codes == c])
        assert np.array_equal(got, want), f"centroid {c}"
    assert rp[0, 8] == rp[0, 7], "empty centroid has no lists"
    print(f"  plain IVF (max_len={max_len}): {len(lens)} lists, E_needed={csr['E_needed']}, every centroid matches brute force")


def test_plain_ivf_tokens(max_len, seed=1):
    """Token-level index: one posting per token, token ids aligned with pids, sorted by token in a list."""
    rng = np.random.default_rng(seed)
    N, NC = 300, 25
    doclens = rng.integers(1, 30, N)
    pids = np.repeat(np.arange(N), doclens)
    codes = rng.integers(0, NC, len(pids))
    csr = plain_ivf_csr(codes, pids, NC, max_len=max_len, tokens=True)
    rp, lens, P, T = csr["row_ptrs"], csr["lengths"], csr["pids_sorted"], csr["token_ids"]
    assert T.dtype == np.uint32 and len(T) == len(P) == len(codes)
    assert np.array_equal(np.sort(T.astype(np.int64)), np.arange(len(codes))), "every token exactly once"
    assert np.array_equal(pids[T.astype(np.int64)], P), "token ids aligned with pids"
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    for c in range(NC):
        lo, hi = rp[0, c], rp[0, c + 1]
        got = np.concatenate([T[starts[i]:starts[i] + lens[i]] for i in range(lo, hi)]) if hi > lo else np.array([], np.uint32)
        assert np.array_equal(got.astype(np.int64), np.flatnonzero(codes == c)), f"centroid {c}"
    print(f"  token-level plain IVF (max_len={max_len}): {len(P)} postings, every centroid's tokens match")


def colbert_reversed_bits(i, nbits=2):
    """ColBERT ResidualCodec.reversed_bit_map entry (bit order reversed within each nbits group)."""
    mask, z = (1 << nbits) - 1, 0
    for j in range(8, 0, -nbits):
        x = (i >> (j - nbits)) & mask
        y = sum(((x >> (nbits - k - 1)) & 1) << k for k in range(nbits - 1, -1, -1))
        z |= y
        if j > nbits:
            z <<= nbits
    return z


def test_residual_lut(seed=0):
    """32 LUT lookups == q . decoded 2-bit residual (same decode as ColBERT / the rerank kernel)."""
    g = torch.Generator().manual_seed(seed)
    # ColBERT nbits=2 tables: reversed_bit_map reverses the 2-bit groups of a byte; decomp_table
    # maps a byte to its 4 bucket indices
    rbm = torch.tensor([colbert_reversed_bits(i) for i in range(256)], dtype=torch.uint8)
    dt = torch.tensor(list(itertools.product(range(4), repeat=4)), dtype=torch.uint8)     # [256, 4]
    bw = torch.tensor([-0.031, -0.009, 0.008, 0.030])
    Q = torch.randn(9, 128, generator=g)
    lut = residual_lut(Q, bw, rbm, dt)
    assert lut.shape == (9, 32, 256)
    res = torch.randint(0, 256, (50, 32), generator=g, dtype=torch.uint8)
    for t in range(9):
        for i in range(50):
            via_lut = float(lut[t, torch.arange(32), res[i].long()].sum())
            ref = residual_dot_reference(Q[t], res[i], bw, rbm, dt)
            assert abs(via_lut - ref) < 1e-4, (t, i, via_lut, ref)
    print("  residual LUT (32 lookups) == q . decoded residual")


def test_compact_prep(seed=3):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(10, 128, generator=g); C = torch.randn(400, 128, generator=g)
    for impute in (False, True):
        t1, s1, m1 = native_prep(Q, C, 9, impute)
        t2, s2, m2 = native_prep(Q, C, 9, impute, compact=True)
        assert torch.equal(t1, t2) and s2.shape == (10, 9)
        assert torch.allclose(torch.gather(s1, 1, t1.long()), s2, atol=1e-6)
        assert (m1 is None) == (m2 is None) and (m1 is None or torch.equal(m1, m2))
    print("  compact prep == full prep gathered at topc")


def test_muvera_matches_reference():
    cfg, P = M.make_params(dim=24, reps=4, k_sim=3, d_proj=6, seed=2)
    rng = np.random.default_rng(1)
    lens = [1, 3, 2, 17, 6]
    docs = [rng.standard_normal((n, 24)) for n in lens]
    X = torch.from_numpy(np.concatenate(docs))
    doc = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
    F = M.doc_fdes(X, doc, len(lens), P, cfg).numpy()
    assert F.shape == (len(lens), cfg["fde_dim"]) == (5, 4 * 8 * 6)
    for i, d in enumerate(docs):
        assert np.allclose(F[i], M.reference_doc_fde(d, P, cfg), atol=1e-10), f"doc {i}"
    q = rng.standard_normal((9, 24))
    assert np.allclose(M.query_fde(torch.from_numpy(q), P, cfg).numpy(), M.reference_query_fde(q, P, cfg), atol=1e-10)
    print("  MUVERA FDEs (vectorized, with empty-bucket fill) == per-passage reference")


def test_muvera_tracks_chamfer():
    """FDE scores must track Chamfer, and finer SimHash partitions must track it better."""
    rng = np.random.default_rng(0)
    norm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    centers = norm(rng.standard_normal((40, 32)))
    docs = [norm(centers[rng.integers(0, 40, 12)] + 0.3 * rng.standard_normal((12, 32))) for _ in range(300)]
    qs = [norm(centers[rng.integers(0, 40, 8)] + 0.3 * rng.standard_normal((8, 32))) for _ in range(10)]
    X = torch.from_numpy(np.concatenate(docs)); doc = torch.repeat_interleave(torch.arange(300), 12)
    cham = [np.array([(q @ d.T).max(1).sum() for d in docs]) for q in qs]
    corr = {}
    for k in (0, 5):
        cfg, P = M.make_params(dim=32, reps=20, k_sim=k, d_proj=32, seed=0, identity=True)
        F = M.doc_fdes(X, doc, 300, P, cfg).numpy()
        corr[k] = np.mean([np.corrcoef(c, F @ M.query_fde(torch.from_numpy(q), P, cfg).numpy())[0, 1]
                           for q, c in zip(qs, cham)])
    print(f"  MUVERA: corr(FDE score, Chamfer) {corr[0]:.3f} with 1 bucket, {corr[5]:.3f} with 32 buckets")
    assert corr[5] > corr[0] + 0.05 and corr[5] > 0.5, corr


def test_native_prep_and_imputation(seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.nn.functional.normalize(torch.randn(12, 128, generator=g), dim=-1)
    C = torch.nn.functional.normalize(torch.randn(500, 128, generator=g), dim=-1)
    nc = 7
    topc, s, m = native_prep(Q, C, nc, impute=False)
    full = Q @ C.T
    assert m is None and torch.allclose(s, full, atol=1e-6)
    assert torch.equal(topc.long(), full.topk(nc, dim=-1).indices)
    topc2, s2, m2 = native_prep(Q, C, nc, impute=True)
    assert torch.equal(topc2, topc)
    assert torch.allclose(m2, full.topk(nc + 1, dim=-1).values[:, nc])
    assert torch.allclose(s2, full - m2[:, None], atol=1e-6)
    probed = torch.gather(s2, 1, topc2.long())
    assert (probed >= -1e-6).all(), "probed centroids must have non-negative shifted scores"
    # ranking by the shifted, found-only sum == ranking by the imputed sum
    P = 300
    found = torch.rand(P, 12, generator=g) < 0.4
    best = torch.where(found, m2[None, :] + torch.rand(P, 12, generator=g), torch.zeros(P, 12))
    imputed = imputed_passage_scores(best, found, m2)
    shifted = torch.where(found, best - m2[None, :], torch.zeros(P, 12)).sum(1)
    assert torch.allclose(imputed - shifted, m2.sum().expand(P), atol=1e-4)
    assert torch.equal(torch.argsort(imputed, descending=True, stable=True),
                       torch.argsort(shifted, descending=True, stable=True))
    print("  native 128D prep matches Q.C^T; imputation shift is rank-equivalent to WARP-style imputation")


if __name__ == "__main__":
    test_plain_ivf(max_len=65535)
    test_plain_ivf(max_len=5)            # forces multi-chunk centroids
    test_plain_ivf_tokens(max_len=65535)
    test_plain_ivf_tokens(max_len=4)
    test_residual_lut()
    test_compact_prep()
    test_muvera_matches_reference()
    test_muvera_tracks_chamfer()
    test_native_prep_and_imputation()
    print("ABLATION CPU TESTS PASSED")
