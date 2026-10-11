"""
CPU tests for the ablation baselines: the plain-IVF CSR (build_plain_ivf.py) and MUVERA FDEs
(muvera_fde.py). Run: python3 tests/test_ablation_cpu.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ragrt_index_lib as L
from build_plain_ivf import plain_ivf_csr
import muvera_fde as M


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


if __name__ == "__main__":
    test_plain_ivf(max_len=65535)
    test_plain_ivf(max_len=5)            # forces multi-chunk centroids
    test_muvera_matches_reference()
    test_muvera_tracks_chamfer()
    print("ABLATION CPU TESTS PASSED")
