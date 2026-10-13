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
from weft_prep import native_prep, imputed_passage_scores, residual_lut, residual_dot_reference, oracle_prep


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


def test_bucket_cap():
    from graph_search import bucket_cap, mask_to_i32
    prev = 0
    for P in [0, 1, 1000, 65535, 65536, 70000, 10**6, 3 * 10**6 + 7]:
        c = bucket_cap(P, 512)
        assert c >= P + 1 and c >= 512 and c >= (1 << 16), (P, c)
        assert c <= max(1 << 16, int((P + 1) * 1.126) + 1), (P, c)      # <= 1/8 octave of padding
        assert c >= prev; prev = c
    assert bucket_cap(10, 100000) >= 100000
    assert mask_to_i32(1 << 31) == -(1 << 31) and mask_to_i32(5) == 5
    print("  graph capacity buckets: >= P + 1, <= 12.5% padding, monotone")


def test_corr_filters():
    import types
    if "engines" not in sys.modules:                       # build_corr_filters imports the GPU engines
        sys.modules["engines"] = types.ModuleType("engines"); sys.modules["engines"].Engines = None
    import build_corr_filters as B
    g = torch.Generator().manual_seed(0)
    Cn = torch.nn.functional.normalize(torch.randn(50, 128, generator=g), dim=-1)
    doclens = torch.tensor([3, 1, 5, 2])
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), doclens.cumsum(0)])
    codes = torch.randint(0, 50, (int(doclens.sum()),), generator=g, dtype=torch.int32)
    M = B.passage_means(codes, offs, Cn, chunk=3)
    for i in range(4):
        ref = torch.nn.functional.normalize(Cn[codes[offs[i]:offs[i + 1]].long()].mean(0), dim=0)
        assert torch.allclose(M[i], ref, atol=1e-5), i
    centers = torch.nn.functional.normalize(torch.randn(16, 128, generator=g), dim=-1)
    q = centers[5] + 0.01 * torch.randn(128, generator=g)
    r = B.rank_filters(torch.nn.functional.normalize(q, dim=0), centers)
    assert r["corr_near"] == 5 and r["corr_far"] == int(torch.argmin(centers @ q)) and len(set(r.values())) == 3
    import tempfile
    d = tempfile.mkdtemp(); path = os.path.join(d, "predicates.npy")
    np.save(path, np.array([0b1, 0b10 | (1 << 20), 0xFFFFFFFF], dtype=np.uint32))
    B.update_predicates(path, np.array([0, 3, 15]), 16)
    got = np.load(path)
    assert got.tolist() == [0b1 | (1 << 16), 0b10 | (1 << 19), 0xFFFF | (1 << 31)], [bin(x) for x in got]
    assert os.path.exists(path + ".bak")
    print("  correlated filters: passage means, topic ranking, predicate bits 16-31 replaced, low bits kept")


def test_adaptive_probe():
    from weft_prep import adaptive_probe
    g = torch.Generator().manual_seed(4)
    C = torch.nn.functional.normalize(torch.randn(200, 128, generator=g), dim=-1)
    Q = torch.nn.functional.normalize(torch.randn(6, 128, generator=g), dim=-1)
    tot = torch.full((200,), 100.0)
    order = (Q @ C.T).argsort(dim=-1, descending=True)
    nc, ncap = 4, 16
    # filter uncorrelated with the query (10% of every list passes): no extra probing
    t, s, n = adaptive_probe(Q, C, nc, ncap, torch.full((200,), 10.0), tot, 0.10)
    assert n == nc and t.shape == (6, nc) and torch.equal(t.long(), order[:, :nc])
    full = Q @ C.T
    assert torch.allclose(s, full.gather(1, order[:, :nc]) - full.gather(1, order[:, nc:nc + 1]), atol=1e-6)
    # anti-correlated: the query's nearest lists hold no passing postings -> probe to the cap
    near = torch.zeros(200, dtype=torch.bool); near[order[:, :12].flatten()] = True
    pass_c = torch.where(near, 0.0, 1.0)
    t, s, n = adaptive_probe(Q, C, nc, ncap, pass_c, tot, 0.10)
    assert n == ncap and (s >= -1e-6).all()
    # positively correlated: everything near the query passes -> nc unchanged
    t, s, n = adaptive_probe(Q, C, nc, ncap, torch.where(near, 100.0, 0.0), tot, 0.06)
    assert n == nc
    # quantile over tokens: only one token is starved -> q = 0.75 keeps nc, q = 1 (max) probes deeper
    others = set(order[1:, :ncap].flatten().tolist())
    starved = torch.zeros(200, dtype=torch.bool)
    starved[[c for c in order[0, :12].tolist() if c not in others]] = True
    pc = torch.where(starved, 0.0, 10.0)
    assert adaptive_probe(Q, C, nc, ncap, pc, tot, 0.10, q=0.75)[2] == nc
    assert adaptive_probe(Q, C, nc, ncap, pc, tot, 0.10, q=1.0)[2] > nc
    # random filter: every list a little under the expected count -> slack keeps the fast path
    pr = torch.full((200,), 9.0)
    assert adaptive_probe(Q, C, nc, ncap, pr, tot, 0.10, q=0.75)[2] > nc
    assert adaptive_probe(Q, C, nc, ncap, pr, tot, 0.10, q=0.75, slack=0.5)[2] == nc
    assert adaptive_probe(Q, C, nc, ncap, pass_c, tot, 0.10, slack=0.5)[2] == ncap
    print("  adaptive probing: uncorrelated / correlated filters keep nc, anti-correlated probes to the cap")


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



def test_oracle_prep():
    """weft_oracle: skip centroids with no passing postings, impute from the next non-empty one."""
    from weft_prep import adaptive_probe
    g = torch.Generator().manual_seed(7)
    C = torch.nn.functional.normalize(torch.randn(300, 128, generator=g), dim=-1)
    Q = torch.nn.functional.normalize(torch.randn(5, 128, generator=g), dim=-1)
    full = Q @ C.T
    order = full.argsort(dim=-1, descending=True)
    nc = 6
    # no empty centroid: identical to native_prep with imputation
    none = torch.zeros(300, dtype=torch.bool)
    t, s, depth = oracle_prep(Q, C, nc, none, 300)
    t0, s0, _ = native_prep(Q, C, nc, impute=True, compact=True)
    assert torch.equal(t, t0) and torch.allclose(s, s0) and abs(float(depth) - nc) < 1e-6
    # empty centroids among each token's nearest: probe the nearest non-empty ones, impute from the next
    empty = torch.zeros(300, dtype=torch.bool)
    empty[order[:, [0, 2, 3]].flatten()] = True
    t, s, depth = oracle_prep(Q, C, nc, empty, int((~empty).sum()))
    for i in range(5):
        ok = [c for c in order[i].tolist() if not empty[c]]
        assert t[i].tolist() == ok[:nc]
        m = float(full[i, ok[nc]])
        assert torch.allclose(s[i], full[i, ok[:nc]] - m, atol=1e-6) and (s[i] >= -1e-6).all()
        # the oracle's imputed value is never above plain weft's (it skips empty lists)
        assert m <= float(full[i, order[i, nc]]) + 1e-6
    assert float(depth) > nc
    # compacted fast path (weft_oracle) == the masked reference
    from weft_prep import compact_centroids, oracle_fast_prep, oracle_ablation, unfiltered_depth
    C_ok, ids = compact_centroids(C, empty)
    tf, sf, last = oracle_fast_prep(Q, C_ok, ids, nc)
    assert torch.equal(tf, t) and torch.allclose(sf, s, atol=1e-6)
    assert abs(float(unfiltered_depth(Q, C, last)) - float(depth)) < 1e-6
    assert compact_centroids(C, none)[1] is None             # random filters: no copy, no remap
    # weft_oracle_adapt: adaptive probing over the compacted centroids, mapped back
    pass_c = torch.where(empty, 0.0, 10.0)
    st = {}
    t2, s2, n2 = adaptive_probe(Q, C_ok, nc, 4 * nc, pass_c[ids], torch.full((len(ids),), 10.0), 1.0,
                                q=0.75, slack=0.5, stats=st)
    assert n2 == nc and torch.equal(ids[t2.long()].int(), t) and torch.allclose(s2, s)
    # ablation halves: "oimp" probes like weft and imputes from the next non-empty centroid below them;
    # "oskip" probes like weft_oracle and imputes from the very next centroid, empty or not
    ti, si, _ = oracle_ablation(Q, C, nc, empty, "oimp")
    tk, sk, _ = oracle_ablation(Q, C, nc, empty, "oskip")
    for i in range(5):
        o = order[i].tolist()
        ok = [c for c in o if not empty[c]]
        mi = float(full[i, next(c for c in o[nc:] if not empty[c])])
        assert ti[i].tolist() == o[:nc] and torch.allclose(si[i], full[i, o[:nc]] - mi, atol=1e-6)
        last_pos = o.index(ok[nc - 1])
        mk = float(full[i, o[last_pos + 1]])
        assert tk[i].tolist() == ok[:nc] and torch.allclose(sk[i], full[i, ok[:nc]] - mk, atol=1e-6)
        assert (si[i] >= -1e-6).all() and (sk[i] >= -1e-6).all()
    print("  oracle prep: skips empty centroids, imputes from the next non-empty one; compacted path and ablations agree")

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
    test_bucket_cap()
    test_corr_filters()
    test_adaptive_probe()
    test_muvera_matches_reference()
    test_muvera_tracks_chamfer()
    test_native_prep_and_imputation()
    test_oracle_prep()
    print("ABLATION CPU TESTS PASSED")
