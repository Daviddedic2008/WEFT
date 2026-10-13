"""
The three retrieval engines behind one interface, for benchmarks and profiling.

  plaid  : stock ColBERTv2/PLAID (searcher.ranker.rank)
  ptms   : PLAID stages 1-3, then fused TileMaxSim rerank
  ragrt  : RT-core candidate generation + fused TileMaxSim rerank
  ragrt_bf : identical pipeline, but Stage 1 runs on CUDA cores: the no-RT ablation.
  ragrt_nofilter : codebook-filter ablation: Stage 1 returns every codeword, so Stage 2
             fetches every list of each probed centroid (plain centroid probing on the same
             index and kernels). Grid parameter eids = "all".
             polar geometry: the same threshold query as a linear scan (identical hits);
             fan geometry: exact top-k codewords per token and subspace.
  ragrt_centroid : codeword-score ablation: same lists as ragrt_bf (CUDA-core Stage 1, top
             eids codewords), but each posting is scored q.c only (no q_s.w term). ragrt_bf vs
             this isolates what the codeword term adds to the approximate score.
  ragrt_ivf : plain IVF on the RAGRT engine: q.c scoring, every list fetched. Meant for a
             plain-IVF index (build_plain_ivf.py: one list per centroid, deduplicated pids),
             so it is PLAID's candidate generation (centroid interaction) run on RAGRT's
             kernels. ptms vs this isolates the engine; this vs ragrt_bf isolates the codebooks.
  weft     : the plain-IVF engine with the two cheap improvements: Stage 0b centroid scores from
             one fp16 GEMM in the native 128 dimensions (no SVD projection), and WARP-style
             missing-score imputation (weft_prep.py). Run on a build_plain_ivf.py index.
  weft_noimpute : weft without the imputation (isolates it; ragrt_ivf vs this isolates the prep).
  Residual-correction ablation (WARP-style implicit decompression), on a TOKEN-level plain-IVF
  index (build_plain_ivf.py --tokens: one posting per token, csr_token_ids.npy):
  weft_tok : weft on the token-level index, no residuals (the cost of not deduplicating).
  weft_resid : each posting scored q.c + q.r (32 LUT lookups, weft_prep.residual_lut) - m, then
             the same exact rerank. Tests whether more precise approximate scores cut ndocs.
  weft_resid_norr : weft_resid without the rerank: the top-k by approximate score, i.e. WARP's
             pipeline (imputation + implicit decompression, no rerank) ported to the GPU.
  weft_graph : weft run as two CUDA graph replays with one mid-query sync (graph_search.py):
             same candidates and results as weft, less launch and sync overhead.
  weft_hash : weft with the hash-table Stage 3 (no radix sort): same scores, less Stage-3 time.
  weft_pairs : weft with the 32-bit-key pair sort in Stage 3 (pid<<5|token keys, fp16 values):
             fewer, narrower radix passes; a reduce-by-key takes the max per (passage, token).
  weft_adapt : weft + adaptive probing under a filter: when the probed centroids' lists hold fewer
             passing postings than an uncorrelated filter would (per-centroid filter-bit counts,
             centroid_bit_counts.npy), probe up to ADAPT_CAP x nc centroids (anti-correlated filters).
  weft_oracle : weft with filter-aware prep under a filter (weft_prep.oracle_prep): each query token
             probes its nc nearest centroids that hold passing postings and imputes from the next such
             centroid, emulating an IVF built over only the passing passages (cf. ACORN's oracle index).
  weft_oracle_adapt : weft_adapt with the same skipping of empty centroids.
  muvera   : MUVERA fixed-dimensional encodings (build_muvera_fde.py), exact GPU inner
             product over all passages, then the same TileMaxSim rerank. Grid: cands.

RAGRT options (Engines(..., geometry=, stage3=, rerank=, rt_quantile=)):
  geometry  "polar" (default): exact-threshold scene, one triangle per codeword in the
            plane x.c = 1; a ray hits exactly the codewords with d.c >= tau_s. tau_s is
            calibrated per subspace and per k_eids level on sample query tokens
            (calibrate()): the rt_quantile quantile of the k-th best d.c, so ~(1 - q) of
            rays get at least k hits. "fan": the original scene (hits ~every codeword with
            d.c > 0, any-hit computes the exact value; RT acts as a half-space filter).
            "polar_iter": the polar scene without hit shaders: k nearest-hit traversals per
            ray (optixTraverse; nearest = best score), exact top-k in descending order.
  stage3    "sparse" (default): sort only the touched postings. "dense": ntok x N table.
  rerank    "wmma" (default): tensor-core TileMaxSim. "simt": the original kernel.
            Applies to every engine that uses TileMaxSim (ptms too).

Every search takes an already-encoded query (Qf: [32, 128] fp16 on the GPU) and
returns a python list of pids, best first. Query encoding (BERT) is common to all
engines and timed separately as stage 0a. RAGRT's own query prep (128->96
projection, centroid scores, top-nc) is stage 0b and is inside RAGRT's search.
"""
import os, sys
from math import ceil
import numpy as np
import torch

import eval_config as C
sys.path.insert(0, C.BASE_DIR)
sys.path.insert(0, C.COLBERT_DIR)

from colbert import Searcher
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.colbert import colbert_score_reduce
import rtrag_corr_3d
import ragrt_index_lib as L
from weft_prep import native_prep, residual_lut, residual_byte_weights, adaptive_probe, oracle_prep

NEG_INF = float("-inf")


def plaid_cfg_key(p):
    return f"nc={p['ncells']} th={p['threshold']} ndocs={p['ndocs']}"


def ragrt_cfg_key(p):
    return f"nc={p['nc']} eids={p['eids']} ndocs={p['ndocs']}"


WEFT_ENGINES = ("weft", "weft_noimpute", "weft_tok", "weft_resid", "weft_resid_norr", "weft_graph", "weft_hash",
                "weft_adapt", "weft_pairs", "weft_oracle", "weft_oracle_adapt")
HASH_ENGINES = ("weft_hash",)
PAIRS_ENGINES = ("weft_pairs",)
ADAPT_ENGINES = ("weft_adapt", "weft_oracle_adapt")   # per-query nc: no batched pipeline
ORACLE_ENGINES = ("weft_oracle", "weft_oracle_adapt")  # skip centroids with no passing postings
ADAPT_CAP = 8                            # probe at most ADAPT_CAP x nc centroids
ADAPT_Q = 0.75                           # quantile of the tokens' needed nc (1.0 = max)
ADAPT_SLACK = 0.5                        # token is starved below this share of the uncorrelated count
GRAPH_ENGINES = ("weft_graph",)          # CUDA-graph path: no per-stage profile, no batched pipeline
RESID_ENGINES = ("weft_resid", "weft_resid_norr")       # need a token-level index (csr_token_ids.npy)
TOKEN_ENGINES = ("weft_tok",) + RESID_ENGINES
RAGRT_ENGINES = ("ragrt", "ragrt_bf", "ragrt_nofilter", "ragrt_centroid", "ragrt_ivf") + WEFT_ENGINES
STAGE1_MODE = {"ragrt": 0, "ragrt_bf": 1, "ragrt_nofilter": 2,      # see CorrIndex3D.set_stage1_mode
               "ragrt_centroid": 1, "ragrt_ivf": 2, **{e: 2 for e in WEFT_ENGINES}}
CODEWORD_WEIGHT = {"ragrt": 1.0, "ragrt_bf": 1.0, "ragrt_nofilter": 1.0,   # weight of q_s.w in Stage 2
                   "ragrt_centroid": 0.0, "ragrt_ivf": 0.0, **{e: 0.0 for e in WEFT_ENGINES}}
# Stage 0b: "svd96" = centroid scores in the 96D SVD space (codebook engines); "native" = fp16 GEMM
# against the 128D ColBERT centroids; impute = missing-score imputation; compact = scores [ntok, nc]
# aligned with topc instead of [ntok, num_centroids] (weft_prep.py; identical results).
# resid = add q.r per posting (residual LUT; token-level index only).
PREP = {"weft": ("native", True, True, False), "weft_noimpute": ("native", False, True, False),
        "weft_tok": ("native", True, True, False), "weft_resid": ("native", True, True, True),
        "weft_resid_norr": ("native", True, True, True), "weft_graph": ("native", True, True, False),
        "weft_hash": ("native", True, True, False), "weft_adapt": ("native", True, True, False),
        "weft_pairs": ("native", True, True, False), "weft_oracle": ("native", True, True, False),
        "weft_oracle_adapt": ("native", True, True, False)}
NO_PREP = ("svd96", False, False, False)


def muvera_cfg_key(p):
    return f"cands={p['cands']}"


def cfg_key(engine, p):
    if engine in RAGRT_ENGINES:
        return ragrt_cfg_key(p)
    if engine == "muvera":
        return muvera_cfg_key(p)
    return plaid_cfg_key(p)


RT_LEVELS = (4, 8, 16, 32, 64, 128)     # k_eids values the polar scene is calibrated for


class Engines:
    def __init__(self, ds, load_ragrt=True, geometry="polar", stage3="sparse", rerank="wmma",
                 rt_quantile=0.05, rt_levels=RT_LEVELS, muvera_dir=None, encoder="stock"):
        assert geometry in ("polar", "polar_iter", "fan") and stage3 in ("sparse", "dense") and rerank in ("wmma", "simt")
        self.ds = ds
        self.geometry, self.stage3, self.rerank, self.rt_quantile = geometry, stage3, rerank, rt_quantile
        self.rt_levels = tuple(rt_levels)
        self.calibration = None
        rtrag_corr_3d.set_rerank_mode(rerank)
        self.searcher = Searcher(index=ds["index"], collection=ds["collection"])
        self.ranker = self.searcher.ranker
        self.N = len(self.ranker.doclens)
        assert encoder in ("stock", "fast")
        self.encoder = encoder
        self.fast_encoder = None
        if encoder == "fast":
            from fast_encoder import FastQueryEncoder
            self.fast_encoder = FastQueryEncoder(self.searcher.checkpoint)

        codec = self.ranker.codec
        doclens = torch.tensor(self.ranker.doclens, dtype=torch.int32, device="cuda")
        self.doclens = doclens
        self.doc_offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device="cuda"),
                                      torch.cumsum(doclens.to(torch.int64), dim=0)])
        self.codes = self.ranker.embeddings.codes[:int(self.doc_offsets[-1])].cuda().to(torch.int32).contiguous()
        self.residuals = self.ranker.embeddings.residuals[:int(self.doc_offsets[-1])].cuda().contiguous()
        self.centroids_fp16 = codec.centroids.cuda().to(torch.float16).contiguous()
        self.bucket_weights = codec.bucket_weights.cuda().to(torch.float16).contiguous()
        self.reversed_bit_map = codec.reversed_bit_map.cuda().to(torch.uint8).contiguous()
        self.decomp_table = codec.decompression_lookup_table.cuda().to(torch.uint8).contiguous()

        pred = np.load(ds["predicates_path"])
        assert len(pred) == self.N, f"{ds['predicates_path']} has {len(pred):,} rows, index has {self.N:,}"
        self.pred_np = pred.astype(np.uint32)
        self.pred_gpu = torch.from_numpy(self.pred_np.astype(np.int64)).cuda()

        self.prep_mode = NO_PREP
        self.has_token_ids = False
        self.graph = None
        self.adapt = False
        self.cur_mask = 0
        self.cbc = None                 # per-centroid filter-bit counts (weft_adapt)
        self.oracle = False             # weft_oracle*: skip centroids with no passing postings
        self.empty_cache = {}           # filter mask -> (bool [NC] empty centroids, non-empty count)
        self.sel_cache = {}
        self.pass_cache = {}
        self.adapt_log = []
        self.zero_sub = None
        self.res_wb = None
        self.index = None
        if load_ragrt:
            self._load_ragrt()
        self.muvera = None
        if muvera_dir:
            self._load_muvera(muvera_dir)

    # ------------------------------------------------------------------ setup
    def _load_ragrt(self):
        d = self.ds["sparse_csr_dir"]
        ld = lambda n: np.load(os.path.join(d, n))
        self.R = torch.from_numpy(ld("svd_rotation_128_to_96.npy")).cuda().float()
        self.C96 = torch.from_numpy(ld("centroids_96d_svd.npy")).cuda().float()
        self.CB = torch.tensor(ld("codebooks.npy"), dtype=torch.float32).contiguous()
        self.E = self.CB.shape[1]
        eids = ld("csr_col_eids.npy")
        assert eids.dtype == L.eid_dtype(self.E), f"csr_col_eids is {eids.dtype} but E={self.E}"
        if eids.dtype == np.uint16:
            eids = eids.view(np.int16)          # same bytes; the kernel reads them as uint16
        self.index = rtrag_corr_3d.CorrIndex3D(C.PTX_PATH)
        self.index.build(self.CB, 0.95, 4)
        self.index.set_geometry({"fan": 0, "polar": 1, "polar_iter": 2}[self.geometry])
        self.index.set_stage3_sparse(self.stage3 == "sparse")
        self.CB = self.CB.cuda()
        self.index.bind_index(
            torch.from_numpy(ld("csr_row_ptrs.npy").astype(np.int64)).cuda(),
            torch.from_numpy(eids).cuda(),
            torch.from_numpy(ld("csr_block_sums_128.npy").astype(np.int64)).cuda(),
            torch.from_numpy(ld("csr_lengths.npy")).cuda(),
            torch.from_numpy(ld("csr_packed_24.npy")).cuda(),
            self.doc_offsets[:self.N + 1], self.doclens[:self.N], self.codes, self.residuals,
            self.centroids_fp16, self.bucket_weights, self.reversed_bit_map, self.decomp_table,
            torch.from_numpy(self.pred_np.astype(np.int64)).cuda().to(torch.uint32), self.N, self.C96.shape[0])
        tp = os.path.join(d, "csr_token_ids.npy")
        if os.path.exists(tp):                  # token-level plain-IVF index (residual ablation)
            tids = np.load(tp)
            assert tids.dtype == np.uint32 and int(tids.max()) < self.codes.shape[0], "csr_token_ids out of range"
            self.index.bind_token_ids(torch.from_numpy(tids.view(np.int32)).cuda())
            self.has_token_ids = True

    def _load_muvera(self, d):
        import muvera_fde as M
        cfg, params = M.load_params(d)
        F = np.load(os.path.join(d, "fde_docs_fp16.npy"), mmap_mode="r")
        assert F.shape[0] == self.N, f"{d}: {F.shape[0]:,} doc FDEs, index has {self.N:,} passages"
        Fg = torch.empty(F.shape, dtype=torch.float16, device="cuda")
        for a in range(0, F.shape[0], 200_000):              # chunked: no full host copy
            Fg[a:a + 200_000] = torch.from_numpy(np.asarray(F[a:a + 200_000])).cuda()
        self.muvera = {"F": Fg, "cfg": cfg, "params": {k: v.cuda() for k, v in params.items()}}
        print(f"  MUVERA: {F.shape[0]:,} x {F.shape[1]} FDEs from {d} ({cfg})", flush=True)

    # ------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, Qs, quantile=None):
        """
        Per-subspace thresholds for the polar scene from sample queries [(Qf, ntok)]
        (use TUNE queries). For level k: tau_s = the `quantile` quantile, over rays, of the
        k-th largest unit-query . codeword in subspace s, clamped to [0.02, 0.999] x max|c_s|.
        """
        if self.index is None or self.geometry == "fan":
            return None
        q = self.rt_quantile if quantile is None else quantile
        X = torch.cat([torch.nn.functional.normalize(Qf[:n].float() @ self.R, p=2, dim=-1).view(n, L.NUM_SUBSPACES, 3)
                       for Qf, n in Qs])
        X = torch.nn.functional.normalize(X, p=2, dim=-1)
        ks = [min(k, self.E) for k in self.rt_levels]
        taus = torch.zeros(len(ks), L.NUM_SUBSPACES)
        hits = torch.zeros(len(ks), L.NUM_SUBSPACES)
        short = torch.zeros(len(ks), L.NUM_SUBSPACES)
        for s in range(L.NUM_SUBSPACES):
            sc = X[:, s] @ self.CB[s].T                                   # [rays, E]
            top = sc.topk(max(ks), dim=1).values
            cmax = float(self.CB[s].norm(dim=1).max())
            for l, k in enumerate(ks):
                t = float(torch.quantile(top[:, k - 1], q))
                t = min(max(t, 0.02 * cmax), 0.999 * cmax)
                taus[l, s] = t
                cnt = (sc >= t).sum(1).float()
                hits[l, s] = cnt.mean()
                short[l, s] = (cnt < k).float().mean()
        self.index.set_polar_levels(taus, list(ks))
        self.calibration = {
            "quantile": q, "rays_per_subspace": int(X.shape[0]), "E": self.E,
            "levels": [{"k": k, "tau_mean": float(taus[l].mean()), "hits_per_ray_mean": float(hits[l].mean()),
                        "frac_rays_short": float(short[l].mean())} for l, k in enumerate(ks)],
        }
        for lv in self.calibration["levels"]:
            print(f"  polar level k={lv['k']:<4} tau~{lv['tau_mean']:.4f}  hits/ray {lv['hits_per_ray_mean']:7.1f}  "
                  f"rays short of k {lv['frac_rays_short']:.1%}  (E={self.E})", flush=True)
        return self.calibration

    def calibrate_from_questions(self, n=300):
        """Calibrate on the first n queries of the TUNE split (by qid hash, C.TUNE_SIZE)."""
        import eval_lib as EL
        qs = EL.load_questions(self.ds["questions_path"])
        tune, _ = EL.make_split([q for q, _ in qs], C.TUNE_SIZE, C.SPLIT_SEED)
        text = dict(qs)
        return self.calibrate([self.encode(text[q]) for q in tune[:n]])

    def ensure_calibrated(self):
        if self.index is not None and self.geometry != "fan" and self.calibration is None:
            print("  calibrating polar thresholds on TUNE queries", flush=True)
            self.calibrate_from_questions()

    def options(self):
        d = {"geometry": self.geometry, "stage3": self.stage3, "rerank": self.rerank, "encoder": self.encoder,
             "rt_quantile": self.rt_quantile, "calibration": self.calibration}
        if self.muvera is not None:
            d["muvera"] = self.muvera["cfg"]
        if self.index is not None:
            d["index"] = {k: v for k, v in dict(self.index.info()).items() if k != "levels"}
            d["sparse_csr_dir"] = self.ds["sparse_csr_dir"]
        return d

    # ------------------------------------------------------------ stage 0a
    @torch.no_grad()
    def encode(self, text):
        """[32, 128] fp16 query matrix on the GPU, and the number of real tokens."""
        if self.fast_encoder is not None:
            return self.fast_encoder.encode(text)
        Qf = self.searcher.encode(text).squeeze(0).cuda().to(torch.float16).contiguous()
        assert Qf.shape == (32, 128), f"expected [32, 128] query, got {tuple(Qf.shape)}"
        return Qf, L.query_ntok(self.searcher, text)

    # ------------------------------------------------------------ helpers
    def tms_scores(self, Qf, pids_i32):
        return rtrag_corr_3d.native_tile_maxsim_fused_decomp(
            Qf, pids_i32, self.doc_offsets, self.doclens, self.codes, self.residuals,
            self.centroids_fp16, self.bucket_weights, self.reversed_bit_map, self.decomp_table)

    def passes(self, pids_list, mask):
        if not mask or not len(pids_list):
            return pids_list
        a = np.asarray(pids_list, dtype=np.int64)
        return a[(self.pred_np[a] & mask) != 0].tolist()

    def set_plaid(self, p):
        cfg = self.searcher.config
        cfg.ncells, cfg.centroid_score_threshold, cfg.ndocs = p["ncells"], p["threshold"], p["ndocs"]

    # ------------------------------------------------------------ PLAID
    @torch.no_grad()
    def plaid(self, Qf, p, mask=0):
        """Stock PLAID; a filter can only be applied after ranking (post-filter)."""
        self.set_plaid(p)
        pids, _ = self.ranker.rank(self.searcher.config, Qf.unsqueeze(0))
        pids = pids.tolist() if torch.is_tensor(pids) else list(pids)
        return self.passes(pids, mask)[:C.TOP_K]

    @torch.no_grad()
    def plaid_stage3(self, Qf, p):
        self.set_plaid(p)
        return get_stage3_pids(self.ranker, self.searcher.config, Qf.unsqueeze(0))

    @torch.no_grad()
    def ptms(self, Qf, p, mask=0):
        """PLAID stages 1-3, post-filter the candidates, fused TileMaxSim rerank. All on GPU."""
        cands = self.plaid_stage3(Qf, p).to(torch.int64)
        if mask:
            cands = cands[(self.pred_gpu[cands] & mask) != 0]
        if cands.numel() == 0:
            return []
        cands = cands.to(torch.int32).contiguous()
        s = self.tms_scores(Qf, cands)
        top = torch.topk(s, min(C.TOP_K, cands.numel())).indices
        return cands[top].tolist()

    # ------------------------------------------------------------ RAGRT
    @torch.no_grad()
    def ragrt_prep(self, Qf, ntok, nc):
        """Stage 0b: 3D sub-queries, centroid scores and the top-nc centroids per token
        (scores in the 96D SVD space, or native 128D fp16 with optional imputation: self.prep_mode)."""
        mode, impute, compact, resid = self.prep_mode
        if mode == "native":
            # WEFT engines: Stage 1 is "every codeword" with codeword weight 0, so the 3D sub-queries
            # are never used; skip the 128 -> 96 projection (zeros of the right shape).
            if self.zero_sub is None:
                self.zero_sub = torch.zeros(32, L.NUM_SUBSPACES, L.SUBSPACE_DIM, device="cuda")
            Q_sub = self.zero_sub[:ntok]
            if self.adapt and self.cur_mask:
                topc, scores = self.adaptive_prep(Qf[:ntok], nc, self.cur_mask)
                return Q_sub, scores, topc
            if self.oracle and self.cur_mask:
                empty, n_ok = self.empty_centroids(self.cur_mask)
                topc, scores, depth = oracle_prep(Qf[:ntok], self.centroids_fp16, nc, empty, n_ok)
                self.adapt_log.append(depth)        # read after the run (no sync here)
                return Q_sub, scores, topc
            topc, scores, _ = native_prep(Qf[:ntok], self.centroids_fp16, nc, impute, compact)
            if resid:   # read by the next Stage 3 (a global in the extension; one query at a time)
                if self.res_wb is None:
                    self.res_wb = residual_byte_weights(self.bucket_weights, self.reversed_bit_map, self.decomp_table)
                self.index.set_residual_lut(residual_lut(Qf[:ntok], None, None, None, wb=self.res_wb))
            return Q_sub, scores, topc
        Q96 = torch.nn.functional.normalize(Qf[:ntok].float() @ self.R, p=2, dim=-1)
        Q_sub = Q96.view(ntok, L.NUM_SUBSPACES, L.SUBSPACE_DIM).contiguous()
        scores = (Q96 @ self.C96.T).contiguous()
        topc = scores.topk(k=nc, dim=-1).indices.to(torch.int32).contiguous()
        return Q_sub, scores, topc

    # ------------------------------------------------------------ adaptive probing (weft_adapt)
    def centroid_bit_counts(self):
        """[num_centroids, 33] int64: per centroid, postings whose passage has predicate bit b (b < 32),
        and all postings (col 32). Cached next to the index, recomputed when predicates.npy changes."""
        d = self.ds["sparse_csr_dir"]
        cache = os.path.join(d, "centroid_bit_counts.npy")
        pp = self.ds["predicates_path"]
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(pp):
            return np.load(cache)
        print(f"  computing per-centroid filter-bit counts for {d} (once)", flush=True)
        rp = np.load(os.path.join(d, "csr_row_ptrs.npy"))[0].astype(np.int64)
        lengths = np.load(os.path.join(d, "csr_lengths.npy")).astype(np.int64)
        packed = np.load(os.path.join(d, "csr_packed_24.npy"), mmap_mode="r")
        NC = len(rp) - 1
        n_lists = int(rp[-1])
        list_c = (np.searchsorted(rp, np.arange(n_lists), side="right") - 1).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths[:n_lists])])
        out = np.zeros((NC, 33), dtype=np.int64)
        a = 0
        while a < n_lists:
            b = int(np.searchsorted(starts, starts[a] + 50_000_000, side="right")) - 1
            b = min(max(b, a + 1), n_lists)
            p0, p1 = int(starts[a]), int(starts[b])
            pids = L.unpack_pids_24(np.asarray(packed[3 * p0:3 * p1]))
            cent = np.repeat(list_c[a:b], lengths[a:b])
            pred = self.pred_np[pids]
            out[:, 32] += np.bincount(cent, minlength=NC)
            for bit in range(32):
                sel = ((pred >> np.uint32(bit)) & np.uint32(1)).astype(bool)
                if sel.any():
                    out[:, bit] += np.bincount(cent[sel], minlength=NC)
            a = b
        np.save(cache, out)
        return out

    def selectivity(self, mask):
        if mask not in self.sel_cache:
            self.sel_cache[mask] = float(((self.pred_gpu & mask) != 0).float().mean())
        return self.sel_cache[mask]

    @torch.no_grad()
    def adaptive_prep(self, Q, nc, mask):
        """Probe deeper when the filter is anti-correlated with the query. For each query token,
        target = (global pass rate) x (postings in its top-nc lists): the passing postings an
        uncorrelated filter would leave. Probe the smallest nc' in [nc, ADAPT_CAP * nc] whose lists
        hold that many passing postings (per-centroid bit counts; exact for one-bit masks, an upper
        bound for several bits), then use the largest nc' over tokens. Imputation uses the
        (nc' + 1)-th score. Returns (topc [ntok, nc'] int32, compact shifted scores [ntok, nc'])."""
        pass_c = self.pass_counts(mask)
        ncap = min(ADAPT_CAP * nc, self.centroids_fp16.shape[0] - 1)
        kw = {}
        if self.oracle:
            empty, n_ok = self.empty_centroids(mask)
            kw = {"empty": empty, "n_ok": n_ok, "stats": {}}
        topc, scores, nc2 = adaptive_probe(Q, self.centroids_fp16, nc, ncap, pass_c, self.cbc["tot"],
                                           self.selectivity(mask), q=ADAPT_Q, slack=ADAPT_SLACK, **kw)
        self.last_adapt_nc = nc2
        self.adapt_log.append(kw["stats"]["depth"] if self.oracle else nc2)
        return topc, scores

    def pass_counts(self, mask):
        """[NC] float: postings per centroid whose passage passes the mask (per mask, once; exact for
        one-bit masks, an upper bound for several bits, so zero always means truly empty)."""
        if self.cbc is None:
            c = torch.from_numpy(self.centroid_bit_counts()).cuda()
            self.cbc = {"bits": c[:, :32].float(), "tot": c[:, 32].float()}
        pass_c = self.pass_cache.get(mask)
        if pass_c is None:
            bits = torch.tensor([float((mask >> b) & 1) for b in range(32)], device=self.cbc["bits"].device)
            pass_c = self.pass_cache[mask] = self.cbc["bits"] @ bits
        return pass_c

    def empty_centroids(self, mask):
        """(bool [NC] centroids with no passing postings, number of non-empty centroids), per mask, once."""
        if mask not in self.empty_cache:
            empty = self.pass_counts(mask) == 0
            self.empty_cache[mask] = (empty, int((~empty).sum()))
        return self.empty_cache[mask]

    @torch.no_grad()
    def ragrt_candidates(self, Qf, ntok, p, mask=0):
        """Stages 0b-3 only: (candidate pids with -1 padding, approximate scores)."""
        self.ensure_calibrated()
        self.cur_mask = mask
        Q_sub, scores, topc = self.ragrt_prep(Qf, ntok, p["nc"])
        return self.index.candidates(Q_sub, topc, scores, topc.shape[1], self.ndocs(p), mask, p["eids"])

    def use_engine(self, engine):
        """Set Stage 1 mode, the Stage 2 codeword weight and the prep for a RAGRT engine."""
        if engine in TOKEN_ENGINES and not self.has_token_ids:
            raise RuntimeError(f"{engine} needs a token-level index (build_plain_ivf.py --tokens); "
                               f"{self.ds['sparse_csr_dir']} has no csr_token_ids.npy")
        self.index.set_stage1_mode(STAGE1_MODE[engine])
        rtrag_corr_3d.set_codeword_weight(CODEWORD_WEIGHT[engine])
        self.prep_mode = PREP.get(engine, NO_PREP)
        rtrag_corr_3d.set_compact_scores(self.prep_mode[2])
        rtrag_corr_3d.set_stage3_mode(2 if engine in PAIRS_ENGINES else (1 if engine in HASH_ENGINES else 0))
        self.adapt = engine in ADAPT_ENGINES
        self.oracle = engine in ORACLE_ENGINES
        if not self.prep_mode[3]:
            self.index.set_residual_lut(torch.empty(0))

    @torch.no_grad()
    def approx_topk(self, Qf, ntok, p, mask=0):
        """No-rerank search (weft_resid_norr): top-k candidates by approximate score."""
        cands, approx = self.ragrt_candidates(Qf, ntok, p, mask)
        ok = cands >= 0
        cands, approx = cands[ok], approx[ok]
        top = torch.topk(approx, min(C.TOP_K, cands.numel())).indices
        return cands[top].tolist()

    def resolve(self, engine, p):
        """eids = "all" (ragrt_nofilter, ragrt_ivf) means every codeword of the index."""
        if p.get("eids") == "all":
            p = dict(p, eids=self.E)
        return p

    def stage1(self, brute_force):
        """Select Stage 1: RT cores (False) or the CUDA-core ablation (True)."""
        self.index.set_stage1_mode(1 if brute_force else 0)

    def ndocs(self, p):
        return min(p["ndocs"], self.N)

    @torch.no_grad()
    def ragrt(self, Qf, ntok, p, mask=0):
        self.ensure_calibrated()
        self.cur_mask = mask
        Q_sub, scores, topc = self.ragrt_prep(Qf, ntok, p["nc"])
        out = self.index.search_single_query_native(
            Qf.float(), Qf, Q_sub, topc, scores, topc.shape[1], self.ndocs(p), C.TOP_K,
            query_mask=mask, k_eids=p["eids"])
        # Padding (-1, when fewer candidates exist than ndocs) and, under a very selective
        # filter, failing pids (scored -inf) can reach the top-k; drop them.
        return self.passes([x for x in out.tolist() if x >= 0], mask)

    @torch.no_grad()
    def ragrt_profiled(self, Qf, ntok, p):
        """(pids, [s0b, s1, s2, s3, s4] ms). s0b by CUDA events around the prep."""
        self.ensure_calibrated()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        self.cur_mask = 0
        Q_sub, scores, topc = self.ragrt_prep(Qf, ntok, p["nc"])
        e1.record()
        out, st = self.index.search_single_query_profiled(
            Qf.float(), Qf, Q_sub, topc, scores, topc.shape[1], self.ndocs(p), C.TOP_K,
            query_mask=0, k_eids=p["eids"])
        torch.cuda.synchronize()
        return [x for x in out.tolist() if x >= 0], [e0.elapsed_time(e1)] + list(st)

    @torch.no_grad()
    def ragrt_pipelined(self, Qfs, ntoks, p, mask=0):
        """Throughput path: prep every query, then the dual-stream pipelined search."""
        self.ensure_calibrated()
        self.cur_mask = 0
        preps = [self.ragrt_prep(q, n, p["nc"]) for q, n in zip(Qfs, ntoks)]
        out = self.index.search_batch_pipelined(
            [q.float() for q in Qfs], list(Qfs), [x[0] for x in preps], [x[2] for x in preps],
            [x[1] for x in preps], p["nc"], self.ndocs(p), C.TOP_K, query_mask=mask, k_eids=p["eids"])
        return out

    @torch.no_grad()
    def graph_search(self, Qf, ntok, p, mask=0):
        if self.graph is None:
            from graph_search import GraphSearch
            self.graph = GraphSearch(self.index, self.centroids_fp16, C.TOP_K, impute=True)
        return self.passes(self.graph.search(Qf, ntok, p["nc"], self.ndocs(p), p["eids"], mask), mask)

    # ------------------------------------------------------------ MUVERA
    @torch.no_grad()
    def muvera_candidates(self, Qf, p):
        import muvera_fde as M
        m = self.muvera
        qv = M.query_fde(Qf.float(), m["params"], m["cfg"]).to(torch.float16)
        s = m["F"] @ qv
        return torch.topk(s, min(p["cands"], self.N)).indices.to(torch.int32).contiguous()

    @torch.no_grad()
    def muvera_search(self, Qf, p, mask=0):
        """FDE inner product over all passages, top cands, TileMaxSim rerank (post-filter)."""
        cands = self.muvera_candidates(Qf, p)
        if mask:
            cands = cands[(self.pred_gpu[cands.long()] & mask) != 0]
        if cands.numel() == 0:
            return []
        s = self.tms_scores(Qf, cands)
        return cands[torch.topk(s, min(C.TOP_K, cands.numel())).indices].tolist()

    def search(self, engine, Qf, ntok, p, mask=0):
        if engine == "muvera":
            return self.muvera_search(Qf, p, mask)
        if engine == "plaid":
            return self.plaid(Qf, p, mask)
        if engine == "ptms":
            return self.ptms(Qf, p, mask)
        if engine in RAGRT_ENGINES:
            self.use_engine(engine)
            try:
                if engine == "weft_resid_norr":
                    return self.approx_topk(Qf, ntok, self.resolve(engine, p), mask)
                if engine == "weft_graph":
                    return self.graph_search(Qf, ntok, self.resolve(engine, p), mask)
                return self.ragrt(Qf, ntok, self.resolve(engine, p), mask)
            finally:
                self.use_engine("ragrt")
        raise ValueError(engine)


def get_stage3_pids(ranker, config, Q):
    """
    PLAID stages 1-3, the same steps as ColBERT's IndexScorer.score_pids (GPU path):
    candidate generation, pruned centroid interaction -> top ndocs, full centroid
    interaction -> top ndocs // 4. (The pre-Oct-2026 copy skipped the middle cut and
    the full re-score, so PLAID+TMS reranked a different candidate set than PLAID.)
    """
    with torch.inference_mode():
        pids, centroid_scores = ranker.retrieve(config, Q)
        if isinstance(pids, list):
            pids = torch.tensor(pids, dtype=torch.int32, device="cuda")
        batch_size = 2 ** 20
        if centroid_scores is not None and ranker.use_gpu:
            centroid_scores = centroid_scores.cuda()
            idx = centroid_scores.max(-1).values >= config.centroid_score_threshold
            approx_scores = []
            for i in range(0, ceil(len(pids) / batch_size)):
                pids_ = pids[i * batch_size: (i + 1) * batch_size]
                codes_packed, codes_lengths = ranker.embeddings_strided.lookup_codes(pids_)
                idx_ = idx[codes_packed.long()]
                pruned_codes_strided = StridedTensor(idx_, codes_lengths, use_gpu=ranker.use_gpu)
                pruned_codes_padded, pruned_codes_mask = pruned_codes_strided.as_padded_tensor()
                pruned_codes_lengths = (pruned_codes_padded * pruned_codes_mask).sum(dim=1)
                codes_packed_ = codes_packed[idx_]
                approx_scores_ = centroid_scores[codes_packed_.long()]
                if approx_scores_.shape[0] == 0:
                    approx_scores.append(torch.zeros((len(pids_),), dtype=approx_scores_.dtype).cuda())
                    continue
                approx_scores_strided = StridedTensor(approx_scores_, pruned_codes_lengths, use_gpu=ranker.use_gpu)
                approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
                approx_scores_ = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
                approx_scores.append(approx_scores_)
            approx_scores = torch.cat(approx_scores, dim=0)
            if config.ndocs < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=config.ndocs).indices]

            # Re-score the survivors with the full (unpruned) centroid scores, keep ndocs // 4.
            codes_packed, codes_lengths = ranker.embeddings_strided.lookup_codes(pids)
            approx_scores = centroid_scores[codes_packed.long()]
            approx_scores_strided = StridedTensor(approx_scores, codes_lengths, use_gpu=ranker.use_gpu)
            approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
            approx_scores = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
            if config.ndocs // 4 < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=(config.ndocs // 4)).indices]
        return pids
