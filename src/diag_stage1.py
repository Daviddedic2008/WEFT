"""
Does Stage 1 return the true top-k codewords, and what does it cost? (real queries, GPU)

For TUNE queries, per ray (query token x subspace) and per k:
  exact   : top-k positive codewords by q.c (brute force, exact)
  polar   : top-k of the polar RT scene's hits (exact-threshold geometry, any-hit)
  polar_iter: k nearest-hit traversals of the same scene (no hit shaders)
  fan     : top-k of the legacy fan scene's hits
Reports the recall of the exact top-k (mean, and the share of rays with all k), hits per
ray (the any-hit work), and the median Stage-1 time of each variant at a typical config.

  python3 diag_stage1.py --dataset lotte [--sparse_csr_dir DIR] [--n_queries 200]
Writes <results>/<dataset>/diag_stage1[_<tag>].json
"""
import os, json, argparse
import numpy as np
import torch

import eval_config as C
import eval_lib as E
from engines import Engines
import rtrag_corr_3d as X


def topk_hits(c, v, n, k):
    mh = c.shape[1]
    valid = torch.arange(mh, device=c.device)[None, :] < n.clamp(max=mh)[:, None]
    v = torch.where(valid, v, torch.full_like(v, float("-inf")))
    top = v.topk(min(k, mh), dim=1)
    ids = c.gather(1, top.indices)
    return torch.where(torch.isfinite(top.values), ids, torch.full_like(ids, -1))


def recall(got, exact):
    """per-ray |got & exact| / |exact| (rays with an empty exact set count as 1)."""
    out = []
    for g, e in zip(got.tolist(), exact.tolist()):
        es = {x for x in e if x >= 0}
        out.append(1.0 if not es else len(es & {x for x in g if x >= 0}) / len(es))
    return np.array(out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--sparse_csr_dir", default=None)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--ks", default="4,8,16,32,64")
    ap.add_argument("--rt_quantile", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    ap.add_argument("--timing_eids", type=int, default=16)
    args = ap.parse_args()
    ds = C.dataset(args.dataset)
    if args.sparse_csr_dir:
        ds["sparse_csr_dir"] = args.sparse_csr_dir
        ds["predicates_path"] = os.path.join(args.sparse_csr_dir, "predicates.npy")
    eng = Engines(ds, geometry="polar", rt_quantile=args.rt_quantile)
    qs = E.load_questions(ds["questions_path"])
    tune, _ = E.make_split([q for q, _ in qs], C.TUNE_SIZE, C.SPLIT_SEED)
    text = dict(qs)
    calib = tune[:300]
    eng.calibrate([eng.encode(text[q]) for q in calib])
    evalq = tune[300:300 + args.n_queries] if len(tune) > 300 + 20 else tune[:args.n_queries]
    bank = [eng.encode(text[q]) for q in evalq]
    ks = [int(k) for k in args.ks.split(",")]
    cw = eng.CB.reshape(-1, 3).contiguous()
    idx = eng.index
    out = {"E": eng.E, "n_queries": len(bank), "calibration": eng.calibration, "recall": {}, "timing_ms": {}}

    for k in ks:
        rec = {"polar": [], "polar_iter": [], "fan": []}
        hits = {"polar": [], "polar_iter": [], "fan": []}
        for Qf, n in bank:
            Q_sub, _, _ = eng.ragrt_prep(Qf, n, 1)
            qp = Q_sub.reshape(-1, 3).contiguous()
            ex_c, _, _ = X.bruteforce_stage1(qp, cw, k, 256)
            exact = ex_c[:, :k]
            for geom, g in (("polar", 1), ("polar_iter", 2), ("fan", 0)):
                idx.set_geometry(g)
                c, v, cnt = idx.stage1_hits(Q_sub, k)
                rec[geom].append(recall(topk_hits(c, v, cnt, k), exact))
                hits[geom].append(cnt.float().cpu().numpy())
        idx.set_geometry(1)
        out["recall"][str(k)] = {}
        for geom in rec:
            r = np.concatenate(rec[geom]); h = np.concatenate(hits[geom])
            out["recall"][str(k)][geom] = {"mean_recall_of_exact_topk": float(r.mean()),
                                           "frac_rays_all_k": float((r == 1).mean()),
                                           "hits_per_ray_mean": float(h.mean()),
                                           "hits_per_ray_p95": float(np.percentile(h, 95))}
            print(f"  k={k:<3} {geom:<5}: recall of exact top-k {r.mean():.4f} (all k on {np.mean(r == 1):.1%} of rays), "
                  f"hits/ray mean {h.mean():.1f} p95 {np.percentile(h, 95):.0f} of E={eng.E}", flush=True)

    # Stage-1 time at a typical config (median over queries), from the profiled search.
    p = {"nc": 16, "eids": args.timing_eids, "ndocs": 4096}
    variants = [("polar_rt", 1, False), ("polar_iter_rt", 2, False), ("polar_cuda_threshold", 1, True),
                ("fan_rt", 0, False), ("exact_topk_cuda", 0, True)]
    for name, g, bf in variants:
        idx.set_geometry(g); eng.stage1(bf)
        ts = []
        for i, (Qf, n) in enumerate(bank):
            _, st = eng.ragrt_profiled(Qf, n, p)
            if i >= 5:
                ts.append(st[1])
        out["timing_ms"][name] = float(np.median(ts))
        print(f"  stage 1 {name:<22} median {np.median(ts):.3f} ms  (nc=16 eids={args.timing_eids} ndocs=4096)", flush=True)
    idx.set_geometry(1); eng.stage1(False)

    os.makedirs(ds["results_dir"], exist_ok=True)
    path = os.path.join(ds["results_dir"], f"diag_stage1{('_' + args.tag) if args.tag else ''}.json")
    E.dump_json(out, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
