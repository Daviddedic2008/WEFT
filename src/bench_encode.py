"""
Stage 0a (query encoding): ColBERT's stock encoder vs fast_encoder.py.

Reports, on the first --n evaluation queries:
  latency    per-query median (ms) of stock, fast without graphs, fast with CUDA graphs
  batched    queries/s encoding --batch queries at once, stock vs fast (graphs)
  agreement  per-token cosine between the two encodings; ntok agreement
  retrieval  (with --sparse_csr_dir <plain IVF index>) top-10 / top-100 overlap of WEFT results
             from the two encodings, and recall@10 of each against the exact ground truth

  python3 bench_encode.py --dataset lotte --sparse_csr_dir /local/scratch/a/cashman3/lotte_plain_ivf
Writes <results>/<dataset>/encode_bench.json.
"""
import os, sys, json, time, argparse
import numpy as np
import torch

import eval_config as C
import eval_lib as E
from engines import Engines
from fast_encoder import FastQueryEncoder


def timed(fn, items, warm=10):
    for x in items[:warm]:
        fn(x)
    out = []
    for x in items:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(x); torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sparse_csr_dir", default=None, help="plain-IVF index: also compare WEFT results")
    ap.add_argument("--nc", type=int, default=32)
    ap.add_argument("--ndocs", type=int, default=4096)
    args = ap.parse_args()

    ds = C.dataset(args.dataset)
    if args.sparse_csr_dir:
        ds["sparse_csr_dir"] = args.sparse_csr_dir
        ds["predicates_path"] = os.path.join(args.sparse_csr_dir, "predicates.npy")
    eng = Engines(ds, load_ragrt=bool(args.sparse_csr_dir))
    fast = FastQueryEncoder(eng.searcher.checkpoint)
    fast_eager = FastQueryEncoder(eng.searcher.checkpoint, use_graphs=False)
    qs = E.load_questions(ds["questions_path"])[:args.n]
    texts = [t for _, t in qs]

    R = {"n": len(texts), "gpu": torch.cuda.get_device_name(0)}
    lat = {"stock": timed(eng.encode, texts), "fast_eager": timed(fast_eager.encode, texts),
           "fast_graph": timed(fast.encode, texts)}
    R["latency_ms"] = {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))} for k, v in lat.items()}

    B = args.batch
    batches = [texts[a:a + B] for a in range(0, len(texts) - B + 1, B)]
    def qps(fn):
        fn(batches[0]); torch.cuda.synchronize(); t0 = time.perf_counter()
        for b in batches:
            fn(b)
        torch.cuda.synchronize()
        return len(batches) * B / (time.perf_counter() - t0)
    R["batched_qps"] = {"batch": B, "stock": qps(lambda b: eng.searcher.encode(b)),
                        "fast_graph": qps(fast.encode_batch)}

    cos, ntok_same, enc = [], 0, []
    for t in texts:
        Qs, ns = eng.encode(t)
        Qf, nf = fast.encode(t)
        c = torch.nn.functional.cosine_similarity(Qs.float(), Qf.float(), dim=-1)
        cos.append(float(c.min()))
        ntok_same += int(ns == nf)
        enc.append(((Qs, ns), (Qf, nf)))
    R["agreement"] = {"min_token_cosine_mean": float(np.mean(cos)), "min_token_cosine_worst": float(np.min(cos)),
                      "ntok_identical": ntok_same / len(texts)}

    if args.sparse_csr_dir:
        p = {"nc": args.nc, "eids": "all", "ndocs": args.ndocs}
        o10, o100, r10s, r10f = [], [], [], []
        gt = None
        if os.path.exists(ds["gt_path"]):
            z = np.load(ds["gt_path"])
            gt = {str(q): z["all"][i] for i, q in enumerate(z["qids"])}
        for (q, _), ((Qs, ns), (Qf, nf)) in zip(qs, enc):
            a = eng.search("weft", Qs, ns, p)
            b = eng.search("weft", Qf, nf, p)
            o10.append(len(set(a[:10]) & set(b[:10])) / 10)
            o100.append(len(set(a) & set(b)) / max(1, len(a)))
            if gt is not None and str(q) in gt:
                g = gt[str(q)]
                r10s.append(E.gt_recall(a, g, 10)); r10f.append(E.gt_recall(b, g, 10))
        R["retrieval"] = {"config": p, "top10_overlap": float(np.mean(o10)), "top100_overlap": float(np.mean(o100))}
        if r10s:
            R["retrieval"].update(gt_r10_stock=float(np.mean(r10s)), gt_r10_fast=float(np.mean(r10f)),
                                  note="ground truth was computed with the stock encoder")

    os.makedirs(ds["results_dir"], exist_ok=True)
    out = os.path.join(ds["results_dir"], "encode_bench.json")
    E.dump_json(R, out)
    print(json.dumps(R, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
