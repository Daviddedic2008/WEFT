"""
Query-correlated filters: the filtered-search workloads where the filter is (or is not) about the
same thing as the query. Random filters (syn5 ... syn60) are independent of relevance, which is
the easy case; real filters usually correlate with the query in one direction or the other.

1. Topic clusters. Every passage gets a topic: the passage's mean token vector (its tokens'
   ColBERT centroids, averaged, normalized) assigned to one of K k-means clusters (default 16, so
   about 6% of the corpus each). Cluster c is predicate bit 16 + c (bits 16-31; build_predicates.py
   uses bits 0-12). Written into predicates.npy of every index dir given (bits 16-31 replaced).
2. Per-query filters. A query's mean token vector ranks the clusters by similarity:
     corr_near  pass = the query's closest topic       (positively correlated: the easy case)
     corr_mid   pass = its (K/2)-th closest topic       (unrelated)
     corr_far   pass = its least similar topic          (anti-correlated: the hard case; the
                                                         unfiltered top results almost all fail)
   Saved to <results>/<dataset>/corr_filters.json as {filter: {qid: mask}}, read by
   build_ground_truth.py (exact filtered answers) and run_benchmarks.py (mask per query).
3. Sanity check (when the ground truth exists): the fraction of each query's unfiltered exact
   top-10 that passes each filter. It should be high for corr_near and near zero for corr_far.

Usage (after a run has finished; it rewrites predicates.npy in place, keeping a .bak):
  python3 build_corr_filters.py --dataset lotte --update_dirs <index dir> [<index dir> ...]
"""
import os, sys, json, argparse, shutil
import numpy as np
import torch

import eval_config as C
import eval_lib as E
from engines import Engines
import gpu_kmeans as K

BIT_BASE = 16
FILTERS = ("corr_near", "corr_mid", "corr_far")


@torch.no_grad()
def passage_means(codes, doc_offsets, centroids, chunk=2_000_000):
    """codes int32 [T] (cuda), doc_offsets int64 [N + 1] (cuda), centroids [nc, 128] -> unit means [N, 128] f32.
    Chunked over tokens (2M tokens x 128 x 4 B = 1 GB of gathered centroids per chunk)."""
    N = doc_offsets.numel() - 1
    T = int(doc_offsets[-1])
    Cf = centroids.float()
    out = torch.zeros(N, Cf.shape[1], dtype=torch.float32, device=codes.device)
    ends = doc_offsets[1:].contiguous()
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        tok = torch.arange(t0, t1, device=codes.device, dtype=torch.int64)
        pid = torch.searchsorted(ends, tok, right=True)          # passage of each token
        out.index_add_(0, pid, Cf[codes[t0:t1].long()])
    return torch.nn.functional.normalize(out, dim=-1)


def rank_filters(qvec, centers):
    """qvec [d] unit, centers [K, d] unit -> {filter: cluster id}."""
    order = torch.argsort(centers @ qvec, descending=True).tolist()
    k = len(order)
    return {"corr_near": order[0], "corr_mid": order[k // 2], "corr_far": order[-1]}


def update_predicates(path, cid, k):
    pred = np.load(path).astype(np.uint32)
    assert len(pred) == len(cid), f"{path}: {len(pred):,} rows, expected {len(cid):,}"
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    keep = np.uint32((1 << BIT_BASE) - 1)
    pred = (pred & keep) | (np.uint32(1) << (BIT_BASE + cid.astype(np.uint32)))
    np.save(path, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sample", type=int, default=1_000_000, help="passages used to fit k-means")
    ap.add_argument("--update_dirs", nargs="+", required=True, help="index dirs whose predicates.npy gets the topic bits")
    args = ap.parse_args()
    assert args.k <= 32 - BIT_BASE, f"at most {32 - BIT_BASE} topics fit in bits {BIT_BASE}-31"

    ds = C.dataset(args.dataset)
    eng = Engines(ds, load_ragrt=False)
    N = eng.N
    print(f"[{args.dataset}] {N:,} passages: passage mean vectors", flush=True)
    means = passage_means(eng.codes, eng.doc_offsets, eng.centroids_fp16)
    rng = np.random.default_rng(0)
    samp = rng.choice(N, size=min(args.sample, N), replace=False)
    centers = K.kmeans(means[torch.from_numpy(samp).cuda()].cpu().numpy(), args.k, iters=25, seed=0)
    ct = torch.nn.functional.normalize(torch.from_numpy(centers).cuda(), dim=-1)
    cid, _ = K.assign(means, torch.from_numpy(centers).cuda())
    cid = cid.cpu().numpy().astype(np.int64)
    frac = np.bincount(cid, minlength=args.k) / N
    print("  topic sizes: " + " ".join(f"{f:.1%}" for f in frac), flush=True)
    del means

    for d in args.update_dirs:
        p = os.path.join(d, "predicates.npy")
        update_predicates(p, cid, args.k)
        print(f"  wrote topic bits {BIT_BASE}-{BIT_BASE + args.k - 1} into {p}", flush=True)

    qrels = E.load_qrels(ds["qrels_path"])
    questions = [(q, t) for q, t in E.load_questions(ds["questions_path"]) if q in qrels]
    out = {f: {} for f in FILTERS}
    sel = {f: [] for f in FILTERS}
    for i, (q, text) in enumerate(questions):
        Qf, ntok = eng.encode(text)
        qv = torch.nn.functional.normalize(Qf[:ntok].float().mean(0), dim=-1)
        for f, c in rank_filters(qv, ct).items():
            out[f][q] = int(1 << (BIT_BASE + c))
            sel[f].append(frac[c])
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{len(questions):,} queries", flush=True)
    meta = {"k": args.k, "bit_base": BIT_BASE, "topic_frac": frac.tolist(), "filters": out,
            "mean_selectivity": {f: float(np.mean(v)) for f, v in sel.items()}}
    path = os.path.join(ds["results_dir"], "corr_filters.json")
    os.makedirs(ds["results_dir"], exist_ok=True)
    json.dump(meta, open(path, "w"))
    print(f"Saved {path}: " + ", ".join(f"{f} mean selectivity {meta['mean_selectivity'][f]:.1%}" for f in FILTERS))

    if os.path.exists(ds["gt_path"]):
        z = np.load(ds["gt_path"])
        gq = {str(q): i for i, q in enumerate(z["qids"])}
        for f in FILTERS:
            share = []
            for q, m in out[f].items():
                if q in gq:
                    top = z["all"][gq[q]][:10]
                    top = top[top >= 0]
                    if len(top):
                        share.append(np.mean((np.uint32(1) << (BIT_BASE + cid[top]).astype(np.uint32)) & m != 0))
            print(f"  {f}: {np.mean(share):.1%} of the unfiltered exact top-10 passes (over {len(share):,} queries)")


if __name__ == "__main__":
    main()
