"""
Exact late-interaction ground truth: for every evaluation query, score ALL passages
with the fused TileMaxSim kernel (all 32 query rows, 2-bit residuals decompressed,
the same scorer every engine reranks with) and keep the top GT_DEPTH pids.

Stored per filter, so filtered runs are judged against the exact filtered answer:
  all            no filter
  syn5 ... syn60 synthetic selectivity bits (build_predicates.py)
  own_domain     LoTTE only: each query restricted to its own domain
  corr_near/mid/far  query-correlated topic filters (build_corr_filters.py), if corr_filters.json exists

Output: <results>/<dataset>/ground_truth.npz with qids [n] and, per filter, pids
[n, GT_DEPTH] (int32, -1 padded) plus the unfiltered scores [n, GT_DEPTH].
Resumable: progress is checkpointed every --save_every queries.

Cost: one full-corpus TileMaxSim pass per query (about 0.5 s on MS MARCO, 0.15 s on
LoTTE on an RTX 6000 Ada), so run it once per dataset inside tmux.

Usage: python3 build_ground_truth.py --dataset msmarco [--limit N]
       python3 build_ground_truth.py --dataset lotte --merge    # only filters the file lacks
"""
import os, sys, json, time, argparse
import numpy as np
import torch

import eval_config as C
import eval_lib as E
from engines import Engines

CHUNK = 2_000_000


def filter_masks(ds, qids, meta):
    """{name: mask or callable(qid)->mask}."""
    masks = {"all": 0}
    for lbl, bit in C.SYNTHETIC_MASKS:
        masks["syn" + lbl.rstrip("%")] = bit
    if ds["has_domains"]:
        masks["own_domain"] = lambda q: 1 << (C.DOMAIN_BIT_BASE + C.LOTTE_DOMAINS.index(meta[q]["domain"]))
    corr = load_corr_filters(ds)
    for name, table in corr.items():
        masks[name] = lambda q, t=table: t[q]
    return masks


def load_corr_filters(ds):
    """{filter name: {qid: mask}} from build_corr_filters.py, or {}."""
    p = os.path.join(ds["results_dir"], "corr_filters.json")
    return json.load(open(p))["filters"] if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (smoke test)")
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--merge", action="store_true",
                    help="keep an existing ground truth file and only add the filters it lacks")
    args = ap.parse_args()

    ds = C.dataset(args.dataset)
    os.makedirs(ds["results_dir"], exist_ok=True)
    out_path = ds["gt_path"] if args.limit is None else ds["gt_path"].replace(".npz", f".limit{args.limit}.npz")
    part_path = out_path + ".partial.npz"

    qrels = E.load_qrels(ds["qrels_path"])
    questions = [(q, t) for q, t in E.load_questions(ds["questions_path"]) if q in qrels]
    if args.limit:
        questions = questions[:args.limit]
    meta = json.load(open(ds["queries_meta_path"])) if ds["has_domains"] else {}
    qids = [q for q, _ in questions]

    eng = Engines(ds, load_ragrt=False)
    N, K = eng.N, C.GT_DEPTH
    masks = filter_masks(ds, qids, meta)
    pass_cache = {}

    def passes(m):
        if m not in pass_cache:
            pass_cache[m] = (eng.pred_gpu & m) != 0
        return pass_cache[m]

    old = {}
    if args.merge and os.path.exists(out_path):
        z = np.load(out_path)
        assert [str(q) for q in z["qids"]] == qids, "existing ground truth is for a different query set"
        old = {k: z[k] for k in z.files if k not in ("qids", "done")}
        masks = {n: m for n, m in masks.items() if n not in old}
        if not masks:
            print(f"{out_path} already has every filter"); return
        print(f"Merging new filters {sorted(masks)} into {out_path}")
        part_path = out_path + ".merge.partial.npz"
    res = {name: np.full((len(qids), K), -1, dtype=np.int32) for name in masks}
    top_scores = old["scores_all"] if old else np.full((len(qids), K), np.nan, dtype=np.float32)
    done = 0
    if os.path.exists(part_path):
        z = np.load(part_path, allow_pickle=False)
        if list(z["qids"]) == qids:
            done = int(z["done"])
            for name in masks:
                res[name] = z[name]
            top_scores = z["scores_all"]
            print(f"Resuming at query {done:,}/{len(qids):,}")

    dev = getattr(eng, "device", "cuda")
    all_pids = torch.arange(N, dtype=torch.int32, device=dev)
    t0 = time.time()

    def save(path, n_done):
        keep = {k: v for k, v in old.items() if k != "scores_all"}
        np.savez(path, qids=np.array(qids), done=n_done, scores_all=top_scores, **keep, **res)

    for i in range(done, len(qids)):
        Qf, _ = eng.encode(questions[i][1])
        with torch.no_grad():
            scores = torch.empty(N, dtype=torch.float32, device=dev)
            for a in range(0, N, CHUNK):
                scores[a:a + CHUNK] = eng.tms_scores(Qf, all_pids[a:a + CHUNK].contiguous())
            for name, m in masks.items():
                mv = m(qids[i]) if callable(m) else m
                s = scores if mv == 0 else torch.where(passes(mv), scores, torch.full_like(scores, float("-inf")))
                v, idx = torch.topk(s, K)
                keep = torch.isfinite(v)
                row = torch.where(keep, idx.to(torch.int32), torch.full_like(idx, -1, dtype=torch.int32))
                res[name][i] = row.cpu().numpy()
                if name == "all":
                    top_scores[i] = v.float().cpu().numpy()
        if (i + 1) % args.save_every == 0:
            save(part_path, i + 1)
            rate = (i + 1 - done) / (time.time() - t0)
            print(f"  {i + 1:,}/{len(qids):,} queries  ({rate:.2f} q/s, "
                  f"{(len(qids) - i - 1) / max(rate, 1e-9) / 60:.0f} min left)", flush=True)

    save(out_path, len(qids))
    if os.path.exists(part_path):
        os.remove(part_path)
    print(f"Saved {out_path}: {len(qids):,} queries x {len(masks)} filters x top-{K}")


if __name__ == "__main__":
    main()
