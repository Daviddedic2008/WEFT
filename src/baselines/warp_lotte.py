"""
WARP (XTR) baseline on LoTTE pooled dev, run inside a checkout of github.com/jlscheerer/xtr-warp
(run_warp.sh sets that up). Two modes:

  index   build the WARP index for lotte/pooled/dev (google/xtr-base-en, nranks=1, one GPU)
  search  for each nprobe in --nprobe: per-query latency (single query, CPU, --threads threads)
          and Success@5 on the pooled dev *search* queries, as WARP's paper reports; rankings
          are saved so RAGRT/PLAID can be compared on the same queries (qid text matched later).

Writes <out>/warp_lotte_pooled.json.
"""
import os, sys, json, time, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["index", "search"])
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--nprobe", default="8,16,32")
    ap.add_argument("--threads", default="1,16")
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))          # the .env in the xtr-warp checkout (cwd), not next to this file
    missing = [k for k in ("INDEX_ROOT", "EXPERIMENT_ROOT", "LOTTE_COLLECTION_PATH", "BEIR_COLLECTION_PATH")
               if k not in os.environ]
    if missing:
        sys.exit(f"FATAL: {missing} not set (run_warp.sh exports them; or run from the xtr-warp checkout)")
    from warp.engine.config import WARPRunConfig

    base = dict(collection="lotte", dataset="pooled", type_="search", datasplit="dev", nbits=args.nbits, k=100)
    if args.mode == "index":
        from warp.engine.utils.collection_indexer import index
        from warp.engine.utils.index_converter import convert_index
        cfg = WARPRunConfig(nranks=1, **base)
        t0 = time.time()
        index(cfg)
        convert_index(os.path.join(cfg.index_root, cfg.index_name))
        print(f"[DONE] WARP index {cfg.index_name} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return

    import torch
    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "warp_lotte_pooled.json")
    R = json.load(open(path)) if os.path.exists(path) else {"runs": {}}
    for threads in [int(x) for x in args.threads.split(",")]:
        torch.set_num_threads(threads)
        for nprobe in [int(x) for x in args.nprobe.split(",")]:
            cfg = WARPRunConfig(nranks=1, nprobe=nprobe, fused_ext=threads != 1, **base)
            searcher = WARPSearcher(cfg)
            queries = WARPQueries(cfg)
            items = list(queries)
            if args.max_queries:
                items = items[:args.max_queries]
            for qid, text in items[:10]:                                  # warm-up
                searcher.search(text, k=100)
            lat, ranks = [], {}
            for qid, text in items:
                t0 = time.perf_counter()
                res = searcher.search(text, k=100)
                lat.append((time.perf_counter() - t0) * 1000)
                pids = res[0] if isinstance(res, tuple) else res
                ranks[str(qid)] = [int(p) for p in list(pids)[:100]]
            qas = queries.qrels.qas.data
            succ = [any(p in qas.get(int(q), set()) for p in r[:5]) for q, r in ranks.items()]
            lat.sort()
            key = f"threads={threads} nprobe={nprobe}"
            R["runs"][key] = {"threads": threads, "nprobe": nprobe, "nbits": args.nbits, "n_queries": len(items),
                              "lat_median_ms": lat[len(lat) // 2], "lat_p95_ms": lat[int(0.95 * len(lat))],
                              "success@5": sum(succ) / max(1, len(succ)),
                              "note": "includes query encoding (XTR, CPU), as WARP reports end-to-end latency"}
            R.setdefault("rankings", {})[key] = ranks
            R["queries"] = {str(q): t for q, t in items}
            json.dump(R, open(path, "w"))
            print(f"  WARP {key}: median {R['runs'][key]['lat_median_ms']:.1f} ms, success@5 "
                  f"{R['runs'][key]['success@5']:.4f} over {len(items)} queries", flush=True)
    print(f"[DONE] {path}")


if __name__ == "__main__":
    main()
