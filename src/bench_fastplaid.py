"""
FastPLAID baseline (LightOn, GPU PLAID in Rust/CUDA) on the same split, ground truth and
metrics as run_benchmarks.py. Runs in FastPLAID's own environment (it pins its own torch):
it imports only eval_config / eval_lib from this repo, never ColBERT or our extension.

Inputs come from export_embeddings.py (same doc vectors the ground truth used, same encoded
queries). FastPLAID builds its own centroids and residual codes (--nbits); all 32 query rows
are used, as ColBERT scores.

Stages:
  build      index the exported embeddings (once)
  sweep      n_ivf_probe x n_full_scores on TUNE: median latency, exact R@10/R@100, official metric
  select     fastest TUNE config reaching each target (same targets as the main run)
  test       selected configs on TEST, median of --repeats per query, 95% bootstrap CIs
  filtered   synthetic and own-domain filters via FastPLAID's `subset`, judged on the filtered GT
  throughput one batched search over --throughput_queries queries
  table      rewrite comparison.md from saved results (no searching)
Results: <results>/<dataset>/fastplaid_nbits<b>/results.json, plus a side-by-side table with
the main run (--compare <results subdir>, e.g. v2_E256).

  python3 bench_fastplaid.py --dataset lotte --emb_dir /local/scratch/a/cashman3/lotte_doc_embs \
      --index_dir /local/scratch/a/cashman3/fastplaid_lotte_nbits2 --nbits 2 --compare v2_E256
"""
import os, sys, json, time, argparse
import numpy as np
import torch

import eval_config as C
import eval_lib as E

PROBES = [1, 2, 4, 8, 16, 32, 64]
FULL = [256, 512, 1024, 2048, 4096, 8192, 16384]
WARMUP = 5


def key(p):
    return f"probe={p['n_ivf_probe']} full={p['n_full_scores']}"


def summarize(lat, r10, r100, off):
    s = {"lat": E.latency_summary(lat)}
    for k, v in (("gt_r10", r10), ("gt_r100", r100), ("official", off)):
        m, lo, hi = E.bootstrap_ci(np.asarray(v, dtype=float))
        s[k] = {"mean": m, "ci95": [lo, hi]}
    return s


class FP:
    def __init__(self, args):
        self.args = args
        from fast_plaid import search as fps
        self.fps = fps
        self.ds = C.dataset(args.dataset)
        self.out_dir = os.path.join(self.ds["results_dir"], f"fastplaid_nbits{args.nbits}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.json_path = os.path.join(self.out_dir, "results.json")
        self.R = json.load(open(self.json_path)) if os.path.exists(self.json_path) else {}
        self.R["meta"] = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu": torch.cuda.get_device_name(0),
                          "torch": torch.__version__, "args": vars(args),
                          "fast_plaid": getattr(__import__("fast_plaid"), "__version__", "?")}
        self.index = None

    def save(self):
        E.dump_json(self.R, self.json_path)

    # -------------------------------------------------------------- data
    def load_eval(self):
        a = self.args
        qd = torch.load(os.path.join(a.emb_dir, "queries.pt"))
        self.Q = {q: qd["Q"][i].float() for i, q in enumerate(qd["qids"])}
        z = np.load(self.ds["gt_path"])
        gt_qids = [str(q) for q in z["qids"]]
        self.gt_names = [k for k in z.files if k not in ("qids", "done", "scores_all")]
        self.gt = {n: {q: z[n][i].tolist() for i, q in enumerate(gt_qids)} for n in self.gt_names}
        self.qrels = E.load_qrels(self.ds["qrels_path"])
        qids = [q for q in gt_qids if q in self.qrels]
        self.tune, self.test = E.make_split(qids, a.tune_size, C.SPLIT_SEED)
        if a.max_test:
            self.test = self.test[:a.max_test]
        self.R["split"] = {"tune": len(self.tune), "test": len(self.test), "seed": C.SPLIT_SEED, "total_with_gt": len(qids)}
        self.meta = json.load(open(self.ds["queries_meta_path"])) if self.ds["has_domains"] else {}
        print(f"[{a.dataset}] tune {len(self.tune):,}, test {len(self.test):,}", flush=True)

    def open_index(self):
        if self.index is None:
            kw = {} if self.args.index_gpu_memory == "default" else {"index_gpu_memory": self.args.index_gpu_memory}
            try:
                self.index = self.fps.FastPlaid(index=self.args.index_dir, device="cuda:0", **kw)
            except TypeError:  # older fast-plaid without the placement option
                print("  note: this fast-plaid has no index_gpu_memory option; using its default placement")
                self.index = self.fps.FastPlaid(index=self.args.index_dir, device="cuda:0")
            self.R["meta"]["index_gpu_memory"] = kw.get("index_gpu_memory", "default")
        return self.index

    # -------------------------------------------------------------- stages
    def stage_build(self):
        a = self.args
        if os.path.isdir(a.index_dir) and os.listdir(a.index_dir):
            print(f"  index exists in {a.index_dir}; delete it to rebuild")
            return
        embs = np.load(os.path.join(a.emb_dir, "doc_embs_fp16.npy"), mmap_mode="r")
        offs = np.load(os.path.join(a.emb_dir, "doc_offsets.npy"))
        # Views into the memory map: pages are read on demand, nothing is copied up front.
        docs = [torch.from_numpy(np.asarray(embs[offs[i]:offs[i + 1]])) for i in range(len(offs) - 1)]
        print(f"  building FastPLAID (nbits={a.nbits}) over {len(docs):,} passages, {int(offs[-1]):,} tokens", flush=True)
        t0 = time.time()
        self.fps.FastPlaid(index=a.index_dir, device="cuda:0").create(documents_embeddings=docs, nbits=a.nbits)
        self.R["build_seconds"] = round(time.time() - t0, 1)
        print(f"  built in {self.R['build_seconds']} s", flush=True)

    def run(self, q, p, subset=None):
        res = self.index.search(queries_embeddings=[self.Q[q]], top_k=C.TOP_K, n_ivf_probe=p["n_ivf_probe"],
                                n_full_scores=p["n_full_scores"], show_progress=False, subset=subset)
        return [int(d) for d, _ in res[0]]

    def evaluate(self, p, qids, repeats=1, filt="all", subsets=None):
        idx = self.open_index()
        sub = (lambda q: subsets(q)) if subsets else (lambda q: None)
        for q in qids[:WARMUP]:
            self.run(q, p, sub(q))
        lat, r10, r100, off = [], [], [], []
        for q in qids:
            ts = []
            for _ in range(repeats):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ranked = self.run(q, p, sub(q))
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            lat.append(float(np.median(ts)))
            gt = self.gt[filt][q]
            r10.append(E.gt_recall(ranked, gt, 10)); r100.append(E.gt_recall(ranked, gt, 100))
            off.append(E.official_metric(self.ds["official_metric"], ranked, self.qrels[q])
                       if filt in ("all", "own_domain") else float("nan"))
        return lat, r10, r100, off

    def stage_sweep(self):
        pts = []
        grid = [{"n_ivf_probe": a, "n_full_scores": b} for a in PROBES for b in FULL]
        t0 = time.time()
        for i, p in enumerate(grid):
            lat, r10, r100, off = self.evaluate(p, self.tune)
            pts.append({"params": p, "key": key(p), "lat_median": float(np.median(lat)),
                        "lat_p95": float(np.percentile(lat, 95)), "gt_r10": float(np.nanmean(r10)),
                        "gt_r100": float(np.nanmean(r100)), "official": float(np.nanmean(off))})
            print(f"    {key(p):<24} {pts[-1]['lat_median']:6.2f} ms  R@10 {pts[-1]['gt_r10']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        self.R["sweep"] = {"fastplaid": pts}

    def compare_results(self):
        if not self.args.compare:
            return None
        p = os.path.join(self.ds["results_dir"], self.args.compare, "results.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def stage_select(self):
        targets = {f"gt_r10>={t:.2f}": t for t in self.args.targets}
        M = self.compare_results()
        if M and "plaid_default" in M.get("selection", {}):
            targets["plaid_default"] = M["selection"]["plaid_default"]["target_gt_r10"]
        sel = {}
        for name, t in targets.items():
            pt = E.select_fastest_at(self.R["sweep"]["fastplaid"], t)
            sel[name] = {"target_gt_r10": t,
                         "fastplaid": None if pt is None else {"params": pt["params"], "key": pt["key"],
                                                               "tune_lat_median": pt["lat_median"], "tune_gt_r10": pt["gt_r10"]}}
            print(f"    {name:<16} {sel[name]['fastplaid']['key'] if sel[name]['fastplaid'] else 'unreachable'}")
        self.R["selection"] = sel

    def stage_test(self):
        test, perq = {}, {}
        seen = {}
        for s in self.R["selection"].values():
            if s["fastplaid"]:
                seen[s["fastplaid"]["key"]] = s["fastplaid"]["params"]
        for k, p in seen.items():
            lat, r10, r100, off = self.evaluate(p, self.test, repeats=self.args.repeats)
            test[f"fastplaid|{k}"] = {"engine": "fastplaid", "params": p, **summarize(lat, r10, r100, off)}
            for n, v in (("lat", lat), ("gt_r10", r10), ("gt_r100", r100), ("official", off)):
                perq[f"fastplaid|{k}|{n}"] = np.asarray(v)
            t = test[f"fastplaid|{k}"]
            print(f"    {k:<24} lat {t['lat']['median']:6.2f} ms  R@10 {t['gt_r10']['mean']:.4f}  "
                  f"{self.ds['official_metric']} {t['official']['mean']:.4f}", flush=True)
        perq["test_qids"] = np.array(self.test)
        np.savez(os.path.join(self.out_dir, "test_perquery.npz"), **perq)
        self.R["test"] = test
        self.print_comparison()

    def print_comparison(self):
        M = self.compare_results()
        om = {"success@5": "S@5", "mrr@10": "MRR@10"}.get(str(self.ds["official_metric"]).lower(), self.ds["official_metric"])
        rows = [f"Cells: median latency (exact R@10 / {om} vs qrels), TEST split, one query at a time.", "",
                "| target | FastPLAID | RAGRT (RT) | RAGRT (CUDA-core Stage 1) | PLAID+TMS | PLAID |", "|---|---|---|---|---|---|"]

        def f(r):
            if not r:
                return "unreachable"
            off = r.get("official", {}).get("mean")
            return f"{r['lat']['median']:.2f} ms ({r['gt_r10']['mean']:.3f} / {off:.3f})" if off is not None else \
                   f"{r['lat']['median']:.2f} ms ({r['gt_r10']['mean']:.3f})"
        for name, s in self.R["selection"].items():
            mine = self.R["test"].get(f"fastplaid|{s['fastplaid']['key']}") if s["fastplaid"] else None
            cells = []
            for e in ("ragrt", "ragrt_bf", "ptms", "plaid"):
                ms = (M or {}).get("selection", {}).get(name, {}).get(e)
                cells.append(f((M or {}).get("test", {}).get(f"{e}|{ms['key']}")) if ms else "n/a")
            rows.append(f"| {name} | {f(mine)} | " + " | ".join(cells) + " |")
        txt = "\n".join(rows)
        open(os.path.join(self.out_dir, "comparison.md"), "w").write(txt + "\n")
        print("\n" + txt)

    def stage_filtered(self):
        pred = np.load(self.ds["predicates_path"]).astype(np.uint32)
        masks = dict(("syn" + l.rstrip("%"), b) for l, b in C.SYNTHETIC_MASKS)
        qids = self.test[:self.args.max_filter_queries]
        name = self.primary()
        sel = self.R["selection"][name]["fastplaid"]
        out = self.R.setdefault("filtered", {})
        self.R["filtered_target"] = name
        for filt in [g for g in self.gt_names if g != "all"]:
            if filt == "own_domain":
                lists = {d: np.flatnonzero(pred & np.uint32(1 << (C.DOMAIN_BIT_BASE + i))).tolist()
                         for i, d in enumerate(C.LOTTE_DOMAINS)}
                subsets = lambda q, L=lists: L[self.meta[q]["domain"]]
            else:
                allowed = np.flatnonzero(pred & np.uint32(masks[filt])).tolist()
                subsets = lambda q, A=allowed: A
            lat, r10, r100, off = self.evaluate(sel["params"], qids, filt=filt, subsets=subsets)
            out[filt] = summarize(lat, r10, r100, off)
            print(f"    filter {filt:<10} {out[filt]['lat']['median']:.2f} ms  R@10 {out[filt]['gt_r10']['mean']:.3f}", flush=True)

    def stage_table(self):
        self.print_comparison()

    def primary(self):
        """The main run's primary target if FastPLAID reaches it, else the highest target it reaches."""
        sel = self.R["selection"]
        M = self.compare_results()
        if M and sel.get(M.get("primary_target"), {}).get("fastplaid"):
            return M["primary_target"]
        reach = [(s["target_gt_r10"], n) for n, s in sel.items() if s["fastplaid"]]
        if not reach:
            sys.exit("FATAL: FastPLAID reaches none of the targets")
        name = max(reach)[1]
        print(f"  using {name} (the main run's primary target is unreachable for FastPLAID)")
        return name

    def stage_throughput(self):
        idx = self.open_index()
        p = self.R["selection"][self.primary()]["fastplaid"]["params"]
        qs = [self.Q[q] for q in self.test[:self.args.throughput_queries]]
        idx.search(queries_embeddings=qs[:WARMUP], top_k=C.TOP_K, show_progress=False, **p)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        idx.search(queries_embeddings=qs, top_k=C.TOP_K, show_progress=False, **p)
        torch.cuda.synchronize()
        self.R["throughput"] = {"n_queries": len(qs), "batched_qps": len(qs) / (time.perf_counter() - t0), "params": p}
        print(f"  throughput {self.R['throughput']['batched_qps']:.0f} queries/s (one batched call, {key(p)})", flush=True)
        M = self.compare_results() or {}
        if "throughput" in M:
            print(f"  main run throughput section: {json.dumps(M['throughput'])[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--nbits", type=int, default=2, help="FastPLAID residual bits (ColBERTv2 index: 2)")
    ap.add_argument("--stages", default="build,sweep,select,test,filtered,throughput")
    ap.add_argument("--tune_size", type=int, default=C.TUNE_SIZE)
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max_filter_queries", type=int, default=2000)
    ap.add_argument("--throughput_queries", type=int, default=1024)
    ap.add_argument("--targets", type=lambda s: [float(x) for x in s.split(",")], default=[0.80, 0.85, 0.90, 0.95, 0.98])
    ap.add_argument("--index_gpu_memory", default="high", choices=["high", "auto", "medium", "low", "default"],
                    help="FastPLAID placement of codes/residuals at query time; 'high' keeps both on the GPU")
    ap.add_argument("--compare", default=None, help="results subdir of the main run to compare against, e.g. v2_E256")
    args = ap.parse_args()
    b = FP(args)
    stages = args.stages.split(",")
    if any(s != "build" for s in stages):
        b.load_eval()
    for st in stages:
        print(f"\n== stage {st}", flush=True)
        t0 = time.time()
        getattr(b, f"stage_{st}")()
        b.save()
        print(f"   ({time.time() - t0:.0f} s, saved {b.json_path})", flush=True)


if __name__ == "__main__":
    main()
