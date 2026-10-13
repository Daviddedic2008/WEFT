"""
RAGRT benchmark harness (replaces generate_figures.py).

Methodology, in order:
  1. Queries: the full evaluation set (MS MARCO dev small; LoTTE pooled dev, all
     domains, search + forum), split by a seeded hash into TUNE (default 1000) and
     TEST (the rest). Configs are only ever chosen on TUNE; every reported number is TEST.
  2. Quality: official metric (MS MARCO MRR@10, LoTTE Success@5) from qrels, plus
     recall of the exact MaxSim top-10/top-100 (build_ground_truth.py), with 95%
     bootstrap CIs. Filtered runs are judged against the exact *filtered* top-k.
  3. Latency: sequential, one query at a time, synchronized. Stage 0a (BERT query
     encoding) is common to all engines and reported separately; RAGRT's own prep
     (projection + centroid top-nc) is inside its retrieval time. Throughput is a
     separate measurement and never reported as latency.
  4. Operating points: for each quality target (exact recall@10 on TUNE, plus the
     quality of PLAID's own k=100 default), the fastest TUNE config of each engine
     that reaches it, then evaluated on TEST.

Stages (each saved to results.json, so they can be rerun independently):
  sweep, select, test, breakdown, filtered, throughput, drops, attribution
  attribution: torch.profiler pass per engine at the primary point: wall ms vs GPU-busy ms,
  kernels, memcpys and host syncs per query (where the time goes: kernels or host overhead).

Ablation engines (engines.py): ragrt_bf (no RT), ragrt_nofilter (codebook filter off),
ragrt_centroid (no codeword score term), ragrt_ivf (plain IVF on the RAGRT kernels; needs a
build_plain_ivf.py index via --sparse_csr_dir), muvera (--muvera_dir). The "attribution"
block of each comparison lists latency ratios and paired quality differences for the
ablation chain ptms -> ragrt_ivf -> ragrt_nofilter / ragrt_centroid -> ragrt_bf -> ragrt.
Residual-correction ablation (weft_tok, weft_resid, weft_resid_norr): needs a token-level index
(build_plain_ivf.py --tokens) via --sparse_csr_dir; run_attribution.sh step "resid".

RAGRT options (see engines.py): --geometry polar|fan, --stage3 sparse|dense,
--rerank wmma|simt (also used by ptms), --rt_quantile, --sparse_csr_dir (another
index, e.g. a different codebook size E). --baseline_from <results dir> imports the
plaid/ptms results of an earlier run on the same split instead of rerunning them
(they do not depend on the RAGRT index), e.g. for the E sweep.

Usage:
  python3 run_benchmarks.py --dataset lotte                      # everything
  python3 run_benchmarks.py --dataset msmarco --stages test,breakdown
  python3 run_benchmarks.py --dataset lotte --quick --max_test 200   # smoke test
Then: python3 plot_results.py --dataset lotte
"""
import os, sys, json, time, argparse, subprocess, datetime
import numpy as np
import torch

import eval_config as C
import eval_lib as E
from engines import Engines, cfg_key, RAGRT_ENGINES, RESID_ENGINES, GRAPH_ENGINES, ADAPT_ENGINES, ORACLE_ENGINES
PERQ_PREP = ADAPT_ENGINES + ORACLE_ENGINES      # per-query prep state: no batched pipeline
import rtrag_corr_3d

CORE = ["plaid", "ptms", "ragrt"]          # the primary operating point must be reachable by these
# ablations: no-RT Stage 1; codebook filter off; no codeword score; plain IVF on RAGRT kernels; MUVERA
ENGINES = CORE + ["ragrt_bf", "ragrt_nofilter", "ragrt_centroid", "ragrt_ivf", "weft_noimpute", "weft",
                  "weft_tok", "weft_resid", "weft_resid_norr", "weft_graph",
                  "weft_hash", "weft_adapt", "weft_pairs", "weft_oracle", "weft_oracle_adapt", "weft_oimp", "weft_oskip", "muvera"]
# engines identical to another when unfiltered: the filtered stage runs them at that engine's selected config
PARAMS_FROM = {"weft_oracle": "weft", "weft_oracle_adapt": "weft_adapt", "weft_oimp": "weft", "weft_oskip": "weft"}
# filtered stage: paired recall@10 differences (b - a, same queries) for these (a, b) when both ran in this run
FILTER_PAIRS = [("weft", "weft_adapt"), ("weft", "weft_oracle"), ("weft_adapt", "weft_oracle_adapt"),
                ("weft", "weft_oimp"), ("weft", "weft_oskip")]
DIAG_QUERIES = 200        # untimed pass per filter that logs the probed depth of filter-aware engines
# (slow, fast, what the step from slow to fast changes); ratio = latency(slow) / latency(fast)
ATTRIBUTION = [
    ("ptms", "ragrt_ivf", "engine: PLAID candidate generation -> same algorithm on RAGRT kernels"),
    ("ragrt_ivf", "ragrt_nofilter", "subspace codeword term in the score (every list)"),
    ("ragrt_nofilter", "ragrt_bf", "codebook filter (fetch only top-eids lists)"),
    ("ragrt_centroid", "ragrt_bf", "codeword term in the score (same lists)"),
    ("ragrt_ivf", "ragrt_bf", "subspace codebooks overall (score + filter)"),
    ("ragrt_bf", "ragrt", "RT cores for Stage 1"),
    ("ptms", "ragrt_bf", "everything except RT"),
    ("ragrt_ivf", "weft_noimpute", "native 128D fp16 centroid scores (no SVD projection)"),
    ("weft_noimpute", "weft", "missing-score imputation"),
    ("ptms", "weft", "WEFT vs PLAID+TMS"),
    ("ragrt", "weft", "WEFT vs the original RT design"),
    ("muvera", "ragrt_bf", "RAGRT (no RT) vs MUVERA"),
    ("weft", "weft_tok", "token-level postings (no per-passage dedup)"),
    ("weft_tok", "weft_resid", "residual correction (WARP implicit decompression) + rerank"),
    ("weft", "weft_resid", "residual correction overall (index + scoring)"),
    ("weft_resid", "weft_resid_norr", "drop the exact rerank"),
    ("weft", "weft_resid_norr", "WEFT vs a WARP-style GPU pipeline (no rerank)"),
    ("weft", "weft_graph", "CUDA graphs + one sync (launch and sync overhead)"),
    ("ptms", "weft_graph", "WEFT (graphs) vs PLAID+TMS"),
    ("weft", "weft_hash", "hash-table Stage 3 instead of a radix sort"),
    ("ptms", "weft_hash", "WEFT (hash Stage 3) vs PLAID+TMS"),
    ("weft", "weft_pairs", "32-bit key / 16-bit value pair sort instead of 64-bit keys"),
    ("ptms", "weft_pairs", "WEFT (pair sort) vs PLAID+TMS"),
]
# filter_budget stage: PLAID / PLAID+TMS post-filtering with growing candidate budgets
FILTER_BUDGETS = [{"ncells": 2, "threshold": 0.45, "ndocs": 1024}, {"ncells": 4, "threshold": 0.4, "ndocs": 4096},
                  {"ncells": 8, "threshold": 0.35, "ndocs": 16384}, {"ncells": 16, "threshold": 0.3, "ndocs": 65536}]
DEFAULT_ENGINES = CORE + ["ragrt_bf"]
PLAID_DEFAULT_K100 = {"ncells": 2, "threshold": 0.45, "ndocs": 1024}   # ColBERT Searcher defaults for k=100

PLAID_GRID = [{"ncells": n, "threshold": t, "ndocs": d}
              for n in [1, 2, 4, 8] for t in [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
              for d in [64, 128, 256, 512, 1024, 2048, 4096, 8192]]
RAGRT_GRID = [{"nc": n, "eids": e, "ndocs": d}
              for n in [4, 8, 16, 32, 64] for e in [4, 8, 16, 32, 64, 128]
              for d in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]]   # eids <= 128 (kernel limit)
QUICK_PLAID = [PLAID_DEFAULT_K100, {"ncells": 1, "threshold": 0.5, "ndocs": 256}]
NOFILTER_GRID = [{"nc": n, "eids": "all", "ndocs": d}
                 for n in [1, 2, 4, 8, 16, 32, 64]
                 for d in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]]
QUICK_RAGRT = [{"nc": 32, "eids": 16, "ndocs": 4096}, {"nc": 16, "eids": 32, "ndocs": 16384}]
# no rerank: ndocs only sets how many candidates leave Stage 3; the top-k by approximate score
# is the same for any ndocs >= k, so one value per nc
NORR_GRID = [{"nc": n, "eids": "all", "ndocs": C.TOP_K} for n in [1, 2, 4, 8, 16, 32, 64, 128]]
MUVERA_GRID = [{"cands": c} for c in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]]
ATTRIB_QUERIES = 200
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
WARMUP = 5


class Bench:
    def __init__(self, args):
        self.args = args
        self.engines = [e for e in ENGINES if e in args.engines]
        self.ds = C.dataset(args.dataset)
        if args.sparse_csr_dir:
            self.ds["sparse_csr_dir"] = args.sparse_csr_dir
            self.ds["predicates_path"] = os.path.join(args.sparse_csr_dir, "predicates.npy")
        if args.results_subdir:
            self.ds["results_dir"] = os.path.join(self.ds["results_dir"], args.results_subdir)
        os.makedirs(self.ds["results_dir"], exist_ok=True)
        self.json_path = os.path.join(self.ds["results_dir"], "results.json")
        self.npz_path = os.path.join(self.ds["results_dir"], "test_perquery.npz")
        self.R = json.load(open(self.json_path)) if os.path.exists(self.json_path) else {}
        self.perq = dict(np.load(self.npz_path)) if os.path.exists(self.npz_path) else {}

        if args.gt_path:
            self.ds["gt_path"] = args.gt_path
        if not os.path.exists(self.ds["gt_path"]):
            sys.exit(f"FATAL: {self.ds['gt_path']} missing. Run build_ground_truth.py --dataset {args.dataset} first.")
        z = np.load(self.ds["gt_path"])
        self.gt_names = [k for k in z.files if k not in ("qids", "done", "scores_all")]
        gt_qids = [str(q) for q in z["qids"]]
        self.gt = {name: {q: z[name][i].tolist() for i, q in enumerate(gt_qids)} for name in self.gt_names}

        self.qrels = E.load_qrels(self.ds["qrels_path"])
        cp = os.path.join(C.dataset(args.dataset)["results_dir"], "corr_filters.json")
        self.corr = json.load(open(cp))["filters"] if os.path.exists(cp) else {}
        missing = [f for f in self.gt_names if f.startswith("corr_") and f not in self.corr]
        if missing:
            sys.exit(f"FATAL: ground truth has {missing} but {cp} is missing")
        text = dict(E.load_questions(self.ds["questions_path"]))
        self.meta = json.load(open(self.ds["queries_meta_path"])) if self.ds["has_domains"] else {}
        qids = [q for q in gt_qids if q in self.qrels]
        self.tune, self.test = E.make_split(qids, args.tune_size, C.SPLIT_SEED)
        if args.max_test:
            self.test = self.test[:args.max_test]
        self.R["split"] = {"tune": len(self.tune), "test": len(self.test), "seed": C.SPLIT_SEED,
                           "total_with_gt": len(qids)}
        print(f"[{args.dataset}] {len(qids):,} queries with ground truth: tune {len(self.tune):,}, "
              f"test {len(self.test):,}", flush=True)

        if "muvera" in self.engines and not args.muvera_dir:
            sys.exit("FATAL: --engines muvera needs --muvera_dir (build_muvera_fde.py)")
        extra = {"muvera_dir": args.muvera_dir} if "muvera" in self.engines else {}
        self.eng = Engines(self.ds, load_ragrt=any(e in RAGRT_ENGINES for e in self.engines),
                           geometry=args.geometry, stage3=args.stage3, rerank=args.rerank,
                           rt_quantile=args.rt_quantile, encoder=args.encoder, **extra)
        self.bank = {}
        enc_ms = []
        need = self.tune + self.test
        for q in need[:WARMUP]:
            self.eng.encode(text[q])
        for q in need:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            self.bank[q] = self.eng.encode(text[q])
            torch.cuda.synchronize()
            enc_ms.append((time.perf_counter() - t0) * 1000)
        self.R["stage0a_encode"] = E.latency_summary(enc_ms)
        print(f"  encoded {len(need):,} queries, stage 0a median {np.median(enc_ms):.2f} ms", flush=True)
        if self.eng.index is not None:
            self.eng.calibrate([self.bank[q] for q in self.tune[:args.calib_queries]])
        if args.baseline_from:
            self.import_baselines(args.baseline_from, args.import_engines)
        self.R["meta"] = {**self.R.get("meta", {}),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "gpu": torch.cuda.get_device_name(0),
            "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.BASE_DIR,
                                  capture_output=True, text=True).stdout.strip(),
            "args": vars(args), "official_metric": self.ds["official_metric"],
            "query_rows": "attention mask (ragrt_index_lib.query_ntok)",
            "ragrt": self.eng.options(),
        }

    def import_baselines(self, src_dir, engines=("plaid", "ptms")):
        """Copy other engines' results (sweep, test, per-query, breakdown, filtered, throughput)."""
        src = json.load(open(os.path.join(src_dir, "results.json")))
        if src.get("split") != self.R["split"]:
            sys.exit(f"FATAL: --baseline_from split {src.get('split')} != this run's {self.R['split']}")
        perq = dict(np.load(os.path.join(src_dir, "test_perquery.npz")))
        base = [e for e in engines if e in src.get("sweep", {})]
        for e in base:
            self.R.setdefault("sweep", {})[e] = src["sweep"][e]
            for k, v in src.get("test", {}).items():
                if v.get("engine") == e:
                    self.R.setdefault("test", {})[k] = v
            for k, v in perq.items():
                if k.startswith(e + "|"):
                    self.perq[k] = v
            if e in src.get("breakdown", {}):
                self.R.setdefault("breakdown", {})[e] = src["breakdown"][e]
            for f, d in src.get("filtered", {}).items():
                if e in d:
                    self.R.setdefault("filtered", {}).setdefault(f, {})[e] = d[e]
            for k, v in src.get("throughput", {}).items():
                if k.startswith(e + "_"):
                    self.R.setdefault("throughput", {})[k] = v
        self.R.setdefault("meta", {})["baselines_from"] = {"dir": src_dir, "engines": base,
                                                           "meta": src.get("meta", {})}
        print(f"  imported baselines {base} from {src_dir}", flush=True)

    # ------------------------------------------------------------------ core
    def mask_for(self, filt, q):
        if filt == "all":
            return 0
        if filt == "own_domain":
            return 1 << (C.DOMAIN_BIT_BASE + C.LOTTE_DOMAINS.index(self.meta[q]["domain"]))
        if filt in self.corr:                       # query-correlated topic filters (build_corr_filters.py)
            return self.corr[filt][q]
        return dict(("syn" + l.rstrip("%"), b) for l, b in C.SYNTHETIC_MASKS)[filt]

    def evaluate(self, engine, p, qids, repeats=1, filt="all"):
        """Per-query latency (median of repeats) and quality for one engine/config."""
        for q in qids[:WARMUP]:
            Qf, n = self.bank[q]
            self.eng.search(engine, Qf, n, p, self.mask_for(filt, q))
        lat, r10, r100, off, nret = [], [], [], [], []
        for q in qids:
            Qf, n = self.bank[q]
            m = self.mask_for(filt, q)
            ts = []
            for _ in range(repeats):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ranked = self.eng.search(engine, Qf, n, p, m)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            lat.append(float(np.median(ts)))
            gt = self.gt[filt][q]
            r10.append(E.gt_recall(ranked, gt, 10))
            r100.append(E.gt_recall(ranked, gt, 100))
            off.append(E.official_metric(self.ds["official_metric"], ranked, self.qrels[q])
                       if filt in ("all", "own_domain") else float("nan"))
            nret.append(len(ranked))
        return {"lat": np.array(lat), "gt_r10": np.array(r10), "gt_r100": np.array(r100),
                "official": np.array(off), "n_returned": np.array(nret),
                "ntok": np.array([self.bank[q][1] for q in qids])}

    @staticmethod
    def summarize(res):
        s = {"lat": E.latency_summary(res["lat"])}
        for k in ("gt_r10", "gt_r100", "official"):
            m, lo, hi = E.bootstrap_ci(res[k])
            s[k] = {"mean": m, "ci95": [lo, hi]}
        s["frac_under_10_results"] = float(np.mean(res["n_returned"] < 10))
        return s

    def save(self):
        E.dump_json(self.R, self.json_path)
        np.savez(self.npz_path, **self.perq)

    # ---------------------------------------------------------------- stages
    def stage_sweep(self):
        plaid_grid = QUICK_PLAID if self.args.quick else PLAID_GRID
        ragrt_grid = QUICK_RAGRT if self.args.quick else RAGRT_GRID
        nof = NOFILTER_GRID[:2] if self.args.quick else NOFILTER_GRID
        grids = {"plaid": plaid_grid, "ptms": plaid_grid, "ragrt": ragrt_grid, "ragrt_bf": ragrt_grid,
                 "ragrt_nofilter": nof, "ragrt_centroid": ragrt_grid, "ragrt_ivf": nof,
                 "weft": nof, "weft_noimpute": nof, "weft_tok": nof, "weft_resid": nof,
                 "weft_resid_norr": NORR_GRID[:2] if self.args.quick else NORR_GRID, "weft_graph": nof, "weft_hash": nof, "weft_adapt": nof, "weft_pairs": nof,
                 "muvera": MUVERA_GRID[:2] if self.args.quick else MUVERA_GRID}
        if PLAID_DEFAULT_K100 not in grids["plaid"]:
            grids["plaid"] = grids["plaid"] + [PLAID_DEFAULT_K100]
        sweep = self.R.setdefault("sweep", {})
        for engine in self.engines:
            pts = []
            t0 = time.time()
            for i, p in enumerate(grids[engine]):
                res = self.evaluate(engine, p, self.tune)
                pts.append({"params": p, "key": cfg_key(engine, p),
                            "lat_median": float(np.median(res["lat"])),
                            "lat_p95": float(np.percentile(res["lat"], 95)),
                            "gt_r10": float(np.nanmean(res["gt_r10"])),
                            "gt_r100": float(np.nanmean(res["gt_r100"])),
                            "official": float(np.nanmean(res["official"]))})
                if (i + 1) % 20 == 0:
                    print(f"    {engine}: {i + 1}/{len(grids[engine])} configs ({time.time() - t0:.0f} s)", flush=True)
            sweep[engine] = pts
            print(f"  swept {engine}: {len(pts)} configs on {len(self.tune):,} tune queries "
                  f"({time.time() - t0:.0f} s)", flush=True)
            self.save()

    def stage_select(self):
        sweep = self.R["sweep"]
        default_pt = next(p for p in sweep["plaid"] if p["params"] == PLAID_DEFAULT_K100)
        targets = {f"gt_r10>={t:.2f}": t for t in self.args.targets}
        targets["plaid_default"] = default_pt["gt_r10"]
        sel = {}
        swept = [e for e in ENGINES if e in sweep]
        for name, t in targets.items():
            sel[name] = {"target_gt_r10": t}
            for engine in swept:
                pt = E.select_fastest_at(sweep[engine], t)
                sel[name][engine] = None if pt is None else {"params": pt["params"], "key": pt["key"],
                                                             "tune_lat_median": pt["lat_median"],
                                                             "tune_gt_r10": pt["gt_r10"]}
        self.R["selection"] = sel
        # Primary point: PLAID's own k=100 default quality if every engine reaches it on TUNE,
        # else the highest target that all three reach (MS MARCO: RAGRT tops out just below it).
        core = self.args.core or CORE
        reached = [n for n, v in sel.items() if all(v.get(e) for e in core)]
        if "plaid_default" in reached:
            primary = "plaid_default"
        elif reached:
            primary = max(reached, key=lambda n: sel[n]["target_gt_r10"])
        else:
            sys.exit("FATAL: no quality target is reached by all engines on TUNE; widen the grids.")
        self.R["primary_target"] = primary
        print(f"  primary operating point: {primary}")
        print("  operating points (fastest TUNE config reaching each target):")
        for name, s in sel.items():
            row = "  ".join(f"{e}={s[e]['key'] if s[e] else 'unreachable'}" for e in swept)
            print(f"    {name:<16} {row}")

    def selected_configs(self, engines=None):
        seen = {}
        for s in self.R["selection"].values():
            for engine in (engines or self.engines):
                if s.get(engine):
                    seen[(engine, s[engine]["key"])] = s[engine]["params"]
        return seen

    def stage_test(self):
        test = self.R.setdefault("test", {})
        for (engine, key), p in self.selected_configs().items():
            res = self.evaluate(engine, p, self.test, repeats=self.args.repeats)
            test[f"{engine}|{key}"] = {"engine": engine, "params": p, **self.summarize(res)}
            for k, v in res.items():
                self.perq[f"{engine}|{key}|{k}"] = v
            s = test[f"{engine}|{key}"]
            print(f"    {engine:<5} {key:<28} lat {s['lat']['median']:6.2f} ms  gt_r10 {s['gt_r10']['mean']:.4f}  "
                  f"{self.ds['official_metric']} {s['official']['mean']:.4f}", flush=True)
        self.perq["test_qids"] = np.array(self.test)
        comps = {}
        for name, s in self.R["selection"].items():
            c = {}
            for engine in ENGINES:
                if s.get(engine) and f"{engine}|{s[engine]['key']}" in test:
                    c[engine] = test[f"{engine}|{s[engine]['key']}"]
            if "ragrt" in c:
                rk = f"ragrt|{s['ragrt']['key']}"
                for base in ("plaid", "ptms", "ragrt_bf", "ragrt_nofilter", "ragrt_centroid", "ragrt_ivf", "weft", "muvera"):
                    if base in c:
                        bk = f"{base}|{s[base]['key']}"
                        c[f"speedup_vs_{base}"] = c[base]["lat"]["median"] / c["ragrt"]["lat"]["median"]
                        for metric in ("gt_r10", "official"):
                            m, lo, hi = E.paired_diff_ci(self.perq[f"{rk}|{metric}"], self.perq[f"{bk}|{metric}"])
                            c[f"{metric}_diff_vs_{base}"] = {"mean": m, "ci95": [lo, hi]}
            att = []
            for slow, fast, what in ATTRIBUTION:
                if slow in c and fast in c:
                    a = {"slow": slow, "fast": fast, "what": what,
                         "latency_ratio": c[slow]["lat"]["median"] / c[fast]["lat"]["median"]}
                    sk, fk = f"{slow}|{s[slow]['key']}", f"{fast}|{s[fast]['key']}"
                    for metric in ("gt_r10", "official"):
                        if f"{sk}|{metric}" in self.perq and f"{fk}|{metric}" in self.perq:
                            m, lo, hi = E.paired_diff_ci(self.perq[f"{fk}|{metric}"], self.perq[f"{sk}|{metric}"])
                            a[f"{metric}_diff"] = {"mean": m, "ci95": [lo, hi]}
                    att.append(a)
            if att:
                c["attribution"] = att
            comps[name] = c
        self.R["comparisons"] = comps

    def primary(self, engines=None):
        """Primary-point config per engine; an engine that cannot reach it (e.g. MUVERA) uses the
        highest target it does reach, recorded in R["primary_fallback"]."""
        engines = engines or self.engines
        sel = self.R["selection"]
        s = sel[self.R["primary_target"]]
        out = {e: s[e]["params"] for e in engines if s.get(e)}
        out.update({e: s[PARAMS_FROM[e]]["params"] for e in engines
                    if e not in out and e in PARAMS_FROM and s.get(PARAMS_FROM[e])})
        fb = self.R.setdefault("primary_fallback", {})
        for e in engines:
            if e in out:
                fb.pop(e, None)
                continue
            reach = [(v["target_gt_r10"], n) for n, v in sel.items() if v.get(e)]
            if reach:
                name = max(reach)[1]
                out[e] = sel[name][e]["params"]
                fb[e] = name
                print(f"  {e} cannot reach {self.R['primary_target']}; using its {name} config", flush=True)
        return out

    def stage_breakdown(self):
        P = self.primary()
        qids = self.test[:self.args.max_filter_queries]
        out = self.R.setdefault("breakdown", {})
        out["stage0a_encode_median"] = self.R["stage0a_encode"]["median"]
        eng = self.eng
        if "plaid" in P:
            s13, tot = [], []
            for q in qids[:WARMUP]:
                eng.plaid(self.bank[q][0], P["plaid"])
            for q in qids:
                Qf, _ = self.bank[q]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                eng.plaid_stage3(Qf, P["plaid"]); torch.cuda.synchronize()
                s13.append((time.perf_counter() - t0) * 1000)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                eng.plaid(Qf, P["plaid"]); torch.cuda.synchronize()
                tot.append((time.perf_counter() - t0) * 1000)
            out["plaid"] = {"s1_3": float(np.median(s13)), "total": float(np.median(tot)),
                            "s4_est": float(max(0.0, np.median(tot) - np.median(s13)))}
        if "ptms" in P:
            s13, s4 = [], []
            for q in qids[:WARMUP]:
                eng.ptms(self.bank[q][0], P["ptms"])
            for q in qids:
                Qf, _ = self.bank[q]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                c = eng.plaid_stage3(Qf, P["ptms"]).to(torch.int32).contiguous(); torch.cuda.synchronize()
                t1 = time.perf_counter()
                s = eng.tms_scores(Qf, c)
                ranked = c[torch.topk(s, min(C.TOP_K, c.numel())).indices].tolist(); torch.cuda.synchronize()
                s13.append((t1 - t0) * 1000); s4.append((time.perf_counter() - t1) * 1000)
            out["ptms"] = {"s1_3": float(np.median(s13)), "s4": float(np.median(s4)),
                           "total": float(np.median(np.array(s13) + np.array(s4)))}
        if "muvera" in P:
            s13, s4 = [], []
            for q in qids[:WARMUP]:
                eng.muvera_search(self.bank[q][0], P["muvera"])
            for q in qids:
                Qf, _ = self.bank[q]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                c = eng.muvera_candidates(Qf, P["muvera"]); torch.cuda.synchronize()
                t1 = time.perf_counter()
                s = eng.tms_scores(Qf, c)
                c[torch.topk(s, min(C.TOP_K, c.numel())).indices].tolist(); torch.cuda.synchronize()
                s13.append((t1 - t0) * 1000); s4.append((time.perf_counter() - t1) * 1000)
            out["muvera"] = {"s1_3": float(np.median(s13)), "s4": float(np.median(s4)),
                             "total": float(np.median(np.array(s13) + np.array(s4)))}
        for e in GRAPH_ENGINES:             # no per-stage events inside a graph: total only
            if e not in P:
                continue
            tot = []
            for q in qids[:WARMUP]:
                eng.search(e, *self.bank[q], P[e])
            for q in qids:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                eng.search(e, *self.bank[q], P[e]); torch.cuda.synchronize()
                tot.append((time.perf_counter() - t0) * 1000)
            out[e] = {"total": float(np.median(tot))}
        for e in RAGRT_ENGINES:
            if e not in P or e in GRAPH_ENGINES:
                continue
            eng.use_engine(e)
            pe = eng.resolve(e, P[e])
            try:
                st, tot = [], []
                for q in qids[:WARMUP]:
                    eng.ragrt_profiled(*self.bank[q], pe)
                for q in qids:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    _, s = eng.ragrt_profiled(*self.bank[q], pe)
                    tot.append((time.perf_counter() - t0) * 1000)
                    st.append(s)
            finally:
                eng.use_engine("ragrt")
            st = np.array(st)
            names = ["s0b_prep", "s1_rt", "s2_gather", "s3_score", "s4_tms"]
            if e == "weft_resid_norr":      # the profiled path always reranks; this engine does not:
                st[:, 4] = 0.0              # drop s4 and time the real (no-rerank) search for the total
                tot = []
                for q in qids:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    eng.search(e, *self.bank[q], P[e]); torch.cuda.synchronize()
                    tot.append((time.perf_counter() - t0) * 1000)
            out[e] = {n: float(np.median(st[:, i])) for i, n in enumerate(names)}
            out[e]["total"] = float(np.median(tot))
            out[e]["gpu_sum"] = float(np.median(st.sum(axis=1)))
        print(f"  breakdown: {json.dumps(out)}", flush=True)

    def stage_filtered(self):
        extra = [e for e in (self.args.filtered_extra or []) if e not in self.engines]
        P = self.primary(self.engines + extra)
        qids = self.test[:self.args.max_filter_queries]
        filts = [f for f in self.gt_names if f != "all" and (not self.args.filters or f in self.args.filters)]
        out = self.R.setdefault("filtered", {})
        pairs = self.R.setdefault("filtered_pairs", {})
        for f in filts:
            out.setdefault(f, {})
            r10 = {}
            for engine, p in P.items():
                depth = None
                if engine in ORACLE_ENGINES:         # untimed pass: how deep the last probed centroid sits
                    self.eng.diag, self.eng.adapt_log = True, []
                    self.evaluate(engine, p, qids[:DIAG_QUERIES], filt=f)
                    self.eng.diag = False
                    depth = float(np.mean([float(x) for x in self.eng.adapt_log])) if self.eng.adapt_log else None
                self.eng.adapt_log = []
                res = self.evaluate(engine, p, qids, filt=f)
                out[f][engine] = self.summarize(res)
                r10[engine] = res["gt_r10"]
                self.perq[f"{engine}|filt:{f}|gt_r10"] = res["gt_r10"]
                log = self.eng.adapt_log
                if depth is not None:                # probed depth (unfiltered centroid rank)
                    out[f][engine]["adapt_nc_mean"] = depth
                elif engine in ADAPT_ENGINES and log:
                    out[f][engine]["adapt_nc_mean"] = float(np.mean([float(x) for x in log]))
            for a, b in FILTER_PAIRS:
                if a in r10 and b in r10:
                    m, lo, hi = E.paired_diff_ci(r10[b], r10[a])
                    pairs.setdefault(f, {})[f"{a} -> {b}"] = {"mean": m, "ci95": [lo, hi]}
            row = "  ".join(f"{e} {out[f][e]['lat']['median']:.2f}ms r10={out[f][e]['gt_r10']['mean']:.3f}"
                            for e in out[f])
            print(f"    filter {f:<10} {row}", flush=True)

    def stage_filter_budget(self):
        """Can post-filtering catch up with a bigger candidate budget, and at what latency? PLAID and
        PLAID+TMS at FILTER_BUDGETS on the hard filters (default corr_mid, corr_far, syn5)."""
        filts = self.args.filters or [f for f in ("corr_mid", "corr_far", "syn5") if f in self.gt_names]
        qids = self.test[:self.args.budget_queries]
        out = self.R.setdefault("filter_budget", {})
        for f in filts:
            for e in ("plaid", "ptms"):
                for b in FILTER_BUDGETS:
                    res = self.evaluate(e, b, qids, filt=f)
                    k = f"nc={b['ncells']} th={b['threshold']} ndocs={b['ndocs']}"
                    out.setdefault(f, {}).setdefault(e, {})[k] = {"params": b, **self.summarize(res)}
                    v = out[f][e][k]
                    print(f"    budget {f:<9} {e:<5} {k:<28} r10={v['gt_r10']['mean']:.3f}  {v['lat']['median']:.2f} ms", flush=True)
                self.save()

    def stage_throughput(self):
        P = self.primary()
        qids = self.test[:self.args.throughput_queries]
        Qs = [self.bank[q] for q in qids]
        out = self.R.setdefault("throughput", {})
        out["n_queries"] = len(qids)

        def timed(fn):
            fn(Qs[:WARMUP])
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(Qs); torch.cuda.synchronize()
            return len(Qs) / (time.perf_counter() - t0)

        for engine in self.engines:
            if engine in P:
                out[f"{engine}_sequential_qps"] = timed(
                    lambda batch, e=engine: [self.eng.search(e, Qf, n, P[e]) for Qf, n in batch])
        for e in RAGRT_ENGINES:
            if e not in P or e in RESID_ENGINES or e in GRAPH_ENGINES or e in PERQ_PREP:   # per-query state
                continue
            self.eng.use_engine(e)
            pe = self.eng.resolve(e, P[e])
            try:
                out[f"{e}_pipelined_qps"] = {}
                for B in BATCH_SIZES:
                    def run(batch, B=B, pe=pe):
                        for a in range(0, len(batch), B):
                            chunk = batch[a:a + B]
                            self.eng.ragrt_pipelined([x[0] for x in chunk], [x[1] for x in chunk], pe)
                    out[f"{e}_pipelined_qps"][str(B)] = timed(run)
            finally:
                self.eng.use_engine("ragrt")
        print(f"  throughput: {json.dumps(out)}", flush=True)

    def stage_attribution(self):
        """
        Where each engine's time goes at the primary point: one torch.profiler (CUPTI) pass over
        ATTRIB_QUERIES TEST queries. GPU-busy = summed device time of kernels and memcpys (the
        engines run on one stream per query, so this is not double counted); host = wall - busy.
        Profiling inflates wall time somewhat, so wall is also measured without the profiler.
        """
        from torch.profiler import profile, ProfilerActivity
        P = self.primary()
        qids = self.test[:ATTRIB_QUERIES]
        out = self.R.setdefault("attribution", {})
        fns = {e: (lambda q, e=e: self.eng.search(e, *self.bank[q], P[e])) for e in P}
        if "ptms" in P:
            fns["ptms_stages1_3"] = lambda q: self.eng.plaid_stage3(self.bank[q][0], P["ptms"])
        for name, fn in fns.items():
            for q in qids[:WARMUP]:
                fn(q)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for q in qids:
                fn(q); torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000 / len(qids)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for q in qids:
                    fn(q); torch.cuda.synchronize()
            out[name] = {"wall_ms": wall, **profile_counts(prof, len(qids))}
            o = out[name]
            print(f"    {name:<16} wall {o['wall_ms']:6.2f} ms  gpu busy {o['gpu_busy_ms']:6.2f} ms  "
                  f"kernels {o['kernels']:6.1f}  memcpy {o['memcpys']:5.1f}  syncs {o['syncs']:5.1f}  per query", flush=True)

    def stage_drops(self):
        qids = self.test[:self.args.max_filter_queries]
        out = self.R.setdefault("drops", {})
        rtrag_corr_3d.set_drop_stats(True)
        try:
            for (engine, key), p in self.selected_configs().items():
                if engine not in RAGRT_ENGINES:
                    continue
                rtrag_corr_3d.reset_drop_stats()
                for q in qids:
                    self.eng.search(engine, *self.bank[q], p)
                torch.cuda.synchronize()
                s = dict(rtrag_corr_3d.get_drop_stats())
                rays = max(s["rays"], 1)
                s["frac_rays_overflow"] = s["rays_overflow"] / rays
                s["frac_combos_found"] = s["combos_found"] / max(s["combos"], 1)
                s["frac_tasks_dropped"] = s["tasks_dropped"] / max(s["tasks"], 1)
                s["hits_per_ray"] = s["hits"] / rays
                s["frac_rays_short_of_k"] = s["rays_short"] / rays
                s["postings_per_query"] = s["postings"] / max(len(qids), 1)
                s["queries"] = len(qids)
                s["engine"] = engine
                out[f"{engine}|{key}"] = s
                print(f"    {engine:<8} {key:<28} hits/ray {s['hits_per_ray']:.1f}  short of k {s['frac_rays_short_of_k']:.1%}  "
                      f"overflow {s['frac_rays_overflow']:.2%}  tasks dropped {s['frac_tasks_dropped']:.2%}  "
                      f"postings/query {s['postings_per_query']:,.0f}", flush=True)
        finally:
            rtrag_corr_3d.set_drop_stats(False)


def profile_counts(prof, n):
    """Per-query kernel / memcpy / sync counts and device time from a torch.profiler run."""
    kern = mem = sync = 0
    busy_us = 0.0
    for ev in prof.events():
        name = ev.name
        dev = str(getattr(ev, "device_type", "")).endswith("CUDA")
        if dev:
            t = getattr(ev, "device_time", None)
            if t is None:
                t = getattr(ev, "cuda_time", 0.0)
            busy_us += t
            if "memcpy" in name.lower() or "memset" in name.lower():
                mem += 1
            else:
                kern += 1
        elif "Synchronize" in name or name in ("cudaMemcpy", "cudaMemcpyAsync"):
            sync += 1
    # syncs exclude the harness's own torch.cuda.synchronize() after every query
    return {"gpu_busy_ms": busy_us / 1000 / n, "kernels": kern / n, "memcpys": mem / n,
            "syncs": max(0.0, sync / n - 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--stages", default="sweep,select,test,breakdown,filtered,throughput,drops")
    ap.add_argument("--tune_size", type=int, default=C.TUNE_SIZE)
    ap.add_argument("--max_test", type=int, default=None, help="cap TEST queries (default: all)")
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats per TEST query (median)")
    ap.add_argument("--max_filter_queries", type=int, default=2000)
    ap.add_argument("--throughput_queries", type=int, default=1024)
    ap.add_argument("--targets", type=lambda s: [float(x) for x in s.split(",")], default=[0.80, 0.85, 0.90, 0.95, 0.98])
    ap.add_argument("--quick", action="store_true", help="tiny grids, for a smoke test")
    ap.add_argument("--engines", type=lambda s: s.split(","), default=DEFAULT_ENGINES,
                    help="subset to run (results merge into an existing results.json), e.g. ragrt_bf")
    ap.add_argument("--gt_path", default=None, help="ground truth file (default: the full one for the dataset)")
    ap.add_argument("--results_subdir", default=None, help="write results to <results>/<dataset>/<subdir> (e.g. smoke)")
    ap.add_argument("--sparse_csr_dir", default=None, help="RAGRT index dir (default: eval_config), e.g. an E-sweep index")
    ap.add_argument("--geometry", default="polar", choices=["polar", "polar_iter", "fan"])
    ap.add_argument("--stage3", default="sparse", choices=["sparse", "dense"])
    ap.add_argument("--rerank", default="wmma", choices=["wmma", "simt"])
    ap.add_argument("--rt_quantile", type=float, default=0.05)
    ap.add_argument("--calib_queries", type=int, default=300, help="TUNE queries used to calibrate polar thresholds")
    ap.add_argument("--baseline_from", default=None, help="results dir whose results to import (see --import_engines)")
    ap.add_argument("--import_engines", type=lambda s: s.split(","), default=["plaid", "ptms"],
                    help="engines imported with --baseline_from (they are not rerun), e.g. plaid,ptms,ragrt,ragrt_bf")
    ap.add_argument("--core", type=lambda s: s.split(","), default=None,
                    help="engines that must reach the primary operating point (default plaid,ptms,ragrt), e.g. plaid,ptms,weft")
    ap.add_argument("--muvera_dir", default=None, help="FDE dir from build_muvera_fde.py (engine muvera)")
    ap.add_argument("--encoder", default="stock", choices=["stock", "fast"],
                    help="query encoder (stage 0a): ColBERT's own, or fast_encoder.py (fp16 + CUDA graph)")
    ap.add_argument("--filters", type=lambda s: s.split(","), default=None,
                    help="filtered stage: only these filters (e.g. corr_near,corr_mid,corr_far)")
    ap.add_argument("--budget_queries", type=int, default=500, help="TEST queries per point in the filter_budget stage")
    ap.add_argument("--filtered_extra", type=lambda s: s.split(","), default=None,
                    help="filtered stage: also run these (e.g. imported) engines at their selected configs, e.g. plaid,ptms")
    args = ap.parse_args()
    if args.baseline_from:
        args.engines = [e for e in args.engines if e not in args.import_engines and e not in ("plaid", "ptms")]

    b = Bench(args)
    for st in args.stages.split(","):
        print(f"\n== stage {st}", flush=True)
        t0 = time.time()
        getattr(b, f"stage_{st}")()
        b.save()
        print(f"   ({time.time() - t0:.0f} s, saved {b.json_path})", flush=True)


if __name__ == "__main__":
    main()
