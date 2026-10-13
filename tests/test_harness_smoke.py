"""
CPU end-to-end smoke test of the evaluation pipeline with a fake engine:
build_ground_truth -> run_benchmarks (all stages) -> plot_results.
Checks plumbing (splits, GT lookups, filters, selection, JSON/npz, figures), not speed.
Needs torch (CPU is fine). Run: python3 tests/test_harness_smoke.py
"""
import os, sys, json, types, tempfile, shutil
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
TMP = tempfile.mkdtemp(prefix="ragrt_smoke_")
os.environ["RAGRT_RESULTS"] = os.path.join(TMP, "results")

import torch
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.get_device_name = lambda *a, **k: "fake-gpu"

# --- stub the compiled extension and the ColBERT-backed engines -------------
ext = types.ModuleType("rtrag_corr_3d")
_stats = {"on": False, "v": dict.fromkeys(["launches", "rays", "rays_overflow", "hits_lost_to_cap", "combos",
                                            "combos_found", "min_score_drops", "tasks", "tasks_dropped",
                                            "hits", "rays_short", "postings"], 0)}
ext.set_drop_stats = lambda on: _stats.__setitem__("on", on)
ext.reset_drop_stats = lambda: _stats["v"].update(dict.fromkeys(_stats["v"], 0))
ext.get_drop_stats = lambda: dict(_stats["v"])
sys.modules["rtrag_corr_3d"] = ext

N_DOCS, DIM = 3000, 16
RNG = np.random.default_rng(0)
DOCS = torch.tensor(RNG.normal(size=(N_DOCS, DIM)), dtype=torch.float32)


FAKE_RESID = ("weft_resid", "weft_resid_norr")
FAKE_RAGRT = ("ragrt", "ragrt_bf", "ragrt_nofilter", "ragrt_centroid", "ragrt_ivf", "weft", "weft_noimpute",
              "weft_tok", "weft_graph", "weft_hash", "weft_adapt", "weft_pairs", "weft_oracle", "weft_oracle_adapt") + FAKE_RESID


def cfg_key(engine, p):
    if engine == "muvera":
        return f"cands={p['cands']}"
    return (f"nc={p['nc']} eids={p['eids']} ndocs={p['ndocs']}" if engine in FAKE_RAGRT
            else f"nc={p['ncells']} th={p['threshold']} ndocs={p['ndocs']}")


class FakeEngines:
    device = "cpu"
    calls = []

    def __init__(self, ds, load_ragrt=True, **opts):
        FakeEngines.calls.append({"load_ragrt": load_ragrt, "sparse_csr_dir": ds["sparse_csr_dir"], **opts})
        self.index = object() if load_ragrt else None
        self.opts = opts
        self.calibrated = 0
        self.N = N_DOCS
        self.pred_np = np.load(ds["predicates_path"]).astype(np.uint32)
        self.pred_gpu = torch.from_numpy(self.pred_np.astype(np.int64))

    def calibrate(self, Qs):
        self.calibrated = len(Qs)

    def options(self):
        return {**self.opts, "calibrated_on": self.calibrated}

    def encode(self, text):
        g = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return torch.tensor(g.normal(size=DIM), dtype=torch.float32), 5 + len(text) % 7

    def tms_scores(self, Qf, pids):
        return DOCS[pids.long()] @ Qf

    def _ranked(self, Qf, depth, noise, mask):
        s = DOCS @ Qf + noise * torch.randn(N_DOCS, generator=torch.Generator().manual_seed(int(depth)))
        cand = torch.topk(s, depth).indices
        if mask:
            cand = cand[(self.pred_gpu[cand] & mask) != 0]
        if cand.numel() == 0:
            return []
        exact = DOCS[cand] @ Qf
        return cand[torch.argsort(exact, descending=True)][:100].tolist()

    def muvera_candidates(self, Qf, p):
        return torch.topk(DOCS @ Qf, min(N_DOCS, p["cands"])).indices.to(torch.int32)

    def muvera_search(self, Qf, p, mask=0):
        return self._ranked(Qf, max(min(N_DOCS, p["cands"]), 10), 3.0, mask)

    def search(self, engine, Qf, ntok, p, mask=0):
        if engine == "muvera":
            return self.muvera_search(Qf, p, mask)
        if engine.startswith("ragrt"):
            return self.ragrt(Qf, ntok, p, mask)
        return self._ranked(Qf, max(min(N_DOCS, p["ndocs"] // 4), 10), 2.0, mask)

    def plaid(self, Qf, p, mask=0):
        return self.search("plaid", Qf, 0, p, mask)

    def ptms(self, Qf, p, mask=0):
        return self.search("ptms", Qf, 0, p, mask)

    def plaid_stage3(self, Qf, p):
        return torch.topk(DOCS @ Qf, min(N_DOCS, p["ndocs"] // 4)).indices.to(torch.int32)

    def stage1(self, brute_force):
        self.bf = brute_force

    def use_engine(self, engine):
        self.engine = engine

    def resolve(self, engine, p):
        return dict(p, eids=256) if p.get("eids") == "all" else p

    def ragrt(self, Qf, ntok, p, mask=0):
        if _stats["on"]:
            _stats["v"]["rays"] += 32 * ntok
            _stats["v"]["tasks"] += 10
        return self._ranked(Qf, max(min(N_DOCS, p["ndocs"]), 10), 2.0, mask)

    def ragrt_profiled(self, Qf, ntok, p):
        return self.ragrt(Qf, ntok, p), [0.01, 0.1, 0.1, 0.2, 0.3]

    def ragrt_pipelined(self, Qfs, ntoks, p, mask=0):
        return [self.ragrt(q, n, p, mask) for q, n in zip(Qfs, ntoks)]


eng_mod = types.ModuleType("engines")
eng_mod.Engines, eng_mod.cfg_key, eng_mod.RAGRT_ENGINES = FakeEngines, cfg_key, FAKE_RAGRT
eng_mod.RESID_ENGINES = FAKE_RESID
eng_mod.GRAPH_ENGINES = ("weft_graph",)
eng_mod.ADAPT_ENGINES = ("weft_adapt", "weft_oracle_adapt")
eng_mod.ORACLE_ENGINES = ("weft_oracle", "weft_oracle_adapt")
sys.modules["engines"] = eng_mod

import eval_config as C


def setup_dataset():
    d = os.path.join(TMP, "data")
    os.makedirs(d)
    C.DATASETS["msmarco"].update(eval_dir=d, sparse_csr_dir=d)
    with open(os.path.join(d, "queries.dev.small.tsv"), "w") as fq, open(os.path.join(d, "qrels.dev.small.tsv"), "w") as fr:
        for i in range(260):
            fq.write(f"{1000 + i}\tquery number {i} about topic {i % 13}\n")
            if i % 50 != 7:                      # a few queries have no qrels and must be skipped
                fr.write(f"{1000 + i}\t0\t{(i * 37) % N_DOCS}\t1\n")
    sys.argv = ["build_predicates.py", "--outdir", d, "--num_passages", str(N_DOCS)]
    import build_predicates
    build_predicates.main()


def main():
    setup_dataset()
    import build_ground_truth
    sys.argv = ["build_ground_truth.py", "--dataset", "msmarco", "--save_every", "50"]
    build_ground_truth.main()
    gt = np.load(C.dataset("msmarco")["gt_path"])
    assert set(gt.files) >= {"qids", "all", "syn5", "syn14", "syn30", "syn60", "scores_all"}
    assert len(gt["qids"]) == 260 - len([i for i in range(260) if i % 50 == 7])
    pred = np.load(C.dataset("msmarco")["predicates_path"])
    for row in gt["syn5"][:20]:                  # filtered GT only contains passing passages
        assert all(pred[p] & 1 for p in row if p >= 0)

    import run_benchmarks
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60",
                "--repeats", "1", "--max_filter_queries", "40", "--throughput_queries", "32", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    # second invocation: only the no-RT engine, merged into the same results
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60", "--engines", "ragrt_bf,ragrt_nofilter",
                "--stages", "sweep,select,test,breakdown,filtered,throughput,drops",
                "--repeats", "1", "--max_filter_queries", "40", "--throughput_queries", "32", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    R = json.load(open(os.path.join(os.environ["RAGRT_RESULTS"], "msmarco", "results.json")))
    assert set(R["sweep"]) == {"plaid", "ptms", "ragrt", "ragrt_bf", "ragrt_nofilter"}, R["sweep"].keys()
    assert "speedup_vs_ragrt_nofilter" in R["comparisons"]["plaid_default"]
    assert {"plaid", "ptms", "ragrt", "ragrt_bf", "ragrt_nofilter"} <= set(R["breakdown"]), R["breakdown"].keys()
    assert "speedup_vs_ragrt_bf" in R["comparisons"]["plaid_default"]
    assert all(set(v) == {"plaid", "ptms", "ragrt", "ragrt_bf", "ragrt_nofilter"} for v in R["filtered"].values())
    assert R["split"]["tune"] == 60 and R["split"]["test"] == R["split"]["total_with_gt"] - 60
    assert "plaid_default" in R["selection"] and R["primary_target"] == "plaid_default"
    for k, v in R["test"].items():
        lo, hi = v["gt_r10"]["ci95"]
        assert lo <= v["gt_r10"]["mean"] <= hi, k
    assert set(R["filtered"]) == {"syn5", "syn14", "syn30", "syn60"}
    assert R["drops"] and all(s["rays"] > 0 for s in R["drops"].values())
    assert "ragrt_pipelined_qps" in R["throughput"]

    # E-sweep style run: other index dir, RAGRT engines only, plaid/ptms imported
    src = os.path.join(os.environ["RAGRT_RESULTS"], "msmarco")
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60",
                "--results_subdir", "E1024", "--sparse_csr_dir", C.DATASETS["msmarco"]["sparse_csr_dir"],
                "--baseline_from", src, "--geometry", "fan", "--stage3", "dense", "--rerank", "simt",
                "--repeats", "1", "--max_filter_queries", "40", "--throughput_queries", "32", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    last = FakeEngines.calls[-1]
    assert last["geometry"] == "fan" and last["stage3"] == "dense" and last["rerank"] == "simt" and last["load_ragrt"]
    R2 = json.load(open(os.path.join(src, "E1024", "results.json")))
    assert set(R2["sweep"]) == {"plaid", "ptms", "ragrt", "ragrt_bf"}, R2["sweep"].keys()
    assert R2["sweep"]["plaid"] == R["sweep"]["plaid"], "plaid sweep must be imported unchanged"
    assert R2["meta"]["baselines_from"]["engines"] == ["plaid", "ptms"]
    assert R2["meta"]["ragrt"]["calibrated_on"] == 60
    assert "speedup_vs_ptms" in R2["comparisons"][R2["primary_target"]]
    assert R2["breakdown"]["ptms"] == R["breakdown"]["ptms"]

    # ablation engines merged into the main run, plus the profiler attribution stage
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60",
                "--engines", "ragrt_centroid,muvera", "--muvera_dir", TMP,
                "--stages", "sweep,select,test,breakdown,throughput,attribution",
                "--repeats", "1", "--max_filter_queries", "40", "--throughput_queries", "32", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    assert FakeEngines.calls[-1]["muvera_dir"] == TMP
    R = json.load(open(os.path.join(os.environ["RAGRT_RESULTS"], "msmarco", "results.json")))
    assert {"ragrt_centroid", "muvera"} <= set(R["sweep"]) and "muvera" in R["breakdown"]
    att = [a for c in R["comparisons"].values() for a in c.get("attribution", [])]
    steps = {(a["slow"], a["fast"]) for a in att}
    assert {("ragrt_centroid", "ragrt_bf"), ("ragrt_nofilter", "ragrt_bf"), ("ragrt_bf", "ragrt"),
            ("muvera", "ragrt_bf"), ("ptms", "ragrt_bf")} <= steps, steps
    assert all("gt_r10_diff" in a for a in att)
    assert {"ragrt_centroid", "muvera"} <= set(R["attribution"]), R["attribution"].keys()
    assert R["primary_fallback"] == {"muvera": "gt_r10>=0.90"}, R["primary_fallback"]

    # plain-IVF style run: other index, every other engine imported, only ragrt_ivf rerun
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60",
                "--results_subdir", "ivf", "--sparse_csr_dir", C.DATASETS["msmarco"]["sparse_csr_dir"],
                "--baseline_from", src, "--import_engines", "plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid",
                "--engines", "ragrt,ragrt_ivf,weft,weft_noimpute", "--stages", "sweep,select,test",
                "--repeats", "1", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    R3 = json.load(open(os.path.join(src, "ivf", "results.json")))
    assert R3["sweep"]["ragrt"] == R["sweep"]["ragrt"], "ragrt must be imported, not rerun on the IVF index"
    steps = {(a["slow"], a["fast"]) for a in R3["comparisons"]["plaid_default"]["attribution"]}
    assert {("ptms", "ragrt_ivf"), ("ragrt_ivf", "ragrt_nofilter"), ("ragrt_ivf", "ragrt_bf"),
            ("ragrt_ivf", "weft_noimpute"), ("weft_noimpute", "weft"), ("ptms", "weft")} <= steps, steps
    sys.argv = ["plot_results.py", "--dataset", "msmarco", "--results_subdir", "ivf"]
    import plot_results
    plot_results.main()
    assert "Attribution chain" in open(os.path.join(src, "ivf", "summary.md")).read()

    # residual-correction ablation on a token-level index, on top of the ivf run
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60",
                "--results_subdir", "resid", "--sparse_csr_dir", C.DATASETS["msmarco"]["sparse_csr_dir"],
                "--baseline_from", os.path.join(src, "ivf"),
                "--import_engines", "plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft,weft_noimpute",
                "--core", "plaid,ptms,weft",
                "--engines", "weft_tok,weft_resid,weft_resid_norr,weft_graph", "--stages", "sweep,select,test,breakdown,throughput",
                "--repeats", "1", "--max_filter_queries", "40", "--throughput_queries", "32", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    R4 = json.load(open(os.path.join(src, "resid", "results.json")))
    assert R4["sweep"]["weft"] == R3["sweep"]["weft"], "weft must be imported"
    assert len(R4["sweep"]["weft_resid_norr"]) == 2 and all(p["params"]["ndocs"] == C.TOP_K for p in R4["sweep"]["weft_resid_norr"])
    steps = {(a["slow"], a["fast"]) for c in R4["comparisons"].values() for a in c.get("attribution", [])}
    assert {("weft", "weft_tok"), ("weft_tok", "weft_resid"), ("weft_resid", "weft_resid_norr")} <= steps, steps
    assert R4["breakdown"]["weft_resid_norr"]["s4_tms"] == 0.0
    assert set(R4["breakdown"]["weft_graph"]) == {"total"} and "weft_graph_pipelined_qps" not in R4["throughput"]
    tp = R4["throughput"]
    assert "weft_tok_pipelined_qps" in tp and not any(k.startswith("weft_resid") and "pipelined" in k for k in tp), tp.keys()
    sys.argv = ["plot_results.py", "--dataset", "msmarco", "--results_subdir", "resid"]
    plot_results.main()

    # query-correlated filters: per-query masks, merged into the ground truth, then the filtered
    # stage for a new engine plus imported baselines (--filtered_extra), restricted with --filters
    gt_path = C.dataset("msmarco")["gt_path"]
    qids_gt = [str(q) for q in np.load(gt_path)["qids"]]
    corr = {"corr_near": {q: 1 << 0 for q in qids_gt}, "corr_far": {q: 1 << (1 + int(q) % 2) for q in qids_gt}}
    json.dump({"k": 2, "bit_base": 0, "filters": corr}, open(os.path.join(src, "corr_filters.json"), "w"))
    before = np.load(gt_path)["syn5"].copy()
    sys.argv = ["build_ground_truth.py", "--dataset", "msmarco", "--merge"]
    build_ground_truth.main()
    gt2 = np.load(gt_path)
    assert {"corr_near", "corr_far", "syn5", "all"} <= set(gt2.files) and np.array_equal(gt2["syn5"], before)
    assert np.array_equal(gt2["corr_near"], gt2["syn5"]), "same mask, same exact answer"
    for i, q in enumerate(qids_gt[:20]):
        assert all(pred[p] & corr["corr_far"][q] for p in gt2["corr_far"][i] if p >= 0)
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60", "--results_subdir", "resid",
                "--engines", "weft_graph", "--filtered_extra", "plaid,ptms", "--filters", "corr_near,corr_far",
                "--stages", "filtered", "--repeats", "1", "--max_filter_queries", "40", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    R5 = json.load(open(os.path.join(src, "resid", "results.json")))
    assert {"plaid", "ptms", "weft_graph"} <= set(R5["filtered"]["corr_far"]) and "corr_mid" not in R5["filtered"]

    # hash / adaptive engines + the post-filter budget sweep (default filters: those present among corr_mid, corr_far, syn5)
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60", "--results_subdir", "tue",
                "--baseline_from", os.path.join(src, "resid"), "--import_engines", "plaid,ptms,weft,weft_graph",
                "--core", "plaid,ptms,weft", "--engines", "weft_hash,weft_adapt",
                "--stages", "sweep,select,test,throughput,filtered,filter_budget", "--repeats", "1",
                "--max_filter_queries", "40", "--budget_queries", "30", "--throughput_queries", "32", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    R6 = json.load(open(os.path.join(src, "tue", "results.json")))
    assert set(R6["filter_budget"]) == {"corr_far", "syn5"}, R6["filter_budget"].keys()
    assert all(len(R6["filter_budget"][f][e]) == 4 for f in R6["filter_budget"] for e in ("plaid", "ptms"))
    assert "weft_adapt_pipelined_qps" not in R6["throughput"] and "weft_hash_pipelined_qps" in R6["throughput"]
    steps = {(a["slow"], a["fast"]) for c in R6["comparisons"].values() for a in c.get("attribution", [])}
    assert ("weft", "weft_hash") in steps, steps
    sys.argv = ["plot_results.py", "--dataset", "msmarco", "--results_subdir", "tue"]
    plot_results.main()
    md6 = open(os.path.join(src, "tue", "summary.md")).read()
    assert "Post-filtering with larger candidate budgets" in md6 and "## Filtered search" in md6

    # oracle engines: no sweep of their own, run in the filtered stage at weft / weft_adapt's selected
    # configs (PARAMS_FROM), next to re-run weft / weft_adapt (--filtered_extra)
    sys.argv = ["run_benchmarks.py", "--dataset", "msmarco", "--quick", "--tune_size", "60", "--results_subdir", "oracle",
                "--baseline_from", os.path.join(src, "tue"), "--import_engines", "plaid,ptms,weft,weft_graph,weft_hash,weft_adapt",
                "--core", "plaid,ptms,weft", "--engines", "weft_oracle,weft_oracle_adapt", "--filtered_extra", "weft,weft_adapt",
                "--filters", "corr_near,corr_far", "--stages", "select,filtered", "--repeats", "1",
                "--max_filter_queries", "40", "--targets", "0.5,0.9"]
    run_benchmarks.main()
    R7 = json.load(open(os.path.join(src, "oracle", "results.json")))
    assert {"weft", "weft_adapt", "weft_oracle", "weft_oracle_adapt"} <= set(R7["filtered"]["corr_far"]), R7["filtered"]["corr_far"].keys()
    sys.argv = ["plot_results.py", "--dataset", "msmarco", "--results_subdir", "oracle"]
    plot_results.main()
    assert "weft_oracle" in open(os.path.join(src, "oracle", "summary.md")).read()

    sys.argv = ["plot_results.py", "--dataset", "msmarco"]
    plot_results.main()
    out = os.path.join(os.environ["RAGRT_RESULTS"], "msmarco")
    for f in ["fig_pareto.png", "fig_operating_point.png", "fig_breakdown.png", "fig_latency_cdf.png",
              "fig_filtered.png", "fig_throughput.png", "summary.md"]:
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    md = open(os.path.join(out, "summary.md")).read()
    assert "Where the time goes" in md and "Attribution chain" in md
    print(md)
    print("SMOKE TEST PASSED", out)


if __name__ == "__main__":
    main()
