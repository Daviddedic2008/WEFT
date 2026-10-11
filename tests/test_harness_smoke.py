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


def cfg_key(engine, p):
    return (f"nc={p['nc']} eids={p['eids']} ndocs={p['ndocs']}" if engine in ("ragrt", "ragrt_bf", "ragrt_nofilter")
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

    def search(self, engine, Qf, ntok, p, mask=0):
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
eng_mod.Engines, eng_mod.cfg_key, eng_mod.RAGRT_ENGINES = FakeEngines, cfg_key, ("ragrt", "ragrt_bf", "ragrt_nofilter")
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

    import plot_results
    sys.argv = ["plot_results.py", "--dataset", "msmarco"]
    plot_results.main()
    out = os.path.join(os.environ["RAGRT_RESULTS"], "msmarco")
    for f in ["fig_pareto.png", "fig_operating_point.png", "fig_breakdown.png", "fig_latency_cdf.png",
              "fig_filtered.png", "fig_throughput.png", "summary.md"]:
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    print(open(os.path.join(out, "summary.md")).read())
    print("SMOKE TEST PASSED", out)


if __name__ == "__main__":
    main()
