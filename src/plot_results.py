"""
Figures and a markdown summary from run_benchmarks.py output. Nothing is measured here.

  python3 plot_results.py --dataset lotte

Writes to <results>/<dataset>/: fig_pareto.png, fig_operating_point.png,
fig_breakdown.png, fig_latency_cdf.png, fig_filtered.png, fig_throughput.png, summary.md
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import eval_config as C

# Fixed per engine (color follows the entity). Validated palette slots 1-3; aqua is
# below 3:1 contrast on white, so every series also has its own marker, a legend and
# direct labels, and summary.md carries the numbers as a table.
STYLE = {
    "ragrt": {"color": "#2a78d6", "marker": "o", "label": "RT-core design"},
    "plaid": {"color": "#eb6834", "marker": "s", "label": "PLAID"},
    "ptms":  {"color": "#1baf7a", "marker": "D", "label": "PLAID+TMS"},
    "plaid_fn": {"color": "#b5532a", "marker": "s", "label": "PLAID, filter on Stage 1 (filter_fn)"},
    "ptms_fn": {"color": "#127a55", "marker": "D", "label": "PLAID+TMS, filter on Stage 1"},
    "ragrt_bf": {"color": "#eda100", "marker": "^", "label": "RT-core design, CUDA-core Stage 1"},
    "ragrt_nofilter": {"color": "#8c564b", "marker": "v", "label": "RT-core design, codebook filter off"},
    "ragrt_centroid": {"color": "#7a5195", "marker": "P", "label": "RT-core design, no codeword score"},
    "ragrt_ivf": {"color": "#5f6b6d", "marker": "X", "label": "Plain IVF on our kernels"},
    "muvera": {"color": "#c2185b", "marker": "*", "label": "MUVERA (FDE + TMS)"},
    "weft_noimpute": {"color": "#0e7c86", "marker": "h", "label": "RAGRT, no imputation"},
    "weft": {"color": "#1b1b8f", "marker": "p", "label": "RAGRT (plain IVF, native prep, imputation)"},
    "weft_tok": {"color": "#7a5195", "marker": "<", "label": "RAGRT, token-level index"},
    "weft_resid": {"color": "#bc5090", "marker": ">", "label": "RAGRT + residual correction"},
    "weft_resid_norr": {"color": "#ef5675", "marker": "d", "label": "WARP-style (residuals, no rerank)"},
    "weft_graph": {"color": "#000060", "marker": "*", "label": "RAGRT, CUDA graphs"},
    "weft_hash": {"color": "#2f4b7c", "marker": "P", "label": "RAGRT, hash Stage 3"},
    "weft_adapt": {"color": "#665191", "marker": "X", "label": "RAGRT, adaptive probing"},
    "weft_pairs": {"color": "#003f5c", "marker": "v", "label": "RAGRT, 32-bit pair sort"},
    "weft_oracle": {"color": "#d45087", "marker": "o", "label": "RAGRT, filter-aware (oracle) probing"},
    "weft_oracle_adapt": {"color": "#a05195", "marker": "D", "label": "RAGRT, oracle + adaptive probing"},
    "weft_oimp": {"color": "#f95d6a", "marker": "^", "label": "RAGRT, filter-aware imputation only"},
    "weft_oskip": {"color": "#ff7c43", "marker": "s", "label": "RAGRT, skip empty centroids only"},
}
ALL_ENGINES = ["plaid", "ptms", "plaid_fn", "ptms_fn", "ragrt", "ragrt_bf", "ragrt_nofilter", "ragrt_centroid", "ragrt_ivf",
               "weft_noimpute", "weft", "weft_tok", "weft_resid", "weft_resid_norr", "weft_graph", "weft_hash", "weft_adapt", "weft_pairs",
               "weft_oracle", "weft_oracle_adapt", "weft_oimp", "weft_oskip", "muvera"]
ENGINES = ALL_ENGINES          # narrowed in main() to the engines present in results.json
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print(f"  wrote {name}")


def fig_pareto(R, out, metric_name):
    sweep, test, sel = R["sweep"], R.get("test", {}), R.get("selection", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, qkey, ylabel in [(axes[0], "gt_r10", "Recall of exact top-10 (TUNE)"),
                             (axes[1], "official", f"{metric_name} (TUNE)")]:
        for e in ENGINES:
            if e not in sweep:
                continue
            st = STYLE[e]
            pts = sweep[e]
            ax.scatter([p["lat_median"] for p in pts], [p[qkey] for p in pts], s=10,
                       color=st["color"], alpha=0.25, linewidths=0)
            front, best = [], -1
            for p in sorted(pts, key=lambda p: p["lat_median"]):
                if p[qkey] > best:
                    front.append(p); best = p[qkey]
            ax.plot([p["lat_median"] for p in front], [p[qkey] for p in front], color=st["color"],
                    marker=st["marker"], markersize=4, label=st["label"])
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel("Retrieval latency, median ms (log)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right")
    fig.suptitle("Quality vs latency over the TUNE sweep (curves = Pareto frontier)", color=INK)
    save(fig, out, "fig_pareto.png")


def fig_operating_point(R, out, metric_name):
    target = R["primary_target"]
    sel, test = R["selection"][target], R["test"]
    engines = [e for e in ENGINES if sel.get(e) and f"{e}|{sel[e]['key']}" in test]
    rows = [test[f"{e}|{sel[e]['key']}"] for e in engines]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    x = np.arange(len(engines))
    labels = [STYLE[e]["label"] for e in engines]
    colors = [STYLE[e]["color"] for e in engines]
    lat = [r["lat"]["median"] for r in rows]
    axes[0].bar(x, lat, color=colors, width=0.6)
    for i, v in enumerate(lat):
        axes[0].text(i, v, f"{v:.2f} ms", ha="center", va="bottom", color=INK, fontsize=8)
    axes[0].set_ylabel("Median retrieval latency (ms, TEST)")
    for ax, k, yl in [(axes[1], "gt_r10", "Recall of exact top-10"), (axes[2], "official", metric_name)]:
        m = [r[k]["mean"] for r in rows]
        err = np.array([[r[k]["mean"] - r[k]["ci95"][0], r[k]["ci95"][1] - r[k]["mean"]] for r in rows]).T
        ax.bar(x, m, color=colors, width=0.6, yerr=err, ecolor=INK2, capsize=3)
        for i, v in enumerate(m):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=8)
        ax.set_ylabel(yl + " (TEST, 95% CI)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis="x", visible=False)
    fig.suptitle(f"Operating point: fastest TUNE config reaching PLAID-default quality "
                 f"(exact R@10 >= {sel['target_gt_r10']:.3f})", color=INK)
    save(fig, out, "fig_operating_point.png")


def fig_breakdown(R, out):
    b = R["breakdown"]
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    bars = []
    if "plaid" in b:
        bars.append(("PLAID", [("Stages 1-3", b["plaid"]["s1_3"]), ("Stage 4 (est.)", b["plaid"]["s4_est"])]))
    if "ptms" in b:
        bars.append(("PLAID+TMS", [("Stages 1-3", b["ptms"]["s1_3"]), ("Stage 4 TMS", b["ptms"]["s4"])]))
    if "muvera" in b:
        bars.append(("MUVERA", [("FDE MIPS", b["muvera"]["s1_3"]), ("Stage 4 TMS", b["muvera"]["s4"])]))
    for e, name, s1 in (("ragrt", "RT-core design", "1 RT"), ("ragrt_bf", "RT-core, CUDA Stage 1", "1 brute"),
                        ("ragrt_nofilter", "filter off", "1 all"), ("ragrt_centroid", "no codeword score", "1 brute"),
                        ("ragrt_ivf", "plain IVF, our kernels", "1 all"),
                        ("weft_noimpute", "RAGRT, no imputation", "1 all"), ("weft", "RAGRT", "1 all"),
                        ("weft_tok", "RAGRT token index", "1 all"), ("weft_resid", "RAGRT + residuals", "1 all"),
                        ("weft_resid_norr", "WARP-style, no rerank", "1 all"), ("weft_hash", "RAGRT, hash Stage 3", "1 all"),
                        ("weft_adapt", "RAGRT, adaptive", "1 all"), ("weft_pairs", "RAGRT, pair sort", "1 all")):
        if e in b:
            r = b[e]
            bars.append((name, [("0b prep", r["s0b_prep"]), (s1, r["s1_rt"]), ("2 gather", r["s2_gather"]),
                                ("3 score", r["s3_score"]), ("4 TMS", r["s4_tms"])]))
    if "weft_graph" in b:          # one graph per phase: no per-stage events, total only
        bars.append(("RAGRT, CUDA graphs", [("total (graph)", b["weft_graph"]["total"])]))
    if not bars:
        print("  skipping fig_breakdown (no breakdown in this run)")
        return
    shades = ["#cde2fb", "#9ec5f4", "#6ba6ea", "#2a78d6", "#1c5299"]
    longest = max(sum(v for _, v in segs) for _, segs in bars)
    for i, (name, segs) in enumerate(bars):
        left = 0.0
        for j, (seg, v) in enumerate(segs):
            ax.barh(i, v, left=left, color=shades[j % len(shades)], edgecolor="white", linewidth=2, height=0.55)
            if v > 0.07 * longest:
                ax.text(left + v / 2, i, f"{seg}\n{v:.2f}", ha="center", va="center", fontsize=7,
                        color=INK if j < 2 else "white")
            left += v
        ax.text(left, i, f"  {left:.2f} ms", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([n for n, _ in bars]); ax.invert_yaxis()
    enc = b.get("stage0a_encode_median", R.get("stage0a_encode", {}).get("median"))
    ax.set_xlabel("Median ms per query (TEST). Stage 0a query encoding"
                  + (f", {enc:.2f} ms," if enc is not None else "") + " is common to all and excluded")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_breakdown.png")


def fig_latency_cdf(R, perq, out):
    sel, test = R["selection"][R["primary_target"]], R["test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for e in ENGINES:
        if not sel.get(e) or f"{e}|{sel[e]['key']}|lat" not in perq:
            continue
        k = f"{e}|{sel[e]['key']}"
        lat = np.sort(perq[f"{k}|lat"])
        st = STYLE[e]
        axes[0].plot(lat, np.arange(1, len(lat) + 1) / len(lat), color=st["color"], label=st["label"])
        ntok = perq[f"{k}|ntok"]
        r = np.corrcoef(ntok, perq[f"{k}|lat"])[0, 1]
        axes[1].scatter(ntok + (ENGINES.index(e) - 1) * 0.15, perq[f"{k}|lat"], s=8, color=st["color"],
                        marker=st["marker"], alpha=0.35, linewidths=0, label=f"{st['label']} (r = {r:+.2f})")
    axes[0].set_xlabel("Retrieval latency (ms)"); axes[0].set_ylabel("Fraction of TEST queries")
    axes[0].legend(loc="lower right")
    axes[1].set_xlabel("Query tokens (attention mask)"); axes[1].set_ylabel("Retrieval latency (ms)")
    axes[1].legend(loc="upper left", markerscale=2)
    save(fig, out, "fig_latency_cdf.png")


def fig_filtered(R, out):
    F = R["filtered"]
    syn = [(int(k[3:]), k) for k in F if k.startswith("syn")]
    syn.sort()
    if not syn:
        return
    test, sel = R["test"], R["selection"][R["primary_target"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for e in ENGINES:
        if e not in F[syn[0][1]]:
            continue
        st = STYLE[e]
        xs = [p for p, _ in syn] + [100]
        unf = test.get(f"{e}|{sel[e]['key']}") if sel.get(e) else None
        if unf is None:
            continue
        r10 = [F[k][e]["gt_r10"]["mean"] for _, k in syn] + [unf["gt_r10"]["mean"]]
        lat = [F[k][e]["lat"]["median"] for _, k in syn] + [unf["lat"]["median"]]
        starve = [F[k][e]["frac_under_10_results"] for _, k in syn] + [unf["frac_under_10_results"]]
        for ax, ys in zip(axes, [r10, lat, starve]):
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=5, label=st["label"])
    axes[0].set_ylabel("Recall of exact filtered top-10")
    axes[1].set_ylabel("Median latency (ms)")
    axes[2].set_ylabel("Fraction of queries with < 10 results")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks([5, 14, 30, 60, 100]); ax.set_xticklabels(["5", "14", "30", "60", "100"])
        ax.set_xlabel("Passages passing the filter (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Synthetic filters (independent of relevance), primary operating point, TEST", color=INK)
    save(fig, out, "fig_filtered.png")


def fig_throughput(R, out):
    T = R["throughput"]
    if "n_queries" not in T:          # only imported baselines, no throughput stage in this run
        print("  skipping fig_throughput (no throughput stage in this run)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
    for e in ENGINES:
        k = f"{e}_sequential_qps"
        if k in T:
            ax.axhline(T[k], color=STYLE[e]["color"], linewidth=1.5, linestyle="--")
            ax.text(1, T[k], f" {STYLE[e]['label']} sequential {T[k]:.0f} QPS", va="bottom", fontsize=8, color=INK2)
    for e in ("ragrt", "ragrt_bf", "ragrt_nofilter", "ragrt_centroid", "ragrt_ivf", "weft_noimpute", "weft", "weft_tok"):
        k = f"{e}_pipelined_qps"
        if k in T:
            B = sorted(int(b) for b in T[k])
            ax.plot(B, [T[k][str(b)] for b in B], color=STYLE[e]["color"], marker=STYLE[e]["marker"],
                    markersize=5, label=f"{STYLE[e]['label']} pipelined")
    ax.set_xscale("log", base=2); ax.set_xlabel("Queries per pipelined call")
    ax.set_ylabel("Throughput (queries / s)")
    ax.legend(loc="lower right")
    ax.set_title(f"Throughput over {T['n_queries']} TEST queries (not latency)", color=INK)
    save(fig, out, "fig_throughput.png")


def ragrt_opts_line(R):
    o = R["meta"].get("ragrt") or {}
    if not o:
        return "- RT-core design options: (not recorded; pre-Oct-2026 run: fan geometry, dense stage 3, simt rerank)"
    idx = o.get("index", {})
    line = (f"- RT-core design: E={idx.get('num_entries', '?')}, geometry {o.get('geometry')}, stage 3 {o.get('stage3')}, "
            f"rerank {o.get('rerank')} (also PLAID+TMS), index {o.get('sparse_csr_dir', '?')}")
    cal = o.get("calibration")
    if cal:
        line += "; polar thresholds at quantile {q}: ".format(q=cal["quantile"]) + ", ".join(
            f"k={lv['k']}: {lv['hits_per_ray_mean']:.0f} hits/ray" for lv in cal["levels"])
    b = R["meta"].get("baselines_from")
    if b:
        line += f"\n- PLAID / PLAID+TMS imported from {b['dir']} (rerank {((b.get('meta') or {}).get('ragrt') or {}).get('rerank', 'simt')})"
    return line


def summary_md(R, out, metric_name):
    L = [f"# RAGRT results: {R['meta']['args']['dataset']}", "",
         f"- {R['meta']['date']} on {R['meta']['gpu']}, git {R['meta']['git']}",
         f"- queries: {R['split']['tune']:,} tune / {R['split']['test']:,} test (seeded hash split)",
         f"- stage 0a query encoding (common to all engines): median {R['stage0a_encode']['median']:.2f} ms",
         ragrt_opts_line(R),
         "", "## Operating points (TEST, 95% bootstrap CI)", "",
         f"| target | engine | config | median ms | p99 ms | exact R@10 | {metric_name} |",
         "|---|---|---|---|---|---|---|"]
    for name, s in R.get("selection", {}).items():
        for e in ENGINES:
            if not s.get(e):
                L.append(f"| {name} | {STYLE[e]['label']} | unreachable on TUNE | | | | |")
                continue
            t = R["test"].get(f"{e}|{s[e]['key']}")
            if not t:
                continue
            ci = lambda m: f"{t[m]['mean']:.4f} [{t[m]['ci95'][0]:.4f}, {t[m]['ci95'][1]:.4f}]"
            L.append(f"| {name} | {STYLE[e]['label']} | {s[e]['key']} | {t['lat']['median']:.2f} | "
                     f"{t['lat']['p99']:.2f} | {ci('gt_r10')} | {ci('official')} |")
    L += ["", "## RT-core design vs baselines at each target", "",
          "| target | speedup vs PLAID | speedup vs PLAID+TMS | exact R@10 diff vs PLAID+TMS | "
          f"{metric_name} diff vs PLAID+TMS | RT vs no-RT speedup | exact R@10 diff vs no-RT | "
          "vs codebook filter off: speedup | exact R@10 diff |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, c in R.get("comparisons", {}).items():
        f = lambda k: f"{c[k]['mean']:+.4f} [{c[k]['ci95'][0]:+.4f}, {c[k]['ci95'][1]:+.4f}]" if k in c else "n/a"
        sp = lambda k: f"{c[k]:.2f}x" if k in c else "n/a"
        L.append(f"| {name} | {sp('speedup_vs_plaid')} | {sp('speedup_vs_ptms')} | "
                 f"{f('gt_r10_diff_vs_ptms')} | {f('official_diff_vs_ptms')} | {sp('speedup_vs_ragrt_bf')} | "
                 f"{f('gt_r10_diff_vs_ragrt_bf')} | {sp('speedup_vs_ragrt_nofilter')} | {f('gt_r10_diff_vs_ragrt_nofilter')} |")
    att = [(name, c["attribution"]) for name, c in R.get("comparisons", {}).items() if c.get("attribution")]
    if att:
        L += ["", "## Attribution chain (latency ratio slow/fast at each target; quality diff = fast - slow)", "",
              f"| target | step | slow -> fast | latency ratio | exact R@10 diff | {metric_name} diff |",
              "|---|---|---|---|---|---|"]
        for name, rows in att:
            for a in rows:
                f = lambda k: f"{a[k]['mean']:+.4f} [{a[k]['ci95'][0]:+.4f}, {a[k]['ci95'][1]:+.4f}]" if k in a else "n/a"
                L.append(f"| {name} | {a['what']} | {a['slow']} -> {a['fast']} | {a['latency_ratio']:.2f}x | "
                         f"{f('gt_r10_diff')} | {f('official_diff')} |")
    if "filtered" in R:
        L += ["", "## Filtered search (recall@10 vs the exact filtered top-10, median latency)", "",
              "| filter | " + " | ".join(e for e in ALL_ENGINES if any(e in d for d in R["filtered"].values())) + " |"]
        cols = [e for e in ALL_ENGINES if any(e in d for d in R["filtered"].values())]
        L.append("|---" * (len(cols) + 1) + "|")
        for f, d in R["filtered"].items():
            cell = lambda e: (f"{d[e]['gt_r10']['mean']:.3f} ({d[e]['lat']['median']:.2f} ms"
                              + (f", nc {d[e]['adapt_nc_mean']:.0f}" if "adapt_nc_mean" in d[e] else "") + ")") if e in d else ""
            L.append(f"| {f} | " + " | ".join(cell(e) for e in cols) + " |")
    if R.get("filtered_pairs"):
        P = R["filtered_pairs"]
        cols = sorted({k for d in P.values() for k in d}, key=lambda k: list(next(iter(P.values()))).index(k)
                      if k in next(iter(P.values())) else 99)
        L += ["", "## Filtered search: paired recall@10 difference (b - a on the same queries, 95% bootstrap CI)", "",
              "| filter | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for f, d in P.items():
            L.append(f"| {f} | " + " | ".join(f"{d[c]['mean']:+.4f} [{d[c]['ci95'][0]:+.4f}, {d[c]['ci95'][1]:+.4f}]"
                                              if c in d else "" for c in cols) + " |")
    if "filter_budget" in R:
        L += ["", "## Post-filtering with larger candidate budgets (PLAID / PLAID+TMS)", "",
              "| filter | engine | budget (ncells, threshold, ndocs) | recall@10 | median ms |", "|---|---|---|---|---|"]
        for f, d in R["filter_budget"].items():
            for e, rows in d.items():
                for k, v in rows.items():
                    L.append(f"| {f} | {e} | {k} | {v['gt_r10']['mean']:.3f} | {v['lat']['median']:.2f} |")
    if "attribution" in R:
        L += ["", "## Where the time goes (torch.profiler, primary point, per query)", "",
              "| engine | wall ms | GPU-busy ms | host/idle ms | kernels | memcpy/memset | host syncs |",
              "|---|---|---|---|---|---|---|"]
        for e, o in R["attribution"].items():
            L.append(f"| {e} | {o['wall_ms']:.2f} | {o['gpu_busy_ms']:.2f} | {max(0, o['wall_ms'] - o['gpu_busy_ms']):.2f} | "
                     f"{o['kernels']:.1f} | {o['memcpys']:.1f} | {o['syncs']:.1f} |")
    if "drops" in R:
        L += ["", "## RT-core design stage counters (diagnostic pass, counters off during timing)", "",
              "| config | stage-1 hits/ray | rays short of k | rays over hit cap | tasks dropped (MAX_TASKS) | "
              "MIN_BASE_SCORE drops | postings/query |",
              "|---|---|---|---|---|---|---|"]
        for k, s in R["drops"].items():
            g = lambda n, fmt: format(s[n], fmt) if n in s else "n/a"
            L.append(f"| {k} | {g('hits_per_ray', '.1f')} | {g('frac_rays_short_of_k', '.1%')} | "
                     f"{s['frac_rays_overflow']:.2%} | {s['frac_tasks_dropped']:.2%} | {s['min_score_drops']:,} | "
                     f"{g('postings_per_query', ',.0f')} |")
    dpath = os.path.join(out, "dram.json")
    if os.path.exists(dpath):
        D = json.load(open(dpath))
        L += ["", f"## DRAM per query (Nsight Compute, {D['n_queries']} TEST queries, calibrated)", "",
              "| cache mode | calibration ratio | PLAID MB/q | PLAID+TMS MB/q | RT-core MB/q |", "|---|---|---|---|---|"]
        for mode in ("none", "all"):
            m = D.get(f"cache_{mode}")
            if m:
                g = lambda e: f"{m[e]['mb_per_query_calibrated']:.1f}" if e in m else "n/a"
                L.append(f"| {'warm (L2 kept)' if mode == 'none' else 'cold (L2 flushed per kernel)'} | "
                         f"{m['calibration_ratio']:.3f} | {g('plaid')} | {g('ptms')} | {g('ragrt')} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("  wrote summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--results_subdir", default=None)
    args = ap.parse_args()
    ds = C.dataset(args.dataset)
    out = os.path.join(ds["results_dir"], args.results_subdir) if args.results_subdir else ds["results_dir"]
    R = json.load(open(os.path.join(out, "results.json")))
    perq_path = os.path.join(out, "test_perquery.npz")
    perq = dict(np.load(perq_path)) if os.path.exists(perq_path) else {}
    metric = ds["official_metric"].replace("mrr", "MRR").replace("success", "Success")
    global ENGINES
    present = set(R.get("sweep", {})) | {t.split("|")[0] for t in R.get("test", {})}
    ENGINES = [e for e in ALL_ENGINES if e in present] or ALL_ENGINES
    if "sweep" in R:
        fig_pareto(R, out, metric)
    if "test" in R and "selection" in R:
        fig_operating_point(R, out, metric)
        if perq:
            fig_latency_cdf(R, perq, out)
    if "breakdown" in R:
        fig_breakdown(R, out)
    if "filtered" in R and "test" in R:
        fig_filtered(R, out)
    if "throughput" in R:
        fig_throughput(R, out)
    summary_md(R, out, metric)


if __name__ == "__main__":
    main()
