"""
Paper-style figures, in the spirit of the old generate_figures.py. To measure everything fresh and
then plot, use run_figures.sh (one new run, no imported results). This script itself only plots: every number comes from results.json files written by run_benchmarks.py,
bench_fastplaid.py, run_attribution.sh (MUVERA), baselines/warp_lotte.py and baselines/juno.

Unlike plot_results.py (one run, every engine), this draws a curated comparison:
PLAID, PLAID+TMS, FastPLAID, MUVERA, WARP (LoTTE, CPU) and RAGRT, optionally the RT-core design.

  python3 make_figures.py --dataset lotte   --main v3_ablation_oracle2
  python3 make_figures.py --dataset msmarco --main v1_oracle2

Baseline runs are found automatically (newest <results>/<ds>/*_muvera and fastplaid_nbits2, and
<results>/lotte/warp_lotte_pooled.json); override with --muvera / --fastplaid / --warp, or pass
"none". Writes <results>/<ds>/<main>/figures/: fig1_headline ... fig9_juno, plus figures.md
(the numbers behind every bar and point, so nothing has to be read off a plot).
"""
import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_config as C

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Colors follow the old generate_figures.py (PLAID red, PLAID+TMS orange, ours blue); new baselines added.
STYLE = {
    "plaid":     ("#c0392b", "s", "PLAID"),
    "ptms":      ("#e67e22", "D", "PLAID+TMS"),
    "fastplaid": ("#8e44ad", "^", "FastPLAID"),
    "muvera":    ("#c2185b", "*", "MUVERA (FDE + TMS)"),
    "warp":      ("#16a085", "v", "WARP (XTR, CPU)"),
    "ragrt":     ("#7f8c8d", "X", "RT-core design"),
    "ours":      ("#2980b9", "o", "RAGRT"),
    "ours_adapt":  ("#1b4f72", "P", "RAGRT, adaptive probing"),
    "ours_oracle": ("#5dade2", "h", "RAGRT, filter-aware + adaptive"),
}
FILTER_LABEL = {"syn5": "random 5%", "syn14": "random 14%", "syn30": "random 30%", "syn60": "random 60%",
                "own_domain": "own domain", "corr_near": "corr near", "corr_mid": "corr mid", "corr_far": "corr far"}
FILTER_ORDER = ["syn5", "syn14", "syn30", "syn60", "own_domain", "corr_near", "corr_mid", "corr_far"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2.0, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ------------------------------------------------------------------ loading
def jload(p):
    return json.load(open(p)) if p and os.path.exists(p) else None


def newest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    return hits[-1] if hits else None


def val(x):
    """Summaries store {'mean': m, 'ci95': [lo, hi]} or {'median': ...}; sweep points store floats."""
    if isinstance(x, dict):
        return x.get("mean", x.get("median"))
    return x


class Results:
    """One view over the main run plus the separately run baselines, keyed by display engine names."""

    def __init__(self, a, ds):
        root = ds["results_dir"]
        self.root = root
        self.M = jload(os.path.join(root, a.main, "results.json"))
        if self.M is None:
            raise SystemExit(f"FATAL: no results.json in {os.path.join(root, a.main)}")
        self.target = a.target or self.M["primary_target"]
        self.metric = ds["official_metric"]
        self.ids = {"plaid": "plaid", "ptms": "ptms", "ragrt": "ragrt", "ours": a.ours,
                    "ours_adapt": "weft_adapt", "ours_oracle": "weft_oracle_adapt", "muvera": "muvera"}
        mu = None if a.muvera == "none" else (a.muvera or newest(os.path.join(root, "*_muvera", "results.json")))
        self.MU = jload(mu if mu and mu.endswith(".json") else os.path.join(mu, "results.json") if mu else None)
        fp = None if a.fastplaid == "none" else (a.fastplaid or os.path.join(root, "fastplaid_nbits2", "results.json"))
        self.FP = jload(fp if fp and fp.endswith(".json") else os.path.join(fp, "results.json") if fp else None)
        wp = None if a.warp == "none" else (a.warp or os.path.join(root, "warp_lotte_pooled.json"))
        self.W = jload(wp)
        self.show_rt = a.with_rt
        self.sources = {"main": os.path.join(root, a.main), "muvera": mu, "fastplaid": fp, "warp": wp}
        # Results from a separate run older than the main run are flagged (and dropped with --drop_stale),
        # so a figure never silently mixes today's numbers with an old run's.
        t_main = os.path.getmtime(os.path.join(root, a.main, "results.json"))
        self.stale = []
        for name, attr in (("muvera", "MU"), ("fastplaid", "FP"), ("warp", "W")):
            p = self.sources[name]
            p = p if p is None or p.endswith(".json") else os.path.join(p, "results.json")
            if getattr(self, attr) is not None and os.path.getmtime(p) < t_main - 6 * 3600:
                self.stale.append(name)
                print(f"  WARNING: {name} results ({p}) are older than the main run"
                      + ("; dropped (--drop_stale)" if a.drop_stale else "; pass --drop_stale to leave them out"))
                if a.drop_stale:
                    setattr(self, attr, None)
        if self.MU is not None and self.M.get("sweep", {}).get("muvera"):
            self.MU = None                      # measured in the main run: use that, not a separate run

    def src(self, name):
        """Which results dict holds this engine, and its id there."""
        if name == "fastplaid":
            return self.FP, "fastplaid"
        if name == "muvera":
            if self.M.get("sweep", {}).get("muvera"):
                return self.M, "muvera"
            return self.MU, "muvera"
        return self.M, self.ids.get(name, name)

    def engines(self, names):
        out = []
        for n in names:
            if n == "ragrt" and not self.show_rt:
                continue
            R, e = self.src(n)
            if R and (e in R.get("sweep", {}) or any(k.startswith(e + "|") for k in R.get("test", {}))):
                out.append(n)
        return out

    def sweep(self, name):
        R, e = self.src(name)
        return (R or {}).get("sweep", {}).get(e, [])

    def point(self, name, target=None):
        """TEST summary at the selected operating point, or None when the engine cannot reach it."""
        R, e = self.src(name)
        if not R:
            return None
        t = target or self.target
        s = R.get("selection", {}).get(t, {}).get(e)
        if s is None and e in ("weft_adapt", "weft_oracle_adapt"):
            return None
        return R["test"].get(f"{e}|{s['key']}") if s else None

    def breakdown(self, name):
        R, e = self.src(name)
        return (R or {}).get("breakdown", {}).get(e)

    def filtered(self, name, f):
        if name == "fastplaid":
            return (self.FP or {}).get("filtered", {}).get(f)
        R, e = self.src(name)
        return (R or {}).get("filtered", {}).get(f, {}).get(e)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print(f"  wrote {name}")


def frontier(pts, key="gt_r10"):
    pts = sorted(pts, key=lambda p: p["lat_median"])
    best, out = -1, []
    for p in pts:
        if p.get(key) is not None and p[key] > best:
            best = p[key]
            out.append(p)
    return out


# ------------------------------------------------------------------ figures
def fig_headline(X, out, md):
    """Old fig1: latency with slowdown labels, quality, and best throughput at one operating point."""
    names = X.engines(["plaid", "ptms", "fastplaid", "muvera", "ragrt", "ours"])
    pts = {n: X.point(n) for n in names}
    names = [n for n in names if pts[n]]
    if "ours" not in names:
        print("  skipping fig1 (our engine has no TEST point at this target)")
        return
    T = X.M.get("throughput", {})
    qps = {n: None for n in names}
    for n in names:
        R, e = X.src(n)
        if n == "fastplaid":
            qps[n] = max((X.FP.get("throughput", {}).get("qps_by_batch") or {"": None}).values() or [None])
        else:
            TT = (R or {}).get("throughput", {})
            cands = [TT.get(f"{e}_sequential_qps")] + list((TT.get(f"{e}_pipelined_qps") or {}).values())
            cands = [c for c in cands if c]
            qps[n] = max(cands) if cands else None
    lat = [pts[n]["lat"]["median"] for n in names]
    r10 = [val(pts[n]["gt_r10"]) for n in names]
    off = [val(pts[n]["official"]) for n in names]
    ours = lat[names.index("ours")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
    x = np.arange(len(names))
    cols = [STYLE[n][0] for n in names]
    axes[0].bar(x, lat, color=cols)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Median latency per query (ms, log)")
    for i, v in enumerate(lat):
        axes[0].text(i, v, f"{v:.2f}\n{v / ours:.1f}x" if names[i] != "ours" else f"{v:.2f}", ha="center",
                     va="bottom", fontsize=7, color=INK)
    axes[1].bar(x, r10, color=cols); axes[1].set_ylabel("Recall of exact top-10")
    axes[1].set_ylim(min(r10) - 0.05, 1.0)
    axes[2].bar(x, off, color=cols); axes[2].set_ylabel(X.metric)
    axes[2].set_ylim(min(off) - 0.05, max(off) + 0.03)
    axes[3].bar(x, [q or 0 for q in qps.values()], color=cols); axes[3].set_ylabel("Best throughput (queries/s)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([STYLE[n][2] for n in names], rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Operating point {X.target}, TEST split, batch 1 latency", color=INK)
    save(fig, out, "fig1_headline.png")
    md.append(f"## fig1: operating point {X.target}\n\n| Engine | ms | x RAGRT | R@10 | {X.metric} | best QPS |\n|---|---|---|---|---|---|")
    for n, l, r, o in zip(names, lat, r10, off):
        md.append(f"| {STYLE[n][2]} | {l:.2f} | {l / ours:.1f} | {r:.3f} | {o:.3f} | {qps[n] or 0:.0f} |")
    md.append("")


def fig_pareto(X, out, md):
    """Old fig4: latency vs quality frontier per engine; WARP joins on the official metric (LoTTE only)."""
    names = X.engines(["plaid", "ptms", "fastplaid", "muvera", "ragrt", "ours"])
    have_warp = X.W is not None and X.metric.startswith("success")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for n in names:
        c, m, lab = STYLE[n]
        sw = X.sweep(n)
        if not sw:
            continue
        for ax, k in zip(axes, ("gt_r10", "official")):
            f = frontier(sw, k)
            ax.plot([p["lat_median"] for p in f], [p[k] for p in f], color=c, marker=m, markersize=4, label=lab)
    if have_warp:
        c, m, lab = STYLE["warp"]
        for thr in sorted({r["threads"] for r in X.W["runs"].values()}):
            rs = sorted((r for r in X.W["runs"].values() if r["threads"] == thr), key=lambda r: r["lat_median_ms"])
            axes[1].plot([r["lat_median_ms"] for r in rs], [r["success@5"] for r in rs], color=c, marker=m,
                         linestyle="-" if thr == 1 else ":", markersize=5, label=f"{lab}, {thr} thread(s), incl. encoding")
        md.append("WARP points include XTR query encoding on CPU and use WARP's own query loader; "
                  "the other engines exclude encoding (GPU). Compare shapes, not absolute gaps.\n")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("Median latency per query (ms, log)")
    axes[0].set_ylabel("Recall of exact top-10 (TUNE)"); axes[1].set_ylabel(f"{X.metric} (TUNE)")
    axes[1].legend(loc="lower right", fontsize=7)
    save(fig, out, "fig2_pareto.png")


def fig_breakdown(X, out, md):
    """Old fig3: per-stage latency, same five stage names for every engine that records them."""
    bars = []
    for n in X.engines(["plaid", "ptms", "muvera", "ragrt", "ours"]):
        b = X.breakdown(n)
        if not b:
            continue
        if n == "plaid":
            bars.append((n, [("candidate generation", b["s1_3"]), ("decompress + rerank (est.)", b["s4_est"])]))
        elif n in ("ptms", "muvera"):
            bars.append((n, [("candidate generation", b["s1_3"]), ("TileMaxSim rerank", b["s4"])]))
        else:
            bars.append((n, [("prep", b["s0b_prep"]), ("codewords" if n == "ragrt" else "tasks", b["s1_rt"]),
                             ("gather", b["s2_gather"]), ("sort-reduce", b["s3_score"]), ("TileMaxSim rerank", b["s4_tms"])]))
    if not bars:
        return
    shades = ["#d6eaf8", "#aed6f1", "#5dade2", "#2e86c1", "#1b4f72"]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(bars) + 1.2), layout="constrained")
    md.append("## fig3: stage breakdown (median ms)\n")
    for i, (n, segs) in enumerate(bars):
        left = 0.0
        for j, (seg, v) in enumerate(segs):
            ax.barh(i, v, left=left, color=shades[j if len(segs) > 2 else (1, 4)[j]], edgecolor="white", height=0.6)
            left += v
        ax.text(left, i, f"  {left:.2f} ms", va="center", fontsize=8)
        md.append(f"- {STYLE[n][2]}: " + ", ".join(f"{s} {v:.3f}" for s, v in segs) + f" (sum {left:.2f})")
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([STYLE[n][2] for n, _ in bars]); ax.invert_yaxis()
    ax.set_xlabel("Median ms per query (TEST); query encoding excluded"); ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=shades[k]) for k in range(5)]
    ax.legend(handles, ["prep / candidate gen.", "tasks", "gather", "sort-reduce", "rerank"], loc="lower right", fontsize=7)
    md.append("")
    save(fig, out, "fig3_breakdown.png")


def fig_throughput(X, out, md):
    """Old fig6: queries/s vs queries per call. Ours pipelined, FastPLAID batched, others sequential."""
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    md.append("## fig4: throughput (queries/s)\n")
    drew = False
    for n in X.engines(["plaid", "ptms", "muvera", "ragrt", "ours"]):
        R, e = X.src(n)
        T = (R or {}).get("throughput", {})
        if f"{e}_pipelined_qps" in T:
            B = sorted(int(b) for b in T[f"{e}_pipelined_qps"])
            ys = [T[f"{e}_pipelined_qps"][str(b)] for b in B]
            ax.plot(B, ys, color=STYLE[n][0], marker=STYLE[n][1], label=f"{STYLE[n][2]}, pipelined")
            md.append(f"- {STYLE[n][2]} pipelined: " + ", ".join(f"B={b} {y:.0f}" for b, y in zip(B, ys)))
            drew = True
        elif f"{e}_sequential_qps" in T:
            ax.axhline(T[f"{e}_sequential_qps"], color=STYLE[n][0], linestyle="--", linewidth=1.5,
                       label=f"{STYLE[n][2]}, one query at a time")
            md.append(f"- {STYLE[n][2]} sequential: {T[f'{e}_sequential_qps']:.0f}")
            drew = True
    if X.FP and X.FP.get("throughput", {}).get("qps_by_batch"):
        q = X.FP["throughput"]["qps_by_batch"]
        B = sorted(int(b) for b in q if int(b) <= 64)
        ys = [q[str(b)] for b in B]
        ax.plot(B, ys, color=STYLE["fastplaid"][0], marker=STYLE["fastplaid"][1], label="FastPLAID, batched search")
        md.append("- FastPLAID batched: " + ", ".join(f"B={b} {y:.0f}" for b, y in zip(B, ys))
                  + f"; all at once {X.FP['throughput'].get('batched_qps', 0):.0f}")
        drew = True
    if not drew:
        plt.close(fig); return
    ax.set_xscale("log", base=2); ax.set_xlabel("Queries per call"); ax.set_ylabel("Queries / s")
    ax.legend(fontsize=7, loc="upper left")
    md.append("")
    save(fig, out, "fig4_throughput.png")


def fig_filters(X, out, md):
    """Old fig2, extended: every filter kind, recall and latency, post-filter baselines vs ours."""
    F = X.M.get("filtered", {})
    filts = [f for f in FILTER_ORDER if f in F]
    if not filts:
        return
    names = [n for n in ["plaid", "ptms", "fastplaid", "ragrt", "ours", "ours_adapt", "ours_oracle"]
             if (n != "ragrt" or X.show_rt) and any(X.filtered(n, f) for f in filts)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), layout="constrained", sharex=True)
    w = 0.8 / len(names)
    md.append("## fig5: filters (TEST, primary point): R@10 / median ms\n")
    md.append("| Filter | " + " | ".join(STYLE[n][2] for n in names) + " |")
    md.append("|---" * (len(names) + 1) + "|")
    for i, n in enumerate(names):
        r = [val((X.filtered(n, f) or {}).get("gt_r10")) for f in filts]
        l = [((X.filtered(n, f) or {}).get("lat") or {}).get("median") for f in filts]
        xs = np.arange(len(filts)) + (i - (len(names) - 1) / 2) * w
        axes[0].bar(xs, [v or 0 for v in r], w, color=STYLE[n][0], label=STYLE[n][2])
        axes[1].bar(xs, [v or 0 for v in l], w, color=STYLE[n][0])
    for f in filts:
        cells = []
        for n in names:
            s = X.filtered(n, f)
            cells.append(f"{val(s['gt_r10']):.3f} / {s['lat']['median']:.2f}" if s else "n/a")
        md.append(f"| {FILTER_LABEL.get(f, f)} | " + " | ".join(cells) + " |")
    md.append("\nFastPLAID covers only filters it can express as a subset list (random and own-domain).\n")
    axes[0].set_ylabel("Recall of exact filtered top-10"); axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("Median latency (ms)"); axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(filts))); axes[1].set_xticklabels([FILTER_LABEL.get(f, f) for f in filts])
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(names), fontsize=7, loc="outside upper center")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    save(fig, out, "fig5_filters.png")


def fig_selectivity(X, out, md):
    """Old fig7: recall and latency against the fraction of passages passing a random filter."""
    F = X.M.get("filtered", {})
    syn = sorted((int(f[3:]), f) for f in F if f.startswith("syn"))
    if not syn:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for n in ["plaid", "ptms", "fastplaid", "ragrt", "ours", "ours_adapt"]:
        if n == "ragrt" and not X.show_rt:
            continue
        unf = X.point(n)
        rows = [(p, X.filtered(n, f)) for p, f in syn]
        rows = [(p, s) for p, s in rows if s]
        if not rows:
            continue
        if unf:
            rows.append((100, unf))
        xs = [p for p, _ in rows]
        c, m, lab = STYLE[n]
        axes[0].plot(xs, [val(s["gt_r10"]) for _, s in rows], color=c, marker=m, label=lab)
        axes[1].plot(xs, [s["lat"]["median"] for _, s in rows], color=c, marker=m)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks([5, 14, 30, 60, 100]); ax.set_xticklabels(["5", "14", "30", "60", "100"])
        ax.set_xlabel("Passages passing the filter (%)")
    axes[0].set_ylabel("Recall of exact filtered top-10"); axes[1].set_ylabel("Median latency (ms)")
    axes[1].set_yscale("log"); axes[0].legend(fontsize=7, loc="lower right")
    save(fig, out, "fig6_selectivity.png")


def fig_budget(X, out, md):
    """Can post-filtering catch up with more candidates? PLAID budgets as curves, ours as single points."""
    B = X.M.get("filter_budget", {})
    if not B:
        return
    filts = [f for f in ("corr_mid", "corr_far", "syn5") if f in B]
    fig, axes = plt.subplots(1, len(filts), figsize=(4.2 * len(filts), 3.4), layout="constrained", squeeze=False)
    md.append("## fig7: post-filter budget sweep (R@10 at ms)\n")
    for ax, f in zip(axes[0], filts):
        for e in ("plaid", "ptms"):
            if e not in B[f]:
                continue
            pts = sorted(B[f][e].values(), key=lambda v: v["params"]["ndocs"])
            ax.plot([p["lat"]["median"] for p in pts], [val(p["gt_r10"]) for p in pts], color=STYLE[e][0],
                    marker=STYLE[e][1], label=f"{STYLE[e][2]}, 1x to 64x candidates")
            md.append(f"- {f} {STYLE[e][2]}: " + ", ".join(f"{val(p['gt_r10']):.3f} at {p['lat']['median']:.1f}" for p in pts))
        for n in ("ours", "ours_adapt", "ours_oracle"):
            s = X.filtered(n, f)
            if s:
                ax.scatter([s["lat"]["median"]], [val(s["gt_r10"])], color=STYLE[n][0], marker=STYLE[n][1], s=60,
                           zorder=3, label=STYLE[n][2])
                md.append(f"- {f} {STYLE[n][2]}: {val(s['gt_r10']):.3f} at {s['lat']['median']:.2f}")
        ax.set_xscale("log"); ax.set_xlabel("Median latency (ms, log)"); ax.set_title(FILTER_LABEL.get(f, f), color=INK)
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("Recall of exact filtered top-10"); axes[0][-1].legend(fontsize=7, loc="lower right")
    md.append("")
    save(fig, out, "fig7_budget.png")


def fig_attribution(X, out, md):
    """Where wall time goes: GPU-busy vs host time, and kernel launches / host syncs per query."""
    rows = []
    for n in X.engines(["plaid", "ptms", "muvera", "ragrt", "ours"]):
        R, e = X.src(n)
        a = (R or {}).get("attribution", {}).get(e)
        if a:
            rows.append((n, a))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(rows) + 1.5), layout="constrained")
    y = np.arange(len(rows))
    busy = [a["gpu_busy_ms"] for _, a in rows]
    host = [max(0.0, a["wall_ms"] - a["gpu_busy_ms"]) for _, a in rows]
    axes[0].barh(y, busy, color="#2e86c1", label="GPU busy"); axes[0].barh(y, host, left=busy, color="#d5d8dc", label="host / idle")
    axes[0].set_xlabel("ms per query"); axes[0].legend(fontsize=7)
    axes[1].barh(y - 0.2, [a["kernels"] for _, a in rows], 0.4, color="#5dade2", label="kernel launches")
    axes[1].barh(y + 0.2, [a["syncs"] for _, a in rows], 0.4, color="#c0392b", label="host syncs")
    axes[1].set_xscale("log"); axes[1].set_xlabel("per query (log)"); axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_yticks(y); ax.set_yticklabels([STYLE[n][2] for n, _ in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    md.append("## fig8: attribution per query\n\n| Engine | wall ms | GPU-busy ms | kernels | memcpys | syncs |\n|---|---|---|---|---|---|")
    for n, a in rows:
        md.append(f"| {STYLE[n][2]} | {a['wall_ms']:.2f} | {a['gpu_busy_ms']:.2f} | {a['kernels']:.0f} | {a.get('memcpys', 0):.0f} | {a['syncs']:.0f} |")
    md.append("")
    save(fig, out, "fig8_attribution.png")


def fig_oracle_ablation(X, out, md):
    """Paired differences in filtered R@10 with 95% bootstrap CIs (filtered_pairs)."""
    P = X.M.get("filtered_pairs", {})
    rows = [(f, k, v) for f in FILTER_ORDER if f in P for k, v in P[f].items()]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(rows) + 1.2), layout="constrained")
    y = np.arange(len(rows))
    m = [v["mean"] for *_, v in rows]
    lo = [v["mean"] - v["ci95"][0] for *_, v in rows]
    hi = [v["ci95"][1] - v["mean"] for *_, v in rows]
    ax.errorbar(m, y, xerr=[lo, hi], fmt="o", color=STYLE["ours"][0], ecolor=INK2, capsize=3)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{FILTER_LABEL.get(f, f)}: {k}" for f, k, _ in rows], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("Change in filtered R@10 (paired, 95% bootstrap CI)")
    md.append("## fig9: paired filtered-R@10 differences\n")
    md += [f"- {f}: {k}: {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for f, k, v in rows]
    md.append("")
    save(fig, out, "fig9_oracle_ablation.png")


def fig_juno(out, md):
    """JUNO's selective-LUT step on RT cores vs CUDA cores, same GPU and inputs (baselines/juno)."""
    files = sorted(glob.glob(os.path.join(C.RESULTS_ROOT, "juno", "juno_rt_vs_cuda_*.jsonl")))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(5.5 * len(files), 3.4), layout="constrained", squeeze=False)
    md.append("## fig10: JUNO selection step, RT vs CUDA cores (ms per batch of queries)\n")
    for ax, p in zip(axes[0], files):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        name = os.path.basename(p)[len("juno_rt_vs_cuda_"):-len(".jsonl")]
        for frac in sorted({r["frac"] for r in rows}):
            rs = sorted((r for r in rows if r["frac"] == frac), key=lambda r: r["nprobe"])
            ax.plot([r["nprobe"] for r in rs], [r["rt_ms"] / r["cuda_ms"] for r in rs], marker="o",
                    label=f"{frac:.0%} of entries selected")
            md += [f"- {name} frac {frac} nprobe {r['nprobe']}: RT {r['rt_ms']:.3f}, CUDA {r['cuda_ms']:.3f}, "
                   f"dense LUT {r.get('dense_lut_ms', float('nan')):.3f}, sets match {r.get('set_match')}" for r in rs]
        ax.axhline(1, color=INK2, linewidth=1)
        ax.set_xscale("log", base=2); ax.set_xlabel("nprobe"); ax.set_ylabel("RT time / CUDA-core time")
        ax.set_title(f"{name} ({rows[0].get('gpu', '')})", color=INK); ax.legend(fontsize=7)
    md.append("")
    save(fig, out, "fig10_juno.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--main", required=True, help="results subdir of the main run_benchmarks.py run")
    ap.add_argument("--ours", default="weft", help="engine id shown as RAGRT (weft, weft_graph, ...)")
    ap.add_argument("--target", default=None, help="selection target name (default: the run's primary)")
    ap.add_argument("--muvera", default=None); ap.add_argument("--fastplaid", default=None); ap.add_argument("--warp", default=None)
    ap.add_argument("--with_rt", action="store_true", help="also show the RT-core design (needs it in the main run)")
    ap.add_argument("--drop_stale", action="store_true", help="leave out baseline results older than the main run")
    a = ap.parse_args()
    ds = C.dataset(a.dataset)
    X = Results(a, ds)
    out = os.path.join(X.root, a.main, "figures")
    os.makedirs(out, exist_ok=True)
    md = [f"# Figures for {a.dataset} / {a.main}\n", "Sources: " + ", ".join(f"{k} = {v}" for k, v in X.sources.items()) + "\n"]
    if X.stale:
        md.append(f"**Stale (older than the main run{', left out' if a.drop_stale else ''}): {', '.join(X.stale)}**\n")
    for f in (fig_headline, fig_pareto, fig_breakdown, fig_throughput, fig_filters, fig_selectivity,
              fig_budget, fig_attribution, fig_oracle_ablation):
        try:
            f(X, out, md)
        except (KeyError, TypeError, ValueError) as err:          # a missing stage should not stop the rest
            print(f"  skipped {f.__name__}: {type(err).__name__} {err}")
    fig_juno(out, md)
    open(os.path.join(out, "figures.md"), "w").write("\n".join(md) + "\n")
    print(f"  wrote figures.md to {out}")


if __name__ == "__main__":
    main()
