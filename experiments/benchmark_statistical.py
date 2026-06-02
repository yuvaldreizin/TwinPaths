"""
benchmark_statistical.py
========================
Statistical runtime benchmark of the three DPT/DPST solvers, designed to back up
the complexity analysis in docs/algorithm_analysis.md.

For every graph size n we generate `--seeds` independent random Euclidean graphs
and time each solver on each one. We then report, per (algorithm, n):

    mean, std, standard error, and a 95% confidence interval

and plot mean runtime vs. n on a log scale with 95% CI error bars.

Because the brute-force solver is super-exponential, it only runs up to
`--brute-max` nodes; the two polynomial solvers run over the full size range.

For large n, dense graphs (--edge-prob) make the matroid solver intractable, so
pass --avg-degree to use SPARSE graphs (edge_prob = avg_degree/(n-1) per size).

Outputs (under results/, optionally prefixed by --out-prefix):
    benchmark_runtimes_raw.csv       one row per (algo, n, seed)
    benchmark_runtimes_summary.csv   one row per (algo, n) with statistics
    benchmark_runtimes.png           log-scale runtime plot with 95% CI bars

Usage (run from the repository root):
    python -m experiments.benchmark_statistical                   # full run (dense)
    python -m experiments.benchmark_statistical --quick           # fast smoke run
    python -m experiments.benchmark_statistical --sizes 4 6 8 --brute-max 8 --seeds 10
    # extended large-n scaling of the two polynomial solvers on sparse graphs:
    python -m experiments.benchmark_statistical --sizes 50 100 200 300 500 750 1000 \
        --seeds 5 --avg-degree 6 --out-prefix large_
"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from twinpaths.graphs.generate_graphs import random_euclidean_graph
from twinpaths.solvers.DPGC_heuristic import dpgc_heuristic
from twinpaths.solvers.matroid_dpt import solve_dpt_matroid
from twinpaths.solvers.ground_truth import mst_k2_generator

# ── solver registry ───────────────────────────────────────────────────────────
# (key, display label, function, only_up_to_n)  —  None = no size cap
ALGOS = [
    ("brute",   "Brute-force (exact)",  mst_k2_generator,  "brute_max"),
    ("matroid", "Matroid DPT (exact)",  solve_dpt_matroid, None),
    ("dpgc",    "DPGC (heuristic)",     dpgc_heuristic,     None),
]

# ── dark-theme palette (matches the rest of the repo) ──────────────────────────
BG, CARD, BORDER, MUTED, TEXT = '#09090b', '#18181b', '#27272a', '#71717a', '#fafafa'
COLORS = {"brute": '#f87171', "matroid": '#a78bfa', "dpgc": '#22d3ee'}
MARKERS = {"brute": '^', "matroid": 's', "dpgc": 'o'}

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': BORDER,
    'axes.labelcolor': TEXT, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT, 'grid.color': BORDER, 'grid.linestyle': '--',
    'grid.alpha': 0.4, 'font.family': 'sans-serif',
    'legend.facecolor': CARD, 'legend.edgecolor': BORDER, 'legend.labelcolor': TEXT,
})


def time_solver(fn, G, s, t):
    """Run a solver once and return (runtime_sec, total_cost) or (None, None)."""
    try:
        t0 = time.perf_counter()
        _, info = fn(G.copy(), s=s, t=t)
        elapsed = time.perf_counter() - t0
        return elapsed, float(info.get("total_cost", float("nan")))
    except Exception as exc:
        print(f"      WARN {fn.__name__}: {exc}")
        return None, None


def summarize(samples):
    """Return summary statistics for a list of runtime samples."""
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    # 95% normal-approx CI half-width (t-value ~1.96 for large n; fine for plots)
    ci = 1.96 * sem
    return {"n_samples": n, "mean": mean, "std": std, "sem": sem,
            "ci95_low": mean - ci, "ci95_high": mean + ci, "ci95_half": ci}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40],
                    help="graph sizes (node counts) to benchmark")
    ap.add_argument("--brute-max", type=int, default=10,
                    help="largest n on which to run the brute-force solver")
    ap.add_argument("--seeds", type=int, default=20,
                    help="number of random graphs (seeds) per size")
    ap.add_argument("--edge-prob", type=float, default=0.5,
                    help="edge probability for the random Euclidean graphs")
    ap.add_argument("--avg-degree", type=float, default=None,
                    help="if set, use SPARSE graphs with edge_prob = avg_degree/(n-1) "
                         "per size (overrides --edge-prob); needed to reach large n")
    ap.add_argument("--out-prefix", default="",
                    help="prefix for output filenames, e.g. 'large_' (avoids "
                         "overwriting the default dense-graph outputs)")
    ap.add_argument("--quick", action="store_true",
                    help="small/fast run (overrides sizes/seeds) for smoke testing")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    if args.quick:
        args.sizes = [4, 5, 6, 7, 8]
        args.brute_max = 8
        args.seeds = 3

    os.makedirs(args.results_dir, exist_ok=True)

    def edge_prob_for(n):
        """Per-size edge probability (sparse if --avg-degree given, else fixed)."""
        if args.avg_degree is not None:
            return min(1.0, args.avg_degree / max(n - 1, 1))
        return args.edge_prob

    density_desc = (f"avg_degree={args.avg_degree} (sparse)"
                    if args.avg_degree is not None else f"edge_prob={args.edge_prob}")

    caps = {"brute_max": args.brute_max, None: None}
    # samples[key][n] = list of runtimes ; costs[key][n] = list of costs
    samples = {key: defaultdict(list) for key, *_ in ALGOS}
    costs = {key: defaultdict(list) for key, *_ in ALGOS}

    raw_path = os.path.join(args.results_dir, f"{args.out_prefix}benchmark_runtimes_raw.csv")
    raw_rows = []

    print("=" * 64)
    print(f"Statistical runtime benchmark  |  seeds={args.seeds}  {density_desc}")
    print("=" * 64)

    for n in args.sizes:
        print(f"\nn = {n}")
        for seed in range(args.seeds):
            G = random_euclidean_graph(n, edge_prob=edge_prob_for(n), seed=seed)
            t_node = max(G.nodes())
            for key, label, fn, cap_key in ALGOS:
                cap = caps[cap_key]
                if cap is not None and n > cap:
                    continue
                rt, cost = time_solver(fn, G, 1, t_node)
                if rt is None:
                    continue
                samples[key][n].append(rt)
                costs[key][n].append(cost)
                raw_rows.append({"algo": key, "label": label, "n": n,
                                 "seed": seed, "runtime_sec": rt,
                                 "total_cost": cost})
            print(f"  seed={seed} done", flush=True)

    # ── write raw CSV ──────────────────────────────────────────────────────────
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "label", "n", "seed",
                                          "runtime_sec", "total_cost"])
        w.writeheader()
        w.writerows(raw_rows)
    print(f"\nWrote raw samples -> {raw_path}")

    # ── summary CSV + console table ─────────────────────────────────────────────
    summary_path = os.path.join(args.results_dir, f"{args.out_prefix}benchmark_runtimes_summary.csv")
    summary_rows = []
    print("\n" + "-" * 78)
    print(f"{'algo':<10}{'n':>4}{'samples':>9}{'mean (s)':>14}"
          f"{'std (s)':>12}{'95% CI half':>14}")
    print("-" * 78)
    for key, label, fn, _ in ALGOS:
        for n in sorted(samples[key]):
            st = summarize(samples[key][n])
            row = {"algo": key, "label": label, "n": n, **st}
            summary_rows.append(row)
            print(f"{key:<10}{n:>4}{st['n_samples']:>9}{st['mean']:>14.6f}"
                  f"{st['std']:>12.6f}{st['ci95_half']:>14.6f}")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "label", "n", "n_samples",
                                          "mean", "std", "sem",
                                          "ci95_low", "ci95_high", "ci95_half"])
        w.writeheader()
        w.writerows(summary_rows)
    print("-" * 78)
    print(f"Wrote summary -> {summary_path}")

    # ── plot: log-scale runtime vs n with 95% CI error bars ─────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)

    for key, label, fn, _ in ALGOS:
        ns = sorted(samples[key])
        if not ns:
            continue
        means = [summarize(samples[key][n])["mean"] for n in ns]
        errs = [summarize(samples[key][n])["ci95_half"] for n in ns]
        ax.errorbar(ns, means, yerr=errs, color=COLORS[key], marker=MARKERS[key],
                    markersize=8, linewidth=2.0, capsize=4, capthick=1.2,
                    elinewidth=1.2, label=label, zorder=3)
        ax.annotate(f"{means[-1]:.4f}s", xy=(ns[-1], means[-1]),
                    xytext=(-6, 8), textcoords="offset points",
                    fontsize=8.5, color=COLORS[key], va="bottom", ha="right")

    ax.set_yscale("log")

    # Only annotate the brute-force cut-off if brute-force actually ran here.
    if samples["brute"] and args.brute_max < max(args.sizes):
        x = args.brute_max + 0.5
        ax.axvline(x=x, color=MUTED, linewidth=1, linestyle=":", alpha=0.6)
        y_lo, y_hi = ax.get_ylim()
        ax.text(x, y_lo * 1.5, " brute-force not run ->",
                fontsize=8, color=MUTED, va="bottom", ha="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.3f}s" if v >= 1e-3 else f"{v:.0e}s"))
    ax.set_xlabel("Number of nodes  (n)", fontsize=13, labelpad=8)
    ax.set_ylabel("Mean runtime  [log scale, 95% CI]", fontsize=13, labelpad=8)
    ax.set_title(f"DPT/DPST solver runtime  —  mean of {args.seeds} random "
                 f"Euclidean graphs per size ({density_desc})",
                 fontsize=13, pad=14, color=TEXT)
    ax.set_xticks(args.sizes)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, which="both")

    fig.tight_layout()
    plot_path = os.path.join(args.results_dir, f"{args.out_prefix}benchmark_runtimes.png")
    fig.savefig(plot_path, dpi=150, facecolor=BG)
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
