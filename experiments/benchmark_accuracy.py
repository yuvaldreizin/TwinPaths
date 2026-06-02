"""
benchmark_accuracy.py
=====================
Statistical accuracy comparison of the DPGC heuristic against the Matroid DPT
solver, as a function of graph size n.

The Matroid solver is *exact* for the metric (triangle-cost) case, so it is used
as the optimal reference. For each size n we generate `--seeds` random Euclidean
graphs and, on each, compute:

    approximation ratio  =  DPGC cost / Matroid cost   (>= 1.0; 1.0 == optimal)
    excess               =  (DPGC cost - Matroid cost) / Matroid cost  [%]
    match                =  DPGC found the optimum (ratio <= 1 + tol)

We report, per n: the match rate (% of graphs solved optimally by DPGC), and the
distribution of the approximation ratio (mean / median / p95 / max).

To keep the Matroid-as-ground-truth claim honest, for n <= `--brute-max` we also
run the exhaustive brute-force solver and report how often Matroid equals the
true optimum (it should be 100%).

Outputs (under results/):
    accuracy_raw.csv        one row per (n, seed)
    accuracy_summary.csv    one row per n with statistics (incl. ratio stats)
    accuracy_comparison.png  match-rate bar chart (% of graphs solved optimally)

Usage (run from the repository root):
    python -m experiments.benchmark_accuracy                 # full run
    python -m experiments.benchmark_accuracy --quick         # fast smoke run
    python -m experiments.benchmark_accuracy --sizes 5 10 20 --seeds 50
"""

import argparse
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from twinpaths.graphs.generate_graphs import random_euclidean_graph
from twinpaths.solvers.DPGC_heuristic import dpgc_heuristic
from twinpaths.solvers.matroid_dpt import solve_dpt_matroid
from twinpaths.solvers.ground_truth import mst_k2_generator

# ── dark-theme palette (matches the rest of the repo) ──────────────────────────
BG, CARD, BORDER, MUTED, TEXT = '#09090b', '#18181b', '#27272a', '#71717a', '#fafafa'
CYAN, VIOLET, RED, GREEN, AMBER = '#22d3ee', '#a78bfa', '#f87171', '#34d399', '#fbbf24'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': BORDER,
    'axes.labelcolor': TEXT, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT, 'grid.color': BORDER, 'grid.linestyle': '--',
    'grid.alpha': 0.4, 'font.family': 'sans-serif',
    'legend.facecolor': CARD, 'legend.edgecolor': BORDER, 'legend.labelcolor': TEXT,
})


def cost_of(fn, G, s, t):
    """Run a solver and return its total cost, or None on failure."""
    try:
        _, info = fn(G.copy(), s=s, t=t)
        return float(info.get("total_cost", float("nan")))
    except Exception as exc:
        print(f"      WARN {fn.__name__}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[5, 8, 10, 15, 20, 30, 50, 75, 100],
                    help="graph sizes (node counts) to evaluate")
    ap.add_argument("--seeds", type=int, default=50,
                    help="number of random graphs (seeds) per size")
    ap.add_argument("--brute-max", type=int, default=10,
                    help="validate Matroid against brute-force for n <= this")
    ap.add_argument("--edge-prob", type=float, default=0.5,
                    help="edge probability for the random Euclidean graphs")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative tolerance for counting a DPGC result as optimal")
    ap.add_argument("--quick", action="store_true",
                    help="small/fast run (overrides sizes/seeds) for smoke testing")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    if args.quick:
        args.sizes = [5, 8, 10, 15]
        args.seeds = 8
        args.brute_max = 8

    os.makedirs(args.results_dir, exist_ok=True)

    raw_rows = []
    per_n_ratios = {}        # n -> list of approximation ratios
    per_n_match = {}         # n -> match rate (%)
    per_n_matroid_ok = {}    # n -> matroid-vs-brute match rate (%) or None

    print("=" * 64)
    print(f"Accuracy benchmark  |  seeds={args.seeds}  edge_prob={args.edge_prob}")
    print("DPGC vs Matroid (exact reference)")
    print("=" * 64)

    for n in args.sizes:
        print(f"\nn = {n}")
        ratios, matched = [], 0
        matroid_checks, matroid_ok = 0, 0
        for seed in range(args.seeds):
            G = random_euclidean_graph(n, edge_prob=args.edge_prob, seed=seed)
            t_node = max(G.nodes())

            cd = cost_of(dpgc_heuristic, G, 1, t_node)
            cm = cost_of(solve_dpt_matroid, G, 1, t_node)
            if cd is None or cm is None or cm <= 0:
                continue

            ratio = cd / cm
            ratios.append(ratio)
            is_match = ratio <= 1.0 + args.tol
            if is_match:
                matched += 1

            cb = None
            if n <= args.brute_max:
                cb = cost_of(mst_k2_generator, G, 1, t_node)
                if cb is not None and cb > 0:
                    matroid_checks += 1
                    if math.isclose(cm, cb, rel_tol=1e-6):
                        matroid_ok += 1

            raw_rows.append({
                "n": n, "seed": seed,
                "dpgc_cost": cd, "matroid_cost": cm,
                "brute_cost": cb if cb is not None else "",
                "ratio": ratio, "excess_pct": (ratio - 1.0) * 100.0,
                "dpgc_optimal": int(is_match),
            })
            print(f"  seed={seed}  ratio={ratio:.4f}", flush=True)

        if not ratios:
            continue
        per_n_ratios[n] = ratios
        per_n_match[n] = matched / len(ratios) * 100.0
        per_n_matroid_ok[n] = (matroid_ok / matroid_checks * 100.0
                               if matroid_checks else None)
        r = np.asarray(ratios)
        print(f"  -> match {per_n_match[n]:.0f}%   mean ratio {r.mean():.4f}   "
              f"p95 {np.percentile(r, 95):.4f}   max {r.max():.4f}")

    # ── raw CSV ─────────────────────────────────────────────────────────────────
    raw_path = os.path.join(args.results_dir, "accuracy_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "seed", "dpgc_cost", "matroid_cost",
                                          "brute_cost", "ratio", "excess_pct",
                                          "dpgc_optimal"])
        w.writeheader()
        w.writerows(raw_rows)
    print(f"\nWrote raw samples -> {raw_path}")

    # ── summary CSV + console table ─────────────────────────────────────────────
    sizes = sorted(per_n_ratios)
    summary_path = os.path.join(args.results_dir, "accuracy_summary.csv")
    summary_rows = []
    print("\n" + "-" * 88)
    print(f"{'n':>4}{'samples':>9}{'match %':>10}{'mean ratio':>13}"
          f"{'median':>10}{'p95':>9}{'max':>9}{'matroid=opt %':>16}")
    print("-" * 88)
    for n in sizes:
        r = np.asarray(per_n_ratios[n])
        mok = per_n_matroid_ok[n]
        row = {
            "n": n, "n_samples": len(r), "match_pct": per_n_match[n],
            "mean_ratio": float(r.mean()), "median_ratio": float(np.median(r)),
            "p95_ratio": float(np.percentile(r, 95)), "max_ratio": float(r.max()),
            "mean_excess_pct": float((r - 1.0).mean() * 100.0),
            "matroid_eq_opt_pct": mok if mok is not None else "",
        }
        summary_rows.append(row)
        mok_str = f"{mok:.0f}" if mok is not None else "-"
        print(f"{n:>4}{len(r):>9}{per_n_match[n]:>9.0f}%{r.mean():>13.4f}"
              f"{np.median(r):>10.4f}{np.percentile(r, 95):>9.4f}{r.max():>9.4f}"
              f"{mok_str:>16}")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "n_samples", "match_pct",
                                          "mean_ratio", "median_ratio",
                                          "p95_ratio", "max_ratio",
                                          "mean_excess_pct", "matroid_eq_opt_pct"])
        w.writeheader()
        w.writerows(summary_rows)
    print("-" * 88)
    print(f"Wrote summary -> {summary_path}")

    # ── plot: match-rate bars (fraction of graphs solved optimally by DPGC) ──────
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)

    colors = [CYAN if per_n_match[n] >= 90 else AMBER if per_n_match[n] >= 70 else RED
              for n in sizes]
    width = max(1.5, (max(sizes) - min(sizes)) / (len(sizes) * 2.2))
    bars = ax1.bar(sizes, [per_n_match[n] for n in sizes], color=colors,
                   edgecolor=BORDER, width=width, alpha=0.9, zorder=3)
    for b, n in zip(bars, sizes):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                 f"{per_n_match[n]:.0f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=TEXT)
    ax1.axhline(100, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.5,
                label="100% (always optimal)")
    ax1.set_xlabel("Number of nodes  (n)", fontsize=13, labelpad=8)
    ax1.set_ylabel("Graphs where DPGC = optimal  (%)", fontsize=13, labelpad=8)
    ax1.set_title(f"DPGC match rate vs. Matroid DPT (exact reference)  —  "
                  f"{args.seeds} random Euclidean graphs per size",
                  fontsize=13, pad=12, color=TEXT)
    ax1.set_ylim(0, 119)
    ax1.set_xticks(sizes)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, axis="y", zorder=0)

    fig.tight_layout()
    plot_path = os.path.join(args.results_dir, "accuracy_comparison.png")
    fig.savefig(plot_path, dpi=150, facecolor=BG)
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
