"""Plot DPGC/Matroid approximation ratio (mean +/- std) vs n.

Reads per-graph ratios from results/accuracy_raw.csv, computes mean and sample
std per size, and plots a single line with std error bars -- in the same
dark-theme style as benchmark_statistical.py.

Run:  python -m experiments.plot_accuracy
"""
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "results", "accuracy_raw.csv")
OUT = os.path.join(ROOT, "results", "accuracy_ratio.png")

# ── dark-theme palette (matches benchmark_statistical.py) ──────────────────────
BG, CARD, BORDER, MUTED, TEXT = '#09090b', '#18181b', '#27272a', '#71717a', '#fafafa'
DPGC_COLOR = '#22d3ee'  # same cyan used for DPGC in the runtime plots

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': BORDER,
    'axes.labelcolor': TEXT, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT, 'grid.color': BORDER, 'grid.linestyle': '--',
    'grid.alpha': 0.4, 'font.family': 'sans-serif',
    'legend.facecolor': CARD, 'legend.edgecolor': BORDER, 'legend.labelcolor': TEXT,
})


def main():
    by_n = defaultdict(list)
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            by_n[int(r["n"])].append(float(r["ratio"]))

    ns = sorted(by_n)
    means, stds = [], []
    for n in ns:
        vals = by_n[n]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        means.append(m)
        stds.append(math.sqrt(var))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)

    # Ratio is bounded below by 1.0 (DPGC never beats the optimum), so clip the
    # lower whisker at that floor -- a symmetric ±std would dip below 1.0 purely
    # as a plotting artifact of right-skewed, one-sided data.
    lower = [min(s, m - 1.0) for m, s in zip(means, stds)]
    upper = stds
    ax.errorbar(ns, means, yerr=[lower, upper], color=DPGC_COLOR, marker='o',
                markersize=8, linewidth=2.0, capsize=4, capthick=1.2,
                elinewidth=1.2, label="DPGC / Matroid  (mean ± std, clipped at 1.0)",
                zorder=3)
    ax.annotate(f"{means[-1]:.3f}", xy=(ns[-1], means[-1]),
                xytext=(-6, 8), textcoords="offset points",
                fontsize=8.5, color=DPGC_COLOR, va="bottom", ha="right")

    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":", alpha=0.8,
               label="optimal (1.00)", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Number of nodes  (n)", fontsize=13, labelpad=8)
    ax.set_ylabel("Approximation ratio  (DPGC cost / Matroid cost)",
                  fontsize=13, labelpad=8)
    ax.set_title("DPGC heuristic accuracy vs. Matroid DPT (exact)",
                 fontsize=14, pad=14, color=TEXT)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, which="both")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=BG)
    print(f"Saved plot -> {OUT}")
    for n, m, s in zip(ns, means, stds):
        print(f"  n={n:4d}  mean={m:.4f}  std={s:.4f}")


if __name__ == "__main__":
    main()
