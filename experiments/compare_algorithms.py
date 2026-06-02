"""
compare_algorithms.py
---------------------
CLI wrapper: loads graphs from disk and delegates to run_batch().

Examples::

    python -m experiments.compare_algorithms --s 1 --t 5
    python -m experiments.compare_algorithms --graph-dir data/example_graphs --s 1 --t 5 --output-csv results/run.csv
    python -m experiments.compare_algorithms --graphs data/example_graphs/dfn-gwin.pkl --s 1 --t 5
"""

import argparse
import os

from experiments.batch_run_algorithms import run_batch
from twinpaths.graphs.graph_sources import load_graphs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DPGC, matroid, and ground-truth solvers on graphs and export CSV."
    )
    parser.add_argument(
        "--graph-dir", default="data/example_graphs",
        help="Directory of .pkl graph files (default: data/example_graphs).",
    )
    parser.add_argument(
        "--graphs", nargs="*", default=[],
        help="Explicit graph file paths; overrides --graph-dir.",
    )
    parser.add_argument("--s", default="1", help="Start node label (default: 1).")
    parser.add_argument("--t", default="2", help="End node label (default: 2).")
    parser.add_argument("--weight", default="weight", help="Edge weight attribute.")
    parser.add_argument(
        "--output-csv",
        default=os.path.join("results", "batch_results.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    graph_entries = load_graphs(graph_dir=args.graph_dir, graphs=args.graphs)
    if not graph_entries:
        raise SystemExit("No graph files found.")

    results = run_batch(
        graph_entries,
        s=args.s,
        t=args.t,
        weight=args.weight,
        output_csv=args.output_csv,
    )

    # Brief console summary
    print("\n--- Summary ---")
    for r in results:
        if r["error"]:
            print(f"  {r['graph']} [{r['algo']}] ERROR: {r['error']}")
        else:
            print(
                f"  {r['graph']} [{r['algo']}]"
                f"  cost={r['total_cost']:.4f}"
                f"  time={float(r['runtime_sec']):.4f}s"
                f"  |E|={r['edge_count']}"
            )


if __name__ == "__main__":
    main()
