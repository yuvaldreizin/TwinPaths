"""
batch_run_algorithms.py
-----------------------
API for running all three DPT/DPST solvers on one or more graphs and
collecting results in a unified format.

Usage (as a module)::

    from experiments.batch_run_algorithms import run_batch
    results = run_batch([("my_graph", G)], s=1, t=14)
    for r in results:
        print(r["algo"], r["total_cost"], r["runtime_sec"])

Usage (as a script): runs the built-in example graph.
"""

import csv
import os
import time
from datetime import datetime
from typing import Any, Iterable, List, Optional, Tuple, Dict

import networkx as nx

from twinpaths.solvers.DPGC_heuristic import dpgc_heuristic
from twinpaths.solvers.matroid_dpt import solve_dpt_matroid
from twinpaths.solvers.ground_truth import mst_k2_generator


# ---------------------------------------------------------------------------
# Registry — add / remove solvers here
# ---------------------------------------------------------------------------

ALGORITHMS: List[Tuple[str, Any]] = [
    ("dpgc_heuristic",          dpgc_heuristic),
    ("matroid_dpt",             solve_dpt_matroid),
    ("ground_truth_bruteforce", mst_k2_generator),
]

CSV_FIELDNAMES = [
    "graph", "algo", "s", "t", "weight_attr",
    "total_cost", "runtime_sec", "edge_count", "edges", "error",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stringify_edges(edges: Iterable[Tuple[Any, Any]]) -> str:
    return ";".join(
        f"{u}-{v}"
        for u, v in sorted(edges, key=lambda e: (str(e[0]), str(e[1])))
    )


def _resolve_terminal(val: Any, nodes) -> Any:
    """Coerce a terminal label to an existing node, trying int/float if needed."""
    if val in nodes:
        return val
    for cast in (int, float):
        try:
            v = cast(val)
            if v in nodes:
                return v
        except Exception:
            pass
    return None


def _has_two_edge_disjoint_paths(g: nx.Graph, s: Any, t: Any) -> bool:
    try:
        return nx.edge_connectivity(g, s, t) >= 2
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_batch(
    graphs: List[Tuple[str, nx.Graph]],
    s: Any = 1,
    t: Any = 2,
    weight: str = "weight",
    output_csv: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run all three algorithms on each graph and write results to a CSV file.

    Parameters
    ----------
    graphs     : list of (name, nx.Graph) pairs
    s, t       : terminal node labels (coerced to graph node type if needed)
    weight     : edge-attribute name used as cost
    output_csv : destination CSV path; auto-generated in results/ if None

    Returns
    -------
    List of result dicts — one per (graph, algorithm) pair — with keys:
        graph, algo, s, t, weight_attr, total_cost, runtime_sec,
        edge_count, edges, error
    """
    results: List[Dict[str, Any]] = []

    for graph_name, g in graphs:
        nodes = set(g.nodes())
        s_use = _resolve_terminal(s, nodes)
        t_use = _resolve_terminal(t, nodes)

        if s_use is None or t_use is None:
            for algo_name, _ in ALGORITHMS:
                results.append(_error_row(
                    graph_name, algo_name, s, t, weight,
                    f"terminal not in graph; available: {sorted(nodes, key=str)}",
                ))
            continue

        if not _has_two_edge_disjoint_paths(g, s_use, t_use):
            for algo_name, _ in ALGORITHMS:
                results.append(_error_row(
                    graph_name, algo_name, s_use, t_use, weight,
                    "infeasible: fewer than 2 edge-disjoint s-t paths",
                ))
            continue

        for algo_name, fn in ALGORITHMS:
            try:
                edges, info = fn(g.copy(), s=s_use, t=t_use, weight=weight)
                results.append({
                    "graph":       graph_name,
                    "algo":        info.get("algo", algo_name),
                    "s":           s_use,
                    "t":           t_use,
                    "weight_attr": weight,
                    "total_cost":  info.get("total_cost", ""),
                    "runtime_sec": info.get("runtime_sec", ""),
                    "edge_count":  len(edges),
                    "edges":       stringify_edges(info.get("edges") or edges),
                    "error":       "",
                })
            except Exception as exc:
                results.append(_error_row(graph_name, algo_name, s_use, t_use, weight, str(exc)))

    _write_csv(results, output_csv)
    return results


def _error_row(graph_name, algo_name, s, t, weight, error_msg) -> Dict[str, Any]:
    return {
        "graph": graph_name, "algo": algo_name,
        "s": s, "t": t, "weight_attr": weight,
        "total_cost": "", "runtime_sec": "", "edge_count": "", "edges": "",
        "error": error_msg,
    }


def _write_csv(results: List[Dict[str, Any]], output_csv: Optional[str]) -> None:
    if output_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join("results", f"batch_results_{ts}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows for {len(results) // max(len(ALGORITHMS), 1)} graph(s) to {output_csv}")


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    G = nx.Graph()
    for u, v, w in [
        (1, 2, 1),  (2, 3, 2),  (3, 14, 3),
        (1, 4, 2),  (4, 5, 2),  (5, 14, 2),
        (1, 6, 1),  (6, 7, 1),  (7, 14, 4),
        (3, 8, 2),  (8, 9, 2),  (9, 14, 2),  (8, 5, 3),
        (2, 8, 3),  (6, 9, 2),  (7, 5, 5),
    ]:
        G.add_edge(u, v, weight=w)

    results = run_batch([("example", G)], s=1, t=14)

    print("\n--- Results ---")
    for r in results:
        if r["error"]:
            print(f"  [{r['algo']}] ERROR: {r['error']}")
        else:
            print(f"  [{r['algo']}] cost={r['total_cost']:.4f}  time={float(r['runtime_sec']):.4f}s  |E|={r['edge_count']}")
