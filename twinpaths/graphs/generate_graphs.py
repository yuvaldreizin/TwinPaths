"""
generate_graphs.py
------------------
Generators for weighted graphs that satisfy the triangle inequality.

The triangle inequality holds for a weighted graph when, for every edge (u, v):
    w(u, v) <= w(u, x) + w(x, v)   for all nodes x

All generators here place nodes in 2D Euclidean space and use Euclidean
distances as edge weights, which inherently satisfies this property.

Functions
---------
random_euclidean_graph(n, edge_prob, seed)
    Sparse random graph; edges sampled with given probability.

complete_euclidean_graph(n, seed)
    Complete graph; every pair of nodes is connected.

grid_graph(rows, cols)
    Rectangular grid with unit edge weights.

Usage::

    from twinpaths.graphs import random_euclidean_graph, complete_euclidean_graph
    G = random_euclidean_graph(8, edge_prob=0.5, seed=0)
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _ensure_connected(
    G: nx.Graph,
    coords: List[Tuple[float, float]],
    node_ids: List[int],
) -> None:
    """Add minimum-weight edges between disconnected components until G is connected."""
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        best: Optional[Tuple[int, int, float]] = None
        for i, c1 in enumerate(comps):
            for c2 in comps[i + 1 :]:
                for u in c1:
                    for v in c2:
                        w = _euclidean(coords[node_ids.index(u)], coords[node_ids.index(v)])
                        if best is None or w < best[2]:
                            best = (u, v, w)
        if best:
            G.add_edge(best[0], best[1], weight=round(best[2], 4))
        else:
            break  # unreachable in practice


def _ensure_2edge_connected(
    G: nx.Graph,
    coords: List[Tuple[float, float]],
    node_ids: List[int],
) -> None:
    """
    Add cheapest missing edges across every bridge until the graph is
    2-edge-connected (edge connectivity >= 2).
    """
    while True:
        bridges = list(nx.bridges(G))
        if not bridges:
            break
        # For each bridge, find the cheapest alternative edge that bypasses it
        added = False
        for u_br, v_br in bridges:
            # Nodes on each side after removing the bridge
            bridge_weight = G[u_br][v_br].get("weight", 1)
            G.remove_edge(u_br, v_br)
            side_u = set(nx.node_connected_component(G, u_br))
            G.add_edge(u_br, v_br, weight=bridge_weight)

            best: Optional[Tuple[int, int, float]] = None
            for u in side_u:
                for v in set(G.nodes()) - side_u:
                    if G.has_edge(u, v):
                        continue
                    w = _euclidean(coords[node_ids.index(u)], coords[node_ids.index(v)])
                    if best is None or w < best[2]:
                        best = (u, v, w)
            if best:
                G.add_edge(best[0], best[1], weight=round(best[2], 4))
                added = True
                break  # recompute bridges after each addition
        if not added:
            break


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def random_euclidean_graph(
    n: int,
    edge_prob: float = 0.4,
    seed: Optional[int] = None,
    coord_range: float = 100.0,
    ensure_2_connected: bool = True,
) -> nx.Graph:
    """
    Random sparse graph with Euclidean edge weights (satisfies triangle inequality).

    Parameters
    ----------
    n                  : number of nodes (labeled 1..n)
    edge_prob          : probability that each pair of nodes shares an edge
    seed               : RNG seed for reproducibility
    coord_range        : nodes are placed in [0, coord_range]^2
    ensure_2_connected : if True, add cheapest missing edges until the graph has
                         edge-connectivity >= 2 (needed by all three solvers)

    Returns
    -------
    nx.Graph with 'weight' edge attributes
    """
    rng = random.Random(seed)
    coords = [(rng.uniform(0, coord_range), rng.uniform(0, coord_range)) for _ in range(n)]
    node_ids = list(range(1, n + 1))

    G = nx.Graph()
    for i, nid in enumerate(node_ids):
        G.add_node(nid, pos=coords[i])

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                w = _euclidean(coords[i], coords[j])
                G.add_edge(node_ids[i], node_ids[j], weight=round(w, 4))

    _ensure_connected(G, coords, node_ids)

    if ensure_2_connected:
        _ensure_2edge_connected(G, coords, node_ids)

    return G


def complete_euclidean_graph(
    n: int,
    seed: Optional[int] = None,
    coord_range: float = 100.0,
) -> nx.Graph:
    """
    Complete graph with Euclidean distances (always satisfies triangle inequality).

    Every pair of nodes is connected, so edge connectivity = n-1.
    Suitable for the matroid solver's triangle-cost assumption.

    Parameters
    ----------
    n          : number of nodes (labeled 1..n)
    seed       : RNG seed
    coord_range: nodes are placed in [0, coord_range]^2
    """
    rng = random.Random(seed)
    coords = [(rng.uniform(0, coord_range), rng.uniform(0, coord_range)) for _ in range(n)]

    G = nx.Graph()
    for i in range(n):
        G.add_node(i + 1, pos=coords[i])

    for i in range(n):
        for j in range(i + 1, n):
            w = _euclidean(coords[i], coords[j])
            G.add_edge(i + 1, j + 1, weight=round(w, 4))

    return G


def grid_graph(rows: int, cols: int) -> nx.Graph:
    """
    Rectangular grid graph with unit edge weights (satisfies triangle inequality).

    Nodes are labeled 1..(rows*cols) in row-major order.
    Edges connect 4-neighbors (left/right/up/down) with weight 1.0.
    """
    G = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c + 1
            G.add_node(nid, pos=(c, r))

    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c + 1
            if c + 1 < cols:
                G.add_edge(nid, nid + 1, weight=1.0)
            if r + 1 < rows:
                G.add_edge(nid, nid + cols, weight=1.0)

    return G


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from experiments.batch_run_algorithms import run_batch

    # Keep n small (<=8) so the brute-force ground truth finishes quickly.
    test_cases = [
        ("random_n5_p05",  random_euclidean_graph(5, edge_prob=0.5, seed=1)),
        ("random_n6_p05",  random_euclidean_graph(6, edge_prob=0.5, seed=2)),
        ("random_n8_p04",  random_euclidean_graph(8, edge_prob=0.4, seed=3)),
        ("complete_n5",    complete_euclidean_graph(5, seed=10)),
        ("complete_n6",    complete_euclidean_graph(6, seed=11)),
        ("grid_3x3",       grid_graph(3, 3)),
    ]

    print("Graph summary:")
    for name, G in test_cases:
        print(f"  {name:<20}  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")

    print("\nRunning all algorithms (s=1, t=last node)...\n")

    all_results = []
    for name, G in test_cases:
        t_node = max(G.nodes())
        all_results += run_batch([(name, G)], s=1, t=t_node,
                                 output_csv=f"results/gen_{name}.csv")

    print("\n{:<22} {:<28} {:>10} {:>12} {:>6}".format(
        "graph", "algo", "cost", "time (s)", "|E|"))
    print("-" * 82)
    for r in all_results:
        if r["error"]:
            print(f"  {r['graph']:<20} {r['algo']:<28} ERROR: {r['error']}")
        else:
            print("{:<22} {:<28} {:>10.4f} {:>12.5f} {:>6}".format(
                r["graph"], r["algo"],
                float(r["total_cost"]), float(r["runtime_sec"]), r["edge_count"],
            ))
