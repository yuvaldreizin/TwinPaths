import os
import time
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from twinpaths.viz.visualize_pyvis import visualize_dpgc_pyvis
from twinpaths.viz.visualize import visualize_graph

"""
Brute-force baseline for k=2 edge-disjoint paths + connecting remaining nodes.
Aligned API with dpgc_heuristic and matroid_dpt:
    mst_k2_generator(graph, s=1, t=2, weight="weight", visualize_dual_paths=False, visualize_final=False, ...)
returns (final_edges_set, info_dict) where info_dict includes runtime_sec and total_cost.
"""


def _canon_edge(a: Any, b: Any) -> Tuple[Any, Any]:
    """Canonical undirected edge key."""
    return (a, b) if str(a) <= str(b) else (b, a)

def are_paths_independant(path1, path2, start_node, end_node):
    """
    True iff path1 and path2 share no edges AND no intermediate nodes
    (terminals start_node / end_node may be shared).
    """
    # Edge-disjointness: reject if any directed step appears in both paths
    edges1 = {(step[0], step[1]) for step in path1}
    for step in path2:
        if (step[0], step[1]) in edges1:
            return False

    # Node-disjointness of intermediate nodes
    terminals = {start_node, end_node}
    occupied = set()
    for step in path1:
        for node in (step[0], step[1]):
            if node not in terminals:
                occupied.add(node)
    for step in path2:
        for node in (step[0], step[1]):
            if node not in terminals and node in occupied:
                return False
    return True

def find_paths_aux(graph, current, end, visited, steps, all_paths):
    if current == end:
        all_paths.append(steps.copy())
        return
    
    n = np.size(graph, 0)
    for nxt in range(n):
        if graph[current][nxt] != 0 and nxt not in visited:
            # take step
            visited.add(nxt)
            steps.append([current, nxt, float(graph[current][nxt])])
            
            find_paths_aux(graph, nxt, end, visited, steps, all_paths)
            
            # backtrack
            steps.pop()
            visited.remove(nxt)

def mst_k2_generator(
    graph: nx.Graph,
    s: Any = 1,
    t: Any = 2,
    weight: str = "weight",
    visualize_dual_paths: bool = False,
    visualize_final: bool = False,
    filename_prefix: str = "ground_truth",
    results_dir: str = "results",
) -> Tuple[Set[Tuple[Any, Any]], Dict[str, Any]]:
    """
    Exhaustive search for two edge-disjoint s-t paths plus cheapest attachments
    of remaining nodes. Returns (final_edges, info) aligned with dpgc_heuristic.
    """
    t_start = time.time()
    filepath = None

    nodes_list = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    if s not in node_to_idx or t not in node_to_idx:
        raise ValueError("s and t must be nodes in the graph")
    s_idx = node_to_idx[s]
    t_idx = node_to_idx[t]

    arr = nx.to_numpy_array(graph, weight=weight)

    all_paths: List[List[List[float]]] = []
    visited = {s_idx}
    steps: List[List[float]] = []
    find_paths_aux(arr, s_idx, t_idx, visited, steps, all_paths)

    best_total_weight = float("inf")
    best_paths: Tuple[List[List[float]], List[List[float]]] = ([], [])
    best_added: List[List[float]] = []

    for i in range(len(all_paths)):
        for j in range(len(all_paths)):
            if not are_paths_independant(all_paths[i], all_paths[j], s_idx, t_idx):
                continue

            added_connections: List[List[float]] = []
            included_nodes: List[int] = [t_idx]
            total_weights = 0.0

            for step in all_paths[i]:
                total_weights += step[2]
                included_nodes.append(step[0])
            for step in all_paths[j]:
                total_weights += step[2]
                if step[0] not in included_nodes:
                    included_nodes.append(step[0])
                if step[1] not in included_nodes:
                    included_nodes.append(step[1])

            unincluded_nodes = [n for n in range(len(arr)) if n not in included_nodes]

            while unincluded_nodes:
                cheapest_connection_weight = 0.0
                cheapest_connection: List[int] = []
                for node in unincluded_nodes:
                    for connection_node in included_nodes:
                        if arr[node][connection_node] != 0:
                            if (
                                not cheapest_connection
                                or cheapest_connection_weight > arr[node][connection_node]
                            ):
                                cheapest_connection = [node, connection_node]
                                cheapest_connection_weight = arr[node][connection_node]

                if cheapest_connection_weight != 0.0:
                    new_connection = [
                        cheapest_connection[0],
                        cheapest_connection[1],
                        float(cheapest_connection_weight),
                    ]
                    included_nodes.append(cheapest_connection[0])
                    unincluded_nodes.remove(cheapest_connection[0])
                    total_weights += cheapest_connection_weight
                    added_connections.append(new_connection)
                else:
                    break

            if total_weights < best_total_weight:
                best_total_weight = total_weights
                best_paths = (all_paths[i], all_paths[j])
                best_added = added_connections

    if best_total_weight == float("inf"):
        info = {
            "paths": [],
            "E1": [],
            "N1": [],
            "contracted_graph": None,
            "metric_closure": None,
            "mst_edges": [],
            "recovered_edges": [],
            "total_cost": 0.0,
            "runtime_sec": time.time() - t_start,
            "algo": "ground_truth_bruteforce",
            "visualization": filepath,
        }
        return set(), info

    def _steps_to_nodes(path_steps: List[List[float]]) -> List[Any]:
        if not path_steps:
            return []
        nodes_seq = [path_steps[0][0]] + [step[1] for step in path_steps]
        return [nodes_list[int(idx)] for idx in nodes_seq]

    path1_nodes = _steps_to_nodes(best_paths[0])
    path2_nodes = _steps_to_nodes(best_paths[1])

    dual_path_edges: Set[Tuple[Any, Any]] = set()
    recovered_edges: Set[Tuple[Any, Any]] = set()

    for path_steps in best_paths:
        for step in path_steps:
            u = nodes_list[int(step[0])]
            v = nodes_list[int(step[1])]
            dual_path_edges.add(_canon_edge(u, v))

    for conn in best_added:
        u = nodes_list[int(conn[0])]
        v = nodes_list[int(conn[1])]
        recovered_edges.add(_canon_edge(u, v))

    final_edges: Set[Tuple[Any, Any]] = set(dual_path_edges) | recovered_edges

    if visualize_dual_paths or visualize_final:
        filepath = visualize_dpgc_pyvis(
            graph,
            dual_path_edges=dual_path_edges,
            mst_edges=recovered_edges,
            filename_prefix=filename_prefix,
            results_dir=results_dir,
        )

    total_cost = best_total_weight
    runtime_sec = time.time() - t_start

    info: Dict[str, Any] = {
        "algo": "ground_truth_bruteforce",
        "setup": {"s": s, "t": t, "weight": weight},
        "runtime_sec": runtime_sec,
        "total_cost": total_cost,
        "edges": sorted(final_edges, key=lambda e: (str(e[0]), str(e[1]))),
        "visualization": filepath,
    }
    return final_edges, info


if __name__ == "__main__":

    # Example graph construction for testing
    graph = nx.Graph()
    edges = [
            # Core block around s=1 and t=14
            (1, 2, 1),
            (2, 3, 2),
            (3, 14, 3),
            (1, 4, 2),
            (4, 5, 2),
            (5, 14, 2),

            # Extra alternate cheap path
            (1, 6, 1),
            (6, 7, 1),
            (7, 14, 4),

            # A diamond cycle in the middle
            (3, 8, 2),
            (8, 9, 2),
            (9, 14, 2),
            (8, 5, 3),

            # A long misleading "tempting" but expensive path
            (1, 10, 5),
            (10, 11, 8),
            (11, 12, 8),
            (12, 13, 3),
            (13, 14, 6),

            # Short cross-links to create competing MST choices
            (2, 8, 3),
            (6, 9, 2),
            (7, 5, 5),
    ]
    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)

    # finding the k=2 shortest path by checking every possible combination
    solution_edges, info = mst_k2_generator(graph, s=1, t=14, visualize_final=False)
    print("Ground-truth solution edges:", sorted(solution_edges))
    print("Total cost:", info["total_cost"])
    print("Runtime (sec):", info["runtime_sec"])