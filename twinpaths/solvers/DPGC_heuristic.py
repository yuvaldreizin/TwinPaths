import time
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Set, Dict, Any, Optional
from twinpaths.viz.visualize_pyvis import visualize_dpgc_pyvis
from twinpaths.viz.visualize import visualize_graph


def _canon_edge(a: Any, b: Any) -> Tuple[Any, Any]:
    """
    Canonical undirected edge representation, robust to mixed types (int, str, etc.).
    """
    return tuple(sorted((a, b), key=str))


def _min_cost_k_edge_disjoint_paths(
    G: nx.Graph,
    s: Any,
    t: Any,
    k: int = 2,
    weight: str = "weight",
    visualize_dual_paths: bool = False,
) -> List[List[Any]]:
    """
    Find k edge-disjoint s-t paths of minimum total weight using Suurballe's algorithm.
    Returns list of k paths (each path is a list of nodes).

    Suurballe's algorithm (k=2):
      1. Find shortest path P1 via Dijkstra.
      2. Build residual graph: reverse P1 edges with negated cost, keep all others.
      3. Find shortest path P2 in residual via Bellman-Ford (negative weights exist).
      4. Cancel edges used in opposite directions; decompose into two s-t paths.

    For k != 2, falls back to sequential Dijkstra (greedy, polynomial).
    """
    if k != 2:
        # Sequential Dijkstra fallback for k != 2
        H = G.copy()
        paths: List[List[Any]] = []
        for _ in range(k):
            try:
                path = nx.dijkstra_path(H, s, t, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                raise RuntimeError(
                    f"Could not find {k} edge-disjoint paths from {s} to {t}"
                )
            paths.append(path)
            for a, b in zip(path[:-1], path[1:]):
                H.remove_edge(a, b)
        return paths

    # ── Suurballe's algorithm for k=2 ─────────────────────────────────────────

    # Step 1: shortest path P1 (Dijkstra). Keep the s-distances as node
    # potentials so step 3 can run Dijkstra instead of Bellman-Ford.
    try:
        dist, paths_from_s = nx.single_source_dijkstra(G, s, weight=weight)
        p1 = paths_from_s[t]
    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
        raise RuntimeError(f"No path from {s} to {t}")

    p1_edges: Set[Tuple[Any, Any]] = set(zip(p1[:-1], p1[1:]))

    # Step 2: build residual directed graph.
    #   - For edges on P1: add the reverse arc with negated cost, drop the forward arc.
    #   - For all other undirected edges: add both directions with original cost.
    # Each arc also carries a *reduced cost* rweight = cost + dist[a] - dist[b]
    # (Johnson reweighting). By the shortest-path property every rweight is >= 0
    # (reverse P1 arcs get exactly 0), so step 3 can use Dijkstra in O(E log V).
    R = nx.DiGraph()

    def _add_arc(a: Any, b: Any, cost: float) -> None:
        rc = cost + dist[a] - dist[b]
        if rc < 0.0:  # guard tiny float negatives from breaking Dijkstra
            rc = 0.0
        R.add_edge(a, b, weight=cost, rweight=rc)

    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1)
        if (u, v) in p1_edges:
            _add_arc(v, u, -w)  # P1 used u->v → reverse arc v->u, negated cost
        elif (v, u) in p1_edges:
            _add_arc(u, v, -w)  # P1 used v->u → reverse arc u->v, negated cost
        else:
            _add_arc(u, v, w)  # not on P1 → both directions, original cost
            _add_arc(v, u, w)

    # Step 3: shortest path P2 in the residual via Dijkstra on the reduced costs.
    # Reduced costs preserve shortest paths (the potential terms telescope), so
    # this returns the same P2 as Bellman-Ford would, in O(E log V).
    try:
        p2 = nx.dijkstra_path(R, s, t, weight="rweight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise RuntimeError(f"Could not find 2 edge-disjoint paths from {s} to {t}")

    p2_edges: Set[Tuple[Any, Any]] = set(zip(p2[:-1], p2[1:]))

    # Step 4: cancel opposing edges and decompose into two s-t paths
    canceled: Set[Tuple[Any, Any]] = set()
    for a, b in p2_edges:
        if (b, a) in p1_edges:
            canceled.add((b, a))
            canceled.add((a, b))

    combined = (p1_edges | p2_edges) - canceled

    # Decompose combined edge set into exactly 2 s-t paths
    H = nx.DiGraph()
    H.add_edges_from(combined)
    paths = []
    for _ in range(k):
        try:
            path = nx.shortest_path(H, s, t)
        except nx.NetworkXNoPath:
            break
        paths.append(path)
        for a, b in zip(path[:-1], path[1:]):
            H.remove_edge(a, b)

    if len(paths) < k:
        raise RuntimeError(
            f"Suurballe decomposition produced only {len(paths)} paths (expected {k})"
        )

    # Optional visualization of the dual paths
    if visualize_dual_paths:
        all_edges: Set[Tuple[Any, Any]] = set()
        for p in paths:
            for a, b in zip(p[:-1], p[1:]):
                all_edges.add(_canon_edge(a, b))

        print("Dual paths found:", paths)
        visualize_graph(G, solution_edges=all_edges, title="Dual Paths (Step 1)")
        plt.show()

    return paths


def _contract_subgraph(
    G: nx.Graph,
    N1: Set[Any],
    contracted_label: Any = "C",
    weight: str = "weight",
) -> nx.Graph:
    """
    Contract induced subgraph G[N1] into a single node 'contracted_label'.
    Keeps edges between nodes outside N1 as they are.
    For edges between N1 and outside node v, creates an edge (contracted_label, v)
    with weight = minimum weight among edges between any n in N1 and v.

    Additionally, for edges (contracted_label, v) we store:
        H[contracted_label][v]["_orig_edge"] = (u, v)
    where (u, v) is the original edge in G that achieved the minimal weight.
    """
    H = nx.Graph()
    # add nodes outside N1
    for n in G.nodes():
        if n not in N1:
            H.add_node(n)
    H.add_node(contracted_label)

    # gather min-edge-cost from N1 to outside nodes AND remember which u gave min
    # map: outside_node -> (min_cost, node_in_N1_that_connects)
    min_cost_to_out: Dict[Any, Tuple[float, Any]] = {}

    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1)
        if u in N1 and v in N1:
            continue  # internal, removed by contraction

        if u in N1 and v not in N1:
            # edge u-v becomes contracted_label-v
            best = min_cost_to_out.get(v)
            if best is None or w < best[0]:
                min_cost_to_out[v] = (w, u)

        elif v in N1 and u not in N1:
            # edge v-u becomes contracted_label-u
            best = min_cost_to_out.get(u)
            if best is None or w < best[0]:
                min_cost_to_out[u] = (w, v)

        else:
            # both outside N1: preserve edge in H (keep min weight if multiple)
            if H.has_edge(u, v):
                if H[u][v].get(weight, float("inf")) > w:
                    H[u][v][weight] = w
            else:
                H.add_edge(u, v, **{weight: w})

    # add edges between contracted node and outside nodes with minimal observed weight
    for v, (wmin, u_orig) in min_cost_to_out.items():
        H.add_edge(
            contracted_label,
            v,
            **{
                weight: wmin,
                "_orig_edge": (u_orig, v),
            },
        )

    return H


def dpgc_heuristic(
    G: nx.Graph,
    s: Any = 1,
    t: Any = 2,
    weight: str = "weight",
    visualize_dual_paths: bool = False,
    visualize_final: bool = False,
) -> Tuple[Set[Tuple[Any, Any]], Dict[str, Any]]:
    """
    Implements the DPGC heuristic described in Balakrishnan et al. (2004) for the DPST/DPT problem.

    Steps:
      1) Find a minimum-cost pair of edge-disjoint s-t paths -> (E1, N1).
      2) Contract subgraph induced by N1 into a node 'C', find an MST of the
         contracted graph directly, recover original edges for the MST edges and
         add those edges once to E1.

    Returns:
      - final_edges: set of undirected edges (u,v) of the solution (u,v are original nodes, no 'C')
      - info: dict with intermediate data (paths, E1, N1, contracted_graph, mst_edges, recovered_edges)
    """
    t_start = time.perf_counter()

    # Step 1: two edge-disjoint min-cost s-t paths
    paths = _min_cost_k_edge_disjoint_paths(
        G,
        s,
        t,
        k=2,
        weight=weight,
        visualize_dual_paths=visualize_dual_paths,
    )

    # E1 and N1
    E1: Set[Tuple[Any, Any]] = set()
    N1: Set[Any] = set()
    for path in paths:
        N1.update(path)
        for a, b in zip(path[:-1], path[1:]):
            edge = _canon_edge(a, b)
            E1.add(edge)

    # Step 2: contract G[N1] into single node 'C'
    contracted_label = "C"
    H = _contract_subgraph(G, N1, contracted_label=contracted_label, weight=weight)

    # MST directly on the contracted graph (no metric closure).
    # H is sparse (O(V) nodes, O(E) edges), so Kruskal is O(E log V).
    mst = nx.minimum_spanning_tree(H, weight=weight)
    mst_edges = list(mst.edges(data=True))

    # Recover original edges for each MST edge. Each MST edge is an edge of H:
    # either a contracted 'C'-v edge (map back via '_orig_edge') or a real edge
    # between two off-path nodes (use as-is). 'C' never leaks into the output.
    recovered_edges: Set[Tuple[Any, Any]] = set()
    for u, v, data in mst_edges:
        orig = data.get("_orig_edge")
        if orig is not None:
            u_orig, v_orig = orig
            edge = _canon_edge(u_orig, v_orig)
        else:
            edge = _canon_edge(u, v)
        recovered_edges.add(edge)

    # final solution edges = E1 U recovered_edges (all on original nodes)
    final_edges: Set[Tuple[Any, Any]] = set(E1) | recovered_edges

    # Optional final visualization
    if visualize_final:
        pos = visualize_graph(G, title="Original Graph")
        visualize_graph(G, solution_edges=final_edges, pos=pos, title="DPGC Solution")
        plt.show()

    # Build info dict
    info: Dict[str, Any] = {
        "algo": "dpgc_heuristic",
        "total_cost": sum(G[u][v].get(weight, 0) for u, v in final_edges),
        "runtime_sec": time.perf_counter() - t_start,
        "edges": sorted(final_edges, key=lambda e: (str(e[0]), str(e[1]))),
        # algorithm-specific details
        "paths": paths,
        "E1": sorted(E1, key=lambda e: (str(e[0]), str(e[1]))),
        "N1": sorted(N1, key=str),
        "contracted_graph": H,
        "metric_closure": None,  # no longer computed (MST runs on H directly)
        "mst_edges": [(u, v, d.get(weight, 0)) for u, v, d in mst_edges],
        "recovered_edges": sorted(recovered_edges, key=lambda e: (str(e[0]), str(e[1]))),
    }
    return final_edges, info


if __name__ == "__main__":
    # Example graph (the "richer" one we discussed)
    G = nx.Graph()
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
        G.add_edge(u, v, weight=w)

    # Run DPGC heuristic with visualization flags
    final_edges, info = dpgc_heuristic(
        G,
        s=1,
        t=14,
        weight="weight",
        visualize_dual_paths=False,
        visualize_final=False,
    )

    print("Final edges in solution:", final_edges)
    print("Intermediate info:")
    print("  paths:", info["paths"])
    print("  E1:", info["E1"])
    print("  N1:", info["N1"])
    print("  mst_edges:", info["mst_edges"])
    print("  recovered_edges:", info["recovered_edges"])


    # Extract visualization sets
    dual = {tuple(e) for e in info["E1"]}
    mst_edges = {tuple(e) for e in info["recovered_edges"]}

    filepath = visualize_dpgc_pyvis(
        G,
        dual_path_edges=dual,
        mst_edges=mst_edges,
        filename_prefix="dpgc",
        results_dir="results"
    )

    print("Visualization written to:", filepath)
