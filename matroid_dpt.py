import math
from collections import deque, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


def _find_two_edge_disjoint_paths(
    G: nx.Graph,
    edges: Set[Tuple[Any, Any]],
    s: Any,
    t: Any,
    weight: str = "weight",
) -> List[List[Any]]:
    """Extract up to two edge-disjoint s-t paths from the provided edge set."""
    H = nx.Graph()
    for u, v in edges:
        w = G[u][v].get(weight, 1)
        H.add_edge(u, v, **{weight: w})
    try:
        paths_gen = nx.edge_disjoint_paths(H, s, t)
        paths = []
        for p in paths_gen:
            paths.append(p)
            if len(paths) == 2:
                break
        return paths
    except Exception:
        return []


def _visualize_pyvis_solution(
    G: nx.Graph,
    solution_edges: Set[Tuple[Any, Any]],
    s: Any,
    t: Any,
    dual_paths: Optional[List[List[Any]]] = None,
    filename_prefix: str = "matroid_dpt",
    results_dir: str = "results",
) -> str:
    """Render solution with two colored paths and legend via PyVis."""
    import os
    from datetime import datetime
    from pyvis.network import Network

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"{filename_prefix}_{ts}.html")

    path1_edges: Set[frozenset] = set()
    path2_edges: Set[frozenset] = set()
    if dual_paths:
        if len(dual_paths) > 0:
            for a, b in zip(dual_paths[0][:-1], dual_paths[0][1:]):
                path1_edges.add(frozenset({a, b}))
        if len(dual_paths) > 1:
            for a, b in zip(dual_paths[1][:-1], dual_paths[1][1:]):
                path2_edges.add(frozenset({a, b}))

    net = Network(height="900px", width="100%", directed=False, bgcolor="#ffffff", font_color="black")
    net.toggle_physics(True)

    # Nodes
    for n in G.nodes():
        is_term = n == s or n == t
        net.add_node(
            n,
            label=str(n),
            shape="circle",
            color={"background": "#2ca02c" if is_term else "white", "border": "black"},
            borderWidth=2,
            font={"size": 26},
        )

    # Edges
    for u, v, data in G.edges(data=True):
        key = frozenset({u, v})
        w = data.get("weight", "")
        in_solution = key in {frozenset(e) for e in solution_edges}
        if key in path1_edges:
            color = "#d62728"
            width = 6
            kind = "Path 1"
        elif key in path2_edges:
            color = "#1f77b4"
            width = 6
            kind = "Path 2"
        elif in_solution:
            color = "#494949"
            width = 3
            kind = "Solution edge"
        else:
            color = "#e0e0e0"
            width = 1.5
            kind = "Non-solution"
        net.add_edge(
            u,
            v,
            label=str(w),
            color=color,
            width=width,
            smooth=False,
            font={"size": 18},
            title=f"{kind}, weight={w}",
        )

    # Legend node (fixed)
    total_cost = sum(G[u][v].get("weight", 0) for u, v in solution_edges)
    legend_lines = [
        "Legend",
        f"Selected Edges: {s}, {t} (green)",
        f"Total cost: {total_cost}",
        f"|E|={len(solution_edges)}, |V|={G.number_of_nodes()}",
    ]
    net.add_node(
        "__legend__",
        label="\n".join(legend_lines),
        shape="box",
        color={"background": "#f7f7f7", "border": "black"},
        fixed=True,
        x=-800,
        y=-800,
        physics=False,
        font={"size": 20},
    )

    net.write_html(filepath)
    return filepath


############################################################
# Utility helpers
############################################################

def _canon_edge(a: Any, b: Any) -> Tuple[Any, Any]:
    """Undirected edge key, robust to mixed hashables."""
    return (a, b) if str(a) <= str(b) else (b, a)


def is_triangle_inequality_satisfied(
    G: nx.Graph, weight: str = "weight", tol: float = 1e-9
) -> Tuple[bool, List[Tuple[Any, Any, float, float]]]:
    """
    Check triangle inequality on all edges (non-edges are ignored).

    Returns (ok, violations) where each violation is (u, v, w_uv, sp_dist_uv)
    meaning direct edge weight w_uv is larger than the shortest-path distance.
    """
    # All-pairs shortest paths (nonnegative assumed)
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    violations: List[Tuple[Any, Any, float, float]] = []
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        sp = dist[u].get(v, math.inf)
        if w - sp > tol:
            violations.append((u, v, w, sp))
    return (len(violations) == 0, violations)


def metric_closure_graph(G: nx.Graph, weight: str = "weight") -> nx.Graph:
    """Return the metric closure (complete graph weighted by shortest paths)."""
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    H = nx.Graph()
    for u in G.nodes():
        for v in G.nodes():
            if u == v:
                continue
            d = dist[u].get(v, math.inf)
            if d < math.inf:
                if H.has_edge(u, v):
                    H[u][v]["weight"] = min(H[u][v]["weight"], d)
                else:
                    H.add_edge(u, v, weight=d)
    return H


############################################################
# Q-restricted 1-tree matroid
############################################################

class QRestrictedOneTreeMatroid:
    """
    Matroid of q-restricted 1-trees: at most one cycle and, if present, it must
    contain the distinguished node q.

    Ground set elements are edges of G in the fixed order provided at init.
    """

    def __init__(self, G: nx.Graph, q: Any, weight_attr: str = "weight"):
        self.G = G
        self.q = q
        self.weight_attr = weight_attr
        self.edges: List[Tuple[Any, Any]] = []
        self.weights: List[float] = []
        self.edge_index: Dict[Tuple[Any, Any], int] = {}
        for idx, (u, v, data) in enumerate(G.edges(data=True)):
            key = _canon_edge(u, v)
            self.edge_index[key] = idx
            self.edges.append((u, v))
            self.weights.append(float(data.get(weight_attr, 1.0)))

        self.n = len(self.edges)

        # State set by set(I)
        self.I: List[bool] = [False] * self.n
        self.adj: Dict[Any, List[Tuple[Any, int]]] = {}
        self.comp_id: Dict[Any, int] = {}
        self.comp_cycle_edges: Dict[int, Set[int]] = {}
        self.comp_cycle_has_q: Dict[int, bool] = {}

    def size(self) -> int:
        return self.n

    def set(self, I: List[bool]) -> None:
        """Update internal state for the current independent set I."""
        assert len(I) == self.n
        self.I = list(I)

        self.adj = defaultdict(list)
        for idx, in_set in enumerate(I):
            if not in_set:
                continue
            u, v = self.edges[idx]
            self.adj[u].append((v, idx))
            self.adj[v].append((u, idx))

        self.comp_id = {}
        self.comp_cycle_edges = {}
        self.comp_cycle_has_q = {}

        visited: Set[Any] = set()
        comp_counter = 0

        for node in self.G.nodes():
            if node in visited:
                continue
            comp_nodes: List[Any] = []
            stack = [(node, None, None)]  # (current, parent, edge_idx_from_parent)
            parent: Dict[Any, Any] = {}
            parent_edge: Dict[Any, int] = {}
            found_cycle_edges: Optional[Set[int]] = None
            found_cycle_has_q = False

            while stack:
                cur, par, pedge = stack.pop()
                if cur in visited:
                    # Found a back edge -> cycle
                    if par is not None:
                        cycle_edges = {pedge}
                        x = par
                        while x != cur and x is not None:
                            pe = parent_edge.get(x)
                            if pe is None:
                                break  # safety: missing parent edge should not crash
                            cycle_edges.add(pe)
                            x = parent.get(x)
                        found_cycle_edges = cycle_edges
                        found_cycle_has_q = (self.q == cur) or any(
                            self.q == x for x in self._path_nodes(cur, par, parent)
                        )
                    continue
                visited.add(cur)
                comp_nodes.append(cur)
                if par is not None:
                    parent[cur] = par
                    parent_edge[cur] = pedge
                for nxt, eidx in self.adj.get(cur, []):
                    if nxt == par:
                        continue
                    stack.append((nxt, cur, eidx))

            for n in comp_nodes:
                self.comp_id[n] = comp_counter

            if found_cycle_edges:
                self.comp_cycle_edges[comp_counter] = found_cycle_edges
                # Ensure cycle nodes list contains q if present
                if not found_cycle_has_q:
                    # Re-evaluate via explicit traversal of the stored cycle
                    nodes_on_cycle = self._nodes_from_edge_set(found_cycle_edges)
                    found_cycle_has_q = self.q in nodes_on_cycle
                self.comp_cycle_has_q[comp_counter] = found_cycle_has_q
            else:
                self.comp_cycle_edges[comp_counter] = set()
                self.comp_cycle_has_q[comp_counter] = False

            comp_counter += 1

    def _nodes_from_edge_set(self, edge_set: Iterable[int]) -> Set[Any]:
        nodes: Set[Any] = set()
        for idx in edge_set:
            u, v = self.edges[idx]
            nodes.add(u)
            nodes.add(v)
        return nodes

    def _path_edges(self, start: Any, goal: Any) -> Tuple[List[int], List[Any]]:
        """Return (edge_idx path, node path) between start and goal in current I."""
        if start == goal:
            return [], [start]
        dq = deque([start])
        prev_node: Dict[Any, Any] = {start: None}
        prev_edge: Dict[Any, int] = {}
        while dq:
            cur = dq.popleft()
            for nxt, eidx in self.adj.get(cur, []):
                if nxt in prev_node:
                    continue
                prev_node[nxt] = cur
                prev_edge[nxt] = eidx
                if nxt == goal:
                    dq.clear()
                    break
                dq.append(nxt)
        if goal not in prev_node:
            return [], []  # Disconnected
        path_edges: List[int] = []
        path_nodes: List[Any] = [goal]
        x = goal
        while prev_node[x] is not None:
            path_edges.append(prev_edge[x])
            x = prev_node[x]
            path_nodes.append(x)
        path_edges.reverse()
        path_nodes.reverse()
        return path_edges, path_nodes

    def _path_nodes(self, start: Any, goal: Any, parent: Dict[Any, Any]) -> List[Any]:
        """Helper used during cycle detection; walk parents until goal."""
        path = [start]
        x = start
        while x != goal and x in parent:
            x = parent[x]
            path.append(x)
        return path

    def circuit(self, e: int) -> List[int]:
        """
        Return the circuit created by adding element e to I, or [] if still independent.
        """
        if self.I[e]:
            return []

        u, v = self.edges[e]
        cu = self.comp_id.get(u, None)
        cv = self.comp_id.get(v, None)

        # Different components -> no cycle introduced
        if cu is None or cv is None or cu != cv:
            return []

        comp = cu
        has_cycle = bool(self.comp_cycle_edges.get(comp))
        cycle_edges = self.comp_cycle_edges.get(comp, set())

        path_edges, path_nodes = self._path_edges(u, v)
        if not path_edges and u != v:
            # Disconnected within component? Treat as independent fallback
            return []

        cycle_candidate = set(path_edges + [e])
        cycle_nodes = set(path_nodes + [u, v])

        if not has_cycle:
            # First cycle being created; must include q to remain independent
            if self.q in cycle_nodes:
                return []
            return list(cycle_candidate)

        # Already have one cycle that (by independence) includes q.
        # Any new cycle yields dependency; circuit = old cycle ∪ new cycle.
        return list(set(cycle_edges) | cycle_candidate)


############################################################
# Weighted matroid intersection (port of cplib algorithm)
############################################################

def _bellman_ford_path(
    n_nodes: int, edges: List[Tuple[int, int, float]], src: int, dst: int
) -> List[int]:
    dist = [math.inf] * n_nodes
    prev = [-1] * n_nodes
    dist[src] = 0.0
    for _ in range(n_nodes - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break
    if dist[dst] == math.inf:
        return []
    path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def weighted_matroid_intersection(
    m1: Any, m2: Any, weights: Optional[List[float]] = None
) -> List[bool]:
    """
    Weighted matroid intersection following cplib's matroid_intersection.hpp.
    m1, m2 must expose size(), set(I), circuit(e).
    """
    n = m1.size()
    assert m2.size() == n
    if weights is None:
        weights = [0.0] * n
    assert len(weights) == n

    I = [False] * n

    # Greedy init for unweighted case
    if all(w == 0 for w in weights):
        m1.set(I)
        m2.set(I)
        for e in range(n):
            if not m1.circuit(e) and not m2.circuit(e):
                I[e] = True
                m1.set(I)
                m2.set(I)

    while True:
        m1.set(I)
        m2.set(I)
        edges: List[Tuple[int, int, float]] = []
        gs = n
        gt = n + 1
        for e in range(n):
            if I[e]:
                continue
            c1 = m1.circuit(e)
            c2 = m2.circuit(e)
            if not c1:
                edges.append((e, gt, 0.0))
            for f in c1:
                if f != e:
                    edges.append((e, f, -weights[f] + 1.0))
            if not c2:
                edges.append((gs, e, weights[e] + 1.0))
            for f in c2:
                if f != e:
                    edges.append((f, e, weights[e] + 1.0))

        path = _bellman_ford_path(n + 2, edges, gs, gt)
        if not path:
            break
        for node in path:
            if node in (gs, gt):
                continue
            I[node] = not I[node]

    return I


############################################################
# Public API
############################################################

def solve_dpt_matroid(
    G: nx.Graph,
    s: Any = 1,
    t: Any = 2,
    weight: str = "weight",
    enforce_metric_closure: bool = False,
    check_metric: bool = False,
    metric_tol: float = 1e-9,
    visualize_final: bool = False,
    visualize_html: bool = False,
    filename_prefix: str = "matroid_dpt",
    results_dir: str = "results",
    layout_seed: int = 42,
) -> Tuple[Set[Tuple[Any, Any]], Dict[str, Any]]:
    """
    Solve DPT with triangle inequality using weighted matroid intersection.

    Returns (edge_set, info).
    """
    if check_metric:
        ok, violations = is_triangle_inequality_satisfied(G, weight=weight, tol=metric_tol)
        if not ok:
            print(f"Triangle inequality violated on {len(violations)} edge(s).")

    G_use = metric_closure_graph(G, weight=weight) if enforce_metric_closure else G

    m1 = QRestrictedOneTreeMatroid(G_use, q=s, weight_attr=weight)
    m2 = QRestrictedOneTreeMatroid(G_use, q=t, weight_attr=weight)
    weights = m1.weights  # identical ordering for both matroids

    I = weighted_matroid_intersection(m1, m2, weights=weights)
    solution_edges: Set[Tuple[Any, Any]] = set()
    for idx, take in enumerate(I):
        if take:
            u, v = m1.edges[idx]
            solution_edges.add(_canon_edge(u, v))

    m1.set(I)
    m2.set(I)

    dual_paths = _find_two_edge_disjoint_paths(G_use, solution_edges, s, t, weight=weight)

    pyvis_file = None
    if visualize_html:
        pyvis_file = _visualize_pyvis_solution(
            G_use,
            solution_edges=solution_edges,
            s=s,
            t=t,
            dual_paths=dual_paths,
            filename_prefix=filename_prefix,
            results_dir=results_dir,
        )

    if visualize_final:
        from utils.visualize import visualize_graph  # lazy import
        import matplotlib.pyplot as plt

        visualize_graph(
            G_use,
            solution_edges=solution_edges,
            title="Matroid DPT solution",
            layout_seed=layout_seed,
        )
        plt.show()

    info = {
        "edge_count": len(solution_edges),
        "expected_edges": G_use.number_of_nodes(),
        "enforce_metric_closure": enforce_metric_closure,
        "metric_checked": check_metric,
        "triangle_ok": ok if check_metric else None,
        "rank1": sum(I),
        "rank2": sum(I),
        "dual_paths": dual_paths,
        "pyvis_file": pyvis_file,
    }
    return solution_edges, info


if __name__ == "__main__":
    # General example to exercise the matroid DPT solver.
    # This mirrors the example used in DPGC_heuristic.__main__.
    edges = [
        (1, 2, 1),
        (2, 3, 2),
        (3, 14, 3),
        (1, 4, 2),
        (4, 5, 2),
        (5, 14, 2),
        (1, 6, 1),
        (6, 7, 1),
        (7, 14, 4),
        (3, 8, 2),
        (8, 9, 2),
        (9, 14, 2),
        (8, 5, 3),
        (1, 10, 5),
        (10, 11, 8),
        (11, 12, 8),
        (12, 13, 3),
        (13, 14, 6),
        (2, 8, 3),
        (6, 9, 2),
        (7, 5, 5),
    ]

    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    solution, info = solve_dpt_matroid(
        G,
        s=1,
        t=14,
        check_metric=True,
        enforce_metric_closure=False,
        visualize_final=False,
        visualize_html=True,
    )

    total_cost = sum(G[u][v]["weight"] for u, v in solution)
    print("Matroid DPT solution edges ({}):".format(len(solution)))
    print(sorted(solution))
    print("Total cost:", total_cost)
    print("Info:", info)


