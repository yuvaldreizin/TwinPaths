import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
from typing import Dict, Tuple, Any, Iterable, Optional, List, Set


def _normalize_undirected_edge(edge: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """Return a canonical (u, v) with u < v for undirected (simple) edges."""
    u, v = edge
    return (u, v) if u <= v else (v, u)


def visualize_graph(
    G: nx.Graph,
    solution_edges: Optional[Iterable[Tuple[Any, Any]]] = None,
    pos: Optional[Dict[Any, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    weight_attr: str = "weight",
    layout_seed: int = 42,
) -> Dict[Any, Tuple[float, float]]:
    """
    Generic graph visualization helper.

    Parameters
    ----------
    G : nx.Graph
        Graph to draw.
    solution_edges : iterable of (u, v), optional
        Edges to highlight (e.g., algorithm solution).
    pos : dict, optional
        Node positions. If None, a spring layout is computed.
        Returned so you can reuse it across calls.
    title : str, optional
        Figure title.
    weight_attr : str
        Edge-attribute name with weights.
    layout_seed : int
        Seed for layout determinism.

    Returns
    -------
    pos : dict
        Node positions used for drawing.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=layout_seed)

    plt.figure(figsize=(6, 4))

    # Nodes + labels
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="#ffffff", edgecolors="black")
    nx.draw_networkx_labels(G, pos)

    # Edges (optionally highlighted)
    if solution_edges is not None:
        sol = {_normalize_undirected_edge(e) for e in solution_edges}
        colors = []
        widths = []
        for u, v in G.edges():
            e = _normalize_undirected_edge((u, v))
            if e in sol:
                colors.append("tab:blue")
                widths.append(3.0)
            else:
                colors.append("lightgray")
                widths.append(1.0)
        nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths)
    else:
        nx.draw_networkx_edges(G, pos)

    # Edge weights
    weights = nx.get_edge_attributes(G, weight_attr)
    if weights:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    return pos
