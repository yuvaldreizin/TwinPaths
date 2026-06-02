import os
from datetime import datetime
from pyvis.network import Network
import networkx as nx


def visualize_dpgc_pyvis(
    G: nx.Graph,
    dual_path_edges=None,
    mst_edges=None,
    filename_prefix="dpgc_solution",
    height="900px",
    width="100%",
    results_dir="results"
):
    """
    Interactive PyVis visualization for DPGC solution.
    Shows:
      - node numbers
      - edge weights (ONLY)
      - colored solution edges (dual paths = red, mst edges = blue)
    """

    dual_path_edges = {frozenset(e) for e in (dual_path_edges or [])}
    mst_edges = {frozenset(e) for e in (mst_edges or [])}

    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.html"
    filepath = os.path.join(results_dir, filename)

    # PyVis graph
    net = Network(
        height=height,
        width=width,
        directed=False,
        bgcolor="#ffffff",
        font_color="black"
    )

    # --- ADD NODES WITH LABELS ---
    for n in G.nodes():
        net.add_node(
            n,
            label=str(n),  # show node number
            shape="circle",
            color={"background": "white", "border": "black"},
            borderWidth=2,
            font={"size": 28},
        )

    # --- ADD EDGES WITH WEIGHTS ONLY ---
    for u, v, data in G.edges(data=True):

        w = data.get("weight", "")
        key = frozenset([u, v])

        # Default style
        color = "#cccccc"
        width = 1.5

        # Highlight dual-path edges
        if key in dual_path_edges:
            color = "#ff3333"
            width = 6

        # Highlight MST completion edges
        elif key in mst_edges:
            color = "#0077ff"
            width = 5

        net.add_edge(
            u,
            v,
            label=str(w),  # <-- ONLY WEIGHT
            color=color,
            width=width,
            smooth=False,
            font={"size": 18}
        )

    net.toggle_physics(True)
    net.write_html(filepath)

    return filepath
