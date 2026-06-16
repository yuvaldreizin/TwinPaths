"""
illustrate_steps.py
===================
Generate step-by-step illustration figures of how the DPGC heuristic and the
Matroid DPT solver build their solution on a small dummy graph — from the
original graph through each intermediate stage to the final dual-path tree.

Outputs (under results/):
    dpgc_steps.png      DPGC: original -> 2 disjoint paths -> contract -> MST -> lift
    matroid_steps.png   Matroid: original -> common independent set grows edge by edge

Usage (from the repository root):
    python -m experiments.illustrate_steps
"""

import math
import os

import matplotlib.pyplot as plt
import networkx as nx

from twinpaths.solvers.DPGC_heuristic import dpgc_heuristic
from twinpaths.solvers.matroid_dpt import solve_dpt_matroid

# ── dark-theme palette (matches the rest of the repo) ──────────────────────────
BG, CARD, BORDER, MUTED, TEXT = '#09090b', '#18181b', '#27272a', '#71717a', '#fafafa'
CYAN, VIOLET, RED, GREEN, AMBER = '#22d3ee', '#a78bfa', '#f87171', '#34d399', '#fbbf24'
GREY = '#3f3f46'
NODE_FACE, NODE_EDGE = '#fafafa', '#27272a'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'text.color': TEXT,
    'font.family': 'sans-serif',
})

S, T = 1, 6


def dummy_graph():
    """Small metric (Euclidean) 2-edge-connected graph with fixed positions."""
    pos = {
        1: (0.0, 1.0),    # s
        2: (2.0, 2.2),    # top path
        3: (2.0, -0.2),   # bottom path
        4: (0.6, 3.0),    # leaf to attach (near top)
        5: (3.4, -1.2),   # leaf to attach (near bottom)
        6: (4.0, 1.0),    # t
    }
    edge_pairs = [(1, 2), (2, 6), (1, 3), (3, 6), (1, 4), (2, 4), (3, 5), (5, 6)]
    G = nx.Graph()
    for n, p in pos.items():
        G.add_node(n, pos=p)
    for u, v in edge_pairs:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        w = round(math.hypot(x1 - x2, y1 - y2), 2)
        G.add_edge(u, v, weight=w)
    return G, pos


# ── drawing helpers ────────────────────────────────────────────────────────────

def _edge_list(edges):
    """Normalize an iterable of (u,v) pairs to plain tuples."""
    return [tuple(e) for e in edges]


def draw_base(ax, G, pos, title, *, highlight=None, node_colors=None,
              show_weights=True, special_nodes=None):
    """Draw G; `highlight` maps a color -> (edge_list, width)."""
    ax.set_facecolor(BG)
    highlight = highlight or {}
    highlighted = set()
    for _, (elist, _) in highlight.items():
        for u, v in elist:
            highlighted.add(frozenset((u, v)))

    # base (non-highlighted) edges
    base = [(u, v) for u, v in G.edges() if frozenset((u, v)) not in highlighted]
    nx.draw_networkx_edges(G, pos, edgelist=base, edge_color=GREY, width=1.2, ax=ax)
    for color, (elist, width) in highlight.items():
        nx.draw_networkx_edges(G, pos, edgelist=_edge_list(elist),
                               edge_color=color, width=width, ax=ax)

    # nodes
    faces = []
    for n in G.nodes():
        if node_colors and n in node_colors:
            faces.append(node_colors[n])
        elif n == S or n == T:
            faces.append(GREEN)
        else:
            faces.append(NODE_FACE)
    nx.draw_networkx_nodes(G, pos, node_color=faces, edgecolors=NODE_EDGE,
                           linewidths=1.6, node_size=620, ax=ax)
    if special_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(special_nodes),
                               node_color=VIOLET, edgecolors=TEXT,
                               linewidths=2.0, node_size=760, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=12,
                            font_weight='bold', ax=ax)

    if show_weights:
        labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7.5,
                                     font_color=MUTED, ax=ax,
                                     bbox=dict(boxstyle='round,pad=0.1',
                                               fc=BG, ec='none', alpha=0.6))
    ax.set_title(title, color=TEXT, fontsize=11, pad=6)
    ax.set_axis_off()
    ax.margins(0.18)


def path_edges(path):
    return [(a, b) for a, b in zip(path[:-1], path[1:])]


def two_disjoint_paths(sol_edges, s, t):
    H = nx.Graph()
    H.add_edges_from(_edge_list(sol_edges))
    paths = []
    try:
        for p in nx.edge_disjoint_paths(H, s, t):
            paths.append(p)
            if len(paths) == 2:
                break
    except Exception:
        pass
    return paths


def solution_cost(G, sol_edges):
    return sum(G[u][v]['weight'] for u, v in _edge_list(sol_edges))


def draw_solution(ax, G, pos, sol_edges, title):
    """Final-solution panel: 2 disjoint s-t paths colored, attachments green."""
    sol = {frozenset(e) for e in sol_edges}
    paths = two_disjoint_paths(sol_edges, S, T)
    p1 = {frozenset(e) for e in path_edges(paths[0])} if len(paths) > 0 else set()
    p2 = {frozenset(e) for e in path_edges(paths[1])} if len(paths) > 1 else set()
    e_p1 = [tuple(fs) for fs in p1]
    e_p2 = [tuple(fs) for fs in p2]
    e_att = [tuple(fs) for fs in sol - p1 - p2]
    draw_base(ax, G, pos, title, show_weights=False,
              highlight={RED: (e_p1, 4.5), CYAN: (e_p2, 4.5), GREEN: (e_att, 4.5)})


def grid(panels, ncols, outpath, suptitle, legend=None, panel_h=4.0):
    """panels: list of (draw_fn). Render into a grid and save."""
    n = len(panels)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * panel_h))
    fig.patch.set_facecolor(BG)
    axes = axes.ravel() if n > 1 else [axes]
    for i, draw_fn in enumerate(panels):
        draw_fn(axes[i])
    for j in range(n, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(suptitle, color=TEXT, fontsize=15, y=0.99)
    if legend:
        handles = [plt.Line2D([0], [0], color=c, lw=4, label=l) for l, c in legend]
        fig.legend(handles=handles, loc='lower center', ncol=len(legend),
                   frameon=False, fontsize=10, labelcolor=TEXT,
                   bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04 if legend else 0, 1, 0.97))
    fig.savefig(outpath, dpi=150, facecolor=BG)
    print(f"Saved -> {outpath}")
    plt.close(fig)


# ── DPGC step figure ────────────────────────────────────────────────────────────

def make_dpgc_figure(G, pos, results_dir):
    final_edges, info = dpgc_heuristic(G, s=S, t=T)
    paths = info['paths']
    N1 = set(info['N1'])
    e_p1 = path_edges(paths[0])
    e_p2 = path_edges(paths[1])

    # contracted graph H (path nodes -> 'C')
    H = info['contracted_graph']
    posH = {n: pos[n] for n in H.nodes() if n != 'C'}
    cx = sum(pos[n][0] for n in N1) / len(N1)
    cy = sum(pos[n][1] for n in N1) / len(N1)
    posH['C'] = (cx, cy)

    # MST on the contracted graph (no metric closure)
    H_mst = info['contracted_graph']
    mst_edges = [(u, v) for u, v, _ in info['mst_edges']]

    def p0(ax):
        draw_base(ax, G, pos, "1. Original graph")

    def p1(ax):
        draw_base(ax, G, pos, "2. Min-cost 2 edge-disjoint s–t paths",
                  show_weights=False,
                  highlight={RED: (e_p1, 4.5), CYAN: (e_p2, 4.5)},
                  special_nodes=N1)

    def p2(ax):
        draw_base(ax, H, posH, "3. Contract path nodes into super-node C",
                  node_colors={'C': VIOLET}, show_weights=True)

    def p3(ax):
        draw_base(ax, H_mst, posH, "4. MST on contracted graph",
                  node_colors={'C': VIOLET}, show_weights=False,
                  highlight={GREEN: (mst_edges, 4.5)})

    def p4(ax):
        c = solution_cost(G, final_edges)
        draw_solution(ax, G, pos, final_edges,
                      f"5. Lift MST edges back + paths = DPGC solution (cost {c:.2f})")

    grid([p0, p1, p2, p3, p4], ncols=3,
         outpath=os.path.join(results_dir, "dpgc_steps.png"),
         suptitle="DPGC heuristic — step by step",
         legend=[("s-t path 1", RED), ("s-t path 2", CYAN),
                 ("MST / attachment", GREEN), ("contracted node C", VIOLET)])


# ── Matroid step figure ──────────────────────────────────────────────────────────

def representative_one_tree(G, anchor):
    """A sample q-restricted 1-tree: an MST plus the cheapest extra edge at the
    anchor (which closes a cycle through the anchor). For illustration only."""
    T = nx.minimum_spanning_tree(G, weight='weight')
    tree = list(T.edges())
    in_tree = {frozenset(e) for e in tree}
    extras = [(G[anchor][x]['weight'], (anchor, x)) for x in G[anchor]
              if frozenset((anchor, x)) not in in_tree]
    extra = min(extras)[1] if extras else None
    return tree, extra


def make_matroid_figure(G, pos, results_dir):
    sol_edges, info = solve_dpt_matroid(G, s=S, t=T)
    t1, x1 = representative_one_tree(G, S)
    t2, x2 = representative_one_tree(G, T)

    def p_orig(ax):
        draw_base(ax, G, pos, "1. Original graph")

    def p_m1(ax):
        draw_base(ax, G, pos, "2. Matroid M₁  (1-tree, cycle through s)",
                  show_weights=False,
                  highlight={VIOLET: (t1, 4.0), AMBER: ([x1] if x1 else [], 4.0)},
                  node_colors={S: AMBER})

    def p_m2(ax):
        draw_base(ax, G, pos, "3. Matroid M₂  (1-tree, cycle through t)",
                  show_weights=False,
                  highlight={VIOLET: (t2, 4.0), AMBER: ([x2] if x2 else [], 4.0)},
                  node_colors={T: AMBER})

    def p_int(ax):
        c = solution_cost(G, sol_edges)
        draw_solution(ax, G, pos, sol_edges,
                      f"4.  M₁ ∩ M₂  =  optimal DPT (cost {c:.2f})")

    grid([p_orig, p_m1, p_m2, p_int], ncols=4,
         outpath=os.path.join(results_dir, "matroid_steps.png"),
         suptitle="Matroid DPT — build two matroids, then intersect them",
         legend=[("1-tree edges", VIOLET), ("cycle edge (∋ anchor)", AMBER),
                 ("s-t path 1", RED), ("s-t path 2", CYAN)])


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    G, pos = dummy_graph()
    make_dpgc_figure(G.copy(), pos, results_dir)
    make_matroid_figure(G.copy(), pos, results_dir)


if __name__ == "__main__":
    main()
