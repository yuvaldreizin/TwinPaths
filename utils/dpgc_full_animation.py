"""
Animated explanation for both the DPGC heuristic and the Matroid DPT solution.

Improved over the previous demo:
- Uses a slightly richer 8-node example with realistic weights.
- Colors dual paths separately (red/blue), MST/recovered edges (green),
  and non-used edges (light gray).
- Adds textual legends with s/t terminals and total cost.
- Provides two separate scenes: DPGC heuristic and Matroid DPT solution.

Render examples (do not run here; Manim not installed in this environment):
    manim -pqh utils/dpgc_full_animation.py DPGCScene
    manim -pqh utils/dpgc_full_animation.py MatroidDPTScene
"""

from manim import *
import networkx as nx


# -----------------------------------------------------------------------------
# Shared helpers and data
# -----------------------------------------------------------------------------

def layout_positions(G: nx.Graph):
    """Use spring layout for reproducible node positions, scaled for Manim."""
    pos = nx.spring_layout(G, seed=7)
    return {k: np.array([v[0] * 6, v[1] * 4, 0]) for k, v in pos.items()}


def manim_edge(u, v, positions, color=GRAY, weight=None, width=4):
    """Create a Manim line + weight text for edge (u,v)."""
    line = Line(positions[u], positions[v], color=color, stroke_width=width)
    if weight is not None:
        mid = (positions[u] + positions[v]) / 2
        label = Text(str(weight), font_size=20).move_to(mid + DOWN * 0.2)
    else:
        label = VGroup()
    return line, label


def build_example_graph():
    """
    Example graph (s=1, t=3):
      - Two edge-disjoint s-t paths:
          P1: 1-2-3 (red)
          P2: 1-4-5-3 (blue)
      - Extra nodes (6,7,8) connect via 5-8, 8-7, 7-6 (for MST completion)
    """
    edges = [
        (1, 2, 1),
        (2, 3, 1),
        (1, 4, 2),
        (4, 5, 2),
        (5, 3, 2),
        (2, 6, 2),
        (6, 7, 1),
        (7, 3, 3),
        (5, 8, 1),
        (8, 7, 1),
        (4, 6, 3),
    ]
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G


def legend_box(lines: list[str], position=UR):
    box = VGroup(*[Text(line, font_size=20) for line in lines]).arrange(DOWN, aligned_edge=LEFT)
    rect = SurroundingRectangle(box, color=WHITE, buff=0.3)
    legend = VGroup(rect, box).to_corner(position)
    return legend


# -----------------------------------------------------------------------------
# DPGC Scene
# -----------------------------------------------------------------------------

class DPGCScene(Scene):
    def construct(self):
        G = build_example_graph()
        pos = layout_positions(G)
        s, t = 1, 3

        # Known dual paths and recovered MST edges for this small example
        dual_path1 = [(1, 2), (2, 3)]
        dual_path2 = [(1, 4), (4, 5), (5, 3)]
        recovered_edges = [(5, 8), (8, 7), (7, 6)]  # from metric-closure MST

        # Draw nodes
        nodes = VGroup()
        for n, p in pos.items():
            color = GREEN if n in (s, t) else WHITE
            circ = Circle(radius=0.25, color=color).move_to(p)
            label = Text(str(n), font_size=26).move_to(p)
            nodes.add(circ, label)
        self.play(FadeIn(nodes))
        self.wait(0.3)

        # Draw all edges
        edge_mobs = []
        for u, v, w in G.edges.data("weight"):
            line, wlabel = manim_edge(u, v, pos, color=LIGHT_GRAY, weight=w)
            edge_mobs.append((u, v, line, wlabel))
            self.play(Create(line), FadeIn(wlabel), run_time=0.1)
        self.wait(0.4)

        title = Text("DPGC Heuristic", font_size=40).to_edge(UP)
        self.play(FadeIn(title))
        subtitle = Text("Step 1: Two min-cost edge-disjoint s–t paths", font_size=28).to_edge(DOWN)
        self.play(FadeIn(subtitle))

        # Highlight path1 (red) and path2 (blue)
        for u, v in dual_path1:
            for (a, b, line, _) in edge_mobs:
                if {a, b} == {u, v}:
                    self.play(line.animate.set_color(RED).set_stroke(width=6), run_time=0.2)
        for u, v in dual_path2:
            for (a, b, line, _) in edge_mobs:
                if {a, b} == {u, v}:
                    self.play(line.animate.set_color(BLUE).set_stroke(width=6), run_time=0.2)
        self.wait(0.6)

        # Contract step (conceptual)
        step2 = Text("Step 2: Contract dual-path nodes into C", font_size=28).to_edge(DOWN)
        self.play(ReplacementTransform(subtitle, step2))
        N1 = {1, 2, 3, 4, 5}
        C_pos = ORIGIN
        C_node = VGroup(Circle(radius=0.35, color=YELLOW), Text("C", font_size=26)).move_to(C_pos)
        # Fade out N1 nodes, replace with C
        self.play(*[FadeOut(nodes[i * 2 : i * 2 + 2]) for i, n in enumerate(G.nodes()) if n in N1], run_time=0.6)
        self.play(FadeIn(C_node))
        self.wait(0.4)

        # Metric closure + MST (shown abstractly)
        step3 = Text("Step 3: Metric closure + MST", font_size=28).to_edge(DOWN)
        self.play(ReplacementTransform(step2, step3))
        self.wait(0.4)

        # Bring original nodes back and highlight recovered edges
        self.play(FadeOut(C_node), run_time=0.4)
        self.play(FadeIn(nodes), run_time=0.6)
        step4 = Text("Step 4: Recover MST edges into original graph", font_size=28).to_edge(DOWN)
        self.play(ReplacementTransform(step3, step4))
        for u, v in recovered_edges:
            for (a, b, line, _) in edge_mobs:
                if {a, b} == {u, v}:
                    self.play(line.animate.set_color(GREEN).set_stroke(width=5), run_time=0.2)
        self.wait(0.5)

        # Final solution: union of dual paths + recovered edges
        final = Text("Final DPGC solution", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(step4, final))
        self.wait(0.3)

        total_cost = sum(G[u][v]["weight"] for (u, v, line, _) in edge_mobs if line.get_color() != LIGHT_GRAY)
        legend = legend_box(
            [
                "Red: Path 1 (s–t)",
                "Blue: Path 2 (s–t)",
                "Green: MST / completion edges",
                "Gray: Unused edges",
                f"Terminals: {s}, {t} (green nodes)",
                f"Total cost (colored edges): {total_cost}",
            ],
            position=UR,
        )
        self.play(FadeIn(legend))
        self.wait(2)


# -----------------------------------------------------------------------------
# Matroid DPT Scene
# -----------------------------------------------------------------------------

class MatroidDPTScene(Scene):
    def construct(self):
        G = build_example_graph()
        pos = layout_positions(G)
        s, t = 1, 3

        # Matroid solution: same final edge set as heuristic in this example
        path1 = [(1, 2), (2, 3)]
        path2 = [(1, 4), (4, 5), (5, 3)]
        tree_edges = [(5, 8), (8, 7), (7, 6)]  # completion edges

        # Draw nodes
        nodes = VGroup()
        for n, p in pos.items():
            color = GREEN if n in (s, t) else WHITE
            circ = Circle(radius=0.25, color=color).move_to(p)
            label = Text(str(n), font_size=26).move_to(p)
            nodes.add(circ, label)
        self.play(FadeIn(nodes))
        self.wait(0.3)

        # Draw all edges
        edge_mobs = []
        for u, v, w in G.edges.data("weight"):
            line, wlabel = manim_edge(u, v, pos, color=LIGHT_GRAY, weight=w)
            edge_mobs.append((u, v, line, wlabel))
            self.play(Create(line), FadeIn(wlabel), run_time=0.1)
        self.wait(0.4)

        title = Text("Matroid DPT (triangle-cost case)", font_size=38).to_edge(UP)
        self.play(FadeIn(title))
        subtitle = Text("Intersection of two q-restricted 1-tree matroids", font_size=26).next_to(title, DOWN)
        self.play(FadeIn(subtitle))
        self.wait(0.5)

        # Highlight path1 (red) and path2 (blue)
        for u, v in path1:
            for (a, b, line, _) in edge_mobs:
                if {a, b} == {u, v}:
                    self.play(line.animate.set_color(RED).set_stroke(width=6), run_time=0.2)
        for u, v in path2:
            for (a, b, line, _) in edge_mobs:
                if {a, b} == {u, v}:
                    self.play(line.animate.set_color(BLUE).set_stroke(width=6), run_time=0.2)

        # Highlight remaining tree edges (gray->green)
        for u, v in tree_edges:
            for (a, b, line, _) in edge_mobs:
                if {a, b} == {u, v}:
                    self.play(line.animate.set_color(GREEN).set_stroke(width=5), run_time=0.2)
        self.wait(0.6)

        total_cost = sum(G[u][v]["weight"] for (u, v, line, _) in edge_mobs if line.get_color() != LIGHT_GRAY)
        legend = legend_box(
            [
                "Red: Path 1 (matroid basis through s)",
                "Blue: Path 2 (matroid basis through t)",
                "Green: Remaining tree edges",
                f"Terminals: {s}, {t} (green nodes)",
                f"Total cost (colored edges): {total_cost}",
                "Computed via weighted matroid intersection",
            ],
            position=UR,
        )
        self.play(FadeIn(legend))
        self.wait(2)

