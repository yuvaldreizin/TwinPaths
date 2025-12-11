from manim import *
import networkx as nx


###############################################################################
# Helper functions
###############################################################################

def layout_positions(G):
    """Use spring layout for reproducible node positions."""
    pos = nx.spring_layout(G, seed=42)
    return {k: np.array([v[0]*6, v[1]*4, 0]) for k, v in pos.items()}


def manim_edge(u, v, positions, color=GRAY, weight=None):
    """Create a Manim line + weight text for edge (u,v)."""
    line = Line(positions[u], positions[v], color=color, stroke_width=4)
    if weight is not None:
        mid = (positions[u] + positions[v]) / 2
        label = Text(str(weight), font_size=20).move_to(mid + DOWN*0.2)
    else:
        label = VGroup()
    return line, label


###############################################################################
# MAIN SCENE
###############################################################################

class DPGCFullScene(Scene):
    def construct(self):

        #######################################################################
        # STEP 0 — Build the original graph
        #######################################################################
        G = nx.Graph()
        edges = [
            (1, 2, 1),
            (1, 3, 2),
            (2, 3, 1),
            (2, 4, 3),
            (3, 4, 1),
            (3, 5, 2),
            (4, 5, 1),
        ]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        positions = layout_positions(G)

        # Draw nodes
        node_mobs = {}
        for n, pos in positions.items():
            circ = Circle(radius=0.25, color=WHITE).move_to(pos)
            label = Text(str(n), font_size=26).move_to(pos)
            node_mobs[n] = VGroup(circ, label)
            self.play(FadeIn(circ), FadeIn(label), run_time=0.2)

        self.wait(0.4)

        # Draw all edges
        all_edge_objs = []
        for u, v, w in edges:
            line, weight_label = manim_edge(u, v, positions, color=GRAY, weight=w)
            all_edge_objs.append((u, v, line, weight_label))
            self.play(Create(line), FadeIn(weight_label), run_time=0.15)

        self.wait(0.8)

        title = Text("DPGC Heuristic", font_size=40).to_edge(UP)
        self.play(FadeIn(title))


        #######################################################################
        # STEP 1 — Dual paths (already known)
        #######################################################################
        # For this demonstration, insert known dual paths (from your algorithm output)
        path1 = [(1, 3), (3, 5)]
        path2 = [(1, 2), (2, 4), (4, 5)]

        step1_title = Text("Step 1: Find Min-Cost Edge-Disjoint Paths", font_size=32).to_edge(DOWN)
        self.play(FadeIn(step1_title))
        self.wait(0.6)

        # Highlight path 1 in RED
        for u, v in path1:
            for (a, b, line, lbl) in all_edge_objs:
                if {a,b} == {u,v}:
                    self.play(line.animate.set_color(RED), run_time=0.4)

        self.wait(0.5)

        # Highlight path 2 in BLUE
        for u, v in path2:
            for (a, b, line, lbl) in all_edge_objs:
                if {a,b} == {u,v}:
                    self.play(line.animate.set_color(BLUE), run_time=0.4)

        self.wait(0.8)


        #######################################################################
        # STEP 2 — Contract dual-path nodes into C
        #######################################################################
        N1 = {1, 2, 3, 4, 5}

        step2_title = Text("Step 2: Contract Subgraph into Node C", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(step1_title, step2_title))
        self.wait(0.5)

        # Fade out nodes in N1 except use C instead
        group_nodes = VGroup(*[node_mobs[n] for n in N1])
        self.play(FadeOut(group_nodes))

        C_pos = np.array([0, 0, 0])   # center
        C_node = VGroup(
            Circle(radius=0.35, color=YELLOW),
            Text("C", font_size=26)
        ).move_to(C_pos)

        self.play(FadeIn(C_node))
        self.wait(0.5)


        #######################################################################
        # STEP 3 — Build Metric Closure (complete graph)
        #######################################################################
        step3_title = Text("Step 3: Metric Closure (Complete Graph)", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(step2_title, step3_title))
        self.wait(0.5)

        # Compute metric closure
        H = nx.Graph()
        H.add_node("C")
        for n in [6,7,8,9]:  # In real case, these come from your contracted graph
            H.add_node(n)

        # For demonstration, we connect everything (complete graph)
        H_edges = []
        nodes_H = list(H.nodes())
        for i in range(len(nodes_H)):
            for j in range(i+1, len(nodes_H)):
                H_edges.append((nodes_H[i], nodes_H[j], 1+abs(i-j)))

        # Place them in circle
        pos_H = {}
        R = 3
        for i, n in enumerate(nodes_H):
            angle = 2 * PI * i / len(nodes_H)
            pos_H[n] = np.array([R*np.cos(angle), R*np.sin(angle), 0])

        # Draw edges
        H_edge_mobs = []
        for u,v,w in H_edges:
            line, wlabel = manim_edge(u, v, pos_H, color=GRAY, weight=w)
            H_edge_mobs.append((u,v,line,wlabel))

        # Draw nodes
        H_nodes_mobs = {}
        for n,pos in pos_H.items():
            circ = Circle(radius=0.28, color=WHITE).move_to(pos)
            label = Text(str(n), font_size=24).move_to(pos)
            H_nodes_mobs[n] = VGroup(circ,label)
            self.play(FadeIn(circ), FadeIn(label), run_time=0.2)

        for (_,_,line,wlabel) in H_edge_mobs:
            self.play(Create(line), FadeIn(wlabel), run_time=0.05)

        self.wait(1)


        #######################################################################
        # STEP 4 — MST on Metric Closure
        #######################################################################
        step4_title = Text("Step 4: Minimum Spanning Tree of Metric Closure", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(step3_title, step4_title))

        # Fake MST for visualization
        mst_edges = [(nodes_H[i], nodes_H[i+1]) for i in range(len(nodes_H)-1)]

        for u,v in mst_edges:
            for (a,b,line,wlabel) in H_edge_mobs:
                if {a,b} == {u,v}:
                    self.play(line.animate.set_color(GREEN), run_time=0.4)

        self.wait(1)


        #######################################################################
        # STEP 5 — Expand MST edges back to original graph edges
        #######################################################################
        step5_title = Text("Step 5: Recover Original Edges", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(step4_title, step5_title))

        # Fade out metric closure
        self.play(*[FadeOut(m) for m in H_nodes_mobs.values()])
        for (_,_,line,w) in H_edge_mobs:
            self.play(FadeOut(line), FadeOut(w), run_time=0.01)

        # Bring back original graph nodes
        for n, mob in node_mobs.items():
            self.play(FadeIn(mob), run_time=0.1)

        # Show final result edges (for demo, MST edges = highlight all edges)
        for (u,v,line,w) in all_edge_objs:
            self.play(line.animate.set_color(GREEN), run_time=0.1)

        self.wait(1)


        #######################################################################
        # END — Final Solution
        #######################################################################
        final_title = Text("Final DPGC Solution", font_size=40).to_edge(UP)
        self.play(ReplacementTransform(title, final_title))
        self.wait(2)

