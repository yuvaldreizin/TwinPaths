"""TwinPaths — algorithms for the k=2 Survivable Network Design (DPT/DPST) problem.

Subpackages:
    solvers  — the three solvers (DPGC heuristic, matroid intersection, brute force)
    graphs   — graph generators and on-disk graph loaders
    viz      — Matplotlib / PyVis / Manim visualization helpers
"""

from twinpaths.solvers.DPGC_heuristic import dpgc_heuristic
from twinpaths.solvers.matroid_dpt import solve_dpt_matroid
from twinpaths.solvers.ground_truth import mst_k2_generator

__all__ = ["dpgc_heuristic", "solve_dpt_matroid", "mst_k2_generator"]
