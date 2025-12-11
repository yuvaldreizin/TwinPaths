# TwinPaths

Algorithms for the k=2 Survivable Network Design problem (DPT/DPST), including:
- **DPGC heuristic** implementation (`DPGC_heuristic.py`) with a runnable example.
- **Matroid-intersection solver** for the triangular-cost DPT case (`matroid_dpt.py`) with optional metric-closure enforcement/checking.
- Visualization helpers (Matplotlib, PyVis) and Manim animation utilities.

## Setup
Requires Python 3.10+.

### Quick start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Run the DPGC example:
```powershell
python DPGC_heuristic.py
```

### One-step venv bootstrap (PowerShell)
A helper script creates/uses `.venv`, installs dependencies, and keeps output concise:
```powershell
.\scripts\setup_venv.ps1
```

## Key components
- `DPGC_heuristic.py`: Balakrishnan et al. dual-path contraction heuristic (min-cost two edge-disjoint s–t paths → contraction → metric closure → MST → lift back). Example graph included under `__main__`.
- `matroid_dpt.py`: Pure-Python weighted matroid intersection solver specialized to DPT with triangle inequality.
  - API: `solve_dpt_matroid(G, s=1, t=2, enforce_metric_closure=False, check_metric=False)`.
  - `check_metric=True` will verify triangle inequality; `enforce_metric_closure=True` will replace weights by shortest-path distances before solving (off by default).
- `utils/visualize.py`, `utils/visualize_pyvis.py`: Static and interactive graph visualizations; PyVis export writes HTML snapshots.


## Matroid DPT usage (triangle-cost case)
```python
import networkx as nx
from matroid_dpt import solve_dpt_matroid

G = nx.Graph()
G.add_edge(1, 2, weight=1)
G.add_edge(2, 3, weight=1)
G.add_edge(1, 3, weight=2)

solution, info = solve_dpt_matroid(G, s=1, t=3, check_metric=True)
print("Edges in optimal DPT:", solution)
```

## Notes
- Triangle inequality: inputs are assumed metric; use `check_metric=True` to assert, or `enforce_metric_closure=True` to metricize before solving.
- Visuals: set `visualize_dual_paths` / `visualize_final` in `dpgc_heuristic` or use the PyVis exporter for interactive HTML.
