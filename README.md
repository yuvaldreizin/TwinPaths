# TwinPaths

Algorithms for the k=2 Survivable Network Design problem (DPT/DPST), including:
- **DPGC heuristic** (`twinpaths/solvers/DPGC_heuristic.py`) with a runnable example.
- **Matroid-intersection solver** for the triangular-cost DPT case (`twinpaths/solvers/matroid_dpt.py`) with optional metric-closure enforcement/checking.
- **Brute-force exact baseline** (`twinpaths/solvers/ground_truth.py`).
- Visualization helpers (Matplotlib, PyVis) and Manim animation utilities.

## Project layout

```
twinpaths/            importable library package
  solvers/            DPGC_heuristic.py, matroid_dpt.py, ground_truth.py
  graphs/             generate_graphs.py (generators), graph_sources.py (.pkl loaders)
  viz/                visualize.py, visualize_pyvis.py, dpgc_full_animation.py
experiments/          runnable drivers (run as `python -m experiments.<name>`)
  benchmark_statistical.py   runtime benchmark (mean +/- 95% CI over seeds)
  benchmark_accuracy.py      DPGC-vs-Matroid accuracy benchmark
  batch_run_algorithms.py    run all 3 solvers on graphs -> CSV
  compare_algorithms.py      CLI over on-disk .pkl graphs
  create_presentation.py     build the slide deck
data/                 example_graphs/ (.pkl), test_graphs_txt_files/ (.txt)
docs/                 algorithm_analysis.md  (complexity + empirical results)
articles/             source papers      results/ (gitignored outputs)
```

## Setup
Requires Python 3.10+.

### Quick start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .          # makes the `twinpaths` package importable
```
Run a solver's built-in example:
```powershell
python -m twinpaths.solvers.DPGC_heuristic
```
Run the benchmarks (from the repository root):
```powershell
python -m experiments.benchmark_statistical --quick    # runtime (all 3 solvers, dense)
python -m experiments.benchmark_accuracy --quick        # accuracy (DPGC vs Matroid)
# extended scaling of the 2 polynomial solvers on sparse graphs, up to n=1000:
python -m experiments.benchmark_statistical --sizes 50 100 200 300 500 750 1000 --seeds 5 --avg-degree 6 --out-prefix large_
```
See `docs/algorithm_analysis.md` for the complexity analysis and all plots
(`results/benchmark_runtimes.png`, `results/large_benchmark_runtimes.png`,
`results/accuracy_comparison.png`).

### One-step venv bootstrap (PowerShell)
A helper script creates/uses `.venv`, installs dependencies, and keeps output concise:
```powershell
.\scripts\setup_venv.ps1
```

## Key components
- `twinpaths/solvers/DPGC_heuristic.py`: Balakrishnan et al. dual-path contraction heuristic (min-cost two edge-disjoint s–t paths → contraction → metric closure → MST → lift back). Example graph included under `__main__`.
- `twinpaths/solvers/matroid_dpt.py`: Pure-Python weighted matroid intersection solver specialized to DPT with triangle inequality.
  - API: `solve_dpt_matroid(G, s=1, t=2, enforce_metric_closure=False, check_metric=False)`.
  - `check_metric=True` will verify triangle inequality; `enforce_metric_closure=True` will replace weights by shortest-path distances before solving (off by default).
- `twinpaths/viz/visualize.py`, `twinpaths/viz/visualize_pyvis.py`: Static and interactive graph visualizations; PyVis export writes HTML snapshots.
- `docs/algorithm_analysis.md`: explanation + runtime-complexity analysis of all three solvers, plus the empirical runtime and accuracy results.


## Matroid DPT usage (triangle-cost case)
```python
import networkx as nx
from twinpaths import solve_dpt_matroid

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
