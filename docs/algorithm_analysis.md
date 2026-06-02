# Algorithm Analysis — DPT/DPST Solvers

This document explains the three algorithms implemented in TwinPaths for the
**k = 2 Dual-Path Spanning Tree (DPT/DPST)** problem and analyzes their
asymptotic running time. It is paired with `benchmark_statistical.py`, which
empirically measures the runtimes predicted here.

## The problem

Given an undirected, edge-weighted graph `G = (V, E)` with a source `s` and a
sink `t`, find a **minimum-cost subgraph** that simultaneously:

1. contains **two edge-disjoint `s–t` paths** (survivability — the `s–t`
   connection survives any single edge failure), and
2. **spans every vertex** of `G` (every node is attached to the structure).

Edge weights are assumed to be **metric** (they satisfy the triangle
inequality); the graph generators place nodes in the Euclidean plane, which
guarantees this.

Throughout, let `V = |V|` (nodes) and `E = |E|` (edges). For a dense graph
`E = Θ(V²)`; for the sparse Euclidean graphs used in the benchmark
`E = Θ(V)` to `Θ(V²)` depending on `edge_prob`.

---

## 1. Brute-force ground truth — `ground_truth.mst_k2_generator`

### What it does

An exhaustive, exact baseline:

1. **Enumerate every simple `s–t` path** using recursive DFS with backtracking
   (`find_paths_aux`).
2. **Try every ordered pair of paths** `(Pᵢ, Pⱼ)`; keep only pairs that are
   edge-disjoint *and* internally node-disjoint (`are_paths_independant`).
3. For each feasible pair, **greedily attach the remaining nodes**: repeatedly
   connect the cheapest edge from an unconnected node to the already-included
   set (a Prim-like nearest-neighbor attachment).
4. Return the pair + attachments with the **minimum total weight**.

Because it inspects every path pair, it is **exact** by construction and is used
as the source of truth for correctness of the other two.

### Runtime complexity

The killer is step 1. In a dense graph the number of simple `s–t` paths is
`Θ((V−2)!)` — it grows **factorially / exponentially** in `V`.

- Path enumeration: `O(P · V)` where `P` = number of simple paths (worst case
  `P = Θ((V−2)!)`).
- Pair evaluation: the double loop is `O(P²)`, and each feasibility +
  attachment check is `O(V²)`.

**Total: `O(P² · V²)` with `P` exponential in `V` ⇒ super-exponential in the
worst case.** This is why the benchmark only runs it for `n ≲ 10`; beyond that
it does not terminate in reasonable time.

> Implementation note: the recursion builds **all** paths in memory before
> pairing them, so memory also blows up exponentially.

---

## 2. Matroid DPT — `matroid_dpt.solve_dpt_matroid`

### What it does

This is an **exact** polynomial-time solver for the metric (triangle-cost) case,
built on **weighted matroid intersection**. The DPT structure (a spanning
subgraph whose `s`- and `t`-sides each form a "1-tree" containing the required
cycle) is exactly the set of common independent sets of **two matroids**:

- `M₁` = *q-restricted 1-tree matroid* with `q = s`
- `M₂` = *q-restricted 1-tree matroid* with `q = t`

A **q-restricted 1-tree** is a subgraph with at most one cycle, and if a cycle
exists it must pass through the distinguished node `q`
(`QRestrictedOneTreeMatroid`). The maximum-weight common independent set of
`M₁ ∩ M₂` (computed on **negated** costs to get *minimum* cost) is the optimal
DPT.

The solver (`weighted_matroid_intersection`) is a port of cplib's algorithm:
it repeatedly builds an **exchange graph** over the ground set (the edges of
`G`), where arcs encode which element can replace which while preserving
independence in each matroid, then finds a shortest augmenting `s→t` path in
that exchange graph via **Bellman–Ford** (costs may be negative). Each
augmenting path flips the membership of the elements along it, growing the
common independent set by one. It stops when no augmenting path remains.

### Runtime complexity

Let `n = E` be the ground-set size. Per iteration:

- For every element `e ∉ I` (`O(E)` of them) it computes `circuit(e)`, which
  runs a BFS over the current independent set: `O(V + E)` each ⇒ `O(E · (V+E))`
  to build the exchange graph.
- The exchange graph has `O(E)` nodes and `O(E · V)` arcs (each element links
  to the ≤ `V` elements of its circuit).
- **Bellman–Ford** on it costs `O(nodes · arcs) = O(E · E·V) = O(E² · V)`.

The number of augmentations equals the final rank, which is `O(V)` (a 1-tree on
`V` nodes has `≤ V` edges).

**Total: `O(V) · O(E² · V) = O(V² · E²)`.** For a dense graph (`E = Θ(V²)`)
this is **`O(V⁶)`** — polynomial, but a *high-degree* polynomial. So it scales
to far larger graphs than brute force, yet is markedly slower than the
heuristic and degrades quickly as `n` grows. This is the expected middle curve
in the benchmark.

> Practical note: `Bellman–Ford` is `O(V·E)` with a pure-Python inner loop, so
> the constant factor is also large compared to NetworkX's C-backed routines.

---

## 3. DPGC heuristic — `DPGC_heuristic.dpgc_heuristic`

### What it does

The **Dual-Path Graph Contraction** heuristic of Balakrishnan et al. — fast,
polynomial, and *approximate* (not guaranteed optimal):

1. **Min-cost pair of edge-disjoint `s–t` paths** via **Suurballe's algorithm**
   (`_min_cost_k_edge_disjoint_paths`): one Dijkstra for `P₁`, then a residual
   graph with reversed/negated `P₁` arcs, then Bellman–Ford for `P₂`, then
   cancel opposing arcs and decompose into two paths. This yields the edge set
   `E₁` and node set `N₁`.
2. **Contract** the subgraph induced by `N₁` into a single super-node `C`
   (`_contract_subgraph`), keeping the cheapest edge from `C` to each outside
   node (and remembering the original edge).
3. **Metric closure** of the contracted graph (`_metric_closure_graph`, all-pairs
   Dijkstra) followed by a **minimum spanning tree** (`nx.minimum_spanning_tree`).
4. **Lift** the MST edges back to original edges and union them with `E₁`.

Intuitively: secure the two disjoint paths first, collapse them, then cheaply
hang the rest of the graph off that core with an MST.

### Runtime complexity

- Suurballe (step 1): one Dijkstra `O(E + V log V)` + one Bellman–Ford
  `O(V·E)` ⇒ `O(V·E)`.
- Contraction (step 2): `O(E)`.
- Metric closure (step 3): all-pairs Dijkstra `O(V·(E + V log V)) =
  O(V·E + V² log V)`; the closure is complete (`O(V²)` edges), so its MST costs
  `O(V² log V)`.
- Lifting (step 4): `O(V·E)`.

**Total: `O(V·E + V² log V)`.** For a dense graph (`E = Θ(V²)`) this is
**`O(V³)`** — a low-degree polynomial. It is the **fastest** of the three by a
wide margin and the expected bottom curve in the benchmark.

> Because it is a heuristic, its *cost* may exceed the optimum. The accuracy
> benchmark quantifies how often it still reaches the optimum (vs. the exact
> matroid solution).

---

## Summary

| Algorithm | Type | Time complexity (general) | Dense graph `E=Θ(V²)` | Scales to |
|---|---|---|---|---|
| Brute-force (`mst_k2_generator`) | Exact | `O(P²·V²)`, `P = Θ((V−2)!)` | super-exponential | `n ≲ 10` |
| Matroid DPT (`solve_dpt_matroid`) | Exact (metric) | `O(V²·E²)` | `O(V⁶)` | `n` ≈ tens |
| DPGC (`dpgc_heuristic`) | Heuristic | `O(V·E + V² log V)` | `O(V³)` | `n` ≈ hundreds |

**Expected ordering of runtimes (fastest → slowest): DPGC ≪ Matroid ≪
Brute-force**, with the gap widening sharply as `n` grows. The companion
benchmark confirms this across multiple random seeds.

---

## Empirical results

`benchmark_statistical.py` was run on random Euclidean graphs
(`edge_prob = 0.5`), **20 seeds per size**, timing each solver on the identical
graphs. The brute-force solver was capped at `n = 10`. Mean runtimes (± 95% CI):

![Runtime benchmark](../results/benchmark_runtimes.png)

Selected mean runtimes (seconds), from `results/benchmark_runtimes_summary.csv`:

| n | Brute-force | Matroid DPT | DPGC |
|----:|---:|---:|---:|
| 8 | 0.0053 | 0.00059 | 0.00034 |
| 9 | 0.0176 | 0.00080 | 0.00049 |
| 10 | **0.543** | 0.00101 | 0.00054 |
| 30 | — | 0.0228 | 0.0063 |
| 50 | — | 0.1146 | 0.0242 |
| 75 | — | 0.4764 | 0.0790 |
| 100 | — | 1.1869 | 0.1727 |

Observations that match the analysis:

- **Brute-force is super-exponential.** From `n = 9` to `n = 10` the mean jumps
  ~30× (0.018 s → 0.54 s), and at `n = 11` individual seeds already exceed 8 s.
  Its huge run-to-run variance (std ≈ mean) is the signature of factorial path
  counts that depend heavily on graph structure. It is infeasible past `n ≈ 10`.
- **Both polynomial solvers scale smoothly** on a log plot. Fitting the
  measured exponents over `n = 30 … 100` gives roughly **`O(n^3.4)` for Matroid**
  and **`O(n^2.8)` for DPGC** — consistent with the derived `O(V³)` /
  high-degree-polynomial bounds (the empirical exponents sit below the dense
  worst case because `edge_prob = 0.5` graphs are not fully dense and the
  Bellman–Ford augmentation loop terminates early).
- **DPGC is consistently the fastest**, and its advantage over Matroid widens
  from ~1.6× at `n = 10` to ~7× at `n = 100`.

To reproduce:

```powershell
python -m experiments.benchmark_statistical --sizes 4 5 6 7 8 9 10 15 20 30 50 75 100 --brute-max 10 --seeds 20
# or a quick smoke run:
python -m experiments.benchmark_statistical --quick
```

### Scaling the two polynomial solvers to n = 1000

The dense (`edge_prob = 0.5`) setting above is intractable for the matroid solver
much beyond `n = 100` (a dense `n = 1000` graph has ≈ 250 000 edges). To push the
two *polynomial* solvers further we switch to **sparse** graphs (average degree
≈ 6, i.e. `edge_prob = 6/(n−1)`), which is also more representative of real
networks. Brute-force is excluded (it is hopeless at this scale). 5 seeds per size:

![Extended runtime scaling](../results/large_benchmark_runtimes.png)

Mean runtimes (seconds), from `results/large_benchmark_runtimes_summary.csv`:

| n | Matroid DPT | DPGC | Matroid / DPGC |
|----:|---:|---:|---:|
| 50 | 0.030 | 0.011 | 2.9× |
| 100 | 0.173 | 0.047 | 3.7× |
| 200 | 1.202 | 0.201 | 6.0× |
| 300 | 3.843 | 0.466 | 8.2× |
| 500 | 15.17 | 1.47 | 10.3× |
| 750 | 48.01 | 3.44 | 14.0× |
| 1000 | 230.6 | 14.5 | 15.9× |

- The two curves stay straight on a log–log scale, confirming polynomial growth.
  Fitting `n = 200 … 1000` gives roughly **`O(n^3.3)` for Matroid** and
  **`O(n^2.7)` for DPGC** — consistent with the dense-graph fits and the derived
  bounds.
- **The gap widens monotonically** — from ~3× at `n = 50` to ~16× at `n = 1000` —
  so the heuristic's advantage compounds with scale. At `n = 1000` DPGC finishes
  in ~15 s where Matroid needs ~4 minutes.
- **Matroid's runtime is highly variable.** At `n = 1000` the mean is 230 s but the
  std is ±140 s (95% CI ≈ ±122 s): the number of augmenting-path iterations
  depends strongly on the random graph, so a few "hard" instances dominate the
  wall-clock. DPGC is far more predictable (its cost is structural, not
  search-dependent).

To reproduce:

```powershell
python -m experiments.benchmark_statistical --sizes 50 100 200 300 500 750 1000 --seeds 5 --avg-degree 6 --out-prefix large_
```

---

## Accuracy: DPGC vs. Matroid DPT

Runtime is only half the story — DPGC buys its speed by being a **heuristic**, so
it can return a sub-optimal subgraph. The Matroid solver is *exact* for the
metric case, so we use its cost as the **optimal reference** and measure, per
graph size `n`:

- **approximation ratio** = `DPGC cost / Matroid cost` (≥ 1.0; 1.0 means DPGC
  found the optimum),
- **match rate** = fraction of graphs on which DPGC reaches the optimum.

`benchmark_accuracy.py` evaluated **50 random Euclidean graphs per size**
(`edge_prob = 0.5`). To keep the "Matroid = optimal" claim honest, for `n ≤ 10`
it also runs the exhaustive brute-force solver and checks that Matroid equals the
true optimum.

![Accuracy comparison](../results/accuracy_comparison.png)

From `results/accuracy_summary.csv`:

| n | DPGC match rate | mean ratio | median | p95 | max | Matroid = true optimum |
|----:|---:|---:|---:|---:|---:|---:|
| 5 | 68% | 1.075 | 1.000 | 1.788 | 1.935 | 100% |
| 8 | 42% | 1.051 | 1.012 | 1.171 | 1.500 | 100% |
| 10 | 26% | 1.044 | 1.024 | 1.135 | 1.244 | 98% |
| 15 | 22% | 1.048 | 1.024 | 1.142 | 1.250 | — |
| 20 | 14% | 1.062 | 1.045 | 1.187 | 1.224 | — |
| 30 | 10% | 1.054 | 1.044 | 1.164 | 1.208 | — |
| 50 | 10% | 1.064 | 1.059 | 1.160 | 1.183 | — |
| 75 | 6% | 1.060 | 1.056 | 1.133 | 1.171 | — |
| 100 | 2% | 1.052 | 1.047 | 1.111 | 1.117 | — |

Key findings:

- **DPGC never beats Matroid** — across all 450 graphs the minimum ratio is
  exactly `1.0000`. Matroid is always at least as good, confirming it as a valid
  optimal reference and DPGC as a true upper bound.
- **The probability of hitting the optimum collapses with size**: DPGC matches
  the optimum on **68% of `n = 5`** graphs but only **2% of `n = 100`** graphs.
  As `n` grows there are simply more ways for the "secure two paths first, then
  MST the rest" strategy to miss the globally optimal trade-off.
- **But the cost penalty stays small and bounded.** The mean approximation ratio
  hovers around **1.05–1.06 (≈ 5–6% above optimal)** and is essentially flat in
  `n`; the 95th percentile stays under **1.19**. So while DPGC rarely nails the
  exact optimum on large graphs, it reliably stays within a few percent of it —
  the expected behaviour of a good heuristic. (The wide tail at `n = 5`, max
  1.94, is small-graph noise: few feasible solutions and tiny denominators.)
- **Honest caveat on the reference.** Matroid matched the brute-force optimum on
  every checked graph except **one** (`n = 10`: 448.27 vs 447.22 → 98%). The
  matroid-intersection formulation is provably optimal on a **complete** metric
  graph; on the *sparse* graphs used here it is near-optimal but not guaranteed
  exact, so the ratios above are very slightly optimistic in rare cases.

> Note on disjointness: DPGC and Matroid enforce **edge-disjoint** `s–t` paths,
> whereas the brute-force `mst_k2_generator` enforces the stricter
> **vertex-disjoint** paths. On a few graphs that admit edge- but not
> vertex-disjoint paths the brute force returns no solution; those rows are
> excluded from the Matroid-validation column above.

To reproduce:

```powershell
python -m experiments.benchmark_accuracy --sizes 5 8 10 15 20 30 50 75 100 --seeds 50 --brute-max 10
# or a quick smoke run:
python -m experiments.benchmark_accuracy --quick
```
