# Kaleidoscopic reorganization of network communities across different scales

A replication of [*Kaleidoscopic reorganization of network communities across
different scales*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.014312)
(Phys. Rev. E **111**, 014312), together with the code behind
[`presentation.pdf`](presentation.pdf).

The paper asks what modularity maximisation actually returns as the resolution
parameter `γ` is varied. The naive expectation is monotone: raise `γ`, pay more
for the null term, get more and smaller communities. The paper's claim is that
this is not what happens — over some ranges of `γ` the number of communities
`n_c` *drops* as `γ` grows, and the partition reorganises rather than simply
refining. This repository reproduces that behaviour on a synthetic benchmark
with known ground truth and on a real collaboration network, and probes how far
it generalises.

The modularity used throughout is the generalised (RB) form

```text
Q(G; γ) = (1/M) · Σ_g ( L_g − γ · K_g² / 4M )
```

with `L_g` the edges inside community `g`, `K_g` its total degree and `M` the
edge count. `Q` is linear in `γ`, which the code exploits in two places: the
exhaustive landscape (two coefficients per partition cover every resolution) and
the `b_p = Σ_g (K_g / 2M)²` statistic plotted alongside `n_c`.

## Layout

```text
src/
  traag_graph.py          two triangles joined by a bridge node
  partition_landscape.py  every partition of a small graph, scored at every γ
  core_periphery_sbm.py   the core–periphery SBM benchmark (32 planted blocks)
  resolution_sweep.py     parallel (method × γ × seed) sweeps, cached to CSV
graphs/
  ca-CondMat.mtx          arXiv cond-mat co-authorship network
understanding_modularity.ipynb   the modularity surface on a 7-node graph
core_periphery_model.ipynb       building and checking the benchmark
resolution_sweep_sbm.ipynb       the sweep against planted ground truth
resolution_sweep_condmat.ipynb   the sweep on a real network
presentation.pdf                 slides
writeup.typ                      Typst notes (partial)
```

## Setup

Python 3.12, dependencies pinned in `uv.lock`:

```sh
uv sync
uv run jupyter lab
```

`uv sync` installs the dev group (`ipykernel`, `ipywidgets`), which the
notebooks need — the 3D surface and the `γ` slider are Plotly figures.

## The notebooks

Read in this order; each one stands alone but they build on each other.

**1. [`understanding_modularity.ipynb`](understanding_modularity.ipynb)** —
what the resolution parameter does, with the optimiser removed. The Traag graph
(two triangles sharing a bridge node) is small enough to enumerate all 877 set
partitions of its 7 nodes and score every one of them at 1000 resolutions.
Because partitions have no natural order, `PartitionLandscape` lays them on a
line that minimises total Mirkin distance between neighbours (greedy chain, then
swap relaxation), which turns the table into a readable surface: partition on
`x`, `γ` on `y`, `Q` as height. Sliding `γ` moves the ridge from "everything in
one community" through "one community per triangle" and onward. `verify()`
checks an entry of the table against `networkx.community.modularity`.

**2. [`core_periphery_model.ipynb`](core_periphery_model.ipynb)** — the
benchmark network. Two dense cores (400 nodes at `p = 0.7`, 200 at `p = 0.5`,
joined at `p = 0.1`) with 30 peripheral communities of 20 nodes hanging off
them: 10 on the large core, 20 on the small one, none joined to each other.
1200 nodes in **32 planted blocks**. The Bernoulli blocks come from
`nx.stochastic_block_model`; the periphery–core bridges do not, because the
specification fixes their *count* (exactly 20 edges per bridge), so they are
drawn afterwards with an exact edge count. `summary()` and `checks()` verify the
realised sample against the specification, and the notebook shows the network
both as one mark per community and as a 32×32 block matrix.

**3. [`resolution_sweep_sbm.ipynb`](resolution_sweep_sbm.ipynb)** — Louvain over
`γ ∈ [0, 1]` in steps of 0.01, 10 seeds per resolution, on the benchmark above.
Because the truth is known, `n_c` can be read against the 32 blocks it should
recover, and only a band of resolutions gets the count right. This is where the
non-monotonicity shows: `n_c` climbs to ~29 near `γ ≈ 0.15`, falls back to ~14
around `γ ≈ 0.42`, then climbs again to settle at 32. The `n_c` and `b_p` panels
turn at the same two resolutions (marked at 0.39 and 0.71).

**4. [`resolution_sweep_condmat.ipynb`](resolution_sweep_condmat.ipynb)** — the
same sweep on [ca-CondMat](graphs/ca-CondMat.mtx) (21,363 authors, 91,286
co-authorship edges, read from Matrix Market with weights dropped). The same
shape appears on real data: a peak near `γ ≈ 0.15` at `n_c ≈ 65`, a trough near
`γ ≈ 0.4` at `n_c ≈ 25`, then a steady climb.

## Modules

### `ResolutionSweep`

The workhorse. Holds a graph, a resolution grid, a list of seeds and one or more
community-detection callables, and evaluates every (method, γ, seed)
combination in a process pool.

```python
sweep = ResolutionSweep(G, GAMMAS, SEEDS, [nx.community.louvain_communities],
                        stats_path="nc_gamma_stats.csv")
print(sweep.pending())     # what is still missing, before anything runs
stats = sweep.run()        # computes only that, merges, writes the CSV
sweep.plot_n_communities(save="nc_vs_gamma.png")
sweep.plot_n_communities_mod_regularizer()   # n_c and b_p side by side
```

Notes on the design:

- **Resumable.** `run()` reads `stats_path`, computes only the missing
  combinations and rewrites the merged table. Rows from earlier runs outside the
  current grid are kept. `run(reuse=False)` recomputes everything.
- **Any method.** Each callable is invoked as `method(graph, resolution=γ)`,
  with `seed=seed` added only when the signature accepts it — so
  `nx.community.greedy_modularity_communities`, a `leidenalg` wrapper and
  `louvain_communities` can be swept together and plotted on shared axes. Pass a
  mapping to control the labels.
- **Fork-based pool.** The child inherits `__main__`, so methods defined
  directly in a notebook cell stay usable.
- Per row it records `n_communities`, `b_p = Σ_g (K_g / 2M)²` and `mean_size`;
  plots show the mean over seeds with the spread as error bars.

### `PartitionLandscape`

Enumerates every partition of a small graph and scores it across a `γ` grid.
Exposes the modularity table, the Mirkin distance matrix, the similarity
ordering and its quality, per-partition drawings, a CSV dump, and the Plotly
surface and slider figures.

### `CorePeripherySBM`

The benchmark generator. Every structural parameter is a keyword argument
(sizes, densities, number of peripheral communities, bridge count, and
`bridge_mode` — `"uniform"` for exact-count bridges, `"per_node"` to give every
peripheral node one edge to its core). Provides `graph`, `blocks`, `sizes`,
`edge_matrix`, `roles`, `summary()`, `checks()`, `plot_communities()` and
`plot_block_matrix()`.

### `TraagGraph`

The fixed two-triangle graph with a bridge node, plus its drawing positions.

## What the replication found

The paper's effect reproduces, but the presentation adds two caveats that the
sweeps here make visible:

1. **It is not the rule.** Sweeping 100 graphs across 21 categories from
   [networkrepository.com](https://networkrepository.com), only 5 showed the
   reorganisation behaviour. On CondMat and on OpenFlights *flights* it is
   clear; on rt-pol and OpenFlights *routes* `n_c` is essentially monotone.
2. **It is optimiser-dependent.** Louvain and Leiden (RB) agree closely across
   `γ`, while CNM greedy modularity traces a completely different curve — often
   several times higher in `n_c`, with its own non-monotone excursions. The
   effect is a property of the modularity landscape *as explored by a given
   optimiser*, not of the landscape alone.

Only `ca-CondMat` is committed under `graphs/`; the OpenFlights, rt-pol and
networkrepository sweeps shown in the slides were run against locally downloaded
copies and are not part of the repository. `ResolutionSweep` runs them unchanged
once the graph is loaded — nothing in it is specific to a dataset.

## Reference

Paper: <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.014312>
Slides: [`presentation.pdf`](presentation.pdf) — Figini Matteo, Politecnico di Milano
