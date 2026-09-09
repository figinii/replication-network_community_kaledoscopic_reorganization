"""Resolution sweeps of three community-detection algorithms over four networks.

Run from the repository root::

    python -m src.run_sweeps                  # every graph, every method
    python -m src.run_sweeps --dry-run        # print the plan and the cost
    python -m src.run_sweeps -g flights rtpol -m louvain leiden_rb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # figures are only ever saved, never shown

from src.resolution_sweep import ResolutionSweep

#: igraph translations of the graphs seen so far, per process; converting a
#: 21k-node graph on every call would cost more than the Leiden run itself.
_IGRAPH_CACHE: dict[int, tuple[nx.Graph, object, list]] = {}

#: Seconds one CNM run took on a 3330-node, 19079-edge graph, used to warn about
#: the cost before a sweep starts. CNM grows worse than linearly, so the estimate
#: scales with n^2 and is a floor rather than a promise.
CNM_SECONDS_AT = (25.9, 3330)


def _igraph_of(graph: nx.Graph):
  """Translate a graph to igraph, reusing the translation across calls.

  Args:
    graph: the graph to translate.

  Returns:
    Tuple of (igraph graph, list of nodes in igraph vertex order).
  """
  key = id(graph)
  if key not in _IGRAPH_CACHE:
    import igraph as ig

    nodes = list(graph.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    translated = ig.Graph(n=len(nodes),
                          edges=[(index[u], index[v]) for u, v in graph.edges()])
    # The graph is kept alive so its id() cannot be reused by another object.
    _IGRAPH_CACHE[key] = (graph, translated, nodes)
  _, translated, nodes = _IGRAPH_CACHE[key]
  return translated, nodes


def leiden_rb(graph: nx.Graph, resolution: float = 1.0, seed: int | None = None):
  """Partition a graph with Leiden under the RB-configuration objective.

  Args:
    graph: the graph to partition.
    resolution: resolution parameter, matching modularity's gamma.
    seed: random seed.

  Returns:
    List of communities, each a list of nodes.
  """
  import leidenalg

  translated, nodes = _igraph_of(graph)
  partition = leidenalg.find_partition(
    translated,
    leidenalg.RBConfigurationVertexPartition,
    resolution_parameter=resolution,
    seed=seed,
  )
  return [[nodes[i] for i in community] for community in partition]


#: Methods whose result depends on a random seed, so they are replicated.
STOCHASTIC_METHODS = {
  "louvain": nx.community.louvain_communities,
  "leiden_rb": leiden_rb,
}

#: Methods that always return the same partition, so one seed is enough.
DETERMINISTIC_METHODS = {
  "cnm_greedy": nx.community.greedy_modularity_communities,
}

METHODS = STOCHASTIC_METHODS | DETERMINISTIC_METHODS


# --- graph loaders -----------------------------------------------------------

def load_condmat(graph_dir: Path) -> nx.Graph:
  """Load the ca-CondMat co-authorship network.

  Args:
    graph_dir: directory holding the graph files.

  Returns:
    The graph.
  """
  from scipy.io import mmread

  matrix = mmread(graph_dir / "ca-CondMat.mtx")
  return nx.from_scipy_sparse_array(matrix, create_using=nx.Graph())


def load_flights(graph_dir: Path) -> nx.Graph:
  """Load the OpenFlights airport network: routes.dat edges, flights.csv labels.

  Args:
    graph_dir: directory holding the graph files.

  Returns:
    The graph, with "name" and "iata" attributes on the airports that
    flights.csv knows about.
  """
  routes = pd.read_csv(
    graph_dir / "routes.dat", header=None,
    names=["airline", "airline_id", "src", "src_id", "dst", "dst_id",
           "codeshare", "stops", "equipment"])
  routes = routes[(routes["src_id"] != r"\N") & (routes["dst_id"] != r"\N")]
  routes = routes.astype({"src_id": int, "dst_id": int}).query("src_id != dst_id")
  graph = nx.from_pandas_edgelist(routes, "src_id", "dst_id", create_using=nx.Graph())

  airports = pd.read_csv(graph_dir / "flights.csv", header=None, usecols=[0, 1, 4],
                         names=["id", "name", "iata"]).set_index("id")
  known = airports.reindex(list(graph.nodes())).dropna(how="all")
  nx.set_node_attributes(graph, known["name"].to_dict(), "name")
  nx.set_node_attributes(graph, known["iata"].to_dict(), "iata")
  return graph


def load_openflights(graph_dir: Path) -> nx.Graph:
  """Load the Network Repository openflights airport network.

  Args:
    graph_dir: directory holding the graph files.

  Returns:
    The graph.
  """
  return nx.read_edgelist(graph_dir / "inf-openflights.edges", comments="%",
                          nodetype=int, create_using=nx.Graph())


def load_rtpol(graph_dir: Path) -> nx.Graph:
  """Load the rt-pol retweet network.

  Args:
    graph_dir: directory holding the graph files.

  Returns:
    The graph.
  """
  edges = pd.read_csv(graph_dir / "rt-pol.txt", sep=",", comment="%", header=None,
                      names=["src", "dst", "timestamp"], dtype=int)
  return nx.from_pandas_edgelist(edges.query("src != dst"), "src", "dst",
                                 create_using=nx.Graph())


#: Loaders in ascending order of size, so the cheap sweeps report first.
GRAPHS = {
  "openflights": load_openflights,
  "flights": load_flights,
  "rtpol": load_rtpol,
  "condmat": load_condmat,
}


def largest_component(graph: nx.Graph) -> nx.Graph:
  """Restrict a graph to its largest connected component.

  Args:
    graph: the graph.

  Returns:
    A copy of the largest component.
  """
  components = sorted(nx.connected_components(graph), key=len, reverse=True)
  return graph.subgraph(components[0]).copy()


# --- the sweep ---------------------------------------------------------------

def describe(name: str, graph: nx.Graph) -> None:
  """Print the size and component structure of a loaded graph.

  Args:
    name: the graph's label.
    graph: the graph.
  """
  components = nx.number_connected_components(graph)
  largest = max(len(c) for c in nx.connected_components(graph))
  print(f"  {name:12s} {graph.number_of_nodes():>7,} nodes  "
        f"{graph.number_of_edges():>7,} edges  "
        f"{components:>5,} components (largest {largest:,})")


def cnm_estimate(graph: nx.Graph, n_gammas: int, workers: int) -> float:
  """Estimate the wall-clock hours one CNM sweep of a graph will take.

  Args:
    graph: the graph to be swept.
    n_gammas: number of resolutions.
    workers: processes running in parallel.

  Returns:
    Estimated hours, from a measured run scaled quadratically in the node count.
  """
  seconds, nodes = CNM_SECONDS_AT
  per_run = seconds * (graph.number_of_nodes() / nodes) ** 2
  return per_run * n_gammas / max(workers, 1) / 3600


def sweep_graph(
  name: str,
  graph: nx.Graph,
  gammas: np.ndarray,
  seeds: list[int],
  methods: dict[str, object],
  out_dir: Path,
  *,
  workers: int,
  plot: bool,
) -> pd.DataFrame:
  """Sweep one graph with every selected method and save the statistics.

  Stochastic methods are replicated over every seed; deterministic ones are run
  once, since further seeds would only repeat the same partition. Both write to
  the same CSV, which :class:`ResolutionSweep` merges.

  Args:
    name: the graph's label, used in the output filenames.
    graph: the graph to sweep.
    gammas: resolutions to evaluate.
    seeds: random seeds for the stochastic methods.
    methods: label to callable, the methods to run.
    out_dir: directory the CSV and figures are written to.
    workers: processes used for each sweep.
    plot: whether to write the figures as well.

  Returns:
    The merged statistics for this graph.
  """
  stats_path = out_dir / f"nc_gamma_stats_{name}.csv"
  stochastic = {k: v for k, v in methods.items() if k in STOCHASTIC_METHODS}
  deterministic = {k: v for k, v in methods.items() if k in DETERMINISTIC_METHODS}

  sweep = None
  for selected, method_seeds in ((stochastic, seeds), (deterministic, seeds[:1])):
    if not selected:
      continue
    sweep = ResolutionSweep(graph, gammas, method_seeds, selected,
                            max_workers=workers, stats_path=stats_path)
    sweep.run()

  stats = sweep.load(stats_path)
  print(f"  -> {stats_path} ({len(stats):,} rows, "
        f"{stats['method'].nunique()} methods)")

  if plot:
    sweep.plot_n_communities(stats, save=out_dir / f"nc_vs_gamma_{name}.png")
    sweep.plot_n_communities_mod_regularizer(
      stats, save=out_dir / f"nc_bp_vs_gamma_{name}.png")
    print(f"  -> {out_dir / f'nc_vs_gamma_{name}.png'}, "
          f"{out_dir / f'nc_bp_vs_gamma_{name}.png'}")
  return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parse the command line.

  Args:
    argv: arguments to parse; defaults to sys.argv.

  Returns:
    The parsed arguments.
  """
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("-g", "--graphs", nargs="+", choices=list(GRAPHS),
                      default=list(GRAPHS), help="graphs to sweep")
  parser.add_argument("-m", "--methods", nargs="+", choices=list(METHODS),
                      default=list(METHODS), help="algorithms to run")
  parser.add_argument("-n", "--n-gammas", type=int, default=100,
                      help="number of resolutions between 0 and 1")
  parser.add_argument("-s", "--seeds", type=int, default=5,
                      help="seeds per resolution, for the stochastic methods")
  parser.add_argument("-w", "--workers", type=int, default=8,
                      help="worker processes")
  parser.add_argument("--graph-dir", type=Path, default=Path("graphs"),
                      help="directory holding the graph files")
  parser.add_argument("-o", "--out-dir", type=Path, default=Path("."),
                      help="directory for the CSVs and figures")
  parser.add_argument("--lcc", action="store_true",
                      help="restrict every graph to its largest connected component")
  parser.add_argument("--no-plot", action="store_true",
                      help="write the CSVs only")
  parser.add_argument("--dry-run", action="store_true",
                      help="load the graphs, print the plan, and stop")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  """Load the graphs, print the plan, and run the sweeps.

  Args:
    argv: command-line arguments; defaults to sys.argv.

  Returns:
    Process exit status.
  """
  args = parse_args(argv)
  gammas = np.round(np.linspace(0.0, 1.0, args.n_gammas), 10)
  seeds = list(range(args.seeds))
  methods = {name: METHODS[name] for name in args.methods}
  args.out_dir.mkdir(parents=True, exist_ok=True)

  print(f"gammas: {args.n_gammas} points from {gammas[0]} to {gammas[-1]} "
        f"(step {gammas[1] - gammas[0]:.6f})")
  print(f"methods: {', '.join(methods)}")
  print(f"seeds: {args.seeds} for stochastic methods, 1 for deterministic ones")
  print(f"workers: {args.workers}\n")

  print("graphs:")
  loaded = {}
  for name in args.graphs:
    graph = GRAPHS[name](args.graph_dir)
    if args.lcc:
      graph = largest_component(graph)
    loaded[name] = graph
    describe(name, graph)

  n_stochastic = sum(1 for m in methods if m in STOCHASTIC_METHODS)
  n_deterministic = sum(1 for m in methods if m in DETERMINISTIC_METHODS)
  runs = len(loaded) * args.n_gammas * (n_stochastic * args.seeds + n_deterministic)
  print(f"\ntotal runs: {runs:,}")

  if "cnm_greedy" in methods:
    hours = sum(cnm_estimate(g, args.n_gammas, args.workers) for g in loaded.values())
    print(f"WARNING: cnm_greedy is the bottleneck; rough estimate for its share "
          f"alone is {hours:.1f} h of wall clock on {args.workers} workers.")
    print("         Drop it with: --methods louvain leiden_rb")

  if args.dry_run:
    print("\ndry run, stopping here")
    return 0

  for name, graph in loaded.items():
    print(f"\n=== {name} ===")
    sweep_graph(name, graph, gammas, seeds, methods, args.out_dir,
                workers=args.workers, plot=not args.no_plot)
  return 0


if __name__ == "__main__":
  sys.exit(main())
