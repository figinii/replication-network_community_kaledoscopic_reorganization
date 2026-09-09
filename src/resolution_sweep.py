"""Resolution sweeps of community-detection methods over a fixed graph."""

from __future__ import annotations

import inspect
import itertools
import multiprocessing as mp
import os
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm.auto import tqdm

type Partition = Iterable[Collection[Hashable]]
type PartitionMethod = Callable[..., Partition]
type StatsSource = pd.DataFrame | str | os.PathLike[str] | None
type Task = tuple[float, int]

STAT_COLUMNS: tuple[str, ...] = (
  "method",
  "gamma",
  "seed",
  "n_communities",
  "b_p",
  "mean_size",
)

#: Columns that must be present for a cached row to count as already computed.
REQUIRED_COLUMNS: tuple[str, ...] = ("n_communities", "b_p", "mean_size")

#: Decimals used to match cached resolutions against the requested grid.
GAMMA_DECIMALS: int = 10

SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                  "#e87ba4", "#008300", "#4a3aa7", "#e34948")

#: Set in each worker process by :func:`_init_worker` before any task runs.
_WORKER: tuple[ResolutionSweep, str, PartitionMethod] | None = None


def _init_worker(sweep: ResolutionSweep, name: str, method: PartitionMethod) -> None:
  """Store the sweep a worker process will run tasks against.

  Args:
    sweep: the sweep holding the graph.
    name: label written to the ``method`` field of every row.
    method: community-detection callable.
  """
  global _WORKER
  _WORKER = (sweep, name, method)


def _worker_stats(task: Task) -> dict[str, object]:
  """Compute one row of the sweep inside a worker process.

  Args:
    task: (gamma, seed) pair to evaluate.

  Returns:
    dict of community statistics.

  Raises:
    RuntimeError: if the worker was started without its initializer.
  """
  if _WORKER is None:
    raise RuntimeError("worker state is unset; the pool was not started by run()")
  sweep, name, method = _WORKER
  gamma, seed = task
  return sweep.community_stats(gamma, seed, method, name=name)


def _method_name(method: PartitionMethod) -> str:
  """Derive a stable label for a community-detection callable.

  Args:
    method: the callable.

  Returns:
    its ``__name__``, unwrapping :class:`functools.partial`, else its repr.
  """
  target = method
  while not hasattr(target, "__name__") and hasattr(target, "func"):
    target = target.func
  return getattr(target, "__name__", repr(method))


@lru_cache(maxsize=None)
def _accepts_seed(method: PartitionMethod) -> bool:
  """Report whether a community-detection callable takes a ``seed`` argument.

  Args:
    method: the callable.

  Returns:
    True when ``seed`` is one of its parameters.
  """
  return "seed" in inspect.signature(method).parameters


class ResolutionSweep:
  """Sweep community-detection methods over a grid of resolutions and seeds.

  Args:
    graph: the graph to partition.
    gammas: resolution values to sweep.
    seeds: random seeds replicated at every resolution.
    methods: community-detection callables, or a mapping of label to callable.
      Each is called as ``method(graph, resolution=gamma)``, with ``seed=seed``
      added when the callable accepts it.
    max_workers: processes used by :meth:`run`.
    chunksize: tasks handed to a worker at a time.
    progress: show a tqdm progress bar per method while :meth:`run` works.
    stats_path: default CSV path for :meth:`run` and the plotting methods.
      When it already holds results, :meth:`run` reuses them and computes only
      the missing (method, gamma, seed) combinations.
  """

  def __init__(
    self,
    graph: nx.Graph,
    gammas: Sequence[float] | np.ndarray,
    seeds: Sequence[int],
    methods: Sequence[PartitionMethod] | Mapping[str, PartitionMethod] = (
      nx.community.louvain_communities,
    ),
    *,
    max_workers: int = 8,
    chunksize: int = 4,
    progress: bool = True,
    stats_path: str | os.PathLike[str] | None = None,
  ) -> None:
    self.graph: nx.Graph = graph
    self.gammas: list[float] = [float(g) for g in gammas]
    self.seeds: list[int] = [int(s) for s in seeds]
    self.methods: dict[str, PartitionMethod] = (
      dict(methods) if isinstance(methods, Mapping)
      else {_method_name(m): m for m in methods}
    )
    self.max_workers: int = max_workers
    self.chunksize: int = chunksize
    self.progress: bool = progress
    self.stats_path: Path | None = Path(stats_path) if stats_path else None
    self.stats: pd.DataFrame | None = None
    self._degrees: dict[Hashable, int] | None = None

  def __repr__(self) -> str:
    return (f"{type(self).__name__}(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, gammas={len(self.gammas)}, "
            f"seeds={len(self.seeds)}, methods={list(self.methods)})")

  # --- one partition ---------------------------------------------------------

  @property
  def degrees(self) -> dict[Hashable, int]:
    """Node degrees of the graph, computed once.

    Returns:
      dict mapping each node to its degree.
    """
    if self._degrees is None:
      self._degrees = dict(self.graph.degree())
    return self._degrees

  def _resolve(self, method: PartitionMethod | str) -> tuple[str, PartitionMethod]:
    """Look up a method by label, or label a callable.

    Args:
      method: a registered method label or a community-detection callable.

    Returns:
      tuple of (label, callable).
    """
    if isinstance(method, str):
      return method, self.methods[method]
    return _method_name(method), method

  def null_weight(self, communities: Iterable[Collection[Hashable]]) -> float:
    """Compute the null-model weight of a partition.

    Args:
      communities: the communities of the partition.

    Returns:
      ``b_p = sum_g (K_g / 2M)**2``, with ``K_g`` the total degree of community
      ``g`` and ``2M`` the total degree of the graph.
    """
    degree = self.degrees
    two_m = float(sum(degree.values()))
    return float(sum((sum(degree[n] for n in c) / two_m) ** 2 for c in communities))

  def partition(
    self,
    gamma: float,
    seed: int,
    method: PartitionMethod | str = nx.community.louvain_communities,
  ) -> list[Collection[Hashable]]:
    """Partition the graph once.

    Args:
      gamma: resolution parameter.
      seed: random seed, passed only if the method accepts one.
      method: a registered method label or a community-detection callable.

    Returns:
      list of communities, each a collection of nodes.
    """
    _, func = self._resolve(method)
    kwargs: dict[str, object] = {"resolution": gamma}
    if _accepts_seed(func):
      kwargs["seed"] = seed
    return list(func(self.graph, **kwargs))

  def community_stats(
    self,
    gamma: float,
    seed: int,
    method: PartitionMethod | str = nx.community.louvain_communities,
    *,
    name: str | None = None,
  ) -> dict[str, object]:
    """Partition the graph and summarize the community sizes.

    Args:
      gamma: resolution parameter.
      seed: random seed.
      method: a registered method label or a community-detection callable.
      name: label written to the ``method`` field; defaults to the method's own.

    Returns:
      dict with the method label, the gamma, the seed, the number of
      communities, the null-model weight ``b_p`` and the mean community size.
    """
    label, func = self._resolve(method)
    communities = self.partition(gamma, seed, func)
    sizes = [len(c) for c in communities]
    return {
      "method": name or label,
      "gamma": gamma,
      "seed": seed,
      "n_communities": len(sizes),
      "b_p": self.null_weight(communities),
      "mean_size": float(np.mean(sizes)),
    }

  # --- the statistics table --------------------------------------------------

  @staticmethod
  def _normalize(stats: pd.DataFrame) -> pd.DataFrame:
    """Coerce a statistics table to the sweep's columns and drop unusable rows.

    Args:
      stats: a raw table, e.g. read from CSV.

    Returns:
      frame with the sweep's columns, numeric statistics, rounded resolutions,
      no duplicate (method, gamma, seed) keys and no partially filled rows.
    """
    stats = stats.reindex(columns=list(STAT_COLUMNS)).copy()
    for column in STAT_COLUMNS[1:]:
      stats[column] = pd.to_numeric(stats[column], errors="coerce")
    stats["gamma"] = stats["gamma"].round(GAMMA_DECIMALS)
    stats = stats.dropna(subset=["method", "gamma", "seed", *REQUIRED_COLUMNS])
    stats["method"] = stats["method"].astype(str)   # after dropna: NaN -> "nan"
    stats["seed"] = stats["seed"].astype(int)
    return stats.drop_duplicates(["method", "gamma", "seed"], keep="last")

  @classmethod
  def _empty(cls) -> pd.DataFrame:
    """An empty statistics table with the sweep's columns.

    Returns:
      the empty frame.
    """
    return cls._normalize(pd.DataFrame(columns=list(STAT_COLUMNS)))

  def _known(self, path: Path | None) -> pd.DataFrame:
    """Collect the statistics already held in memory and on disk.

    Args:
      path: CSV path to read earlier results from; may be missing or None.

    Returns:
      the normalized statistics, empty when nothing has been computed yet.
    """
    frames = [frame for frame in (self.stats, self._read(path)) if frame is not None]
    if not frames:
      return self._empty()
    return self._normalize(pd.concat(frames, ignore_index=True))

  @classmethod
  def _read(cls, path: Path | None) -> pd.DataFrame | None:
    """Read a statistics table from disk.

    Args:
      path: CSV path; may be None or point at a missing file.

    Returns:
      the normalized statistics, or None when there is nothing to read.
    """
    if path is None or not path.exists():
      return None
    return cls._normalize(pd.read_csv(path))

  def load(self, source: StatsSource = None) -> pd.DataFrame:
    """Resolve a statistics table from a frame, a CSV path or this sweep.

    Args:
      source: a DataFrame, a CSV path, or None to use the last run and then
        ``stats_path``.

    Returns:
      the statistics DataFrame.

    Raises:
      ValueError: if no source is given and no sweep has been run or saved.
    """
    if isinstance(source, pd.DataFrame):
      return source
    if source is not None:
      return self._normalize(pd.read_csv(source))
    if self.stats is not None:
      return self.stats
    stats = self._read(self.stats_path)
    if stats is None:
      raise ValueError("no statistics available; call run() or pass a source")
    return stats

  def pending(
    self,
    path: str | os.PathLike[str] | None = None,
  ) -> dict[str, list[Task]]:
    """List the (gamma, seed) pairs still missing for each method.

    Args:
      path: CSV path holding earlier results; defaults to ``stats_path``.

    Returns:
      dict mapping each method label to the pairs that would be computed.
    """
    return self._tasks(self._known(Path(path) if path else self.stats_path))

  def _tasks(self, known: pd.DataFrame) -> dict[str, list[Task]]:
    """Work out which grid points each method still needs.

    Args:
      known: statistics already computed; an empty frame asks for the full grid.

    Returns:
      dict mapping each method label to its outstanding (gamma, seed) pairs.
    """
    done = set(zip(known["method"], known["gamma"], known["seed"]))
    grid = list(itertools.product(self.gammas, self.seeds))
    return {
      name: [(gamma, seed) for gamma, seed in grid
             if (name, round(gamma, GAMMA_DECIMALS), seed) not in done]
      for name in self.methods
    }

  def run(
    self,
    path: str | os.PathLike[str] | None = None,
    *,
    reuse: bool = True,
  ) -> pd.DataFrame:
    """Run every method over every (gamma, seed) pair and save the statistics.

    Args:
      path: CSV path to read earlier results from and write to; defaults to the
        instance's ``stats_path``. Nothing is written when both are None.
      reuse: when True, keep the rows already present in that CSV (or in the
        last run) and compute only the missing combinations; when False,
        recompute the whole grid and overwrite.

    Returns:
      DataFrame with one row per method, gamma and seed, sorted in that order.
      Rows carried over from earlier runs, including combinations outside the
      current grid, are kept.
    """
    target = Path(path) if path else self.stats_path
    known = self._known(target) if reuse else self._empty()

    rows: list[dict[str, object]] = []
    for name, tasks in self._tasks(known).items():
      if tasks:
        rows.extend(self._map(name, self.methods[name], tasks))

    computed = self._normalize(pd.DataFrame(rows, columns=list(STAT_COLUMNS)))
    stats = self._normalize(pd.concat([known, computed], ignore_index=True))
    stats = stats.sort_values(["method", "gamma", "seed"], ignore_index=True)

    if target is not None:
      stats.to_csv(target, index=False)
      self.stats_path = target
    self.stats = stats
    return stats

  def _map(
    self,
    name: str,
    method: PartitionMethod,
    tasks: Sequence[Task],
  ) -> list[dict[str, object]]:
    """Evaluate one method over a list of grid points, in parallel.

    Args:
      name: label written to the ``method`` field of every row.
      method: community-detection callable.
      tasks: the (gamma, seed) pairs to evaluate.

    Returns:
      list of statistics dicts, one per task.
    """
    # Forking keeps methods defined in a notebook usable: the child inherits
    # __main__, so the initializer's arguments unpickle by reference there.
    with ProcessPoolExecutor(
      max_workers=self.max_workers,
      mp_context=mp.get_context("fork"),
      initializer=_init_worker,
      initargs=(self, name, method),
    ) as pool:
      results = pool.map(_worker_stats, tasks, chunksize=self.chunksize)
      return list(tqdm(results, total=len(tasks), desc=name, unit="run",
                       disable=not self.progress))

  @staticmethod
  def aggregate(stats: pd.DataFrame) -> pd.DataFrame:
    """Average the sweep over seeds.

    Args:
      stats: frame returned by :meth:`run`.

    Returns:
      frame indexed by (method, gamma) with a mean and std column per statistic.
    """
    return stats.groupby(["method", "gamma"]).agg(["mean", "std"])

  # --- figures ---------------------------------------------------------------

  def _series(self, source: StatsSource) -> list[tuple[str, pd.DataFrame, str | None]]:
    """Split the aggregated statistics into one block per method.

    Args:
      source: a DataFrame, a CSV path, or None to use the saved sweep.

    Returns:
      list of (method label, gamma-indexed aggregate, color); the color is
      black for a lone method, so a single series is not needlessly coloured.
    """
    agg = self.aggregate(self.load(source))
    names = list(dict.fromkeys(agg.index.get_level_values("method")))
    single = len(names) == 1
    return [(name, agg.loc[name],
             "black" if single else SERIES_COLORS[i % len(SERIES_COLORS)])
            for i, name in enumerate(names)]

  @staticmethod
  def _axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Reuse the given axes or open a new figure.

    Args:
      ax: axes to draw on; a new figure is created when None.
      figsize: size of that new figure.

    Returns:
      tuple of (figure, axes).
    """
    if ax is not None:
      return ax.figure, ax
    return plt.subplots(figsize=figsize)

  @staticmethod
  def _finish(fig: Figure, save: str | os.PathLike[str] | None) -> Figure:
    """Lay a figure out and optionally write it to disk.

    Args:
      fig: the figure.
      save: path to write it to; None to skip.

    Returns:
      the figure.
    """
    fig.tight_layout()
    if save is not None:
      fig.savefig(save, dpi=200)
    return fig

  def _plot_statistic(
    self,
    series: Sequence[tuple[str, pd.DataFrame, str | None]],
    column: str,
    ylabel: str,
    ax: Axes,
    panel: str | None,
    *,
    corner: str = "top",
  ) -> None:
    """Draw one statistic against the resolution, one error bar per method.

    Args:
      series: blocks returned by :meth:`_series`.
      column: statistic to plot.
      ylabel: y-axis label.
      ax: axes to draw on.
      panel: panel label drawn in a corner; None to omit it.
      corner: "top" or "bottom", the corner the panel label goes in.
    """
    for name, agg, color in series:
      ax.errorbar(
        agg.index,
        agg[(column, "mean")],
        yerr=agg[(column, "std")],
        fmt="s",
        color=color,
        markersize=4,
        elinewidth=0.8,
        capsize=2,
        label=name,
      )
    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    gammas = np.concatenate([agg.index.to_numpy() for _, agg, _ in series])
    margin = 0.02 * ((gammas.max() - gammas.min()) or 1.0)
    ax.set_xlim(gammas.min() - margin, gammas.max() + margin)
    if panel is not None:
      y = 0.95 if corner == "top" else 0.05
      ax.text(0.03, y, panel, transform=ax.transAxes,
              va="top" if corner == "top" else "bottom", fontsize=13)
    if len(series) > 1:
      ax.legend()

  def plot_n_communities(
    self,
    source: StatsSource = None,
    *,
    save: str | os.PathLike[str] | None = None,
    ax: Axes | None = None,
    panel: str | None = "(a)",
  ) -> Figure:
    """Plot the number of communities against the resolution.

    Args:
      source: a DataFrame, a CSV path, or None to use the saved sweep.
      save: path to write the figure to.
      ax: axes to draw on; a new figure is created when omitted.
      panel: panel label drawn in the top corner.

    Returns:
      the figure.
    """
    fig, ax = self._axes(ax, (4.2, 3.8))
    self._plot_statistic(self._series(source), "n_communities", r"$n_c$", ax,
                         panel, corner="top")
    ax.set_ylim(0, None)
    return self._finish(fig, save)

  def plot_b_p(
    self,
    source: StatsSource = None,
    *,
    save: str | os.PathLike[str] | None = None,
    ax: Axes | None = None,
    panel: str | None = "(b)",
  ) -> Figure:
    """Plot the null-model weight of the partition against the resolution.

    Args:
      source: a DataFrame, a CSV path, or None to use the saved sweep.
      save: path to write the figure to.
      ax: axes to draw on; a new figure is created when omitted.
      panel: panel label drawn in the bottom corner.

    Returns:
      the figure.
    """
    fig, ax = self._axes(ax, (4.2, 3.8))
    # bottom left: b_p sits at 1 for the low resolutions, where n_c is small
    self._plot_statistic(self._series(source), "b_p", r"$m$", ax,
                         panel, corner="bottom")
    ax.set_ylim(0, 1.02)
    return self._finish(fig, save)

  def plot_n_communities_mod_regularizer(
    self,
    source: StatsSource = None,
    *,
    save: str | os.PathLike[str] | None = None,
  ) -> Figure:
    """Plot the number of communities and the null-model weight side by side.

    Args:
      source: a DataFrame, a CSV path, or None to use the saved sweep.
      save: path to write the figure to.

    Returns:
      the figure, with n_c on the left panel and b_p on the right one.
    """
    stats = self.load(source)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    self.plot_n_communities(stats, ax=axes[0])
    self.plot_b_p(stats, ax=axes[1])
    return self._finish(fig, save)
