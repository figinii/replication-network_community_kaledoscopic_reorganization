"""Core-periphery benchmark network drawn from a stochastic block model."""

from __future__ import annotations

import itertools
import os
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure

type BridgeMode = Literal["uniform", "per_node"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
BLUE, BLUE_TINT = "#2a78d6", "#9ec5f4"
ORANGE, ORANGE_TINT = "#eb6834", "#f6bfa8"
EDGE_C = "#cfcec8"

ROLE_COLOR: dict[str, str] = {"core_large": BLUE, "core_small": ORANGE,
                              "periph_large": BLUE_TINT, "periph_small": ORANGE_TINT}

BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


class CorePeripherySBM:
  """Two dense core communities with peripheral communities hanging off them.

  Blocks are numbered with the large core first, then the small core, then the
  peripheral communities attached to the large core, then those attached to the
  small one.

  Args:
    n_core_large: nodes in the large core community.
    n_core_small: nodes in the small core community.
    p_core_large: edge probability within the large core community.
    p_core_small: edge probability within the small core community.
    p_core_core: edge probability between the two core communities.
    size_periph: nodes in each peripheral community.
    n_periph_on_large: peripheral communities attached to the large core.
    n_periph_on_small: peripheral communities attached to the small core.
    p_periph: edge probability within a peripheral community.
    bridge_edges: edges joining a peripheral community to its core community.
    bridge_mode: "uniform" draws bridge_edges pairs uniformly at random,
      "per_node" gives every peripheral node one edge to a random core node.
    seed: random seed for the sample.
  """

  def __init__(
    self,
    *,
    n_core_large: int = 400,
    n_core_small: int = 200,
    p_core_large: float = 0.7,
    p_core_small: float = 0.5,
    p_core_core: float = 0.1,
    size_periph: int = 20,
    n_periph_on_large: int = 10,
    n_periph_on_small: int = 20,
    p_periph: float = 0.5,
    bridge_edges: int = 20,
    bridge_mode: BridgeMode = "uniform",
    seed: int = 7,
  ) -> None:
    self.n_core_large = n_core_large
    self.n_core_small = n_core_small
    self.p_core_large = p_core_large
    self.p_core_small = p_core_small
    self.p_core_core = p_core_core
    self.size_periph = size_periph
    self.n_periph_on_large = n_periph_on_large
    self.n_periph_on_small = n_periph_on_small
    self.p_periph = p_periph
    self.bridge_edges = bridge_edges
    self.bridge_mode: BridgeMode = bridge_mode
    self.seed = seed

    self.large, self.small = 0, 1
    self.n_periph = n_periph_on_large + n_periph_on_small
    self.periph_on_large = list(range(2, 2 + n_periph_on_large))
    self.periph_on_small = list(range(2 + n_periph_on_large, 2 + self.n_periph))
    self.master: dict[int, int] = (
      {b: self.large for b in self.periph_on_large}
      | {b: self.small for b in self.periph_on_small}
    )
    self.sizes = [n_core_large, n_core_small] + [size_periph] * self.n_periph
    self.n_blocks = len(self.sizes)

    self.probs = self.block_probs()
    #: The pure-SBM matrix, with the bridges written as probabilities instead of
    #: exact counts; pass it to nx.stochastic_block_model for a Bernoulli model.
    self.probs_nominal = self.block_probs(bridges_as_probability=True)
    self.graph, self.blocks = self._build()
    self._edge_matrix: np.ndarray | None = None

  def __repr__(self) -> str:
    return (f"{type(self).__name__}(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, blocks={self.n_blocks}, "
            f"seed={self.seed})")

  # --- the model -------------------------------------------------------------

  def block_probs(self, bridges_as_probability: bool = False) -> np.ndarray:
    """Build the block-to-block connection-probability matrix.

    Args:
      bridges_as_probability: when True, fill the periphery-core blocks with the
        probability matching bridge_edges in expectation instead of 0.

    Returns:
      (n_blocks, n_blocks) array of connection probabilities.
    """
    p = np.zeros((self.n_blocks, self.n_blocks))
    p[self.large, self.large] = self.p_core_large
    p[self.small, self.small] = self.p_core_small
    p[self.large, self.small] = p[self.small, self.large] = self.p_core_core
    for b, master in self.master.items():
      p[b, b] = self.p_periph
      if bridges_as_probability:
        rate = self.bridge_edges / (self.sizes[b] * self.sizes[master])
        p[b, master] = p[master, b] = rate
    return p

  def _add_bridges(self, graph: nx.Graph, blocks: list[list[int]],
                   rng: np.random.Generator) -> None:
    """Attach each peripheral community to its core community, in place.

    Args:
      graph: graph to add the bridge edges to.
      blocks: list of node lists, one per block.
      rng: random generator used to pick the endpoints.

    Raises:
      ValueError: if bridge_mode is not a known mode.
    """
    for b, master in self.master.items():
      periph, core = blocks[b], blocks[master]
      if self.bridge_mode == "per_node":
        for u in periph:
          graph.add_edge(u, core[rng.integers(len(core))], bridge=True)
      elif self.bridge_mode == "uniform":
        pairs = list(itertools.product(periph, core))
        for i in rng.choice(len(pairs), size=self.bridge_edges, replace=False):
          u, v = pairs[i]
          graph.add_edge(u, v, bridge=True)
      else:
        raise ValueError(f"unknown bridge mode: {self.bridge_mode!r}")

  def _build(self) -> tuple[nx.Graph, list[list[int]]]:
    """Sample the network.

    Returns:
      Tuple of (graph, blocks): the graph carries a "block" attribute on every
      node, and blocks is the list of node lists per block.
    """
    # The specification fixes the number of periphery-core edges rather than
    # their probability, so those are drawn here instead of by the SBM.
    rng = np.random.default_rng(self.seed)
    graph = nx.stochastic_block_model(self.sizes, self.probs, seed=self.seed)
    blocks = [sorted(part) for part in graph.graph["partition"]]
    self._add_bridges(graph, blocks, rng)
    return graph, blocks

  # --- checking the sample against the specification -------------------------

  @property
  def edge_matrix(self) -> np.ndarray:
    """Edge counts between every pair of blocks, computed once.

    Returns:
      (n_blocks, n_blocks) integer array; the diagonal holds internal edges.
    """
    if self._edge_matrix is None:
      block_of = nx.get_node_attributes(self.graph, "block")
      matrix = np.zeros((self.n_blocks, self.n_blocks), dtype=int)
      for u, v in self.graph.edges():
        i, j = block_of[u], block_of[v]
        matrix[i, j] += 1
        if i != j:
          matrix[j, i] += 1
      self._edge_matrix = matrix
    return self._edge_matrix

  @property
  def roles(self) -> list[str]:
    """Role of each block, as a key into ROLE_COLOR.

    Returns:
      List of role names, one per block.
    """
    return ["core_large", "core_small"] + [
      "periph_large" if self.master[b] == self.large else "periph_small"
      for b in range(2, self.n_blocks)
    ]

  def summary(self) -> pd.DataFrame:
    """Per-block sizes, densities and external edge counts.

    Returns:
      DataFrame with one row per block: its kind, node count, internal edges,
      realised and target internal density, and its edges to other blocks.
    """
    matrix = self.edge_matrix
    internal = np.diag(matrix)
    max_internal = np.array([n * (n - 1) / 2 for n in self.sizes])
    kind = [f"core ({self.n_core_large})", f"core ({self.n_core_small})"] + [
      f"periphery -> {self.n_core_large if self.master[b] == self.large else self.n_core_small}"
      for b in range(2, self.n_blocks)
    ]
    return pd.DataFrame({
      "block": range(self.n_blocks),
      "kind": kind,
      "nodes": self.sizes,
      "internal_edges": internal,
      "internal_density": internal / max_internal,
      "target_density": np.diag(self.probs),
      "external_edges": matrix.sum(axis=1) - internal,
    })

  def checks(self) -> dict[str, object]:
    """Compare the realised network with the values the specification fixes.

    Returns:
      Dict with the node and edge counts, the core-core edges and their density,
      the distinct bridge-edge counts, and the edges between different
      peripheral communities (which should be zero).
    """
    matrix = self.edge_matrix
    core_core = int(matrix[self.large, self.small])
    periph = np.array(sorted(self.master))
    cross = matrix[np.ix_(periph, periph)].copy()
    np.fill_diagonal(cross, 0)
    return {
      "nodes": self.graph.number_of_nodes(),
      "edges": self.graph.number_of_edges(),
      "core_core_edges": core_core,
      "core_core_density": core_core / (self.n_core_large * self.n_core_small),
      "bridge_edges_per_community": sorted({int(matrix[b, m]) for b, m in self.master.items()}),
      "inter_periphery_edges": int(cross.sum()),
    }

  # --- figures ---------------------------------------------------------------

  def _community_layout(self) -> np.ndarray:
    """Place the two cores at the centre and their peripheries on arcs around them.

    Returns:
      (n_blocks, 2) array of positions.
    """
    pos = np.zeros((self.n_blocks, 2))
    pos[self.large] = (-1.6, 0.0)
    pos[self.small] = (1.5, 0.0)
    for arc, centre, radius, (a0, a1) in (
      (self.periph_on_large, pos[self.large], 2.6, (108, 252)),
      (self.periph_on_small, pos[self.small], 2.9, (-72, 72)),
    ):
      for b, angle in zip(arc, np.linspace(a0, a1, len(arc))):
        t = np.deg2rad(angle)
        pos[b] = centre + radius * np.array([np.cos(t), np.sin(t)])
    return pos

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

  def plot_communities(self, ax: Axes | None = None,
                       save: str | os.PathLike[str] | None = None) -> Figure:
    """Draw one mark per community, sized by nodes and joined by edge counts.

    Args:
      ax: axes to draw on; a new figure is created when omitted.
      save: path to write the figure to.

    Returns:
      the figure.
    """
    matrix = self.edge_matrix
    pos = self._community_layout()
    roles = self.roles

    if ax is None:
      fig, ax = plt.subplots(figsize=(11, 8), facecolor=SURFACE)
    else:
      fig = ax.figure
    ax.set_facecolor(SURFACE)

    # Edges, width on a log scale across the observed range.
    inter = [(i, j, matrix[i, j])
             for i in range(self.n_blocks) for j in range(i + 1, self.n_blocks)
             if matrix[i, j]]
    counts = np.array([c for _, _, c in inter])
    span = np.log(counts.max()) - np.log(counts.min())
    widths = 0.9 + 6.5 * (np.log(counts) - np.log(counts.min())) / (span or 1.0)
    for (i, j, count), width in zip(inter, widths):
      ax.plot(*zip(pos[i], pos[j]), lw=width,
              color=INK_3 if count > 100 else EDGE_C,
              solid_capstyle="round", zorder=1)

    # Communities: disc area proportional to the node count.
    ax.scatter(pos[:, 0], pos[:, 1], s=[n * 10 for n in self.sizes],
               c=[ROLE_COLOR[r] for r in roles], edgecolors=SURFACE,
               linewidths=2, zorder=2)

    for b, label in ((self.large, f"{self.n_core_large}\n$p$ = {self.p_core_large}"),
                     (self.small, f"{self.n_core_small}\n$p$ = {self.p_core_small}")):
      ax.annotate(label, pos[b], ha="center", va="center", color="white",
                  fontsize=9, fontweight="bold", zorder=3)

    mid = (pos[self.large] + pos[self.small]) / 2
    ax.annotate(f"{matrix[self.large, self.small]:,} edges\n$p$ = {self.p_core_core}",
                mid + (0, 0.34), ha="center", va="bottom", color=INK_2, fontsize=9.5)
    ax.annotate(f"{self.bridge_edges} edges",
                pos[self.periph_on_large[0]] * 0.55 + pos[self.large] * 0.45,
                ha="center", va="center", color=INK_2, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    ax.annotate(f"{self.n_periph} peripheral communities\n"
                f"{self.size_periph} nodes each, $p$ = {self.p_periph}",
                (0.01, 0.96), xycoords="axes fraction",
                ha="left", va="top", color=INK_2, fontsize=9.5)

    handles = [
      mpl.lines.Line2D([], [], marker="o", ls="", color=ROLE_COLOR["core_large"],
                       ms=15, label=f"core community, {self.n_core_large} nodes"),
      mpl.lines.Line2D([], [], marker="o", ls="", color=ROLE_COLOR["core_small"],
                       ms=11, label=f"core community, {self.n_core_small} nodes"),
      mpl.lines.Line2D([], [], marker="o", ls="", color=ROLE_COLOR["periph_large"],
                       ms=7, label=f"{self.n_periph_on_large} peripheral communities "
                                   f"on the {self.n_core_large}-node core"),
      mpl.lines.Line2D([], [], marker="o", ls="", color=ROLE_COLOR["periph_small"],
                       ms=7, label=f"{self.n_periph_on_small} peripheral communities "
                                   f"on the {self.n_core_small}-node core"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, labelspacing=0.8, handletextpad=1.0,
              labelcolor=INK_2, fontsize=9.5)

    ax.set_title("Core–periphery model network, one mark per community",
                 color=INK, fontsize=14, loc="left", pad=14)
    ax.annotate("disc area ∝ community size · edge width ∝ number of edges "
                "between the two communities (log scale)",
                (0, 1.005), xycoords="axes fraction", color=INK_2,
                fontsize=9.5, va="bottom")
    ax.set_aspect("equal")
    ax.axis("off")
    return self._finish(fig, save)

  def plot_block_matrix(self, ax: Axes | None = None,
                        save: str | os.PathLike[str] | None = None) -> Figure:
    """Draw the edge counts between blocks as a matrix.

    Args:
      ax: axes to draw on; a new figure is created when omitted.
      save: path to write the figure to.

    Returns:
      the figure.
    """
    matrix = self.edge_matrix
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)

    if ax is None:
      fig, ax = plt.subplots(figsize=(7.2, 6), facecolor=SURFACE)
    else:
      fig = ax.figure
    im = ax.imshow(np.ma.masked_equal(matrix, 0), cmap=cmap,
                   norm=LogNorm(vmin=1, vmax=matrix.max()))
    for edge in (1.5, 1.5 + self.n_periph_on_large):
      ax.axhline(edge, color=INK_3, lw=0.8)
      ax.axvline(edge, color=INK_3, lw=0.8)

    # The matrix is symmetric, so the two core blocks are spelled out once, on y.
    groups = [1.5 + self.n_periph_on_large / 2,
              1.5 + self.n_periph_on_large + self.n_periph_on_small / 2]
    periph_labels = [f"periphery on {self.n_core_large}",
                     f"periphery on {self.n_core_small}"]
    ax.set_xticks(groups)
    ax.set_xticklabels(periph_labels, fontsize=9)
    ax.set_yticks([0, 1] + groups)
    ax.set_yticklabels([f"core {self.n_core_large}", f"core {self.n_core_small}",
                        *periph_labels], fontsize=9)
    ax.tick_params(colors=INK_2, length=0)
    for spine in ax.spines.values():
      spine.set_visible(False)

    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label("edges between blocks", color=INK_2, fontsize=9.5)
    bar.ax.tick_params(colors=INK_2, length=2)
    bar.outline.set_visible(False)
    ax.set_title(f"Edge counts between the {self.n_blocks} blocks",
                 color=INK, fontsize=13, loc="left", pad=12)
    return self._finish(fig, save)
