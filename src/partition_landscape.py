"""Modularity of every partition of a small graph, across resolutions."""

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from more_itertools import set_partitions


class PartitionLandscape:
  """Every partition of a graph, scored by modularity at every resolution.

  Attributes:
    graph: the graph the partitions are of.
    nodes: list of nodes, in the order used to enumerate the partitions.
    partitions: list of every partition, each a list of communities.
    gammas: array of resolutions the modularity is evaluated at.
    internal_fraction: per-partition share of edges falling inside communities.
    degree_fraction: per-partition sum of squared community degree shares.
    modularity: (partition, gamma) array of modularity values.
    distance: (partition, partition) array of Mirkin distances.
    order: array mapping each position along the chain to a partition id.
    position: array mapping each partition id to its position along the chain.
  """

  def __init__(self, graph, positions=None, n_gammas=1000, max_gamma=1.0):
    """Enumerate the partitions, score them, and order them by similarity.

    Args:
      graph: graph to partition; small enough for every partition to be listed.
      positions: optional dict of node positions used to seed partition layouts.
      n_gammas: number of resolutions sampled between 0 and max_gamma.
      max_gamma: largest resolution sampled.
    """
    self.graph = graph
    self.positions = dict(positions) if positions else None
    self.nodes = list(graph.nodes())
    self.node_index = {n: i for i, n in enumerate(self.nodes)}
    self.partitions = list(set_partitions(self.nodes))
    self.n_partitions = len(self.partitions)
    self.gammas = np.linspace(0, max_gamma, n_gammas)

    self._score_partitions()
    self.distance = self._mirkin_distances()
    self.order, self.position = self.order_by_similarity()

  # --- computation -----------------------------------------------------------

  def _score_partitions(self):
    """Fill in the modularity coefficients and the modularity array."""
    # Modularity is linear in gamma:
    #   Q(gamma) = sum_c L_c / m - gamma * sum_c (d_c / 2m)^2
    # so two coefficients per partition cover every gamma, instead of one
    # nx.community.modularity call per (partition, gamma) pair.
    m = self.graph.number_of_edges()
    degree = dict(self.graph.degree())
    edges = list(self.graph.edges())

    self.internal_fraction = np.empty(self.n_partitions)
    self.degree_fraction = np.empty(self.n_partitions)
    for pid, partition in enumerate(self.partitions):
      internal = 0.0
      degree_sq = 0.0
      for community in partition:
        members = set(community)
        internal += sum(1 for u, v in edges if u in members and v in members) / m
        degree_sq += (sum(degree[n] for n in members) / (2 * m)) ** 2
      self.internal_fraction[pid] = internal
      self.degree_fraction[pid] = degree_sq

    self.modularity = (self.internal_fraction[:, None]
                       - np.outer(self.degree_fraction, self.gammas))

  def modularity_at(self, partition_id, gamma):
    """Modularity of one partition at one resolution.

    Args:
      partition_id: index into partitions.
      gamma: resolution parameter.

    Returns:
      The modularity value.
    """
    return self.internal_fraction[partition_id] - gamma * self.degree_fraction[partition_id]

  def verify(self, partition_id=10, gamma_index=500):
    """Check one modularity value against networkx.

    Args:
      partition_id: index into partitions.
      gamma_index: index into gammas.

    Returns:
      True when the values agree.

    Raises:
      AssertionError: if they disagree.
    """
    gamma = self.gammas[gamma_index]
    expected = nx.community.modularity(
      self.graph, self.partitions[partition_id], resolution=gamma)
    assert np.isclose(self.modularity[partition_id, gamma_index], expected)
    return True

  def _mirkin_distances(self):
    """Pairwise Mirkin distances between the partitions.

    Returns:
      A (partition, partition) array of distances.
    """
    # Mirkin distance: the number of node pairs two partitions disagree on
    # (together in one partition, apart in the other). It is a metric.
    node_pairs = [(i, j) for i in range(len(self.nodes))
                  for j in range(i + 1, len(self.nodes))]

    comembership = np.zeros((self.n_partitions, len(node_pairs)))
    for pid, partition in enumerate(self.partitions):
      labels = np.empty(len(self.nodes), dtype=int)
      for cid, community in enumerate(partition):
        for n in community:
          labels[self.node_index[n]] = cid
      comembership[pid] = [labels[i] == labels[j] for i, j in node_pairs]

    same_count = comembership.sum(axis=1)
    distance = (same_count[:, None] + same_count[None, :]
                - 2 * (comembership @ comembership.T))
    distance = np.maximum(distance, 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance

  # --- ordering --------------------------------------------------------------

  def chain_cost(self, order):
    """Total partition distance between consecutive positions.

    Args:
      order: array mapping each position to the partition placed there.

    Returns:
      The summed distance.
    """
    return 0.5 * (self.distance[np.ix_(order, order)] * self._adjacent()).sum()

  def _adjacent(self):
    """Indicator array of positions one step apart along the chain.

    Returns:
      A (position, position) array of 1.0 for neighbours and 0.0 elsewhere.
    """
    index = np.arange(self.n_partitions)
    return (np.abs(index[:, None] - index[None, :]) == 1).astype(float)

  def _greedy_chain(self, start=0):
    """Chain the partitions by repeatedly stepping to the nearest unused one.

    Args:
      start: partition to start from.

    Returns:
      Array mapping each position to the partition placed there.
    """
    order = [start]
    remaining = set(range(self.n_partitions)) - {start}
    while remaining:
      candidates = np.fromiter(remaining, dtype=int)
      nearest = int(candidates[np.argmin(self.distance[order[-1], candidates])])
      order.append(nearest)
      remaining.discard(nearest)
    return np.array(order)

  def _relax(self, order, max_passes=300, shortlist_size=4000):
    """Swap partitions between positions until no swap lowers the chain cost.

    Args:
      order: array mapping each position to the partition placed there.
      max_passes: maximum number of swap rounds.
      shortlist_size: number of best candidate swaps considered per round.

    Returns:
      The improved order array.
    """
    order = order.copy()
    adjacent = self._adjacent()
    neighbours = [np.flatnonzero(adjacent[p]) for p in range(self.n_partitions)]
    shortlist_size = min(shortlist_size, self.n_partitions ** 2 - 1)

    for _ in range(max_passes):
      reach = self.distance[:, order] @ adjacent   # reach[p, i]: cost of putting p at position i
      current = reach[order, np.arange(self.n_partitions)]
      here = reach[order]
      delta = (here.T + here - current[:, None] - current[None, :]
               + 2 * adjacent * self.distance[np.ix_(order, order)])
      np.fill_diagonal(delta, 0.0)

      shortlist = np.argpartition(delta, shortlist_size, axis=None)[:shortlist_size]
      shortlist = shortlist[np.argsort(delta.flat[shortlist])]
      blocked = np.zeros(self.n_partitions, dtype=bool)
      applied = 0
      for flat in shortlist:
        if delta.flat[flat] > -1e-9:
          break
        i, j = divmod(int(flat), self.n_partitions)
        if blocked[i] or blocked[j]:
          continue          # only swap positions whose neighbours are untouched this
        order[[i, j]] = order[[j, i]]      # round, so the deltas stay exact
        blocked[[i, j]] = True
        blocked[neighbours[i]] = True
        blocked[neighbours[j]] = True
        applied += 1
      if applied == 0:
        break
    return order

  def order_by_similarity(self, start=0, max_passes=300):
    """Lay the partitions on a line so neighbouring positions are similar.

    Args:
      start: partition placed at the first position of the initial chain.
      max_passes: maximum number of swap rounds used to improve the chain.

    Returns:
      Tuple of (order, position): order maps a position to a partition id,
      position maps a partition id back to its position.
    """
    order = self._relax(self._greedy_chain(start), max_passes=max_passes)
    position = np.empty(self.n_partitions, dtype=int)
    position[order] = np.arange(self.n_partitions)
    return order, position

  def ordering_quality(self):
    """Distances between consecutive partitions, against two baselines.

    Returns:
      Dict with the mean distance between neighbours on the chain, the mean
      distance between two random partitions, and the mean distance to the
      closest other partition.
    """
    upper = self.distance[np.triu_indices(self.n_partitions, 1)]
    padded = self.distance + np.eye(self.n_partitions) * 1e9
    return {
      "mean_neighbour": self.chain_cost(self.order) / (self.n_partitions - 1),
      "mean_random": upper.mean(),
      "mean_closest": np.sort(padded, axis=1)[:, 0].mean(),
    }

  # --- persistence -----------------------------------------------------------

  def to_dataframe(self):
    """The landscape as one row per (gamma, partition) pair.

    Returns:
      DataFrame with Partition, Partition_Order, Communities, Modularity and
      Gamma columns, grouped by gamma.
    """
    communities = np.array([str(p) for p in self.partitions], dtype=object)
    n_gammas = len(self.gammas)
    return pd.DataFrame({
      "Partition": np.tile(np.arange(self.n_partitions), n_gammas),
      "Partition_Order": np.tile(self.position, n_gammas),
      "Communities": np.tile(communities, n_gammas),
      "Modularity": self.modularity.T.ravel(),
      "Gamma": np.repeat(self.gammas, self.n_partitions),
    })

  def save_csv(self, path):
    """Write the landscape to a csv file.

    Args:
      path: destination file path.

    Returns:
      The number of rows written.
    """
    frame = self.to_dataframe()
    frame.to_csv(path, index=False)
    return len(frame)

  # --- partition drawing -----------------------------------------------------

  def community_layout(self, partition, intra_weight=8.0, k=1.15, seed=0):
    """Spring layout with invisible intra-community edges added.

    Args:
      partition: list of communities, each a collection of nodes.
      intra_weight: weight of the added intra-community edges, relative to the
        real edges (weight 1.0).
      k: target distance between nodes passed to nx.spring_layout.
      seed: random seed for the layout.

    Returns:
      Dict mapping each node to its (x, y) position.
    """
    H = nx.Graph()
    H.add_nodes_from(self.graph)
    H.add_edges_from(self.graph.edges(), weight=1.0)
    for community in partition:
      for u, v in itertools.combinations(community, 2):
        H.add_edge(u, v, weight=intra_weight)   # invisible: never drawn, only pulls
    return nx.spring_layout(H, weight="weight", k=k, pos=self.positions, seed=seed)

  def draw_partition(self, partition_id, gamma=None, ax=None, spread=0.55,
                     base=0.14, seed=0):
    """Draw one partition with a coloured outline around each community.

    Args:
      partition_id: index into partitions.
      gamma: resolution to report in the title; None to omit it.
      ax: axes to draw on; a new figure is created when None.
      spread: outline width relative to the community's radius.
      base: minimum outline width, used for singleton communities.
      seed: random seed for the layout.

    Returns:
      The axes drawn on.
    """
    partition = self.partitions[partition_id]
    if ax is None:
      _, ax = plt.subplots(figsize=(7, 6))
    pos = self.community_layout(partition, seed=seed)
    palette = plt.get_cmap("tab10").colors
    node_color = {n: palette[cid % len(palette)]
                  for cid, community in enumerate(partition) for n in community}

    # Each community gets a metaball outline: a sum of Gaussians centred on its
    # nodes, drawn at a fixed level, so nearby nodes merge into one smooth blob.
    points = [np.array([pos[n] for n in community]) for community in partition]
    sigmas = [max(base, spread * np.hypot(*(pts - pts.mean(axis=0)).T).max())
              for pts in points]
    xy = np.array([pos[n] for n in self.graph.nodes()])
    pad = 1.8 * max(sigmas)
    grid_x = np.linspace(xy[:, 0].min() - pad, xy[:, 0].max() + pad, 320)
    grid_y = np.linspace(xy[:, 1].min() - pad, xy[:, 1].max() + pad, 320)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    for cid, (pts, sigma) in enumerate(zip(points, sigmas)):
      field = np.exp(-((mesh_x[..., None] - pts[:, 0]) ** 2
                       + (mesh_y[..., None] - pts[:, 1]) ** 2)
                     / (2 * sigma ** 2)).sum(axis=-1)
      color = palette[cid % len(palette)]
      ax.contourf(mesh_x, mesh_y, field, levels=[0.5, field.max() + 1],
                  colors=[color], alpha=0.15)
      ax.contour(mesh_x, mesh_y, field, levels=[0.5], colors=[color], linewidths=2)

    nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color="0.4", width=2)
    nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_size=650, edgecolors="black",
                           node_color=[node_color[n] for n in self.graph.nodes()])
    nx.draw_networkx_labels(self.graph, pos, ax=ax, font_weight="bold")

    title = f"#{partition_id} at position {self.position[partition_id]}: {partition}"
    if gamma is not None:
      title += f"\nQ({gamma:.3f}) = {self.modularity_at(partition_id, gamma):.3f}"
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax

  def draw_partitions(self, partition_ids, gamma=None, ncols=3, panel=(5.5, 5.0),
                      **kwargs):
    """Draw several partitions side by side, each at its own resolution.

    Args:
      partition_ids: indices into partitions.
      gamma: one resolution per partition id, in the same order and of the same
        length; a single number applies to every panel, None omits them.
      ncols: number of columns of panels.
      panel: (width, height) of one panel in inches.
      **kwargs: passed on to draw_partition.

    Returns:
      The figure drawn on.

    Raises:
      ValueError: if gamma is a sequence of a different length than partition_ids.
    """
    if gamma is None or np.isscalar(gamma):
      gamma = [gamma] * len(partition_ids)
    elif len(gamma) != len(partition_ids):
      raise ValueError(f"{len(gamma)} gammas for {len(partition_ids)} partition ids")

    nrows = -(-len(partition_ids) // ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(ncols * panel[0], nrows * panel[1]))
    for ax, partition_id, resolution in zip(axes.ravel(), partition_ids, gamma):
      self.draw_partition(partition_id, gamma=resolution, ax=ax, **kwargs)
    for ax in axes.ravel()[len(partition_ids):]:
      ax.set_axis_off()
    fig.tight_layout()
    return fig

  # --- landscape figures -----------------------------------------------------

  def _ordered_modularity(self):
    """Modularity with the partitions laid out in the similarity order.

    Returns:
      A (position, gamma) array.
    """
    return self.modularity[self.order]

  def surface_figure(self, width=950, height=800):
    """Modularity as a surface over partition and resolution.

    Args:
      width: figure width in pixels.
      height: figure height in pixels.

    Returns:
      The plotly figure.
    """
    q = self._ordered_modularity()
    fig = go.Figure(
      go.Surface(
        x=np.arange(self.n_partitions),
        y=self.gammas,
        z=q.T,                       # one row per gamma, one column per position
        colorscale="Viridis",
        colorbar=dict(title="Modularity (Q)"),
      )
    )
    fig.update_layout(
      title="Modularity landscape",
      autosize=False,
      width=width,
      height=height,
      margin=dict(l=65, r=50, b=65, t=90),
      scene=dict(
        xaxis_title="Partition (ordered by similarity)",
        yaxis_title="Resolution (gamma)",
        zaxis_title="Modularity (Q)",
        aspectratio=dict(x=1, y=1, z=0.7),
      ),
    )
    return fig

  def slider_figure(self, n_steps=50, width=950, height=550):
    """Modularity against partition, with a slider over the resolution.

    Args:
      n_steps: number of resolutions the slider steps through.
      width: figure width in pixels.
      height: figure height in pixels.

    Returns:
      The plotly figure.
    """
    q = self._ordered_modularity()
    stride = max(1, len(self.gammas) // n_steps)
    step_indices = np.arange(0, len(self.gammas), stride)
    positions = np.arange(self.n_partitions)
    labels = np.array(
      [[pid, str(self.partitions[pid])] for pid in self.order], dtype=object)
    hover = ("position %{x}, partition %{customdata[0]}"
             "<br>%{customdata[1]}<br>Q = %{y:.4f}<extra></extra>")

    fig = go.Figure()
    for index in step_indices:
      column = q[:, index]
      best = np.flatnonzero(column >= column.max() - 1e-12)   # ties are common: the
                                                             # graph is symmetric, so
                                                             # node 0 can join either
                                                             # triangle
      fig.add_trace(go.Scatter(
        x=positions, y=column, mode="lines", visible=False, showlegend=False,
        line=dict(color="#2a6fdb", width=1), customdata=labels, hovertemplate=hover,
      ))
      fig.add_trace(go.Scatter(
        x=positions[best], y=column[best], mode="markers", visible=False,
        name="best partition", marker=dict(color="crimson", size=10),
        customdata=labels[best], hovertemplate=hover,
      ))
    fig.data[0].visible = True
    fig.data[1].visible = True

    steps = []
    for i, index in enumerate(step_indices):
      visible = [False] * len(fig.data)
      visible[2 * i] = visible[2 * i + 1] = True
      steps.append(dict(method="update", label=f"{self.gammas[index]:.3f}",
                        args=[{"visible": visible}]))

    fig.update_layout(
      title="Modularity across partitions",
      width=width,
      height=height,
      margin=dict(l=65, r=50, b=80, t=90),
      xaxis_title="Partition (ordered by similarity)",
      yaxis_title="Modularity (Q)",
      yaxis_range=[q.min() - 0.02, q.max() + 0.02],
      sliders=[dict(active=0, currentvalue=dict(prefix="gamma = "),
                    pad=dict(t=50), steps=steps)],
    )
    return fig
