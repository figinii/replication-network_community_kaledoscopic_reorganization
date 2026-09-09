"""The Traag two-triangle graph."""

import matplotlib.pyplot as plt
import networkx as nx


class TraagGraph:
  """Two triangles joined through a single bridge node.

  Attributes:
    graph: the undirected networkx graph.
    positions: dict mapping each node to its (x, y) drawing position.
  """

  LEFT_EDGES = ((1, 2), (2, 3), (1, 3))
  RIGHT_EDGES = ((4, 5), (5, 6), (4, 6))
  BRIDGE_EDGES = ((3, 0), (0, 4))

  POSITIONS = {
    1: (-2, 1),
    2: (-2, -1),
    3: (-1, 0),
    0: (0, 0),
    4: (1, 0),
    5: (2, 1),
    6: (2, -1),
  }

  def __init__(self):
    self.graph = nx.Graph()
    self.graph.add_edges_from(self.LEFT_EDGES + self.RIGHT_EDGES + self.BRIDGE_EDGES)
    self.positions = dict(self.POSITIONS)

  @property
  def nodes(self):
    """List of the nodes, in the order they were added."""
    return list(self.graph.nodes())

  def draw(self, ax=None, title="Traag graph: two triangles and a bridge"):
    """Draw the graph at its fixed node positions.

    Args:
      ax: axes to draw on; a new figure is created when None.
      title: title for the axes; None omits it.

    Returns:
      The axes drawn on.
    """
    if ax is None:
      _, ax = plt.subplots(figsize=(8, 4))
    nx.draw(
      self.graph,
      pos=self.positions,
      ax=ax,
      with_labels=True,
      node_color="lightblue",
      node_size=800,
      font_weight="bold",
      edge_color="gray",
      width=2,
    )
    if title:
      ax.set_title(title)
    ax.margins(0.2)
    return ax
