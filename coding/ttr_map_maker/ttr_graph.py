"""
A class to represent a graph for the Ticket to Ride game.

Graph consists of nodes (locations) and edges (paths).
Each edge has a length and a color.
"""
from typing import List, Tuple


class TTR_Graph:
  def __init__(self, locations: List[str], paths: List[Tuple[str, str, int, str]]):
    self.nodes = locations
    *self.edges, self.edge_lengths, self.edge_colors = paths

  def __str__(self):
    return f"Graph with {len(self.nodes)} nodes and {len(self.edges)} edges."