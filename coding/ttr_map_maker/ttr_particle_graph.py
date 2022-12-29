"""
TTR particle graph class. This represents the graph of a Ticket to Ride map as a system of particles for layout optimization.

A particle graph consists of nodes (locations), edges (paths) and node labels.
Each edge has a length and a color. Each node has a label close to it.

A particle graph's layout can be optimized using a simple particle method.
"""
from typing import List, Tuple


class TTR_Particle_Graph:
  def __init__(self,
        locations: List[str],
        paths: List[Tuple[str, str, int, str]]):
    self.nodes = locations
    *self.edges, self.edge_lengths, self.edge_colors = paths

  def create_particle_system(self):
    

  def __str__(self):
      return f"Particle graph with {len(self.nodes)} nodes and {len(self.edges)} edges."