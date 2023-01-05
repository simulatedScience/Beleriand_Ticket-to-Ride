"""
TTR particle graph class. This represents the graph of a Ticket to Ride map as a system of particles for layout optimization.

A particle graph consists of nodes (locations), edges (paths) and node labels.
Each edge has a length and a color. Each node has a label close to it.

A particle graph's layout can be optimized using a simple particle method.
"""
import pickle
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from particle_node import Particle_Node
from particle_label import Particle_Label
from particle_edge import Particle_Edge


class TTR_Particle_Graph:
  def __init__(self,
        locations: List[str],
        paths: List[Tuple[str, str, int, str]],
        location_positions: Dict[str, np.ndarray] = None):
    """
    Initialize a TTR particle graph made from labeled locations and paths.
    Each path is made from several rectangular edge particles.

    Args:
        locations (List[str]): list of location labels
        paths (List[Tuple[str, str, int, str]]): list of paths. Each path is a tuple of (node_1, node_2, length, color)
        location_positions (Dict[str, np.ndarray], optional): dictionary of location positions. Keys are location labels, values are 2D numpy arrays representing the position of the location. Defaults to None.
    """
    self.node_labels = locations
    *self.edges, self.edge_lengths, self.edge_colors = list(zip(*paths))
    self.edges = list(zip(*self.edges))
    self.location_positions = location_positions

    self.particle_nodes = dict()
    self.particle_edges = dict()
    self.particle_labels = dict()
    
    self.create_particle_system()


  def create_particle_system(self):
    for i, label in enumerate(self.node_labels):
      if self.location_positions is not None:
        position = self.location_positions[label]
      else:
        position = np.array([5*i, 0], dtype=np.float64)
      self.particle_nodes[label] = Particle_Node(
          label,
          position=position)
      self.particle_labels[label] = Particle_Label(
          label,
          position=position + np.array([0, 2], dtype=np.float64))

    for ((location_1, location_2), length, color) in zip(self.edges, self.edge_lengths, self.edge_colors):
      node_1 = self.particle_nodes[location_1]
      node_2 = self.particle_nodes[location_2]
      last_particle = node_1
      
      for i in range(length):
        edge_position = node_1.position + (node_2.position - node_1.position) * (i+1) / (length+1)
        edge_rotation = -np.arctan2(node_2.position[1] - node_1.position[1], node_2.position[0] - node_1.position[0])
        edge_particle = Particle_Edge(
            color,
            edge_position,
            edge_rotation)
        edge_particle.add_connected_particle(last_particle)
        if i >= 1:
          last_particle.add_connected_particle(edge_particle)
        self.particle_edges[(location_1, location_2, i)] = edge_particle
        last_particle = edge_particle
      last_particle.add_connected_particle(node_2)


  def optimize_layout(self, iterations: int = 1000, dt: float = 0.02):
    """
    optimize layout of particle graph by calling the interact() and update() methods of each particle.
    Use Cell lists and Verlet lists to speed up the computation.

    Args:
        iterations (int, optional): number of iterations to perform. Defaults to 1000.
        dt (float, optional): timestep. Defaults to 0.02.
    """
    all_particles = list(self.particle_nodes.values()) + list(self.particle_labels.values()) + list(self.particle_edges.values())
    for i in range(iterations):
      for particle in all_particles:
        particle.reset_acceleration()

      for particle_1 in all_particles:
        for particle_2 in all_particles:
          if particle_1 != particle_2:
            particle_1.interact(particle_2)

      for particle in all_particles:
        particle.update(dt)


  def draw(self, ax: plt.Axes, alpha_multiplier: float = 1.0):
    """draw particle graph

    Args:
        ax (_type_): _description_
    """
    for particle_node in self.particle_nodes.values():
      particle_node.draw(ax, color="#222222", alpha=0.7 * alpha_multiplier)
    for particle_label in self.particle_labels.values():
      particle_label.draw(ax, color="#222222", alpha=1.0 * alpha_multiplier)
    for particle_edge in self.particle_edges.values():
      particle_edge.draw(ax, color=particle_edge.color, alpha=0.7 * alpha_multiplier)


  def __str__(self):
      return f"Particle graph with {len(self.node_labels)} nodes and {len(self.edges)} edges."


  def save(self, filename: str):
    """
    save particle graph as a file using pickle

    Args:
        filename (str): filename to save particle graph to
    """
    with open(filename, "wb") as file:
      pickle.dump(self, file)


if __name__ == "__main__":
  locations = [
      "Menegroth",
      "Nargothrond",
      "Hithlum",
  ]
  location_positions = {
    "Menegroth": np.array([0, 5], dtype=np.float64),
    "Nargothrond": np.array([-4, -2], dtype=np.float64),
    "Hithlum": np.array([4, -4], dtype=np.float64),
  }

  paths = [
    ("Menegroth", "Nargothrond", 2, "#dd0000"),
    ("Menegroth", "Hithlum", 3, "#5588ff"),
    ("Nargothrond", "Hithlum", 2, "#22dd22"),
  ]

  particle_graph = TTR_Particle_Graph(locations, paths, location_positions)

  fig, ax = plt.subplots()
  particle_graph.draw(ax, alpha_multiplier=0.3)
  particle_graph.optimize_layout(iterations=5000, dt=0.1)
  particle_graph.draw(ax, alpha_multiplier=1.0)
  
  ax.set_xlim(-30, 30)
  ax.set_ylim(-10, 10)
  ax.legend()
  plt.grid(color="#dddddd", linestyle="--", linewidth=1)
  plt.show()