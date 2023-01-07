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
        location_positions: Dict[str, np.ndarray] = None,
        particle_parameters: dict = {
          "velocity_decay": 0.99,
          "edge-edge": 0.01,
          "edge-node": 0.01,
          "node-label": 0.001,
          "node-target": 0.001,
          "node_mass": 1,
          "edge_mass": 1,
          "label_mass": 0.2,
          "interaction_radius": 15,
          "repulsion_strength": 2,
        }):
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
    self.particle_parameters = particle_parameters

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
          position=position,
          mass=self.particle_parameters["node_mass"],
          target_attraction=self.particle_parameters["node-target"],
          interaction_radius=self.particle_parameters["interaction_radius"],
          velocity_decay=self.particle_parameters["velocity_decay"],
          repulsion_strength=self.particle_parameters["repulsion_strength"],
          )
      self.particle_labels[label] = Particle_Label(
          label,
          position=position,
          mass=self.particle_parameters["label_mass"],
          node_attraction=self.particle_parameters["node-label"],
          interaction_radius=self.particle_parameters["interaction_radius"],
          velocity_decay=self.particle_parameters["velocity_decay"],
          repulsion_strength=self.particle_parameters["repulsion_strength"],)
      self.particle_labels[label].position = \
          self.particle_nodes[label].position + np.array([self.particle_labels[label].bounding_box_size[0], 0]) + np.array([1, 0], dtype=np.float64)
      self.particle_labels[label].add_connected_particle(self.particle_nodes[label])
      print(f"node position {label}: {self.particle_nodes[label].position}")
      print(f"label position {label}: {self.particle_labels[label].position}")

    # create edges and add connections between particles
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
            edge_rotation,
            mass=self.particle_parameters["edge_mass"],
            node_attraction=self.particle_parameters["edge-node"],
            edge_attraction=self.particle_parameters["edge-edge"],
            interaction_radius=self.particle_parameters["interaction_radius"],
            velocity_decay=self.particle_parameters["velocity_decay"],
            repulsion_strength=self.particle_parameters["repulsion_strength"],)
        edge_particle.add_connected_particle(last_particle)
        if i >= 1:
          last_particle.add_connected_particle(edge_particle)
        self.particle_edges[(location_1, location_2, i)] = edge_particle
        last_particle = edge_particle
      last_particle.add_connected_particle(node_2)


  def optimize_layout(self,
      iterations: int = 1000,
      dt: float = 0.02):
    """
    optimize layout of particle graph by calling the interact() and update() methods of each particle.
    Use Cell lists and Verlet lists to speed up the computation.

    Args:
        iterations (int, optional): number of iterations to perform. Defaults to 1000.
        dt (float, optional): timestep. Defaults to 0.02.
    """
    all_particles = list(self.particle_nodes.values()) + list(self.particle_labels.values()) + list(self.particle_edges.values())
    cell_list = dict()
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
      particle_edge.draw(ax, color=particle_edge.color, alpha=0.8 * alpha_multiplier)

  def erase(self):
    """
    erase particle graph
    """
    for particle_node in self.particle_nodes.values():
      particle_node.erase()
    for particle_label in self.particle_labels.values():
      particle_label.erase()
    for particle_edge in self.particle_edges.values():
      particle_edge.erase()

  def draw_nodes(self, ax: plt.Axes, alpha_multiplier: float = 1.0):
    """draw nodes of particle graph

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha_multiplier (float, optional): transparency multiplier. Defaults to 1.0.
    """
    for particle_node in self.particle_nodes.values():
      particle_node.draw(ax, color="#222222", alpha=0.7 * alpha_multiplier)

  def erase_nodes(self):
    """
    erase nodes of particle graph
    """
    for particle_node in self.particle_nodes.values():
      particle_node.erase()

  def draw_labels(self, ax: plt.Axes, alpha_multiplier: float = 1.0):
    """draw labels of particle graph

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha_multiplier (float, optional): transparency multiplier. Defaults to 1.0.
    """
    for particle_label in self.particle_labels.values():
      particle_label.draw(ax, color="#222222", alpha=1.0 * alpha_multiplier)

  def erase_labels(self):
    """
    erase labels of particle graph
    """
    for particle_label in self.particle_labels.values():
      particle_label.erase()

  def draw_edges(self, ax: plt.Axes, alpha_multiplier: float = 1.0):
    """draw edges of particle graph

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha_multiplier (float, optional): transparency multiplier. Defaults to 1.0.
    """
    for particle_edge in self.particle_edges.values():
      particle_edge.draw(ax, color=particle_edge.color, alpha=0.8 * alpha_multiplier)

  def erase_edges(self):
    """
    erase edges of particle graph
    """
    for particle_edge in self.particle_edges.values():
      particle_edge.erase()

  def draw_connections(self, ax: plt.Axes, alpha_multiplier: float = 1.0):
    """
    draw arrows between all particles that are connected to each other.
    """
    all_particles = list(self.particle_nodes.values()) + list(self.particle_labels.values()) + list(self.particle_edges.values())
    for particle in all_particles:
      for connected_particle in particle.connected_particles:
        ax.arrow(particle.position[0], particle.position[1],
            connected_particle.position[0] - particle.position[0],
            connected_particle.position[1] - particle.position[1],
            color="#222222",
            alpha=alpha_multiplier,
            width=0.1,
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            zorder=0)

  def draw_edge_attractors(self, ax: plt.Axes, alpha_multiplier: float = 1.0):
    """
    draw circles around each edge particle that shows the area where other particles are attracted to it.
    """
    for particle_edge in self.particle_edges.values():
      for connected_particle in particle_edge.connected_particles:
        _, anchor_1 = particle_edge.get_attraction_forces(connected_particle)
        _, anchor_2 = connected_particle.get_attraction_forces(particle_edge)
        ax.arrow(anchor_1[0], anchor_1[1],
            anchor_2[0] - anchor_1[0],
            anchor_2[1] - anchor_1[1],
            color="#222222",
            alpha=alpha_multiplier,
            width=0.1,
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            zorder=0)

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
    "Menegroth": np.array([0, 6], dtype=np.float64),
    "Nargothrond": np.array([-6, -2], dtype=np.float64),
    "Hithlum": np.array([4, -5], dtype=np.float64),
  }

  paths = [
    ("Menegroth", "Nargothrond", 2, "#dd0000"),
    ("Menegroth", "Hithlum", 3, "#5588ff"),
    ("Nargothrond", "Hithlum", 2, "#22dd22"),
  ]

  particle_graph = TTR_Particle_Graph(locations, paths, location_positions)

  n_iter = 100
  dt = 0.1
  print_at = 0

  fig, ax = plt.subplots()
  particle_graph.draw(ax, alpha_multiplier=0.1)
  # particle_graph.optimize_layout(iterations=print_at, dt=dt)
  # particle_graph.draw(ax, alpha_multiplier=0.5)
  particle_graph.optimize_layout(iterations=n_iter-print_at, dt=dt)
  particle_graph.draw(ax, alpha_multiplier=1.0)
  # particle_graph.draw_connections(ax, alpha_multiplier=0.5)
  particle_graph.draw_edge_attractors(ax, alpha_multiplier=0.5)
  
  ax.set_xlim(-15, 15)
  ax.set_ylim(-10, 10)
  ax.legend()
  ax.set_aspect("equal")
  plt.grid(color="#dddddd", linestyle="--", linewidth=1)
  plt.show()