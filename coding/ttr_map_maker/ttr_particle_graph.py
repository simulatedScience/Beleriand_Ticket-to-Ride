"""
TTR particle graph class. This represents the graph of a Ticket to Ride map as a system of particles for layout optimization.

A particle graph consists of nodes (locations), edges (paths) and node labels.
Each edge has a length and a color. Each node has a label close to it.

A particle graph's layout can be optimized using a simple particle method.
"""
import pickle
import json
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from graph_particle import Graph_Particle
from particle_node import Particle_Node
from particle_label import Particle_Label
from particle_edge import Particle_Edge


class TTR_Particle_Graph:
  def __init__(self,
        locations: List[str],
        paths: List[Tuple[str, str, int, str]],
        node_positions: Dict[str, np.ndarray] = None,
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
    self.paths = paths
    self.node_positions = node_positions
    self.particle_parameters = particle_parameters

    self.particle_nodes = dict()
    self.particle_edges = dict()
    self.particle_labels = dict()
    
    self.create_particle_system()


  def create_particle_system(self):
    """
    Create the particle system from the given locations and paths connecting them.
    """
    for i, label in enumerate(self.node_labels):
      if self.node_positions is not None:
        position = self.node_positions[label]
      else:
        position = np.array([3*i, 0], dtype=np.float64)
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
          self.particle_nodes[label].position + np.array([self.particle_labels[label].bounding_box_size[0] / 2, 0]) + np.array([1, 0], dtype=np.float64)
      self.particle_labels[label].add_connected_particle(self.particle_nodes[label])
      # print(f"node position {label}: {self.particle_nodes[label].position}")
      # print(f"label position {label}: {self.particle_labels[label].position}")

    # create edges and add connections between particles
    for (location_1, location_2, length, color) in self.paths:
      node_1: Graph_Particle = self.particle_nodes[location_1]
      node_2: Graph_Particle = self.particle_nodes[location_2]
      last_particle: Graph_Particle = node_1 # connect the first edge particle to the first node
      # create `length` edge particles between `node_1` and `node_2``
      for i in range(length):
        edge_position: np.ndarray = node_1.position + (node_2.position - node_1.position) * (i+1) / (length+1)
        edge_rotation: float = -np.arctan2(node_2.position[1] - node_1.position[1], node_2.position[0] - node_1.position[0])
        edge_particle: Graph_Particle = Particle_Edge(
            color=color,
            location_1_name=location_1,
            location_2_name=location_2,
            path_index=i,
            position=edge_position,
            rotation=edge_rotation,
            mass=self.particle_parameters["edge_mass"],
            node_attraction=self.particle_parameters["edge-node"],
            edge_attraction=self.particle_parameters["edge-edge"],
            interaction_radius=self.particle_parameters["interaction_radius"],
            velocity_decay=self.particle_parameters["velocity_decay"],
            repulsion_strength=self.particle_parameters["repulsion_strength"],)
        edge_particle.add_connected_particle(last_particle)
        if i >= 1: # if not the first edge particle, add connection to previous edge particle
          last_particle.add_connected_particle(edge_particle)
        self.particle_edges[(location_1, location_2, i)] = edge_particle
        last_particle = edge_particle
      # connect last edge particle to node_2
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
    all_particles = self.get_particle_list()
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


  def move_labels_to_nodes(self,
      ax: plt.Axes,
      x_offset: float = 0.,
      y_offset: float = 2.) -> None:
    """
    move all labels to the position of their connected node with an offset. Then erase and redraw the labels.

    Args:
        ax (plt.Axes): axes to draw on
        x_offset (float, optional): x offset. Defaults to 0..
        y_offset (float, optional): y offset. Defaults to 2.
    """
    for particle_label in self.particle_labels.values():
      particle_label.position = particle_label.connected_particles[0].position + np.array([x_offset, y_offset], dtype=np.float64)
      particle_label.erase()
      particle_label.draw(ax)


  def move_edges_to_nodes(self, ax: plt.Axes, **draw_kwargs) -> None:
    """
    move all edges such that tey form straight lines between their connected nodes. Then erase and redraw the edges.

    Args:
        ax (plt.Axes): axes to draw on
        draw_kwargs (dict): kwargs to pass to the draw method of the edge particles
    """
    for path in self.paths:
      location_1 = path[0]
      location_2 = path[1]
      length = path[2]
      node_1 = self.particle_nodes[location_1]
      node_2 = self.particle_nodes[location_2]
      edge_particles = [self.particle_edges[(location_1, location_2, i)] for i in range(length)]
      for i, particle in enumerate(edge_particles):
        particle.set_position(
          node_1.position + (node_2.position - node_1.position) * (i+1) / (length+1)
        )
        particle.set_rotation(
          -np.arctan2(node_2.position[1] - node_1.position[1], node_2.position[0] - node_1.position[0])
        )
        particle.erase()
        particle.draw(ax, **draw_kwargs)


  def scale_node_positions(self, ax, scale_factor: float = 0.8):
    """
    scale the positions of all nodes by a scale factor

    Args:
        scale_factor (float, optional): scale factor. Defaults to 0.8.
    """
    for particle_node in self.particle_nodes.values():
      particle_node.set_position(
        particle_node.position*scale_factor)
      particle_node.erase()
      particle_node.draw(ax)


  def get_locations(self) -> list:
    """
    get all location labels in particle graph

    Returns:
        list: list of location labels
    """
    return self.particle_nodes

  def get_paths(self) -> list:
    """
    get all paths in particle graph

    Returns:
        list: list of paths
    """
    return self.paths


  def get_particle_list(self):
    """
    get all particles in particle graph

    Returns:
        [type]: [description]
    """
    return list(self.particle_nodes.values()) + list(self.particle_labels.values()) + list(self.particle_edges.values())


  def set_parameters(self, particle_parameters: dict):
    """
    update all given particle parameters
    These changes are applied to all particles in the particle graph.

    Args:
        particle_parameters (dict): dictionary of particle parameters. See __init__ for allowed parameters and details.
    """
    for particle in self.get_particle_list():
      particle.set_parameters(particle_parameters)


  def get_edge_colors(self) -> List[str]:
    """
    return a list of all edge colors occuring in the graph.
    edge border colors are not included.

    Returns:
        List[str]: list of edge colors
    """
    edge_colors = set()
    for particle_edge in self.particle_edges.values():
      edge_colors.add(particle_edge.color)
    return list(edge_colors)

  def set_edge_colors(self, edge_color_map: dict):
    """
    set the colors of all edges in the graph according to a color map.
    The color map is a dictionary mapping current edge colors to new colors.
    To ensure the correct dict keys are used, use the get_edge_colors method.

    Args:
        edge_color_map (dict): dictionary mapping edge colors to new colors
    """
    for particle_edge in self.particle_edges.values():
      if particle_edge.color in edge_color_map.keys():
        particle_edge.color = edge_color_map[particle_edge.color]
        particle_edge.set_image(None)

  def set_edge_images(self, edge_color_map: dict):
    """
    set edges to display images instead of flat colored rectangles.
    The color map is a dictionary mapping current edge colors to image file paths.
    Each color corresponds to one image. Different images for edges of the same color are not supported.

    Args:
        edge_color_map (dict): dictionary mapping edge colors to image file paths
    """
    for particle_edge in self.particle_edges.values():
      try:
        particle_edge.set_image(edge_color_map[particle_edge.color])
      except KeyError:
        raise ValueError(f"no image file path specified for edge color '{particle_edge.color}'")


  def draw(self, ax: plt.Axes, alpha_multiplier: float = 1.0):
    """
    draw particle graph

    Args:
        ax (plt.Axes): axes to draw on
    """
    for particle_node in self.particle_nodes.values():
      particle_node.draw(ax, color="#222222", alpha=0.7 * alpha_multiplier)
    for particle_label in self.particle_labels.values():
      particle_label.draw(ax, color="#222222", alpha=1.0 * alpha_multiplier)
    for particle_edge in self.particle_edges.values():
      particle_edge.draw(ax, color=particle_edge.color, border_color="#555555", alpha=0.8 * alpha_multiplier)

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
    all_particles = self.get_particle_list()
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

  # def __str__(self): # TODO
  #     return f"Particle graph with {len(self.node_labels)} nodes and {len(self.edges)} edges."


  def to_json(self) -> str:
    """
    converts a particle graph `self` into a JSON string containing info about the graph itself as well as all the particles it contains.
    Particles are saved in three separate arrays:
    - `nodes`: contains all node particles
    - `edges`: contains all edge particles
    - `labels`: contains all label particles

    Returns:
        str: JSON string representing the particle graph
    """
    all_particles = self.get_particle_list()
    # add id's to all particles
    particle_id = 0
    for particle_node in all_particles:
      try:
        particle_node.set_id(particle_id)
      except AttributeError:
        particle_node.particle_id = particle_id
      particle_id += 1
    # get json string for all particles
    all_particles_json = []
    for particle in all_particles:
      all_particles_json.append(particle.to_dict())

    particle_graph = {
        "particle_graph": {
            "n_nodes": len(self.particle_nodes),
            "n_edges": len(self.particle_edges),
            "n_labels": len(self.particle_labels),
            "particles": all_particles_json,
            "particle_parameters": self.particle_parameters,
        }
    }
    return json.dumps(particle_graph, indent=2)

  def save_json(self, filepath: str) -> None:
    """
    Save particle graph as JSON file.

    Args:
        filepath (str): filepath to save particle graph to. If the filepath does not end with `.json`, the extension will be added automatically.
    """
    if not filepath.endswith(".json"):
      filepath += ".json"
    with open(filepath, "w", encoding="utf-8") as file:
      file.write(self.to_json())


  def add_particle(self, particle: Graph_Particle) -> None:
    """
    add a particle to the particle graph.

    Args:
        particle (Graph_Particle): particle to add
    """
    if isinstance(particle, Particle_Node):
      self.particle_nodes[particle.label] = particle
    elif isinstance(particle, Particle_Edge):
      loc_1 = particle.location_1_name
      loc_2 = particle.location_2_name
      path_index = particle.path_index
      self.particle_edges[(loc_1, loc_2, path_index)] = particle
    elif isinstance(particle, Particle_Label):
      self.particle_labels[particle.label] = particle

  def build_paths(self) -> None:
    """
    calculate a list of all paths in the particle graph as tuples (location_1, location_2, length, color) from the list of edges
    """
    locations_to_lengths = dict()
    for (loc_1, loc_2, min_path_length), edge in self.particle_edges.items():
      temp_key = (loc_1, loc_2, edge.color)
      if not temp_key in locations_to_lengths:
        locations_to_lengths[temp_key] = min_path_length
        continue
      if min_path_length > locations_to_lengths[temp_key]:
        locations_to_lengths[temp_key] = min_path_length
    self.paths = []
    for (loc_1, loc_2, color), length in locations_to_lengths.items():
      self.paths.append((loc_1, loc_2, length+1, color))

  @staticmethod
  def load_json(filepath: str) -> None:
    """
    create a particle graph from a JSON string.

    Args:
        filepath (str): filepath to JSON file containing info for a particle graph

    Returns:
        TTR_Particle_Graph: particle graph
    """
    # read JSON string from file
    with open(filepath, "r", encoding="utf-8") as file:
      json_string = file.read()
    # load JSON string into python dictionary
    graph_info = json.loads(json_string)
    particle_dicts = graph_info["particle_graph"]["particles"]
    if len(particle_dicts) == 0:
      print("Warning: no particles found in JSON")
      return None
    particle_parameters = graph_info["particle_graph"]["particle_parameters"]
    particle_graph = TTR_Particle_Graph(
        locations = [],
        paths = [],
        node_positions = [],
        particle_parameters = particle_parameters,
      )
    particle_list = []
    for particle_info in particle_dicts:
      connected_particles = particle_info.pop("connected_particles")
      particle_type = particle_info.pop("particle_type")
      id = particle_info.pop("id")
      
      particle_info["position"] = np.array(particle_info["position"])
      particle_info["bounding_box_size"] = tuple(particle_info["bounding_box_size"])
      if particle_type == "Particle_Node":
        particle_info.pop("rotation")
        particle_info.pop("angular_velocity_decay")
        particle_info["target_position"] = np.array(particle_info["target_position"])
        particle = Particle_Node(**particle_info)
        particle_graph.add_particle(particle)
      elif particle_type == "Particle_Edge":
        particle_info.pop("target_position")
        particle = Particle_Edge(**particle_info)
        particle_graph.add_particle(particle)
      elif particle_type == "Particle_Label":
        particle_info.pop("bounding_box_size")
        particle_info.pop("target_position")
        particle = Particle_Label(**particle_info)
        particle_graph.add_particle(particle)
      # add connected particles back to particle info
      particle_info["connected_particles"] = connected_particles
      particle_list.append(particle)
    # add connections to particles
    for particle, particle_info in zip(particle_list, particle_dicts):
      for connected_particle_id in particle_info["connected_particles"]:
        particle.add_connected_particle(particle_list[connected_particle_id])
    # build list of paths in graph
    particle_graph.build_paths()
    return particle_graph


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

  fig, ax = plt.subplots(dpi=300)
  # particle_graph.draw(ax, alpha_multiplier=0.1)
  # particle_graph.optimize_layout(iterations=print_at, dt=dt)
  # particle_graph.draw(ax, alpha_multiplier=0.5)
  # particle_graph.optimize_layout(iterations=n_iter-print_at, dt=dt)
  particle_graph.draw(ax, alpha_multiplier=1.0)
  # particle_graph.draw_connections(ax, alpha_multiplier=0.5)
  particle_graph.draw_edge_attractors(ax, alpha_multiplier=0.5)

  
  particle_graph.save_json("test_particle_graph.json")

  ax.set_xlim(-15, 15)
  ax.set_ylim(-10, 10)
  ax.legend()
  ax.set_aspect("equal")
  plt.grid(color="#dddddd", linestyle="--", linewidth=1)
  plt.show()

  recovered_graph = TTR_Particle_Graph.load_json("test_particle_graph.json")

  fig, ax = plt.subplots(dpi=100)
  recovered_graph.draw(ax, alpha_multiplier=1.0)
  ax.set_xlim(-15, 15)
  ax.set_ylim(-10, 10)
  ax.set_title("recovered graph")
  ax.set_aspect("equal")
  plt.grid(color="#dddddd", linestyle="--", linewidth=1)
  plt.show()