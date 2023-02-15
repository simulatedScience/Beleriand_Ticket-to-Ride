"""
TTR particle graph class. This represents the graph of a Ticket to Ride map as a system of particles for layout optimization.

A particle graph consists of nodes (locations), edges (paths) and node labels.
Each edge has a length and a color. Each node has a label close to it.

A particle graph's layout can be optimized using a simple particle method.
"""
import json
from typing import List, Tuple, Dict
from math import isinf

import numpy as np
import matplotlib.pyplot as plt

from graph_particle import Graph_Particle
from particle_node import Particle_Node
from particle_label import Particle_Label
from particle_edge import Particle_Edge
from graph_analysis import TTR_Graph_Analysis


class TTR_Particle_Graph:
  def __init__(self,
        locations: List[str],
        paths: List[Tuple[str, str, int, str]],
        tasks: List[Tuple[str, str]],
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
        tasks (List[Tuple[str, str]]): list of tasks. Each task is a tuple of (node_1, node_2)
        node_positions (Dict[str, np.ndarray], optional): dictionary of location positions. Keys are location labels, values are 2D numpy arrays representing the position of the location. Defaults to None.
        location_positions (Dict[str, np.ndarray], optional): dictionary of location positions. Keys are location labels, values are 2D numpy arrays representing the position of the location. Defaults to None.
    """
    self.paths: List[Tuple[str, str, int, str]] = paths
    self.tasks: List[Tuple[str, str]] = tasks
    self.particle_parameters: dict = particle_parameters
    self.node_labels: List[str] = locations # deleted after graph creation
    self.node_positions: Dict[str, np.ndarray] = node_positions # deleted after graph creation

    self.max_particle_id = 0

    self.particle_nodes: dict[str, Particle_Node] = dict()
    self.particle_edges: dict[Tuple[str, str, int, int], Particle_Edge] = dict()
    self.particle_labels: dict[str, Particle_Label] = dict()
    self.analysis_graph: TTR_Graph_Analysis = None

    self.edge_attractor_artists: List[plt.Artist] = list()
    
    self.create_particle_system()


  def create_particle_system(self) -> None:
    """
    Create the particle system from the given locations and paths connecting them.
    """
    particle_id: int = self.max_particle_id + 1
    self.label_height_scale: float = Particle_Label.get_label_height_scale()
    n_nodes: int = len(self.node_labels)
    for path_index, label in enumerate(self.node_labels):
      if self.node_positions is not None:
        position = self.node_positions[label]
      else:
        position = np.array([3*path_index, 0], dtype=np.float64)
      self.particle_nodes[label] = Particle_Node(
          label,
          id=particle_id,
          position=position,
          mass=self.particle_parameters["node_mass"],
          target_attraction=self.particle_parameters["node-target"],
          interaction_radius=self.particle_parameters["interaction_radius"],
          velocity_decay=self.particle_parameters["velocity_decay"],
          repulsion_strength=self.particle_parameters["repulsion_strength"],
          )
      self.particle_labels[label] = Particle_Label(
          label,
          id=particle_id+n_nodes,
          position=position,
          mass=self.particle_parameters["label_mass"],
          node_attraction=self.particle_parameters["node-label"],
          interaction_radius=self.particle_parameters["interaction_radius"],
          velocity_decay=self.particle_parameters["velocity_decay"],
          repulsion_strength=self.particle_parameters["repulsion_strength"],
          height_scale_factor=self.label_height_scale,
          )
      self.particle_labels[label].position = \
          self.particle_nodes[label].position + np.array([self.particle_labels[label].bounding_box_size[0] / 2, 0]) + np.array([1, 0], dtype=np.float64)
      self.particle_labels[label].add_connected_particle(self.particle_nodes[label])
      particle_id += 1
      # print(f"node position {label}: {self.particle_nodes[label].position}")
      # print(f"label position {label}: {self.particle_labels[label].position}")

    # create edges and add connections between particles
    for (location_1, location_2, length, color) in self.paths:
      self.add_connection(location_1, location_2, length, color)
    
    del self.node_positions
    del self.node_labels


  def optimize_layout(self,
      iterations: int = 1000,
      dt: float = 0.02) -> None:
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
      y_offset: float = 2.,
      movable: bool = False) -> None:
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
      particle_label.draw(ax, movable=movable)


  def straighten_connections(self,
      ax: plt.Axes,
      movable: bool = False,
      edge_border_color: str = None,
      alpha = 0.8,
      **draw_kwargs) -> None:
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
  
  def straighten_connection(self,
      location_1: str,
      location_2: str,
      connection_index: int,
      ax: plt.Axes,
      movable: bool = False,
      edge_border_color: str = None,
      alpha = 0.8,
      **draw_kwargs) -> None:
    """
    move an edge such that it forms a straight line between its connected nodes. Then erase and redraw the edge.

    Args:
        location_1 (str): label of the first node
        location_2 (str): label of the second node
        connection_index (int): index of the connection
        ax (plt.Axes): axes to draw on
        draw_kwargs (dict): kwargs to pass to the draw method of the edge particle
    """
    node_1 = self.particle_nodes[location_1]
    node_2 = self.particle_nodes[location_2]
    edge_particles: List[Particle_Edge] = []
    length = 0
    while True:
      if (location_1, location_2, length, connection_index) in self.particle_edges:
        edge_particles.append(self.particle_edges[(location_1, location_2, length, connection_index)])
        length += 1
      else:
        break
    for i, edge_particle in enumerate(edge_particles):
      edge_particle.set_position(
        node_1.position + (node_2.position - node_1.position) * (i+1) / (length+1)
      )
      edge_particle.set_rotation(
        np.arctan2(node_2.position[1] - node_1.position[1], node_2.position[0] - node_1.position[0])
      )
      edge_particle.erase()
      edge_particle.draw(
          ax,
          movable=movable,
          border_color=edge_border_color,
          alpha=alpha,
          **draw_kwargs)


  def scale_graph_positions(self, ax: plt.Axes, scale_factor: float = 0.8) -> None:
    """
    scale the positions of all graph particles by a given factor. Then redraw the particles.

    Args:
        ax (plt.Axes): axes to draw on
        scale_factor (float, optional): scale factor. Defaults to 0.8.
    """
    for particle_node in self.get_particle_list():
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
    return list(self.particle_nodes.keys())

  def get_paths(self) -> list:
    """
    get all paths in particle graph

    Returns:
        list: list of paths
    """
    return self.paths

  def get_particle_list(self) -> List[Graph_Particle]:
    """
    get all particles in particle graph

    Returns:
        [type]: [description]
    """
    return list(self.particle_nodes.values()) + list(self.particle_labels.values()) + list(self.particle_edges.values())


  def set_parameters(self, particle_parameters: dict) -> None:
    """
    update all given particle parameters
    These changes are applied to all particles in the particle graph.

    Args:
        particle_parameters (dict): dictionary of particle parameters. See __init__ for allowed parameters and details.
    """
    for particle in self.get_particle_list():
      particle.set_parameters(particle_parameters)

  def update_tasks(self, task_list: List[Tuple[str, str]]) -> None:
    """
    update the list of tasks saved in the particle graph.
    The previous list of tasks is overwritten.

    Args:
        task_list (List[str, str]): list of tasks. Each task is a tuple of the form (location_1, location_2)
    """
    self.tasks = task_list

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

  def set_edge_colors(self, edge_color_map: dict) -> None:
    """
    set the colors of all edges in the graph according to a color map.
    The color map is a dictionary mapping current edge colors to new colors.
    To ensure the correct dict keys are used, use the get_edge_colors method.

    Args:
        edge_color_map (dict): dictionary mapping edge colors to new colors
    """
    self.analysis_graph = None
    for particle_edge in self.particle_edges.values():
      if particle_edge.color in edge_color_map.keys():
        particle_edge.color = edge_color_map[particle_edge.color]
        particle_edge.set_image_file_path(None)

  def set_edge_images(self, edge_color_map: dict) -> None:
    """
    set edges to display images instead of flat colored rectangles.
    The color map is a dictionary mapping current edge colors to image file paths.
    Each color corresponds to one image. Different images for edges of the same color are not supported.

    Args:
        edge_color_map (dict): dictionary mapping edge colors to image file paths
    """
    for particle_edge in self.particle_edges.values():
      try:
        particle_edge.set_image_file_path(edge_color_map[particle_edge.color])
      except KeyError:
        raise ValueError(f"no image file path specified for edge color '{particle_edge.color}'")

  def toggle_move_nodes(self, move_nodes: bool = None) -> None:
    """
    toggle the ability to move nodes in the graph.
    If `move_nodes` is None, the current state is toggled.
    If `move_nodes` is True or False, the state is set to the given value.

    Args:
        move_nodes (bool, optional): new state of node movement. Defaults to None (toggle).
    """
    print(f"toggle movability to {move_nodes} for {len(self.particle_nodes.values())} ndoes")
    for particle_node in self.particle_nodes.values():
      particle_node.set_particle_movable(move_nodes)

  def toggle_move_labels(self, move_labels: bool = None) -> None:
    """
    toggle the ability to move labels in the graph.
    If `move_labels` is None, the current state is toggled.
    If `move_labels` is True or False, the state is set to the given value.

    Args:
        move_labels (bool, optional): new state of label movement. Defaults to None (toggle).
    """
    for particle_label in self.particle_labels.values():
      particle_label.set_particle_movable(move_labels)

  def toggle_move_edges(self, move_edges: bool = None) -> None:
    """
    toggle the ability to move edges in the graph.
    If `move_edges` is None, the current state is toggled.
    If `move_edges` is True or False, the state is set to the given value.

    Args:
        move_edges (bool, optional): new state of edge movement. Defaults to None (toggle).
    """
    for particle_edge in self.particle_edges.values():
      particle_edge.set_particle_movable(move_edges)

  def draw(self, ax: plt.Axes, alpha_multiplier: float = 1.0, movable: bool = False, edge_border_color: str = "#555555") -> None:
    """
    draw particle graph

    Args:
        ax (plt.Axes): axes to draw on
    """
    for particle_edge in self.particle_edges.values():
      particle_edge.draw(ax, color=particle_edge.color, border_color=edge_border_color, alpha=0.8 * alpha_multiplier, movable=movable)
    for particle_node in self.particle_nodes.values():
      particle_node.draw(ax, color="#222222", alpha=0.8 * alpha_multiplier, movable=movable)
    for particle_label in self.particle_labels.values():
      particle_label.draw(ax, color="#222222", alpha=1.0 * alpha_multiplier, movable=movable)

  def erase(self) -> None:
    """
    erase particle graph
    """
    for particle_node in self.particle_nodes.values():
      particle_node.erase()
    for particle_label in self.particle_labels.values():
      particle_label.erase()
    for particle_edge in self.particle_edges.values():
      particle_edge.erase()

  def draw_nodes(self, ax: plt.Axes, alpha: float = 0.8, movable: bool = False) -> None:
    """draw nodes of particle graph

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency multiplier. Defaults to 1.0.
    """
    for particle_node in self.particle_nodes.values():
      particle_node.draw(ax, color="#222222", alpha=alpha, movable=movable)

  def erase_nodes(self) -> None:
    """
    erase nodes of particle graph
    """
    for particle_node in self.particle_nodes.values():
      particle_node.erase()

  def draw_labels(self, ax: plt.Axes, alpha: float = 1.0, movable: bool = False) -> None:
    """draw labels of particle graph

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency multiplier. Defaults to 1.0.
    """
    for particle_label in self.particle_labels.values():
      particle_label.draw(ax, color="#222222", alpha=alpha, movable=movable)

  def erase_labels(self) -> None:
    """
    erase labels of particle graph
    """
    for particle_label in self.particle_labels.values():
      particle_label.erase()

  def draw_edges(self, ax: plt.Axes, alpha: float = 0.8, movable: bool = False, border_color: str = "#555555") -> None:
    """draw edges of particle graph

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency. Defaults to 0.8.
    """
    for particle_edge in self.particle_edges.values():
      particle_edge.draw(ax, color=particle_edge.color, alpha=alpha, movable=movable, border_color=border_color)

  def erase_edges(self) -> None:
    """
    erase edges of particle graph
    """
    for particle_edge in self.particle_edges.values():
      particle_edge.erase()

  def draw_connections(self, ax: plt.Axes, alpha: float = 1.0) -> None:
    """
    draw arrows between all particles that are connected to each other.

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency. Defaults to 1.0.
    """
    all_particles = self.get_particle_list()
    for particle in all_particles:
      for connected_particle in particle.connected_particles:
        ax.arrow(particle.position[0], particle.position[1],
            connected_particle.position[0] - particle.position[0],
            connected_particle.position[1] - particle.position[1],
            color="#222222",
            alpha=alpha,
            width=0.1,
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            zorder=0)

  def draw_edge_attractors(self, ax: plt.Axes, alpha: float = 1.0) -> None:
    """
    draw circles around each edge particle that shows the area where other particles are attracted to it.

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency. Defaults to 1.0.
    """
    for particle_edge in self.particle_edges.values():
      for connected_particle in particle_edge.connected_particles:
        _, anchor_1 = particle_edge.get_attraction_forces(connected_particle)
        _, anchor_2 = connected_particle.get_attraction_forces(particle_edge)
        self.edge_attractor_artists.append(ax.arrow(
            anchor_1[0],
            anchor_1[1],
            anchor_2[0] - anchor_1[0],
            anchor_2[1] - anchor_1[1],
            color="#222222",
            alpha=alpha,
            width=0.1,
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            zorder=0))

  def erase_edge_attractors(self) -> None:
    """
    erase edge attractor visualization
    """
    for artist in self.edge_attractor_artists:
      artist.remove()
    self.edge_attractor_artists = []

  def draw_tasks(self,
      ax: plt.Axes,
      alpha: float = 1.0,
      base_color: str = "#cc00cc",
      movable: bool = None,
      border_color: str = "#555555") -> None:
    """
    draw tasks of particle graph.
    1. Calculate the shortest route(s) for each task.
    2. Count how many shortest routes go through each connection.
    3. Color all connections according to this number.
    Bonus: If there are multiple shortest routes for tasks, add a button that displays a random combination of these routes.

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency. Defaults to 1.0.
        base_color (str, optional): base color for the gradient. Defaults to "#ff00ff".
    """
    if self.analysis_graph is None:
      self.init_analysis_graph()

    edge_weights: dict[tuple[str, str, int], int] = self.analysis_graph.get_random_shortest_task_paths_edge_counts()
    if len(edge_weights) == 0:
      print("No tasks to draw.")
      self.analysis_graph = None
      return
    max_weight = max(edge_weights.values())
    
    for (edge_key, particle_edge) in self.particle_edges.items():
      locations_key = (edge_key[0], edge_key[1], edge_key[3])
      if locations_key in edge_weights:
        edge_weight = edge_weights[locations_key]
      else:
        locations_key = (edge_key[1], edge_key[0], edge_key[3])
        edge_weight = edge_weights.get(locations_key, 0) # if the edge is not in the dict, set the weight to 0
      color = get_gradient_color(base_color, edge_weight, max_weight)
      particle_edge.draw(ax, color=color, alpha=alpha, movable=movable, border_color=border_color)

  def draw_edge_importance(self,
      ax: plt.Axes,
      alpha: float = 1.0,
      base_color: str = "#cc00cc",
      movable: bool = None,
      border_color: str = "#555555") -> None:
    """
    draw edge importance of particle graph. Importance is measured by the increase in task lengths if the edge is removed.

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency multiplier. Defaults to 1.0.
        base_color (str, optional): base color for the gradient. Defaults to "#ff00ff".
    """
    if self.analysis_graph is None:
      self.init_analysis_graph()

    edge_weights: dict[tuple[str, str, int], int] = self.analysis_graph.get_edge_importance()
    if len(edge_weights) == 0:
      print("No tasks to analyse edges with.")
      self.analysis_graph = None
      return
    # get maximum finite weight
    max_weight = max([weight for weight in edge_weights.values() if weight != float("inf")])
    
    for (edge_key, particle_edge) in self.particle_edges.items():
      locations_key = (edge_key[0], edge_key[1], edge_key[3])
      if locations_key in edge_weights:
        edge_weight = edge_weights[locations_key]
      else:
        locations_key = (edge_key[1], edge_key[0], edge_key[3])
        edge_weight = edge_weights.get(locations_key, 0)
      color = get_gradient_color(base_color, edge_weight, max_weight)
      particle_edge.draw(ax, color=color, alpha=alpha, movable=movable, border_color=border_color)

  def draw_graph_analysis(self, axs: "np.ndarray[plt.Axes]", grid_color: str = None, base_color="#cc00cc") -> None:
    """
    draw analysis of the graph. This includes:
    1. node degree distribution: how often each node degree occurs
    2. edge length distribution: how often each edge length occurs
    3. edge color distribution: how many edges have a certain color
    4. edge color length distribution: how often each edge length occurs for each color
    5. edge color total length distribution: combined edge lengths for each color
    6. edge importance: how much the average task length increases if an edge is removed
    7. task length distribution: how often each task length occurs
    8. task color average distribution: how often each color is needed on average for all tasks
    9. task visualisation: how many shortest task routes go through each edge

    Args:
        axs (np.ndarray[plt.Axes]): 3x3 array of matplotlib axes to draw on
        grid_color (str, optional): color of the grid. Defaults to None (no grid).
        base_color (str, optional): base color for the gradients. This is only used for the edge importance (6.)and task visualisation (9.). Defaults to "#cc00cc" (pink).
    """
    if self.analysis_graph is None:
      self.init_analysis_graph()

    # plot node degree distribution
    self.analysis_graph.plot_node_degree_distribution(ax=axs[0, 0], grid_color=grid_color)
    # plot edge length distribution
    self.analysis_graph.plot_edge_length_distribution(ax=axs[0, 1], grid_color=grid_color)
    # plot edge color length distribution
    self.analysis_graph.plot_edge_color_length_distribution(ax=axs[0, 2], grid_color=grid_color)
    # plot edge color distribution
    self.analysis_graph.plot_edge_color_distribution(ax=axs[1, 0], grid_color=grid_color)
    # plot edge color total length distribution
    self.analysis_graph.plot_edge_color_total_length_distribution(ax=axs[1, 1], grid_color=grid_color)
    # plot edge importance
    axs[1, 2].set_title("Edge importance")
    self.draw_nodes(ax=axs[1, 2])
    self.draw_labels(ax=axs[1, 2])
    self.draw_edge_importance(ax=axs[1, 2])
    # plot task length distribution
    self.analysis_graph.plot_task_length_distribution(ax=axs[2, 0], grid_color=grid_color)
    # plot task color distribution
    self.analysis_graph.plot_task_color_avg_distribution(ax=axs[2, 1], grid_color=grid_color)
    # plot shortest task paths
    axs[2, 2].set_title("Task visualisation")
    self.draw_nodes(ax=axs[2, 2])
    self.draw_labels(ax=axs[2, 2])
    self.draw_tasks(ax=axs[2, 2], alpha=1, base_color=base_color)


  def init_analysis_graph(self) -> None:
    """
    initialize the analysis graph.
    """
    self.analysis_graph: TTR_Graph_Analysis = TTR_Graph_Analysis(
        list(self.particle_nodes.keys()),
        self.particle_edges,
        self.tasks)

  def __str__(self) -> str:
    """
    return information about the particle graph:
    - number of nodes
    - number of edges

    Returns:
        str: information about the particle graph
    """
    return f"Particle graph with {len(self.node_labels)} nodes and {len(self.paths)} edges."

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
    particle_id = self.max_particle_id
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
            "tasks": self.tasks
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
    add an existing particle to the particle graph.
    This is mainly used to load particle graphs from JSON files.

    Args:
        particle (Graph_Particle): particle to add
    """
    if isinstance(particle, Particle_Node):
      self.particle_nodes[particle.label] = particle
    elif isinstance(particle, Particle_Edge):
      loc_1: str = particle.location_1_name
      loc_2: str = particle.location_2_name
      path_index: int = particle.path_index
      connection_index: int = particle.connection_index
      self.particle_edges[(loc_1, loc_2, path_index, connection_index)] = particle
    elif isinstance(particle, Particle_Label):
      self.particle_labels[particle.label] = particle

  def add_connection(self, location_1: str, location_2: str, length: int, color: str, add_path: bool=False, ax:plt.Axes = None) -> None:
    """
    Add a connection between two locations to the particle graph.
    This will create a new edge particle and add it to the particle graph.

    Args:
        location_1 (str): name of the first location
        location_2 (str): name of the second location
        length (int): length of the connection
        color (str): color of the connection
        add_path (bool, optional): whether to add the connection to the path list. Defaults to False.
        ax (plt.Axes, optional): axis to draw the connection on. Defaults to None (no drawing)
    """
    node_1: Graph_Particle = self.particle_nodes[location_1]
    node_2: Graph_Particle = self.particle_nodes[location_2]
    last_particle: Graph_Particle = node_1 # connect the first edge particle to the first node
    # determine number of connections that already exist between the two nodes
    connection_index: int = 0
    while (location_1, location_2, 0, connection_index) in self.particle_edges or (location_2, location_1, 0, connection_index) in self.particle_edges:
      connection_index += 1
    print(f"Adding connection {connection_index} between {location_1} and {location_2} with {length} particles.")
    # create `length` edge particles between `node_1` and `node_2``
    for path_index in range(length):
      edge_position: np.ndarray = node_1.position + (node_2.position - node_1.position) * (path_index+1) / (length+1)
      edge_rotation: float = np.arctan2(node_2.position[1] - node_1.position[1], node_2.position[0] - node_1.position[0])
      edge_particle: Graph_Particle = Particle_Edge(
          color=color,
          location_1_name=location_1,
          location_2_name=location_2,
          id=self.max_particle_id,
          path_index=path_index,
          position=edge_position,
          rotation=edge_rotation,
          mass=self.particle_parameters["edge_mass"],
          node_attraction=self.particle_parameters["edge-node"],
          edge_attraction=self.particle_parameters["edge-edge"],
          interaction_radius=self.particle_parameters["interaction_radius"],
          velocity_decay=self.particle_parameters["velocity_decay"],
          repulsion_strength=self.particle_parameters["repulsion_strength"],)
      edge_particle.add_connected_particle(last_particle)
      if path_index >= 1: # if not the first edge particle, add connection to previous edge particle
        last_particle.add_connected_particle(edge_particle)
      self.particle_edges[(location_1, location_2, path_index, connection_index)] = edge_particle
      last_particle = edge_particle
      self.max_particle_id += 1
      if ax is not None:
        edge_particle.draw(ax)
    # connect last edge particle to node_2
    last_particle.add_connected_particle(node_2)
    if add_path:
      self.paths.append((location_1, location_2, length, color))


  def delete_node(self, particle_node: Particle_Node) -> None:
    """
    Remove a single node particle and it's associated label particle from the particle graph.
    This also deletes:
    - all edges connected to the node
    - all labels associated with the node (same label as the node)
    - all tasks starting or ending in the node
    Deleting nodes may make some tasks impossible to complete. They are not removed since new edges can be added to make them possible again.

    Args:
        particle_node (Particle_Node): node particle to be deleted
    """
    self.analysis_graph = None
    # delete connected edges
    to_be_deleted: List[Particle_Edge] = []
    # find edges to be deleted (all that are associated with the node)
    for particle_edge in self.particle_edges.values():
      if particle_edge.location_1_name == particle_node.label or \
          particle_edge.location_2_name == particle_node.label:
        to_be_deleted.append(particle_edge)
    print(f"Deleted {len(to_be_deleted)} edge particles.")
    # actually delete edges
    for particle_edge in to_be_deleted:
      particle_edge.erase()
      self.delete_edge(particle_edge)

    # delete connected labels
    self.particle_labels[particle_node.label].erase()
    del self.particle_labels[particle_node.label]

    # delete tasks starting or ending in the node
    to_be_deleted: List[Tuple[str, str]] = []
    # find tasks to delelte
    for i, task in enumerate(self.tasks):
      if particle_node.label in task:
        to_be_deleted.insert(0, i) # delete in reverse order to avoid index errors
    print(f"Deleted {len(to_be_deleted)} tasks.")
    for i in to_be_deleted:
      print(f"Deleting task {i}: {self.tasks[i]}")
      self.tasks.pop(i)

    # delete node
    # self.node_labels.remove(particle_node.label)
    self.particle_nodes[particle_node.label].erase()
    del self.particle_nodes[particle_node.label]
    

  def delete_edge(self, particle_edge: Particle_Edge) -> None:
    """
    remove a single edge particle from the particle graph. If there are other edges connected to the particle, their `connected_particles` attribute gets updated to reflect the change. For this the connected particle `particle_edge` is removed from the list of connected particles and replaced by the other particle `particle_edge` is connected to (either another edge or a node).

    Args:
        particle_edge (Particle_Edge): edge particle to be deleted
    """
    self.analysis_graph = None
    loc_1 = particle_edge.location_1_name
    loc_2 = particle_edge.location_2_name
    path_index = particle_edge.path_index
    connection_index = particle_edge.connection_index
    path_length = 0
    # update path in self.paths
    for i, path in enumerate(self.paths):
      if path[0] == loc_1 and path[1] == loc_2 and path[3] == particle_edge.color:
        if path_length == 0:
          path_length = path[2]
        if path[2] == 1:
          del self.paths[i]
          break
        self.paths[i] = (loc_1, loc_2, path[2] - 1, particle_edge.color)
        break
    # update path indices of all edge particles with higher path index
    changed_particle_edges: dict[tuple[str, str, int, int], Particle_Edge] = dict()
    for particle_key, particle_edge in self.particle_edges.items():
      if particle_edge.location_1_name == loc_1 and \
          particle_edge.location_2_name == loc_2 and \
          particle_edge.connection_index == connection_index and \
          particle_edge.path_index > path_index:
        particle_edge.path_index -= 1
        changed_particle_edges[particle_key] = particle_edge
    # update path indices of all edges that needed to change
    for particle_key, particle_edge in changed_particle_edges.items():
      self.particle_edges[(particle_edge.location_1_name, particle_edge.location_2_name, particle_edge.path_index, connection_index)] = particle_edge
      del self.particle_edges[particle_key]
    # remove edge particle
    if (loc_1, loc_2, path_index) in self.particle_edges:
      del self.particle_edges[(loc_1, loc_2, path_index, connection_index)]
    elif (loc_2, loc_1, path_index) in self.particle_edges:
      del self.particle_edges[(loc_2, loc_1, path_index, connection_index)]
    else:
      print(f"Warning: Could not find edge particle {loc_1} -> {loc_2} ({path_index})")
      print()


  def rename_label(self, old_name: str, new_name: str, ax: plt.Axes) -> None:
    """
    rename a given label in the particle graph.

    Args:
        old_name (str): current label name
        new_name (str): new label name
        ax (plt.Axes): axis to draw the new label on
    """
    for label, particle_label in self.particle_labels.items():
      if label == old_name:
        particle_label.set_text(new_name, ax)
        self.particle_labels[new_name] = particle_label
        del self.particle_labels[label]
        break

  def update_path_color(self, particle_edge: Particle_Edge, old_color: str) -> None:
    """
    update the color of the path in the particle graph.

    Args:
        particle_edge (Particle_Edge): edge particle whose color has changed
        old_color (str): old color of the path the edge particle belongs to
    """
    self.analysis_graph = None
    # update path in self.paths
    for i, path in enumerate(self.paths):
      loc_1 = particle_edge.location_1_name
      loc_2 = particle_edge.location_2_name
      if path[0] == loc_1 and path[1] == loc_2 and path[3] == old_color:
        self.paths[i] = (loc_1, loc_2, path[2], particle_edge.color)
        print(f"updated path {loc_1} <-> {loc_2} from {old_color} to {particle_edge.color}")
        break

  def build_paths(self) -> None:
    """
    calculate a list of all paths in the particle graph as tuples (location_1, location_2, length, color) from the list of edges
    """
    self.analysis_graph = None
    locations_to_lengths = dict()
    for (loc_1, loc_2, min_path_length, connection_index), edge in self.particle_edges.items():
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
        tasks = [],
        node_positions = [],
        particle_parameters = particle_parameters,
      )
    particle_dict: dict[int, Graph_Particle] = dict()
    for particle_info in particle_dicts:
      connected_particles = particle_info.pop("connected_particles")
      particle_type = particle_info.pop("particle_type")
      
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
      particle_dict[particle.get_id()] = particle
    # add connections to particles
    for particle, particle_info in zip(particle_dict.values(), particle_dicts):
      for connected_particle_id in particle_info["connected_particles"]:
        particle.add_connected_particle(particle_dict[connected_particle_id])
    # build list of paths in graph
    particle_graph.build_paths()
    particle_graph.update_tasks(graph_info["particle_graph"]["tasks"])
    return particle_graph



def get_gradient_color(
    color: str,
    weight: int,
    max_weight: int,
    min_color_factor: float = 0.1,
    weight_zero_color: str = "#aaaaaa",
    inf_color: str = "#111111") -> str:
  """
  get a color that is a gradient between white and the given color

  Args:
      color (str): color as hex string to use for maximum weight
      weight (int): weight of current object
      max_weight (int): maximum weight of all objects
      min_color_factor (float, optional): proportion of color to use for weight 1. Defaults to 0.3.

  Returns:
      str: color in hex format
  """
  # convert color to rgb
  rgb_tuple = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
  if weight == 0:
    return weight_zero_color
  if isinf(weight):
    return inf_color
  # calculate color factor
  color_factor = min_color_factor + (1-min_color_factor) * weight / max_weight
  # calculate gradient color
  gradient_color = tuple(int(color_factor * rgb_tuple[i] + (1-color_factor) * 255) for i in range(3))
  # convert gradient color to hex
  gradient_color = f"#{''.join([hex(component)[2:].zfill(2) for component in gradient_color])}"
  return gradient_color

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
  tasks = [
    ("Menegroth", "Nargothrond"),
    ("Menegroth", "Hithlum")
  ]

  particle_graph = TTR_Particle_Graph(
      locations,
      paths,
      tasks,
      location_positions)

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
  particle_graph.draw_edge_attractors(ax, alpha=0.5)
  
  # # plot midpoints for all edges
  # for edge in particle_graph.particle_edges.values():
  #   midpoints = edge.get_edge_midpoints()
  #   ax.scatter(midpoints[:, 0], midpoints[:, 1], color="#000000", s=1)

  particle_graph.save_json("test_particle_graph.json")

  ax.set_xlim(-15, 15)
  ax.set_ylim(-10, 10)
  ax.legend()
  ax.set_aspect("equal")
  plt.grid(color="#dddddd", linestyle="--", linewidth=1)
  plt.show()

  recovered_graph: TTR_Particle_Graph = TTR_Particle_Graph.load_json("test_particle_graph.json")

  fig, ax = plt.subplots(dpi=100)
  recovered_graph.draw(ax, alpha_multiplier=1.0)
  ax.set_xlim(-15, 15)
  ax.set_ylim(-10, 10)
  ax.set_title("recovered graph")
  ax.set_aspect("equal")
  plt.grid(color="#dddddd", linestyle="--", linewidth=1)
  plt.show()