"""
TTR particle graph class. This represents the graph of a Ticket to Ride map as a system of particles for layout optimization.

A particle graph consists of nodes (locations), edges (paths) and node labels.
Each edge has a length and a color. Each node has a label close to it.

A particle graph's layout can be optimized using a simple particle method.
"""
import json
from typing import List, Tuple, Dict, Union, Any
from math import isinf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from graph_particle import Graph_Particle
from particle_node import Particle_Node
from particle_label import Particle_Label
from particle_edge import Particle_Edge
from graph_analysis import TTR_Graph_Analysis
from ttr_task import TTR_Task


class TTR_Particle_Graph:
  def __init__(self,
        locations: List[str],
        paths: List[Tuple[str, str, int, str]],
        tasks: dict[str, TTR_Task],
        node_positions: Dict[str, np.ndarray] = None,
        color_config: dict = {
          "label_text_color": "#eeeeee",
          "label_outline_color": "#000000",
        },
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
        },
        project_setup: dict = None,
        ):
    """
    Initialize a TTR particle graph made from labeled locations and paths.
    Each path is made from several rectangular edge particles.

    Args:
        locations (List[str]): list of location labels
        paths (List[Tuple[str, str, int, str]]): list of paths. Each path is a tuple of (node_1, node_2, length, color)
        tasks (List[Tuple[str, str]]): list of tasks. Each task is a tuple of (node_1, node_2)
        node_positions (Dict[str, np.ndarray], optional): dictionary of location positions. Keys are location labels, values are 2D numpy arrays representing the position of the location. Defaults to None.
        color_config (dict, optional): dictionary of color settings. Defaults to {
            "label_text_color": "#eeeeee",
            "label_outline_color": "#000000",
          }.
        particle_parameters (dict, optional): dictionary of particle simmulation parameters. Defaults to {
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
          }.
        project_setup (dict, optional): dictionary of project setup parameters, including bg image path, size, task export settings, etc. Defaults to None (create default setup)
    """
    self.color_config: dict = color_config
    self.paths: List[Tuple[str, str, int, str]] = paths
    if tasks and isinstance(tasks, (list, tuple)) and isinstance(tasks[0], (tuple, list)):
      self.tasks: dict[str, TTR_Task] = {}
      for task in tasks:
        self.tasks[task.name] = TTR_Task([task[0], task[-1]])
      print(f"Converted tasks given as tuples to TTR_Task objects.")
    self.tasks:  dict[str, TTR_Task] = tasks
    self.project_setup_dict: dict = self.setup_project_dict() if project_setup is None else project_setup
    # fill in missing keys with default values
    default_project_setup = self.setup_project_dict()
    for key in default_project_setup.keys():
      if key not in self.project_setup_dict.keys():
        self.project_setup_dict[key] = default_project_setup[key]
    
    self.particle_parameters: dict = particle_parameters
    self.node_labels: List[str] = locations # deleted after graph creation
    self.node_positions: Dict[str, np.ndarray] = node_positions # deleted after graph creation

    self.max_particle_id = 0

    self.particle_nodes: dict[str, Particle_Node] = dict()
    self.particle_edges: dict[Tuple[str, str, int, int], Particle_Edge] = dict()
    self.particle_labels: dict[str, Particle_Label] = dict()
    self.analysis_graph: TTR_Graph_Analysis = None

    self.graph_extent: np.ndarray = np.array([0, 0, 0, 0], dtype=np.float16)

    self.edge_attractor_artists: List[plt.Artist] = list()
    
    self.create_particle_system()


  def setup_project_dict(self) -> dict:
    project_setup_dict: dict = {
        # "bg_image_settings": {
            "bg_image_path": "",
            "bg_image_size": (52.4, 39.4),
            "bg_image_offset": (0, 0),
        #    }
        # "task_export_settings": {
            # background map on task card (None = use bg image)
            "task_bg_image_path": "",
            # task bg size (used to scale down graph)
            "task_bg_image_size": (49.25, 39.4),
            "task_bg_image_offset": (0, 0),
            # task card frame (pasted on top of map)
            "task_frame_image_path": "",
            "task_frame_image_offset": (0, 0),
            # task card size (used to scale the task card frame)
            "task_card_size": (9.1, 6.5),
            # task card text settings
            "task_label_position": (4.4, 5.8),
            "task_label_scale": 0.4,
            # task card node settings (locations)
            "task_node_scale": 0.5,
            "task_node_override_image_path": "",
            "task_node_override": False,
            "task_node_connection_lines": True,
            "task_node_connection_line_width": 10,
            "task_node_connection_line_color": "#cc5500",
            "task_node_connection_line_alpha": 0.7,
            # task card points settings
            "task_points_position": (1.1, 1.3),
            "task_points_scale": 1.15,
            "task_bonus_points_position": (2.4, 0.8),
            "task_bonus_points_scale": 0.8,
            "task_penalty_points_position": (8.0, 0.8),
            "task_penalty_points_scale": 0.8,
            "task_points_directory": "assets/points_images",
            # card folder settings
            "task_card_folder_path": "",
          # other settings
            "label_font": "assets/fonts/Stamp.ttf",
            "label_fontsize": 200,
            "edge_images_path": "assets/edge_images",
        #    }
    }
    return project_setup_dict

  def get_bg_info(self) -> dict:
    """
    Get all background image settings from the project setup dict.

    Returns:
        dict: Dictionary containing all settings for background image.
    """
    return {
        "bg_image_path": self.project_setup_dict["bg_image_path"],
        "bg_image_size": self.project_setup_dict["bg_image_size"],
        "bg_image_offset": self.project_setup_dict["bg_image_offset"],
      }

  def set_bg_info(self, **kwargs) -> None:
    """
    Set background image settings (update internal project settings). Possible keys are `bg_image_path`, `bg_image_size`, `bg_image_offset`.
    To set size or offset values seperately use the keys `width`, `height`, `x_offset`, `y_offset`.
    """
    if "bg_image_path" in kwargs.keys():
      self.project_setup_dict["bg_image_path"] = kwargs["bg_image_path"]
      # print(f"bg_settings: Set bg_image_path to {kwargs['bg_image_path']}")
    if "bg_image_size" in kwargs.keys():
      self.project_setup_dict["bg_image_size"] = kwargs["bg_image_size"]
      # print(f"bg_settings: Set bg_image_size to {kwargs['bg_image_size']}")
    if "bg_image_offset" in kwargs.keys():
      self.project_setup_dict["bg_image_offset"] = kwargs["bg_image_offset"]
      # print(f"bg_settings: Set bg_image_offset to {kwargs['bg_image_offset']}")
    if "width" in kwargs.keys():
      self.project_setup_dict["bg_image_size"] = (
          kwargs["width"],
          self.project_setup_dict["bg_image_size"][1])
      # print(f"bg_settings: Set bg_image_size to {self.project_setup_dict['bg_image_size']}")
    if "height" in kwargs.keys():
      self.project_setup_dict["bg_image_size"] = (
          self.project_setup_dict["bg_image_size"][0],
          kwargs["height"])
      # print(f"bg_settings: Set bg_image_size to {self.project_setup_dict['bg_image_size']}")
    if "x_offset" in kwargs.keys():
      self.project_setup_dict["bg_image_offset"] = (
          kwargs["x_offset"],
          self.project_setup_dict["bg_image_offset"][1])
      # print(f"bg_settings: Set bg_image_offset to {self.project_setup_dict['bg_image_offset']}")
    if "y_offset" in kwargs.keys():
      self.project_setup_dict["bg_image_offset"] = (
          self.project_setup_dict["bg_image_offset"][0],
          kwargs["y_offset"])
      # print(f"bg_settings: Set bg_image_offset to {self.project_setup_dict['bg_image_offset']}")

  def get_task_info(self, key=None) -> dict | Any:
    """
    Get all task card settings from the project setup dict.
    If a key is given, only that key's value is returned. Otherwise, the whole dictionary with all settings is returned.
    
    Args:
        key (str, optional): key of the setting to return. Defaults to None.

    Returns:
        dict: Dictionary containing all settings for task card export.
    """
    if key is None:
      task_dict = self.project_setup_dict.copy() # copy dict
      # remove bg image settings
      for key in self.project_setup_dict.keys():
        if not "task_" == key[0:5]:
          del task_dict[key]
      return task_dict
    return self.project_setup_dict[key]

  def set_task_info(self, **kwargs) -> None:
    """
    Set task card settings (update internal project settings). Possible keys are:
      - `task_bg_image_path` (str)
      - `task_bg_image_size` (tuple[float, float])
      - `task_bg_image_offset` (tuple[float, float])
      - `task_frame_image_path` (str)
      - `task_frame_image_offset` (tuple[float, float])
      - `task_card_size` (tuple[float, float])
      - `task_label_position` (tuple[float, float])
      - `task_label_scale` (float)
      - `task_node_scale` (float)
      - `task_node_override_image_path` (str)
      - `task_node_override` (bool)
      - `task_node_connection_lines` (bool)
      - `task_node_connection_line_width` (float)
      - `task_points_position` (tuple[float, float])
      - `task_points_scale` (float)
      - `task_bonus_points_position` (tuple[float, float])
      - `task_bonus_points_scale` (float)
      - `task_penalty_points_position` (tuple[float, float])
      - `task_penalty_points_scale` (float)
      - `task_points_directory` (str)
      - `task_card_folder_path` (str)
    """
    for key in kwargs.keys():
      if key in self.project_setup_dict.keys():
        self.project_setup_dict[key] = kwargs[key]
        # print(f"task_settings: Set {key} to {kwargs[key]}")
      else:
        print(f"task_settings: Ignoring unknown key {key}.")

  def get_misc_info(self, key=None) -> dict | Any:
    """
    Get all misc project settings from the project setup dict.
    If a key is given, only that key's value is returned. Otherwise, the whole dictionary with all settings is returned.
    
    Args:
        key (str, optional): key of the setting to return. Defaults to None.

    Returns:
        dict: Dictionary containing all misc settings.
    """
    if key is None:
      misc_dict: dict = self.project_setup_dict.copy() # copy dict
      # remove bg image settings
      for key in self.get_bg_info().keys():
        del misc_dict[key]
      # remove task card settings
      for key in self.get_task_info().keys():
        del misc_dict[key]
      return misc_dict
    return self.project_setup_dict[key]
  
  def set_misc_info(self, **kwargs) -> None:
    """
    Set misc project settings (update internal project settings). Possible keys are `label_font`, `edge_images_path`.
    """
    for key in kwargs.keys():
      if key in self.project_setup_dict.keys():
        self.project_setup_dict[key] = kwargs[key]
        # print(f"misc_settings: Set {key} to {kwargs[key]}")
      else:
        print(f"misc_settings: Ignoring unknown key {key}.")


  def create_particle_system(self) -> None:
    """
    Create the particle system from the given locations and paths connecting them.
    """
    particle_id: int = self.max_particle_id + 1
    self.label_height_scale: float = Particle_Label.get_label_height_scale(font_path=self.project_setup_dict.get("label_font", "assets/fonts/Stamp.ttf"))
    n_nodes: int = len(self.node_labels)
    for path_index, label in enumerate(self.node_labels):
      if self.node_positions is not None:
        position = self.node_positions[label]
      else:
        position = np.array([3*path_index, 0], dtype=np.float16)
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
          self.particle_nodes[label].position + np.array([self.particle_labels[label].bounding_box_size[0] / 2, 0], dtype=np.float16) + np.array([1, 0], dtype=np.float16)
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
    for i in range(iterations):
      for particle in all_particles:
        particle.reset_acceleration()

      for particle_1 in all_particles:
        for particle_2 in all_particles:
          if particle_1 != particle_2:
            particle_1.interact(particle_2)

      for particle in all_particles:
        particle.update(dt)

  def set_graph_extent(self, graph_extent: np.ndarray) -> None:
    """
    set the extent of the graph. This is used to automatically position edges when periodic boundary conditions are used. It's recommended to set this to the extent of the background image used for the graph.

    Args:
        graph_extent (np.ndarray): extent of the graph (xmin, xmax, ymin, ymax)
    """
    self.graph_extent = graph_extent


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
      particle_label.position = particle_label.connected_particles[0].position + np.array([x_offset, y_offset], dtype=np.float16)
      particle_label.erase()
      particle_label.draw(ax, movable=movable)

  def get_connection_count(self, location_1: str, location_2: str) -> int:
    """
    Count the number of edges between two locations.

    Args:
        location_1 (str): label of the first node
        location_2 (str): label of the second node

    Returns:
        int: number of edges between the two locations
    """
    n_edges: int = 0
    connection_indices: set[int] = set()
    for edge_key in self.particle_edges.keys():
      if {location_1, location_2} == {edge_key[0], edge_key[1]}:
        if not edge_key[3] in connection_indices:
          n_edges += 1
          connection_indices.add(edge_key[3])
    return n_edges

  def straighten_connections(self,
      ax: plt.Axes,
      x_periodic: bool = False,
      y_periodic: bool = False,
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
    updated_edges = set()
    for edge_key in self.particle_edges.keys():
      location_1 = edge_key[0]
      location_2 = edge_key[1]
      connection_index = edge_key[3]
      connection_identifier = (location_1, location_2, connection_index)
      # only straighten each edge once
      if connection_identifier not in updated_edges:
      # find how many connections there are between the two nodes
        max_connection_index: int = max(val for (str1, str2, _, val) in self.particle_edges if {str1, str2} == {location_1, location_2})
        self.straighten_connection(
            location_1,
            location_2,
            connection_index=connection_index,
            max_connection_index=max_connection_index,
            ax=ax,
            x_periodic=x_periodic,
            y_periodic=y_periodic,
            movable=movable,
            edge_border_color=edge_border_color,
            alpha=alpha,
            **draw_kwargs
        )
        updated_edges.add(connection_identifier)

  def straighten_connection(self,
      location_1: str,
      location_2: str,
      connection_index: int,
      ax: plt.Axes,
      x_periodic: bool = True,
      y_periodic: bool = True,
      movable: bool = False,
      max_connection_index: int = 1,
      edge_border_color: str = None,
      alpha: float = 0.8,
      **draw_kwargs) -> None:
    """
    move an edge such that it forms a straight line between its connected nodes. Then erase and redraw the edge.

    Args:
        location_1 (str): label of the first node
        location_2 (str): label of the second node
        connection_index (int): index of the connection
        ax (plt.Axes): axes to draw on
        x_periodic (bool, optional): whether the graph is periodic in x direction. If so, the edge will be drawn in the shortest direction even if that means crossing the periodic boundary. Defaults to False.
        y_periodic (bool, optional): whether the graph is periodic in y direction. If so, the edge will be drawn in the shortest direction even if that means crossing the periodic boundary. Defaults to False.
        movable (bool, optional): whether the edge is movable. Defaults to False.
        max_connection_index (int, optional): maximum number of connections between the two nodes. Defaults to 1.
        edge_border_color (str, optional): color of the edge border. Defaults to None.
        alpha (float, optional): alpha value of the edge. Defaults to 0.8.
        draw_kwargs (dict): kwargs to pass to the draw method of the edge particle
    """
    node_1 = self.particle_nodes[location_1]
    node_2 = self.particle_nodes[location_2]
    node_distance_vec: np.ndarray = node_2.position - node_1.position
    if x_periodic and abs(node_distance_vec[0]) > (self.graph_extent[1] - self.graph_extent[0]) / 2:
      # if the distance in x direction is larger than half the graph extent, the shortest distance is across the periodic boundary
      node_distance_vec[0] = node_distance_vec[0] - np.sign(node_distance_vec[0]) * (self.graph_extent[1] - self.graph_extent[0])
    if y_periodic and abs(node_distance_vec[1]) > (self.graph_extent[3] - self.graph_extent[2]) / 2:
      # if the distance in y direction is larger than half the graph extent, the shortest distance is across the periodic boundary
      node_distance_vec[1] = node_distance_vec[1] - np.sign(node_distance_vec[1]) * (self.graph_extent[3] - self.graph_extent[2])
    # calculate orthogonal vector
    offset_normal_vec: np.ndarray = np.array([-node_distance_vec[1], node_distance_vec[0]])
    offset_normal_vec: np.ndarray = offset_normal_vec / np.linalg.norm(offset_normal_vec)
    edge_particles: List[Particle_Edge] = []
    length = 0
    # find all edge particles that belong to the current connection
    while True:
      if (location_1, location_2, length, connection_index) in self.particle_edges:
        edge_particles.append(self.particle_edges[(location_1, location_2, length, connection_index)])
        length += 1
      else:
        break
    # define offset vector for multiple connections
    offset_vec: np.ndarray = offset_normal_vec * min(edge_particles[0].bounding_box_size)
    # reposition the edge particles
    for i, edge_particle in enumerate(edge_particles):
      # calculate new position through linear interpolation along node_distance_vec
      new_position: np.ndarray = node_1.position + node_distance_vec * (i+1) / (length+1)
      # make sure the new position is within the graph extent
      new_position: np.ndarray = (new_position - self.graph_extent[0:3:2]) \
          % np.array([
              self.graph_extent[1] - self.graph_extent[0],
              self.graph_extent[3] - self.graph_extent[2]]) \
          + self.graph_extent[0:3:2]
      # avoid overlap of multiple connections between the same nodes by offsetting the particles
      new_position += offset_vec * (connection_index - max_connection_index / 2)
      edge_particle.set_position(new_position)

      new_rotation = np.arctan2(node_distance_vec[1], node_distance_vec[0])
      edge_particle.set_rotation(new_rotation)

      edge_particle.erase()
      edge_particle.draw(
          ax,
          movable=movable,
          border_color=edge_border_color,
          alpha=alpha,
          **draw_kwargs)


  def repair_connections(self):
    """
    Update the `connected_particles` property of all edge particle based on the current state of the graph (as defined by the keys in self.particle_edges).
    This is useful to repair graphs with wrong connection settings.
    """
    # nodes should never be connected to other particles
    for node in self.particle_nodes.values():
      node.connected_particles = []
    # labls should only be connected to corresponding nodes
    for location_name, particle_label in self.particle_labels.items():
      particle_label.set_connected_particles([self.particle_nodes[location_name]])
    # calculate connected particles for all edge particles based on keys in self.particle_edges
    # 1. loop through all stored connections
    # 2. find all edges that belong to the current connection
    # 3. update `connected_particles` property of particles in the current connection
    updated_edges = set()
    for edge_key in self.particle_edges.keys():
      location_1, location_2, connection_index = edge_key[0], edge_key[1], edge_key[3] # ignore path index for now
      connection_identifier = (location_1, location_2, connection_index)
      if connection_identifier not in updated_edges:
        self.repair_edge_connection(location_1, location_2, connection_index)
        updated_edges.add(connection_identifier)
    print("Repaired connections for all particles.")


  def repair_edge_connection(self, location_1: str, location_2: str, connection_index: int):
    """
    Repair the `connected_particles` property of all edge particles in a connection based on the current state of the graph (as defined by the keys in self.particle_edges).

    Args:
        location_1 (str): label of the first node
        location_2 (str): label of the second node
        connection_index (int): index of the connection
    """
    # find all edges that belong to the current connection
    edge_particles: list[Particle_Edge] = []
    length = 0
    while True: # find all edge particles that belong to the current connection sorted by path index (increasing)
      if (location_1, location_2, length, connection_index) in self.particle_edges:
          edge_particles.append(self.particle_edges[(location_1, location_2, length, connection_index)])
          length += 1
      else:
          break
    # update connected_particles property of particles in the current connection
    node_1 = self.particle_nodes[location_1]
    node_2 = self.particle_nodes[location_2]
    # sort edges based on distance of first edge to each node to assign correct node to the end edge particles
    if np.linalg.norm(edge_particles[0].position - node_1.position) > np.linalg.norm(edge_particles[0].position - node_2.position):
      edge_particles.reverse()
    if length == 1: # handle length 1 connections
      edge_particles[0].connected_particles = [node_1, node_2]
      return
    for i, edge_particle in enumerate(edge_particles):
        if i == 0:
            edge_particle.connected_particles = [node_1, edge_particles[i+1]]
        elif i == length - 1:
            edge_particle.connected_particles = [edge_particles[i-1], node_2]
        else:
            edge_particle.connected_particles = [edge_particles[i-1], edge_particles[i+1]]

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
    # update cell list radius


  def get_locations(self) -> List[str]:
    """
    get all location labels in particle graph

    Returns:
        List[str]: list of location labels
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
        particle_parameters (dict): dictionary of particle parameters. Possible keys (and ecommended values) are:
          "velocity_decay": 0.99,
          "edge-edge": 0.01 - edge-edge attraction strength
          "edge-node": 0.01 - edge-node attraction strength
          "node-label": 0.001 - node-label attraction strength
          "node-target": 0.001 - node-target attraction strength
          "node_mass": 1 - mass of node particles
          "edge_mass": 1 - mass of edge particles
          "label_mass": 0.2 - mass of label particles
          "interaction_radius": 15 - maximum distance between particles to interact
          "repulsion_strength": 2 - repulsion strength between particles
    """
    for particle in self.get_particle_list():
      if isinstance(particle, Particle_Node):
        particle.set_simulation_parameters(
            mass = particle_parameters["node_mass"],
            target_attraction = particle_parameters["node-target"],
            interaction_radius = particle_parameters["interaction_radius"],
            velocity_decay = particle_parameters["velocity_decay"],
            repulsion_strength = particle_parameters["repulsion_strength"],
        )
      elif isinstance(particle, Particle_Edge):
        particle.set_simulation_parameters(
            mass = particle_parameters["edge_mass"],
            node_attraction = particle_parameters["edge-node"],
            edge_attraction = particle_parameters["edge-edge"],
            interaction_radius = particle_parameters["interaction_radius"],
            velocity_decay = particle_parameters["velocity_decay"],
            angular_velocity_decay = particle_parameters["velocity_decay"],
            repulsion_strength = particle_parameters["repulsion_strength"],
        )
      elif isinstance(particle, Particle_Label):
        particle.set_simulation_parameters(
            mass = particle_parameters["label_mass"],
            node_attraction = particle_parameters["node-label"],
            interaction_radius = particle_parameters["interaction_radius"],
            velocity_decay = particle_parameters["velocity_decay"],
            angular_velocity_decay = particle_parameters["velocity_decay"],
            repulsion_strength = 0, # particle_parameters["repulsion_strength"])
        )


  def update_tasks(self, new_tasks: dict[str, TTR_Task]) -> None:
    """
    update the list of tasks saved in the particle graph.
    The previous list of tasks is overwritten.

    Args:
        task_list (dict[str, TTR_Task]) or (List[dict]) or (List[Tuple[str, str]]): list/ dict of tasks.
            Each task is given as a tuple of the form (location_1, location_2), or a TTR_Task object. All list elements must be of the same type.
            If the input is not of these types, it is assumed to be a dict of TTR_Task objects with the tasks' names as keys.
    """
    if not new_tasks:
      self.tasks: dict[str, TTR_Task] = {}
      return
    if isinstance(new_tasks, list) and isinstance(new_tasks[0], (tuple, list)): # TODO: remove this case when it is no longer needed
      self.tasks: dict[str, TTR_Task] = {}
      for task in new_tasks:
        task: TTR_Task = TTR_Task([task[0], task[1]])
        self.tasks[task.name] = task
    elif isinstance(new_tasks, list) and isinstance(new_tasks[0], dict): # for loading from json
      self.tasks: dict[str, TTR_Task] = {}
      for task in new_tasks:
        task = TTR_Task.from_dict(task)
        self.tasks[task.name] = task
    else:
      self.tasks: dict[str, TTR_Task] = new_tasks

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

  def set_node_sizes(self, node_sizes: Union[float, list[float]]) -> None:
    """
    Set the sizes of all nodes in the graph to the given value(s).

    Args:
        node_sizes (Union[float, list[float]]): size(s) of nodes
          If a single float is given, all nodes are set to this size
          If a list of floats is given, each node is set to the corresponding size.
          The list must have the same length as the number of nodes.
    """
    if isinstance(node_sizes, float):
      node_sizes: list[float] = [node_sizes] * len(self.particle_nodes)
    elif len(node_sizes) != len(self.particle_nodes):
      raise ValueError(f"Mismatch between number of nodes ({len(self.particle_nodes)}) and number of node sizes ({len(node_sizes)}).")
    for particle_node, node_size in zip(self.particle_nodes.values(), node_sizes):
      particle_node.set_size(node_size)
      
  def set_label_settings(self, ax: plt.Axes, label_font: str, label_fontsize: int) -> None:
    """
    set the font and font size of all labels in the graph.
    Redraw the labels on the given axes.

    Args:
        ax (plt.Axes): axes to draw on
        label_font (str): font name/ path to .ttf file
        label_fontsize (int): font size
    """
    self.erase_labels()
    Particle_Label.get_label_height_scale(fontsize=label_fontsize, font_path=label_font)
    for particle_label in self.particle_labels.values():
      particle_label.set_font(label_fontsize, label_font)
    self.draw_labels(ax, movable=False)


  def toggle_move_nodes(self, move_nodes: bool = None) -> None:
    """
    toggle the ability to move nodes in the graph.
    If `move_nodes` is None, the current state is toggled.
    If `move_nodes` is True or False, the state is set to the given value.

    Args:
        move_nodes (bool, optional): new state of node movement. Defaults to None (toggle).
    """
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
      particle_label.draw(
          ax,
          color=self.color_config["label_text_color"],
          border_color=self.color_config["label_outline_color"],
          alpha=1.0 * alpha_multiplier,
          movable=movable)

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
      particle_label.draw(
          ax,
          color=self.color_config["label_text_color"],
          border_color=self.color_config["label_outline_color"],
          alpha=alpha,
          movable=movable)

  def erase_labels(self) -> None:
    """
    erase labels of particle graph
    """
    for particle_label in self.particle_labels.values():
      particle_label.erase()

  def draw_edges(self,
      ax: plt.Axes,
      alpha: float = 0.8,
      movable: bool = False,
      color: str = None,
      border_color: str = "#555555") -> None:
    """draw edges of particle graph

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency. Defaults to 0.8.
        movable (bool, optional): whether the edges are movable. Defaults to False.
        color (str, optional): color of the edges. Defaults to None (use color of edge).
        border_color (str, optional): color of the edge border. Defaults to "#555555".
    """
    for particle_edge in self.particle_edges.values():
      if color is None:
        edge_color = particle_edge.color
      else:
        edge_color = color
      particle_edge.draw(ax, color=edge_color, alpha=alpha, movable=movable, border_color=border_color)

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
        force_1, anchor_1 = particle_edge.get_attraction_forces(connected_particle)
        # force_2, anchor_2 = connected_particle.get_attraction_forces(particle_edge)
        # if isinstance(connected_particle, Particle_Node):
        #   print(f"connected to node: {connected_particle.id} at {connected_particle.position}")
        #   print(f"anchor_1: {anchor_1}")
        #   print(f"anchor_2: {anchor_2}")
        # arrow_length = np.linalg.norm(anchor_2-anchor_1)
        # if arrow_length > max(particle_edge.bounding_box_size):
        #   print(f"Warning: edge attractor is unusually long between particles: {particle_edge.id} and {connected_particle.id} ({arrow_length} cm).")
        self.edge_attractor_artists.append(
          ax.arrow(
            anchor_1[0],
            anchor_1[1],
            3*force_1[0],
            3*force_1[1],
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
      movable: bool = None,
      base_color: str = "#cc00cc",
      border_color: str = "#555555",
      neutral_color: str = "#aaaaaa") -> tuple[plt.Axes, mpl.colorbar.Colorbar]:
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
        movable (bool, optional): whether the tasks are movable. Defaults to None (use movable of particle graph).
        border_color (str, optional): color of the task border. Defaults to "#555555".
        neutral_color (str, optional): color of connections that are not part of any shortest route. Defaults to "#aaaaaa".

    Returns:
        plt.Axes: matplotlib axes containing the colorbar
        mpl.colorbar.Colorbar: colorbar
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
      color = get_gradient_color(base_color, edge_weight, max_weight, weight_zero_color=neutral_color)
      particle_edge.draw(ax, color=color, alpha=alpha, movable=movable, border_color=border_color)
      
    cbar_ax, cbar = add_colorbar(
      ax,
      min_value=0,
      max_value=max_weight,
      min_color=neutral_color,
      max_color=base_color,
      label="Avg. number of shortest routes through edge")
    return cbar_ax, cbar

  def draw_edge_importance(self,
      ax: plt.Axes,
      alpha: float = 1.0,
      movable: bool = False,
      base_color: str = "#cc00cc",
      border_color: str = "#555555",
      neutral_color: str = "#aaaaaa") -> tuple[plt.Axes, mpl.colorbar.Colorbar]:
    """
    draw edge importance of particle graph. Importance is measured by the increase in task lengths if the edge is removed.

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        alpha (float, optional): transparency multiplier. Defaults to 1.0.
        movable (bool, optional): whether each edge is set to be movable or not. Defaults to False
        base_color (str, optional): base color for the gradient. Defaults to "#ff00ff".
        border_color (str, optional): color of each edges border. Defaults to "#555555" (gray).
        neutral_color (str, optional): color of edges that are not part of any task. Defaults to "#aaaaaa" (light gray).

    Returns:
        plt.Axes: matplotlib axes containing the colorbar
        mpl.colorbar.Colorbar: colorbar
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
    
    cbar_axes, cbar = add_colorbar(
      ax,
      min_value=0,
      max_value=max_weight,
      min_color=neutral_color,
      max_color=base_color,
      label="Max. task length increase when removing edge")

    for (edge_key, particle_edge) in self.particle_edges.items():
      locations_key = (edge_key[0], edge_key[1], edge_key[3])
      if locations_key in edge_weights:
        edge_weight = edge_weights[locations_key]
      else:
        locations_key = (edge_key[1], edge_key[0], edge_key[3])
        edge_weight = edge_weights.get(locations_key, 0)
      color = get_gradient_color(base_color, edge_weight, max_weight, weight_zero_color=neutral_color)
      particle_edge.draw(ax, color=color, alpha=alpha, movable=movable, border_color=border_color)

    return cbar_axes, cbar

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
    self.draw_nodes(ax=axs[1, 2])
    self.draw_labels(ax=axs[1, 2])
    if not self.tasks: # no graphs exist yet
      axs[1, 2].set_title("graph")
      print("No tasks found to analyse edges with.")
      self.draw_edges(ax=axs[1, 2])
    else: # tasks are already defined
      axs[1, 2].set_title("Edge importance")
      self.draw_edge_importance(ax=axs[1, 2])
      # plot task length distribution
      self.analysis_graph.plot_task_points_distribution(ax=axs[2, 0], grid_color=grid_color)
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
      if loc_1 > loc_2: # sort locations alphabetically
        loc_1, loc_2 = loc_2, loc_1
        print(f"Warning: locations in path {loc_1} -> {loc_2} were not sorted alphabetically")
      path_index: int = particle.path_index
      path_index: int = particle.path_index
      connection_index: int = particle.connection_index
      # handle incorrect connection indices:
      particle_edge_identifier = (loc_1, loc_2, path_index, connection_index)
      if particle_edge_identifier in self.particle_edges:
        print(f"Warning: edge particle with identifier {particle_edge_identifier} already exists.")
        while particle_edge_identifier in self.particle_edges:
          connection_index += 1
          particle_edge_identifier = (loc_1, loc_2, path_index, connection_index)
        print(f"New connection index: {connection_index}")
        particle.connection_index = connection_index
      self.particle_edges[particle_edge_identifier] = particle
    elif isinstance(particle, Particle_Label):
      self.particle_labels[particle.label] = particle
    self.max_particle_id += 1

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
      self.max_particle_id += 1
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
      if location_1 > location_2: # sort locations alphabetically
        location_1, location_2 = location_2, location_1
      self.particle_edges[(location_1, location_2, path_index, connection_index)] = edge_particle
      last_particle = edge_particle
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

    # delete tasks containing the node
    to_be_deleted: List[str] = []
    # find tasks to delelte
    for task_key, task in self.tasks.items():
      if particle_node.label in task.node_names: # TODO: consider just deleting tasks that start or end in the deleted node
        to_be_deleted.insert(0, task_key) # delete in reverse order to avoid index errors
    print(f"Deleted {len(to_be_deleted)} tasks.")
    for task_key in to_be_deleted:
      print(f"Deleting task: {task_key}")
      del self.tasks[task_key]

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
    if loc_1 > loc_2: # sort locations alphabetically
      loc_1, loc_2 = loc_2, loc_1
      print(f"Warning: locations in path {loc_1} -> {loc_2} were not sorted alphabetically")
    path_index = particle_edge.path_index
    connection_index = particle_edge.connection_index
    path_length = 0
    # update path length in self.paths
    for i, path in enumerate(self.paths):
      # TODO: this only works if there are no two connections with the same color but different lengths
      if path[0] == loc_1 and path[1] == loc_2 and path[3] == particle_edge.color:
        if path_length == 0:
          path_length = path[2]
        if path[2] == 1:
          del self.paths[i]
          break
        self.paths[i] = (loc_1, loc_2, path[2] - 1, particle_edge.color)
        break
    # remove edge particle
    if (loc_1, loc_2, path_index, connection_index) in self.particle_edges:
      del self.particle_edges[(loc_1, loc_2, path_index, connection_index)]
    else:
      print(f"Warning: Could not find edge particle to remove: {loc_1} -> {loc_2} ({path_index})")
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

  def rename_node(self, old_name: str, new_name: str) -> None:
    """
    rename a given node in the particle graph.
    
    Args:
        old_name (str): current node name
        new_name (str): new node name
    """
    self.particle_nodes[new_name] = self.particle_nodes.pop(old_name)
    self.particle_nodes[new_name].label = new_name
    # rename node in all edges
    modified_edges: dict[tuple, tuple[tuple, Particle_Edge]] = dict()
    for edge_key, particle_edge in self.particle_edges.items():
      if old_name in edge_key: # edge needs to be renamed
        # replace name in edge key (tuple)
        new_edge_key = tuple(new_name if x == old_name else x for x in edge_key)
        particle_edge.location_1_name = new_name if particle_edge.location_1_name == old_name else particle_edge.location_1_name
        particle_edge.location_2_name = new_name if particle_edge.location_2_name == old_name else particle_edge.location_2_name
        modified_edges[edge_key] = (new_edge_key, particle_edge)
    for old_edge_key, (new_edge_key, particle_edge) in modified_edges.items():
      del self.particle_edges[old_edge_key]
      self.particle_edges[new_edge_key] = particle_edge
    # rename node in all tasks
    modified_tasks: dict[str, tuple[str, TTR_Task]] = dict()
    for task_key, task in self.tasks.items():
      if old_name in task.node_names:
        new_node_names = [new_name if x == old_name else x for x in task.node_names]
        task.set_node_names(new_node_names, update_name=True)
        modified_tasks[task_key] = (task.name, task)
    for old_task_key, (new_task_key, task) in modified_tasks.items():
      del self.tasks[old_task_key]
      self.tasks[new_task_key] = task
        

  def rename_label(self, old_name: str, new_name: str, ax: plt.Axes) -> None:
    """
    rename a given label in the particle graph.

    Args:
        old_name (str): current label name
        new_name (str): new label name
        ax (plt.Axes): axis to draw the new label on
    """
    for label, particle_label in self.particle_labels.items():
      if label == old_name: # found the label to be renamed
        particle_label.set_text(new_name, ax)
        del self.particle_labels[label]
        self.particle_labels[new_name] = particle_label
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
      if loc_1 > loc_2: # sort locations alphabetically
        loc_1, loc_2 = loc_2, loc_1
        print(f"Warning: locations in path {loc_1} -> {loc_2} were not sorted alphabetically")
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
    # particle_id = self.max_particle_id
    # for particle_node in all_particles:
    #   try:
    #     particle_node.set_id(particle_id)
    #   except AttributeError:
    #     particle_node.particle_id = particle_id
    #   particle_id += 1
    # get json string for all particles
    all_particles_dict: List[Dict[str, Any]] = [particle.to_dict() for particle in all_particles]
    # get json string for all tasks
    all_tasks_dict = [task.to_dict() for task in self.tasks.values()]
    
    project_info = {
        "particle_graph": {
            "n_nodes": len(self.particle_nodes),
            "n_connections": len(self.paths),
            "n_tasks": len(self.tasks),
            "n_edges": len(self.particle_edges),
            "n_labels": len(self.particle_labels),
            "graph_extent": [float(e) for e in self.graph_extent],
            "particles": all_particles_dict,
            "particle_parameters": self.particle_parameters,
            "tasks": all_tasks_dict
        }
    }
    project_info["project_setup"] = self.project_setup_dict
    return json.dumps(project_info, indent=2)

  def save_json(self, filepath: str, **kwargs) -> None:
    """
    Save particle graph as JSON file.

    Args:
        filepath (str): filepath to save particle graph to. If the filepath does not end with `.json`, the extension will be added automatically.
    """
    if not filepath.endswith(".json"):
      filepath += ".json"
    with open(filepath, "w", encoding="utf-8") as file:
      file.write(self.to_json())


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
    # load project setup if possible
    project_setup_dict = graph_info.get("project_setup", None)
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
        project_setup = project_setup_dict,
      )
    particles_by_id: dict[int, Graph_Particle] = dict()
    max_particle_id: int = 0
    # load particles one at a time
    for particle_info in particle_dicts:
      # keep track of maximum particle ID
      if particle_info["id"] > max_particle_id:
        max_particle_id = particle_info["id"]
      # (temporarily) remove some particle info that isn't needed for the next step
      connected_particles = particle_info.pop("connected_particles")
      particle_type = particle_info.pop("particle_type")
      
      particle_info["position"] = np.array(particle_info["position"], dtype=np.float16)
      particle_info["bounding_box_size"] = tuple(particle_info["bounding_box_size"])
      # add to graph: Particle Node (Location)
      if particle_type == "Particle_Node":
        particle_info.pop("rotation")
        particle_info.pop("angular_velocity_decay")
        particle_info["target_position"] = np.array(particle_info["target_position"], dtype=np.float16)
        particle = Particle_Node(**particle_info)
        particle_graph.add_particle(particle)
      # add to graph: Particle Edge (partial connection between two locations)
      elif particle_type == "Particle_Edge":
        particle_info.pop("target_position")
        particle = Particle_Edge(**particle_info)
        particle_graph.add_particle(particle)
      # add to graph: Particle Label (text label for locations)
      elif particle_type == "Particle_Label":
        particle_info.pop("bounding_box_size")
        particle_info.pop("target_position")
        particle = Particle_Label(**particle_info)
        particle_graph.add_particle(particle)
      # add connected particles back to particle info
      particle_info["connected_particles"] = connected_particles
      # add particle to ID
      particles_by_id[particle.get_id()] = particle
    # all particles have been added to graph and to particles_by_id
    # next: add connections to particles from their ids
    for particle, particle_info in zip(particles_by_id.values(), particle_dicts):
      particle.set_connected_particles(
          [particles_by_id[connected_particle_id] for connected_particle_id in particle_info["connected_particles"]]
      )
    # set max id
    particle_graph.max_particle_id = max_particle_id
    if "graph_extent" in graph_info["particle_graph"]:
      particle_graph.set_graph_extent(np.array(graph_info["particle_graph"]["graph_extent"]))
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
  get a color that is a gradient between `weight_zero_color` and the given `color`.
  If `weight` is 0, `weight_zero_color` is returned.
  If `weight` is `inf`, `inf_color` is returned.
  otherwise, the color is a gradient between `weight_zero_color` and `color` depending on the weight.
  The minimum weight should be 0. The maximum possible weight should be `max_weight`.

  Args:
      color (str): color as hex string to use for maximum weight
      weight (int): weight of current object
      max_weight (int): maximum weight of all objects
      min_color_factor (float, optional): proportion of color to use for weight 1. Defaults to 0.3.
      weight_zero_color (str, optional): color to use for weight 0. Defaults to "#aaaaaa".
      inf_color (str, optional): color to use for infinite weight. Defaults to "#111111".

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


def add_colorbar(
    ax: plt.Axes,
    min_value: int,
    max_value: int,
    min_color: str,
    max_color: str,
    label: str = "Edge importance",
) -> tuple[plt.Axes, mpl.colorbar.Colorbar]:
  """
  Add a colorbar to the plot.

  Args:
      ax (plt.Axes): matplotlib axes to draw on
      min_value (int): minimum value for the colorbar
      max_value (int): maximum value for the colorbar
      min_color (str): color for min_value
      max_color (str): color for max_value
      label (str, optional): label for the colorbar. Defaults to "Edge importance".
      
  Returns:
      plt.Axes: matplotlib axes containing the colorbar
      mpl.colorbar.Colorbar: colorbar
  """
  n_steps = max_value+1 if isinstance(max_value, int) else max(round(10*max_value), 100)
  cmap = mpl.colors.ListedColormap(
    [
      get_gradient_color(
        max_color,
        value,
        max_value,
        weight_zero_color=min_color)
      for value in np.linspace(min_value, max_value, n_steps, endpoint=True)])
  norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

  colorbar_axes = ax.inset_axes([1.0, 0.1, 0.005, 0.8])
  colorbar = mpl.colorbar.Colorbar(colorbar_axes, cmap=cmap, norm=norm, label=label, location="right")
  return colorbar_axes, colorbar


if __name__ == "__main__":
  locations = [
      "Menegroth",
      "Nargothrond",
      "Hithlum",
  ]
  location_positions = {
    "Menegroth": np.array([0, 6], dtype=np.float16),
    "Nargothrond": np.array([-6, -2], dtype=np.float16),
    "Hithlum": np.array([4, -5], dtype=np.float16),
  }

  paths = [
    ("Menegroth", "Nargothrond", 2, "#dd0000"),
    ("Menegroth", "Hithlum", 3, "#5588ff"),
    ("Nargothrond", "Hithlum", 2, "#22dd22"),
  ]
  tasks = {
    "Menegroth - Nargothrond": TTR_Task(node_names=["Menegroth", "Nargothrond"]),
    "Menegroth - Hithlum": TTR_Task(node_names=["Menegroth", "Hithlum"])
  }

  particle_graph = TTR_Particle_Graph(
      locations,
      paths,
      tasks,
      location_positions)

  n_iter = 100
  dt = 0.1
  print_at = 0

  fig, ax = plt.subplots(dpi=300)
  particle_graph.draw(ax, alpha_multiplier=0.1)
  particle_graph.optimize_layout(iterations=print_at, dt=dt)
  particle_graph.draw(ax, alpha_multiplier=0.5)
  particle_graph.optimize_layout(iterations=n_iter-print_at, dt=dt)
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