"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle

from graph_particle import Graph_Particle
from particle_node import Particle_Node


class Particle_Edge(Graph_Particle):
  def __init__(self,
        color: str,
        location_1_name: str,
        location_2_name: str,
        id: int,
        position: np.ndarray = np.array([0, 0], dtype=np.float16),
        rotation: float = 0,
        mass: float = 0.1,
        bounding_box_size: tuple = (3.2, 0.8),
        border_color: str = "#555555",
        node_attraction: float = 0.1,
        edge_attraction: float = 0.1,
        interaction_radius: float = 5,
        velocity_decay: float = 0.9999,
        angular_velocity_decay: float = 0.9999,
        repulsion_strength: float = 1,
        path_index: int = 0,
        connection_index: int = 0,
        image_override_filepath: str = "",
        ):
    """
    initialize a particle edge as a part of a connection two nodes `location_1` and `location_2`.  `path_index` is the index of the edge along the path between the two given locations. Counting starts at index 0, up  to path length -1.

    Args:
        color (str): color of the edge
        location_1_name (str): name of the first location
        location_2_name (str): name of the second location
        id (int): unique numeric id of the particle
        position (np.ndarray, optional): position of the edge. Defaults to np.array([0, 0], dtype=np.float16).
        rotation (float, optional): rotation of the edge in radians. Defaults to 0.
        mass (float, optional): mass of the edge. Defaults to 0.1.
        bounding_box_size (tuple, optional): size of the bounding box of the edge. Defaults to (4, 1).
        border_color (str, optional): color of the border of the edge. Defaults to "#555555".
        node_attraction (float, optional): attraction force between the edge and connected nodes. Defaults to 0.1.
        edge_attraction (float, optional): attraction force between the edge and the other edges. Defaults to 0.1.
        interaction_radius (float, optional): interaction radius of the edge for repulsion. Defaults to 5.
        velocity_decay (float, optional): velocity decay factor of the edge. Defaults to 0.9999.
        repulsion_strength (float, optional): repulsion strength of the edge. Defaults to 1.
        path_index (int, optional): index of the edge along the path between the two given locations. Defaults to 1.
        connection_index (int, optional): index of the edge along the connection between the two given locations. Defaults to 0.
    """
    super().__init__(
        id,
        position=position,
        rotation=rotation,
        target_position=None,
        mass=mass,
        bounding_box_size=bounding_box_size,
        interaction_radius=interaction_radius,
        velocity_decay=velocity_decay,
        angular_velocity_decay=angular_velocity_decay,
        repulsion_strength=repulsion_strength,
    )
    self.node_attraction = node_attraction
    self.edge_attraction = edge_attraction
    self.color = color
    self.border_color = border_color
    if location_1_name > location_2_name: # sort location names alphabetically
      location_1_name, location_2_name = location_2_name, location_1_name
    self.location_1_name = location_1_name
    self.location_2_name = location_2_name
    self.path_index = path_index
    self.connection_index = connection_index
    self.image_file_path = None
    self.image_override_filepath = image_override_filepath


  def get_adjustable_settings(self) -> dict[str, object]:
    """
    Get the adjustable settings of the particle.
    Subclasses should override this method to return a dictionary of adjustable settings. Position and rotation should always be included.
    dict keys:
      - position (np.ndarray)
      - rotation (float)
      - color (str)
      - image_file_path (str) or (None)

    returns:
      (dict[str, object]): dictionary of adjustable settings. type of value depends on the key.
    """
    return {
      "position": self.position,
      "rotation": self.rotation,
      "color": self.color,
      "image_file_path": self.image_file_path,
      "image_override_filepath": self.image_override_filepath
    }

  def set_adjustable_settings(self,
      ax: plt.Axes,
      position: np.ndarray = None,
      rotation: float = None,
      color: str = None,
      image_file_path: str = "",
      image_override_filepath: str = "") -> None:
    """
    Set the adjustable settings of the edge particle. If a new image file path is given, the image is loaded and drawn. Allowed settings are:
      - position (np.ndarray)
      - rotation (float)
      - color (str)
      - image_file_path (str) or (None)

    Args:
        ax (plt.Axes): Axes to draw the edge on
        position (np.ndarray, optional): new position. Defaults to None (keep current value)
        rotation (float, optional): new rotation. Defaults to None (keep current value)
        color (str, optional): new color. Defaults to None (keep current value)
        image_file_path (str, optional): new image file path. Defaults to "" (keep current value)
    """
    redraw = False
    if position is not None and np.any(np.not_equal(position, self.position)):
      self.set_position(position)
      redraw = True
    if rotation is not None and rotation != self.rotation:
      self.set_rotation(rotation)
      redraw = True
    if color is not None and color != self.color:
      self.color = color

      if len(self.plotted_objects) > 0 and not redraw:
        for artist in self.plotted_objects:
          if isinstance(artist, Rectangle):
            artist.set_facecolor(self.color)
            artist.set_edgecolor(self.border_color)
    if image_file_path not in ("", self.image_file_path) \
        or not image_override_filepath in (None, self.image_override_filepath):
      self.set_image_file_path(image_file_path, image_override_filepath=image_override_filepath)
      redraw = True

    if redraw:
      self.erase()
      self.draw(ax)

  def set_simulation_parameters(self,
      mass: float = None,
      node_attraction: float = None,
      edge_attraction: float = None,
      interaction_radius: float = None,
      velocity_decay: float = None,
      angular_velocity_decay: float = None,
      repulsion_strength: float = None,
      ) -> None:
    """
    Set the simulation parameters of the edge particle.

    Args:
        mass (float, optional): mass of the particle
        node_attraction (float, optional): attraction force between the edge and connected nodes
        edge_attraction (float, optional): attraction force between the edge and connected edges
        interaction_radius (float, optional): maxumum distance to interaction partners
        velocity_decay (float, optional): velocity decay factor
        angular_velocity_decay (float, optional): angular velocity decay factor
        repulsion_strength (float, optional): repulsion strength
    """
    if mass is not None:
      self.mass = mass
    if node_attraction is not None:
      self.node_attraction = node_attraction
    if edge_attraction is not None:
      self.edge_attraction = edge_attraction
    if interaction_radius is not None:
      self.interaction_radius = interaction_radius
    if velocity_decay is not None:
      self.velocity_decay = velocity_decay
    if angular_velocity_decay is not None:
      self.angular_velocity_decay = angular_velocity_decay
    if repulsion_strength is not None:
      self.repulsion_strength = repulsion_strength

  def get_attraction_forces(self, other_particle) -> Tuple[np.ndarray, np.ndarray]:
    """get attraction force between this particle and the other particle

    Args:
        other_particle (Graph_Particle): other particle

    Returns:
        np.ndarray: attraction force
        np.ndarray: closest point on this particle to the other particle
    """
    if isinstance(other_particle, Particle_Edge):
      return self.get_edge_attraction_force(other_particle)
    else: # other_particle is Particle_Node
      return self.get_node_attraction_force(other_particle)

  def get_edge_attraction_force(self, other_edge: "Particle_Edge") -> Tuple[np.ndarray, np.ndarray]:
    """
    get attraction force between this particle and the other edge depending on the minimum distance between midpoints of edge's bounding boxes shortest edges.
    This uses the helper function `get_edge_midpoints()`

    Args:
        other_edge (Particle_Edge): other edge

    Returns:
        np.ndarray: attraction force vector
        np.ndarray: closest point on this edge to the other edge
    """
    min_distance = np.inf
    closest_points = np.zeros((2, 2))
    for point_1 in self.get_edge_midpoints():
      for point_2 in other_edge.get_edge_midpoints():
        distance = np.linalg.norm(point_1 - point_2)
        if distance < min_distance:
          min_distance = distance
          closest_points[0, :] = point_1
          closest_points[1, :] = point_2
    
    force_direction = (closest_points[1, :] - closest_points[0, :]) / min_distance
    translation_force = self.edge_attraction * self.attraction_from_distance(min_distance) * force_direction
    return translation_force, closest_points[0, :]

  def get_node_attraction_force(self, node: Graph_Particle) -> Tuple[np.ndarray, np.ndarray]:
    """
    get attraction force between this particle and the node depending on the minimum distance between the node and the edge's bounding box's shortest edges.
    This uses the helper function `get_edge_midpoints()`

    Args:
        node (Particle_Node): node

    Returns:
        np.ndarray: attraction force vector
        np.ndarray: closest midpoint to the node (midpoints are on a short side of this edge particle)
    """
    min_distance = np.inf
    closest_point = np.zeros(2)
    for point in self.get_edge_midpoints():
      distance = np.linalg.norm(point - node.position)
      if distance < min_distance:
        min_distance = distance
        closest_point = point

    force_direction = (node.position - closest_point) / min_distance
    translation_force = self.node_attraction * self.attraction_from_distance(min_distance) * force_direction

    force_anchor = closest_point

    return translation_force, force_anchor

  def get_edge_midpoints(self, eps=1e-8):
    """
    get the midpoints of the edges of the bounding box of this edge

    Returns:
        np.ndarray: midpoints of the edges of the bounding box of this edge
    """
    midpoints = np.zeros((2, 2))
    found_n_shortest_edges = 0
    for i in range(4):
      corner_1 = self.bounding_box[i]
      corner_2 = self.bounding_box[(i + 1) % 4]
      if np.linalg.norm(corner_1 - corner_2) - np.min(self.bounding_box_size) < eps:
        midpoints[found_n_shortest_edges, :] = (corner_1 + corner_2) / 2
        found_n_shortest_edges += 1
        if found_n_shortest_edges == 2:
          break
    return midpoints

  def attraction_from_distance(self, distance):
    """
    get attraction force depending on the distance

    Args:
        distance (float): distance

    Returns:
        float: attraction force
    """
    return distance**2 / 2
    # return (np.exp(distance) - 1) / 3


  # def set_parameters(self, edge_parameters):
  #   """
  #   set parameters of this edge

  #   Args:
  #       edge_parameters (dict): dictionary with parameters for this edge
  #   """
  #   self.node_attraction = edge_parameters.get("edge-node", self.node_attraction)
  #   self.edge_attraction = edge_parameters.get("edge-edge", self.edge_attraction)
  #   self.mass = edge_parameters.get("edge_mass", self.mass)
  #   self.color = edge_parameters.get("color", self.color)
  #   self.velocity_decay = edge_parameters.get("velocity_decay", self.velocity_decay)
  #   self.repulsion_strength = edge_parameters.get("repulsion_strength", self.repulsion_strength)
  #   self.interaction_radius = edge_parameters.get("interaction_radius", self.interaction_radius)


  def set_image_file_path(self, image_file_path: str = None, image_override_filepath: str = None):
    """
    Set edge to display the image at the given filepath when drawn.
    If `image_file_path` is `None`, the image will be removed and the edge is drawn as a flat colored rectangle.

    Args:
        image_file_path (str, optional): filepath to image file. Image aspect ratio should match bounding box aspect ratio. Otherwise the image gets stretched. Defaults to None.
        override (str, optional): Sets a filepath for an image to draw instead of the image at `image_file_path`. Defaults to None.
            - if a nonempty string is given, that is used as the new override filepath
            - if an empty string is given, the override filepath is deleted and the image at `image_file_path` will be drawn
            - if `None` is given, the override filepath is not changed
    """
    if image_override_filepath is not None:
      self.image_override_filepath = image_override_filepath
    self.image_file_path = image_file_path


  def draw(self,
      ax: plt.Axes,
      color: str = None,
      border_color: str = None,
      alpha: float = 0.7,
      zorder: int = 2,
      movable: bool = True) -> None:
    """
    draw this edge as a rectangle
    If there is an image stored in self.image_file_path or self.image_override_filepath, draw the edge as that. The override_filpath takes priority.

    Args:
        ax (plt.Axes): matplotlib axes
        color (str, optional): color. Defaults to None.
        alpha (float, optional): alpha. Defaults to 0.7.
        zorder (int, optional): zorder. Defaults to 4.
    """
    if self.image_file_path is None: # draw mpl Rectangle
      if color is None:
        color = self.color
      if border_color is None:
        # if particle has no  border color, initialize it as gray
        try:
          border_color = self.border_color
        except AttributeError:
          border_color = "#555555"
          self.border_color = border_color
      super().draw_bounding_box(ax, color, border_color, alpha, zorder, movable)
    else:
      if self.image_override_filepath: # use override image
        mpl_image = mpimg.imread(self.image_override_filepath)
      else: # use image at `self.image_file_path`
        mpl_image = mpimg.imread(self.image_file_path)
      edge_extent = (
        self.position[0] - self.bounding_box_size[0] / 2,
        self.position[0] + self.bounding_box_size[0] / 2,
        self.position[1] - self.bounding_box_size[1] / 2,
        self.position[1] + self.bounding_box_size[1] / 2)
      plotted_image = ax.imshow(mpl_image, extent=edge_extent, zorder=zorder, picker=True)
      # rotate image using transformation
      # keep image upright
      image_rotation = self.get_image_rotation()
      # print(f"image rotation {self.location_1_name}-{self.location_2_name}-{self.path_index}: {image_rotation}")
      plotted_image.set_transform(
        transforms.Affine2D().rotate_around(self.position[0], self.position[1], image_rotation) + ax.transData
      )
      self.plotted_objects.append(plotted_image)
      # set movability of particle
      super().set_particle_movable(movable)

  # def highlight(self, ax: plt.Axes,  highlight_color: str = "#cc00cc"):
  #   self.erase()
  #   self.draw(ax=ax, border_color=highlight_color)

  # def remove_highlight(self):
  #   return super().remove_highlight()

  def get_image_rotation(self) -> float:
    """
    calculate image rotation based on `self.rotation` and the location of the nodes this edge is connected to.

    Returns:
        float: rotation in radians
    """
    # find noes this edge is connected to
    visited_particle_ids = {self.get_id()}
    connected_nodes = [self, self]
    for i in range(2):
      while True: # find a connected node
        connected_index = 0
        new_node = connected_nodes[i].connected_particles[connected_index]
        while True: # find a connected node that has not been visited yet
          if not new_node.get_id() in visited_particle_ids:
            break
          connected_index += 1
          if connected_index >= len(connected_nodes[i].connected_particles):
            raise ValueError("Could not find a connected particle that has not been visited yet. Ensure that the graph is connected properly.")
          new_node = connected_nodes[i].connected_particles[connected_index]
        connected_nodes[i] = new_node
        visited_particle_ids.add(connected_nodes[i].get_id())
        if isinstance(connected_nodes[i], Particle_Node):
          break
    # calculate normal vector of direct connection between nodes
    node_1_position = connected_nodes[0].position
    node_2_position = connected_nodes[1].position
    node_1_to_node_2 = node_2_position - node_1_position
    norm = np.linalg.norm(node_1_to_node_2)
    if norm == 0:
      print(f"WARNING: edge {self.location_1_name}-{self.location_2_name} has length zero. Using original rotation of edge particle.")
      return self.rotation
    node_1_to_node_2 = node_1_to_node_2 / norm
    normal_vector = np.array([-node_1_to_node_2[1], node_1_to_node_2[0]], dtype=np.float16) # rotate by 90°
    # ensure that normal vector direction is always pointing upwards
    if normal_vector[1] < 0:
      normal_vector = -normal_vector
    # if normal vector is to the right of the current rotation vector, rotate by 180°
    # this aligns the image with the normal vector
    if np.cross(normal_vector, np.array([np.cos(self.rotation), np.sin(self.rotation)], dtype=np.float16)) > 0:
      return self.rotation + np.pi
    return self.rotation


  def add_json_info(self, particle_info: dict) -> dict:
    """
    add edge-specific particle information to json dictionary for saving.

    Args:
        particle_info (dict): json dictionary

    Returns:
        dict: json dictionary with edge-specific information
    """
    particle_info["node_attraction"] = self.node_attraction
    particle_info["edge_attraction"] = self.edge_attraction
    particle_info["color"] = self.color
    particle_info["border_color"] = self.border_color
    particle_info["location_1_name"] = self.location_1_name
    particle_info["location_2_name"] = self.location_2_name
    particle_info["path_index"] = self.path_index
    particle_info["connection_index"] = self.connection_index
    particle_info["image_override_filepath"] = self.image_override_filepath
    return particle_info

def get_adjacent_nodes(particle_edge: Particle_Edge) -> Tuple[List[Particle_Node], int]:
  """
  Find the two nodes that are connected to the given edge (directly or indirectly via connected edges).
  These nodes should always correspond to `particle_edge.location_1_name` and `particle_edge.location_2_name`.

  Findnodes by traversing the graph starting from the given edge and following the connected edges until a node is reached. Then follow the other connected particles and repeat the process until the second node has been found.

  Args:
      particle_edge (Particle_Edge): edge to find connected nodes for

  Raises:
      ValueError: If the graph is not connected properly, this function will not be able to find the second node and raise an error.

  Returns:
      List[Particle_Node]: list of two nodes that are connected to the given edge
      int
  """
  # find noes this edge is connected to
  visited_particle_ids = {particle_edge.get_id()}
  connected_nodes = [particle_edge, particle_edge]
  edge_length = 0
  for i in range(2):
    while True:
      connected_index = 0
      new_node = connected_nodes[i].connected_particles[connected_index]
      while True: # find a connected node that has not been visited yet
        if not new_node.get_id() in visited_particle_ids:
          break
        connected_index += 1
        if connected_index >= len(connected_nodes[i].connected_particles):
          raise ValueError("Could not find connected particle that has not been visited yet. Ensure that the graph is connected properly.")
        new_node = connected_nodes[i].connected_particles[connected_index]
      connected_nodes[i] = new_node
      visited_particle_ids.add(connected_nodes[i].get_id())
      if isinstance(connected_nodes[i], Particle_Node):
        break

    return connected_nodes, edge_length