"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
from matplotlib.patches import Circle

from graph_particle import Graph_Particle


class Particle_Node(Graph_Particle):
  def __init__(self,
        location_name: str,
        id: int,
        color: str = "#222222",
        position: np.ndarray = np.array([0, 0], dtype=np.float16),
        mass: float = 0.1,
        bounding_box_size: tuple = (1, 1),
        target_attraction: float = 0.001,
        target_position: np.ndarray = None,
        interaction_radius: float = 5,
        velocity_decay: float = 0.9999,
        repulsion_strength: float = 1,
        image_file_path: str = None,):
    """
    Initialize a particle node

    Args:
        location_name (str): name of the location
        id (int): unique numeric id of the particle
        color (str, optional): color of the node. Defaults to "#222222".
        position (np.ndarray, optional): position of the particle. Defaults to np.array([0, 0], dtype=np.float16).
        mass (float, optional): mass of the particle. Defaults to 0.1.
        bounding_box_size (tuple, optional): size of the bounding box. Defaults to (1, 1).
        target_attraction (float, optional): attraction force to the target position. Defaults to 0.001.
        target_position (np.ndarray, optional): target position of the particle. Defaults to None.
        interaction_radius (float, optional): radius of the interaction. Defaults to 5.
        velocity_decay (float, optional): decay of the velocity. Defaults to 0.9999.
        repulsion_strength (float, optional): strength of the repulsion force. Defaults to 1.
    """
    if target_position is None:
      target_position = position
    super().__init__(
        id,
        position=position,
        rotation=0,
        target_position=target_position,
        mass=mass,
        bounding_box_size=bounding_box_size,
        interaction_radius=interaction_radius,
        velocity_decay=velocity_decay,
        repulsion_strength=repulsion_strength,
    )
    self.label = location_name
    self.target_attraction = target_attraction
    self.color = color
    self.image_file_path = image_file_path

  def get_adjustable_settings(self) -> dict[str, object]:
    return {
      "position": self.position,
      "rotation": self.rotation,
      "label": self.label,
      "image_file_path": self.image_file_path,
    }

  def set_adjustable_settings(self,
      ax: plt.Axes,
      position: np.ndarray = None,
      rotation: float = None,
      label: str = None,
      image_file_path: str = "") -> None:
    """
    Set the adjustable settings of the edge particle. If a new image file path is given, the image is loaded and drawn. Allowed settings are:
      - position (np.ndarray)
      - rotation (float)
      - label (str)
      - image_file_path (str) or (None)

    Args:
        ax (plt.Axes): Axes to draw the edge on
        position (np.ndarray, optional): new position. Defaults to None (keep current value)
        rotation (float, optional): new rotation. Defaults to None (keep current value)
        label (str, optional): new label. Defaults to None (keep current value)
        image_file_path (str, optional): new image file path. Defaults to "" (keep current value)
    """
    redraw = False
    if position is not None and np.any(np.not_equal(position, self.position)):
      self.set_position(position)
      redraw = True
    if rotation is not None and rotation != self.rotation:
      self.set_rotation(rotation)
      redraw = True
    if label is not None and label != self.label:
      self.label = label
    if image_file_path not in ("", self.image_file_path):
      self.set_image_file_path(image_file_path)
      redraw = True

    if redraw:
      self.erase()
      self.draw(ax)

  def draw(self,
      ax: plt.Axes,
      color: str = "#222222",
      alpha: float = 1,
      zorder: int = 4,
      scale: float = 1,
      override_image_path: str = None,
      movable: bool = True):
    """draw node as circle on given axes

    Args:
        ax (plt.Axes): axes to draw on
        color (str, optional): color of the node. Defaults to "#222222".
        alpha (float, optional): alpha value of the node. Defaults to 1.
        zorder (int, optional): zorder of the node. Defaults to 4.
    """
    if self.image_file_path is None and override_image_path is None: # draw mpl Circle
      self.plotted_objects.append(
          ax.add_patch(
              Circle(
                  self.position,
                  radius=0.5 * scale,
                  color=color,
                  alpha=alpha,
                  zorder=zorder,
                  picker=True,
              )
          )
      )
    else:
      if override_image_path is None: # draw image saved in self.image_file_path
        override_image_path = self.image_file_path
      mpl_image = mpimg.imread(override_image_path)
      img_extent = self.get_extent(scale)
      plotted_image = ax.imshow(mpl_image, extent=img_extent, zorder=zorder, picker=True)
      # rotate image using transformation
      # keep image upright
      image_rotation = self.rotation
      # print(f"image rotation {self.location_1_name}-{self.location_2_name}-{self.path_index}: {image_rotation}")
      plotted_image.set_transform(
        transforms.Affine2D().rotate_around(self.position[0], self.position[1], image_rotation) + ax.transData
      )
      self.plotted_objects.append(plotted_image)
    # set movability of particle
    super().set_particle_movable(movable)

  def get_extent(self, scale):
    """
    Get the extent of the node with the given scale applied.

    Args:
        scale (float): scaling factor

    Returns:
        Tuple[float, float, float, float]: extent of the node as (left, right, bottom, top)
    """
    return (
        self.position[0] - self.bounding_box_size[0] / 2 * scale,
        self.position[0] + self.bounding_box_size[0] / 2 * scale,
        self.position[1] - self.bounding_box_size[1] / 2 * scale,
        self.position[1] + self.bounding_box_size[1] / 2 * scale)


  def set_image_file_path(self, image_file_path: str = None):
    """
    Set node to display the image at the given filepath when drawn.
    If `image_file_path` is `None`, the image will be removed and the node is drawn as a flat colored circle.

    Args:
        image_file_path (str, optional): filepath to image file. Image aspect ratio should match bounding box aspect ratio. Otherwise the image gets stretched. Defaults to None.
    """
    self.image_file_path = image_file_path


  def get_attraction_force(self, other):
    """calculate attraction force to other particle

    Args:
        other (Graph_Particle): other particle

    Returns:
        np.ndarray: attraction force
    """
    return np.zeros(2), self.position


  def set_parameters(self, node_parameters):
    """
    set parameters of the node

    Args:
        node_parameters (dict): dictionary with parameters for the node
    """
    self.color = node_parameters.get("color", self.color)
    self.mass = node_parameters.get("node_mass", self.mass)
    self.target_attraction = node_parameters.get("target_attraction", self.target_attraction)
    self.velocity_decay = node_parameters.get("node_velocity_decay", self.velocity_decay)
    self.interaction_radius = node_parameters.get("interaction_radius", self.interaction_radius)
    self.repulsion_strength = node_parameters.get("repulsion_strength", self.repulsion_strength)

  def add_json_info(self, particle_info: dict) -> dict:
    """
    add node-specific particle information to json dictionary for saving.

    Args:
        particle_info (dict): json dictionary

    Returns:
        dict: json dictionary with node-specific information
    """
    particle_info["color"] = self.color
    particle_info["target_attraction"] = self.target_attraction
    particle_info["location_name"] = self.label
    particle_info["image_file_path"] = self.image_file_path
    return particle_info
