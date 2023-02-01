"""
base class for particles in a particle graph
"""
from typing import Tuple, List
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon

class Graph_Particle:
  def __init__(self,
        id: int,
        position: np.ndarray = np.array([0, 0]),
        rotation: float = 0,
        target_position: np.ndarray = None,
        mass: float = 1,
        bounding_box_size: Tuple[float, float] = (1, 1),
        interaction_radius: float = 5,
        velocity_decay: float = 0.999,
        angular_velocity_decay: float = 0.999,
        repulsion_strength: float = 1,):
    """
    initialize a particle
    moment of inertia is calculated from bounding box size and mass assuming a uniform density and a rectangular shape

    args:
      id (int): unique numeric id of the particle
      position (np.ndarray): position of particle's center
      rotation (float): rotation of particle in radians, counter-clockwise from positive x-axis
      mass (float): mass of particle
      bounding_box_size (Tuple[float, float]): size of particle's bounding box (width, height)
      interaction_radius (float): particle's interaction radius. Particles will only interact with other particles where the distance between their bounding boxes is less than `interaction_radius`.
      velocity_decay (float): factor by which the particle's velocity is multiplied each time step. This is used to simulate friction.
    """
    self.id = id
    # translation properties
    self.position = position.astype(np.float64) # position of particle's center
    self.velocity = np.zeros(2)
    self.acceleration = np.zeros(2)
    self.mass = mass
    # rotation properties
    self.rotation = rotation # rotation of particle in radians
    self.angular_velocity = 0
    self.angular_acceleration = 0
    self.inertia = mass * (bounding_box_size[0] ** 2 + bounding_box_size[1] ** 2) / 12 # moment of inertia

    self.repulsion_strength = repulsion_strength
    self.velocity_decay = velocity_decay # factor by which the particle's velocity is multiplied each time step. This is used to simulate friction.
    self.angular_velocity_decay = angular_velocity_decay # factor by which the particle's angular velocity is multiplied each time step. This is used to simulate friction.
    
    self.connected_particles = list() # particles that this particle is attracted to
    self.attraction_strength = 0.001 # strength of attraction force between this particle and its connected particles
    self.target_position: np.ndarray = target_position # target location for this particle
    # bounding box properties
    self.bounding_box_size: Tuple[float, float] = bounding_box_size # size of particle's bounding box (width, height)
    self.bounding_box, self.bounding_box_polygon = self.update_bounding_box() # bounding box as a list of points and as a shapely polygon

    self.plotted_objects: list = list() # objects that are plotted a graph
    # variables for Verlet list algorithm
    self.neighbors = []
    self.interaction_radius = interaction_radius


  def set_connected_particles(self, particles: List["Graph_Particle"]):
    """
    Set the particles that this particle is connected to.
    There is an attraction force between the particle and each of its connected particles. This attraction force is calculated in the `interact` method and is not symmetric. For example a node can attract a label, but the label may not attract the node.

    args:
      particles (List[Graph_Particle]): list of particles that this particle is connected to
    """
    self.connected_particles = particles


  def add_connected_particle(self, particle: "Graph_Particle"):
    """
    Add a particle that this particle is connected to.
    There is an attraction force between the particle and each of its connected particles. This attraction force is calculated in the `interact` method and is not symmetric. For example a node can attract a label, but the label may not attract the node.

    args:
      particle (Graph_Particle): particle that this particle is connected to
    """
    if not self.connected_particles: # if list is empty
      self.connected_particles = [particle]
    else:
      self.connected_particles.append(particle)


  def set_neighbors(self, neighbors: List["Graph_Particle"]):
    """
    Set the neighbors of this particle.
    This is used for the Verlet list algorithm.

    args:
      neighbors (List[Graph_Particle]): list of neighbors of this particle
    """
    self.neighbors = neighbors


  def get_neighbors(self):
    """
    Get the neighbors of this particle.
    This is used for the Verlet list algorithm.

    returns:
      (List[Graph_Particle]): list of neighbors of this particle
    """
    return self.neighbors

  def set_position(self, position: np.ndarray):
    """
    Set the position of the particle.

    args:
      position (np.ndarray): new position of particle
    """
    self.position = position
    self.bounding_box, self.bounding_box_polygon = self.update_bounding_box()

  def set_rotation(self, rotation: float):
    """
    Set the rotation of the particle.

    args:
      rotation (float): new rotation of particle
    """
    self.rotation = rotation % (2*np.pi)
    self.bounding_box, self.bounding_box_polygon = self.update_bounding_box()

  def get_rotation(self):
    """
    Get the rotation of the particle.

    returns:
      (float): rotation of particle
    """
    return self.rotation

  def __str__(self):
    return f"Particle at\t {self.position} with mass\t {self.mass} and inertia\t {self.inertia}."


  def reset_acceleration(self):
    """
    Reset acceleration and angular acceleration to zero.
    """
    self.acceleration = np.zeros(2)
    self.angular_acceleration = 0


  def interact(self, other: "Graph_Particle") -> Tuple[np.ndarray, float]:
    """
    Interact with another particle.
    - Only interact with particles where the distance between their center points is less than `self.interaction_radius`.
    - Repulsion forces are calculated from the amount of overlap between the particles' bounding boxes using the `get_repulsion_forces` method.
    - Attraction forces are calculated using the `get_attraction_forces` method.
    - Attraction forces are only calculated if `other` is in `self.connected_particles`.
    - This interaction is not symmetric, so `self` particle will be affected by `other` particle differently than `other` particle will be affected by `self` particle.

    args:
      other (Graph_Particle): other particle to interact with

    returns:
      (np.ndarray) acceleration to apply to `self` particle
      (float) angular acceleration to apply to `self` particle
    """
    # check if particles are close enough to interact
    if np.linalg.norm(self.position - other.position) <= self.interaction_radius:
      # get repulsion force
      repulsion_force, repulsion_anchor = self.get_repulsion_forces(other)
      repulsion_force_radial, repulsion_torque = split_force(repulsion_force, repulsion_anchor, self.position)
      # get attraction force
      if other in self.connected_particles:
        attraction_force, attraction_anchor = self.get_attraction_forces(other)
        attraction_force_radial, attraction_torque = split_force(attraction_force, attraction_anchor, self.position)
      else:
        attraction_force_radial = np.zeros(2)
        attraction_torque = 0
      # split attraction force into translation and rotation components
      # add forces to acceleration
      self.acceleration += (self.repulsion_strength * repulsion_force_radial + attraction_force_radial) / self.mass
      self.angular_acceleration += (self.repulsion_strength * repulsion_torque + attraction_torque) / self.inertia


  def get_repulsion_forces(self, other: "Graph_Particle") -> Tuple[np.ndarray, np.ndarray]:
    """
    Get repulsion force between this particle and another particle.
    Forces are calculated from the amount of overlap between the particles' bounding boxes.


    args:
      other (Graph_Particle): other particle to interact with

    returns:
      (np.ndarray) translation repulsion force to apply to `self` particle
      (np.ndarray) rotation repulsion force to apply to `self` particle
    """
    # get overlap
    self_polygon = self.get_bounding_box_polygon()
    other_polygon = other.get_bounding_box_polygon()
    overlap_center, overlap_area = get_box_overlap(self_polygon, other_polygon)

    if not overlap_area > 0: # no overlap => no repulsion force
      return np.zeros(2), 0

    overlap_vector = overlap_center - self.position # vector from self center to center of overlap area
    translation_force = -overlap_vector * overlap_area

    return translation_force, overlap_center


  def get_attraction_forces(self, other: "Graph_Particle") -> Tuple[np.ndarray, np.ndarray]:
    """
    Get attraction force between this particle and another particle.
    Forces are calculated from the minimum distance between the particles' bounding boxes.

    args:
      other (Graph_Particle): other particle to interact with

    returns:
      (np.ndarray) translation attraction force to apply to `self` particle
      (np.ndarray) rotation attraction force to apply to `self` particle
    """
    # get distance between particles
    distance = np.linalg.norm(self.position - other.position)
    # calculate forces
    attraction_vector = other.position - self.position
    translation_force = attraction_vector * distance

    return translation_force, other.position

  def update(self, dt: float) -> float:
    """
    update particle's position and velocity

    args:
      dt (float): time step

    returns:
      (float) distance traveled
    """
    if np.linalg.norm(self.velocity) > 0 or \
          np.linalg.norm(self.acceleration) > 0 or \
          self.angular_velocity > 0 or \
          self.angular_acceleration > 0:
      # update velocity
      if self.target_position is not None:
        # if target location is set, apply force towards target location
        target_vector = self.target_position - self.position
        if np.linalg.norm(target_vector) > 0:
          target_vector /= np.linalg.norm(target_vector)
        self.acceleration += target_vector * self.attraction_strength / self.mass
      self.velocity += self.acceleration * dt
      self.angular_velocity += self.angular_acceleration * dt

      # update position
      self.position += self.velocity * dt
      self.rotation += self.angular_velocity * dt

      # update bounding box
      self.bounding_box, self.bounding_box_polygon = self.update_bounding_box()

      # reduce velocities using friction
      self.velocity *= self.velocity_decay * dt
      self.angular_velocity *= self.angular_velocity_decay * dt

    return np.linalg.norm(self.velocity) * dt


  def get_bounding_box(self) -> np.ndarray:
    """
    get bounding box of particle stored in `self.bounding_box`

    returns:
      (np.ndarray): bounding box as numpy array containing it's corners, in counter-clockwise order
    """
    return self.bounding_box


  def get_bounding_box_polygon(self) -> Polygon:
    """
    get bounding box of particle stored in `self.bounding_box_polygon`

    returns:
      (shapely.geometry.Polygon): bounding box as shapely polygon
    """
    return self.bounding_box_polygon


  def update_bounding_box(self) -> Tuple[np.ndarray, Polygon]:
    """
    recalculate bounding box of particle from `self.position`, `self.rotation`, and `self.bounding_box_size`

    returns:
      (np.ndarray): bounding box as numpy array containing it's corners, in counter-clockwise order
      (shapely.geometry.Polygon): bounding box as shapely polygon
    """
    width, height = self.bounding_box_size
    bounding_box = np.array([
      [width / 2, height / 2],
      [width / 2, -height / 2],
      [-width / 2, -height / 2],
      [-width / 2, height / 2]
    ])

    # rotate bounding box
    rotation_matrix = np.array([
      [np.cos(self.rotation), -np.sin(self.rotation)],
      [np.sin(self.rotation), np.cos(self.rotation)]
    ])
    bounding_box = bounding_box @ rotation_matrix

    # translate bounding box
    bounding_box += self.position

    return bounding_box, Polygon(bounding_box)


  def draw_bounding_box(self,
      ax: plt.Axes, 
      color: str = None,
      border_color: str = None,
      alpha: float = 0.3,
      zorder: int = 3,
      movable: bool = True) -> None:
    """
    draw bounding box of particle on `ax`

    args:
      ax (matplotlib.axes.Axes): axis to draw on
      color (str): color of particle
      alpha (float): alpha value of particle
      zorder (int): zorder of particle
      movable (bool): whether particle is movable via drag & drop
    """
    if border_color is None:
      border_color = color
    if color is None:
      print("Warning: Particle not shown since no color was given")
      return
    polygon_patch = Rectangle(
        self.position - np.array([self.bounding_box_size[0] / 2, self.bounding_box_size[1] / 2]),
        width=self.bounding_box_size[0],
        height=self.bounding_box_size[1],
        angle=np.rad2deg(self.rotation),
        rotation_point="center",
        facecolor=color,
        edgecolor=border_color,
        alpha=alpha,
        zorder=zorder,
        picker=True,
    )
    self.plotted_objects.append(ax.add_patch(polygon_patch))
    self.set_particle_movable(movable)

  def draw(self,
      ax: plt.Axes,
      color: str = "",
      alpha: float = 0.7,
      zorder: int = 4,
      movable: bool = True):
    """
    draw particle on `ax`. This should be overwritten by subclasses.

    args:
      ax (matplotlib.axes.Axes): axis to draw on
      color (str): color of particle
      alpha (float): alpha value of particle
      zorder (int): zorder of particle
    """
    print("Warning: draw() method of Particle class should be overwritten by subclasses. Falling back to draw_bounding_box().")
    self.draw_bounding_box(ax, color, alpha, zorder, movable)

  def highlight(self, ax: plt.Axes, highlight_color: str = "#cc00cc"):
    """
    highlight particle `self` 

    Args:
        highlilght_color (str, optional): color used to highlight the particle. Defaults to "#cc00cc".
    """
    if not self.plotted_objects: # no artists drawn -> can't highlight anything
      return
    # print("Warning: using highlight method of `Graph_Particle`. This method should be overwritten by subclasses for better performance and more control.")
    for artist in self.plotted_objects:
      artist.set_path_effects([
          path_effects.Stroke(linewidth=20, foreground=highlight_color),
          path_effects.Normal()])

  def remove_highlight(self, ax: plt.Axes):
    """
    remove highlight of particle `self` and set `self.plotted_objects` appropriately.
    """
    for artist in self.plotted_objects:
      artist.set_path_effects([])
    # gid = self.plotted_objects[0].get_gid()
    # self.erase()
    # self.draw(ax, movable=gid)

  def erase(self):
    """
    erase drawn particle
    """
    for obj in self.plotted_objects:
      obj.remove()
    self.plotted_objects = []

  def set_particle_movable(self, movable: bool = None):
    """
    Toggle whether a particle can be moved with drag and drop. This is done by setting the group id of the artists to `movable`.

    Args:
        movable (bool, optional): whether to enable picking. Defaults to None (toggle)
    """
    for obj in self.plotted_objects:
      if movable is None: # toggle movability
        current_gid = obj.get_gid()
        movable_gid = "movable" if current_gid is None else None
      else: # set movability
        movable_gid = "movable" if movable else None
      obj.set_gid(movable_gid)


  def to_json(self) -> str:
    """
    converts a particle to a JSON string representation containing all information required to recreate the particle.
    This requires particle ids to be set.

    args:
      file_path (str): path to json file

    
    """
    particle_info = self.to_dict()
    return json.dumps(particle_info, indent=2)

  def to_dict(self) -> dict:
    particle_info = {
      "particle_type": self.__class__.__name__,
      "id": self.id,
      "position": self.position.tolist(),
      "rotation": self.rotation,
      "mass": self.mass,
      "bounding_box_size": list(self.bounding_box_size),
      "velocity_decay": self.velocity_decay,
      "angular_velocity_decay": self.angular_velocity_decay,
      "interaction_radius": self.interaction_radius,
      "connected_particles": [p.get_id() for p in self.connected_particles],
      "repulsion_strength": self.repulsion_strength,
    }
    if self.target_position is not None:
      particle_info["target_position"] = self.target_position.tolist()
    else:
      particle_info["target_position"] = None
    particle_info = self.add_json_info(particle_info)
    return particle_info

  def add_json_info(self, particle_info: dict) -> dict:
    """
    add additional information to the particle info dictionary. This should be overwritten by subclasses to add subclass-specific information.

    args:
      particle_info (dict): dictionary containing particle information

    returns:
      (dict): dictionary containing particle information
    """
    print("Warning: add_json_info() method of Particle class should be overwritten by subclasses. Falling back to default implementation.")
    return particle_info

  def set_id(self, id: int) -> None:
    """
    set id of particle

    args:
      id (int): id of particle
    """
    self.id = id

  def get_id(self) -> int:
    """
    get id of particle

    returns:
      (int): id of particle
    """
    return self.id
      

def get_box_overlap(box1_poly: Polygon, box2_poly: Polygon) -> Tuple[np.ndarray, float]:
    """
    get overlap between two bounding boxes

    args:
      box1_poly (shapely.geometry.Polygon): first bounding box as shapely polygon
      box2_poly (shapely.geometry.Polygon): second bounding box as shapely polygon

    returns:
      (Tuple[np.ndarray, float]): center of overlap and area of overlap
    """
    # Calculate overlapping polygon
    overlap_poly = box1_poly.intersection(box2_poly)

    # Check if there is no overlap
    if overlap_poly.is_empty:
        return np.zeros((2,)), 0.0

    # Calculate area of overlap
    overlap_area = overlap_poly.area

    # Calculate center of overlap
    overlap_center = np.array([overlap_poly.centroid.x, overlap_poly.centroid.y])

    return overlap_center, overlap_area

def split_force(
        force: np.ndarray,
        anchor: np.ndarray,
        center_of_mass: np.ndarray,
        eps=1e-8) -> Tuple[np.ndarray, float]:
    """
    split given force into radial and tangential components to calculate translation and rotation forces

    args:
      force (np.ndarray): force vector
      anchor (np.ndarray): anchor point
      center_of_mass (np.ndarray): center of mass of particle

    returns:
      (np.ndarray) translation force
      (float) torque ("rotation force")
    """
    # get vector from center of mass to anchor
    anchor_vector = anchor - center_of_mass
    # get radial component of force
    radial_component = np.dot(force, anchor_vector) * anchor_vector / np.linalg.norm(anchor_vector)
    # get tangential component of force
    tangential_component = force - radial_component
    # get torque
    torque = np.linalg.norm(anchor_vector) * np.linalg.norm(tangential_component)
    if np.cross(anchor_vector, tangential_component) < 0:
      torque *= -1

    return radial_component, torque


if __name__ == "__main__":
  # create particles
  particle1 = Graph_Particle(position=np.array([5, 3]), rotation=0, mass=1, bounding_box_size=np.array([8, 2]))
  particle2 = Graph_Particle(position=np.array([0.5, 0]), rotation=np.pi/2, mass=1, bounding_box_size=np.array([8, 2]))

  particle1.add_connected_particle(particle2)
  particle2.add_connected_particle(particle1)

  # plot before interaction
  fig, ax = plt.subplots()
  ax.set_aspect('equal')
  particle1.draw_bounding_box(ax, color="#ff0000", alpha=0.1)
  particle2.draw_bounding_box(ax, color="#0000ff", alpha=0.1)

  print("before interaction")
  print(f"particle1.position: {particle1.position}")
  print(f"particle1.rotation: {particle1.rotation}")
  print(f"particle2.position: {particle2.position}")
  print(f"particle2.rotation: {particle2.rotation}")

  print(f"bounding box 1: {particle1.get_bounding_box()}")
  # interact
  n_steps = 500
  dt = 2e-1
  position_1 = np.zeros((n_steps, 2))
  position_2 = np.zeros((n_steps, 2))
  for n in range(n_steps):
    particle1.reset_acceleration()
    particle2.reset_acceleration()
    translation_acceleration_1, rot_acceleration_1 = particle1.interact(particle2)
    translation_acceleration_2, rot_acceleration_2 = particle2.interact(particle1)
    particle1.update(dt)
    particle2.update(dt)
    position_1[n] = particle1.position
    position_2[n] = particle2.position

  ax.plot(position_1[:, 0], position_1[:, 1], color="#ff0000", label="particle 1 center")
  ax.plot(position_2[:, 0], position_2[:, 1], color="#0000ff", label="particle 2 center")

  print("after interaction")
  print(f"particle1.position: {particle1.position}")
  print(f"particle1.rotation: {particle1.rotation}")
  print(f"particle2.position: {particle2.position}")
  print(f"particle2.rotation: {particle2.rotation}")

  print(f"bounding box 1: {particle1.get_bounding_box()}")
  # plot after interaction
  particle1.draw_bounding_box(ax, color="#aa0000")
  particle2.draw_bounding_box(ax, color="#0000aa")

  ax.set_xlim(-10, 10)
  ax.set_ylim(-10, 10)
  ax.legend()
  plt.show()