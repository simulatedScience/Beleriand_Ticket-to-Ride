"""
base class for particles in a particle graph
"""
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import Polygon

class Graph_Particle:
  def __init__(self,
        position: np.ndarray = np.array([0, 0]),
        rotation: float = 0,
        target_position: np.ndarray = None,
        mass: float = 1,
        bounding_box_size: Tuple[float, float] = (1, 1),
        interaction_radius: float = 5):
    """
    initialize a particle
    moment of inertia is calculated from bounding box size and mass assuming a uniform density and a rectangular shape

    args:
      position (np.ndarray): position of particle's center
      rotation (float): rotation of particle in radians, counter-clockwise from positive x-axis
      mass (float): mass of particle
      bounding_box_size (Tuple[float, float]): size of particle's bounding box (width, height)
      interaction_radius (float): particle's interaction radius. Particles will only interact with other particles where the distance between their bounding boxes is less than `interaction_radius`.
    """
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
    
    self.velocity_decay = 0.9999 # velocity decay factor
    self.connected_particles = list() # particles that this particle is attracted to
    self.attraction_strength = 0.001 # strength of attraction force between this particle and its connected particles
    self.target_location = target_position
    # bounding box properties
    self.bounding_box_size = bounding_box_size # corners of bounding box, in counter-clockwise order
    self.bounding_box, self.bounding_box_polygon = self.update_bounding_box()

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
      repulsion_force, repulsion_torque = self.get_repulsion_forces(other)
      # get attraction force
      if other in self.connected_particles:
        attraction_force, attraction_torque = self.get_attraction_forces(other)
      else:
        attraction_force = np.zeros(2)
        attraction_torque = 0
      # add forces to acceleration
      self.acceleration += (repulsion_force + attraction_force) / self.mass
      self.angular_acceleration += (repulsion_torque + attraction_torque) / self.inertia


  def old_interact(self, other: "Graph_Particle") -> Tuple[np.ndarray, float]:
    """
    old interact method
    """
    # check if particles are close enough to interact
    self_polygon = self.get_bounding_box_polygon()
    other_polygon = other.get_bounding_box_polygon()
    min_distance = self_polygon.distance(other_polygon)
    # REPULSION
    if min_distance <= self.interaction_radius:
      # if distance between bounding boxes is within interaction radius
      # get overlap
      overlap_center, overlap_area = get_box_overlap(self_polygon, other_polygon)
      if overlap_area > 0:
        # calculate forces
        overlap_vector = overlap_center - self.position
        translation_force = -overlap_vector * overlap_area
        translation_acceleration = translation_force / self.mass
        # rotation force perpendicular to translation force
        center_center = other.position - self.position
        rotation_axis = np.array([0, 0, np.cross(center_center, overlap_vector)])
        if np.linalg.norm(rotation_axis) > 0:
          rotation_axis /= np.linalg.norm(rotation_axis)
        
        tangential_force = np.cross(rotation_axis, overlap_vector) * overlap_area
        torque = np.linalg.norm(overlap_vector) * np.linalg.norm(tangential_force) * (-rotation_axis[2])
        angular_acceleration = torque / self.inertia

        self.acceleration += translation_acceleration
        self.angular_acceleration += angular_acceleration

    # ATTRACTION TO CONNECTED PARTICLES
    if other in self.connected_particles:
      # if other particle is connected to this particle, apply attraction force towards other particle
      self.acceleration += self.attraction_strength * min_distance / self.mass
      # encourage similar orientation of connected particles by applying torque
      self.angular_acceleration += (other.rotation - self.rotation) * self.attraction_strength / self.inertia
    # return translation_acceleration, angular_acceleration


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
    # rotation force perpendicular to translation force
    center_center = other.position - self.position
    rotation_axis = np.array([0, 0, np.cross(center_center, overlap_vector)])
    if np.linalg.norm(rotation_axis) > 0: # normalize rotation axis if it is defined
      rotation_axis /= np.linalg.norm(rotation_axis)
    
    tangential_force = np.cross(rotation_axis, overlap_vector) * overlap_area
    torque = np.linalg.norm(overlap_vector) * np.linalg.norm(tangential_force) * (-rotation_axis[2])

    return translation_force, torque


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
      if self.target_location is not None:
        # if target location is set, apply force towards target location
        target_vector = self.target_location - self.position
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
      self.angular_velocity *= self.velocity_decay * dt

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
      color: str = "",
      alpha: float = 0.3,
      zorder: int = 3):
    """
    draw bounding box of particle on `ax`

    args:
      ax (matplotlib.axes.Axes): axis to draw on
      color (str): color of particle
      alpha (float): alpha value of particle
      zorder (int): zorder of particle
    """
    polygon_patch = PolygonPatch(
        self.get_bounding_box(),
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(polygon_patch)


  def draw(self,
      ax: plt.Axes,
      color: str = "",
      alpha: float = 0.7,
      zorder: int = 4):
    """
    draw particle on `ax`. This should be overwritten by subclasses.

    args:
      ax (matplotlib.axes.Axes): axis to draw on
      color (str): color of particle
      alpha (float): alpha value of particle
      zorder (int): zorder of particle
    """
    print("Warning: draw() method of Particle class should be overwritten by subclasses. Falling back to draw_bounding_box().")
    self.draw_bounding_box(ax, color, alpha, zorder)

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