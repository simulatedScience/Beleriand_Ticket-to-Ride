"""
base class for particles in a particle graph
"""
from typing import Tuple

import numpy as np

class graph_particle:
  def __init__(self,
        position: Tuple[float, float],
        rotation: float = 0,
        mass: float = 1,
        bounding_box_size: Tuple[float, float] = (1, 1)):
    self.position = position # position of particle's center
    self.rotation = rotation # rotation of particle in radians
    self.velocity = (0, 0)
    self.angular_velocity = 0
    self.acceleration = (0, 0)
    self.angular_acceleration = 0
    self.mass = mass
    self.bounding_box_size = bounding_box_size # corners of bounding box, in counter-clockwise order
    self.bounding_box = self.update_bounding_box()


  def __str__(self):
    return f"Particle at {self.position}"


  def interact(self, other: "graph_particle"):
    """
    interact with another particle
    forces are calculated from the amount of overlap between the particles' bounding boxes

    args:
      other (graph_particle): other particle to interact with

    returns:
      (Tuple[float, float]): force to apply to self particle
    """
    # get bounding boxes
    self_bounding_box = self.get_bounding_box()
    other_bounding_box = other.get_bounding_box()

    # get overlap
    overlap = self.get_overlap(self_bounding_box, other_bounding_box)

    # calculate force
    force = overlap * 1000

    return force



  def update(self, dt: float):
    """
    update particle's position and velocity
    """
    if np.linalg.norm(self.velocity) > 0 or np.linalg.norm(self.acceleration) > 0 or self.angular_velocity > 0 or self.angular_acceleration > 0:
      self.moved = True

    # update velocity
    self.velocity += self.acceleration * dt
    self.angular_velocity += self.angular_acceleration * dt



  def update_bounding_box(self) -> np.ndarray:
    """
    update bounding box of particle stored in `self.bounding_box`

    returns:
      (np.ndarray): bounding box as numpy array containing it's corners, in counter-clockwise order
    """
    width, height = self.bounding_box_size
    self.bounding_box = np.array([
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
    self.bounding_box = self.bounding_box @ rotation_matrix

    # translate bounding box
    self.bounding_box += self.position

    return self.bounding_box