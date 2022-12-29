"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
import numpy as np

from graph_particle import Graph_Particle


class Particle_Edge(Graph_Particle):
  def __init__(self,
        color: str,
        position: np.ndarray = np.array([0, 0]),
        rotation: float = 0,
        ):
    super().__init__(
        position,
        rotation = rotation,
        target_position = None,
        mass = 1,
        bounding_box_size = (4, 1),
        interaction_radius = 3,
    )
    self.color = color