"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
import numpy as np

from graph_particle import Graph_Particle


class Particle_Node(Graph_Particle):
  def __init__(self,
        label: str,
        position: np.ndarray = np.array([0, 0])):
    super().__init__(
        position,
        rotation = 0,
        target_position = position,
        mass = 0.1,
        bounding_box_size = (1, 1),
        interaction_radius = 3,
    )
    self.label = label
    self.color = "#222222"