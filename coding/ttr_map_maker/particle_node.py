"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
import numpy as np
import matplotlib.pyplot as plt

from graph_particle import Graph_Particle


class Particle_Node(Graph_Particle):
  def __init__(self,
        label: str,
        position: np.ndarray = np.array([0, 0]),
        mass: float = 0.1,
        target_attraction: float = 0.001,
        target_position: np.ndarray = None,
        interaction_radius: float = 5,
        velocity_decay: float = 0.9999,
        repulsion_strength: float = 1,):
    if target_position is None:
      target_position = position
    super().__init__(
        position,
        rotation = 0,
        target_position = target_position,
        mass = mass,
        bounding_box_size = (1, 1),
        interaction_radius = interaction_radius,
        velocity_decay = velocity_decay,
        repulsion_strength=repulsion_strength,
    )
    self.label = label
    self.color = "#222222"


  def draw(self,
      ax: plt.Axes,
      color: str = "#222222",
      alpha: float = 1,
      zorder: int = 4):
    """draw node as circle on given axes

    Args:
        ax (plt.Axes): axes to draw on
        color (str, optional): color of the node. Defaults to "#222222".
        alpha (float, optional): alpha value of the node. Defaults to 1.
        zorder (int, optional): zorder of the node. Defaults to 4.
    """
    ax.add_patch(plt.Circle(self.position, 0.5, color = color, alpha = alpha, zorder = zorder))


  def get_attraction_force(self, other):
    """calculate attraction force to other particle

    Args:
        other (Graph_Particle): other particle

    Returns:
        np.ndarray: attraction force
    """
    return np.zeros(2)