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
    self.target_attraction = target_attraction
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
    self.plotted_objects.append(
        ax.add_patch(
            plt.Circle(
                self.position,
                0.5,
                color=color,
                alpha=alpha,
                zorder=zorder,
                picker=True
            )
        )
    )


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