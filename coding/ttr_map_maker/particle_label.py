"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
import os

from PIL import ImageFont
import numpy as np
import matplotlib.pyplot as plt

from graph_particle import Graph_Particle


class Particle_Label(Graph_Particle):
  def __init__(self,
        label: str,
        position: np.ndarray = np.array([0, 0]),
        fontsize: int = 20,
        font: str = "beleriand_ttr\\MiddleEarth.ttf"):
    width, height = self.get_label_size(label, fontsize, font)
    super().__init__(
        position,
        rotation = 0,
        target_position = position,
        mass = 1,
        bounding_box_size = (width, height),
        interaction_radius = 3,
    )
    self.label = label
    self.color = "#222222"


  def draw(self, 
      ax: plt.Axes,
      color: str = "#222222",
      alpha: float = 1,
      zorder: int = 4):
    """
    draw the particle on the canvas

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        color (str, optional): color of the particle. Defaults to "".
        alpha (float, optional): alpha value of the particle. Defaults to 0.7.
        zorder (int, optional): zorder of the particle. Defaults to 4.
    """
    ax.text(
        self.position[0],
        self.position[1],
        self.label,
        color = color,
        alpha = alpha,
        zorder = zorder,
        horizontalalignment = "center",
        verticalalignment = "center",
        fontsize = 20,
        fontproperties = ImageFont.truetype("beleriand_ttr\\MiddleEarth.ttf", 20),
    )


  def get_label_size(self, label: str, fontsize: int, font: str):
    """
    get size of a label with a given font size

    Args:
        label (str): _description_
        fontsize (int): _description_

    Returns:
        _type_: _description_
    """
    font = ImageFont.truetype(font, fontsize)
    width, height = font.getsize(label)
    # normalize height
    width /= height
    height = 1
    return width, height


if __name__ == "__main__":
  locations = [
      "Taur-im-Duinath",
      "Menegroth",
      "Nargothrond",
      "Nogrod",
  ]
  for location in locations:
    label = Particle_Label(location)
    print(f"{location}: {label.bounding_box_size}")