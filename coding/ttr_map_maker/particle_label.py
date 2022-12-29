"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
import os

from PIL import ImageFont
import numpy as np

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