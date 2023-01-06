"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
from typing import Tuple

from PIL import ImageFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

from graph_particle import Graph_Particle


class Particle_Label(Graph_Particle):
  def __init__(self,
        label: str,
        position: np.ndarray = np.array([0, 0]),
        fontsize: int = 20,
        font_name: str = None,
        font_path: str = "beleriand_ttr\\MiddleEarth.ttf"):
    """
    Initialize a particle label
    It's bounding box is calculated from the label text, fontsize and font at the given path.

    Args:
        label (str): text for the particle
        position (np.ndarray, optional): position of the particle. Defaults to np.array([0, 0]).
        fontsize (int, optional): fontsize of the label. Defaults to 20.
        font_name (str, optional): name of the font. Defaults to None.
        font_path (str, optional): path to the font file as `.ttf`. This is only used  Defaults to "beleriand_ttr\\MiddleEarth.ttf".
    """
    if font_name is None:
      font_name = font_path.split("\\")[-1].strip(".ttf")
      fontManager.addfont(font_path)
      width, height = self.get_label_size(label, fontsize, font_path)
    else:
      width, height = self.get_label_size(label, fontsize, font_name)
    self.font_name = font_name
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
      bg_color: str = None,
      alpha: float = 1,
      zorder: int = 4):
    """
    draw the particle on the canvas.
    If a background color is given, a rectangle is drawn behind the label with the given color.

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        color (str, optional): color of the particle. Defaults to "#222222".
        bg_color (str, optional): background color of the label. Defaults to None.
        alpha (float, optional): alpha value of the particle. Defaults to 0.7.
        zorder (int, optional): zorder of the particle. Defaults to 4.
    """
    if bg_color is not None:
      ax.add_patch(plt.Rectangle(
        self.position - self.bounding_box_size / 2,
        self.bounding_box_size[0]*1.1,
        self.bounding_box_size[1]*1.1,
        color = bg_color,
        alpha = alpha,
        zorder = zorder - 1,
      ))
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
        fontname = self.font_name,
    )


  def get_label_size(self, label: str, fontsize: int, font: str) -> Tuple[float, float]:
    """
    get size of a label with a given font size

    Args:
        label (str): text of the label
        fontsize (int): fontsize of the label
        font (str): name or path of the font to use

    Returns:
        float: width of the label, normalized to height=1
        float: height of the label, always 1
    """
    if ".ttf" in font:
      img_font = ImageFont.truetype(font, fontsize)
    else:
      # load installed font
      img_font = ImageFont.load(font)
      img_font.set_size(fontsize)
    width, height = img_font.getsize(label)
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