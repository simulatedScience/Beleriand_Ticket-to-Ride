"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
from typing import Tuple

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

from graph_particle import Graph_Particle
from particle_node import Particle_Node


class Particle_Label(Graph_Particle):
  def __init__(self,
        label: str,
        id: int,
        position: np.ndarray = np.array([0, 0]),
        rotation: float = 0,
        mass: float = 1,
        color: str = "#222222",
        interaction_radius: float = 5,
        velocity_decay: float = 0.9999,
        angular_velocity_decay: float = 0.9999,
        repulsion_strength: float = 1,
        node_attraction: float = 0.1,
        fontsize: int = 150,
        font_name: str = None,
        font_path: str = "beleriand_ttr\\MiddleEarth.ttf",
        height_scale_factor: float = None):
    """
    Initialize a particle label
    It's bounding box is calculated from the label text, fontsize and font at the given path.

    Args:
        label (str): text for the particle
        id (int): unique numeric id of the particle
        position (np.ndarray, optional): position of the particle. Defaults to np.array([0, 0]).
        fontsize (int, optional): fontsize of the label. Defaults to 20.
        font_name (str, optional): name of the font. Defaults to None.
        font_path (str, optional): path to the font file as `.ttf`. This is only used  Defaults to "beleriand_ttr\\MiddleEarth.ttf".
        height_scale_factor (float, optional) # TODO: complete docstring Args
    """
    self.label = label
    self.fontsize = fontsize
    self.color = color
    self.node_attraction = node_attraction
    if height_scale_factor is None:
      self.height_scale_factor = Particle_Label.get_label_height_scale()
    else:
      self.height_scale_factor = height_scale_factor
    self.inside_stroke_width = fontsize // 25
    self.outline_stroke_width = fontsize // 8
    if font_name is None:
      # font_name = font_path.split("\\")[-1].strip(".ttf")
      fontManager.addfont(font_path)
      width, height, pix_width, pix_height, *offset = self.get_label_size(label, fontsize, font_path)
      self.font_name = font_path
    else:
      width, height, pix_width, pix_height, *offset = self.get_label_size(label, fontsize, font_name)
      self.font_name = font_name
    super().__init__(
        id,
        position=position,
        rotation=rotation,
        target_position=None,
        mass=mass,
        bounding_box_size=(width, height),
        interaction_radius=interaction_radius,
        velocity_decay=velocity_decay,
        angular_velocity_decay=angular_velocity_decay,
        repulsion_strength=repulsion_strength,
    )
    self.width_pixels = pix_width
    self.height_pixels = pix_height
    self.text_x_offset = offset[0]
    self.text_y_offset = offset[1]


  def get_attraction_force(self, other: Particle_Node) -> Tuple[np.ndarray, np.ndarray]:
    """calculate attraction force to other particle

    Args:
        other (Particle_Node): particle node that the label is attracted to

    Returns:
        np.ndarray: attraction force
        np.ndarray: force anchor point
    """
    return self.node_attraction * (other.position - self.position), self.position


  def set_parameters(self, label_parameters: dict):
    """
    set parameters for the label

    Args:
        label_parameters (dict): dictionary with parameters for the label
    """
    self.color = label_parameters.get("color", self.color)
    # self.fontsize = label_parameters.get("fontsize", self.fontsize)
    # self.font_name = label_parameters.get("font_name", self.font_name)
    self.node_attraction = label_parameters.get("node-label", self.node_attraction)
    self.mass = label_parameters.get("label_mass", self.mass)
    self.velocity_decay = label_parameters.get("label_velocity_decay", self.velocity_decay)
    self.interaction_radius = label_parameters.get("interaction_radius", self.interaction_radius)
    self.repulsion_strength = label_parameters.get("repulsion_strength", self.repulsion_strength)

    # self.img_font = ImageFont.truetype(self.font_name, self.fontsize)
  def set_text(self, new_label: str, ax: plt.Axes):
    """
    set the text of the label

    Args:
        text (str): text to set
        ax (plt.Axes): matplotlib axes to draw on
    """
    if new_label != self.label:
      self.label = new_label
      # update bounding box size
      width, height, pix_width, pix_height, *offset = self.get_label_size(new_label, self.fontsize, self.font_name)
      self.bounding_box_size = (width, height)
      self.width_pixels = pix_width
      self.height_pixels = pix_height
      self.text_x_offset = offset[0]
      self.text_y_offset = offset[1]
      # redraw label
      self.erase()
      self.draw(ax)

  def draw(self, 
      ax: plt.Axes,
      color: str = None,
      border_color: str = "#eeeeee",
      alpha: float = 1,
      zorder: int = 4,
      movable: bool = True) -> None:
    """
    draw the particle on the canvas.
    If a border color is given, the label text will have an outline with the given color.

    Args:
        ax (plt.Axes): matplotlib axes to draw on
        color (str, optional): color of the particle. Defaults to "#222222".
        border_color (str, optional): background color of the label. Defaults to None.
        alpha (float, optional): alpha value of the particle. Defaults to 0.7.
        zorder (int, optional): zorder of the particle. Defaults to 4.
    """
    if color is None:
      color = self.color
    text_image_size = (self.width_pixels, self.height_pixels)
    if border_color is not None:
      text_image = self.draw_label_outline(text_image_size, border_color)
    else:
      text_image = Image.new("RGBA", text_image_size, (0, 0, 0, 0))

    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((self.text_x_offset, self.text_y_offset), self.label, font=self.img_font, fill=color, stroke_width=self.inside_stroke_width)

    label_extent = (
        self.position[0] - self.bounding_box_size[0] / 2,
        self.position[0] + self.bounding_box_size[0] / 2,
        self.position[1] - self.bounding_box_size[1] / 2,
        self.position[1] + self.bounding_box_size[1] / 2)
    self.plotted_objects.append(ax.imshow(
        text_image,
        extent=label_extent,
        zorder=zorder,
        alpha=alpha,
        picker=True))
    # set movability of particle
    super().set_particle_movable(movable)

  def draw_label_outline(self,
      text_image_size: Tuple[int, int],
      border_color: str = "#ff00ff") -> Image:
    
    outline_image = Image.new("RGBA", text_image_size, (0,0,0,0))
    text_draw = ImageDraw.Draw(outline_image)
    text_draw.text((self.text_x_offset, self.text_y_offset), self.label, font=self.img_font, fill=border_color, border=1, borderfill=border_color, stroke_width=self.outline_stroke_width, stroke_fill=border_color)

    return outline_image

  def get_label_size(self, label: str, fontsize: int, font: str, image_padding: int = 10) -> Tuple[float, float, float, float]:
    """
    get size of a label with a given font size

    Args:
        label (str): text of the label
        fontsize (int): fontsize of the label
        font (str): name or path of the font to use
        image_padding (int): number of pixels to add to width and height as padding to ensure text is not cut off. Defaults to 0.1

    Returns:
        float: width of the label, normalized to height=1
        float: height of the label, always 1
        int: width in pixels
        int: height in pixels
        int: x offset for text in image in pixels
        int: y offset for text in image in pixels
    """
    if ".ttf" in font:
      self.img_font = ImageFont.truetype(font, fontsize)
    else:
      # load installed font
      self.img_font = ImageFont.load(font)
      self.img_font.set_size(fontsize)
    width_pixels, height_pixels = self.img_font.getsize(label, stroke_width=self.outline_stroke_width)
    width_pixels = int(width_pixels + image_padding)
    height_pixels = int(height_pixels + image_padding)
    small_width_pixels, small_height_pixels = self.img_font.getsize(label, stroke_width=self.inside_stroke_width)
    text_x_offset = (width_pixels - small_width_pixels) // 2
    text_y_offset = (height_pixels - small_height_pixels) // 2
    # normalize height
    width = width_pixels * self.height_scale_factor
    height = height_pixels * self.height_scale_factor
    return width, height, width_pixels, height_pixels, text_x_offset, text_y_offset

  def add_json_info(self, particle_info: dict) -> dict:
    """
    Add label-specific particle information to json dictionary for saving.

    Args:
        particle_info (dict): json dictionary

    Returns:
        (dict): json dictionary with label-specific information
    """
    particle_info["label"] = self.label
    particle_info["color"] = self.color
    particle_info["fontsize"] = self.fontsize
    particle_info["font_name"] = self.font_name
    particle_info["node_attraction"] = self.node_attraction
    particle_info["height_scale_factor"] = self.height_scale_factor
    return particle_info

  def get_label_height_scale(
      fontsize: int = 150,
      font_name: str = None,
      font_path: str = "beleriand_ttr\\MiddleEarth.ttf") -> float:
    """
    calculate a scale factor to be used for all labels to normalize their height but keep all text the same size.
    This is necessary because a label "Xy" being scaled to the same height as "xx", would result in a much smaller fontsize because of the capital letter and low reaching y.

    Args:
        fontsize (int, optional): font size to use. Defaults to 150.
        font_name (str, optional): font to use if it is a default font Leave this as None when using a custom font. Defaults to None.
        font_path (str, optional): path to a custom font to use (as ttf file). Defaults to "beleriand_ttr\MiddleEarth.ttf".

    Returns:
        float: scale factor to use such that no label will have height >1. Labels without vertically large letters may be smaller, but the displayed text will have the same size for all labels.
    """
    # load image_font
    if font_name is None:
      img_font = ImageFont.truetype(font_path, fontsize)
    else:
      # load installed font
      img_font = ImageFont.load(font_name)
      img_font.set_size(fontsize)
    # determine sroke widths
    outline_stroke_width = fontsize // 8
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-abcdefghijklmnopqrstuvwxyz"
    
    width, height =  img_font.getsize(text, stroke_width=outline_stroke_width)
    return 1/height


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