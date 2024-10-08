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
from matplotlib.patches import Rectangle

from graph_particle import Graph_Particle
from particle_node import Particle_Node


class Particle_Label(Graph_Particle):
  def __init__(self,
        label: str,
        id: int,
        position: np.ndarray = np.array([0, 0]),
        rotation: float = 0,
        mass: float = 1,
        color: str = "#eeeeee",#"#222222",
        node_attraction: float = 0.1,
        interaction_radius: float = 5,
        velocity_decay: float = 0.9999,
        angular_velocity_decay: float = 0.9999,
        repulsion_strength: float = 0,
        fontsize: int = 250,
        font_name: str = None,
        # font_path: str = "assets\\fonts\\MiddleEarth.ttf",
        # font_path: str = "assets\\fonts\\ringbearer.ttf",
        font_path: str = None,
        height_scale_factor: float = None,
        ignore_linebreaks: bool = False):
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
    self.ignore_linebreaks = ignore_linebreaks
    if ignore_linebreaks:
      self.label = self.label.replace("\n", " ")
    self.fontsize = fontsize
    self.color = color
    self.node_attraction = node_attraction
    if height_scale_factor is None:
      self.height_scale_factor = Particle_Label.get_label_height_scale(font_path=font_path, fontsize=fontsize)
    else:
      self.height_scale_factor = height_scale_factor
    # self.inside_stroke_width = fontsize // 25
    self.inside_stroke_width = fontsize // 75
    self.outline_stroke_width = fontsize // 8
    if font_name is None and font_path is None:
      font_path = "assets/fonts/Stamp.ttf"
    if font_name is None:
      # font_name = font_path.split("\\")[-1].strip(".ttf")
      fontManager.addfont(font_path)
      width, height, pix_width, pix_height, *offset = self._get_label_size(
          label = label,
          fontsize = fontsize,
          font = font_path)
      self.font_name = font_path
    else:
      width, height, pix_width, pix_height, *offset = self._get_label_size(
          label = label,
          fontsize = fontsize,
          font = font_name)
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
        np.ndarray: attraction force vector
        np.ndarray: force anchor point
    """
    if not isinstance(other, Particle_Node):
      return np.zeros(2), self.position
    return self.node_attraction * (other.position - self.position), self.position


  def set_simulation_parameters(self,
      mass: float = None,
      node_attraction: float = None,
      interaction_radius: float = None,
      velocity_decay: float = None,
      angular_velocity_decay: float = None,
      repulsion_strength: float = None) -> None:
    """
    set simulation parameters for the label particle

    Args:
        mass (float, optional): mass of the particle
        node_attraction (float, optional): attraction strength to nodes
        interaction_radius (float, optional): maximum distance to interaction partners
        velocity_decay (float, optional): velocity decay factor
        angular_velocity_decay (float, optional): angular velocity decay factor
        repulsion_strength (float, optional): repulsion strength
    """
    if mass is not None:
      self.mass = mass
    if node_attraction is not None:
      self.node_attraction = node_attraction
    if interaction_radius is not None:
      self.interaction_radius = interaction_radius
    if velocity_decay is not None:
      self.velocity_decay = velocity_decay
    if angular_velocity_decay is not None:
      self.angular_velocity_decay = angular_velocity_decay
    if repulsion_strength is not None:
      self.repulsion_strength = repulsion_strength


  def set_text(self, new_label: str, ax: plt.Axes):
    """
    set the text of the label

    Args:
        text (str): text to set
        ax (plt.Axes): matplotlib axes to draw on
    """
    if new_label != self.label:
      self.label = new_label
      if self.ignore_linebreaks:
        self.label = self.label.replace("\n", " ")
      # update bounding box size
      width, height, pix_width, pix_height, *offset = self._get_label_size(
          new_label,
          self.fontsize,
          self.font_path)
      self.bounding_box_size = (width, height)
      self.width_pixels = pix_width
      self.height_pixels = pix_height
      self.text_x_offset = offset[0]
      self.text_y_offset = offset[1]
      # redraw label
      self.erase()
      self.draw(ax)

  def set_font(self,
    font_size: int = None,
    font_path: str = None,
    label_height_scale: float = None,
  ) -> None:
    """
    set the font and fontsize of the label

    Args:
        font_size (int, optional): fontsize to set. Defaults to None.
        font_path (str, optional): path to a font file. Defaults to None.
    """
    if label_height_scale is not None:
      self.height_scale_factor: float = label_height_scale
    if font_size is not None:
      self.fontsize = font_size
    if font_path is not None:
      self.font_path = font_path
    width, height, pix_width, pix_height, *offset = self._get_label_size(
        label = self.label,
        fontsize = self.fontsize,
        font = self.font_path)
    self.bounding_box_size = (width, height)
    self.width_pixels = pix_width
    self.height_pixels = pix_height
    self.text_x_offset = offset[0]
    self.text_y_offset = offset[1]
    

  def draw(self, 
      ax: plt.Axes,
      color: str = None,
      border_color: str = "#222222",#"#eeeeee",
      alpha: float = 1,
      zorder: int = 4,
      scale: float = 1,
      override_position: np.ndarray = None,
      movable: bool = True,
      debug: bool = False) -> None:
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
    # if border_color is not None:
    #   text_image = self.draw_label_outline(text_image_size, border_color)
    # else:
    #   text_image = Image.new("RGBA", text_image_size, (0, 0, 0, 0))
    # text_draw = ImageDraw.Draw(text_image)
    # text_draw.text(
    #   (self.text_x_offset, self.text_y_offset),
    #   self.label,
    #   font=self.img_font,
    #   fill=color,
    #   stroke_width=self.inside_stroke_width)
    
    # draw bounding box
    if debug:
      ax.add_patch(Rectangle(
        (self.position[0] - self.bounding_box_size[0] / 2, self.position[1] - self.bounding_box_size[1] / 2),
        self.bounding_box_size[0],
        self.bounding_box_size[1],
        linewidth=1,
        edgecolor="black",
        facecolor="#ff00ff",
        alpha=0.3,
        zorder=5
        ))
    
    text_image = self.draw_label_outline(text_image_size, color, border_color)

    label_extent = self.get_extent(scale, override_position)
    self.plotted_objects.append(ax.imshow(
        text_image,
        extent=label_extent,
        zorder=zorder,
        alpha=alpha,
        picker=True))
    super().set_particle_movable(movable)

  def draw_label_outline(self,
      text_image_size: Tuple[int, int],
      inner_color: str = "#dddddd",
      border_color: str = "#222222") -> Image:
    
    outline_image = Image.new("RGBA", text_image_size, (0,0,0,0))
    text_draw = ImageDraw.Draw(outline_image)
    text_draw.text(
        # (50, -50),
        (self.text_x_offset, self.text_y_offset),
        self.label,
        align="center",
        font=self.img_font,
        fill=inner_color,
        stroke_width=self.outline_stroke_width,
        stroke_fill=border_color)

    return outline_image

  def get_extent(self, scale: float = 1, override_position: np.ndarray = None) -> Tuple[float, float, float, float]:
    """
    get the extent of the label

    Args:
        scale (float, optional): _description_. Defaults to 1.
        override_position (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        Tuple[float, float, float, float]: extent of the label as (left, right, bottom, top)
    """
    if override_position is None:
      override_position = self.position
    return (
        override_position[0] - self.bounding_box_size[0] / 2 * scale,
        override_position[0] + self.bounding_box_size[0] / 2 * scale,
        override_position[1] - self.bounding_box_size[1] / 2 * scale,
        override_position[1] + self.bounding_box_size[1] / 2 * scale)


  def _get_label_size(self, label: str, fontsize: int, font: str, image_padding: int = 0.3) -> Tuple[float, float, float, float]:
    """
    get size of a label with a given font size

    Args:
        label (str): text of the label
        fontsize (int): fontsize of the label
        font (str): name or path of the font to use
        image_padding (int): number of pixels to add to width and height in each direction as padding to ensure text is not cut off. Defaults to 0.3 (30%)

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
    # calculate bounding box of outline stroke
    bbox_size = get_multiline_bbox_size(label, self.img_font, stroke_width=self.outline_stroke_width)
    width_pixels, height_pixels = bbox_size
    # add padding
    width_pixels = int(width_pixels + image_padding*width_pixels)
    height_pixels = int(height_pixels + image_padding*height_pixels)
    # calculate bounding box of inside stroke
    # bbox_size = get_multiline_bbox_size(label, self.img_font, stroke_width=self.inside_stroke_width)
    small_width_pixels, small_height_pixels = bbox_size
    # calculate offsets of inner text
    text_x_offset = (width_pixels - small_width_pixels)//2
    text_y_offset = (height_pixels - small_height_pixels)//2
    # text_x_offset = image_padding // 2
    # text_y_offset = -image_padding // 2
    # normalize height
    # line_count: int = label.count("\n") + 1
    width = width_pixels * self.height_scale_factor# * line_count
    height = height_pixels * self.height_scale_factor# * line_count

    return width, height, width_pixels, height_pixels, text_x_offset, text_y_offset



  def get_label_height_scale(
      fontsize: int = 250,
      font_name: str = None,
      font_path: str = None) -> float:
      # font_path: str = "beleriand_ttr\\MiddleEarth.ttf") -> float:
    """
    calculate a scale factor to be used for all labels to normalize their height but keep all text the same size.
    This is necessary because a label "Xy" being scaled to the same height as "xx", would result in a much smaller fontsize because of the capital letter and low reaching y.

    Args:
        fontsize (int, optional): font size to use. Defaults to 200.
        font_name (str, optional): font to use if it is a default font Leave this as None when using a custom font. Defaults to None.
        font_name (str, optional): font to use if it is a default font Leave this as None when using a custom font. Defaults to None.
        font_path (str, optional): path to a custom font to use (as ttf file). Defaults to "beleriand_ttr\MiddleEarth.ttf".

    Returns:
        float: scale factor to use such that no label will have height >1. Labels without vertically large letters may be smaller, but the displayed text will have the same size for all labels.
    """
    if font_name is None and font_path is None:
      font_name = "assets/fonts/Stamp.ttf"
    # load image_font from file
    if font_name is None:
      img_font = ImageFont.truetype(font_path, fontsize)
    else:
      # load installed font
      img_font = ImageFont.load(font_name)
      img_font.set_size(fontsize)
    # determine sroke widths
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-abcdefghijklmnopqrstuvwxyz_0123456789"
    outline_stroke_width = fontsize // 8
    bbox_size = get_multiline_bbox_size(text, img_font, stroke_width=outline_stroke_width)
    height = bbox_size[1]
    return 1/height


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



def get_multiline_bbox_size(label: str, img_font: ImageFont, stroke_width: int) -> Tuple[int, int]:
  """
  get the bounding box size (width, height) of a (multiline) label in px.

  Args:
      label (str): string to get bounding box size of. May contain `\n` for new lines.
      img_font (ImageFont): ImageFont object to use for calculating the bounding box
      stroke_width (int): stroke width to use for calculating the bounding box

  Returns:
      tuple[int]: (width, height) of the bounding box in px
  """
  width_pixels: int = 0
  height_pixels: int = 0
  height_multiplier: float = 1 # used to add line spacing
  for line in label.split("\n"):
    line_bbox = img_font.getbbox(line, stroke_width=stroke_width)
    line_width_pixels = abs(line_bbox[2] - line_bbox[0])
    line_height_pixels = abs(line_bbox[1] - line_bbox[3])
    width_pixels = max(width_pixels, line_width_pixels)
    height_pixels += line_height_pixels * height_multiplier
    height_multiplier = 1.1
  return width_pixels, height_pixels


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