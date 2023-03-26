
from typing import Tuple
import os

from PIL import Image, ImageFont, ImageDraw

def generate_all_points_images(
    font_size=150,
    font_name: str = None,
    font_path: str = "beleriand_ttr\\MiddleEarth.ttf",
    standard_color: str = "#3366dd", # blue
    bonus_color: str = "#33dd33", # green
    penalty_color: str = "#dd3333", # red
    stroke_width: int = None,
    outline_color: str = None,
    outline_stroke_width: int = None,
    min_number: int = 1,
    max_number: int = 30,
    square_image: bool = True,
    ):
  """
  Generate images of numbers from min_number to max_number using the given font settings. These numbers are generated with three different text colors: standard_color, bonus_color, and penalty_color. For penalty_color, all numbers are multiplied by -1. The images are saved in `./points_standard`, `./points_bonus`, and `./points_penalty` respectively.

  Args:
    font_size (int, optional): The font size to use. Defaults to 150.
    font_name (str, optional): The name of the font to use. This only works for standard fonts. If None, the font_path must be provided. Defaults to None (use `font_path`)
    font_path (str, optional): The path to the .ttf font file to use (necessary for non-standard fonts). If None, the font_name must be provided. Defaults to "beleriand_ttr\\MiddleEarth.ttf".
    standard_color (str, optional): The color to use for standard points. Defaults to "#3366dd".
    bonus_color (str, optional): The color to use for bonus points. Defaults to "#33dd33".
    penalty_color (str, optional): The color to use for penalty points. Defaults to "#dd3333".
    stroke_width (int, optional): The width of the stroke. Defaults to font_size // 25
    outline_color (str, optional): The color to use for the outline of the text. Defaults to None (no outline).
    outline_stroke_width (int, optional): The width of the outline. Defaults to font_size // 8.
    min_number (int, optional): The minimum number to generate images for. Defaults to 1.
    max_number (int, optional): The maximum number to generate images for. Defaults to 30.
    square_image (bool, optional): Whether to make the image square. Defaults to True.
  """
  # get current directory
  current_dir = os.path.dirname(os.path.abspath(__file__))
  generate_points_images(
    save_filepath=os.path.join(current_dir, "points_standard"),
    font_size=font_size,
    font_name=font_name,
    font_path=font_path,
    text_color=standard_color,
    stroke_width=stroke_width,
    outline_color=outline_color,
    outline_stroke_width=outline_stroke_width,
    min_number=min_number,
    max_number=max_number,
    )
  generate_points_images(
    save_filepath=os.path.join(current_dir, "points_bonus"),
    font_size=font_size,
    font_name=font_name,
    font_path=font_path,
    text_color=bonus_color,
    stroke_width=stroke_width,
    outline_color=outline_color,
    outline_stroke_width=outline_stroke_width,
    min_number=min_number,
    max_number=max_number,
    )
  generate_points_images(
    save_filepath=os.path.join(current_dir, "points_penalty"),
    font_size=font_size,
    font_name=font_name,
    font_path=font_path,
    text_color=penalty_color,
    stroke_width=stroke_width,
    outline_color=outline_color,
    outline_stroke_width=outline_stroke_width,
    min_number= -max_number, # penalty points are negative
    max_number= -min_number,
    )

def generate_points_images(
    save_filepath: str,
    font_size=150,
    font_name: str = None,
    font_path: str = "beleriand_ttr\\MiddleEarth.ttf",
    text_color: str = "#3366dd", # blue
    stroke_width: int = None,
    outline_color: str = None,
    outline_stroke_width: int = None,
    min_number: int = 1,
    max_number: int = 30,
    square_image: bool = True,
    ):
  """
  Generate images of numbers from min_number to max_number using the given font settings. The images are saved in `save_filepath`.

  Args:
    save_filepath (str): The directory to save the images to.
    font_size (int, optional): The font size to use. Defaults to 150.
    font_name (str, optional): The name of the font to use. This only works for standard fonts. If None, the font_path must be provided. Defaults to None (use `font_path`)
    font_path (str, optional): The path to the .ttf font file to use (necessary for non-standard fonts). If None, the font_name must be provided. Defaults to "beleriand_ttr\\MiddleEarth.ttf".
    text_color (str, optional): The color to use for the text. Defaults to "#3366dd".
    stroke_width (int, optional): The width of the stroke. Defaults to font_size // 25
    outline_color (str, optional): The color to use for the outline of the text. Defaults to None (no outline).
    outline_stroke_width (int, optional): The width of the outline. Defaults to font_size // 8.
    min_number (int, optional): The minimum number to generate images for. Defaults to 1.
    max_number (int, optional): The maximum number to generate images for. Defaults to 30.
    square_image (bool, optional): Whether to make the image square. Defaults to True.

  Raises:
    ValueError: If neither `font_name` nor `font_path` are provided.
  """
  # determine font
  if font_name is not None:
    font = ImageFont.truetype(font_name, font_size)
  elif font_path is not None:
    font = ImageFont.truetype(font_path, font_size)
  else:
    raise ValueError("Either `font_name` or `font_path` must be provided.")
  # set stroke width
  if stroke_width is None:
    stroke_width = font_size // 25
  if outline_stroke_width is None:
    outline_stroke_width = font_size // 8
  # create directory if it doesn't exist
  if not os.path.exists(save_filepath):
    os.makedirs(save_filepath)
  # calculate image size
  image_size = get_image_size(
      font=font,
      stroke_width=max(stroke_width, outline_stroke_width),
      min_number=min_number,
      max_number=max_number,
      square_image=square_image,
      )
  # generate images
  for number in range(min_number, max_number + 1):
    # create image
    image = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # draw text
    text = str(number)
    text_size = draw.textsize(text, font=font)
    text_position = ((image_size[0] - text_size[0]) / 2, (image_size[1] - text_size[1]) / 2)
    if outline_color is not None:
      draw.text(text_position, text, font=font, fill=outline_color, stroke_width=outline_stroke_width)
    draw.text(text_position, text, font=font, fill=text_color, stroke_width=stroke_width)
    # save image
    image.save(os.path.join(save_filepath, f"{number}.png"))

def get_image_size(
      font: ImageFont,
      stroke_width: int,
      min_number: int,
      max_number: int,
      padding: int = 10,
      square_image: bool = True,):
  """
  Calculate the size of the image needed to fit all of the given numbers.

  Args:
    font (ImageFont): The font to use with font_size already set.
    stroke_width (int): The stroke width to use for the font.
    min_number (int): The minimum number to generate images for.
    max_number (int): The maximum number to generate images for.
    padding (int, optional): The padding to use around the text. Defaults to 10.
    square_image (bool, optional): Whether to make the image square. Defaults to True.

  Returns:
    tuple: The size of the image needed to fit the given numbers.
  """
  # create image
  image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
  # for each number, calculate the size of the image needed to fit the number
  max_image_size = (0, 0)
  for number in range(min_number, max_number + 1):
    draw = ImageDraw.Draw(image)
    text = str(number)
    text_size = font.getsize(text, stroke_width=stroke_width)
    max_image_size = (
        max(max_image_size[0], text_size[0]),
        max(max_image_size[1], text_size[1]))
  # add padding
  if square_image:
    max_image_size = (max(max_image_size) + padding, max(max_image_size) + padding)
  else:
    max_image_size = (max_image_size[0] , max_image_size[1] + padding)
  return max_image_size

if __name__ == "__main__":
  # generate_all_points_images(outline_color="#222222")
  generate_all_points_images(
    font_size=250,
    outline_color="#222222",
    square_image=False,)