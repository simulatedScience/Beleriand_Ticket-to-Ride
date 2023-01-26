import os

def type_to_edge_image(energy_type: str) -> str:
  """
  returns the filepath to an image corresponding to the given energy type string.

  Args:
      energy_type (str): energy type string. Allowed values are:
        - grass
        - fire
        - water
        - electric
        - fighting
        - psychic
        - dark
        - steel
        - fairy
        - neutral
  """
  if not energy_type in ("grass", "fire", "water", "electric", "fighting", "psychic", "dark", "steel", "fairy", "neutral"):
    raise ValueError(f"energy_type was {energy_type} but must be one of the following: 'grass', 'fire', 'water', 'electric', 'fighting', 'psychic', 'dark', 'steel', 'fairy', 'neutral'")
  image_name = energy_type + ".png"
  return os.path.join(os.path.dirname(__file__), "edge_images", image_name)

def is_pokemon_energy_type(color: str) -> bool:
  """
  checks if the given color is a pokemon TCG energy type.
  energy types are:
  'grass', 'fire', 'water', 'electric', 'fighting', 'psychic', 'dark', 'steel', 'fairy', 'neutral'

  Args:
      color (str): any string

  Returns:
      bool: True if the given color is a pokemon TCG energy type, False otherwise
  """
  if color in ("grass", "fire", "water", "electric", "fighting", "psychic", "dark", "steel", "fairy", "neutral"):
    return True
  return False

def color_to_energy_type(color: str) -> bool:
  """
  returns the energy type corresponding to the given color name

  Args:
      color (str): a color name

  Returns:
      bool: energy type corresponding to the given color
  """
  if color == "green":
    return "grass"
  elif color == "red":
    return "fire"
  elif color == "blue":
    return "water"
  elif color == "yellow":
    return "electric"
  elif color == "orange":
    return "fighting"
  elif color == "purple":
    return "psychic"
  elif color == "black":
    return "dark"
  elif color in ("grey", "gray"):
    return "neutral"
  elif color == "white":
    return "fairy"
  else:
    raise ValueError(f"Color '{color}' could not be converted. Color must be one of the following: 'green', 'red', 'blue', 'yellow', 'orange', 'purple', 'black', 'gray'/'grey', 'white'")