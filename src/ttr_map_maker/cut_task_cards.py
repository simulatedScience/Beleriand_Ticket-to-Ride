import os

import numpy as np
from PIL import Image

def find_borders_in_file(img):
    img_array = np.array(img)
    mask = img_array[..., 3] == 0  # Alpha value is the 4th channel in RGBA

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    min_coords = np.min(coords, axis=1)
    max_coords = np.max(coords, axis=1)

    return (min_coords[1], min_coords[0], max_coords[1], max_coords[0])  # (left, upper, right, lower)

def remove_borders_from_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_path, filename)
        output_path = os.path.join(output_folder, filename)
        remove_borders_from_file(input_folder, output_path)

def remove_borders_from_file(input_path, output_path):
    
    if input_path.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(input_path).convert("RGBA")
        box = find_borders_in_file(img)
        img_cropped = img.crop(box)
        img_cropped.save(output_path)
    else:
        raise ValueError(f"input_path must be a path to a .png or .jpg file, but was {input_path}")

# Usage
if __name__ == "__main__":
#   origin_folder = r"C:\future_D\private\programming\python\Beleriand_TTR\coding\ttr_map_maker\beleriand_ttr\task_cards"
  origin_folder = r"C:\future_D\private\programming\python\Beleriand_TTR\coding\ttr_map_maker\beleriand_ttr\task_cards\updated_tasks"
  output_folder = r"C:\future_D\private\programming\python\Beleriand_TTR\coding\ttr_map_maker\beleriand_ttr\task_cards_cut"
  remove_borders_from_folder(origin_folder, output_folder)
