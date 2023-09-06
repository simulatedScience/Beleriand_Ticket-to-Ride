import os

import numpy as np
from PIL import Image

def find_borders(img):
    img_array = np.array(img)
    mask = img_array[..., 3] == 0  # Alpha value is the 4th channel in RGBA

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    min_coords = np.min(coords, axis=1)
    max_coords = np.max(coords, axis=1)

    return (min_coords[1], min_coords[0], max_coords[1], max_coords[0])  # (left, upper, right, lower)

def remove_borders(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_folder, filename)).convert("RGBA")
            box = find_borders(img)
            img_cropped = img.crop(box)
            img_cropped.save(os.path.join(output_folder, filename))

# Usage
if __name__ == "__main__":
  origin_folder = r"C:\future_D\private\programming\python\Beleriand_TTR\coding\ttr_map_maker\beleriand_ttr\task_cards"
  output_folder = r"C:\future_D\private\programming\python\Beleriand_TTR\coding\ttr_map_maker\beleriand_ttr\task_cards_cut"
  remove_borders(origin_folder, output_folder)
