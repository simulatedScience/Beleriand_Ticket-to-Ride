import os
from typing import List, Tuple
from PIL import Image
import numpy as np
from skimage.measure import label, regionprops

def load_image(image_path: str) -> Image.Image:
    """
    Load an image and convert it to RGBA.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        Image.Image: The loaded image in RGBA format.
    """
    img = Image.open(image_path)
    img_rgba = img.convert("RGBA")
    return img_rgba

def create_binary_image(img_rgba: Image.Image) -> np.ndarray:
    """
    Create a binary image from the alpha channel of an RGBA image.
    
    Args:
        img_rgba (Image.Image): The source image in RGBA format.

    Returns:
        np.ndarray: A binary image where non-transparent pixels are True and transparent pixels are False.
    """
    alpha = np.array(img_rgba)[:, :, 3]
    binary = alpha > 0
    return binary

def find_connected_components(binary: np.ndarray) -> np.ndarray:
    """
    Find connected components in a binary image.
    
    Args:
        binary (np.ndarray): The binary image.

    Returns:
        np.ndarray: An array with the same shape as the input, where each connected component is labeled with a unique integer.
    """
    labels = label(binary)
    return labels

def extract_regions_get_centers_and_bboxes(
        img_rgba: Image.Image,
        labels: np.ndarray,
        output_folder: str) -> Tuple[List[Tuple[float]], List[Tuple[int]]]:
    """
    Extract regions corresponding to connected components from the source image, save them as separate images, and calculate their centers and bounding boxes.
    
    Args:
        img_rgba (Image.Image): The source image in RGBA format.
        labels (np.ndarray): An array with the connected components labels.
        output_folder (str): Path to the output folder where the extracted circles will be saved.

    Returns:
        List[Tuple[float]]: List of center points of the extracted circles.
        List[Tuple[int]]: List of bounding boxes of the extracted circles as (min_row, min_col, max_row, max_col).
    """
    props = regionprops(labels)
    centers = []
    boxes = []

    for i, prop in enumerate(props, start=1):
        minr, minc, maxr, maxc = prop.bbox
        region = img_rgba.crop((minc, minr, maxc, maxr))
        output_path = os.path.join(output_folder, f"circle_{i}.png")
        region.save(output_path)
        center = ((minc + maxc) / 2, (minr + maxr) / 2)
        centers.append(center)
        boxes.append((minr, minc, maxr, maxc))

    return centers, boxes

def extract_circles(image_path: str, output_folder: str) -> Tuple[List[Tuple[float]], List[Tuple[int]]]:
    """
    Extracts circular sections from an otherwise transparent image and saves them to individual files.
    The source image should have a transparent background and the circles should be filled with non-transparent pixels. Bounding boxes of circles should not overlap.

    Args:
        image_path (str): Path to the source image (containing circular images on a transparent background).
        output_folder (str): Path to the output folder where the extracted circles will be saved.

    Returns:
        List[Tuple[float]]: List of center points of the extracted circles.
        List[Tuple[int]]: List of bounding boxes of the extracted circles as (min_row, min_col, max_row, max_col).
    """
    img_rgba = load_image(image_path)
    binary = create_binary_image(img_rgba)
    labels = find_connected_components(binary)
    centers, boxes = extract_regions_get_centers_and_bboxes(img_rgba, labels, output_folder)
    return centers, boxes

if __name__ == "__main__":
    all_locations_image: str = "beleriand_all_locations2.png"
    output_folder: str = "auto_locations"
    centerpoints, bounding_boxes = extract_circles(all_locations_image, output_folder)
    # print bboxes and centers
    for i, (bbox, center) in enumerate(zip(bounding_boxes, centerpoints), start=1):
        print(f"Circle {i+1}: bbox={bbox}, center={center}")
        