from PIL import Image
import os

def add_frame_to_image(
        image_path: str,
        frame_path: str,
        output_path: str,
        target_width: int = None,
        target_height: int = None,
        position: str | tuple[int] = "center") -> None:
    """
    Add a frame to an image.

    Args:
        image_path (str): Path to the input image (including the file name and extension).
        frame_path (str): Path to the frame image (including the file name and extension).
        output_path (str): Path to save the framed image (including the file name and extension).
        target_width (int): Desired width of the framed image. Defaults to None (keep original width)
        target_height (int): Desired height of the framed image. Defaults to None (keep original height)
        position (str |tuple[int], optional): Position to align the inner image within the frame.
            If a tuple is provided, it should contain the x and y coordinates of the top left corner.
            If a string is provided, it should be one of the following, used for automatic positioning:
            Valid values: "center", "ne", "nw", "sw", "se", "n", "w", "s", "e".
            Defaults to "center".

    Returns:
        None

    Raises:
        ValueError: If the position string is invalid.
    """
    # Open the images
    image = Image.open(image_path)
    frame = Image.open(frame_path)

    # Resize the image to the specified width and height
    resized_image = image.resize((target_width, target_height))

    # Calculate the position to place the resized image inside the frame
    if isinstance(position, str):
        x_offset, y_offset = string_pos_to_ints(target_width, target_height, position, frame)
    elif isinstance(position, tuple):
        x_offset, y_offset = position
    else:
        raise ValueError("Invalid position type")

    # Paste the resized image onto the frame
    frame.paste(resized_image, (x_offset, y_offset), mask=resized_image)

    # Save the result
    frame.save(output_path)

def string_pos_to_ints(target_width: int, target_height: int, position: str, frame: Image):
    """
    Convert a string position to x and y offsets.

    Args:
        target_width (int): Desired width of the framed image.
        target_height (int): Desired height of the framed image.
        position (str): Position to align the inner image within the frame.
            Valid values: "center", "ne", "nw", "sw", "se", "n", "w", "s", "e".
        frame (Image): The frame image.

    Returns:
        tuple[int]: The x and y offsets.

    Raises:
        ValueError: If the position string is invalid.
    """
    if position == "center":
        x_offset = (frame.width - target_width) // 2
        y_offset = (frame.height - target_height) // 2
    elif position == "ne":
        x_offset = frame.width - target_width
        y_offset = 0
    elif position == "nw":
        x_offset = 0
        y_offset = 0
    elif position == "sw":
        x_offset = 0
        y_offset = frame.height - target_height
    elif position == "se":
        x_offset = frame.width - target_width
        y_offset = frame.height - target_height
    elif position == "n":
        x_offset = (frame.width - target_width) // 2
        y_offset = 0
    elif position == "w":
        x_offset = 0
        y_offset = (frame.height - target_height) // 2
    elif position == "s":
        x_offset = (frame.width - target_width) // 2
        y_offset = frame.height - target_height
    elif position == "e":
        x_offset = frame.width - target_width
        y_offset = (frame.height - target_height) // 2
    else:
        raise ValueError("Invalid position string")
    return x_offset,y_offset

def process_images_in_folder(input_folder, frame_path, output_folder=None, target_width=None, target_height=None, position="center"):
    if output_folder is None:
        folder_name = os.path.basename(input_folder)
        output_folder = os.path.join(os.path.dirname(input_folder), f"{folder_name}_framed")

    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(input_folder):
        if image_name.endswith('.png'):
            image_path = os.path.join(input_folder, image_name)
            output_path = os.path.join(output_folder, image_name)
            add_frame_to_image(image_path, frame_path, output_path, target_width, target_height, position)


if __name__ == "__main__":
    source_folder = r"C:\future_D\private\programming\python\Beleriand_TTR\board_map_design\location_designs\location_images\auto_locations"
    frame_image = r"C:\future_D\private\programming\python\Beleriand_TTR\board_map_design\location_designs\blank_location_design_hole_small.png"
    # target_width = 390
    # target_height = 390
    target_width = 257
    target_height = 257
    # target_width = 334
    # target_height = 334
    process_images_in_folder(
        source_folder,
        frame_image,
        target_width=target_width,
        target_height=target_height,
        position="center")
