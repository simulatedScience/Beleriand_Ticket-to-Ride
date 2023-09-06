from PIL import Image, ImageChops

def generate_counting_strip(
            min_number: int,
            max_number: int,
            length_px: int,
            cell_images: tuple[str],
            number_folders: tuple[str],
            number_heights: tuple[float] = (0.8),
            number_offset: tuple[float] = (0, 0),
            number_rotation: float = 0,
            empty_first_cell: bool = False,
            save_path_prefix: str = "assets\\"):
    # Normalizing inputs
    cell_images = cell_images if isinstance(cell_images, list) else [cell_images]
    number_folders = number_folders if isinstance(number_folders, list) else [number_folders]
    
    # Calculate the number of cells and the size of each cell
    num_cells = max_number - min_number + 1 + empty_first_cell
    cell_size = length_px // num_cells  # This ensures each cell is as square as possible
    extra_pixels = length_px % num_cells  # Calculate the remaining pixels after dividing the length evenly among the cells
    
    # Handle the case where cell_images=None
    if cell_images[0] is None:
        cell_images = [Image.new('RGBA', (cell_size, cell_size))]

    # Resize the cell images
    cell_images_resized = [img.resize((cell_size, cell_size)) if isinstance(img, Image.Image) else Image.open(img).resize((cell_size, cell_size)) for img in cell_images]
    
    # Initialize the image
    strip = Image.new('RGBA', (length_px, cell_size))
    
    img = cell_images_resized[0].copy()
    num_img = Image.open(f"{number_folders[0]}\\1.png")
    
    if empty_first_cell:
        min_number -= 1
    for i, num in enumerate(range(min_number, max_number + 1)):
        if i == 0 and empty_first_cell:
            continue
        # load correct cell image as number background
        if num % 10 == 0 and len(cell_images) >= 3:
            img = cell_images_resized[2].copy()
        elif num % 5 == 0 and len(cell_images) >= 2:
            img = cell_images_resized[1].copy()
        else:
            img = cell_images_resized[0].copy()
        # Load correct number image
        if num % 10 == 0 and len(number_folders) >= 3:
            num_img = Image.open(f"{number_folders[2]}/{num}.png")
        elif num % 5 == 0 and len(number_folders) >= 2:
            num_img = Image.open(f"{number_folders[1]}/{num}.png")
        else:
            num_img = Image.open(f"{number_folders[0]}/{num}.png")
        # get correct number height
        if num % 10 == 0 and len(number_heights) >= 3:
            number_height = number_heights[2]
        elif num % 5 == 0 and len(number_heights) >= 2:
            number_height = number_heights[1]
        else:
            number_height = number_heights[0]

        # Resize the number image to fit the specified height and adjust the width accordingly
        orig_width, orig_height = num_img.size
        new_height = int(cell_size * number_height)
        new_width = int(new_height * orig_width / orig_height)
        num_img_resized = num_img.resize((new_width, new_height))
        # Calculate the offset based on the resized image and the number_offset tuple
        offset_x = int(new_width * number_offset[0])
        offset_y = -int(new_height * number_offset[1])
        num_img_resized = ImageChops.offset(num_img_resized, offset_x, offset_y)
        # Rotate the number image
        num_img_resized = num_img_resized.rotate(number_rotation, expand=True)
        # move number relative to background
        # Calculate the center of the rotated image
        rotated_width, rotated_height = num_img_resized.size
        center_x = rotated_width // 2
        center_y = rotated_height // 2

        # Calculate the new offset based on the center of the rotated image and the number_offset tuple
        offset_x = (new_width // 2) - center_x
        offset_y = (new_height // 2) - center_y

        num_img_resized = ImageChops.offset(num_img_resized, offset_x, offset_y)
        
        # Calculate the position to place the number image in the center of the cell
        num_pos = ((cell_size - new_width) // 2, (cell_size - new_height) // 2)
        img.paste(num_img_resized, num_pos, num_img_resized)
        
        # Paste the cell onto the strip
        pos = i * cell_size + min(i, extra_pixels)  # Distribute the extra pixels evenly between the cells
        # add first column of pixels to fill the gap between cells
        if i > 0:
            strip.paste(img.crop((0, 0, 1, cell_size)), (pos - 1, 0))
        strip.paste(img, (pos, 0))
    
    strip.save(save_path_prefix + f"counting_strip_{min_number}-{max_number}_{length_px}.png")
    return strip # return saved image


if __name__ == "__main__":
    # Example usage
    strip_image = generate_counting_strip(
        min_number=1,
        max_number=25,
        length_px=9490,
        cell_images=["assets\\counting_strips\\cell.png"],
        # cell_images=["cell.png", "cell5.png", "cell10.png"],
        number_folders=["assets\\points_images\\points_standard"],
        # number_folders=["assets\\points_images\\points_standard", "assets\\points_images\\points_5", "assets\\points_images\\points_10"])
        number_height=0.8)
    # strip_image.save("assets\\counting_strip_1_25.png")
    # strip_image.save("assets\\counting_strip_26_50.png")

    # ##### Harry Potter counting strip #####
    # strip_image = generate_counting_strip(
    #     min_number=1,
    #     max_number=25,
    #     length_px=9490,
    #     cell_images=["assets\\counting_strips\\cell.png"],
    #     number_folders=["assets\\points_images\\points_standard"],
    #     number_heights=[0.8])
    # # strip_image.save("assets\\counting_strip_1_25.png")
    # # strip_image.save("assets\\counting_strip_26_50.png")
