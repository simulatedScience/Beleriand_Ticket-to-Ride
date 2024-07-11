import os
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def split_image_into_4_parts(image_path, output_folder):
    """
    
    """
    # If no file or folder was selected, return
    if not image_path or not output_folder:
        print("File or output folder not selected. Exiting.")
        return

    # Load the image
    original_image = Image.open(image_path)

    # Calculate dimensions for the split
    width, height = original_image.size
    center_x, center_y = width // 2, height // 2

    # Check for odd dimensions and calculate overlap if needed
    overlap_x = width % 2
    overlap_y = height % 2

    # Define the coordinates for the 4 parts
    top_left = (0, 0, center_x + overlap_x, center_y + overlap_y)
    top_right = (center_x, 0, width, center_y + overlap_y)
    bottom_left = (0, center_y, center_x + overlap_x, height)
    bottom_right = (center_x, center_y, width, height)

    # Crop the image into four parts
    top_left_part = original_image.crop(top_left)
    top_right_part = original_image.crop(top_right)
    bottom_left_part = original_image.crop(bottom_left)
    bottom_right_part = original_image.crop(bottom_right)

    # Save each part to a separate file
    top_left_part.save(os.path.join(output_folder, 'top_left.png'))
    top_right_part.save(os.path.join(output_folder, 'top_right.png'))
    bottom_left_part.save(os.path.join(output_folder, 'bottom_left.png'))
    bottom_right_part.save(os.path.join(output_folder, 'bottom_right.png'))

    print("The image has been split and saved to the output folder.")

def split_image_nxm(
        image_path,
        n,
        m,
        total_width: float = 832, # in mm
        total_height: float = 589, # in mm
        outer_margin: float = 5, # in mm
        inner_margin: float = 1 # in mm
    ):
    # Open the image
    img = Image.open(image_path)
    
    # Calculate effective width and height
    effective_width = total_width - 2 * outer_margin
    effective_height = total_height - 2 * outer_margin
    
    # Calculate tile widths and heights
    tile_widths = [total_width / n - 2*inner_margin] * n
    tile_heights = [total_height / m - 2*inner_margin] * m
    # Correct outer cell sizes to account for outer margin
    tile_widths[0] -= outer_margin - inner_margin
    tile_widths[-1] -= outer_margin - inner_margin
    tile_heights[0] -= outer_margin - inner_margin
    tile_heights[-1] -= outer_margin - inner_margin

    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Draw the total size rectangle
    total_rect = patches.Rectangle((0, 0), total_width, total_height, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(total_rect)
    
    # Plot rectangles for tiles
    for i in range(n):
        for j in range(m):
            x = outer_margin + sum(tile_widths[:i]) + 2*i * inner_margin
            y = outer_margin + sum(tile_heights[:j]) + 2*j * inner_margin
            rect = patches.Rectangle((x, y), tile_widths[i], tile_heights[j], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    # Plot the image centered within the effective area
    img_x = outer_margin
    img_y = outer_margin
    img_extent = [img_x, img_x + effective_width, img_y + effective_height, img_y]
    ax.imshow(img, extent=img_extent)
    
    # Draw cut lines for inner margins
    for i in range(1, n): # vertical cut lines
        cut_x = outer_margin + sum(tile_widths[:i]) + (2*i-1) * inner_margin
        print(f"vert cut {i} at {cut_x}")
        ax.axvline(x=cut_x, color='g', linestyle='--')
    
    for j in range(1, m): # horizontal cut lines
        cut_y = outer_margin + sum(tile_heights[:j]) + (2*j-1) * inner_margin
        print(f"horz cut {j} at {cut_y}")
        ax.axhline(y=cut_y, color='g', linestyle='--')
    
    # Set limits and aspect ratio
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.set_aspect('equal')
    
    plt.show()


if __name__ == "__main__":
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()

    # Open a file picker dialog to choose the image file
    image_path = filedialog.askopenfilename(
        title='Select an image file',
        filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp')]
    )

    # # Open a folder picker dialog to choose the output directory
    # output_folder = filedialog.askdirectory(title='Select an output folder')

    # split_image_into_4_parts(image_path, output_folder)
    
    split_image_nxm(
        image_path,
        3, 3,
        total_width = 832, # in mm
        total_height = 589, # in mm
        outer_margin = 2, # in mm
        inner_margin = 1 # in mm
    )
