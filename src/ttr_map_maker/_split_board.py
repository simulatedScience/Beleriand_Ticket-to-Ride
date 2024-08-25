import os
import sys
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pylatex import Document, NoEscape, MiniPage, Command, Package

from _task_card_pdf_generation import add_horizontal_crop_marks, add_vertical_crop_marks

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

def add_fullpage_image(
        doc: Document,
        x_pos: float,
        y_pos: float,
        image_path: str,
        tile_width: float = 88.9,
        tile_height: float = 63.5,):
    """
    Adds a single card (front and back) to the document.

    Args:
        doc (Document): The LaTeX document.
        front_path (str): File path to the front image.
        back_path (str): File path to the back image.
        flip_backside (bool): Whether to flip the back image.
        card_width (float): Width of the card in mm.
        card_height (float): Height of the card in mm.
        gap (float): Gap between front and back in mm.
        vspace (float): Vertical space between cards in mm.
    """
    
    # Start tikzpicture for precise image placement
    doc.append(NoEscape(r'\begin{tikzpicture}[overlay, remember picture]'))

    # Place the image at the specified location with rotation if needed
    if tile_width > tile_height:
        x_shift = x_pos + tile_height / 2
        y_shift = y_pos + tile_width / 2
        doc.append(NoEscape(
            f'\\node[anchor=center,rotate=-90] at ([xshift={x_shift}mm,yshift=-{y_shift}mm]current page.north west) '
            f'{{\\includegraphics[height={tile_height}mm,width={tile_width}mm,keepaspectratio=FALSE]{{{image_path}}}}};'
        ))
    else:
        x_shift = x_pos + tile_width / 2
        y_shift = y_pos + tile_height / 2
        doc.append(NoEscape(
            f'\\node[anchor=center] at ([xshift={x_shift}mm,yshift=-{y_shift}mm]current page.north west) '
            f'{{\\includegraphics[height={tile_height}mm,width={tile_width}mm,keepaspectratio=FALSE]{{{image_path}}}}};'
        ))


    # End tikzpicture
    doc.append(NoEscape(r'\end{tikzpicture}'))

    # Add crop marks
    add_horizontal_crop_marks(doc, x_pos, y_pos, tile_width, tile_height)
    add_vertical_crop_marks(doc, x_pos, y_pos, tile_width, tile_height)
    

def calculate_cut_lines_and_tiles(
        tile_columns: int,
        tile_rows: int,
        total_board_width: float = 832, # in mm
        total_board_height: float = 589, # in mm
        outer_margin: float = 5, # in mm
        inner_margin: float = 1 # in mm
    ) -> tuple[list[tuple[float, float, float, float]], list[float], list[float]]:
    # Calculate tile widths and heights
    tile_widths = [total_board_width / tile_columns - 2 * inner_margin] * tile_columns
    tile_heights = [total_board_height / tile_rows - 2 * inner_margin] * tile_rows

    tile_widths[0] -= outer_margin - inner_margin
    tile_widths[-1] -= outer_margin - inner_margin
    tile_heights[0] -= outer_margin - inner_margin
    tile_heights[-1] -= outer_margin - inner_margin

    tiles = [] # list of bboxes as (left, bottom, width, height)
    vertical_cut_lines = []
    horizontal_cut_lines = []
    
    for i in range(tile_columns):
        for j in range(tile_rows):
            x = outer_margin + sum(tile_widths[:i]) + 2 * i * inner_margin
            y = outer_margin + sum(tile_heights[:j]) + 2 * j * inner_margin
            tile = (x, y, tile_widths[i], tile_heights[j])
            tiles.append(tile)
    
    for i in range(1, tile_columns):
        cut_x = outer_margin + sum(tile_widths[:i]) + (2 * i - 1) * inner_margin
        vertical_cut_lines.append(cut_x)
    
    for j in range(1, tile_rows):
        cut_y = outer_margin + sum(tile_heights[:j]) + (2 * j - 1) * inner_margin
        horizontal_cut_lines.append(cut_y)
    
    return tiles, horizontal_cut_lines, vertical_cut_lines

def plot_tiles_and_cut_lines(
        image_path: str,
        tile_bboxes: list[tuple[float, float, float, float]],
        horizontal_cut_lines: list[float],
        vertical_cut_lines: list[float],
        total_board_width: float,
        total_board_height: float,
    ):
    """
    Given an image and the bounding boxes of tiles and cut lines plot all on a board of the given site to visualize tiles and cut lines

    Args:
        image_path (str): 
        tile_bboxes (list[tuple[float, float, float, float]]): 
        horizontal_cut_lines (list[float]): 
        vertical_cut_lines (list[float]): 
        total_board_width (float): 
        total_board_height (float): 
    """
    # Open the image
    img = Image.open(image_path)
    
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Draw the total board size rectangle
    total_rect = patches.Rectangle((0, 0), total_board_width, total_board_height, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(total_rect)
    
    # Plot rectangles for tiles
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
    for tile in tile_bboxes:
        x, y, w, h = tile
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x + w > max_x:
            max_x = x + w
        if y + h > max_y:
            max_y = y + h
        patch = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(patch)
    
    # Get image extent as min and max in rectangles
    img_extent = [min_x, max_x, min_y, max_y] # (left, right, bottom, top)
    ax.imshow(img, extent=img_extent)
    
    # Draw cut lines for inner margins
    for cut_x in vertical_cut_lines:  # vertical cut lines
        ax.axvline(x=cut_x, color='g', linestyle='--')
    
    for cut_y in horizontal_cut_lines:  # horizontal cut lines
        ax.axhline(y=cut_y, color='g', linestyle='--')
    
    # Set limits and aspect ratio
    ax.set_xlim(-2, total_board_width+2)
    ax.set_ylim(-2, total_board_height+2)
    ax.set_aspect('equal')
    
    plt.show()

def export_sub_images(
        image_path: str,
        output_folder: str,
        tiles,
        total_board_width,
        total_board_height,
        outer_margin: float,
        output_prefix='cut') -> list[str]:
    img = Image.open(image_path)
    img_width, img_height = img.size
    tile_columns = len(set([x for x, *_ in tiles]))
    tile_rows = len(set([y for _, y, *_ in tiles]))
    # convert mm to pixels
    scale_x = img_width / (total_board_width - 2*outer_margin)
    scale_y = img_height / (total_board_height - 2*outer_margin)
    
    first_rect_x: float = tiles[0][0]
    first_rect_y: float = tiles[0][1]
    
    image_paths = []
    for idx, tile in enumerate(tiles):
        x, y, w, h = tile
        x -= first_rect_x
        y -= first_rect_y
        left = int(x * scale_x)
        upper = int(y * scale_y)
        right = int((x + w) * scale_x)
        lower = int((y + h) * scale_y)
        
        cropped_img = img.crop((left, upper, right, lower))
        filename = f"{output_prefix}_{idx%tile_columns+1}_{idx//tile_rows+1}.png"
        filepath = os.path.join(output_folder, filename)
        image_paths.append(filepath)
        cropped_img.save(filepath)
        print(f"Saved sub-image {filename} to {output_folder}")

    return image_paths

def generate_board_latex(
        image_paths: list[str],
        tile_bboxes: list[tuple[float, float, float, float]],
        target_filepath: str,
        border: tuple[float, float] = (10, 10),
    ):
    if border[0] != border[1]:
        raise NotImplementedError("Different borders for left and top are not supported yet.")

    geometry_options = {
        "margin": f"{border[1]}mm"
    }
    doc = Document(documentclass='article', document_options='a4paper', geometry_options=geometry_options)
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('tikz'))
    doc.append(NoEscape(r'\pagestyle{empty}'))

    x_pos = border[0]
    y_pos = border[1]
    
    for tile, image_path in zip(tile_bboxes, image_paths):
        image_path: str = image_path.replace("\\", "/")
        add_fullpage_image(
            doc=doc,
            x_pos=x_pos,
            y_pos=y_pos,
            image_path=image_path,
            tile_width=tile[2],
            tile_height=tile[3]
        )
        doc.append(NoEscape(r'\newpage'))
    # Output as .tex file
    target_filepath_stripped = target_filepath[:-4] if ".tex" in target_filepath else target_filepath
    doc.generate_tex(target_filepath_stripped)
    print(f"Generated LaTeX file at {target_filepath}")

if __name__ == "__main__":
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()

    image_path = None
    # image_path = "../../projects/MiddleEarth_TTR/board_samples/10_board_preview.png"
    image_paths = None
    # image_paths = [
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_1_1.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_1_2.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_1_3.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_2_1.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_2_2.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_2_3.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_3_1.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_3_2.png",
    #     "../../projects/MiddleEarth_TTR/printing/board_08/Middle_Earth_3_3.png"
    # ]

    if not image_path:
        # Open a file picker dialog to choose the image file
        image_path = filedialog.askopenfilename(
            title='Select an image file',
            filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp')]
        )

    if not image_paths:
        # # Open a folder picker dialog to choose the output directory
        output_folder = filedialog.askdirectory(title='Select an output folder')
    else:
        output_folder = os.path.dirname(image_paths[0])
    if not output_folder:
        sys.exit("No output folder selected. Exiting.")

    # split_image_into_4_parts(image_path, output_folder)
    tile_columns = 3
    tile_rows = 3
    total_board_width = 832 # in mm
    total_board_height = 589 # in mm
    outer_margin = 2 # in mm
    inner_margin = 1 # in mm
    output_prefix = "Beleriand_v2.08"
    
    tile_bboxes, horz_cuts, vert_cuts = calculate_cut_lines_and_tiles(
        tile_columns = tile_columns,
        tile_rows = tile_rows,
        total_board_width = total_board_width, # in mm
        total_board_height = total_board_height, # in mm
        outer_margin = outer_margin, # in mm
        inner_margin = inner_margin # in mm
    )
    # plot_tiles_and_cut_lines(
    #     image_path,
    #     tile_bboxes,
    #     horz_cuts,
    #     vert_cuts,
    #     total_board_width = total_board_width,
    #     total_board_height = total_board_height,
    # )
    if not image_paths:
        image_paths = export_sub_images(
            image_path,
            output_folder,
            tile_bboxes,
            total_board_width,
            total_board_height,
            outer_margin=outer_margin,
            output_prefix=output_prefix
        )
    generate_board_latex(
        image_paths=image_paths,
        tile_bboxes=tile_bboxes,
        target_filepath=f"{output_folder}/{output_prefix}_board.tex",
        border=(10, 10),
    )
