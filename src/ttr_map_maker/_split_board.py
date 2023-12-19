import os
from tkinter import Tk, filedialog
from PIL import Image

def split_image_into_parts():
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()

    # Open a file picker dialog to choose the image file
    file_path = filedialog.askopenfilename(
        title='Select an image file',
        filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp')]
    )

    # Open a folder picker dialog to choose the output directory
    output_folder = filedialog.askdirectory(title='Select an output folder')

    # If no file or folder was selected, return
    if not file_path or not output_folder:
        print("File or output folder not selected. Exiting.")
        return

    # Load the image
    original_image = Image.open(file_path)

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

# Run the function
split_image_into_parts()
