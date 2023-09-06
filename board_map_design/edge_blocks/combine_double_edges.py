import os
from PIL import Image

def combine_images(input_images, border_image_name, output_folder):
  """Combines two images with transparency.

  Args:
    input_images: A list of images to combine.
    border_image: The border image to paste on top.
    output_folder: The folder to save the combined images to.
  """
  for i, image1_name in enumerate(input_images):
    for image2_name in input_images[i:]:
      image1 = Image.open(image1_name)
      image2 = Image.open(image2_name)
      # rotate image2
      image2 = image2.rotate(180)

      combined_image = Image.new('RGBA', (image1.width, image1.height))
      combined_image.paste(image1, (0, 0, image1.width, image1.height), mask=image1.convert('RGBA'))
      combined_image.paste(image2, (0, 0, image2.width, image2.height), mask=image2.convert('RGBA'))

      border_image = Image.open(border_image_name)
      combined_image.paste(border_image, (0, 0, border_image.width, border_image.height), mask=border_image.convert('RGBA'))

      filename = f'{os.path.basename(image1_name).split(".")[0]}-{os.path.basename(image2_name).split(".")[0]}.png'
      combined_image.save(os.path.join(output_folder, filename))
      # close images
      image1.close()
      image2.close()
      border_image.close()
      combined_image.close()

def main():
    input_image_paths = [
        "fire.png",
        "fighting.png",
        "electric.png",
        "grass.png",
        "water.png",
        "psychic.png",
        "dark.png",
        "fairy.png",
        "neutral.png",
    ]
    input_iumage_folder = "double_edge_bases_2"
    input_image_paths = [os.path.join(input_iumage_folder, path) for path in input_image_paths]
    # mask_image_path = "mask.jpg"
    border_image_path = "double_edge_border_2.png"
    output_folder = "double_edges_auto_2"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    combine_images(input_image_paths, border_image_path, output_folder)

if __name__ == "__main__":
    main()
