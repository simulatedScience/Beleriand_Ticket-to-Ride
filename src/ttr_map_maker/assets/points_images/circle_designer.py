
"""
Use a loaded template image as alpha mask to cut out a portion of a texture image.
Then apply filters for 3D effect
"""
from PIL import Image, ImageDraw, ImageFilter
import os
from image_filters import apply_gradient_shading, apply_highlights, apply_texture


# mask_image = Image.open("assets\\points_images\\circle_template.png")
# mask_image = Image.open("assets\\points_images\\disk_template.png")
# mask_image = Image.open("assets\\points_images\\circle_template_thin.png")
mask_image = Image.open("assets\\points_images\\circle_design_flowery.png")
# mask_image = Image.open("assets\\points_images\\circle_template_slotted.png")
# make mask white where alpha > 0

save_filepath = "assets\\points_images"
# texture_path = "assets\\points_images\\stone_texture.png"
# texture_path = "assets\\points_images\\dark_stone_texture.png"
texture_path = "assets\\points_images\\gold.jpg"
image_size = mask_image.size
print(image_size)
image = Image.new("RGBA", image_size, (0, 0, 0, 0))
# apply texture
image = apply_texture(mask_image, texture_path)
# apply shading
image = apply_gradient_shading(image, top_opacity=0, bottom_opacity=80)
# apply highlights
light_direction: tuple = (1,1) # light from top left
image = apply_highlights(image, light_direction, intensity=200, blur_radius=41)
# image = apply_highlights(image, light_direction, intensity=200, blur_radius=19)
# save image
# image.save(os.path.join(save_filepath, f"circle_design_python_thick.png"))
# image.save(os.path.join(save_filepath, f"circle_design_python.png"))
image.save(os.path.join(save_filepath, f"circle_design_flowery_gold.png"))