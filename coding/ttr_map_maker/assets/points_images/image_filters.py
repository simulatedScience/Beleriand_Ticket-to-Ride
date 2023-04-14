import random

import numpy as np
from PIL import Image
import cv2


def apply_texture(image, texture_path):
    texture = Image.open(texture_path)

    # Randomly crop the texture to match the image size
    crop_x = random.randint(0, texture.width - image.width)
    crop_y = random.randint(0, texture.height - image.height)
    texture = texture.crop((crop_x, crop_y, crop_x + image.width, crop_y + image.height))

    # Invert the alpha channel of the original image to create the mask
    mask = image.split()[3].point(lambda x: 255 - x)

    # Composite the original image and the texture using the mask
    result = Image.composite(image, texture, mask)

    # Preserve the original alpha channel
    result.putalpha(image.split()[3])

    return result


def apply_gradient_shading(image, top_opacity=20, bottom_opacity=150):
    # Crop the image to the non-transparent parts
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)

    # Create a custom linear gradient for the cropped image
    gradient = Image.new("L", cropped_image.size)
    for y in range(cropped_image.height):
        opacity = y * (bottom_opacity - top_opacity) // cropped_image.height + top_opacity
        for x in range(cropped_image.width):
            gradient.putpixel((x, y), opacity)

    # Combine the gradient with the cropped image's alpha channel
    alpha_gradient = Image.new("RGBA", cropped_image.size)
    alpha_gradient.putalpha(gradient)

    # Apply the gradient to the cropped image
    shaded_image = Image.alpha_composite(cropped_image, alpha_gradient)

    # Paste the shaded image back onto the original image
    result = Image.new("RGBA", image.size, (0, 0, 0, 0))
    result.paste(shaded_image, bbox)

    # Apply the alpha values of the original image
    result.putalpha(image.getchannel("A"))

    return result

def apply_highlights(image: Image, light_direction: tuple, intensity=128, blur_radius=15):
    np_image = np.array(image)
    light_kernel = np.array([[-1, -1, -0.5],
                             [-0.5, 0, 0.5],
                             [0.5, 1, 1]])
    dark_kernel = -light_kernel
    # Calculate light and dark highlights using the respective kernels
    light_highlights = calculate_highlights(np_image, light_kernel, intensity, blur_radius)
    dark_highlights = calculate_highlights(np_image, dark_kernel, intensity, blur_radius)
    # Create the highlights and shadows arrays
    highlights = np.zeros(np_image.shape, dtype=np.uint8)
    highlights[:, :, :3] = np.array([light_highlights, light_highlights, light_highlights]).transpose((1, 2, 0))
    shadows = np.array([dark_highlights, dark_highlights, dark_highlights]).transpose((1, 2, 0))
    # Create a black image with the same shape as np_image
    black_image = np.zeros(np_image.shape, dtype=np.uint8)
    # Create a shadow mask by using the shadow array as the alpha channel
    shadow_mask = shadows[:, :, 0]
    # Normalize the shadow mask to be in the range [0, 1]
    shadow_mask = shadow_mask / 255.0
    # Blend between the black image and np_image using the shadow mask as the alpha channel
    blended_image = (1 - shadow_mask)[:, :, np.newaxis] * np_image + shadow_mask[:, :, np.newaxis] * black_image
    # Add the highlights to the blended image
    blended_image[:, :, :3] += highlights[:, :, :3]
    # Clip the values to the valid range (0-255)
    blended_image = np.clip(blended_image, 0, 255)
    # Convert the resulting numpy array back to a PIL Image
    result = Image.fromarray(blended_image.astype(np.uint8))
    return result







def calculate_highlights(np_image, kernel, intensity=128, blur_radius=15):
    alpha_channel = np_image[:, :, 3]
    filtered_alpha = cv2.filter2D(alpha_channel, -1, kernel)
    # Create a mask that selects only the edges
    mask = filtered_alpha.copy()
    mask[mask <= 0] = 0
    mask[mask > 0] = intensity
    blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    return blurred_mask

# if __name__ == "__main__":
    # image = Image.open("assets/points_images/texture.png")
    # image = apply_shading(image)
    # image.show()