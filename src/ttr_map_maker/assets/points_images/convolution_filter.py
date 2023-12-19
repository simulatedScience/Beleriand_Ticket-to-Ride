import cv2
import numpy as np
from PIL import Image

# Load an image
img = cv2.imread("assets/points_images/points_standard/1.png", cv2.IMREAD_UNCHANGED)

# Extract the alpha channel
alpha_channel = img[:, :, 3]

# Define a convolution kernel for detecting upward edges
kernel = np.array([[-1, -1, -.5],
                   [-.5,  -.1, .5],
                   [ .5,  1, 1]])

# Apply the kernel to the alpha channel using cv2.filter2D()
filtered_alpha = cv2.filter2D(alpha_channel, -1, kernel)

# Create a mask that selects only the upward edges
mask = filtered_alpha.copy()
mask[mask <= 0] = 0
mask[mask > 0] = 255

# Create a new image with the same size as the original image, with a white background
white_bg = Image.new('RGBA', (img.shape[1], img.shape[0]), (255, 255, 255, 255))

# Create a new image with the mask as the alpha channel
edges = Image.fromarray(alpha_channel)
edges.putalpha(Image.fromarray(mask))

# Show the original image and the edges that point upwards side by side
Image.fromarray(img).show()
edges.show()
