import cv2
import numpy as np
from PIL import Image
from tkinter import filedialog
from tkinter import Tk
import noise
import opensimplex
from scipy.spatial import cKDTree

def select_files():
    root = Tk()
    root.withdraw()  # Hides the small tk window.
    files = filedialog.askopenfilenames(parent=root, title='Choose files')
    return files

# def apply_stamp_effect(image):
#     # Split the image into color channels
#     b, g, r, a = cv2.split(image)

#     # Generate noise
#     noise = np.random.randint(0, 255, a.shape).astype(np.uint8)

#     # Update the alpha channel with the minimum of the original alpha and the noise
#     a = np.minimum(a, noise)

#     # Merge the channels back together
#     final = cv2.merge([b, g, r, a])

#     return final

import cv2
import numpy as np
import noise
import opensimplex
from scipy.spatial import cKDTree

def apply_noise(width, height, noise_type="perlin"):
    if noise_type == "perlin":
        # Perlin noise
        scale = 100
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        world = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                world[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        return world

    elif noise_type == "simplex":
        # OpenSimplex noise
        simplex = opensimplex.OpenSimplex()
        world = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                world[i][j] = simplex.noise2(i, j)
        return world

    elif noise_type == "worley":
        # Worley noise
        points = [[np.random.randint(0, height), np.random.randint(0, width)] for _ in range(100)]  # Generates Points(y, x)
        coord = np.dstack(np.mgrid[0:height, 0:width])  # Makes array with coordinates as values
        tree = cKDTree(points)  # Build Tree
        world = tree.query(coord, workers=-1)[0]  # Calculate distances (workers=-1: Uses all CPU Cores)
        return world

def apply_stamp_effect(image, noise_type="perlin"):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate noise
    noise = apply_noise(*gray.shape, noise_type)
    
    # Normalize noise to the range 0-255
    noise = cv2.normalize(noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply noise to the alpha channel of the image
    image[..., 3] = np.minimum(image[..., 3], noise)

    return image



def main():
    files = select_files()

    for file in files:
        # Read the image
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # Apply the stamp effect
        stamp_img = apply_stamp_effect(img, noise_type="perlin")

        # Save the image
        filename = file.split('/')[-1]
        cv2.imwrite('assets\\points_images\\points_standard\\' + filename, stamp_img)

if __name__ == '__main__':
    main()
