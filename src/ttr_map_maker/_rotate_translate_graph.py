import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter.filedialog import askopenfilename, asksaveasfilename


def rotate_position(pos, angle):
    rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return np.dot(rotation_matrix, pos)

def fit_to_aspect_ratio(positions, aspect_ratio):
    x_coords, y_coords = zip(*positions)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    width = max_x - min_x
    height = max_y - min_y

    if width / height > aspect_ratio:
        new_height = width / aspect_ratio
        height_center = (min_y + max_y) / 2
        min_y = height_center - new_height / 2
        max_y = height_center + new_height / 2
    else:
        new_width = height * aspect_ratio
        width_center = (min_x + max_x) / 2
        min_x = width_center - new_width / 2
        max_x = width_center + new_width / 2

    return max_x - min_x, max_y - min_y

def rotate_and_center_particles(data, angle, aspect_ratio, center_point):
    particles = data['particle_graph']['particles']
    rotated_positions = [rotate_position(np.array(particle['position']), angle) for particle in particles if 'position' in particle]

    bbox_width, bbox_height = fit_to_aspect_ratio(rotated_positions, aspect_ratio)
    rotated_positions = np.array(rotated_positions)
    current_center = np.mean(rotated_positions, axis=0)
    translation_vector = center_point - current_center
    translated_positions = rotated_positions + translation_vector

    min_x, min_y = np.min(translated_positions, axis=0)
    translated_positions -= np.array([min_x, min_y])

    return translated_positions, (bbox_width, bbox_height)

def update_positions_in_json(data, translated_positions):
    particles = data['particle_graph']['particles']
    for particle, pos in zip(particles, translated_positions):
        if 'position' in particle:
            particle['position'] = pos.tolist()

def plot_particles_with_corrected_approach(data, translated_positions, bbox_size):
    particles = data['particle_graph']['particles']

    for particle, pos in zip(particles, translated_positions):
        if 'position' in particle:
            particle_type = particle['particle_type']
            if particle_type == 'Particle_Node':
                plt.scatter(pos[0], pos[1], c='black', marker='o', s=50)
            elif particle_type == 'Particle_Label':
                plt.scatter(pos[0], pos[1], c='blue', marker='s', s=30)
            elif particle_type == 'Particle_Edge':
                plt.scatter(pos[0], pos[1], c='grey', marker='x', s=20)

    rect = patches.Rectangle((0, 0), bbox_size[0], bbox_size[1], linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

    plt.title("Particles with Correctly Positioned Bounding Box")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal')
    plt.show()

def main():
    # file picker
    filepath = askopenfilename(title="Select JSON particle graph file")
    # Load the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Define the optimal rotation angle and aspect ratio
    optimal_angle = -10  # Replace with your calculated angle
    aspect_ratio = 49.25 / 39.4
    center_point = np.array([49.25 / 2, 39.4 / 2])

    # Rotate and translate the positions
    translated_positions, bbox_size = rotate_and_center_particles(data, optimal_angle, aspect_ratio, center_point)

    # Update the JSON data
    update_positions_in_json(data, translated_positions)

    # let user choose where to save the file, default: same name as input file
    new_filepath = asksaveasfilename(
        title="Save JSON particle graph file",
        defaultextension=".json",
        initialfile=filepath.split('/')[-1],
    )
    # Save the updated JSON data to a file
    with open(new_filepath, 'w') as file:
        json.dump(data, file, indent=4)

    # Plot the particles
    plot_particles_with_corrected_approach(data, translated_positions, bbox_size)

if __name__ == "__main__":
    main()
