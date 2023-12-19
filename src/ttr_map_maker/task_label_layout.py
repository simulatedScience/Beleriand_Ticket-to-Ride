from typing import List, Tuple

import numpy as np

def calculate_label_layout(
    node_extents: List[Tuple[float, float, float, float]],
    label_extents: List[Tuple[float, float, float, float]],
    node_centers: List[np.ndarray],
    label_centers: List[np.ndarray],
    bounding_rectangle: Tuple[float, float, float, float]
) -> List[np.ndarray]:
    """
    Calculate the positions of the labels such that:
        - no two labels overlap each other,
        - no label overlaps any node,
        - all labels are within the bounding rectangle.

    Args:
        node_extents: A list of (x_min, x_max, y_min, y_max) tuples for each node.
        label_extents: A list of (x_min, x_max, y_min, y_max) tuples for each label.
        node_centers: A list of (x, y) coordinates for each node.
        label_centers: A list of (x, y) coordinates for each label.
        bounding_rectangle: The (x_min, x_max, y_min, y_max) tuple for the bounding rectangle.

    Returns:
        A list of (x, y) coordinates for each label's center.
    """
    # Calculate the minimum distance between two labels (in pixels)
    min_distance = np.min([np.min(np.abs(np.subtract.outer(lc, label_centers))) for lc in label_centers])

    # Initialize the label positions to their centers
    label_positions = label_centers.copy()

    # Loop until all labels are properly positioned
    while True:
        # Initialize a flag to indicate if any label has moved
        moved = False

        # Iterate over each label
        for i, label_extent in enumerate(label_extents):
            # Calculate the indices of the nodes that intersect with the label
            node_indices = [j for j, node_extent in enumerate(node_extents) if intersects(node_extent, label_extent)]

            # Calculate the indices of the labels that intersect with the current label
            intersecting_labels = [j for j, other_extent in enumerate(label_extents) if i != j and intersects(other_extent, label_extent)]

            # Calculate the new position of the label
            new_position = get_new_position(
                label_extent,
                label_positions[i],
                node_centers[node_indices],
                label_centers[intersecting_labels],
                min_distance,
                bounding_rectangle
            )

            # Check if the label has moved
            if not np.array_equal(new_position, label_positions[i]):
                label_positions[i] = new_position
                moved = True

        # If no label has moved, exit the loop
        if not moved:
            break

    return label_positions

def intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    """
    Check if two rectangles intersect.

    Args:
        a: A tuple (x_min, x_max, y_min, y_max) representing the first rectangle.
        b: A tuple (x_min, x_max, y_min, y_max) representing the second rectangle.

    Returns:
        True if the rectangles intersect, False otherwise.
    """
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

def get_new_position(label_extent, node_extent, node_center, label_center):
    """
    Calculate the new center position of a label.

    Args:
        label_extent: A (left, right, bottom, top) tuple representing the extent of the label.
        node_extent: A (left, right, bottom, top) tuple representing the extent of the node.
        node_center: A (x, y) tuple representing the center of the node.
        label_center: A (x, y) tuple representing the current center of the label.

    Returns:
        A (x, y) tuple representing the new center position of the label.
    """
    # Calculate the maximum distance that the label can be moved horizontally and vertically
    dx = min(node_extent[0] - label_extent[0], 0) + max(node_extent[1] - label_extent[1], 0)
    dy = min(node_extent[2] - label_extent[2], 0) + max(node_extent[3] - label_extent[3], 0)

    # Calculate the ideal position of the label (centered on the node)
    ideal_position = node_center

    # Calculate the new position of the label (based on the ideal position and maximum allowable movement)
    new_position = np.array([ideal_position[0] + dx / 2, ideal_position[1] + dy / 2])

    # Make sure the new position is within the bounding rectangle
    new_position[0] = max(min(new_position[0], node_extent[1] - (label_extent[1] - label_extent[0]) / 2), node_extent[0] + (label_extent[1] - label_extent[0]) / 2)
    new_position[1] = max(min(new_position[1], node_extent[3] - (label_extent[3] - label_extent[2]) / 2), node_extent[2] + (label_extent[3] - label_extent[2]) / 2)
    
    return new_position
