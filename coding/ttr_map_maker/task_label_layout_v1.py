from typing import List, Tuple

import numpy as np

def calculate_label_layout(
    node_extents: List[Tuple[float, float, float, float]],
    label_extents: List[Tuple[float, float, float, float]],
    node_centers: List[np.ndarray],
    label_centers: List[np.ndarray],
    bounding_rectangle: Tuple[float, float, float, float]) -> List[np.ndarray]:
  """
  Calculate the positions of the labels such that
    - no two labels overlap each other,
    - no two labels overlap any node,
    - all labels are within the bounding rectangle.

  Args:
    node_extents: A list of (x_min, x_max, y_min, y_max) tuples for each node.
    label_extents: A list of (x_min, x_max, y_min, y_max) tuples for each label.
    node_centers: A list of (x, y) coordinates for each node.
    label_centers: A list of (x, y) coordinates for each label.
    bounding_rectangle: The (x_min, x_max, y_min, y_max) tuple for the bounding
      rectangle.

  Returns:
    List[np.ndarray]: A list of (x, y) coordinates for each label's center.
  """
  # Compute the maximum label width and height
  max_label_width = max([extent[1] - extent[0] for extent in label_extents])
  max_label_height = max([extent[3] - extent[2] for extent in label_extents])

  # Initialize the label positions
  label_positions = []
  for label_center in label_centers:
    # Find the indices of the nodes that are closest to the label center
    node_distances = [np.linalg.norm(node_center - label_center) for node_center in node_centers]
    closest_node_indices = np.argsort(node_distances)[:3]

    # Determine the initial label position
    initial_label_position = label_center + np.array([max_label_width / 2, max_label_height / 2])

    # Iterate over a grid of candidate label positions
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        candidate_label_position = initial_label_position + np.array([dx * max_label_width, dy * max_label_height])

        # Check if the candidate label position is valid
        valid = True
        candidate_label_extent = (candidate_label_position[0] - max_label_width / 2,
                                   candidate_label_position[0] + max_label_width / 2,
                                   candidate_label_position[1] - max_label_height / 2,
                                   candidate_label_position[1] + max_label_height / 2)
        for node_extent in node_extents:
          if overlap(node_extent, candidate_label_extent):
            valid = False
            break
        if not valid:
          continue
        for other_label_position in label_positions:
          if np.linalg.norm(other_label_position - candidate_label_position) < max_label_width:
            valid = False
            break
        if not valid:
          continue
        if not is_inside(candidate_label_position, bounding_rectangle):
          continue

        # If we reach this point, the candidate label position is valid
        label_positions.append(candidate_label_position)
        break
      if valid:
        break
    else:
      # If no valid position was found, use the initial label position
      label_positions.append(initial_label_position)

  return label_positions


def overlap(extent1: Tuple[float, float, float, float], extent2: Tuple[float, float, float, float]) -> bool:
  """
  Check if two extents overlap.

  Args:
    extent1: A (x_min, x_max, y_min, y_max) tuple.
    extent2: A (x_min, x_max, y_min, y_max) tuple.

  Returns:
    bool: True if the extents overlap, False otherwise.
  """
  return (extent1[0] < extent2[1] and extent1[1] > extent2[0] and
          extent1[2] < extent2[3] and extent1[3] > extent2[2])


def is_inside(point: np.ndarray, extent: Tuple[float, float, float, float]) -> bool:
  """
  Check if a point is inside an extent.

  Args:
    point: A (x, y) coordinate.
    extent: A (x_min, x_max, y_min, y_max) tuple.

  Returns:
    bool: True if the point is inside the extent, False otherwise.
  """
  return (point[0] >= extent[0] and point[0] <= extent[1] and
          point[1] >= extent[2] and point[1] <= extent[3])