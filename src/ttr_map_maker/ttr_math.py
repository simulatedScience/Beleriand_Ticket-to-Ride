import numpy as np

def get_2d_rotation_matrix(theta: float) -> np.ndarray:
  """
  Get a 2D rotation matrix for a given angle.

  Args:
      theta (float): Angle in radians.

  Returns:
      np.ndarray: 2D rotation matrix.
  """
  sin_theta = np.sin(theta)
  cos_theta = np.cos(theta)
  return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

def rotate_point_around_point(
    point: np.ndarray,
    rotation_center: np.ndarray,
    rotation_angle: float) -> np.ndarray:
  """
  Rotate a point around another point by a given angle.

  Args:
      point (np.ndarray): point to be rotated
      rotation_center (np.ndarray): point around which the other point should be rotated
      rotation_angle (float): angle by which the point should be rotated

  Returns:
      np.ndarray: rotated point
  """
  return get_2d_rotation_matrix(rotation_angle) @ (point - rotation_center) + rotation_center