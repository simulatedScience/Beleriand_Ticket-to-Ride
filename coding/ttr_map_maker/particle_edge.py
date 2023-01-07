"""
A class representing nodes of a particle graph. Each node has a current and target position and a label.
There is an attraction force between the node and target position as well as between the node and the label.

A node can be connected to other nodes by edges.
"""
import numpy as np
import matplotlib.pyplot as plt

from graph_particle import Graph_Particle


class Particle_Edge(Graph_Particle):
  def __init__(self,
        color: str,
        position: np.ndarray = np.array([0, 0]),
        rotation: float = 0,
        mass: float = 0.1,
        node_attraction: float = 0.1,
        edge_attraction: float = 0.1,
        interaction_radius: float = 5,
        velocity_decay: float = 0.9999,
        repulsion_strength: float = 1,
        ):
    super().__init__(
        position,
        rotation = rotation,
        target_position = None,
        mass = mass,
        bounding_box_size = (4, 1),
        interaction_radius = interaction_radius,
        velocity_decay = velocity_decay,
        repulsion_strength=repulsion_strength,
    )
    self.node_attraction = node_attraction
    self.edge_attraction = edge_attraction
    self.color = color


  def get_attraction_forces(self, other_particle):
    """get attraction force between this particle and the other particle

    Args:
        other_particle (Graph_Particle): other particle

    Returns:
        np.ndarray: attraction force
    """
    if isinstance(other_particle, Particle_Edge):
      return self.get_edge_attraction_force(other_particle)
    else: # other_particle is Particle_Node
      return self.get_node_attraction_force(other_particle)


  def get_edge_attraction_force(self, other_edge: "Particle_Edge"):
    """
    get attraction force between this particle and the other edge depending on the minimum distance between midpoints of edge's bounding boxes shortest edges.
    This uses the helper function `get_edge_midpoints()`

    Args:
        other_edge (Particle_Edge): other edge

    Returns:
        np.ndarray: attraction force
    """
    min_distance = np.inf
    closest_points = np.zeros((2, 2))
    for point_1 in self.get_edge_midpoints():
      for point_2 in other_edge.get_edge_midpoints():
        distance = np.linalg.norm(point_1 - point_2)
        if distance < min_distance:
          min_distance = distance
          closest_points[0, :] = point_1
          closest_points[1, :] = point_2
    
    force_direction = (closest_points[1, :] - closest_points[0, :]) / min_distance
    translation_force = self.edge_attraction * self.attraction_from_distance(min_distance) * force_direction
    return translation_force, closest_points[0, :]


  def get_node_attraction_force(self, node: Graph_Particle):
    """
    get attraction force between this particle and the node depending on the minimum distance between the node and the edge's bounding box's shortest edges.
    This uses the helper function `get_edge_midpoints()`

    Args:
        node (Particle_Node): node

    Returns:
        np.ndarray: attraction force
    """
    min_distance = np.inf
    closest_point = np.zeros(2)
    for point in self.get_edge_midpoints():
      distance = np.linalg.norm(point - node.position)
      if distance < min_distance:
        min_distance = distance
        closest_point = point

    force_direction = (node.position - closest_point) / min_distance
    translation_force = self.node_attraction * self.attraction_from_distance(min_distance) * force_direction

    force_anchor = closest_point

    return translation_force, force_anchor


  def get_edge_midpoints(self, eps=1e-8):
    """
    get the midpoints of the edges of the bounding box of this edge

    Returns:
        np.ndarray: midpoints of the edges of the bounding box of this edge
    """
    midpoints = np.zeros((2, 2))
    found_n_shortest_edges = 0
    for i in range(4):
      corner_1 = self.bounding_box[i]
      corner_2 = self.bounding_box[(i + 1) % 4]
      if np.linalg.norm(corner_1 - corner_2) - np.min(self.bounding_box_size) < eps:
        midpoints[found_n_shortest_edges, :] = (corner_1 + corner_2) / 2
        found_n_shortest_edges += 1
        if found_n_shortest_edges == 2:
          break
    return midpoints

  
  def attraction_from_distance(self, distance):
    """
    get attraction force depending on the distance

    Args:
        distance (float): distance

    Returns:
        float: attraction force
    """
    return distance**2 / 2
    return (np.exp(distance) - 1) / 3


  def draw(self,
      ax: plt.Axes,
      color: str = "",
      alpha: float = 0.7,
      zorder: int = 4):
    super().draw_bounding_box(ax, color, alpha, zorder)
    # midpoints = self.get_edge_midpoints()
    # self.plotted_objects.append(
    #     ax.plot(midpoints[:, 0], midpoints[:, 1], color=color, alpha=alpha, zorder=zorder)
    # )