"""
This module contains graph analysis class `TTR_Graph_Analysis`, which is used to analyze a given graph.
Key features:
- find the shortest path between two nodes - done (untested)
- find the shortest paths for all tasks - done (untested)
- analyze importance of nodes
- analyze importance of edges
- calculate statistics for the graph:
  - distribution of node degrees - done (untested)
  - distribution of edge lengths - done (untested)
  - distribution of edge colors - done (untested)
  - distribution of task lengths - done (untested)
  - distribution of colors needed for tasks
- tools for visualization of the analysis
"""
import random
from typing import List, Tuple

import networkx as nx

class TTR_Graph_Analysis:
  """
  This class is used to analyze a given graph.
  """
  def __init__(self,
      locations: List[str],
      paths: List[Tuple[str, str, int, str]],
      tasks: List[Tuple[str, str, int]]
      ):
    """
    Initialize the class with a graph.
    """
    self.locations = locations
    self.paths = paths
    self.tasks = tasks
    self.networkx_graph = create_graph(locations, paths)


  def get_path_cost(self, location_list: List[str]) -> int:
    """
    Calculate the cost of a path.

    Args:
        location_list (List[str]): list of locations on the path

    Returns:
        int: cost of the path
    """
    cost = 0
    for i in range(len(location_list) - 1):
      cost += self.networkx_graph[location_list[i]][location_list[i+1]]["length"]
    return cost


  def get_shortest_path(self, loc1: str, loc2: str) -> List[str]:
    """
    Find the shortest path between two locations.

    Args:
        loc1 (str): start location
        loc2 (str): end location

    Returns:
        List[str]: list of locations on the shortest path
        int: length of the shortest path
    """
    path = nx.shortest_path(self.networkx_graph, loc1, loc2)
    return path, self.get_path_cost(path)

  def get_all_shortest_paths(self, loc1: str, loc2: str) -> List[Tuple[List[str], int]]:
    """
    Find all paths between the given locations that have the minimum possible length.

    Args:
        loc1 (str): start location
        loc2 (str): end location

    Returns:
        List[Tuple[List[str], int]]: list of shortest paths and their lengths
    """
    paths = nx.all_shortest_paths(self.networkx_graph, loc1, loc2)
    shortest_paths = []
    for path in paths:
      shortest_paths.append((path, self.get_path_cost(path)))
    return shortest_paths

  def get_shortest_task_paths(self) -> List[Tuple[List[str], int]]:
    """
    Find the shortest path for all tasks.

    Returns:
        List[Tuple[List[str], int]]: list of shortest paths and their lengths for each task
    """
    shortest_task_paths = []
    for (loc1, loc2, length) in self.tasks:
      shortest_task_paths.append(self.get_shortest_path(loc1, loc2))
    return shortest_task_paths

  def get_random_shortest_task_paths(self) -> List[Tuple[List[str], int]]:
    """
    Find shortest path for all tasks. If there are multiple shortest paths, choose one (uniformly) randomly.

    Returns:
        List[Tuple[List[str], int]]: list of shortest paths and their lengths for each task
    """
    shortest_task_paths = []
    for (loc1, loc2, length) in self.tasks:
      shortest_paths = self.get_shortest_path(loc1, loc2)
      shortest_task_paths.append(random.choice(shortest_paths))
    return shortest_task_paths


  def get_node_degree_distribution(self) -> dict[int, int]:
    """
    Calculate the degree distribution of the nodes.

    Returns:
        dict[int, int]: dictionary of degrees and their counts
    """
    degree_distribution = {}
    for node in self.networkx_graph.nodes:
      degree = self.networkx_graph.degree[node]
      if degree in degree_distribution:
        degree_distribution[degree] += 1
      else:
        degree_distribution[degree] = 1
    return degree_distribution

  def get_edge_length_distribution(self, color: str = None) -> dict[int, int]:
    """
    Calculate the distribution of edge lengths. If a color is given, only edges of that color are considered.

    Args:
        color (str, optional): color of the edges to consider. Defaults to None (all edges are considered).

    Returns:
        dict[int, int]: dictionary of edge lengths and their counts
    """
    edge_length_distribution = {}
    for edge in self.networkx_graph.edges:
      if color is not None and self.networkx_graph.edges[edge]["color"] != color:
        continue
      length = self.networkx_graph.edges[edge]["length"]
      if length in edge_length_distribution:
        edge_length_distribution[length] += 1
      else:
        edge_length_distribution[length] = 1
    return edge_length_distribution

  def get_edge_color_length_distribution(self) -> dict[str, dict[int, int]]:
    """
    Calculate the distribution lengths of edges for each color.

    Returns:
        dict[str, dict[int, int]]: dictionary of edge colors and their length distributions. See get_edge_length_distribution for the format of the length distributions.
    """
    edge_color_length_distribution: dict = {}
    for edge in self.networkx_graph.edges:
      color: str = self.networkx_graph.edges[edge]["color"]
      length: int = self.networkx_graph.edges[edge]["length"]
      if color not in edge_color_length_distribution:
        edge_color_length_distribution[color]: dict = {}
      if length in edge_color_length_distribution[color]:
        edge_color_length_distribution[color][length] += 1
      else:
        edge_color_length_distribution[color][length]: int = 1
    return edge_color_length_distribution

  def get_edge_color_distribution(self) -> dict[str, int]:
    """
    Calculate the distribution of edge colors counting each edge once towards the color regardless of its length.

    Returns:
        dict[str, int]: dictionary of edge colors and their counts
    """
    edge_color_length_distribution = self.get_edge_color_length_distribution()
    edge_color_distribution = {}
    for color in edge_color_length_distribution:
      edge_color_distribution[color] = sum(edge_color_length_distribution[color].values())
    return edge_color_distribution

  def get_edge_color_total_length_distribution(self) -> dict[str, int]:
    """
    Count the total length of edges for each color.

    Returns:
        dict[str, int]: dictionary of edge colors and their total lengths
    """
    edge_color_length_distribution = self.get_edge_color_length_distribution()
    edge_color_total_length_distribution = {}
    for color in edge_color_length_distribution:
      edge_color_total_length_distribution[color] = sum([length * count for length, count in edge_color_length_distribution[color].items()])
    return edge_color_total_length_distribution

  def get_task_length_distribution(self) -> dict[int, int]:
    """
    Calculate the distribution of task lengths.

    Returns:
        dict[int, int]: dictionary of task lengths and their counts
    """
    task_length_distribution = {}
    for (loc1, loc2, length) in self.tasks:
      if length in task_length_distribution:
        task_length_distribution[length] += 1
      else:
        task_length_distribution[length] = 1
    return task_length_distribution

  def get_task_color_avg_distribution(self) -> dict[str, Tuple(float, float)]:
    """
    Calculate how often each color is required to complete all tasks. Assume that the shortest path for each task is used. If there are multiple shortest paths, choose one (uniformly) randomly.
    Repeat this process for a large number of times and calculate the average and standard deviation of the distribution.

    Returns:
        dict[str, Tuple(float, float)]: dictionary of edge colors and their average and standard deviation of the number of times they are required to complete all tasks
    """
    task_color_avg_distribution = {}
    for (loc1, loc2, length) in self.tasks:
      shortest_paths = self.get_all_shortest_paths(loc1, loc2)
      # TODO: complete this function



def create_graph(locations: List[str], paths: List[Tuple[str, str, int, str]]) -> nx.Graph:
  """
  create a networkx graph from locations and paths

  Args:
      locations (List[str]): list of location names for the nodes
      paths (List[Tuple[str, str, int, str]]): list of edges specified by the two locations, the length and the color

  Returns:
      networkx.Graph: the graph
  """
  # Create an empty graph
  nx_graph = nx.Graph()
  # Add nodes and edges to the graph
  nx_graph.add_nodes_from(locations)
  for (loc1, loc2, length, color) in paths:
      nx_graph.add_edge(loc1, loc2, length=length, color=color)
  return nx_graph