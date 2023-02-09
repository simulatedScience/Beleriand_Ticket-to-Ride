"""
This module contains graph analysis class `TTR_Graph_Analysis`, which is used to analyze a given graph.
Key features:
- find the shortest path between two nodes - done
- find the shortest paths for all tasks - done
- visulization tool for tasks - done (untested)
- analyze importance of nodes - done (untested)
- analyze importance of edges - done (untested)
- calculate statistics for the graph: - done
  - distribution of node degrees - done
  - distribution of edge lengths - done
  - distribution of edge colors - done
  - distribution of task lengths - done
  - distribution of colors needed for tasks - done
- tools for visualization of the analysis - done
"""
import random
from math import sqrt, ceil
from typing import List, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class TTR_Graph_Analysis:
  """
  This class is used to analyze a given graph.
  """
  def __init__(self,
      locations: List[str],
      paths: List[Tuple[str, str, int, str]],
      tasks: List[Tuple[str, str]]
      ):
    """
    Initialize the class with a graph.

    Args:
        locations (List[str]): list of location names (nodes)
        paths (List[Tuple[str, str, int, str]]): list of paths (start location, end location, length, color)
        tasks (List[Tuple[str, str]]): list of tasks (start location, end location)
    """
    self.locations: List[str] = locations # list of location names (nodes)
    self.paths: List[Tuple[str, str, int, str]] = paths # list of paths (start location, end location, length, color)
    self.tasks: List[Tuple[str, str]] = tasks # list of tasks (start location, end location)

    self.networkx_graph: nx.Graph = create_graph(locations, paths) # networkx graph object containing location and path information
    self.task_lengths: dict[Tuple[str, str], int] = self.get_task_lengths() # shortest path lengths for all tasks

# access methods for basic graph information

  def get_locations(self) -> List[str]:
    """
    Get the list of location names (nodes).

    Returns:
        List[str]: list of locations
    """
    return self.locations

  def get_paths(self) -> List[Tuple[str, str, int, str]]:
    """
    Get the list of paths (edges)

    Returns:
        List[Tuple[str, str, int, str]]: list of edges
    """
    return self.paths

  def get_tasks_with_lengths(self) -> List[Tuple[str, str]]:
    """
    Get the list of tasks.

    Returns:
        List[Tuple[str, str, int]]: list of tasks (start location, end location)
    """
    tasks_with_lengths = [(loc1, loc2, length) for (loc1, loc2), length in zip(self.tasks, self.task_lengths)]
    return tasks_with_lengths

  def number_of_locations(self) -> int:
    """
    Get the number of locations (nodes) in the graph.

    Returns:
        int: number of locations
    """
    return len(self.locations)

  def number_of_paths(self) -> int:
    """
    Get the number of paths (edges) in the graph.

    Returns:
        int: number of paths
    """
    return len(self.paths)

  def number_of_tasks(self) -> int:
    """
    Get the number of tasks in the graph.

    Returns:
        int: number of tasks
    """
    return len(self.tasks)

# methods to calculate basic graph information

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
      edge = (location_list[i], location_list[i+1])
      cost += self.networkx_graph.edges[edge]["length"]
    return cost

  def get_task_lengths(self, graph: nx.Graph = None) -> dict[Tuple[str, str], int]:
    """
    Get the lengths of all tasks.

    Args:
        graph (nx.Graph, optional): graph to use for the analysis. Defaults to None (use the graph from the class)

    Returns:
        dict[Tuple[str, str], int]: dictionary of lengths for each task.
    """
    if graph is None:
      graph = self.networkx_graph
    task_lengths = {}
    for (loc1, loc2) in self.tasks:
      path, length = self.get_shortest_path(loc1, loc2, graph)
      task_lengths[(loc1, loc2)] = length
    return task_lengths

# pathfinding methods

  def get_shortest_path(self, loc1: str, loc2: str, graph: nx.Graph = None) -> Tuple[List[str], int]:
    """
    Find the shortest path between two locations.

    Args:
        loc1 (str): start location
        loc2 (str): end location
        graph (nx.Graph, optional): graph to use for the analysis. Defaults to None (use the graph from the class)

    Returns:
        List[str]: list of locations on the shortest path
        int: length of the shortest path
    """
    if graph is None:
      graph = self.networkx_graph
    try:
      path = nx.shortest_path(graph, loc1, loc2, weight="length")
    except nx.exception.NetworkXNoPath:
      return [], float("inf")
    return path, self.get_path_cost(path)

  def get_all_shortest_paths(self, loc1: str, loc2: str) -> List[Tuple[List[str], int]]:
    """
    Find all paths between the given locations that have the minimum possible length.

    Args:
        loc1 (str): start location
        loc2 (str): end location

    Returns:
        List[Tuple[List[str], int]]: list of shortest paths between te given locations and their lengths
    """
    paths = nx.all_shortest_paths(self.networkx_graph, loc1, loc2, weight="length")
    shortest_paths = []
    for path in paths:
      shortest_paths.append((path, self.get_path_cost(path)))
    return shortest_paths

  def get_shortest_task_paths(self) -> List[Tuple[List[str], int]]:
    """
    Find a shortest path for all tasks. If there are multiple shortest paths, there is no guarantee which one will be returned.

    Returns:
        List[Tuple[List[str], int]]: list of a shortest path and it's length for each task
    """
    shortest_task_paths = []
    for (loc1, loc2) in self.tasks:
      shortest_path = self.get_shortest_path(loc1, loc2)
      length = self.get_path_cost(shortest_path)
      shortest_task_paths.append((shortest_path, length))
    return shortest_task_paths

  def get_random_shortest_task_paths(self) -> List[Tuple[List[str], int]]:
    """
    Find shortest path for all tasks. If there are multiple shortest paths, choose one (uniformly) randomly. Multiple calls to this function will likely return different paths.

    Returns:
        List[Tuple[List[str], int]]: list of a random shortest path and it's length for each task
    """
    shortest_task_paths = []
    for (loc1, loc2) in self.tasks:
      shortest_paths = self.get_all_shortest_paths(loc1, loc2)
      shortest_path, length = random.choice(shortest_paths)
      shortest_task_paths.append((shortest_path, length))
    return shortest_task_paths

  def get_random_shortest_task_paths_edge_counts(self, n_random_paths: int = 10000) -> dict[Tuple[str, str], int]:
    """
    For each edge, count how many of the shortest paths for all tasks go through that edge.
    Returns a dictionary with edges as keys (pairs of location names) and the number of shortest paths that go through that edge as values.
    If there are multiple shortest paths for a task, choose one (uniformly) randomly.

    Args:
        n_random_paths (int, optional): number of random paths to use. Defaults to 1000.

    Returns:
        dict[[str, str], int]: dictionary of edges and the number of shortest paths that go through that edge
    """
    # shortest_task_paths = self.get_random_shortest_task_paths()
    # edge_counts = {}
    # for (path, length) in shortest_task_paths:
    #   for i in range(len(path) - 1):
    #     edge = (path[i], path[i+1])
    #     if edge in edge_counts:
    #       edge_counts[edge] += 1
    #     else:
    #       edge_counts[edge] = 1
    # return edge_counts
    task_edge_counts: dict[Tuple[str, str], int] = {}
    for task in self.tasks:
      shortest_paths = self.get_all_shortest_paths(loc1=task[0], loc2=task[1])
      for _ in range(n_random_paths):
        path, length = random.choice(shortest_paths)
        for (loc1, loc2) in zip(path[:-1], path[1:]):
          edge = (loc1, loc2)
          if edge in task_edge_counts:
            task_edge_counts[edge] += 1
          else:
            task_edge_counts[edge] = 1
    return task_edge_counts

# connectedness analysis methods

  def get_node_importance(self) -> dict[str, float]:
    """
    Calculate the importance of each node in the graph: how much the average task length is increased if that node is removed.

    Returns:
        dict[str, float]: dictionary of nodes and the avergae increase in task length if that node is removed
    """
    node_importance = {}
    average_task_length = self.get_average_task_length()
    for node in self.networkx_graph.nodes:
      # copy graph and remove node
      new_graph = self.networkx_graph.copy()
      new_graph.remove_node(node)
      node_importance[node] = self.get_average_task_length(new_graph) - average_task_length
    return node_importance

  def get_edge_importance(self) -> dict[Tuple[str, str], float]:
    """
    Calculate the importance of each edge in the graph: how much the average task length is increased if that edge is removed.

    Returns:
        dict[Tuple[str, str], float]: dictionary of edges and the avergae increase in task length if that edge is removed
    """
    edge_importance = {}
    average_task_length = self.get_average_task_length()
    for edge in self.networkx_graph.edges:
      # copy graph and remove edge
      new_graph = self.networkx_graph.copy()
      new_graph.remove_edge(*edge)
      edge_importance[edge] = self.get_average_task_length(new_graph) - average_task_length
    return edge_importance

  def get_average_task_length(self, graph: nx.Graph = None) -> float:
    """
    Calculate the average task length in the graph.

    Args:
        graph (nx.Graph, optional): graph to use for the analysis. Defaults to None (use the graph from the class)

    Returns:
        float: average task length
    """
    if graph is None:
      graph = self.networkx_graph
    task_lengths = list(self.get_task_lengths(graph).values())
    return sum(task_lengths) / len(task_lengths)

# plottable graph information (various distributions)

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

  def plot_node_degree_distribution(self, ax: plt.Axes, color: str = "#5588ff", grid_color: str = None, **bar_plot_kwargs):
    """
    Plot the degree distribution of the nodes as a bar plot.

    Args:
        ax (plt.Axes, optional): axis to plot on. Defaults to None (a new figure is created).
        color (str, optional): color of the bars. Defaults to "#5588ff".
        grid_color (str, optional): color of the grid. Defaults to None (no grid).
        bar_plot_kwargs: keyword arguments passed to the `matplotlib.pyplot.bar` plot function
    """
    degree_distribution = self.get_node_degree_distribution()
    ax.bar(degree_distribution.keys(), degree_distribution.values(), color=color, **bar_plot_kwargs)
    x_ticks = get_ticks(1, max(degree_distribution.keys()) + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    y_ticks = get_ticks(0, max(degree_distribution.values()) + 1, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("node degree")
    ax.set_ylabel("count")
    ax.set_title("Node degree distribution")
    if grid_color is not None:
      ax.grid(axis="y", color=grid_color)

  def get_edge_length_distribution(self, color: str = None) -> dict[int, int]:
    """
    Calculate the distribution of edge lengths. If a color is given, only edges of that color are considered.

    Args:
        color (str, optional): color of the edges to consider. Defaults to None (all edges are considered).

    Returns:
        dict[int, int]: dictionary of edge lengths and how often they occur in the graph
    """
    edge_length_distribution = {}
    for edge in self.networkx_graph.edges:
      if color is not None and self.networkx_graph.edges[edge]["color"] != color:
        # skip edges of the wrong color
        continue
      length = self.networkx_graph.edges[edge]["length"]
      if length in edge_length_distribution:
        edge_length_distribution[length] += 1
      else:
        edge_length_distribution[length] = 1
    return edge_length_distribution

  def plot_edge_length_distribution(self, ax: plt.Axes, color: str = None, plot_color: str = None, grid_color: str = None, **bar_plot_kwargs):
    """
    Plot the distribution of edge lengths as a bar plot. If a color is given, only edges of that color are considered.

    Args:
        ax (plt.Axes, optional): axis to plot on. Defaults to None (a new figure is created).
        color (str, optional): color of the edges to consider. Defaults to None (all edges are considered).
        plot_color (str, optional): color of the bars. Defaults to None (the color of the edges is used).
        grid_color (str, optional): color of the grid. Defaults to None (no grid).
        bar_plot_kwargs: keyword arguments passed to the `matplotlib.pyplot.bar` plot function
    """
    if color is not None and plot_color is None:
      plot_color = color
    if color is None and plot_color is None:
      plot_color = "#5588ff"
    edge_length_distribution = self.get_edge_length_distribution(color)
    ax.bar(edge_length_distribution.keys(), edge_length_distribution.values(), color=plot_color, **bar_plot_kwargs)
    x_ticks = get_ticks(1, max(edge_length_distribution.keys()) + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    y_ticks = get_ticks(0, max(edge_length_distribution.values()) + 1, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("edge length")
    ax.set_ylabel("count")
    title = "Edge length distribution"
    if color is not None:
      title += f" for color {color}"
    ax.set_title(title)
    if grid_color is not None:
      ax.grid(axis="y", color=grid_color)

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

  def plot_edge_color_length_distribution(self,
      ax: plt.Axes,
      color_map: dict[str, str] = None,
      alpha: float = 0.7,
      grid_color: str = None,
      **plot_kwargs):
    """
    Plot the distribution of edge lengths for each color. Each color is plotted as a line.

    Args:
        ax (plt.Axes, optional): axis to plot on. Defaults to None (a new figure is created).
        color_map (dict[str, str], optional): mapping from edge colors to colors for plotting. Defaults to None (each color is plotted in its own color).
        alpha (float, optional): alpha value for the lines. Defaults to 0.7.
        plot_kwargs: additional keyword arguments for the plot function
    """
    edge_color_length_distribution: dict[str, dict[int, int]] = self.get_edge_color_length_distribution()
    if color_map is None:
      color_map = {color: color for color in edge_color_length_distribution}
    max_length = max([max(edge_color_length_distribution[color].keys()) for color in edge_color_length_distribution])
    # add value 0 for missing lengths, then plot
    for color in edge_color_length_distribution:
      for length in range(1, max_length + 1):
        if length not in edge_color_length_distribution[color]:
          edge_color_length_distribution[color][length] = 0
      # sort by length
      x_values = sorted(edge_color_length_distribution[color].keys())
      y_values = [edge_color_length_distribution[color][length] for length in x_values]
      ax.plot(
          x_values,
          y_values,
          color=color_map[color],
          linestyle="--",
          marker=".",
          alpha=alpha,
          label=color,
          **plot_kwargs)
    x_ticks = get_ticks(1, max_length + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    y_ticks = get_ticks(0, max([max(edge_color_length_distribution[color].values()) for color in edge_color_length_distribution]) + 1, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("edge length")
    ax.set_ylabel("count")
    ax.set_title("Edge length distribution for each color")
    # ax.legend()
    if grid_color is not None:
      ax.grid(axis="both", color=grid_color)

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

  def plot_edge_color_distribution(self, ax: plt.Axes, color_map: dict[str, str] = None, grid_color: str = None, **bar_plot_kwargs):
    """
    Plot the distribution of edge colors counting each edge once towards the color regardless of its length as a bar plot.

    Args:
        ax (plt.Axes, optional): axis to plot on. Defaults to None (a new figure is created).
        color_map (dict[str, str], optional): mapping from edge colors to colors for plotting. Defaults to None (each color is plotted in its own color).
        grid_color (str, optional): color of the grid. Defaults to None (no grid).
        bar_plot_kwargs: keyword arguments passed to the `matplotlib.pyplot.bar` plot function
    """
    edge_color_distribution = self.get_edge_color_distribution()
    if color_map is None:
      color_map = {color: color for color in edge_color_distribution}
    ax.bar(
        edge_color_distribution.keys(),
        edge_color_distribution.values(),
        color=[color_map[color] for color in edge_color_distribution],
        **bar_plot_kwargs)
    y_ticks = get_ticks(0, max(edge_color_distribution.values()) + 1, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("edge color")
    ax.set_ylabel("count")
    ax.set_title("Edge color distribution")
    if grid_color is not None:
      ax.grid(axis="y", color=grid_color)

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

  def plot_edge_color_total_length_distribution(self, ax: plt.Axes, color_map: dict[str, str] = None, grid_color: str = None, **bar_plot_kwargs):
    """
    Plot the total length of edges for each color as a bar plot.

    Args:
        ax (plt.Axes, optional): axis to plot on. Defaults to None (a new figure is created).
        color_map (dict[str, str], optional): mapping from edge colors to colors for plotting. Defaults to None (each color is plotted in its own color).
        grid_color (str, optional): color of the grid. Defaults to None (no grid).
        bar_plot_kwargs: keyword arguments passed to the `matplotlib.pyplot.bar` plot function
    """
    edge_color_total_length_distribution = self.get_edge_color_total_length_distribution()
    if color_map is None:
      color_map = {color: color for color in edge_color_total_length_distribution}
    ax.bar(edge_color_total_length_distribution.keys(), edge_color_total_length_distribution.values(), color=[color_map[color] for color in edge_color_total_length_distribution], **bar_plot_kwargs)
    y_ticks = get_ticks(0, max(edge_color_total_length_distribution.values()) + 1, 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("edge color")
    ax.set_ylabel("total length")
    ax.set_title("Edge color total length distribution")
    if grid_color is not None:
      ax.grid(axis="y", color=grid_color)

  def get_task_length_distribution(self) -> dict[int, int]:
    """
    Calculate the distribution of task lengths.

    Returns:
        dict[int, int]: dictionary of task lengths and their counts
    """
    task_length_distribution = {}
    for length in self.task_lengths.values():
      if length in task_length_distribution:
        task_length_distribution[length] += 1
      else:
        task_length_distribution[length] = 1
    return task_length_distribution

  def plot_task_length_distribution(self, ax: plt.Axes, plot_color: str = "#5588ff", grid_color: str = None, **bar_plot_kwargs):
    """
    Plot the distribution of task lengths as a bar plot.

    Args:
        ax (plt.Axes, optional): axis to plot on. Defaults to None (a new figure is created).
        plot_color (str, optional): color of the bars. Defaults to "#5588ff".
        grid_color (str, optional): color of the grid. Defaults to None (no grid).
        bar_plot_kwargs: keyword arguments passed to the `matplotlib.pyplot.bar` plot function
    """
    task_length_distribution = self.get_task_length_distribution()
    ax.bar(
        task_length_distribution.keys(),
        task_length_distribution.values(),
        color=plot_color,
        **bar_plot_kwargs)
    x_ticks = get_ticks(1, max(task_length_distribution.keys()), 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    y_ticks = get_ticks(0, max(task_length_distribution.values()), 10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("task length")
    ax.set_ylabel("count")
    ax.set_title("Task length distribution")
    if grid_color is not None:
      ax.grid(axis="y", color=grid_color)

  def get_task_color_avg_distribution(self, n_random_paths: int = 1000) -> dict[str, Tuple[float, float]]:
    """
    Calculate how often each color is required to complete all tasks.
    Assume that the shortest path for each task is used.
    If there are multiple shortest paths, choose one (uniformly) randomly.
    Repeat this process for a large number of times and calculate the average and standard deviation of the distribution.

    Algorithm used:
      For each task:
        Get all shortest paths
        For k in {1,…,n_repetitions}:
          Choose random shortest path
          For each edge in shortest path:
            Add length to it's color
      Calculate average & (corrected sample) standard deviation

    Args:
        n_random_paths (int, optional): number of times to repeat the process. Defaults to 1000.

    Returns:
        dict[str, Tuple(float, float)]: dictionary of edge colors and their average and standard deviation of the number of times they are required to complete all tasks
          key: color
          value: tuple of average and standard deviation
    """
    task_color_length_counts: dict[str, dict[int, int]] = {}
    for task in self.tasks:
      shortest_paths = self.get_all_shortest_paths(loc1=task[0], loc2=task[1])
      for _ in range(n_random_paths):
        path, length = random.choice(shortest_paths)
        for (loc1, loc2) in zip(path[:-1], path[1:]):
          edge = (loc1, loc2)
          color = self.networkx_graph.edges[edge]["color"]
          length = self.networkx_graph.edges[edge]["length"]
          if color not in task_color_length_counts: # initialize dict for color
            task_color_length_counts[color] = {}
          if length not in task_color_length_counts[color]: # initialize dict for length
            task_color_length_counts[color][length] = 0
          task_color_length_counts[color][length] += 1
    # calculate average and standard deviation for each color
    task_color_avg_distribution: dict[str, Tuple(float, float)] = {}
    for color in task_color_length_counts:
      # calculate average
      avg = sum([length * count for length, count in task_color_length_counts[color].items()]) / n_random_paths
      # calculate standard deviation
      std = sqrt(sum([((length - avg) ** 2) * count for length, count in task_color_length_counts[color].items()]) / (n_random_paths - 1))
      task_color_avg_distribution[color] = (avg, std)

    return task_color_avg_distribution

  def plot_task_color_avg_distribution(self,
      ax: plt.Axes,
      color_map: dict[str, str] = None,
      errorbar_color: str = "#222222",
      grid_color: str = None,
      **bar_plot_kwargs):
    """
    Plot the average distribution of task colors as a bar plot.

    Args:
        ax (plt.Axes, optional): axis to plot on. Defaults to None (a new figure is created).
        color_map (dict[str, str], optional): mapping from edge colors to colors for plotting. Defaults to None (each color is plotted in its own color).
        bar_plot_kwargs: keyword arguments passed to the `matplotlib.pyplot.bar` plot function
    """
    task_color_avg_distribution = self.get_task_color_avg_distribution()
    if color_map is None:
      color_map = {color: color for color in task_color_avg_distribution}
    mapped_colors = [color_map[color] for color in task_color_avg_distribution]
    ax.bar(
        task_color_avg_distribution.keys(),
        [avg for avg, _ in task_color_avg_distribution.values()],
        color=mapped_colors,
        # yerr=[std for _, std in task_color_avg_distribution.values()],
        # capsize=5,
        # ecolor=errorbar_color,
        **bar_plot_kwargs)
    ax.set_xlabel("task color")
    ax.set_ylabel("avg color count for all tasks")
    ax.set_title("Task color avg requirements distribution")
    if grid_color is not None:
      ax.grid(axis="y", color=grid_color)


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

def get_ticks(min_val: float, max_val: float, max_n_ticks: int = 10, int_ticks: bool = True):
  """
  get ticks for a plot

  Args:
      min_val (float): minimum value
      max_val (float): maximum value
      max_n_ticks (int, optional): maximum number of ticks. Defaults to 10.
      int_ticks (bool, optional): whether to use integer ticks. Defaults to False.

  Returns:
      List[float]: list of ticks
  """
  # get ticks
  if not int_ticks:
    ticks = np.linspace(min_val, max_val, max_n_ticks)
    return ticks

  tick_step = max(1, int((max_val - min_val) // max_n_ticks))
  ticks = np.arange(min_val, max_val + 1, tick_step)
  return ticks


if __name__ == "__main__":
  # Create a graph
  locations = ["Menegroth", "Nargothrond", "Dor-Lomin", "Amon-Rudh"]
  paths = [
    ("Menegroth", "Nargothrond", 4, "blue"),
    ("Menegroth", "Dor-Lomin", 4, "red"),
    ("Menegroth", "Amon-Rudh", 3, "blue"),
    ("Nargothrond", "Dor-Lomin", 2, "blue"),
    ("Nargothrond", "Amon-Rudh", 2, "red"),
  ]
  tasks = [
    ("Amon-Rudh", "Dor-Lomin"),
    ("Nargothrond", "Menegroth"),
  ]

  ttr_graph = TTR_Graph_Analysis(locations=locations, paths=paths, tasks=tasks)

  # test plot functions
  grid_color: str = "#dddddd"
  # create 3x3 grid of plots
  fig, axs = plt.subplots(3, 3, figsize=(15, 15))
  # plot node degree distribution
  ttr_graph.plot_node_degree_distribution(ax=axs[0, 0], grid_color=grid_color)
  # plot edge length distribution
  ttr_graph.plot_edge_length_distribution(ax=axs[1, 0], grid_color=grid_color)
  # plot edge color distribution
  ttr_graph.plot_edge_color_distribution(ax=axs[2, 0], grid_color=grid_color)
  # plot edge color length distribution
  ttr_graph.plot_edge_color_length_distribution(ax=axs[0, 1], grid_color=grid_color)
  # plot edge color total length distribution
  ttr_graph.plot_edge_color_total_length_distribution(ax=axs[1, 1], grid_color=grid_color)
  # plot task length distribution
  ttr_graph.plot_task_length_distribution(ax=axs[1, 2], grid_color=grid_color)
  # plot task color distribution
  ttr_graph.plot_task_color_avg_distribution(ax=axs[0, 2], grid_color=grid_color)
  
  fig.tight_layout()
  fig.subplots_adjust(left=0.025, bottom=0.05, right=0.975, top=0.95, wspace=0.15, hspace=0.3)
  plt.show()