"""
This module contains the TTR_task class, which is used to represent a task/ ticket/ destination card.

Tasks contain the following information:
  - name of the task?
  - contained node names
  - number of nodes in the task (if > 2, task has bonus nodes in between start and end node)
  - length of the task (shortest path between start and end node)
  - points for completing the task
  - points for completing the task with bonus nodes
  - negative points for not completing the task
"""
from typing import List
import json

import numpy as np
import matplotlib.pyplot as plt

class TTR_Task:
  def __init__(self,
      node_names: List[str],
      length: int = None,
      points: int = None,
      points_bonus: int = None,
      points_penalty: int = None,
      name: str = None):
    """
    This method initializes a TTR_task object.
    if no nodes or name is given, the task will be named "_empty_task_"

    Args:
        node_names (List[str]): The names of the nodes in the task.
        length (int, optional): The length of the task. Defaults to None.
        points (int, optional): The points for completing the task. Defaults to None.
        points_bonus (int, optional): The points for completing the task with bonus nodes. Defaults to None.
        points_penalty (int, optional): The negative points for not completing the task. Defaults to None.
        name (str, optional): The name of the task. Defaults to None (will be set to "{first node name} - {last node name}").
    """
    self.node_names: List[str] = node_names
    if not node_names and name is None:
      self.name = "_empty_task_"
    else:
      self.name = name if name is not None else f"{node_names[0]} - {node_names[-1]}"
    self.length: int = length
    self.points: int = points if points is not None else len(node_names)
    self.points_bonus: int = points_bonus
    self.points_penalty: int = points_penalty
    self.plotted_objects: List[plt.Line2D] = []


  def set_node_names(self, node_names: List[str], update_name: bool = True) -> None:
    self.node_names = node_names
    if update_name:
      self.name = f"{node_names[0]} - {node_names[-1]}"

  def set_length(self, length: int) -> None:
    """
    Set the length of the task.

    Args:
        length (int): 
    """
    self.length = length

  def set_points(self, points: int, points_bonus: int = None, points_penalty: int = None) -> None:
    """
    Set the points for completing the task (`points`)),
      the points for completing the task with bonus nodes (`points_bonus`) and
      the (negative) points for not completing the task (`points_penalty`).

    Args:
        points (int): The points for completing the task.
        points_bonus (int, optional): The points for completing the task with bonus nodes. Defaults to None.
        points_penalty (int, optional): The (negative) points for not completing the task. Defaults to None.
    """
    self.points = points
    self.points_bonus = points_bonus
    self.points_penalty = points_penalty


  def is_empty(self):
    return self.name == "_empty_task_"

  def __bool__(self):
    return not self.is_empty()


  def draw(self,
      ax: plt.Axes,
      particle_graph: "TTR_Particle_Graph",
      color: str = "#cc5500",
      linewidth: float = 6.0,
      linestyle: str = "--",
      alpha: float = 0.8,
      zorder: int = 6,
      override_positions: List[np.ndarray] = None,) -> None:
    """
    Draw the task on the given axes using straight lines between the nodes contained in the task. The lines

    Args:
        ax (plt.Axes): The axes to draw on.
        particle_graph (TTR_Particle_Graph): The particle graph to draw.
        color (str, optional): The color to draw the task in. Defaults to "black".
        linewidth (float, optional): The linewidth to draw the task in. Defaults to 5.0.
        linestyle (str, optional): The linestyle to draw the task in. Defaults to "--".
        alpha (float, optional): The alpha value to draw the task in. Defaults to 1.0.
        zorder (int, optional): The zorder to draw the task in. Defaults to 6.
        override_positions (List[np.ndarray], optional): If given, the positions of the nodes will be overridden by the given positions. Defaults to None.
    """
    # get node positions
    if override_positions:
      node_positions = override_positions
    else:
      node_positions = [particle_graph.particle_nodes[location].position for location in self.node_names]
    positions_x, positions_y = list(zip(*node_positions))
    # draw lines between nodes
    self.plotted_objects.append(ax.plot(
        positions_x,
        positions_y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder
        )[0]
    )

  def erase(self):
    """
    Erase the task from the axes.
    """
    if self.plotted_objects:
      for obj in self.plotted_objects:
        obj.remove()
      self.plotted_objects: List[plt.Line2D] = []


  def __str__(self):
    return f"Task: {self.name}: {self.node_names}"


  def to_dict(self) -> dict:
    """
    Prepare the task for json serialization.

    Returns:
        dict: The task as dict.
    """
    # summarize task in dict
    task_info: dict = {
      "name": self.name,
      "node_names": self.node_names,
      "length": self.length,
      "points": self.points,
      "points_bonus": self.points_bonus,
      "points_penalty": self.points_penalty
    }
    return task_info

  def to_json(self) -> str:
    """
    Prepare the task for json serialization.

    Returns:
        str: The task as json string.
    """
    return json.dumps(self.to_dict(), indent=2)


  @staticmethod
  def from_dict(task_info: dict) -> "TTR_Task":
    """
    Create a task from a dict.

    Args:
        task_info (dict): The task as dict (as created by `to_dict()`):
            name (str): The name of the task.
            node_names (List[str]): The names of the nodes in the task.
            length (int): The length of the task.
            points (int): The points for completing the task.
            points_bonus (int): The points for completing the task with bonus nodes.
            points_penalty (int): The negative points for not completing the task.

    Returns:
        TTR_Task: The task as `TTR_Task` object.
    """
    return TTR_Task(
      node_names=task_info["node_names"],
      length=task_info["length"],
      points=task_info["points"],
      points_bonus=task_info["points_bonus"],
      points_penalty=task_info["points_penalty"],
      name=task_info["name"]
    )



if __name__ == "__main__":
  empty_task = TTR_Task(node_names=[])
  print(empty_task, "\t\tis empty:", empty_task.is_empty())