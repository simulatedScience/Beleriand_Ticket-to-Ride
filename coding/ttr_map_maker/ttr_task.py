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

    Args:
        node_names (List[str]): The names of the nodes in the task.
        length (int, optional): The length of the task. Defaults to None.
        points (int, optional): The points for completing the task. Defaults to None.
        points_bonus (int, optional): The points for completing the task with bonus nodes. Defaults to None.
        points_penalty (int, optional): The negative points for not completing the task. Defaults to None.
        name (str, optional): The name of the task. Defaults to None (will be set to "{first node name} - {last node name}").
    """
    self.node_names: List[str] = node_names
    self.length: int = length
    self.points: int = points
    self.points_bonus: int = points_bonus
    self.points_penalty: int = points_penalty
    self.name: str = name if name is not None else f"{node_names[0]} - {node_names[-1]}"
