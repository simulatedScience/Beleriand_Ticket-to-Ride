"""
this module contains functions to read the input files for TTR maps represented as graphs
"""
import pickle
import json

from ttr_particle_graph import TTR_Particle_Graph
from ttr_task import TTR_Task

def read_locations(location_file: str) -> list[str]:
  """
  read locations from txt file
  Each line contains the name of exactly one location.
  whitespace is stripped from the beginning and end of each line.
  empty lines are ignored.

  Args:
      location_file (str): path to the location file
  """
  locations = []
  try:
    with open(location_file, "r") as loc_file:
      for line in loc_file:
        line = line.strip()
        if line: # skip empty lines
          locations.append(line)
  except FileNotFoundError:
    print(f"Warning: could not read locations from {location_file}")
  return locations


def read_paths(path_file: str):
  """
  read edges from paths file: list of pairs of location names, lengths and colors, seperated by ` ; `

  Args:
      path_file (str): path to the path file

  Returns:
      List[Tuple[str, str, int, str]]: list of edges as tuples of two location names, path lengths and a color name
  """
  paths = []
  try:
    with open(path_file, "r") as path_file:
      for line in path_file:
        line = line.strip()
        if line:
          loc_id1, loc_id2, length, color = line.split(" ; ")
          paths.append((loc_id1, loc_id2, int(length), color))
  except FileNotFoundError:
    print(f"Warning: could not read paths from {path_file}")
  return paths


def read_tasks(task_filepath: str) -> dict[str, TTR_Task]:
  """
  read tasks from tasks file: list of location names and lengths, seperated by ` ; `

  Args:
      task_file (str): path to the task file

  Returns:
      Dict[Tuple[str, str, int]]: list of tasks as tuples of two location names and the length of the shortest path between them
  """
  tasks: dict[str, TTR_Task] = {}
  try:
    with open(task_filepath, "r") as task_file:
      for line in task_file:
        line = line.strip()
        if line:
          loc_1, loc_2, *length = line.split(" ; ")
          task = TTR_Task(node_names=[loc_1, loc_2])
          tasks[task.name] = task
  except FileNotFoundError:
    print(f"Warning: could not read tasks from {task_filepath}")
  return tasks


def load_particle_graph_pickle(particle_graph_file):
  """
  load a particle graph from a pickle file

  Args:
      particle_graph_file (str): path to the pickle file

  Returns:
      ParticleGraph: particle graph
  """
  with open(particle_graph_file, "rb") as file:
    particle_graph = pickle.load(file)

  return particle_graph
