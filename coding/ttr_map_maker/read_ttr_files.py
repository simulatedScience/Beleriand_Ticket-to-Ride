"""
this module contains functions to read the input files for TTR maps represented as graphs
"""
import pickle
import json

from ttr_particle_graph import TTR_Particle_Graph

def read_locations(location_file: str):
  """
  read locations from txt file
  Each line contains the name of exactly one location.
  whitespace is stripped from the beginning and end of each line.
  empty lines are ignored.

  Args:
      location_file (str): path to the location file
  """
  locations = []
  with open(location_file, "r") as loc_file:
    for line in loc_file:
      line = line.strip()
      if line: # skip empty lines
        locations.append(line)

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
  with open(path_file, "r") as path_file:
    for line in path_file:
      line = line.strip()
      if line:
        loc_id1, loc_id2, length, color = line.split(" ; ")
        paths.append((loc_id1, loc_id2, int(length), color))

  return paths


def read_tasks(task_file: str):
  """
  read tasks from tasks file: list of location names and lengths, seperated by ` ; `

  Args:
      task_file (str): path to the task file

  Returns:
      List[Tuple[str, str, int]]: list of tasks as tuples of two location names and the length of the shortest path between them
  """
  tasks = []
  with open(task_file, "r") as task_file:
    for line in task_file:
      line = line.strip()
      if line:
        loc_1, loc_2, length = line.split(" ; ")
        tasks.append((loc_1, loc_2, int(length)))

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


def load_particle_graph_json(particle_graph_file: str) -> TTR_Particle_Graph:
  """
  load a particle graph from a JSON file

  Args:
      particle_graph_file (str): path to the pickle file

  Returns:
      ParticleGraph: particle graph
  """
  with open(particle_graph_file, "r") as file:
    particle_graph = TTR_Particle_Graph.load_json(file.read())

  return particle_graph
