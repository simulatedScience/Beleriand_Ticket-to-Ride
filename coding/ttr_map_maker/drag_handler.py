"""
This module implements a class to handle drag and drop events for matplotlib artists.

It also provides tools to find a particle associated to that artist.
"""
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from graph_particle import Graph_Particle

class Drag_Handler:
  def __init__(self,
      canvas: FigureCanvasTkAgg,
      axis: plt.Axes,
      particle_list: List[Graph_Particle],
      particle_cell_list: List[List[int]] = None,
      cell_size: float = None,
      max_pick_range: float = 2.):
    """
    Initialize the Drag_Handler object.
    if a cell list is provided, the cell size must also be provided. When both are provided, the handler will use the cell list to find the particle associated to the artist more efficiently.

    Args:
      canvas (FigureCanvasTkAgg): The canvas to which the artists are drawn.
      particle_list (List[Graph_Particle]): The list of particles to which the artists are associated.
      particle_cell_list (List[List[int]], optional): a grid of cells and the particles in each cell as references to `particle_list`. Defaults to None.
      cell_size (float, optional): The size of the cells in the cell list. Defaults to None.
      max_pick_range (float, optional): The maximum distance from the mouse click to the artist's center that will be considered a pick. Defaults to 2.
    
    """
    self.canvas = canvas
    self.ax = axis
    self.particle_list = particle_list
    self.max_pick_range = max_pick_range
    self.particle_cell_list = None
    # if a cell list is provided, the cell size must also be provided
    if particle_cell_list is None or cell_size is None:
      self.use_cell_list: bool = False
    else:
      self.use_cell_list: bool = True
      self.particle_cell_list = particle_cell_list
      self.cell_size = cell_size

    self.cid_1: int = None # the id of the motion event
    self.cid_2: int = None # the id of the release event
    self.current_artist: plt.Artist = None # the artist that is currently being dragged
    self.current_particle: Graph_Particle = None # the particle associated to the artist that is currently being dragged
    self.canvas.mpl_connect("pick_event", self.on_pick)
    print("Drag handler initialized.")

  def on_pick(self, event):
    """
    This function is called when an artist is picked.
    It binds the motion and release events to the canvas.

    Args:
      event (matplotlib.backend_bases.PickEvent): The pick event.
    """
    if self.current_artist is not None:
      print("Warning: an artist was already being dragged.")
      event.xdata = event.mouseevent.x
      event.ydata = event.mouseevent.y
      self.on_release(event)
    # Get the rectangle artist that was picked
    self.current_artist = event.artist
    event_position = self.current_artist.get_center()

    # Bind the motion and button release events to the canvas
    self.cid_1 = self.canvas.mpl_connect("motion_notify_event", self.on_motion)
    self.cid_2 = self.canvas.mpl_connect("button_release_event", self.on_release)

    # find the particle associated to the artist
    if self.use_cell_list:
      potential_particles = self.find_cell_particles(event_position)
      self.current_particle = self.find_particle_in_list(event_position, potential_particles)
    else:
      self.current_particle = self.find_particle_in_list(event_position, self.particle_list)

    # print(f"picked artist: {event.artist}")
    # print(f"picked particle: {self.current_particle}, type: {type(self.current_particle)}")

  def on_motion(self, event):
    """
    This function is called when the mouse is moved while a button is pressed.
    It moves the artist to the mouse position.

    Args:
      event (matplotlib.backend_bases.MouseEvent): The mouse event.
    """
    if event.inaxes:
      self.current_artist.set_center(np.array([event.xdata, event.ydata]))
      # self.current_artist.set_ydata(event.mouseevent.y)
      self.canvas.draw_idle()


  def on_release(self, event):
    """
    This function is called when a mouse button is released.
    It unbinds the motion and release events from the canvas.

    Args:
      event (matplotlib.backend_bases.MouseEvent): The mouse event.
    """
    if self.cid_1 is not None:
      self.canvas.mpl_disconnect(self.cid_1)
      self.cid_1 = None
    if self.cid_2 is not None:
      self.canvas.mpl_disconnect(self.cid_2)
      self.cid_2 = None

    # update the particle's position
    x_pos, y_pos = event.xdata, event.ydata
    self.current_particle.set_position(np.array([x_pos, y_pos]))
    self.current_particle.erase()
    self.current_particle.draw(self.ax)

    self.current_artist = None
    self.current_particle = None


  def find_particle_in_list(self, event_position: np.ndarray, particle_list: List[Graph_Particle]) -> Graph_Particle:
    """
    Find the particle associated to the artist in the list of particles.
    Choose the particle that is closest to the click event but within the maximum pick range.

    Args:
      event_position (List[float]): The position of the click event.
      particle_list (List[Graph_Particle]): The list of particles to search in.

    Returns:
      Graph_Particle: The particle associated to the artist.
    """
    min_distance = np.inf
    for particle in particle_list:
      distance = np.linalg.norm(particle.position - event_position)
      if distance < min_distance:
        min_distance = distance
        closest_particle = particle
    if min_distance < self.max_pick_range:
      return closest_particle
    return None # if no particle is close enough


  def find_cell_particles(self, event_position: np.ndarray) -> List[Graph_Particle]:
    """
    Find all particles that can possibly be associated to the click event.
    The particles are found in the cell that contains the click event and in the surrounding cells.

    Args:
        event_position (np.ndarray): position of the click event.

    Returns:
        List[Graph_Particle]: list of particles that can be associated to the click event.
    """
    # find the cell that contains the click event
    cell_x = int(event_position[0] / self.cell_size)
    cell_y = int(event_position[1] / self.cell_size)
    # find the particles in the cell and the surrounding cells
    potential_particles = []
    for i in range(cell_x - 1, cell_x + 2):
      for j in range(cell_y - 1, cell_y + 2):
        if i >= 0 and j >= 0 and i < len(self.particle_cell_list) and j < len(self.particle_cell_list[0]):
          potential_particles.extend(self.particle_cell_list[i][j])
    # convert the list of indices to a list of particles
    return [self.particle_list[i] for i in potential_particles]