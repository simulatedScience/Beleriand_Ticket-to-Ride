"""
This module implements a class to handle editing TTR particle graphs through a GUI.

The class is called ParticleGraphEditor and can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.

This requires a tkinter frame where particle settings are displayed and edited as well as a matplotlib Axes object where the particle graph is displayed.
"""
import tkinter as tk
from typing import Tuple, List, Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import PickEvent, MouseEvent
import numpy as np

from ttr_particle_graph import TTR_Particle_Graph
from graph_particle import Graph_Particle
from particle_node import Particle_Node
from particle_label import Particle_Label
from particle_edge import Particle_Edge
from drag_handler import find_particle_in_list, get_artist_center

class Graph_Editor_GUI:
  def __init__(self,
      color_config: dict[str, str],
      tk_config_methods: dict[str, Callable],
      particle_graph: TTR_Particle_Graph,
      settings_frame: tk.Frame,
      ax: plt.Axes,
      canvas: FigureCanvasTkAgg,
      max_pick_range: float = 2.
      ):
    """
    Args:
        color_config: Dictionary of colors for the UI (see `Board_Layout_GUI`)
        tk_config_methods: Dictionary of methods to configure tkinter widgets (see `Board_Layout_GUI`). These should add styles to the widgets. Expect the following keys:
            - `add_frame_style(frame: tk.Frame)`
            - `add_label_style(label: tk.Label, headline_level: int, font_type: str)
            - `add_button_style(button: tk.Button)`
            - `add_entry_style(entry: tk.Entry, justify: str)`
            - `add_checkbutton_style(checkbutton: tk.Checkbutton)
            - `add_radiobutton_style(radiobutton: tk.Radiobutton)
        particle_graph: The particle graph to edit.
        settings_frame: The tkinter frame where the particle settings are displayed.
        ax: The matplotlib Axes object where the particle graph is displayed.
        canvas: the tkinter canvas where the mpl figure is shown
        max_pick_range: The maximum distance from a particle to the click event to still consider it as a click on the particle.
    """
    self.particle_graph = particle_graph
    self.settings_frame = settings_frame
    self.ax = ax
    self.canvas = canvas

    self.max_pick_range = max_pick_range
    # save color config
    self.color_config = color_config
    # Initialize methods for configuring tkinter widgets
    self.add_frame_style = tk_config_methods["add_frame_style"]
    self.add_label_style = tk_config_methods["add_label_style"]
    self.add_button_style = tk_config_methods["add_button_style"]
    self.add_entry_style = tk_config_methods["add_entry_style"]
    self.add_checkbutton_style = tk_config_methods["add_checkbutton_style"]
    self.add_radiobutton_style = tk_config_methods["add_radiobutton_style"]

    self.highlighted_particles: List[Graph_Particle] = [] # particles that are highlighted
    self.selected_particle: Graph_Particle = None # particle for which settings are displayed
    self.preselected_particle: Graph_Particle = None # intermediate variable to store selected particle between pick and release events

    self.pick_event_cid: int = None # connection id for pick event
    self.release_event_cid: int = None # connection id for release event

    self.bind_mouse_events()


  def bind_mouse_events(self):
    """
    Bind mouse events to the matplotlib Axes object.
    """
    self.pick_event_cid: int = self.canvas.mpl_connect("pick_event", self.on_mouse_click)

  def unbind_mouse_events(self):
    if self.pick_event_cid is not None:
      self.canvas.mpl_disconnect(self.pick_event_cid)
    if self.release_event_cid is not None:
      self.canvas.mpl_disconnect(self.release_event_cid)
    self.pick_event_cid = None

  def on_mouse_click(self, event: PickEvent):
    """
    Handle mouse clicks on the matplotlib Axes object.
    """
    # ignore pick events from scrolling and mouse buttons other than left click
    if not event.mouseevent.button == 1:
      return
    ## if event.artist.get_gid() == "background" and self.selected_particle is not None:
    ##   self.clear_selection()
    ##   return
    # get the center of the artist
    artist_center = get_artist_center(event.artist)
    ## check if click is on a particle
    ## TODO: use cell list for faster search
    ## if self.use_cell_list: # search using cell list
    ##   potential_particles = self.find_cell_particles(artist_center)
    ##   particle = find_particle_in_list(artist_center, potential_particles, max_pick_range=self.max_pick_range)
    ## else: # search in all particles
    particle = find_particle_in_list(artist_center, self.particle_graph.get_particle_list())#, max_pick_range=self.max_pick_range)
    
    # if no particle was clicked or the selected one was clicked again, deselect all particles
    print(f"previous selection: {self.selected_particle.get_id() if self.selected_particle is not None else None}")
    print(f"current highlighted particles: {[p.get_id() for p in self.highlighted_particles]}")
    print(f"selected particle: {particle.get_id()}")
    if particle is None or particle == self.selected_particle:
      print(f"remove highlights from {self.selected_particle.get_id()} (clicked highlighted particle)")
      self.clear_selection()
      self.canvas.draw_idle()
      print(f"new selection: {self.selected_particle.get_id() if self.selected_particle is not None else None}")
      print(f"new highlighted particles: {[p.get_id() for p in self.highlighted_particles]}")
      print("-"*25)
      return
    
    # select clicked particle
    self.preselected_particle: Graph_Particle = particle
    self.release_event_cid: int = self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
    self.canvas.mpl_disconnect(self.pick_event_cid)
    

  def on_mouse_release(self, event: MouseEvent):
    """
    Handle mouse releases on the matplotlib Axes object.
    """
    self.canvas.mpl_disconnect(self.release_event_cid)
    self.release_event_cid = None
    # if a particle was clicked, select it
    self.select_particle(self.preselected_particle)
    self.preselected_particle = None
    self.canvas.draw_idle()
    print(f"new selection: {self.selected_particle.get_id() if self.selected_particle is not None else None}")
    print(f"new highlighted particles: {[p.get_id() for p in self.highlighted_particles]}")
    print("-"*25)
    self.canvas.mpl_connect("pick_event", self.on_mouse_click)

  def select_particle(self, particle: Graph_Particle, add_to_selection: bool = False, highlight_color: str = "#cc00cc"):
    """
    Select a particle to display its settings and highlight it

    Args:
        particle (Graph_Particle): The particle to select.
        add_to_selection (bool, optional): If True, the particle will be added to the current selection. Otherwise the current selection will be replaced. Defaults to False.
        highlight_color (str, optional): The color to highlight the particle with. Defaults to "#cc00cc".
    """
    if len(self.highlighted_particles) == 0: # no particle was selected yet
      # select clicked particle, show it's settings and highlight it.
      print("show particle settings")
      self.highlighted_particles = [particle]
    elif add_to_selection:
      self.highlighted_particles.append(particle)
    else:
      self.clear_selection() # remove highlights from previous selection
      self.highlighted_particles = [particle]
    # select particle and highlight it
    self.selected_particle = particle
    print(f"highlight particle: {particle.get_id()}")
    particle.highlight(ax=self.ax, highlight_color=highlight_color)
    # show settings
    if isinstance(particle, Particle_Node):
      self.show_node_settings(particle)
    elif isinstance(particle, Particle_Label):
      self.show_label_settings(particle)
    elif isinstance(particle, Particle_Edge):
      self.show_edge_settings(particle)


  def clear_selection(self):
    """
    remove highlights from all highlighted particles, reset internal variables and hide particle settings
    """
    for highlight_particle in self.highlighted_particles:
      highlight_particle.remove_highlight(ax=self.ax)
    self.highlighted_particles: List[Graph_Particle] = [] # particles that are highlighted
    if self.selected_particle is not None:
      # self.selected_particle.remove_highlight(ax=self.ax)
      self.selected_particle: Graph_Particle = None


  def show_node_settings(self, particle_node: Particle_Node):
    """
    Display the settings of a node.

    """
    pass

  def show_label_settings(self, particle_label: Particle_Label):
    """
    Display the settings of a label.
    """
    pass

  def show_edge_settings(self, particle_edge: Particle_Edge):
    """
    Display the settings of an edge.
    """
    pass