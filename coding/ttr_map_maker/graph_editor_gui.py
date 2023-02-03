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
      grid_padding: Tuple[float, float],
      particle_graph: TTR_Particle_Graph,
      settings_frame: tk.Frame,
      ax: plt.Axes,
      canvas: FigureCanvasTkAgg,
      max_pick_range: float = 2.,
      edge_color_list: List[str] = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]
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
    self.particle_graph: TTR_Particle_Graph = particle_graph
    self.settings_frame: tk.Frame = settings_frame
    self.ax: plt.Axes = ax
    self.canvas: FigureCanvasTkAgg = canvas

    self.max_pick_range: float = max_pick_range
    # save color config
    self.color_config: dict[str, str] = color_config
    self.edge_color_list: List[str] = edge_color_list # list of all edge colors
    # save grid padding settings
    self.grid_pad_x: int = grid_padding[0]
    self.grid_pad_y: int = grid_padding[1]
    # Initialize methods for configuring tkinter widgets
    self.add_frame_style: Callable = tk_config_methods["add_frame_style"]
    self.add_label_style: Callable = tk_config_methods["add_label_style"]
    self.add_button_style: Callable = tk_config_methods["add_button_style"]
    self.add_entry_style: Callable = tk_config_methods["add_entry_style"]
    self.add_checkbutton_style: Callable = tk_config_methods["add_checkbutton_style"]
    self.add_radiobutton_style: Callable = tk_config_methods["add_radiobutton_style"]

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
    self.pick_event_cid: int = None
    self.clear_selection()

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
    if particle is None or particle == self.selected_particle:
      self.clear_selection()
      self.canvas.draw_idle()
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
      self.highlighted_particles: List[Graph_Particle] = [particle]
    elif add_to_selection:
      # clear settings frame
      for widget in self.settings_frame.winfo_children():
        widget.destroy()
      self.highlighted_particles.append(particle)
    else:
      self.clear_selection() # remove highlights from previous selection
      self.highlighted_particles = [particle]
    # select particle and highlight it
    self.selected_particle = particle
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
    # clear settings frame
    for widget in self.settings_frame.winfo_children():
      widget.destroy()


  def add_position_setting(self, position: np.ndarray, row_index: int) -> Tuple[tk.DoubleVar, tk.DoubleVar]:
    """
    Add a position setting to the settings panel.

    Args:
        position (np.ndarray): The position to display.

    Returns:
        (tk.DoubleVar): tk variable for x coordinate of position
        (tk.DoubleVar): tk variable for y coordinate of position
    """
    position_x_var = tk.DoubleVar(value=position[0])
    position_y_var = tk.DoubleVar(value=position[1])

    position_label = tk.Label(self.settings_frame, text="Position")
    self.add_label_style(position_label)
    position_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    # frame for position inputs
    position_inputs_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(position_inputs_frame)
    position_inputs_frame.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=self.grid_pad_y)
    # x position input
    position_x_label = tk.Label(position_inputs_frame, text="x:")
    self.add_label_style(position_x_label)
    position_x_label.grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=0)
    position_x_input = tk.Entry(position_inputs_frame, textvariable=position_x_var, width=5)
    self.add_entry_style(position_x_input)
    position_x_input.grid(
        row=0,
        column=1,
        sticky="w",
        padx=(0, 2*self.grid_pad_x),
        pady=0)
    # y position input
    position_y_label = tk.Label(position_inputs_frame, text="y:")
    self.add_label_style(position_y_label)
    position_y_label.grid(
        row=0,
        column=2,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=0)
    position_y_input = tk.Entry(position_inputs_frame, textvariable=position_y_var, width=5)
    self.add_entry_style(position_y_input)
    position_y_input.grid(
        row=0,
        column=3,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=0)

    return [position_x_var, position_y_var]

  def add_rotation_setting(self, rotation: float, row_index: int) -> tk.DoubleVar:
    """
    Add a rotation setting to the settings panel in the specified row.

    Args:
        rotation (float): The rotation to display.

    Returns:
        (tk.DoubleVar): tk variable for particle rotation in degrees.
    """
    rotation_var_deg = tk.DoubleVar(value=np.rad2deg(rotation))

    rotation_label = tk.Label(self.settings_frame, text="Rotation")
    self.add_label_style(rotation_label)
    rotation_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    # frame for rotation input
    rotation_input_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(rotation_input_frame)
    rotation_input_frame.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=self.grid_pad_y)
    # rotation input
    rotation_input = tk.Entry(rotation_input_frame, textvariable=rotation_var_deg, width=5)
    self.add_entry_style(rotation_input)
    rotation_input.grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=0)
    # rotation unit label
    rotation_unit_label = tk.Label(rotation_input_frame, text="°")
    self.add_label_style(rotation_unit_label)
    rotation_unit_label.grid(
        row=0,
        column=1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=0)

    return rotation_var_deg


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

  def show_edge_settings(self, particle_edge: Particle_Edge) -> None:
    """
    Display the settings of an edge.

    Args:
        particle_edge (Particle_Edge): The edge to display the settings for.
    """
    edge_settings = particle_edge.get_adjustable_settings()
    row_index: int = 0
    # headline
    edge_headline_label = tk.Label(self.settings_frame, text=f"Settings for Edge {particle_edge.get_id()}", justify="center", anchor="n")
    self.add_label_style(edge_headline_label, headline_level=5, font_type="bold")
    edge_headline_label.grid(
      row=row_index,
      column=0,
      sticky="new",
      columnspan=2,
      padx=self.grid_pad_x,
      pady=self.grid_pad_y
    )
    row_index += 1
    # particle position
    posisition_x_var, position_y_var = self.add_position_setting(edge_settings["position"], row_index)
    row_index += 1
    # particle rotation
    rotation_var_deg = self.add_rotation_setting(edge_settings["rotation"], row_index)
    row_index += 1
    # edge color
    edge_color_var = self.add_edge_color_setting(edge_settings["color"], row_index)
    row_index += 1
    # edge image file path

    # edge button frame
    edge_buttons_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(edge_buttons_frame)
    edge_buttons_frame.grid(
      row=row_index,
      column=0,
      sticky="new",
      columnspan=2,
      padx=self.grid_pad_x,
      pady=self.grid_pad_y
    )
    edge_buttons_frame.grid_columnconfigure(0, weight=1)
    edge_buttons_frame.grid_columnconfigure(1, weight=1)
    # apply settings button
    apply_edge_settings_button = tk.Button(
        edge_buttons_frame,
        text="Apply",
        command=lambda: self.apply_edge_settings(
            particle_edge,
            posisition_x_var,
            position_y_var,
            rotation_var_deg,
            edge_color_var))
    self.add_button_style(apply_edge_settings_button)
    apply_edge_settings_button.grid(
        row=0,
        column=0,
        sticky="new",
        padx=(0, self.grid_pad_x),
        pady=0,
    )
    # edge delete button
    delete_edge_button = tk.Button(
        edge_buttons_frame,
        text="Delete",
        command=lambda: self.delete_edge(particle_edge))
    self.add_button_style(delete_edge_button)
    delete_edge_button.config(
        bg=self.color_config["delete_button_bg_color"],
        fg=self.color_config["delete_button_fg_color"])
    delete_edge_button.grid(
        row=0,
        column=1,
        sticky="new",
        padx=0,
        pady=0,
    )

  def apply_edge_settings(self,
      particle_edge: Particle_Edge,
      posisition_x_var: tk.DoubleVar,
      position_y_var: tk.DoubleVar,
      rotation_var_deg: tk.DoubleVar,
      edge_color_var: tk.IntVar):
    """
    Apply the settings of an edge.

    Args:
        particle_edge (Particle_Edge): The edge to apply the settings to.
        posisition_x_var (tk.DoubleVar): The x position variable.
        position_y_var (tk.DoubleVar): The y position variable.
        rotation_var_deg (tk.DoubleVar): The rotation variable in degrees.
        edge_color_var (tk.IntVar): The color variable.
    """
    new_position = np.array([posisition_x_var.get(), position_y_var.get()])
    new_rotation = np.deg2rad(rotation_var_deg.get())
    new_color = self.edge_color_list[edge_color_var.get()]
    particle_edge.set_adjustable_settings(
        self.ax,
        position=new_position,
        rotation=new_rotation)
    for connected_edge in get_connected_edges(particle_edge)[0]:
      connected_edge.set_adjustable_settings(
        self.ax,
        color=new_color)
    self.canvas.draw_idle()

  def delete_edge(self, particle_edge: Particle_Edge, reposition_edges: bool = False) -> None:
    """
    Delete an edge. If there are other edges connected to the giben one, the `reposition_edges` parameter decides whether they are automatically moved to fill the newly empty space.
    This is  done by calculating the vector(s) from the 

    Args:
        particle_edge (Particle_Edge): The edge to delete.
        reposition_edges (bool): Whether to automatically recalculae positions and rotations of remaining particles along the same edge.
    """
    connected_edges, edge_length = get_connected_edges(particle_edge)
    if len(connected_edges) == 1: # remove connection between the two nodes entirely
      self.remove_connection(edge_particles=connected_edges)
    particle_edge.erase()
    connected_types = [type(particle) for particle in particle_edge.connected_particles]
    # if Particle_Node in connected_types:
    #   particle_edge.connected_particles[0].remove_edge(particle_edge)

  def remove_connection(self, particle_edge: Particle_Edge, edge_particles: List[Particle_Edge] = None):
    if edge_particles is None:
      edge_particles, *_ = get_connected_edges(particle_edge)

    for end_edge_particle in (edge_particles[0], edge_particles[-1]):
      for connected_node in particle_edge.connected_particles:
        connected_node.connected_particles.remove(particle_edge) # remove connection to edge
    for edge_particle in edge_particles:
      edge_particle.erase()
      del edge_particle
    # TODO: remove connection from particle graph


  def add_edge_color_setting(self, color: str, row_index: int) -> tk.IntVar:
    """
    Add a color settings to the settings frame in the specified row.

    Args:
        rotation (float): The rotation to display.
        row_index (int): row index where to show the widgets.

    Returns:
        (tk.IntVar): tk variable for the index of the particle's color in `self.edge_color_list`
    """
    edge_color_index_var = tk.IntVar(value=self.edge_color_list.index(color))

    # color label
    edge_color_label = tk.Label(self.settings_frame, text="Edge color")
    self.add_label_style(edge_color_label)
    edge_color_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y
    )
    # color selector frame
    color_selector_frame = tk.Frame(self.settings_frame, cursor="hand2")
    self.add_frame_style(color_selector_frame)
    color_selector_frame.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=self.grid_pad_y
    )
    # color selector
    color_display_border = tk.Frame(color_selector_frame, bg=self.color_config["edge_border_color"])
    color_display_border.grid(
      row=0,
      column=1,
      sticky="w",
      padx=0,
      pady=0,
    )
    color_display_label = tk.Label(
        color_display_border,
        width=5,
        height=1,
        cursor="hand2")
    self.add_label_style(color_display_label)
    color_display_label.config(bg=self.edge_color_list[edge_color_index_var.get()])
    color_display_label.grid(
      row=0,
      column=0,
      sticky="w",
      padx=self.grid_pad_x/2,
      pady=self.grid_pad_y/2,
    )
    # add bindings to change color of the label (mousewheel and buttons)
    color_display_border.bind(
        "<MouseWheel>",
        func = lambda event, tk_var=edge_color_index_var, tk_widget=color_display_label: 
            self.change_widget_color(event.delta, tk_var, tk_widget))
    color_display_border.bind(
        "<Button-1>",
        func = lambda event, tk_var=edge_color_index_var, tk_widget=color_display_label:
            self.change_widget_color(-1, tk_var, tk_widget))
    color_display_border.bind(
        "<Button-3>",
        func = lambda event, tk_var=edge_color_index_var, tk_widget=color_display_label:
            self.change_widget_color(1, tk_var, tk_widget))
    color_display_label.bind(
        "<MouseWheel>",
        func = lambda event, tk_var=edge_color_index_var, tk_widget=color_display_label: 
            self.change_widget_color(event.delta, tk_var, tk_widget))
    color_display_label.bind(
        "<Button-1>",
        func = lambda event, tk_var=edge_color_index_var, tk_widget=color_display_label:
            self.change_widget_color(-1, tk_var, tk_widget))
    color_display_label.bind(
        "<Button-3>",
        func = lambda event, tk_var=edge_color_index_var, tk_widget=color_display_label:
            self.change_widget_color(1, tk_var, tk_widget))
    # left color change arrow (same as right click/ scroll up)
    left_color_change_button = tk.Button(
        color_selector_frame,
        text="❮",
        command=lambda: self.change_widget_color(1, edge_color_index_var, color_display_label))
    self.add_button_style(left_color_change_button)
    left_color_change_button.grid(
        row=0,
        column=0,
        sticky="w",
        padx=0,
        pady=0,
    )
    # right color change arrow (same as left click/ scroll down)
    right_color_change_button = tk.Button(
        color_selector_frame,
        text="❯",
        command=lambda: self.change_widget_color(-1, edge_color_index_var, color_display_label))
    self.add_button_style(right_color_change_button)
    right_color_change_button.grid(
        row=0,
        column=2,
        sticky="w",
        padx=0,
        pady=0,
    )
    return edge_color_index_var

  def change_widget_color(self, change_dir: int, tk_index_var: tk.IntVar, tk_widget: tk.Widget):
    """
    change background color of the given widget according to the current value of the given index variable, the change direction and `self.edge_color_list`.
    First change the index variable by `change_dir` (+-1), then update the background color to the corresponding color in `self.edge_color_list`.

    Args:
        change_dir (int): direction of the color change (+-1)
        tk_index_var (tk.IntVar): color index variable
        tk_widget (tk.Widget): widget of which to change the background color
    """
    if change_dir < 0:
      tk_index_var.set((tk_index_var.get() + 1) % len(self.edge_color_list))
    elif change_dir > 0:
      tk_index_var.set((tk_index_var.get() - 1) % len(self.edge_color_list))
    else:
      return
    tk_widget.config(background=self.edge_color_list[tk_index_var.get()])


def get_connected_edges(particle_edge: Particle_Edge) -> Tuple[List[Particle_Edge], int]:
  """
  get all edge particles connected to the given edge particle (directly or indirectly through other edges).
  Returned edges are sorted such that the first edge is connected to `particle_edge.location_1_name` and the last edge is connected to `particle_edge.location_2_name`.

  Args:
      particle_edge (Particle_Edge): edge particle to get the connected edges of

  Returns:
      (List[Particle_Edge]): list of connected edges for the given edge particle
      (int): length of the connection the given edge belongs to

  Raises:
      TypeError: if the given edge is connected (directly or indirectly through other edges) to a particle other than nodes or edges
  """
  visited_edges: set[int] = {particle_edge.get_id()}
  connected_edges: List[Particle_Edge] = [particle_edge]
  connection_length = 1
  connected_nodes: List[Particle_Node] = []
  i = 0
  while True: # add edges to the end of the list
    connected_particle = connected_edges[-1].connected_particles[i]
    if isinstance(connected_particle, Particle_Node):
      # end of edge in this direction
      connected_nodes.append(connected_particle)
      break
    elif isinstance(connected_particle, Particle_Edge):
      if connected_particle.get_id() in visited_edges:
        i += 1 # edge was already visited
        if i > 1:
          print(f"Warning: Encountered unexpected state in `get_connected_edges()` starting from edge {particle_edge.get_id()}.")
        continue
      visited_edges.add(connected_particle.get_id())
      connected_edges.append(connected_particle)
      connection_length += 1
      i = 0
    else:
      raise TypeError(f"Unexpected type of connected particle: {type(connected_particle)}")
  i = 1
  while True: # add edges to the beginning of the list
    connected_particle = connected_edges[0].connected_particles[i]
    if isinstance(connected_particle, Particle_Node):
      # end of edge in this direction
      connected_nodes.insert(0, connected_particle)
      break
    elif isinstance(connected_particle, Particle_Edge):
      if connected_particle.get_id() in visited_edges:
        i -= 1
        if i < 0:
          print(f"Warning: Encountered unexpected state in `get_connected_edges()` starting from edge {particle_edge.get_id()}.")
        continue
      visited_edges.add(connected_particle.get_id())
      connected_edges.insert(0, connected_particle)
      connection_length += 1
      i = 1
    else:
      raise TypeError(f"Unexpected type of connected particle: {type(connected_particle)}")

  if connected_nodes[0].label != particle_edge.location_1_name: # reverse list if necessary
    connected_edges.reverse()

  return connected_edges, connection_length