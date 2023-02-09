"""
This module implements a class to handle editing TTR particle graphs through a GUI.

The class is called ParticleGraphEditor and can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.

This requires a tkinter frame where particle settings are displayed and edited as well as a matplotlib Axes object where the particle graph is displayed.
"""
import tkinter as tk
from typing import Tuple, List, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import PickEvent, MouseEvent

from ttr_particle_graph import TTR_Particle_Graph
from graph_particle import Graph_Particle
from particle_node import Particle_Node
from particle_label import Particle_Label
from particle_edge import Particle_Edge
from drag_handler import find_particle_in_list, get_artist_center
from ttr_math import rotate_point_around_point

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

  def add_settings_buttons(self, row_index: int, apply_function: Callable, delete_function: Callable) -> None:
    """
    Add the apply and delete buttons to the settings panel in the specified row.

    Args:
        row_index (int): The row index to add the buttons to.
        apply_function (Callable): The function to call when the apply button is pressed.
        delete_function (Callable): The function to call when the delete button is pressed.
    """
    
    # button frame
    buttons_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(buttons_frame)
    buttons_frame.grid(
      row=row_index,
      column=0,
      sticky="new",
      columnspan=2,
      padx=self.grid_pad_x,
      pady=self.grid_pad_y
    )
    buttons_frame.grid_columnconfigure(0, weight=1)
    buttons_frame.grid_columnconfigure(1, weight=1)
    # apply settings button
    apply_settings_button = tk.Button(
        buttons_frame,
        text="Apply",
        command=apply_function)
    self.add_button_style(apply_settings_button)
    apply_settings_button.grid(
        row=0,
        column=0,
        sticky="new",
        padx=(0, self.grid_pad_x),
        pady=0,
    )
    # delete button
    delete_button = tk.Button(
        buttons_frame,
        text="Delete",
        command=delete_function)
    self.add_button_style(delete_button)
    delete_button.config(
        bg=self.color_config["delete_button_bg_color"],
        fg=self.color_config["delete_button_fg_color"])
    delete_button.grid(
        row=0,
        column=1,
        sticky="new",
        padx=0,
        pady=0,
    )

  def show_node_settings(self, particle_node: Particle_Node):
    """
    Display the settings of a node.
    """
    node_settings = particle_node.get_adjustable_settings()
    row_index: int = 0
    # headline
    node_headline_label = tk.Label(self.settings_frame, text=f"Settings for node {particle_node.get_id()}", justify="center", anchor="n")
    self.add_label_style(node_headline_label, headline_level=5, font_type="bold")
    node_headline_label.grid(
      row=row_index,
      column=0,
      sticky="new",
      columnspan=2,
      padx=self.grid_pad_x,
      pady=self.grid_pad_y
    )
    row_index += 1
    # particle position
    posisition_x_var, position_y_var = self.add_position_setting(node_settings["position"], row_index)
    row_index += 1
    # particle rotation
    rotation_var_deg = self.add_rotation_setting(node_settings["rotation"], row_index)
    row_index += 1
    # node label
    node_label_var = self.add_label_setting(node_settings["label"], row_index)

    # add edge buttons (apply & delete)
    apply_function = lambda: self.apply_node_settings(
            particle_node,
            posisition_x_var,
            position_y_var,
            rotation_var_deg,
            node_label_var)
    delete_function = lambda: self.delete_node(particle_node)
    self.add_settings_buttons(row_index, apply_function, delete_function)
    row_index += 1

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
    edge_headline_label = tk.Label(self.settings_frame, text=f"Settings for edge {particle_edge.get_id()}", justify="center", anchor="n")
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

    # add edge buttons (apply & delete)
    apply_function = lambda: self.apply_edge_settings(
            particle_edge,
            posisition_x_var,
            position_y_var,
            rotation_var_deg,
            edge_color_var)
    delete_function = lambda: self.delete_edge(particle_edge)
    self.add_settings_buttons(row_index, apply_function, delete_function)
    row_index += 1
    # add button to delete entire connection
    delete_connection_button = tk.Button(
        self.settings_frame,
        text="Delete Connection",
        cursor="hand2",
        command=lambda: self.remove_connection(particle_edge))
    self.add_button_style(delete_connection_button)
    delete_connection_button.config(
        bg=self.color_config["delete_button_bg_color"],
        fg=self.color_config["delete_button_fg_color"])
    delete_connection_button.grid(
        row=row_index,
        column=0,
        sticky="new",
        columnspan=2,
        padx=self.grid_pad_x,
        pady=self.grid_pad_y,
    )
    row_index += 1

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
    if new_color != particle_edge.color:
      old_color = particle_edge.color
      for connected_edge in get_edge_connected_particles(particle_edge)[0][1:-1]:
        connected_edge.set_adjustable_settings(
          self.ax,
          color=new_color)
      self.particle_graph.update_path_color(particle_edge, old_color)
    self.canvas.draw_idle()

  def delete_edge(self, particle_edge: Particle_Edge, reposition_edges: bool = True) -> None:
    """
    Delete an edge. If there are other edges connected to the giben one, the `reposition_edges` parameter decides whether they are automatically moved to fill the newly empty space.
    To do this, consider three cases:
    1. edge has length 1 -> delete connection between the two nodes entirely and remove the edge particle (see `remove_connection()`)
    2. delete an edge that's connected to a node -> Rearange the remaining edges to form a connection between the two nodes.
    3. delete an edge that's only connected to other edges -> Rearange the remaining edges such that the edges connected to the deleted particle are close to each other.

    Args:
        particle_edge (Particle_Edge): The edge to delete.
        reposition_edges (bool): Whether to automatically recalculae positions and rotations of remaining particles along the same edge.
    """
    connected_particles, edge_length = get_edge_connected_particles(particle_edge)
    self.clear_selection()
    # case 1: remove connection between the two nodes entirely
    if len(connected_particles) == 1:
      self.remove_connection(edge_particles=connected_particles[1:-1])
      return
    particle_edge.erase()
    self.particle_graph.remove_particle(particle_edge)
    # reposition edges if necessary
    if not reposition_edges:
      return
    connected_types = [type(particle) for particle in particle_edge.connected_particles]
    deletion_index = connected_particles.index(particle_edge)
    # case 2: delete an edge that's connected to a node
    if Particle_Node in connected_types:
      delete_end_edge_particle(
        deleted_edge=particle_edge,
        connected_particles=connected_particles,
        deletion_index=deletion_index,
        ax=self.ax,
        canvas=self.canvas)
    # case 3: delete an edge that's only connected to other edges
    else:
      delete_middle_edge_particle(
        deleted_edge=particle_edge,
        connected_particles=connected_particles,
        deletion_index=deletion_index,
        ax=self.ax,
        canvas=self.canvas)

  def remove_connection(self, particle_edge: Particle_Edge, edge_particles: List[Particle_Edge] = None):
    if edge_particles is None:
      edge_particles, *_ = get_edge_connected_particles(particle_edge)
      edge_particles = edge_particles[1:-1]
      print(f"deleting connection {particle_edge.location_1_name} -> {particle_edge.location_2_name}")
      print(f"deleting {len(edge_particles)} edge particles with path indices: {[particle.path_index for particle in edge_particles]}")

    # for end_edge_particle in (edge_particles[0], edge_particles[-1]):
    #   for connected_node in end_edge_particle.connected_particles:
    #     try:
    #       connected_node.connected_particles.remove(end_edge_particle) # remove connection of node to edge
    #     except ValueError:
    #       pass
    for particle_edge in reversed(edge_particles):
      particle_edge.erase()
      self.particle_graph.remove_particle(particle_edge)
      # del edge_particle
    self.canvas.draw_idle()


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



def get_edge_connected_particles(particle_edge: Particle_Edge) -> Tuple[List[Particle_Edge], int]:
  """
  get all edge particles connected to the given edge particle (directly or indirectly through other edges).
  Returned edges are sorted such that the first edge is connected to `particle_edge.location_1_name` and the last edge is connected to `particle_edge.location_2_name`.

  Args:
      particle_edge (Particle_Edge): edge particle to get the connected edges of

  Returns:
      (List[Particle_Edge]): list of connected edges and nodes for the given edge particle. The connected nodes are the first and last elements of the list. The edges are sorted in between them.
      (int): length of the connection the given edge belongs to

  Raises:
      TypeError: if the given edge is connected (directly or indirectly through other edges) to a particle other than nodes or edges
  """
  visited_edges: set[int] = {particle_edge.get_id()}
  connected_particles: List[Graph_Particle] = [particle_edge]
  connection_length = 1
  i = 0
  while True: # add edges to the end of the list
    connected_particle = connected_particles[-1].connected_particles[i]
    if isinstance(connected_particle, Particle_Node):
      # end of edge in this direction
      connected_particles.append(connected_particle)
      break
    elif isinstance(connected_particle, Particle_Edge):
      if connected_particle.get_id() in visited_edges:
        i += 1 # edge was already visited
        if i > 1:
          print(f"Warning: Encountered unexpected state in `get_connected_edges()` starting from edge {particle_edge.get_id()}.")
        continue
      visited_edges.add(connected_particle.get_id())
      connected_particles.append(connected_particle)
      connection_length += 1
      i = 0
    else:
      raise TypeError(f"Unexpected type of connected particle: {type(connected_particle)}")
  i = 1
  while True: # add edges to the beginning of the list
    connected_particle = connected_particles[0].connected_particles[i]
    if isinstance(connected_particle, Particle_Node):
      # end of edge in this direction
      connected_particles.insert(0, connected_particle)
      break
    elif isinstance(connected_particle, Particle_Edge):
      if connected_particle.get_id() in visited_edges:
        i -= 1
        if i < 0:
          print(f"Warning: Encountered unexpected state in `get_connected_edges()` starting from edge {particle_edge.get_id()}.")
        continue
      visited_edges.add(connected_particle.get_id())
      connected_particles.insert(0, connected_particle)
      connection_length += 1
      i = 1
    else:
      raise TypeError(f"Unexpected type of connected particle: {type(connected_particle)}")

  if connected_particles[0].label != particle_edge.location_1_name: # reverse list if necessary
    connected_particles.reverse()

  return connected_particles, connection_length


def get_connection_endpoints(edge_list: List[Graph_Particle], added_gap_at_end: float = 0) -> Tuple[np.ndarray, np.ndarray]:
  """
  Calculate the endpoints of the given edges. The first and last particle in the list should not be the ones just outside the connection (i.e. a connected node and the deleted edge).
  The endpoints are the points furthest away from other edges in the given list, but are always in the center of one of the short sides of an edge particle.
  If `added_gap_at_end > 0`, the second output is moved by `added_gap_at_end` in the direction of the connection between the two outputs.
  The code uses Particle_Edge.get_attraction_forces() to get the endpoints.

  Args:
      edge_list (List[Graph_Particle]): list of edge particles including the first particles to either side that are not part of the connection of which the endpoints should be calculated.
      added_gap_at_end (float, optional): added distance at the end of the connection corresponding to the last edge. Defaults to 0.

  Returns:
      Tuple[np.ndarray, np.ndarray]: the two endpoints of the connection as numpy arrays with shape (2,)
  """
  start_point = edge_list[1].get_attraction_forces(edge_list[0])[1]
  end_point = edge_list[-2].get_attraction_forces(edge_list[-1])[1]
  if added_gap_at_end > 0:
    end_point += added_gap_at_end * (end_point - start_point) / np.linalg.norm(end_point - start_point)
  return start_point, end_point


def rotate_rescale_edges(
    edge_list: List[Particle_Edge],
    target_connection: Tuple[np.ndarray, np.ndarray],
    current_connection: Tuple[np.ndarray, np.ndarray],
    rotation_center: np.ndarray,
    ax: plt.Axes,
    debug: bool = False):
  """
  Rotate remaining particles around start of new connection to match the old connection, then rescale to match the old connection length.

  Args:
      edge_list (List[Particle_Edge]): list of edge particles to be rotated and rescaled
      target_connection (Tuple[np.ndarray, np.ndarray]): start and end point of the new connection
      current_connection (Tuple[np.ndarray, np.ndarray]): start and end point of the old connection
      rotation_center (np.ndarray): point around which the edges should be rotated
      ax (plt.Axes): axes to draw the particles on
  """
  target_connection_vector = target_connection[1] - target_connection[0]
  current_connection_vector = current_connection[1] - current_connection[0]
  rotation_angle = np.arctan2(target_connection_vector[1], target_connection_vector[0]) - np.arctan2(current_connection_vector[1], current_connection_vector[0])

  if debug:
    # plot rotation center
    ax.plot(rotation_center[0], rotation_center[1], 'o', color='#ff00ff', markersize=10)
    # plot target connection
    ax.plot([target_connection[0][0], target_connection[1][0]], [target_connection[0][1], target_connection[1][1]], color='#ff0000', linewidth=5)
    # plot current connection
    ax.plot([current_connection[0][0], current_connection[1][0]], [current_connection[0][1], current_connection[1][1]], color='#ff00ff', linewidth=5)
    # show start points of connections
    ax.plot([target_connection[0][0], current_connection[0][0]], [target_connection[0][1], current_connection[0][1]], 'go')

  for particle in edge_list:
    new_position = rotate_point_around_point(particle.position, rotation_center, rotation_angle)
    # scale new position to match old connection length
    new_position = target_connection[0] + (new_position - current_connection[0]) / np.linalg.norm(current_connection_vector) * np.linalg.norm(target_connection_vector)
    particle.set_adjustable_settings(
      ax,
      position=new_position,
      rotation=particle.rotation + rotation_angle)

# deletion functions
def delete_end_edge_particle(deleted_edge: Particle_Edge, connected_particles: List[Graph_Particle], deletion_index: int, ax: plt.Axes, canvas: FigureCanvasTkAgg) -> None:
  """
  Delete the edge particle at the end of the connection and rotate the remaining particles to match the old connection.

  Args:
      deleted_edge (Particle_Edge): edge particle to be deleted
      connected_particles (List[Graph_Particle]): list of particles connected to the edge to be deleted
      deletion_index (int): index of the edge to be deleted in the list of connected particles
      ax (plt.Axes): axes to draw the particles on
  """
  # get endpoints of current connection
  target_connection = get_connection_endpoints(connected_particles, added_gap_at_end=0)
  # get endpoints of new connection (with the deleted edge removed)
  # rotate around the the node where the connected edge is not deleted
  if deletion_index == 1:
    particle_list: List[Graph_Particle] = connected_particles[1:]
    rotation_center: np.ndarray = connected_particles[-1].position
    particle_list.reverse()
  else:
    particle_list: List[Graph_Particle] = connected_particles[:-1]
    rotation_center: np.ndarray = connected_particles[0].position
  new_connection: Tuple[np.ndarray, np.ndarray] = get_connection_endpoints(particle_list, added_gap_at_end=0)
  if deletion_index == 1:
    target_connection = (target_connection[1], target_connection[0])
  # rotate and rescale the edges
  rotate_rescale_edges(
      particle_list[1:-1],
      target_connection,
      new_connection,
      rotation_center=rotation_center,
      ax=ax)
  relink_delete_edge(deleted_edge)
  canvas.draw_idle()
  return

def delete_middle_edge_particle(
    deleted_edge: Particle_Edge,
    connected_particles: List[Graph_Particle],
    deletion_index: int,
    ax: plt.Axes,
    canvas: FigureCanvasTkAgg) -> None:
  """
  Delete the edge particle in the middle of the connection and rotate the remaining particles to match the old connection.

  Args:
      deleted_edge (Particle_Edge): edge to be deleted
      connected_particles (List[Graph_Particle]): list of particles connected to the edge to be deleted
      deletion_index (int): index of the edge to be deleted in the list of connected particles
      ax (plt.Axes): axes to draw the particles on
      canvas (FigureCanvasTkAgg): canvas to draw the particles on
  """
  left_particle_list: List[Graph_Particle] = connected_particles[:deletion_index + 1]
  right_particle_list: List[Graph_Particle] = connected_particles[deletion_index:]
  # calculate the gap between the deleted edge and the left edge
  deleted_edge_anchor = deleted_edge.get_attraction_forces(connected_particles[deletion_index - 1])[1]
  left_edge_anchor = connected_particles[deletion_index - 1].get_attraction_forces(deleted_edge)[1]
  edge_gap: float = np.linalg.norm(deleted_edge_anchor - left_edge_anchor)
  # get endpoints of current connection
  left_connection = get_connection_endpoints(left_particle_list, added_gap_at_end=edge_gap)
  right_connection = get_connection_endpoints(right_particle_list, added_gap_at_end=0)
  # get endpoints of new connection (with the deleted edge removed) using the intersection point of circles with radii equal to the connection lengths
  target_point = get_circle_intersection(
      center_a=left_connection[0],
      radius_a=left_connection[1] - left_connection[0],
      center_b=right_connection[-1],
      radius_b=right_connection[1] - right_connection[0],
      )
  if target_point is None:
    # cannot find a good connection between the two edges
    print(f"Could not find a good connection between the remaining edges")
    relink_delete_edge(deleted_edge)
    canvas.draw_idle()
    return
  # rotate and rescale the edges
  left_target_connection = (left_connection[0], target_point)
  right_target_connection = (right_connection[-1], target_point)
  rotate_rescale_edges(
    left_particle_list[1:-1],
    left_target_connection,
    left_connection,
    rotation_center=left_particle_list[0].position,
    ax=ax,
    debug=False)
  rotate_rescale_edges(
    right_particle_list[1:-1],
    right_target_connection,
    (right_connection[1], right_connection[0]),
    rotation_center=right_particle_list[-1].position,
    ax=ax,
    debug=False)
  relink_delete_edge(deleted_edge)
  canvas.draw_idle()

def relink_delete_edge(particle_edge: Particle_Edge) -> None:
  """
  Update the particles connected to the given edge particle when it is deleted.

  Args:
      edge_particle (Particle_Edge): edge particle to be deleted
  """
  # get the particles connected to the edge
  particle_1, particle_2 = particle_edge.connected_particles
  # update the connected particles of the connected particles
  for i, particle in enumerate(particle_1.connected_particles):
    if particle == particle_edge:
      particle_1.connected_particles[i] = particle_2
      break
  for i, particle in enumerate(particle_2.connected_particles):
    if particle == particle_edge:
      particle_2.connected_particles[i] = particle_1
      break

# helper function for automatic edge repositioning
def get_circle_intersection(center_a: np.ndarray, radius_a: np.ndarray, center_b: np.ndarray, radius_b: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
  """
  Calculate the intersection of two circles given by their centers and radii. Radii are given as vectors pointing from the center to the edge of the circle. The returned intersection point is the one closest to the first center plus the first radius vector.
  If the circles do not intersect, `None` is returned.

  Args:
      center_a (np.ndarray): center of the first circle
      radius_a (np.ndarray): radius vector of the first circle
      center_b (np.ndarray): center of the second circle
      radius_b (np.ndarray): radius vector of the second circle

  Returns:
      (np.ndarray): intersection point of the two circles
  """
  # calculate the distance between the two centers
  center_distance: float = np.linalg.norm(center_b - center_a)
  # check if the circles are too far away to intersect
  if center_distance > np.linalg.norm(radius_a) + np.linalg.norm(radius_b):
      return None
  # calculate the distance along the line between the two centers to the closest point to center_a
  a: float = (np.dot(radius_a, radius_a) - np.dot(radius_b, radius_b) + center_distance ** 2) / (2 * center_distance)
  # calculate the perpendicular distance from the closest point to the line between the intersection points
  h: float = np.sqrt(np.dot(radius_a, radius_a) - a ** 2)
  # handle case where only one intersection point exists due to numerical instabilities
  if h < epsilon:
    return center_a + a * (center_b - center_a) / center_distance
  # calculate both intersection points by 
  x0: float = center_a[0] + a * (center_b[0] - center_a[0]) / center_distance
  y0: float = center_a[1] + a * (center_b[1] - center_a[1]) / center_distance
  rx: float = -(center_b[1] - center_a[1]) * (h / center_distance)
  ry: float = (center_b[0] - center_a[0]) * (h / center_distance)
  intersection_point_1: np.ndarray = np.array([x0 + rx, y0 + ry])
  intersection_point_2: np.ndarray = np.array([x0 - rx, y0 - ry])
  # return the intersection point closest to the center_a plus radius_a
  if np.linalg.norm(intersection_point_1 - (center_a + radius_a)) < np.linalg.norm(intersection_point_2 - (center_a + radius_a)):
    return intersection_point_1
  else:
    return intersection_point_2


def plot_circles_and_intersection(center_a, radius_a, center_b, radius_b, intersection_point):
    # Plot the two circles and their centers
    angle = np.linspace(0, 2 * np.pi, 100)
    x_a = center_a[0] + np.linalg.norm(radius_a) * np.cos(angle)
    y_a = center_a[1] + np.linalg.norm(radius_a) * np.sin(angle)
    x_b = center_b[0] + np.linalg.norm(radius_b) * np.cos(angle)
    y_b = center_b[1] + np.linalg.norm(radius_b) * np.sin(angle)
    plt.plot(x_a, y_a, 'r')
    plt.plot(x_b, y_b, 'b')
    plt.plot(center_a[0], center_a[1], 'ro')
    plt.plot(center_b[0], center_b[1], 'bo')

    # Plot the radius vectors and connections between center points and intersection point
    plt.plot([center_a[0], center_a[0]+radius_a[0]], [center_a[1], center_a[1]+radius_a[1]], 'r', alpha=0.5)
    plt.plot([center_b[0], center_b[0]+radius_b[0]], [center_b[1], center_b[1]+radius_b[1]], 'b', alpha=0.5)
    plt.plot([center_a[0], intersection_point[0]], [center_a[1], intersection_point[1]], 'g', alpha=0.5)
    plt.plot([center_b[0], intersection_point[0]], [center_b[1], intersection_point[1]], 'g', alpha=0.5)

    # Plot the intersection point
    plt.plot(intersection_point[0], intersection_point[1], 'go', label="intersection point")

    # Show the plot
    plt.grid(color="#dddddd")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


if __name__ == "__main__":
  # center_a = np.array([1, 1])
  # radius_a = np.array([0.7, -1])
  # center_b = np.array([4, -1])
  # radius_b = np.array([-1.7, -1.8])
  center_a = np.array([1, 1])
  radius_a = np.array([0.7, -1])
  center_b = np.array([4, 2])
  radius_b = np.array([-1.2, 1.8])
  intersection_point = get_circle_intersection(center_a, radius_a, center_b, radius_b)
  plot_circles_and_intersection(center_a, radius_a, center_b, radius_b, intersection_point)