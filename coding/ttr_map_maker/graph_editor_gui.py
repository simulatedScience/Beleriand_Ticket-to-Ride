"""
This module implements a class to handle editing TTR particle graphs through a GUI.

The class is called Graph_Editor_GUI and can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.

This requires a tkinter frame where particle settings are displayed and edited as well as a matplotlib Axes object where the particle graph is displayed.
"""
import tkinter as tk
from typing import Tuple, List, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import PickEvent, MouseEvent
from matplotlib.patches import Circle

from file_browsing import browse_image_file
from drag_handler import Drag_Handler, find_particle_in_list, get_artist_center
from ttr_math import rotate_point_around_point
from ttr_particle_graph import TTR_Particle_Graph
from graph_particle import Graph_Particle
from particle_node import Particle_Node
from particle_label import Particle_Label
from particle_edge import Particle_Edge

class Graph_Editor_GUI:
  def __init__(self,
      master: tk.Tk,
      color_config: dict[str, str],
      grid_padding: Tuple[int, int],
      tk_config_methods: dict[str, Callable],
      movability_tk_variables: List[tk.BooleanVar],
      particle_graph: TTR_Particle_Graph,
      graph_edit_frame: tk.Frame,
      ax: plt.Axes,
      canvas: FigureCanvasTkAgg,
      max_pick_range: float = 0.2,
      edge_color_list: List[str] = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]
      ):
    """
    Initialize the graph editor GUI. This class handles the editing of particle graphs through a GUI and shows all controls in the given frame `graph_edit_frame`. Changes are made to the given particle graph `particle_graph` and the graph is displayed in the given matplotlib Axes object `ax` and tkinter canvas `canvas`.

    Args:
        master (tk.Tk): The tkinter master widget.
        color_config (dict[str, str]): Dictionary of colors for the UI (see `Board_Layout_GUI`)
        tk_config_methods (dict[str, Callable]): Dictionary of methods to configure tkinter widgets (see `Board_Layout_GUI`). These should add styles to the widgets. Expect the following keys:
            - `add_frame_style(frame: tk.Frame)`
            - `add_label_style(label: tk.Label, headline_level: int, font_type: str)
            - `add_button_style(button: tk.Button)`
            - `add_entry_style(entry: tk.Entry, justify: str)`
            - `add_checkbutton_style(checkbutton: tk.Checkbutton)
            - `add_radiobutton_style(radiobutton: tk.Radiobutton)
            - `add_browse_button(frame: tk.Frame, row_index: int, column_index: int, command: Callable) -> tk.Button`
        movability_tk_variables (List[tk.BooleanVar]): List of tkinter boolean variables that control the movability of the particles.
        particle_graph (TTR_Particle_Graph): The particle graph to edit.
        graph_edit_frame (tk.Frame): The tkinter frame where the edit mode widgets are displayed.
        ax (plt.Axes):The matplotlib Axes object where the particle graph is displayed.
        canvas (FigureCanvasTkAgg): The tkinter canvas where the mpl figure is shown
        max_pick_range (float, optional): The maximum distance from a particle to the click event to still consider it as a click on the particle. Defaults to 0.2.
    """
    self.master: tk.Tk = master
    self.particle_graph: TTR_Particle_Graph = particle_graph
    self.graph_edit_frame: tk.Frame = graph_edit_frame
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
    self.add_browse_button: Callable = tk_config_methods["add_browse_button"]

    self.move_nodes_enabled = movability_tk_variables[0]
    self.move_labels_enabled = movability_tk_variables[1]
    self.move_edges_enabled = movability_tk_variables[2]

    self.highlighted_particles: List[Graph_Particle] = [] # particles that are highlighted
    self.selected_particle: Graph_Particle = None # particle for which settings are displayed
    self.preselected_particle: Graph_Particle = None # intermediate variable to store selected particle between pick and release events

    self.pick_event_cid: int = None # connection id for pick event
    self.release_event_cid: int = None # connection id for release event

    self.add_node_mode: bool = False # flag to indicate if a node is being added
    self.add_edge_mode: bool = False # flag to indicate if an edge is being added

    self.init_graph_edit_mode()


  def init_graph_edit_mode(self):
    """
    Create the static edit buttons for the graph edit mode and bind a mouse pick event to the canvas.
    """
    row_index = 0
    # create frame for graph edit buttons
    graph_edit_buttons_frame: tk.Frame = tk.Frame(self.graph_edit_frame)
    self.add_frame_style(graph_edit_buttons_frame)
    graph_edit_buttons_frame.grid(
        row=row_index,
        column=0,
        sticky="new",
        pady=(0, self.grid_pad_y))
    row_index += 1
    graph_edit_headline: tk.Label = tk.Label(
        graph_edit_buttons_frame,
        text="Graph editing controls",
        justify="left",
        anchor="w",)
    self.add_label_style(graph_edit_headline, font_type="bold")
    graph_edit_headline.grid(
        row=0,
        column=0,
        columnspan=4,
        sticky="new",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    self.create_static_edit_buttons(graph_edit_buttons_frame)
    # create frame to show particle settings of selected particle
    self.settings_frame: tk.Frame = tk.Frame(self.graph_edit_frame)
    self.add_frame_style(self.settings_frame)
    self.settings_frame.grid(
        row=row_index,
        column=0,
        sticky="new",
        pady=(0, self.grid_pad_y))
    self.settings_frame.columnconfigure(0, weight=1)
    self.settings_frame.columnconfigure(1, weight=1)
    row_index += 1
    # init drag handler
    self.drag_handler = Drag_Handler(self.canvas, self.ax, self.particle_graph.get_particle_list())
    self.bind_mouse_events()

  def create_static_edit_buttons(self, button_frame: tk.Frame) -> None:
    """
    Create static buttons for graph editing.
    Add buttons to:
    - add a node
    - add an edge
    Add toggles to enable moving
    - nodes
    - edges
    - labels

    Args:
        button_frame (tk.Frame): Frame to add the buttons to.
    """
    def add_particle_move_toggle(
        particle_type: str,
        row_index: int,
        column_index: int,
        toggle_var: tk.BooleanVar) -> None:
      """
      Add toggle to enable moving of a particle type.

      Args:
          particle_type (str): Type of particle to move.
          row_index (int): Row index to add the toggle to.
          column_index (int): Column index to add the toggle to.
          toggle_var (tk.BooleanVar): Variable to store the toggle state.
          toggle_command (Callable): Command to execute when the toggle is changed.
      """
      particle_move_toggle: tk.Checkbutton = tk.Checkbutton(
          button_frame,
          text=particle_type,
          variable=toggle_var,
          command=self.toggle_move_particle_type)
      self.add_checkbutton_style(particle_move_toggle)
      particle_move_toggle.grid(
          row=row_index,
          column=column_index,
          sticky="new",
          padx=(self.grid_pad_x, 0),
          pady=(0, self.grid_pad_y))

    for i in range(1,4):
      button_frame.columnconfigure(i, weight=1)

    row_index = 1
    add_particle_frame: tk.Frame = tk.Frame(button_frame)
    self.add_frame_style(add_particle_frame)
    add_particle_frame.grid(
        row=row_index,
        column=0,
        sticky="nsew",
        columnspan=4,)
    add_particle_frame.columnconfigure(0, weight=1)
    add_particle_frame.columnconfigure(1, weight=1)
    add_node_button: tk.Button = tk.Button(
        add_particle_frame,
        text="Add node",
        command=self.start_node_adding_mode)
    self.add_button_style(add_node_button)
    add_node_button.grid(
        row=0,
        column=0,
        sticky="new",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    row_index += 1
    add_edge_button: tk.Button = tk.Button(
        add_particle_frame,
        text="Add edge",
        command=self.start_edge_adding_mode)
    self.add_button_style(add_edge_button)
    add_edge_button.grid(
        row=0,
        column=1,
        sticky="new",
        padx=(0, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    # bind CTRL-E to add edge
    self.master.bind("<Control-e>", lambda event: self.start_edge_adding_mode())

    column_index = 0
    move_particle_label: tk.Label = tk.Label(button_frame, text="Move")
    self.add_label_style(move_particle_label)
    move_particle_label.grid(
        row=row_index,
        column=column_index,
        sticky="nw",
        padx=(self.grid_pad_x, 0),
        pady=(0, self.grid_pad_y))
    column_index += 1
    add_particle_move_toggle("nodes", row_index, column_index, self.move_nodes_enabled)
    column_index += 1
    add_particle_move_toggle("edges", row_index, column_index, self.move_edges_enabled)
    column_index += 1
    add_particle_move_toggle("labels", row_index, column_index, self.move_labels_enabled)
    column_index += 1

  def start_node_adding_mode(self) -> None:
    """
    Add a node to the graph. This initiates a two-step process:
    1. A new node is created and attached to the mouse cursor. When the cursor moves, the node moves with it.
    2. Upon clicking on the plot, the node is placed there and the user is prompted to enter a location name.
    3. (Automatic) Once the location name is entered, the node and corresponding label are added to the graph.
    """
    self.add_node_mode: bool = True
    # self.unbind_mouse_events()
    # disable all movability toggles
    for movability_var in [self.move_nodes_enabled, self.move_edges_enabled, self.move_labels_enabled]:
      movability_var.set(False)
    self.toggle_move_particle_type() # apply movability changes
    self.clear_selection()
    print(f"Node adding mode started: {self.add_node_mode}")
    # step 2: add click listener to canvas
    self.click_listener_id: int = self.canvas.mpl_connect("button_press_event", self.add_node_on_click)
    
  def add_node_on_click(self, event: MouseEvent) -> None:
    """
    When the user clicks on the canvas while in node adding mode, a node is placed at the clicked location and the user is prompted to enter a location name.

    Args:
        event (MouseEvent): Mouse event containing the clicked location.
    """
    self.canvas.mpl_disconnect(self.click_listener_id)
    del self.click_listener_id
    location_indicator: Circle = self.ax.add_patch(
      Circle(
        xy=(event.xdata, event.ydata),
        radius=0.5,
        color=self.color_config["node_color"],
      )
    )
    self.canvas.draw_idle()

    node_name_var: tk.StringVar = tk.StringVar()
    
    def finish_node_adding():
      """
      Finish adding a node to the graph. This creates a new node at the given location and with the given name.
      Hide the given lodation_indicator and destroy the given node_name_var. Then draw the new node and add it to the graph.
      
      Args:
          location_indicator (Circle): Indicator for the location of the new node.
          node_name_var (tk.StringVar): Variable containing the name of the new node.
      """
      # hide location indicator
      location_indicator.remove()
      
      # create node
      new_node: Particle_Node = Particle_Node(
        location_name=node_name_var.get(),
        id=self.particle_graph.max_particle_id + 1,
        color=self.color_config["node_color"],
        position=np.array([event.xdata, event.ydata]),
      )
      new_node.draw(ax=self.ax, color=self.color_config["node_color"], movable=False)
      self.particle_graph.add_particle(new_node) # also increments max_particle_id

      new_label: Particle_Label = Particle_Label(
        label=node_name_var.get(),
        id=self.particle_graph.max_particle_id + 1,
        color=self.color_config["label_text_color"],
        position=np.array([event.xdata, event.ydata]) + np.array([0, 2]),
        height_scale_factor=self.particle_graph.label_height_scale,
      )
      new_label.add_connected_particle(new_node)
      new_label.draw(
          ax=self.ax,
          color=self.color_config["label_text_color"],
          border_color=self.color_config["label_outline_color"],
          movable=False)
      self.particle_graph.add_particle(new_label)
      self.canvas.draw_idle()
      self.add_node_mode: bool = False
      
      # print(f"Node added: {new_node}")
      # print(f"Graph now has nodes: {list(self.particle_graph.particle_nodes.keys())}")
      return
    
    # add prompt for location name
    self.prompt_location_name(node_name_var=node_name_var, confirm_callback=finish_node_adding)
    
  def prompt_location_name(self, node_name_var: tk.StringVar, confirm_callback: Callable):
    """
    Prompt the user to enter a location name for a new node.
    When the user confirms their input, the given confirm_callback is called.
    The input is stored in `node_name_var`.
    
    Args:
        node_name_var (tk.StringVar): Variable to store the user input in.
        confirm_callback (Callable): Function to call when the user confirms their input.
    """
    # create a frame floating above the canvas
    location_prompt_frame: tk.Frame = tk.Frame(self.master)
    self.add_frame_style(location_prompt_frame)
    location_prompt_frame.place(relx=0.5, rely=0.5, anchor="center")
    # add input field
    location_label: tk.Label = tk.Label(location_prompt_frame, text="new location name:")
    self.add_label_style(location_label)
    location_label.grid(
      row=0,
      column=0,
      sticky="nw",
      padx=self.grid_pad_x,
      pady=self.grid_pad_y)
    location_input: tk.Entry = tk.Entry(location_prompt_frame, textvariable=node_name_var)
    self.add_entry_style(location_input)
    location_input.grid(
      row=0,
      column=1,
      sticky="nw",
      padx=(0, self.grid_pad_x),
      pady=self.grid_pad_y)
    location_input.focus_set()

    def validate_input():
      """
      Check if the input is valid. If it is, call the confirm_callback.
      Valid inputs are nonempty strings that are not already used as location names.
      """
      valid_input: bool = True
      if node_name_var.get() == "":
        valid_input = False
      elif node_name_var.get() in self.particle_graph.particle_nodes.keys():
        valid_input = False
      if valid_input:
        # destroy prompt frame and call confirm callback
        location_prompt_frame.destroy()
        print(f"Valid location name confirmed: {node_name_var.get()}")
        confirm_callback()
      else:
        notification_label: tk.Label = tk.Label(location_prompt_frame, text="Input a unique, nonempty location name.")
        self.add_label_style(notification_label)
        notification_label.grid(
          row=2,
          column=0,
          columnspan=2,
          sticky="n",
          padx=self.grid_pad_x,
          pady=(0, self.grid_pad_y))
          
    # add confirm button
    confirm_button: tk.Button = tk.Button(
      location_prompt_frame,
      text="confirm",
      command=validate_input)
    self.add_button_style(confirm_button)
    confirm_button.grid(
      row=1,
      column=0,
      columnspan=2,
      sticky="n",
      padx=self.grid_pad_x,
      pady=(0, self.grid_pad_y))


  def start_edge_adding_mode(self) -> None:
    """
    Add an edge to the graph. This initiates a two-step process:
    1. Enter node selection mode. The user can select two nodes by clicking on them. Clicking on a node that is already selected deselects it.
    2. When selecting the second node, the user is prompted to enter a length for the connection.
    3, (Automatic) Once the length is entered, the edge is added to the graph with the default color.
    """
    self.add_edge_mode: bool = True
    # disable all movability toggles
    for movability_var in [self.move_nodes_enabled, self.move_edges_enabled, self.move_labels_enabled]:
      movability_var.set(False)
    self.toggle_move_particle_type() # apply movability changes
    self.clear_selection()
    print(f"Edge adding mode started: {self.add_edge_mode}")
    self.add_edge_node_indices: List[int] = [0, 0] # indices of the nodes to connect
    self.add_edge_node_labels: List[tk.Widget] = [] # labels to show the selected nodes
    self.new_edge_length: tk.IntVar = tk.IntVar(value=3, name="new_edge_length")
    self.new_edge_color_index: tk.IntVar = None
    self.node_names: List[str] = ["None"] + sorted(self.particle_graph.get_locations())

    row_index: int = 0
    # display instructions in the settings frame
    edge_adding_headline = tk.Label(self.settings_frame, text="Edge adding mode")
    self.add_label_style(edge_adding_headline, font_type="bold")
    edge_adding_headline.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    row_index += 1
    edge_adding_instructions = tk.Label(self.settings_frame, text="Click on two nodes to\nconnect them with an edge.", justify="left")
    self.add_label_style(edge_adding_instructions)
    edge_adding_instructions.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y))
    row_index += 1
    # prepare displaying selected nodes
    for i in range(2):
      selected_node_label = tk.Label(self.settings_frame, text=f"Node {i + 1}")
      self.add_label_style(selected_node_label)
      selected_node_label.grid(
          row=row_index + i,
          column=0,
          sticky="nw",
          padx=self.grid_pad_x,
          pady=(0, self.grid_pad_y))
      node_selector_frame = tk.Frame(self.settings_frame)
      self.add_frame_style(node_selector_frame)
      node_selector_frame.grid(
          row=row_index + i,
          column=1,
          sticky="w",
          padx=(0, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
      # display the name of the currently selected node
      selected_node_indicator = tk.Label(node_selector_frame, text="None", width = max([len(name) for name in self.node_names]), cursor="hand2")
      self.add_label_style(selected_node_indicator, font_type="italic")
      selected_node_indicator.grid(
          row=row_index + i,
          column=1,
          sticky="w",
          padx=0,
          pady=(0, self.grid_pad_y))
      self.add_edge_node_labels.append(selected_node_indicator)
      # add bindings to change text of the label (mousewheel and buttons)
      selected_node_indicator.bind(
          "<MouseWheel>",
          func = lambda event, location_indicator_index=i: 
              self.change_node_label(event.delta, location_indicator_index))
      selected_node_indicator.bind(
          "<Button-1>",
          func = lambda event, location_indicator_index=i:
              self.change_node_label(-1, location_indicator_index))
      selected_node_indicator.bind(
          "<Button-3>",
          func = lambda event, location_indicator_index=i:
              self.change_node_label(1, location_indicator_index))
      # add arrow buttons to change the selected node
      left_arrow_button = self.add_arrow_button("left", node_selector_frame, lambda location_indicator_index=i: self.change_node_label(1, location_indicator_index))
      left_arrow_button.grid(
          row=row_index + i,
          column=0,
          sticky="e",
          padx=0,
          pady=0)
      right_arrow_button = self.add_arrow_button("right", node_selector_frame, lambda location_indicator_index=i: self.change_node_label(-1, location_indicator_index))
      right_arrow_button.grid(
          row=row_index + i,
          column=2,
          sticky="w",
          padx=0,
          pady=0)
    row_index += 2

    # add input for the edge length
    edge_length_label = tk.Label(self.settings_frame, text="Edge length")
    self.add_label_style(edge_length_label)
    edge_length_label.grid(
        row=row_index,
        column=0,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y))
    edge_length_input_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(edge_length_input_frame)
    edge_length_input_frame.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    edge_length_input = tk.Label(edge_length_input_frame, width=5, justify="center", textvariable=self.new_edge_length, cursor="hand2")
    self.add_label_style(edge_length_input, font_type="italic")
    edge_length_input.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=0,
        pady=0)
    # add bindings to change text of the label (mousewheel and buttons)
    edge_length_input.bind(
        "<MouseWheel>",
        func = lambda event: self.change_edge_length(event.delta))
    edge_length_input.bind(
        "<Button-1>",
        func = lambda event: self.change_edge_length(1))
    edge_length_input.bind(
        "<Button-3>",
        func = lambda event: self.change_edge_length(-1))
    # add arrow buttons to change the edge length
    left_arrow_button = self.add_arrow_button("left", edge_length_input_frame, lambda: self.change_edge_length(-1))
    left_arrow_button.grid(
        row=row_index,
        column=0,
        sticky="e",
        padx=0,
        pady=0)
    right_arrow_button = self.add_arrow_button("right", edge_length_input_frame, lambda: self.change_edge_length(1))
    right_arrow_button.grid(
        row=row_index,
        column=2,
        sticky="w",
        padx=0,
        pady=0)
    row_index += 1

    # add input for the edge color
    self.new_edge_color_index: tk.IntVar = self.add_edge_color_setting(color=len(self.edge_color_list) - 1, row_index=row_index)
    row_index += 1

    # add button to add the edge or abort the process
    buttons_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(buttons_frame)
    buttons_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y))
    buttons_frame.grid_columnconfigure(0, weight=1)
    buttons_frame.grid_columnconfigure(1, weight=1)
    # add button to add the edge
    add_edge_button = tk.Button(buttons_frame, text="Add edge", command=self.add_connection)
    self.add_button_style(add_edge_button)
    add_edge_button.grid(
        row=row_index,
        column=0,
        sticky="new",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y))
    # add button to abort the process
    abort_button = tk.Button(buttons_frame, text="Abort", command=self.abort_edge_adding)
    self.add_button_style(abort_button)
    abort_button.config(
        bg=self.color_config["delete_button_bg_color"],
        fg=self.color_config["delete_button_fg_color"])
    abort_button.grid(
        row=row_index,
        column=1,
        sticky="new",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    # bind ESC key to abort the process
    self.master.bind("<Escape>", lambda event: self.abort_edge_adding())


  def change_node_label(self, change_direction: int, location_indicator_index: int) -> None:
    """
    change the label of the selected location
    First change the index variable by `change_direction` (+-1), then update the label text to the corresponding location name.

    Args:
        change_direction (int): direction of the color change (+-1)
        location_indicator_index (int): index of the location indicator to change
    """
    # remove highlight from the current node
    if change_direction != 0:
      current_node_name: str = self.node_names[self.add_edge_node_indices[location_indicator_index]]
      if current_node_name != "None":
        old_node: Particle_Node = self.particle_graph.particle_nodes[current_node_name]
        old_node.remove_highlight(self.ax)
        self.highlighted_particles.remove(old_node)

    if change_direction < 0:
      self.add_edge_node_indices[location_indicator_index] = (self.add_edge_node_indices[location_indicator_index] + 1) % len(self.node_names)
      # skip a node if it is already selected (except None)
      if self.add_edge_node_indices[location_indicator_index] == self.add_edge_node_indices[(location_indicator_index + 1) % 2] and \
          self.add_edge_node_indices[location_indicator_index] != 0: # don't skip None (index 0)
        self.add_edge_node_indices[location_indicator_index] = (self.add_edge_node_indices[location_indicator_index] + 1) % len(self.node_names)
    elif change_direction > 0:
      self.add_edge_node_indices[location_indicator_index] = (self.add_edge_node_indices[location_indicator_index] - 1) % len(self.node_names)
      # skip a node if it is already selected (except None)
      if self.add_edge_node_indices[location_indicator_index] == self.add_edge_node_indices[(location_indicator_index + 1) % 2] and \
          self.add_edge_node_indices[location_indicator_index] != 0:  # don't skip None (index 0)
        self.add_edge_node_indices[location_indicator_index] = (self.add_edge_node_indices[location_indicator_index] - 1) % len(self.node_names)
    else:
      return
    # update the label text
    self.add_edge_node_labels[location_indicator_index].config(text=self.node_names[self.add_edge_node_indices[location_indicator_index]])
    # highlight the node corresponding to the new label
    current_node_name = self.node_names[self.add_edge_node_indices[location_indicator_index]]
    if current_node_name != "None":
      new_node = self.particle_graph.particle_nodes[current_node_name]
      new_node.highlight(self.ax)
      self.highlighted_particles.append(new_node)
    self.canvas.draw_idle()

  def change_edge_length(self, change_direction: int) -> None:
    """
    Change the length of the new edge by `change_direction` (+-1).

    Args:
        change_direction (int): direction of the color change (+-1)
    """
    current_value = int(self.new_edge_length.get())
    if change_direction > 0:
      self.new_edge_length.set(str(current_value + 1))
    elif change_direction < 0 and current_value > 1:
      self.new_edge_length.set(str(current_value - 1))
    else:
      return

  def add_connection(self) -> None:
    """
    Add the new edge to the graph.
    """
    # get connection properties
    node_names = [self.node_names[i] for i in self.add_edge_node_indices]
    edge_length = int(self.new_edge_length.get())
    edge_color = self.edge_color_list[self.new_edge_color_index.get()]
    # add the edge to the graph
    self.particle_graph.add_connection(*node_names, edge_length, edge_color, add_path=True, ax=self.ax)
    self.drag_handler.update_particle_list(self.particle_graph.get_particle_list())
    # abort the edge adding process
    self.abort_edge_adding() # this also redraws the canvas

  def abort_edge_adding(self) -> None:
    """
    Abort the edge adding process:
    - unbind ESC key
    - remove the highlight from the nodes
    - hide the edge adding widgets
    - delete the edge adding variables
    - update the canvas
    """
    # unbind ESC key
    self.master.unbind("<Escape>")
    # remove the highlight from the nodes
    for particle_node in self.highlighted_particles:
      particle_node.remove_highlight(self.ax)
    # hide the edge adding widgets
    for widget in self.settings_frame.winfo_children():
      widget.destroy()
    # delete the edge adding variables
    del self.add_edge_node_indices
    del self.add_edge_node_labels
    del self.new_edge_length
    del self.new_edge_color_index
    del self.node_names
    self.add_edge_mode: bool = False
    # redraw the canvas
    self.canvas.draw_idle()


  def toggle_move_particle_type(self) -> None:
    """
    Set movability of all particle types to the current settings.
    """
    self.particle_graph.toggle_move_nodes(self.move_nodes_enabled.get())
    self.particle_graph.toggle_move_labels(self.move_labels_enabled.get())
    self.particle_graph.toggle_move_edges(self.move_edges_enabled.get())


  def bind_mouse_events(self):
    """
    Bind mouse events to the matplotlib Axes object.
    """
    self.pick_event_cid: int = self.canvas.mpl_connect("pick_event", self.on_mouse_click)

  def unbind_mouse_events(self):
    """
    Unbind mouse events from the matplotlib Axes object and clear the selection.
    """
    if self.pick_event_cid is not None:
      self.canvas.mpl_disconnect(self.pick_event_cid)
      print(f"disconnecting pick event")
    if self.release_event_cid is not None:
      self.canvas.mpl_disconnect(self.release_event_cid)
      print(f"disconnecting release event")
    self.pick_event_cid: int = None
    del self.drag_handler
    self.drag_handler: Drag_Handler = None
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
    if (particle is None or particle in self.highlighted_particles):
      if self.add_edge_mode:
        # remove highlight if node was already selected
        delete_index = self.add_edge_node_indices.index(particle.get_id() + 1)
        particle.remove_highlight(self.ax)
        self.highlighted_particles.remove(particle)
        self.add_edge_node_labels[delete_index].config(text="None")
        self.add_edge_node_indices[delete_index] = 0
      else:
      # if not self.add_edge_mode:
        self.clear_selection()
      self.canvas.draw_idle()
      return
    
    # select clicked particle
    self.preselected_particle: Graph_Particle = particle
    self.release_event_cid: int = self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
    self.canvas.mpl_disconnect(self.pick_event_cid)
    print(f"disconnecting pick event")


  def on_mouse_release(self, event: MouseEvent):
    """
    Handle mouse releases on the matplotlib Axes object.
    Select the particle that was clicked and update the canvas.
    Then rebind the mouse click event.
    """
    self.canvas.mpl_disconnect(self.release_event_cid)
    print(f"disconnecting release event")
    self.release_event_cid = None
    # if a particle was clicked, select it
    self.select_particle(self.preselected_particle, add_to_selection=self.add_edge_mode)
    self.preselected_particle = None
    self.canvas.draw_idle()
    self.pick_event_cid = self.canvas.mpl_connect("pick_event", self.on_mouse_click)


  def select_particle(self, particle: Graph_Particle, add_to_selection: bool = False, highlight_color: str = "#cc00cc"):
    """
    Select a particle to display its settings and highlight it

    Args:
        particle (Graph_Particle): The particle to select.
        add_to_selection (bool, optional): If True, the particle will be added to the current selection. Otherwise the current selection will be replaced. Defaults to False.
        highlight_color (str, optional): The color to highlight the particle with. Defaults to "#cc00cc".
    """
    print(f"{self.move_edges_enabled.get() = }")
    print(f"{self.add_edge_mode = }, {isinstance(particle, Particle_Node)=}")
    if self.add_edge_mode and (not isinstance(particle, Particle_Node)):
      print(f"2 {self.add_edge_mode = }, {isinstance(particle, Particle_Node)=}")
      return
    if len(self.highlighted_particles) == 0: # no particle was selected yet
      # select clicked particle, show it's settings and highlight it.
      self.highlighted_particles: List[Graph_Particle] = [particle]
    elif add_to_selection:
      # clear settings frame
      # for widget in self.settings_frame.winfo_children():
      #   widget.destroy()
      self.highlighted_particles.append(particle)
    else:
      self.clear_selection() # remove highlights from previous selection
      self.highlighted_particles = [particle]
    # select particle and highlight it
    self.selected_particle = particle
    particle.highlight(ax=self.ax, highlight_color=highlight_color)
    if not self.add_edge_mode: # handle clicks in edge adding mode separately
      # show settings
      if isinstance(particle, Particle_Node):
        self.show_node_settings(particle)
      elif isinstance(particle, Particle_Label):
        self.show_label_settings(particle)
      elif isinstance(particle, Particle_Edge):
        self.show_edge_settings(particle)
    elif isinstance(particle, Particle_Node): # add edge mode and particle is a node
      self.add_edge_node_selection(particle)

  def add_edge_node_selection(self, particle_node: Particle_Node):
    """
    Handle clicks in edge adding mode. Set the node indicator to the clicked node. If it was the second click, add an edge length setting.

    Args:
        particle_node (Particle_Node): The particle node that was clicked.
        highlight_color (str, optional): The color to highlight the particle with. Defaults to "#cc00cc".
    """
    # find the first empty node indicator
    if self.add_edge_node_indices[0] == 0:
      write_index = 0
    else: # no empty node indicator was found, overwrite the second node indicator
      if self.add_edge_node_indices[1] != 0: # remove highlight from second node
        old_node = self.particle_graph.particle_nodes[self.node_names[self.add_edge_node_indices[1]]]
        self.highlighted_particles.remove(old_node)
        old_node.remove_highlight(ax=self.ax)
      write_index = 1
    # set node indicator to node name
    self.add_edge_node_labels[write_index].config(text=particle_node.label)
    # add node to edge node list
    self.add_edge_node_indices[write_index] = self.node_names.index(particle_node.label)



  def clear_selection(self):
    """
    remove highlights from all highlighted particles, reset internal variables and hide particle settings
    """
    # unbind mouse events
    if self.release_event_cid is not None:
      self.canvas.mpl_disconnect(self.release_event_cid)
      print(f"disconnecting release event")
      self.release_event_cid = None
    # if self.pick_event_cid is not None:
    #   self.canvas.mpl_disconnect(self.pick_event_cid)
    #   print(f"disconnecting pick event")
    #   self.pick_event_cid = None
    # remove highlights from all highlighted particles
    for highlight_particle in self.highlighted_particles:
      highlight_particle.remove_highlight(ax=self.ax)
    self.highlighted_particles: List[Graph_Particle] = [] # particles that are highlighted
    if self.selected_particle is not None:
      # self.selected_particle.remove_highlight(ax=self.ax)
      self.selected_particle: Graph_Particle = None
    # clear settings frame
    for widget in list(self.settings_frame.winfo_children()):
      # widget.grid_forget()
      try:
        widget.destroy()
      except tk.TclError as e:
        print(f"Warning: An error occured while deleting widget {widget}:\n{e}")
    self.settings_frame.update()
    self.canvas.draw_idle()


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
    position_x_label = tk.Label(position_inputs_frame, text="x")
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
    position_y_label = tk.Label(position_inputs_frame, text="y")
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

  def add_label_setting(self, text: str, label: str, row_index: int, width=20) -> tk.StringVar:
    """
    Add a label setting to the settings panel in the specified row.

    Args:
        text (str): The text to display in front of the input.
        label (str): The label to display.
        row_index (int): The row index to display the setting in (in the `self.settings_frame`)

    Returns:
        (tk.StringVar): tk variable for particle label.
    """
    label_var = tk.StringVar(value=label)

    label_label = tk.Label(self.settings_frame, text=text)
    self.add_label_style(label_label)
    label_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    # frame for label input
    label_input_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(label_input_frame)
    label_input_frame.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=self.grid_pad_y)
    # label input
    label_input = tk.Entry(label_input_frame, textvariable=label_var, width=width)
    self.add_entry_style(label_input)
    label_input.config(justify="left")
    label_input.grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=0)

    return label_var

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
    node_label_var = self.add_label_setting("Label", node_settings["label"], row_index)
    row_index += 1
    # node image
    node_image_path_var = self.add_node_image_setting("Node image", node_settings["image_file_path"], row_index, width=15)
    row_index += 1

    # add edge buttons (apply & delete)
    apply_function = lambda: self.apply_node_settings(
            particle_node,
            posisition_x_var,
            position_y_var,
            rotation_var_deg,
            node_label_var,
            node_image_path_var)
    delete_function = lambda: self.delete_node(particle_node)
    self.add_settings_buttons(row_index, apply_function, delete_function)
    row_index += 1

  def apply_node_settings(self,
      particle_node: Particle_Node,
      posisition_x_var: tk.StringVar,
      position_y_var: tk.StringVar,
      rotation_var_deg: tk.StringVar,
      node_label_var: tk.StringVar,
      node_image_path_var: tk.StringVar) -> None:
    """
    Apply the settings of a node.

    Args:
        particle_node (Particle_Node): The node to apply the settings to.
        posisition_x_var (tk.StringVar): The x position variable.
        position_y_var (tk.StringVar): The y position variable.
        rotation_var_deg (tk.StringVar): The rotation variable.
        node_label_var (tk.StringVar): The label variable.
        node_image_path_var (tk.StringVar): The image path variable.
    """
    new_position = np.array([posisition_x_var.get(), position_y_var.get()], dtype=np.float16)
    new_rotation = np.deg2rad(rotation_var_deg.get())
    old_name: str = particle_node.label
    new_name: str = node_label_var.get()
    particle_node.set_adjustable_settings(
        self.ax,
        position=new_position,
        rotation=new_rotation,
        label=new_name,
        image_file_path=node_image_path_var.get()
    )
    # check if node has been renamed
    if old_name != new_name:
      # change node name in particle graph
      self.particle_graph.rename_node(old_name, new_name)
      # find Particle_Label corresponding to the node and change its text
      self.particle_graph.rename_label(old_name, new_name, self.ax)

    self.canvas.draw_idle()

  def delete_node(self, particle_node: Particle_Node) -> None:
    """
    Delete a node and all connected edges.

    Args:
        particle_node (Particle_Node): The node to delete.
    """
    self.clear_selection()
    self.particle_graph.delete_node(particle_node)
    # update drag handler
    self.drag_handler.update_particle_list(self.particle_graph.get_particle_list())
    self.canvas.draw_idle()

  def add_node_image_setting(self, text: str, file_path: str, row_index: int, width=10) -> tk.StringVar:
    """
    Add a label setting to the settings panel in the specified row.

    Args:
        text (str): The text to display in front of the input.
        file_path (str): The file path to display in the input.
        row_index (int): The row index to display the setting in (in the `self.settings_frame`)
        width (int, optional): The width of the input. Defaults to 10.

    Returns:
        (tk.StringVar): tk variable for particle label.
    """
    image_path_var = tk.StringVar(value=file_path)

    label_label = tk.Label(self.settings_frame, text=text)
    self.add_label_style(label_label)
    label_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    # frame for label input
    label_input_frame = tk.Frame(self.settings_frame)
    self.add_frame_style(label_input_frame)
    label_input_frame.grid(
        row=row_index,
        column=1,
        sticky="ew",
        padx=(0, self.grid_pad_x),
        pady=self.grid_pad_y)
    label_input_frame.columnconfigure(1, weight=1)
    # label input
    label_input = tk.Entry(label_input_frame, textvariable=image_path_var, width=width)
    self.add_entry_style(label_input)
    label_input.config(justify="left")
    label_input.grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=0)
    # browse button
    browse_button: tk.Button = self.add_browse_button(
        label_input_frame,
        row_index=0,
        column_index=1,
        command=lambda: browse_image_file("browse node image file", image_path_var))
    browse_button.grid(
        row=0,
        column=1,
        sticky="e",
        padx=0,
        pady=0)
    return image_path_var


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
        command=lambda: self.delete_connection(particle_edge))
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
    new_position = np.array([posisition_x_var.get(), position_y_var.get()], dtype=np.float16)
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
      self.delete_connection(edge_particles=connected_particles[1:-1])
      return
    particle_edge.erase()
    self.particle_graph.delete_edge(particle_edge)
    # update drag handler
    self.drag_handler.update_particle_list(self.particle_graph.get_particle_list())
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

  def delete_connection(self, particle_edge: Particle_Edge, edge_particles: List[Particle_Edge] = None):
    self.clear_selection()
    if edge_particles is None:
      edge_particles, *_ = get_edge_connected_particles(particle_edge)
      edge_particles = edge_particles[1:-1]

    for particle_edge in reversed(edge_particles):
      particle_edge.erase()
      self.particle_graph.delete_edge(particle_edge)
      # del edge_particle
    # update drag handler
    self.drag_handler.update_particle_list(self.particle_graph.get_particle_list())
    self.canvas.draw_idle()


  def add_edge_color_setting(self, color, row_index: int) -> tk.IntVar:
    """
    Add a color settings to the settings frame in the specified row.

    Args:
        color (str) or (int): The color to display or its index in `self.edge_color_list`.
        row_index (int): row index where to show the widgets.

    Returns:
        (tk.IntVar): tk variable for the index of the particle's color in `self.edge_color_list`
    """
    if isinstance(color, str):
      edge_color_index_var = tk.IntVar(value=self.edge_color_list.index(color))
    elif isinstance(color, int):
      edge_color_index_var = tk.IntVar(value=color)

    # color label
    edge_color_label = tk.Label(self.settings_frame, text="Edge color")
    self.add_label_style(edge_color_label)
    edge_color_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y)
    )
    # color selector frame
    color_selector_frame = tk.Frame(self.settings_frame, cursor="hand2")
    self.add_frame_style(color_selector_frame)
    color_selector_frame.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y)
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
    left_color_change_button = self.add_arrow_button(
        "left",
        color_selector_frame,
        lambda: self.change_widget_color(1, edge_color_index_var, color_display_label)
    )
    left_color_change_button.grid(
        row=0,
        column=0,
        sticky="w",
        padx=0,
        pady=0,
    )
    # right color change arrow (same as left click/ scroll down)
    right_color_change_button = self.add_arrow_button(
        "right",
        color_selector_frame,
        lambda: self.change_widget_color(-1, edge_color_index_var, color_display_label)
    )
    right_color_change_button.grid(
        row=0,
        column=2,
        sticky="w",
        padx=0,
        pady=0,
    )
    return edge_color_index_var
  
  def add_arrow_button(self, direction: str, parent_frame: tk.Frame, command: Callable) -> tk.Button:
    """
    add a button displaying an arrow in the given direction to the given parent frame and bind the command to it.
    This automatically applies the button style to the button.

    Args:
        direction (str): direction of the arrow (left, right, up, down)
        parent_frame (tk.Frame): frame to add the button to
        command (Callable): function to bind to the button

    Returns:
        (tk.Button): the created button to be placed with a geometry manager

    Raises:
        ValueError: if the given direction is not one of 'left', 'right', 'up', 'down'
    """
    if direction == "left":
      button_text = "❮"
    elif direction == "right":
      button_text = "❯"
    elif direction == "up":
      button_text = "︿"
    elif direction == "down":
      button_text = "﹀"
    else:
      raise ValueError(f"Invalid direction: {direction}. Must be one of 'left', 'right', 'up', 'down'.")
    button = tk.Button(
        parent_frame,
        text=button_text,
        command=command)
    self.add_button_style(button)
    return button


  def change_widget_color(self, change_direction: int, tk_index_var: tk.IntVar, tk_widget: tk.Widget):
    """
    change background color of the given widget according to the current value of the given index variable, the change direction and `self.edge_color_list`.
    First change the index variable by `change_direction` (+-1), then update the background color to the corresponding color in `self.edge_color_list`.

    Args:
        change_direction (int): direction of the color change (+-1)
        tk_index_var (tk.IntVar): color index variable
        tk_widget (tk.Widget): widget of which to change the background color
    """
    if change_direction < 0:
      tk_index_var.set((tk_index_var.get() + 1) % len(self.edge_color_list))
    elif change_direction > 0:
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
  intersection_point_1: np.ndarray = np.array([x0 + rx, y0 + ry], dtype=np.float16)
  intersection_point_2: np.ndarray = np.array([x0 - rx, y0 - ry], dtype=np.float16)
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