"""
This module implements a GUI for optimizing board layouts for custom Ticket to Ride boards.

The GUI is implemented using tkinter with an embedded matplotlib animation.

Upon startup, the GUI will load an example board layout for a LotR themed TTR board.

User interactions:
- load other board layouts from files
  -> file for locations (nodes)
  -> file for paths (edges)
  -> file for tasks (pairs of locations)
  -> background image (map)
- change available edge colors with hex-code inputs
- change color of each edge to one of the available colors
- change parameters of the particle simulation
  -> change size of time step per iteration using a slider
  -> change the number of iterations per frame using the slider
  -> change velocity decay multiplier
  -> change edge-edge force multiplier
  -> change edge-node force multiplier
  -> change node-label force multiplier
  -> change node-target force multiplier
  -> change mass of nodes
  -> change mass of labels
  -> change mass of edges
- toggle what is currently shown in graph
  -> visualization of (shortest) task paths
  -> play/stop animation
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as anim
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ttr_particle_graph import TTR_Particle_Graph
from particle_edge import Particle_Edge
import read_ttr_files as ttr_reader
from drag_handler import Drag_Handler

class Board_Layout_GUI:
  def __init__(self,
      color_config: dict = {
          "bg_color":               "#1e1e1e", # darkest grey
          "fg_color":               "#d4d4d4", # light grey
          "frame_bg_color":         "#252526", # darker grey
          "label_bg_color":         "#252526", # darker grey
          "label_fg_color":         "#d4d4d4", # light grey
          "button_bg_color":        "#333333", # dark grey
          "button_fg_color":        "#f5f5f5", # white
          "button_hover_bg_color":  "#444444", # dark grey
          "button_hover_fg_color":  "#f5f5f5", # white
          "button_active_bg_color": "#555555", # dark grey
          "button_active_fg_color": "#f5f5f5", # white
          "entry_bg_color":         "#3c3c3c", # dark grey
          "entry_fg_color":         "#f5f5f5", # white
          "entry_select_color":     "#4477aa", # blue
          "plot_bg_color":          "#cccccc", # darker grey
          "plot_fg_color":          "#000000", # black
          "plot_grid_color":        "#aaaaaa", # black
          }):
    self.color_config = color_config
    # create master window in fullscreen
    self.master = tk.Tk()
    # maximize window
    self.master.configure(bg=self.color_config["bg_color"])
    self.master.minsize(800, 700)
    self.master.state("zoomed")
    # add fullscreen toggle
    self.master.bind("<F11>", self.toggle_fullscreen)
    self.master.title("Ticket-to-Ride board layout optimizer")

    self.grid_pad_x = 5
    self.grid_pad_y = 5
    self.font = "Roboto"
    self.font_size = 11

    self.background_image_mpl: np.ndarray = None
    self.plotted_background_image: plt.AxesImage = None
    self.background_image_extent: np.ndarray = None
    self.graph_data: dict = None
    self.particle_graph: TTR_Particle_Graph = None

    self.init_tk_variables()
    self.init_frames()
    self.init_animation()

    self.master.mainloop()

  def toggle_fullscreen(self, event=None):
    self.master.attributes("-fullscreen", not self.master.attributes("-fullscreen"))

  def add_frame_style(self, frame: tk.Frame):
    frame.configure(
      bg=self.color_config["frame_bg_color"],
      )

  def add_label_style(self, label: tk.Label, headline_level=5):
    if headline_level == 1:
      font_size = self.font_size + 6
    elif headline_level == 2:
      font_size = self.font_size + 4
    elif headline_level == 3:
      font_size = self.font_size + 3
    elif headline_level == 4:
      font_size = self.font_size + 1
    if headline_level == 5:
      font_size = self.font_size
    label.configure(
      bg=self.color_config["label_bg_color"],
      fg=self.color_config["label_fg_color"],
      font=(self.font, font_size),
      )

  def add_button_style(self, button: tk.Button):
    button.configure(
      bg=self.color_config["button_bg_color"],
      fg=self.color_config["button_fg_color"],
      activebackground=self.color_config["button_active_bg_color"],
      activeforeground=self.color_config["button_active_fg_color"],
      relief="flat",
      border=0,
      font=(self.font, self.font_size, "bold"),
      padx=self.grid_pad_x//2,
      pady=self.grid_pad_y//2,
      )

  def add_entry_style(self, entry: tk.Entry, justify="right"):
    entry.configure(
      bg=self.color_config["entry_bg_color"],
      fg=self.color_config["entry_fg_color"],
      insertbackground=self.color_config["button_fg_color"],
      selectbackground=self.color_config["entry_select_color"],
      relief="flat",
      border=0,
      justify=justify,
      font=(self.font, self.font_size),
      )

  def add_checkbutton_style(self, checkbutton: tk.Checkbutton):
    checkbutton.configure(
      bg=self.color_config["label_bg_color"],
      fg=self.color_config["label_fg_color"],
      activebackground=self.color_config["bg_color"],
      activeforeground=self.color_config["fg_color"],
      selectcolor=self.color_config["button_active_bg_color"],
      )


  def init_frames(self):
    """
    Create frames for the matplotlib animation and controls.
    """
    self.main_frame = tk.Frame(self.master, background=self.color_config["bg_color"])
    self.main_frame.place(relx=0.5,rely=0.5,anchor="c")
    # configure grid
    self.main_frame.columnconfigure(0, weight=1)
    self.main_frame.columnconfigure(1, weight=0)
    # create frame for matplotlib animation
    self.animation_frame = tk.Frame(self.main_frame, background=self.color_config["bg_color"])
    self.animation_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(2*self.grid_pad_x, 2*self.grid_pad_x),
        pady=(2*self.grid_pad_y, 2*self.grid_pad_y))

    # create frame for controls
    self.control_frame = tk.Frame(self.main_frame, background=self.color_config["bg_color"])
    self.control_frame.grid(
        row=0,
        column=1,
        sticky="e",
        padx=(2*self.grid_pad_x, 0),
        pady=(0,0),
        ipadx=2*self.grid_pad_x,
        ipady=2*self.grid_pad_y)

    self.draw_control_widgets()

    # create matplotlib figure
    self.fig = plt.figure(figsize=(15, 10), dpi=100)
    self.fig.patch.set_facecolor(self.color_config["plot_bg_color"])
    self.ax = self.fig.add_subplot(111)
    self.ax.set_facecolor(self.color_config["plot_bg_color"])
    # hide frame, axis and ticks
    self.ax.set_xlim(0, 20)
    self.ax.set_ylim(0, 15)
    self.ax.axis("scaled")
    self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    self.update_frame()
    
    # create canvas for matplotlib figure
    self.canvas = FigureCanvasTkAgg(self.fig, master=self.animation_frame)
    self.canvas.get_tk_widget().grid(
        row=0,
        column=0,
        sticky="nsew")
    self.canvas.draw()
    self.ax.plot(0, 0, "o", color=self.color_config["plot_fg_color"])

  def init_tk_variables(self):
    """
    Initialize variables for user inputs
    """
    # variables for user inputs
    self.node_file = tk.StringVar(value="beleriand_ttr//beleriand_locations.txt", name="node_file")
    self.edge_file = tk.StringVar(value="beleriand_ttr//beleriand_paths.txt", name="edge_file")
    self.task_file = tk.StringVar(value="beleriand_ttr//beleriand_tasks.txt", name="task_file")
    self.background_file = tk.StringVar(value="beleriand_ttr//beleriand_map.png", name="background_file")

    base_colors = [
      "#000000", # black
      "#ffffff", # white
      "#ff0000", # red
      "#00ff00", # green
      "#0000ff", # blue
      "#ffff00", # yellow
      "#ff00ff", # magenta
      "#00ffff", # cyan
    ]
    self.edge_colors = [tk.StringVar(value=color, name=f"edge_color_{i}") for i, color in enumerate(base_colors)]
    
    # variables for particle simulation
    self.time_step = tk.DoubleVar(value=0.1, name="time_step")
    self.iterations_per_frame = tk.IntVar(value=10, name="iterations_per_frame")
    self.velocity_decay = tk.DoubleVar(value=0.99, name="velocity_decay")
    self.edge_edge_force = tk.DoubleVar(value=0.1, name="edge_edge_force")
    self.edge_node_force = tk.DoubleVar(value=0.1, name="edge_node_force")
    self.node_label_force = tk.DoubleVar(value=0.1, name="node_label_force")
    self.node_target_force = tk.DoubleVar(value=0.1, name="node_target_force")
    self.node_mass = tk.DoubleVar(value=1.0, name="node_mass")
    self.label_mass = tk.DoubleVar(value=1.0, name="label_mass")
    self.edge_mass = tk.DoubleVar(value=1.0, name="edge_mass")
    self.interaction_radius = tk.DoubleVar(value=15.0, name="interaction_radius")
    self.repulsion_strength = tk.DoubleVar(value=2.0, name="repulsion_strength")

    # variables for toggles
    self.show_nodes = tk.BooleanVar(value=True, name="show_nodes")
    self.show_edges = tk.BooleanVar(value=False, name="show_edges")
    self.show_labels = tk.BooleanVar(value=True, name="show_labels")
    self.show_targets = tk.BooleanVar(value=False, name="show_targets")
    self.show_background_image = tk.BooleanVar(value=True, name="show_background")
    self.show_task_paths = tk.BooleanVar(value=True, name="show_task_paths")
    self.simulation_paused = tk.BooleanVar(value=True, name="simulation_paused")
    self.show_plot_frame = tk.BooleanVar(value=True, name="show_plot_frame")

  def draw_control_widgets(self):
    """
    Draw widgets for user inputs. Place them in the control frame using grid layout.
    """
    row_index = 0
    # frame for file inputs
    file_frame = tk.Frame(self.control_frame)
    self.add_frame_style(file_frame)
    file_frame.grid(
        row=row_index,
        column=0,
        sticky="nsew",
        pady=(0, self.grid_pad_y))
    row_index += 1
    self.draw_file_widgets(file_frame)

    # frame for particle simulation parameters
    particle_frame = tk.Frame(self.control_frame)
    self.add_frame_style(particle_frame)
    particle_frame.grid(
        row=row_index,
        column=0,
        sticky="ew",
        pady=(0, self.grid_pad_y))
    row_index += 1
    self.draw_particle_widgets(particle_frame)

    # frame for toggles
    toggle_frame = tk.Frame(self.control_frame)
    self.add_frame_style(toggle_frame)
    toggle_frame.grid(
        row=row_index,
        column=0,
        sticky="nsew",
        pady=(0, self.grid_pad_y))
    row_index += 1
    self.draw_toggle_widgets(toggle_frame)

  def draw_file_widgets(self, file_frame: tk.Frame):
    """
    Draw widgets for file inputs. Place them in the given frame using grid layout.

    Args:
        file_frame (tk.Frame): frame to place widgets in
    """
    # configure grid layout
    file_frame.columnconfigure(0, weight=1)
    file_frame.columnconfigure(1, weight=1)
    file_frame.columnconfigure(2, weight=1)
    row_index = 0
    # headline for file inputs
    category_label_files = tk.Label(file_frame, text="File Inputs", justify="center")
    self.add_label_style(category_label_files, headline_level=3)
    category_label_files.grid(
        row=row_index,
        column=0,
        columnspan=3,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    row_index += 1
    # node file input
    node_file_label = tk.Label(file_frame, text="Node File")
    self.add_label_style(node_file_label)
    node_file_label.grid(
        row=row_index,
        column=0,
        sticky="ne",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    node_file_entry = tk.Entry(file_frame, textvariable=self.node_file, width=12)
    self.add_entry_style(node_file_entry)
    node_file_entry.grid(
        row=row_index,
        column=1,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    node_file_button = tk.Button(file_frame, 
        text="Browse",
        command=lambda: self.browse_txt_file("browse locations file (.txt)", self.node_file))
    self.add_button_style(node_file_button)
    node_file_button.grid(
        row=row_index,
        column=2,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    row_index += 1

    # edge file input
    edge_file_label = tk.Label(file_frame, text="Edge File")
    self.add_label_style(edge_file_label)
    edge_file_label.grid(
        row=row_index,
        column=0,
        sticky="ne",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    edge_file_entry = tk.Entry(file_frame, textvariable=self.edge_file, width=12)
    self.add_entry_style(edge_file_entry)
    edge_file_entry.grid(
        row=row_index,
        column=1,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    edge_file_button = tk.Button(file_frame, 
        text="Browse",
        command=lambda: self.browse_txt_file("browse paths file (.txt)", self.edge_file))
    self.add_button_style(edge_file_button)
    edge_file_button.grid(
        row=row_index,
        column=2,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    row_index += 1

    # task file input
    task_file_label = tk.Label(file_frame, text="Task File")
    self.add_label_style(task_file_label)
    task_file_label.grid(
        row=row_index,
        column=0,
        sticky="ne",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    task_file_entry = tk.Entry(file_frame, textvariable=self.task_file, width=12)
    self.add_entry_style(task_file_entry)
    task_file_entry.grid(
        row=row_index,
        column=1,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    task_file_button = tk.Button(file_frame,
        text="Browse",
        command=lambda: self.browse_txt_file("browse tasks file (.txt)", self.task_file))
    self.add_button_style(task_file_button)
    task_file_button.grid(
        row=row_index,
        column=2,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    row_index += 1

    # background file input
    background_file_label = tk.Label(file_frame, text="Background File")
    self.add_label_style(background_file_label)
    background_file_label.grid(
        row=row_index,
        column=0,
        sticky="ne",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    background_file_entry = tk.Entry(file_frame, textvariable=self.background_file, width=12)
    self.add_entry_style(background_file_entry)
    background_file_entry.grid(
        row=row_index,
        column=1,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    background_file_button = tk.Button(file_frame,
        text="Browse",
        command=lambda: self.browse_image_file("browse background file", self.background_file))
    self.add_button_style(background_file_button)
    background_file_button.grid(
        row=row_index,
        column=2,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    row_index += 1

    # load button
    load_button = tk.Button(file_frame, text="Load", command=self.load_files)
    self.add_button_style(load_button)
    load_button.grid(
        row=row_index,
        column=0,
        columnspan=3,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))

  def browse_txt_file(self, browse_request: str, var: tk.StringVar):
    """
    Open a file dialog to select a txt file.

    Args:
        browse_request (str): text to display in the file dialog
        var (tk.StringVar): variable to store the file path in
    """
    file_path = tk.filedialog.askopenfilename(filetypes=[(browse_request, "*.txt")])
    if file_path:
      var.set(file_path)

  def browse_image_file(self, browse_request: str, var: tk.StringVar):
    """
    Open a file dialog to select an image file.

    Args:
        browse_request (str): text to display in the file dialog
        var (tk.StringVar): variable to store the file path in
    """
    file_path = tk.filedialog.askopenfilename(filetypes=[(browse_request, ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"])])
    if file_path:
      var.set(file_path)

  def load_files(self):
    """
    Load the files specified in the file inputs.
    """
    # load node file (locations)
    locations = ttr_reader.read_locations(self.node_file.get())
    # load edge file (paths)
    paths = ttr_reader.read_paths(self.edge_file.get())
    # load task file (tasks)
    tasks = ttr_reader.read_tasks(self.task_file.get())

    self.graph_data = {
      "locations": locations,
      "paths": paths,
      "tasks": tasks}
    # load background image
    try:
      self.background_image_mpl = mpimg.imread(self.background_file.get())
    except FileNotFoundError:
      print("Background image file not found.")
    self.update_background_image()
    self.init_particle_graph()

  def draw_particle_widgets(self, particle_frame: tk.Frame):
    """
    Draw widgets for particle simulation parameters. Place them in the given frame using grid layout.

    Inputs for:
    - time step size
    - number of time steps per frame
    - velocity decay factor
    - edge-edge attraction factor
    - edge-node attraction factor
    - node-label attraction factor
    - node-target attraction factor
    - node mass
    - edge mass
    - label mass

    Args:
        particle_frame (tk.Frame): frame to place widgets in
    """
    def add_label_and_entry(row_index: int, text: str, var: tk.StringVar):
      """
      Add a label and entry widget to the given frame.

      Args:
          row_index (int): row index to place the widgets in
          text (str): text to display in the label
          var (tk.StringVar): variable to store the entry value in
      """
      label = tk.Label(particle_frame, text=text)
      self.add_label_style(label)
      label.grid(
          row=row_index,
          column=0,
          sticky="ne",
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
      entry = tk.Entry(particle_frame, textvariable=var, width=4)
      self.add_entry_style(entry)
      entry.grid(
          row=row_index,
          column=1,
          sticky="nw",
          padx=(0, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
    
    # configure grid layout
    particle_frame.columnconfigure(0, weight=1)
    particle_frame.columnconfigure(1, weight=1)
    row_index = 0
    # headline for particle graph parameters
    headline = tk.Label(particle_frame, text="Particle Graph Parameters", justify="center")
    self.add_label_style(headline, headline_level=3)
    headline.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    row_index += 1
    add_label_and_entry(row_index, "Time Step Size", self.time_step)
    row_index += 1
    add_label_and_entry(row_index, "Time Steps Per Frame", self.iterations_per_frame)
    row_index += 1
    add_label_and_entry(row_index, "Velocity Decay", self.velocity_decay)
    row_index += 1
    add_label_and_entry(row_index, "Edge-Edge Attraction", self.edge_edge_force)
    row_index += 1
    add_label_and_entry(row_index, "Edge-Node Attraction", self.edge_node_force)
    row_index += 1
    add_label_and_entry(row_index, "Node-Label Attraction", self.node_label_force)
    row_index += 1
    add_label_and_entry(row_index, "Node-Target Attraction", self.node_target_force)
    row_index += 1
    add_label_and_entry(row_index, "Node Mass", self.node_mass)
    row_index += 1
    add_label_and_entry(row_index, "Edge Mass", self.edge_mass)
    row_index += 1
    add_label_and_entry(row_index, "Label Mass", self.label_mass)
    row_index += 1
    add_label_and_entry(row_index, "interaction_radius", self.interaction_radius)
    row_index += 1
    add_label_and_entry(row_index, "repulsion_strength", self.repulsion_strength)
    row_index += 1
    # add button to load parameters into the particle simulation
    load_button = tk.Button(particle_frame, text="Load", command=self.load_particle_parameters)
    self.add_button_style(load_button)
    load_button.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))

  def load_particle_parameters(self):
    """
    Load the parameters for the particle simulation from the tkinter variables.
    """
    raise NotImplementedError # TODO

  def draw_toggle_widgets(self, toggle_frame: tk.Frame):
    """
    Draw widgets for toggling the display of different elements. Place them in the given frame using grid layout.

    Toggles for:
    - edges
    - nodes
    - labels
    - targets
    - background image
    - play/pause simulation
    - show shortest paths for tasks

    Args:
        toggle_frame (tk.Frame): frame to place widgets in
    """
    def add_label_and_checkbutton(
        row_index: int,
        text: str,
        var: tk.BooleanVar,
        command: Callable = self.update_canvas):
      """
      Add a label and checkbutton widget to the given frame.

      Args:
          row_index (int): row index to place the widgets in
          text (str): text to display in the label
          var (tk.BooleanVar): variable to store the checkbutton value in
      """
      # label = tk.Label(toggle_frame, text=text)
      # self.add_label_style(label)
      # label.grid(
      #     row=row_index,
      #     column=0,
      #     sticky="ne",
      #     padx=(self.grid_pad_x, self.grid_pad_x),
      #     pady=(0, self.grid_pad_y))
      checkbutton = tk.Checkbutton(toggle_frame, 
          text=text,
          variable=var,
          command=command)
      self.add_checkbutton_style(checkbutton)
      checkbutton.grid(
          row=row_index,
          column=1,
          sticky="nw",
          padx=(0, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
    
    # configure grid layout
    # toggle_frame.columnconfigure(0, weight=1)
    toggle_frame.columnconfigure(1, weight=1)
    row_index = 0
    # headline for the toggle widgets
    label = tk.Label(toggle_frame, text="Toggles", justify="center")
    self.add_label_style(label, headline_level=3)
    label.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, 0))
    row_index += 1
    add_label_and_checkbutton(row_index, "Nodes", self.show_nodes, command=self.update_nodes)
    row_index += 1
    add_label_and_checkbutton(row_index, "Edges", self.show_edges, command=self.update_edges)
    row_index += 1
    add_label_and_checkbutton(row_index, "Labels", self.show_labels, command=self.update_labels)
    row_index += 1
    add_label_and_checkbutton(row_index, "Targets", self.show_targets)
    row_index += 1
    add_label_and_checkbutton(row_index, "Show Tasks", self.show_task_paths)
    row_index += 1
    add_label_and_checkbutton(row_index, "Play/Pause", self.simulation_paused)
    row_index += 1 # TODO: prettify play/pause button
    add_label_and_checkbutton(row_index, "Background Image", self.show_background_image, command=self.update_background_image)
    row_index += 1
    add_label_and_checkbutton(row_index, "Show Plot Frame", self.show_plot_frame, command=self.update_frame)
    row_index += 1

  def update_canvas(self, *args):
    """
    Update the canvas with the current settings.
    """
    pass # TODO

  def update_nodes(self, *args):
    """
    Update the nodes with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.show_nodes.get():
      self.particle_graph.draw_nodes(self.ax)
    else:
      self.particle_graph.erase_nodes()

  def update_edges(self, *args):
    """
    Update the edges with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.show_edges.get():
      self.particle_graph.draw_edges(self.ax)
    else:
      self.particle_graph.erase_edges()

  def update_labels(self, *args):
    """
    Update the labels with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.show_labels.get():
      self.particle_graph.draw_labels(self.ax)
    else:
      self.particle_graph.erase_labels()

  def update_frame(self):
    """
    Update the frame with the current settings.
    """
    self.ax.set_frame_on(self.show_plot_frame.get())
    # show/hid ticks
    self.ax.tick_params(
        axis="both",
        which="both",
        bottom=self.show_plot_frame.get(),
        top=self.show_plot_frame.get(),
        left=self.show_plot_frame.get(),
        right=self.show_plot_frame.get(),
        labelbottom=self.show_plot_frame.get(),
        labelleft=self.show_plot_frame.get())
    if self.show_plot_frame.get():
      self.ax.grid(color=self.color_config["plot_grid_color"])
    else:
      self.ax.grid(False)

  def update_background_image(self):
    """
    Update the background image in self.ax if the background image is shown.
    """
    # set background image
    if self.show_background_image.get() and \
        self.background_image_mpl is not None:
      if self.plotted_background_image is not None:
        try:
          self.plotted_background_image.remove()
        except ValueError: # ignore if image is not plotted
          pass
      self.plotted_background_image = self.ax.imshow(self.background_image_mpl, extent=self.background_image_extent)
    elif self.plotted_background_image is not None:
      self.plotted_background_image.remove()

  def init_animation(self):
    """
    Initialize the animation.
    """
    # pass
    self.animation = anim.FuncAnimation(
        self.fig,
        self.update_canvas,
        interval=100,
        blit=False)

  def init_particle_graph(self):
    """
    Initialize the particle graph.
    arange nodes along left edge and corresponding labels to their right
    """
    print("init_particle_graph")
    if self.graph_data is None:
      print("No graph data available.")
      return
    node_positions = self.init_node_positions()
    if self.particle_graph is not None:
      self.particle_graph.erase()
    self.particle_graph = TTR_Particle_Graph(
        locations = self.graph_data["locations"],
        paths = self.graph_data["paths"],
        location_positions = node_positions,
        particle_parameters = self.get_particle_parameters()
    )
    if self.show_nodes.get():
      self.particle_graph.draw_nodes(self.ax)
    if self.show_labels.get():
      self.particle_graph.draw_labels(self.ax)
    if self.show_edges.get():
      self.particle_graph.draw_edges(self.ax)
    
    self.drag_handler = Drag_Handler(self.canvas, self.ax, self.particle_graph.get_particle_list())
    

  def init_node_positions(self, node_spacing: float = 3) -> dict:
    """
    Initialize the node positions along the left edge of the plot.

    Args:
      node_spacing: the spacing between the nodes in the y direction

    Returns:
      (dict): a dictionary with the node positions. keys are the node names, values are the node positions as numpy arrays
    """
    num_nodes = len(self.graph_data["locations"])
    image_height = (num_nodes + 1) * node_spacing
    node_positions: dict = dict()
    for i, location_name in enumerate(self.graph_data["locations"]):
      node_positions[location_name] = np.array([
          10,
          image_height - node_spacing*(i+1)])

    # rescale background image to fit the node positions along the y axis
    if self.background_image_mpl is not None:
      scale_factor = image_height / self.background_image_mpl.shape[1]
      self.background_image_extent = np.array([
          0,
          image_height,
          0,
          self.background_image_mpl.shape[0]*scale_factor])
      # update plot limits
      self.ax.set_xlim(0, self.background_image_mpl.shape[0]*scale_factor)
      self.ax.set_ylim(0, self.background_image_mpl.shape[1]*scale_factor)
      self.update_background_image()
    else:
      self.ax.set_xlim(0, image_height*2)
      self.ax.set_ylim(0, image_height)
    return node_positions


  def get_particle_parameters(self):
    """
    Get the particle parameters from the tkinter variables.
    """
    return {
        "velocity_decay": self.velocity_decay.get(),
        "edge-edge": self.edge_edge_force.get(),
        "edge-node": self.edge_node_force.get(),
        "node-label": self.node_label_force.get(),
        "node-target": self.node_target_force.get(),
        "node_mass": self.node_mass.get(),
        "edge_mass": self.edge_mass.get(),
        "label_mass": self.label_mass.get(),
        "interaction_radius": self.interaction_radius.get(),
        "repulsion_strength": self.repulsion_strength.get(),
    }


if __name__ == "__main__":
  gui = Board_Layout_GUI()
  # tk.mainloop()