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
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as anim
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from multi_monitor_fullscreen import toggle_full_screen
from auto_scroll_frame import Auto_Scroll_Frame
from ttr_particle_graph import TTR_Particle_Graph
from particle_edge import Particle_Edge
import read_ttr_files as ttr_reader
from drag_handler import Drag_Handler
import pokemon_colors as pkmn_colors

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
    self.master.title("Ticket-to-Ride board layout optimizer")
    self.master.state("zoomed")
    self.master.minsize(500, 400)
    # self.master.resizable(False, False)
    # add fullscreen toggle
    self.master.bind("<F11>", lambda event: toggle_full_screen(self.master))

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
    # self.init_animation() # TODO: implement animation

    self.master.mainloop()


  def add_frame_style(self, frame: tk.Frame):
    frame.configure(
      bg=self.color_config["frame_bg_color"],
      )

  def add_label_style(self, label: tk.Label, headline_level: int = 5, font_type: str = "normal"):
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
      font=(self.font, font_size, font_type),
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
    self.main_frame.rowconfigure(0, weight=1)
    # create frame for matplotlib animation
    self.animation_frame = tk.Frame(self.main_frame, background=self.color_config["bg_color"])
    self.animation_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(2*self.grid_pad_x, 2*self.grid_pad_x),
        pady=(2*self.grid_pad_y, 2*self.grid_pad_y))
    # stretch animation frame to fill window
    self.animation_frame.grid_columnconfigure(0, weight=1)
    self.animation_frame.grid_rowconfigure(0, weight=1)
    self.animation_frame.grid_rowconfigure(1, weight=0)

    # create frame for controls
    control_outer_frame = tk.Frame(self.main_frame, background=self.color_config["bg_color"], height=self.master.winfo_height()-2*self.grid_pad_y, )
    self.master.bind("<Configure>", lambda event: self.control_frame_size_update(event, control_outer_frame))
    control_outer_frame.grid(
        row=0,
        column=1,
        sticky="nse")
    self.control_frame = Auto_Scroll_Frame(control_outer_frame, 
        frame_kwargs=dict(background=self.color_config["bg_color"]),
        scrollbar_kwargs=dict(
            troughcolor=self.color_config["bg_color"],
            activebackground=self.color_config["button_active_bg_color"],
            bg=self.color_config["button_bg_color"],
            border=0,
            width=10,
            highlightthickness=0,
            highlightbackground=self.color_config["bg_color"],
            highlightcolor=self.color_config["bg_color"],
            )
        )
    self.control_frame.grid(
        row=0,
        column=0,
        sticky="e",
        padx=(2*self.grid_pad_x, 0),
        pady=(0,0))
        # ipadx=2*self.grid_pad_x,
        # ipady=2*self.grid_pad_y)
    self.control_frame = self.control_frame.scrollframe
    self.master.update()

    self.draw_control_widgets()
    # control_outer_frame.grid_propagate(False)

    # create matplotlib figure
    # self.master.update()
    # self.animation_frame.update()
    # px = 1 / plt.rcParams["figure.dpi"]
    # total_width = self.master.winfo_width()
    # total_height = self.master.winfo_height()
    # navigation_bar_height: int = 150 # height of navigation bar in pixels # TODO: get height of navigation bar
    # width = (total_width - (self.control_frame.winfo_width() + 2 * self.grid_pad_x))
    # height = (total_height - (2 * self.grid_pad_y + navigation_bar_height))
    # print(f"window size: {total_width}x{total_height}")
    # print(f"px to inches factor: {px}")
    # print(f"plot size: {width}x{height}")
    self.fig = plt.figure(figsize=(15,10), dpi=100)
    # self.fig = plt.figure(figsize=(width*px, height*px), dpi=100)
    self.fig.patch.set_facecolor(self.color_config["plot_bg_color"])
    self.ax = self.fig.add_subplot(111)
    self.ax.set_facecolor(self.color_config["plot_bg_color"])
    # hide frame, axis and ticks
    self.ax.set_xlim(0, 20)
    self.ax.set_ylim(0, 15)
    self.ax.axis("scaled")
    
    # create canvas for matplotlib figure
    self.canvas = FigureCanvasTkAgg(self.fig, master=self.animation_frame)
    self.canvas.get_tk_widget().grid(
        row=0,
        column=0,
        sticky="nsew")
    # add toolbar
    self.toolbar = NavigationToolbar2Tk(self.canvas, self.animation_frame, pack_toolbar=False)
    self.toolbar.update()
    self.toolbar.grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=(0, 0),
        pady=(self.grid_pad_y, 0))

    self.fig.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.975, wspace=None, hspace=None)
    self.toggle_mpl_frame_visibility()
    # self.canvas.draw_idle()

  def control_frame_size_update(self, event: tk.Event, control_outer_frame: tk.Frame):
    """
    Update control frame size when window size changes
    """
    if event.widget != self.master:
      return
    height = self.master.winfo_height() - 2*self.grid_pad_y
    control_outer_frame.config(height=height, width=control_outer_frame.winfo_width() - 2*self.grid_pad_x)
    control_outer_frame.update()

  def init_tk_variables(self):
    """
    Initialize variables for user inputs
    """
    # variables for user inputs
    self.node_file = tk.StringVar(value="beleriand_ttr//beleriand_locations.txt", name="node_file")
    self.edge_file = tk.StringVar(value="beleriand_ttr//beleriand_paths.txt", name="edge_file")
    self.task_file = tk.StringVar(value="beleriand_ttr//beleriand_tasks.txt", name="task_file")
    self.background_file = tk.StringVar(value="beleriand_ttr//beleriand_map.png", name="background_file")
    self.particle_graph_file = tk.StringVar(value="beleriand_ttr//beleriand_particle_graph.json", name="particle_graph_file")

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
    self.show_edges = tk.BooleanVar(value=True, name="show_edges")
    self.show_labels = tk.BooleanVar(value=True, name="show_labels")
    self.show_targets = tk.BooleanVar(value=False, name="show_targets") # unused # TODO: implement
    self.show_background_image = tk.BooleanVar(value=True, name="show_background")
    self.show_task_paths = tk.BooleanVar(value=False, name="show_task_paths")
    self.show_plot_frame = tk.BooleanVar(value=False, name="show_plot_frame")
    self.use_edge_images = tk.BooleanVar(value=False, name="use_edge_images")
    self.simulation_paused = tk.BooleanVar(value=True, name="simulation_paused") # unused # TODO: implement

    # variables for plot
    self.board_width = tk.DoubleVar(value=83.1, name="board_width")
    self.board_height = tk.DoubleVar(value=54.0, name="board_height")
    self.background_image_offset_x = tk.DoubleVar(value=0.0, name="background_image_offset_x")
    self.background_image_offset_y = tk.DoubleVar(value=0.0, name="background_image_offset_y")
    self.board_scale_factor = tk.DoubleVar(value=1.25, name="board_scale_factor")
    self.node_scale_factor = tk.DoubleVar(value=0.8, name="node_scale_factor")

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

    # frame for other buttons
    button_frame = tk.Frame(self.control_frame)
    self.add_frame_style(button_frame)
    button_frame.grid(
        row=row_index,
        column=0,
        sticky="nsew",
        pady=(0, self.grid_pad_y))
    row_index += 1
    self.draw_button_widgets(button_frame)

    # frame for plot parameters
    plot_param_frame = tk.Frame(self.control_frame)
    self.add_frame_style(plot_param_frame)
    plot_param_frame.grid(
        row=row_index,
        column=0,
        sticky="nsew",
        pady=(0, self.grid_pad_y))
    row_index += 1
    self.draw_board_scaling_widgets(plot_param_frame)

  def create_submenu_button(self,
      master: tk.Widget,
      text: str,
      widget_list: list,
      row_index: int = 0,
      column_index: int = 0,
      columnspan: int = 1,
      ) -> Tuple[tk.Button, Callable]:
    """
    Draw a button that opens a submenu with the given widgets.

    Args:
        master (tk.Widget): parent widget
        text (str): text to display on the button
        widget_list (list): list of widgets to show/hide
        row_index (int, optional): row index to place the button. Defaults to 0.
        column_index (int, optional): column index to place the button. Defaults to 0.
        column_span (int, optional): column span of the button. Defaults to 1.

    Returns:
        Callable: function to toggle the visibility of the submenu
    """
    button = tk.Button(
        master,
        anchor="w",
        text="⯆ " + text)
    def toggle_function():
      """
      toggle the visibility of the submenu and change the button text
      """
      if button.cget("text").startswith("⯆"):
        new_text = "⯈ " + text
      else:
        new_text = "⯆ " + text
      toggle_widget_visibility(widget_list)
      button.config(text=new_text)
    button.config(command=toggle_function)
    self.add_button_style(button)
    button.grid(
        row=row_index,
        column=column_index,
        columnspan=columnspan,
        sticky="nsew",
        pady=(0, self.grid_pad_y))

    return button, toggle_function

  def draw_file_widgets(self, file_frame: tk.Frame):
    """
    Draw widgets for file inputs. Place them in the given frame using grid layout.

    Args:
        file_frame (tk.Frame): frame to place widgets in
    """
    file_widgets: list = list()
    # configure grid layout
    file_frame.columnconfigure(0, weight=1)
    file_frame.columnconfigure(1, weight=1)
    file_frame.columnconfigure(2, weight=1)
    def add_browse_button(row_index: int, command: Callable):
      """
      Add a button to browse for a file of the given type.
      """
      button = tk.Button(
          file_frame,
          text="Browse",
          command=command)
      self.add_button_style(button)
      button.grid(
          row=row_index,
          column=2,
          sticky="nsew",
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(self.grid_pad_y, self.grid_pad_y))
      file_widgets.append(button)

    def add_file_input(row_index: int, label_text: str, variable: tk.StringVar, command: Callable):
      """
      Add a file input widget to the given frame.
      """
      label = tk.Label(file_frame, text=label_text)
      self.add_label_style(label)
      label.grid(
          row=row_index,
          column=0,
          sticky="nsew",
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(self.grid_pad_y, self.grid_pad_y))
      file_widgets.append(label)
      entry = tk.Entry(file_frame, textvariable=variable)
      self.add_entry_style(entry)
      entry.grid(
          row=row_index,
          column=1,
          sticky="nsew",
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(self.grid_pad_y, self.grid_pad_y))
      file_widgets.append(entry)
      add_browse_button(row_index, command)

    row_index = 0
    # add submenu button for file inputs
    file_input_toggle_button, toggle_file_input_submenu = self.create_submenu_button(
      master=file_frame,
      text="File Inputs",
      widget_list=file_widgets,
      row_index=row_index,
      column_index=0,
      columnspan=3)
    row_index += 1
    add_file_input(row_index, "Node File", self.node_file, lambda: self.browse_txt_file("browse locations file", self.node_file))
    row_index += 1
    add_file_input(row_index, "Edge File", self.edge_file, lambda: self.browse_txt_file("browse paths file", self.edge_file))
    row_index += 1
    add_file_input(row_index, "Task File", self.task_file, lambda: self.browse_txt_file("browse tasks file", self.task_file))
    row_index += 1
    add_file_input(row_index, "Particle Graph", self.particle_graph_file, lambda: self.browse_json_file("browse particle graph file", self.particle_graph_file))
    row_index += 1
    add_file_input(row_index, "Background File", self.background_file, lambda: self.browse_image_file("browse background file", self.background_file))
    row_index += 1

    # load button
    def load_files_hide_menu():
      """
      Load the files and hide the submenu afterwards.
      """
      self.load_files()
      toggle_file_input_submenu()
    load_button = tk.Button(file_frame, text="Load", command=load_files_hide_menu)
    self.add_button_style(load_button)
    load_button.grid(
        row=row_index,
        column=0,
        columnspan=3,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    file_widgets.append(load_button)

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

  def browse_json_file(self, browse_request: str, var: tk.StringVar):
    """
    Open a file dialog to select a json file.

    Args:
        browse_request (str): text to display in the file dialog
        var (tk.StringVar): variable to store the file path in
    """
    file_path = tk.filedialog.askopenfilename(filetypes=[(browse_request, "*.json")])
    if file_path:
      var.set(file_path)

  def load_files(self):
    """
    Load the files specified in the file inputs.
    """
    # print info about self.master position
    print("self.master.winfo_rootx():", self.master.winfo_rootx())
    print("self.master.winfo_rooty():", self.master.winfo_rooty())
    print("self.master.winfo_x():", self.master.winfo_x())
    print("self.master.winfo_y():", self.master.winfo_y())
    print("self.master.winfo_width():", self.master.winfo_width())
    print("self.master.winfo_height():", self.master.winfo_height())
    print("self.master.winfo_screenwidth():", self.master.winfo_screenwidth())
    print("self.master.winfo_screenheight():", self.master.winfo_screenheight())
    print("self.master.winfo_geometry():", self.master.winfo_geometry())
    # clear figure
    # self.fig.clear()
    # reset variables
    self.particle_graph = None
    self.background_image = None
    self.graph_data = None
    # try loading particle graph
    try:
      self.particle_graph = TTR_Particle_Graph.load_json(self.particle_graph_file.get())
      locations = self.particle_graph.get_locations()
      paths = self.particle_graph.get_paths()
    except FileNotFoundError:
      print("Particle graph file not found. Constructing new particle graph.")
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
    self.ax.clear()
    self.toggle_background_image_visibility()

    self.init_particle_graph()

    self.drag_handler = Drag_Handler(self.canvas, self.ax, self.particle_graph.get_particle_list())


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
    particle_parameter_widgets: list = list()
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
      particle_parameter_widgets.append(label)
      entry = tk.Entry(particle_frame, textvariable=var, width=4)
      self.add_entry_style(entry)
      entry.grid(
          row=row_index,
          column=1,
          sticky="nw",
          padx=(0, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
      particle_parameter_widgets.append(entry)
    
    # configure grid layout
    particle_frame.columnconfigure(0, weight=1)
    particle_frame.columnconfigure(1, weight=1)
    row_index = 0
    # add submenu button for particle simulation parameters
    particle_parameter_toggle_button, toggle_particle_parameter_submenu = self.create_submenu_button(
        master=particle_frame,
        text="Particle Simulation Parameters",
        widget_list=particle_parameter_widgets,
        row_index=row_index,
        column_index=0,
        columnspan=2)
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
    add_label_and_entry(row_index, "Interaction Radius", self.interaction_radius)
    row_index += 1
    add_label_and_entry(row_index, "Repulsion Strength", self.repulsion_strength)
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
    particle_parameter_widgets.append(load_button)
    
    # hide particle parameter widgets
    # schedule the hiding of the widgets to be executed after the mainloop has started
    particle_frame.after(1000, toggle_particle_parameter_submenu)

  def load_particle_parameters(self):
    """
    Load the parameters for the particle simulation from the tkinter variables.
    """
    particle_parameters = {
      "velocity_decay": float(self.velocity_decay.get()),
      "edge-edge": float(self.edge_edge_force.get()),
      "edge-node": float(self.edge_node_force.get()),
      "node-label": float(self.node_label_force.get()),
      "node-target": float(self.node_target_force.get()),
      "node_mass": float(self.node_mass.get()),
      "edge_mass": float(self.edge_mass.get()),
      "label_mass": float(self.label_mass.get()),
      "interaction_radius": float(self.interaction_radius.get()),
      "repulsion_strength": float(self.repulsion_strength.get())}
    self.particle_graph.set_parameters(particle_parameters)	

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
    - show plot frame

    Args:
        toggle_frame (tk.Frame): frame to place widgets in
    """
    checkbox_toggle_widgets: list = list()
    def add_checkbutton(
        row_index: int,
        column_index: int,
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
          column=column_index,
          sticky="nw",
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
      checkbox_toggle_widgets.append(checkbutton)
    
    # configure grid layout
    toggle_frame.columnconfigure(0, weight=1)
    toggle_frame.columnconfigure(1, weight=1)
    row_index = 0
    column_index = 0
    # add submenu button for toggles
    toggles_button, toggle_toggles_submenu = self.create_submenu_button(
        master=toggle_frame,
        text="Toggles",
        widget_list=checkbox_toggle_widgets,
        row_index=row_index,
        column_index=0,
        columnspan=2)
    row_index += 1
    add_checkbutton(row_index, column_index, "Nodes", self.show_nodes, command=self.toggle_node_visibility)
    row_index += 1
    add_checkbutton(row_index, column_index, "Edges", self.show_edges, command=self.toggle_edge_visibility)
    row_index += 1
    add_checkbutton(row_index, column_index, "Labels", self.show_labels, command=self.toggle_label_visibility)
    row_index += 1
    add_checkbutton(row_index, column_index, "Targets", self.show_targets)
    row_index = 1 # reset for second column
    column_index = 1
    add_checkbutton(row_index, column_index, "Show Tasks", self.show_task_paths)
    row_index += 1
    add_checkbutton(row_index, column_index, "Background Image", self.show_background_image, command=self.toggle_background_image_visibility)
    row_index += 1
    add_checkbutton(row_index, column_index, "Show Plot Frame", self.show_plot_frame, command=self.toggle_mpl_frame_visibility)
    row_index += 1
    add_checkbutton(row_index, column_index, "edge images", self.use_edge_images, command=self.toggle_edge_images)
  
  def toggle_toggle_widgets(self):
    """
    Toggle the visibility of the toggle widgets (checkboxes).
    """
    for widget in checkbox_toggle_widgets:
      if widget.winfo_ismapped():
        widget.grid_remove()
      else:
        widget.grid()

  def update_canvas(self, *args):
    """
    Update the canvas with the current settings.
    This method will likely not be necessary in the future but will be replaced by more specific methods.
    """
    pass # TODO

  def toggle_node_visibility(self, *args) -> None:
    """
    Update the nodes with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.show_nodes.get():
      self.particle_graph.draw_nodes(self.ax)
    else:
      self.particle_graph.erase_nodes()
    self.canvas.draw_idle()

  def toggle_edge_visibility(self, *args) -> None:
    """
    Update the edges with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.show_edges.get():
      self.particle_graph.draw_edges(self.ax)
    else:
      self.particle_graph.erase_edges()
    self.canvas.draw_idle()

  def toggle_label_visibility(self, *args) -> None:
    """
    Update the labels with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.show_labels.get():
      self.particle_graph.draw_labels(self.ax)
    else:
      self.particle_graph.erase_labels()
    self.canvas.draw_idle()

  def toggle_task_visibility(self, *args) -> None:
    """
    Update the tasks with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.show_task_paths.get():
      self.use_edge_images.set(False)
      self.particle_graph.draw_tasks(self.ax)
    elif self.show_edges.get():
      self.particle_graph.draw_edges(self.ax)
    else:
      self.particle_graph.erase_edges()
    self.canvas.draw_idle()

  def toggle_mpl_frame_visibility(self) -> None:
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
      self.fig.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.975, wspace=None, hspace=None)
    else:
      self.ax.grid(False)
      self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    self.canvas.draw_idle()

  def toggle_background_image_visibility(self) -> None:
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
    self.canvas.draw_idle()

  def toggle_edge_images(self) -> None:
    """
    Update the edge images with the current settings.
    """
    if self.particle_graph is None:
      return
    if self.use_edge_images.get():
      self.show_task_paths.set(False)
      edge_colors = self.particle_graph.get_edge_colors()
      image_map = self.get_edge_color_map(edge_colors)
      self.particle_graph.set_edge_images(image_map)
      self.particle_graph.erase_edges()
      self.particle_graph.draw_edges(self.ax)
    else:
      edge_colors = self.particle_graph.get_edge_colors()
      color_map = {color: color for color in edge_colors}
      self.particle_graph.set_edge_colors(color_map)
      self.particle_graph.erase_edges()
      self.particle_graph.draw_edges(self.ax)
    
    self.canvas.draw_idle()

  def get_edge_color_map(self, edge_colors) -> dict:
    """
    Get the edge color map.

    Returns:
        dict: edge color map mapping colors to edge image paths
    """
    return {color: pkmn_colors.type_to_edge_image(
        pkmn_colors.color_to_energy_type(color)) for color in edge_colors}


  def draw_button_widgets(self, button_frame: tk.Frame) -> None:
    """
    Draw the button widgets in the given frame.
    """
    def add_control_button(
        row_index: int,
        column_index: int,
        text: str,
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
      button = tk.Button(button_frame, 
          text=text,
          command=command)
      self.add_button_style(button)
      padx = (self.grid_pad_x, self.grid_pad_x) if column_index == 0 else (0, self.grid_pad_x)
      pady = (self.grid_pad_y, self.grid_pad_y) if row_index == 0 else (0, self.grid_pad_y)
      button.grid(
          row=row_index,
          column=column_index,
          sticky="nsew",
          padx=padx,
          pady=pady)
      button_frame.columnconfigure(column_index, weight=1)

    row_index = 0
    column_index = 0
    # add buttons
    add_control_button(row_index, column_index, "Save graph", self.save_graph)
    column_index += 1
    add_control_button(row_index, column_index, "Snap labels", self.move_labels_to_nodes)
    column_index += 1
    add_control_button(row_index, column_index, "Scale graph", self.scale_graph_posistions)
    row_index += 1
    column_index = 0
    add_control_button(row_index, column_index, "Save img", self.save_image)
    column_index += 1
    add_control_button(row_index, column_index, "Snap edges", self.move_edges_to_nodes)
    column_index += 1
    add_control_button(row_index, column_index, "Scale img", self.scale_background_image)
    column_index += 1

  def play_pause(self):
    """
    Start/stop the particle simulation.
    """
    raise NotImplementedError # TODO

  def save_graph(self):
    """
    Save the current canvas.
    """
    if self.particle_graph is None:
      return
    # save particle graph object
    filepath = tk.filedialog.asksaveasfilename(
        title="Save particle graph as JSON",
        filetypes=(("JSON", "*.json"), ("all files", "*.*")))
    if filepath == "":
      return
    self.particle_graph.save_json(filepath)
  
  def save_image(self):
    # save canvas as image
    filepath = tk.filedialog.asksaveasfilename(
        title="Save canvas as image",
        filetypes=(("PNG", "*.png"), ("all files", "*.*")))
    if filepath == "":
      return
    self.fig.savefig(filepath, dpi=600, transparent=True)

  def move_labels_to_nodes(self):
    """
    Move the labels to the nodes.
    """
    if self.particle_graph is None:
      return
    self.particle_graph.move_labels_to_nodes(self.ax)
    self.canvas.draw_idle()

  def move_edges_to_nodes(self):
    """
    Move the edges to the nodes.
    """
    if self.particle_graph is None:
      return
    self.particle_graph.move_edges_to_nodes(self.ax, alpha=0.7)
    self.show_edges.set(True)
    self.canvas.draw_idle()

  def scale_background_image(self):
    """
    Resize the background image and canvas to ensure the background fits real lego pieces.
    new size will be the board size * scale_factor.
    """
    if self.background_image_mpl is None:
      return
    # get new size
    new_width = self.board_width.get() * self.board_scale_factor.get()
    new_height = self.board_height.get() * self.board_scale_factor.get()
    x_offset = self.background_image_offset_x.get()
    y_offset = self.background_image_offset_y.get()

    self.background_image_extent = np.array([
        x_offset,
        new_width + x_offset,
        y_offset,
        new_height + y_offset])
    # update plot limits
    self.ax.set_xlim(x_offset, new_width + x_offset)
    self.ax.set_ylim(y_offset, new_height + y_offset)
    self.toggle_background_image_visibility()

  def scale_graph_posistions(self):
    """
    Scale the node positions to fit the new background image size.
    """
    if self.particle_graph is None:
      return
    self.particle_graph.scale_graph_positions(self.ax, self.node_scale_factor.get())
    self.canvas.draw_idle()

  def draw_board_scaling_widgets(self, plot_param_frame: tk.Frame):
    """
    draw widgets for plot parameter settings.
    settings include:
    wdith and height of physical board
    """
    board_size_widgets: list = list()
    def add_label_and_entry(row_index: int, column_index: int, text: str, var: tk.StringVar):
      """
      Add a label and entry widget to the given frame.

      Args:
          row_index (int): row index to place the widgets in
          text (str): text to display in the label
          var (tk.StringVar): variable to store the entry value in
      """
      label = tk.Label(plot_param_frame, text=text)
      self.add_label_style(label)
      label.grid(
          row=row_index,
          column=column_index,
          sticky="ne",
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
      board_size_widgets.append(label)
      entry = tk.Entry(plot_param_frame, textvariable=var, width=4)
      self.add_entry_style(entry)
      entry.grid(
          row=row_index,
          column=column_index+1,
          sticky="nw",
          padx=(0, self.grid_pad_x),
          pady=(0, self.grid_pad_y))
      # configure used columns to stretch
      plot_param_frame.columnconfigure(column_index, weight=1)
      plot_param_frame.columnconfigure(column_index+1, weight=1)
      board_size_widgets.append(entry)
    
    row_index = 0
    column_index = 0
    # add submenu button for physical board size
    board_scaling_button, toggle_board_scaling_submenu = self.create_submenu_button(
        master=plot_param_frame,
        text="Board scaling",
        widget_list=board_size_widgets,
        row_index=row_index,
        column_index=0,
        columnspan=4)

    row_index += 1
    # add Entries for width and height of physical board
    add_label_and_entry(row_index, column_index, "width", self.board_width)
    column_index += 2
    add_label_and_entry(row_index, column_index, "height", self.board_height)
    column_index += 2
    row_index += 1
    # add label for background image offset
    label = tk.Label(plot_param_frame, text="Background image offset", justify="center")
    self.add_label_style(label)
    label.grid(
        row=row_index,
        column=0,
        columnspan=4,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    board_size_widgets.append(label)
    row_index += 1
    column_index = 0
    # add Entries for width and height of physical board
    add_label_and_entry(row_index, column_index, "x", self.background_image_offset_x)
    column_index += 2
    add_label_and_entry(row_index, column_index, "y", self.background_image_offset_y)
    column_index += 2
    row_index += 1
    # add label for node and background scale factor
    label = tk.Label(plot_param_frame, text="scale factors", justify="center")
    self.add_label_style(label)
    label.grid(
        row=row_index,
        column=0,
        columnspan=4,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    board_size_widgets.append(label)
    row_index += 1
    column_index = 0
    # add Entry for board scale factor
    add_label_and_entry(row_index, column_index, "board", self.board_scale_factor)
    column_index += 2
    # add Entry for node scale factor
    add_label_and_entry(row_index, column_index, "nodes", self.node_scale_factor)


  def init_animation(self):
    """
    Initialize the animation.
    """
    # pass
    # self.animation = anim.FuncAnimation(
    #     self.fig,
    #     self.update_canvas,
    #     interval=10000,
    #     blit=False)

  def init_particle_graph(self):
    """
    Initialize the particle graph.
    arange nodes along left edge and corresponding labels to their right
    """
    print("init_particle_graph")
    node_positions = self.init_node_positions()
    if self.particle_graph is None:
      if self.graph_data is None:
        print("No graph data available.")
        return
      print("new graph")
      if self.particle_graph is not None:
        self.particle_graph.erase()
      self.particle_graph = TTR_Particle_Graph(
          locations = self.graph_data["locations"],
          paths = self.graph_data["paths"],
          node_positions = node_positions,
          particle_parameters = self.get_particle_parameters()
      )
    if self.show_nodes.get():
      self.particle_graph.draw_nodes(self.ax)
    if self.show_labels.get():
      self.particle_graph.draw_labels(self.ax)
    if self.show_edges.get():
      self.particle_graph.draw_edges(self.ax)


  def init_node_positions(self, node_spacing: float = 1.5) -> dict:
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
          2,
          image_height - node_spacing*(i+1)])

    # rescale background image to fit the node positions along the y axis
    if self.background_image_mpl is not None:
      scale_factor = image_height / self.background_image_mpl.shape[0]
      self.background_image_extent = np.array([
          0,
          self.background_image_mpl.shape[1]*scale_factor,
          0,
          image_height,])
      # update plot limits
      self.ax.set_xlim(0, self.background_image_mpl.shape[1]*scale_factor)
      self.ax.set_ylim(0, image_height)
      self.toggle_background_image_visibility()
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



def toggle_widget_visibility(widget_list: List[tk.Widget]):
  """
  Toggle the visibility of a list of widgets. Widgets should be displayed using the grid layout manager.
  """
  for widget in widget_list:
    if widget.winfo_ismapped():
      widget.grid_remove()
    else:
      widget.grid()


if __name__ == "__main__":
  vscode_colors = {
      "bg_color":               "#1e1e1e", # darkest grey
      "fg_color":               "#d4d4d4", # light grey
      "frame_bg_color":         "#252526", # darker grey
      "label_bg_color":         "#252526", # darker grey
      "label_fg_color":         "#d4d4d4", # light grey
      "button_bg_color":        "#333333", # dark grey
      "button_fg_color":        "#f5f5f5", # white
      "button_hover_bg_color":  "#444444", # lighter dark grey
      "button_hover_fg_color":  "#f5f5f5", # white
      "button_active_bg_color": "#555555", # light dark grey
      "button_active_fg_color": "#f5f5f5", # white
      "entry_bg_color":         "#3c3c3c", # dark grey
      "entry_fg_color":         "#f5f5f5", # white
      "entry_select_color":     "#4477aa", # blue
      "plot_bg_color":          "#cccccc", # darker grey
      "plot_fg_color":          "#000000", # black
      "plot_grid_color":        "#aaaaaa", # black
      }
  blender_colors = {
      "bg_color":               "#1d1d1d", # darkest grey
      "fg_color":               "#d4d4d4", # light grey
      "frame_bg_color":         "#303030", # darker grey
      "label_bg_color":         "#303030", # darker grey
      "label_fg_color":         "#ffffff", # light grey
      "button_bg_color":        "#3d3d3d", # dark grey
      "button_fg_color":        "#ffffff", # white
      "button_hover_bg_color":  "#444444", # lighter dark grey
      "button_hover_fg_color":  "#f5f5f5", # white
      "button_active_bg_color": "#555555", # light dark grey
      "button_active_fg_color": "#f5f5f5", # white
      "entry_bg_color":         "#232323", # dark grey
      "entry_fg_color":         "#f5f5f5", # white
      "entry_select_color":     "#4772b3", # blue
      "plot_bg_color":          "#cccccc", # darker grey
      "plot_fg_color":          "#000000", # black
      "plot_grid_color":        "#aaaaaa", # black
      }
  gui = Board_Layout_GUI(color_config=blender_colors)
  # tk.mainloop()