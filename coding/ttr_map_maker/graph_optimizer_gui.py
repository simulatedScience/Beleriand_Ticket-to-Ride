"""
This module contains a class that allows users to controol the automatic graph optimizer.


"""

import time
import tkinter as tk
from typing import Callable, Tuple, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ttr_particle_graph import TTR_Particle_Graph

class Graph_Optimzer_GUI:
  def __init__(self,
      master: tk.Tk,
      color_config: dict[str, str],
      grid_padding: Tuple[int, int],
      tk_config_methods: dict[str, Callable],
      particle_graph: TTR_Particle_Graph,
      graph_optimizer_frame: tk.Frame,
      fig: plt.Figure,
      ax: plt.Axes,
      canvas: FigureCanvasTkAgg,
      ):
    """
    Add GUI elements to control the graph optimizer. Set simulation parameters and start/stop the simulation.

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
        particle_graph (TTR_Particle_Graph): The particle graph to edit.
        graph_edit_frame (tk.Frame): The tkinter frame where the edit mode widgets are displayed.
        ax (plt.Axes):The matplotlib Axes object where the particle graph is displayed.
        canvas (FigureCanvasTkAgg): The tkinter canvas where the mpl figure is shown
    """
    self.master: tk.Tk = master
    self.graph_optimizer_frame: tk.Frame = graph_optimizer_frame
    self.particle_graph: TTR_Particle_Graph = particle_graph
    self.ax: plt.Axes = ax
    self.fig: plt.Figure = fig
    self.canvas: FigureCanvasTkAgg = canvas
    
    self.color_config: dict[str, str] = color_config
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
    
    self.init_simulation_settings()
    
    
  def init_simulation_settings(self) -> None:
    """
    Initialize variables for simulation settings.
    """
    self.simulation_running: bool = False
    # variables for particle simulation
    self.time_step = tk.DoubleVar(value=0.1, name="time_step")
    self.iterations_per_frame = tk.IntVar(value=3, name="iterations_per_frame")
    self.velocity_decay = tk.DoubleVar(value=0.99, name="velocity_decay")
    self.edge_edge_force = tk.DoubleVar(value=0.3, name="edge_edge_force")
    self.edge_node_force = tk.DoubleVar(value=0.3, name="edge_node_force")
    self.node_label_force = tk.DoubleVar(value=0.03, name="node_label_force")
    self.node_target_force = tk.DoubleVar(value=0.5, name="node_target_force")
    self.node_mass = tk.DoubleVar(value=3.0, name="node_mass")
    self.label_mass = tk.DoubleVar(value=0.1, name="label_mass")
    self.edge_mass = tk.DoubleVar(value=1.0, name="edge_mass")
    self.interaction_radius = tk.DoubleVar(value=15.0, name="interaction_radius")
    self.repulsion_strength = tk.DoubleVar(value=2.0, name="repulsion_strength")
    
    # self.simulation_paused = tk.BooleanVar(value=True, name="simulation_paused") # unused # TODO: implement
    
    self.draw_particle_widgets(self.graph_optimizer_frame)


  def draw_particle_widgets(self, particle_frame: tk.Frame) -> None:
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
    row_index: int = 0
    # add submenu button for particle simulation parameters
    simulation_parameter_label: tk.Label = tk.Label(
        particle_frame,
        text="Particle Simulation Parameters")
    self.add_label_style(simulation_parameter_label, font_type="bold")
    simulation_parameter_label.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
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
    self.start_stop_button = tk.Button(particle_frame, text="Start", command=self.start_stop_optimization)
    self.add_button_style(self.start_stop_button)
    self.start_stop_button.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    particle_parameter_widgets.append(self.start_stop_button)

  def start_stop_optimization(self):
    """
    First, load the current parameters for the particle simulation. Then, start the particle simulation to optimize the graph layout.
    """
    if self.simulation_running:
      self.simulation_running = False
      self.start_stop_button.config(text="Start")
      return
    # start simulation
    self.start_stop_button.config(text="loading parameters")
    try:
      self.load_particle_parameters()
    except ValueError:
      self.start_stop_button.config(text="loading failed")
      time.sleep(0.5)
      self.start_stop_button.config(text="Start")
      return
      
    self.start_stop_button.config(text="Stop")
    self.simulation_running = True
    self.master.after(1, self.run_smulation_frame)
    
  
  def run_smulation_frame(self):
    """
    run the simulation for one frame (specified number of timesteps) and update the canvas with the results.
    """
    if not self.simulation_running:
      return
    print(f"running {self.iterations_per_frame.get()} simulation frame(s)")
    self.particle_graph.optimize_layout(
        iterations = self.iterations_per_frame.get(),
        dt = self.time_step.get())
    print("drawing simulation frame")
    self.particle_graph.erase()
    self.particle_graph.draw(ax=self.ax)
    self.canvas.draw_idle()
    self.master.after(1, self.run_smulation_frame)

  def load_particle_parameters(self):
    """
    Load the parameters for the particle simulation from the tkinter variables.
    """
    try:
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
    except tk.TclError:
      raise ValueError("Some parameters aren't numbers.")
    self.particle_graph.set_parameters(particle_parameters)	
