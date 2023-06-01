"""
This module implements task card export functionality for the TTR board layout GUI.

This is done via a class Task_Export_GUI which can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.
"""
from typing import List, Tuple, Callable
import os
import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from file_browsing import browse_image_file, browse_directory
from ttr_particle_graph import TTR_Particle_Graph
from particle_node import Particle_Node
from particle_label import Particle_Label
from ttr_task import TTR_Task

class Task_Export_GUI:
  def __init__(self,
      master: tk.Tk,
      color_config: dict[str, str],
      grid_padding: Tuple[float, float],
      tk_config_methods: dict[str, Callable],
      particle_graph: TTR_Particle_Graph,
      task_export_frame: tk.Frame,
      ax: plt.Axes,
      fig: plt.Figure,
      canvas: FigureCanvasTkAgg,
      background_image_mpl: np.ndarray = None,
      current_directory: str = None,
      ):
    """
    Args:
        master (tk.Tk): The master widget of the GUI.
        color_config (dict[str, str]): A dictionary containing the colors used in the GUI.
        grid_padding (Tuple[float, float]): The padding of the grid in the GUI.
        tk_config_methods (dict[str, Callable]):  Dictionary of methods to configure tkinter widgets (see `Board_Layout_GUI`). These should add styles to the widgets. Expect the following keys:
            - `add_frame_style(frame: tk.Frame)`
            - `add_label_style(label: tk.Label, headline_level: int, font_type: str)
            - `add_button_style(button: tk.Button)`
            - `add_entry_style(entry: tk.Entry, justify: str)`
            - `add_checkbutton_style(checkbutton: tk.Checkbutton)
            - `add_radiobutton_style(radiobutton: tk.Radiobutton)
            - `add_browse_button(frame: tk.Frame, row_index: int, column_index: int, command: Callable) -> tk.Button`
        particle_graph (TTR_Particle_Graph): The particle graph of the GUI.
        task_settings_frame (tk.Frame): The frame containing the task edit GUI.
        ax (plt.Axes): The axes where the graph is drawn.
        fig (plt.Figure): The figure where the graph is drawn.
        canvas (FigureCanvasTkAgg): The canvas for the graph.
        background_image_mpl (np.ndarray, optional): The background image of the graph. Defaults to None.
        current_directory (str, optional): The current directory of the GUI. Defaults to None.
    """
    self.master: tk.Tk = master
    self.color_config: dict[str, str] = color_config
    self.particle_graph: TTR_Particle_Graph = particle_graph
    self.task_export_frame: tk.Frame = task_export_frame
    self.ax: plt.Axes = ax
    self.fig: plt.Figure = fig
    self.canvas: FigureCanvasTkAgg = canvas
    
    self.background_image_mpl: np.ndarray = background_image_mpl
    self.plotted_images: dict[str, plt.AxesImage] = dict()
    self.card_background_image_extent: Tuple[float, float, float, float] = None
    self.graph_scale_factors: np.ndarray = np.ones(2) # scaling factors for the graph (x, y)

    # save grid padding for later use
    self.grid_pad_x: int = grid_padding[0]
    self.grid_pad_y: int = grid_padding[1]

    # extract tkinter style methods
    self.add_frame_style: Callable = tk_config_methods["add_frame_style"]
    self.add_label_style: Callable = tk_config_methods["add_label_style"]
    self.add_button_style: Callable = tk_config_methods["add_button_style"]
    self.add_entry_style: Callable = tk_config_methods["add_entry_style"]
    self.add_checkbutton_style: Callable = tk_config_methods["add_checkbutton_style"]
    self.add_radiobutton_style: Callable = tk_config_methods["add_radiobutton_style"]
    self.add_browse_button: Callable = tk_config_methods["add_browse_button"]

    # variables for task export
    self.card_frame_filepath: tk.StringVar = tk.StringVar(value="")
    self.card_frame_width: tk.DoubleVar = tk.DoubleVar(value=8.9)
    self.card_frame_height: tk.DoubleVar = tk.DoubleVar(value=6.4)
    self.background_image_width: tk.DoubleVar = tk.DoubleVar(value=8.9)
    self.background_image_height: tk.DoubleVar = tk.DoubleVar(value=5.6)
    self.background_image_offset_x: tk.DoubleVar = tk.DoubleVar(value=0.15)
    self.background_image_offset_y: tk.DoubleVar = tk.DoubleVar(value=0.3)

    self.label_scale: tk.DoubleVar = tk.DoubleVar(value=0.4)
    self.node_scale: tk.DoubleVar = tk.DoubleVar(value=0.25)
    self.node_image_filepath: tk.StringVar = tk.StringVar(value="")
    self.label_position_x: tk.DoubleVar = tk.DoubleVar(value=4.45)
    self.label_position_y: tk.DoubleVar = tk.DoubleVar(value=5.55)

    self.node_image_override: tk.BooleanVar = tk.BooleanVar(value=False)
    self.node_connector_lines: tk.BooleanVar = tk.BooleanVar(value=True)
    self.node_connector_line_width: tk.DoubleVar = tk.DoubleVar(value=10)

    self.points_image_directory: tk.StringVar = tk.StringVar(value=os.path.join(os.getcwd(), "assets", "points_images"))
    self.points_font_size: tk.DoubleVar = tk.DoubleVar(value=1.15)
    self.points_position_x: tk.DoubleVar = tk.DoubleVar(value=1.05)
    self.points_position_y: tk.DoubleVar = tk.DoubleVar(value=1.3)

    self.bonus_font_size: tk.DoubleVar = tk.DoubleVar(value=1.0)
    self.bonus_position_x: tk.DoubleVar = tk.DoubleVar(value=2.4)
    self.bonus_position_y: tk.DoubleVar = tk.DoubleVar(value=0.8)

    self.penalty_font_size: tk.DoubleVar = tk.DoubleVar(value=1.0)
    self.penalty_position_x: tk.DoubleVar = tk.DoubleVar(value=8.0)
    self.penalty_position_y: tk.DoubleVar = tk.DoubleVar(value=0.8)
    self.points_labels: dict[str, Particle_Node] = dict()

    self.selected_task: tk.IntVar = tk.IntVar(value=0)
    self.task_list: List[TTR_Task] = list(self.particle_graph.tasks.values())
    self.task_label: Particle_Label = None

    if current_directory is None:
      current_directory = os.getcwd()
    self.current_directory: str = current_directory
    self.export_filepath: tk.StringVar = tk.StringVar(value=os.path.join(current_directory, "task_cards"))

    # create widgets
    self.create_task_export_widgets()
  

  def create_task_export_widgets(self) -> None:
    """
    Creates the widgets for the task export GUI and place them in `self.task_settings_frame`
    """
    def add_numeric_input(
        parent: tk.Frame,
        row_index: int,
        column_index: int,
        label_text: str,
        variable: tk.DoubleVar,
        width: int = 4) -> Tuple[tk.Label, tk.Entry]:
      """
      Adds a label and entry widget to the given frame using the grid layout manager.

      Args:
          parent (tk.Frame): The frame to add the widgets to.
          row_index (int): The row index of the widgets.
          column_index (int): The column index of the widgets.
          label_text (str): The text of the label.
          variable (tk.DoubleVar): The variable of the entry widget.
          width (int, optional): The width of the entry widget. Defaults to 4.

      Returns:
          tk.Label: The label widget added to (row_index, column_index)
          tk.Entry: The entry widget added to (row_index, column_index + 1)
      """
      # create label
      label = tk.Label(parent, text=label_text)
      self.add_label_style(label)
      label.grid(
          row=row_index,
          column=column_index,
          sticky="e",
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(0, self.grid_pad_y),
          )
      # create entry
      entry = tk.Entry(parent, textvariable=variable, width=width)
      self.add_entry_style(entry, justify="right")
      entry.grid(
          row=row_index,
          column=column_index + 1,
          sticky="w",
          padx=(0, self.grid_pad_x),
          pady=(0, self.grid_pad_y),
          )
      return label, entry


    row_index: int = 0
    # create task export headline
    task_export_headline = tk.Label(self.task_export_frame, text="Task Export Settings")
    self.add_label_style(task_export_headline, font_type="bold")
    task_export_headline.grid(
        row=row_index,
        column=0,
        columnspan=2,
        padx=self.grid_pad_x,
        pady=self.grid_pad_y,
        sticky="nw",
        )
    row_index += 1
    # card frame settings
    # frame image file input
    border_image_frame: tk.Frame = tk.Frame(self.task_export_frame)
    self.add_frame_style(border_image_frame)
    border_image_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        padx=self.grid_pad_x,
        pady=(self.grid_pad_y, 0),
        sticky="new",
        )
    border_image_frame.grid_columnconfigure(1, weight=1)
    label = tk.Label(border_image_frame, text="Card frame image", justify="left")
    self.add_label_style(label)
    label.grid(
        row=row_index,
        column=0,
        sticky="nsw",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    entry = tk.Entry(border_image_frame, textvariable=self.card_frame_filepath, width=10)
    self.add_entry_style(entry)
    entry.xview_moveto(1)
    entry.grid(
        row=row_index,
        column=1,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    browse_button = self.add_browse_button(
        frame=border_image_frame,
        row_index=row_index,
        column_index=2,
        command=lambda: browse_image_file("Select a task card frame image.", self.card_frame_filepath),
        )
    row_index += 1
    # card frame image configuration (width, height, offset)
    border_configuration_frame: tk.Frame = tk.Frame(self.task_export_frame)
    self.add_frame_style(border_configuration_frame)
    border_configuration_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        padx=self.grid_pad_x,
        pady=0,
        sticky="new",
        )
    # card frame image size
    card_border_size_label: tk.Label = tk.Label(border_configuration_frame, text="Card frame size", anchor="w")
    self.add_label_style(card_border_size_label)
    card_border_size_label.grid(
        row=row_index,
        column=0,
        sticky="new",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    border_width_label, border_width_entry = add_numeric_input(
        parent=border_configuration_frame,
        row_index=row_index,
        column_index=1,
        label_text="width:",
        variable=self.card_frame_width,
        width=4,
        )
    border_height_label, border_height_entry = add_numeric_input(
        parent=border_configuration_frame,
        row_index=row_index,
        column_index=3,
        label_text="height:",
        variable=self.card_frame_height,
        width=4,
        )
    row_index += 1
    # background image width and height
    background_image_size_label: tk.Label = tk.Label(border_configuration_frame, text="Background image size", anchor="w")
    self.add_label_style(background_image_size_label)
    background_image_size_label.grid(
        row=row_index,
        column=0,
        sticky="new",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    background_image_width_label, background_image_width_entry = add_numeric_input(
        parent=border_configuration_frame,
        row_index=row_index,
        column_index=1,
        label_text="width:",
        variable=self.background_image_width,
        width=4,
        )
    background_image_height_label, background_image_height_entry = add_numeric_input(
        parent=border_configuration_frame,
        row_index=row_index,
        column_index=3,
        label_text="height:",
        variable=self.background_image_height,
        width=4,
        )
    row_index += 1
    # background image offset
    background_image_offset_label: tk.Label = tk.Label(border_configuration_frame, text="Background image offset", anchor="w")
    self.add_label_style(background_image_offset_label)
    background_image_offset_label.grid(
        row=row_index,
        column=0,
        sticky="new",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    background_image_offset_label, background_image_offset_entry = add_numeric_input(
        parent=border_configuration_frame,
        row_index=row_index,
        column_index=1,
        label_text="x:",
        variable=self.background_image_offset_x,
        width=4,
        )
    background_image_offset_label, background_image_offset_entry = add_numeric_input(
        parent=border_configuration_frame,
        row_index=row_index,
        column_index=3,
        label_text="y:",
        variable=self.background_image_offset_y,
        width=4,
        )
    row_index += 1
    # bind update methods to variables
    self.card_frame_width.trace_add("write", lambda *_: self.update_frame_image())
    self.card_frame_height.trace_add("write", lambda *_: self.update_frame_image())
    self.background_image_width.trace_add("write", lambda *_: self.update_background_image())
    self.background_image_height.trace_add("write", lambda *_: self.update_background_image())
    self.background_image_offset_x.trace_add("write", lambda *_: self.update_background_image())
    self.background_image_offset_y.trace_add("write", lambda *_: self.update_background_image())
    
    self.update_background_image()
    # settings for graph elements
    graph_settings_frame: tk.Frame = tk.Frame(self.task_export_frame)
    self.add_frame_style(graph_settings_frame)
    graph_settings_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=self.grid_pad_x,
        pady=(3 * self.grid_pad_y, self.grid_pad_y),
        )
    graph_settings_frame.grid_columnconfigure(1, weight=1)
    graph_settings_frame.grid_columnconfigure(3, weight=1)
    label_scale_label, label_scale_entry = add_numeric_input(
        parent=graph_settings_frame,
        row_index=0,
        column_index=0,
        label_text="Label scale:",
        variable=self.label_scale,
        width=4,
        )
    node_scale_label, node_scale_entry = add_numeric_input(
        parent=graph_settings_frame,
        row_index=0,
        column_index=2,
        label_text="Node scale:",
        variable=self.node_scale,
        width=4,
        )
    row_index += 1
    # add label position settings
    label_position_frame: tk.Frame = tk.Frame(graph_settings_frame)
    self.add_frame_style(label_position_frame)
    label_position_frame.grid(
        row=row_index,
        column=0,
        columnspan=4,
        sticky="new",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y),
        )
    label_position_frame.grid_columnconfigure(0, weight=1)
    label_position_label: tk.Label = tk.Label(label_position_frame, text="Label position", anchor="w")
    self.add_label_style(label_position_label)
    label_position_label.grid(
        row=row_index,
        column=0,
        sticky="new",
        padx=0,
        pady=0,
        )
    label_position_x_label, label_position_x_entry = add_numeric_input(
        parent=label_position_frame,
        row_index=row_index,
        column_index=1,
        label_text="x:",
        variable=self.label_position_x,
        width=4,
        )
    label_position_y_label, label_position_y_entry = add_numeric_input(
        parent=label_position_frame,
        row_index=row_index,
        column_index=3,
        label_text="y:",
        variable=self.label_position_y,
        width=4,
        )
    row_index += 1
    # optional node image override
    node_image_input_frame: tk.Frame = tk.Frame(graph_settings_frame)
    self.add_frame_style(node_image_input_frame)
    node_image_input_frame.grid(
        row=row_index,
        column=0,
        columnspan=4,
        sticky="new",
        padx=0,
        pady=0,
        )
    node_image_input_frame.grid_columnconfigure(1, weight=1)
    node_image_label = tk.Label(node_image_input_frame, text="Node image", justify="left")
    self.add_label_style(node_image_label)
    node_image_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    node_image_entry = tk.Entry(node_image_input_frame, textvariable=self.node_image_filepath, width=10)
    self.add_entry_style(node_image_entry)
    node_image_entry.xview_moveto(1)
    node_image_entry.grid(
        row=row_index,
        column=1,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    browse_button = self.add_browse_button(
        frame=node_image_input_frame,
        row_index=row_index,
        column_index=2,
        command=lambda: browse_image_file("Select a node image for task cards.", self.node_image_filepath),
        )
    row_index += 1
    # checkbutton for node image override and node connection lines
    node_image_override_checkbox = tk.Checkbutton(
        graph_settings_frame,
        text="Node image override",
        variable=self.node_image_override,
        command=self.update_node_image_override,
        anchor="w",
        )
    self.add_checkbutton_style(node_image_override_checkbox)
    node_image_override_checkbox.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    node_connection_lines_checkbox = tk.Checkbutton(
        graph_settings_frame,
        text="Node connection lines",
        variable=self.node_connector_lines,
        command=self.update_node_connector_lines,
        anchor="w",
        )
    self.add_checkbutton_style(node_connection_lines_checkbox)
    node_connection_lines_checkbox.grid(
        row=row_index,
        column=2,
        columnspan=2,
        sticky="ew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    row_index += 1
    # add inputs for points font size, color and positions
    points_inputs_frame: tk.Frame = tk.Frame(self.task_export_frame)
    self.add_frame_style(points_inputs_frame)
    points_inputs_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=(3 * self.grid_pad_y, self.grid_pad_y),
        )
    points_inputs_headline: tk.Label = tk.Label(points_inputs_frame, text="Points font sizes:", anchor="w")
    self.add_label_style(points_inputs_headline)
    points_inputs_headline.grid(
        row=0,
        column=0,
        columnspan=6,
        sticky="new",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=0,
        )
    # widgets for standard points
    standard_points_frame: tk.Frame = tk.Frame(points_inputs_frame)
    self.add_frame_style(standard_points_frame)
    standard_points_frame.grid(
        row=1,
        column=0,
        sticky="new",
        padx=0,
        pady=0,
        )
    points_font_size_label, points_font_size_entry = add_numeric_input(
        parent=standard_points_frame,
        row_index=0,
        column_index=0,
        label_text="points:",
        variable=self.points_font_size,
        width=3,
        )
    standard_points_position_frame: tk.Frame = tk.Frame(standard_points_frame)
    self.add_frame_style(standard_points_position_frame)
    standard_points_position_frame.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=0,
        )
    _ = add_numeric_input(
        parent=standard_points_position_frame,
        row_index=0,
        column_index=0,
        label_text="x:",
        variable=self.points_position_x,
        width=4,
        )
    _ = add_numeric_input(
        parent=standard_points_position_frame,
        row_index=0,
        column_index=2,
        label_text="y:",
        variable=self.points_position_y,
        width=4,
        )
    # widgets for bonus points
    bonus_points_frame: tk.Frame = tk.Frame(points_inputs_frame)
    self.add_frame_style(bonus_points_frame)
    bonus_points_frame.grid(
        row=1,
        column=1,
        sticky="new",
        padx=0,
        pady=0,
        )
    bonus_font_size_label, bonus_font_size_entry = add_numeric_input(
        parent=bonus_points_frame,
        row_index=0,
        column_index=0,
        label_text="bonus:",
        variable=self.bonus_font_size,
        width=3,
        )
    bonus_points_position_frame: tk.Frame = tk.Frame(bonus_points_frame)
    self.add_frame_style(bonus_points_position_frame)
    bonus_points_position_frame.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=0,
        )
    _ = add_numeric_input(
        parent=bonus_points_position_frame,
        row_index=0,
        column_index=0,
        label_text="x:",
        variable=self.bonus_position_x,
        width=4,
        )
    _ = add_numeric_input(
        parent=bonus_points_position_frame,
        row_index=0,
        column_index=2,
        label_text="y:",
        variable=self.bonus_position_y,
        width=4,
        )
    # widgets for penalty points
    penalty_points_frame: tk.Frame = tk.Frame(points_inputs_frame)
    self.add_frame_style(penalty_points_frame)
    penalty_points_frame.grid(
        row=1,
        column=2,
        sticky="new",
        padx=0,
        pady=0,
        )
    penalty_font_size_label, penalty_font_size_entry = add_numeric_input(
        parent=penalty_points_frame,
        row_index=0,
        column_index=0,
        label_text="penalty:",
        variable=self.penalty_font_size,
        width=3,
        )
    penalty_points_position_frame: tk.Frame = tk.Frame(penalty_points_frame)
    self.add_frame_style(penalty_points_position_frame)
    penalty_points_position_frame.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=0,
        )
    _ = add_numeric_input(
        parent=penalty_points_position_frame,
        row_index=0,
        column_index=0,
        label_text="x:",
        variable=self.penalty_position_x,
        width=4,
        )
    _ = add_numeric_input(
        parent=penalty_points_position_frame,
        row_index=0,
        column_index=2,
        label_text="y:",
        variable=self.penalty_position_y,
        width=4,
        )
    row_index += 1
    # add selector for the shown task
    task_selector_frame: tk.Frame = tk.Frame(self.task_export_frame)
    self.add_frame_style(task_selector_frame)
    task_selector_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=(3 * self.grid_pad_y, self.grid_pad_y),
        )
    task_selector_frame.grid_columnconfigure(1, weight=1)
    row_index += 1
    task_selector_headline: tk.Label = tk.Label(task_selector_frame, text="Shown task:", anchor="w")
    self.add_label_style(task_selector_headline)
    task_selector_headline.grid(
        row=0,
        column=0,
        columnspan=3,
        sticky="new",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=0,
        )
    
    task_selector_label: tk.Label = tk.Label(task_selector_frame, text="...loading...", anchor="center", cursor="hand2")
    self.add_label_style(task_selector_label)
    task_selector_label.grid(
        row=1,
        column=1,
        sticky="nsew",
        padx=0,
        pady=(0, self.grid_pad_y),
        )
    self.change_selected_task(0, task_selector_label)
    # add bindings to change the task (mousewheel and buttons)
    task_selector_label.bind(
        "<MouseWheel>",
        func = lambda event: self.change_selected_task(-event.delta, task_selector_label))
    task_selector_label.bind(
        "<Button-1>",
        func = lambda event: self.change_selected_task(-1, task_selector_label))
    task_selector_label.bind(
        "<Button-3>",
        func = lambda event: self.change_selected_task(1, task_selector_label))
    task_selector_left: tk.Button = self.add_arrow_button(
        direction="left",
        parent=task_selector_frame,
        command=lambda: self.change_selected_task(-1, task_selector_label),
        )
    task_selector_left.grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y),
        )
    task_selector_right: tk.Button = self.add_arrow_button(
        direction="right",
        parent=task_selector_frame,
        command=lambda: self.change_selected_task(1, task_selector_label),
        )
    task_selector_right.grid(
        row=1,
        column=2,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y),
        )
    # add buttons to export task cards
    export_buttons_frame: tk.Frame = tk.Frame(self.task_export_frame)
    self.add_frame_style(export_buttons_frame)
    export_buttons_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=(3 * self.grid_pad_y, self.grid_pad_y),
        )
    export_buttons_frame.grid_columnconfigure(0, weight=1)
    export_buttons_frame.grid_columnconfigure(1, weight=1)
    # add selector for the export directory
    export_filepath_selector_frame: tk.Frame = tk.Frame(export_buttons_frame)
    self.add_frame_style(export_filepath_selector_frame)
    export_filepath_selector_frame.grid(
        row=0,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=(0, self.grid_pad_y),
        )
    export_filepath_selector_frame.grid_columnconfigure(1, weight=1)
    export_filepath_label: tk.Label = tk.Label(export_filepath_selector_frame, text="Task card directory:", anchor="w")
    self.add_label_style(export_filepath_label)
    export_filepath_label.grid(
        row=0,
        column=0,
        sticky="w",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=0,
        )
    export_filepath_entry: tk.Entry = tk.Entry(export_filepath_selector_frame, textvariable=self.export_filepath, width=10)
    self.add_entry_style(export_filepath_entry)
    export_filepath_entry.xview_moveto(1)
    export_filepath_entry.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=0,
        )
    export_filepath_button: tk.Button = self.add_browse_button(
        frame=export_filepath_selector_frame,
        row_index=0,
        column_index=2,
        command=lambda: browse_directory("Select task card directory", self.export_filepath),
        )
    # add buttons to export the current or all task cards
    export_current_button: tk.Button = tk.Button(
        export_buttons_frame,
        text="Export current",
        command=lambda task=self.task_list[self.selected_task.get()]: self.export_task_card(task),
        )
    self.add_button_style(export_current_button)
    export_current_button.grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(0, self.grid_pad_y),
        )
    export_all_button: tk.Button = tk.Button(
        export_buttons_frame,
        text="Export all task cards",
        command=lambda: self.export_all_task_cards(task_selector_label),
        )
    self.add_button_style(export_all_button)
    export_all_button.grid(
        row=1,
        column=1,
        sticky="nsew",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y),
        )
    
    # set trace for frame path
    self.card_frame_filepath.trace("w", lambda *_: self.load_card_frame())
    # configure variables to update points images when the font size changes
    self.points_font_size.trace("w", lambda *_: self.update_points_image(points_type="standard"))
    self.bonus_font_size.trace("w", lambda *_: self.update_points_image(points_type="bonus"))
    self.penalty_font_size.trace("w", lambda *_: self.update_points_image(points_type="penalty"))
    # update points images
    for points_type in ["standard", "bonus", "penalty"]:
      self.update_points_image(points_type=points_type)



  def update_node_image_override(self) -> None:
    """
    Toggle whether nodes use the image specified in the node image override field or their own image.
    """
    self.show_current_task()

  def update_node_connector_lines(self) -> None:
    """
    Toggle the visibility of connection lines between all nodes included in the currently shown task.
    """
    self.show_current_task()

  def change_selected_task(self, direction: int, task_label: tk.Label) -> None:
    """
    Change the currently selected task by the given direction.
    Update the task label accordingly.
    Update the plot accordingly.

    Args:
        direction (int): the direction to change the task by. -1 for previous task, 1 for next task.
        task_label (tk.Label): the label displaying the currently selected task.
    """
    if direction > 0:
      self.selected_task.set((self.selected_task.get() + 1) % len(self.task_list))
    elif direction < 0:
      self.selected_task.set((self.selected_task.get() - 1) % len(self.task_list))
    task_label.config(text=f"{self.selected_task.get() + 1}. {self.task_list[self.selected_task.get()].name}")
    # update shown task
    self.show_current_task()
    # update points images
    for points_type in ["standard", "bonus", "penalty"]:
      self.update_points_image(points_type=points_type)

  def export_all_task_cards(self, task_label: tk.Label) -> None: # TODO: to be tested
    """
    Export all task cards to the directory given in `self.task_export_dir`.
    """
    for _ in range(len(self.task_list)):
      self.export_task_card(self.task_list[self.selected_task.get()])
      self.change_selected_task(1, task_label)
      self.task_export_frame.update()
      self.canvas.draw_idle()

  def export_task_card(self, task: TTR_Task) -> None:
    """
    Export the current task images to the directory given in `self.task_export_dir`.

    Exporting is done using the following steps:
    - hide all nodes, labels, edges and task indicators
    - show background image

    Args:
        task (TTR_Task): the task to export.
    """
    filepath: str = os.path.join(self.export_filepath.get(), f"{task.name}.png")
    self.fig.savefig(
        filepath,
        dpi=300,
        format="png",
        bbox_inches="tight",
        transparent=True)


  def get_current_settings(self) -> dict:
    """
    Get the current settings of the GUI.

    Returns:
        (dict): a dictionary containing all current settings.
    """
    return {
        "card_frame_filepath": self.card_frame_filepath.get(),
        "card_frame_width": self.card_frame_width.get(),
        "card_frame_height": self.card_frame_height.get(),
        "background_image_width": self.background_image_width.get(),
        "background_image_height": self.background_image_height.get(),
        "background_image_offset_x": self.background_image_offset_x.get(),
        "background_image_offset_y": self.background_image_offset_y.get(),

        "label_scale": self.label_scale.get(),
        "node_scale": self.node_scale.get(),
        "node_image_filepath": self.node_image_filepath.get(),
        "node_image_override": self.node_image_override.get(),
        "node_connector_lines": self.node_connector_lines.get(),

        "points_font_size": self.points_font_size.get(),
        "bonus_font_size": self.bonus_font_size.get(),
        "penalty_font_size": self.penalty_font_size.get(),

        "export_filepath": self.export_filepath.get(),
        "selected_task": self.selected_task.get(),
    }


  def show_current_task(self) -> None:
    """
    Show the currently selected task in the GUI by showing all nodes and corresponding labels belonging to the task.
    If self.node_connector_lines is True, also show the connection lines between the nodes.
    """
    if self.task_label is not None:
      self.task_label.erase()
    for task in self.task_list:
      task.erase()
    ttr_task = self.task_list[self.selected_task.get()]
    self.particle_graph.erase()
    if self.node_image_override.get():
      override_image: str = self.node_image_filepath.get() if self.node_image_filepath.get() else None
      if override_image is None:
        self.node_image_override.set(False)
    else:
      override_image: str = None
    # define variables to scale nodes to the background image
    old_background_offset: np.ndarray = np.array([
        self.particle_graph.graph_extent[0],
        self.particle_graph.graph_extent[2],
        ])
    new_background_offset: np.ndarray = np.array([
        self.background_image_offset_x.get(),
        self.background_image_offset_y.get(),
        ])
    node_override_positions: List[np.ndarray] = []
    # draw nodes and calculate their new positions
    for location in ttr_task.node_names:
      particle_node: Particle_Node = self.particle_graph.particle_nodes[location] 
      new_position: np.ndarray = (particle_node.position - old_background_offset) * self.graph_scale_factors + new_background_offset
      particle_node.draw(
          self.ax,
          scale=self.node_scale.get(),
          override_image_path=override_image,
          override_position=new_position,
          movable=False,
          )
      node_override_positions.append(new_position)
    # draw connection line(s) between nodes
    if self.node_connector_lines.get():
      ttr_task.draw(
          ax=self.ax,
          particle_graph=self.particle_graph,
          # color=self.task_connector_color,
          linewidth=self.node_connector_line_width.get(),
          zorder=1,
          override_positions=node_override_positions,
          )
    # create a particle_label to show the task name
    self.task_label = Particle_Label(
        label=ttr_task.name,
        id=-1,
        position=np.array([self.label_position_x.get(), self.label_position_y.get()]),
        rotation=0,
        height_scale_factor=self.particle_graph.label_height_scale,
        )
    self.task_label.draw(
        ax=self.ax,
        scale=self.label_scale.get(),
        movable=True)
    self.canvas.draw_idle()

  def load_card_frame(self) -> None:
    """
    Load the card frame image from the filepath set in `self.card_frame_filepath` unless it's empty.
    """
    # delete the old image if it exists
    if "frame" in self.plotted_images:
        self.plotted_images["frame"].remove()
        del self.plotted_images["frame"]
    filepath: str = get_tk_var(self.card_frame_filepath, default="")
    if filepath is "":
      if "frame" in self.plotted_images:
        self.plotted_images["frame"].remove()
        del self.plotted_images["frame"]
        self.canvas.draw_idle()
      return
    try:
      self.card_frame_image_mpl = mpimg.imread(filepath)
    except FileNotFoundError: # if the file is not found, erase the image
      self.card_frame_image_mpl = None
      if "frame" in self.plotted_images:
        self.plotted_images["frame"].remove()
        del self.plotted_images["frame"]
        self.canvas.draw_idle()
    self.update_frame_image()

  def update_points_image(self, points_type: str) -> None:
    """
    Update the points image to the one given in `self.points_image_directory`.

    Args:
        points_type (str): type of points to be shown. Can be "points", "standard", "bonus" or "penalty", where "points" and "standard" are equivalent.

    Raises:
        ValueError: if `points_type` is not one of "points", "standard", "bonus" or "penalty".
    """
    task: TTR_Task = self.task_list[self.selected_task.get()]
    if points_type == "points" or points_type == "standard":
        points: int = task.points
        filename: str = f"{points}.png"
        filepath: str = os.path.join(self.points_image_directory.get(), "points_standard", filename)
        points_scale: float = get_tk_var(self.points_font_size, 0)
        position: np.ndarray = np.array([get_tk_var(self.points_position_x, 0), get_tk_var(self.points_position_y, 0)])
    elif points_type == "bonus":
        if task.points_bonus == 0:
            return # don't show bonus points if there are none
        points: int = task.points_bonus
        filename: str = f"{points}.png"
        filepath: str = os.path.join(self.points_image_directory.get(), "points_bonus", filename)
        points_scale: float = get_tk_var(self.bonus_font_size, 0)
        position: np.ndarray = np.array([get_tk_var(self.bonus_position_x, 0), get_tk_var(self.bonus_position_y, 0)])
    elif points_type == "penalty":
        if task.points_penalty == 0:
            return # don't show penalty points if there are none
        points: int = task.points_penalty
        filename: str = f"{points}.png"
        filepath: str = os.path.join(self.points_image_directory.get(), "points_penalty", filename)
        points_scale: float = get_tk_var(self.penalty_font_size, 0)
        position: np.ndarray = np.array([get_tk_var(self.penalty_position_x, 0), get_tk_var(self.penalty_position_y, 0)])
    else:
        raise ValueError(f"Invalid points type: {points_type}. Expected one of 'points', 'standard', 'bonus' or 'penalty'.")
    # erase the points label if the scale is 0 or the points label already exists
    if points_type in self.points_labels:
        self.points_labels[points_type].erase()
        del self.points_labels[points_type]
    if points_scale == 0:
        return
    if points_type == "penalty" and task.points_penalty == -task.points:
        # don't show penalty points if they are the same as the standard points
        return
    points_image_mpl: np.ndarray = mpimg.imread(filepath) # shape: (height, width, 4)
    # create/ update a Particle_Node to show the points
    self.points_labels[points_type] = Particle_Node(
        location_name=str(points),
        id=-1,
        position=position,
        rotation=0,
        )
    # scale node to height 1
    bounding_box_size: Tuple[float, float] = (points_image_mpl.shape[1] / points_image_mpl.shape[0], 1)
    self.points_labels[points_type].bounding_box_size = bounding_box_size
    # draw the points label
    self.points_labels[points_type].draw(
        ax=self.ax,
        scale=points_scale,
        override_image_path=filepath,
        movable=True)


  def update_frame_image(self) -> None:
    """
    Update the frame image to the one given in `self.card_frame_filepath`.
    """
    self.card_frame_image_extent = np.array([
        0,
        self.card_frame_width.get(),
        0,
        self.card_frame_height.get(),
        ])
    self.plotted_images["frame"] = self.ax.imshow(
        self.card_frame_image_mpl,
        extent=self.card_frame_image_extent,
        zorder=1,
        )
    # update plot limits
    self.ax.set_xlim(self.card_frame_image_extent[0], self.card_frame_image_extent[1])
    self.ax.set_ylim(self.card_frame_image_extent[2], self.card_frame_image_extent[3])
    self.canvas.draw_idle()

  def update_background_image(self) -> None:
    """
    Resize the background image and canvas to ensure the background fits real lego pieces.
    new size will be the board size * scale_factor.
    """
    if "background" in self.plotted_images: # remove old background image
        self.plotted_images["background"].remove()
        del self.plotted_images["background"]
    # get size and offset inputs
    try:
      new_width: float = self.background_image_width.get() if self.background_image_width.get() > 0 \
        else self.card_background_image_extent[1] - self.card_background_image_extent[0]
    except tk.TclError: # non-numeric input
      new_width: float = self.card_background_image_extent[1] - self.card_background_image_extent[0]
    try:
      new_height: float = self.background_image_height.get() if self.background_image_height.get() > 0 \
        else self.card_background_image_extent[3] - self.card_background_image_extent[2]
    except tk.TclError: # non-numeric input
      new_height: float = self.card_background_image_extent[3] - self.card_background_image_extent[2]
    try:
      x_offset: float = self.background_image_offset_x.get()
    except tk.TclError: # non-numeric input
      x_offset: float = self.card_background_image_extent[0]
    try:
      y_offset: float = self.background_image_offset_y.get()
    except tk.TclError: # non-numeric input
      y_offset: float = self.card_background_image_extent[2]

    self.card_background_image_extent = np.array([
        x_offset,
        new_width + x_offset,
        y_offset,
        new_height + y_offset], dtype=np.float16)
    self.graph_scale_factors: np.ndarray = np.array([
        new_width / (self.particle_graph.graph_extent[1] - self.particle_graph.graph_extent[0]),
        new_height / (self.particle_graph.graph_extent[3] - self.particle_graph.graph_extent[2])])
    # update plot limits
    if "frame" in self.plotted_images:
      self.ax.set_xlim(self.card_frame_image_extent[0], self.card_frame_image_extent[1])
      self.ax.set_ylim(self.card_frame_image_extent[2], self.card_frame_image_extent[3])
    else:
      self.ax.set_xlim(self.card_background_image_extent[0], self.card_background_image_extent[1])
      self.ax.set_ylim(self.card_background_image_extent[2], self.card_background_image_extent[3])
    self.plotted_images["background"] = self.ax.imshow(self.background_image_mpl, extent=self.card_background_image_extent, zorder=0)

    # update the graph to fit on the new background image

    self.canvas.draw_idle()


  def add_arrow_button(self, direction: str, parent: tk.Frame, command: Callable) -> tk.Button:
    """
    add a button displaying an arrow in the given direction to the given parent frame and bind the command to it.
    This automatically applies the button style to the button.

    Args:
        direction (str): direction of the arrow (left, right, up, down)
        parent (tk.Frame): frame to add the button to
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
        parent,
        text=button_text,
        command=command)
    self.add_button_style(button)
    return button

def get_tk_var(tk_var, default=None):
    """
    Get the value of a tkinter variable. If the variable has an invalid value, return the default value.
    """
    try:
      return tk_var.get()
    except tk.TclError:
      return default