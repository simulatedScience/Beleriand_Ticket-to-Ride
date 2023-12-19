"""
This module implements task card export functionality for the TTR board layout GUI.

This is done via a class Task_Export_GUI which can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.
"""
from typing import List, Tuple, Callable
import os
import tkinter as tk
from tkinter import colorchooser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from file_browsing import browse_image_file, browse_directory
from cut_task_cards import remove_borders_from_file
# from _task_card_pdf_generation import task_cards_to_latex
from ttr_particle_graph import TTR_Particle_Graph
from particle_node import Particle_Node
from particle_label import Particle_Label
from ttr_task import TTR_Task

class Task_Export_GUI:
  def __init__(self,
      master: tk.Tk,
      color_config: dict[str, str],
      grid_padding: Tuple[int, int],
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
    self.card_frame_filepath: tk.StringVar = tk.StringVar(value=self.particle_graph.get_task_info("task_frame_image_path"))
    self.card_frame_width: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_card_size")[0])
    self.card_frame_height: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_card_size")[1])
    self.background_image_width: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_bg_image_size")[0])
    self.background_image_height: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_bg_image_size")[1])
    self.background_image_offset_x: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_bg_image_offset")[0])
    self.background_image_offset_y: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_bg_image_offset")[1])

    self.node_scale: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_node_scale"))
    self.node_image_filepath: tk.StringVar = tk.StringVar(value=self.particle_graph.get_task_info("task_node_override_image_path"))
    self.label_scale: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_label_scale"))
    self.label_position_x: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_label_position")[0])
    self.label_position_y: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_label_position")[1])

    self.node_image_override: tk.BooleanVar = tk.BooleanVar(value=self.particle_graph.get_task_info("task_node_override"))
    self.node_connection_lines: tk.BooleanVar = tk.BooleanVar(value=self.particle_graph.get_task_info("task_node_connection_lines"))
    self.node_connection_width: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_node_connection_line_width"))
    self.node_connection_color: tk.StringVar = tk.StringVar(value=self.particle_graph.get_task_info("task_node_connection_line_color"))
    self.node_connection_alpha: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_node_connection_line_alpha"))

    self.points_image_directory: tk.StringVar = tk.StringVar(value=self.particle_graph.get_task_info("task_points_directory"))
    self.points_font_scale: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_points_scale"))
    self.points_position_x: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_points_position")[0])
    self.points_position_y: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_points_position")[1])

    self.bonus_font_scale: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_bonus_points_scale"))
    self.bonus_position_x: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_bonus_points_position")[0])
    self.bonus_position_y: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_bonus_points_position")[1])

    self.penalty_font_scale: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_penalty_points_scale"))
    self.penalty_position_x: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_penalty_points_position")[0])
    self.penalty_position_y: tk.DoubleVar = tk.DoubleVar(value=self.particle_graph.get_task_info("task_penalty_points_position")[1])
    self.points_labels: dict[str, Particle_Node] = dict() # points are shown using particle nodes

    self.selected_task: tk.IntVar = tk.IntVar(value=0)
    self.task_list: List[TTR_Task] = list(self.particle_graph.tasks.values())
    self.task_label: Particle_Label = None

    if current_directory is None:
      current_directory = os.getcwd()
    self.current_directory: str = current_directory
    self.export_folderpath: tk.StringVar = tk.StringVar(value=self.particle_graph.get_task_info("task_card_folder_path"))

    # create widgets
    self.create_task_export_widgets()
  
  def save_settings(self) -> None:
    """
    Save all task export settings to the particle graph.
    """
    self.particle_graph.set_task_info(
        # task background image settings
        task_bg_image_offset=(self.background_image_offset_x.get(), self.background_image_offset_y.get()),
        task_bg_image_size=(self.background_image_width.get(), self.background_image_height.get()),
        task_card_size=(self.card_frame_width.get(), self.card_frame_height.get()),
        task_frame_image_path=self.card_frame_filepath.get(),
        # task label settings
        task_label_position=(self.label_position_x.get(), self.label_position_y.get()),
        task_label_scale=self.label_scale.get(),
        # task card node settings (locations)
        task_node_scale=self.node_scale.get(),
        task_node_override=self.node_image_override.get(),
        task_node_override_image_path=self.node_image_filepath.get(),
        task_node_connection_lines=self.node_connection_lines.get(),
        task_node_connection_line_width=self.node_connection_width.get(),
        task_node_connection_line_color=self.node_connection_color.get(),
        task_node_connection_line_alpha=self.node_connection_alpha.get(),
        # task card points settings
        task_points_scale=self.points_font_scale.get(),
        task_points_position=(self.points_position_x.get(), self.points_position_y.get()),
        task_bonus_points_scale=self.bonus_font_scale.get(),
        task_bonus_points_position=(self.bonus_position_x.get(), self.bonus_position_y.get()),
        task_penalty_points_scale=self.penalty_font_scale.get(),
        task_penalty_points_position=(self.penalty_position_x.get(), self.penalty_position_y.get()),
        task_points_directory=self.points_image_directory.get(),
        # card folder settings
        task_card_folder_path=self.export_folderpath.get(),
        )

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
        sticky_label: str = "w",
        sticky_entry: str = "w",
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
          sticky=sticky_label,
          padx=(self.grid_pad_x, self.grid_pad_x),
          pady=(0, self.grid_pad_y),
          )
      # create entry
      entry = tk.Entry(parent, textvariable=variable, width=width)
      self.add_entry_style(entry, justify="right")
      entry.grid(
          row=row_index,
          column=column_index + 1,
          sticky=sticky_entry,
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
        variable=self.node_connection_lines,
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
    # add node connection line settings
    node_connection_width_label, node_connection_width_entry = add_numeric_input(
        parent=graph_settings_frame,
        row_index=row_index,
        column_index=0,
        label_text="Line width:",
        variable=self.node_connection_width,
        width=4,
        )
    node_connection_color_label: tk.Label = tk.Label(graph_settings_frame, text="Line color:", anchor="w")
    self.add_label_style(node_connection_color_label)
    node_connection_color_label.grid(
        row=row_index,
        column=2,
        sticky="ew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    # add color picker for node connection lines
    node_connection_color_picker: tk.Button = tk.Button(
        graph_settings_frame,
        text="",
        width=5,
        )
    self.add_button_style(node_connection_color_picker)
    node_connection_color_picker.config(
        bg=self.node_connection_color.get(),
        command=lambda: self.change_node_connector_color(
            self.node_connection_color,
            node_connection_color_picker,),
    )
    node_connection_color_picker.grid(
        row=row_index,
        column=3,
        sticky="w",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    row_index += 1
    # add alpha slider for node connection lines
    node_connection_alpha_label: tk.Label = tk.Label(graph_settings_frame, text="Line alpha:", anchor="w")
    self.add_label_style(node_connection_alpha_label)
    node_connection_alpha_label.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    # add label for alpha slider
    node_connection_alpha_value_label: tk.Label = tk.Label(
        graph_settings_frame,
        textvariable=self.node_connection_alpha,
        anchor="w")
    self.add_label_style(node_connection_alpha_value_label)
    node_connection_alpha_value_label.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y),
        )
    node_connection_alpha_slider: tk.Scale = tk.Scale(
        graph_settings_frame,
        bg=self.color_config["button_bg_color"],
        # fg irrelevant
        troughcolor=self.color_config["entry_bg_color"],
        highlightthickness=0,
        activebackground=self.color_config["button_fg_color"],
        showvalue=False,
        sliderlength=20,
        sliderrelief="flat",
        cursor="hand2",
        borderwidth=0,
        from_=0,
        to=1,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=self.node_connection_alpha,
        command=lambda _: self.update_node_connector_lines(),
        )
    node_connection_alpha_slider.grid(
        row=row_index,
        column=2,
        columnspan=2,
        sticky="new",
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
        variable=self.points_font_scale,
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
        variable=self.bonus_font_scale,
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
        variable=self.penalty_font_scale,
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
    # add input for points image directory
    points_image_directory_frame: tk.Frame = tk.Frame(self.task_export_frame)
    self.add_frame_style(points_image_directory_frame)
    points_image_directory_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=0,
        pady=(3 * self.grid_pad_y, self.grid_pad_y),
        )
    points_image_directory_frame.grid_columnconfigure(1, weight=1)
    points_image_directory_label: tk.Label = tk.Label(points_image_directory_frame, text="Points image directory:", anchor="w")
    self.add_label_style(points_image_directory_label)
    points_image_directory_label.grid(
        row=0,
        column=0,
        sticky="w",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=0,
        )
    points_image_directory_entry: tk.Entry = tk.Entry(points_image_directory_frame, textvariable=self.points_image_directory, width=10)
    self.add_entry_style(points_image_directory_entry)
    points_image_directory_entry.xview_moveto(1)
    points_image_directory_entry.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=0,
        )
    points_image_directory_button: tk.Button = self.add_browse_button(
        frame=points_image_directory_frame,
        row_index=0,
        column_index=2,
        command=lambda: browse_directory("Select points image directory", self.points_image_directory),
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
    export_filepath_entry: tk.Entry = tk.Entry(export_filepath_selector_frame, textvariable=self.export_folderpath, width=10)
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
        command=lambda: browse_directory("Select task card directory", self.export_folderpath),
        )
    # add buttons to export the current or all task cards
    def export_current_task_card():
        task = self.task_list[self.selected_task.get()]
        self.export_task_card(task)
    export_current_button: tk.Button = tk.Button(
        export_buttons_frame,
        text="Export current",
        command=export_current_task_card,
        # command=lambda task=self.task_list[self.selected_task.get()]: self.export_task_card(task),
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
    self.points_font_scale.trace("w", lambda *_: self.update_points_image(points_type="standard"))
    self.bonus_font_scale.trace("w", lambda *_: self.update_points_image(points_type="bonus"))
    self.penalty_font_scale.trace("w", lambda *_: self.update_points_image(points_type="penalty"))
    # update points images
    for points_type in ["standard", "bonus", "penalty"]:
      self.update_points_image(points_type=points_type)
    
    self.update_background_image()

  def change_node_connector_color(self,
        color_var: tk.StringVar,
        button: tk.Button,
        ) -> None:
    """
    Change the color of the node connection lines.

    Args:
        color_var (tk.StringVar): the variable containing the color.
    """
    color: str = colorchooser.askcolor(color=color_var.get())[1]
    if color is not None:
      print(f"Changing node connection line color to {color}")
      color_var.set(color)
      # update button
      button.config(bg=color)
      # update plot
      self.update_node_connector_lines()
      # update particle graph
      self.particle_graph.set_task_info(
        task_node_connection_line_width=self.node_connection_width.get(),
        task_node_connection_line_color=color,
      )

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
    print("Exported all tasks!")

  def export_task_card(self, task: TTR_Task) -> None:
    """
    Export the current task images to the directory given in `self.task_export_dir`.

    Exporting is done using the following steps:
    - hide all nodes, labels, edges and task indicators
    - show background image

    Args:
        task (TTR_Task): the task to export.
    """
    print(f"Exporting task {task.name} to {task.name}.png", end = "\n...")
    filepath: str = os.path.join(self.export_folderpath.get(), f"{task.name}.png")
    # save export filepath to particle graph
    self.particle_graph.set_task_info(
        task_card_folder_path=filepath,
    )
    self.fig.savefig(
        filepath,
        dpi=150,
        format="png",
        bbox_inches="tight",
        transparent=True)
    print("\rtrimming transparent edges...", end="")
    remove_borders_from_file(
        input_path=filepath,
        output_path=filepath)
    print("\rDone!                        ")


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
        "node_connector_lines": self.node_connection_lines.get(),

        "points_font_size": self.points_font_scale.get(),
        "bonus_font_size": self.bonus_font_scale.get(),
        "penalty_font_size": self.penalty_font_scale.get(),

        "export_filepath": self.export_folderpath.get(),
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
    # prepare node override image
    if self.node_image_override.get():
      override_image: str = self.node_image_filepath.get() if self.node_image_filepath.get() else None
      if override_image is None:
        self.node_image_override.set(False)
      # save node ovverride info to particle graph
      self.particle_graph.set_task_info(
        task_node_override_image_path=override_image,
        task_node_override=self.node_image_override.get(),
      )
    else:
      override_image: str = None
      self.particle_graph.set_task_info(
        task_node_override=self.node_image_override.get(),
      )
    # define variables to scale nodes to the background image
    old_background_offset: np.ndarray = np.array(
        self.particle_graph.get_bg_info()["bg_image_offset"]
    )
    new_background_offset: np.ndarray = np.array([
        self.background_image_offset_x.get(),
        self.background_image_offset_y.get(),
        ])
    # save task bg image offset to particle graph
    self.particle_graph.set_bg_info(
        task_bg_image_offset=tuple(new_background_offset),
    )
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
          zorder=4,
          )
      node_override_positions.append(new_position)
    # draw connection line(s) between nodes
    if self.node_connection_lines.get():
      ttr_task.draw(
          ax=self.ax,
          particle_graph=self.particle_graph,
          color=self.node_connection_color.get(),
          linewidth=self.node_connection_width.get(),
          alpha=self.node_connection_alpha.get(),
          zorder=1,
          override_positions=node_override_positions,
          )
    # save node connection line info to particle graph
    self.particle_graph.set_task_info(
        task_node_connection_lines=self.node_connection_lines.get(),
        task_node_connection_line_width=self.node_connection_width.get(),
        task_node_connection_line_color=self.node_connection_color.get(),
        task_node_connection_line_alpha=self.node_connection_alpha.get(),
        )
    # create a particle_label to show the task name
    self.task_label = Particle_Label(
        label=ttr_task.name,
        id=-1,
        position=np.array([self.label_position_x.get(), self.label_position_y.get()]),
        rotation=0,
        height_scale_factor=self.particle_graph.label_height_scale,
        ignore_linebreaks=True,
        font_path=self.particle_graph.get_misc_info("label_font"),
        )
    # save task label info to particle graph
    self.particle_graph.set_task_info(
        task_label_position=(self.label_position_x.get(), self.label_position_y.get()),
        task_label_scale=self.label_scale.get(),
        task_card_folder_path=self.export_folderpath.get(),
        )
    self.task_label.draw(
        color="#eeeeee",
        # border_color="#629bb4",
        border_color=(1, 0, 1, 0),
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
    if filepath == "":
      if "frame" in self.plotted_images:
        self.plotted_images["frame"].remove()
        del self.plotted_images["frame"]
        self.canvas.draw_idle()
      return
    try:
      self.card_frame_image_mpl = mpimg.imread(filepath)
      # save card frame path to particle graph
      self.particle_graph.set_task_info(
        task_frame_image_path=filepath)
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
    self.particle_graph.set_task_info(
        task_points_directory=self.points_image_directory.get(),
    )
    if points_type == "points" or points_type == "standard":
        points: int = task.points
        filename: str = f"{points}.png"
        filepath: str = os.path.join(self.points_image_directory.get(), "points_standard", filename)
        points_scale: float = get_tk_var(self.points_font_scale, 0)
        position: np.ndarray = np.array([get_tk_var(self.points_position_x, 0), get_tk_var(self.points_position_y, 0)])
    elif points_type == "bonus":
        if task.points_bonus == 0:
            return # don't show bonus points if there are none
        points: int = task.points_bonus
        filename: str = f"{points}.png"
        filepath: str = os.path.join(self.points_image_directory.get(), "points_bonus", filename)
        points_scale: float = get_tk_var(self.bonus_font_scale, 0)
        position: np.ndarray = np.array([get_tk_var(self.bonus_position_x, 0), get_tk_var(self.bonus_position_y, 0)])
    elif points_type == "penalty":
        if task.points_penalty == 0:
            return # don't show penalty points if there are none
        points: int = task.points_penalty
        filename: str = f"{points}.png"
        filepath: str = os.path.join(self.points_image_directory.get(), "points_penalty", filename)
        points_scale: float = get_tk_var(self.penalty_font_scale, 0)
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
    # save current points settings to the particle graph
    points_type_to_param: dict[str, str] = {
      "points": "task_points",
      "standard": "task_points",
      "bonus": "task_bonus_points",
      "penalty": "task_penalty_points",
    }
    prefix: str = points_type_to_param[points_type]
    self.particle_graph.set_task_info(**{
      prefix + "_position": tuple(position),
      prefix + "_scale": points_scale,
    })
    # draw the points label
    self.points_labels[points_type].draw(
        ax=self.ax,
        scale=points_scale,
        override_image_path=filepath,
        zorder=1,
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
        zorder=2,
        )
    # save card frame size to particle graph
    self.particle_graph.set_task_info(
      task_frame_image_path=self.card_frame_filepath.get(),
      task_card_size=(self.card_frame_width.get(), self.card_frame_height.get())
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
    except tk.TclError: # non-numeric input => keep previous value
      new_width: float = self.card_background_image_extent[1] - self.card_background_image_extent[0]
    try:
      new_height: float = self.background_image_height.get() if self.background_image_height.get() > 0 \
        else self.card_background_image_extent[3] - self.card_background_image_extent[2]
    except tk.TclError: # non-numeric input => keep previous value
      new_height: float = self.card_background_image_extent[3] - self.card_background_image_extent[2]
    self.particle_graph.set_task_info(task_bg_image_size=(new_width, new_height))
    try:
      x_offset: float = self.background_image_offset_x.get()
    except tk.TclError: # non-numeric input => keep previous value
      x_offset: float = self.card_background_image_extent[0]
    try:
      y_offset: float = self.background_image_offset_y.get()
    except tk.TclError: # non-numeric input => keep previous value
      y_offset: float = self.card_background_image_extent[2]
    self.particle_graph.set_task_info(task_bg_image_offset=(x_offset, y_offset))

    self.card_background_image_extent = np.array([
        x_offset,
        new_width + x_offset,
        y_offset,
        new_height + y_offset], dtype=np.float16)
    # get original bg image size
    bg_info: dict = self.particle_graph.get_bg_info()
    board_width, board_height = bg_info["bg_image_size"]
    self.graph_scale_factors: np.ndarray = np.array([
        new_width / board_width,
        new_height / board_height,
        ])
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