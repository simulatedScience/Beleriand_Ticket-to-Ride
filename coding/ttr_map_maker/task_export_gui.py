"""
This module implements task card export functionality for the TTR board layout GUI.

This is done via a class Task_Export_GUI which can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.
"""
from typing import List, Tuple, Callable
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from file_browsing import browse_image_file
from ttr_particle_graph import TTR_Particle_Graph

class Task_Export_GUI:
  def __init__(self,
      master: tk.Tk,
      color_config: dict[str, str],
      grid_padding: Tuple[float, float],
      tk_config_methods: dict[str, Callable],
      particle_graph: TTR_Particle_Graph,
      task_settings_frame: tk.Frame,
      ax: plt.Axes,
      canvas: FigureCanvasTkAgg,
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
        canvas (FigureCanvasTkAgg): The canvas for the graph.
    """
    self.master: tk.Tk = master
    self.color_config: dict[str, str] = color_config
    self.particle_graph: TTR_Particle_Graph = particle_graph
    self.task_settings_frame: tk.Frame = task_settings_frame
    self.ax: plt.Axes = ax
    self.canvas: FigureCanvasTkAgg = canvas

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
    self.card_frame_width: tk.DoubleVar = tk.StringVar(value=8.9)
    self.card_frame_height: tk.DoubleVar = tk.StringVar(value=6.4)

    # create widgets
    self.create_task_export_widgets()
  

  def create_task_export_widgets(self) -> None:
    """
    Creates the widgets for the task export GUI and place them in `self.task_settings_frame`
    """
    row_index: int = 0
    # create task export headline
    task_export_headline = tk.Label(self.task_settings_frame, text="Task export settings"),
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
    border_settings_frame: tk.Frame = tk.Frame(self.task_settings_frame)
    self.add_frame_style(border_settings_frame)
    border_settings_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        padx=self.grid_pad_x,
        pady=self.grid_pad_y,
        sticky="new",
        )
    # frame image file input
    label = tk.Label(border_settings_frame, text="Card frame image:")
    self.add_label_style(label)
    label.grid(
        row=row_index,
        column=0,
        sticky="nsw",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    entry = tk.Entry(border_settings_frame, textvariable=self.card_frame_filepath, width=10)
    self.add_entry_style(entry)
    entry.xview_moveto(1)
    entry.grid(
        row=row_index,
        column=1,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    browse_button = self.add_browse_button(
        frame=border_settings_frame,
        row_index=row_index,
        column_index=2,
        command=lambda: browse_image_file("Select a task card frame image.", self.card_frame_filepath),
        )
