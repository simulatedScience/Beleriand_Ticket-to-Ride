"""
This module implements task card export functionality for the TTR board layout GUI.

This is done via a class Task_Export_GUI which can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.
"""
from typing import List, Tuple, Callable
import os
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from file_browsing import browse_image_file, browse_directory
from ttr_particle_graph import TTR_Particle_Graph
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
      canvas: FigureCanvasTkAgg,
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
        canvas (FigureCanvasTkAgg): The canvas for the graph.
    """
    self.master: tk.Tk = master
    self.color_config: dict[str, str] = color_config
    self.particle_graph: TTR_Particle_Graph = particle_graph
    self.task_export_frame: tk.Frame = task_export_frame
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
    self.card_frame_width: tk.DoubleVar = tk.DoubleVar(value=8.9)
    self.card_frame_height: tk.DoubleVar = tk.DoubleVar(value=6.4)
    self.background_image_width: tk.DoubleVar = tk.DoubleVar(value=8.5)
    self.background_image_height: tk.DoubleVar = tk.DoubleVar(value=6.2)
    self.background_image_offset_x: tk.DoubleVar = tk.StringVar(value=0.0)
    self.background_image_offset_y: tk.DoubleVar = tk.StringVar(value=0.0)

    self.label_scale: tk.DoubleVar = tk.DoubleVar(value=4.0)
    self.node_scale: tk.DoubleVar = tk.DoubleVar(value=2.0)
    self.node_image_filepath: tk.StringVar = tk.StringVar(value="")

    self.node_image_override: tk.BooleanVar = tk.BooleanVar(value=False)
    self.node_connector_lines: tk.BooleanVar = tk.BooleanVar(value=True)

    self.points_font_size: tk.DoubleVar = tk.DoubleVar(value=8.0)
    self.bonus_font_size: tk.DoubleVar = tk.DoubleVar(value=5.0)
    self.penalty_font_size: tk.DoubleVar = tk.DoubleVar(value=5.0)

    self.selected_task: tk.IntVar = tk.IntVar(value=0)
    self.task_list: List[TTR_Task] = list(self.particle_graph.tasks.values())

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
        sticky="nsw",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
    node_image_entry = tk.Entry(node_image_input_frame, textvariable=self.node_image_filepath, width=10)
    self.add_entry_style(node_image_entry)
    node_image_entry.xview_moveto(1)
    node_image_entry.grid(
        row=row_index,
        column=1,
        sticky="nsew",
        padx=(self.grid_pad_x, self.grid_pad_x),
        pady=(self.grid_pad_y, self.grid_pad_y))
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
    # add inputs for points font size and color
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
    points_font_size_label, points_font_size_entry = add_numeric_input(
        parent=points_inputs_frame,
        row_index=1,
        column_index=0,
        label_text="points:",
        variable=self.points_font_size,
        width=4,
        )
    bonus_font_size_label, bonus_font_size_entry = add_numeric_input(
        parent=points_inputs_frame,
        row_index=1,
        column_index=2,
        label_text="bonus:",
        variable=self.bonus_font_size,
        width=4,
        )
    penalty_font_size_label, penalty_font_size_entry = add_numeric_input(
        parent=points_inputs_frame,
        row_index=1,
        column_index=4,
        label_text="penalty:",
        variable=self.penalty_font_size,
        width=4,
        )
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
        command=lambda task=self.selected_task: self.export_task_card(task),
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
        command=self.export_all_task_cards,
        )
    self.add_button_style(export_all_button)
    export_all_button.grid(
        row=1,
        column=1,
        sticky="nsew",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y),
        )



  def update_node_image_override(self) -> None:
    """
    Toggle whether nodes use the image specified in the node image override field or their own image.
    """
    raise NotImplementedError

  def update_node_connector_lines(self) -> None:
    """
    Toggle the visibility of connection lines between all nodes included in the currently shown task.
    """
    self.show_current_task()

  def change_selected_task(self, direction, task_label) -> None:
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

  def export_all_task_cards(self) -> None:
    """
    Export all task cards to the directory given in `self.task_export_dir`.
    """
    raise NotImplementedError

  def export_task_card(self, task: TTR_Task) -> None:
    """
    Export the current task images to the directory given in `self.task_export_dir`.

    Exporting is done using the following steps:
    - hide all nodes, labels, edges and task indicators
    - show background image
    """
    raise NotImplementedError


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
    for task in self.task_list:
      task.erase()
    ttr_task = self.task_list[self.selected_task.get()]
    self.particle_graph.erase()
    for location in ttr_task.node_names:
      self.particle_graph.particle_nodes[location].draw(
          self.ax,
          # scale=self.node_scale.get(),
          )
      self.particle_graph.particle_labels[location].draw(
          self.ax,
          # scale=self.label_scale.get(),
          )
    if self.node_connector_lines.get():
      ttr_task.draw(
          ax=self.ax,
          particle_graph=self.particle_graph,
          # color=self.task_connector_color,
          # linewidth=self.task_connector_linewidth,
          )
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
