"""
This module implements task edit functionality for the TTR board layout GUI.

This is done via a class Task_Editor_Gui which can be seen as an extension of `Board_Layout_GUI`, which is the intended way to use it.

"""
import tkinter as tk
from typing import Tuple, List, Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import PickEvent, MouseEvent

from auto_scroll_frame import Auto_Scroll_Frame
from ttr_particle_graph import TTR_Particle_Graph
from ttr_task import TTR_Task
from particle_node import Particle_Node


class Task_Editor_GUI:
  def __init__(self,
      master: tk.Tk,
      color_config: dict[str, str],
      grid_padding: Tuple[float, float],
      tk_config_methods: dict[str, Callable],
      particle_graph: TTR_Particle_Graph,
      task_edit_frame: tk.Frame,
      ax: plt.Axes,
      canvas: FigureCanvasTkAgg,
      max_pick_range: float = 0.2
      ):
    """
    This class implements task edit functionality for the TTR board layout GUI. Changes are made to the given particle graph `particle_graph` and the graph is displayed in the given matplotlib Axes object `ax` and tkinter canvas `canvas`.

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
        task_edit_frame (tk.Frame): The frame containing the task edit GUI.
        ax (plt.Axes): The axes where the graph is drawn.
        canvas (FigureCanvasTkAgg): The canvas for the graph.
        max_pick_range (float, optional): The maximum range for picking a node. Defaults to 0.2.
    """
    self.master: tk.Tk = master
    self.color_config: dict[str, str] = color_config
    self.particle_graph: TTR_Particle_Graph = particle_graph
    self.task_edit_frame: tk.Frame = task_edit_frame
    self.ax: plt.Axes = ax
    self.canvas: FigureCanvasTkAgg = canvas
    self.max_pick_range: float = max_pick_range

    # save grid padding for later use
    self.grid_pad_x = grid_padding[0]
    self.grid_pad_y = grid_padding[1]

    # extract tkinter style methods
    self.add_frame_style = tk_config_methods["add_frame_style"]
    self.add_label_style = tk_config_methods["add_label_style"]
    self.add_button_style = tk_config_methods["add_button_style"]
    self.add_entry_style = tk_config_methods["add_entry_style"]
    self.add_checkbutton_style = tk_config_methods["add_checkbutton_style"]
    self.add_radiobutton_style = tk_config_methods["add_radiobutton_style"]
    self.add_browse_button = tk_config_methods["add_browse_button"]

    # set up task edit variables
    self.node_names: List[str] = ["None"] + sorted(self.particle_graph.get_locations())
    self.highlighted_particles: List[Particle_Node] = []

    self.init_task_edit_gui()


  def init_task_edit_gui(self):
    """
    This method initializes the task edit GUI with all its widgets.
    """
    # set up task edit variables
    self.task_visibility_vars: dict[str, tk.BooleanVar] = {}

    # set up task edit frame
    self.open_task_overview()


  def clear_task_edit_frame(self):
    """
    Clears the task edit frame.
    """
    for widget in self.task_edit_frame.winfo_children():
      widget.destroy()


  def open_task_overview(self):
    """
    Draws an overview of all tasks in the task edit frame. This includes:
      - a headline
      - a button to calculate all task lengths
      - a checkbutton to toggle visibility of all tasks
      - a button to add a new task, and
      - a list of all tasks.
    """
    # clear task edit frame
    self.clear_task_edit_frame()

    row_index: int = 0
    # create task edit headline
    headline_label = tk.Label(self.task_edit_frame, text="Task Overview")
    self.add_label_style(headline_label, font_type="bold")
    headline_label.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    row_index += 1
    # add button to calculate all task lengths
    calc_task_lengths_button = tk.Button(
        self.task_edit_frame,
        text="Calculate Task Lengths",
        command=self.calculate_all_task_lengths)
    self.add_button_style(calc_task_lengths_button)
    calc_task_lengths_button.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    row_index += 1
    # add checkbutton to toggle visibility of all tasks
    all_task_visibility_var: tk.BooleanVar = tk.BooleanVar()
    self.task_visibility_vars["all"] = all_task_visibility_var
    toggle_task_visibility_button = tk.Checkbutton(
        self.task_edit_frame,
        text="Show/hide all",
        variable=all_task_visibility_var,
        command=self.toggle_all_tasks_visibility)
    self.add_checkbutton_style(toggle_task_visibility_button)
    toggle_task_visibility_button.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    # add button to add new task
    add_task_button = tk.Button(
        self.task_edit_frame,
        text="Add Task",
        command=self.add_task)
    self.add_button_style(add_task_button)
    add_task_button.grid(
        row=row_index,
        column=1,
        sticky="e",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    row_index += 1
    
    # add task list
    task_list_outer_frame = tk.Frame(
        self.task_edit_frame,
        background=self.color_config["frame_bg_color"],
        height=250,
        width=self.task_edit_frame.winfo_width())
    task_list_outer_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=0,#self.grid_pad_x,
        pady=0)
    task_list_outer_frame.grid_rowconfigure(0, weight=1)
    task_list_auto_frame = Auto_Scroll_Frame(
        task_list_outer_frame,
        canvas_kwargs=dict(background=self.color_config["bg_color"]),
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
    task_list_auto_frame.scrollframe.grid_columnconfigure(0, weight=1)
    task_list_frame: tk.Frame = task_list_auto_frame.scrollframe
    row_index += 1
    # add tasks to subframe
    task_row_index: int = 0
    self.task_list_widgets: List[Tuple[tk.Frame, tk.Label, tk.Checkbutton, tk.Label, tk.Button, tk.Button]] = []
    self.task_list: List[TTR_Task] = []
    for ttr_task in self.particle_graph.tasks.values():
      task_widgets = self._show_single_task_summary(ttr_task, task_list_frame, task_row_index)
      self.task_list_widgets.append(task_widgets)
      self.task_list.append(ttr_task)
      task_row_index += 1
    task_list_auto_frame._on_configure()

  def _show_single_task_summary(self,
      ttr_task: TTR_Task,
      task_list_frame: tk.Frame,
      task_row_index: int) -> Tuple[tk.Frame, tk.Label, tk.Checkbutton, tk.Label, tk.Button, tk.Button]:
    """
    Adds the widgets for a single task to the task list frame.

    Args:
        ttr_task (TTR_Task): The task to add the widgets for.
        task_list_frame (tk.Frame): The frame to add the widgets to.
        task_row_index (int): The row index to add the widgets to.

    Returns:
        tk.Frame: The frame containing the widgets for the task.
    """
    task_var: tk.BooleanVar = tk.BooleanVar(value=False)
    task_name = ttr_task.name
    self.task_visibility_vars[task_name]: tk.BooleanVar = task_var
    task_frame = tk.Frame(task_list_frame)
    self.add_frame_style(task_frame)
    task_frame.grid(
        row=task_row_index,
        column=0,
        sticky="nsew",
        padx=0,
        pady=(self.grid_pad_y, 0))
    task_frame.grid_columnconfigure(1, weight=1)
    # add task number
    task_number_label = tk.Label(task_frame, text=f"{task_row_index+1}.")
    self.add_label_style(task_number_label, font_type="bold")
    task_number_label.grid(
        row=0,
        column=0,
        rowspan=2,
        sticky="w",
        padx=(self.grid_pad_x, 0),
        pady=0)
    # task_row_index += 1
    # add checkbutton to toggle visibility of task
    if " - " in task_name:
      task_name = "\n".join(task_name.split(" - "))
    task_visibility_button = tk.Checkbutton(
        task_frame,
        text=task_name,
        justify="left",
        variable=task_var,
        command=lambda task_var=task_var, task=ttr_task: self.toggle_task_visibility(task_var, task))
    self.add_checkbutton_style(task_visibility_button)
    task_visibility_button.grid(
        row=0,
        column=1,
        rowspan=2,
        sticky="w",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    # add label to show task length. If task has bonus points, show them as well
    task_length_text = f"length: {ttr_task.points}"
    if ttr_task.points_bonus is not None: # add task bonus points
      task_length_text += " + " + str(ttr_task.points_bonus)
    task_length_label = tk.Label(
        task_frame,
        text=task_length_text)
    self.add_label_style(task_length_label)
    task_length_label.grid(
        row=0,
        column=2,
        columnspan=2,
        sticky="nw",
        padx=(0, self.grid_pad_x),
        pady=(self.grid_pad_y,0))
    # add button to edit task
    edit_task_button = tk.Button(
        task_frame,
        text="Edit",
        command=lambda task=ttr_task: self.edit_task(task))
    self.add_button_style(edit_task_button)
    edit_task_button.grid(
        row=1,
        column=2,
        sticky="ne",
        padx=0, # (0, self.grid_pad_x),
        pady=0)
    # add button to delete task
    delete_task_button = tk.Button(
        task_frame,
        text="Delete",
        command=lambda task=ttr_task, task_index=task_row_index: self.delete_task(task, task_index))
    self.add_button_style(delete_task_button)
    delete_task_button.config(
        bg=self.color_config["delete_button_bg_color"],
        fg=self.color_config["delete_button_fg_color"])
    delete_task_button.grid(
        row=1,
        column=3,
        sticky="nw",
        padx=0,
        pady=0)

    return (task_frame, task_number_label, task_visibility_button, task_length_label, edit_task_button, delete_task_button)


  def calculate_all_task_lengths(self):
    """
    Calculates the length of all tasks in the task edit frame.
    """
    raise NotImplementedError() # TODO: implement


  def add_task(self):
    """
    Adds a new task to the task edit frame.
    """
    new_task: TTR_Task = TTR_Task(node_names=[]) # create new, empty task
    self.edit_task(new_task) # open edit mode for new task


  def toggle_all_tasks_visibility(self):
    """
    Toggles the visibility of all tasks.
    """
    for task_name, task_var in self.task_visibility_vars.items():
      if task_name != "all":
        task_var.set(self.task_visibility_vars["all"].get())
        self.toggle_task_visibility(task_var, self.particle_graph.tasks[task_name], update_canvas=False, update_all_tasks=False)
    self.canvas.draw_idle()

  def toggle_task_visibility(self, task_visibility_var: tk.BooleanVar, task: TTR_Task, update_canvas: bool = True, update_all_tasks: bool = True):
    """
    Toggles the visibility of the given task.
    If all task visibility variables were True and this one is now False, uncheck the "Show/hide all" checkbutton.
    If all task visibility variables are now True, check the "Show/hide all" checkbutton.
    """
    if task_visibility_var.get():
      task.erase()
      task.draw(self.ax, self.particle_graph)
    else:
      task.erase()
    # check if all tasks are now visible
    if update_all_tasks:
      all_tasks_visible = True
      for task_name, task_var in self.task_visibility_vars.items():
        if task_name != "all" and not task_var.get():
          all_tasks_visible = False
          self.task_visibility_vars["all"].set(False)
          break
      else:
        self.task_visibility_vars["all"].set(True)
    if update_canvas:
      self.canvas.draw_idle()


  def edit_task(self, task: TTR_Task):
    """
    Open edit mode for the given task.

    1. Clear the task edit frame
    2. Create headline for edit mode
    3. Show task location widgets
    4. Show task length/ points widgets
    5. Show buttons to save changes or abort editing

    Args:
        task (TTR_Task): The task to edit.
    """
    self.task_node_indices: List[int] = [] # indices of the nodes in the task
    self.task_location_widgets: List[Tuple[tk.Label, tk.Frame, tk.Button, tk.Label, tk.Button, tk.Button]] = []
    task_points_vars: List[tk.IntVar] = [] # variables for task length and points
    # 1. Clear the task edit frame
    self.clear_task_edit_frame()
    row_index: int = 0
    # 2. Create headline for edit mode
    edit_mode_headline = tk.Label(
        self.task_edit_frame,
        text="Edit Task")
    self.add_label_style(edit_mode_headline, font_type="bold")
    edit_mode_headline.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=(self.grid_pad_y, 0))
    row_index += 1
    # 3. Show task location widgets
    # 3.1. Show task location headline
    task_location_headline = tk.Label(
        self.task_edit_frame,
        text="Task Locations")
    self.add_label_style(task_location_headline)
    task_location_headline.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=(self.grid_pad_y, 0))
    row_index += 1
    # 3.2. Show task location widgets
    self.task_locations_frame: tk.Frame = tk.Frame(self.task_edit_frame)
    self.add_frame_style(self.task_locations_frame)
    self.task_locations_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=0)
    row_index += 1
    location_index: int = -1
    for location_index, location in enumerate(task.node_names):
      self.add_task_location(location_index, location)

    self.canvas.draw_idle()
    # 3.3. Show button to add new task location
    self.add_task_location_button = tk.Button(
        self.task_edit_frame,
        text="Add Location",
        command=lambda task_index=location_index+1: self.add_task_location(
            task_index,
            location_name="None",
            update_add_location_button=True))
    self.add_button_style(self.add_task_location_button)
    self.add_task_location_button.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y))
    row_index += 1
    # 4. Show task length/ points widgets
    # 4.1. Create task points headline
    task_length_headline = tk.Label(
        self.task_edit_frame,
        text="Task points")
    self.add_label_style(task_length_headline)
    task_length_headline.grid(
        row=row_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=(self.grid_pad_y, 0))
    # 4.2. Create button to calculate task length
    calculate_task_length_button = tk.Button(
        self.task_edit_frame,
        text="Calculate task length",
        command=lambda task=task, task_points_vars=task_points_vars: self.calculate_task_length(task, task_points_vars))
    self.add_button_style(calculate_task_length_button)
    calculate_task_length_button.grid(
        row=row_index,
        column=1,
        sticky="e",
        padx=(0, self.grid_pad_x),
        pady=(self.grid_pad_y, 0))
    row_index += 1
    # 4.3. Create task points 
    task_points_frame: tk.Frame = tk.Frame(self.task_edit_frame)
    self.add_frame_style(task_points_frame)
    task_points_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=self.grid_pad_x,
        pady=0)
    # task_points_frame.grid_columnconfigure(1, weight=1)
    # task_points_frame.grid_columnconfigure(3, weight=1)
    row_index += 1
    # 4.3.1. Create label for task length
    task_length_var: tk.IntVar = tk.IntVar(value=task.length)
    task_length_label = tk.Label(
        task_points_frame,
        text="length")
    self.add_label_style(task_length_label)
    task_length_label.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    task_length_indicator = tk.Label(
        task_points_frame,
        textvariable=task_length_var)
    self.add_label_style(task_length_indicator, font_type="bold italic")
    task_length_indicator.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=0,
        pady=0)
    task_points_vars.append(task_length_var)
    # 4.3.2. Create label for task points
    task_points_var: tk.IntVar = tk.IntVar(value=task.points)
    self.add_int_input(
        task_points_frame,
        row_index=0,
        column_index=2,
        label_text="points",
        int_var=task_points_var,
        input_width=3,
        input_justify="center",
    )
    task_points_vars.append(task_points_var)
    # 4.3.3. Create label for task bonus points
    task_bonus_points_var: tk.IntVar = tk.IntVar(value=task.points_bonus)
    self.add_int_input(
        task_points_frame,
        row_index=1,
        column_index=0,
        label_text="bonus",
        int_var=task_bonus_points_var,
        input_width=3,
        input_justify="center",
    )
    task_points_vars.append(task_bonus_points_var)
    # 4.3.4. Create label for task penalty points
    task_penalty_points_var: tk.IntVar = tk.IntVar(value=task.points_penalty)
    self.add_int_input(
        task_points_frame,
        row_index=1,
        column_index=2,
        label_text="penalty",
        int_var=task_penalty_points_var,
        input_width=3,
        input_justify="center",
    )
    task_points_vars.append(task_penalty_points_var)
    # 5. Show task edit buttons
    # 5.1 Create task edit buttons frame
    task_edit_buttons_frame: tk.Frame = tk.Frame(self.task_edit_frame)
    self.add_frame_style(task_edit_buttons_frame)
    task_edit_buttons_frame.grid(
        row=row_index,
        column=0,
        columnspan=2,
        sticky="new",
        padx=self.grid_pad_x,
        pady=self.grid_pad_y)
    task_edit_buttons_frame.grid_columnconfigure(0, weight=1)
    task_edit_buttons_frame.grid_columnconfigure(1, weight=1)
    row_index += 1
    # 5.2. Create button to apply changes
    apply_name = "Create" if task.is_empty() else "Apply"
    apply_changes_button = tk.Button(
        task_edit_buttons_frame,
        text=apply_name,
        command=lambda task=task, task_points_vars=task_points_vars, task_node_indices=self.task_node_indices: self.apply_task_changes(task, task_points_vars, task_node_indices))
    self.add_button_style(apply_changes_button)
    apply_changes_button.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=self.grid_pad_x,
        pady=0)
    # 5.3. Create button to cancel changes
    cancel_changes_button = tk.Button(
        task_edit_buttons_frame,
        text="Abort",
        command=lambda task_points_vars=task_points_vars, task_node_indices=self.task_node_indices: self.cancel_task_changes(task_points_vars, task_node_indices))
    self.add_button_style(cancel_changes_button)
    cancel_changes_button.config(
        background=self.color_config["delete_button_bg_color"],
        foreground=self.color_config["delete_button_fg_color"]
    )
    cancel_changes_button.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=(0, self.grid_pad_x),
        pady=0)
    # bind ESC to cancel changes
    self.master.bind("<Escape>", lambda event, task_points_vars=task_points_vars, task_node_indices=self.task_node_indices: self.cancel_task_changes(task_points_vars, task_node_indices))


  def add_task_location(self,
      location_index: int,
      location_name: str = "None",
      update_add_location_button: bool = False) -> None:
    """
    Add a location input field to the task edit frame. This consists of a
    - label with the node number
    - a label to show the selected node name (scrolling on the label or clicking it will change the selected node)
    - two buttons to change the selected node
    - a button to remove the location

    Args:
        location_index (int): The index of the location to be added (= current number of locations)
        location_name (str, optional): The initial name displayed in the location label. Defaults to "None".
        update_add_location_button (bool, optional): If True, the add location button will be updated such that it will add a new location in the correct position. Defaults to False.
    """
    node_number_label = tk.Label(self.task_locations_frame, text=f"{location_index + 1}.")
    self.add_label_style(node_number_label, font_type="bold")
    node_number_label.grid(
        row=location_index,
        column=0,
        sticky="w",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y))
    node_selector_frame = tk.Frame(self.task_locations_frame)
    self.add_frame_style(node_selector_frame)
    node_selector_frame.grid(
        row=location_index,
        column=1,
        sticky="w",
        padx=0,
        pady=(0, self.grid_pad_y))
    # display the name of the currently selected node
    # add node to highlighted particles
    self.task_node_indices.append(self.node_names.index(location_name))
    if location_name != "None": # highlight selected node
      self.highlighted_particles.append(self.particle_graph.particle_nodes[location_name])
      self.highlighted_particles[-1].highlight(self.ax)
    selected_node_indicator = tk.Label(node_selector_frame, text=location_name, width = max([len(name) for name in self.node_names]), cursor="hand2")
    self.add_label_style(selected_node_indicator, font_type="italic")
    selected_node_indicator.grid(
        row=0,
        column=1,
        sticky="w",
        padx=0,
        pady=(0, self.grid_pad_y))
    # add bindings to change text of the label (mousewheel and buttons)
    selected_node_indicator.bind(
        "<MouseWheel>",
        func = lambda event, location_indicator_index=location_index: 
            self.change_node_label(event.delta, location_indicator_index))
    selected_node_indicator.bind(
        "<Button-1>",
        func = lambda event, location_indicator_index=location_index:
            self.change_node_label(-1, location_indicator_index))
    selected_node_indicator.bind(
        "<Button-3>",
        func = lambda event, location_indicator_index=location_index:
            self.change_node_label(1, location_indicator_index))
    # add arrow buttons to change the selected node
    left_arrow_button = self.add_arrow_button("left", node_selector_frame, lambda location_indicator_index=location_index: self.change_node_label(1, location_indicator_index))
    left_arrow_button.grid(
        row=0,
        column=0,
        sticky="e",
        padx=0,
        pady=0)
    right_arrow_button = self.add_arrow_button("right", node_selector_frame, lambda location_indicator_index=location_index: self.change_node_label(-1, location_indicator_index))
    right_arrow_button.grid(
        row=0,
        column=2,
        sticky="w",
        padx=0,
        pady=0)
    # add button to remove the location
    remove_location_button = tk.Button(
        node_selector_frame,
        text="Remove",
        command=lambda task_index=location_index: self.remove_task_location(task_index))
    self.add_button_style(remove_location_button)
    remove_location_button.config(
      background=self.color_config["delete_button_bg_color"],
      foreground=self.color_config["delete_button_fg_color"]
    )
    remove_location_button.grid(
        row=0,
        column=3,
        sticky="w",
        padx=(self.grid_pad_x, 0),
        pady=0)
    self.task_location_widgets.append(
      (node_number_label, node_selector_frame, left_arrow_button, selected_node_indicator, right_arrow_button, remove_location_button)
    )
    # update the add location button
    if update_add_location_button:
      self.add_task_location_button.config(
        command=lambda task_index=location_index+1: self.add_task_location(
            task_index,
            location_name="None",
            update_add_location_button=True)
      )

  def remove_task_location(self, task_index: int):
    """
    remove the task location at the given index.
    - Remove highlight from the corresponding node,
    - remove the associated widgets, and
    - update the index of the following locations.

    Args:
        task_index (int): index of the task location to remove
    """
    # remove highlight from the current node
    current_node_name = self.node_names[self.task_node_indices[task_index]]
    if current_node_name != "None":
      current_node = self.particle_graph.particle_nodes[current_node_name]
      current_node.remove_highlight(self.ax)
      self.canvas.draw_idle()
      self.highlighted_particles.remove(current_node)
    # remove the associated widgets
    for widget in self.task_location_widgets[task_index]:
      widget.destroy()
    # move the widgets below the removed location up and change their task number
    for row_index in range(task_index, len(self.task_node_indices)-1):
      # repostion the task number label
      task_number_label: tk.Label = self.task_location_widgets[row_index+1][0]
      task_number_label.config(text=f"{row_index+1}.")
      task_number_label.grid_remove()
      task_number_label.grid(row=row_index)
      # repostion the node selector frame
      node_selector_frame: tk.Frame = self.task_location_widgets[row_index+1][1]
      node_selector_frame.grid_remove()
      node_selector_frame.grid(row=row_index)
      # update the index of the location indicator
      location_indicator: tk.Label = self.task_location_widgets[row_index+1][3]
      location_indicator.bind(
          "<MouseWheel>",
          func = lambda event, location_indicator_index=row_index:
              self.change_node_label(event.delta, location_indicator_index))
      location_indicator.bind(
          "<Button-1>",
          func = lambda event, location_indicator_index=row_index:
              self.change_node_label(-1, location_indicator_index))
      location_indicator.bind(
          "<Button-3>",
          func = lambda event, location_indicator_index=row_index:
              self.change_node_label(1, location_indicator_index))
      # update the index of the left arrow button
      left_arrow_button: tk.Button = self.task_location_widgets[row_index+1][2]
      left_arrow_button.config(
        command=lambda task_index=row_index: self.change_node_label(-1, task_index))
      # update the index of the right arrow button
      right_arrow_button: tk.Button = self.task_location_widgets[row_index+1][4]
      right_arrow_button.config(
        command=lambda task_index=row_index: self.change_node_label(1, task_index))
      # update the index of the remove location button
      remove_location_button: tk.Button = self.task_location_widgets[row_index+1][5]
      remove_location_button.config(
        command=lambda task_index=row_index: self.remove_task_location(task_index))
    # delete task from variables
    del self.task_node_indices[task_index]
    del self.task_location_widgets[task_index]
    # update the add location button
    self.add_task_location_button.config(
      command=lambda task_index=len(self.task_node_indices): self.add_task_location(
          task_index,
          location_name="None",
          update_add_location_button=True)
    )

  def apply_task_changes(self, task: TTR_Task, task_points_vars: List[tk.IntVar], task_node_indices: List[int]):
    """
    Apply changes to the currently selected task. If it did not exist before, add it to the task list. Then re-open the task overview.

    Args:
        task (TTR_Task): task to change
        task_points_vars (List[tk.IntVar]): list of IntVars for the task points (these will be deleted)
        task_node_indices (List[int]): list of node indices for the task locations
    """
    # get new node names
    new_node_names: List[str] = [self.node_names[i] for i in task_node_indices if i != 0] # remove the placeholder "None" (index 0)
    # update task nodes and name
    if task.is_empty():
      task.set_node_names(new_node_names, update_name=True)
      self.particle_graph.tasks[task.name] = task
      self.task_list.append(task) # potentially unnecessary
    elif task.node_names != new_node_names:
      task.set_node_names(new_node_names, update_name=True) # TODO: consider adding a name input
      del self.particle_graph.tasks[task.name]
      self.particle_graph.tasks[task.name] = task
    # update task points
    task.set_length(task_points_vars[0].get())
    task.set_points(*[var.get() for var in task_points_vars[1:]])
    # remove highlights from selected nodes
    self.cancel_task_changes(task_points_vars, task_node_indices)
    # clear the task frame and go back to the task overview
    self.open_task_overview()

  def cancel_task_changes(self, task_points_vars: List[tk.IntVar], task_node_indices: List[int]):
    """
    Abort changing the currently selected task. Then re-open the task overview.

    Args:
        task_points_vars (List[tk.IntVar]): list of IntVars for the task points (these will be deleted)
        task_node_indices (List[int]): list of node indices for the task locations (remove highlight from these nodes)
    """
    # unbind the escape key
    self.master.unbind("<Escape>")
    # remove highlight from selected nodes
    for node_index in task_node_indices:
      self.particle_graph.particle_nodes[self.node_names[node_index]].remove_highlight(self.ax)
    # delete task points variables
    for task_points_var in task_points_vars:
      del task_points_var
    # clear the task frame and go back to the task overview
    self.open_task_overview()

  def add_int_input(self,
      partent: tk.Widget,
      row_index: int,
      column_index: int,
      label_text: str,
      int_var: tk.IntVar,
      input_width: int = 3,
      input_justify: str = "center") -> None:
    """
    Add an integer input to the given parent widget at the specified row and column (using the grid layout manager). The input consists of a label describing the input, a label showing the current value, and two buttons to increase and decrease the value. The user can also adjust the value by scrolling the mouse wheel over the value label or clicking on it (left click to decrease, right click to increase).
    This method adds widgets to the given (row, column) as well as (row, column + 1) positions.

    Args:
        partent (tk.Widget): parent widget
        row_index (int): row index
        column_index (int): column index where the name label should be placed. The buttons and indicator will be placed in the next column.
        label_text (str): text to display in the label
        int_var (tk.IntVar): integer variable to store the value. This variable's value will be shown in the indicator label and will be updated when the user changes the value.
        input_width (int, optional): width of the indicator label in text units. Defaults to 3.
        input_justify (str, optional): justification of the indicator label. Defaults to "center".
    """
    input_name_label = tk.Label(partent, text=label_text)
    self.add_label_style(input_name_label)
    input_name_label.grid(
        row=row_index,
        column=column_index,
        sticky="nw",
        padx=self.grid_pad_x,
        pady=(0, self.grid_pad_y))
    number_input_frame = tk.Frame(partent)
    self.add_frame_style(number_input_frame)
    number_input_frame.grid(
        row=row_index,
        column=column_index + 1,
        sticky="w",
        padx=(0, self.grid_pad_x),
        pady=(0, self.grid_pad_y))
    number_input_label = tk.Label(number_input_frame, width=input_width, justify=input_justify, textvariable=int_var, cursor="hand2")
    self.add_label_style(number_input_label, font_type="italic")
    number_input_label.grid(
        row=row_index,
        column=1,
        sticky="w",
        padx=0,
        pady=0)
    # add bindings to change text of the label (mousewheel and buttons)
    number_input_label.bind(
        "<MouseWheel>",
        func = lambda event, int_var=int_var: change_edge_length(event.delta, int_var))
    number_input_label.bind(
        "<Button-1>",
        func = lambda event, int_var=int_var: change_edge_length(-1, int_var))
    number_input_label.bind(
        "<Button-3>",
        func = lambda event, int_var=int_var: change_edge_length(1, int_var))
    # add arrow buttons to change the edge length
    left_arrow_button = self.add_arrow_button("left", number_input_frame, lambda int_var=int_var: change_edge_length(-1, int_var))
    left_arrow_button.grid(
        row=row_index,
        column=0,
        sticky="e",
        padx=0,
        pady=0)
    right_arrow_button = self.add_arrow_button("right", number_input_frame, lambda int_var=int_var: change_edge_length(1, int_var))
    right_arrow_button.grid(
        row=row_index,
        column=2,
        sticky="w",
        padx=0,
        pady=0)

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
      current_node_name = self.node_names[self.task_node_indices[location_indicator_index]]
      if current_node_name != "None":
        old_node = self.particle_graph.particle_nodes[current_node_name]
        old_node.remove_highlight(self.ax)
        self.highlighted_particles.remove(old_node)

    if change_direction < 0:
      new_location_index = (self.task_node_indices[location_indicator_index] + 1) % len(self.node_names)
      # skip a node if it is already selected (except None)
      while new_location_index in self.task_node_indices and new_location_index != 0:
        new_location_index = (new_location_index + 1) % len(self.node_names)
      self.task_node_indices[location_indicator_index] = new_location_index
    elif change_direction > 0:
      new_location_index = (self.task_node_indices[location_indicator_index] - 1) % len(self.node_names)
      # skip a node if it is already selected (except None)
      while new_location_index in self.task_node_indices and new_location_index != 0:
        new_location_index = (new_location_index - 1) % len(self.node_names)
      self.task_node_indices[location_indicator_index] = new_location_index
    else:
      return
    # update the label text
    self.task_location_widgets[location_indicator_index][3].config(text=self.node_names[self.task_node_indices[location_indicator_index]])
    # highlight the node corresponding to the new label
    current_node_name = self.node_names[self.task_node_indices[location_indicator_index]]
    if current_node_name != "None":
      new_node = self.particle_graph.particle_nodes[current_node_name]
      new_node.highlight(self.ax)
      self.highlighted_particles.append(new_node)
    self.canvas.draw_idle()


  def delete_task(self, task: TTR_Task, task_index: int):
    """
    Deletes the given task:
    - if it was shown, remove it from the canvas
    - remove the corresponding task visibility variable
    - remove it from the particle graph
    - remove it from the task edit frame
    - update number labels of all other tasks in list

    Args:
        task (TTR_Task): task to delete
        task_index (int): index of the task in the task list
    """
    print(f"deleting task {task.name}")
    # remove the task from the canvas
    if self.task_visibility_vars[task.name].get():
      task.erase()
      self.canvas.draw_idle()
    else:
      # check if all task toggle needs to be activated if a hidden task is deleted
      for task_name, task_visibility_var in self.task_visibility_vars.items():
        if task_name not in ("all", task.name) and not task_visibility_var.get():
          break
      else:
        self.task_visibility_vars["all"].set(True)
    # remove the task from the list of highlighted tasks
    del self.task_visibility_vars[task.name]
    del self.particle_graph.tasks[task.name]
    self.task_list.pop(task_index)
    # remove the task widgets from the frame
    deleted_task_widgets = self.task_list_widgets.pop(task_index)
    for widget in deleted_task_widgets[:2]:
      widget.destroy()
    # update the task number labels
    for row_index, task_widgets in enumerate(self.task_list_widgets[task_index:]):
      # move frame one row up
      task_widgets[0].grid_remove()
      task_widgets[0].grid(row=row_index)
      # change task number in label
      task_widgets[1].config(text=f"{row_index+1}.")
      # update task delete button command
      task_widgets[5].config(command=lambda task=self.task_list[row_index], task_index=row_index: self.delete_task(task, task_index))


  def export_task_images(self) -> None:
    """
    Export the current task images to the directory given in `self.task_export_dir`.

    Exporting is done using the following steps:
    - hide all nodes, labels, edges and task indicators
    - show background image
    """

  
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



def change_edge_length(change_direction: int, int_var: tk.IntVar) -> None:
  """
  Change the value of the given variable by `change_direction` (+-1).

  Args:
      change_direction (int): direction of the color change (+-1)
  """
  current_value = int(int_var.get())	
  if change_direction > 0:
    int_var.set(current_value + 1)
  elif change_direction < 0:
    int_var.set(current_value - 1)
  else:
    return # no change
