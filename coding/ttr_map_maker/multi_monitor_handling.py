"""
This file conttains the function `toggle_full_screen` which can be used to toggle a tkinter window between full screen and windowed mode. Bind this function to a key press event to make it easy to toggle full screen mode. Contrary to `tk.Tk.attributes("-fullscreen", True)`, this function will also work on multi-monitor setups and always keep the window on the same monitor it was on before toggling full screen mode.

To achieve this, the module implements a few helper functions as well as functions to plot the current monitor setup.

Example:
    import tkinter as tk
    from multi_monitor_handling import toggle_full_screen

    root = tk.Tk()
    root.bind("<F11>", lambda event: toggle_full_screen(root))
    root.mainloop()
"""


from  typing import Tuple
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import screeninfo


def toggle_full_screen(root):
  monitor_geom = get_monitor_geometry(root)
  if root.winfo_width() == monitor_geom[2] and root.winfo_height() == monitor_geom[3]:
    root.geometry(f"{monitor_geom[2]}x{monitor_geom[3]}+{monitor_geom[0]}+{monitor_geom[1]}")
    root.state("zoomed")
    root.overrideredirect(False)
  else:
    root.geometry(f"{monitor_geom[2]}x{monitor_geom[3]}+{monitor_geom[0]}+{monitor_geom[1]}")
    root.overrideredirect(True)


def get_monitor_geometry(tk_window: tk.Tk) -> Tuple[int, int, int, int]:
  """
  find the monitor that the given window is currently on and return it's x, y, width and height,
    which can be used to set the tk window to full screen mode on that monitor by calling `tk_window.geometry(f"{width}x{height}+{x}+{y}")`
    
  Args:
      tk_window (tk.Tk): tkinter window to get the monitor geometry for

  Returns:
      (int): x-coordinate of top left corner of monitor
      (int): y-coordinate of top left corner of monitor
      (int): width of current monitor
      (int): height of current monitor
  """
  # get the monitor that the window is currently on
  window_offset = (9, 9)
  # first try to get the monitor that the top left corner of the window is on
  monitor_index = get_monitor_from_coord(tk_window.winfo_x() + window_offset[0], tk_window.winfo_y() + window_offset[1])
  if monitor_index is None: # if that didn't work, try the top right corner
    monitor_index = get_monitor_from_coord(tk_window.winfo_x() + tk_window.winfo_width() + window_offset[0],
        tk_window.winfo_y() + window_offset[1])
  if monitor_index is None: # if that didn't work, try the bottom right corner
    monitor_index = get_monitor_from_coord(tk_window.winfo_x() + tk_window.winfo_width() + window_offset[0],
        tk_window.winfo_y() + tk_window.winfo_height() + window_offset[1])
  if monitor_index is None: # if that didn't work, try the bottom left corner
    monitor_index = get_monitor_from_coord(tk_window.winfo_x() + window_offset[9], 
        tk_window.winfo_y() + tk_window.winfo_height() + window_offset[1])
  if monitor_index is None: # if that didn't work, set monitor_index to 0
    monitor_index = 0
  monitors = screeninfo.get_monitors()[monitor_index]

  monitor_coords = (monitors.x, monitors.y, monitors.width, monitors.height)
  # print(f"index: {monitor_index}, {monitor_coords = }")
  return monitor_coords


def get_monitor_from_coord(x: int, y: int) -> int:
    """
    find the monitor that the given coordinates are on and return it's index in `screeninfo.get_monitors()`

    Args:
        x (int): x coordinate
        y (int): y coordinate

    Returns:
        int: index of the monitor that the given coordinates are on. None if no monitor was found
    """
    monitors = screeninfo.get_monitors()

    for i, screen in enumerate(monitors):
        if screen.x <= x <= screen.width + screen.x and screen.y <= y <= screen.height + screen.y:
            return i
    print(f"Warning: no monitor found for coordinates {x}, {y}")
    return None


def show_monitors(ax: plt.Axes, root: tk.Tk) -> None:
  """
  plot a rectangle for all detected monitors and the given root window

  Args:
      ax (plt.Axes): matplotlib axes to draw on
      root (tk.Tk): tkinter root window to plot
  """
  ax.set_aspect('equal')
  ax.set_autoscale_on(True)
  monitors = screeninfo.get_monitors()
  colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "grey", "black"]
  for i, monitor in enumerate(monitors, ):
    print(f"monitor {i}: {monitor.width}x{monitor.height} at {monitor.x}, {monitor.y}")
    monitor_center = (monitor.x + monitor.width/2, monitor.y + monitor.height/2)
    ax.add_patch(Rectangle(
        (monitor.x, monitor.y),
        monitor.width,
        monitor.height,
        fill=True,
        label=f"monitor {i}",
        alpha=0.5,
        facecolor=colors[i%len(colors)],
        edgecolor="black",))
    ax.text(monitor_center[0], monitor_center[1], f"{i}", ha="center", va="center")
  ax.add_patch(Rectangle(
      (root.winfo_x(), root.winfo_y()),
      root.winfo_width(),
      root.winfo_height(),
      fill=True,
      label="root",
      alpha=0.5,
      facecolor="white",
      edgecolor="black",))
  print(f"root: {root.winfo_width()}x{root.winfo_height()} at {root.winfo_x()}, {root.winfo_y()}")
  # plot root position
  ax.plot(root.winfo_x(), root.winfo_y(), "o", label="root position")
  ax.set_autoscale_on(True)


def plot_points(points, ax):
  symbols = ["o", "x", "s", "^", "p", "+", "d"]
  colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "grey", "black"]
  for point in points:
    monitor_index = get_monitor_from_coord(point[0], point[1])
    ax.plot(point[0], point[1], symbols[monitor_index%len(symbols)], color=colors[monitor_index%len(colors)], label=f"monitor {monitor_index}")


def draw_monitor_arrangement(ax, root):
  ax.clear()
  show_monitors(ax, root)
  monitor_geom = get_monitor_geometry(root)
  # draw rectangle representing the new geometry of root
  ax.add_patch(Rectangle(
    (monitor_geom[0], monitor_geom[1]),
    monitor_geom[2],
    monitor_geom[3],
    fill=True,
    label="new root",
    alpha=0.5,
    facecolor="white",
    edgecolor="black",
  ))
  ax.legend()
  ax.get_figure().canvas.draw()



if __name__ == "__main__":
  root = tk.Tk()
  root.minsize(700, 500)
  root.state("zoomed")
  # root.geometry("500x400+1250+100")
  root.update()
  
  import numpy as np
  test_points = ((np.random.randint(-1000, 1000, (1500, 2)) -0.5 ) + np.array([500,0])) * np.array([2,2])
  
  fig, ax = plt.subplots()

  draw_monitor_arrangement(ax, root)
  # plot_points(test_points, ax)

  # show figure in root window
  canvas = FigureCanvasTkAgg(fig, master=root)
  canvas.draw()
  canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
  root.columnconfigure(0, weight=1)
  root.rowconfigure(0, weight=1)

  # add button to redraw figure
  button = tk.Button(master=root, text="redraw", command=lambda: draw_monitor_arrangement(ax, root))
  button.grid(row=1, column=0, sticky="nsew")

  # bind fullscreen toggle to F11
  root.bind("<F11>", lambda event: toggle_full_screen(root))

  root.mainloop()