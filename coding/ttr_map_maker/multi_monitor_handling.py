
from  typing import Tuple
import tkinter as tk

import screeninfo


def get_current_monitor(tk_window: tk.Tk) -> Tuple[int, int, int, int]:
  """
  find the monitor that the given window is currently on and return it's x, y, width and height,
    which can be used to set the window to full screen mode on that monitor.
  """
  monitor_index = get_monitor_from_coord(tk_window.winfo_x(), tk_window.winfo_y())
  monitors = screeninfo.get_monitors()
  x_offset = 0
  for monitor in monitors[-1:-monitor_index-1:-1]:
    print(f"monitor: {monitor.width}x{monitor.height}")
    x_offset += monitor.width
  monitor = monitors[-monitor_index-1]
  monitor_coords = x_offset, 0, monitor.width, monitor.height
  print(f"index: {monitor_index}, {monitor_coords = }")
  return monitor_coords

def get_monitor_from_coord(x, y):
    monitors = screeninfo.get_monitors()

    for i, screen in enumerate(reversed(monitors)):
        if screen.x <= x <= screen.width + screen.x and screen.y <= y <= screen.height + screen.y:
            return i
    return 0