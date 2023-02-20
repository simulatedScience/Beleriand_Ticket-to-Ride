import tkinter as tk
from tkinter import ttk

class Auto_Scroll_Frame(tk.Frame):
  """A Frame that automatically adds a scrollbar when necessary"""
  def __init__(self,
      parent,
      scrollbar_kwargs=dict(),
      canvas_grid_kwargs=dict(),
      canvas_kwargs=dict(),
      frame_kwargs=dict()):
    super().__init__(parent, **frame_kwargs)
    # Create a canvas and a scrollbar

    self.canvas = tk.Canvas(parent,
        borderwidth=0,
        highlightthickness=0,
        **canvas_kwargs)
    self.canvas.grid(column=0, row=0, sticky="new", **canvas_grid_kwargs)
    self.canvas.grid_columnconfigure(0, weight=1)
    self.vbar = tk.Scrollbar(parent, orient="vertical", command=self.canvas.yview, elementborderwidth=0, relief="flat", **scrollbar_kwargs)
    self.vbar.grid(column=2, row=0, sticky="ens")
    self.vbar.grid_remove()
    self.canvas.configure(yscrollcommand=self.vbar.set)
    # self.canvas.grid_propagate(False)
    self.master.update()

    # Create a frame and place it inside the canvas
    self.scrollframe = tk.Frame(self.canvas, **frame_kwargs)
    self.window = self.canvas.create_window((0, 0), window=self.scrollframe, anchor="ne")
    # self.scrollframe.grid(row=0, column=0, sticky="nsew")

    self.scrollframe.bind("<Configure>", self._on_configure)
    self.master.bind("<Configure>", self._on_configure)
    self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

  def _on_configure(self, event: tk.Event = None):
    """Update the scrollbar and canvas when the size of the frame changes"""
    # restrict height of scrollframe to height of parent
    height = min(self.scrollframe.winfo_height(), self.master.winfo_height())
    self.canvas.configure(
        scrollregion=self.canvas.bbox("all"),
        width=self.scrollframe.winfo_width(),
        height=height)
    self.canvas.update()
    self._hide_or_show_scrollbar()

  def _on_mousewheel(self, event: tk.Event):
    """Scroll the canvas with the mousewheel"""
    self.canvas.yview_scroll(-1*(event.delta//120), "units")

  def _hide_or_show_scrollbar(self):
    """Hide or show the scrollbar based on whether the contents of the frame exceed the height of the window"""
    if self.master.winfo_height() < self.scrollframe.winfo_reqheight():
      self.vbar.grid(column=1, row=0, sticky="wns")
      self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    else:
      self.vbar.grid_remove()
      self.canvas.unbind_all("<MouseWheel>")
      self.master.update()


if __name__ == "__main__":
  # simple demonstration of the scrollable frame
  def add_label(parent, label_list):
    """Adding a label to the scrollable frame"""
    row = len(label_list) + 1
    label = ttk.Label(parent, text = "Label", justify='center')
    label.grid(column = 0, row = row, sticky = 'n', columnspan=2)
    label_list.append(label)

  root = tk.Tk()
  root.geometry("400x200")
  root.columnconfigure(1, weight = 1)
  root.rowconfigure(1, weight = 1)
  scrollableframe = Auto_Scroll_Frame(root,
      scrollbar_kwargs=dict(troughcolor='white'),
      canvas_kwargs=dict(bg="#0000dd"),
      frame_kwargs=dict(bg="#00dd00"))
  scrollableframe.grid(column = 0, row = 0, sticky = 'nsew')

  # add button to insert labels into scrollable frame
  label_list = []
  btn = ttk.Button(scrollableframe.scrollframe, text = "add", command = lambda: add_label(scrollableframe.scrollframe, label_list))
  btn.grid(column = 0, row = 0, sticky = 'nw')

  # add button to delete labels from scrollable frame
  btn = ttk.Button(scrollableframe.scrollframe, text = "remove", command = lambda: label_list.pop().destroy())
  btn.grid(column = 1, row = 0, sticky = 'nw')

  root.mainloop()
