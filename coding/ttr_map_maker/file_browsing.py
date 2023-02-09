"""
This module implements several functions to open file dialogs for various file types.
"""

import tkinter as tk


def browse_txt_file(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select a txt file. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = tk.filedialog.askopenfilename(filetypes=[(browse_request, "*.txt")])
  if file_path:
    var.set(file_path)

def browse_image_file(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select an image file. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = tk.filedialog.askopenfilename(filetypes=[(browse_request, ["*.png", "*.jpg", "*.jpeg"])])
  if file_path:
    var.set(file_path)

def browse_json_file(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select a json file. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = tk.filedialog.askopenfilename(filetypes=[(browse_request, "*.json")])
  if file_path:
    var.set(file_path)