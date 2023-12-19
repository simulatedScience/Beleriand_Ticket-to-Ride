"""
This module implements several functions to open file dialogs for various file types.
"""

import tkinter as tk
from tkinter import filedialog


def browse_txt_file(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select a txt file. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = filedialog.askopenfilename(
      initialdir="../../projects",
      filetypes=[(browse_request, "*.txt")])
  if file_path:
    var.set(file_path)

def browse_image_file(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select an image file. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = filedialog.askopenfilename(
      initialdir="../../projects",
      filetypes=[(browse_request, ["*.png", "*.jpg", "*.jpeg", "*.webp"])])
  if file_path:
    var.set(file_path)

def browse_json_file(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select a json file. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = filedialog.askopenfilename(
      initialdir="../../projects",
      filetypes=[(browse_request, "*.json")])
  if file_path:
    var.set(file_path)

def browse_directory(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select a directory. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = filedialog.askdirectory(initialdir="../../projects",)
  if file_path:
    var.set(file_path)
    
def browse_ttf_file(browse_request: str, var: tk.StringVar):
  """
  Open a file dialog to select a ttf file. The file path is stored in the given variable.

  Args:
      browse_request (str): text to display in the file dialog
      var (tk.StringVar): variable to store the file path in
  """
  file_path = filedialog.askopenfilename(
      initialdir="../../projects",
      filetypes=[(browse_request, "*.ttf")])
  if file_path:
    var.set(file_path)