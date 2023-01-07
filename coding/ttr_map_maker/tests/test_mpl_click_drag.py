import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Create the Tkinter window and Matplotlib figure
window = tk.Tk()
figure, axes = plt.subplots()
# set axis limits
axes.set_xlim(-10, 10)
axes.set_ylim(-10, 10)

# Add a single rectangle artist to the figure
rectangle = plt.Rectangle((0, 0), 1, 1, picker=True)
axes.add_artist(rectangle)


# Create the FigureCanvasTkAgg object and pack it into the window
canvas = FigureCanvasTkAgg(figure, window)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Define the function that will be called when the rectangle is picked
def on_pick(event):
    # Get the rectangle artist that was picked
    artist = event.artist

    # Bind the motion and button release events to the canvas
    cid_1 = canvas.mpl_connect("motion_notify_event", lambda event: on_motion(event, artist))
    cid_2 = canvas.mpl_connect("button_release_event", lambda event: on_release(event, cid_1))
    print(f"cids: {cid_1}, {cid_2}")

# Define the function that will be called when the mouse is moved
# with the rectangle artist picked
def on_motion(event, artist):
    if event.inaxes != artist.axes:
        return
    # get artist height and width
    width = artist.get_width()
    height = artist.get_height()
    # Update the position of the rectangle artist based on the mouse position
    artist.set_x(event.xdata - width / 2)
    artist.set_y(event.ydata - height / 2)

    # Redraw the figure
    canvas.draw()

# Define the function that will be called when the mouse button is released
# with the rectangle artist picked
def on_release(event, cid_1):
    print("disconnecting")
    # Unbind the motion and button release events from the canvas
    canvas.mpl_disconnect(cid_1)
    canvas.mpl_disconnect(cid_1+1)
    

# Bind the pick event to the canvas
canvas.mpl_connect("pick_event", on_pick)

# Run the Tkinter event loop
window.mainloop()
