# Custom Ticket to Ride
## Overview & current features
This project implements tools to create your very own, custom ticket to ride game:
- Load your own background map.
- Create your own graph from locations and "rail" connections.
- All graph elements (nodes, edges and labels) can be moved in the UI to improve the layout.
- edit colors of existing edges and text of nodes through the UI
- add and remove edges and nodes through the UI
- delete nodes through the UI
- The background and graph elements can be rescaled such that everything fits well when printed.
- Show the graph on top of the background map to export it as a large image, which can then be printed. 
- visualize the tasks on the graph
- automatic graph analysis tools ([inspired by this article](https://towardsdatascience.com/playing-ticket-to-ride-like-a-computer-programmer-2129ac4909d9))
- select an image for each node in the UI

### planned features:
- allow adding nodes through the UI
- support for different monitor resolutions
- automatically generate images for task cards
- allow adding and removing tasks through the UI
- add overlay for reward points for each edge length (possibly implemented via mpl legend?)
- automatically optimze the graph layout from the UI
- documentation and user manual

Currently the project is still unfinished, but it is already possible to create a custom game, although some steps are still manual and rather tedious. Due to the libraries being used, the program runs quite slow, so it will require some patience. The program is only tested on Windows (Win11 22h2). While most of the code should be cross-platform compatible, some features may not work on other operating systems and require modifications (mostly UI related).

## How to make your physical game
The program mostly helps just with creating the game board. However, it is still necessary to create the game cards and pieces yourself.
If you already have the original Ticket to ride game, you can use the cards and pieces from that game. Otherwise, you will have to create your own cards and pieces. Here are some suggestions for that:

For some easily accessible train cards and pieces, we suggest using Pokémon Energy cards and LEGO 1x4 bricks. Pokémon cards come in a variety of different colors and already include icons to make it easy to distinguish different colors, even in bad lighting conditions or for colorblind players. LEGO bricks are a convenient size, fun to play around with and convenient to store. Both Pokémon Energy cards and LEGO bricks are cheap and easy to obtain in sufficient quantities all around the world. You may even already have enough lying around.

For 1x4 LEGO bricks to fit well onto the board, I recommend printing it onto 8 or 9 Din A4 sheets. The sheets can then be glued together to form a large sheet of paper. For a stronger board, glue the paper onto cardboard.

To store the game board, cards and LEGO bricks, you can find or buy a box (approximately Din A4 base and 5cm tall). Then fold a large piece of thick paper (200-300g/m²) into an insert for the box to securely keep everything in place. A foldable 3D model for this insert can be found in `box_design/TTR_box_model.blend`. Adjust this to the required size, copy the plan onto the paper, cut and fold it.

## How to use the program
### Prerequisites
At the current stage, I highly recommend being familiar with Python and some image editing software like GIMP.
- Python 3.6 or higher
#### Dependencies
- matplotlib 3.6 or higher -> essential for the UI
- numpy -> for efficient calculations
- shapely -> for particle simulation
- screeninfo -> to get proper multi-monitor support
- networkx -> for graph analysis

### How to run the program
The code is currently not packaged as a standalone application, so you will have to run it from the source code (written in python). You can find the code in `coding/ttr_map_maker`. There, execute `board_layout_gui.py` to start the program, which will open a new window. Try out the many buttons!


## how the program was created
Most of the code was written with some level of support or inspiration from GitHub Copilot and/or OpenAI's ChatGPT. Other than that, if files contain code that is not my own, I tried to mention it in the file header and/or function docstring. Most of the code is my own though.

A note on my code style: empty lines between functions/ methods are usually not random. If Methods are closely related, they are grouped together - signified by just one empty line between them. Otherwise they are separated by two empty lines. At the end of a class, there are three empty lines.
This is a personal preference, but I find it helps with readability.