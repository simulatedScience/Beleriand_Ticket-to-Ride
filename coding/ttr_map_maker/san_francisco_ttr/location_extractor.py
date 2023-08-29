import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the main tkinter window
Tk().withdraw()

# Open the file picker
filename = askopenfilename()

# Initialize a set to store the unique locations
unique_locations = set()

# Open the file and read the data
with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        # Add each location to the set. The set will automatically eliminate any duplicates.
        unique_locations.add(row[0].strip())  # location1
        unique_locations.add(row[1].strip())  # location2

# Print the unique locations
for location in unique_locations:
    print(location)
