"""
read graph nodes from locations file: list of location names
read edges from paths file: list of pairs of location names, lengths and colors, seperated by ` ; `
read tasks from tasks file: list of location names and lengths, seperated by ` ; `
"""
from typing import List, Tuple

import networkx as nx
import matplotlib.pyplot as plt

def read_locations(location_file: str):
    """
    read locations from file
    """
    locations = []
    with open(location_file, "r") as loc_file:
        for line in loc_file:
            line = line.strip()
            if line: # skip empty lines
                locations.append(line)

    return locations


def read_paths(path_file: str):
    """
    read edges from paths file: list of pairs of location names, lengths and colors, seperated by ` ; `
    """
    paths = []
    with open(path_file, "r") as path_file:
        for line in path_file:
            line = line.strip()
            if line:
                loc_id1, loc_id2, length, color = line.split(" ; ")
                paths.append((loc_id1, loc_id2, int(length), color))

    return paths


def read_tasks(task_file: str):
    """
    read tasks from tasks file: list of location names and lengths, seperated by ` ; `
    """
    tasks = []
    with open(task_file, "r") as task_file:
        for line in task_file:
            line = line.strip()
            if line:
                loc_1, loc_2, length = line.split(" ; ")
                tasks.append((loc_1, loc_2, int(length)))

    return tasks


def create_graph(locations: List[str], paths: List[Tuple[str, str, int, str]]):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(locations)

    # Add edges to the graph
    for (loc1, loc2, length, color) in paths:
        # Add an edge between loc1 and loc2 with the specified length and color
        G.add_edge(loc1, loc2, length=length, color=color)

    return G


def draw_graph(G: nx.Graph, tasks: List[Tuple[str, str, int]]):
    # Get the positions of all nodes
    pos = nx.fruchterman_reingold_layout(G)

    # Draw all edges
    for (loc1, loc2, length) in G.edges.data("length", "color"):
        nx.draw_networkx_edges(G, pos, edgelist=[(loc1, loc2)], width=length/10)#, edge_color=color)

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # Draw all node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # Draw all task labels
    for (loc1, loc2, length) in tasks:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(loc1, loc2): length}, font_size=10, font_family="sans-serif")

    # Show the graph
    plt.show()


def print_statistics(locations: List[str], paths: List[Tuple[str, str, int, str]], tasks: List[Tuple[str, str, int]]):
    print("Number of locations: ", len(locations))
    print("Number of paths: ", len(paths))
    print("Number of tasks: ", len(tasks))

    # for each color in path labels, print the number of paths with that color
    colors = set()
    for (loc1, loc2, length, color) in paths:
        colors.add(color)
    for color in colors:
        colored_paths = [path for path in paths if path[3] == color]
        print(f"Number of paths with color {color}: ", len(colored_paths))
        print(f"total length of paths with color {color}: ", sum([path[2] for path in colored_paths]))
        print(f"average length of paths with color {color}: ", sum([path[2] for path in colored_paths]) / len(colored_paths))

    print(f"total average length of paths: ", sum([path[2] for path in paths]) / len(paths))

    # for each length in task labels, print the number of tasks with that length
    lengths = set()
    for (loc1, loc2, length) in tasks:
        lengths.add(length)
    for length in lengths:
        print(f"Number of tasks with length {length}: ", len([task for task in tasks if task[2] == length]))


def main():
    location_file = "beleriand_locations.txt"
    path_file = "beleriand_paths.txt"
    task_file = "beleriand_tasks.txt"

    locations = read_locations(location_file)
    paths = read_paths(path_file)
    tasks = read_tasks(task_file)

    print_statistics(locations, paths, tasks)

    G = create_graph(locations, paths)
    draw_graph(G, tasks)

if __name__ == "__main__":
    main()