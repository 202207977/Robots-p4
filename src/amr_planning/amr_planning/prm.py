import datetime
import numpy as np
import os
import pytz
import random
import time
import math
from typing import List
import heapq
import copy

# This try-except enables local debugging of the PRM class
try:
    from amr_planning.maps import Map
except ImportError:
    from maps import Map

from matplotlib import pyplot as plt


class PRM:
    """Class to plan a path to a given destination using probabilistic roadmaps (PRM)."""

    def __init__(
        self,
        map_path: str,
        obstacle_safety_distance=0.08,
        use_grid: bool = False,
        node_count: int = 50,
        grid_size=0.1,
        connection_distance: float = 0.15,
        sensor_range_max: float = 8.0,
        ):
        """Probabilistic roadmap (PRM) class initializer.

        Args:
            map_path: Path to the map of the environment.
            obstacle_safety_distance: Distance to grow the obstacles by [m].
            use_grid: Sample from a uniform distribution when False.
                Use a fixed step grid layout otherwise.
            node_count: Number of random nodes to generate. Only considered if use_grid is False.
            grid_size: If use_grid is True, distance between consecutive nodes in x and y.
            connection_distance: Maximum distance to consider adding an edge between two nodes [m].
            sensor_range_max: Sensor measurement range [m].
        """
        self._map: Map = Map(
            map_path,
            sensor_range=sensor_range_max,
            safety_distance=obstacle_safety_distance,
            compiled_intersect=False,
            use_regions=False,
        )

        self._graph: dict[tuple[float, float], list[tuple[float, float]]] = self._create_graph(
            use_grid,
            node_count,
            grid_size,
            connection_distance,
        )

        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )


    def find_path(
        self, start: tuple[float, float], goal: tuple[float, float]
        ) -> list[tuple[float, float]]:
        """Computes the shortest path from a start to a goal location using the A* algorithm.

        Args:
            start: Initial location in (x, y) [m] format.
            goal: Destination in (x, y) [m] format.

        Returns:
            Path to the destination. The first value corresponds to the initial location.

        """
        # Check if the target points are valid
        if not self._map.contains(start):
            raise ValueError("Start location is outside the environment.")

        if not self._map.contains(goal):
            raise ValueError("Goal location is outside the environment.")

        ancestors: dict[tuple[float, float], tuple[float, float]] = {}  # {(x, y): (x_prev, y_prev)}

        # TODO: 4.3. Complete the function body (i.e., replace the code below).
        
        # 1. Find the closest nodes to the start and goal points
        possibles = self._graph.keys()
        node_start = min(possibles, key = lambda k: self._distance(start, k))
        node_goal = min(possibles, key = lambda k: self._distance(k, goal))

        # 2. Initialize the open and closed lists
        open_list = {node_start: (self._distance(node_start, node_goal), 0)} # {(node): (f, g)}
        closed_list = set()

        while open_list:

            # 3a. Get the node with the lowest f-value
            expanded_node = min(open_list, key=lambda k: open_list[k][0]) 

            # 3b. Get the node's g-value
            g_value = open_list[expanded_node][1]
            del open_list[expanded_node]

            # Check if it is the goal node:
            if expanded_node == node_goal:

                # If the goal has been reached, reconstruct the path 
                if node_start != start:
                    ancestors[node_start] = start
                if node_goal != goal:
                    ancestors[goal] = node_goal

                return self._reconstruct_path(node_start, node_goal, ancestors)

            # 3c. Add open nodes to the list 
            for node in self._graph[expanded_node]:
                if node in closed_list:
                    continue

                new_g = g_value + self._distance(expanded_node, node)
                h = self._distance(node, node_goal)
                new_f = new_g + h

                # If the node hadn´t been reached yet
                if not node in open_list:
                    open_list[node] = (new_f, new_g)
                    ancestors[node] = expanded_node
                
                # If the node had been reached
                else: 
                    old_g = open_list[node][1]

                    # Update if the new path is shorter
                    if old_g > new_g:
                        open_list[node] = (new_f, new_g)
                        ancestors[node] = expanded_node
            
            closed_list.add(expanded_node)

        # If goal is not reached
        raise ValueError("Enable to find a path between start and goal.")


    def _distance(self, node1: tuple[float, float], node2: tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points (node1 and node2).

        Args:
            node1: First point in (x, y) format.
            node2: Second point in (x, y) format.

        Returns:
            Euclidean distance between the two points.
        """
        x1, y1 = node1
        x2, y2 = node2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    @staticmethod
    def smooth_path(
        path: list[tuple[float, float]],
        data_weight: float = 0.1,
        smooth_weight: float = 0.3,
        additional_smoothing_points: int = 0,
        tolerance: float = 1e-6,
        ) -> list[tuple[float, float]]:
        """Computes a smooth path from a piecewise linear path.

        Args:
            path: Non-smoothed path to the goal (start location first).
            data_weight: The larger, the more similar the output will be to the original path.
            smooth_weight: The larger, the smoother the output path will be.
            additional_smoothing_points: Number of equally spaced intermediate points to add
                between two nodes of the original path.
            tolerance: The algorithm will stop when after an iteration the smoothed path changes
                less than this value.

        Returns: Smoothed path (initial location first) in (x, y) format.

        """
        # TODO: 4.5. Complete the function body (i.e., load smoothed_path).
        
        original_path: list[tuple[float, float]] = copy.deepcopy(path)

        # Interpolation of extra points if necessary 
        if additional_smoothing_points > 0:
            extended_path = []
            for i in range(len(path) - 1):
                extended_path.append(path[i])  # Add original point

                for j in range(1, additional_smoothing_points + 1):
                    alpha = j / (additional_smoothing_points + 1)
                    x_interpolated = path[i][0] + (path[i + 1][0] - path[i][0]) * alpha
                    y_interpolated = path[i][1] + (path[i + 1][1] - path[i][1]) * alpha

                    extended_path.append((x_interpolated, y_interpolated))

            extended_path.append(path[-1])  # Add last point
            original_path = copy.deepcopy(extended_path)

        # Gradient descent to smooth the path 
        smoothed_path = copy.deepcopy(original_path)
        change =  float('inf') # Initialize threshold with a high value
        while change >= tolerance:
            change = 0
            for i in range(1, len(smoothed_path) - 1):  # Do not modify the first nor last node
                x, y = original_path[i]

                # Calculate new position of each node
                curr_x, curr_y = smoothed_path[i]

                # Calculate smoothing (average of adjacent nodes)
                prev_x, prev_y = smoothed_path[i - 1]
                next_x, next_y = smoothed_path[i + 1]

                # Smoothing based on the gradient
                new_x = curr_x + data_weight * (x-curr_x) + smooth_weight * (prev_x + next_x - 2 * curr_x)
                new_y = curr_y + data_weight * (y-curr_y) + smooth_weight * (prev_y + next_y - 2 * curr_y)

                # Update the node
                smoothed_path[i] = (new_x, new_y)

                # Calculate difference between new and previous position
                change += math.sqrt((new_x - curr_x) ** 2 + (new_y - curr_y) ** 2)

        return smoothed_path


    def plot(
        self,
        axes,
        path: list[tuple[float, float]] = (),
        smoothed_path: list[tuple[float, float]] = (),
        ):
        """Draws particles.

        Args:
            axes: Figure axes.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).

        Returns:
            axes: Modified axes.

        """
        # Plot the nodes
        x, y = zip(*self._graph.keys())
        axes.plot(list(x), list(y), "co", markersize=1)

        # Plot the edges
        for node, neighbors in self._graph.items():
            x_start, y_start = node

            if neighbors:
                for x_end, y_end in neighbors:
                    axes.plot([x_start, x_end], [y_start, y_end], "c-", linewidth=0.25)

        # Plot the path
        if path:
            x_val = [x[0] for x in path]
            y_val = [x[1] for x in path]

            axes.plot(x_val, y_val)  # Plot the path
            axes.plot(x_val[1:-1], y_val[1:-1], "bo", markersize=4)  # Draw nodes as blue circles

        # Plot the smoothed path
        if smoothed_path:
            x_val = [x[0] for x in smoothed_path]
            y_val = [x[1] for x in smoothed_path]

            axes.plot(x_val, y_val, "y")  # Plot the path
            axes.plot(x_val[1:-1], y_val[1:-1], "yo", markersize=2)  # Draw nodes as yellow circles

        if path or smoothed_path:
            axes.plot(
                x_val[0], y_val[0], "rs", markersize=7
            )  # Draw a red square at the start location
            axes.plot(
                x_val[-1], y_val[-1], "g*", markersize=12
            )  # Draw a green star at the goal location

        return axes


    def show(
        self,
        title: str = "",
        path=(),
        smoothed_path=(),
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
        ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, path, smoothed_path)

        axes.set_title(title)
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait for 1 ms or the figure won't be displayed

        if display:
            plt.show(block=block)

        if save_figure:
            save_path = os.path.join(os.path.dirname(__file__), "..", save_dir)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = f"{self._timestamp} {title.lower()}.png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)


    def _connect_nodes(
        self,
        graph: dict[tuple[float, float], list[tuple[float, float]]],
        connection_distance: float = 0.15,
        ) -> dict[tuple[float, float], list[tuple[float, float]]]:
        """Connects every generated node with all the nodes that are closer than a given threshold.

        Args:
            graph: A dictionary with (x, y) [m] tuples as keys and empty lists as values.
            connection_distance: Maximum distance to consider adding an edge between two nodes [m].

        Returns: A modified graph with lists of connected nodes as values.

        """
        # TODO: 4.2. Complete the missing function body with your code.

        # Obtain all nodes as a list of tuples
        nodes = list(graph.keys())

        # Iterate over each node in the graph
        for i, node_a in enumerate(nodes):
            # Iterate over the remaining nodes to check connections
            for j, node_b in enumerate(nodes):
                
                # Not necessary to compare the node with itself or previous ones
                if i >= j:
                    continue  

                # Calculate distance between nodes
                dist = self._distance(node_a, node_b)

                # If the distance is equal or less than the connection_distance threshold
                if dist <= connection_distance:
                    
                    # Then, verify wether the segment joining the two nodes crosses any obstacle
                    if not self._map.crosses([node_a, node_b]):
                        
                        # If it does not cross any obstacle, add connection in both directions
                        graph[node_a].append(node_b)
                        graph[node_b].append(node_a)

        return graph


    def _create_graph(
        self,
        use_grid: bool = False,
        node_count: int = 50,
        grid_size=0.1,
        connection_distance: float = 0.15,
        ) -> dict[tuple[float, float], list[tuple[float, float]]]:
        """Creates a roadmap as a graph with edges connecting the closest nodes.

        Args:
            use_grid: Sample from a uniform distribution when False.
                Use a fixed step grid layout otherwise.
            node_count: Number of random nodes to generate. Only considered if use_grid is False.
            grid_size: If use_grid is True, distance between consecutive nodes in x and y.
            connection_distance: Maximum distance to consider adding an edge between two nodes [m].

        Returns: A dictionary with (x, y) [m] tuples as keys and lists of connected nodes as values.
            Key elements are rounded to a fixed number of decimal places to allow comparisons.

        """
        graph = self._generate_nodes(use_grid, node_count, grid_size)
        graph = self._connect_nodes(graph, connection_distance)

        return graph


    def _generate_nodes(
        self, use_grid: bool = False, node_count: int = 50, grid_size=0.1
        ) -> dict[tuple[float, float], list[tuple[float, float]]]:
        """Creates a set of valid nodes to build a roadmap with.

        Args:
            use_grid: Sample from a uniform distribution when False.
                Use a fixed step grid layout otherwise.
            node_count: Number of random nodes to generate. Only considered if use_grid is False.
            grid_size: If use_grid is True, distance between consecutive nodes in x and y.

        Returns: A dictionary with (x, y) [m] tuples as keys and empty lists as values.
            Key elements are rounded to a fixed number of decimal places to allow comparisons.

        """
        graph: dict[tuple[float, float], list[tuple[float, float]]] = {}

        # TODO: 4.1. Complete the missing function body with your code.
        min_x, min_y, max_x, max_y = self._map.bounds()

        if use_grid:
            x_vals = np.arange(min_x, max_x, grid_size)
            y_vals = np.arange(min_y, max_y, grid_size)

            for x in x_vals:
                for y in y_vals:
                    if self._map.contains((x, y)):
                        graph[(x, y)] = []

        else:
            while len(graph) < node_count:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                if self._map.contains((x, y)):
                    graph[(x, y)] = []

        return graph


    def _reconstruct_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        ancestors: dict[tuple[int, int], tuple[int, int]],
        ) -> list[tuple[float, float]]:
        """Computes the path from the start to the goal given the ancestors of a search algorithm.

        Args:
            start: Initial location in (x, y) [m] format.
            goal: Goal location in (x, y) [m] format.
            ancestors: Dictionary with (x, y) [m] tuples as keys and the node (x_prev, y_prev) [m]
                from which it was added to the open list as values.

        Returns: Path to the goal (start location first) in (x, y) [m] format.

        """
        path: list[tuple[float, float]] = []

        # TODO: 4.4. Complete the missing function body with your code.
        
        # Start at the goal node
        current_node = goal

        # Follow ancestors until reaching the initial node 
        while current_node != start:
            path.append(current_node)
            current_node = ancestors[current_node]

        # Add initial node
        path.append(start)

        # Invert the list to obtain the list from start to final goal 
        path.reverse()

        return path


if __name__ == "__main__":
    map_name = "project"
    map_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "maps", map_name + ".json")
    )
        
    # Create the roadmap
    start_time = time.perf_counter()
    prm = PRM(map_path, use_grid=True, node_count=250, grid_size=0.1, connection_distance=0.15)
    roadmap_creation_time = time.perf_counter() - start_time

    print(f"Roadmap creation time: {roadmap_creation_time:1.3f} s")

    # Find the path
    start_time = time.perf_counter()
    path = prm.find_path(start=(-1.0, -1.0), goal=(-0.6, 1.0))
    pathfinding_time = time.perf_counter() - start_time

    print(f"Pathfinding time: {pathfinding_time:1.3f} s")

    # Smooth the path
    start_time = time.perf_counter()
    smoothed_path = prm.smooth_path(
        path, data_weight=0.1, smooth_weight=0.3, additional_smoothing_points=3
    )
    smoothing_time = time.perf_counter() - start_time

    print(f"Smoothing time: {smoothing_time:1.3f} s")

    prm.show(path=path, smoothed_path=smoothed_path, save_figure=True)

