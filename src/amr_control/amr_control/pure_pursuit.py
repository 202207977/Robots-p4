import math
import numpy as np

class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(self, dt: float, lookahead_distance: float = 0.4):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].

        """
        self._dt: float = dt
        self._lookahead_distance: float = lookahead_distance
        
        # Se implementa lookahead adaptativa
        self._lookahead_distance_adaptative: float = lookahead_distance
        self._path: list[tuple[float, float]] = []

    def compute_commands(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Pure pursuit controller implementation.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].
        """

        # TODO: 4.11. Complete the function body with your code (i.e., compute v and w).
        ###################################################
        v = 0.0
        w = 0.0

        # If we don't have path yet, the velocities are zero
        if  self._path:

            v = 0.21
            # Find the closest point to the robot
            # If the point is the goal, we can stop
            _, closest_idx = self._find_closest_point(x, y)

            # Compute the target point
            target_xy = self._find_target_point((x, y), closest_idx)

            # Get angle the robot should have to reach the target point
            beta = np.arctan2(target_xy[1] - y, target_xy[0] - x)
            
            # Compute ω to turn towards the target
            alpha = beta - theta
            
            w = 2 * v * np.sin(beta - theta) / self._lookahead_distance

            # If the angle is too big, stop the robot. And only move forward if the angle is small
            if abs(np.sin(abs(alpha))) > 0.7:
                v = 0.0


        return v, w

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value

    def _find_closest_point(self, x: float, y: float) -> tuple[tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # TODO: 4.9. Complete the function body (i.e., find closest_xy and closest_idx).
        closest_xy = (0.0, 0.0)
        closest_idx = 0
        min_dist = float("inf")  # Initialize minimum distance to a high value

        # Iterate over the points in the path and calculate distance
        for idx, (path_x, path_y) in enumerate(self._path):
            dist = math.sqrt((path_x - x) ** 2 + (path_y - y) ** 2)

            # If we find a distance smaller than the minimum, update closest point
            if dist < min_dist:
                min_dist = dist
                closest_xy = (path_x, path_y)
                closest_idx = idx

        return closest_xy, closest_idx

    def _find_target_point(
        self, origin_xy: tuple[float, float], origin_idx: int
    ) -> tuple[float, float]:
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # TODO: 4.10. Complete the function body with your code (i.e., determine target_xy).

        # Trim path from the current index
        path_segment = self._path[origin_idx:]

        # Search for lookahead point
        for i in range(1, len(path_segment)):
            # Get the current path point
            current_point = path_segment[i]

            # Calculate the accumulated distance from the origin
            dist_current = math.sqrt(
                (current_point[0] - origin_xy[0]) ** 2 + (current_point[1] - origin_xy[1]) ** 2
            )

            # If the accumulated distance reaches or exceeds the lookahead distance
            if dist_current >= self._lookahead_distance:
                return current_point

        # If no point meets the lookahead distance condition, return the last point in the segment
        return path_segment[-1]
