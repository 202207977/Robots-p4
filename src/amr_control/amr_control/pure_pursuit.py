import math


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
        if not self.path:
            return 0.0, 0.0  # Evitar problemas si la ruta no está definida

        # Encontrar el punto más cercano
        closest_xy, closest_idx = self._find_closest_point(x, y)

        # Encontrar el punto objetivo en la ruta usando la distancia de mirada adelante
        goal_x, goal_y = self._find_target_point((x, y), closest_idx)

        # Parámetros del robot
        v_max = 0.5  # Velocidad máxima lineal [m/s]
        w_max = 1.0  # Velocidad máxima angular [rad/s]

        # Calcular el ángulo de error con respecto al punto objetivo
        alpha = math.atan2(goal_y - y, goal_x - x) - theta

        print(goal_x, goal_y, alpha, theta, closest_idx, closest_xy, flush=True)

        # alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # Normalización entre -pi y pi

        # Si el error angular es grande, girar en el sitio
        # if abs(alpha) > math.pi / 4:
        #     return 0.0, w_max * (1 if alpha > 0 else -1)

        # Calcular la distancia al punto objetivo
        # distance = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

        # Aplicar la ecuación de persecución pura para calcular omega
        v = 0.15
        # v = min(v_max, distance)  # Reducir velocidad cerca del objetivo
        w = 2 * v * math.sin(alpha) / self._lookahead_distance  # Curvatura
        # w = min(w_max, gamma * v)

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

        # Recortar el camino desde el índice actual
        path_segment = self._path[origin_idx:]

        # Inicializamos el objetivo con el origen y la distancia previa
        previous_point = origin_xy
        previous_dist = 0.0

        # Buscamos el lookahead point
        for i in range(1, len(path_segment)):
            # Obtenemos el punto actual del camino
            current_point = path_segment[i]

            # Calcular la distancia acumulada desde el origen
            dist_current = math.sqrt(
                (current_point[0] - origin_xy[0]) ** 2 + (current_point[1] - origin_xy[1]) ** 2
            )

            if dist_current >= self._lookahead_distance:
                return current_point

        return path_segment[-1]
