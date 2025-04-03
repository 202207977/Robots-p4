import datetime
import math
import numpy as np
import os
import pytz

from amr_localization.maps import Map
from matplotlib import pyplot as plt


class ExtendedKalmanFilter:
    """Extended Kalman Filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensor_range_max=8.0,
        Qt=1,
        sigma_v: float = 0.05,
        sigma_w: float = 0.1,
        sigma_z: float = 0.2,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_covariance: np.ndarray = np.diag([0.1, 0.1, 0.1]),
    ):
        """Initializes the Extended Kalman Filter.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].
            initial_pose: Initial pose estimate (x, y, theta) [m, m, rad].
            initial_covariance: Initial covariance matrix.
        """
        self._dt = dt
        self._sigma_v = sigma_v
        self._sigma_w = sigma_w
        self._sigma_z = sigma_z

        self._sensor_range_max: float = sensor_range_max
        self._map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )

        self._state = np.array(initial_pose)
        self._covariance = initial_covariance

        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self._Rt = np.diag([self._sigma_v**2, self._sigma_v**2, self._sigma_w**2])
        self._Qt = Qt

    def predict(self, v: float, w: float):
        """Predicts the next state using the motion model.

        Args:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].
        """
        theta = self._state[-1]

        if abs(w) > 1e-6:
            dx = -v / w * np.sin(theta) + v / w * np.sin(theta + w * self._dt)
            dy = v / w * np.cos(theta) - v / w * np.cos(theta + w * self._dt)
        else:
            dx = v * self._dt * np.cos(theta)
            dy = v * self._dt * np.sin(theta)

        dtheta = w * self._dt

        # Se añade el ruído
        noise = np.random.multivariate_normal([0, 0, 0], self._Rt)
        self._state += np.array([dx, dy, dtheta]) + noise

        self._state[2] = (self._state[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize angle

        # Actualizamos covarianza
        Gt = np.array([[1, 0, -dy], [0, 1, dx], [0, 0, 1]])

        self._covariance = Gt @ self._covariance @ Gt.T + self._Rt

    def correction(self, measurements: list[float]):
        """Updates the state estimate based on sensor measurements.

        Args:
            measurements: Sensor measurements [m].
        """

        zt = measurements[::30]
        z_hat, Ht, Kt = self._correction_auxiliary()
        dt = np.array(zt) - z_hat

        for i in range(len(zt)):
            real, pred = zt[i], z_hat[i]
            Ht_i, Kt_i = Ht[i], Kt[i]
            dt = real - pred
            self._state += Kt_i @ np.array([[dt], [1]])

            self._covariance = (np.eye(len(self._state)) - Kt_i @ Ht_i.T) @ self._covariance

    def _correction_auxiliary(self) -> list[float]:
        """Simulates sensor readings given the current state.

        Returns: List of predicted measurements; nan if a sensor is out of range.
        """
        x, y, theta = self._state
        indices = range(0, 240, 30)
        rays = self._lidar_rays((x, y, theta), indices)

        z_hat = []
        Ht = []  # lista de las Ht_i
        Kt = []

        for ray in rays:
            object, distance = self._map.check_collision(ray, True)
            if distance is not None:
                distance += np.random.normal(0.0, self._sigma_z)
            else:
                distance = self._sensor_range_min

            Ht_i = self._compute_Ht_i(object, distance)
            St_i = Ht_i @ self._covariance @ Ht_i.T + self._Qt
            Kt_i = self._covariance @ Ht_i.T @ np.linalg.inv(St_i)
            z_hat.append(np.array(distance))
            Ht.append(Ht_i)
            Kt.append(Kt_i)

        return np.array(z_hat), Ht, Kt

    def _lidar_rays(
        self, pose: tuple[float, float, float], indices: tuple[float], degree_increment: float = 1.5
    ) -> list[list[tuple[float, float]]]:
        """Determines the simulated LiDAR ray segments for a given robot pose.

        Args:
            pose: Robot pose (x, y, theta) in [m] and [rad].
            indices: Rays of interest in counterclockwise order (0 for to the forward-facing ray).
            degree_increment: Angle difference of the sensor between contiguous rays [degrees].

        Returns: Ray segments. Format:
                 [[(x0_start, y0_start), (x0_end, y0_end)],
                  [(x1_start, y1_start), (x1_end, y1_end)],
                  ...]

        """
        x, y, theta = pose

        # Convert the sensor origin to world coordinates
        x_start = x - 0.035 * math.cos(theta)
        y_start = y - 0.035 * math.sin(theta)

        rays = []

        for index in indices:
            ray_angle = math.radians(degree_increment * index)
            x_end = x_start + self._sensor_range_max * math.cos(theta + ray_angle)
            y_end = y_start + self._sensor_range_max * math.sin(theta + ray_angle)
            rays.append([(x_start, y_start), (x_end, y_end)])

        return rays

    def set_initial_state(self, x, y, theta):
        self._state = x, y, theta

    def _compute_Ht_i(self, object, distance):
        """Predice la medición esperada phi_pred basada en el estado [x, y, theta].

        Args:
            state (np.ndarray): El estado actual del sistema [x, y, theta].

        Returns:
            np.ndarray: La medición predicha [r_pred, phi_pred], donde:
                r_pred = sqrt(x^2 + y^2)
                phi_pred = atan2(y, x)
        """
        mx, my = object
        x, y, _ = self._state

        Ht_i = [
            [(-mx + x) / np.sqrt(distance), (-my + y) / np.sqrt(distance), 0],
            [(my - y) / distance, (-mx + x) / distance, -1],
        ]

        return np.array(Ht_i)

    # def _measurement_probability(
    #     self, measurements: list[float], particle: tuple[float, float, float]
    # ) -> float:
    #     """Computes the probability of a set of measurements given a particle's pose.

    #             If a measurement is unavailable (usually because it is out of range), it is replaced with
    #             the minimum sensor range to perform the computation because the environment is smaller
    #             than the maximum range.

    #             Args:
    #                 measurements: Sensor measurements [m].
    #                 particle: Particle pose (x, y, theta) [m, m, rad].

    #             Returns:
    #                 float: Probability.
    #     s
    #     """
    #     Ht = np.zeros((2, 3))

    #     z_hat = self._obtain_z_hat()

    #     # TODO: 3.8. Complete the missing function body with your code.

    #     # Calcular la probabilidad acumulada
    #     for real, pred in zip(measurements[::30], z_hat):
    #         if math.isnan(real):
    #             real = self._sensor_range_min
    #         if math.isnan(pred):
    #             pred = self._sensor_range_min

    #         dt = real - pred
    #         # probability *= prob

    #     return probability

    # def process_noice_cov(self, u, dt) -> None:
    #     """
    #     Compute the process noise covriance using the control input
    #     """

    #     sigma_x2 = (self._sigma_v) ** 2
    #     sigma_y2 = (self._sigma_v) ** 2 * 0.1  # much smaller than the noise in the x axis
    #     sigma_theta2 = (self._sigma_w) ** 2

    #     self._Rt = np.diag([sigma_x2, sigma_y2, sigma_theta2])
