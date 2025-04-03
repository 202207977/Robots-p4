import datetime
import math
import numpy as np
import os
import pytz
import random

from amr_localization.maps import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class ExtendedKalmanFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        particle_count: int,
        sigma_v: float = 0.05,
        sigma_w: float = 0.1,
        sigma_z: float = 0.2,
        sensor_range_max: float = 8.0,
        sensor_range_min: float = 0.16,
        global_localization: bool = True,
        initial_pose: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
        initial_pose_sigma: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].
            sensor_range_max: Maximum sensor measurement range [m].
            sensor_range_min: Minimum sensor measurement range [m].
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

        """
        self._dt: float = dt
        self._initial_particle_count: int = particle_count
        self._particle_count: int = particle_count
        self._sensor_range_max: float = sensor_range_max
        self._sensor_range_min: float = sensor_range_min
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0
        self._sensor_range_max_artificial = float(1)

        self._map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        self._particles = self._init_particles(
            particle_count, global_localization, initial_pose, initial_pose_sigma
        )
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    def move(self, v, w):
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
        self._jacobianG(dx, dy)

        self._state[2] = (self._state[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize angle


    def _jacobianG(self, dx, dy):
        # Actualizamos covarianza
        self._Gt = np.array([[1, 0, -dy], [0, 1, dx], [0, 0, 1]])


    def process_noice_cov(self, u, dt) -> None:
        """
        Compute the process noise covriance using the control input
        """

        sigma_x2 = (self._sigma_v) ** 2
        sigma_y2 = (self._sigma_v) ** 2 * 0.1  # much smaller than the noise in the x axis
        sigma_theta2 = (self._sigma_w) ** 2

        self._Rt = np.diag([sigma_x2, sigma_y2, sigma_theta2])


    def prediction(self, x_prev, cov_prev, u, dt):
        pass


    def encoder_kalman_gain(self, cov_t_, C):
        pass

    def lidar_kalman_gain(self, cov_t_t, C):
        pass

    def correction(self, x_t_, z, C, cov_t_, K):
        pass
