import datetime
import math
import numpy as np
import os
import pytz
import random

from amr_localization.maps import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class ParticleFilter:
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
        self._global_localization = global_localization
        self._initial_pose = initial_pose
        self._initial_pose_sigma = initial_pose_sigma

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

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        # TODO: 3.10. Complete the missing function body with your code.
        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        scan = DBSCAN(eps=0.2, min_samples=10)
        scan.fit(self._particles[:, :2])
        clusters = len(set(scan.labels_)) - (1 if -1 in scan.labels_ else 0)

        if clusters == 1:
            if not self._detect_incorrect_position():
                localized = True
                self._particle_count = 100  # Reducir partículas para acelerar el cálculo

                angles = self._particles[:, 2]

                # para evitar el problema de los angulos 0 y 2pi
                unit_vectors = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
                mean_vector = np.mean(unit_vectors, axis=0)
                mean_angle = np.arctan2(mean_vector[1], mean_vector[0])

                pose = (
                    np.mean(self._particles[:, 0]),  # Promedio de X
                    np.mean(self._particles[:, 1]),  # Promedio de Y
                    (mean_angle + 2 * np.pi) % (2 * np.pi),  # Normalizar el ángulo entre 0 y 2π
                )
            else:
                self._particles = self._init_particles(
                    particle_count=self._initial_particle_count,
                    global_localization=self._global_localization,
                    initial_pose=self._initial_pose,
                    initial_pose_sigma=self._initial_pose_sigma,
                )

        return localized, pose

    def _detect_incorrect_position(self):
        likelihood_mean = np.mean(self._weights)  # Verosimilitud media
        threshold = 0.01  # Umbral para considerar que el robot está perdido

        if likelihood_mean < threshold:
            print("¡Robot perdido! Reiniciando partículas...")
            return True

        return False

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        # TODO: 3.5. Complete the function body with your code.
        for particle in self._particles:
            v_norm = np.random.normal(v, self._sigma_v)
            w_norm = np.random.normal(w, self._sigma_w)

            x, y, theta = particle
            theta += w_norm * self._dt % (2 * np.pi)

            pos_x = x + v_norm * np.cos(theta) * self._dt
            pos_y = y + v_norm * np.sin(theta) * self._dt

            if not self._map.contains((pos_x, pos_y)):
                obstacle, _ = self._map.check_collision([(x, y), (x + pos_x, y + pos_y)])
                if obstacle:
                    pos_x, pos_y = obstacle

            particle[0] = pos_x
            particle[1] = pos_y
            particle[2] = theta

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 3.9. Complete the function body with your code (i.e., replace the pass statement).

        # Compute the weight of each particle based on the measurement probability
        self._weights = np.array(
            [self._measurement_probability(measurements, particle) for particle in self._particles]
        )

        # Compute normalization
        weights = self._weights / np.sum(self._weights)

        # Compute the cumulative sum of weights for resampling
        cum_weights = np.cumsum(weights)
        N = len(self._particles)

        # Perform systematic resampling
        u1 = np.random.uniform(0, 1 / N)
        new_sample = np.zeros(N, dtype=int)
        for i in range(1, N + 1):
            u = u1 + (i - 1) / N
            new_sample[i - 1] = np.searchsorted(cum_weights, u)

        self._particles = self._particles[new_sample]

        return None

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
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
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(
        self,
        particle_count: int,
        global_localization: bool,
        initial_pose: tuple[float, float, float],
        initial_pose_sigma: tuple[float, float, float],
    ) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=object)

        # TODO: 3.4. Complete the missing function body with your code.

        # Orientations
        allowed_orientations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])

        # Coors of the map
        x_min, y_min, x_max, y_max = self._map.bounds()

        idx = 0
        while idx < particle_count:
            # with info
            if global_localization:
                x_particle = np.random.uniform(x_min, x_max)
                y_particle = np.random.uniform(y_min, y_max)
                theta_particle = np.random.choice(allowed_orientations)

            # Without info
            else:
                x_particle = np.random.normal(initial_pose[0], initial_pose_sigma[0])
                y_particle = np.random.normal(initial_pose[1], initial_pose_sigma[1])
                theta_particle = np.random.normal(initial_pose[2], initial_pose_sigma[2])

            # Check generated particle is within bounds
            if self._map.contains((x_particle, y_particle)):
                particles[idx] = (x_particle, y_particle, theta_particle)
                idx += 1

        return particles

    def _sense(self, particle: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every LiDAR ray given the robot's pose.

        Args:
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; nan if a sensor is out of range.

        """
        z_hat: list[float] = []

        # TODO: 3.6. Complete the  missing function body with your code.
        indices = range(0, 240, 30)
        rays = self._lidar_rays(particle, indices)

        # Vemos si nos quedamos con los rayos
        for ray in rays:
            obstacle, distance = self._map.check_collision(ray, True)

            if distance is not None:
                distance += random.gauss(0.0, self._sigma_z)
            z_hat.append(distance)

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        # TODO: 3.7. Complete the function body (i.e., replace the code below).
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

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

    def _measurement_probability(
        self, measurements: list[float], particle: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        the minimum sensor range to perform the computation because the environment is smaller
        than the maximum range.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        probability = 1.0
        measurements_predicted = self._sense(particle=particle)

        # TODO: 3.8. Complete the missing function body with your code.

        # Calcular la probabilidad acumulada
        for real, pred in zip(measurements[::30], measurements_predicted):
            if math.isnan(real):
                real = self._sensor_range_min
            if math.isnan(pred):
                pred = self._sensor_range_min

            prob = self._gaussian(real, self._sigma_z, pred)
            probability *= prob

        return probability
