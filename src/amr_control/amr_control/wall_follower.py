import math
import numpy as np

class WallFollower:
    def __init__(self, dt: float) -> None:
        """Initializes the Wall Follower controller."""

        self._dt: float = dt
        self._target_distance   = 0.2  
        self._safety_threshold   = 0.22  

        # PID control parameters
        self._Kp = 2.0  
        self._Ki = 0.005  
        self._Kd = 1.0  
        
        self._error_sum = 0.0
        self._previous_error = 0.0
        

        # State flags for maneuvering
        self._turning_left = False
        self._turning_right = False
        self._escape_mode = False
        self._rotation_progress = 0.0

    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        """Algoritmo de seguimiento de paredes.

        Args:
            z_scan: Lista de distancias LiDAR a los obstáculos [m].
            z_v: Estimación de velocidad lineal del robot [m/s].
            z_w: Estimación de velocidad angular del robot [rad/s].

        Returns:
            v: Velocidad lineal [m/s].
            w: Velocidad angular [rad/s].
        """
        z_scan = np.array(z_scan)
        z_scan = np.where(np.isnan(z_scan), 9.0, z_scan)

        forward_dist = z_scan[0] 
        right_dist = z_scan[-59]  
        left_dist = z_scan[59] 
        

        # Set initial velocity values
        z_v = 0.15  
        z_w = 0.0

        # Check if robot is trapped
        if forward_dist < self._safety_threshold and left_dist < 0.23 and right_dist < 0.23:
            self._escape_mode = True
        

        # Perform 180-degree escape turn
        if self._escape_mode:
            self._rotation_progress += self._dt
            if self._rotation_progress >= math.pi: 
                self._reset_states()
            return 0.0, 1.0

        # Decide turn direction based on obstacle proximity
        if forward_dist <= self._safety_threshold:
            if right_dist <= left_dist:
                self._turning_left = True
            else:
                self._turning_right = True
               

        # Execute right turn
        if self._turning_right:
            self._rotation_progress += self._dt
            if self._rotation_progress  >= math.pi / 2: 
                self._reset_states() 
            return 0.0, -1.0

        # Execute left turn
        if self._turning_left:
            self._rotation_progress  += self._dt
            if self._rotation_progress  >= math.pi / 2:
                self._reset_states() 
            return 0.0, 1.0

        # Wall following using PID control when path is mostly clear
        if abs(right_dist  - left_dist ) < 0.2:
            error = left_dist - self._target_distance
            derivative = (error - self._previous_error) / self._dt
            self._error_sum += error * self._dt

            # Control PID para ajustar el giro
            z_w = self._Kp * error + self._Ki * self._error_sum + self._Kd * derivative
            self._previous_error = error
        
        return z_v, z_w
    
    def _reset_states(self):
        """Resets internal flags and error accumulations."""
        self._turning_right = False
        self._turning_left = False
        self._escape_mode = False
        self._rotation_progress = 0.0
        self._error_sum = 0.0
        self._previous_error = 0.0