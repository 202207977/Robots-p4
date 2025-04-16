import math
import numpy as np

class WallFollower:
    def __init__(self, dt: float) -> None:
        """Inicializador de la clase WallFollower.

        Args:
            dt: Periodo de muestreo [s].
        """
        self._dt: float = dt
        self._distancia_objetivo = 0.2  # queremos estar a esta distancia de la pared [m]
        self.Kp = 2  # ganancia para correcciones en la trayectoria
        self.Kd = 1  # ganancia para suavizar el movimiento
        self.Ki = 0.005  # ganancia para corregir errores acumulados
        self.error_integral = 0.0
        self.error_anterior = 0.0
        self._distancia_seguridad = 0.22  

        # Modos para decidir cómo girar o si estamos atrapados
        self._modo_giro_izquierda = False
        self._modo_giro_derecha = False
        self._modo_callejon_salida = False
        self._giro_completado = 0.0

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

        distancia_frontal = z_scan[0]  
        distancia_izquierda = z_scan[59] 
        distancia_derecha = z_scan[-59] 

        # Valor inicial de la velocidad
        z_v = 0.15  
        z_w = 0.0

        # 1. Si estamos atrapados, girar.
        if (distancia_frontal <= self._distancia_seguridad
            and distancia_izquierda <= 0.23
            and distancia_derecha <= 0.23
        ):
            self._modo_callejon_salida = True

        # Giro de 180 grados para salir 
        if self._modo_callejon_salida:
            z_v = 0.0
            z_w = 1.0  
            self._giro_completado += abs(z_w) * self._dt
            if self._giro_completado >= math.pi:  # giro de 180 grados
                self._modo_callejon_salida = False
                self.error_anterior = 0
                self.error_integral = 0
                self._giro_completado = 0.0
            return z_v, z_w

        # 2. Si hay un obstáculo al frente, decidimos a qué lado girar
        if distancia_frontal <= self._distancia_seguridad:
            if distancia_derecha >= distancia_izquierda:
                self._modo_giro_derecha = True
            else:
                self._modo_giro_izquierda = True

        # 3. Si vamos a girar a la derecha
        if self._modo_giro_derecha:
            z_v = 0.0
            z_w = -1.0
            self._giro_completado += abs(z_w) * self._dt
            if self._giro_completado >= math.pi / 2:  
                self._modo_giro_derecha = False
                self.error_anterior = 0
                self.error_integral = 0
                self._giro_completado = 0.0
            return z_v, z_w

        # 4. Si vamos a girar a la izquierda
        elif self._modo_giro_izquierda:
            z_v = 0.0
            z_w = 1.0
            self._giro_completado += abs(z_w) * self._dt
            if self._giro_completado >= math.pi / 2:
                self._modo_giro_izquierda = False
                self.error_anterior = 0
                self.error_integral = 0
                self._giro_completado = 0.0
            return z_v, z_w

        # 5. Si no hay obstáculos cercanos, usamos un control PID para mantener la distancia a la pared
        if abs(distancia_izquierda - distancia_derecha) < 0.2:
            error = distancia_izquierda - self._distancia_objetivo  # cuánto nos estamos alejando de la pared
            derivada = (error - self.error_anterior) / self._dt
            self.error_integral += error * self._dt

            # Control PID para ajustar el giro
            z_w = self.Kp * error + self.Kd * derivada + self.Ki * self.error_integral
            self.error_anterior = error

        return z_v, z_w