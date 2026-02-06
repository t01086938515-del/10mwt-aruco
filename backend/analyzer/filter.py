# analyzer/filter.py

import numpy as np
from typing import Tuple, Optional


class KalmanFilter2D:
    """2D 좌표 안정화를 위한 칼만 필터"""

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        self.state = np.zeros(4)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4)
        self.initialized = False

    def initialize(self, x: float, y: float):
        self.state = np.array([x, y, 0, 0])
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.state[0], self.state[1])

    def update(
        self,
        measurement: Optional[Tuple[float, float]]
    ) -> Tuple[float, float]:
        if not self.initialized:
            if measurement:
                self.initialize(measurement[0], measurement[1])
            return (0, 0)

        self.predict()

        if measurement is None:
            return (self.state[0], self.state[1])

        z = np.array(measurement)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.state
        self.state = self.state + K @ y

        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return (self.state[0], self.state[1])

    def get_velocity(self) -> Tuple[float, float]:
        return (self.state[2], self.state[3])


class DualFootKalmanFilter:
    """양발 개별 칼만 필터"""

    def __init__(self):
        self.left_filter = KalmanFilter2D()
        self.right_filter = KalmanFilter2D()

    def update(
        self,
        left_pos: Optional[Tuple[float, float]],
        right_pos: Optional[Tuple[float, float]]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        filtered_left = self.left_filter.update(left_pos)
        filtered_right = self.right_filter.update(right_pos)
        return (filtered_left, filtered_right)
