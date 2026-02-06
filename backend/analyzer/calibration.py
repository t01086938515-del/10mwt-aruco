# analyzer/calibration.py

import numpy as np
from typing import Tuple, Optional


class DistanceCalibrator:
    """픽셀-실제 거리 변환 캘리브레이터"""

    def __init__(self):
        self.pixels_per_meter: Optional[float] = None
        self.calibration_points: Optional[Tuple] = None

    def calibrate_with_known_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        real_distance_meters: float = 10.0
    ):
        pixel_distance = np.sqrt(
            (point2[0] - point1[0]) ** 2 +
            (point2[1] - point1[1]) ** 2
        )
        self.pixels_per_meter = pixel_distance / real_distance_meters
        self.calibration_points = (point1, point2)
        return self.pixels_per_meter

    def calibrate_with_person_height(
        self,
        head_point: Tuple[float, float],
        foot_point: Tuple[float, float],
        person_height_meters: float = 1.7
    ):
        pixel_height = np.sqrt(
            (foot_point[0] - head_point[0]) ** 2 +
            (foot_point[1] - head_point[1]) ** 2
        )
        self.pixels_per_meter = pixel_height / person_height_meters
        return self.pixels_per_meter

    def pixels_to_meters(self, pixel_distance: float) -> float:
        if self.pixels_per_meter is None:
            raise ValueError("캘리브레이션이 필요합니다")
        return pixel_distance / self.pixels_per_meter

    def meters_to_pixels(self, meter_distance: float) -> float:
        if self.pixels_per_meter is None:
            raise ValueError("캘리브레이션이 필요합니다")
        return meter_distance * self.pixels_per_meter


class HeightBasedValidator:
    """환자 키 기반 캘리브레이션 검증"""

    def __init__(self, patient_height_cm: float):
        self.patient_height_m = patient_height_cm / 100.0

    def validate_calibration(
        self,
        head_point_topview: Tuple[float, float],
        foot_point_topview: Tuple[float, float],
        pixels_per_meter: float
    ) -> dict:
        pixel_height = abs(head_point_topview[1] - foot_point_topview[1])
        measured_height_m = pixel_height / pixels_per_meter
        error = abs(measured_height_m - self.patient_height_m)
        error_percent = (error / self.patient_height_m) * 100

        return {
            'measured_height_m': round(measured_height_m, 3),
            'expected_height_m': self.patient_height_m,
            'error_percent': round(error_percent, 1),
            'is_valid': error_percent < 10
        }
