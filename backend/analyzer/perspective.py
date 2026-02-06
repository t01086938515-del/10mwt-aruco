# analyzer/perspective.py

import cv2
import numpy as np
from typing import Tuple, List, Optional


class PerspectiveCorrector:
    """호모그래피 기반 원근 보정"""

    def __init__(self):
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_matrix: Optional[np.ndarray] = None
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None
        self.output_size: Tuple[int, int] = (400, 800)

        self.start_line_y: float = 0.0
        self.end_line_y: float = 800.0
        self.pixels_per_meter: float = 80.0

    def calibrate_with_4points(
        self,
        points: List[Tuple[int, int]],
        real_width_m: float = 2.0,
        real_length_m: float = 10.0
    ):
        self.src_points = np.float32(points)

        aspect_ratio = real_length_m / real_width_m
        output_width = 400
        output_height = int(output_width * aspect_ratio)
        self.output_size = (output_width, output_height)

        self.dst_points = np.float32([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ])

        self.homography_matrix = cv2.getPerspectiveTransform(
            self.src_points, self.dst_points
        )
        self.inverse_matrix = cv2.getPerspectiveTransform(
            self.dst_points, self.src_points
        )

        self.pixels_per_meter = output_height / real_length_m
        self.start_line_y = 0.0
        self.end_line_y = output_height

        return self.homography_matrix

    def transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        if self.homography_matrix is None:
            raise ValueError("캘리브레이션이 필요합니다")
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        return (transformed[0][0][0], transformed[0][0][1])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if self.homography_matrix is None:
            raise ValueError("캘리브레이션이 필요합니다")
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
        return transformed.reshape(-1, 2)

    def inverse_transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        if self.inverse_matrix is None:
            raise ValueError("캘리브레이션이 필요합니다")
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_matrix)
        return (transformed[0][0][0], transformed[0][0][1])

    def get_bird_eye_view(self, frame: np.ndarray) -> np.ndarray:
        if self.homography_matrix is None:
            raise ValueError("캘리브레이션이 필요합니다")
        return cv2.warpPerspective(
            frame, self.homography_matrix, self.output_size
        )

    def get_distance_from_start(
        self, point_topview: Tuple[float, float]
    ) -> float:
        pixel_distance = point_topview[1] - self.start_line_y
        return pixel_distance / self.pixels_per_meter

    def is_past_start_line(
        self, point_topview: Tuple[float, float]
    ) -> bool:
        return point_topview[1] > self.start_line_y

    def is_past_end_line(
        self, point_topview: Tuple[float, float]
    ) -> bool:
        return point_topview[1] >= self.end_line_y
