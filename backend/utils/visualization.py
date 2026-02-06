# utils/visualization.py

import cv2
import numpy as np
from typing import Dict, List, Tuple


class GaitVisualizer:
    """보행 분석 결과 시각화 (디버그용)"""

    COLORS = {
        'skeleton': (0, 255, 0),
        'ankle': (0, 0, 255),
        'trajectory': (255, 165, 0),
        'interpolated': (128, 128, 128),
        'text': (255, 255, 255),
        'background': (0, 0, 0)
    }

    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    def draw_pose(
        self, frame: np.ndarray, keypoints: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        result = frame.copy()
        for connection in self.SKELETON_CONNECTIONS:
            pt1_idx, pt2_idx = connection
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            if pt1[2] >= confidence_threshold and pt2[2] >= confidence_threshold:
                cv2.line(result,
                         (int(pt1[0]), int(pt1[1])),
                         (int(pt2[0]), int(pt2[1])),
                         self.COLORS['skeleton'], 2)

        for i, kpt in enumerate(keypoints):
            if kpt[2] >= confidence_threshold:
                color = self.COLORS['ankle'] if i in [15, 16] else self.COLORS['skeleton']
                cv2.circle(result, (int(kpt[0]), int(kpt[1])),
                           5 if i in [15, 16] else 3, color, -1)
        return result

    def draw_info_overlay(
        self, frame: np.ndarray, speed_mps: float,
        cadence_spm: float = None, fall_risk: str = None,
        elapsed_time: float = None
    ) -> np.ndarray:
        result = frame.copy()
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), self.COLORS['background'], -1)
        result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)

        y_offset = 35
        if elapsed_time is not None:
            cv2.putText(result, f"Time: {elapsed_time:.2f}s", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 2)
            y_offset += 30

        cv2.putText(result, f"Speed: {speed_mps:.2f} m/s", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 2)

        if cadence_spm is not None:
            y_offset += 30
            cv2.putText(result, f"Cadence: {cadence_spm:.0f} steps/min", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 2)

        if fall_risk:
            y_offset += 30
            risk_color = (0, 0, 255) if fall_risk == '높음' else (0, 255, 255)
            cv2.putText(result, f"Fall Risk: {fall_risk}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        return result
