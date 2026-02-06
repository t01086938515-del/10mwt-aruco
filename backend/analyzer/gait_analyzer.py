# analyzer/gait_analyzer.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from .calibration import DistanceCalibrator


class GaitAnalyzer:
    """보행 속도 및 임상 해석"""

    def __init__(
        self,
        calibrator: DistanceCalibrator,
        fps: float = 30.0
    ):
        self.calibrator = calibrator
        self.fps = fps
        self.frame_duration = 1.0 / fps

    def calculate_speed(
        self,
        elapsed_time_s: float,
        distance_m: float = 10.0
    ) -> Dict:
        if elapsed_time_s <= 0:
            return self._empty_result()

        speed_mps = distance_m / elapsed_time_s

        return {
            'speed_mps': round(speed_mps, 3),
            'speed_kmph': round(speed_mps * 3.6, 3),
            'total_distance_m': round(distance_m, 3),
            'total_time_s': round(elapsed_time_s, 3),
        }

    def _empty_result(self) -> Dict:
        return {
            'speed_mps': 0, 'speed_kmph': 0,
            'total_distance_m': 0, 'total_time_s': 0
        }

    def get_clinical_interpretation(
        self,
        speed_mps: float,
        cadence_spm: float = None
    ) -> Dict:
        interpretation = {
            'speed_category': '',
            'fall_risk': '',
            'community_ambulation': '',
            'recommendations': []
        }

        if speed_mps >= 1.2:
            interpretation['speed_category'] = '빠른 보행'
        elif speed_mps >= 0.8:
            interpretation['speed_category'] = '정상 보행'
        elif speed_mps >= 0.6:
            interpretation['speed_category'] = '느린 보행'
        elif speed_mps >= 0.4:
            interpretation['speed_category'] = '매우 느린 보행'
        else:
            interpretation['speed_category'] = '기능적 제한'

        if speed_mps >= 0.8:
            interpretation['community_ambulation'] = '지역사회 보행 가능'
        elif speed_mps >= 0.4:
            interpretation['community_ambulation'] = '제한적 지역사회 보행'
        else:
            interpretation['community_ambulation'] = '가정 내 보행만 가능'

        if speed_mps < 0.6:
            interpretation['fall_risk'] = '높음'
            interpretation['recommendations'].append('낙상 예방 프로그램 권장')
        elif speed_mps < 0.8:
            interpretation['fall_risk'] = '중간'
            interpretation['recommendations'].append('균형 훈련 고려')
        else:
            interpretation['fall_risk'] = '낮음'

        if cadence_spm:
            if cadence_spm < 80:
                interpretation['recommendations'].append('보행 리듬 훈련 권장')
            elif cadence_spm > 140:
                interpretation['recommendations'].append('과도한 보행 속도 주의')

        return interpretation
