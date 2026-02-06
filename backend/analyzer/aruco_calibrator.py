# analyzer/aruco_calibrator.py

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from .config import AnalyzerConfig


class ArucoCalibrator:
    """ArUco 마커 기반 보정 + 진행률 방식 타이밍 (횡방향 분석용)

    호모그래피 대신 **진행률(progress)** 방식:
    - 시작마커 중심 → 끝마커 중심 연결 벡터에 발 위치를 투영
    - progress=0 → 시작선, progress=1 → 끝선

    횡방향 분석 특화:
    - 카메라가 측면에서 촬영 (사람이 좌→우 또는 우→좌로 이동)
    - X좌표 기반 진행률 계산이 주축
    - 깊이(Z) 대신 X축 이동을 추적

    정적 카메라 가정: 마커가 서로 다른 프레임에서 감지되어도 보정 가능
    """

    def __init__(
        self,
        marker_size_m: float = 0.2,
        start_marker_id: int = 0,
        finish_marker_id: int = 1,
        marker_distance_m: float = 10.0,
        corridor_width_m: float = 2.0,
    ):
        self.marker_size_m = marker_size_m
        self.start_marker_id = start_marker_id
        self.finish_marker_id = finish_marker_id
        self.marker_distance_m = marker_distance_m
        self.corridor_width_m = corridor_width_m

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # 마커 3D 좌표 (마커 좌표계, 마커 중심이 원점)
        half = marker_size_m / 2
        self.marker_3d_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        # 카메라 행렬 (영상 로드 시 설정)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        # 보정 상태
        self.calibrated: bool = False
        self.pixels_per_meter: float = 80.0

        # 마커 이미지 좌표 (진행률 계산용)
        self.start_center_px: Optional[Tuple[float, float]] = None
        self.finish_center_px: Optional[Tuple[float, float]] = None
        self.direction_vec: Optional[Tuple[float, float]] = None  # start→finish 방향
        self.direction_length_sq: float = 0.0  # |direction|^2

        # 프레임 간 마커 저장 (정적 카메라에서 서로 다른 프레임의 마커 사용)
        self._stored_markers: Dict[str, Dict] = {}

        # 횡방향 분석: X축 기반 진행률 사용
        self._use_x_progress: bool = True  # 횡방향은 항상 X축 기반
        self._start_x: float = 0.0
        self._finish_x: float = 0.0

        # 보행 방향 (자동 감지)
        self._walking_direction: Optional[str] = None
        self._initial_progress: Optional[float] = None
        self._progress_samples: List[float] = []

    def set_camera_params(self, frame_width: int, frame_height: int):
        """영상 해상도에 맞는 카메라 행렬 설정"""
        focal_length = frame_width * 0.65
        cx = frame_width / 2
        cy = frame_height / 2
        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def detect_markers(self, frame: np.ndarray) -> Dict:
        """단일 프레임에서 ArUco 마커 감지"""
        if self.camera_matrix is None:
            h, w = frame.shape[:2]
            self.set_camera_params(w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, rejected = self.detector.detectMarkers(gray)

        markers = []
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                corners_2d = corners_list[i][0]  # shape: (4, 2)

                # solvePnP로 rvec/tvec 계산
                success, rvec, tvec = cv2.solvePnP(
                    self.marker_3d_points,
                    corners_2d.astype(np.float64),
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if not success:
                    continue

                rvec = rvec.flatten()
                tvec = tvec.flatten()
                distance = float(np.linalg.norm(tvec))

                # 마커 역할 결정
                role = 'unknown'
                if int(marker_id) == self.start_marker_id:
                    role = 'start'
                elif int(marker_id) == self.finish_marker_id:
                    role = 'finish'

                # 마커 중심 (4 코너 평균)
                center = corners_2d.mean(axis=0).tolist()

                # 마커 하단 중심 (corners[2]와 corners[3]의 중점)
                bottom_center = (
                    (corners_2d[2] + corners_2d[3]) / 2
                ).tolist()

                marker_data = {
                    'id': int(marker_id),
                    'corners': corners_2d.tolist(),
                    'rvec': rvec.tolist(),
                    'tvec': tvec.tolist(),
                    'distance': round(distance, 3),
                    'role': role,
                    'center': center,
                    'bottom_center': bottom_center,
                }
                markers.append(marker_data)

                # 프레임 간 저장 (첫 감지 시 저장, 정적 카메라 가정)
                if role in ('start', 'finish') and role not in self._stored_markers:
                    self._stored_markers[role] = marker_data
                    print(f"[ArUco] Stored {role} marker (ID {marker_id}) "
                          f"center=({center[0]:.0f},{center[1]:.0f})")

        return {
            'markers': markers,
            'num_markers': len(markers),
        }

    def try_calibrate(self) -> bool:
        """저장된 마커로 보정 시도"""
        if self.calibrated:
            return True

        start_m = self._stored_markers.get('start')
        finish_m = self._stored_markers.get('finish')

        if start_m is None or finish_m is None:
            return False

        return self._do_calibration(start_m, finish_m)

    def _do_calibration(self, start_marker: Dict, finish_marker: Dict) -> bool:
        """두 마커로 진행률 벡터 계산 (횡방향 분석)

        횡방향 분석: X좌표 기반 진행률 계산
        - 마커가 좌/우로 배치됨
        - 사람이 좌→우 또는 우→좌로 이동
        """
        # 마커 하단 중심 (2D 픽셀 좌표)
        sx, sy = start_marker['bottom_center']
        fx, fy = finish_marker['bottom_center']

        self.start_center_px = (sx, sy)
        self.finish_center_px = (fx, fy)

        # 3D 위치 (solvePnP tvec) 사용하여 방향 확인
        s_tvec = np.array(start_marker['tvec'])
        f_tvec = np.array(finish_marker['tvec'])

        # tvec: [x, y, z] - z가 카메라로부터의 깊이
        x_diff_3d = abs(s_tvec[0] - f_tvec[0])
        depth_diff = abs(s_tvec[2] - f_tvec[2])

        print(f"[ArUco] 3D positions: start={s_tvec}, finish={f_tvec}")
        print(f"[ArUco] X diff: {x_diff_3d:.2f}m, Depth diff: {depth_diff:.2f}m")

        # 2D 방향 벡터
        dx = fx - sx
        dy = fy - sy
        self.direction_vec = (dx, dy)
        self.direction_length_sq = dx * dx + dy * dy

        # 픽셀/미터 비율
        pixel_dist = np.sqrt(self.direction_length_sq)
        if pixel_dist < 10:
            print(f"[ArUco] Calibration failed: markers too close")
            return False
        self.pixels_per_meter = pixel_dist / self.marker_distance_m

        # 횡방향 분석: X좌표 기반 (측면 카메라)
        self._use_x_progress = True
        self._start_x = float(sx)
        self._finish_x = float(fx)

        # X축 이동 방향 확인
        if abs(dx) > abs(dy):
            print(f"[ArUco] Using X-axis progress (lateral camera) - Primary axis")
        else:
            print(f"[ArUco] Using X-axis progress (lateral camera) - Y is larger but forcing X")
            # 횡방향이므로 Y가 커도 X를 사용

        self.calibrated = True
        print(f"[ArUco] Calibrated! start=({sx:.0f},{sy:.0f}) "
              f"finish=({fx:.0f},{fy:.0f}) "
              f"pixel_dist={pixel_dist:.0f} ppm={self.pixels_per_meter:.1f}")
        print(f"[ArUco] X range: {self._start_x:.0f} -> {self._finish_x:.0f}")
        return True

    def add_perspective_sample(self, ankle_y: float, person_height_px: float):
        """횡방향 분석에서는 원근 보정이 덜 중요 (측면에서 촬영)

        측면 카메라에서는 사람이 카메라와 거의 일정한 거리를 유지하므로
        복잡한 원근 보정이 필요 없음.
        """
        pass  # 횡방향 분석에서는 사용하지 않음

    def get_walking_progress(self, foot_x: float, foot_y: float,
                             person_height_px: float = 0) -> Optional[float]:
        """발 위치의 진행률 계산 (0=시작, 1=끝) - 횡방향 분석

        횡방향 분석:
        - X좌표 기반 진행률 계산
        - 시작 마커 X → 끝 마커 X 사이에서 발의 X좌표 비율
        """
        if not self.calibrated:
            return None

        # X좌표 기반 진행률 계산
        x_range = self._finish_x - self._start_x
        if abs(x_range) < 1:
            return None

        # 발의 X좌표를 시작-끝 범위에 투영
        t = (foot_x - self._start_x) / x_range
        return float(t)

    def get_distance_from_start(self, progress: float) -> float:
        """진행률에서 시작선으로부터의 거리(m)"""
        return progress * self.marker_distance_m

    def check_progress_crossing(
        self,
        prev_progress: float,
        curr_progress: float,
    ) -> Optional[str]:
        """진행률 기반 라인 통과 감지 (양방향 자동 감지)

        Phase 1: 초기 위치를 수집하여 보행 방향 자동 결정
          - 최소 8프레임 수집 후 중앙값으로 진입 방향 판별
          - 노이즈/급격한 점프 필터링
        Phase 2: 결정된 방향에 따라 START/FINISH 라인 통과 감지
          - forward (0→1): 0=START, 1=FINISH
          - reverse (1→0): 1=START, 0=FINISH

        Returns:
            'start' | 'finish' | None
        """
        if not self.calibrated:
            return None

        # ── Phase 1: 방향 자동 감지 (초기 위치 수집) ──
        if self._walking_direction is None:
            # 노이즈 필터: 극단적인 progress 값 무시
            if abs(curr_progress) > 2.0 or abs(prev_progress) > 2.0:
                return None
            # 급격한 점프 무시 (프레임 간 0.3 이상 변화는 노이즈)
            if abs(curr_progress - prev_progress) > 0.3:
                return None

            self._progress_samples.append(curr_progress)

            MIN_SAMPLES = 8
            if len(self._progress_samples) < MIN_SAMPLES:
                return None

            # 초기 위치의 중앙값으로 진입 방향 결정
            initial_median = float(np.median(self._progress_samples[:5]))
            recent_median = float(np.median(self._progress_samples[-3:]))

            if initial_median < 0.5:
                self._walking_direction = 'forward'  # 0→1
            else:
                self._walking_direction = 'reverse'  # 1→0

            print(f"[ArUco] Walking direction: {self._walking_direction} "
                  f"(initial={initial_median:.3f}, recent={recent_median:.3f})")

            # 방향 결정 시점에서 이미 START를 지났는지 확인 (retroactive)
            if self._walking_direction == 'forward' and curr_progress > 0.0:
                print(f"[ArUco] Retroactive START (forward): "
                      f"progress={curr_progress:.3f} > 0.0")
                return 'start'
            elif self._walking_direction == 'reverse' and curr_progress < 1.0:
                print(f"[ArUco] Retroactive START (reverse): "
                      f"progress={curr_progress:.3f} < 1.0")
                return 'start'

            return None

        # ── Phase 2: 라인 통과 감지 ──
        crossed_0 = (prev_progress <= 0.0 < curr_progress) or \
                     (prev_progress >= 0.0 > curr_progress)
        crossed_1 = (prev_progress <= 1.0 < curr_progress) or \
                     (prev_progress >= 1.0 > curr_progress)

        if not crossed_0 and not crossed_1:
            return None

        if self._walking_direction == 'forward':
            # 0→1: progress=0이 START, progress=1이 FINISH
            if crossed_0:
                return 'start'
            if crossed_1:
                return 'finish'
        else:  # reverse
            # 1→0: progress=1이 START, progress=0이 FINISH
            if crossed_1:
                return 'start'
            if crossed_0:
                return 'finish'

        return None

    def get_calibration_info(self) -> Dict:
        """현재 보정 상태 정보"""
        return {
            'calibrated': self.calibrated,
            'marker_size_m': self.marker_size_m,
            'marker_distance_m': self.marker_distance_m,
            'start_marker_id': self.start_marker_id,
            'finish_marker_id': self.finish_marker_id,
            'pixels_per_meter': self.pixels_per_meter,
            'start_center_px': self.start_center_px,
            'finish_center_px': self.finish_center_px,
            'stored_markers': list(self._stored_markers.keys()),
            'analysis_mode': 'lateral',  # 횡방향 분석
            'start_x': self._start_x if hasattr(self, '_start_x') else None,
            'finish_x': self._finish_x if hasattr(self, '_finish_x') else None,
        }
