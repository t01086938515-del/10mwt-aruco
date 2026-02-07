# backend/processor.py - Updated with stride L/R debug
"""프레임별 분석 코디네이터 (횡방향 분석용)

MediaPipe Pose Heavy + ArUco 마커 → 진행률(progress) 기반 타이밍.
사람의 발 X좌표를 시작마커→끝마커 벡터에 투영하여 0~1 진행률 산출.

횡방향 분석:
- 카메라가 측면에서 촬영 (사람이 좌→우 또는 우→좌로 이동)
- X좌표 기반 진행률 계산
- 케이던스/보폭 추정 (발목 좌표 분석)
"""

import cv2
import numpy as np
import time
import os
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from scipy import signal

from analyzer.aruco_calibrator import ArucoCalibrator
from analyzer.pose_detector import PoseDetector
from analyzer.gait_analyzer import GaitAnalyzer
from analyzer.gait_judgment import judge_all
from analyzer.calibration import DistanceCalibrator
from analyzer.filter import KalmanFilter2D
from analyzer.solution_2_homography import PerspectiveCorrector
from utils.video_utils import VideoReader


class FrameProcessor:
    """프레임별 분석 코디네이터 (횡방향 분석)"""

    def __init__(
        self,
        model_path: str = "",
        marker_size_m: float = 0.2,
        start_marker_id: int = 0,
        finish_marker_id: int = 1,
        marker_distance_m: float = 10.0,
    ):
        # ArUco 보정기
        self.aruco = ArucoCalibrator(
            marker_size_m=marker_size_m,
            start_marker_id=start_marker_id,
            finish_marker_id=finish_marker_id,
            marker_distance_m=marker_distance_m,
        )

        # MediaPipe Pose Heavy (단일 사람 감지)
        self.pose_detector = PoseDetector(
            model_path=model_path,
            confidence_threshold=0.3,
        )

        # 칼만 필터 (process_noise 높여서 빠른 응답)
        self.kalman = KalmanFilter2D(process_noise=0.05, measurement_noise=0.1)

        # 거리 보정기 + 임상 분석
        self.calibrator = DistanceCalibrator()
        self.gait_analyzer: Optional[GaitAnalyzer] = None

        # 원근 보정 (#2 Homography)
        self.perspective_corrector = PerspectiveCorrector()
        self._first_frame: Optional[np.ndarray] = None

        # 상태
        self.prev_progress: Optional[float] = None
        self.timer_state = 'standby'  # standby | running | finished
        self.timer_start_time_s: float = 0.0
        self.timer_elapsed_s: float = 0.0
        self.analysis_results: Optional[Dict] = None

        # 크로싱 이벤트 기록
        self.crossing_events: List[Dict] = []

        # 케이던스/보폭 추정을 위한 발목 좌표 기록
        self.ankle_history: List[Dict] = []  # [{timestamp_s, left_ankle, right_ankle, left_y, right_y}]
        self.fps: float = 30.0

        # 디버그
        self._debug_frame_count = 0

    def reset(self):
        """상태 초기화"""
        self.pose_detector.reset_tracking()
        self.kalman = KalmanFilter2D(process_noise=0.05, measurement_noise=0.1)
        self.prev_progress = None
        self.timer_state = 'standby'
        self.timer_start_time_s = 0.0
        self.timer_elapsed_s = 0.0
        self.analysis_results = None
        self.crossing_events = []
        self.ankle_history = []
        self._debug_frame_count = 0
        self.perspective_corrector = PerspectiveCorrector()
        self._first_frame = None
        # 보행 방향 초기화
        if hasattr(self.aruco, '_walking_direction'):
            self.aruco._walking_direction = None
            self.aruco._progress_samples = []
            self.aruco._initial_progress = None

    def set_fps(self, fps: float):
        """영상 FPS 설정"""
        self.fps = fps

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
    ) -> Dict:
        """단일 프레임 분석 (횡방향)"""
        result = {
            'frame_idx': frame_idx,
            'timestamp_s': round(timestamp_s, 4),
            'markers': [],
            'num_markers': 0,
            'person': None,
            'timer': {
                'state': self.timer_state,
                'elapsed_s': round(self.timer_elapsed_s, 3),
            },
            'crossing_event': None,
            'calibration': {'calibrated': self.aruco.calibrated},
            'results': None,
        }

        # 첫 프레임 저장 (원근 보정용)
        if self._first_frame is None:
            self._first_frame = frame.copy()

        # 1) ArUco 마커 감지 (보정 완료 후에는 매 30프레임만 - 성능 최적화)
        run_aruco = (not self.aruco.calibrated) or (frame_idx % 30 == 0)
        if run_aruco:
            marker_result = self.aruco.detect_markers(frame)
            result['markers'] = marker_result['markers']
            result['num_markers'] = marker_result['num_markers']
            self._last_markers = marker_result
        elif hasattr(self, '_last_markers'):
            result['markers'] = self._last_markers['markers']
            result['num_markers'] = self._last_markers['num_markers']

        # 2) 보정 시도
        if not self.aruco.calibrated:
            if self.aruco.try_calibrate():
                result['calibration'] = self.aruco.get_calibration_info()
                print(f"[Processor] Calibration done at frame {frame_idx} ({timestamp_s:.2f}s)")
                # 원근 보정기 캘리브레이션 (#2 Homography)
                if self._first_frame is not None:
                    marker_info = {}
                    sc = self.aruco.start_center_px
                    fc = self.aruco.finish_center_px
                    if sc and fc:
                        marker_info[0] = {'cx': sc[0], 'cy': sc[1], 'size': 20}
                        marker_info[1] = {'cx': fc[0], 'cy': fc[1], 'size': 20}
                        self.perspective_corrector.calibrate(
                            self._first_frame, marker_info,
                            actual_distance_m=self.aruco.marker_distance_m
                        )

        # 3) MediaPipe Pose 감지 (단일 사람)
        pose_result = self.pose_detector.detect(frame)
        tracks = pose_result['tracks']

        best_track = tracks[0] if tracks else None

        if best_track is not None and best_track['ankle_center'] is not None:
            person = {
                'detected': True,
                'track_id': best_track.get('track_id', 0),
                'bbox': best_track['bbox'],
                'keypoints': best_track['keypoints'],
                'ankle_center': best_track['ankle_center'],
                'hip_center': best_track.get('hip_center'),
                'ground_point': None,
                'progress': None,
                'distance_from_start_m': None,
            }

            # 발 접지점 (MediaPipe 발뒤꿈치/발끝 기반)
            ground_point = best_track['ankle_center']
            person['ground_point'] = ground_point

            # 칼만 필터 적용
            filtered = self.kalman.update((ground_point[0], ground_point[1]))
            filtered_ground = [float(filtered[0]), float(filtered[1])]

            # 양발 발목/뒤꿈치/발끝 좌표 수집 (케이던스/보폭/스윙 분석용)
            keypoints = best_track['keypoints']
            # MediaPipe Pose 인덱스:
            # 27: left_ankle, 28: right_ankle
            # 29: left_heel, 30: right_heel
            # 31: left_foot_index (발끝), 32: right_foot_index (발끝)
            left_ankle = keypoints[27] if len(keypoints) > 27 else None
            right_ankle = keypoints[28] if len(keypoints) > 28 else None
            left_heel = keypoints[29] if len(keypoints) > 29 else None
            right_heel = keypoints[30] if len(keypoints) > 30 else None
            left_toe = keypoints[31] if len(keypoints) > 31 else None
            right_toe = keypoints[32] if len(keypoints) > 32 else None

            if left_ankle and right_ankle and left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
                history_entry = {
                    'timestamp_s': timestamp_s,
                    'frame_idx': frame_idx,
                    'left_x': left_ankle[0],
                    'left_y': left_ankle[1],
                    'right_x': right_ankle[0],
                    'right_y': right_ankle[1],
                }
                # heel 좌표 추가 (신뢰도 체크)
                if left_heel and left_heel[2] > 0.3:
                    history_entry['left_heel_x'] = left_heel[0]
                    history_entry['left_heel_y'] = left_heel[1]
                if right_heel and right_heel[2] > 0.3:
                    history_entry['right_heel_x'] = right_heel[0]
                    history_entry['right_heel_y'] = right_heel[1]
                # toe (foot_index) 좌표 추가 - 스윙 감지에 중요
                if left_toe and left_toe[2] > 0.3:
                    history_entry['left_toe_x'] = left_toe[0]
                    history_entry['left_toe_y'] = left_toe[1]
                if right_toe and right_toe[2] > 0.3:
                    history_entry['right_toe_x'] = right_toe[0]
                    history_entry['right_toe_y'] = right_toe[1]

                self.ankle_history.append(history_entry)

            # 진행률 계산 (0=시작, 1=끝) - 횡방향은 X좌표 기반
            if self.aruco.calibrated:
                # bbox 높이 (사람 키 pixel) - 횡방향에서는 참고용
                bbox = best_track['bbox']
                person_height_px = bbox[3] - bbox[1]

                # 횡방향 진행률 계산 (X좌표 기반)
                progress = self.aruco.get_walking_progress(
                    filtered_ground[0], filtered_ground[1],
                    person_height_px=person_height_px,
                )

                if progress is not None:
                    person['progress'] = round(progress, 4)
                    dist_m = self.aruco.get_distance_from_start(progress)
                    person['distance_from_start_m'] = round(dist_m, 3)

                    # 디버그 로그
                    self._debug_frame_count += 1
                    if self._debug_frame_count <= 20 or self._debug_frame_count % 20 == 0:
                        print(f"[Processor] frame={frame_idx} "
                              f"foot_x={filtered_ground[0]:.0f} "
                              f"progress={progress:.3f} dist={dist_m:.2f}m "
                              f"timer={self.timer_state}")

                    # 라인 통과 감지 (진행률 기반)
                    if self.prev_progress is not None:
                        crossing = self.aruco.check_progress_crossing(
                            self.prev_progress, progress
                        )
                        if crossing == 'start' and self.timer_state == 'standby':
                            self.timer_state = 'running'
                            self.timer_start_time_s = timestamp_s
                            result['crossing_event'] = 'start'
                            self.crossing_events.append({
                                'line': 'start',
                                'timestamp_s': round(timestamp_s, 4),
                                'frame_idx': frame_idx,
                                'distance_m': round(dist_m, 3),
                            })
                            print(f"[Processor] ★★★ START at frame {frame_idx}, "
                                  f"t={timestamp_s:.2f}s, progress={progress:.3f}")
                        elif crossing == 'finish' and self.timer_state == 'running':
                            self.timer_elapsed_s = timestamp_s - self.timer_start_time_s
                            self.timer_state = 'finished'
                            result['crossing_event'] = 'finish'
                            self.crossing_events.append({
                                'line': 'finish',
                                'timestamp_s': round(timestamp_s, 4),
                                'frame_idx': frame_idx,
                                'elapsed_s': round(self.timer_elapsed_s, 3),
                                'distance_m': round(dist_m, 3),
                            })
                            try:
                                self._compute_results()
                                result['results'] = self.analysis_results
                            except Exception as e:
                                import traceback
                                print(f"[Processor] _compute_results error: {e}")
                                traceback.print_exc()
                                result['results'] = None
                            print(f"[Processor] ★★★ FINISH at frame {frame_idx}, "
                                  f"t={timestamp_s:.2f}s, elapsed={self.timer_elapsed_s:.3f}s")

                    self.prev_progress = progress

            result['person'] = person

        # 타이머 업데이트
        if self.timer_state == 'running':
            self.timer_elapsed_s = timestamp_s - self.timer_start_time_s
        result['timer'] = {
            'state': self.timer_state,
            'elapsed_s': round(self.timer_elapsed_s, 3),
        }

        return result

    def _analyze_gait_parameters(self) -> Dict:
        """상세 보행 분석 - 횡방향 영상에서 발목 좌표 분석

        분석 항목:
        - 보행 시간 (step time, stride time)
        - 보행 속도 (gait speed)
        - 보폭 (step length) - 한 발 디딤 거리
        - 활보장 (stride length) - 같은 발 두 번 닿는 거리
        - 양측 대칭성 (SI index)
        - 스윙/스탠스 비율

        Returns:
            Dict with all gait parameters
        """
        result = {
            'step_count': None,
            'cadence_spm': None,
            'step_length_m': None,
            'stride_length_m': None,
            # 좌우 활보장 (stride = 같은 발 연속 착지 거리)
            'left_stride_length_m': None,
            'right_stride_length_m': None,
            'stride_length_si': None,
            'step_time_s': None,
            'stride_time_s': None,
            'left_stride_time_s': None,
            'right_stride_time_s': None,
            'stride_time_si': None,
            # 좌우 보폭
            'left_step_length_m': None,
            'right_step_length_m': None,
            'step_length_si': None,
            # 좌우 스텝 시간
            'left_step_time_s': None,
            'right_step_time_s': None,
            'step_time_si': None,
            # 좌우 스윙 시간
            'left_swing_time_s': None,
            'right_swing_time_s': None,
            'swing_time_si': None,
            # 좌우 스탠스 시간
            'left_stance_time_s': None,
            'right_stance_time_s': None,
            'stance_time_si': None,
            # 좌우 스윙/스탠스 비율
            'left_swing_stance_ratio': None,
            'right_swing_stance_ratio': None,
            'swing_stance_si': None,
            # 스윙/스탠스 백분율 (정상: Swing 40%, Stance 60%)
            'left_swing_pct': None,
            'right_swing_pct': None,
            'left_stance_pct': None,
            'right_stance_pct': None,
            # 종합 대칭성
            'overall_symmetry_index': None,
            'step_events': [],
        }

        if len(self.ankle_history) < 20:
            return result

        # START~FINISH 구간 데이터
        start_ev = next((e for e in self.crossing_events if e['line'] == 'start'), None)
        finish_ev = next((e for e in self.crossing_events if e['line'] == 'finish'), None)

        if not start_ev or not finish_ev:
            return result

        start_t = start_ev['timestamp_s']
        finish_t = finish_ev['timestamp_s']
        elapsed = finish_t - start_t

        if elapsed <= 0:
            return result

        # 구간 내 데이터 필터링
        segment = [h for h in self.ankle_history if start_t <= h['timestamp_s'] <= finish_t]
        if len(segment) < 10:
            return result

        times = np.array([h['timestamp_s'] for h in segment])
        left_x = np.array([h['left_x'] for h in segment])
        right_x = np.array([h['right_x'] for h in segment])
        left_y = np.array([h['left_y'] for h in segment])
        right_y = np.array([h['right_y'] for h in segment])

        # 스무딩
        kernel = np.ones(5) / 5
        if len(left_x) > 5:
            left_x_smooth = np.convolve(left_x, kernel, mode='same')
            right_x_smooth = np.convolve(right_x, kernel, mode='same')
            left_y_smooth = np.convolve(left_y, kernel, mode='same')
            right_y_smooth = np.convolve(right_y, kernel, mode='same')
        else:
            left_x_smooth, right_x_smooth = left_x, right_x
            left_y_smooth, right_y_smooth = left_y, right_y

        # ═══ Step Detection: 적응형 3중 필터 Heel Strike 감지 ═══
        # 필터 1: heel Y 로컬 최대값 (발뒤꿈치가 바닥에 닿는 순간)
        # 필터 2: heel X 속도 ≈ 0 (실제 HS에서는 해당 발의 수평 이동 정지)
        # 필터 3: 최소 간격 0.35초 (정상 보행 기준 false positive 제거)

        effective_fps = len(times) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else self.fps
        min_distance = max(3, int(effective_fps * 0.3))

        # heel X/Y 좌표 추출 및 스무딩
        left_heel_x_raw = np.array([h.get('left_heel_x', h['left_x']) for h in segment])
        left_heel_y_raw = np.array([h.get('left_heel_y', h['left_y']) for h in segment])
        right_heel_x_raw = np.array([h.get('right_heel_x', h['right_x']) for h in segment])
        right_heel_y_raw = np.array([h.get('right_heel_y', h['right_y']) for h in segment])
        if len(left_heel_y_raw) > 5:
            left_heel_x_sm = np.convolve(left_heel_x_raw, kernel, mode='same')
            left_heel_y_sm = np.convolve(left_heel_y_raw, kernel, mode='same')
            right_heel_x_sm = np.convolve(right_heel_x_raw, kernel, mode='same')
            right_heel_y_sm = np.convolve(right_heel_y_raw, kernel, mode='same')
        else:
            left_heel_x_sm, left_heel_y_sm = left_heel_x_raw, left_heel_y_raw
            right_heel_x_sm, right_heel_y_sm = right_heel_x_raw, right_heel_y_raw

        def detect_heel_strikes(heel_x_smooth, heel_y_smooth, times_arr, min_dist, min_gap_s=0.35):
            """적응형 3중 필터 Heel Strike 감지

            Args:
                heel_x_smooth: 스무딩된 발꿈치 X좌표 배열
                heel_y_smooth: 스무딩된 발꿈치 Y좌표 배열
                times_arr: 시간 배열 (초)
                min_dist: find_peaks 최소 거리 (프레임)
                min_gap_s: 같은 발 연속 HS 최소 간격 (초)

            Returns:
                heel strike 프레임 인덱스 리스트
            """
            if len(heel_y_smooth) < 5:
                return []

            # ── 필터 1: heel Y 로컬 최대값 (화면 좌표: Y↑ = 아래 = 바닥) ──
            peaks, _ = signal.find_peaks(heel_y_smooth, distance=min_dist, prominence=3)
            if len(peaks) == 0:
                return []

            # ── 필터 2: heel X 속도 필터 (접지 시 수평 이동 ≈ 0) ──
            heel_vx = np.gradient(heel_x_smooth, times_arr)
            abs_heel_vx = np.abs(heel_vx)
            vx_median = np.median(abs_heel_vx)
            vx_threshold = vx_median * 2.0  # 중앙값의 2배 이하만 유지

            filtered_peaks = []
            for idx in peaks:
                if abs_heel_vx[idx] <= vx_threshold:
                    filtered_peaks.append(idx)

            # ── 필터 3: 최소 시간 간격 (0.35초) ──
            final_peaks = []
            for idx in filtered_peaks:
                if final_peaks:
                    dt = times_arr[idx] - times_arr[final_peaks[-1]]
                    if dt < min_gap_s:
                        continue
                final_peaks.append(idx)

            return final_peaks

        step_events = []
        try:
            left_hs = detect_heel_strikes(left_heel_x_sm, left_heel_y_sm, times, min_distance)
            right_hs = detect_heel_strikes(right_heel_x_sm, right_heel_y_sm, times, min_distance)

            print(f"[Processor] Adaptive 3-filter HS detection: L={len(left_hs)}, R={len(right_hs)}, min_dist={min_distance}", flush=True)

            # 모든 HS 이벤트를 시간순으로 합치기
            all_hs = []
            for idx in left_hs:
                if idx < len(times):
                    all_hs.append({
                        'time': float(times[idx]),
                        'frame_idx': segment[idx]['frame_idx'],
                        'leading_foot': 'left',
                        'left_x': float(left_x[idx]),
                        'right_x': float(right_x[idx]),
                        'left_y': float(left_y[idx]),
                        'right_y': float(right_y[idx]),
                    })
            for idx in right_hs:
                if idx < len(times):
                    all_hs.append({
                        'time': float(times[idx]),
                        'frame_idx': segment[idx]['frame_idx'],
                        'leading_foot': 'right',
                        'left_x': float(left_x[idx]),
                        'right_x': float(right_x[idx]),
                        'left_y': float(left_y[idx]),
                        'right_y': float(right_y[idx]),
                    })

            all_hs.sort(key=lambda e: e['time'])

            # ── START 직후 진입 중 스텝 제외 ──
            step_buffer_s = 0.3
            all_hs = [e for e in all_hs if e['time'] >= start_t + step_buffer_s]
            print(f"[Processor] After start buffer ({step_buffer_s}s): {len(all_hs)} heel strikes", flush=True)

            step_events = all_hs
        except Exception as e:
            print(f"[Processor] Adaptive HS detection error: {e}", flush=True)

        # 폴백: X좌표 교차 방식
        if len(step_events) < 2:
            diff_x = left_x_smooth - right_x_smooth
            sign_changes = np.where(np.diff(np.sign(diff_x)))[0]
            for idx in sign_changes:
                if idx < len(times):
                    leading_foot = 'left' if diff_x[idx] > 0 else 'right'
                    t_val = float(times[idx])
                    if t_val >= start_t + 0.3:  # 동일 버퍼 적용
                        step_events.append({
                            'time': t_val,
                            'frame_idx': segment[idx]['frame_idx'],
                            'leading_foot': leading_foot,
                            'left_x': float(left_x[idx]),
                            'right_x': float(right_x[idx]),
                            'left_y': float(left_y[idx]),
                            'right_y': float(right_y[idx]),
                        })
            print(f"[Processor] X-crossing fallback: {len(step_events)} steps", flush=True)

        step_count = len(step_events)
        result['step_events'] = step_events
        result['step_count'] = step_count
        print(f"[Processor] Total step count: {step_count}", flush=True)

        if step_count < 2:
            return result

        # ═══ 기본 보행 지표 계산 ═══
        distance_m = self.aruco.marker_distance_m
        ppm = self.aruco.pixels_per_meter if self.aruco.pixels_per_meter else 100
        print(f"[Processor] Homography calibrated: {self.perspective_corrector.calibrated}", flush=True)

        # 케이던스 (steps per minute)
        cadence = (step_count / elapsed) * 60
        if 30 <= cadence <= 200:
            result['cadence_spm'] = round(cadence, 1)

        # 평균 보폭 (step length) - 연속된 step 사이 거리
        step_length = distance_m / step_count if step_count > 0 else None
        if step_length and 0.2 <= step_length <= 1.5:
            result['step_length_m'] = round(step_length, 3)

        # 활보장 (stride length) - 같은 발 2회 = 2 steps
        stride_count = step_count // 2
        stride_length = distance_m / stride_count if stride_count > 0 else None
        if stride_length and 0.4 <= stride_length <= 2.5:
            result['stride_length_m'] = round(stride_length, 3)

        # 좌우 활보장 개별 계산 (같은 발 연속 착지 간 거리)
        print(f"[Processor] === Stride L/R Calculation ===")
        print(f"[Processor] Step events count: {len(step_events)}, ppm: {ppm}")

        # 최소 3개 이벤트만 있어도 계산 시도 (조건 완화)
        if len(step_events) >= 3:
            left_strides = []
            right_strides = []
            all_strides_debug = []

            # 같은 발의 연속 스텝 간 거리 계산
            for i in range(len(step_events) - 2):
                curr = step_events[i]
                # 2스텝 후 = 같은 발 (left -> right -> left)
                next_same = step_events[i + 2]

                if curr['leading_foot'] == next_same['leading_foot']:
                    # leading_foot에 맞는 발의 좌표 사용
                    if curr['leading_foot'] == 'left':
                        x1, y1 = curr['left_x'], curr['left_y']
                        x2, y2 = next_same['left_x'], next_same['left_y']
                    else:
                        x1, y1 = curr['right_x'], curr['right_y']
                        x2, y2 = next_same['right_x'], next_same['right_y']
                    dx = abs(x2 - x1)
                    # Homography 원근 보정 적용
                    if self.perspective_corrector.calibrated:
                        stride_m = self.perspective_corrector.real_distance_x(x1, y1, x2, y2)
                    else:
                        stride_m = dx / ppm

                    all_strides_debug.append({
                        'foot': curr['leading_foot'],
                        'dx_px': dx,
                        'stride_m': stride_m
                    })

                    # 범위 완화: 0.3m ~ 4.0m
                    if 0.3 <= stride_m <= 4.0:
                        if curr['leading_foot'] == 'left':
                            left_strides.append(stride_m)
                        else:
                            right_strides.append(stride_m)

            print(f"[Processor] All strides (debug): {all_strides_debug[:5]}")  # 처음 5개만
            print(f"[Processor] Valid strides - L: {len(left_strides)}, R: {len(right_strides)}")

            if left_strides:
                result['left_stride_length_m'] = round(np.mean(left_strides), 3)
                print(f"[Processor] Left stride: {result['left_stride_length_m']}m (from {len(left_strides)} samples)")
            else:
                print(f"[Processor] Left stride: No valid samples")

            if right_strides:
                result['right_stride_length_m'] = round(np.mean(right_strides), 3)
                print(f"[Processor] Right stride: {result['right_stride_length_m']}m (from {len(right_strides)} samples)")
            else:
                print(f"[Processor] Right stride: No valid samples")

            # Stride SI 계산
            if result['left_stride_length_m'] and result['right_stride_length_m']:
                left_s = result['left_stride_length_m']
                right_s = result['right_stride_length_m']
                si = round(abs(left_s - right_s) / (0.5 * (left_s + right_s)) * 100, 1)
                result['stride_length_si'] = si
                print(f"[Processor] Stride SI: {si}%")
        else:
            print(f"[Processor] Not enough step events for stride calculation (need >= 3)")

        # ═══ 시간 지표 ═══
        if len(step_events) >= 2:
            step_times = []
            for i in range(1, len(step_events)):
                dt = step_events[i]['time'] - step_events[i-1]['time']
                if 0.2 <= dt <= 2.0:  # 합리적인 범위
                    step_times.append(dt)

            if step_times:
                result['step_time_s'] = round(np.mean(step_times), 3)
                result['stride_time_s'] = round(np.mean(step_times) * 2, 3)

        # ═══ SI 계산 헬퍼 함수 ═══
        def calc_si(left_val, right_val):
            """Symmetry Index 계산: |L-R| / (0.5*(L+R)) * 100"""
            if left_val is None or right_val is None:
                return None
            if left_val + right_val <= 0:
                return None
            return round(abs(left_val - right_val) / (0.5 * (left_val + right_val)) * 100, 1)

        # ═══ 좌우 대칭성 분석 ═══
        si_values = []  # 종합 SI 계산용

        if len(step_events) >= 4:
            left_steps = [e for e in step_events if e['leading_foot'] == 'left']
            right_steps = [e for e in step_events if e['leading_foot'] == 'right']

            # 1) 좌우 보폭 (Step Length)
            # Step = 연속된 step event 간 거리 (다른 발 사이)
            # left_step = '왼발이 앞서는 스텝'의 길이 = 이전 event → left leading event
            # right_step = '오른발이 앞서는 스텝'의 길이 = 이전 event → right leading event
            left_step_dists = []
            right_step_dists = []

            for i in range(1, len(step_events)):
                curr = step_events[i]
                prev = step_events[i - 1]
                # 연속된 두 이벤트는 다른 발이어야 step
                if curr['leading_foot'] != prev['leading_foot']:
                    # 두 발의 중점(midpoint) 변화로 step 거리 계산
                    curr_mid_x = (curr['left_x'] + curr['right_x']) / 2
                    prev_mid_x = (prev['left_x'] + prev['right_x']) / 2
                    curr_mid_y = (curr['left_y'] + curr['right_y']) / 2
                    prev_mid_y = (prev['left_y'] + prev['right_y']) / 2
                    dx = abs(curr_mid_x - prev_mid_x)
                    # Homography 원근 보정 적용
                    if self.perspective_corrector.calibrated:
                        step_m = self.perspective_corrector.real_distance_x(
                            prev_mid_x, prev_mid_y, curr_mid_x, curr_mid_y)
                    else:
                        step_m = dx / ppm
                    if 0.2 <= step_m <= 2.0:  # 합리적 범위
                        if curr['leading_foot'] == 'left':
                            left_step_dists.append(step_m)
                        else:
                            right_step_dists.append(step_m)

            print(f"[Processor] Step L/R - L: {len(left_step_dists)} samples, R: {len(right_step_dists)} samples")

            if left_step_dists and right_step_dists:
                left_len = np.mean(left_step_dists)
                right_len = np.mean(right_step_dists)
                result['left_step_length_m'] = round(float(left_len), 3)
                result['right_step_length_m'] = round(float(right_len), 3)
                si = calc_si(float(left_len), float(right_len))
                result['step_length_si'] = si
                if si is not None:
                    si_values.append(si)
                print(f"[Processor] Step L: {result['left_step_length_m']}m, R: {result['right_step_length_m']}m, SI: {si}%")

            # 2) 좌우 스텝 시간 (Step Time = 연속 이벤트 간, 다른 발)
            left_step_times = []
            right_step_times = []
            for i in range(1, len(step_events)):
                curr = step_events[i]
                prev = step_events[i - 1]
                if curr['leading_foot'] != prev['leading_foot']:
                    dt = curr['time'] - prev['time']
                    if 0.2 <= dt <= 2.0:
                        if curr['leading_foot'] == 'left':
                            left_step_times.append(dt)
                        else:
                            right_step_times.append(dt)

            if left_step_times and right_step_times:
                left_time = float(np.mean(left_step_times))
                right_time = float(np.mean(right_step_times))
                result['left_step_time_s'] = round(left_time, 3)
                result['right_step_time_s'] = round(right_time, 3)
                si = calc_si(left_time, right_time)
                result['step_time_si'] = si
                if si is not None:
                    si_values.append(si)

            # 3) 좌우 활보장 시간 (Stride Time = 같은 발 연속, 같은 발)
            def calc_stride_times(steps):
                if len(steps) < 2:
                    return []
                times_list = []
                for i in range(1, len(steps)):
                    dt = steps[i]['time'] - steps[i-1]['time']
                    if 0.4 <= dt <= 4.0:
                        times_list.append(dt)
                return times_list

            left_stride_times = calc_stride_times(left_steps)
            right_stride_times = calc_stride_times(right_steps)

            if left_stride_times and right_stride_times:
                left_st = float(np.mean(left_stride_times))
                right_st = float(np.mean(right_stride_times))
                result['left_stride_time_s'] = round(left_st, 3)
                result['right_stride_time_s'] = round(right_st, 3)
                si = calc_si(left_st, right_st)
                result['stride_time_si'] = si
                if si is not None:
                    si_values.append(si)
                print(f"[Processor] Stride Time L: {result['left_stride_time_s']}s, R: {result['right_stride_time_s']}s, SI: {si}%")

        # ═══ IC/TO 이벤트 기반 스윙/스탠스 분석 (통합 임계값) ═══
        print(f"[Processor] === Starting IC/TO Event-Based Swing/Stance Analysis ===", flush=True)
        try:
            frame_time = 1.0 / self.fps if self.fps > 0 else 1/30

            # toe(foot_index) Y좌표 추출 - 스윙 감지에 ankle보다 효과적
            left_toe_y = np.array([h.get('left_toe_y', h['left_y']) for h in segment])
            right_toe_y = np.array([h.get('right_toe_y', h['right_y']) for h in segment])
            left_heel_y = np.array([h.get('left_heel_y', h['left_y']) for h in segment])
            right_heel_y = np.array([h.get('right_heel_y', h['right_y']) for h in segment])

            # 스무딩
            if len(left_toe_y) > 5:
                left_toe_y_smooth = np.convolve(left_toe_y, kernel, mode='same')
                right_toe_y_smooth = np.convolve(right_toe_y, kernel, mode='same')
                left_heel_y_smooth = np.convolve(left_heel_y, kernel, mode='same')
                right_heel_y_smooth = np.convolve(right_heel_y, kernel, mode='same')
            else:
                left_toe_y_smooth = left_toe_y
                right_toe_y_smooth = right_toe_y
                left_heel_y_smooth = left_heel_y
                right_heel_y_smooth = right_heel_y

            # 발 속도 계산 (X축 속도)
            left_vx = np.gradient(left_x_smooth, times)
            right_vx = np.gradient(right_x_smooth, times)

            # ── 통합 임계값 계산 (좌우 동일하게 적용) ──
            all_vx = np.concatenate([left_vx, right_vx])
            min_step_frames = max(3, int(self.fps * 0.25))

            # ── 통합 지면 기준 Y좌표 계산 ──
            all_toe_y = np.concatenate([left_toe_y_smooth, right_toe_y_smooth])
            all_heel_y = np.concatenate([left_heel_y_smooth, right_heel_y_smooth])
            # 지면 Y = heel/toe의 상위 값 (화면에서 아래쪽 = Y 큰 값이 지면)
            ground_y = np.percentile(all_heel_y, 80)
            # 마진: 발이 지면에서 얼마나 떠야 swing인지 (Y가 ground_y보다 작으면 떠있음)
            swing_margin_px = max(10, (np.max(all_heel_y) - np.min(all_heel_y)) * 0.15)

            print(f"[Processor] Ground Y: {ground_y:.1f}, swing margin: {swing_margin_px:.1f}px", flush=True)

            # ── IC/TO 이벤트 검출 함수 (개선 v3 - 지면 접촉 기반) ──
            def detect_gait_events_v2(heel_y, toe_y, ankle_vx, ground_y, vel_threshold, timestamps):
                """Stance/Swing 판별 - 지면 접촉 기반"""
                events = []

                # 각 프레임의 stance/swing 상태 판별
                # Stance = 발이 지면에 닿아있음 (heel이 ground 근처)
                # Swing = 발이 지면에서 떨어짐 (heel이 ground보다 위쪽 = Y값 작음)
                is_stance_arr = []
                for i in range(len(timestamps)):
                    # heel이 지면 근처에 있으면 stance
                    heel_on_ground = heel_y[i] >= ground_y - swing_margin_px
                    is_stance_arr.append(heel_on_ground)

                is_stance_arr = np.array(is_stance_arr)

                # 상태 변화 감지
                for i in range(1, len(is_stance_arr)):
                    if is_stance_arr[i] and not is_stance_arr[i-1]:
                        # Swing -> Stance = Initial Contact (착지)
                        events.append({'type': 'IC', 'time': timestamps[i], 'frame': i})
                    elif not is_stance_arr[i] and is_stance_arr[i-1]:
                        # Stance -> Swing = Toe Off (발 떼기)
                        events.append({'type': 'TO', 'time': timestamps[i], 'frame': i})

                # stance/swing 프레임 수 계산
                stance_frames = np.sum(is_stance_arr)
                swing_frames = len(is_stance_arr) - stance_frames

                return events, stance_frames, swing_frames

            # 양발 각각 이벤트 검출
            left_events, left_stance_frames, left_swing_frames = detect_gait_events_v2(
                left_heel_y_smooth, left_toe_y_smooth, left_vx,
                ground_y, 0, times
            )
            right_events, right_stance_frames, right_swing_frames = detect_gait_events_v2(
                right_heel_y_smooth, right_toe_y_smooth, right_vx,
                ground_y, 0, times
            )

            total_frames = len(times)
            print(f"[Processor] Events - L: {len(left_events)}, R: {len(right_events)}", flush=True)
            print(f"[Processor] Frames - L stance:{left_stance_frames} swing:{left_swing_frames}, R stance:{right_stance_frames} swing:{right_swing_frames}", flush=True)

            # ── Swing % / Stance % 계산 (백분율) ──
            left_swing_pct = (left_swing_frames / total_frames) * 100 if total_frames > 0 else 0
            left_stance_pct = (left_stance_frames / total_frames) * 100 if total_frames > 0 else 0
            right_swing_pct = (right_swing_frames / total_frames) * 100 if total_frames > 0 else 0
            right_stance_pct = (right_stance_frames / total_frames) * 100 if total_frames > 0 else 0

            print(f"[Processor] Swing % - L: {left_swing_pct:.1f}%, R: {right_swing_pct:.1f}%", flush=True)
            print(f"[Processor] Stance % - L: {left_stance_pct:.1f}%, R: {right_stance_pct:.1f}%", flush=True)

            # 3) 좌우 스윙 시간 (프레임 기반)
            left_swing_time = (left_swing_frames * frame_time) / max(1, step_count // 2)
            right_swing_time = (right_swing_frames * frame_time) / max(1, step_count // 2)

            if 0.1 <= left_swing_time <= 1.5 and 0.1 <= right_swing_time <= 1.5:
                result['left_swing_time_s'] = round(left_swing_time, 3)
                result['right_swing_time_s'] = round(right_swing_time, 3)
                si = calc_si(left_swing_time, right_swing_time)
                result['swing_time_si'] = si
                if si is not None:
                    si_values.append(si)
                print(f"[Processor] Swing time - L: {left_swing_time:.3f}s, R: {right_swing_time:.3f}s", flush=True)

            # 4) 좌우 스탠스 시간 (프레임 기반)
            left_stance_time = (left_stance_frames * frame_time) / max(1, step_count // 2)
            right_stance_time = (right_stance_frames * frame_time) / max(1, step_count // 2)

            if 0.1 <= left_stance_time <= 2.0 and 0.1 <= right_stance_time <= 2.0:
                result['left_stance_time_s'] = round(left_stance_time, 3)
                result['right_stance_time_s'] = round(right_stance_time, 3)
                si = calc_si(left_stance_time, right_stance_time)
                result['stance_time_si'] = si
                if si is not None:
                    si_values.append(si)
                print(f"[Processor] Stance time - L: {left_stance_time:.3f}s, R: {right_stance_time:.3f}s", flush=True)

            # 5) 좌우 스윙/스탠스 비율 (백분율로 저장)
            # 정상: Swing 40%, Stance 60% → ratio ≈ 0.67
            if left_stance_frames > 0 and right_stance_frames > 0:
                # 백분율 저장 (프론트엔드에서 % 표시용)
                result['left_swing_pct'] = round(left_swing_pct, 1)
                result['right_swing_pct'] = round(right_swing_pct, 1)
                result['left_stance_pct'] = round(left_stance_pct, 1)
                result['right_stance_pct'] = round(right_stance_pct, 1)

                # 비율도 계산 (참고용)
                left_ratio = left_swing_frames / left_stance_frames
                right_ratio = right_swing_frames / right_stance_frames
                result['left_swing_stance_ratio'] = round(left_ratio, 2)
                result['right_swing_stance_ratio'] = round(right_ratio, 2)

                # SI는 Swing % 기준으로 계산 (더 직관적)
                si = calc_si(left_swing_pct, right_swing_pct)
                result['swing_stance_si'] = si
                if si is not None:
                    si_values.append(si)

                print(f"[Processor] Swing/Stance ratio - L: {left_ratio:.2f}, R: {right_ratio:.2f}", flush=True)

        except Exception as e:
            import traceback
            print(f"[Processor] Swing/Stance analysis error: {e}", flush=True)
            traceback.print_exc()

        # ═══ 종합 대칭성 지수 (모든 SI의 평균) ═══
        if si_values:
            result['overall_symmetry_index'] = round(np.mean(si_values), 1)

        return result

    def _compute_results(self):
        """최종 임상 결과 계산"""
        if self.timer_elapsed_s <= 0:
            return

        distance_m = self.aruco.marker_distance_m
        speed_mps = distance_m / self.timer_elapsed_s

        # 상세 보행 분석
        gait_params = self._analyze_gait_parameters()

        print(f"[Processor] Gait analysis: steps={gait_params['step_count']}, "
              f"cadence={gait_params['cadence_spm']}, "
              f"step_len={gait_params['step_length_m']}, "
              f"stride_len={gait_params['stride_length_m']}, "
              f"overall_SI={gait_params['overall_symmetry_index']}")

        # 임상 해석
        if self.gait_analyzer is None:
            self.calibrator.pixels_per_meter = self.aruco.pixels_per_meter
            self.gait_analyzer = GaitAnalyzer(self.calibrator)

        interpretation = self.gait_analyzer.get_clinical_interpretation(
            speed_mps, gait_params['cadence_spm']
        )

        self.analysis_results = {
            # 기본 정보
            'elapsed_time_s': round(self.timer_elapsed_s, 3),
            'distance_m': distance_m,
            'speed_mps': round(speed_mps, 3),
            'speed_kmph': round(speed_mps * 3.6, 3),

            # 보행 지표
            'step_count': gait_params['step_count'],
            'cadence_spm': gait_params['cadence_spm'],
            'step_length_m': gait_params['step_length_m'],
            'stride_length_m': gait_params['stride_length_m'],

            # 시간 지표
            'step_time_s': gait_params['step_time_s'],
            'stride_time_s': gait_params['stride_time_s'],
            'left_stride_time_s': gait_params['left_stride_time_s'],
            'right_stride_time_s': gait_params['right_stride_time_s'],
            'stride_time_si': gait_params['stride_time_si'],

            # 좌우 보폭 + SI
            'left_step_length_m': gait_params['left_step_length_m'],
            'right_step_length_m': gait_params['right_step_length_m'],
            'step_length_si': gait_params['step_length_si'],

            # 좌우 스텝 시간 + SI
            'left_step_time_s': gait_params['left_step_time_s'],
            'right_step_time_s': gait_params['right_step_time_s'],
            'step_time_si': gait_params['step_time_si'],

            # 좌우 스윙 시간 + SI
            'left_swing_time_s': gait_params['left_swing_time_s'],
            'right_swing_time_s': gait_params['right_swing_time_s'],
            'swing_time_si': gait_params['swing_time_si'],

            # 좌우 스탠스 시간 + SI
            'left_stance_time_s': gait_params['left_stance_time_s'],
            'right_stance_time_s': gait_params['right_stance_time_s'],
            'stance_time_si': gait_params['stance_time_si'],

            # 좌우 스윙/스탠스 비율 + SI
            'left_swing_stance_ratio': gait_params['left_swing_stance_ratio'],
            'right_swing_stance_ratio': gait_params['right_swing_stance_ratio'],
            'swing_stance_si': gait_params['swing_stance_si'],

            # 스윙/스탠스 백분율
            'left_swing_pct': gait_params['left_swing_pct'],
            'right_swing_pct': gait_params['right_swing_pct'],
            'left_stance_pct': gait_params['left_stance_pct'],
            'right_stance_pct': gait_params['right_stance_pct'],

            # 좌우 활보장 + SI
            'left_stride_length_m': gait_params['left_stride_length_m'],
            'right_stride_length_m': gait_params['right_stride_length_m'],
            'stride_length_si': gait_params['stride_length_si'],

            # 종합 대칭성 지수
            'overall_symmetry_index': gait_params['overall_symmetry_index'],

            # 상세 이벤트 (선택적)
            'step_events': gait_params['step_events'][:20],  # 처음 20개만

            # 임상 해석
            'clinical': interpretation,
            'analysis_mode': 'lateral',
        }

        # ═══ 판정 로직 ═══
        # 측정값을 판정 모듈 키에 매핑 (m → cm 변환 포함, 좌우 분리)
        left_step_cm = (gait_params['left_step_length_m'] * 100) if gait_params.get('left_step_length_m') else None
        right_step_cm = (gait_params['right_step_length_m'] * 100) if gait_params.get('right_step_length_m') else None
        left_stride_cm = (gait_params['left_stride_length_m'] * 100) if gait_params.get('left_stride_length_m') else None
        right_stride_cm = (gait_params['right_stride_length_m'] * 100) if gait_params.get('right_stride_length_m') else None

        # 좌우 평균 stance/swing 비율
        l_stance = gait_params.get('left_stance_pct')
        r_stance = gait_params.get('right_stance_pct')
        l_swing = gait_params.get('left_swing_pct')
        r_swing = gait_params.get('right_swing_pct')
        stance_pct = round((l_stance + r_stance) / 2, 1) if (l_stance and r_stance) else None
        swing_pct = round((l_swing + r_swing) / 2, 1) if (l_swing and r_swing) else None

        measured = {
            'gait_velocity_ms': round(speed_mps, 2),
            'cadence_spm': gait_params['cadence_spm'],
            'left_step_length_cm': round(left_step_cm, 1) if left_step_cm else None,
            'right_step_length_cm': round(right_step_cm, 1) if right_step_cm else None,
            'left_stride_length_cm': round(left_stride_cm, 1) if left_stride_cm else None,
            'right_stride_length_cm': round(right_stride_cm, 1) if right_stride_cm else None,
            'step_time_s': gait_params['step_time_s'],
            'stride_time_s': gait_params['stride_time_s'],
            'stance_ratio_pct': stance_pct,
            'swing_ratio_pct': swing_pct,
            'double_support_pct': None,
            'single_support_pct': None,
        }

        self.analysis_results['judgment'] = judge_all(measured)

        # ═══ Evidence Clips (각 변수별 영상 구간 정보) ═══
        fps = self.fps if self.fps > 0 else 30
        start_ev = next((e for e in self.crossing_events if e['line'] == 'start'), None)
        finish_ev = next((e for e in self.crossing_events if e['line'] == 'finish'), None)
        start_t = start_ev['timestamp_s'] if start_ev else 0
        finish_t = finish_ev['timestamp_s'] if finish_ev else 0
        step_events = gait_params.get('step_events', [])

        evidence_clips = {}

        # 보행속도 / 케이던스 → 전체 구간
        evidence_clips['gait_velocity_ms'] = {
            'start_s': round(start_t, 2), 'end_s': round(finish_t, 2),
            'label': '전체 보행 구간',
        }
        evidence_clips['cadence_spm'] = evidence_clips['gait_velocity_ms']

        # 보폭 / 활보장 → 해당 발이 앞서는 step event 구간
        for foot in ['left', 'right']:
            foot_steps = [e for e in step_events if e.get('leading_foot') == foot]
            if len(foot_steps) >= 2:
                # 가운데 step event 근처를 보여줌
                mid_idx = len(foot_steps) // 2
                mid_event = foot_steps[mid_idx]
                t = mid_event['time']
                clip = {'start_s': round(t - 0.5, 2), 'end_s': round(t + 0.5, 2), 'label': f'{foot} step'}
                evidence_clips[f'{foot}_step_length_cm'] = clip
            if len(foot_steps) >= 3:
                # stride = 같은 발 2연속 → i번째와 i+2번째 사이
                mid_idx = len(foot_steps) // 2
                stride_start = foot_steps[mid_idx - 1] if mid_idx > 0 else foot_steps[0]
                stride_end = foot_steps[min(mid_idx + 1, len(foot_steps) - 1)]
                clip = {
                    'start_s': round(stride_start['time'] - 0.3, 2),
                    'end_s': round(stride_end['time'] + 0.3, 2),
                    'label': f'{foot} stride',
                }
                evidence_clips[f'{foot}_stride_length_cm'] = clip

        # step_time / stride_time → 연속 step event 구간
        if len(step_events) >= 3:
            mid = len(step_events) // 2
            evidence_clips['step_time_s'] = {
                'start_s': round(step_events[mid - 1]['time'] - 0.3, 2),
                'end_s': round(step_events[mid + 1]['time'] + 0.3, 2),
                'label': 'step time 구간',
            }
            evidence_clips['stride_time_s'] = {
                'start_s': round(step_events[mid - 1]['time'] - 0.3, 2),
                'end_s': round(step_events[min(mid + 2, len(step_events) - 1)]['time'] + 0.3, 2),
                'label': 'stride time 구간',
            }

        # stance / swing → 보행 중반부
        mid_t = (start_t + finish_t) / 2
        evidence_clips['stance_ratio_pct'] = {
            'start_s': round(mid_t - 1.5, 2), 'end_s': round(mid_t + 1.5, 2),
            'label': 'stance/swing 구간',
        }
        evidence_clips['swing_ratio_pct'] = evidence_clips['stance_ratio_pct']

        self.analysis_results['evidence_clips'] = evidence_clips

        # ═══ Analysis Timeline (프레임별 오버레이 데이터) ═══
        # 프론트엔드에서 영상 위에 실시간 오버레이를 그리기 위한 데이터
        ppm = self.aruco.pixels_per_meter if self.aruco.pixels_per_meter else 100

        # step_events에 개별 거리(cm) 추가
        raw_steps = gait_params['step_events']
        enriched_steps = []
        for i, ev in enumerate(raw_steps):
            entry = dict(ev)  # copy
            entry['step_num'] = i + 1
            # 이전 스텝과의 거리 = step length
            if i > 0:
                prev = raw_steps[i - 1]
                # 두 발의 중점 X 변화
                curr_mid = (ev['left_x'] + ev['right_x']) / 2
                prev_mid = (prev['left_x'] + prev['right_x']) / 2
                dx = abs(curr_mid - prev_mid)
                step_cm = (dx / ppm) * 100
                entry['step_length_cm'] = round(float(step_cm), 1)
                entry['step_time_s'] = round(float(ev['time'] - prev['time']), 3)
            else:
                entry['step_length_cm'] = None
                entry['step_time_s'] = None
            # 같은 발 2칸 전 = stride
            if i >= 2 and raw_steps[i - 2].get('leading_foot') == ev.get('leading_foot'):
                prev2 = raw_steps[i - 2]
                foot_key = 'left_x' if ev['leading_foot'] == 'left' else 'right_x'
                dx2 = abs(ev[foot_key] - prev2[foot_key])
                stride_cm = (dx2 / ppm) * 100
                entry['stride_length_cm'] = round(float(stride_cm), 1)
            else:
                entry['stride_length_cm'] = None
            enriched_steps.append(entry)

        self.analysis_results['step_events'] = enriched_steps

        # 프레임별 타임라인 데이터 생성
        timeline = []
        segment = [h for h in self.ankle_history if start_t <= h['timestamp_s'] <= finish_t]

        # step events를 시간순 정렬 (누적 카운트용)
        sorted_steps = sorted(step_events, key=lambda e: e['time'])

        for i, h in enumerate(segment):
            t = h['timestamp_s']
            elapsed_from_start = t - start_t

            # 누적 걸음 수: 현재 시간 이전의 step event 수
            cum_steps = sum(1 for s in sorted_steps if s['time'] <= t)

            # 순간 속도 (rolling window ~1초)
            # 현재까지 이동 거리 / 경과 시간
            if elapsed_from_start > 0.1:
                # X좌표 기반 이동 거리 계산
                mid_x = (h['left_x'] + h['right_x']) / 2
                first_h = segment[0]
                first_mid_x = (first_h['left_x'] + first_h['right_x']) / 2
                dist_so_far = abs(mid_x - first_mid_x) / ppm
                inst_speed = dist_so_far / elapsed_from_start
            else:
                inst_speed = 0

            # 순간 케이던스 (최근 걸음수 기반)
            if elapsed_from_start > 0.5:
                inst_cadence = (cum_steps / elapsed_from_start) * 60
            else:
                inst_cadence = 0

            entry = {
                't': round(t, 3),
                'fi': h['frame_idx'],
                'lx': round(h['left_x'], 1),
                'ly': round(h['left_y'], 1),
                'rx': round(h['right_x'], 1),
                'ry': round(h['right_y'], 1),
                'cs': cum_steps,
                'spd': round(inst_speed, 2),
                'cad': round(inst_cadence, 0),
            }
            # heel 좌표 (있으면 추가 - 스텝 라인 그리기용)
            if 'left_heel_x' in h:
                entry['lhx'] = round(h['left_heel_x'], 1)
                entry['lhy'] = round(h['left_heel_y'], 1)
            if 'right_heel_x' in h:
                entry['rhx'] = round(h['right_heel_x'], 1)
                entry['rhy'] = round(h['right_heel_y'], 1)

            timeline.append(entry)

        self.analysis_results['analysis_timeline'] = timeline
        self.analysis_results['timeline_meta'] = {
            'start_t': round(start_t, 3),
            'finish_t': round(finish_t, 3),
            'fps': self.fps,
            'ppm': round(ppm, 2),
            'total_steps': len(sorted_steps),
        }
        print(f"[Processor] Timeline: {len(timeline)} frames, {len(sorted_steps)} step events")
