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

        # 2.5) 원근 보정기 캘리브레이션 (ArUco 보정 완료 후 1회)
        # 2-pass 분석에서 ArUco가 Pass 1에서 보정 완료되면
        # Pass 2의 process_frame에서 캘리브레이션 블록을 건너뛸 수 있으므로 별도 체크
        if (self.aruco.calibrated and not self.perspective_corrector.calibrated
                and self._first_frame is not None):
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

        # ═══ Step Detection: Y-peak Primary + Vx-crossing Fallback ═══
        # Primary: heel Y 로컬 최대값 기반 (timing 정밀도 우수)
        # Fallback: cadence < 70 시 ankle X velocity crossing (crosstalk 없는 counting)

        effective_fps = len(times) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else self.fps
        min_distance = max(3, int(effective_fps * 0.3))

        # 보행 방향 결정 (vx fallback에서도 사용)
        mid_x = (left_x_smooth + right_x_smooth) / 2
        walk_dir_sign = 1.0 if mid_x[-1] > mid_x[0] else -1.0

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

        # ═══ C. Homography vx 보정: 원근보정된 X좌표 (vx 비대칭 해소) ═══
        # 원근 효과: 카메라에 가까운 발의 pixel vx가 과대 → L/R vx 비대칭
        # Homography로 X좌표 보정 후 vx 계산 → 실세계 기준 대칭
        if self.perspective_corrector.calibrated and self.perspective_corrector.H is not None:
            _H = self.perspective_corrector.H
            # Batch transform: ankle smooth X
            pts_left = np.column_stack([left_x_smooth, left_y]).reshape(-1, 1, 2).astype(np.float32)
            pts_right = np.column_stack([right_x_smooth, right_y]).reshape(-1, 1, 2).astype(np.float32)
            left_x_corrected = cv2.perspectiveTransform(pts_left, _H)[:, 0, 0]
            right_x_corrected = cv2.perspectiveTransform(pts_right, _H)[:, 0, 0]
            # Batch transform: heel smooth X
            pts_lheel = np.column_stack([left_heel_x_sm, left_y]).reshape(-1, 1, 2).astype(np.float32)
            pts_rheel = np.column_stack([right_heel_x_sm, right_y]).reshape(-1, 1, 2).astype(np.float32)
            left_heel_x_corrected = cv2.perspectiveTransform(pts_lheel, _H)[:, 0, 0]
            right_heel_x_corrected = cv2.perspectiveTransform(pts_rheel, _H)[:, 0, 0]
            print(f"[Processor] Homography vx correction: X coordinates perspective-corrected", flush=True)
        else:
            left_x_corrected = left_x_smooth
            right_x_corrected = right_x_smooth
            left_heel_x_corrected = left_heel_x_sm
            right_heel_x_corrected = right_heel_x_sm

        # ── HS Onset Ratio: valley→peak 구간에서 Initial Contact 위치 ──
        # 1.0 = Y-peak 그대로 (loading response), 0.3 = IC에 가까움, 0.0 = valley
        HS_ONSET_RATIO = 0.3

        def detect_heel_strikes(heel_x_smooth, heel_y_smooth, times_arr, min_dist, min_gap_s=0.25):
            """적응형 Heel Strike 감지 - Y-peak onset (Initial Contact 시점)"""
            if len(heel_y_smooth) < 5:
                return []
            y_range = np.max(heel_y_smooth) - np.min(heel_y_smooth)
            for prom_factor in [0.06, 0.03, 0.015]:
                prom = max(1, y_range * prom_factor)
                peaks, props = signal.find_peaks(heel_y_smooth, distance=min_dist, prominence=prom)
                if len(peaks) >= 4:
                    break
            if len(peaks) == 0:
                return []
            heel_vx = np.gradient(heel_x_smooth, times_arr)
            abs_heel_vx = np.abs(heel_vx)
            vx_median = np.median(abs_heel_vx)
            for vx_mult in [2.5, 4.0, 6.0]:
                vx_threshold = vx_median * vx_mult
                filtered_peaks = [idx for idx in peaks if abs_heel_vx[idx] <= vx_threshold]
                if len(filtered_peaks) >= 4:
                    break
            else:
                filtered_peaks = list(peaks)
            final_peaks = []
            for idx in filtered_peaks:
                if final_peaks:
                    dt = times_arr[idx] - times_arr[final_peaks[-1]]
                    if dt < min_gap_s:
                        if heel_y_smooth[idx] > heel_y_smooth[final_peaks[-1]]:
                            final_peaks[-1] = idx
                        continue
                final_peaks.append(idx)

            # ── Onset detection: Y-peak → Initial Contact 보정 ──
            if HS_ONSET_RATIO >= 1.0:
                return final_peaks

            onset_peaks = []
            total_shift_ms = 0
            for peak_idx in final_peaks:
                # peak 이전 구간에서 valley (Y 최소) 찾기
                search_start = max(0, peak_idx - int(min_dist * 2))
                region = heel_y_smooth[search_start:peak_idx + 1]
                if len(region) < 3:
                    onset_peaks.append(peak_idx)
                    continue

                valley_local = int(np.argmin(region))
                valley_idx = search_start + valley_local

                # onset = valley + (peak - valley) * ratio
                onset_idx = int(valley_idx + (peak_idx - valley_idx) * HS_ONSET_RATIO)
                onset_idx = max(valley_idx, min(onset_idx, peak_idx))

                shift_ms = (times_arr[peak_idx] - times_arr[onset_idx]) * 1000
                total_shift_ms += shift_ms
                onset_peaks.append(onset_idx)

            avg_shift = total_shift_ms / len(onset_peaks) if onset_peaks else 0
            print(f"[HS Onset] ratio={HS_ONSET_RATIO}, peaks={len(final_peaks)}, avg_shift={avg_shift:.0f}ms", flush=True)
            return onset_peaks

        def detect_hs_vx_crossing(heel_x_sm, times_arr, walk_sign, shared_threshold, min_gap_s=0.3):
            """Ankle X velocity crossing 기반 HS 감지 (sub-frame 보간 포함)
            Y-peak이 crosstalk으로 실패할 때 사용하는 fallback"""
            if len(heel_x_sm) < 10:
                return []
            vx_raw = np.gradient(heel_x_sm, times_arr)
            vx_norm = vx_raw * walk_sign
            vk = np.ones(5) / 5
            vx = np.convolve(vx_norm, vk, mode='same')
            above = vx > shared_threshold
            crossings = []
            for i in range(1, len(above)):
                if above[i - 1] and not above[i]:
                    crossings.append(i)
            eff_fps = len(times_arr) / (times_arr[-1] - times_arr[0]) if len(times_arr) > 1 else 20
            window = max(3, int(eff_fps * 0.2))
            refined = []
            for idx in crossings:
                search_end = min(len(vx), idx + window)
                zero_found = False
                for j in range(idx, search_end):
                    if vx[j] <= 0:
                        if j > 0 and vx[j - 1] > 0:
                            alpha = vx[j - 1] / (vx[j - 1] - vx[j])
                            interp_time = times_arr[j - 1] + alpha * (times_arr[j] - times_arr[j - 1])
                            refined.append((j, interp_time))
                        else:
                            refined.append((j, times_arr[j]))
                        zero_found = True
                        break
                if not zero_found:
                    local_abs_vx = np.abs(vx[idx:search_end])
                    if len(local_abs_vx) > 0:
                        best = idx + int(np.argmin(local_abs_vx))
                        refined.append((best, times_arr[best]))
                    else:
                        refined.append((idx, times_arr[idx]))
            filtered = []
            for frame_idx, interp_t in refined:
                if filtered:
                    dt = interp_t - filtered[-1][1]
                    if dt < min_gap_s:
                        continue
                filtered.append((frame_idx, interp_t))
            return filtered

        def build_step_events_from_indices(left_indices, right_indices, use_interp_time=False):
            """per-foot 인덱스 → 시간순 step event 리스트 생성 + bilateral dedup"""
            all_hs = []
            for item in left_indices:
                if use_interp_time:
                    idx, t = item
                else:
                    idx, t = item, float(times[item])
                if idx < len(times):
                    all_hs.append({
                        'time': float(t), 'frame_idx': segment[idx]['frame_idx'],
                        'leading_foot': 'left',
                        'left_x': float(left_x[idx]), 'right_x': float(right_x[idx]),
                        'left_y': float(left_y[idx]), 'right_y': float(right_y[idx]),
                    })
            for item in right_indices:
                if use_interp_time:
                    idx, t = item
                else:
                    idx, t = item, float(times[item])
                if idx < len(times):
                    all_hs.append({
                        'time': float(t), 'frame_idx': segment[idx]['frame_idx'],
                        'leading_foot': 'right',
                        'left_x': float(left_x[idx]), 'right_x': float(right_x[idx]),
                        'left_y': float(left_y[idx]), 'right_y': float(right_y[idx]),
                    })
            all_hs.sort(key=lambda e: e['time'])
            # bilateral dedup
            deduped = []
            for ev in all_hs:
                if not deduped:
                    deduped.append(ev)
                    continue
                dt = ev['time'] - deduped[-1]['time']
                if dt < 0.15:
                    curr_y = ev['left_y'] if ev['leading_foot'] == 'left' else ev['right_y']
                    prev_y = deduped[-1]['left_y'] if deduped[-1]['leading_foot'] == 'left' else deduped[-1]['right_y']
                    if curr_y > prev_y:
                        deduped[-1] = ev
                    continue
                if dt < 0.25 and ev['leading_foot'] == deduped[-1]['leading_foot']:
                    curr_y = ev['left_y'] if ev['leading_foot'] == 'left' else ev['right_y']
                    prev_y = deduped[-1]['left_y'] if deduped[-1]['leading_foot'] == 'left' else deduped[-1]['right_y']
                    if curr_y > prev_y:
                        deduped[-1] = ev
                    continue
                deduped.append(ev)
            # boundary filter
            deduped = [e for e in deduped if e['time'] >= start_t + 0.3]
            if len(deduped) >= 4:
                dts = [deduped[i]['time'] - deduped[i-1]['time'] for i in range(1, len(deduped))]
                dt_median = float(np.median(dts))
                if dts[-1] > dt_median * 2.5:
                    deduped = deduped[:-1]
            return deduped

        def detect_foot_separation_steps(l_x_sm, r_x_sm, times_arr, min_gap_s=0.2):
            """Foot Separation (|left_x - right_x|) 피크 기반 step detection
            Y축을 사용하지 않으므로 heel 부재 + crosstalk 문제를 우회"""
            if len(l_x_sm) < 10:
                return []

            foot_sep = np.abs(l_x_sm - r_x_sm)

            # 스무딩 (7-point MA)
            sep_kernel = np.ones(7) / 7
            foot_sep_smooth = np.convolve(foot_sep, sep_kernel, mode='same')

            sep_range = np.max(foot_sep_smooth) - np.min(foot_sep_smooth)
            if sep_range < 1:
                return []

            eff_fps = len(times_arr) / (times_arr[-1] - times_arr[0]) if len(times_arr) > 1 else 20
            min_dist = max(2, int(eff_fps * 0.15))

            # prominence 3단계 시도
            best_peaks = np.array([], dtype=int)
            for prom_factor in [0.15, 0.10, 0.05]:
                prom = max(0.5, sep_range * prom_factor)
                peaks, _ = signal.find_peaks(foot_sep_smooth, distance=min_dist, prominence=prom)
                if len(peaks) >= 6:
                    best_peaks = peaks
                    break
                if len(peaks) > len(best_peaks):
                    best_peaks = peaks

            if len(best_peaks) < 2:
                return []

            # min_gap dedup + boundary filter
            filtered_peaks = []
            for idx in best_peaks:
                t = times_arr[idx]
                if t < start_t + 0.3:
                    continue
                if filtered_peaks:
                    dt = t - times_arr[filtered_peaks[-1]]
                    if dt < min_gap_s:
                        if foot_sep_smooth[idx] > foot_sep_smooth[filtered_peaks[-1]]:
                            filtered_peaks[-1] = idx
                        continue
                filtered_peaks.append(idx)

            # step events 생성
            events = []
            for idx in filtered_peaks:
                if idx < len(times_arr):
                    l_ahead = l_x_sm[idx] * walk_dir_sign
                    r_ahead = r_x_sm[idx] * walk_dir_sign
                    leading = 'left' if l_ahead > r_ahead else 'right'
                    events.append({
                        'time': float(times_arr[idx]),
                        'frame_idx': segment[idx]['frame_idx'],
                        'leading_foot': leading,
                        'left_x': float(left_x[idx]), 'right_x': float(right_x[idx]),
                        'left_y': float(left_y[idx]), 'right_y': float(right_y[idx]),
                    })

            # tail filter
            if len(events) >= 4:
                dts = [events[i]['time'] - events[i-1]['time'] for i in range(1, len(events))]
                dt_median = float(np.median(dts))
                if dts[-1] > dt_median * 2.5:
                    events = events[:-1]

            return events

        def step_regularity_score(events):
            """Step interval CV (Coefficient of Variation) — 낮을수록 규칙적"""
            if len(events) < 3:
                return float('inf')
            intervals = [events[i]['time'] - events[i-1]['time'] for i in range(1, len(events))]
            valid = [dt for dt in intervals if 0.3 <= dt <= 2.0]
            if len(valid) < 2:
                return float('inf')
            return float(np.std(valid) / np.mean(valid))

        step_events = []
        detection_method = 'none'
        try:
            # ── Primary: Y-peak HS detection ──
            left_hs = detect_heel_strikes(left_heel_x_sm, left_heel_y_sm, times, min_distance)
            right_hs = detect_heel_strikes(right_heel_x_sm, right_heel_y_sm, times, min_distance)
            print(f"[Processor] Y-peak HS detection: L={len(left_hs)}, R={len(right_hs)}", flush=True)

            ypeak_events = build_step_events_from_indices(left_hs, right_hs)
            ypeak_cadence = (len(ypeak_events) / elapsed) * 60 if len(ypeak_events) >= 2 else 0
            print(f"[Processor] Y-peak result: {len(ypeak_events)} events, cadence={ypeak_cadence:.0f} spm", flush=True)

            # ── Vx-crossing detection (always run) ──
            vk = np.ones(5) / 5
            # C: Homography 보정된 X좌표로 vx 계산 (원근 비대칭 해소)
            left_vx_n = np.convolve(np.gradient(left_heel_x_corrected, times) * walk_dir_sign, vk, mode='same')
            right_vx_n = np.convolve(np.gradient(right_heel_x_corrected, times) * walk_dir_sign, vk, mode='same')
            all_pos_vx = np.concatenate([left_vx_n[left_vx_n > 0], right_vx_n[right_vx_n > 0]])
            swing_med = float(np.median(all_pos_vx)) if len(all_pos_vx) >= 5 else 100.0

            best_l, best_r = [], []
            for thr_f in [0.35, 0.25, 0.15]:
                thr = swing_med * thr_f
                l_hs = detect_hs_vx_crossing(left_heel_x_corrected, times, walk_dir_sign, thr)
                r_hs = detect_hs_vx_crossing(right_heel_x_corrected, times, walk_dir_sign, thr)
                if len(l_hs) >= 4 and len(r_hs) >= 4:
                    best_l, best_r = l_hs, r_hs
                    break
                if len(l_hs) + len(r_hs) > len(best_l) + len(best_r):
                    best_l, best_r = l_hs, r_hs

            vx_events = build_step_events_from_indices(best_l, best_r, use_interp_time=True)
            vx_cadence = (len(vx_events) / elapsed) * 60 if len(vx_events) >= 2 else 0
            print(f"[Processor] Vx-crossing: L={len(best_l)}, R={len(best_r)} → {len(vx_events)} events, cadence={vx_cadence:.0f}", flush=True)

            # ── Foot Separation detection (Homography 보정된 좌표 사용) ──
            sep_events = detect_foot_separation_steps(left_x_corrected, right_x_corrected, times)
            sep_cadence = (len(sep_events) / elapsed) * 60 if len(sep_events) >= 2 else 0
            print(f"[Processor] Foot Separation: {len(sep_events)} events, cadence={sep_cadence:.0f} spm", flush=True)

            # ── 3-way quality selection ──
            candidates = []
            if len(ypeak_events) >= 2:
                candidates.append(('y-peak', ypeak_events, ypeak_cadence))
            if len(vx_events) >= 2:
                candidates.append(('vx-crossing', vx_events, vx_cadence))
            if len(sep_events) >= 2:
                candidates.append(('foot-separation', sep_events, sep_cadence))

            # cadence >= 70인 후보 중 regularity 최적 선택
            good = [(n, ev, c) for n, ev, c in candidates if c >= 70]
            if good:
                best = min(good, key=lambda x: step_regularity_score(x[1]))
                step_events, detection_method = best[1], best[0]
                print(f"[Processor] → Selected {detection_method} (cadence {best[2]:.0f} >= 70, regularity={step_regularity_score(best[1]):.3f})", flush=True)
            else:
                # 모두 <70 → events >= 6이고 regularity 좋은 쪽
                ok = [(n, ev, c) for n, ev, c in candidates if len(ev) >= 6]
                if ok:
                    best = min(ok, key=lambda x: step_regularity_score(x[1]))
                    step_events, detection_method = best[1], best[0]
                    print(f"[Processor] → All <70 spm, selected {detection_method} (events >= 6, regularity={step_regularity_score(best[1]):.3f})", flush=True)
                elif candidates:
                    best = max(candidates, key=lambda x: len(x[1]))
                    step_events, detection_method = best[1], best[0]
                    print(f"[Processor] → Fallback: selected {detection_method} ({len(best[1])} events)", flush=True)

        except Exception as e:
            print(f"[Processor] Step detection error: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # 최종 폴백: X좌표 교차 방식 (모든 방법 실패 시)
        if len(step_events) < 4:
            prev_count = len(step_events)
            diff_x = left_x_smooth - right_x_smooth
            sign_changes = np.where(np.diff(np.sign(diff_x)))[0]
            fallback_events = []
            for idx in sign_changes:
                if idx < len(times):
                    leading_foot = 'left' if diff_x[idx] > 0 else 'right'
                    t_val = float(times[idx])
                    if t_val >= start_t + 0.3:
                        fallback_events.append({
                            'time': t_val, 'frame_idx': segment[idx]['frame_idx'],
                            'leading_foot': leading_foot,
                            'left_x': float(left_x[idx]), 'right_x': float(right_x[idx]),
                            'left_y': float(left_y[idx]), 'right_y': float(right_y[idx]),
                        })
            if len(fallback_events) > len(step_events):
                step_events = fallback_events
                detection_method = 'x-crossing'
            print(f"[Processor] X-crossing last-resort: prev={prev_count} → {len(step_events)}", flush=True)

        # ═══ Leading Foot 재판별: 공간 위치 기반 ═══
        # 측면 카메라에서 HS 감지기의 L/R 라벨이 부정확한 문제 해결
        # 원리: 각 HS 시점에서 보행 방향으로 더 앞에 있는 발 = leading foot
        if len(step_events) >= 2:
            # 보행 방향 부호 결정 (X 증가=+1, X 감소=-1)
            first_mid = (step_events[0]['left_x'] + step_events[0]['right_x']) / 2
            last_mid = (step_events[-1]['left_x'] + step_events[-1]['right_x']) / 2
            walk_sign = 1.0 if last_mid > first_mid else -1.0

            reassign_count = 0
            for ev in step_events:
                # 보행 방향으로 더 앞에 있는 발 = leading foot
                left_ahead = ev['left_x'] * walk_sign
                right_ahead = ev['right_x'] * walk_sign
                new_foot = 'left' if left_ahead > right_ahead else 'right'
                if ev['leading_foot'] != new_foot:
                    reassign_count += 1
                ev['leading_foot'] = new_foot

            if reassign_count > 0:
                print(f"[Processor] Spatial position reassign: {reassign_count}/{len(step_events)} events relabeled", flush=True)

            # 교대 강제: 같은 발 연속이면 두 번째를 반대 발로 전환
            opposite = {'left': 'right', 'right': 'left'}
            flip_count = 0
            for i in range(1, len(step_events)):
                if step_events[i]['leading_foot'] == step_events[i-1]['leading_foot']:
                    step_events[i]['leading_foot'] = opposite[step_events[i]['leading_foot']]
                    flip_count += 1
            if flip_count > 0:
                print(f"[Processor] Alternation fix: flipped {flip_count} remaining consecutive same-foot", flush=True)

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

        # 케이던스 (steps per minute) - 중앙값 기반 (이상치 강건)
        cadence = (step_count / elapsed) * 60  # 기본 계산
        # step interval 중앙값으로 보강 (0.3s~2.0s 범위만)
        _step_intervals = []
        for i in range(1, len(step_events)):
            dt = step_events[i]['time'] - step_events[i-1]['time']
            if 0.3 <= dt <= 2.0:
                _step_intervals.append(dt)
        if _step_intervals:
            cadence = 60.0 / np.median(_step_intervals)
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

        # 좌우 활보장 개별 계산 (2-step 간격 = stride)
        print(f"[Processor] === Stride L/R Calculation ===")
        print(f"[Processor] Step events count: {len(step_events)}, ppm: {ppm}")

        # 라벨 독립: event[i]→event[i+2] 거리 = stride (같은 발 1회전)
        # 홀수 위치(0,2,4,..) = foot A stride, 짝수 위치(1,3,5,..) = foot B stride
        if len(step_events) >= 3:
            left_strides = []
            right_strides = []

            for i in range(len(step_events) - 2):
                curr = step_events[i]
                next2 = step_events[i + 2]
                # 중점(midpoint)으로 stride 거리 계산
                x1 = (curr['left_x'] + curr['right_x']) / 2
                y1 = (curr['left_y'] + curr['right_y']) / 2
                x2 = (next2['left_x'] + next2['right_x']) / 2
                y2 = (next2['left_y'] + next2['right_y']) / 2
                dx = abs(x2 - x1)
                if self.perspective_corrector.calibrated:
                    stride_m = self.perspective_corrector.real_distance_x(x1, y1, x2, y2)
                else:
                    stride_m = dx / ppm

                if 0.3 <= stride_m <= 4.0:
                    if i % 2 == 0:
                        left_strides.append(stride_m)
                    else:
                        right_strides.append(stride_m)

            print(f"[Processor] Valid strides - L: {len(left_strides)}, R: {len(right_strides)}")

            def filter_outliers(values):
                if len(values) < 3:
                    return values
                med = np.median(values)
                return [v for v in values if 0.5 * med <= v <= 2.0 * med]

            left_strides = filter_outliers(left_strides)
            right_strides = filter_outliers(right_strides)
            print(f"[Processor] Valid strides (filtered) - L: {len(left_strides)}, R: {len(right_strides)}")

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

        # stride_length_si는 위에서 이미 계산됨 → si_values에 포함
        if result.get('stride_length_si') is not None:
            si_values.append(result['stride_length_si'])

        if len(step_events) >= 4:
            left_steps = [e for e in step_events if e['leading_foot'] == 'left']
            right_steps = [e for e in step_events if e['leading_foot'] == 'right']

            # 1) 좌우 보폭 (Step Length)
            # Step = 연속된 step event 간 거리 (다른 발 사이)
            left_step_dists = []
            right_step_dists = []

            for i in range(1, len(step_events)):
                curr = step_events[i]
                prev = step_events[i - 1]
                if curr['leading_foot'] != prev['leading_foot']:
                    curr_mid_x = (curr['left_x'] + curr['right_x']) / 2
                    prev_mid_x = (prev['left_x'] + prev['right_x']) / 2
                    curr_mid_y = (curr['left_y'] + curr['right_y']) / 2
                    prev_mid_y = (prev['left_y'] + prev['right_y']) / 2
                    dx = abs(curr_mid_x - prev_mid_x)
                    if self.perspective_corrector.calibrated:
                        step_m = self.perspective_corrector.real_distance_x(
                            prev_mid_x, prev_mid_y, curr_mid_x, curr_mid_y)
                    else:
                        step_m = dx / ppm
                    if 0.2 <= step_m <= 2.0:
                        if curr['leading_foot'] == 'left':
                            left_step_dists.append(step_m)
                        else:
                            right_step_dists.append(step_m)

            print(f"[Processor] Step L/R (raw) - L: {len(left_step_dists)} samples, R: {len(right_step_dists)} samples")

            # 이상치 제거 (L/R 합쳐서 median 기준)
            all_step_dists = left_step_dists + right_step_dists
            if len(all_step_dists) >= 3:
                step_med = np.median(all_step_dists)
                left_step_dists = [v for v in left_step_dists if 0.5 * step_med <= v <= 2.0 * step_med]
                right_step_dists = [v for v in right_step_dists if 0.5 * step_med <= v <= 2.0 * step_med]
            print(f"[Processor] Step L/R (filtered) - L: {len(left_step_dists)} samples, R: {len(right_step_dists)} samples")

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

            # 3) 좌우 활보장 시간 (Stride Time = event[i]→event[i+2] 간격)
            left_stride_times = []
            right_stride_times = []
            for i in range(len(step_events) - 2):
                dt = step_events[i + 2]['time'] - step_events[i]['time']
                if 0.4 <= dt <= 4.0:
                    if i % 2 == 0:
                        left_stride_times.append(dt)
                    else:
                        right_stride_times.append(dt)

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

        # ═══ 발목 X속도 기반 스윙/스탠스 분석 ═══
        # 측면 카메라에서 가장 확실한 swing/stance 구분 = 발의 X방향 이동 속도
        # Stance: 발이 바닥에 고정 → ankle_vx ≈ 0
        # Swing: 발이 앞으로 이동 → |ankle_vx| 크게 증가
        print(f"[Processor] === Starting Ankle X-Velocity Swing/Stance Analysis ===", flush=True)
        try:
            frame_time = 1.0 / self.fps if self.fps > 0 else 1/30

            # C: Homography 보정된 X좌표로 vx 계산 (원근 비대칭 해소)
            left_vx = np.gradient(left_x_corrected, times)
            right_vx = np.gradient(right_x_corrected, times)

            # 보행 방향 부호 결정 (좌→우: +, 우→좌: -)
            walk_dir = np.sign(np.mean(left_vx) + np.mean(right_vx))
            if walk_dir == 0:
                walk_dir = 1.0  # 기본값

            # D: Per-foot threshold — 각 발별 별도 threshold
            # 기존: 양발 합산 median → 카메라 가까운 발이 과대 → 편향
            # 개선: 발별 median → 원근 차이와 무관하게 각 발의 swing 판별
            left_vx_threshold = np.median(np.abs(left_vx)) * 1.2
            right_vx_threshold = np.median(np.abs(right_vx)) * 1.2

            left_is_swing = (left_vx * walk_dir) > left_vx_threshold
            right_is_swing = (right_vx * walk_dir) > right_vx_threshold

            left_swing_frames = int(np.sum(left_is_swing))
            left_stance_frames = len(left_is_swing) - left_swing_frames
            right_swing_frames = int(np.sum(right_is_swing))
            right_stance_frames = len(right_is_swing) - right_swing_frames

            total_frames = len(times)
            print(f"[Processor] Walk direction: {'L→R' if walk_dir > 0 else 'R→L'}, vx_threshold L:{left_vx_threshold:.1f} R:{right_vx_threshold:.1f}", flush=True)
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

        # 보폭 / 활보장 → 전체 step_events 기준으로 앵커 HS ~ 타겟 HS 포함
        for foot in ['left', 'right']:
            foot_steps = [e for e in step_events if e.get('leading_foot') == foot]
            if len(foot_steps) >= 2:
                # Step: 타겟 HS 기준, 바로 이전 HS(반대발)부터 포함
                mid_idx = len(foot_steps) // 2
                target_event = foot_steps[mid_idx]
                target_all_idx = step_events.index(target_event)
                # 앵커 = 전체 step_events에서 1칸 전 (반대발 HS)
                anchor_idx = max(0, target_all_idx - 1)
                anchor_event = step_events[anchor_idx]
                clip = {
                    'start_s': round(anchor_event['time'] - 0.4, 2),
                    'end_s': round(target_event['time'] + 0.4, 2),
                    'label': f'{foot} step',
                }
                evidence_clips[f'{foot}_step_length_cm'] = clip
            if len(foot_steps) >= 3:
                # Stride: 같은 발 HS → 반대발 HS → 같은 발 HS (2칸 전부터)
                mid_idx = len(foot_steps) // 2
                target_event = foot_steps[min(mid_idx + 1, len(foot_steps) - 1)]
                target_all_idx = step_events.index(target_event)
                # 앵커 = 전체 step_events에서 2칸 전 (같은 발 이전 HS)
                anchor_idx = max(0, target_all_idx - 2)
                anchor_event = step_events[anchor_idx]
                clip = {
                    'start_s': round(anchor_event['time'] - 0.4, 2),
                    'end_s': round(target_event['time'] + 0.4, 2),
                    'label': f'{foot} stride',
                }
                evidence_clips[f'{foot}_stride_length_cm'] = clip

        # step_time / stride_time → L/R 별도 (distance와 동일 HS 구간)
        for foot in ['left', 'right']:
            foot_steps = [e for e in step_events if e.get('leading_foot') == foot]
            if len(foot_steps) >= 2:
                # Step time: 반대발 HS → 이 발 HS (distance step과 동일 구간)
                mid_idx = len(foot_steps) // 2
                target_event = foot_steps[mid_idx]
                target_all_idx = step_events.index(target_event)
                anchor_idx = max(0, target_all_idx - 1)
                anchor_event = step_events[anchor_idx]
                evidence_clips[f'{foot}_step_time_s'] = {
                    'start_s': round(anchor_event['time'] - 0.4, 2),
                    'end_s': round(target_event['time'] + 0.4, 2),
                    'label': f'{foot} step time',
                }
            if len(foot_steps) >= 3:
                # Stride time: 같은 발 HS → 같은 발 HS (distance stride와 동일 구간)
                mid_idx = len(foot_steps) // 2
                target_event = foot_steps[min(mid_idx + 1, len(foot_steps) - 1)]
                target_all_idx = step_events.index(target_event)
                anchor_idx = max(0, target_all_idx - 2)
                anchor_event = step_events[anchor_idx]
                evidence_clips[f'{foot}_stride_time_s'] = {
                    'start_s': round(anchor_event['time'] - 0.4, 2),
                    'end_s': round(target_event['time'] + 0.4, 2),
                    'label': f'{foot} stride time',
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
