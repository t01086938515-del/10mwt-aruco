# backend/processor.py
"""프레임별 분석 코디네이터 (횡방향 분석용)

키(신장) 기반 캘리브레이션 + MediaPipe Pose Heavy.
사람의 키 픽셀 높이로 pixels_per_meter를 추정하고,
전신 프레임 진입/퇴장으로 보행 구간을 자동 감지.

횡방향 분석:
- 카메라가 측면에서 촬영 (사람이 좌→우 또는 우→좌로 이동)
- X좌표 기반 진행률 계산
- 케이던스/보폭 추정 (발목 좌표 분석)
- 동적 PPM으로 원근 자동 보정
"""

import cv2
import numpy as np
import time
import os
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from scipy import signal

from analyzer.pose_detector import PoseDetector
from analyzer.gait_analyzer import GaitAnalyzer
from analyzer.gait_judgment import judge_all
from analyzer.calibration import DistanceCalibrator
from analyzer.filter import KalmanFilter2D
from utils.video_utils import VideoReader


class FrameProcessor:
    """프레임별 분석 코디네이터 (횡방향 분석)"""

    def __init__(
        self,
        model_path: str = "",
        walk_distance_m: float = 10.0,
        patient_height_m: float | None = None,
    ):
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

        # 보행 거리 (설정값)
        self.walk_distance_m = walk_distance_m

        # 키 기반 캘리브레이션
        self.patient_height_m = patient_height_m
        self.calibrated = False
        self.height_based_ppm: Optional[float] = None
        self.height_samples: List[float] = []

        # 전신 감지용 (시작/종료)
        self.body_was_visible = False
        self.start_foot_x: Optional[float] = None
        self.finish_foot_x: Optional[float] = None
        self.virtual_start_x: Optional[float] = None
        self.virtual_finish_x: Optional[float] = None
        self._height_mode_distance: Optional[float] = None
        self._last_visible_foot_x: Optional[float] = None
        self._invisible_count = 0  # 연속 미감지 프레임 수 (디바운스)
        self._min_elapsed_s = max(walk_distance_m / 3.0, 3.0)  # 최소 경과시간 (10m → 3.3s)

        # 상태
        self.prev_progress: Optional[float] = None
        self.timer_state = 'standby'  # standby | running | finished
        self.timer_start_time_s: float = 0.0
        self.timer_elapsed_s: float = 0.0
        self.analysis_results: Optional[Dict] = None

        # 크로싱 이벤트 기록
        self.crossing_events: List[Dict] = []

        # 케이던스/보폭 추정을 위한 발목 좌표 기록
        self.ankle_history: List[Dict] = []
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
        self.calibrated = False
        self.height_based_ppm = None
        self.height_samples = []
        self.body_was_visible = False
        self.start_foot_x = None
        self.finish_foot_x = None
        self.virtual_start_x = None
        self.virtual_finish_x = None
        self._height_mode_distance = None
        self._last_visible_foot_x = None
        self._invisible_count = 0

    def set_fps(self, fps: float):
        """영상 FPS 설정"""
        self.fps = fps

    def _is_full_body_visible(self, keypoints: List, bbox: List, frame_w: int, frame_h: int) -> bool:
        """전신이 프레임 안에 완전히 보이는지 판별

        조건:
        - nose(0), hips(23,24), ankles(27,28) 신뢰도 > 0.5
        - bbox가 프레임 경계에서 5% 이상 안쪽
        """
        # 주요 관절 신뢰도 체크
        required_indices = [0, 23, 24, 27, 28]  # nose, L/R hip, L/R ankle
        for idx in required_indices:
            if idx >= len(keypoints):
                return False
            kp = keypoints[idx]
            if len(kp) < 3 or kp[2] < 0.5:
                return False

        # bbox 경계 체크 (5% 마진)
        margin_x = frame_w * 0.05
        margin_y = frame_h * 0.05
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        if x1 < margin_x or x2 > frame_w - margin_x:
            return False
        if y1 < margin_y or y2 > frame_h - margin_y:
            return False

        return True

    def _sample_person_height(self, keypoints: List) -> Optional[float]:
        """키포인트에서 사람 키 픽셀 높이 측정

        nose_y ~ max(ankle_left_y, ankle_right_y) 픽셀 차이
        """
        if len(keypoints) <= 28:
            return None

        nose = keypoints[0]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]

        # 신뢰도 체크
        if len(nose) < 3 or nose[2] < 0.5:
            return None
        if len(left_ankle) < 3 or left_ankle[2] < 0.5:
            return None
        if len(right_ankle) < 3 or right_ankle[2] < 0.5:
            return None

        # 발목 중 더 낮은(큰 y) 위치 사용
        ankle_y = max(left_ankle[1], right_ankle[1])
        height_px = abs(ankle_y - nose[1])

        # 범위 체크 (100~800px)
        if 100 <= height_px <= 800:
            return height_px
        return None

    def try_calibrate_with_height(self) -> bool:
        """키 샘플에서 ppm 계산 시도

        height_samples 10개 이상 필요
        ppm = median(height_px) / patient_height_m
        """
        if not self.patient_height_m or self.patient_height_m <= 0:
            return False
        if len(self.height_samples) < 10:
            return False

        median_height_px = float(np.median(self.height_samples))
        self.height_based_ppm = median_height_px / self.patient_height_m
        self.calibrated = True
        print(f"[Processor] Height-based calibration: "
              f"median_height_px={median_height_px:.1f}, "
              f"patient_height_m={self.patient_height_m:.2f}, "
              f"ppm={self.height_based_ppm:.1f}")
        return True

    def _get_effective_ppm(self) -> Optional[float]:
        """캘리브레이션된 PPM 반환"""
        return self.height_based_ppm

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
            'person': None,
            'timer': {
                'state': self.timer_state,
                'elapsed_s': round(self.timer_elapsed_s, 3),
            },
            'crossing_event': None,
            'calibration': {'calibrated': self.calibrated},
            'results': None,
        }

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

                # height 모드: 매 프레임 키 픽셀 높이 기록 (동적 PPM → 원근보정)
                if self.patient_height_m:
                    h_px = self._sample_person_height(keypoints)
                    if h_px is not None:
                        history_entry['height_px'] = h_px

                self.ankle_history.append(history_entry)

            # 진행률 계산 — 전신 감지 기반 시작/종료
            if self.calibrated and self.height_based_ppm:
                frame_h, frame_w = frame.shape[:2]
                keypoints = best_track['keypoints']
                bbox = best_track['bbox']
                foot_x = filtered_ground[0]
                is_visible = self._is_full_body_visible(keypoints, bbox, frame_w, frame_h)

                # START: 전신 처음 보임 (body_was_visible: False → True)
                if self.timer_state == 'standby' and is_visible and not self.body_was_visible:
                    self.start_foot_x = foot_x
                    self.timer_state = 'running'
                    self.timer_start_time_s = timestamp_s
                    self.virtual_start_x = foot_x
                    # 가상 끝점: 반대쪽 프레임 경계 (10% 마진)
                    self.virtual_finish_x = frame_w * 0.9 if foot_x < frame_w / 2 else frame_w * 0.1
                    result['crossing_event'] = 'start'
                    self.crossing_events.append({
                        'line': 'start',
                        'timestamp_s': round(timestamp_s, 4),
                        'frame_idx': frame_idx,
                        'distance_m': 0.0,
                    })
                    print(f"[Processor] ★★★ HEIGHT MODE START at frame {frame_idx}, "
                          f"t={timestamp_s:.2f}s, foot_x={foot_x:.0f}")

                # FINISH: 전신 사라짐 (디바운스 + 최소 경과시간)
                if self.timer_state == 'running' and not is_visible:
                    self._invisible_count += 1
                elif self.timer_state == 'running' and is_visible:
                    self._invisible_count = 0  # 다시 보이면 리셋

                elapsed_since_start = timestamp_s - self.timer_start_time_s if self.timer_state == 'running' else 0
                finish_ready = (
                    self.timer_state == 'running'
                    and not is_visible
                    and self._invisible_count >= 5  # 연속 5프레임 미감지
                    and elapsed_since_start >= self._min_elapsed_s  # 최소 경과시간
                )

                if finish_ready:
                    self.finish_foot_x = self._last_visible_foot_x or foot_x
                    self.timer_elapsed_s = timestamp_s - self.timer_start_time_s
                    if self.start_foot_x is not None and self.finish_foot_x is not None:
                        distance_m = abs(self.finish_foot_x - self.start_foot_x) / self.height_based_ppm
                    else:
                        distance_m = 0.0
                    self._height_mode_distance = distance_m
                    self.timer_state = 'finished'
                    result['crossing_event'] = 'finish'
                    self.crossing_events.append({
                        'line': 'finish',
                        'timestamp_s': round(timestamp_s, 4),
                        'frame_idx': frame_idx,
                        'elapsed_s': round(self.timer_elapsed_s, 3),
                        'distance_m': round(distance_m, 3),
                    })
                    try:
                        self._compute_results()
                        result['results'] = self.analysis_results
                    except Exception as e:
                        import traceback
                        print(f"[Processor] _compute_results error: {e}")
                        traceback.print_exc()
                        result['results'] = None
                    print(f"[Processor] ★★★ HEIGHT MODE FINISH at frame {frame_idx}, "
                          f"t={timestamp_s:.2f}s, distance={distance_m:.2f}m, "
                          f"elapsed={self.timer_elapsed_s:.3f}s")

                # 전신 가시 상태 업데이트
                if is_visible:
                    self._last_visible_foot_x = foot_x
                self.body_was_visible = is_visible

                # 가상 progress (UI 표시용)
                if self.virtual_start_x is not None and self.virtual_finish_x is not None:
                    total_range = self.virtual_finish_x - self.virtual_start_x
                    if abs(total_range) > 1:
                        progress = (foot_x - self.virtual_start_x) / total_range
                        progress = max(0.0, min(1.0, progress))
                        person['progress'] = round(progress, 4)
                        if self.height_based_ppm:
                            dist_m = abs(foot_x - self.virtual_start_x) / self.height_based_ppm
                            person['distance_from_start_m'] = round(dist_m, 3)

                # 디버그 로그
                self._debug_frame_count += 1
                if self._debug_frame_count <= 20 or self._debug_frame_count % 20 == 0:
                    print(f"[Processor] frame={frame_idx} "
                          f"foot_x={foot_x:.0f} "
                          f"visible={is_visible} "
                          f"timer={self.timer_state}")

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

        # X좌표 보정 없이 스무딩된 값 그대로 사용
        # (동적 PPM이 거리 계산 시 원근보정 대체)
        left_x_corrected = left_x_smooth
        right_x_corrected = right_x_smooth
        left_heel_x_corrected = left_heel_x_sm
        right_heel_x_corrected = right_heel_x_sm

        def detect_heel_strikes(heel_x_smooth, heel_y_smooth, times_arr, min_dist, min_gap_s=0.25):
            """적응형 Heel Strike 감지 - Y-velocity 기반 IC 시점 감지

            각 Y-peak마다:
            1. valley→peak 구간에서 Y-velocity(vy) 계산
            2. vy가 최대인 지점 = 발이 가장 빠르게 내려오는 순간 (IC 직전)
            3. vy가 최대의 50%로 감소한 지점 = 발이 바닥에 닿으며 감속 (≈IC)
            """
            if len(heel_y_smooth) < 5:
                return []  # returns list of (idx, interp_time) tuples
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

            # ── 적응형 IC 감지: Y-velocity 기반 ──
            # Y-velocity = heel Y의 시간 미분 (양수 = 하강 중)
            vy = np.gradient(heel_y_smooth, times_arr)
            # 약간 스무딩 (노이즈 방지)
            vy_sm = np.convolve(vy, np.ones(3) / 3, mode='same')

            onset_peaks = []
            total_shift_ms = 0
            for peak_idx in final_peaks:
                search_start = max(0, peak_idx - int(min_dist * 2))
                region = heel_y_smooth[search_start:peak_idx + 1]
                if len(region) < 5:
                    onset_peaks.append((peak_idx, float(times_arr[peak_idx])))
                    continue

                valley_local = int(np.argmin(region))
                valley_idx = search_start + valley_local

                # valley→peak 구간의 vy
                region_vy = vy_sm[valley_idx:peak_idx + 1]
                if len(region_vy) < 3:
                    onset_peaks.append((peak_idx, float(times_arr[peak_idx])))
                    continue

                # vy 최대 = 발이 가장 빠르게 내려오는 순간 (IC 직전)
                max_vy_local = int(np.argmax(region_vy))
                max_vy_val = region_vy[max_vy_local]

                if max_vy_val <= 0:
                    onset_peaks.append((peak_idx, float(times_arr[peak_idx])))
                    total_shift_ms += 0
                    continue

                # vy가 최대의 40%로 감소한 지점 = IC (발이 바닥에 닿으며 감속)
                # 서브프레임 보간: threshold crossing을 프레임 사이에서 정밀하게 잡음
                threshold = max_vy_val * 0.4
                ic_local = max_vy_local
                interp_time = None
                for j in range(max_vy_local + 1, len(region_vy)):
                    if region_vy[j] < threshold:
                        ic_local = j
                        # j-1 (above threshold) ~ j (below threshold) 사이 선형 보간
                        vy_above = region_vy[j - 1]
                        vy_below = region_vy[j]
                        denom = vy_above - vy_below
                        if denom > 0:
                            alpha = (vy_above - threshold) / denom
                        else:
                            alpha = 0.5
                        idx_a = valley_idx + j - 1
                        idx_b = valley_idx + j
                        idx_a = max(0, min(idx_a, len(times_arr) - 1))
                        idx_b = max(0, min(idx_b, len(times_arr) - 1))
                        interp_time = times_arr[idx_a] + alpha * (times_arr[idx_b] - times_arr[idx_a])
                        break

                onset_idx = valley_idx + ic_local
                onset_idx = max(valley_idx, min(onset_idx, peak_idx))

                if interp_time is None:
                    interp_time = times_arr[onset_idx]

                shift_ms = (times_arr[peak_idx] - interp_time) * 1000
                total_shift_ms += shift_ms
                # (frame_idx, interpolated_time) 튜플로 반환
                onset_peaks.append((onset_idx, float(interp_time)))

            avg_shift = total_shift_ms / len(onset_peaks) if onset_peaks else 0
            print(f"[HS Onset] adaptive vy-based, peaks={len(final_peaks)}, avg_shift={avg_shift:.0f}ms", flush=True)
            return onset_peaks

        # ── Vx-crossing HS refinement 모드 ──
        # 'peak_decel': 최대 감속 시점 (Initial Contact에 가장 가까움)
        # 'threshold': threshold crossing 시점 그대로 사용
        # 'zero':      vx=0 까지 추적 (기존 방식, ~300ms 늦음)
        VX_HS_MODE = 'peak_decel'

        def detect_hs_vx_crossing(heel_x_sm, times_arr, walk_sign, shared_threshold, min_gap_s=0.3):
            """Ankle X velocity crossing 기반 HS 감지
            VX_HS_MODE로 refinement 방식 선택"""
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
                if VX_HS_MODE == 'peak_decel':
                    # 최대 감속 시점: threshold crossing 전후 구간에서
                    # 가속도(dvx/dt)가 가장 음수인 지점 = 발이 가장 급격히 감속하는 순간
                    search_start = max(0, idx - window)
                    search_end = min(len(vx), idx + window // 2)
                    ax = np.gradient(vx[search_start:search_end])
                    if len(ax) > 0:
                        peak_decel_local = int(np.argmin(ax))  # 가장 큰 음의 가속도
                        best_idx = search_start + peak_decel_local
                        refined.append((best_idx, times_arr[best_idx]))
                    else:
                        refined.append((idx, times_arr[idx]))

                elif VX_HS_MODE == 'threshold':
                    # threshold crossing 시점 그대로
                    refined.append((idx, times_arr[idx]))

                else:  # 'zero' - 기존 방식
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

            ypeak_events = build_step_events_from_indices(left_hs, right_hs, use_interp_time=True)
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
        distance_m = self._height_mode_distance if self._height_mode_distance else self.walk_distance_m
        ppm = self._get_effective_ppm() or 100

        # ── 동적 PPM (원근보정) ──
        # 매 프레임의 키 픽셀 높이로 위치별 ppm 계산
        # 카메라에 가까울수록 키가 크게 보임 → ppm 높음 → 자동 원근보정
        _height_px_by_frame = {}
        if self.patient_height_m:
            for h in segment:
                if 'height_px' in h:
                    _height_px_by_frame[h['frame_idx']] = h['height_px']
            if _height_px_by_frame:
                print(f"[Processor] Dynamic PPM: {len(_height_px_by_frame)} height samples for perspective correction", flush=True)

        def get_local_ppm(frame_idx):
            """위치별 동적 ppm 반환. 해당 프레임에 height_px 없으면 글로벌 ppm 폴백."""
            if _height_px_by_frame and self.patient_height_m:
                h_px = _height_px_by_frame.get(frame_idx)
                if h_px:
                    return h_px / self.patient_height_m
            return ppm

        print(f"[Processor] ppm: {ppm:.1f}, dynamic_ppm_samples: {len(_height_px_by_frame)}", flush=True)

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

        # 평균 보폭/활보장은 L/R 계산 후 평균으로 설정 (아래에서 덮어씀)
        # fallback: 총거리 / 스텝수
        step_length = distance_m / step_count if step_count > 0 else None
        if step_length and 0.2 <= step_length <= 1.5:
            result['step_length_m'] = round(step_length, 3)
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
                # 동적 PPM: 두 위치의 평균 local ppm 사용 (원근보정)
                lp = (get_local_ppm(curr['frame_idx']) + get_local_ppm(next2['frame_idx'])) / 2
                stride_m = dx / lp

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
                    # 동적 PPM: 두 위치의 평균 local ppm 사용
                    lp = (get_local_ppm(curr['frame_idx']) + get_local_ppm(prev['frame_idx'])) / 2
                    step_m = dx / lp
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

        # ═══ 통합 step/stride를 L/R 평균으로 덮어쓰기 (거리 일관성) ═══
        l_step = result.get('left_step_length_m')
        r_step = result.get('right_step_length_m')
        if l_step and r_step:
            result['step_length_m'] = round((l_step + r_step) / 2, 3)
        elif l_step:
            result['step_length_m'] = l_step
        elif r_step:
            result['step_length_m'] = r_step

        l_stride = result.get('left_stride_length_m')
        r_stride = result.get('right_stride_length_m')
        if l_stride and r_stride:
            result['stride_length_m'] = round((l_stride + r_stride) / 2, 3)
        elif l_stride:
            result['stride_length_m'] = l_stride
        elif r_stride:
            result['stride_length_m'] = r_stride

        # ═══ 종합 대칭성 지수 (모든 SI의 평균) ═══
        if si_values:
            result['overall_symmetry_index'] = round(np.mean(si_values), 1)

        return result

    def _compute_results(self):
        """최종 임상 결과 계산"""
        if self.timer_elapsed_s <= 0:
            return

        distance_m = self._height_mode_distance if self._height_mode_distance else self.walk_distance_m
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
            self.calibrator.pixels_per_meter = self._get_effective_ppm() or 100
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
            'height_based_ppm': self.height_based_ppm,
            'actual_distance_m': distance_m,
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

        # 동적 PPM lookup (evidence clips에서도 사용)
        ppm_ev = self._get_effective_ppm() or 100
        _hpx_by_frame_ev = {}
        if self.patient_height_m:
            for h in self.ankle_history:
                if 'height_px' in h:
                    _hpx_by_frame_ev[h['frame_idx']] = h['height_px']

        def _ev_local_ppm(frame_idx):
            if _hpx_by_frame_ev and self.patient_height_m:
                h_px = _hpx_by_frame_ev.get(frame_idx)
                if h_px:
                    return h_px / self.patient_height_m
            return ppm_ev

        # 보행속도 / 케이던스 → 전체 구간
        evidence_clips['gait_velocity_ms'] = {
            'start_s': round(start_t, 2), 'end_s': round(finish_t, 2),
            'label': '전체 보행 구간',
        }
        evidence_clips['cadence_spm'] = evidence_clips['gait_velocity_ms']

        # 보폭 / 활보장 → 평균에 가장 가까운 대표 스텝 선택
        for foot in ['left', 'right']:
            foot_steps = [e for e in step_events if e.get('leading_foot') == foot]
            avg_step_cm = left_step_cm if foot == 'left' else right_step_cm
            avg_stride_cm = left_stride_cm if foot == 'left' else right_stride_cm

            if len(foot_steps) >= 2 and avg_step_cm:
                # 각 foot_step의 개별 step_length 계산 → 평균에 가장 가까운 것 선택
                best_target = None
                best_diff = float('inf')
                for fs in foot_steps:
                    fs_all_idx = step_events.index(fs)
                    if fs_all_idx < 1:
                        continue
                    prev = step_events[fs_all_idx - 1]
                    curr_mid = (fs['left_x'] + fs['right_x']) / 2
                    prev_mid = (prev['left_x'] + prev['right_x']) / 2
                    dx = abs(curr_mid - prev_mid)
                    lp = (_ev_local_ppm(fs['frame_idx']) + _ev_local_ppm(prev['frame_idx'])) / 2
                    step_cm = (dx / lp) * 100
                    diff = abs(step_cm - avg_step_cm)
                    if diff < best_diff:
                        best_diff = diff
                        best_target = fs

                if best_target:
                    target_all_idx = step_events.index(best_target)
                    anchor_idx = max(0, target_all_idx - 1)
                    anchor_event = step_events[anchor_idx]
                    evidence_clips[f'{foot}_step_length_cm'] = {
                        'start_s': round(anchor_event['time'] - 0.4, 2),
                        'end_s': round(best_target['time'] + 0.4, 2),
                        'label': f'{foot} step',
                        'target_step_num': target_all_idx + 1,
                    }

            if len(foot_steps) >= 3 and avg_stride_cm:
                # Stride: 같은 발 event[i-2]→event[i] 거리 → 평균에 가장 가까운 것 선택
                best_target = None
                best_diff = float('inf')
                for fs in foot_steps:
                    fs_all_idx = step_events.index(fs)
                    if fs_all_idx < 2:
                        continue
                    prev2 = step_events[fs_all_idx - 2]
                    foot_key = 'left_x' if foot == 'left' else 'right_x'
                    dx = abs(fs[foot_key] - prev2[foot_key])
                    lp = (_ev_local_ppm(fs['frame_idx']) + _ev_local_ppm(prev2['frame_idx'])) / 2
                    stride_cm = (dx / lp) * 100
                    diff = abs(stride_cm - avg_stride_cm)
                    if diff < best_diff:
                        best_diff = diff
                        best_target = fs
                        best_anchor = prev2

                if best_target:
                    evidence_clips[f'{foot}_stride_length_cm'] = {
                        'start_s': round(best_anchor['time'] - 0.4, 2),
                        'end_s': round(best_target['time'] + 0.4, 2),
                        'label': f'{foot} stride',
                        'target_step_num': step_events.index(best_target) + 1,
                    }

        # step_time / stride_time → 평균에 가장 가까운 대표 스텝 선택
        avg_step_time = {
            'left': gait_params.get('left_step_time_s'),
            'right': gait_params.get('right_step_time_s'),
        }
        avg_stride_time = {
            'left': gait_params.get('left_stride_time_s'),
            'right': gait_params.get('right_stride_time_s'),
        }
        for foot in ['left', 'right']:
            foot_steps = [e for e in step_events if e.get('leading_foot') == foot]
            if len(foot_steps) >= 2 and avg_step_time[foot]:
                best_target = None
                best_diff = float('inf')
                for fs in foot_steps:
                    fs_all_idx = step_events.index(fs)
                    if fs_all_idx < 1:
                        continue
                    prev = step_events[fs_all_idx - 1]
                    dt = fs['time'] - prev['time']
                    if 0.2 <= dt <= 2.0:
                        diff = abs(dt - avg_step_time[foot])
                        if diff < best_diff:
                            best_diff = diff
                            best_target = fs
                            best_anchor = prev

                if best_target:
                    evidence_clips[f'{foot}_step_time_s'] = {
                        'start_s': round(best_anchor['time'] - 0.4, 2),
                        'end_s': round(best_target['time'] + 0.4, 2),
                        'label': f'{foot} step time',
                        'target_step_num': step_events.index(best_target) + 1,
                    }

            if len(foot_steps) >= 3 and avg_stride_time[foot]:
                best_target = None
                best_diff = float('inf')
                for fs in foot_steps:
                    fs_all_idx = step_events.index(fs)
                    if fs_all_idx < 2:
                        continue
                    prev2 = step_events[fs_all_idx - 2]
                    dt = fs['time'] - prev2['time']
                    if 0.4 <= dt <= 4.0:
                        diff = abs(dt - avg_stride_time[foot])
                        if diff < best_diff:
                            best_diff = diff
                            best_target = fs
                            best_anchor = prev2

                if best_target:
                    evidence_clips[f'{foot}_stride_time_s'] = {
                        'start_s': round(best_anchor['time'] - 0.4, 2),
                        'end_s': round(best_target['time'] + 0.4, 2),
                        'label': f'{foot} stride time',
                        'target_step_num': step_events.index(best_target) + 1,
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
        ppm = self._get_effective_ppm() or 100

        # 동적 PPM lookup (원근보정)
        _hpx_by_frame = {}
        if self.patient_height_m:
            for h in self.ankle_history:
                if 'height_px' in h:
                    _hpx_by_frame[h['frame_idx']] = h['height_px']

        def _local_ppm(frame_idx):
            if _hpx_by_frame and self.patient_height_m:
                h_px = _hpx_by_frame.get(frame_idx)
                if h_px:
                    return h_px / self.patient_height_m
            return ppm

        # step_events에 개별 거리(cm) 추가
        raw_steps = gait_params['step_events']
        enriched_steps = []
        for i, ev in enumerate(raw_steps):
            entry = dict(ev)  # copy
            entry['step_num'] = i + 1
            # 이전 스텝과의 거리 = step length (반드시 다른 발이어야 step)
            if i > 0:
                prev = raw_steps[i - 1]
                dt = ev['time'] - prev['time']
                if ev.get('leading_foot') != prev.get('leading_foot') and 0.2 <= dt <= 2.0:
                    curr_mid = (ev['left_x'] + ev['right_x']) / 2
                    prev_mid = (prev['left_x'] + prev['right_x']) / 2
                    dx = abs(curr_mid - prev_mid)
                    lp = (_local_ppm(ev['frame_idx']) + _local_ppm(prev['frame_idx'])) / 2
                    step_cm = (dx / lp) * 100
                    if 10 <= step_cm <= 200:  # 합리적 범위 필터
                        entry['step_length_cm'] = round(float(step_cm), 1)
                    else:
                        entry['step_length_cm'] = None
                    entry['step_time_s'] = round(float(dt), 3)
                else:
                    entry['step_length_cm'] = None
                    entry['step_time_s'] = None
            else:
                entry['step_length_cm'] = None
                entry['step_time_s'] = None
            # 같은 발 2칸 전 = stride (midpoint 기준 — 평균 계산과 동일)
            if i >= 2 and raw_steps[i - 2].get('leading_foot') == ev.get('leading_foot'):
                prev2 = raw_steps[i - 2]
                curr_mid2 = (ev['left_x'] + ev['right_x']) / 2
                prev_mid2 = (prev2['left_x'] + prev2['right_x']) / 2
                dx2 = abs(curr_mid2 - prev_mid2)
                lp2 = (_local_ppm(ev['frame_idx']) + _local_ppm(prev2['frame_idx'])) / 2
                stride_cm = (dx2 / lp2) * 100
                dt2 = ev['time'] - prev2['time']
                if 30 <= stride_cm <= 400 and 0.4 <= dt2 <= 4.0:
                    entry['stride_length_cm'] = round(float(stride_cm), 1)
                    entry['stride_time_s'] = round(float(dt2), 3)
                else:
                    entry['stride_length_cm'] = None
                    entry['stride_time_s'] = None
            else:
                entry['stride_length_cm'] = None
                entry['stride_time_s'] = None
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
                local_p = _local_ppm(h['frame_idx'])
                dist_so_far = abs(mid_x - first_mid_x) / local_p
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
