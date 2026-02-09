"""분석 오버레이 영상 생성 스크립트

영상을 분석하면서 스켈레톤 + 메트릭스 오버레이를 그려 영상으로 출력.
전체 영상 + evidence clip별 개별 클립 생성.

Usage:
    python generate_overlay.py <video_path> [--output-dir DIR]
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np
from processor import FrameProcessor
from utils.video_utils import VideoReader


def find_model_path():
    candidates = [
        BACKEND_DIR.parent / "yolov8n-pose.pt",
        BACKEND_DIR / "yolov8n-pose.pt",
        Path("yolov8n-pose.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolov8n-pose.pt"


# YOLOv8 COCO 17 keypoint skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

# 색상 정의
COLOR_SKELETON = (0, 255, 0)       # green
COLOR_LEFT = (255, 100, 100)       # blue-ish (BGR)
COLOR_RIGHT = (100, 100, 255)      # red-ish (BGR)
COLOR_ANKLE_LEFT = (255, 50, 50)   # blue
COLOR_ANKLE_RIGHT = (50, 50, 255)  # red
COLOR_TEXT = (255, 255, 255)       # white
COLOR_BG = (0, 0, 0)              # black
COLOR_TIMER = (0, 255, 255)        # yellow
COLOR_START_LINE = (0, 255, 0)     # green
COLOR_FINISH_LINE = (0, 0, 255)    # red
COLOR_STEP_EVENT = (0, 200, 255)   # orange


def draw_skeleton(frame: np.ndarray, keypoints: list, conf_threshold: float = 0.3) -> np.ndarray:
    """스켈레톤 + 관절 점 그리기"""
    if not keypoints or len(keypoints) < 17:
        return frame

    # 연결선
    for (i, j) in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            pt1, pt2 = keypoints[i], keypoints[j]
            if len(pt1) >= 3 and len(pt2) >= 3 and pt1[2] > conf_threshold and pt2[2] > conf_threshold:
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                        COLOR_SKELETON, 2, cv2.LINE_AA)

    # 관절 점
    for idx, kp in enumerate(keypoints):
        if len(kp) >= 3 and kp[2] > conf_threshold:
            if idx == 15:    # left ankle
                color, radius = COLOR_ANKLE_LEFT, 7
            elif idx == 16:  # right ankle
                color, radius = COLOR_ANKLE_RIGHT, 7
            elif idx in [13, 14]:  # knees
                color, radius = (100, 255, 100), 5
            else:
                color, radius = COLOR_SKELETON, 3
            cv2.circle(frame, (int(kp[0]), int(kp[1])), radius, color, -1, cv2.LINE_AA)

    return frame


def draw_metrics_panel(frame: np.ndarray, metrics: dict, y_start: int = 10) -> np.ndarray:
    """왼쪽 상단 메트릭스 패널"""
    h, w = frame.shape[:2]
    panel_w = 340
    panel_h = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y_start), (10 + panel_w, y_start + panel_h), COLOR_BG, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    y = y_start + 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_s = 0.6
    thick = 1

    lines = []
    if 'elapsed_s' in metrics:
        lines.append(f"Time: {metrics['elapsed_s']:.2f}s")
    if 'speed_mps' in metrics:
        lines.append(f"Speed: {metrics['speed_mps']:.2f} m/s ({metrics.get('speed_kmph', 0):.1f} km/h)")
    if 'step_count' in metrics:
        lines.append(f"Steps: {metrics['step_count']}")
    if 'cadence_spm' in metrics:
        lines.append(f"Cadence: {metrics['cadence_spm']:.0f} steps/min")
    if 'step_length_m' in metrics:
        lines.append(f"Step Length: {metrics['step_length_m']:.3f} m")
    if 'stride_length_m' in metrics:
        lines.append(f"Stride Length: {metrics['stride_length_m']:.3f} m")

    for line in lines:
        cv2.putText(frame, line, (20, y), font, font_s, COLOR_TEXT, thick, cv2.LINE_AA)
        y += 26

    return frame


def draw_timer(frame: np.ndarray, state: str, elapsed_s: float) -> np.ndarray:
    """우측 상단 타이머"""
    h, w = frame.shape[:2]
    if state == 'running':
        text = f"{elapsed_s:.2f}s"
        color = COLOR_TIMER
    elif state == 'finished':
        text = f"{elapsed_s:.2f}s DONE"
        color = (0, 255, 0)
    else:
        text = "STANDBY"
        color = (128, 128, 128)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.2, 2)
    x = w - tw - 20
    cv2.putText(frame, text, (x, 50), font, 1.2, color, 2, cv2.LINE_AA)
    return frame


def draw_aruco_lines(frame: np.ndarray, start_x: int, finish_x: int) -> np.ndarray:
    """ArUco 시작/끝 마커 수직선"""
    h = frame.shape[0]
    if start_x > 0:
        cv2.line(frame, (start_x, 0), (start_x, h), COLOR_START_LINE, 2, cv2.LINE_AA)
        cv2.putText(frame, "START", (start_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, COLOR_START_LINE, 2, cv2.LINE_AA)
    if finish_x > 0:
        cv2.line(frame, (finish_x, 0), (finish_x, h), COLOR_FINISH_LINE, 2, cv2.LINE_AA)
        cv2.putText(frame, "FINISH", (finish_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, COLOR_FINISH_LINE, 2, cv2.LINE_AA)
    return frame


def draw_step_indicator(frame: np.ndarray, step_events: list, current_time: float) -> np.ndarray:
    """화면 하단에 스텝 이벤트 타임라인 표시"""
    h, w = frame.shape[:2]
    bar_y = h - 40
    bar_h = 30

    # 배경 바
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, bar_y + bar_h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    if not step_events:
        return frame

    # 시간 범위
    times = [e['time'] for e in step_events]
    t_min = min(times) - 1
    t_max = max(times) + 1

    for ev in step_events:
        t = ev['time']
        x = int((t - t_min) / (t_max - t_min) * w)
        foot = ev.get('leading_foot', 'unknown')
        color = COLOR_ANKLE_LEFT if foot == 'left' else COLOR_ANKLE_RIGHT
        cv2.line(frame, (x, bar_y), (x, bar_y + bar_h), color, 2)

        # 작은 라벨
        label = "L" if foot == 'left' else "R"
        cv2.putText(frame, label, (x - 4, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.35, color, 1, cv2.LINE_AA)

    # 현재 위치 표시
    if t_min < current_time < t_max:
        cx = int((current_time - t_min) / (t_max - t_min) * w)
        cv2.drawMarker(frame, (cx, bar_y + bar_h // 2), COLOR_TIMER,
                      cv2.MARKER_TRIANGLE_DOWN, 10, 2)

    return frame


def draw_foot_trails(frame: np.ndarray, ankle_history: list, current_idx: int,
                     trail_length: int = 30) -> np.ndarray:
    """최근 N프레임의 발목 궤적 그리기"""
    start = max(0, current_idx - trail_length)

    for i in range(start + 1, current_idx + 1):
        if i >= len(ankle_history):
            break
        prev = ankle_history[i - 1]
        curr = ankle_history[i]

        alpha = (i - start) / trail_length  # 0 → 1 (최근일수록 진하게)
        thickness = max(1, int(alpha * 3))

        # Left ankle trail
        if prev.get('left_x') and curr.get('left_x'):
            color_l = tuple(int(c * alpha) for c in COLOR_ANKLE_LEFT)
            cv2.line(frame,
                    (int(prev['left_x']), int(prev['left_y'])),
                    (int(curr['left_x']), int(curr['left_y'])),
                    color_l, thickness, cv2.LINE_AA)

        # Right ankle trail
        if prev.get('right_x') and curr.get('right_x'):
            color_r = tuple(int(c * alpha) for c in COLOR_ANKLE_RIGHT)
            cv2.line(frame,
                    (int(prev['right_x']), int(prev['right_y'])),
                    (int(curr['right_x']), int(curr['right_y'])),
                    color_r, thickness, cv2.LINE_AA)

    return frame


def generate_overlay_video(video_path: str, output_dir: str) -> dict:
    """오버레이 영상 생성 (전체 + evidence clips)"""
    video_name = Path(video_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Generating Overlay: {video_name}")
    print(f"{'=' * 60}")

    # ─── 분석 실행 (Pass 1: 캘리브레이션, Pass 2: 전체 분석) ───
    processor = FrameProcessor(
        model_path=find_model_path(),
        marker_size_m=0.2,
        start_marker_id=0,
        finish_marker_id=1,
        marker_distance_m=10.0,
    )

    reader = VideoReader(video_path)
    fps = reader.fps
    total_frames = reader.total_frames
    w, h = reader.width, reader.height
    processor.set_fps(fps)
    processor.aruco.set_camera_params(w, h)

    frame_skip = 3 if fps > 45 else 2
    marker_scan_skip = 10 if fps > 45 else 5

    # Pass 1: ArUco
    print(f"  Pass 1: ArUco calibration...")
    for frame_idx, frame in reader.frames():
        if frame_idx % marker_scan_skip != 0:
            continue
        processor.aruco.detect_markers(frame)
        if processor.aruco.try_calibrate():
            print(f"  Calibrated at frame {frame_idx}")
            break
    reader.release()

    if not processor.aruco.calibrated:
        return {"error": "Calibration failed", "video": video_name}

    cal_info = processor.aruco.get_calibration_info()
    start_x = int(cal_info.get('start_x', 0))
    finish_x = int(cal_info.get('finish_x', 0))

    # Pass 2: 전체 분석 + 오버레이 프레임 수집
    print(f"  Pass 2: Full analysis with overlay...")
    reader2 = VideoReader(video_path)
    output_fps = fps / frame_skip  # 실제 출력 FPS

    # 전체 오버레이 영상 Writer
    full_output_path = str(output_dir / f"{video_name}_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(full_output_path, fourcc, output_fps, (w, h))

    # 프레임 데이터 수집 (evidence clip 추출용)
    frame_data_list = []  # [(frame_idx, timestamp_s, annotated_frame), ...]
    ankle_trail = []
    processed = 0

    start_time = time.time()

    for frame_idx, frame in reader2.frames():
        if frame_idx % frame_skip != 0:
            continue

        timestamp_s = frame_idx / fps
        frame_data = processor.process_frame(frame, frame_idx, timestamp_s)

        # ─── 오버레이 그리기 ───
        annotated = frame.copy()

        # 1. ArUco 라인
        annotated = draw_aruco_lines(annotated, start_x, finish_x)

        # 2. 스켈레톤
        person = frame_data.get('person')
        if person and person.get('detected') and person.get('keypoints'):
            kps = person['keypoints']
            annotated = draw_skeleton(annotated, kps)

            # 발목 궤적 데이터 수집
            if len(kps) > 16:
                ankle_trail.append({
                    'left_x': kps[15][0] if kps[15][2] > 0.3 else None,
                    'left_y': kps[15][1] if kps[15][2] > 0.3 else None,
                    'right_x': kps[16][0] if kps[16][2] > 0.3 else None,
                    'right_y': kps[16][1] if kps[16][2] > 0.3 else None,
                })
            else:
                ankle_trail.append({})

            # 3. 발목 궤적
            annotated = draw_foot_trails(annotated, ankle_trail, len(ankle_trail) - 1)

        else:
            ankle_trail.append({})

        # 4. 타이머
        timer = frame_data.get('timer', {})
        annotated = draw_timer(annotated, timer.get('state', 'standby'), timer.get('elapsed_s', 0))

        # 5. 프레임 정보 (좌하단)
        info_text = f"Frame: {frame_idx}/{total_frames}  t={timestamp_s:.2f}s"
        cv2.putText(annotated, info_text, (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # Writer에 쓰기
        writer.write(annotated)
        frame_data_list.append((frame_idx, timestamp_s, annotated))

        processed += 1
        if processed % 100 == 0:
            elapsed = time.time() - start_time
            pct = frame_idx / total_frames * 100
            print(f"  {pct:.0f}% ({processed} frames, {elapsed:.1f}s)")

    reader2.release()
    writer.release()

    elapsed_total = time.time() - start_time
    print(f"  Full overlay: {processed} frames in {elapsed_total:.1f}s → {full_output_path}")

    # ─── 분석 결과로 메트릭 패널 포함 영상 생성 ───
    results = processor.analysis_results or {}

    # ─── Evidence Clips 추출 ───
    evidence_clips = results.get('evidence_clips', {})
    step_events = results.get('step_events', [])
    clip_paths = {}

    if evidence_clips:
        print(f"\n  Extracting {len(evidence_clips)} evidence clips...")

        for clip_name, clip_info in evidence_clips.items():
            start_s = clip_info.get('start_s', 0)
            end_s = clip_info.get('end_s', 0)
            label = clip_info.get('label', clip_name)

            if start_s >= end_s:
                continue

            # 해당 시간 범위의 프레임 필터
            clip_frames = [(fi, ts, frm) for fi, ts, frm in frame_data_list
                          if start_s <= ts <= end_s]

            if not clip_frames:
                continue

            # 클립 파일명 (안전한 이름으로)
            safe_name = clip_name.replace('.', '_').replace('/', '_')
            clip_path = str(output_dir / f"{video_name}_clip_{safe_name}.mp4")

            clip_writer = cv2.VideoWriter(clip_path, fourcc, output_fps,
                                         (clip_frames[0][2].shape[1], clip_frames[0][2].shape[0]))

            for fi, ts, frm in clip_frames:
                # 클립 라벨 추가
                labeled = frm.copy()
                overlay = labeled.copy()
                cv2.rectangle(overlay, (0, 0), (labeled.shape[1], 35), COLOR_BG, -1)
                labeled = cv2.addWeighted(overlay, 0.6, labeled, 0.4, 0)
                cv2.putText(labeled, f"[{label}] {clip_name}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TIMER, 2, cv2.LINE_AA)
                clip_writer.write(labeled)

            clip_writer.release()
            clip_paths[clip_name] = clip_path
            print(f"    {clip_name}: {len(clip_frames)} frames → {clip_path}")

    # ─── 메트릭스 요약 영상 (정지 프레임 + 결과 표시) ───
    if results:
        summary_path = str(output_dir / f"{video_name}_summary.mp4")
        summary_writer = cv2.VideoWriter(summary_path, fourcc, 1, (w, h))  # 1fps

        # 마지막 프레임 사용
        if frame_data_list:
            last_frame = frame_data_list[-1][2].copy()

            # 메트릭스 패널
            metrics = {
                'elapsed_s': results.get('elapsed_time_s', 0),
                'speed_mps': results.get('speed_mps', 0),
                'speed_kmph': results.get('speed_kmph', 0),
                'step_count': results.get('step_count', 0),
                'cadence_spm': results.get('cadence_spm', 0),
                'step_length_m': results.get('step_length_m', 0),
                'stride_length_m': results.get('stride_length_m', 0),
            }
            last_frame = draw_metrics_panel(last_frame, metrics)

            # 스텝 이벤트 바
            if step_events:
                last_frame = draw_step_indicator(last_frame, step_events, 999)

            # 판정 결과 (우측)
            judgment = results.get('judgment', {})
            velocity_j = judgment.get('velocity', {})
            if velocity_j:
                perry = velocity_j.get('perry_class', '')
                y_j = 250
                cv2.putText(last_frame, f"Perry: {perry}", (w - 300, y_j),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TIMER, 2, cv2.LINE_AA)

            # 5초간 표시
            for _ in range(5):
                summary_writer.write(last_frame)

            summary_writer.release()
            print(f"  Summary: {summary_path}")

    # 결과 JSON
    result_info = {
        'video': video_name,
        'full_overlay': full_output_path,
        'summary': str(output_dir / f"{video_name}_summary.mp4"),
        'evidence_clips': clip_paths,
        'total_frames_processed': processed,
        'generation_time_s': round(elapsed_total, 1),
        'analysis_results': {
            'step_count': results.get('step_count'),
            'speed_mps': results.get('speed_mps'),
            'cadence_spm': results.get('cadence_spm'),
            'elapsed_time_s': results.get('elapsed_time_s'),
        }
    }
    info_path = output_dir / f"{video_name}_overlay_info.json"
    info_path.write_text(json.dumps(result_info, ensure_ascii=False, indent=2), encoding='utf-8')

    return result_info


def main():
    parser = argparse.ArgumentParser(description='분석 오버레이 영상 생성')
    parser.add_argument('video_path', help='영상 파일 경로')
    parser.add_argument('--output-dir', default=None, help='출력 디렉토리')
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(__file__).parent.parent / "overlay_clips")

    result = generate_overlay_video(args.video_path, output_dir)

    if 'error' in result:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)
    else:
        print(f"\nDone! Files in: {output_dir}")
        print(f"  Full overlay: {result['full_overlay']}")
        print(f"  Evidence clips: {len(result['evidence_clips'])}")


if __name__ == '__main__':
    main()
