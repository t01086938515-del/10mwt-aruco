"""3-way 선택 + MediaPipe 힐좌표 품질 테스트

4개 영상(5199, 5200, 5201, 5203)에서:
1. MediaPipe heel 좌표 감지율
2. 3-way 선택 결과 (Y-peak / Vx-crossing / Foot Separation)
3. cadence, step count, regularity score
"""

import sys
import os
import time
import io
from pathlib import Path
from contextlib import redirect_stdout

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

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


def analyze_video(video_path: str):
    """영상 분석 + 진단 정보 수집"""
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
    processor.set_fps(fps)
    processor.aruco.set_camera_params(reader.width, reader.height)

    frame_skip = 3 if fps > 45 else 2
    marker_scan_skip = 10 if fps > 45 else 5

    # Pass 1: ArUco 마커 스캔
    for frame_idx, frame in reader.frames():
        if frame_idx % marker_scan_skip != 0:
            continue
        processor.aruco.detect_markers(frame)
        if processor.aruco.try_calibrate():
            break
    reader.release()

    if not processor.aruco.calibrated:
        return None, "ArUco 캘리브레이션 실패"

    # Pass 2: 전체 프레임 분석 (stdout 캡처)
    log_buf = io.StringIO()
    reader2 = VideoReader(video_path)
    processed = 0
    with redirect_stdout(log_buf):
        for frame_idx, frame in reader2.frames():
            if frame_idx % frame_skip != 0:
                continue
            timestamp_s = frame_idx / fps
            processor.process_frame(frame, frame_idx, timestamp_s)
            processed += 1
    reader2.release()

    logs = log_buf.getvalue()

    # === 진단 정보 수집 ===

    # 1) MediaPipe 힐좌표 가용률
    total_history = len(processor.ankle_history)
    heel_left_count = sum(1 for h in processor.ankle_history if 'left_heel_x' in h)
    heel_right_count = sum(1 for h in processor.ankle_history if 'right_heel_x' in h)
    toe_left_count = sum(1 for h in processor.ankle_history if 'left_toe_x' in h)
    toe_right_count = sum(1 for h in processor.ankle_history if 'right_toe_x' in h)

    heel_info = {
        'total_frames': total_history,
        'left_heel': heel_left_count,
        'right_heel': heel_right_count,
        'left_toe': toe_left_count,
        'right_toe': toe_right_count,
        'left_heel_pct': round(heel_left_count / total_history * 100, 1) if total_history else 0,
        'right_heel_pct': round(heel_right_count / total_history * 100, 1) if total_history else 0,
        'left_toe_pct': round(toe_left_count / total_history * 100, 1) if total_history else 0,
        'right_toe_pct': round(toe_right_count / total_history * 100, 1) if total_history else 0,
    }

    # 2) 로그에서 선택 결과 파싱
    detection_method = 'unknown'
    for line in logs.split('\n'):
        if '→ Selected' in line or '→ Using' in line or '→ All <70' in line or '→ Fallback' in line:
            detection_method = line.strip()
        if 'Y-peak result:' in line:
            ypeak_line = line.strip()
        if 'Vx-crossing:' in line and 'L=' in line:
            vx_line = line.strip()
        if 'Foot Separation:' in line:
            sep_line = line.strip()

    # 3) Homography 보정 상태
    homography = processor.perspective_corrector.calibrated

    results = processor.analysis_results
    return {
        'results': results,
        'heel_info': heel_info,
        'detection_method': detection_method,
        'homography': homography,
        'fps': fps,
        'total_frames': total_frames,
        'processed': processed,
        'logs': logs,
    }, None


def main():
    video_dir = BACKEND_DIR.parent / "sigital"
    videos = ['IMG_5199.MOV', 'IMG_5200.MOV', 'IMG_5201.MOV', 'IMG_5203.MOV']

    print("=" * 80)
    print("  3-Way Selection + MediaPipe Heel 좌표 품질 테스트")
    print("=" * 80)

    for video_name in videos:
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"\n[SKIP] {video_name} - 파일 없음")
            continue

        print(f"\n{'─' * 80}")
        print(f"  {video_name}")
        print(f"{'─' * 80}")

        start = time.time()
        data, error = analyze_video(str(video_path))
        elapsed = time.time() - start

        if error:
            print(f"  [ERROR] {error}")
            continue

        results = data['results']
        heel = data['heel_info']

        # MediaPipe Heel 좌표 품질
        print(f"\n  [MediaPipe Heel 좌표]")
        print(f"    전체 프레임: {heel['total_frames']}")
        print(f"    Left  Heel: {heel['left_heel']}/{heel['total_frames']} ({heel['left_heel_pct']}%)")
        print(f"    Right Heel: {heel['right_heel']}/{heel['total_frames']} ({heel['right_heel_pct']}%)")
        print(f"    Left  Toe:  {heel['left_toe']}/{heel['total_frames']} ({heel['left_toe_pct']}%)")
        print(f"    Right Toe:  {heel['right_toe']}/{heel['total_frames']} ({heel['right_toe_pct']}%)")

        # 3-way 선택 결과
        print(f"\n  [3-Way 선택]")
        # 로그에서 각 방법별 결과 추출
        for line in data['logs'].split('\n'):
            if 'Y-peak result:' in line:
                print(f"    {line.strip()}")
            if 'Vx-crossing:' in line and 'L=' in line:
                print(f"    {line.strip()}")
            if 'Foot Separation:' in line:
                print(f"    {line.strip()}")
            if '→ Selected' in line or '→ All <70' in line or '→ Fallback' in line:
                print(f"    ★ {line.strip()}")

        # Homography 보정
        print(f"\n  [Homography] {'활성' if data['homography'] else '비활성'}")

        # 분석 결과
        if results:
            print(f"\n  [결과]")
            print(f"    소요시간: {results.get('elapsed_time_s', '?')}s")
            print(f"    속도: {results.get('speed_mps', '?')} m/s ({results.get('speed_kmph', '?')} km/h)")
            print(f"    Step count: {results.get('step_count', '?')}")
            print(f"    Cadence: {results.get('cadence_spm', '?')} spm")
            print(f"    Step length: {results.get('step_length_m', '?')} m")
            print(f"    Stride length: {results.get('stride_length_m', '?')} m")
            print(f"    L stride: {results.get('left_stride_length_m', '?')}m | R stride: {results.get('right_stride_length_m', '?')}m | SI: {results.get('stride_length_si', '?')}%")
            print(f"    L step: {results.get('left_step_length_m', '?')}m | R step: {results.get('right_step_length_m', '?')}m | SI: {results.get('step_length_si', '?')}%")
            print(f"    Overall SI: {results.get('overall_symmetry_index', '?')}%")
            print(f"    Swing % - L: {results.get('left_swing_pct', '?')}% | R: {results.get('right_swing_pct', '?')}%")
            print(f"    Stride Time SI: {results.get('stride_time_si', '?')}%")
        else:
            print(f"\n  [결과] 분석 실패 (START/FINISH 미감지)")

        print(f"\n  분석 시간: {elapsed:.1f}s")

    print(f"\n{'=' * 80}")
    print("  테스트 완료")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
