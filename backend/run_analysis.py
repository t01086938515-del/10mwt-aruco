"""독립 실행 분석 스크립트 - 일관성 검증용

Usage:
    python run_analysis.py <video_path> [--runs N] [--output-dir DIR]

각 영상을 N회 반복 실행하고 결과를 JSON으로 저장합니다.
"""

import sys
import os
import json
import time
import argparse
import hashlib
from pathlib import Path

# backend 디렉토리를 path에 추가
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
from processor import FrameProcessor
from utils.video_utils import VideoReader


def find_model_path():
    """YOLOv8-Pose 모델 경로 탐색 (server.py와 동일)"""
    candidates = [
        BACKEND_DIR.parent / "yolov8n-pose.pt",
        BACKEND_DIR / "yolov8n-pose.pt",
        Path("yolov8n-pose.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolov8n-pose.pt"  # ultralytics가 자동 다운로드


def convert_numpy(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_single_analysis(video_path: str) -> dict:
    """영상 1회 분석 실행 → 결과 dict 반환"""
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
        return {"error": "ArUco calibration failed"}

    # Pass 2: 전체 프레임 분석
    reader2 = VideoReader(video_path)
    processed = 0
    for frame_idx, frame in reader2.frames():
        if frame_idx % frame_skip != 0:
            continue
        timestamp_s = frame_idx / fps
        processor.process_frame(frame, frame_idx, timestamp_s)
        processed += 1
    reader2.release()

    results = convert_numpy(processor.analysis_results) if processor.analysis_results else {}
    results['_meta'] = {
        'video': os.path.basename(video_path),
        'total_frames': total_frames,
        'processed_frames': processed,
        'fps': fps,
    }
    return results


def compare_results(results_list: list, video_name: str) -> dict:
    """N회 분석 결과를 비교하여 불일치 항목을 찾음"""
    if len(results_list) < 2:
        return {"consistent": True, "message": "비교할 결과가 부족합니다"}

    # 비교할 핵심 키들
    key_metrics = [
        'elapsed_time_s', 'speed_mps', 'step_count', 'cadence_spm',
        'step_length_m', 'stride_length_m', 'step_time_s', 'stride_time_s',
        'left_step_length_m', 'right_step_length_m', 'step_length_si',
        'left_stride_length_m', 'right_stride_length_m', 'stride_length_si',
        'left_swing_pct', 'right_swing_pct', 'left_stance_pct', 'right_stance_pct',
        'left_swing_time_s', 'right_swing_time_s', 'swing_time_si',
        'left_stance_time_s', 'right_stance_time_s', 'stance_time_si',
        'left_step_time_s', 'right_step_time_s', 'step_time_si',
        'left_stride_time_s', 'right_stride_time_s', 'stride_time_si',
        'overall_symmetry_index',
    ]

    differences = {}
    consistent_keys = []
    missing_keys = []

    for key in key_metrics:
        values = []
        for i, r in enumerate(results_list):
            val = r.get(key)
            values.append(val)

        # None 체크
        non_none = [v for v in values if v is not None]
        if not non_none:
            missing_keys.append(key)
            continue

        if len(non_none) != len(values):
            differences[key] = {
                'values': values,
                'issue': 'some_runs_missing',
                'detail': f'{len(values) - len(non_none)}/{len(values)} runs had None'
            }
            continue

        # 값 비교
        unique_vals = set(values)
        if len(unique_vals) == 1:
            consistent_keys.append(key)
        else:
            min_v = min(non_none)
            max_v = max(non_none)
            mean_v = sum(non_none) / len(non_none)
            spread = max_v - min_v
            spread_pct = (spread / abs(mean_v) * 100) if mean_v != 0 else float('inf')

            differences[key] = {
                'values': values,
                'min': min_v,
                'max': max_v,
                'mean': round(mean_v, 4),
                'spread': round(spread, 4),
                'spread_pct': round(spread_pct, 2),
            }

    # step_events 비교 (step_count 일치 여부)
    step_counts = [len(r.get('step_events', [])) for r in results_list]
    if len(set(step_counts)) > 1:
        differences['step_events_count'] = {
            'values': step_counts,
            'issue': 'different_step_event_counts'
        }

    report = {
        'video': video_name,
        'num_runs': len(results_list),
        'consistent': len(differences) == 0,
        'num_consistent': len(consistent_keys),
        'num_different': len(differences),
        'num_missing': len(missing_keys),
        'consistent_keys': consistent_keys,
        'differences': differences,
        'missing_keys': missing_keys,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description='보행 분석 일관성 검증')
    parser.add_argument('video_path', help='영상 파일 경로')
    parser.add_argument('--runs', type=int, default=5, help='반복 실행 횟수 (기본: 5)')
    parser.add_argument('--output-dir', default=None, help='결과 저장 디렉토리')
    args = parser.parse_args()

    video_path = args.video_path
    num_runs = args.runs
    video_name = Path(video_path).stem

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "consistency_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"  Consistency Test: {video_name}")
    print(f"  Runs: {num_runs}")
    print(f"=" * 60)

    all_results = []
    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        start = time.time()
        result = run_single_analysis(video_path)
        elapsed = time.time() - start
        result['_meta']['run_idx'] = run_idx
        result['_meta']['analysis_time_s'] = round(elapsed, 2)
        all_results.append(result)
        print(f"  Completed in {elapsed:.1f}s | steps={result.get('step_count', '?')} | "
              f"speed={result.get('speed_mps', '?')} | cadence={result.get('cadence_spm', '?')}")

        # 각 실행 결과 저장
        run_path = output_dir / f"{video_name}_run{run_idx}.json"
        run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    # 비교 리포트 생성
    report = compare_results(all_results, video_name)
    report_path = output_dir / f"{video_name}_comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    # 요약 출력
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON REPORT: {video_name}")
    print(f"{'=' * 60}")
    print(f"  Consistent: {report['consistent']}")
    print(f"  Consistent keys: {report['num_consistent']}")
    print(f"  Different keys: {report['num_different']}")
    print(f"  Missing keys: {report['num_missing']}")

    if report['differences']:
        print(f"\n  DIFFERENCES FOUND:")
        for key, info in report['differences'].items():
            if 'spread_pct' in info:
                print(f"    {key}: spread={info['spread_pct']}% "
                      f"(min={info['min']}, max={info['max']}, mean={info['mean']})")
            else:
                print(f"    {key}: {info.get('issue', 'unknown')} -> {info['values']}")

    print(f"\n  Results saved to: {output_dir}")
    return report


if __name__ == '__main__':
    main()
