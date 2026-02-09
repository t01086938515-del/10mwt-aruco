"""WebSocket 경로 일관성 검증 스크립트

서버를 띄우고 → 각 영상을 WebSocket으로 분석 → 직접 실행 결과와 비교.

Usage:
    python ws_consistency_test.py <video_dir> [--runs N] [--output-dir DIR]
"""

import sys
import os
import json
import time
import asyncio
import argparse
import subprocess
import signal
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

import requests
import websockets
import numpy as np


def start_server():
    """백그라운드로 FastAPI 서버 시작"""
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-u", "server.py"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    # 서버 준비 대기
    for _ in range(30):
        try:
            r = requests.get("http://localhost:8000/api/videos", timeout=2)
            if r.status_code == 200:
                print("[WS-Test] Server ready")
                return proc
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Server failed to start within 30s")


def upload_video(video_path: str) -> dict:
    """POST /api/upload로 영상 업로드"""
    with open(video_path, 'rb') as f:
        r = requests.post(
            "http://localhost:8000/api/upload",
            files={"file": (os.path.basename(video_path), f, "video/quicktime")},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


async def run_ws_analysis(video_id: str) -> dict:
    """WebSocket으로 분석 실행 → 결과 반환"""
    uri = f"ws://localhost:8000/ws/analyze/{video_id}"

    async with websockets.connect(uri, max_size=50 * 1024 * 1024,
                                     ping_interval=None, ping_timeout=None) as ws:
        # video_info 수신
        msg = json.loads(await ws.recv())
        assert msg["type"] == "video_info", f"Expected video_info, got {msg['type']}"

        # configure
        await ws.send(json.dumps({
            "type": "configure",
            "marker_size": 0.2,
            "start_id": 0,
            "finish_id": 1,
            "distance": 10.0,
        }))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "configured"

        # start_analysis
        await ws.send(json.dumps({"type": "start_analysis"}))

        # 결과 수신 (analysis_complete까지)
        results = None
        crossing_events = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=300)
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "analysis_complete":
                results = msg.get("results", {})
                crossing_events = msg.get("crossing_events", [])
                break
            elif msg_type == "error":
                return {"error": msg.get("message", "Unknown error")}
            elif msg_type == "crossing_event":
                crossing_events.append(msg)

    return results


def compare_ws_vs_direct(ws_result: dict, direct_result: dict, video_name: str) -> dict:
    """WebSocket 결과와 직접 실행 결과 비교"""
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

    matches = []
    differences = {}
    missing = []

    for key in key_metrics:
        ws_val = ws_result.get(key)
        direct_val = direct_result.get(key)

        if ws_val is None and direct_val is None:
            missing.append(key)
            continue

        if ws_val is None or direct_val is None:
            differences[key] = {
                'ws': ws_val,
                'direct': direct_val,
                'issue': 'one_side_missing',
            }
            continue

        if ws_val == direct_val:
            matches.append(key)
        else:
            diff = abs(ws_val - direct_val)
            mean_val = (abs(ws_val) + abs(direct_val)) / 2
            diff_pct = (diff / mean_val * 100) if mean_val > 0 else float('inf')
            differences[key] = {
                'ws': ws_val,
                'direct': direct_val,
                'diff': round(diff, 6),
                'diff_pct': round(diff_pct, 4),
            }

    return {
        'video': video_name,
        'consistent': len(differences) == 0,
        'num_matches': len(matches),
        'num_differences': len(differences),
        'num_missing': len(missing),
        'matched_keys': matches,
        'differences': differences,
        'missing_keys': missing,
    }


async def main():
    parser = argparse.ArgumentParser(description='WebSocket 일관성 검증')
    parser.add_argument('video_dir', help='영상 디렉토리 경로')
    parser.add_argument('--runs', type=int, default=1, help='WebSocket 반복 횟수 (기본: 1)')
    parser.add_argument('--output-dir', default=None, help='결과 저장 디렉토리')
    parser.add_argument('--direct-results-dir', default=None, help='직접 실행 결과 디렉토리')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    # Windows는 case-insensitive → 중복 제거
    seen = set()
    videos = []
    for pattern in ["*.MOV", "*.mov", "*.mp4", "*.MP4"]:
        for p in sorted(video_dir.glob(pattern)):
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                videos.append(p)
    videos.sort()

    if not videos:
        print(f"No videos found in {video_dir}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "ws_consistency_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_dir = Path(args.direct_results_dir) if args.direct_results_dir else \
                 Path(__file__).parent.parent / "consistency_results"

    print(f"=" * 60)
    print(f"  WebSocket Consistency Test")
    print(f"  Videos: {len(videos)}")
    print(f"  Runs per video: {args.runs}")
    print(f"=" * 60)

    # 서버 시작
    server_proc = start_server()

    try:
        all_reports = []

        for video_path in videos:
            video_name = video_path.stem
            print(f"\n{'─' * 40}")
            print(f"  Video: {video_name}")
            print(f"{'─' * 40}")

            for run_idx in range(args.runs):
                print(f"\n  --- WS Run {run_idx + 1}/{args.runs} ---")

                # 업로드
                start = time.time()
                upload_info = upload_video(str(video_path))
                video_id = upload_info.get("video_id")
                if not video_id:
                    print(f"  Upload failed: {upload_info}")
                    continue
                print(f"  Uploaded: {video_id} ({time.time() - start:.1f}s)")

                # WebSocket 분석
                start = time.time()
                ws_result = await run_ws_analysis(video_id)
                elapsed = time.time() - start

                if "error" in ws_result:
                    print(f"  Analysis error: {ws_result['error']}")
                    continue

                print(f"  Analyzed in {elapsed:.1f}s | steps={ws_result.get('step_count', '?')} | "
                      f"speed={ws_result.get('speed_mps', '?')}")

                # WS 결과 저장
                ws_path = output_dir / f"{video_name}_ws_run{run_idx}.json"
                ws_path.write_text(json.dumps(ws_result, ensure_ascii=False, indent=2), encoding='utf-8')

                # 직접 실행 결과와 비교 (파일명 패턴이 다를 수 있음)
                direct_path = direct_dir / f"{video_name}_run0.json"
                if not direct_path.exists():
                    # prefix가 붙은 파일 탐색 (예: 23839366_IMG_5203_run0.json)
                    candidates = list(direct_dir.glob(f"*{video_name}_run0.json")) + \
                                 list(direct_dir.glob(f"*{video_name}_run1.json"))
                    if candidates:
                        direct_path = candidates[0]
                if direct_path.exists():
                    direct_result = json.loads(direct_path.read_text(encoding='utf-8'))
                    report = compare_ws_vs_direct(ws_result, direct_result, video_name)
                    report['run_idx'] = run_idx
                    all_reports.append(report)

                    report_path = output_dir / f"{video_name}_ws_vs_direct_run{run_idx}.json"
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

                    if report['consistent']:
                        print(f"  vs Direct: MATCH ({report['num_matches']}/{report['num_matches']} keys)")
                    else:
                        print(f"  vs Direct: DIFF! {report['num_differences']} keys differ:")
                        for key, info in report['differences'].items():
                            print(f"    {key}: ws={info['ws']} vs direct={info['direct']} "
                                  f"(diff={info.get('diff_pct', '?')}%)")
                else:
                    print(f"  (No direct result found at {direct_path})")

        # 종합 리포트
        summary = {
            'total_videos': len(videos),
            'total_runs': sum(1 for r in all_reports),
            'all_consistent': all(r['consistent'] for r in all_reports) if all_reports else None,
            'per_video': {},
        }
        for r in all_reports:
            vname = r['video']
            if vname not in summary['per_video']:
                summary['per_video'][vname] = []
            summary['per_video'][vname].append({
                'consistent': r['consistent'],
                'matches': r['num_matches'],
                'diffs': r['num_differences'],
                'differences': r['differences'],
            })

        summary_path = output_dir / "WS_SUMMARY.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

        print(f"\n{'=' * 60}")
        print(f"  WS CONSISTENCY SUMMARY")
        print(f"{'=' * 60}")
        print(f"  All consistent: {summary['all_consistent']}")
        for vname, runs in summary['per_video'].items():
            status = "PASS" if all(r['consistent'] for r in runs) else "DIFF"
            print(f"  {vname}: {status}")
        print(f"\n  Results saved to: {output_dir}")

    finally:
        # 서버 종료
        print("\n[WS-Test] Stopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == '__main__':
    asyncio.run(main())
