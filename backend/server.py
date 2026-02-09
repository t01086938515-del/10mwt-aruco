# backend/server.py
"""FastAPI + WebSocket 서버

엔드포인트:
- POST /api/upload         영상 업로드 → video_id 반환
- GET  /api/frame/{id}     첫 프레임 JPEG (미리보기)
- GET  /                   프론트엔드 HTML 서빙
- WS   /ws/analyze/{id}    양방향 분석 통신

WebSocket 프로토콜:
  Client → Server: configure, start_analysis, pause, seek, reset
  Server → Client: calibration, frame_data, crossing_event, analysis_complete
"""

import asyncio
import base64
import json
import os
import sys
import uuid
import traceback
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 프로젝트 루트를 sys.path에 추가
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from processor import FrameProcessor
from utils.video_utils import VideoReader

app = FastAPI(title="AR Gait Timer Backend (Lateral/Horizontal)")

# CORS 허용 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리
UPLOAD_DIR = PROJECT_DIR / "uploads"
RESULTS_DIR = PROJECT_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 영상 메타데이터 저장
video_registry: Dict[str, Dict] = {}

# MediaPipe는 모델 자동 다운로드 (별도 경로 불필요)
def find_model_path():
    """YOLOv8-Pose 모델 경로 탐색"""
    candidates = [
        Path(__file__).parent.parent / "yolov8n-pose.pt",
        Path(__file__).parent / "yolov8n-pose.pt",
        Path("yolov8n-pose.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolov8n-pose.pt"  # ultralytics가 자동 다운로드


# ─── REST API ───

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """영상 업로드 → video_id 반환"""
    video_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    save_path = UPLOAD_DIR / f"{video_id}{ext}"

    content = await file.read()
    save_path.write_bytes(content)

    # 영상 정보 추출
    try:
        reader = VideoReader(str(save_path))
        info = reader.get_info()
        reader.release()
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return {"error": f"영상 로드 실패: {str(e)}"}

    video_registry[video_id] = {
        "id": video_id,
        "path": str(save_path),
        "filename": file.filename,
        **info,
    }

    return {
        "video_id": video_id,
        "filename": file.filename,
        **info,
    }


@app.get("/api/frame/{video_id}")
async def get_first_frame(video_id: str):
    """첫 프레임 JPEG 반환 (미리보기)"""
    if video_id not in video_registry:
        return Response(content="Video not found", status_code=404)

    video_path = video_registry[video_id]["path"]
    try:
        reader = VideoReader(video_path)
        frame = reader.get_first_frame()
        reader.release()

        if frame is None:
            return Response(content="Failed to read frame", status_code=500)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except Exception as e:
        return Response(content=str(e), status_code=500)


@app.get("/api/videos")
async def list_videos():
    """업로드된 영상 목록"""
    return list(video_registry.values())


@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    """영상 파일 스트리밍 (확장자 자동 감지)"""
    # video_registry에서 먼저 찾기
    if video_id in video_registry:
        video_path = video_registry[video_id]["path"]
    else:
        # registry에 없으면 uploads 디렉토리에서 직접 찾기
        video_path = None
        for ext in [".mp4", ".mov", ".avi", ".webm", ".mkv", ".MP4", ".MOV"]:
            candidate = UPLOAD_DIR / f"{video_id}{ext}"
            if candidate.exists():
                video_path = str(candidate)
                break

    if not video_path or not Path(video_path).exists():
        return Response(content="Video not found", status_code=404)

    # 확장자에 따른 media_type 결정
    ext = Path(video_path).suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }
    media_type = media_types.get(ext, "video/mp4")

    return FileResponse(video_path, media_type=media_type)


# ─── WebSocket 분석 ───

@app.websocket("/ws/analyze/{video_id}")
async def websocket_analyze(websocket: WebSocket, video_id: str):
    """양방향 분석 WebSocket

    Client → Server 메시지:
        { "type": "configure", "marker_size": 0.2, "start_id": 0, "finish_id": 1, "distance": 10.0 }
        { "type": "start_analysis" }
        { "type": "pause" }
        { "type": "reset" }

    Server → Client 메시지:
        { "type": "video_info", ... }
        { "type": "calibration", ... }
        { "type": "frame_data", ... }
        { "type": "crossing_event", "line": "start"|"finish", ... }
        { "type": "analysis_complete", "results": {...} }
        { "type": "error", "message": "..." }
    """
    await websocket.accept()

    if video_id not in video_registry:
        await websocket.send_json({"type": "error", "message": "Video not found"})
        await websocket.close()
        return

    video_info = video_registry[video_id]
    video_path = video_info["path"]

    # 기본 설정
    config = {
        "marker_size": 0.2,
        "start_id": 0,
        "finish_id": 1,
        "distance": 10.0,
    }

    processor: Optional[FrameProcessor] = None
    analyzing = False
    paused = False

    try:
        # 영상 정보 전송
        await websocket.send_json({
            "type": "video_info",
            **video_info,
        })

        while True:
            # 메시지 수신 (분석 중이면 non-blocking)
            if analyzing and not paused:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.001
                    )
                    msg = json.loads(raw)
                except asyncio.TimeoutError:
                    msg = None
                except Exception:
                    msg = None
            else:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

            if msg:
                msg_type = msg.get("type", "")

                if msg_type == "configure":
                    config["marker_size"] = msg.get("marker_size", 0.2)
                    config["start_id"] = msg.get("start_id", 0)
                    config["finish_id"] = msg.get("finish_id", 1)
                    config["distance"] = msg.get("distance", 10.0)
                    await websocket.send_json({
                        "type": "configured",
                        "config": config,
                    })

                elif msg_type == "start_analysis":
                    model_path = find_model_path()
                    processor = FrameProcessor(
                        model_path=model_path,
                        marker_size_m=config["marker_size"],
                        start_marker_id=config["start_id"],
                        finish_marker_id=config["finish_id"],
                        marker_distance_m=config["distance"],
                    )
                    analyzing = True
                    paused = False

                    # 배치 분석 시작
                    await _run_analysis(
                        websocket, processor, video_path, video_info
                    )
                    analyzing = False

                elif msg_type == "pause":
                    paused = not paused

                elif msg_type == "reset":
                    if processor:
                        processor.reset()
                    analyzing = False
                    paused = False
                    await websocket.send_json({"type": "reset_done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass


async def _run_analysis(
    websocket: WebSocket,
    processor: FrameProcessor,
    video_path: str,
    video_info: Dict,
):
    """2-pass 비동기 배치 분석 실행

    Pass 1: ArUco 마커만 빠르게 스캔하여 보정 완료
    Pass 2: 보정 완료 상태에서 프레임 0부터 사람 감지 + 진행률 계산
    """
    import time as _time

    reader = VideoReader(video_path)
    fps = reader.fps
    total_frames = reader.total_frames

    processor.aruco.set_camera_params(reader.width, reader.height)
    processor.set_fps(fps)

    frame_skip = 3 if fps > 45 else 2  # 60fps → 20fps, 30fps → 15fps
    # Pass 1에서는 더 큰 간격으로 스캔 (마커만 찾으면 됨)
    marker_scan_skip = 10 if fps > 45 else 5

    print(f"[Analysis] Starting 2-pass: {total_frames} frames, {fps:.1f}fps")

    analysis_start = _time.time()

    # ═══ PASS 1: ArUco 마커 스캔 (보정 완료까지) ═══
    await websocket.send_json({
        "type": "progress",
        "frame_idx": 0,
        "total_frames": total_frames,
        "percent": 0,
        "phase": "calibration_scan",
    })
    print(f"[Analysis] Pass 1: Scanning for ArUco markers...")

    try:
        for frame_idx, frame in reader.frames():
            if frame_idx % marker_scan_skip != 0:
                continue

            processor.aruco.detect_markers(frame)
            if processor.aruco.try_calibrate():
                print(f"[Analysis] Pass 1: Calibration done at frame {frame_idx}")
                break

            # 진행률 업데이트 (10% 단위)
            if frame_idx % (marker_scan_skip * 20) == 0:
                percent = round(frame_idx / total_frames * 50, 1)  # Pass 1은 0~50%
                await websocket.send_json({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "percent": percent,
                    "phase": "calibration_scan",
                })
                await asyncio.sleep(0)

    except Exception as e:
        print(f"[Analysis] Pass 1 error: {e}")
        traceback.print_exc()
    finally:
        reader.release()

    if not processor.aruco.calibrated:
        print(f"[Analysis] Pass 1: Calibration FAILED - no markers found")
        await websocket.send_json({
            "type": "error",
            "message": "ArUco 마커를 찾을 수 없습니다. 영상에 시작/끝 마커가 보이는지 확인하세요.",
        })
        return

    # 보정 정보 전송
    await websocket.send_json({
        "type": "calibration",
        **processor.aruco.get_calibration_info(),
    })

    pass1_time = _time.time() - analysis_start
    print(f"[Analysis] Pass 1 complete in {pass1_time:.1f}s. Starting Pass 2...")

    # ═══ PASS 2: 프레임 0부터 전체 분석 (보정 완료 상태) ═══
    reader2 = VideoReader(video_path)
    processed_count = 0
    pass2_start = _time.time()

    # numpy 타입을 Python 기본 타입으로 변환
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
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

    try:
        for frame_idx, frame in reader2.frames():
            if frame_idx % frame_skip != 0:
                continue

            timestamp_s = frame_idx / fps

            try:
                frame_data = processor.process_frame(frame, frame_idx, timestamp_s)
            except Exception as e:
                print(f"[Analysis] Error at frame {frame_idx}: {e}")
                traceback.print_exc()
                continue

            # 프레임 데이터 전송 (매 3프레임마다 이미지 포함)
            send_image = (processed_count % 3 == 0)
            if send_image:
                # 프레임에 스켈레톤/마커 어노테이션 그리기
                annotated = frame.copy()
                # 포즈 키포인트 그리기
                person = frame_data.get('person')
                if person and person.get('detected') and person.get('keypoints'):
                    kps = person['keypoints']
                    # 주요 관절 점 그리기
                    for kp in kps:
                        if len(kp) >= 3 and kp[2] > 0.3:
                            cv2.circle(annotated, (int(kp[0]), int(kp[1])), 4, (0, 255, 0), -1)
                    # 발목/발 강조
                    if len(kps) > 28:
                        for idx in [27, 28, 29, 30, 31, 32]:  # ankles, heels, toes
                            if idx < len(kps) and len(kps[idx]) >= 3 and kps[idx][2] > 0.3:
                                color = (255, 100, 100) if idx % 2 == 1 else (100, 100, 255)  # R=blue, L=red
                                cv2.circle(annotated, (int(kps[idx][0]), int(kps[idx][1])), 6, color, -1)

                # ArUco 마커 라인 그리기
                if processor.aruco.calibrated:
                    cal = processor.aruco.get_calibration_info()
                    sx = int(cal.get('start_x', 0))
                    fx = int(cal.get('finish_x', 0))
                    cv2.line(annotated, (sx, 0), (sx, frame.shape[0]), (0, 255, 0), 2)
                    cv2.line(annotated, (fx, 0), (fx, frame.shape[0]), (0, 0, 255), 2)

                # 타이머 표시
                timer_state = frame_data.get('timer', {}).get('state', '')
                elapsed = frame_data.get('timer', {}).get('elapsed_s', 0)
                if timer_state == 'running':
                    cv2.putText(annotated, f"{elapsed:.2f}s", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)

                # JPEG 인코딩 (저해상도로 축소)
                h, w = annotated.shape[:2]
                scale = 640 / max(h, w)
                small = cv2.resize(annotated, (int(w * scale), int(h * scale)))
                _, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                img_b64 = base64.b64encode(jpeg.tobytes()).decode('ascii')

                safe_data = convert_numpy({k: v for k, v in frame_data.items() if k != 'person'})
                await websocket.send_json({
                    "type": "frame_data",
                    **safe_data,
                    "frame_image": img_b64,
                })
            else:
                safe_data = convert_numpy({k: v for k, v in frame_data.items() if k != 'person'})
                await websocket.send_json({
                    "type": "frame_data",
                    **safe_data,
                })

            # 라인 통과 이벤트
            if frame_data['crossing_event']:
                await websocket.send_json(convert_numpy({
                    "type": "crossing_event",
                    "line": frame_data['crossing_event'],
                    "timestamp_s": frame_data['timestamp_s'],
                    "timer": frame_data['timer'],
                }))
                print(f"[Analysis] Crossing: {frame_data['crossing_event']} at {timestamp_s:.2f}s")

            processed_count += 1

            # 매 10프레임마다 진행률 전송
            if processed_count % 10 == 0:
                percent = round(50 + frame_idx / total_frames * 50, 1)  # Pass 2는 50~100%
                elapsed = _time.time() - pass2_start
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                await websocket.send_json({
                    "type": "progress",
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "percent": percent,
                })
                print(f"[Analysis] {percent}% ({frame_idx}/{total_frames}) - {fps_actual:.1f} frames/s")

            await asyncio.sleep(0)

    except Exception as e:
        print(f"[Analysis] Pass 2 error (non-fatal): {e}")
        traceback.print_exc()
        # 에러가 발생해도 이미 처리된 프레임으로 결과 생성 시도
    finally:
        reader2.release()

    elapsed_total = _time.time() - analysis_start
    print(f"[Analysis] Complete: {processed_count} frames in {elapsed_total:.1f}s (2-pass)")

    results = convert_numpy(processor.analysis_results) if processor.analysis_results else {}
    crossing_events = convert_numpy(processor.crossing_events) if processor.crossing_events else []

    # 분석 완료 (크로싱 이벤트 포함)
    await websocket.send_json({
        "type": "analysis_complete",
        "total_frames_analyzed": processed_count,
        "results": results,
        "crossing_events": crossing_events,
    })

    # 결과 저장
    if results:
        result_path = RESULTS_DIR / f"{video_info['id']}_results.json"
        result_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )


# ─── 프론트엔드 서빙 ───

@app.get("/")
async def serve_frontend():
    """프론트엔드 HTML 서빙"""
    index_path = PROJECT_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>AR Gait Timer</h1><p>index.html not found</p>")


# 정적 파일 (영상 등)
if (PROJECT_DIR / "uploads").exists():
    app.mount("/uploads", StaticFiles(directory=str(PROJECT_DIR / "uploads")), name="uploads")


# ─── 메인 ───

if __name__ == "__main__":
    print("=" * 60)
    print("  AR Gait Timer Backend (Lateral/Horizontal Analysis)")
    print("  - Mode: Left-Right Walking Analysis")
    print("  - Camera: Side view (camera perpendicular to walking path)")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
    )
