# utils/video_utils.py

import cv2
import numpy as np
from typing import Tuple, Optional, Generator
from pathlib import Path


class VideoReader:
    """영상 읽기 유틸리티"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def get_info(self) -> dict:
        return {
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'resolution': f"{self.width}x{self.height}",
            'duration_s': round(self.duration, 2)
        }

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_first_frame(self) -> Optional[np.ndarray]:
        return self.get_frame(0)

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """영상 쓰기 유틸리티"""

    def __init__(
        self, output_path: str, fps: float,
        width: int, height: int, codec: str = 'mp4v'
    ):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise ValueError(f"VideoWriter를 초기화할 수 없습니다: {output_path}")

    def write(self, frame: np.ndarray):
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def resize_frame(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720
) -> np.ndarray:
    h, w = frame.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame


def extract_thumbnail(
    video_path: str,
    output_path: str,
    frame_idx: int = 0
) -> bool:
    try:
        with VideoReader(video_path) as reader:
            frame = reader.get_frame(frame_idx)
            if frame is not None:
                cv2.imwrite(output_path, frame)
                return True
    except Exception:
        pass
    return False
