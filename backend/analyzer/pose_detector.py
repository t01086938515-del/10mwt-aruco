# analyzer/pose_detector.py

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple


class PoseDetector:
    """YOLOv8-Pose GPU 기반 포즈 감지기

    MediaPipe 대신 YOLOv8-Pose 사용 → RTX 3050 GPU 가속.
    17 keypoints (COCO format):
      0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
      5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
      9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
      13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

    프론트엔드 호환을 위해 33개 keypoint 슬롯으로 매핑 출력.
    """

    # YOLOv8-Pose 17 keypoints (COCO)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # MediaPipe 33 keypoint 인덱스 (프론트엔드 호환용 매핑 대상)
    MP_NOSE = 0
    MP_LEFT_EYE = 2
    MP_RIGHT_EYE = 5
    MP_LEFT_EAR = 7
    MP_RIGHT_EAR = 8
    MP_LEFT_SHOULDER = 11
    MP_RIGHT_SHOULDER = 12
    MP_LEFT_ELBOW = 13
    MP_RIGHT_ELBOW = 14
    MP_LEFT_WRIST = 15
    MP_RIGHT_WRIST = 16
    MP_LEFT_HIP = 23
    MP_RIGHT_HIP = 24
    MP_LEFT_KNEE = 25
    MP_RIGHT_KNEE = 26
    MP_LEFT_ANKLE = 27
    MP_RIGHT_ANKLE = 28

    # YOLO → MediaPipe 33슬롯 매핑 테이블
    YOLO_TO_MP = {
        0: 0,    # nose
        1: 2,    # left_eye → mp index 2
        2: 5,    # right_eye → mp index 5
        3: 7,    # left_ear
        4: 8,    # right_ear
        5: 11,   # left_shoulder
        6: 12,   # right_shoulder
        7: 13,   # left_elbow
        8: 14,   # right_elbow
        9: 15,   # left_wrist
        10: 16,  # right_wrist
        11: 23,  # left_hip
        12: 24,  # right_hip
        13: 25,  # left_knee
        14: 26,  # right_knee
        15: 27,  # left_ankle
        16: 28,  # right_ankle
    }

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.3,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"[YOLOv8-Pose] Loading model: {model_path}")
        print(f"[YOLOv8-Pose] Device: {self.device}"
              f"{' (' + torch.cuda.get_device_name(0) + ')' if self.device == 'cuda' else ''}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        # warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print(f"[YOLOv8-Pose] Ready (GPU warmup done)")

        self.prev_positions: List[List[float]] = []
        self._tracked_id: Optional[int] = None  # 추적 중인 사람의 track ID
        self._prev_bbox: Optional[List[float]] = None  # 이전 bbox (연속성 체크)

    def detect(self, frame: np.ndarray) -> Dict:
        """YOLOv8-Pose로 사람 감지 (트래킹 기반)

        Returns:
            {'tracks': [{'bbox', 'keypoints', 'ankle_center', 'hip_center', 'confidence', 'track_id'}]}
            keypoints는 MediaPipe 33슬롯 호환 포맷 [x_px, y_px, visibility] × 33
        """
        h, w = frame.shape[:2]

        # 트래킹 모드 사용 - 사람 ID를 유지하여 다른 객체로 점프 방지
        results = self.model.track(frame, verbose=False, conf=self.confidence_threshold,
                                    persist=True, tracker="bytetrack.yaml")
        result = results[0]

        tracks = []

        if result.keypoints is not None and len(result.keypoints) > 0:
            n_detections = len(result.boxes)

            # 추적 ID 기반으로 동일 인물 선택
            best_idx = self._select_best_person(result, n_detections)

            if best_idx is None:
                return {'tracks': tracks}

            # bbox
            box = result.boxes[best_idx]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            bbox = [float(x1), float(y1), float(x2), float(y2)]

            # track ID 저장
            if box.id is not None:
                self._tracked_id = int(box.id[0].cpu().numpy())
            self._prev_bbox = bbox

            # 17 COCO keypoints → 33 MediaPipe 슬롯으로 변환
            kp_data = result.keypoints[best_idx]
            kp_xy = kp_data.xy[0].cpu().numpy()     # (17, 2)
            kp_conf = kp_data.conf[0].cpu().numpy()  # (17,)

            # 33슬롯 초기화 (모두 [0, 0, 0])
            keypoints_33 = [[0.0, 0.0, 0.0] for _ in range(33)]

            for yolo_idx, mp_idx in self.YOLO_TO_MP.items():
                x_val = float(kp_xy[yolo_idx][0])
                y_val = float(kp_xy[yolo_idx][1])
                c_val = float(kp_conf[yolo_idx])
                keypoints_33[mp_idx] = [x_val, y_val, c_val]

            # 발 접지점 (발목 기반 - YOLO에는 heel/toe 없음)
            ankle_center = self._get_ankle_center(kp_xy, kp_conf)
            hip_center = self._get_hip_center(kp_xy, kp_conf)

            tracks.append({
                'bbox': bbox,
                'keypoints': keypoints_33,
                'ankle_center': ankle_center,
                'hip_center': hip_center,
                'confidence': conf,
                'track_id': self._tracked_id or 0,
            })

        return {'tracks': tracks}

    def _select_best_person(self, result, n_detections: int) -> Optional[int]:
        """추적 중인 사람 선택 (ID 기반, 없으면 가장 큰 bbox)"""
        # tracked ID가 있으면 해당 ID를 찾음
        if self._tracked_id is not None:
            for i in range(n_detections):
                box = result.boxes[i]
                if box.id is not None:
                    tid = int(box.id[0].cpu().numpy())
                    if tid == self._tracked_id:
                        return i

        # tracked ID를 못 찾으면 → IoU 기반 매칭 (이전 bbox와 가장 비슷한 것)
        if self._prev_bbox is not None:
            best_iou = 0.3  # 최소 IoU 임계값
            best_idx = None
            for i in range(n_detections):
                box = result.boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                iou = self._compute_iou(self._prev_bbox, [x1, y1, x2, y2])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx is not None:
                return best_idx

        # fallback: 가장 큰 bbox
        if n_detections > 0:
            areas = []
            for i in range(n_detections):
                box = result.boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                areas.append((x2 - x1) * (y2 - y1))
            return int(np.argmax(areas))

        return None

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """두 bbox의 IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _get_ankle_center(self, kp_xy: np.ndarray, kp_conf: np.ndarray) -> Optional[List[float]]:
        """발목 중심점 (COCO: 15=left_ankle, 16=right_ankle)"""
        valid = []
        for idx in [self.LEFT_ANKLE, self.RIGHT_ANKLE]:
            if kp_conf[idx] >= self.confidence_threshold:
                valid.append(kp_xy[idx])

        if not valid:
            return None
        pts = np.array(valid)
        return [float(pts[:, 0].mean()), float(pts[:, 1].max())]

    def _get_hip_center(self, kp_xy: np.ndarray, kp_conf: np.ndarray) -> Optional[List[float]]:
        """골반 중심점 (COCO: 11=left_hip, 12=right_hip)"""
        valid = []
        for idx in [self.LEFT_HIP, self.RIGHT_HIP]:
            if kp_conf[idx] >= self.confidence_threshold:
                valid.append(kp_xy[idx])

        if not valid:
            return None
        pts = np.array(valid)
        return [float(pts[:, 0].mean()), float(pts[:, 1].mean())]

    def get_walking_person(self, tracks: List[Dict]) -> Optional[Dict]:
        """가장 큰 (가까운) 사람 1명 선택"""
        if not tracks:
            return None
        return tracks[0]

    def reset_tracking(self):
        self.prev_positions.clear()
        self._tracked_id = None
        self._prev_bbox = None
        # warmup 다시
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
