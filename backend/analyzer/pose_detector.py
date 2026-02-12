# analyzer/pose_detector.py

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple


class PoseDetector:
    """YOLO+MediaPipe 하이브리드 포즈 감지기

    YOLO: 사람 감지 + ByteTrack 추적 (GPU 가속, 강건한 detection)
    MediaPipe: YOLO bbox crop에서 33 keypoints 추출 (heel/toe 포함)

    YOLO만으로는 17 COCO keypoints (heel/toe 없음).
    MediaPipe 33 keypoints에는 heel(29,30)과 toe(31,32) 포함.
    → Heel Strike 감지 정확도 향상의 핵심.

    MediaPipe 33 keypoints:
      0:nose, 1:left_eye_inner, 2:left_eye, 3:left_eye_outer,
      4:right_eye_inner, 5:right_eye, 6:right_eye_outer,
      7:left_ear, 8:right_ear, 9:mouth_left, 10:mouth_right,
      11:left_shoulder, 12:right_shoulder, 13:left_elbow, 14:right_elbow,
      15:left_wrist, 16:right_wrist, 17:left_pinky, 18:right_pinky,
      19:left_index, 20:right_index, 21:left_thumb, 22:right_thumb,
      23:left_hip, 24:right_hip, 25:left_knee, 26:right_knee,
      27:left_ankle, 28:right_ankle,
      29:left_heel, 30:right_heel,
      31:left_foot_index, 32:right_foot_index
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

    @staticmethod
    def _patch_mediapipe_paths(mp_module):
        """Windows 한글 사용자 경로 문제 우회

        MediaPipe C++ 바인딩이 non-ASCII 경로를 처리 못하는 문제.
        ASCII junction (C:\\mp_root\\mediapipe → 실제 mediapipe)을 사용.
        SolutionBase.__init__이 __file__에서 root_path = [:-3] 으로 계산하므로
        solution_base.__file__을 junction 경로로 교체하면 root_path가 ASCII가 됨.
        """
        import os
        junction_root = r'C:\mp_root'
        junction_mp = os.path.join(junction_root, 'mediapipe')
        if not os.path.isdir(junction_mp):
            return
        mp_dir = os.path.dirname(mp_module.__file__)
        try:
            mp_dir.encode('ascii')
            return  # ASCII 경로면 패치 불필요
        except UnicodeEncodeError:
            pass
        # solution_base.__file__을 junction 경로로 교체
        # root_path = __file__.split(sep)[:-3] → C:\mp_root
        import mediapipe.python.solution_base as sb
        patched = os.path.join(junction_mp, 'python', 'solution_base.py')
        sb.__file__ = patched
        print(f"[MediaPipe] solution_base path patched → {patched}")

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.3,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1) YOLO: 사람 감지 + 추적
        print(f"[YOLOv8-Pose] Loading model: {model_path}")
        print(f"[YOLOv8-Pose] Device: {self.device}"
              f"{' (' + torch.cuda.get_device_name(0) + ')' if self.device == 'cuda' else ''}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        # warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print(f"[YOLOv8-Pose] Ready (GPU warmup done)")

        # 2) MediaPipe: YOLO crop에서 상세 33 keypoints (heel/toe 포함)
        self._mp_available = False
        self._mp_pose = None
        try:
            import mediapipe as mp
            # Windows 한글 경로 우회: junction 링크 경로로 리소스 재매핑
            self._patch_mediapipe_paths(mp)
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,  # Heavy: heel/toe 정확도 최대
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self._mp_available = True
            print(f"[MediaPipe] Pose Heavy initialized (33 keypoints with heel/toe)")
        except ImportError:
            print(f"[MediaPipe] Not available — falling back to YOLO-only (no heel/toe)")
        except Exception as e:
            print(f"[MediaPipe] Initialization failed: {e} — falling back to YOLO-only (no heel/toe)")

        self.prev_positions: List[List[float]] = []
        self._tracked_id: Optional[int] = None
        self._prev_bbox: Optional[List[float]] = None

        # MediaPipe 성공률 추적
        self._mp_success = 0
        self._mp_fail = 0

    def detect(self, frame: np.ndarray) -> Dict:
        """사람 감지 (YOLO 추적 + MediaPipe 키포인트)

        Returns:
            {'tracks': [{'bbox', 'keypoints', 'ankle_center', 'hip_center', 'confidence', 'track_id'}]}
            keypoints는 MediaPipe 33슬롯 [x_px, y_px, visibility] x 33
        """
        h, w = frame.shape[:2]

        # 1) YOLO 트래킹 — 사람 감지 + ID 유지
        results = self.model.track(frame, verbose=False, conf=self.confidence_threshold,
                                    persist=True, tracker="bytetrack.yaml")
        result = results[0]

        tracks = []

        if result.keypoints is not None and len(result.keypoints) > 0:
            n_detections = len(result.boxes)

            best_idx = self._select_best_person(result, n_detections)

            if best_idx is None:
                return {'tracks': tracks}

            # bbox & tracking
            box = result.boxes[best_idx]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            bbox = [float(x1), float(y1), float(x2), float(y2)]

            if box.id is not None:
                self._tracked_id = int(box.id[0].cpu().numpy())
            self._prev_bbox = bbox

            # 2) MediaPipe로 33 keypoints 추출 시도
            keypoints_33 = None
            if self._mp_available:
                keypoints_33 = self._run_mediapipe(frame, bbox)

            if keypoints_33 is not None:
                self._mp_success += 1
            else:
                # Fallback: YOLO 17 → MediaPipe 33슬롯 매핑 (heel/toe 없음)
                kp_data = result.keypoints[best_idx]
                keypoints_33 = self._yolo_to_33(kp_data)
                self._mp_fail += 1

            # 성공률 로그 (첫 프레임 + 매 100프레임)
            total = self._mp_success + self._mp_fail
            if total == 1 or total % 100 == 0:
                rate = self._mp_success / total * 100 if total > 0 else 0
                print(f"[PoseDetector] MediaPipe heel: {self._mp_success}/{total} ({rate:.0f}%)")

            # ankle/hip center from 33-slot keypoints
            ankle_center = self._center_from_33(keypoints_33, [27, 28], use_max_y=True)
            hip_center = self._center_from_33(keypoints_33, [23, 24], use_max_y=False)

            tracks.append({
                'bbox': bbox,
                'keypoints': keypoints_33,
                'ankle_center': ankle_center,
                'hip_center': hip_center,
                'confidence': conf,
                'track_id': self._tracked_id or 0,
            })

        return {'tracks': tracks}

    def _run_mediapipe(self, frame: np.ndarray, bbox: List[float]) -> Optional[List[List[float]]]:
        """YOLO bbox crop → MediaPipe Pose → 33 keypoints (full-frame 좌표)

        Args:
            frame: 전체 프레임 (BGR)
            bbox: YOLO detection [x1, y1, x2, y2]

        Returns:
            33 keypoints [[x, y, visibility], ...] in full-frame pixel coords
            또는 None (실패 시)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # bbox에 패딩 추가 (MediaPipe가 전신 컨텍스트 필요)
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.1)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        # MediaPipe는 RGB 입력
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_result = self._mp_pose.process(crop_rgb)

        if mp_result.pose_landmarks is None:
            return None

        # 핵심 키포인트 visibility 체크 (양발 ankle 최소 1개는 보여야)
        landmarks = mp_result.pose_landmarks.landmark
        left_ankle_vis = landmarks[27].visibility
        right_ankle_vis = landmarks[28].visibility
        if left_ankle_vis < 0.2 and right_ankle_vis < 0.2:
            return None

        # normalized coords → full-frame pixel coords
        crop_h, crop_w = crop.shape[:2]
        keypoints_33 = []
        for lm in landmarks:
            px = lm.x * crop_w + cx1
            py = lm.y * crop_h + cy1
            keypoints_33.append([float(px), float(py), float(lm.visibility)])

        return keypoints_33

    def _yolo_to_33(self, kp_data) -> List[List[float]]:
        """YOLO 17 keypoints → MediaPipe 33슬롯 매핑 (heel/toe = [0,0,0])"""
        kp_xy = kp_data.xy[0].cpu().numpy()     # (17, 2)
        kp_conf = kp_data.conf[0].cpu().numpy()  # (17,)

        keypoints_33 = [[0.0, 0.0, 0.0] for _ in range(33)]
        for yolo_idx, mp_idx in self.YOLO_TO_MP.items():
            keypoints_33[mp_idx] = [
                float(kp_xy[yolo_idx][0]),
                float(kp_xy[yolo_idx][1]),
                float(kp_conf[yolo_idx]),
            ]
        return keypoints_33

    def _center_from_33(
        self,
        kp33: List[List[float]],
        indices: List[int],
        use_max_y: bool = False,
    ) -> Optional[List[float]]:
        """33-slot keypoints에서 지정 인덱스들의 중심점 계산

        Args:
            kp33: 33-slot keypoints
            indices: 사용할 keypoint 인덱스 목록
            use_max_y: True면 Y는 max (접지점), False면 mean (중심)
        """
        valid = []
        for idx in indices:
            if kp33[idx][2] >= self.confidence_threshold:
                valid.append(kp33[idx][:2])

        if not valid:
            return None

        pts = np.array(valid)
        y_val = float(pts[:, 1].max()) if use_max_y else float(pts[:, 1].mean())
        return [float(pts[:, 0].mean()), y_val]

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

    def get_walking_person(self, tracks: List[Dict]) -> Optional[Dict]:
        """가장 큰 (가까운) 사람 1명 선택"""
        if not tracks:
            return None
        return tracks[0]

    def reset_tracking(self):
        self.prev_positions.clear()
        self._tracked_id = None
        self._prev_bbox = None
        self._mp_success = 0
        self._mp_fail = 0
        # warmup 다시
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
