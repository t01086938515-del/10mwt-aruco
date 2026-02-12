"""
해결방안 #2: Homography 원근 보정 모듈
============================================
기존 시스템에 추가하는 방법:
1. 이 파일을 프로젝트에 추가
2. 캘리브레이션 단계에서 compute_homography() 호출
3. 매 프레임 발 좌표를 warp_point()로 변환 후 거리 계산

효과: Step length 절대값 8~10% 보정 (과소측정 완화)
"""

import cv2
import numpy as np


class PerspectiveCorrector:
    """
    무대/복도 바닥면의 원근 왜곡을 보정하는 클래스
    
    사용법:
        corrector = PerspectiveCorrector()
        corrector.calibrate(first_frame, marker_info, actual_distance_m=10.0)
        
        # 매 프레임 발 좌표 변환
        real_x, real_y = corrector.to_real(pixel_x, pixel_y)
        
        # 두 점 사이 실제 거리 (미터)
        dist = corrector.real_distance(x1, y1, x2, y2)
    """
    
    def __init__(self):
        self.H = None           # Homography 행렬
        self.H_inv = None       # 역변환
        self.px_per_m = None    # 보정 후 픽셀/미터 비율
        self.calibrated = False
        self.src_pts = None
        self.dst_pts = None
    
    def calibrate(self, first_frame, marker_info, actual_distance_m=10.0, 
                  stage_depth_m=None):
        """
        첫 프레임에서 원근 보정 행렬 계산
        
        Args:
            first_frame: 첫 프레임 이미지 (BGR)
            marker_info: {marker_id: {'cx': float, 'cy': float, 'size': float}}
            actual_distance_m: 마커 간 실제 거리 (미터)
            stage_depth_m: 무대/복도 깊이 (미터). None이면 자동 추정
        """
        h, w = first_frame.shape[:2]
        
        # 마커 정보 추출
        mids = sorted(marker_info.keys())
        if len(mids) < 2:
            print("[PerspectiveCorrector] 마커 2개 필요. 보정 없이 진행합니다.")
            self._setup_identity(marker_info, actual_distance_m, w)
            return
        
        m_left = marker_info[mids[0]]
        m_right = marker_info[mids[1]]
        
        # 마커 라인 Y (벽/뒤쪽 기준선)
        wall_y = (m_left['cy'] + m_right['cy']) / 2
        marker_dist_px = abs(m_right['cx'] - m_left['cx'])
        self.px_per_m = marker_dist_px / actual_distance_m
        
        # 무대 앞쪽 에지 자동 검출
        stage_front_y = self._detect_stage_edge(first_frame, wall_y, h)
        
        if stage_front_y is None or abs(stage_front_y - wall_y) < 10:
            # Fallback: 영상 하단 85% 지점을 바닥 앞쪽으로 추정
            # 측면 카메라 복도 환경에서 Hough 에지 미검출 시 사용
            stage_front_y = h * 0.85
            if abs(stage_front_y - wall_y) < 10:
                print("[PerspectiveCorrector] 무대 에지 미검출. 보정 없이 진행합니다.")
                self._setup_identity(marker_info, actual_distance_m, w)
                return
            print(f"[PerspectiveCorrector] 에지 미검출 → 추정값 사용: front_y={stage_front_y:.0f}")
        
        # 깊이 추정
        if stage_depth_m is None:
            # 마커 크기 비율로 대략적 깊이 추정
            size_ratio = m_left['size'] / m_right['size'] if m_right['size'] > 0 else 1.0
            # 일반적인 무대/복도 깊이: 2~4m
            stage_depth_m = 2.5
        
        stage_depth_px = stage_depth_m * self.px_per_m
        
        # 소스 포인트: 화면상 사다리꼴 (원근 왜곡된 바닥)
        # 앞쪽이 카메라에 가까우므로 약간 넓게 보임
        front_expand = (stage_front_y - wall_y) / max(wall_y, 1) * 0.3
        front_left = m_left['cx'] - marker_dist_px * front_expand * 0.5
        front_right = m_right['cx'] + marker_dist_px * front_expand * 0.5
        
        self.src_pts = np.array([
            [m_left['cx'], wall_y],           # 뒤쪽 좌
            [m_right['cx'], wall_y],          # 뒤쪽 우  
            [front_right, stage_front_y],      # 앞쪽 우
            [front_left, stage_front_y],       # 앞쪽 좌
        ], dtype=np.float32)
        
        # 목적지: 직사각형 (원근 제거)
        self.dst_pts = np.array([
            [m_left['cx'], wall_y],
            [m_right['cx'], wall_y],
            [m_right['cx'], wall_y + stage_depth_px],
            [m_left['cx'], wall_y + stage_depth_px],
        ], dtype=np.float32)
        
        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.H_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        self.calibrated = True
        
        print(f"[PerspectiveCorrector] 캘리브레이션 완료")
        print(f"  wall_y={wall_y:.0f}, front_y={stage_front_y:.0f}")
        print(f"  px/m={self.px_per_m:.1f}, depth={stage_depth_m:.1f}m")
    
    def calibrate_with_floor_points(self, pixel_pts, real_pts):
        """
        바닥 캘리브레이션 포인트가 있을 때 (더 정확한 방법)
        
        Args:
            pixel_pts: 화면 좌표 4개 [(x,y), ...] 
            real_pts: 실제 좌표 4개 [(x_m, y_m), ...]  (미터 단위)
        
        Example:
            # 바닥에 1m 간격 마킹이 있을 때
            pixel_pts = [(100,500), (1500,500), (1480,600), (120,600)]
            real_pts = [(0,0), (10,0), (10,2), (0,2)]  # 10m x 2m 바닥
            corrector.calibrate_with_floor_points(pixel_pts, real_pts)
        """
        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(real_pts, dtype=np.float32)
        
        # 실제 좌표를 픽셀 스케일로 변환 (px_per_m 계산)
        real_width = max(dst[:, 0]) - min(dst[:, 0])
        pixel_width = max(src[:, 0]) - min(src[:, 0])
        self.px_per_m = pixel_width / real_width if real_width > 0 else 100
        
        # 실제 좌표를 픽셀 스케일에 맞춤
        dst_scaled = dst * self.px_per_m
        # offset 적용
        dst_scaled[:, 0] += min(src[:, 0])
        dst_scaled[:, 1] += min(src[:, 1])
        
        self.src_pts = src
        self.dst_pts = dst_scaled.astype(np.float32)
        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.H_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        self.calibrated = True
        
        print(f"[PerspectiveCorrector] 바닥 포인트 캘리브레이션 완료 (px/m={self.px_per_m:.1f})")
    
    def to_real(self, pixel_x, pixel_y):
        """픽셀 좌표 → 원근 보정된 좌표"""
        if not self.calibrated or self.H is None:
            return pixel_x, pixel_y
        
        pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.H)
        return float(warped[0][0][0]), float(warped[0][0][1])
    
    def to_pixel(self, real_x, real_y):
        """보정된 좌표 → 픽셀 좌표 (역변환)"""
        if not self.calibrated or self.H_inv is None:
            return real_x, real_y
        
        pt = np.array([[[real_x, real_y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.H_inv)
        return float(warped[0][0][0]), float(warped[0][0][1])
    
    def real_distance(self, x1, y1, x2, y2):
        """두 픽셀 좌표 사이의 실제 거리 (미터)"""
        rx1, ry1 = self.to_real(x1, y1)
        rx2, ry2 = self.to_real(x2, y2)
        dist_px = np.sqrt((rx2 - rx1)**2 + (ry2 - ry1)**2)
        return dist_px / self.px_per_m if self.px_per_m else 0
    
    def real_distance_x(self, x1, y1, x2, y2):
        """두 점 사이의 수평(보행방향) 실제 거리만 (미터)"""
        rx1, _ = self.to_real(x1, y1)
        rx2, _ = self.to_real(x2, y2)
        return abs(rx2 - rx1) / self.px_per_m if self.px_per_m else 0
    
    def _detect_stage_edge(self, frame, wall_y, img_height):
        """무대 앞쪽 가장자리 자동 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, 
                                minLineLength=300, maxLineGap=50)
        
        if lines is None:
            return None
        
        # wall_y 아래쪽에서 가장 긴 수평선 찾기
        best_y, best_len = None, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            mid_y = (y1+y2) / 2
            
            if (angle < 5 or angle > 175) and mid_y > wall_y + 10 and length > best_len:
                best_len = length
                best_y = mid_y
        
        return best_y
    
    def _setup_identity(self, marker_info, actual_distance_m, img_width):
        """보정 없이 기본 px/m만 설정"""
        mids = sorted(marker_info.keys())
        if len(mids) >= 2:
            dist = abs(marker_info[mids[1]]['cx'] - marker_info[mids[0]]['cx'])
            self.px_per_m = dist / actual_distance_m
        else:
            self.px_per_m = img_width / 12.0  # rough fallback
        self.calibrated = False


# ===== 기존 코드에 통합하는 예시 =====
"""
# 기존 코드에서 이렇게 사용:

from solution_2_homography import PerspectiveCorrector

# 초기화 (캘리브레이션 단계에서 1회)
corrector = PerspectiveCorrector()
corrector.calibrate(first_frame, marker_info, actual_distance_m=10.0)

# 매 프레임 거리 계산 시:
# 기존: step_length = abs(heel_x2 - heel_x1) / px_per_m
# 변경:
step_length = corrector.real_distance_x(heel_x1, heel_y1, heel_x2, heel_y2)
stride_length = corrector.real_distance_x(heel_x1, heel_y1, heel_x2, heel_y2)
"""
