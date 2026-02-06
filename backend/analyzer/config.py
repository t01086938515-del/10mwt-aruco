# analyzer/config.py

class AnalyzerConfig:
    """분석기 설정값"""

    # 키포인트 신뢰도 임계값
    CONFIDENCE_THRESHOLD = 0.5

    # 사용할 키포인트 (우선순위)
    PRIMARY_KEYPOINTS = ['left_ankle', 'right_ankle']
    BACKUP_KEYPOINTS = ['left_hip', 'right_hip']

    # 키포인트별 가중치
    KEYPOINT_WEIGHTS = {
        'ankle': 1.0,
        'hip': 0.7,
        'knee': 0.5
    }

    # 보간 설정
    MAX_GAP_FRAMES = 10
    SMOOTHING_WINDOW = 5

    # 칼만 필터 설정
    PROCESS_NOISE = 0.01
    MEASUREMENT_NOISE = 0.1

    # 기본값
    DEFAULT_CORRIDOR_WIDTH_M = 2.0
    DEFAULT_TEST_DISTANCE_M = 10.0
    DEFAULT_PATIENT_HEIGHT_CM = 170.0

    # 임상 기준
    COMMUNITY_AMBULATION_THRESHOLD = 0.8  # m/s
    HOME_AMBULATION_THRESHOLD = 0.4  # m/s
    FALL_RISK_THRESHOLD = 0.6  # m/s

    # ArUco 설정
    ARUCO_DICT_ID = 0  # cv2.aruco.DICT_4X4_50
    DEFAULT_MARKER_SIZE_M = 0.2
    DEFAULT_START_MARKER_ID = 0
    DEFAULT_FINISH_MARKER_ID = 1
    DEFAULT_MARKER_DISTANCE_M = 10.0
