# analyzer/gait_judgment.py
"""보행 분석 판정 모듈

측정된 보행 변수를 정상 참고치와 비교하여 판정 결과를 산출.
- 보행속도: Perry et al.(1995) 기능적 보행 분류
- 나머지 변수: 정상 범위 대비 편차 + 임상 코멘트
"""

from typing import Dict, List, Optional, Any

# ═══ 정상 참고치 (한국인 성인 기준) ═══
NORMAL_RANGES = {
    'gait_velocity_ms':   {'min': 1.0,  'max': 1.4,  'unit': 'm/s'},
    'cadence_spm':        {'min': 100,  'max': 120,  'unit': 'steps/min'},
    'left_step_length_cm':  {'min': 55,   'max': 70,   'unit': 'cm'},
    'right_step_length_cm': {'min': 55,   'max': 70,   'unit': 'cm'},
    'left_stride_length_cm':  {'min': 110,  'max': 140,  'unit': 'cm'},
    'right_stride_length_cm': {'min': 110,  'max': 140,  'unit': 'cm'},
    'step_time_s':        {'min': 0.50, 'max': 0.60, 'unit': 's'},
    'stride_time_s':      {'min': 1.00, 'max': 1.20, 'unit': 's'},
    'stance_ratio_pct':   {'min': 58,   'max': 62,   'unit': '%'},
    'swing_ratio_pct':    {'min': 38,   'max': 42,   'unit': '%'},
    'double_support_pct': {'min': 18,   'max': 24,   'unit': '%'},
    'single_support_pct': {'min': 38,   'max': 40,   'unit': '%'},
}

# ═══ 임상 해석 코멘트 ═══
CLINICAL_COMMENTS = {
    'stance_ratio_pct': {
        'high': '입각기 증가: 보행 안정성 보상 전략 가능성',
        'low':  '입각기 감소: 빠른 보행 또는 측정 오차 확인 필요',
    },
    'swing_ratio_pct': {
        'high': '유각기 증가: 빠른 보행 시 정상적 변화',
        'low':  '유각기 감소: 낙상 위험 증가 방향 (Verghese 2009)',
    },
    'double_support_pct': {
        'high': '양하지 지지기 증가: 안정성 보상 패턴, 낙상 위험 증가 방향 (RR 1.165, Verghese 2009)',
        'low':  '양하지 지지기 감소: 빠른 보행 시 정상적 변화',
    },
    'single_support_pct': {
        'high': '단하지 지지기 증가: 빠른 보행 시 정상적 변화',
        'low':  '단하지 지지기 감소: 균형 능력 저하 가능성',
    },
    'left_step_length_cm': {
        'high': '좌측 보폭 증가: 정상 범위 확인',
        'low':  '좌측 보폭 감소: 낙상 이력자에서 관찰되는 패턴 (Verghese 2009)',
    },
    'right_step_length_cm': {
        'high': '우측 보폭 증가: 정상 범위 확인',
        'low':  '우측 보폭 감소: 낙상 이력자에서 관찰되는 패턴 (Verghese 2009)',
    },
    'left_stride_length_cm': {
        'high': '좌측 활보장 증가: 정상 범위 확인',
        'low':  '좌측 활보장 감소: 노화 또는 보행 장애 시 10~20% 감소 (AAFP 2010)',
    },
    'right_stride_length_cm': {
        'high': '우측 활보장 증가: 정상 범위 확인',
        'low':  '우측 활보장 감소: 노화 또는 보행 장애 시 10~20% 감소 (AAFP 2010)',
    },
    'cadence_spm': {
        'high': '분속수 증가: 짧은 보폭 보상 가능성 (한국인 노인 특징, Kim & Lee 2020)',
        'low':  '분속수 감소: 느린 보행 반영',
    },
    'step_time_s': {
        'high': 'Step time 증가: 느린 보행 반영',
        'low':  'Step time 감소: 빠른 보행 반영',
    },
    'stride_time_s': {
        'high': 'Stride time 증가: 느린 보행 반영',
        'low':  'Stride time 감소: 빠른 보행 반영',
    },
    'gait_velocity_ms': {
        'high': '',
        'low':  '',
    },
}

# ═══ 한글 변수명 매핑 ═══
DISPLAY_NAMES = {
    'gait_velocity_ms':   '보행속도',
    'cadence_spm':        '분속수 (Cadence)',
    'left_step_length_cm':  '좌측 보폭 (L Step)',
    'right_step_length_cm': '우측 보폭 (R Step)',
    'left_stride_length_cm':  '좌측 활보장 (L Stride)',
    'right_stride_length_cm': '우측 활보장 (R Stride)',
    'step_time_s':        'Step Time',
    'stride_time_s':      'Stride Time (보행주기)',
    'stance_ratio_pct':   '입각기 비율 (Stance)',
    'swing_ratio_pct':    '유각기 비율 (Swing)',
    'double_support_pct': '양하지 지지기 (Double Support)',
    'single_support_pct': '단하지 지지기 (Single Support)',
}

# ═══ 변수 그룹 (UI 표시용) ═══
VARIABLE_GROUPS = {
    'distance': {
        'label': '거리 변수',
        'variables': ['left_step_length_cm', 'right_step_length_cm', 'left_stride_length_cm', 'right_stride_length_cm'],
    },
    'time': {
        'label': '시간 변수',
        'variables': ['step_time_s', 'stride_time_s'],
    },
    'ratio': {
        'label': '비율 변수',
        'variables': ['stance_ratio_pct', 'swing_ratio_pct'],
    },
    'cadence': {
        'label': '보행 지표',
        'variables': ['cadence_spm'],
    },
}

# ═══ 속도 경고 임계값 ═══
VELOCITY_ALERTS = [
    {'threshold': 0.6, 'message': '0.6 m/s 미만: 신체 장애 고위험, 입원 예측인자 (Cesari 2005)'},
    {'threshold': 0.7, 'message': '0.7 m/s 이하: 낙상 위험 1.5배 증가 (Verghese 2009)'},
]

REFERENCE_NOTE = (
    "[참고사항]\n"
    "- 보행속도 판정은 Perry et al.(1995) 기능적 보행 분류에 근거합니다.\n"
    "- 보행속도를 제외한 변수의 경/중/심 판정은 현재 합의된 임상 근거가 없어 "
    "정상 범위 대비 편차만 제공합니다.\n"
    "- 정상 참고치는 한국인 성인 대상 연구(Kim et al. 2024; Kim & Lee 2020; "
    "Cho et al. 2004)와 국제 문헌을 종합한 대략적 범위입니다.\n"
    "- 측면 영상 기반 측정은 GAITRite 등 gold standard 대비 오차가 있을 수 있습니다.\n"
    "- 최종 임상 해석은 반드시 전문가(물리치료사, 재활의학과)가 수행해야 합니다."
)


def judge_velocity(speed_mps: float) -> Dict[str, Any]:
    """보행속도 Perry 분류 판정

    Returns:
        dict with classification, color, alerts
    """
    if speed_mps >= 1.0:
        classification = '정상'
        color = 'green'
    elif speed_mps >= 0.8:
        classification = '경도 저하'
        color = 'yellow'
    elif speed_mps >= 0.4:
        classification = '중등도 저하'
        color = 'orange'
    else:
        classification = '심각'
        color = 'red'

    # 속도 경고
    alerts = []
    for alert in VELOCITY_ALERTS:
        if speed_mps <= alert['threshold']:
            alerts.append(alert['message'])

    return {
        'speed_mps': round(speed_mps, 2),
        'classification': classification,
        'color': color,
        'alerts': alerts,
    }


def judge_variable(name: str, value: float) -> Dict[str, Any]:
    """개별 변수 판정

    Args:
        name: 변수 영문명 (NORMAL_RANGES 키)
        value: 측정값

    Returns:
        dict with status, deviation, direction, color, clinical_comment
    """
    if name not in NORMAL_RANGES:
        return {
            'variable_name': name,
            'display_name': DISPLAY_NAMES.get(name, name),
            'measured_value': round(value, 1),
            'unit': '',
            'normal_range': 'N/A',
            'status': 'N/A',
            'deviation': None,
            'direction': None,
            'clinical_comment': '',
            'color': 'gray',
        }

    ref = NORMAL_RANGES[name]
    unit = ref['unit']
    normal_range_str = f"{ref['min']}~{ref['max']}{unit}"

    if ref['min'] <= value <= ref['max']:
        status = '정상'
        deviation = None
        direction = None
        color = 'green'
        comment = ''
    elif value > ref['max']:
        status = '상한 초과'
        deviation = round(value - ref['max'], 1)
        direction = '↑'
        color = 'orange'
        comments = CLINICAL_COMMENTS.get(name, {})
        comment = comments.get('high', '')
    else:
        status = '하한 미달'
        deviation = round(ref['min'] - value, 1)
        direction = '↓'
        color = 'orange'
        comments = CLINICAL_COMMENTS.get(name, {})
        comment = comments.get('low', '')

    return {
        'variable_name': name,
        'display_name': DISPLAY_NAMES.get(name, name),
        'measured_value': round(value, 1),
        'unit': unit,
        'normal_range': normal_range_str,
        'status': status,
        'deviation': deviation,
        'direction': direction,
        'clinical_comment': comment,
        'color': color,
    }


def detect_patterns(judgments: List[Dict]) -> List[Dict[str, str]]:
    """변수 간 패턴 분석

    Args:
        judgments: judge_variable 결과 리스트

    Returns:
        감지된 패턴 코멘트 목록
    """
    # 판정 결과를 dict로 변환 (빠른 조회용)
    status_map = {}
    for j in judgments:
        name = j['variable_name']
        status_map[name] = {
            'status': j['status'],
            'direction': j['direction'],
        }

    patterns = []

    # 패턴 1: 안정성 보상 보행
    stance_high = (status_map.get('stance_ratio_pct', {}).get('direction') == '↑')
    swing_low = (status_map.get('swing_ratio_pct', {}).get('direction') == '↓')
    ds_high = (status_map.get('double_support_pct', {}).get('direction') == '↑')
    vel_low = (status_map.get('gait_velocity_ms', {}).get('direction') == '↓')

    if stance_high and swing_low and ds_high and vel_low:
        patterns.append({
            'pattern': 'stability_compensation',
            'label': '안정성 보상 보행 패턴',
            'comment': (
                '입각기 증가, 유각기 감소, 양하지 지지기 증가가 동시에 관찰되며, '
                '이는 균형 능력 저하에 대한 보상 전략일 수 있습니다.'
            ),
            'color': 'orange',
        })

    # 패턴 2: 짧은 보폭 패턴
    step_low = (status_map.get('left_step_length_cm', {}).get('direction') == '↓' or
                status_map.get('right_step_length_cm', {}).get('direction') == '↓')
    stride_low = (status_map.get('left_stride_length_cm', {}).get('direction') == '↓' or
                  status_map.get('right_stride_length_cm', {}).get('direction') == '↓')
    cadence_high = (status_map.get('cadence_spm', {}).get('direction') == '↑')

    if step_low and stride_low and cadence_high:
        patterns.append({
            'pattern': 'short_step_pattern',
            'label': '짧은 보폭 패턴',
            'comment': (
                '보폭과 활보장이 감소하면서 분속수가 증가하는 패턴으로, '
                '한국인 노인에서 흔히 관찰되는 안정성 확보 전략입니다 (Kim & Lee 2020).'
            ),
            'color': 'yellow',
        })

    return patterns


def judge_all(measured: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """전체 판정 수행

    Args:
        measured: 측정값 딕셔너리 (키: 변수 영문명, 값: 측정값 or None)

    Returns:
        dict with variables, patterns, velocity_alert, reference_note
    """
    judgments = []
    velocity_result = None

    for name in NORMAL_RANGES:
        value = measured.get(name)
        if value is None:
            # N/A 항목
            judgments.append({
                'variable_name': name,
                'display_name': DISPLAY_NAMES.get(name, name),
                'measured_value': None,
                'unit': NORMAL_RANGES[name]['unit'],
                'normal_range': f"{NORMAL_RANGES[name]['min']}~{NORMAL_RANGES[name]['max']}{NORMAL_RANGES[name]['unit']}",
                'status': 'N/A',
                'deviation': None,
                'direction': None,
                'clinical_comment': '',
                'color': 'gray',
            })
            continue

        j = judge_variable(name, value)
        judgments.append(j)

    # 보행속도 Perry 분류
    speed = measured.get('gait_velocity_ms')
    if speed is not None:
        velocity_result = judge_velocity(speed)

    # 패턴 분석
    patterns = detect_patterns(judgments)

    return {
        'variables': judgments,
        'patterns': patterns,
        'velocity': velocity_result,
        'reference_note': REFERENCE_NOTE,
    }
