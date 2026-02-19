# 섹션 5: 좌우 거리 지표 (L/R + SI) - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` → `_analyze_gait_parameters()`

---

## 5-1. Step Length L/R (line 804-848)

### 원리
연속 step event 간 양발 중점(midpoint) X거리

### 코드 (line 809-827)
```python
for i in range(1, len(step_events)):
    curr = step_events[i]
    prev = step_events[i - 1]
    if curr['leading_foot'] != prev['leading_foot']:  # 다른 발 사이만
        curr_mid_x = (curr['left_x'] + curr['right_x']) / 2
        prev_mid_x = (prev['left_x'] + prev['right_x']) / 2
        # Homography 보정 적용
        if self.perspective_corrector.calibrated:
            step_m = self.perspective_corrector.real_distance_x(...)
        else:
            step_m = dx / ppm
        if 0.2 <= step_m <= 2.0:
            if curr['leading_foot'] == 'left':
                left_step_dists.append(step_m)
            else:
                right_step_dists.append(step_m)
```

### 이상치 제거 (line 831-836)
```python
all_step_dists = left_step_dists + right_step_dists
step_med = np.median(all_step_dists)
left_step_dists = [v for v in left_step_dists if 0.5 * step_med <= v <= 2.0 * step_med]
right_step_dists = [v for v in right_step_dists if 0.5 * step_med <= v <= 2.0 * step_med]
```

### 핵심 특성
- **라벨 의존**: curr.leading_foot 기준으로 L/R 분류 → 라벨 오판 시 긴 step이 한쪽에 몰림 → SI 급등
- **중점(midpoint) 사용**: 양발 X 평균으로 위치 추정
- **Homography 적용**: ✅

#### Q&A

**Q: Step Length L/R 코드 설명**

- 연속 착지 사이 양발 중점(midpoint) X거리 측정
- 중점 사용 이유: 측면 카메라에서 한쪽 발 X좌표를 믿기 어려워 양발 평균으로 "몸의 위치" 추정
- 다른 발 사이만 계산 (같은 발 연속 = stride이지 step이 아님)
- L/R 분류: 도착한 발의 leading_foot 라벨 기준

**Q: 이상치 제거 방식**

- L/R 합쳐서 median 구한 후 ×0.5~×2.0 범위 외 제거
- 한쪽에만 이상치 있어도 전체 기준으로 판단

**Q: Step Length L/R의 근본 해결책은?**

- 라벨 의존 → 인덱스 기반으로 바꿔도 HS 누락 시 인덱스 뒤집혀서 같은 문제
- 측정 방법을 바꿔도 HS 감지가 부정확하면 해결 안 됨
- 근본 해결 = HS 감지 정확도 향상 (A. Foot Separation, B. MediaPipe, E. 카메라)이 유일한 해법
- 모든 하류 지표(Step/Stride Length/Time L/R, Swing/Stance, SI)가 전부 HS 감지에 의존

---

## 5-2. Stride Length L/R (line 712-769)

### 원리
event[i] → event[i+2] 간 중점 X거리 (라벨 독립)

### 코드 (line 722-740)
```python
for i in range(len(step_events) - 2):
    curr = step_events[i]
    next2 = step_events[i + 2]
    x1 = (curr['left_x'] + curr['right_x']) / 2
    x2 = (next2['left_x'] + next2['right_x']) / 2
    # Homography 보정 적용
    if self.perspective_corrector.calibrated:
        stride_m = self.perspective_corrector.real_distance_x(x1, y1, x2, y2)
    else:
        stride_m = dx / ppm
    if 0.3 <= stride_m <= 4.0:
        if i % 2 == 0:
            left_strides.append(stride_m)
        else:
            right_strides.append(stride_m)
```

### 핵심 특성
- **라벨 독립**: 인덱스 홀짝으로 L/R 분리 (foot 라벨에 의존하지 않음)
- **이상치 제거**: median 기준 ×0.5~×2.0
- **Homography 적용**: ✅
- **샘플 수**: step 수의 절반 → 이상치 1개의 영향이 큼

---

## 5-3. Homography 보정

```
조건: perspective_corrector.calibrated == True
방법: real_distance_x(x1, y1, x2, y2) → 원근 보정된 실제 거리
미보정 시: dx / ppm (pixels_per_meter)
```

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| Step Length 중점 사용 | ✅ | line 813-814 | ✅ |
| 다른 발 사이만 | ✅ | line 812 | ✅ |
| leading_foot 라벨 기반 | ✅ | line 824-827 | ✅ |
| 필터 0.2~2.0m | ✅ | line 823 | ✅ |
| 이상치 median ×0.5~×2.0 | ✅ | line 834-836 | ✅ |
| Stride 라벨 독립 (인덱스 기반) | ✅ | line 737-740 | ✅ |
| Stride 필터 0.3~4.0m | ✅ | line 736 | ✅ |
| Homography 적용 | ✅ | line 818-820, 731-732 | ✅ |

**결론: 문서와 코드 일치**

