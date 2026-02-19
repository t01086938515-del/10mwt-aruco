# 섹션 6: 좌우 시간 지표 (Step Time L/R, Stride Time L/R) - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` → `_analyze_gait_parameters()`

---

## 6-1. Step Time L/R (line 850-862)

### 원리
연속 step event 간 시간 차이, leading_foot 라벨로 L/R 분류

### 코드 (line 850-862)
```python
for i in range(1, len(step_events)):
    curr = step_events[i]
    prev = step_events[i - 1]
    if curr['leading_foot'] != prev['leading_foot']:  # 다른 발 사이만
        dt = curr['time'] - prev['time']
        if 0.2 <= dt <= 2.0:
            if curr['leading_foot'] == 'left':
                left_step_times.append(dt)
            else:
                right_step_times.append(dt)
```

### 핵심 특성
- **라벨 의존**: curr.leading_foot 기준으로 L/R 분류 → Step Length L/R과 동일 구조
- **다른 발 사이만** 계산 (같은 발 연속 = stride time)
- **필터**: 0.2~2.0초
- **HS 감지 의존**: HS 타이밍이 틀리면 시간 간격도 틀림

#### Q&A

**Q: Step Time L/R 쉽게 설명**

- 왼발 착지 → 오른발 착지 사이 시간 = 한 스텝의 시간
- 예시: 왼발(0.0s) → 오른발(0.5s) = 0.5초 → right step time (도착한 발 기준)
- Step Length L/R과 완전히 같은 구조, 거리 대신 시간을 재는 것뿐
- 근본 한계도 동일: HS 감지 정확도에 의존

---

## 6-2. Stride Time L/R (line 874-893)

### 원리
event[i] → event[i+2] 간 시간 차이, 인덱스 홀짝으로 L/R 분류

### 코드 (line 874-893)
```python
for i in range(len(step_events) - 2):
    dt = step_events[i + 2]['time'] - step_events[i]['time']
    if 0.4 <= dt <= 4.0:
        if i % 2 == 0:
            left_stride_times.append(dt)
        else:
            right_stride_times.append(dt)
```

### 핵심 특성
- **라벨 독립**: Stride Length L/R과 동일 — 인덱스 홀짝 기반
- **필터**: 0.4~4.0초
- **HS 감지 의존**: HS 누락/추가 시 인덱스가 밀림 → L/R 뒤집힘

---

## 6-3. 근본 한계

| 지표 | 방식 | HS 부정확 시 |
|------|------|-------------|
| Step Time L/R | 라벨 기반 | 라벨 오판 → L/R 뒤집힘 |
| Stride Time L/R | 인덱스 기반 | HS 누락 → 인덱스 밀림 → L/R 뒤집힘 |

**근본 해결 = A (Foot Separation), B (MediaPipe), E (카메라 가이드)**

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| Step Time 라벨 기반 | ✅ | line 855-860 | ✅ |
| 다른 발 사이만 | ✅ | line 854 | ✅ |
| 필터 0.2~2.0s | ✅ | line 857 | ✅ |
| Stride Time 인덱스 기반 | ✅ | line 885-888 | ✅ |
| 필터 0.4~4.0s | ✅ | line 884 | ✅ |

**결론: 문서와 코드 일치. 섹션 5(거리)와 구조 동일 — 거리→시간만 바뀜.**
