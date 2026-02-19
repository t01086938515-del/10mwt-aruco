# Gait Analysis Variable Logic

> `processor.py` → `_analyze_gait_parameters()` 기준 (2026-02-10)

---

## 1. 입력 데이터

| 항목 | 설명 |
|------|------|
| **구간** | START~FINISH ArUco 마커 crossing 사이 (`start_t` ~ `finish_t`) |
| **ankle_history** | 프레임별 `{left_x, left_y, right_x, right_y, left_heel_x, left_heel_y, ...}` |
| **스무딩** | 5-point moving average (`np.convolve(kernel=ones(5)/5)`) |
| **FPS** | 원본 60fps, frame_skip=3 → 실효 ~20fps |

---

## 2. Step Detection (Heel Strike 감지)

### Primary: Y-peak (heel Y 로컬 최대값)
```
입력: heel_y_smooth (per-foot)
방법: scipy.signal.find_peaks()
prominence: 3단계 (0.06 → 0.03 → 0.015) × y_range, 최소 4피크
vx 필터: 3단계 (×2.5 → ×4.0 → ×6.0), HS 시점에 |vx| 큰 피크 제거
min_gap: 같은 발 0.25s
```
- **장점**: timing 정밀도 우수 (Y축 피크 = 실제 착지 시점)
- **한계**: bilateral crosstalk (양발 Y피크 동시 발생 → dedup에서 50% 소실)

### Fallback: Vx-crossing (ankle X velocity 하향 교차)
```
조건: Y-peak cadence < 70 spm 일 때만 발동
입력: heel_x_smooth → vx = gradient() × walk_direction
threshold: 양발 합산 positive vx의 median × [0.35, 0.25, 0.15] (3단계)
정밀화: 하향 교차 후 vx=0 지점 탐색 + sub-frame 선형 보간
min_gap: 0.3s
```
- **장점**: L/R 독립 (crosstalk 없음) → 정확한 counting
- **한계**: timing 정밀도 낮음 (감속 프로필 L/R 비대칭)

### Last Resort: X-crossing (좌우 발 X좌표 교차)
```
조건: 위 두 방법 모두 step_events < 4개일 때
방법: sign(left_x - right_x) 변화 시점
```

### Bilateral Dedup (양발 병합 후 중복 제거)
```
L/R 이벤트 시간순 병합 후:
- dt < 0.15s: 무조건 병합 (더 높은 heel_y 유지)
- dt < 0.25s + 같은 발: 병합
- boundary filter: start_t + 0.3s 이후만
- tail filter: 마지막 interval이 median × 2.5 초과 시 제거
```

---

## 3. Leading Foot 라벨링

```
방법: 공간 위치 기반
  walk_sign = +1(X증가) or -1(X감소)
  각 HS 시점에서 left_x × walk_sign > right_x × walk_sign → leading = 'left'
후처리: 교대 강제 (연속 같은 발이면 두 번째를 반대로 flip)
```
- **한계**: 측면 카메라에서 두 발의 X 분리도가 낮으면 오판 가능

---

## 4. 기본 지표

| 변수 | 로직 | 비고 |
|------|------|------|
| **step_count** | `len(step_events)` | 감지된 전체 HS 수 |
| **speed_mps** | `10m / elapsed_time_s` | ArUco 마커 간 거리 ÷ 소요 시간 |
| **cadence_spm** | `60 / median(step_intervals)` | step interval 0.3~2.0s 범위만 사용 |
| **step_length_m** | `10m / step_count` | 거리 기반 평균 (전체) |
| **stride_length_m** | `10m / (step_count // 2)` | 거리 기반 평균 (전체) |

---

## 5. 좌우 거리 지표 (L/R + SI)

### Step Length L/R
```
방법: 연속 step event 간 중점(midpoint) X거리
  curr_mid_x = (left_x + right_x) / 2  ← 양발 중점 사용
  step_m = |curr_mid_x - prev_mid_x| / ppm
라벨: curr.leading_foot == 'left' → left_step_dist에 추가
필터: 0.2m ≤ step_m ≤ 2.0m
이상치: L+R 합쳐서 median 기준 ×0.5~×2.0 범위 외 제거
결과: mean(left_steps), mean(right_steps)
```
- **의존**: leading_foot 라벨 정확도에 직접 의존
- **문제점**: 라벨이 틀리면 긴 step이 한쪽에 몰림 → SI 급등

### Stride Length L/R
```
방법: event[i] → event[i+2] 간 중점 X거리 (라벨 독립)
  i % 2 == 0 → left_strides, i % 2 == 1 → right_strides
필터: 0.3m ≤ stride_m ≤ 4.0m
이상치: median 기준 ×0.5~×2.0 범위 외 제거
결과: mean(left_strides), mean(right_strides)
```
- **라벨 독립**: foot 라벨에 의존하지 않음 (인덱스 홀짝으로 분리)
- **문제점**: 샘플 수 = step 수의 절반 → 이상치 1개의 영향이 큼

### Homography 보정
```
조건: perspective_corrector.calibrated == True 일 때
방법: real_distance_x(x1, y1, x2, y2) → 원근 보정된 실제 거리
미보정 시: dx / ppm (pixels_per_meter)
```

---

## 6. 좌우 시간 지표 (L/R + SI)

### Step Time L/R
```
방법: 연속 step event 간 시간 차 (다른 발 사이만)
  dt = curr.time - prev.time (curr.foot != prev.foot)
라벨: curr.leading_foot 기준으로 L/R 분류
필터: 0.2s ≤ dt ≤ 2.0s
결과: mean(left_times), mean(right_times)
```
- **의존**: HS 타이밍 정밀도 + leading_foot 라벨

### Stride Time L/R
```
방법: event[i] → event[i+2] 간 시간 차 (라벨 독립)
  i % 2 == 0 → left, i % 2 == 1 → right
필터: 0.4s ≤ dt ≤ 4.0s
결과: mean(left_stride_times), mean(right_stride_times)
```
- **라벨 독립**: stride length와 동일하게 인덱스 기반

---

## 7. Swing / Stance 분석

```
방법: ankle X-velocity 기반
  vx = gradient(ankle_x_smooth, times)
  walk_dir = sign(mean(left_vx) + mean(right_vx))
  threshold = median(|all_vx|) × 1.2
  swing = (vx × walk_dir) > threshold
  stance = 나머지
```

| 변수 | 계산식 |
|------|--------|
| **swing_pct** | `swing_frames / total_frames × 100` |
| **stance_pct** | `stance_frames / total_frames × 100` |
| **swing_time_s** | `(swing_frames × frame_time) / (step_count // 2)` |
| **stance_time_s** | `(stance_frames × frame_time) / (step_count // 2)` |
| **swing_stance_ratio** | `swing_frames / stance_frames` |

- **정상 범위**: Swing ~40%, Stance ~60%
- **한계**: 프레임 단위 이진 분류 → 해상도 제약 (1 frame ≈ 50ms at 20fps)

---

## 8. SI (Symmetry Index) 계산

```
공식: SI = |L - R| / (0.5 × (L + R)) × 100 (%)
적용 대상: step_length, step_time, stride_length, stride_time, swing_time, stance_time, swing_pct
```

### Overall Symmetry Index
```
= mean(모든 개별 SI 값)
포함: step_length_si, step_time_si, stride_time_si, swing_time_si, stance_time_si, swing_stance_si
미포함: stride_length_si (별도 계산이지만 si_values에 미추가)
```

---

## 9. 변수별 주요 의존성 & 취약점

| 변수 | 핵심 의존 | 취약점 |
|------|----------|--------|
| step_count | HS 감지 | Y-peak crosstalk → 과소, X-crossing → 과다 |
| cadence | step interval median | HS 누락 시 interval 2배 → cadence 절반 |
| step_length L/R | leading_foot 라벨 | 라벨 오판 → 한쪽에 긴 step 몰림 |
| step_time L/R | HS timing + 라벨 | vx-crossing의 L/R 감속 비대칭 |
| stride_length L/R | HS 위치 (라벨 무관) | 샘플 수 적음 → 이상치 민감 |
| stride_time L/R | HS timing (라벨 무관) | 동일 |
| swing/stance | vx threshold | 전역 threshold → 개인차 미반영 |
| overall_SI | 모든 SI 평균 | 한 지표의 극단값이 전체를 끌어올림 |
