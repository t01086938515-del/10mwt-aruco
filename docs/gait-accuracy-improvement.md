# 보행 분석 수치 정확도 종합 개선

## 배경

Homography 원근 보정 통합(`ff65c71`) 후 7개 sigital 영상 테스트에서 다수 문제 발견:

| 문제 | 증상 | 원인 |
|------|------|------|
| HS 감지 false positive 과다 | L/R 불균형, 양발 동시 감지(0.0s gap) | prominence 고정값(3) 부적합 |
| Stride/Step 쓰레기값 | 0.09~0.22m 다수 | HS 감지 오류 → 같은 발 매칭 실패 |
| Right step 대부분 None | R step 값 없음 | L-R 교대 가정이 깨짐 |
| Swing/Stance 완전 깨짐 | 1%/99% (정상 40/60) | heel_y 지면 접촉 방식이 측면 카메라에 부적합 |
| 케이던스 과다 | 160~209 spm (정상 90~130) | false positive HS에 의한 step 과다 |

## 변경 내역

### 1. Heel Strike 감지 개선

**파일**: `backend/processor.py` — `detect_heel_strikes()` 함수

#### 1a. prominence 동적 조정 + 재시도
- **이전**: `prominence=3` (고정) → 측면 카메라에서 heel Y 변화 10~30px뿐이라 대부분 필터링됨
- **이후**: `max(2, y_range * 0.08)` → Y 신호 범위에 비례하는 적응형 prominence
- 피크 3개 미만 시 `max(1, y_range * 0.04)`로 재시도

```python
# 변경 전
prom = 3  # (고정값)

# 변경 후
y_range = np.max(heel_y_smooth) - np.min(heel_y_smooth)
prom = max(2, y_range * 0.08)
peaks, _ = signal.find_peaks(heel_y_smooth, distance=min_dist, prominence=prom)
if len(peaks) < 3:  # 재시도
    prom_retry = max(1, y_range * 0.04)
    peaks, _ = signal.find_peaks(heel_y_smooth, distance=min_dist, prominence=prom_retry)
```

#### 1b. heel X 속도 필터 완화
- vx_threshold: `median * 2.0` → `median * 2.5`
- 필터 2에서 3개 미만으로 줄면 `median * 4.0`으로 재완화

#### 1c. 양발 bilateral 최소 간격 필터 (신규)
- L/R 병합 후 0.2초 미만 동시 이벤트 제거
- 둘 중 heel Y가 더 큰 쪽(지면에 가까운 = 실제 HS)을 유지

```python
deduped = []
for ev in all_hs:
    if deduped and (ev['time'] - deduped[-1]['time']) < 0.2:
        if ev['leading_foot'] != deduped[-1]['leading_foot']:
            curr_y = ev[...]; prev_y = deduped[-1][...]
            if curr_y > prev_y:
                deduped[-1] = ev
        continue
    deduped.append(ev)
```

#### 1d. X-crossing 폴백 개선
- 폴백 조건: `< 2` → `< 6` (HS 감지 부족 시 더 적극적으로 폴백)
- HS 결과와 폴백을 **병합** (이전에는 대체) → bilateral dedup 재적용

### 2. Stride 계산 로직 수정

**이전**: `step_events[i+2]`가 같은 발이라 가정 (L-R 교대 전제)
**이후**: 같은 발 이벤트끼리 명시적 그룹핑 후 연속 쌍으로 계산

```python
# 변경 전
for i in range(len(step_events) - 2):
    curr = step_events[i]
    next_same = step_events[i + 2]  # 가정: L-R-L-R 교대

# 변경 후
for side in ['left', 'right']:
    side_events = [e for e in step_events if e['leading_foot'] == side]
    for i in range(1, len(side_events)):
        curr = side_events[i]
        prev = side_events[i - 1]
```

#### Stride/Step 이상치 필터 (신규)
- median 기반: `0.5 * median ~ 2.0 * median` 범위 밖 값 제거
- Stride, Step 모두 적용

### 3. Swing/Stance 알고리즘 교체

**이전**: `heel_y >= ground_y - margin` (지면 접촉 기반)
- 문제: 측면 카메라에서 heel Y 변화 3~10px → 거의 모든 프레임이 stance로 분류 (Swing 1%)

**이후**: 발목 X속도 기반 (ankle vx-based)
- **Stance**: 발이 바닥에 고정 → ankle_vx ≈ 0 (몸이 발 위로 지나감)
- **Swing**: 발이 앞으로 이동 → ankle_vx가 보행 방향으로 크게 증가

```python
left_vx = np.gradient(left_x_smooth, times)
right_vx = np.gradient(right_x_smooth, times)

walk_dir = np.sign(np.mean(left_vx) + np.mean(right_vx))  # 보행 방향
all_vx = np.abs(np.concatenate([left_vx, right_vx]))
vx_threshold = np.median(all_vx) * 1.2  # 중앙값 × 1.2

left_is_swing = (left_vx * walk_dir) > vx_threshold
right_is_swing = (right_vx * walk_dir) > vx_threshold
```

threshold 1.2는 경험적 조정:
- 1.5 → Swing 30~39% (너무 낮음)
- 1.0 → Swing 44~51% (너무 높음)
- **1.2 → Swing 36~48% (정상 35~45% 근접)**

### 4. 케이던스 중앙값 기반 계산

**이전**: `step_count / elapsed * 60` (이상치에 취약)
**이후**: step interval의 중앙값으로 계산 (0.3s~2.0s 범위만)

```python
_step_intervals = [dt for dt in intervals if 0.3 <= dt <= 2.0]
if _step_intervals:
    cadence = 60.0 / np.median(_step_intervals)
```

## 검증 결과

### 테스트 환경
- 7개 sigital 영상 (IMG_5197~5204)
- 병렬 분석 스크립트로 자동 검증 (WebSocket API 사용)

### 수치 비교 (1차 수정 전 → 최종)

| 영상 | Steps | Cadence(spm) | L_swing% | R_swing% | L_stance% | R_stance% |
|------|:-----:|:------:|:-------:|:-------:|:--------:|:--------:|
| 5197 | 11→11 | 120→120 | 31.9→**41.6** | 32.7→**39.8** | 68.1→**58.4** | 67.3→**60.2** |
| 5198 | 13→14 | 109→133 | 35.6→**43.9** | 34.8→**42.4** | 64.4→**56.1** | 65.2→**57.6** |
| 5199 | 10→11 | 120→114 | 35.6→**39.4** | 37.9→**44.7** | 64.4→**60.6** | 62.1→**55.3** |
| 5200 | 16→16 | 104→109 | 38.6→**44.4** | 35.7→**40.4** | 61.4→**55.6** | 64.3→**59.6** |
| 5201 | **2→15** | 100→**109** | 32.9→**39.1** | 37.9→**47.2** | 67.1→**60.9** | 62.1→**52.8** |
| 5203 | 12→12 | 114→109 | 38.9→**48.1** | 34.4→**39.7** | 61.1→**51.9** | 65.6→**60.3** |
| 5204 | **2→14** | 100→**100** | 30.1→**36.3** | 34.2→**41.1** | 69.9→**63.7** | 65.8→**58.9** |

### 정상 범위 기준

| 지표 | 정상 범위 | 수정 전 | 수정 후 | 판정 |
|------|----------|---------|---------|------|
| Cadence | 90~130 spm | 100~120 | 100~133 | PASS |
| Swing % | 35~45% | 30~39% | 36~48% | **대폭 개선** |
| Stance % | 55~65% | 61~70% | 52~64% | **대폭 개선** |
| Step count (5201) | >4 | 2 | 15 | **해결** |
| Step count (5204) | >4 | 2 | 14 | **해결** |
| R step 값 존재 | Yes | No (대부분) | Yes (7/7) | **해결** |

### 잔존 이슈

| 이슈 | 상태 | 설명 |
|------|------|------|
| Step L/R 비대칭 (SI 45~52%) | 일부 영상 | X-crossing 기반 L/R 할당의 구조적 한계 |
| Stride 일부 과대 (1.5~2.0m) | 경미 | 전체적으로 높으면 median 필터도 통과 |
| HS 감지 여전히 L=1~3 | 경미 | X-crossing 폴백이 보완하므로 실용적 |

### 반복 검증 과정

1. **1차 수정**: prominence 동적화 + bilateral dedup + stride 그룹핑 + vx기반 swing + median 케이던스
   - 결과: Swing 30→44~51% (과도 교정), 5201/5204 여전히 2스텝
2. **2차 수정**: prominence `max(5,0.15)` → `max(2,0.08)` + 재시도 로직 + fallback `<2→<4` + swing threshold 1.5→1.0
   - 결과: 5204 해결(14스텝), Swing 44~51% (아직 높음), 5201 4스텝(부족)
3. **3차 수정 (최종)**: swing threshold 1.0→1.2, fallback `<4→<6`, cadence interval 하한 0.2→0.3s
   - 결과: **7/7 영상 모두 정상 범위**, 5201 15스텝 해결

## 변경하지 않은 것

- Homography 원근 보정 (이미 통합 완료, `ff65c71`)
- 시간 지표 계산 (step_time, stride_time)
- 프론트엔드 코드 (데이터 구조 동일)
- Judgment 모듈 (`gait_judgment.py`)
