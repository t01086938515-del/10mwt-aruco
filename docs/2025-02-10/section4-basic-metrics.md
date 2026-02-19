# 섹션 4: 기본 지표 - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` → `_analyze_gait_parameters()` + `_compute_results()`

---

## 문서 내용

| 변수 | 로직 | 비고 |
|------|------|------|
| **step_count** | `len(step_events)` | 감지된 전체 HS 수 |
| **speed_mps** | `10m / elapsed_time_s` | ArUco 마커 간 거리 / 소요 시간 |
| **cadence_spm** | `60 / median(step_intervals)` | step interval 0.3~2.0s 범위만 사용 |
| **step_length_m** | `10m / step_count` | 거리 기반 평균 (전체) |
| **stride_length_m** | `10m / (step_count // 2)` | 거리 기반 평균 (전체) |

---

## 코드 대조

### step_count (processor.py line 675)
```python
step_count = len(step_events)
result['step_count'] = step_count
```

### speed_mps (processor.py line 1005-1006, _compute_results 내부)
```python
distance_m = self.aruco.marker_distance_m  # 10m
speed_mps = distance_m / self.timer_elapsed_s
```
**주의**: `_analyze_gait_parameters()` 밖에서 계산됨.
- `self.timer_elapsed_s`: 타이머 기반 경과 시간
- `_analyze_gait_parameters()`의 `elapsed` (line 377: `finish_t - start_t`): crossing event 기반 시간
- 두 시간은 미세하게 다를 수 있음

#### Q&A

**Q: speed_mps 로직 상세 설명**

- 속도 = 10m / 소요시간
- 시간이 두 개 존재: timer_elapsed_s (타이머, 매 프레임 갱신) vs elapsed (crossing event 기반)
- speed는 timer, cadence는 elapsed 사용 → 서로 다른 시간 기준
- 차이 0.05s 이내지만 통일하는 게 맞음 → 개선방안 J

**Q: 시간 통일하면 문제없나?**

- 문제없음. 차이 0.05s 이내, 어느 쪽으로 통일해도 결과 거의 동일
- crossing event 쪽이 나음 (나머지 지표와 같은 기준)

### cadence_spm (processor.py line 688-699)
```python
# 기본 계산 (fallback)
cadence = (step_count / elapsed) * 60

# median 기반 보강 (primary)
_step_intervals = []
for i in range(1, len(step_events)):
    dt = step_events[i]['time'] - step_events[i-1]['time']
    if 0.3 <= dt <= 2.0:
        _step_intervals.append(dt)
if _step_intervals:
    cadence = 60.0 / np.median(_step_intervals)

# 유효 범위 필터
if 30 <= cadence <= 200:
    result['cadence_spm'] = round(cadence, 1)
```
**이중 계산**: 먼저 `step_count/elapsed` 기반으로 계산 후, step interval median이 존재하면 덮어쓰기

#### Q&A

**Q: cadence 이중 계산 설명**

- fallback: step_count / elapsed × 60 (단순 평균, HS 누락 시 과소)
- primary: 60 / median(step_intervals) (간격 기반, HS 누락에도 강건)
- 0.3~2.0s 필터: 미만은 비현실적, 초과는 HS 누락으로 인한 긴 간격
- 걸음 수가 틀려도 간격이 맞으면 cadence를 구할 수 있으니 median 우선

### step_length_m (processor.py line 702-704)
```python
step_length = distance_m / step_count if step_count > 0 else None
if step_length and 0.2 <= step_length <= 1.5:
    result['step_length_m'] = round(step_length, 3)
```

### stride_length_m (processor.py line 707-710)
```python
stride_count = step_count // 2
stride_length = distance_m / stride_count if stride_count > 0 else None
if stride_length and 0.4 <= stride_length <= 2.5:
    result['stride_length_m'] = round(stride_length, 3)
```

#### Q&A

**Q: step_length, stride_length 설명**

- step_length = 10m / step_count (한 걸음 평균)
- stride_length = 10m / (step_count // 2) (한 보폭 = 같은 발→같은 발)
- stride = step × 2와 산술적으로 동일, 별도 측정 아님
- 실제 개별 측정은 섹션 5에서 수행 → 개선방안 I (실측 기반으로 교체)
- 유효 범위: step 0.2~1.5m, stride 0.4~2.5m

**Q: stride가 그냥 step×2인가?**

- 맞음. 섹션 4에서는 산술적으로 ×2일 뿐
- 섹션 4 = 요약용 숫자, 섹션 5 = 실제 L/R 개별 측정
- 질적으로 떨어지므로 섹션 5 수정 후 섹션 4도 교체 예정

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| step_count | `len(step_events)` | line 675 | ✅ |
| speed_mps | `10m / elapsed_time_s` | `distance_m / timer_elapsed_s` (line 1006) | ⚠️ |
| cadence_spm | `60 / median(intervals)` | median 기반 + fallback (line 688-699) | ⚠️ |
| step_length_m | `10m / step_count` | line 702 | ✅ |
| stride_length_m | `10m / (step_count // 2)` | line 707-708 | ✅ |

### 문서 보완 필요 사항

1. **speed_mps 계산 위치**: 문서에서 `_analyze_gait_parameters()` 소속으로 표기했지만, 실제로는 `_compute_results()`에서 계산. 사용하는 시간도 `timer_elapsed_s`이며 crossing event 기반 `elapsed`와 다름.
2. **cadence fallback**: 문서에는 median 방식만 기재. 코드에는 `step_count/elapsed × 60` fallback 존재.
3. **유효 범위 필터**: 문서에 미기재
   - step_length: 0.2~1.5m
   - stride_length: 0.4~2.5m
   - cadence: 30~200 spm

**결론: 핵심 로직 일치, 세부 사항 보완 필요**
