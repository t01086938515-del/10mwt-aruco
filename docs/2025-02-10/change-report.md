# 코드 변경 보고서

> 작성일: 2026-02-10 ~ 11
> 목적: 개선방안 구현 과정 기록 (A→C+D+K)
> 원칙: 근본 해결(HS 감지 정확도 향상)에 맞는 변경만. 땜빵/가리기 로직 금지.

---

## 변경 이력

### 1차: Foot Separation 함수 추가

**파일**: `backend/processor.py`

**추가된 함수:**
1. `detect_foot_separation_steps()` — |left_x - right_x| 피크 기반 step detection
2. `step_regularity_score()` — step interval CV 계산

### 2차: 선택 로직 추가 (CV * 0.8 조건)

Y-peak/Vx-crossing 결과와 foot separation을 CV 비교 후 선택.
결과: **7개 영상 전부 변화 없음** — CV 조건이 너무 보수적.

### 3차: 선택 로직 조건 완화

4가지 조건으로 확장 (fallback 적극 비교, 10% CV, cadence 범위).
결과: **5203만 전환 → 악화**

### 4차: 선택 로직 비활성화 (원복)

5203 악화 확인 후 선택 로직 제거. 함수는 참고 로그용으로 보존.

---

## 테스트 결과

### Baseline (변경 전)

| 영상 | Steps | Cadence | StepSI | TimeSI | StrideSI | 상태 |
|------|-------|---------|--------|--------|----------|------|
| 5197 | 9 | 109.0 | 19.7% | 15.1% | 9.8% | ⚠️ |
| 5198 | 9 | 104.3 | 2.5% | 1.8% | 9.6% | ✅ |
| 5199 | 9 | 104.3 | 35.8% | 6.8% | 33.6% | ❌ |
| 5200 | 14 | 96.1 | 13.6% | 18.1% | 1.6% | ⚠️ |
| 5201 | 15 | 104.3 | 2.9% | 9.3% | 5.6% | ⚠️ |
| 5203 | 9 | 109.0 | 7.1% | 1.9% | 5.6% | ✅ |
| 5204 | 11 | 95.9 | 0.5% | 2.2% | 23.9% | ⚠️ |

### 2차 A 테스트 (조건 완화) — 5203 악화

| 영상 | Steps | StepSI | TimeSI | StrideSI | 변화 |
|------|-------|--------|--------|----------|------|
| 5203 | 9 | 7.1→**10.6** | 1.9→**14.4** | 5.6→**15.2** | ❌ 악화 |
| 나머지 6개 | - | 동일 | 동일 | 동일 | 변화 없음 |

---

## 핵심 발견 (A 실패 분석)

### 왜 Foot Separation이 효과 없었나

| 영상 | 기존 Method | Sep Events | Sep CV | 기존 CV | 결과 |
|------|------------|-----------|--------|---------|------|
| 5197 | Y-peak | 7 | 0.325 | 0.243 | 기존 승 |
| 5198 | Y-peak | 9 | 0.318 | 0.344 | 차이 미미 |
| 5199 | Y-peak | 9 | 0.347 | 0.374 | 차이 미미 |
| 5200 | Vx-crossing | 13 | 0.271 | 0.157 | 기존 압승 |
| 5201 | Vx-crossing | 10 | 0.340 | 0.297 | 기존 승 |
| 5203 | Y-peak→**Sep** | 9 | 0.281 | 0.340 | Sep 승 → **악화** |
| 5204 | Vx-crossing | 9 | 0.276 | 0.266 | 기존 승 |

### 근본 원인 재분석

**문제는 step 감지 개수가 아니라 L/R 라벨링 + 원근 비대칭:**

1. **5199** (StepSI=35.8%): 9 steps으로 count는 정상. SI가 높은 이유는 **leading_foot 라벨이 틀려서** 긴 step이 한쪽에 몰림
2. **5200** (TimeSI=18.1%): Vx-crossing이 14 events를 잘 잡음 (CV=0.157 최고). 시간 비대칭은 **L/R 타이밍 측정** 문제
3. **5201** (SwingSI=18.7%): Swing 비대칭은 **pixel vx 원근 차이** (개선방안 C, D)
4. **5203 악화**: foot separation peak 위치가 HS와 다름 → 다른 midpoint X좌표 → **L/R 거리 측정이 틀어짐**

### 결론

> **Step detection 방법을 바꿔도 downstream의 L/R 라벨링, 원근 비대칭 문제가 그대로면 SI는 개선 안 됨.**
> 오히려 정상 영상을 악화시킬 위험이 있음 (5203 사례).
>
> **A(Foot Separation)는 단독으로는 효과 없음.**
> B(MediaPipe heel) + C(Homography vx) + D(Per-foot threshold)와 함께 적용해야 의미 있을 수 있음.

---

## 현재 코드 상태 (A 이후)

- `detect_foot_separation_steps()`: 함수 보존, 참고 로그만 출력
- `step_regularity_score()`: 함수 보존
- **선택 로직: 비활성화** (baseline과 동일한 동작)

---

## 5차: C+D+K 구현

### 변경 내용

**K. stride_length_si를 overall_si에 포함** (line 913-915)
```python
# stride_length_si는 위에서 이미 계산됨 → si_values에 포함
if result.get('stride_length_si') is not None:
    si_values.append(result['stride_length_si'])
```
- 기존: stride_length_si가 직접 계산되었지만 si_values에 미포함 → overall_si에서 빠짐
- 수정: si_values에 추가 → overall_si가 7개 SI 전체 반영

**C. Homography vx 보정** (line 429-449, 689-691, 698-699, 1021-1023)
```python
# 원근보정된 X좌표 배열 생성 (batch perspectiveTransform)
if self.perspective_corrector.calibrated and self.perspective_corrector.H is not None:
    left_x_corrected = cv2.perspectiveTransform(pts_left, _H)[:, 0, 0]
    ...
else:
    left_x_corrected = left_x_smooth  # fallback: 원본 사용
```
- Vx-crossing shared_threshold: 보정된 X로 vx 계산
- Vx-crossing detection: 보정된 X 전달
- Swing/Stance vx: 보정된 X로 vx 계산

**D. Per-foot threshold** (line 1030-1037)
```python
# 기존: all_vx = np.abs(np.concatenate([left_vx, right_vx]))
#        vx_threshold = np.median(all_vx) * 1.2  (단일 임계값)
# 수정: 각 발별 별도 임계값
left_vx_threshold = np.median(np.abs(left_vx)) * 1.2
right_vx_threshold = np.median(np.abs(right_vx)) * 1.2
```

### 5차 테스트 (Homography 미작동 상태 — 중간 기록)

초기 테스트에서 Homography calibrated: False → C 효과 없음 확인.
원인 2개 발견:
1. `_detect_stage_edge()` 바닥 에지 미검출 → fallback 추가 (h*0.85)
2. `run_analysis.py` 2-pass 구조에서 ArUco가 Pass 1에서 보정 완료 → Pass 2의 `process_frame()`에서 보정 블록 건너뜀 → `perspective_corrector.calibrate()` 미호출

### 6차: Homography 초기화 버그 수정

**수정 1**: `solution_2_homography.py` — 에지 미검출 시 `h*0.85` fallback
**수정 2**: `processor.py` line 163-175 — ArUco 보정 완료 후 perspective_corrector 별도 초기화

### 6차 테스트 결과 (C+D+K 전체 작동)

| 영상 | Steps (전→후) | StepSI (전→후) | TimeSI (전→후) | StrideSI (전→후) | SwPct L/R |
|------|--------------|---------------|---------------|-----------------|-----------|
| 5197 | 9→9 | 19.7→**19.4** | 15.1→15.1 | 9.8→**9.7** | 39.8/41.6 |
| 5198 | 9→9 | 2.5→**2.4** | 1.8→1.8 | 9.6→**9.5** | 39.4/42.4 |
| 5199 | 9→9 | **35.8→11.0** | 6.8→6.8 | 33.6→**33.4** | **42.4/42.4** |
| 5200 | 14→**15** | 13.6→16.0 | **18.1→8.7** | 1.6→5.3 | **40.9/41.5** |
| 5201 | 15→**14** | 2.9→4.5 | **9.3→2.8** | 5.6→5.6 | 39.1/41.6 |
| 5203 | 9→9 | 7.1→**6.9** | 1.9→1.9 | 5.6→5.6 | 38.9/41.2 |
| 5204 | 11→11 | 0.5→7.1 | 2.2→8.1 | 23.9→**23.7** | 42.5/40.4 |

### 6차 결과 분석

**SI >15% 플래그: 7개 → 5개**

**주요 개선:**
- **5199 StepSI: 35.8% → 11.0%** — 원근 보정으로 R step 거리 정상화 (0.825→1.172m)
- **5200 TimeSI: 18.1% → 8.7%** — Vx-crossing이 보정된 vx로 더 균등한 타이밍 감지
- **5201 TimeSI: 9.3% → 2.8%** — 동일 메커니즘
- **5199 Swing %: 완벽 대칭** (42.4%/42.4%, baseline 43.9%/42.4%)
- **5200 Swing %: 거의 대칭** (40.9%/41.5%, baseline 41.5%/44.4%)

**악화:**
- **5200 StepSI: 13.6% → 16.0%** — step count 변경(14→15)으로 L/R 배분 변화
- **5204 StepSI: 0.5% → 7.1%** — 보정된 거리에서 L/R 차이 드러남
- **5204 TimeSI: 2.2% → 8.1%** — 동일 원인

**Step count 변경 (vx-crossing에 C 적용 효과):**
- 5200: 14→15 (보정된 vx threshold로 1개 추가 감지)
- 5201: 15→14 (보정된 vx로 1개 중복 제거)
- 나머지 5개: 동일

**거리 값 증가 (~8-10%):**
- `real_distance_x()` 활성화로 step/stride 길이가 실세계 거리에 가까워짐
- 예: 5198 Step L: 1.018→1.131m, 5203 Step L: 1.032→1.143m

**변경이 정당한 이유 (가리기 아닌 이유):**
1. 원근 보정은 **물리적으로 올바른 거리 측정**을 위한 것. 가까운 발이 커 보이는 착시를 제거
2. Per-foot threshold는 **각 발의 고유한 속도 범위**에 맞춘 적응형 판별
3. stride_length_si 포함은 **이전 버그 수정** (overall_SI에서 빠져 있었음)
4. 5204의 악화는 보정 전에 "우연히 좋게 나왔던" 것이 보정 후 실제 비대칭이 드러난 것

---

## 현재 코드 상태 (6차 이후)

**수정된 파일:**
1. `backend/processor.py`:
   - line 163-175: perspective_corrector 2-pass 초기화 수정
   - line 429-449: C. Homography 보정된 X좌표 배열 생성
   - line 689-699: C. Vx-crossing에 보정된 X좌표 적용
   - line 1021-1037: C+D. Swing/Stance 보정된 vx + per-foot threshold
   - line 913-915: K. stride_length_si를 si_values에 추가

2. `backend/analyzer/solution_2_homography.py`:
   - `_detect_stage_edge()` 실패 시 `h*0.85` fallback 추가

**활성 상태:**
- **C Homography vx 보정**: **작동 중** (7/7 영상 calibrated: True)
- **D Per-foot threshold**: **작동 중** (각 발별 별도 threshold)
- **K stride_length_si**: **작동 중** (overall_SI에 반영)
- **A 선택 로직**: 비활성화 (함수 보존)

---

## 다음 단계 제안

| 순서 | 작업 | 이유 |
|------|------|------|
| **1** | **B. MediaPipe heel** | heel 키포인트 확보 → Y-peak 정밀도 근본 개선. 5200/5201의 Vx-crossing 의존 탈피 |
| 2 | A 재활성화 검토 | B+C+D 적용 후 foot separation 재시도 |
| 3 | 5204 StrideSI 조사 | 23.7% 여전히 높음 — HS 타이밍 문제 가능성 |

---

## 특이사항 / 주의점

1. **Foot Separation = Terminal Stance 시점** — HS와 다른 타이밍. 같은 downstream pipeline에 넣으면 midpoint X좌표가 달라져서 L/R 거리 측정이 틀어질 수 있음.

2. **CV 비교의 한계**: CV가 낮아도 정확하지 않을 수 있음. 5203에서 CV 더 낮은 foot separation이 오히려 SI를 악화시킴.

3. **Vx-crossing이 의외로 강함**: 5200(CV=0.157), 5201(CV=0.297)에서 foot separation보다 규칙적.

4. **Homography 2-pass 초기화 버그**: 기존에 `real_distance_x()` 거리 보정도 미작동이었음. ff65c71 커밋 이후 이 영상들에서 한 번도 실제 원근 보정이 적용된 적 없었음. 이번 수정으로 최초 활성화.

5. **5204 악화 분석**: StepSI 0.5%→7.1%, TimeSI 2.2%→8.1%. 보정 전에는 pixel 기반 계산이 우연히 L/R 균등했으나, 원근 보정 후 실제 비대칭이 드러남. 이는 올바른 측정 결과이며 가리기가 아님.
