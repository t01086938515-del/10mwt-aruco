# 섹션 2: Step Detection (Heel Strike 감지) - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` line 404~641

---

## 전체 흐름 (3단계 폴백 구조)

```
Y-peak (1차) → Vx-crossing (2차) → X-crossing (최후수단)
```

선택 기준: cadence(분당 걸음 수)가 70 이상이면 OK, 미만이면 다음 방법 시도

---

## 2-1. Primary: Y-peak (heel Y 로컬 최대값)

### 원리
발이 바닥에 닿으면 발꿈치 Y좌표가 최대 (화면 아래 = Y 큰 값)

### 입력 준비 (line 416-427)
- `left_heel_y`, `right_heel_y` 추출 (heel 키포인트 없으면 ankle로 대체)
- 5-point 스무딩 적용

### 피크 찾기 (line 434-438)
```
prominence를 3단계로 시도:
  0.06 × Y범위 → 너무 엄격, 보통 1-3개만 찾음
  0.03 × Y범위 → 중간
  0.015 × Y범위 → 완화, 5-12개 찾음
→ 4개 이상 나오면 그 단계에서 멈춤
```

### Vx 필터 (line 441-450)
```
원리: 발이 바닥에 닿는 순간 = 수평 이동 속도가 느림
heel X의 속도(vx)를 구해서, |vx|가 너무 큰 피크는 제거

threshold를 3단계로 시도:
  median(|vx|) × 2.5 → 엄격 (정지에 가까운 피크만)
  median(|vx|) × 4.0 → 중간
  median(|vx|) × 6.0 → 완화
→ 4개 이상 살아남으면 그 단계에서 멈춤
→ 전부 실패하면 필터 없이 전부 유지
```

### 최소 간격 필터 (line 451-459)
```
같은 발 연속 HS 간격 < 0.25s이면:
  Y값이 더 큰 쪽(바닥에 더 가까운)만 남기고 나머지 제거
```

### 평가
- **장점**: 실제 착지 시점을 정확히 포착 (timing 정밀도 우수)
- **한계**: 양발 Y피크가 동시 발생 → dedup에서 50% 소실 → cadence 절반

### Q&A

**Q: 왜 heel이 닿을 때 Y가 높다는 거야?**

카메라 영상 좌표계는 **좌상단이 (0,0)**, 아래로 갈수록 Y가 커지는 구조:
```
(0,0) ────────────→ X (1920)
  │
  │   머리: Y ≈ 200 (위쪽 = 작은 값)
  │
  │   허리: Y ≈ 500
  │
  │   발목: Y ≈ 800 (아래쪽 = 큰 값)
  │
  ↓
  Y (1080)        ← 바닥
```
발이 공중에 떠있을 때 → Y가 약간 작아짐 (위로 올라감)
발이 바닥에 닿을 때 → Y가 최대 (가장 아래)
→ `heel Y의 로컬 최대값 = 발이 바닥에 닿는 순간(Heel Strike)`

**Q: heel 키포인트가 왜 없고, 대체는 어떻게?**

YOLOv8n-pose 모델의 키포인트 구성:
```
0: 코  1-4: 눈/귀  5-6: 어깨  7-8: 팔꿈치  9-10: 손목
11-12: 엉덩이  13-14: 무릎  15-16: 발목(ankle)
17-18: 눈  19-20: 귀  ...
```
heel(발꿈치), toe(발가락) 키포인트는 모델에 있긴 하지만, **측면 카메라에서 신뢰도가 매우 낮음** (가려지거나 작아서). 진단 결과 7개 영상 전부 **heel 가용률 0%**.

대체 방식 (line 416):
```python
left_heel_y_raw = np.array([h.get('left_heel_y', h['left_y']) for h in segment])
```
`h.get('left_heel_y', h['left_y'])` = heel_y가 있으면 쓰고, **없으면 ankle_y를 대신 사용**.
현실적으로 지금은 **100% ankle Y로 동작**하고 있음.

ankle과 heel은 위치가 비슷하지만 완벽히 같진 않아서, ankle Y의 피크가 실제 heel strike 시점과 약간 다를 수 있음.

**Q: 5-point 스무딩이 뭐야?**

원본 데이터에는 **노이즈(떨림)**가 있음:
```
원본:  480, 485, 472, 490, 478, 495, 480, 498, 475, 500
       ↑ 울퉁불퉁 → 가짜 피크 많아짐
```

5-point 스무딩 = **연속 5개 값의 평균**으로 교체:
```
[480, 485, 472, 490, 478] → 평균 = 481
[485, 472, 490, 478, 495] → 평균 = 484
[472, 490, 478, 495, 480] → 평균 = 483
```

```python
kernel = np.ones(5) / 5  # = [0.2, 0.2, 0.2, 0.2, 0.2]
smoothed = np.convolve(raw_data, kernel, mode='same')
```

결과:
```
스무딩: 481, 484, 483, 488, 486, 490, 488, 491, ...
       ↑ 부드러워짐 → 진짜 피크만 남음
```
숫자가 클수록 더 부드러워지지만 반응이 느려짐. 5는 적당한 균형점.

**Q: threshold = median(|vx|) × 2.5 쉬운 설명**

vx는 발목의 가로 이동 속도(px/s). median(|vx|)는 "보통 속도"이고, ×2.5는 "보통의 2.5배 이상이면 비정상"이라는 기준. HS 착지 순간은 발이 멈춰야 하는데 threshold 초과하면 swing 중이라 판단 → 해당 피크 제거.

**Q: vx 필터(×2.5)와 Swing/Stance(×1.2) 차이**

- vx 필터(×2.5): HS 후보 중 가짜 제거 — "보통의 2.5배 넘게 빠르면 착지가 아님"
- Swing/Stance(×1.2): 매 프레임 분류 — "보통보다 조금만 빨라도 swing"
- 기준이 다른 이유: HS는 거의 멈추는 순간이라 엄격, swing은 조금만 움직여도 해당이니 느슨.

**Q: Y-peak 감지 실패 이유 (5199/5200/5201)**

세 가지 원인:
1. heel 키포인트 0% 가용률 → ankle 대체 → Y 진폭 작음 (10~15px vs 정상 30~40px)
2. 진폭이 작아 노이즈에 묻힘 → prominence 낮춰도 구분 불가
3. bilateral crosstalk → 양발 동시 피크 → dedup에서 절반 제거

---

## 2-2. Fallback: Vx-crossing (ankle X 속도 하향 교차)

### 발동 조건 (line 578)
Y-peak의 cadence가 70 spm 미만일 때

### 원리 (line 462-506)
```
걸을 때 발이 앞으로 스윙하면 X속도가 양수(빠름)
→ 착지하면 X속도가 0으로 떨어짐
→ 이 "양수→0 하향 교차" 지점 = Heel Strike
```

### 상세 과정
1. heel X의 속도를 구하고, 보행 방향으로 정규화 (line 586-588)
2. 양발 합산 positive vx의 median 계산
3. threshold를 3단계로 시도: median × [0.35, 0.25, 0.15]
4. 하향 교차 후 vx=0 지점을 정밀 탐색 (line 479-498)
5. sub-frame 선형 보간: 프레임 사이 정확한 시간 추정 (line 485-486)
6. min_gap 0.3s로 중복 제거

### 선택 로직 (line 606-614)
```
Vx cadence >= 70 → Vx 사용
Vx도 < 70 → Y-peak과 Vx 중 이벤트가 더 많은 쪽 사용
```

### 평가
- **장점**: L/R 독립적으로 감지 → crosstalk 없음, 정확한 counting
- **한계**: 감속 프로필 L/R 비대칭이면 타이밍 차이 발생

### Q&A

**Q: Y-peak의 cadence가 70 spm 미만일 때 — 이게 왜 조건이 필요하지?**

정상 보행 cadence 범위는 80~120 spm. cadence가 70 미만이면 Y-peak이 실패했다는 신호.

실패 원인 = bilateral crosstalk:
- 왼발/오른발 Y피크가 거의 동시에 발생
- dedup에서 0.15s 이내를 하나로 합침
- step의 절반이 소실 → cadence가 정상의 절반으로 떨어짐

```
정상: 10m 구간에서 ~12 step → cadence 100 spm ✅
실패: 10m 구간에서 ~6 step → cadence 50 spm ❌ (< 70 → fallback 발동)
```

70이라는 수치:
- 정상 최저 ~80 spm에서 여유를 둔 값
- 노인/장애 보행도 보통 60~80 spm
- 50 이하면 거의 확실히 감지 오류

실제 사례 — 5200: Y-peak cadence 50 → Vx-crossing 발동

**Q: cadence 80 이상에서도 Vx-crossing을 쓰면 오류가 많은가?**

counting(개수) 측면에서는 Vx-crossing이 더 좋음 (crosstalk 없음). 문제는 timing(시점 정밀도):

| | Y-peak | Vx-crossing |
|--|--------|-------------|
| counting (개수) | crosstalk에 취약 | 정확함 |
| timing (시점) | 정확함 | L/R 비대칭 가능 |

Vx-crossing의 타이밍 문제:
- threshold 값에 따라 교차 시점이 달라짐
- L/R 감속 프로필이 다르면 같은 threshold에서도 시점이 비대칭
- → Step Time SI가 높아질 수 있음

현재 로직: Y-peak 충분(cadence >= 70) → timing 우선으로 Y-peak 사용 / Y-peak 실패 → counting이라도 맞추자 → Vx-crossing

잠재적 개선 포인트: Vx-crossing으로 개수 확보 + Y-peak으로 시점 보정 = 양쪽 장점 결합 가능 (현재 미구현)

**Q: "속도가 threshold 이하로 떨어지는 순간 → threshold에 따라 시점이 달라짐 → L/R 감속 프로필이 다르면 시점도 다름" — 이게 무슨 말?**

발이 스윙→착지로 넘어갈 때 속도가 줄어드는 패턴(감속 프로필)이 왼발/오른발이 다를 수 있음:

```
왼발 vx:  500 → 400 → 300 → 200 → 100 → 0  (균일 감속)
                              ↑ threshold=200 교차

오른발 vx: 500 → 450 → 400 → 350 → 100 → 0  (마지막 급감속)
                                     ↑ threshold=200 교차
```

같은 threshold인데 왼발은 착지 0.15s 전, 오른발은 0.05s 전에 교차 → 타이밍 비대칭 발생.

원인: 근력 차이, 보행 습관, 측면 카메라 원근(가까운 발 vs 먼 발의 X변화량 차이).

SI에 미치는 영향:
```
실제 step time:     0.55s / 0.55s → SI = 0% (대칭)
Vx 감지 step time:  0.65s / 0.45s → SI = 36% (비대칭으로 오인!)
```

Y-peak은 "바닥 닿는 순간" 자체를 측정하므로 이 문제 없음.

**Q: threshold가 무슨 의미? 속도 변화가 일정하지 않은 사람이 문제가 되는 건가?**

threshold(임계값) = "이 속도 이하면 착지로 판정"하는 기준선.
```
속도 600→500→400→300→200→100→0
                     ↑ threshold=200: 이 순간을 "착지 시작"으로 판정
```
threshold 위 = 아직 스윙 중, threshold 아래 = 착지 시작.

문제의 주 원인은 사람의 비대칭보다 **카메라 원근**:
- 카메라에 가까운 발: X좌표 변화가 크게 보임 → threshold를 늦게 통과
- 카메라에서 먼 발: X좌표 변화가 작게 보임 → threshold를 일찍 통과
- 같은 속도인데 다르게 측정됨 → L/R 타이밍 비대칭

부차적 원인: 왼발/오른발의 실제 착지 습관 차이 (부드러운 착지 vs 탁 찍는 착지).

**Q: 카메라 원근 때문에 가까운 발/먼 발이 다르게 보인다는 것 더 설명해줘**

측면 카메라에서 두 발은 카메라로부터 거리가 다름:
```
위에서 본 모습:
    [카메라] ──0.5m── 가까운 발
             ──2.5m── 먼 발 (벽 쪽)
```

같은 30cm 이동이라도:
- 가까운 발: 화면에서 80px 이동 (크게 보임)
- 먼 발: 화면에서 20px 이동 (작게 보임)

속도(vx) 계산 시:
```
실제 동일한 2.0 m/s인데:
  가까운 발: 1600 px/s (화면상)
  먼 발:     400 px/s (화면상)
```

threshold = 500 px/s 적용 시:
- 가까운 발: 착지 직전에야 통과 (정확)
- 먼 발: 스윙 중인데도 이미 threshold 이하 (너무 일찍 판정)

→ 타이밍 비대칭 발생

참고: Homography 보정이 거리 계산에는 적용되지만, vx 속도 계산에는 미적용 상태 → 잠재적 개선 포인트.

**Q: Homography 보정을 이미 구현했는데 왜 속도 계산에는 안 되는 거야?**

PerspectiveCorrector.real_distance_x()는 **거리 계산**에만 적용됨:
- ✅ Step Length L/R (line 818-819)
- ✅ Stride Length L/R (line 731-732)

속도(vx) 계산은 전부 pixel 기반으로 Homography 미적용:
- ❌ Y-peak vx 필터 (line 441): `np.gradient(heel_x_smooth)` → pixel 속도
- ❌ Vx-crossing 감지 (line 467, 586-587): pixel 속도
- ❌ Swing/Stance 분석 (line 905-906): pixel 속도

원인: Homography를 "최종 거리 결과"에만 적용하고, "중간 과정(속도)"에는 적용하지 않은 구조.

**→ 개선 포인트: vx 계산에도 Homography 보정 적용 시 L/R 비대칭 감소 기대**

**Q: Homography 빼면 안 되나? 잘 되나?**

- 빼면 안 됨: 거리 보정 ~8% 차이. 빼면 step/stride length 과소측정.
- 잘 작동함: 7개 영상에서 일관된 보정. ArUco 인식 안정적.
- 적용 범위: 거리(Step/Stride Length)만 적용, vx 계산에는 미적용.

**Q: 다른 포즈 모델(MediaPipe 등)로 바꾸면?**

- MediaPipe: 33개 키포인트 (heel #29,30 + toe #31,32), 빠름, 교체 쉬움
- 예상 개선: heel 가용률 0% → 70~90%, Y range 2~3배 증가, 3개 문제 영상 중 2개 해결 기대
- 단점: 사람 감지 약함 → YOLO+MediaPipe 하이브리드가 이상적

**Q: 노이즈를 영상 보정으로 해결?**

CLAHE, 감마 보정, 샤프닝 등 가능하지만, 현재 문제 영상은 영상 품질이 아닌 모델 한계(heel 미지원)가 원인 → 직접 효과 낮음. 어두운 환경 등에는 효과적 → 개선방안 G.

---

## 2-3. Last Resort: X-crossing (좌우 발 X좌표 교차)

### 발동 조건 (line 622)
위 두 방법 모두 step_events < 4개일 때

### 원리 (line 624-641)
```
left_x - right_x 의 부호 변화 = 두 발이 교차하는 순간
부호 > 0 → 왼발이 앞 → leading = 'left'
부호 < 0 → 오른발이 앞 → leading = 'right'
```

### 평가
- **장점**: 항상 step을 찾아줌 (신호만 있으면 작동)
- **한계**: 교차 시점 ≠ 실제 HS 시점 → 타이밍 부정확, 교대 short-long 패턴 발생

### Q&A

**Q: X-crossing 상세 원리**

- `left_x - right_x` 부호 변화 = 두 발이 교차하는 순간
- Perry 기준으로 Mid-Swing에 해당 (착지가 아님)
- 교차 시점 ≠ 실제 HS → short-long 교대 패턴 발생 → Step Time SI 악화
- 5201: 16개 과다 감지 → cadence 133 spm

**Q: X-crossing 단점 있는데 왜 사용?**

"틀린 답 vs 답 없음" 선택. X-crossing 없으면 step_count=3으로 거의 모든 지표 계산 불가. 목표는 발동 안 되게 만드는 것 (Foot Separation 도입으로).

**Q: Perry Mid-Stance와 X-crossing 관계**

X-crossing은 Mid-Swing 감지. 반대 발의 Mid-Stance 시점에서 swing 다리가 stance 다리를 추월. MSw + MSt 동시 감지하면 보행주기 재구성 가능 → 개선방안 F.

**Q: step 4개 미감지면 이미 늦은 거 아닌가?**

아님. 실시간이 아니라 사후 분석. 10m 다 걷고 나서 저장된 전체 ankle_history로 세 방법을 순서대로 시도. 데이터 부족이 아니라 신호 품질 문제.

---

## 2-4. Bilateral Dedup (양발 병합 후 중복 제거)

`build_step_events_from_indices()` 함수 내 (line 536-563)

### 규칙
```
L/R 이벤트를 시간순으로 합친 후:

1. dt < 0.15s → 무조건 하나만 남김 (더 높은 heel_y 유지)
   이유: 너무 가까우면 같은 보행 이벤트의 중복

2. 0.15s ≤ dt < 0.25s + 같은 발 → 중복 제거
   이유: 같은 발이 0.25s 안에 두 번 닿을 수 없음
   다른 발이면 유지 (실제 연속 step 가능)

3. boundary filter: start_t + 0.3s 이전 이벤트 제거
   이유: START 마커 통과 직후는 불안정

4. tail filter: 마지막 interval이 median × 2.5 초과 → 마지막 step 제거
   이유: FINISH 마커 근처의 잘못된 감지
```

### Q&A

**Q: Bilateral dedup heel_y 비교의 문제**

dt < 0.15s 병합 시 heel_y 높은 쪽 유지 → 카메라에 가까운 발이 항상 이김 (편향). 대안: vx가 낮은 쪽(진짜 착지), prominence가 큰 쪽(확실한 피크). Leading Foot은 이후 재판별되므로 라벨 영향 없고 타이밍에만 미세 편향 → 개선방안 H.

**Q: 교대 강제 로직 문제**

연속 같은 발 → 무조건 flip은 원인 분석 없이 덮는 방식. 라벨링 오류/허위 이벤트/비정상 보행 구분 안 함. 임상 검사에서 환자의 비정상 패턴을 강제로 정상으로 만들 위험.

**Q: Boundary/Tail filter 영향**

- Boundary (start_t + 0.3s): 최대 1스텝 제거, 영향 작음
- Tail (마지막 interval > median × 2.5): 최대 1개 제거, 영향 작음
- 진짜 문제는 중간 구간의 HS 누락/crosstalk/이상 interval → 양 끝 필터로는 해결 안 됨 → 근본 해결(A, B) 필요

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| Y-peak prominence 3단계 | 0.06→0.03→0.015 | line 434 | ✅ |
| Y-peak vx 필터 3단계 | ×2.5→×4.0→×6.0 | line 444 | ✅ |
| Y-peak min_gap | 0.25s | line 429 | ✅ |
| Vx-crossing 발동조건 | cadence < 70 | line 578 | ✅ |
| Vx threshold 3단계 | median × 0.35→0.25→0.15 | line 592 | ✅ |
| Vx sub-frame 보간 | 있음 | line 485-486 | ✅ |
| Vx min_gap | 0.3s | line 462 | ✅ |
| X-crossing 발동조건 | step < 4개 | line 622 | ✅ |
| Dedup 0.15s 무조건 | 있음 | line 543 | ✅ |
| Dedup 0.25s 같은발 | 있음 | line 549 | ✅ |
| Boundary 0.3s | 있음 | line 557 | ✅ |
| Tail filter ×2.5 | 있음 | line 561 | ✅ |

**결론: 문서와 코드 일치. 로직 자체에 문제 없음.**

### 알려진 한계 (진단 결과 기반)
- Heel keypoint 가용률 **0%** (YOLOv8n-pose) → ankle Y로 대체됨
- 5200: Y-peak L=6,R=6 → dedup 후 7개 → cadence 50 (crosstalk 문제)
- 5201: Y-peak 5개 → Vx-crossing도 부족 → X-crossing 16개 (과다)
- Foot separation `|left_x - right_x|` peaks가 대안으로 유망 (진단 결과)
