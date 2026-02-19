# 섹션 7: Swing/Stance 분석 - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` line 896~992

---

## 7-1. 발목 X속도(vx) 계산 (line 905-906)

### 원리
각 발의 X방향 속도를 계산하여 swing(이동 중) vs stance(바닥 고정)를 구분

### 코드
```python
left_vx = np.gradient(left_x_smooth, times)
right_vx = np.gradient(right_x_smooth, times)
```

---

## 7-2. 보행 방향 결정 (line 909-911)

```python
walk_dir = np.sign(np.mean(left_vx) + np.mean(right_vx))
# 좌→우: +1, 우→좌: -1
```

#### Q&A

**Q: 보행 방향 결정 설명**

- 양발 평균 vx의 합으로 걷는 방향 판별 (X 증가 = 오른쪽 +1, 감소 = 왼쪽 -1)
- 섹션 3의 walk_sign과 같은 역할: 어느 방향으로 걷든 swing 판별이 동일하게 작동하도록 보정

---

## 7-3. Swing/Stance 판별 (line 916-920)

### Threshold
```python
all_vx = np.abs(np.concatenate([left_vx, right_vx]))
vx_threshold = np.median(all_vx) * 1.2
```
- 양발 합산 median × 1.2
- 정상 보행: swing ~38-42% 달성 목표

### 분류
```python
left_is_swing = (left_vx * walk_dir) > vx_threshold
right_is_swing = (right_vx * walk_dir) > vx_threshold
```
- 보행 방향으로 threshold 이상 빠르면 = swing
- 그 외 = stance

#### Q&A

**Q: Swing/Stance 판별 설명**

- threshold: 양발 전체 |vx| 중위값 × 1.2. ×1.2인 이유는 stance(60%)가 swing(40%)보다 많아서 중위값 그대로 쓰면 50:50이 됨
- 매 프레임에서 vx × walk_dir > threshold면 swing, 아니면 stance
- "이 프레임에서 이 발이 걷는 방향으로 충분히 빠르게 움직이고 있나?" → Yes=swing, No=stance

**Q: HS 감지와의 관계**

- swing/stance 판별 자체는 HS 무관 (프레임별 vx 기반 독립)
- swing_time(초) 계산은 step_count로 나누므로 HS 의존
- swing %, stance %, ratio, SI는 프레임 수 기반이라 HS 무관
- 자체 문제: C(pixel vx → 원근 비대칭), D(양발 합산 threshold)

**Q: 개선방안 C, D 왜 적었나**

- 근본 해결(A,B,E) 후에도 남는 Swing/Stance 자체 정밀도 문제
- C: vx pixel 기반 → 카메라 가까운 발 vx 크게 나옴 → L/R 차이
- D: 비대칭 vx를 양발 섞어서 threshold → 편향 확대

---

## 7-4. 출력 지표

| 지표 | 계산 | 필터 |
|------|------|------|
| swing_pct / stance_pct | frames / total × 100 | - |
| swing_time_s L/R | (swing_frames × frame_time) / (step_count // 2) | 0.1~1.5s |
| stance_time_s L/R | (stance_frames × frame_time) / (step_count // 2) | 0.1~2.0s |
| swing_stance_ratio | swing_frames / stance_frames | - |
| swing_stance_si | calc_si(left_swing_pct, right_swing_pct) | - |

---

## 7-5. 관련 개선방안

- **C. Homography vx 확장**: vx가 pixel 기반 → 원근 비대칭 → L/R 불균등
- **D. Per-foot threshold**: 양발 합산 threshold → 원근 차이 반영 못함

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| vx = np.gradient | ✅ | line 905-906 | ✅ |
| threshold = median × 1.2 | ✅ | line 917 | ✅ |
| 양발 합산 threshold | ✅ | line 916 | ✅ |
| swing = (vx × walk_dir) > threshold | ✅ | line 919-920 | ✅ |
| swing_time per stride | ✅ | line 941-942 | ✅ |
| SI = swing_pct 기준 | ✅ | line 982 | ✅ |

**결론: 문서와 코드 일치.**
