# 섹션 3: Leading Foot 라벨링 - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` line 643~662

---

## 3-1. 공간 위치 기반 판별 (line 646-660)

### 원리
각 HS 시점에서 보행 방향으로 더 앞에 있는 발 = 그 step의 leading foot

### 1단계: 보행 방향 결정
```python
first_mid = (첫 step의 left_x + right_x) / 2
last_mid  = (마지막 step의 left_x + right_x) / 2
walk_sign = +1 (오른쪽으로 걸음) or -1 (왼쪽으로 걸음)
```

### 2단계: 각 step에서 판단
```python
left_ahead  = left_x × walk_sign
right_ahead = right_x × walk_sign
→ left_ahead가 더 크면 → leading = 'left'
→ right_ahead가 더 크면 → leading = 'right'
```

예시 (오른쪽으로 걷는 경우, walk_sign = +1):
```
step 1: left_x=500, right_x=480 → 왼발 20px 앞 → 'left'
step 2: left_x=530, right_x=550 → 오른발 20px 앞 → 'right'
step 3: left_x=590, right_x=570 → 왼발 20px 앞 → 'left'
```

### Q&A

**Q: 공간 위치 판별 쉬운 설명**

- 보행 방향 파악: 첫 스텝 vs 마지막 스텝 중점 비교 → X 증가하면 오른쪽 (walk_sign=+1)
- 각 착지 순간: 걷는 방향으로 X좌표가 큰 쪽 = 앞에 있는 발 = leading foot
- walk_sign을 곱하는 이유: 왼쪽으로 걸으면 X가 작은 쪽이 앞인데, ×(-1) 하면 "큰 쪽이 앞"으로 통일됨

**Q: walk_sign이 왜 필요한가?**

- 카메라 앞에서 어느 방향으로 걷든 leading foot을 맞게 잡기 위한 장치
- walk_sign 없으면 왼쪽으로 걷는 셋업에서 leading foot이 전부 뒤집힘
- 예: 왼쪽으로 걸으면 X가 작은 쪽이 앞인데, 단순 비교하면 X 큰 쪽이 앞이라고 오판

**Q: 첫 발만 봐도 방향 알 수 있지 않나?**

- 한 점으로는 위치만 알고 방향은 모름. 두 점(첫+마지막)이 있어야 방향 결정
- 10MWT 직진이면 시작 위치로 추론 가능하지만, 현재 로직이 가정 없이 데이터로 확인하는 방식이라 더 안전

**Q: 2단계 판단도 1단계와 같은 말?**

- walk_sign=+1이면 곱해도 값 안 바뀜 → "X 큰 쪽이 앞"
- walk_sign이 의미 있는 건 -1일 때(왼쪽으로 걸을 때)뿐

---

## 3-2. 교대 강제 (line 620-628)

정상 보행은 반드시 L-R-L-R 교대. 같은 발 연속이면 두 번째를 뒤집음:

```python
opposite = {'left': 'right', 'right': 'left'}
for i in range(1, len(step_events)):
    if step_events[i]['leading_foot'] == step_events[i-1]['leading_foot']:
        step_events[i]['leading_foot'] = opposite[step_events[i]['leading_foot']]
```

앞에서부터 순차 1회 패스. 대부분의 경우 충분하지만, 완전 교대가 보장되지는 않음.

> ⚠️ **주의**: 현재 코드에 존재하나, 원인 분석 없이 무조건 flip하는 방식이라 임상적으로 비정상 보행 패턴을 숨길 수 있음. 대안 로직 준비 후 교체 필요. 상세: 섹션 2-4 Q18 참조.

---

## 3-3. 한계

1. **측면 카메라에서 X 분리가 작음**: 두 발의 X좌표 차이가 2-5px → 노이즈 수준에서 오판 가능
2. **double support phase**: 양발 다 바닥에 있으면 X 위치 거의 동일
3. **교대 강제의 오류 전파**: 첫 라벨이 틀리면 이후 전부 뒤집힘
4. **라벨 오판의 영향**: Step Length L/R에서 긴 step이 한쪽에 몰림 → SI 급등

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| walk_sign 계산 | first/last midpoint 비교 | line 648-650 | ✅ |
| 공간 위치 판별 | left_x × sign vs right_x × sign | line 653-660 | ✅ |
| 교대 강제 | 연속 같은 발 → flip | line 620-628 | ✅ |

**결론: 문서와 코드 일치.**
