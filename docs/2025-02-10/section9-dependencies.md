# 섹션 9: 변수별 주요 의존성 & 취약점 - 검증 결과

> 검증일: 2026-02-10
> 대상: `gait-variable-logic.md` 섹션 9 vs 실제 코드

---

## 9-1. 원문 의존성 테이블

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

---

## 9-2. 검증 및 수정 사항

### ⚠️ "라벨 무관" 표기 보완 필요

stride_length/time L/R은 "라벨 무관(인덱스 기반)"이 맞지만, **HS 감지가 부정확하면 인덱스가 밀려서 L/R이 뒤집힘**. "라벨 무관"이라는 표현이 "HS에 의존하지 않는다"로 오해될 수 있음.

→ 수정 제안: "라벨 무관, 단 HS 감지 정확도에 의존 (HS 누락/추가 시 인덱스 밀림)"

### ⚠️ swing/stance 취약점 보완 필요

현재: "전역 threshold → 개인차 미반영"
추가 필요:
- pixel 기반 vx → 원근 비대칭 (개선방안 C)
- 양발 합산 threshold → 원근 차이 미반영 (개선방안 D)

### ⚠️ overall_SI 구성 명시 필요

- stride_length_si가 제외되어 있음 (6개 SI만 평균)
- 문서에 포함/미포함 목록 명시 필요

#### Q&A

**Q: 고정형 threshold들이 개인 신체조건/상황을 고려할 수 있나?**

- 적응형 (median 기반): Y-peak prominence, vx 필터, Swing/Stance threshold, 이상치 제거 → 개인 속도/신호에 자동 조절됨
- 고정형 (하드코딩): step_length 0.2~2.0m, step_time 0.2~2.0s, stride 0.3~4.0m, cadence 30~200 spm, min_gap 0.25s/0.15s → 극단 케이스(소아, 중증 장애, 초고속) 문제
- 현재 환자 프로필 입력 구조 없음
- 필터에 걸린 값을 "이상 보행 후보"로 보관하면 임상 참고 가능
- 개선방안 L(장기)로 등록됨

---

## 9-3. 근본 의존성 요약

**모든 하류 지표의 최상위 의존:**

```
HS 감지 정확도
  ├── step_count
  ├── cadence (step interval)
  ├── step_length/time L/R (라벨 + HS)
  ├── stride_length/time L/R (인덱스 + HS)
  ├── swing_time_s (step_count로 나눔)
  └── overall_SI (위 지표들의 파생)

vx 정밀도 (HS 독립)
  ├── swing/stance %
  ├── swing/stance ratio
  └── swing_stance_si
```

---

## 검증 결과

| 항목 | 원문 | 검증 | 일치? |
|------|------|------|------|
| step_count 의존 | HS 감지 | ✅ | ✅ |
| cadence 의존 | interval median | ✅ | ✅ |
| step L/R 의존 | 라벨 | ✅ | ✅ |
| stride L/R "라벨 무관" | 라벨 무관 | ⚠️ HS 의존 명시 필요 | ⚠️ |
| swing/stance 취약점 | 전역 threshold | ⚠️ pixel vx + 합산 threshold 추가 | ⚠️ |
| overall_SI | 모든 SI 평균 | ⚠️ stride_length_si 제외 명시 | ⚠️ |

**결론: 핵심 일치, 3개 항목 보완 필요.**
