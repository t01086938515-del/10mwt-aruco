# 섹션 8: SI (Symmetry Index) 계산 - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` line 789~996

---

## 8-1. calc_si 함수 (line 789-795)

### 공식
```python
def calc_si(left_val, right_val):
    """SI = |L-R| / (0.5*(L+R)) × 100 (%)"""
    if left_val is None or right_val is None:
        return None
    if left_val + right_val <= 0:
        return None
    return round(abs(left_val - right_val) / (0.5 * (left_val + right_val)) * 100, 1)
```

- 0% = 완전 대칭, 높을수록 비대칭
- None 체크 + 0 나눗셈 방지

---

## 8-2. SI 적용 대상

| SI 변수 | 입력 L/R | calc_si 사용 | si_values 포함 |
|---------|---------|-------------|---------------|
| step_length_si | step_length L/R | ✅ | ✅ (line 847) |
| step_time_si | step_time L/R | ✅ | ✅ (line 872) |
| stride_time_si | stride_time L/R | ✅ | ✅ (line 893) |
| swing_time_si | swing_time L/R | ✅ | ✅ (line 950) |
| stance_time_si | stance_time L/R | ✅ | ✅ (line 963) |
| swing_stance_si | swing_pct L/R | ✅ | ✅ (line 985) |
| stride_length_si | stride_length L/R | ❌ 직접 계산 (line 770) | ❌ 미포함 |

### ⚠️ stride_length_si 불일치
- calc_si() 사용하지 않고 직접 같은 공식으로 계산
- **si_values에 추가되지 않음** → overall_si에 반영 안 됨
- 의도인지 실수인지 확인 필요

#### Q&A

**Q: stride_length_si 미포함 해결하면 되나? 근본 해결되면 잘 나오나?**

- 단순 코드 실수. calc_si() 사용 + si_values.append(si) 추가하면 끝 (2줄 수정)
- 근본 해결(A,B,E) 후 stride_length L/R이 정확해지면 → SI도 정확해짐
- 개선방안 K로 등록됨

---

## 8-3. Overall Symmetry Index (line 995-996)

```python
if si_values:
    result['overall_symmetry_index'] = round(np.mean(si_values), 1)
```

- 6개 SI의 단순 평균 (stride_length_si 제외)
- 한 지표의 극단값이 전체를 끌어올릴 수 있음

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| SI 공식 | |L-R|/(0.5*(L+R))×100 | line 795 | ✅ |
| stride_length_si 미포함 | ✅ 기재됨 | line 770-772 (si_values 미추가) | ✅ |
| overall_si = mean(si_values) | ✅ | line 996 | ✅ |

**결론: 문서와 코드 일치. stride_length_si의 si_values 미포함은 의도 확인 필요.**
