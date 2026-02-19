# 섹션 1: 입력 데이터 - 검증 결과

> 검증일: 2026-02-10
> 대상 코드: `processor.py` → `_analyze_gait_parameters()`

---

## 문서 내용

| 항목 | 설명 |
|------|------|
| **구간** | START~FINISH ArUco 마커 crossing 사이 (`start_t` ~ `finish_t`) |
| **ankle_history** | 프레임별 `{left_x, left_y, right_x, right_y, left_heel_x, left_heel_y, ...}` |
| **스무딩** | 5-point moving average (`np.convolve(kernel=ones(5)/5)`) |
| **FPS** | 원본 60fps, frame_skip=3 → 실효 ~20fps |

---

## 코드 대조

### 구간 필터링 (processor.py line 368-385)
```python
start_ev = next((e for e in self.crossing_events if e['line'] == 'start'), None)
finish_ev = next((e for e in self.crossing_events if e['line'] == 'finish'), None)
start_t = start_ev['timestamp_s']
finish_t = finish_ev['timestamp_s']
segment = [h for h in self.ankle_history if start_t <= h['timestamp_s'] <= finish_t]
```

### ankle_history 구조 (processor.py line 204-227)
```python
history_entry = {
    'timestamp_s': timestamp_s,
    'frame_idx': frame_idx,
    'left_x': left_ankle[0],   'left_y': left_ankle[1],
    'right_x': right_ankle[0], 'right_y': right_ankle[1],
    # 조건부 추가:
    'left_heel_x', 'left_heel_y',   # heel 신뢰도 > 0.3
    'right_heel_x', 'right_heel_y', # heel 신뢰도 > 0.3
    'left_toe_x', 'left_toe_y',     # toe 신뢰도 > 0.3
    'right_toe_x', 'right_toe_y',   # toe 신뢰도 > 0.3
}
```

### 스무딩 (processor.py line 394)
```python
kernel = np.ones(5) / 5
left_x_smooth = np.convolve(left_x, kernel, mode='same')
```

### FPS (server.py line 307)
```python
frame_skip = 3 if fps > 45 else 2  # 60fps → 20fps, 30fps → 15fps
```

---

## 검증 결과

| 항목 | 문서 | 코드 | 일치? |
|------|------|------|------|
| 구간 | START~FINISH crossing | `crossing_events` 필터 | ✅ |
| ankle_history 필드 | left/right x/y + heel + toe | line 204-227 | ✅ |
| 스무딩 | 5-point MA | `np.ones(5)/5` | ✅ |
| FPS | 60fps, skip=3 → ~20fps | `3 if fps > 45 else 2` | ✅ |

**결론: 문제 없음**
