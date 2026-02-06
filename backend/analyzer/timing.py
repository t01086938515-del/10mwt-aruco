# analyzer/timing.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CrossingEvent:
    """라인 통과 이벤트"""
    frame_index: int
    precise_frame: float
    precise_time_s: float
    foot: str
    position: Tuple[float, float]


class PrecisionTimer:
    """서브픽셀 정밀도의 시간 측정기"""

    def __init__(self, fps: float):
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.start_event: Optional[CrossingEvent] = None
        self.end_event: Optional[CrossingEvent] = None
        self.position_history: List[Tuple[int, float, float]] = []

    def add_position(self, frame_idx: int, position: Tuple[float, float]):
        self.position_history.append((frame_idx, position[0], position[1]))

    def detect_line_crossing(
        self,
        line_y: float,
        direction: str = 'forward'
    ) -> Optional[CrossingEvent]:
        if len(self.position_history) < 2:
            return None

        for i in range(1, len(self.position_history)):
            prev = self.position_history[i - 1]
            curr = self.position_history[i]
            prev_frame, prev_x, prev_y = prev
            curr_frame, curr_x, curr_y = curr

            crossed = False
            if direction == 'forward':
                crossed = prev_y < line_y <= curr_y
            else:
                crossed = prev_y > line_y >= curr_y

            if crossed:
                if curr_y != prev_y:
                    t = (line_y - prev_y) / (curr_y - prev_y)
                else:
                    t = 0.5

                precise_frame = prev_frame + t * (curr_frame - prev_frame)
                precise_time = precise_frame * self.frame_duration
                precise_x = prev_x + t * (curr_x - prev_x)

                return CrossingEvent(
                    frame_index=curr_frame,
                    precise_frame=round(precise_frame, 3),
                    precise_time_s=round(precise_time, 4),
                    foot='unknown',
                    position=(precise_x, line_y)
                )

        return None

    def measure_10m_time(
        self,
        start_line_y: float,
        end_line_y: float
    ) -> Dict:
        start_event = self.detect_line_crossing(start_line_y, 'forward')
        if start_event is None:
            return {'error': '시작선 통과 감지 실패'}
        self.start_event = start_event

        end_event = self.detect_line_crossing(end_line_y, 'forward')
        if end_event is None:
            return {'error': '종료선 통과 감지 실패'}
        self.end_event = end_event

        elapsed_time = end_event.precise_time_s - start_event.precise_time_s
        speed_mps = 10.0 / elapsed_time if elapsed_time > 0 else 0

        return {
            'start_time_s': start_event.precise_time_s,
            'end_time_s': end_event.precise_time_s,
            'elapsed_time_s': round(elapsed_time, 3),
            'speed_mps': round(speed_mps, 3),
            'start_frame': start_event.precise_frame,
            'end_frame': end_event.precise_frame,
            'precision': 'sub-frame'
        }


class DualFootTimer(PrecisionTimer):
    """양발 개별 추적 타이머"""

    def __init__(self, fps: float):
        super().__init__(fps)
        self.left_history: List[Tuple[int, float, float]] = []
        self.right_history: List[Tuple[int, float, float]] = []

    def add_foot_positions(
        self,
        frame_idx: int,
        left_pos: Optional[Tuple[float, float]],
        right_pos: Optional[Tuple[float, float]]
    ):
        if left_pos:
            self.left_history.append((frame_idx, left_pos[0], left_pos[1]))
        if right_pos:
            self.right_history.append((frame_idx, right_pos[0], right_pos[1]))

    def _detect_movement_direction(self) -> str:
        all_history = self.left_history + self.right_history
        if len(all_history) < 10:
            return 'forward'

        all_history.sort(key=lambda x: x[0])
        n = len(all_history)
        early_y = np.mean([p[2] for p in all_history[:n//10+1]])
        late_y = np.mean([p[2] for p in all_history[-n//10-1:]])

        return 'forward' if late_y > early_y else 'backward'

    def measure_10m_time_dual(
        self,
        start_line_y: float,
        end_line_y: float
    ) -> Dict:
        direction = self._detect_movement_direction()

        if direction == 'forward':
            first_line = start_line_y
            second_line = end_line_y
        else:
            first_line = end_line_y
            second_line = start_line_y

        self.position_history = self.left_history
        left_start = self.detect_line_crossing(first_line, direction)
        left_end = self.detect_line_crossing(second_line, direction)

        self.position_history = self.right_history
        right_start = self.detect_line_crossing(first_line, direction)
        right_end = self.detect_line_crossing(second_line, direction)

        start_event = None
        if left_start and right_start:
            if left_start.precise_time_s < right_start.precise_time_s:
                start_event = left_start
                start_event.foot = 'left'
            else:
                start_event = right_start
                start_event.foot = 'right'
        elif left_start:
            start_event = left_start
            start_event.foot = 'left'
        elif right_start:
            start_event = right_start
            start_event.foot = 'right'

        end_event = None
        if left_end and right_end:
            if left_end.precise_time_s < right_end.precise_time_s:
                end_event = left_end
                end_event.foot = 'left'
            else:
                end_event = right_end
                end_event.foot = 'right'
        elif left_end:
            end_event = left_end
            end_event.foot = 'left'
        elif right_end:
            end_event = right_end
            end_event.foot = 'right'

        if start_event is None or end_event is None:
            return {'error': '라인 통과 감지 실패', 'direction': direction}

        elapsed_time = end_event.precise_time_s - start_event.precise_time_s
        if elapsed_time < 0:
            elapsed_time = abs(elapsed_time)
            start_event, end_event = end_event, start_event

        speed_mps = 10.0 / elapsed_time if elapsed_time > 0 else 0

        return {
            'start_time_s': start_event.precise_time_s,
            'start_foot': start_event.foot,
            'end_time_s': end_event.precise_time_s,
            'end_foot': end_event.foot,
            'elapsed_time_s': round(elapsed_time, 3),
            'speed_mps': round(speed_mps, 3),
            'direction': direction,
            'precision': 'sub-frame-dual-foot'
        }
