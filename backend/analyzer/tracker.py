# analyzer/tracker.py

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


class TrajectoryProcessor:
    """궤적 데이터 처리 (보간, 평활화)"""

    def __init__(
        self,
        max_gap_frames: int = 10,
        smoothing_window: int = 5
    ):
        self.max_gap_frames = max_gap_frames
        self.smoothing_window = smoothing_window

    def interpolate_trajectory(
        self,
        positions: List[Tuple[int, float, float]],
        total_frames: int
    ) -> np.ndarray:
        if len(positions) < 2:
            return np.array(positions) if positions else np.array([])

        frames = np.array([p[0] for p in positions])
        x_vals = np.array([p[1] for p in positions])
        y_vals = np.array([p[2] for p in positions])

        all_frames = np.arange(total_frames)

        x_interp_func = interp1d(
            frames, x_vals, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        y_interp_func = interp1d(
            frames, y_vals, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )

        x_interp = x_interp_func(all_frames)
        y_interp = y_interp_func(all_frames)

        result = np.column_stack([all_frames, x_interp, y_interp])
        result = self._mark_long_gaps(result, frames)

        return result

    def _mark_long_gaps(
        self, trajectory: np.ndarray, valid_frames: np.ndarray
    ) -> np.ndarray:
        result = trajectory.copy()
        for i in range(len(valid_frames) - 1):
            gap = valid_frames[i + 1] - valid_frames[i]
            if gap > self.max_gap_frames:
                start = int(valid_frames[i]) + 1
                end = int(valid_frames[i + 1])
                result[start:end, 1:] = np.nan
        return result

    def smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        result = trajectory.copy()
        valid_mask = ~np.isnan(result[:, 1])
        if np.sum(valid_mask) > self.smoothing_window:
            result[valid_mask, 1] = uniform_filter1d(
                result[valid_mask, 1], size=self.smoothing_window, mode='nearest'
            )
            result[valid_mask, 2] = uniform_filter1d(
                result[valid_mask, 2], size=self.smoothing_window, mode='nearest'
            )
        return result

    def get_interpolation_mask(
        self, original_frames: List[int], total_frames: int
    ) -> np.ndarray:
        mask = np.ones(total_frames, dtype=bool)
        mask[original_frames] = False
        return mask
