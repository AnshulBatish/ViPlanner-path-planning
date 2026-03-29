"""Centerline path planning with Kalman-filter fallback.

Extracts a drivable centerline from the binary traversable-road mask and
smooths it with a linear Kalman filter. When the mask is degraded or
empty the filter predicts forward using a constant-curvature motion model,
providing graceful fallback instead of zero output.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from offroad_autonomy.types import PathPlan, PipelineConfig, StabilizedResult

logger = logging.getLogger("offroad_autonomy.planning")


class _KalmanTracker:
    """Linear Kalman filter over [lateral_offset, heading, curvature]."""

    DIM_X = 3
    DIM_Z = 2

    def __init__(self, q: float, r: float) -> None:
        self.x = np.zeros(self.DIM_X)
        self.P = np.eye(self.DIM_X) * 100.0
        self.F = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.Q = np.eye(self.DIM_X) * q
        self.R = np.eye(self.DIM_Z) * r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(self.DIM_X) - k @ self.H) @ self.P
        return self.x.copy()


class CenterlinePlanner:
    """Extract a centerline from the road mask and track it with a Kalman filter."""

    def __init__(self, config: PipelineConfig) -> None:
        self._n_samples = config.centerline_samples
        self._min_road_px = config.min_road_pixels
        self._max_misses = config.fallback_after_n_misses
        self._kf = _KalmanTracker(
            q=config.kalman_process_noise,
            r=config.kalman_measurement_noise,
        )
        self._consecutive_misses = 0

    def plan(self, stabilized: StabilizedResult) -> PathPlan:
        """Compute a centered path through the traversable region."""
        mask = stabilized.mask
        h, w = mask.shape[:2]
        road_px = int(mask.sum())

        if road_px < self._min_road_px:
            return self._fallback(h, w)

        centerline, road_width = self._extract_centerline(mask, h, w)
        if len(centerline) < 2:
            return self._fallback(h, w)

        lateral_offset = centerline[-1, 0] - w / 2.0
        heading = self._estimate_heading(centerline)

        z = np.array([lateral_offset, heading])
        self._kf.predict()
        state = self._kf.update(z)
        self._consecutive_misses = 0

        return PathPlan(
            planner_mode="centerline",
            centerline=centerline,
            overlay_pixels=centerline,
            heading_rad=float(state[1]),
            curvature=float(state[2]),
            road_width_px=road_width,
            kalman_active=False,
            valid=True,
        )

    def _fallback(self, h: int, w: int) -> PathPlan:
        """Use Kalman prediction when the mask is absent or degraded."""
        self._consecutive_misses += 1
        state = self._kf.predict()

        if self._consecutive_misses > self._max_misses:
            logger.warning(
                "No valid road mask for %d frames - Kalman coasting",
                self._consecutive_misses,
            )

        cx = w / 2.0 + state[0]
        fallback_pts = np.array(
            [
                [cx, h * 0.9],
                [cx, h * 0.5],
                [cx, h * 0.1],
            ],
            dtype=np.float32,
        )

        return PathPlan(
            planner_mode="centerline",
            centerline=fallback_pts,
            overlay_pixels=fallback_pts,
            heading_rad=float(state[1]),
            curvature=float(state[2]),
            road_width_px=0.0,
            kalman_active=True,
            valid=True,
        )

    def _extract_centerline(
        self,
        mask: np.ndarray,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, float]:
        row_indices = np.linspace(int(h * 0.1), int(h * 0.95), self._n_samples, dtype=int)
        points: list[tuple[float, float]] = []
        widths: list[float] = []

        for y in row_indices:
            cols = np.where(mask[y, :])[0]
            if len(cols) < 2:
                continue
            left = float(cols[0])
            right = float(cols[-1])
            points.append(((left + right) / 2.0, float(y)))
            widths.append(right - left)

        if not points:
            return np.empty((0, 2), dtype=np.float32), 0.0

        centerline = np.array(points, dtype=np.float32)
        avg_width = float(np.mean(widths)) if widths else 0.0
        return centerline, avg_width

    @staticmethod
    def _estimate_heading(centerline: np.ndarray) -> float:
        if len(centerline) < 2:
            return 0.0
        pt_near = centerline[-1]
        pt_far = centerline[max(0, len(centerline) - max(2, len(centerline) // 3))]
        dx = pt_far[0] - pt_near[0]
        dy = pt_near[1] - pt_far[1]
        if dy < 1e-6:
            return 0.0
        return float(math.atan2(dx, dy))

    def reset(self) -> None:
        self._kf = _KalmanTracker(q=self._kf.Q[0, 0], r=self._kf.R[0, 0])
        self._consecutive_misses = 0
