"""Local ViPlanner integration for BeamNG."""

from __future__ import annotations

import logging
import math

import numpy as np

from offroad_autonomy.planning.viplanner_adapter import ViPlannerAdapter
from offroad_autonomy.types import FramePacket, PathPlan, PipelineConfig, StabilizedResult, VehicleState

logger = logging.getLogger("offroad_autonomy.planning")


class ViPlannerLocalPlanner:
    """Generate a local trajectory from depth, semantics, and an auto goal."""

    def __init__(self, config: PipelineConfig, adapter: ViPlannerAdapter | None = None) -> None:
        self._config = config
        self._adapter = adapter or ViPlannerAdapter(config)

    def plan(
        self,
        frame: FramePacket,
        stabilized: StabilizedResult,
        vehicle_state: VehicleState,
    ) -> PathPlan:
        if frame.depth_m is None or frame.camera_intrinsics is None:
            return self._invalid_plan("Depth or camera intrinsics unavailable for ViPlanner.")

        goal = self._select_goal(frame, stabilized.mask)
        if goal is None:
            return self._invalid_plan("Unable to recover a valid local goal from the road mask.")

        goal_vehicle, goal_pixel = goal
        semantic_proxy = self._mask_to_semantic(stabilized.mask)

        try:
            trajectory_camera, fear_score = self._adapter.plan(
                frame.depth_m,
                semantic_proxy,
                goal_vehicle,
            )
        except Exception as exc:
            logger.warning("ViPlanner inference failed: %s", exc)
            return self._invalid_plan(str(exc))

        if len(trajectory_camera) < 2:
            return self._invalid_plan("ViPlanner returned an empty trajectory.")

        trajectory_vehicle = self._camera_to_vehicle(trajectory_camera, frame)
        overlay_pixels = self._project_vehicle_to_image(trajectory_vehicle, frame)
        trajectory_world = self._vehicle_to_world(trajectory_vehicle, vehicle_state)
        heading = self._estimate_heading(trajectory_vehicle)
        curvature = self._estimate_curvature(trajectory_vehicle)

        return PathPlan(
            planner_mode="viplanner",
            trajectory_vehicle=trajectory_vehicle,
            trajectory_world=trajectory_world,
            overlay_pixels=overlay_pixels,
            goal_vehicle=tuple(float(v) for v in goal_vehicle),
            goal_overlay_pixel=goal_pixel,
            fear_score=fear_score,
            valid=True,
            centerline=overlay_pixels,
            heading_rad=heading,
            curvature=curvature,
        )

    def reset(self) -> None:
        """Stateless planner hook for API symmetry."""

    def _invalid_plan(self, reason: str) -> PathPlan:
        return PathPlan(
            planner_mode="viplanner",
            valid=False,
            failure_reason=reason,
            fear_score=1.0,
        )

    @staticmethod
    def _mask_to_semantic(mask: np.ndarray) -> np.ndarray:
        semantic = (mask.astype(np.uint8) * 255)[:, :, None]
        return np.repeat(semantic, 3, axis=2)

    def _select_goal(
        self,
        frame: FramePacket,
        mask: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float]] | None:
        h, _ = mask.shape[:2]
        row_start = int(h * self._config.auto_goal_min_row_frac)
        row_end = int(h * self._config.auto_goal_max_row_frac)

        for y in range(row_start, max(row_end, row_start + 1)):
            segments = self._segments(mask[y, :])
            if not segments:
                continue
            segment = max(segments, key=lambda item: item[1] - item[0])
            x = int(round((segment[0] + segment[1]) / 2.0))
            point = self._recover_goal_vehicle(frame, x, y)
            if point is None:
                continue
            point = (max(point[0], self._config.auto_goal_fallback_forward_m), point[1], point[2])
            return point, (float(x), float(y))

        return None

    @staticmethod
    def _segments(row: np.ndarray) -> list[tuple[int, int]]:
        cols = np.where(row)[0]
        if len(cols) < 2:
            return []
        splits = np.where(np.diff(cols) > 1)[0] + 1
        groups = np.split(cols, splits)
        return [(int(group[0]), int(group[-1])) for group in groups if len(group) >= 2]

    def _recover_goal_vehicle(self, frame: FramePacket, x: int, y: int) -> tuple[float, float, float] | None:
        depth = frame.depth_m
        if depth is None or frame.camera_intrinsics is None:
            return None

        h, w = depth.shape[:2]
        for dy in range(self._config.auto_goal_search_margin_px + 1):
            yy = min(y + dy, h - 1)
            for dx in (-2, -1, 0, 1, 2):
                xx = int(np.clip(x + dx, 0, w - 1))
                depth_value = float(depth[yy, xx])
                if not np.isfinite(depth_value) or depth_value <= 1e-3:
                    continue
                camera_point = self._pixel_to_camera(frame, xx, yy, depth_value)
                vehicle_point = self._camera_to_vehicle(camera_point[None, :], frame)[0]
                return tuple(float(v) for v in vehicle_point)
        return None

    @staticmethod
    def _pixel_to_camera(frame: FramePacket, x: int, y: int, depth_m: float) -> np.ndarray:
        intr = frame.camera_intrinsics
        assert intr is not None
        x_forward = depth_m
        y_left = -((x - intr.cx) * depth_m / max(intr.fx, 1e-6))
        z_up = -((y - intr.cy) * depth_m / max(intr.fy, 1e-6))
        return np.array([x_forward, y_left, z_up], dtype=np.float32)

    @staticmethod
    def _camera_to_vehicle(points_camera: np.ndarray, frame: FramePacket) -> np.ndarray:
        rotation = np.asarray(frame.camera_rotation_camera_to_vehicle, dtype=np.float32)
        translation = np.asarray(frame.camera_translation_vehicle, dtype=np.float32)
        return (rotation @ points_camera.T).T + translation

    @staticmethod
    def _vehicle_to_camera(points_vehicle: np.ndarray, frame: FramePacket) -> np.ndarray:
        rotation = np.asarray(frame.camera_rotation_camera_to_vehicle, dtype=np.float32)
        translation = np.asarray(frame.camera_translation_vehicle, dtype=np.float32)
        return (rotation.T @ (points_vehicle - translation).T).T

    def _project_vehicle_to_image(self, points_vehicle: np.ndarray, frame: FramePacket) -> np.ndarray:
        intr = frame.camera_intrinsics
        if intr is None or len(points_vehicle) == 0:
            return np.empty((0, 2), dtype=np.float32)

        points_camera = self._vehicle_to_camera(points_vehicle, frame)
        valid = points_camera[:, 0] > 1e-3
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float32)

        points_camera = points_camera[valid]
        u = intr.cx - (points_camera[:, 1] * intr.fx / points_camera[:, 0])
        v = intr.cy - (points_camera[:, 2] * intr.fy / points_camera[:, 0])
        return np.stack([u, v], axis=1).astype(np.float32)

    @staticmethod
    def _vehicle_to_world(points_vehicle: np.ndarray, state: VehicleState) -> np.ndarray:
        if len(points_vehicle) == 0:
            return np.empty((0, 3), dtype=np.float32)
        c = math.cos(state.heading_rad)
        s = math.sin(state.heading_rad)
        world = np.empty_like(points_vehicle, dtype=np.float32)
        world[:, 0] = state.position[0] + c * points_vehicle[:, 0] - s * points_vehicle[:, 1]
        world[:, 1] = state.position[1] + s * points_vehicle[:, 0] + c * points_vehicle[:, 1]
        world[:, 2] = state.position[2] + points_vehicle[:, 2]
        return world

    @staticmethod
    def _estimate_heading(points_vehicle: np.ndarray) -> float:
        if len(points_vehicle) < 2:
            return 0.0
        delta = points_vehicle[min(1, len(points_vehicle) - 1)] - points_vehicle[0]
        return float(math.atan2(delta[1], max(delta[0], 1e-6)))

    @staticmethod
    def _estimate_curvature(points_vehicle: np.ndarray) -> float:
        if len(points_vehicle) < 3:
            return 0.0
        a = points_vehicle[1] - points_vehicle[0]
        b = points_vehicle[2] - points_vehicle[1]
        angle_a = math.atan2(a[1], max(a[0], 1e-6))
        angle_b = math.atan2(b[1], max(b[0], 1e-6))
        return float(angle_b - angle_a)
