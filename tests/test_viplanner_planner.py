"""Unit tests for the ViPlanner local planner."""

import numpy as np

from offroad_autonomy.planning.viplanner_planner import ViPlannerLocalPlanner
from offroad_autonomy.types import CameraIntrinsics, FramePacket, PipelineConfig, StabilizedResult, VehicleState


class _FakeAdapter:
    def plan(self, depth, semantic, goal_vehicle):
        assert depth.shape == (360, 640)
        assert semantic.shape == (360, 640, 3)
        assert goal_vehicle[0] >= 4.0
        return np.array([[1.0, 0.0, 0.0], [3.0, 0.4, 0.0], [5.0, 1.0, 0.0]], dtype=np.float32), 0.2


def _make_frame(depth_value: float) -> FramePacket:
    return FramePacket(
        raw=np.zeros((360, 640, 3), dtype=np.uint8),
        preprocessed=np.zeros((360, 640, 3), dtype=np.uint8),
        timestamp=0.0,
        height=360,
        width=640,
        depth_m=np.full((360, 640), depth_value, dtype=np.float32),
        camera_intrinsics=CameraIntrinsics(width=640, height=360, fx=120.0, fy=120.0, cx=320.0, cy=180.0),
    )


def test_viplanner_planner_generates_valid_plan():
    config = PipelineConfig(planning_mode="viplanner")
    planner = ViPlannerLocalPlanner(config, adapter=_FakeAdapter())
    mask = np.zeros((360, 640), dtype=bool)
    mask[90:300, 220:420] = True

    plan = planner.plan(_make_frame(6.0), StabilizedResult(mask=mask), VehicleState())

    assert plan.valid is True
    assert plan.planner_mode == "viplanner"
    assert len(plan.trajectory_vehicle) == 3
    assert len(plan.overlay_pixels) == 3
    assert plan.goal_vehicle is not None
    assert plan.goal_overlay_pixel is not None


def test_viplanner_planner_fails_when_no_valid_goal_depth_exists():
    config = PipelineConfig(planning_mode="viplanner")
    planner = ViPlannerLocalPlanner(config, adapter=_FakeAdapter())
    mask = np.zeros((360, 640), dtype=bool)
    mask[90:300, 220:420] = True

    plan = planner.plan(_make_frame(0.0), StabilizedResult(mask=mask), VehicleState())

    assert plan.valid is False
    assert "goal" in plan.failure_reason.lower()
