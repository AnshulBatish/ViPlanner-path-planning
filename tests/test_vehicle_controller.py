"""Unit tests for the planner-aware vehicle controller."""

import numpy as np

from offroad_autonomy.control.vehicle_controller import VehicleController
from offroad_autonomy.types import PathPlan, PipelineConfig, VehicleState


def test_pure_pursuit_tracks_straight_trajectory():
    controller = VehicleController(PipelineConfig())
    plan = PathPlan(
        planner_mode="viplanner",
        valid=True,
        trajectory_vehicle=np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32),
    )

    command = controller.compute(plan, VehicleState(speed_mps=2.0))

    assert abs(command.steering) < 1e-6
    assert command.throttle > 0.0
    assert command.brake == 0.0


def test_pure_pursuit_turns_toward_left_offset_waypoints():
    controller = VehicleController(PipelineConfig())
    plan = PathPlan(
        planner_mode="viplanner",
        valid=True,
        trajectory_vehicle=np.array([[1.0, 0.1, 0.0], [3.0, 0.6, 0.0], [5.0, 1.2, 0.0]], dtype=np.float32),
    )

    command = controller.compute(plan, VehicleState(speed_mps=2.0))

    assert command.steering > 0.0


def test_invalid_viplanner_plan_triggers_brake_failsafe():
    config = PipelineConfig(max_brake=0.8)
    controller = VehicleController(config)
    command = controller.compute(PathPlan(planner_mode="viplanner", valid=False), VehicleState(speed_mps=4.0))

    assert command.throttle == 0.0
    assert command.brake == 0.8
