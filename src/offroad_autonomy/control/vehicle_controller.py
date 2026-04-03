"""Planner-aware vehicle controller implementations."""

from __future__ import annotations

import math

import numpy as np

from offroad_autonomy.types import ControlCommand, PathPlan, PipelineConfig, VehicleState


class VehicleController:
    """Use Stanley for centerlines and pure pursuit for ViPlanner trajectories."""

    _REVERSE_RECOVERY_SPEED_MPS = -0.1

    def __init__(self, config: PipelineConfig) -> None:
        self._k = config.stanley_gain_k
        self._k_soft = config.stanley_softening
        self._target_speed = config.target_speed_mps
        self._max_throttle = config.max_throttle
        self._max_brake = config.max_brake
        self._speed_kp = config.speed_kp
        self._frame_w = float(config.preprocess_width)
        self._lookahead_min = config.pure_pursuit_lookahead_min_m
        self._lookahead_gain = config.pure_pursuit_lookahead_gain
        self._wheelbase = config.pure_pursuit_wheelbase_m
        self._max_steer_cmd = config.pure_pursuit_max_steer_cmd

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        if plan.planner_mode == "viplanner":
            return self._compute_viplanner(plan, state)

        steering = self._stanley_lateral_control(plan, state)
        throttle, brake, gear = self._longitudinal_control(state)
        return ControlCommand(
            steering=float(np.clip(steering, -1.0, 1.0)),
            throttle=throttle,
            brake=brake,
            gear=gear,
        )

    def _compute_viplanner(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        if not plan.valid or len(plan.trajectory_vehicle) < 2:
            return ControlCommand(steering=0.0, throttle=0.0, brake=self._max_brake, gear=1)

        steering = self._pure_pursuit_lateral_control(plan, state)
        throttle, brake, gear = self._longitudinal_control(state)
        return ControlCommand(
            steering=float(np.clip(steering, -1.0, 1.0)),
            throttle=throttle,
            brake=brake,
            gear=gear,
        )

    def _stanley_lateral_control(self, plan: PathPlan, state: VehicleState) -> float:
        heading_err = plan.heading_rad
        if len(plan.centerline) > 0:
            cx_bottom = plan.centerline[-1, 0]
            cte = (cx_bottom - self._frame_w / 2.0) / (self._frame_w / 2.0)
        else:
            cte = 0.0

        speed = max(state.speed_mps, 0.1)
        stanley_term = math.atan2(self._k * cte, speed + self._k_soft)
        return heading_err + stanley_term

    def _pure_pursuit_lateral_control(self, plan: PathPlan, state: VehicleState) -> float:
        speed = max(state.forward_speed_mps, 0.0)
        lookahead = max(self._lookahead_min, self._lookahead_min + self._lookahead_gain * speed)
        target = self._select_lookahead_point(plan.trajectory_vehicle, lookahead)
        if target is None:
            return 0.0

        alpha = math.atan2(target[1], max(target[0], 1e-6))
        steering = math.atan2(2.0 * self._wheelbase * math.sin(alpha), max(lookahead, 1e-6))
        return float(np.clip(steering, -self._max_steer_cmd, self._max_steer_cmd))

    @staticmethod
    def _select_lookahead_point(trajectory_vehicle: np.ndarray, lookahead: float) -> np.ndarray | None:
        if len(trajectory_vehicle) == 0:
            return None

        for point in trajectory_vehicle:
            if point[0] <= 0.0:
                continue
            if np.linalg.norm(point[:2]) >= lookahead:
                return point

        forward_points = trajectory_vehicle[trajectory_vehicle[:, 0] > 0.0]
        if len(forward_points) == 0:
            return None
        return forward_points[-1]

    def _longitudinal_control(self, state: VehicleState) -> tuple[float, float, int]:
        if state.forward_speed_mps < self._REVERSE_RECOVERY_SPEED_MPS:
            return 0.0, self._max_brake, 1

        speed_err = self._target_speed - state.forward_speed_mps
        if speed_err > 0:
            throttle = min(self._speed_kp * speed_err, self._max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self._speed_kp * abs(speed_err), self._max_brake)
        return throttle, brake, 1
