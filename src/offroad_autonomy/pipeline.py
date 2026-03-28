"""Pipeline orchestration for the offroad_autonomy stack."""

from __future__ import annotations

import logging

import numpy as np

from offroad_autonomy.control.vehicle_controller import VehicleController
from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.planning.viplanner_planner import ViPlannerLocalPlanner
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.preprocessing.image_preprocessor import ImagePreprocessor
from offroad_autonomy.types import CameraIntrinsics, ControlCommand, PipelineConfig, PipelineStepResult, VehicleState

logger = logging.getLogger("offroad_autonomy.pipeline")


class AutonomyPipeline:
    """Single-step orchestrator for the full autonomy stack."""

    def __init__(self, config: PipelineConfig) -> None:
        logger.info("Initialising pipeline stages")
        self._planning_mode = config.planning_mode
        self.preprocessor = ImagePreprocessor(config)
        self.segmenter = RoadSegmenter(config)
        self.stabilizer = TemporalStabilizer(config)
        self.centerline_planner = CenterlinePlanner(config)
        self.viplanner_planner = (
            ViPlannerLocalPlanner(config) if config.planning_mode == "viplanner" else None
        )
        self.controller = VehicleController(config)

    def step(
        self,
        raw_frame: np.ndarray,
        vehicle_state: VehicleState,
        depth_frame: np.ndarray | None = None,
        camera_intrinsics: CameraIntrinsics | None = None,
        camera_translation_vehicle: tuple[float, float, float] = (0.0, 0.0, 0.0),
        camera_rotation_camera_to_vehicle: np.ndarray | None = None,
    ) -> ControlCommand:
        return self.step_result(
            raw_frame,
            vehicle_state,
            depth_frame=depth_frame,
            camera_intrinsics=camera_intrinsics,
            camera_translation_vehicle=camera_translation_vehicle,
            camera_rotation_camera_to_vehicle=camera_rotation_camera_to_vehicle,
        ).command

    def step_result(
        self,
        raw_frame: np.ndarray,
        vehicle_state: VehicleState,
        depth_frame: np.ndarray | None = None,
        camera_intrinsics: CameraIntrinsics | None = None,
        camera_translation_vehicle: tuple[float, float, float] = (0.0, 0.0, 0.0),
        camera_rotation_camera_to_vehicle: np.ndarray | None = None,
    ) -> PipelineStepResult:
        frame = self.preprocessor.process(
            raw_frame,
            raw_depth=depth_frame,
            camera_intrinsics=camera_intrinsics,
            camera_translation_vehicle=camera_translation_vehicle,
            camera_rotation_camera_to_vehicle=camera_rotation_camera_to_vehicle,
        )
        perception = self.segmenter.predict(frame)
        stabilized = self.stabilizer.stabilize(perception)
        if self._planning_mode == "viplanner":
            planner = self.viplanner_planner
            assert planner is not None
            plan = planner.plan(frame, stabilized, vehicle_state)
        else:
            plan = self.centerline_planner.plan(stabilized)
        command = self.controller.compute(plan, vehicle_state)

        logger.debug(
            "step: infer=%.0fms stability=%.3f planner=%s valid=%s steer=%.3f",
            perception.inference_time_ms,
            stabilized.stability_score,
            plan.planner_mode,
            plan.valid,
            command.steering,
        )

        return PipelineStepResult(
            frame=frame,
            perception=perception,
            stabilized=stabilized,
            plan=plan,
            command=command,
        )

    def reset(self) -> None:
        self.stabilizer.reset()
        self.centerline_planner.reset()
        if self.viplanner_planner is not None:
            self.viplanner_planner.reset()
