"""Unit tests for the pipeline with mocked stages."""

from unittest.mock import patch

import numpy as np

from offroad_autonomy.types import (
    CameraIntrinsics,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    PipelineStepResult,
    StabilizedResult,
    VehicleState,
)


def _make_config(mode: str = "centerline") -> PipelineConfig:
    return PipelineConfig(
        planning_mode=mode,
        model_weights="dummy.pt",
        preprocess_width=640,
        preprocess_height=360,
    )


def _make_frame_packet(frame: np.ndarray) -> FramePacket:
    return FramePacket(
        raw=frame,
        preprocessed=frame,
        timestamp=0.0,
        height=360,
        width=640,
        depth_m=np.ones((360, 640), dtype=np.float32),
        camera_intrinsics=CameraIntrinsics(width=640, height=360, fx=100.0, fy=100.0, cx=320.0, cy=180.0),
    )


def test_pipeline_step_produces_control_command_centerline():
    config = _make_config("centerline")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    mask = np.zeros((360, 640), dtype=bool)
    mask[200:300, 200:440] = True

    mock_frame_packet = _make_frame_packet(frame)
    mock_perception = PerceptionResult(mask=mask, confidences=[0.9], num_detections=1, inference_time_ms=15.0)
    mock_stabilized = StabilizedResult(mask=mask, stability_score=0.95)
    mock_plan = PathPlan(planner_mode="centerline", centerline=np.array([[320.0, 300.0], [320.0, 100.0]], dtype=np.float32))
    mock_command = ControlCommand(steering=0.0, throttle=0.3, brake=0.0)

    with (
        patch("offroad_autonomy.pipeline.ImagePreprocessor") as MockPre,
        patch("offroad_autonomy.pipeline.RoadSegmenter") as MockSeg,
        patch("offroad_autonomy.pipeline.TemporalStabilizer") as MockStab,
        patch("offroad_autonomy.pipeline.CenterlinePlanner") as MockPlan,
        patch("offroad_autonomy.pipeline.VehicleController") as MockCtrl,
    ):
        MockPre.return_value.process.return_value = mock_frame_packet
        MockSeg.return_value.predict.return_value = mock_perception
        MockStab.return_value.stabilize.return_value = mock_stabilized
        MockPlan.return_value.plan.return_value = mock_plan
        MockCtrl.return_value.compute.return_value = mock_command

        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)
        result = pipeline.step(frame, VehicleState())

        assert isinstance(result, ControlCommand)
        assert result.throttle == 0.3
        MockPlan.return_value.plan.assert_called_once_with(mock_stabilized)


def test_pipeline_step_result_exposes_stage_outputs_viplanner():
    config = _make_config("viplanner")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    mask = np.zeros((360, 640), dtype=bool)
    mask[80:300, 200:440] = True

    mock_frame_packet = _make_frame_packet(frame)
    mock_perception = PerceptionResult(mask=mask, confidences=[0.9], num_detections=1, inference_time_ms=15.0)
    mock_stabilized = StabilizedResult(mask=mask, stability_score=0.95)
    mock_plan = PathPlan(planner_mode="viplanner", valid=True, trajectory_vehicle=np.array([[1.0, 0.0, 0.0], [2.0, 0.2, 0.0]], dtype=np.float32))
    mock_command = ControlCommand(steering=0.1, throttle=0.3, brake=0.0)

    with (
        patch("offroad_autonomy.pipeline.ImagePreprocessor") as MockPre,
        patch("offroad_autonomy.pipeline.RoadSegmenter") as MockSeg,
        patch("offroad_autonomy.pipeline.TemporalStabilizer") as MockStab,
        patch("offroad_autonomy.pipeline.ViPlannerLocalPlanner") as MockPlan,
        patch("offroad_autonomy.pipeline.VehicleController") as MockCtrl,
    ):
        MockPre.return_value.process.return_value = mock_frame_packet
        MockSeg.return_value.predict.return_value = mock_perception
        MockStab.return_value.stabilize.return_value = mock_stabilized
        MockPlan.return_value.plan.return_value = mock_plan
        MockCtrl.return_value.compute.return_value = mock_command

        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)
        result = pipeline.step_result(frame, VehicleState())

        assert isinstance(result, PipelineStepResult)
        assert result.frame is mock_frame_packet
        assert result.perception is mock_perception
        assert result.stabilized is mock_stabilized
        assert result.plan is mock_plan
        assert result.command is mock_command
        MockPlan.return_value.plan.assert_called_once_with(mock_frame_packet, mock_stabilized, VehicleState())
