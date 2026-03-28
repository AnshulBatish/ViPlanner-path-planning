"""Unit tests for the ViPlanner adapter."""

import numpy as np

from offroad_autonomy.planning.viplanner_adapter import ViPlannerAdapter
from offroad_autonomy.types import PipelineConfig


class _FakeBackend:
    def __init__(self) -> None:
        self.calls = []

    def plan(self, depth, semantic, goal):
        self.calls.append((depth, semantic, goal))
        return np.array([[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]], dtype=np.float32), np.array([0.25], dtype=np.float32)


def test_adapter_prepares_inputs_and_returns_numpy_outputs():
    backend = _FakeBackend()
    config = PipelineConfig(viplanner_input_width=640, viplanner_input_height=360)
    adapter = ViPlannerAdapter(config, inference_factory=lambda cfg: backend)

    depth = np.full((180, 320), 5.0, dtype=np.float32)
    semantic = np.ones((180, 320, 3), dtype=np.uint8) * 255
    trajectory, fear = adapter.plan(depth, semantic, (4.0, 0.5, 0.0))

    assert trajectory.shape == (2, 3)
    assert fear == 0.25
    depth_in, semantic_in, goal_in = backend.calls[0]
    assert depth_in.shape == (360, 640)
    assert semantic_in.shape == (360, 640, 3)
    assert tuple(goal_in.shape) == (1, 3)
