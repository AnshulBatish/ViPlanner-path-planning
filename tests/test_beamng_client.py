"""Unit tests for BeamNG observation capture helpers."""

from unittest.mock import MagicMock

import numpy as np

from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import PipelineConfig


class _FakeColour:
    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def convert(self, mode: str):
        assert mode == "RGB"
        return self

    def __array__(self, dtype=None):
        return self._image


def test_capture_observation_returns_colour_depth_and_intrinsics():
    config = PipelineConfig(camera_width=640, camera_height=360, camera_fov=90.0)
    client = BeamNGClient(config)

    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    depth = np.full((360, 640), 6.0, dtype=np.float32)

    client._camera = MagicMock()
    client._camera.stream.return_value = {
        "colour": _FakeColour(rgb),
        "depth": depth,
    }

    observation = client.capture_observation()

    assert observation is not None
    assert observation.color_bgr.shape == (360, 640, 3)
    assert observation.depth_m.shape == (360, 640)
    assert observation.camera_intrinsics is not None
    assert observation.camera_intrinsics.width == 640
    assert observation.camera_intrinsics.height == 360
