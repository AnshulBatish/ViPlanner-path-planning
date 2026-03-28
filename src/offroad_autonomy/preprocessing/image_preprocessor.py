"""Frame preprocessing for the perception pipeline."""

from __future__ import annotations

import time

import cv2
import numpy as np

from offroad_autonomy.types import CameraIntrinsics, FramePacket, PipelineConfig


class ImagePreprocessor:
    """Prepare raw camera frames for downstream perception."""

    def __init__(self, config: PipelineConfig) -> None:
        self._target_w = config.preprocess_width
        self._target_h = config.preprocess_height
        self._enable_clahe = config.enable_clahe
        self._clahe: cv2.CLAHE | None = None

        if self._enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
            )

    def process(
        self,
        raw_bgr: np.ndarray,
        raw_depth: np.ndarray | None = None,
        camera_intrinsics: CameraIntrinsics | None = None,
        camera_translation_vehicle: tuple[float, float, float] = (0.0, 0.0, 0.0),
        camera_rotation_camera_to_vehicle: np.ndarray | None = None,
    ) -> FramePacket:
        """Convert raw simulator images into a ``FramePacket``."""
        timestamp = time.perf_counter()
        raw_h, raw_w = raw_bgr.shape[:2]

        frame = raw_bgr
        if (raw_w, raw_h) != (self._target_w, self._target_h):
            frame = cv2.resize(
                frame,
                (self._target_w, self._target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        if self._clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        depth = None
        if raw_depth is not None:
            depth = raw_depth.astype(np.float32, copy=False)
            if depth.shape[:2] != (self._target_h, self._target_w):
                depth = cv2.resize(
                    depth,
                    (self._target_w, self._target_h),
                    interpolation=cv2.INTER_LINEAR,
                )

        scaled_intrinsics = self._scale_intrinsics(camera_intrinsics, raw_w, raw_h)
        rotation = (
            np.eye(3, dtype=np.float32)
            if camera_rotation_camera_to_vehicle is None
            else np.asarray(camera_rotation_camera_to_vehicle, dtype=np.float32)
        )

        return FramePacket(
            raw=raw_bgr,
            preprocessed=frame,
            timestamp=timestamp,
            height=self._target_h,
            width=self._target_w,
            depth_m=depth,
            camera_intrinsics=scaled_intrinsics,
            camera_translation_vehicle=tuple(camera_translation_vehicle),
            camera_rotation_camera_to_vehicle=rotation,
        )

    def _scale_intrinsics(
        self,
        intrinsics: CameraIntrinsics | None,
        raw_w: int,
        raw_h: int,
    ) -> CameraIntrinsics | None:
        if intrinsics is None:
            return None

        sx = self._target_w / max(raw_w, 1)
        sy = self._target_h / max(raw_h, 1)
        return CameraIntrinsics(
            width=self._target_w,
            height=self._target_h,
            fx=intrinsics.fx * sx,
            fy=intrinsics.fy * sy,
            cx=intrinsics.cx * sx,
            cy=intrinsics.cy * sy,
        )
