"""BeamNG.tech simulator client.

Encapsulates all BeamNG-specific I/O: launching the simulator, loading
a scenario, capturing camera frames, polling vehicle state, and sending
control commands. Nothing outside this module should import ``beamngpy``.
"""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np

from offroad_autonomy.types import (
    CameraIntrinsics,
    ControlCommand,
    PipelineConfig,
    SimulationObservation,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.simulation")


class BeamNGClient:
    """Manages the BeamNG.tech session lifecycle and sensor I/O."""

    _BEAMNG_TO_VEHICLE = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._bng = None
        self._vehicle = None
        self._camera = None
        self._camera_intrinsics = self._build_camera_intrinsics(config)
        self._camera_translation_vehicle = tuple(
            float(v) for v in self._beamng_to_vehicle_coords(config.camera_pos)
        )
        self._camera_rotation_camera_to_vehicle = self._build_camera_rotation(config.camera_dir)

    def connect(self) -> None:
        """Launch BeamNG, load the configured map, spawn the vehicle, and attach a camera."""
        from beamngpy import BeamNGpy, Scenario, Vehicle
        from beamngpy.sensors.camera import Camera

        cfg = self._config

        logger.info("Launching BeamNG from %s", cfg.beamng_home)
        self._bng = BeamNGpy(cfg.beamng_host, cfg.beamng_port, home=cfg.beamng_home)
        self._bng.open(launch=True)

        spawn_pos, spawn_rot = self._resolve_spawn(cfg)

        logger.info("Loading map '%s'", cfg.beamng_map)
        scenario = Scenario(cfg.beamng_map, "offroad_autonomy")
        vehicle = Vehicle("ego", model=cfg.beamng_vehicle, licence="OFFROAD")
        scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot, cling=True)

        scenario.make(self._bng)
        self._bng.scenario.load(scenario)
        self._bng.scenario.start()
        vehicle.connect(self._bng)

        logger.info("Waiting for physics to settle")
        time.sleep(3.0)

        logger.info("Attaching front camera")
        self._camera = Camera(
            name="front_cam",
            bng=self._bng,
            vehicle=vehicle,
            pos=tuple(cfg.camera_pos),
            dir=tuple(cfg.camera_dir),
            up=(0, 0, 1),
            resolution=(cfg.camera_width, cfg.camera_height),
            field_of_view_y=cfg.camera_fov,
            near_far_planes=(0.1, 500.0),
            requested_update_time=0.01,
            update_priority=1.0,
            is_render_colours=True,
            is_render_annotations=False,
            is_render_depth=cfg.beamng_render_depth,
            is_using_shared_memory=True,
            is_streaming=True,
        )
        time.sleep(1.0)

        self._vehicle = vehicle
        logger.info("BeamNG session ready")

    def capture_observation(self) -> SimulationObservation | None:
        """Grab the latest colour and depth images from the front camera."""
        if self._camera is None:
            return None

        try:
            images = self._camera.stream()
            color = self._decode_colour(images.get("colour"))
            if color is None:
                return None

            depth = self._decode_depth(images.get("depth"))
            return SimulationObservation(
                color_bgr=color,
                depth_m=depth,
                camera_intrinsics=self._camera_intrinsics,
                camera_translation_vehicle=self._camera_translation_vehicle,
                camera_rotation_camera_to_vehicle=self._camera_rotation_camera_to_vehicle,
            )
        except Exception as exc:  # pragma: no cover - depends on BeamNG runtime
            logger.debug("Observation capture failed: %s", exc)
            return None

    def capture_frame(self) -> np.ndarray | None:
        """Backwards-compatible colour-only frame capture."""
        observation = self.capture_observation()
        return observation.color_bgr if observation is not None else None

    def get_vehicle_state(self) -> VehicleState:
        """Poll the latest vehicle telemetry."""
        if self._vehicle is None:
            return VehicleState()

        try:
            self._vehicle.sensors.poll()
            st = self._vehicle.state
            pos = tuple(st.get("pos", (0, 0, 0)))
            rot = tuple(st.get("rotation", (0, 0, 0, 1)))
            vel = tuple(st.get("vel", (0, 0, 0)))
            direction = np.asarray(st.get("dir", (0, -1, 0)), dtype=np.float32)
            speed = math.sqrt(sum(v ** 2 for v in vel))
            forward_speed = self._forward_speed(vel, direction)

            _, _, yaw = self._quat_to_euler(rot)

            return VehicleState(
                position=pos,
                rotation=rot,
                velocity=vel,
                speed_mps=speed,
                forward_speed_mps=forward_speed,
                heading_rad=yaw,
            )
        except Exception as exc:  # pragma: no cover - depends on BeamNG runtime
            logger.debug("State poll failed: %s", exc)
            return VehicleState()

    def send_controls(self, cmd: ControlCommand) -> None:
        """Send steering, throttle, and brake to the vehicle."""
        if self._vehicle is None:
            return
        self._vehicle.control(
            steering=cmd.steering,
            throttle=cmd.throttle,
            brake=cmd.brake,
            gear=cmd.gear,
        )

    def disconnect(self) -> None:
        """Tear down the session gracefully."""
        if self._camera is not None:
            try:
                self._camera.remove()
            except Exception:
                pass
            self._camera = None

        if self._bng is not None:
            try:
                self._bng.close()
            except Exception:
                pass
            self._bng = None

        self._vehicle = None
        logger.info("BeamNG session closed")

    @staticmethod
    def _build_camera_intrinsics(config: PipelineConfig) -> CameraIntrinsics:
        half_fov = math.radians(config.camera_fov) / 2.0
        fy = config.camera_height / max(2.0 * math.tan(half_fov), 1e-6)
        fx = fy
        return CameraIntrinsics(
            width=config.camera_width,
            height=config.camera_height,
            fx=fx,
            fy=fy,
            cx=config.camera_width / 2.0,
            cy=config.camera_height / 2.0,
        )

    @staticmethod
    def _decode_colour(colour) -> np.ndarray | None:
        if colour is None:
            return None
        if isinstance(colour, np.ndarray):
            image = colour
        else:
            image = np.array(colour.convert("RGB"))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image

    @classmethod
    def _beamng_to_vehicle_coords(cls, vector: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
        raw = np.asarray(vector, dtype=np.float32)
        return cls._BEAMNG_TO_VEHICLE @ raw

    @classmethod
    def _build_camera_rotation(cls, camera_dir: list[float]) -> np.ndarray:
        forward = cls._beamng_to_vehicle_coords(camera_dir)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-6:
            return np.eye(3, dtype=np.float32)
        forward = forward / forward_norm

        up_hint = cls._beamng_to_vehicle_coords((0.0, 0.0, 1.0))
        left = np.cross(up_hint, forward)
        left_norm = float(np.linalg.norm(left))
        if left_norm <= 1e-6:
            left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            left = left / left_norm

        up = np.cross(forward, left)
        up_norm = float(np.linalg.norm(up))
        if up_norm <= 1e-6:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up = up / up_norm

        return np.stack([forward, left, up], axis=1).astype(np.float32)

    @staticmethod
    def _forward_speed(velocity: tuple[float, float, float], direction: np.ndarray) -> float:
        direction_xy = np.asarray(direction[:2], dtype=np.float32)
        norm = float(np.linalg.norm(direction_xy))
        if norm <= 1e-6:
            return 0.0
        direction_xy = direction_xy / norm
        velocity_xy = np.asarray(velocity[:2], dtype=np.float32)
        return float(np.dot(velocity_xy, direction_xy))

    @staticmethod
    def _decode_depth(depth) -> np.ndarray | None:
        if depth is None:
            return None
        if isinstance(depth, np.ndarray):
            image = depth.astype(np.float32)
        else:
            image = np.array(depth).astype(np.float32)
        if image.ndim == 3:
            image = image[:, :, 0]
        if image.dtype.kind in {"u", "i"} and float(image.max(initial=0.0)) > 255.0:
            image = image / 1000.0
        return image

    def _resolve_spawn(self, cfg: PipelineConfig) -> tuple[tuple, tuple]:
        """Look up spawn position/rotation from the config map table."""
        map_cfg = cfg.map_spawns.get(cfg.beamng_map)
        if map_cfg and "spawns" in map_cfg:
            spawns = map_cfg["spawns"]
            idx = min(cfg.beamng_spawn_index, len(spawns) - 1)
            s = spawns[idx]
            pos = tuple(s.get("pos", [0, 0, 0]))
            rot = tuple(s.get("rot", [0, 0, 0, 1]))
            logger.info("Spawn #%d: pos=%s", idx, pos)
            return pos, rot

        logger.warning("No spawn data for map '%s' - using origin", cfg.beamng_map)
        return (0, 0, 0), (0, 0, 0, 1)

    @staticmethod
    def _quat_to_euler(q: tuple) -> tuple[float, float, float]:
        """Convert quaternion (x, y, z, w) to Euler (roll, pitch, yaw)."""
        x, y, z, w = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
