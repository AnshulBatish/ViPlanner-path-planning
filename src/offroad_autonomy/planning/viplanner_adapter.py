"""Planner adapter for upstream ViPlanner inference."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np

from offroad_autonomy.types import PipelineConfig


class _UpstreamViPlannerBackend:
    """Minimal upstream-compatible inference wrapper."""

    def __init__(self, model_dir: str) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
            from viplanner.config.learning_cfg import TrainCfg
            from viplanner.plannernet import AutoEncoder, DualAutoEncoder
            from viplanner.traj_cost_opt.traj_opt import TrajOpt
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "ViPlanner is not installed. Install the upstream package before using planning.mode='viplanner'."
            ) from exc

        model_path = os.path.join(model_dir, "model.pt")
        config_path = os.path.join(model_dir, "model.yaml")
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise RuntimeError(
                f"ViPlanner checkpoint directory must contain model.pt and model.yaml: {model_dir}"
            )

        self._torch = torch
        self._transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((360, 640)),
            ]
        )
        self.train_cfg = TrainCfg.from_yaml(config_path)
        if self.train_cfg.rgb:
            raise RuntimeError(
                "This integration expects a depth+semantic ViPlanner checkpoint, not an RGB-only checkpoint."
            )

        if self.train_cfg.sem:
            self._net = DualAutoEncoder(train_cfg=self.train_cfg, m2f_cfg=None)
        else:
            self._net = AutoEncoder(
                encoder_channel=self.train_cfg.in_channel,
                k=self.train_cfg.knodes,
            )

        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, tuple):
            state_dict = state[0]
        else:
            state_dict = state
        self._net.load_state_dict(state_dict)
        self._net.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._net = self._net.to(self._device)
        self._traj_opt = TrajOpt()

    def _to_tensor(self, image: np.ndarray):
        tensor = self._transforms(image).float().unsqueeze(0)
        return tensor.to(self._device)

    def plan(self, depth_image: np.ndarray, semantic_image: np.ndarray, goal_input):
        with self._torch.no_grad():
            depth = self._to_tensor(depth_image)
            goal = goal_input.to(self._device)
            if self.train_cfg.sem:
                semantic = self._to_tensor(semantic_image.astype(np.uint8))
                keypoints, fear = self._net(depth, semantic, goal)
            else:
                keypoints, fear = self._net(depth, goal)

            trajectory = self._traj_opt.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return trajectory.cpu().squeeze(0).numpy(), fear.cpu().numpy()


class ViPlannerAdapter:
    """Thin adapter around the upstream ViPlanner runtime."""

    def __init__(
        self,
        config: PipelineConfig,
        inference_factory: Callable[[PipelineConfig], object] | None = None,
    ) -> None:
        self._config = config
        self._input_size = (config.viplanner_input_width, config.viplanner_input_height)
        factory = inference_factory or self._build_backend
        self._backend = factory(config)

    def _build_backend(self, config: PipelineConfig):
        return _UpstreamViPlannerBackend(config.viplanner_model_dir)

    def plan(
        self,
        depth_image: np.ndarray,
        semantic_image: np.ndarray,
        goal_vehicle: tuple[float, float, float],
    ) -> tuple[np.ndarray, float]:
        depth = self._prepare_depth(depth_image)
        semantic = self._prepare_semantic(semantic_image)
        goal = self._make_goal_input(goal_vehicle)
        trajectory, fear = self._backend.plan(depth, semantic, goal)

        trajectory_np = np.asarray(trajectory, dtype=np.float32)
        if trajectory_np.ndim == 1:
            trajectory_np = trajectory_np.reshape(-1, 3)
        if trajectory_np.shape[-1] == 2:
            trajectory_np = np.concatenate(
                [trajectory_np, np.zeros((len(trajectory_np), 1), dtype=np.float32)],
                axis=1,
            )

        return trajectory_np, float(np.asarray(fear).squeeze())

    def _prepare_depth(self, depth_image: np.ndarray) -> np.ndarray:
        depth = np.nan_to_num(depth_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth, 0.0, self._config.viplanner_max_depth_m)
        if depth.shape[:2] != self._input_size[::-1]:
            depth = cv2.resize(depth, self._input_size, interpolation=cv2.INTER_LINEAR)
        return depth

    def _prepare_semantic(self, semantic_image: np.ndarray) -> np.ndarray:
        semantic = semantic_image.astype(np.uint8)
        if semantic.shape[:2] != self._input_size[::-1]:
            semantic = cv2.resize(semantic, self._input_size, interpolation=cv2.INTER_NEAREST)
        return semantic

    @staticmethod
    def _make_goal_input(goal_vehicle: tuple[float, float, float]):
        import torch

        return torch.tensor(goal_vehicle, dtype=torch.float32)[None, ...]
