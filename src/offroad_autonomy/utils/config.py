"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from offroad_autonomy.types import (
    DEFAULT_DASHBOARD_COLORS,
    DEFAULT_PERCEPTION_PROMPTS,
    PipelineConfig,
)


def _parse_color(raw: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return default
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return default


def _load_dashboard_colors(raw: object) -> dict[str, tuple[int, int, int]]:
    colors = DEFAULT_DASHBOARD_COLORS.copy()
    if not isinstance(raw, dict):
        return colors

    for key, default in DEFAULT_DASHBOARD_COLORS.items():
        colors[key] = _parse_color(raw.get(key), default)
    return colors


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file into a ``PipelineConfig``."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    bng = raw.get("beamng", {})
    cam = bng.get("camera", {})
    perc = raw.get("perception", {})
    pre = raw.get("preprocessing", {})
    post = raw.get("postprocessing", {})
    plan = raw.get("planning", {})
    viplanner = raw.get("viplanner", {})
    auto_goal = raw.get("auto_goal", {})
    pure_pursuit = raw.get("pure_pursuit", {})
    ctrl = raw.get("control", {})
    viz = raw.get("visualization", {})
    dashboard = viz.get("dashboard", {})

    return PipelineConfig(
        beamng_home=bng.get("home", ""),
        beamng_host=bng.get("host", "localhost"),
        beamng_port=bng.get("port", 64256),
        beamng_map=bng.get("map", "automation_test_track"),
        beamng_vehicle=bng.get("vehicle", "pickup"),
        beamng_spawn_index=bng.get("spawn_index", 0),
        camera_width=cam.get("width", 1280),
        camera_height=cam.get("height", 720),
        camera_fov=cam.get("fov", 120.0),
        camera_pos=cam.get("pos", [0, -2.5, 0.8]),
        camera_dir=cam.get("dir", [0, -1, -0.1]),
        beamng_render_depth=cam.get("render_depth", True),
        map_spawns=bng.get("maps", {}),
        model_weights=perc.get("model_weights", "models/yoloe-26x-seg.pt"),
        confidence_threshold=perc.get("confidence_threshold", 0.25),
        perception_input_size=perc.get("input_size", 640),
        perception_prompts=perc.get("prompts", DEFAULT_PERCEPTION_PROMPTS.copy()),
        preprocess_width=pre.get("target_width", 640),
        preprocess_height=pre.get("target_height", 360),
        enable_clahe=pre.get("enable_clahe", False),
        clahe_clip_limit=pre.get("clahe_clip_limit", 2.0),
        clahe_grid_size=pre.get("clahe_grid_size", 8),
        ema_alpha=post.get("ema_alpha", 0.7),
        min_mask_area_fraction=post.get("min_mask_area_fraction", 0.001),
        morphology_kernel_size=post.get("morphology_kernel_size", 5),
        planning_mode=plan.get("mode", "centerline"),
        centerline_samples=plan.get("centerline_samples", 20),
        kalman_process_noise=plan.get("kalman_process_noise", 1e-3),
        kalman_measurement_noise=plan.get("kalman_measurement_noise", 1e-1),
        fallback_after_n_misses=plan.get("fallback_after_n_misses", 3),
        min_road_pixels=plan.get("min_road_pixels", 500),
        viplanner_model_dir=viplanner.get("model_dir", "models/viplanner"),
        viplanner_max_depth_m=viplanner.get("max_depth_m", 15.0),
        viplanner_input_height=viplanner.get("input_height", 360),
        viplanner_input_width=viplanner.get("input_width", 640),
        auto_goal_min_row_frac=auto_goal.get("min_row_frac", 0.1),
        auto_goal_max_row_frac=auto_goal.get("max_row_frac", 0.85),
        auto_goal_search_margin_px=auto_goal.get("search_margin_px", 24),
        auto_goal_fallback_forward_m=auto_goal.get("fallback_forward_m", 4.0),
        stanley_gain_k=ctrl.get("stanley_gain_k", 2.5),
        stanley_softening=ctrl.get("stanley_softening", 1.0),
        pure_pursuit_lookahead_min_m=pure_pursuit.get("lookahead_min_m", 2.5),
        pure_pursuit_lookahead_gain=pure_pursuit.get("lookahead_gain", 0.4),
        pure_pursuit_wheelbase_m=pure_pursuit.get("wheelbase_m", 2.8),
        pure_pursuit_max_steer_cmd=pure_pursuit.get("max_steer_cmd", 1.0),
        target_speed_mps=ctrl.get("target_speed_mps", 5.0),
        max_throttle=ctrl.get("max_throttle", 0.6),
        max_brake=ctrl.get("max_brake", 0.8),
        speed_kp=ctrl.get("speed_kp", 0.3),
        dashboard_colors=_load_dashboard_colors(dashboard.get("colors")),
    )
