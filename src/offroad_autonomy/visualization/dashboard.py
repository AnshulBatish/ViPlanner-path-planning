"""Development dashboard for autonomy visualization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from offroad_autonomy.types import DEFAULT_DASHBOARD_COLORS, PathPlan


@dataclass
class DashboardTelemetry:
    """Compact status values shown in the dashboard sidebar."""

    speed_mps: float
    steering: float
    throttle: float
    brake: float
    perception_confidence: float
    stability_score: float
    kalman_active: bool
    fps: float
    latency_ms: float
    planner_mode: str
    fear_score: float
    plan_valid: bool


class AutonomyDashboard:
    """Render a dashboard view for the autonomy stack."""

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        colors: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self._colors = DEFAULT_DASHBOARD_COLORS.copy()
        if colors is not None:
            self._colors.update(colors)

    def render(
        self,
        camera_bgr: np.ndarray,
        postprocessed_mask: np.ndarray,
        plan: PathPlan | None,
        telemetry: DashboardTelemetry,
    ) -> np.ndarray:
        camera = self._ensure_bgr(camera_bgr)
        mask_shape = postprocessed_mask.shape[:2]
        overlay = self._build_overlay(camera, postprocessed_mask, plan, mask_shape)

        canvas = np.full((self.height, self.width, 3), self._colors["BG"], dtype=np.uint8)
        sidebar_w = 360
        image_rect = (24, 24, self.width - sidebar_w - 48, self.height - 48)
        sidebar_rect = (self.width - sidebar_w - 24, 24, sidebar_w, self.height - 48)

        self._draw_panel(canvas, image_rect)
        self._draw_panel(canvas, sidebar_rect)

        fitted = self._fit_image(overlay, image_rect[2] - 24, image_rect[3] - 24)
        self._blit(canvas, fitted, image_rect[0] + 12, image_rect[1] + 12)
        self._draw_sidebar(canvas, sidebar_rect, telemetry, plan)
        return canvas

    def _build_overlay(
        self,
        camera: np.ndarray,
        mask: np.ndarray,
        plan: PathPlan | None,
        plan_shape: tuple[int, int],
    ) -> np.ndarray:
        display = camera.copy()
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.shape[:2] != camera.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (camera.shape[1], camera.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_u8.astype(bool)

        if mask_bool.any():
            layer = display.copy()
            layer[mask_bool] = self._colors["MASK_FILL"]
            display = cv2.addWeighted(layer, 0.25, display, 0.75, 0.0)

        if plan is not None:
            path = plan.overlay_pixels if len(plan.overlay_pixels) >= 2 else plan.centerline
            if len(path) >= 2:
                pts = path.astype(np.float32).copy()
                src_h, src_w = plan_shape
                dst_h, dst_w = camera.shape[:2]
                if src_h > 0 and src_w > 0 and (src_h, src_w) != (dst_h, dst_w):
                    pts[:, 0] *= dst_w / src_w
                    pts[:, 1] *= dst_h / src_h
                pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [pts_i], False, self._colors["PATH_GLOW"], 10, lineType=cv2.LINE_AA)
                cv2.polylines(display, [pts_i], False, self._colors["PATH_CORE"], 4, lineType=cv2.LINE_AA)

            if plan.goal_overlay_pixel is not None:
                gx, gy = plan.goal_overlay_pixel
                if plan_shape != camera.shape[:2]:
                    gx *= camera.shape[1] / max(plan_shape[1], 1)
                    gy *= camera.shape[0] / max(plan_shape[0], 1)
                goal = (int(round(gx)), int(round(gy)))
                cv2.circle(display, goal, 8, self._colors["PATH_HIGHLIGHT"], -1, lineType=cv2.LINE_AA)
                cv2.circle(display, goal, 12, self._colors["PATH_CORE"], 2, lineType=cv2.LINE_AA)

        return display

    def _draw_sidebar(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
        plan: PathPlan | None,
    ) -> None:
        x, y, w, h = rect
        lines = [
            "OFF-ROAD AUTONOMY",
            f"Planner: {telemetry.planner_mode}",
            f"Plan valid: {'yes' if telemetry.plan_valid else 'no'}",
            f"Fear score: {telemetry.fear_score:.2f}",
            f"Speed: {telemetry.speed_mps:.2f} m/s",
            f"Steering: {telemetry.steering:+.2f}",
            f"Throttle: {telemetry.throttle:.2f}",
            f"Brake: {telemetry.brake:.2f}",
            f"Perception conf: {telemetry.perception_confidence:.2f}",
            f"Mask stability: {telemetry.stability_score:.2f}",
            f"Kalman fallback: {'yes' if telemetry.kalman_active else 'no'}",
            f"FPS: {telemetry.fps:.1f}",
            f"Loop latency: {telemetry.latency_ms:.0f} ms",
        ]
        if plan is not None and plan.failure_reason:
            lines.append(f"Reason: {plan.failure_reason}")
        if plan is not None and plan.goal_vehicle is not None:
            gx, gy, gz = plan.goal_vehicle
            lines.append(f"Goal vehicle: ({gx:.1f}, {gy:.1f}, {gz:.1f})")

        yy = y + 36
        for idx, line in enumerate(lines):
            color = self._colors["TEXT_PRIMARY"] if idx == 0 else self._colors["TEXT_SECONDARY"]
            scale = 0.72 if idx == 0 else 0.52
            cv2.putText(
                canvas,
                line,
                (x + 18, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )
            yy += 30 if idx == 0 else 24
            if yy > y + h - 24:
                break

    def _draw_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["PANEL_BG"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BORDER"], 1)

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _fit_image(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out = np.full((target_h, target_w, 3), self._colors["PANEL_BG"], dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return out

    @staticmethod
    def _blit(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
        h, w = image.shape[:2]
        canvas[y:y + h, x:x + w] = image
