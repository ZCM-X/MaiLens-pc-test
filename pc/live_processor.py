#!/usr/bin/env python3
"""Low-latency per-frame processor used by the live PC receiver."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .process_session import (
        GeometryDetector,
        apply_geometry_lock,
        build_remap,
        draw_debug,
        make_output_rays,
        rotation_for_row,
        transform_box,
    )
except ImportError:  # Running from `python pc/pc_receiver.py`.
    from process_session import (
        GeometryDetector,
        apply_geometry_lock,
        build_remap,
        draw_debug,
        make_output_rays,
        rotation_for_row,
        transform_box,
    )


class LiveProcessor:
    """Stateful stabilizer: one input JPEG becomes one processed BGR frame."""

    def __init__(
        self,
        *,
        crop: float = 0.74,
        fov: float = 106.4583,
        center_x: float = 0.501753869,
        center_y: float = 0.499423644,
        k1: float = 0.0893163,
        k2: float = -0.0174637,
        model: Path | None = None,
        detect_every: int = 3,
        debug: bool = False,
    ) -> None:
        self.crop = crop
        self.fov = fov
        self.center_x = center_x
        self.center_y = center_y
        self.k1 = k1
        self.k2 = k2
        self.detect_every = max(1, detect_every)
        self.debug = debug
        self.detector = GeometryDetector(model)
        self.frame_index = 0
        self.reference = None
        self.output_shape: tuple[int, int] | None = None
        self.output_rays: np.ndarray | None = None
        self.previous_center = np.array([0.5, 0.5], dtype=np.float32)
        self.previous_zoom = 1.0
        self.previous_outer = None
        self.previous_inner = None

    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        height, width = frame.shape[:2]
        if self.output_shape != (width, height):
            self.output_shape = (width, height)
            self.output_rays = make_output_rays(width, height, self.crop, self.fov)

        rotation, self.reference = rotation_for_row(metadata, self.reference)
        map_x, map_y = build_remap(
            width, height, rotation, self.crop, self.fov,
            self.k1, self.k2, self.center_x, self.center_y,
            output_rays=self.output_rays,
        )
        stabilized = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

        if self.detector.model is not None and (
            self.frame_index % self.detect_every == 0 or self.previous_inner is None
        ):
            self.previous_outer, self.previous_inner = self.detector.detect(stabilized)
        outer, inner = self.previous_outer, self.previous_inner

        center = self.previous_center.copy()
        zoom = self.previous_zoom
        if inner:
            center_target = np.array([
                (inner[0] + inner[2]) / (2 * width),
                (inner[1] + inner[3]) / (2 * height),
            ], dtype=np.float32)
            center = self.previous_center * 0.82 + center_target * 0.18
            inner_width = max(inner[2] - inner[0], 1)
            zoom_target = max(0.75, min(1.25, 0.30 * width / inner_width))
            zoom = self.previous_zoom * 0.93 + zoom_target * 0.07
            self.previous_center, self.previous_zoom = center, zoom

        stabilized, geometry_matrix = apply_geometry_lock(stabilized, center, zoom)
        outer = transform_box(outer, geometry_matrix)
        inner = transform_box(inner, geometry_matrix)
        self.frame_index += 1

        debug = {
            "frame_id": metadata.get("frame_id", self.frame_index),
            "timestamp": metadata.get("timestamp"),
            "sensor_delta_ms": self._sensor_delta_ms(metadata),
            "center": center.tolist(),
            "zoom": zoom,
            "detected_outer": outer,
            "detected_inner": inner,
        }
        if self.debug:
            stabilized = draw_debug(
                stabilized,
                outer,
                inner,
                f"{self.frame_index}  crop={self.crop:.2f}  zoom={zoom:.2f}",
            )
        return stabilized, debug

    @staticmethod
    def _sensor_delta_ms(metadata: dict) -> float | None:
        pose = metadata.get("pose") or {}
        if pose.get("timestamp") is None or metadata.get("timestamp") is None:
            return None
        return (float(pose["timestamp"]) - float(metadata["timestamp"])) * 1000.0

