#!/usr/bin/env python3
"""Low-latency per-frame processor used by the live PC receiver."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .process_session import (
        GeometryDetector,
        GeometryLockTracker,
        apply_geometry_lock,
        build_remap,
        draw_debug,
        map_fisheye_box_to_output,
        make_output_rays,
        rotation_for_row,
        transform_box,
        update_geometry_lock_state,
    )
except ImportError:  # Running from `python pc/pc_receiver.py`.
    from process_session import (
        GeometryDetector,
        GeometryLockTracker,
        apply_geometry_lock,
        build_remap,
        draw_debug,
        map_fisheye_box_to_output,
        make_output_rays,
        rotation_for_row,
        transform_box,
        update_geometry_lock_state,
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
        detect_every: int = 4,
        lock_fill: float = 0.64,
        debug: bool = False,
    ) -> None:
        self.crop = crop
        self.fov = fov
        self.center_x = center_x
        self.center_y = center_y
        self.k1 = k1
        self.k2 = k2
        self.detect_every = max(1, detect_every)
        self.lock_fill = float(min(max(lock_fill, 0.35), 0.90))
        self.debug = debug
        self.detector = GeometryDetector(model)
        self.frame_index = 0
        self.reference = None
        self.output_shape: tuple[int, int] | None = None
        self.output_rays: np.ndarray | None = None
        self.previous_center = np.array([0.5, 0.5], dtype=np.float32)
        self.previous_zoom = 1.0
        self.lock_tracker = GeometryLockTracker(self.detect_every)
        self.previous_gray: np.ndarray | None = None
        self.detection_age = 0
        self.lock_source = "none"

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
        current_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        self.lock_tracker.update_flow(self.previous_gray, current_gray)

        detector_ran = False
        if self.detector.enabled and (
            self.frame_index % self.detect_every == 0
            or self.lock_tracker.box is None
        ):
            detector_ran = True
            # The model was trained on the raw clip-on-lens geometry. Detect
            # there, then map the resulting boxes through the same fisheye
            # inverse used by the output renderer.
            detected_outer_raw, detected_inner_raw = self.detector.detect(frame)
            detected_outer = map_fisheye_box_to_output(
                detected_outer_raw, width, height, rotation, self.crop, self.fov,
                self.k1, self.k2, self.center_x, self.center_y,
            )
            detected_inner = map_fisheye_box_to_output(
                detected_inner_raw, width, height, rotation, self.crop, self.fov,
                self.k1, self.k2, self.center_x, self.center_y,
            )
            self.lock_tracker.ingest(detected_outer, detected_inner, width, height)

        if not self.detector.enabled:
            self.lock_tracker.reset()

        outer, inner = self.lock_tracker.boxes()
        self.detection_age = self.lock_tracker.age_frames

        center = self.previous_center.copy()
        zoom = self.previous_zoom
        acquiring = self.lock_source in ("none", "searching") and self.lock_tracker.box is not None
        center, zoom, lock_source = update_geometry_lock_state(
            self.previous_center,
            self.previous_zoom,
            outer,
            inner,
            width,
            height,
            self.lock_fill,
            snap=acquiring,
        )
        if lock_source != "none":
            self.previous_center, self.previous_zoom = center, zoom
            self.lock_source = lock_source
        elif detector_ran or self.lock_tracker.box is None:
            self.lock_source = "searching"

        stabilized, geometry_matrix = apply_geometry_lock(stabilized, center, zoom)
        outer = transform_box(outer, geometry_matrix)
        inner = transform_box(inner, geometry_matrix)
        self.previous_gray = current_gray
        self.frame_index += 1

        debug = {
            "frame_id": metadata.get("frame_id", self.frame_index),
            "timestamp": metadata.get("timestamp"),
            "sensor_delta_ms": self._sensor_delta_ms(metadata),
            "center": center.tolist(),
            "zoom": zoom,
            "detected_outer": outer,
            "detected_inner": inner,
            "lock_source": self.lock_source,
            "lock_anchor": "outer_frame" if outer is not None else ("inner_screen" if inner is not None else "none"),
            "detection_age_frames": self.detection_age,
        }
        if self.debug:
            anchor = "outer" if outer is not None else ("inner" if inner is not None else "none")
            stabilized = draw_debug(
                stabilized,
                outer,
                inner,
                f"{self.frame_index}  lock={self.lock_source}  anchor={anchor}  zoom={zoom:.2f}",
            )
        return stabilized, debug

    @staticmethod
    def _sensor_delta_ms(metadata: dict) -> float | None:
        pose = metadata.get("pose") or {}
        if pose.get("timestamp") is None or metadata.get("timestamp") is None:
            return None
        return (float(pose["timestamp"]) - float(metadata["timestamp"])) * 1000.0
