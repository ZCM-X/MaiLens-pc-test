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
        PlaneLockTracker,
        apply_geometry_lock,
        apply_plane_lock,
        build_remap,
        draw_debug,
        geometry_margins,
        map_fisheye_box_to_output,
        make_output_rays,
        rotation_for_row,
        transform_box,
        transform_box_homography,
        expand_box,
        geometry_reference,
        update_geometry_lock_state,
    )
except ImportError:  # Running from `python pc/pc_receiver.py`.
    from process_session import (
        GeometryDetector,
        GeometryLockTracker,
        PlaneLockTracker,
        apply_geometry_lock,
        apply_plane_lock,
        build_remap,
        draw_debug,
        geometry_margins,
        map_fisheye_box_to_output,
        make_output_rays,
        rotation_for_row,
        transform_box,
        transform_box_homography,
        expand_box,
        geometry_reference,
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
        detect_every: int = 12,
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
        self.reference_target_size: float | None = None
        self.reference_zoom: float | None = None
        self.lock_tracker = GeometryLockTracker(self.detect_every)
        self.plane_tracker = PlaneLockTracker(self.detect_every)
        self.previous_gray: np.ndarray | None = None
        self.detection_age = 0
        self.lock_source = "none"
        # While a hand covers the machine the flow estimate is meaningless, so
        # the last good warp is reused for a few frames instead of snapping to
        # the detector box (which lurched the whole preview).
        self.plane_hold: np.ndarray | None = None
        self.plane_hold_frames = 0
        self.plane_hold_limit = 12

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
        # A phone can change capture dimensions when the camera rotates or a
        # sender renegotiates its stream.  Sparse LK flow cannot compare
        # pyramids with different shapes; discard the old flow/lock state and
        # reacquire the cabinet on the new geometry instead.
        if self.previous_gray is not None and self.previous_gray.shape != current_gray.shape:
            self.previous_gray = None
            self.lock_tracker.reset()
            self.previous_center = np.array([0.5, 0.5], dtype=np.float32)
            self.previous_zoom = 1.0
            self.reference_target_size = None
            self.reference_zoom = None
            self.lock_source = "searching"
            self.plane_tracker.reset()
        # Once the plane lock is active it owns the LK pass.  Keeping the old
        # box tracker only during acquisition avoids doing two optical-flow
        # solves for every 60-fps frame.
        if not self.plane_tracker.locked:
            self.lock_tracker.update_flow(self.previous_gray, current_gray)

        detector_ran = False
        if self.detector.enabled and self.frame_index % self.detect_every == 0:
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
            # Both boxes came from one raw-frame detector result.  Their
            # inverse-fisheye corners can cross by a few percent, so allow the
            # mapped-pair gate while still rejecting unrelated duplicates.
            self.lock_tracker.ingest(
                detected_outer,
                detected_inner,
                width,
                height,
                allow_soft_pair=True,
            )

        if not self.detector.enabled:
            self.lock_tracker.reset()
            self.plane_tracker.reset()

        outer, inner = self.lock_tracker.boxes()
        plane_locked = self.plane_tracker.update(
            current_gray,
            inner,
            outer,
            width,
            height,
            self.lock_fill,
        )
        if self.lock_tracker.box is None:
            # Keep the last displayed transform until a new target is found,
            # but do not carry its distance reference to a different target.
            self.reference_target_size = None
            self.reference_zoom = None
        elif inner is not None and (self.reference_target_size is None or self.reference_zoom is None):
            self.reference_target_size, self.reference_zoom = geometry_reference(
                inner, width, height, self.lock_fill,
            )
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
            reference_target_size=self.reference_target_size,
            reference_zoom=self.reference_zoom,
        )
        if lock_source != "none":
            self.previous_center, self.previous_zoom = center, zoom
            self.lock_source = lock_source
        elif detector_ran or self.lock_tracker.box is None:
            self.lock_source = "searching"

        plane_matrix = self.plane_tracker.output_homography if plane_locked else None
        fixed_inner = (
            transform_box_homography(
                self.plane_tracker.reference_box,
                self.plane_tracker.reference_to_output,
            )
            if self.plane_tracker.locked else None
        )
        if plane_matrix is not None and fixed_inner is not None:
            self.plane_hold = plane_matrix
            self.plane_hold_frames = 0
            self.lock_source = "plane_homography"
        elif (self.plane_hold is not None and fixed_inner is not None
              and self.plane_hold_frames < self.plane_hold_limit):
            plane_matrix = self.plane_hold
            self.plane_hold_frames += 1
            self.lock_source = "plane_hold"
        else:
            plane_matrix = None
            self.plane_hold = None
            self.plane_hold_frames = 0

        if plane_matrix is not None and fixed_inner is not None:
            stabilized = apply_plane_lock(stabilized, plane_matrix, fixed_inner)
            outer = expand_box(fixed_inner, 1.45)
            # The current detector rectangle is allowed to jitter.  The
            # green debug frame represents the latched output plane instead,
            # so the overlay visualizes the lock rather than detector noise.
            inner = fixed_inner
            # The perspective transform already contains translation, scale
            # and tilt compensation.  Do not apply a second affine crop.
            center = np.array([0.5, 0.5], dtype=np.float32)
            zoom = 1.0
        else:
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
            "geometry_margins": geometry_margins(outer, inner),
            "lock_source": self.lock_source,
            "lock_anchor": "outer_buttons" if outer is not None else ("inner_screen" if inner is not None else "none"),
            "detection_age_frames": self.detection_age,
            "plane_lock": plane_matrix is not None,
            "plane_age_frames": self.plane_tracker.age_frames,
            "plane_inliers": self.plane_tracker.inliers,
            "plane_inlier_ratio": self.plane_tracker.inlier_ratio,
            "plane_reprojection_error": self.plane_tracker.reprojection_error,
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
