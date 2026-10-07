#!/usr/bin/env python3
"""Low-latency per-frame processor used by the live PC receiver."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .process_session import (
        FrameClock,
        GeometryDetector,
        GeometryLockTracker,
        PlaneCornersSmoother,
        PlaneLockTracker,
        _box_points,
        _project_points,
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
        plausible_geometry_pair,
        soft_geometry_pair,
        update_geometry_lock_state,
        lens_remap,
        rotation_homography,
    )
    from . import canonical
    from .shot_authority import AuthorityGovernor, AuthorityLimits
except ImportError:  # Running from `python pc/pc_receiver.py`.
    from process_session import (
        FrameClock,
        GeometryDetector,
        GeometryLockTracker,
        PlaneCornersSmoother,
        PlaneLockTracker,
        _box_points,
        _project_points,
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
        plausible_geometry_pair,
        soft_geometry_pair,
        update_geometry_lock_state,
        lens_remap,
        rotation_homography,
    )
    import canonical
    from shot_authority import AuthorityGovernor, AuthorityLimits


def screen_circle(box) -> tuple[np.ndarray, float]:
    """Centre and mean half-size of a screen box, in the frame it arrives in."""
    x0, y0, x1, y1 = (float(value) for value in box)
    return (np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64),
            0.25 * ((x1 - x0) + (y1 - y0)))


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
        lock_fill: float = 0.71,
        debug: bool = False,
        fast_remap: bool = True,
        lock_authority: AuthorityLimits | None = None,
        full_warp: bool = True,
        plane_smooth: float = 0.35,
        correction_gain: float = 0.15,
        ring_round: bool = True,
        ring_gain: float = 1.8,
        ring_order: int = 3,
        ring_ratio: float = 0.0,
        ring_coarse: int = 8,
        ring_blend: float = 0.35,
        ring_hold: int = 12,
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
        # The lens map is constant, so applying it once and rotating the
        # rectified frame with a perspective warp is exactly equivalent to
        # rebuilding the fisheye map every frame, at a fraction of the cost.
        self.fast_remap = bool(fast_remap)
        # One viewpoint for the whole frame.  The older path pasted a
        # feathered ellipse of the machine over the live background, which
        # put a visible boundary around the cabinet.
        self.full_warp = bool(full_warp)
        # The tracker rebuilds its homography every frame, so the corners
        # carry that noise straight to the picture unless they are blended
        # over time.  The offline render already did this; the live preview
        # did not, which is why the same lock looked twitchier on screen.
        self.corner_smoother = PlaneCornersSmoother(plane_smooth)
        self.lens_map: tuple[np.ndarray, np.ndarray] | None = None
        self.lens_map_shape: tuple[int, int] | None = None
        self.detector = GeometryDetector(model)
        self.frame_index = 0
        self.reference = None
        self.output_shape: tuple[int, int] | None = None
        self.output_rays: np.ndarray | None = None
        self.previous_center = np.array([0.5, 0.5], dtype=np.float32)
        self.previous_zoom = 1.0
        self.reference_target_size: float | None = None
        self.reference_zoom: float | None = None
        # The plane lock puts the machine in the middle at the size it was
        # locked at, but it cannot know the screen and the raised buttons are
        # at different depths.  ``pull_back_maps`` walks the eight button slots
        # onto one circle around the delivered screen, so the four side gaps
        # come out equal; that is the correction ``docs/margin-spec.md``
        # measures.  It runs on the delivered frame, using the delivered
        # screen box for the radius, because a radius from any other space
        # scales the ramp and turns the pull into a global zoom.
        self.ring_round = bool(ring_round)
        self.ring_gain = float(min(max(ring_gain, 0.0), 4.0))
        self.ring_order = max(1, int(ring_order))
        #: Absolute radius ratio to target, or 0 to even the ring out around
        #: the mean it already has.  The lock, not this correction, owns the
        #: size; this one owns the equality of the four gaps.
        self.ring_ratio = float(ring_ratio)
        self.ring_blend = float(min(max(ring_blend, 0.05), 1.0))
        self.ring_hold = max(0, int(ring_hold))
        self.ring_coarse = max(1, int(ring_coarse))
        self.ring_coefficients: np.ndarray | None = None
        self.ring_centre = np.array([0.5, 0.5], dtype=np.float64)
        self.ring_screen: tuple[np.ndarray, float] | None = None
        self.ring_radius = 0.0
        self.ring_missing = 0
        self.ring_margins = 0.0
        self.lock_zoom_ratio = 1.0
        self.lock_reason = "none"
        self.lock_tracker = GeometryLockTracker(self.detect_every)
        self.plane_tracker = PlaneLockTracker(
            self.detect_every,
            correction_gain=float(min(max(correction_gain, 0.0), 1.0)),
        )
        self.previous_gray: np.ndarray | None = None
        self.detection_age = 0
        self.lock_source = "none"
        # While a hand covers the machine the flow estimate is meaningless, so
        # the last good warp is reused for a few frames instead of snapping to
        # the detector box (which lurched the whole preview).
        self.plane_hold: np.ndarray | None = None
        self.plane_hold_frames = 0
        self.plane_hold_limit = 12
        # Bounded lens travel: inside the range the machine is pinned, past it
        # the picture rides along with the phone again.  See shot_authority.
        self.authority = AuthorityGovernor(lock_authority)
        self.clock = FrameClock()
        self.lock_mode = "none"
        self.lock_travel = 0.0
        self.lock_limit = 0.0

    #: Radius band around the median that decides which purple blobs are the
    #: button ring.  Loose enough that a lit or partly hidden button still
    #: counts, tight enough to drop the artwork fragments floating around it.
    RING_BAND = 0.25

    def _measure_ring(self, frame: np.ndarray, screen_box):
        """Ring centre, radius scale and shape coefficients for one frame.

        The radius only sets the band that selects ring blobs and the point
        where the correction ramp starts, so the screen box the lock placed is
        good enough for both.  The *centre* is not: on a delivered frame that
        box sits 20-30 px off the button ring, and at a ring radius of ~300 px
        that offset alone reads as 8-10% of spread -- the same size as the
        defect being corrected.  The ring is therefore centred on its own
        blobs, and its shape is described relative to the mean radius it
        already has, so no radius borrowed from another space can turn the
        pull into a global zoom.
        """
        blob = canonical.screen_blob(frame)
        if blob is not None:
            centre0, radius = blob[0], blob[1]
            if self.ring_screen is not None:
                last_centre, last_radius = self.ring_screen
                if not 0.6 * last_radius <= radius <= 1.7 * last_radius:
                    blob = None
                else:
                    centre0 = 0.5 * (centre0 + last_centre)
                    radius = 0.5 * (radius + last_radius)
            if blob is not None:
                self.ring_screen = (centre0, radius)
        if blob is None:
            # No clean play field on this frame: fall back to the box the lock
            # placed, which at least keeps the ramp in delivered pixels.
            centre0, radius = screen_circle(screen_box)
        if not np.isfinite(radius) or radius <= 5.0:
            return None
        points = canonical.purple_points(frame)
        if points.shape[0] < 4:
            return None
        span = np.hypot(points[:, 0] - centre0[0], points[:, 1] - centre0[1]) / radius
        rough = float(np.median(span))
        if rough <= 0.0:
            return None
        ring = points[np.abs(span - rough) <= self.RING_BAND * rough]
        if ring.shape[0] < 4:
            return None
        centre = centre0 if blob is not None else ring.mean(axis=0)
        ratios = canonical.slot_ratios(centre, radius, ring)
        fitted = canonical.ring_profile(ratios, self.ring_order)
        if fitted is None or not np.isfinite(fitted).all() or fitted[0] <= 0.0:
            return None
        return centre, float(radius), fitted, canonical.ring_error(ratios)

    def _pull_margins_back(self, frame: np.ndarray, screen_box):
        """Even out the four side gaps on the delivered frame.

        A button is a finite patch, so the ramp moves its near half further
        than its far half and the blob centroid follows only part of the way;
        that is why the same correction is worth running above unity gain.  The
        knob that matters is ``ring_gain``, and the shape it targets is the
        mean radius the ring already has, which keeps the size the lock chose.
        """
        if not self.ring_round:
            return frame
        measured = self._measure_ring(frame, screen_box)
        if measured is None:
            self.ring_missing += 1
        else:
            centre, radius, fitted, spread = measured
            self.ring_centre = centre
            self.ring_radius = radius
            if self.ring_coefficients is None:
                self.ring_coefficients = fitted
            else:
                self.ring_coefficients = ((1.0 - self.ring_blend) * self.ring_coefficients
                                          + self.ring_blend * fitted)
            self.ring_missing = 0
            self.ring_margins = spread or 0.0
        if self.ring_coefficients is None or self.ring_missing > self.ring_hold:
            return frame
        target = float(self.ring_ratio) if self.ring_ratio > 0.0 else float(self.ring_coefficients[0])
        if target <= 0.0:
            return frame
        maps = canonical.pull_back_maps(
            frame.shape[1], frame.shape[0], self.ring_centre, self.ring_radius,
            self.ring_coefficients, strength=self.ring_gain,
            order=self.ring_order, target=target, coarse=self.ring_coarse,
        )
        return canonical.apply(frame, maps)

    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        height, width = frame.shape[:2]
        if self.output_shape != (width, height):
            self.output_shape = (width, height)
            self.output_rays = make_output_rays(width, height, self.crop, self.fov)
            self.lock_limit = self.authority.limits.shift_limit(width, height)

        rotation, self.reference = rotation_for_row(metadata, self.reference)
        frame_dt = self.clock.dt(metadata)

        if self.fast_remap:
            if self.lens_map is None or self.lens_map_shape != (width, height):
                self.lens_map = lens_remap(
                    width, height, self.crop, self.fov,
                    self.k1, self.k2, self.center_x, self.center_y,
                )
                self.lens_map_shape = (width, height)
            rectified = cv2.remap(
                frame, self.lens_map[0], self.lens_map[1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101,
            )
            homography = rotation_homography(rotation, width, height, self.crop, self.fov)
            stabilized = cv2.warpPerspective(
                rectified, homography, (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
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
            self.lens_map = None
            self.lens_map_shape = None
            self.lock_tracker.reset()
            self.previous_center = np.array([0.5, 0.5], dtype=np.float32)
            self.previous_zoom = 1.0
            self.reference_target_size = None
            self.reference_zoom = None
            self.lock_source = "searching"
            self.plane_tracker.reset()
            self.lock_mode = "none"
            self.lock_travel = 0.0
            self.lock_limit = self.authority.limits.shift_limit(width, height)
        # Once the plane lock is active it owns the LK pass.  Keeping the old
        # box tracker only during acquisition avoids doing two optical-flow
        # solves for every 60-fps frame.
        if not self.plane_tracker.locked:
            self.lock_tracker.update_flow(self.previous_gray, current_gray, dt=frame_dt)

        detector_ran = False
        fresh_inner_detection = False
        fresh_inner_box = None
        fresh_outer_box = None
        #: The absolute anchor does not need the continuity check the box
        #: tracker applies; it needs a detector pair that really is the
        #: machine.  Gating it on the tracker's jump check threw the reading
        #: away exactly when the phone moved fast, which is when the
        #: accumulated transform needed it most.
        anchor_box = None
        anchor_outer_box = None
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
                dt=frame_dt,
            )
            fresh_inner_detection = (
                detected_inner is not None
                and self.lock_tracker.outer_age_frames == 0
                and self.lock_tracker.inner_age_frames == 0
            )
            if fresh_inner_detection:
                fresh_inner_box = detected_inner
                fresh_outer_box = detected_outer
            if (detected_inner is not None and detected_outer is not None
                    and (plausible_geometry_pair(detected_outer, detected_inner)
                         or soft_geometry_pair(detected_outer, detected_inner))):
                anchor_box = detected_inner
                anchor_outer_box = detected_outer

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
            dt=frame_dt,
        )
        if self.plane_tracker.locked and anchor_box is not None:
            # The flow transform is accumulated frame by frame and drifts;
            # this absolute anchor ties it back to the detected machine.
            self.plane_tracker.reanchor(anchor_box, anchor_outer_box)
        plane_reacquired = self.plane_tracker.reacquire_if_stale(
            current_gray,
            inner,
            outer,
            width,
            height,
            self.lock_fill,
            fresh_detection=fresh_inner_detection,
        )
        if plane_reacquired:
            plane_locked = True
            self.plane_hold = None
            self.plane_hold_frames = 0
            self.corner_smoother.reset()
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
            reference_corners = _box_points(self.plane_tracker.reference_box)
            raw_corners = _project_points(reference_corners, plane_matrix)
            smoothed = self.corner_smoother.update(raw_corners, dt=frame_dt)
            if reference_corners is not None and smoothed is not None:
                plane_matrix = cv2.getPerspectiveTransform(
                    reference_corners, smoothed.astype(np.float32),
                )
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
            self.corner_smoother.reset()

        if plane_matrix is not None and fixed_inner is not None:
            # The lock is only allowed the travel a real lens has.  Whatever it
            # cannot take out stays visible, so the machine follows the phone
            # past the range instead of the picture tearing itself apart
            # reaching for a machine that is no longer there.
            decision = self.authority.decide(
                self.plane_tracker.lock_travel(width, height), width, height,
                dt=frame_dt,
            )
            plane_matrix = decision.apply(plane_matrix)
            fixed_inner = decision.move_box(fixed_inner)
            self.lock_mode = decision.mode
            self.lock_travel = decision.travelled
            self.lock_limit = decision.limit
            self.lock_zoom_ratio = decision.zoom_ratio
            self.lock_reason = decision.reason
            stabilized = apply_plane_lock(
                stabilized, plane_matrix, fixed_inner, full_warp=self.full_warp,
            )
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
            self.lock_mode = "geometry"
            stabilized, geometry_matrix = apply_geometry_lock(stabilized, center, zoom)
            outer = transform_box(outer, geometry_matrix)
            inner = transform_box(inner, geometry_matrix)
        if self.ring_round and plane_matrix is not None and inner is not None:
            stabilized = self._pull_margins_back(stabilized, inner)
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
            "lock_mode": self.lock_mode,
            "lock_travel_px": round(float(self.lock_travel), 2),
            "lock_travel_limit_px": round(float(self.lock_limit), 2),
            "lock_zoom_ratio": round(float(self.lock_zoom_ratio), 4),
            "lock_reason": self.lock_reason,
            "ring_round": self.ring_round,
            "ring_margin_spread": round(float(self.ring_margins), 4),
            "plane_matrix": (
                np.asarray(plane_matrix, dtype=np.float64).reshape(-1).round(6).tolist()
                if plane_matrix is not None else None
            ),
            "plane_reacquired": plane_reacquired,
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
                f"{self.frame_index}  lock={self.lock_source}  {self.lock_mode}  anchor={anchor}  zoom={zoom:.2f}",
            )
        return stabilized, debug

    @staticmethod
    def _sensor_delta_ms(metadata: dict) -> float | None:
        pose = metadata.get("pose") or {}
        if pose.get("timestamp") is None or metadata.get("timestamp") is None:
            return None
        return (float(pose["timestamp"]) - float(metadata["timestamp"])) * 1000.0
