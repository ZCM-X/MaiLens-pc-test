#!/usr/bin/env python3
"""Bounded travel for the machine lock: pin inside the range, follow outside.

The clip the operator supplied shows the reference app doing something our
lock has never done: it gives up.  While the phone stays inside a working
range the cabinet is nailed to the middle of the frame; once the operator pans
or walks past that range the cabinet stops being pinned and travels with the
room again, then slides back to its locked spot as soon as the phone comes
back inside.  The operator describes it as "out of range it becomes follow
mode, close it locks".

That is a lens with finite travel.  Of the correction the lock asks for, only
the part inside the travel can be taken out of the picture; the rest has to
stay visible, and a visible leftover is exactly what makes the cabinet follow
the phone.  The leftover is ``moved - clamp(moved)``, which is continuous, so
the hand-over needs no state machine and no snap: it is one clamp on the shift
and one on the zoom.  The mode reported here is only a label for the operator.

Keeping the arithmetic in one place lets the live receiver, the offline
renderer and the iOS port share a single definition of the range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AuthorityLimits:
    """How far the lens may travel while holding the machine still.

    ``max_shift`` is the fraction of the frame's short side that the crop may
    be dragged away from the locked target.  ``min_zoom``/``max_zoom`` bound
    the distance compensation and are ratios of the machine's size now to its
    size when the lock was taken: ``max_zoom`` is how much closer the operator
    may walk before the picture starts changing scale with them.
    """

    max_shift: float = 0.30
    min_zoom: float = 0.62
    max_zoom: float = 1.75
    #: Dead band kept when the picture comes back in range, as a fraction of
    #: the travel.  Without it a machine sitting on the limit flips between
    #: lock and follow every few frames and the clamp reads as chatter.
    hysteresis: float = 0.15
    #: Seconds of smoothing on the leftover correction.  The measured zoom of
    #: a hand-held fisheye lock wobbles several percent per frame; an
    #: unsmoothed clamp turns that into the picture breathing.
    smooth_seconds: float = 0.25

    def shift_limit(self, width: int, height: int) -> float:
        return max(float(self.max_shift), 0.0) * float(min(int(width), int(height)))

    def clamp_zoom(self, ratio: float) -> float:
        low = min(float(self.min_zoom), float(self.max_zoom))
        high = max(float(self.min_zoom), float(self.max_zoom))
        return float(min(max(float(ratio), low), high))


#: Empty travel, used before a lock exists or when the tracker cannot measure.
NO_TRAVEL = {
    "shift": np.zeros(2, dtype=np.float64),
    "zoom": 1.0,
    "output_scale": 1.0,
    "target_centre": np.zeros(2, dtype=np.float64),
}


@dataclass(frozen=True)
class AuthorityDecision:
    """What the lens can honour this frame, and what it has to leave visible.

    ``residual_shift`` is in output pixels and ``residual_zoom`` is a factor
    about ``target_centre``.  Both are zero/one while the machine is inside
    the travel, which is the ordinary lock.
    """

    mode: str
    travelled: float
    limit: float
    zoom_ratio: float
    residual_shift: np.ndarray
    residual_zoom: float
    target_centre: np.ndarray
    #: Which budget ran out: ``travel``, ``zoom``, ``travel+zoom`` or none.
    #: The two are separate budgets and only one of them is usually to blame,
    #: so the debug log records which one so a follow stretch can be tuned.
    reason: str = "lock"

    @property
    def limited(self) -> bool:
        return self.mode != "lock"

    def follow_matrix(self) -> np.ndarray:
        """Projective correction composed on top of the lock transform."""
        if not self.limited:
            return np.eye(3, dtype=np.float32)
        zoom = float(self.residual_zoom)
        shift_x, shift_y = (float(value) for value in self.residual_shift)
        centre_x, centre_y = (float(value) for value in self.target_centre)
        return np.array(
            [
                [zoom, 0.0, shift_x + centre_x * (1.0 - zoom)],
                [0.0, zoom, shift_y + centre_y * (1.0 - zoom)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Compose the bounded correction on top of a lock homography."""
        if matrix is None:
            return matrix
        if not self.limited:
            return matrix
        return (self.follow_matrix().astype(np.float64) @ np.asarray(matrix, dtype=np.float64)).astype(np.float32)

    def move_box(self, box):
        """Where a box that the lock pinned to its target really lands."""
        if box is None:
            return None
        if not self.limited:
            return box
        zoom = float(self.residual_zoom)
        shift_x, shift_y = (float(value) for value in self.residual_shift)
        centre_x, centre_y = (float(value) for value in self.target_centre)
        x0, y0, x1, y1 = (float(value) for value in box)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        corners *= zoom
        corners[:, 0] += shift_x + centre_x * (1.0 - zoom)
        corners[:, 1] += shift_y + centre_y * (1.0 - zoom)
        return (
            int(round(corners[:, 0].min())),
            int(round(corners[:, 1].min())),
            int(round(corners[:, 0].max())),
            int(round(corners[:, 1].max())),
        )


class AuthorityGovernor:
    """Turn a lock measurement into the part of it the lens can honour.

    The limit is latched and the leftover is smoothed, because an honest clamp
    on a noisy measurement is its own jitter source.  Once the travel runs out
    the lock stays in follow until the picture comes back comfortably inside
    the range, so a machine parked on the boundary cannot flip modes every few
    frames.  The leftover itself is low-passed, because the zoom a hand-held
    fisheye lock reports swings by several percent from frame to frame.
    """

    def __init__(self, limits: AuthorityLimits | None = None) -> None:
        self.limits = limits or AuthorityLimits()
        self.last_mode = "none"
        self.engaged = False
        self._shift_state = np.zeros(2, dtype=np.float64)
        self._zoom_state = 1.0

    def reset(self) -> None:
        """Drop the latch and the smoothing, e.g. once the lock is lost."""
        self.last_mode = "none"
        self.engaged = False
        self._shift_state = np.zeros(2, dtype=np.float64)
        self._zoom_state = 1.0

    def _alpha(self, dt: float | None) -> float:
        tau = float(getattr(self.limits, "smooth_seconds", 0.0) or 0.0)
        if dt is None or tau <= 0.0:
            return 1.0
        step = float(dt)
        if not step > 0.0:
            return 1.0
        return float(1.0 - math.exp(-min(step, 1.0) / tau))

    def decide(self, travel: dict | None, width: int, height: int,
               dt: float | None = None) -> AuthorityDecision:
        limit = self.limits.shift_limit(width, height)
        if not travel:
            self.reset()
            return AuthorityDecision(
                mode="lock",
                travelled=0.0,
                limit=limit,
                zoom_ratio=1.0,
                residual_shift=np.zeros(2, dtype=np.float32),
                residual_zoom=1.0,
                target_centre=np.zeros(2, dtype=np.float32),
            )
        shift = np.asarray(travel.get("shift", NO_TRAVEL["shift"]), dtype=np.float64).reshape(2)
        travelled = float(np.hypot(shift[0], shift[1]))
        output_scale = float(travel.get("output_scale", 1.0) or 1.0)
        zoom_ratio = float(travel.get("zoom", 1.0) or 1.0)
        centre = np.asarray(travel.get("target_centre", NO_TRAVEL["target_centre"]),
                            dtype=np.float64).reshape(2)
        hysteresis = float(min(max(float(getattr(self.limits, "hysteresis", 0.0) or 0.0), 0.0), 0.6))
        low = min(float(self.limits.min_zoom), float(self.limits.max_zoom))
        high = max(float(self.limits.min_zoom), float(self.limits.max_zoom))
        if not self.engaged:
            if (limit > 0.0 and travelled > limit) or zoom_ratio > high or zoom_ratio < low:
                self.engaged = True
        else:
            back_inside = limit <= 0.0 or travelled <= limit * (1.0 - hysteresis)
            zoom_inside = low * (1.0 + hysteresis) <= zoom_ratio <= high * (1.0 - hysteresis)
            if back_inside and zoom_inside:
                self.engaged = False
        if self.engaged:
            if travelled > limit > 0.0:
                # ``shift`` is the correction the crop asks for at the machine,
                # so the part the lens cannot pay is ``clamp(shift) - shift``:
                # the machine keeps the direction its own motion had.  Taking
                # it the other way round would flip the follow drift.
                residual = shift * ((limit - travelled) / travelled)
            else:
                residual = np.zeros(2, dtype=np.float64)
            zoom_clamped = self.limits.clamp_zoom(zoom_ratio)
            raw_zoom = zoom_ratio / zoom_clamped if zoom_clamped > 1e-6 else 1.0
            raw_shift = residual * output_scale
        else:
            raw_shift = np.zeros(2, dtype=np.float64)
            raw_zoom = 1.0
        alpha = self._alpha(dt)
        # Geometric smoothing for the scale, since it is a ratio, and linear
        # for the shift, so one time constant reads the same in both.
        self._zoom_state = float(self._zoom_state ** (1.0 - alpha) * raw_zoom ** alpha)
        self._shift_state = self._shift_state * (1.0 - alpha) + raw_shift * alpha
        residual_shift = self._shift_state.astype(np.float32)
        residual_zoom = float(self._zoom_state)
        limited = bool(np.any(np.abs(residual_shift) > 1e-6) or abs(residual_zoom - 1.0) > 1e-6)
        mode = "follow" if limited else "lock"
        self.last_mode = mode
        # Travel and zoom are separate budgets; saying which one ran out turns
        # a follow stretch into something that can be tuned.
        zoom_clamped_reason = self.limits.clamp_zoom(zoom_ratio)
        travel_limited = bool(limit > 0.0 and travelled > limit)
        zoom_limited = bool(abs(zoom_ratio - zoom_clamped_reason) > 1e-6)
        if travel_limited and zoom_limited:
            reason = "travel+zoom"
        elif travel_limited:
            reason = "travel"
        elif zoom_limited:
            reason = "zoom"
        elif limited:
            # Neither budget is out, but a leftover from the last excursion is
            # still draining through the smoothing, so the picture is neither
            # a clean lock nor a fresh follow.
            reason = "latched"
        else:
            reason = "lock"
        return AuthorityDecision(
            mode=mode,
            travelled=travelled,
            limit=limit,
            zoom_ratio=zoom_ratio,
            residual_shift=residual_shift,
            residual_zoom=residual_zoom,
            target_centre=centre.astype(np.float32),
            reason=reason,
        )
