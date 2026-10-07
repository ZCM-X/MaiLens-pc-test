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
    """Turn a lock measurement into the part of it the lens can honour."""

    def __init__(self, limits: AuthorityLimits | None = None) -> None:
        self.limits = limits or AuthorityLimits()
        self.last_mode = "none"

    def decide(self, travel: dict | None, width: int, height: int) -> AuthorityDecision:
        limit = self.limits.shift_limit(width, height)
        if not travel:
            self.last_mode = "lock"
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
        if travelled > limit > 0.0:
            # ``shift`` is the correction the crop asks for at the machine, so
            # the part the lens cannot pay is ``clamp(shift) - shift``: the
            # machine keeps the direction its own motion had.  Taking it the
            # other way round would flip the follow drift left-right.
            residual_source = shift * ((limit - travelled) / travelled)
        else:
            residual_source = np.zeros(2, dtype=np.float64)
        output_scale = float(travel.get("output_scale", 1.0) or 1.0)
        zoom_ratio = float(travel.get("zoom", 1.0) or 1.0)
        zoom_clamped = self.limits.clamp_zoom(zoom_ratio)
        residual_zoom = zoom_ratio / zoom_clamped if zoom_clamped > 1e-6 else 1.0
        centre = np.asarray(travel.get("target_centre", NO_TRAVEL["target_centre"]),
                            dtype=np.float64).reshape(2)
        residual_shift = (residual_source * output_scale).astype(np.float32)
        limited = bool(np.any(np.abs(residual_shift) > 1e-6) or abs(residual_zoom - 1.0) > 1e-6)
        mode = "follow" if limited else "lock"
        self.last_mode = mode
        return AuthorityDecision(
            mode=mode,
            travelled=travelled,
            limit=limit,
            zoom_ratio=zoom_ratio,
            residual_shift=residual_shift,
            residual_zoom=float(residual_zoom),
            target_centre=centre.astype(np.float32),
        )
