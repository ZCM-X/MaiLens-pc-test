#!/usr/bin/env python3
"""Machine lock for the fisheye phone footage (PC validation).

Stage 1  fisheye unwarp with the calibrated lens profile
Stage 2  HSV measurement of the inner play field and the purple button ring
Stage 3  zero-phase temporal smoothing of centre / size / roll
Stage 4  one affine warp per frame that pins the machine in the middle

The blogger's idea is used directly: the inner screen and the outer button
ring are measured separately, and the four gaps between them drive the virtual
camera.  Here those gaps collapse into (centre, size, roll), which is enough
to hold the cabinet still in the frame and to keep its apparent distance
constant while the operator moves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

RAW_FOCAL_CONST = 772.4089 / 4032.0

LENS = dict(
    crop=0.74,
    fov=106.4583,
    k1=0.0893163,
    k2=-0.0174637,
    center_x=0.501753869,
    center_y=0.499423644,
)

PURPLE_LOW = (125, 80, 60)
PURPLE_HIGH = (145, 255, 255)
SCREEN_LOW = (70, 60, 40)
SCREEN_HIGH = (115, 255, 255)


# --------------------------------------------------------------------- optics
def output_rays(width: int, height: int, crop: float, fov_deg: float) -> np.ndarray:
    virtual_focal = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xx - width * 0.5) / max(virtual_focal * crop, 1.0)
    y = (yy - height * 0.5) / max(virtual_focal * crop, 1.0)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def build_remap(out_w: int, out_h: int, raw_w: int, raw_h: int,
                lens: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rectilinear output grid -> raw fisheye pixel coordinates.

    ``source_focal`` must come from the *raw* sensor size, otherwise a
    downscaled output would silently rescale the lens model.
    """
    lens = dict(LENS if lens is None else lens)
    rays = output_rays(out_w, out_h, lens["crop"], lens["fov"])
    source_focal = max(raw_w, raw_h) * RAW_FOCAL_CONST
    radial = np.hypot(rays[..., 0], rays[..., 1])
    theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
    t2 = theta * theta
    theta_d = theta * (1.0 + lens["k1"] * t2 + lens["k2"] * t2 * t2)
    safe = np.maximum(radial, 1e-6)
    map_x = (lens["center_x"] * raw_w
             + source_focal * (rays[..., 0] / safe) * theta_d).astype(np.float32)
    map_y = (lens["center_y"] * raw_h
             + source_focal * (rays[..., 1] / safe) * theta_d).astype(np.float32)
    return map_x, map_y


def unwarp(frame: np.ndarray, remap: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(frame, remap[0], remap[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------- measurement
def _fit_circle(points: np.ndarray):
    """Least-squares circle with outlier rejection."""
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(3):
        if len(pts) < 5:
            return None
        x, y = pts[:, 0], pts[:, 1]
        design = np.c_[2.0 * x, 2.0 * y, np.ones(len(x))]
        solution, *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
        cx, cy = float(solution[0]), float(solution[1])
        radius = math.sqrt(max(solution[2] + cx * cx + cy * cy, 1e-6))
        residual = np.hypot(x - cx, y - cy) - radius
        rms = float(np.sqrt((residual ** 2).mean()))
        if len(pts) <= 5:
            return (cx, cy), radius, rms
        keep = np.abs(residual) <= max(2.5 * rms, 3.0)
        if keep.all():
            return (cx, cy), radius, rms
        pts = pts[keep]
    return None


def ring_roll(angles, step: float = 45.0) -> float | None:
    """Phase of the button ring against its uniform 45 degree layout.

    The buttons sit on an 8-fold symmetric ring, so the phase of the 8th
    harmonic is a much better estimator than any per-button average: it uses
    all eight centres at once and stays stable when one blob is missed.
    """
    if len(angles) < 4:
        return None
    radians = np.radians(np.asarray(angles, dtype=np.float64))
    order = max(1, int(round(360.0 / step)))
    phase = math.atan2(float(np.sin(order * radians).sum()),
                       float(np.cos(order * radians).sum()))
    return float(math.degrees(phase) / order)


def ellipse_frame(inner: dict):
    """Map image points into the inner screen's own unit-circle frame."""
    radians = math.radians(inner["angle"])
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    rotate = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    scale = np.array([2.0 / inner["major"], 2.0 / inner["minor"]])
    centre = np.array(inner["center"], dtype=np.float64)

    def normalise(points):
        delta = np.asarray(points, dtype=np.float64) - centre
        return (delta @ rotate.T) * scale
    return normalise


# The buttons ring sits at roughly 1.25 screen radii; artwork specks live well
# inside that and other purple cabinets well outside it, so a radial band in
# the screen's own frame is what separates the eight real buttons from the
# junk.  A raw pixel-space band cannot do this, because the ring is about 30%
# wider along one axis than the other.
BUTTON_BAND = (0.85, 1.95)

# How hard the eight button centres pull on the shared shape against the 72
# screen-contour samples.  Sweeping it on the hand-held clip gives:
#   0.15 -> buttons round (3.7%), screen squash visible (7.6% radius error)
#   0.50 -> buttons 7.7%, screen 3.5%     <- the balanced default
#   2.00 -> buttons 9.1%, screen 2.5%
# The two rings genuinely disagree this much, so it has to be a taste call.
RING_WEIGHT = 0.5


def measure_button_ring(frame: np.ndarray, inner: dict | None = None) -> dict | None:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_LOW, PURPLE_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs = [(int(stats[i, cv2.CC_STAT_AREA]), centroids[i])
             for i in range(1, count)]
    blobs.sort(key=lambda item: -item[0])
    blobs = [b for b in blobs if b[0] >= 12]
    if inner is not None:
        normalise = ellipse_frame(inner)
        banded = []
        for area, point in blobs:
            radius = float(np.linalg.norm(normalise(point)))
            if BUTTON_BAND[0] < radius < BUTTON_BAND[1]:
                banded.append((area, point))
        if len(banded) >= 5:
            blobs = banded
    blobs = blobs[:12]
    if len(blobs) < 5:
        return None
    points = np.array([b[1] for b in blobs], dtype=np.float64)
    fit = _fit_circle(points)
    if fit is None:
        return None
    (cx, cy), radius, rms = fit
    if radius < 0.04 * min(width, height):
        return None
    angles = np.degrees(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    major = 2.0 * radius
    if len(points) >= 5:
        # fitEllipse is unstable on a sparse ring; only trust it when it stays
        # close to the circle fit, otherwise it invents a huge major axis.
        ellipse = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
        candidate = float(max(ellipse[1]))
        if 0.9 * major <= candidate <= 1.6 * major:
            major = candidate
    return {
        "center": [cx, cy],
        "radius": float(radius),
        "major": major,
        "rms": rms,
        "count": int(len(points)),
        "angles": [float(a) for a in angles],
        "points": points.tolist(),
        "roll": ring_roll(angles),
    }


def fit_ellipse_robust(contour: np.ndarray, rounds: int = 3, band: float = 0.06):
    """Ellipse fit that survives the character art punching holes in the mask.

    A missing chunk of the cyan mask drags ``cv2.fitEllipse`` towards itself,
    which is what makes the screen centre wander while a song plays.  Points
    are scored by their radius in the current ellipse's own frame and the
    stragglers are dropped before refitting.
    """
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    (ex, ey), (axis_a, axis_b), angle = cv2.fitEllipse(points)
    for _ in range(rounds):
        if len(points) < 60:
            break
        radians = math.radians(angle)
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        rotate = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        scale = np.array([2.0 / max(axis_a, 1e-6), 2.0 / max(axis_b, 1e-6)])
        delta = points[:, 0, :].astype(np.float64) - np.array([ex, ey])
        radius = np.linalg.norm((delta @ rotate.T) * scale, axis=1)
        keep = np.abs(radius - 1.0) <= band
        if keep.sum() < max(24, int(0.5 * len(points))) or keep.all():
            break
        points = points[keep]
        (ex, ey), (axis_a, axis_b), angle = cv2.fitEllipse(points)
    return points[:, 0, :].astype(np.float64), (ex, ey), (axis_a, axis_b), angle


def measure_inner_screen(frame: np.ndarray) -> dict | None:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SCREEN_LOW, SCREEN_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None
    best = None
    for index in range(1, count):
        x, y, bw, bh = (int(stats[index, k]) for k in range(4))
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 0.004 * width * height:
            continue
        fill = area / max(bw * bh, 1)
        aspect = bw / max(bh, 1)
        # The play field is a fat disc; the cabinet's top LCD is a wide slab.
        if fill < 0.55 or not (0.6 < aspect < 1.7):
            continue
        score = area * (1.0 - abs(math.log(max(aspect, 1e-3))))
        if best is None or score > best[0]:
            best = (score, index)
    if best is None:
        return None
    component = (labels == best[1]).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour, (ex, ey), (axis_a, axis_b), angle = fit_ellipse_robust(contour)
    major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
    step = max(1, len(contour) // 72)
    return {
        "center": [float(ex), float(ey)],
        "major": float(major),
        "minor": float(minor),
        "angle": float(angle),
        "area": float(len(contour)),
        # Kept for the joint two-ring fit: the screen edge on its own cannot
        # pin down the ellipse orientation, because it is nearly circular.
        "points": contour[::step].astype(np.float64).tolist(),
    }


def measure_frame(frame: np.ndarray) -> dict:
    inner = measure_inner_screen(frame)
    outer = measure_button_ring(frame, inner)
    joint = None
    if inner is not None and outer is not None:
        joint = fit_two_ring(inner["points"], outer["points"])
    return {"inner": inner, "outer": outer, "joint": joint}


# ------------------------------------------------------------ joint two-ring
#
# The blogger's own correction rule -- "a distortion free picture has equal
# gaps between the outer buttons and the inner screen on all four sides" -- is
# exactly the statement that the button circle and the screen circle are
# concentric after the warp.  The screen edge alone cannot deliver that: it is
# almost a circle, so its ellipse orientation is barely observable and the
# buttons come out skewed.  Fitting both rings *at once* against one affine
# pins the squash direction down, which is what straightens the buttons.


def _chol_q(params: np.ndarray) -> np.ndarray:
    """Shape matrix Q = L L^T from an unconstrained parameter triple."""
    l1 = math.exp(float(params[0]))
    off = float(params[1])
    l2 = math.exp(float(params[2]))
    return np.array([[l1 * l1, l1 * off],
                     [l1 * off, off * off + l2 * l2]])


def _q_chol_params(q: np.ndarray) -> np.ndarray:
    l1 = math.sqrt(max(float(q[0, 0]), 1e-12))
    off = float(q[0, 1]) / l1
    l2 = math.sqrt(max(float(q[1, 1]) - off * off, 1e-12))
    return np.array([math.log(l1), off, math.log(l2)])


def _lm_solve(func, x0: np.ndarray, iterations: int = 40, lam0: float = 1e-3):
    """Small Levenberg-Marquardt with a numeric Jacobian."""
    x = np.asarray(x0, dtype=np.float64).copy()
    residual = func(x)
    cost = float(residual @ residual)
    lam = lam0
    for _ in range(iterations):
        jacobian = np.empty((len(residual), len(x)))
        for k in range(len(x)):
            step = 1e-6 * max(1.0, abs(x[k]))
            trial = x.copy()
            trial[k] += step
            jacobian[:, k] = (func(trial) - residual) / step
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        improved = False
        for _ in range(8):
            damped = normal + lam * np.diag(np.maximum(np.diag(normal), 1e-9))
            try:
                delta = np.linalg.solve(damped, -gradient)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            trial = x + delta
            trial_residual = func(trial)
            trial_cost = float(trial_residual @ trial_residual)
            if trial_cost < cost:
                x, residual, cost = trial, trial_residual, trial_cost
                lam = max(lam * 0.3, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved or cost < 1e-13:
            break
    return x, cost, residual


def matrix_sqrt(q: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(q, dtype=np.float64))
    values = np.maximum(values, 1e-12)
    return (vectors * np.sqrt(values)) @ vectors.T


def fit_two_ring(inner_points, outer_points, inner_weight: float | None = None,
                 robust_rounds: int = 3):
    """One affine that makes the screen a circle and the buttons concentric.

    Unknowns are the shape matrix Q = M^T M, the shared centre and the squared
    button radius.  The screen radius is normalised to one, so ``M`` maps the
    measured frame into the cabinet's own metric up to a rotation.

    The screen contour is redrawn every frame, so a few of its samples always
    sit on a missing chunk of the mask or on the character art.  Those rounds
    of reweighting are what keep one bad sample from tilting the cabinet.
    """
    inner = np.asarray(inner_points, dtype=np.float64)
    outer = np.asarray(outer_points, dtype=np.float64)
    if len(inner) < 12 or len(outer) < 5:
        return None
    if inner_weight is None:
        inner_weight = RING_WEIGHT * math.sqrt(len(outer) / len(inner))

    centre = inner.mean(axis=0)
    delta = inner - centre
    covariance = (delta.T @ delta) / len(delta)
    q0 = 2.0 * np.linalg.inv(covariance + 1e-9 * np.eye(2))
    q0 /= np.mean(np.einsum("ij,jk,ik->i", delta, q0, delta))
    outer_delta = outer - centre
    rho0 = max(float(np.mean(np.einsum("ij,jk,ik->i", outer_delta, q0, outer_delta))), 1e-6)
    x0 = np.concatenate([_q_chol_params(q0), centre, [math.log(rho0)]])

    inner_scale = np.ones(len(inner))
    outer_scale = np.ones(len(outer))

    def make_residuals():
        def residuals(params):
            q = _chol_q(params)
            shared = params[3:5]
            rho2 = math.exp(float(params[5]))
            inner_delta = inner - shared
            values = np.einsum("ij,jk,ik->i", inner_delta, q, inner_delta)
            inner_residual = (values - 1.0) * inner_weight * inner_scale
            outer_delta = outer - shared
            values = np.einsum("ij,jk,ik->i", outer_delta, q, outer_delta)
            return np.concatenate([inner_residual,
                                   (values / rho2 - 1.0) * outer_scale])
        return residuals

    solution = x0
    for _ in range(max(1, robust_rounds)):
        inner_scale = np.ones(len(inner))
        outer_scale = np.ones(len(outer))
        solution, cost, residual = _lm_solve(make_residuals(), solution)
        # Tukey-style hard rejection against the spread of each block.
        inner_residual = residual[:len(inner)]
        outer_residual = residual[len(inner):]
        inner_limit = max(4.0 * float(np.median(np.abs(inner_residual))), 0.02)
        outer_limit = max(4.0 * float(np.median(np.abs(outer_residual))), 0.02)
        inner_scale = (np.abs(inner_residual) <= inner_limit).astype(np.float64)
        outer_scale = (np.abs(outer_residual) <= outer_limit).astype(np.float64)
        if inner_scale.sum() < 12 or outer_scale.sum() < 5:
            break

    solution, cost, residual = _lm_solve(make_residuals(), solution)
    q = _chol_q(solution)
    count = len(residual)
    return {
        # Only the three shape parameters: the centre and the button radius are
        # carried separately so the series can be filtered on their own.
        "chol": [float(v) for v in solution[:3]],
        "q": q.tolist(),
        "center": [float(solution[3]), float(solution[4])],
        "rho": float(math.sqrt(math.exp(float(solution[5])))),
        "rms": float(math.sqrt(cost / count)),
        "buttons": int(len(outer)),
    }


# ------------------------------------------------------------------- smoothing
def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (axis / sigma) ** 2)
    return kernel / kernel.sum()


def fill_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    index = np.arange(len(values))
    good = np.isfinite(values)
    if not good.any():
        return np.full(len(values), np.nan)
    return np.interp(index, index[good], values[good])


def rolling_max(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out[i] = np.max(values[lo:hi])
    return out


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase smoothing: a symmetric FIR kernel has no group delay."""
    values = fill_nan(values)
    if sigma <= 0.0 or len(values) < 3 or not np.isfinite(values).all():
        return values
    # Median pre-pass so a single-frame detector spike cannot smear into a
    # visible wobble.  Both passes are symmetric, so neither adds lag.
    if len(values) >= 5:
        padded = np.pad(values, (2, 2), mode="edge")
        stacked = np.stack([padded[k:k + len(values)] for k in range(5)])
        values = np.median(stacked, axis=0)
    kernel = _gaussian_kernel(sigma)
    radius = (len(kernel) - 1) // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_periodic(values: np.ndarray, sigma: float, period: float = 45.0) -> np.ndarray:
    """Smooth an angle that is only observable modulo ``period``.

    The button ring has 8-fold symmetry, so its phase repeats every 45 deg.
    Unwrapping first keeps a wrap-around from being read as a huge rotation.
    """
    values = fill_nan(values)
    if not np.isfinite(values).all():
        return np.zeros(len(values))
    scale = 2.0 * math.pi / period
    unwrapped = np.unwrap(values * scale) / scale
    smoothed = smooth(unwrapped, max(float(sigma), 6.0))
    return ((smoothed + period * 0.5) % period) - period * 0.5


# ------------------------------------------------------------------- the lock
def state_series(rows: list[dict]) -> dict:
    """Turn raw measurements into per-frame centre / size / roll values."""
    inner_major = np.array([r["inner"]["major"] if r["inner"] else np.nan for r in rows])
    outer_major = np.array([
        r["outer"]["major"] if r["outer"] and r["outer"]["major"]
        else (2.0 * r["outer"]["radius"] if r["outer"] else np.nan)
        for r in rows
    ])
    both = np.isfinite(inner_major) & np.isfinite(outer_major)
    ratio = float(np.median(outer_major[both] / inner_major[both])) if both.sum() >= 5 else None

    count = len(rows)
    cx = np.full(count, np.nan)
    cy = np.full(count, np.nan)
    size = np.full(count, np.nan)
    minor = np.full(count, np.nan)
    angle = np.full(count, np.nan)
    outer_cx = np.full(count, np.nan)
    outer_cy = np.full(count, np.nan)
    roll = np.full(count, np.nan)
    for i, row in enumerate(rows):
        inner, outer = row["inner"], row["outer"]
        if inner is not None:
            cx[i], cy[i] = inner["center"]
            # The major axis is the tilt-invariant distance signal.
            size[i] = inner["major"]
            minor[i] = inner["minor"]
            angle[i] = inner["angle"]
        elif outer is not None and ratio:
            cx[i], cy[i] = outer["center"]
            size[i] = outer_major[i] / ratio
        if outer is not None:
            outer_cx[i], outer_cy[i] = outer["center"]
            if outer.get("roll") is not None:
                roll[i] = outer["roll"]
    return dict(cx=cx, cy=cy, size=size, minor=minor, angle=angle, roll=roll,
                outer_cx=outer_cx, outer_cy=outer_cy, ratio=ratio)


def lock_matrices(series: dict, small_w: int, small_h: int, lock_fill: float,
                  sigma: float, roll_gain: float, roll_sign: float,
                  max_zoom: float, margin: float = 1.0) -> tuple[np.ndarray, dict]:
    """Place the machine in the middle of the working canvas.

    ``margin`` is the extra fisheye field unwarped around the frame.  The
    working canvas is ``margin`` times the delivered frame, and the final
    result is its centre crop.  The headroom is what lets a cabinet sitting
    near the raw frame edge be moved to the middle without the warp having to
    invent pixels; ``need`` below is the fallback when even that is not enough.
    """
    count = len(series["size"])
    if not np.isfinite(series["size"]).any():
        return np.zeros((count, 2, 3), dtype=np.float32), dict(zoom=np.zeros(count))
    cx = smooth(series["cx"], sigma)
    cy = smooth(series["cy"], sigma)
    size = smooth(series["size"], sigma)
    roll = smooth_periodic(series["roll"], sigma) if roll_gain else np.zeros(count)

    frame_frac = 1.0 / max(float(margin), 1.0)
    # The delivered frame is the centre crop, so the machine must fill
    # lock_fill of that crop rather than of the whole canvas.
    target = float(lock_fill) * small_w * frame_frac
    zoom = target / np.maximum(size, 1e-6)
    zoom = np.clip(zoom, 1.0, max(max_zoom, 1.0))
    zoom = smooth(zoom, sigma)

    half_w = small_w * frame_frac * 0.5
    half_h = small_h * frame_frac * 0.5
    # A rotated sampling window reaches further than an axis-aligned one, so
    # the room check has to use its rotated bounding box.
    angle = np.radians(-float(roll_gain) * roll_sign * roll)
    reach_x = half_w * np.abs(np.cos(angle)) + half_h * np.abs(np.sin(angle))
    reach_y = half_w * np.abs(np.sin(angle)) + half_h * np.abs(np.cos(angle))
    room_x = np.minimum(cx, small_w - cx)
    room_y = np.minimum(cy, small_h - cy)
    need = np.maximum(reach_x / np.maximum(room_x, 1e-3),
                      reach_y / np.maximum(room_y, 1e-3))
    need = rolling_max(need, int(max(2.0 * sigma, 4.0)))
    zoom = smooth(np.maximum(zoom, need * 1.03), 2.0)
    zoom = np.maximum(zoom, need * 1.01)
    zoom = np.clip(zoom, 1.0, max(max_zoom, 1.0))

    matrices = np.zeros((count, 2, 3), dtype=np.float32)
    out_cx, out_cy = small_w * 0.5, small_h * 0.5
    for i in range(count):
        if not (np.isfinite(cx[i]) and np.isfinite(cy[i]) and np.isfinite(zoom[i])):
            matrices[i] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            continue
        angle = math.radians(-float(roll_gain) * roll_sign * float(roll[i]))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        scale = float(zoom[i])
        linear = np.array([[scale * cos_a, -scale * sin_a],
                           [scale * sin_a, scale * cos_a]], dtype=np.float64)
        tx = out_cx - (linear[0, 0] * cx[i] + linear[0, 1] * cy[i])
        ty = out_cy - (linear[1, 0] * cx[i] + linear[1, 1] * cy[i])
        matrices[i] = np.array([[linear[0, 0], linear[0, 1], tx],
                                [linear[1, 0], linear[1, 1], ty]], dtype=np.float32)
    return matrices, dict(zoom=zoom, cx=cx, cy=cy, roll=roll, need=need)


def scale_matrix(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Move a matrix measured in the small frame into the render frame."""
    out = matrix.copy()
    out[0, 2] *= scale
    out[1, 2] *= scale
    return out


def scale_homography(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Conjugate a source->dest homography by a uniform pixel rescale."""
    out = matrix.copy()
    out[0, 2] *= scale
    out[1, 2] *= scale
    out[2, 0] /= scale
    out[2, 1] /= scale
    return out


def world_lock(values: np.ndarray, deadband, tau: float, tau_fast: float,
               fast_ratio: float = 4.0, dt: float = 1.0 / 60.0) -> np.ndarray:
    """Hold the machine in the world instead of chasing every measurement.

    Anything inside ``deadband`` is detector noise and is ignored outright, so
    the cabinet sits dead still while the operator's hands shake.  Bigger
    errors are followed -- gently while they are ambiguous, briskly once they
    clearly mean the operator moved rather than the detector twitched.
    """
    values = fill_nan(values)
    if len(values) < 2 or not np.isfinite(values).all():
        return values
    band = np.broadcast_to(np.asarray(deadband, dtype=np.float64), values.shape)
    out = np.empty_like(values)
    anchor = float(values[0])
    for i in range(len(values)):
        error = float(values[i]) - anchor
        excess = math.copysign(max(abs(error) - float(band[i]), 0.0), error)
        if excess:
            rate = tau if abs(excess) <= fast_ratio * float(band[i]) else tau_fast
            anchor += excess * (1.0 - math.exp(-dt / max(rate, 1e-4)))
        out[i] = anchor
    return out


def stabilise(values: np.ndarray, sigma: float, lock: dict | None = None,
              dt: float = 1.0 / 60.0, deadband=0.0) -> np.ndarray:
    if lock is None:
        return smooth(values, sigma)
    return world_lock(values, deadband, lock["tau"], lock["tau_fast"],
                      lock["fast_ratio"], dt)


def joint_position(rows: list[dict]) -> dict:
    """Pull the per-frame two-ring solution out of the raw measurements."""
    count = len(rows)
    cx = np.full(count, np.nan)
    cy = np.full(count, np.nan)
    chol = np.full((count, 3), np.nan)
    rho = np.full(count, np.nan)
    for i, row in enumerate(rows):
        joint = row.get("joint")
        if joint is None:
            continue
        cx[i], cy[i] = joint["center"]
        chol[i] = joint["chol"]
        rho[i] = joint["rho"]
    return dict(cx=cx, cy=cy, chol=chol, rho=rho)


def shape_maps(series: dict, joint: dict | None, sigma: float, rectify: float,
               lock: dict | None, dt: float, deadband_frac: float,
               shape_sigma: float | None = None):
    """Per-frame centre plus the linear map from measurement to screen metric.

    With ``joint`` the shape comes from the two-ring fit, so the button ring
    lands concentric with the screen.  Without it the old single-ellipse path
    is kept as the fallback.

    The two halves of the warp are filtered differently on purpose.  Where the
    cabinet *is* has to follow the measurement, otherwise the cabinet slides
    around inside the frame.  How the cabinet is *squashed*, on the other hand,
    barely changes while the operator moves, so it is worth a much longer
    window: that is what stops the cabinet from pulsing and rolling with the
    detector noise.
    """
    count = len(series["size"])
    cx = np.full(count, np.nan)
    cy = np.full(count, np.nan)
    maps = np.tile(np.eye(2), (count, 1, 1))
    radius = np.full(count, np.nan)
    blend = float(np.clip(rectify, 0.0, 1.0))
    shape_sigma = float(shape_sigma) if shape_sigma else max(sigma, 12.0)

    if joint is not None and np.isfinite(joint["rho"]).any():
        chol = np.column_stack([
            stabilise(joint["chol"][:, k], shape_sigma, lock, dt)
            for k in range(3)
        ])
        raw = np.array([_chol_q(joint["chol"][i])
                        if np.isfinite(joint["chol"][i]).all() else np.eye(2)
                        for i in range(count)])
        # Geometric mean screen radius in measured pixels, used to size the
        # deadband in units the operator would recognise.
        spread = np.array([math.sqrt(abs(np.linalg.det(m))) for m in raw])
        radius = np.where(spread > 0.0, 1.0 / np.sqrt(np.maximum(spread, 1e-9)), np.nan)
        band = np.where(np.isfinite(radius), deadband_frac * radius, 0.0)
        cx = stabilise(joint["cx"], sigma, None, dt, band)
        cy = stabilise(joint["cy"], sigma, None, dt, band)
        for i in range(count):
            if not np.isfinite(chol[i]).all():
                continue
            m = matrix_sqrt(_chol_q(chol[i]))
            isotropic = math.sqrt(abs(np.linalg.det(m)))
            maps[i] = blend * m + (1.0 - blend) * isotropic * np.eye(2)
        usable = np.isfinite(chol).all(axis=1)
        return maps, cx, cy, radius, usable

    a = stabilise(series["size"] * 0.5, shape_sigma, lock, dt,
                  deadband_frac * series["size"] * 0.5)
    semi_minor = stabilise(series["minor"] * 0.5, shape_sigma, lock, dt)
    angle = smooth_periodic(series["angle"], shape_sigma, 180.0)
    cx = stabilise(series["cx"], sigma, None, dt, deadband_frac * series["size"] * 0.5)
    cy = stabilise(series["cy"], sigma, None, dt, deadband_frac * series["size"] * 0.5)
    radius = a
    usable = np.isfinite(series["minor"]) & np.isfinite(series["angle"])
    semi_minor = np.where(np.isfinite(semi_minor), semi_minor, a)
    geometric = 1.0 / np.sqrt(np.maximum(a * semi_minor, 1e-9))
    inv_a = blend / np.maximum(a, 1e-6) + (1.0 - blend) * geometric
    inv_b = blend / np.maximum(semi_minor, 1e-6) + (1.0 - blend) * geometric
    for i in range(count):
        if not (np.isfinite(inv_a[i]) and np.isfinite(inv_b[i]) and np.isfinite(angle[i])):
            continue
        radians = math.radians(float(angle[i]))
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        # cv2.fitEllipse: `angle` is the rotation of the width axis.
        rotate = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        maps[i] = np.diag([inv_a[i], inv_b[i]]) @ rotate
    return maps, cx, cy, radius, usable


def rectify_matrices(series: dict, small_w: int, small_h: int, margin: float,
                     lock_fill: float, sigma: float,
                     rectify: float, roll_gain: float, roll_sign: float,
                     max_k: float = 8.0, joint: dict | None = None,
                     lock: dict | None = None, fps: float = 60.0,
                     deadband_frac: float = 0.0,
                     shape_sigma: float | None = None):
    """Turn the inner play field back into a circle and pin it to the middle.

    The screen is a circle on the cabinet face, so its image is an ellipse
    whose aspect ratio is the cosine of the tilt.  Undoing that ellipse is the
    whole tilt correction: the two-concentric-circle pencil in the same clip
    reports its rank-1 member at (0, 0, 1) on every frame, which is exactly the
    statement that the remaining perspective term is zero and only the affine
    squash needs removing.

    ``rectify`` blends between the plain similarity lock (0) and the full
    un-squash (1).  The blend interpolates the *inverse* semi-axes so the
    linear part stays positive definite for every value in between.
    """
    count = len(series["size"])
    dt = 1.0 / max(float(fps), 1e-6)
    maps, cx, cy, radius, usable = shape_maps(
        series, joint, sigma, rectify, lock, dt, deadband_frac, shape_sigma)
    if not np.isfinite(cx).any():
        return None, None
    roll = smooth_periodic(series["roll"], sigma) if roll_gain else np.zeros(count)

    # Everything here lives in the measured frame's units.  The delivered
    # frame is the centre 1/margin of the canvas, so a lock_fill wide screen
    # is lock_fill * small_w / margin pixels across in this frame.
    margin = max(float(margin), 1.0)
    base_k = float(lock_fill) * small_w * 0.5 / margin
    half_out = (small_w / (2.0 * margin), small_h / (2.0 * margin))
    corners = [(-half_out[0], -half_out[1]), (half_out[0], -half_out[1]),
               (-half_out[0], half_out[1]), (half_out[0], half_out[1])]

    linear = np.zeros((count, 2, 2))
    offset = np.zeros((count, 2))
    need = np.ones(count)
    ok = np.zeros(count, dtype=bool)
    for i in range(count):
        if not (np.isfinite(cx[i]) and np.isfinite(cy[i]) and usable[i]):
            linear[i] = np.eye(2)
            continue
        ok[i] = True
        m0 = maps[i]
        centre = np.array([cx[i], cy[i]])
        t0 = -m0 @ centre
        down = m0 @ np.array([0.0, 1.0])
        # Keep the cabinet's own up direction pointing up the delivered frame.
        phi = math.radians(90.0 - math.degrees(math.atan2(down[1], down[0])))
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        upright = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
        if roll_gain:
            spin = math.radians(-float(roll_gain) * roll_sign * float(roll[i]))
            cos_s, sin_s = math.cos(spin), math.sin(spin)
            upright = np.array([[cos_s, -sin_s], [sin_s, cos_s]]) @ upright
        linear[i] = upright @ m0
        offset[i] = upright @ t0
        # How far the crop corners reach, expressed as the scale they need.
        inverse = np.linalg.inv(linear[i])
        for dx, dy in corners:
            reach = inverse @ np.array([dx, dy])
            if reach[0] > 0 and small_w - centre[0] > 1e-6:
                need[i] = max(need[i], reach[0] / (small_w - centre[0]))
            elif reach[0] < 0 and centre[0] > 1e-6:
                need[i] = max(need[i], -reach[0] / centre[0])
            if reach[1] > 0 and small_h - centre[1] > 1e-6:
                need[i] = max(need[i], reach[1] / (small_h - centre[1]))
            elif reach[1] < 0 and centre[1] > 1e-6:
                need[i] = max(need[i], -reach[1] / centre[1])

    need = rolling_max(need, int(max(2.0 * sigma, 4.0))) * 1.03
    scale = smooth(np.maximum(base_k, need), 2.0)
    scale = np.maximum(scale, np.minimum(need, max_k * base_k))
    scale = np.clip(scale, base_k * 0.25, max_k * base_k)

    centre_small = np.array([small_w * 0.5, small_h * 0.5])
    matrices = np.zeros((count, 3, 3))
    for i in range(count):
        matrices[i, :2, :2] = scale[i] * linear[i]
        matrices[i, :2, 2] = scale[i] * offset[i] + centre_small
        matrices[i, 2, 2] = 1.0
    return matrices, dict(scale=scale, need=need, usable=ok, cx=cx, cy=cy,
                          radius=radius, maps=maps)


# ------------------------------------------------------------------------- io
def measure_clip(path: Path, measure_scale: float, max_frames: int | None,
                 margin: float = 1.0, ring_weight: float | None = None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    raw_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    ring_weight = RING_WEIGHT if ring_weight is None else float(ring_weight)
    cache = path.with_suffix(path.suffix + ".measure.json")
    if max_frames is None and cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if (payload.get("scale") == measure_scale and payload.get("margin") == margin
                and payload.get("raw_w") == raw_w and payload.get("raw_h") == raw_h
                and payload.get("ring_weight") == ring_weight):
            capture.release()
            return dict(raw_w=raw_w, raw_h=raw_h, fps=payload["fps"],
                        small_w=payload["small_w"], small_h=payload["small_h"],
                        rows=payload["rows"], count=len(payload["rows"]),
                        cached=True)
    small_w = max(2, int(round(raw_w * measure_scale)))
    small_h = max(2, int(round(raw_h * measure_scale)))
    # Measure on exactly the same (wider) field the render canvas uses, so a
    # measured pixel means the same ray in both passes.
    lens = dict(LENS, crop=LENS["crop"] / max(float(margin), 1.0))
    remap = build_remap(small_w, small_h, raw_w, raw_h, lens)
    rows: list[dict] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rows.append(measure_frame(unwarp(frame, remap)))
        if max_frames and len(rows) >= max_frames:
            break
    capture.release()
    meta = dict(raw_w=raw_w, raw_h=raw_h, fps=fps, small_w=small_w,
                small_h=small_h, rows=rows, count=len(rows), cached=False)
    if max_frames is None:
        cache.write_text(json.dumps(dict(name=str(path), scale=measure_scale,
                                         margin=margin, fps=fps, raw_w=raw_w,
                                         raw_h=raw_h, small_w=small_w,
                                         small_h=small_h, ring_weight=ring_weight,
                                         rows=rows)),
                         encoding="utf-8")
    return meta


def lock_report(rows: list[dict], project, middle: np.ndarray) -> dict:
    """How well the delivered frame holds up, measured on the warp itself.

    ``ring cv`` is the spread of the button radii around the middle of the
    frame: that is the number to watch when tuning ``--ring-weight``, because
    it is exactly the "the buttons must not be skewed" criterion.  ``machine``
    is how far the cabinet wanders from the middle of the frame.
    """
    machine, ring_cv, screen_cv, gap_lr, gap_tb = [], [], [], [], []
    for i, row in enumerate(rows):
        inner, outer = row["inner"], row["outer"]
        if inner is None or outer is None:
            continue
        inner_points = np.array([project(i, p[0], p[1]) for p in inner["points"]])
        outer_points = np.array([project(i, p[0], p[1]) for p in outer["points"]])
        inner_radius = np.linalg.norm(inner_points - middle, axis=1)
        outer_radius = np.linalg.norm(outer_points - middle, axis=1)
        # The contour's own centroid is biased by the hole the character art
        # punches in the mask, so the cabinet's position comes from the fitted
        # centre instead -- the same point the warp actually pins down.
        anchor = row.get("joint") or inner
        machine.append(float(np.linalg.norm(
            project(i, anchor["center"][0], anchor["center"][1]) - middle)))
        ring_cv.append(float(outer_radius.std() / max(outer_radius.mean(), 1e-6)))
        screen_cv.append(float(inner_radius.std() / max(inner_radius.mean(), 1e-6)))
        gap = outer_radius - inner_radius.mean()
        angles = np.degrees(np.arctan2(outer_points[:, 1] - middle[1],
                                       outer_points[:, 0] - middle[0]))
        side = dict(left=angles > 135, right=(angles <= 45) & ~(angles >= 45),
                    top=(angles <= -45) & (angles > -135), bottom=angles >= 45)
        if all(mask.any() for mask in side.values()):
            edges = {name: float(gap[mask].mean()) for name, mask in side.items()}
            gap_lr.append(abs(edges["left"] - edges["right"]))
            gap_tb.append(abs(edges["top"] - edges["bottom"]))

    def q(values):
        return float(np.median(values)) if values else float("nan")

    return dict(machine=q(machine), ring_cv=q(ring_cv), screen_cv=q(screen_cv),
                gap_lr=q(gap_lr), gap_tb=q(gap_tb))


def _draw_overlay(frame: np.ndarray, target: tuple[float, float],
                  error_px: float) -> None:
    """Show where the measured machine actually lands after the lock.

    A red circle sitting on the green cross means the cabinet is glued to the
    middle; the gap between them is the residual lock error in output pixels.
    """
    height, width = frame.shape[:2]
    cv2.drawMarker(frame, (width // 2, height // 2), (0, 255, 0),
                   cv2.MARKER_CROSS, 48, 2)
    point = (int(round(target[0])), int(round(target[1])))
    cv2.circle(frame, point, 22, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(frame, point, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(frame, f"lock err {error_px:5.1f}px", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)


def contact_sheet(path: Path, output: Path, columns: int = 4,
                  rows_wanted: int = 3, tile_height: int = 360) -> None:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = columns * rows_wanted
    picks = np.linspace(0, max(total - 1, 0), wanted).astype(int)
    tiles = []
    for pick in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(pick))
        ok, frame = capture.read()
        if not ok:
            continue
        scale = tile_height / frame.shape[0]
        tiles.append(cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), tile_height)))
    capture.release()
    if not tiles:
        return
    width = min(tile.shape[1] for tile in tiles)
    tiles = [tile[:, :width] for tile in tiles]
    while len(tiles) < wanted:
        tiles.append(np.zeros_like(tiles[0]))
    grid = [np.hstack(tiles[i * columns:(i + 1) * columns]) for i in range(rows_wanted)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(grid))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--measure-scale", type=float, default=0.25)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--lock-fill", type=float, default=0.72)
    parser.add_argument("--margin", type=float, default=1.4)
    parser.add_argument("--rectify", type=float, default=1.0)
    parser.add_argument("--fit", choices=("joint", "ellipse"), default="joint",
                        help="joint: two-ring fit; ellipse: inner screen only")
    parser.add_argument("--ring-weight", type=float, default=RING_WEIGHT,
                        help="button-ring pull on the shared shape (0.15..2)")
    parser.add_argument("--smooth-sigma", type=float, default=0.5,
                        help="centre window; keep it short so the cabinet stays put")
    parser.add_argument("--shape-sigma", type=float, default=14.0,
                        help="squash/scale window; long enough to kill the pulse")
    parser.add_argument("--world-lock", action="store_true",
                        help="causal anchor for the squash/scale (live-safe)")
    parser.add_argument("--lock-deadband", type=float, default=0.01,
                        help="ignored error, as a fraction of the screen radius")
    parser.add_argument("--lock-tau", type=float, default=0.35)
    parser.add_argument("--lock-tau-fast", type=float, default=0.10)
    parser.add_argument("--lock-fast-ratio", type=float, default=3.0)
    parser.add_argument("--roll-gain", type=float, default=0.0)
    parser.add_argument("--roll-sign", type=float, default=1.0)
    parser.add_argument("--max-zoom", type=float, default=3.5)
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--no-fisheye", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    global RING_WEIGHT
    RING_WEIGHT = float(args.ring_weight)
    margin = max(float(args.margin), 1.0)
    meta = measure_clip(args.input, args.measure_scale, args.max_frames, margin,
                        RING_WEIGHT)
    rows = meta["rows"]
    series = state_series(rows)
    out_w = max(2, int(round(meta["raw_w"] * args.output_scale)))
    out_h = max(2, int(round(meta["raw_h"] * args.output_scale)))
    canvas_w = max(out_w, int(round(out_w * margin)))
    canvas_h = max(out_h, int(round(out_h * margin)))
    scale = canvas_w / meta["small_w"]
    crop_x = (canvas_w - out_w) // 2
    crop_y = (canvas_h - out_h) // 2
    base_k = float(args.lock_fill) * out_w * 0.5
    base_k_small = float(args.lock_fill) * meta["small_w"] * 0.5 / margin

    rectified, rect_info = None, None
    joint = joint_position(rows) if args.fit == "joint" else None
    if joint is not None and not np.isfinite(joint["rho"]).any():
        joint = None
    lock = (dict(tau=float(args.lock_tau), tau_fast=float(args.lock_tau_fast),
                 fast_ratio=float(args.lock_fast_ratio))
            if args.world_lock else None)
    if args.rectify > 0.0:
        rectified, rect_info = rectify_matrices(
            series, meta["small_w"], meta["small_h"], margin,
            args.lock_fill, args.smooth_sigma, args.rectify,
            args.roll_gain, args.roll_sign, joint=joint, lock=lock,
            fps=meta["fps"], deadband_frac=float(args.lock_deadband),
            shape_sigma=float(args.shape_sigma))
    matrices = info = None
    if rectified is None:
        matrices, info = lock_matrices(
            series, meta["small_w"], meta["small_h"], args.lock_fill,
            args.smooth_sigma, args.roll_gain, args.roll_sign, args.max_zoom,
            margin)

    def project(index: int, x: float, y: float) -> np.ndarray:
        """Map a point of the measured frame into the delivered frame."""
        if rectified is not None:
            point = rectified[index] @ np.array([x, y, 1.0])
            point = point[:2] / point[2]
        else:
            point = matrices[index] @ np.array([x, y, 1.0])
        return point * scale - np.array([crop_x, crop_y])

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        middle = np.array([out_w, out_h], dtype=np.float64) * 0.5
        with args.trace.open("w", encoding="utf-8") as handle:
            for i in range(len(rows)):
                row = rows[i]
                if joint is not None and np.isfinite(joint["cx"][i]):
                    raw_cx, raw_cy = joint["cx"][i], joint["cy"][i]
                else:
                    raw_cx, raw_cy = series["cx"][i], series["cy"][i]
                resid = None
                if np.isfinite(raw_cx) and np.isfinite(raw_cy):
                    resid = project(i, raw_cx, raw_cy) - middle
                ring = None
                outer_points = row["outer"]["points"] if row["outer"] else []
                inner_points = row["inner"]["points"] if row["inner"] else []
                gaps = dict(gap_lr=None, gap_tb=None, ring_cv=None, screen_cv=None)
                if outer_points and inner_points:
                    outer = np.array([project(i, p[0], p[1]) for p in outer_points])
                    inner = np.array([project(i, p[0], p[1]) for p in inner_points])
                    outer_radius = np.linalg.norm(outer - middle, axis=1)
                    inner_radius = np.linalg.norm(inner - middle, axis=1)
                    angles = np.degrees(np.arctan2(outer[:, 1] - middle[1],
                                                   outer[:, 0] - middle[0]))
                    gap = outer_radius - inner_radius.mean()
                    top = (angles <= -45) & (angles > -135)
                    bottom = angles >= 45
                    left = angles > 135
                    right = angles <= 45
                    right = right & ~bottom
                    side = dict(left=left, right=right, top=top, bottom=bottom)
                    if all(mask.any() for mask in side.values()):
                        edges = {name: float(gap[mask].mean())
                                 for name, mask in side.items()}
                        gaps["gap_lr"] = abs(edges["left"] - edges["right"])
                        gaps["gap_tb"] = abs(edges["top"] - edges["bottom"])
                    gaps["ring_cv"] = float(outer_radius.std() / max(outer_radius.mean(), 1e-6))
                    gaps["screen_cv"] = float(inner_radius.std() / max(inner_radius.mean(), 1e-6))
                if outer_points:
                    ring = np.array([project(i, p[0], p[1]) for p in outer_points])
                    ring = ring.mean(axis=0) - middle
                handle.write(json.dumps({
                    "frame": i,
                    "cx": float(raw_cx) if np.isfinite(raw_cx) else None,
                    "cy": float(raw_cy) if np.isfinite(raw_cy) else None,
                    "size": float(series["size"][i]) if np.isfinite(series["size"][i]) else None,
                    "minor": float(series["minor"][i]) if np.isfinite(series["minor"][i]) else None,
                    "angle": float(series["angle"][i]) if np.isfinite(series["angle"][i]) else None,
                    "zoom": (float(rect_info["scale"][i] / base_k_small) if rect_info
                             else float(info["zoom"][i])),
                    "roll_meas": float(series["roll"][i]) if np.isfinite(series["roll"][i]) else None,
                    "resid_x": None if resid is None else float(resid[0]),
                    "resid_y": None if resid is None else float(resid[1]),
                    "ring_dx": None if ring is None else float(ring[0]),
                    "ring_dy": None if ring is None else float(ring[1]),
                    "outer_ratio": series["ratio"],
                    "buttons": int(row["outer"]["count"]) if row["outer"] else 0,
                    **gaps,
                }) + "\n")

    capture = cv2.VideoCapture(str(args.input))
    lens = dict(LENS, crop=LENS["crop"] / margin)
    remap = None if args.no_fisheye else build_remap(
        canvas_w, canvas_h, meta["raw_w"], meta["raw_h"], lens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             meta["fps"], (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {args.output}")

    index = 0
    while index < len(rows):
        ok, frame = capture.read()
        if not ok:
            break
        base = frame if remap is None else unwarp(frame, remap)
        if base.shape[1] != canvas_w or base.shape[0] != canvas_h:
            base = cv2.resize(base, (canvas_w, canvas_h))
        if rectified is not None:
            matrix = scale_homography(rectified[index], scale)
            locked = cv2.warpPerspective(base, matrix, (canvas_w, canvas_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT101)
        else:
            matrix = scale_matrix(matrices[index], scale)
            locked = cv2.warpAffine(base, matrix, (canvas_w, canvas_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT101)
        locked = locked[crop_y:crop_y + out_h, crop_x:crop_x + out_w]
        if args.draw:
            row = rows[index]
            source = (row["inner"]["center"] if row["inner"] is not None
                      else (row["outer"]["center"] if row["outer"] is not None else None))
            if source is not None:
                target = project(index, source[0], source[1])
                error = float(np.hypot(target[0] - out_w * 0.5,
                                       target[1] - out_h * 0.5))
                _draw_overlay(locked, (target[0], target[1]), error)
        writer.write(locked)
        index += 1
    capture.release()
    writer.release()
    print(f"wrote {args.output} ({index} frames, {out_w}x{out_h})")
    report = lock_report(rows, project, np.array([out_w, out_h], dtype=np.float64) * 0.5)
    print(f"  cabinet off centre : {report['machine']:6.2f}px  (of {out_w}px wide)")
    print(f"  button ring cv     : {report['ring_cv'] * 100:6.2f}%  "
          f"(0% = perfect circle, tune --ring-weight)")
    print(f"  screen edge cv     : {report['screen_cv'] * 100:6.2f}%")
    print(f"  four-side gap |L-R|: {report['gap_lr']:6.2f}px  "
          f"|T-B|: {report['gap_tb']:6.2f}px")
    if args.contact is not None:
        contact_sheet(args.output, args.contact)
        print(f"wrote {args.contact}")


if __name__ == "__main__":
    main()
