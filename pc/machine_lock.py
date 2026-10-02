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


def measure_button_ring(frame: np.ndarray) -> dict | None:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_LOW, PURPLE_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs = [(int(stats[i, cv2.CC_STAT_AREA]), centroids[i])
             for i in range(1, count)]
    blobs.sort(key=lambda item: -item[0])
    blobs = [b for b in blobs if b[0] >= 12][:12]
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
        "roll": ring_roll(angles),
    }


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
    (ex, ey), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
    return {
        "center": [float(ex), float(ey)],
        "major": float(major),
        "minor": float(minor),
        "angle": float(angle),
        "area": float(cv2.contourArea(contour)),
    }


def measure_frame(frame: np.ndarray) -> dict:
    return {"inner": measure_inner_screen(frame),
            "outer": measure_button_ring(frame)}


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
    roll = np.full(count, np.nan)
    for i, row in enumerate(rows):
        inner, outer = row["inner"], row["outer"]
        if inner is not None:
            cx[i], cy[i] = inner["center"]
            # The major axis is the tilt-invariant distance signal.
            size[i] = inner["major"]
        elif outer is not None and ratio:
            cx[i], cy[i] = outer["center"]
            size[i] = outer_major[i] / ratio
        if outer is not None and outer.get("roll") is not None:
            roll[i] = outer["roll"]
    return dict(cx=cx, cy=cy, size=size, roll=roll, ratio=ratio)


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
    room_x = np.minimum(cx, small_w - cx)
    room_y = np.minimum(cy, small_h - cy)
    need = np.maximum(half_w / np.maximum(room_x, 1e-3),
                      half_h / np.maximum(room_y, 1e-3))
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


# ------------------------------------------------------------------------- io
def measure_clip(path: Path, measure_scale: float, max_frames: int | None,
                 margin: float = 1.0):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    raw_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
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
    return dict(raw_w=raw_w, raw_h=raw_h, fps=fps, small_w=small_w,
                small_h=small_h, rows=rows, count=len(rows))


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
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    parser.add_argument("--roll-gain", type=float, default=0.0)
    parser.add_argument("--roll-sign", type=float, default=1.0)
    parser.add_argument("--max-zoom", type=float, default=3.5)
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--no-fisheye", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    margin = max(float(args.margin), 1.0)
    meta = measure_clip(args.input, args.measure_scale, args.max_frames, margin)
    rows = meta["rows"]
    series = state_series(rows)
    matrices, info = lock_matrices(
        series, meta["small_w"], meta["small_h"], args.lock_fill,
        args.smooth_sigma, args.roll_gain, args.roll_sign, args.max_zoom, margin,
    )

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        out_cx, out_cy = meta["small_w"] * 0.5, meta["small_h"] * 0.5
        with args.trace.open("w", encoding="utf-8") as handle:
            for i in range(len(rows)):
                raw_cx = series["cx"][i]
                raw_cy = series["cy"][i]
                if np.isfinite(raw_cx) and np.isfinite(raw_cy):
                    matrix = matrices[i]
                    resid_x = (matrix[0, 0] * raw_cx + matrix[0, 1] * raw_cy
                               + matrix[0, 2]) - out_cx
                    resid_y = (matrix[1, 0] * raw_cx + matrix[1, 1] * raw_cy
                               + matrix[1, 2]) - out_cy
                else:
                    resid_x = resid_y = None
                handle.write(json.dumps({
                    "frame": i,
                    "cx": float(raw_cx) if np.isfinite(raw_cx) else None,
                    "cy": float(raw_cy) if np.isfinite(raw_cy) else None,
                    "cx_smooth": float(info["cx"][i]),
                    "cy_smooth": float(info["cy"][i]),
                    "size": float(series["size"][i]) if np.isfinite(series["size"][i]) else None,
                    "zoom": float(info["zoom"][i]),
                    "roll": float(info["roll"][i]),
                    "roll_meas": float(series["roll"][i]) if np.isfinite(series["roll"][i]) else None,
                    "resid_x": None if resid_x is None else float(resid_x),
                    "resid_y": None if resid_y is None else float(resid_y),
                    "outer_ratio": series["ratio"],
                }) + "\n")

    out_w = max(2, int(round(meta["raw_w"] * args.output_scale)))
    out_h = max(2, int(round(meta["raw_h"] * args.output_scale)))
    canvas_w = max(out_w, int(round(out_w * margin)))
    canvas_h = max(out_h, int(round(out_h * margin)))
    scale = canvas_w / meta["small_w"]
    capture = cv2.VideoCapture(str(args.input))
    lens = dict(LENS, crop=LENS["crop"] / margin)
    remap = None if args.no_fisheye else build_remap(
        canvas_w, canvas_h, meta["raw_w"], meta["raw_h"], lens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             meta["fps"], (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {args.output}")
    crop_x = (canvas_w - out_w) // 2
    crop_y = (canvas_h - out_h) // 2

    index = 0
    while index < len(rows):
        ok, frame = capture.read()
        if not ok:
            break
        base = frame if remap is None else unwarp(frame, remap)
        if base.shape[1] != canvas_w or base.shape[0] != canvas_h:
            base = cv2.resize(base, (canvas_w, canvas_h))
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
                point = matrices[index] @ np.array([source[0], source[1], 1.0])
                target = (point[0] * scale - crop_x, point[1] * scale - crop_y)
                error = float(np.hypot(point[0] - meta["small_w"] * 0.5,
                                       point[1] - meta["small_h"] * 0.5) * scale)
                _draw_overlay(locked, target, error)
        writer.write(locked)
        index += 1
    capture.release()
    writer.release()
    print(f"wrote {args.output} ({index} frames, {out_w}x{out_h})")
    if args.contact is not None:
        contact_sheet(args.output, args.contact)
        print(f"wrote {args.contact}")


if __name__ == "__main__":
    main()
