#!/usr/bin/env python3
"""Pull a delivered (locked) clip the rest of the way to equal side margins.

``pc/canonical.pull_back_maps`` already knows how to walk all eight button slots
onto one circle: the screen edge does not move, the ring lands on ``--ratio`` in
every direction, and the screen-to-ring gap comes out the same on all four
sides.  The live path does not run it.  This tool does, on a clip that is
already locked, so the fix can be judged before it is wired into the renderer.

One pass at unity gain only removes about half of the measured error, because a
button is a finite patch: the ramp is evaluated per pixel, so the near and far
halves of the same blob are pushed by different amounts and the blob centroid
moves less than the ramp predicts.  ``--gain`` above 1, or a second pass over
the first output, closes the rest.

Measure the result with ``tools/check_margins.py``; the spec is in
``docs/margin-spec.md``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc import canonical  # noqa: E402
from pc.machine_lock import fill_nan, smooth  # noqa: E402
from pc.process_session import GeometryDetector  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models" / "frame-geometry-yolo11n-v5.onnx")
    parser.add_argument("--every", type=int, default=6,
                        help="detect one frame in N; the series is filled between")
    parser.add_argument("--gain", type=float, default=1.8,
                        help="correction strength; 1.0 leaves about half the error")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=canonical.TILE_RATIO)
    parser.add_argument("--sigma", type=float, default=6.0,
                        help="zero-phase smoothing window for the coefficients")
    parser.add_argument("--coarse", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    return parser


def measure_series(path: Path, model: Path, every: int, max_frames: int | None):
    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        total = min(total, max_frames)
    detector = GeometryDetector(model)
    cx = np.full(total, np.nan)
    cy = np.full(total, np.nan)
    radius = np.full(total, np.nan)
    ratios = np.full((total, canonical.SLOT_COUNT), np.nan)
    index = 0
    while index < total:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if index % max(1, every) == 0:
            _outer, inner = detector.detect(frame)
            if inner is not None:
                centre = np.array([(inner[0] + inner[2]) * 0.5,
                                   (inner[1] + inner[3]) * 0.5])
                screen = 0.25 * ((inner[2] - inner[0]) + (inner[3] - inner[1]))
                if screen > 5.0:
                    cx[index], cy[index], radius[index] = centre[0], centre[1], screen
                    ratios[index] = canonical.slot_ratios(
                        centre, screen, canonical.purple_points(frame))
        index += 1
    capture.release()
    for series in (cx, cy, radius):
        known = np.isfinite(series)
        if known.sum() < 2:
            raise RuntimeError(f"{path}: not enough screen detections to correct")
        series[:] = np.interp(np.arange(total), np.flatnonzero(known), series[known])
    return width, height, fps, total, cx, cy, radius, ratios


def fit_series(ratios: np.ndarray, order: int, sigma: float) -> np.ndarray:
    coef = np.full((ratios.shape[0], 2 * order + 1), np.nan)
    for index in range(ratios.shape[0]):
        found = canonical.ring_profile(ratios[index], order)
        if found is not None:
            coef[index] = found
    for column in range(coef.shape[1]):
        values = fill_nan(coef[:, column])
        coef[:, column] = smooth(values, sigma) if sigma > 0 else values
    return coef


def main() -> None:
    args = build_parser().parse_args()
    width, height, fps, total, cx, cy, radius, ratios = measure_series(
        args.input, args.model, args.every, args.max_frames)
    measured = int(np.isfinite(ratios).any(axis=1).sum())
    coef = fit_series(ratios, args.order, args.sigma)
    print(f"{args.input.name}: {total} frames, {measured} with a ring measurement, "
          f"{width}x{height} @ {fps:.1f}fps")

    capture = cv2.VideoCapture(str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    index = 0
    while index < total:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        maps = canonical.pull_back_maps(
            width, height, (cx[index], cy[index]), float(radius[index]),
            coef[index], strength=args.gain, order=args.order,
            target=args.ratio, coarse=args.coarse)
        writer.write(canonical.apply(frame, maps))
        index += 1
    capture.release()
    writer.release()
    print(f"wrote {args.output} ({index} frames, gain {args.gain})")
    print(f"verify: python tools/check_margins.py \"{args.output}\"")


if __name__ == "__main__":
    main()
