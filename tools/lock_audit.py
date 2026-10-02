"""Measure how still a locked clip really is, from its own pixels.

The renderer knows where it *tried* to put the cabinet, so asking it where the
cabinet ended up proves nothing.  This tool ignores the renderer and tracks the
delivered pixels: a patch over the cabinet and a ring over the background are
phase-correlated frame to frame, and the accumulated drift of each is reported.

``cabinet drift`` is the number the operator sees as "the machine is nailed
down".  ``background drift`` says how much of that is the whole picture moving
with the warp.  On a rigid scene the two have to agree; if the cabinet is much
noisier than the background the lock is inventing motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def hann(shape: tuple[int, int]) -> np.ndarray:
    return cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)


def accum(values: np.ndarray) -> np.ndarray:
    return np.cumsum(np.nan_to_num(values, nan=0.0), axis=0)


def describe(name: str, track: np.ndarray, steps: np.ndarray, width: int) -> dict:
    drift = np.hypot(track[:, 0], track[:, 1])
    step = np.hypot(steps[:, 0], steps[:, 1])
    return dict(
        name=name,
        drift_median=float(np.median(drift)),
        drift_p95=float(np.percentile(drift, 95)),
        drift_max=float(drift.max()),
        step_std=float(step.std()),
        step_p99=float(np.percentile(step, 99)),
        drift_pct=float(np.percentile(drift, 95) / width * 100.0),
    )


def audit(path: Path, fill: float, limit: int | None, caption: str = "") -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    radius = fill * width * 0.5
    centre = np.array([width * 0.5, height * 0.5])
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.hypot(xx - centre[0], yy - centre[1])
    masks = {
        "cabinet": ((distance < 1.6 * radius) & (distance > 0.25 * radius)),
        "background": ((distance > 1.35 * radius) & (distance < 3.4 * radius)),
    }
    windows = {}
    for key, mask in masks.items():
        soft = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0.0)
        windows[key] = hann((height, width)) * soft

    previous = None
    steps: dict[str, list[tuple[float, float, float]]] = {k: [] for k in masks}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or (limit and index >= limit):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            for key, mask in masks.items():
                (dx, dy), response = cv2.phaseCorrelate(
                    previous * mask, gray * mask, windows[key])
                steps[key].append((dx, dy, response))
        previous = gray
        index += 1
    capture.release()

    report = dict(path=str(path), frames=index, width=width, height=height,
                  fps=fps, caption=caption)
    for key in masks:
        track = accum(np.asarray(steps[key])[:, :2])
        report[key] = describe(key, track, np.asarray(steps[key])[:, :2], width)
        report[key]["response"] = float(np.median(np.asarray(steps[key])[:, 2]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--fill", type=float, default=0.72,
                        help="lock-fill used by the renderer")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    payload = [audit(path, args.fill, args.limit) for path in args.videos]
    header = f"{'clip':<38}{'cab drift p95':>14}{'cab step std':>14}" \
             f"{'bg drift p95':>14}{'bg step std':>13}"
    print(header)
    print("-" * len(header))
    for item in payload:
        print(f"{item['path'].split(chr(92))[-1]:<38}"
              f"{item['cabinet']['drift_p95']:>11.2f}px"
              f"{item['cabinet']['step_std']:>11.2f}px"
              f"{item['background']['drift_p95']:>11.2f}px"
              f"{item['background']['step_std']:>10.2f}px")
    for item in payload:
        print(f"  {Path(item['path']).name}: cabinet moves "
              f"{item['cabinet']['drift_pct']:.2f}% of the width (p95 of "
              f"{item['frames']} frames @ {item['fps']:.0f}fps), "
              f"mean step {item['cabinet']['step_std']:.2f}px, "
              f"p99 step {item['cabinet']['step_p99']:.2f}px")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
