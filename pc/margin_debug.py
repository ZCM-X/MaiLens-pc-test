#!/usr/bin/env python3
"""Draw the outer/inner machine geometry and its four margin errors.

This is deliberately a diagnostic pass.  It performs the calibrated fisheye
unwarp and runs the detector, but it does not apply a lock transform.  The
overlay makes it possible to distinguish a bad detection from a bad stabilizer
before changing the camera controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .process_session import (
        GeometryDetector,
        build_remap,
        geometry_margins,
        make_output_rays,
        map_fisheye_box_to_output,
    )
except ImportError:
    from process_session import (  # type: ignore
        GeometryDetector,
        build_remap,
        geometry_margins,
        make_output_rays,
        map_fisheye_box_to_output,
    )


def _draw_box(frame: np.ndarray, box, color: tuple[int, int, int], label: str) -> None:
    if box is None:
        return
    x0, y0, x1, y1 = (int(value) for value in box)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 4, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (max(8, x0 + 8), max(30, y0 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA,
    )


def _draw_margin_text(frame: np.ndarray, outer, inner, margins) -> None:
    if outer is None or inner is None or margins is None:
        return
    ox0, oy0, ox1, oy1 = (int(value) for value in outer)
    ix0, iy0, ix1, iy1 = (int(value) for value in inner)
    left, top, right, bottom = (int(margins[key]) for key in ("left", "top", "right", "bottom"))
    # Put each value near the corresponding gap.  Negative values are useful:
    # they show when inverse-fisheye box mapping makes the boxes cross.
    cv2.putText(frame, f"L {left:+d}", (max(8, ox0), max(30, (oy0 + oy1) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 80), 3, cv2.LINE_AA)
    cv2.putText(frame, f"R {right:+d}", (min(frame.shape[1] - 180, ix1 + 12), max(30, (oy0 + oy1) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 80), 3, cv2.LINE_AA)
    cv2.putText(frame, f"T {top:+d}", (max(8, (ox0 + ox1) // 2 - 60), max(35, oy0 - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 80), 3, cv2.LINE_AA)
    cv2.putText(frame, f"B {bottom:+d}", (max(8, (ox0 + ox1) // 2 - 60), min(frame.shape[0] - 12, iy1 + 38)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 80), 3, cv2.LINE_AA)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--crop", type=float, default=0.74)
    parser.add_argument("--fov", type=float, default=106.4583)
    parser.add_argument("--center-x", type=float, default=0.501753869)
    parser.add_argument("--center-y", type=float, default=0.499423644)
    parser.add_argument("--k1", type=float, default=0.0893163)
    parser.add_argument("--k2", type=float, default=-0.0174637)
    parser.add_argument("--detect-every", type=int, default=6)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-fisheye", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = float(args.fps or (source_fps if 0.5 < source_fps < 240.0 else 30.0))
    detector = GeometryDetector(args.model)
    output_rays = make_output_rays(width, height, args.crop, args.fov)
    remap = None
    if not args.no_fisheye:
        remap = build_remap(
            width,
            height,
            np.eye(3, dtype=np.float32),
            args.crop,
            args.fov,
            args.k1,
            args.k2,
            args.center_x,
            args.center_y,
            output_rays=output_rays,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{args.output}")
    log_path = args.output.with_suffix(".jsonl")
    last_outer = None
    last_inner = None
    last_margins = None
    frame_index = 0
    try:
        with log_path.open("w", encoding="utf-8") as log:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
                raw_frame = frame
                detected_outer = detected_inner = None
                if frame_index % max(1, args.detect_every) == 0:
                    raw_outer, raw_inner = detector.detect(raw_frame)
                    if args.no_fisheye:
                        detected_outer, detected_inner = raw_outer, raw_inner
                    else:
                        identity = np.eye(3, dtype=np.float32)
                        detected_outer = map_fisheye_box_to_output(
                            raw_outer,
                            width,
                            height,
                            identity,
                            args.crop,
                            args.fov,
                            args.k1,
                            args.k2,
                            args.center_x,
                            args.center_y,
                        )
                        detected_inner = map_fisheye_box_to_output(
                            raw_inner,
                            width,
                            height,
                            identity,
                            args.crop,
                            args.fov,
                            args.k1,
                            args.k2,
                            args.center_x,
                            args.center_y,
                        )
                    if detected_outer is not None:
                        last_outer = detected_outer
                    if detected_inner is not None:
                        last_inner = detected_inner
                if remap is not None:
                    frame = cv2.remap(
                        frame,
                        remap[0],
                        remap[1],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT101,
                    )
                last_margins = geometry_margins(last_outer, last_inner)
                _draw_box(frame, last_outer, (40, 210, 255), "outer_buttons")
                _draw_box(frame, last_inner, (80, 255, 170), "inner_screen")
                _draw_margin_text(frame, last_outer, last_inner, last_margins)
                cv2.drawMarker(
                    frame,
                    (width // 2, height // 2),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    max(24, min(width, height) // 20),
                    3,
                    cv2.LINE_AA,
                )
                if last_margins is None:
                    error_x = error_y = None
                    status = "waiting for outer + inner"
                else:
                    error_x = int(last_margins["left"] - last_margins["right"])
                    error_y = int(last_margins["top"] - last_margins["bottom"])
                    status = f"L-R={error_x:+d}  T-B={error_y:+d}"
                cv2.putText(
                    frame,
                    f"frame={frame_index}  {status}",
                    (18, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.15,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                log.write(json.dumps({
                    "frame": frame_index,
                    "timestamp": frame_index / fps,
                    "outer": last_outer,
                    "inner": last_inner,
                    "margins": last_margins,
                    "error_x": error_x,
                    "error_y": error_y,
                    "detected": frame_index % max(1, args.detect_every) == 0,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                writer.write(frame)
                frame_index += 1
    finally:
        writer.release()
        capture.release()
    return args.output


def main() -> None:
    args = build_parser().parse_args()
    if args.detect_every < 1:
        raise SystemExit("--detect-every 必须大于 0")
    run(args)


if __name__ == "__main__":
    main()
