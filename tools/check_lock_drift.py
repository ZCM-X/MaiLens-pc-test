#!/usr/bin/env python3
"""Does the lock hold the machine, or just hold it roughly?

``tools/lock_audit.py`` measures frame-to-frame jitter on the delivered clip.
This answers the other question: over a whole segment, how much of the machine's
movement is the operator's hand, and how much is the pipeline's own drift.

It aligns the delivered clip to the source session by ``frame_id`` (the session's
``capture.jsonl`` order is the raw video's frame order), detects the machine
centre in both with the same detector, and reports, per contiguous run of one
``lock_mode``:

* ``in std`` / ``out std`` -- spread of the machine centre in the raw frames and
  in the delivered frames;
* ``ratio`` -- delivered spread over input spread, i.e. how much of the hand
  motion survived;
* ``corr`` -- correlation between the two.  A lock that under-corrects leaves a
  residual proportional to the input, so ``corr`` stays near +1.  A delivered
  wobble that is *uncorrelated* with the input is being generated inside the
  pipeline and cannot be blamed on the operator.

A ``lock`` segment with a large ``out std`` and a low ``corr`` is the signature
of self-generated drift: the machine is moving even though nobody moved it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc.process_session import GeometryDetector  # noqa: E402


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def centres(path: Path, wanted: set, detector) -> dict:
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    out = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if index in wanted:
            _outer, inner = detector.detect(frame)
            if inner is not None:
                out[index] = ((inner[0] + inner[2]) * 0.5,
                              (inner[1] + inner[3]) * 0.5)
        index += 1
    capture.release()
    return out


def segments(modes):
    """Contiguous runs of one lock_mode, as (mode, first, last) over trace rows."""
    out = []
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            out.append((modes[start], start, i - 1))
            start = i
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="session folder with raw.mp4 + capture.jsonl")
    parser.add_argument("clip", type=Path, help="delivered locked clip")
    parser.add_argument("--trace", type=Path,
                        help="per-frame jsonl for the clip (default: clip .jsonl next to it)")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models" / "frame-geometry-yolo11n-v5.onnx")
    parser.add_argument("--step", type=int, default=2,
                        help="detect one frame in N inside each segment")
    parser.add_argument("--json", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace_path = args.trace or args.clip.with_suffix(".jsonl")
    trace = read_jsonl(trace_path)
    capture = read_jsonl(args.session / "capture.jsonl")
    raw_of = {}
    for index, row in enumerate(capture):
        raw_of.setdefault(row.get("frame_id"), index)

    modes = [row.get("lock_mode") for row in trace]
    detector = GeometryDetector(args.model)
    print(f"{args.clip.name}: {len(trace)} frames, trace {trace_path.name}, "
          f"session {args.session.name}")

    payload = []
    for mode, lo, hi in segments(modes):
        rows = [(i, raw_of[trace[i].get("frame_id")])
                for i in range(lo, hi + 1, max(1, args.step))
                if trace[i].get("frame_id") in raw_of]
        if len(rows) < 5:
            continue
        delivered = centres(args.clip, {i for i, _ in rows}, detector)
        raw = centres(args.session / "raw.mp4", {j for _, j in rows}, detector)
        both = [(i, j) for i, j in rows if i in delivered and j in raw]
        if len(both) < 5:
            continue
        dx = np.array([delivered[i][0] for i, _ in both])
        dy = np.array([delivered[i][1] for i, _ in both])
        rx = np.array([raw[j][0] for _, j in both])
        ry = np.array([raw[j][1] for _, j in both])
        item = dict(mode=mode, first_index=lo, last_index=hi, samples=len(both),
                    delivered_std=float(np.hypot(dx.std(), dy.std())),
                    input_std=float(np.hypot(rx.std(), ry.std())),
                    delivered_range=float(max(np.ptp(dx), np.ptp(dy))),
                    input_range=float(max(np.ptp(rx), np.ptp(ry))),
                    ratio=float(np.hypot(dx.std(), dy.std())
                                / max(np.hypot(rx.std(), ry.std()), 1e-6)),
                    corr_x=float(np.corrcoef(rx, dx)[0, 1]),
                    corr_y=float(np.corrcoef(ry, dy)[0, 1]))
        item["corr"] = float(np.mean([item["corr_x"], item["corr_y"]]))
        payload.append(item)
        print(f"  {mode:<6} idx {lo:>4}-{hi:<4} n={len(both):<4} "
              f"in std {item['input_std']:6.1f}px  out std {item['delivered_std']:6.1f}px  "
              f"ratio {item['ratio']:5.2f}  corr {item['corr_x']:+.2f}/{item['corr_y']:+.2f}"
              f"  out range {item['delivered_range']:5.1f}px")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"wrote {args.json}")
    print("corr near +1 = the leftover is the operator's motion; corr near 0 with a "
          "large out std = the pipeline is moving the machine on its own")


if __name__ == "__main__":
    main()
