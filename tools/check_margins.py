#!/usr/bin/env python3
"""Score the four side margins of a delivered (locked) clip against the spec.

Why this tool exists: the lock's own bookkeeping cannot answer the equal-margin
question.  The live pipeline models the screen as a circle, so its
``geometry_margins`` field comes out left == right and top == bottom on every
frame by construction; it says nothing about how the machine landed.

Two measurements are taken on the *delivered* pixels:

``box``
    The annotator's convention: the gaps between the ``outer_buttons`` box and
    the ``inner_screen`` box, renormalised so their mean is the machine's real
    ``--target`` millimetres (four 75.0s is dead on).  Handy for labelling, but
    the current detector's outer box also swallows the cabinet body below the
    ring on locked frames, so it reads bottom-heavy; treat it as a sanity check.

``ring``
    The spec measurement.  Find the play field, take the eight purple button
    blobs around it, and measure each slot's radius over the screen radius.
    A head-on machine puts every slot on one circle (``--ratio``, the head-on
    reference), which is exactly "left/right/top/bottom margins all the same".
    This is the number the lock has to move.

Verdict: ``PASS`` when the ring spread is within ``--tolerance`` on at least
``--pass-rate`` of the frames that produced a valid measurement.
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

from pc import canonical  # noqa: E402
from pc.process_session import (  # noqa: E402
    GeometryDetector,
    geometry_margins,
    plausible_geometry_pair,
)

SIDES = ("left", "right", "top", "bottom")
MM_PER_SIDE = 75.0


def purple_points(image: np.ndarray) -> np.ndarray:
    """Centroids of the button-sized purple blobs in a delivered frame.

    The live path measures the same points with the same code, so this is a
    re-export rather than a second implementation.
    """
    return canonical.purple_points(image)


def ring_ratios(image: np.ndarray, inner) -> np.ndarray:
    """Distance of each of the eight slots from the screen centre, over radius."""
    centre = np.array([(inner[0] + inner[2]) * 0.5, (inner[1] + inner[3]) * 0.5])
    radius = 0.25 * ((inner[2] - inner[0]) + (inner[3] - inner[1]))
    points = purple_points(image)
    return canonical.slot_ratios(centre, radius, points)


def side_values(ratios: np.ndarray) -> dict:
    out = {}
    for name, slots in canonical.SIDE_SLOTS.items():
        values = np.asarray(ratios, dtype=np.float64)[list(slots)]
        known = values[np.isfinite(values)]
        out[name] = float(known.mean()) if known.size else np.nan
    return out


def ring_aspect(points) -> float:
    """Minor/major ratio of the eight button centroids about their own mean.

    This is the anti-tautology guard.  The spec ratio is measured against the
    screen centre and radius, and the ring-round correction targets exactly
    that number, so a clip can look right there while the button ring itself is
    being squashed.  This one uses the button points alone, so trading one axis
    for the other cannot hide.  1.0 is a circle.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 8:
        return float("nan")
    delta = points - points.mean(axis=0)
    values = np.sort(np.linalg.eigvalsh(np.cov(delta.T)))[::-1]
    if values[0] <= 0.0:
        return float("nan")
    return float(np.sqrt(max(values[1], 0.0) / values[0]))


def _draw(frame, outer, inner, margins, target):
    for box, color, label in ((outer, (40, 210, 255), "outer_buttons"),
                              (inner, (80, 255, 170), "inner_screen")):
        if box is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
        cv2.putText(frame, label, (max(8, x0 + 6), max(24, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if inner is None:
        return frame
    centre = ((inner[0] + inner[2]) * 0.5, (inner[1] + inner[3]) * 0.5)
    radius = 0.25 * ((inner[2] - inner[0]) + (inner[3] - inner[1]))
    for point in purple_points(frame):
        cv2.circle(frame, (int(point[0]), int(point[1])), 6, (255, 0, 255), -1)
    if margins is not None:
        mean = float(np.mean([margins[s] for s in SIDES]))
        scale = target / mean if mean > 0 else 0.0
        for side, anchor in (("left", outer[0]), ("right", outer[2]),
                             ("top", outer[1]), ("bottom", outer[3])):
            value = margins[side] * scale
            good = abs(value - target) <= target * 0.08
            color = (120, 255, 120) if good else (80, 80, 255)
            if side in ("left", "right"):
                pos = (max(8, int(anchor) - 150), frame.shape[0] // 2)
            else:
                pos = (frame.shape[1] // 2 - 70,
                       max(24, int(anchor) - 12) if side == "top"
                       else min(frame.shape[0] - 12, int(anchor) + 34))
            cv2.putText(frame, f"{side[0].upper()} {value:4.1f}", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict(median=None, p10=None, p90=None, mean=None, std=None)
    return dict(median=float(np.median(values)),
                p10=float(np.percentile(values, 10)),
                p90=float(np.percentile(values, 90)),
                mean=float(values.mean()), std=float(values.std()))


def measure(path: Path, model, every: int, target: float, ratio: float,
            tolerance: float, pass_rate: float, min_aspect: float, limit,
            contact, contact_scale):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    detector = GeometryDetector(model)

    box_gaps = {side: [] for side in SIDES}
    ring_side = {side: [] for side in SIDES}
    ring_all = []
    centres = []
    aspects = []
    sheets = []
    frames = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if limit is not None and index >= limit:
            break
        if index % max(1, every) == 0:
            outer, inner = detector.detect(frame)
            if inner is not None:
                centre = ((inner[0] + inner[2]) * 0.5, (inner[1] + inner[3]) * 0.5)
                radius = 0.25 * ((inner[2] - inner[0]) + (inner[3] - inner[1]))
                points = purple_points(frame)
                ratios = canonical.slot_ratios(np.asarray(centre), radius, points)
                if np.isfinite(ratios).sum() >= 6:
                    sides = side_values(ratios)
                    if all(np.isfinite(v) for v in sides.values()):
                        for side in SIDES:
                            ring_side[side].append(sides[side])
                        ring_all.append(ratios)
                        centres.append(centre)
                        aspects.append(ring_aspect(points))
                        frames += 1
                if contact is not None and len(sheets) < 6 and outer is not None \
                        and plausible_geometry_pair(outer, inner):
                    sheets.append(_draw(frame.copy(), outer, inner,
                                        geometry_margins(outer, inner), target))
            if outer is not None and inner is not None and plausible_geometry_pair(outer, inner):
                margins = geometry_margins(outer, inner)
                if margins is not None:
                    for side in SIDES:
                        box_gaps[side].append(float(margins[side]))
        index += 1
    capture.release()

    report = dict(path=str(path), width=width, height=height, fps=fps,
                  frames=index, measured=frames, model=str(model) if model else None,
                  target_mm=target, ratio=ratio, tolerance=tolerance,
                  pass_rate=pass_rate)
    if frames == 0:
        report.update(verdict="NO-DETECTION", dead_on_fraction=0.0,
                      spread_median=None, ring_aspect=_stats(np.array([])),
                      aspect_ok=False)
        return report

    table = {side: np.asarray(ring_side[side]) for side in SIDES}
    stack = np.stack([table[s] for s in SIDES], axis=1)
    mean = stack.mean(axis=1)
    spread = (stack.max(axis=1) - stack.min(axis=1)) / np.maximum(mean, 1e-6)
    lr = np.abs(stack[:, 0] - stack[:, 1]) / np.maximum(mean, 1e-6)
    tb = np.abs(stack[:, 2] - stack[:, 3]) / np.maximum(mean, 1e-6)

    report["ring"] = {side: _stats(table[side]) for side in SIDES}
    report["ring_mean"] = _stats(np.stack([table[s] for s in SIDES], axis=1).mean(axis=0))
    report["ring_spread"] = _stats(spread)
    report["ring_imbalance_lr"] = _stats(lr)
    report["ring_imbalance_tb"] = _stats(tb)
    report["ring_target_error"] = _stats(np.abs(stack - ratio).mean(axis=1))
    report["ring_centres"] = dict(cx=_stats([c[0] for c in centres]),
                                  cy=_stats([c[1] for c in centres]))
    report["ring_aspect"] = _stats(np.asarray(aspects, dtype=np.float64))
    report["dead_on_fraction"] = float(np.mean(spread <= tolerance))
    report["spread_median"] = float(np.median(spread))
    aspect_ok = (report["ring_aspect"]["median"] is not None
                 and report["ring_aspect"]["median"] >= min_aspect)
    report["aspect_ok"] = bool(aspect_ok)
    report["verdict"] = ("PASS"
                         if report["dead_on_fraction"] >= pass_rate and aspect_ok
                         else "FAIL")

    if box_gaps["left"]:
        box = {side: np.asarray(box_gaps[side]) for side in SIDES}
        box_mean = np.stack([box[s] for s in SIDES], axis=1).mean(axis=1)
        report["box_mm"] = {side: _stats(target * box[side] / np.maximum(box_mean, 1e-6))
                            for side in SIDES}

    if contact is not None and sheets:
        contact.parent.mkdir(parents=True, exist_ok=True)
        grid = _stack(sheets, contact_scale)
        cv2.imwrite(str(contact), grid)
        report["contact"] = str(contact)
    return report


def _stack(frames, scale):
    if scale != 1.0:
        frames = [cv2.resize(f, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA) for f in frames]
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    rows = []
    for start in range(0, len(frames), 2):
        chunk = frames[start:start + 2]
        if len(chunk) == 1:
            chunk.append(np.zeros_like(chunk[0]))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models" / "frame-geometry-yolo11n-v5.onnx")
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--target", type=float, default=MM_PER_SIDE)
    parser.add_argument("--ratio", type=float, default=canonical.TILE_RATIO)
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument("--pass-rate", type=float, default=0.95)
    parser.add_argument("--min-aspect", type=float, default=0.95,
                        help="floor for the button ring's own minor/major ratio; "
                             "the independent guard against squashing the ring "
                             "to satisfy the spec ratio")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contact", type=Path)
    parser.add_argument("--contact-scale", type=float, default=0.5)
    parser.add_argument("--json", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    payload = []
    header = (f"{'clip':<30}{'frames':>7}{'ring n':>7}{'L/R/T/B ratio':>26}"
              f"{'spread':>9}{'dead-on':>9}")
    print(header)
    print("-" * len(header))
    for path in args.videos:
        contact = args.contact
        if contact is not None and len(args.videos) > 1:
            contact = contact.with_name(f"{contact.stem}-{path.stem}{contact.suffix}")
        report = measure(path, args.model, args.every, args.target, args.ratio,
                         args.tolerance, args.pass_rate, args.min_aspect,
                         args.limit, contact, args.contact_scale)
        payload.append(report)
        if report["measured"] == 0:
            print(f"{path.name:<30}{report['frames']:>7}{0:>7}{'no ring measurement':>26}")
            continue
        sides = "/".join(f"{report['ring'][s]['median']:.2f}" for s in SIDES)
        print(f"{path.name:<30}{report['frames']:>7}{report['measured']:>7}"
              f"{sides:>26}"
              f"{report['ring_spread']['median'] * 100:>8.1f}%"
              f"{report['dead_on_fraction'] * 100:>8.1f}%")
        line = (f"  {report['verdict']}  {path.name}: ring ratio {sides} (L/R/T/B), "
                f"target {report['ratio']:.2f}, spread "
                f"{report['ring_spread']['median'] * 100:.1f}% "
                f"(p90 {report['ring_spread']['p90'] * 100:.1f}%), "
                f"|L-R| {report['ring_imbalance_lr']['median'] * 100:.1f}% "
                f"|T-B| {report['ring_imbalance_tb']['median'] * 100:.1f}%, "
                f"ring aspect {report['ring_aspect']['median']:.3f}, "
                f"centre ({report['ring_centres']['cx']['median']:.0f}, "
                f"{report['ring_centres']['cy']['median']:.0f})")
        print(line)
        if "box_mm" in report:
            box = "/".join(f"{report['box_mm'][s]['median']:.0f}" for s in SIDES)
            print(f"      box-gap cross-check {box} mm (L/R/T/B) -- outer box "
                  f"includes the cabinet body, not part of the verdict")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
