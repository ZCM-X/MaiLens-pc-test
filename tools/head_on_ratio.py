#!/usr/bin/env python3
"""Read the head-on tile ratio off a reference photo, the pipeline's way.

``pc/canonical.py`` pulls the eight button slots back onto one circle.  How
far out that circle sits is a property of the cabinet, not of the correction,
so it has to be measured once on a picture taken square in front of the
machine -- and with the *same* two detectors the correction uses, because the
denominator matters:

* the screen is the cyan play field the HSV mask finds, a little inside the
  bezel;
* the button is the colour centroid of the tile face, a little inside the
  part.

Measured by hand against the bezel's ring of white dots the eight frames come
out at 1.365; measured by this pipeline on the same picture they come out at
1.23.  Mixing the two numbers costs about 9% of zoom, so this command exists
to keep the two tied together.

    python tools/head_on_ratio.py "H:/IMG_8839(20261002-154029).JPG"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Running this as ``python tools/head_on_ratio.py`` puts tools/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc import canonical, machine_lock  # noqa: E402
from tools.measure_button_frames import read_image


def measure(frame: np.ndarray) -> dict | None:
    """Screen radius, per-slot tile ratios and the ring spread for one photo."""
    found = machine_lock.measure_frame(frame)
    inner, outer = found["inner"], found["outer"]
    if inner is None or outer is None:
        return None
    centre = np.array(inner["center"], dtype=np.float64)
    radius = 0.5 * float(inner["major"])
    points = np.array(outer["points"], dtype=np.float64)
    ratios = canonical.slot_ratios(centre, radius, points)
    known = ratios[np.isfinite(ratios)]
    if known.size < 5 or radius <= 0.0:
        return None
    return dict(screen_radius=float(radius),
                blobs=int(len(points)),
                ratios=[float(v) for v in ratios],
                centre=[float(centre[0]), float(centre[1])],
                mean=float(known.mean()),
                spread=canonical.ring_error(known),
                missing=int(np.isnan(ratios).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+", help="正对机台的照片")
    parser.add_argument("--json", type=Path, help="把读数写成 json")
    args = parser.parse_args()

    report = {}
    for path in args.input:
        frame = read_image(path)
        found = measure(frame)
        if found is None:
            print(f"{path.name}: 认不出内屏或按键圈（正面照要够亮、够完整）")
            continue
        report[path.name] = found
        print(f"--- {path.name}  {frame.shape[1]}x{frame.shape[0]}")
        print(f"    内屏半径（青色游戏区）{found['screen_radius']:8.1f}px"
              f"    按键点 {found['blobs']} 个，缺 {found['missing']} 槽")
        print("    每槽 框心半径/内屏半径  "
              + " ".join("--" if not np.isfinite(v) else f"{v:.3f}"
                         for v in found["ratios"]))
        print(f"    均值 {found['mean']:.3f}   绕圈离散"
              f" {100 * found['spread']:.2f}%"
              "   （离散里有一部分是这张照片自己的倾斜）")
    if args.json is not None and report:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print("wrote", args.json)
    if report:
        means = [value["mean"] for value in report.values()]
        print(f"把这组照片的均值 {np.mean(means):.3f} 写进"
              " pc/canonical.TILE_RATIO")


if __name__ == "__main__":
    main()
