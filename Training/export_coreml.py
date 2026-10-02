#!/usr/bin/env python3
"""Export the trained geometry detector to Core ML for the iOS app.

Core ML conversion needs the native coremltools backend, which only ships for
macOS and Linux, and Ultralytics additionally refuses Core ML export on
Windows.  The iOS build therefore runs this script on the macOS builder
before ``xcodegen``, and the app bundles the resulting ``.mlpackage``.

The exported pipeline takes ``image`` (3x640x640), ``iouThreshold`` and
``confidenceThreshold``, and returns:

* ``confidence``  - boxes x class scores
* ``coordinates`` - boxes x [x, y, width, height], normalized to the image

which is what ``MachineDetector.swift`` decodes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("Training/machine-detector.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("iOS/Resources/MachineDetector.mlpackage"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"error: missing weights {args.weights}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: install ultralytics and coremltools first", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    exported = model.export(format="coreml", imgsz=args.imgsz, nms=True,
                            half=False, quantize=16)
    exported_path = Path(exported[0] if isinstance(exported, (list, tuple)) else exported)
    if not exported_path.exists():
        print(f"error: ultralytics reported {exported_path} but it is missing", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.is_dir():
            shutil.rmtree(args.output)
        else:
            args.output.unlink()
    if exported_path.is_dir():
        shutil.copytree(exported_path, args.output)
    else:
        shutil.copyfile(exported_path, args.output)
    print(f"Core ML detector ready: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
