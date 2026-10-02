#!/usr/bin/env python3
"""Build a larger MaiMoller geometry dataset from photos and session footage.

The old model is reliable for ``outer_buttons`` but weak for ``inner_screen``
on fisheye frames, so this script mixes three sources:

* the hand-labelled dataset, copied through untouched,
* extra photo folders (iPhone HEIC and rectified PNG crops) auto-labelled by
  the current model,
* real raw-fisheye frames sampled from the captured sessions, auto-labelled
  the same way.

Whenever the model is not confident about the inner screen, the box is derived
from the reliable outer box with the median inner/outer margins learned from
the hand-labelled set.  Windows paths with non-ASCII characters are read
through Pillow/``imdecode`` because ``cv2.imread`` cannot open them.
"""

from __future__ import annotations

import argparse
import statistics
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("outer_buttons", "inner_screen")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

try:
    from PIL import Image
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:  # HEIC photos are then skipped with a warning.
    HEIC_SUPPORT = False


def read_image(path: Path) -> np.ndarray | None:
    """Read an image, including HEIC and non-ASCII Windows paths."""
    if path.suffix.lower() in {".heic", ".heif"}:
        if not HEIC_SUPPORT:
            return None
        try:
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        except (OSError, ValueError):
            return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray, max_side: int) -> None:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        path.write_bytes(encoded.tobytes())


def box_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5


def box_from_xyxy(xyxy, width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in xyxy)
    return ((x0 + x1) * 0.5 / width, (y0 + y1) * 0.5 / height,
            (x1 - x0) / width, (y1 - y0) / height)


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        labels.append((int(fields[0]), *(float(item) for item in fields[1:])))
    return labels


def write_label(path: Path, labels) -> None:
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def learn_margins(dataset: Path) -> dict[str, float]:
    """Median normalized inner/outer margins from the hand-labelled images."""
    left: list[float] = []
    top: list[float] = []
    right: list[float] = []
    bottom: list[float] = []
    for label_path in (dataset / "labels").rglob("*.txt"):
        labels = parse_yolo_label(label_path)
        outer = next((box for box in labels if box[0] == 0), None)
        inner = next((box for box in labels if box[0] == 1), None)
        if outer is None or inner is None:
            continue
        ox0, oy0, ox1, oy1 = box_to_xyxy(*outer[1:])
        ix0, iy0, ix1, iy1 = box_to_xyxy(*inner[1:])
        left.append(max(0.0, ix0 - ox0))
        top.append(max(0.0, iy0 - oy0))
        right.append(max(0.0, ox1 - ix1))
        bottom.append(max(0.0, oy1 - iy1))
    if not left:
        return {"left": 0.03, "top": 0.03, "right": 0.03, "bottom": 0.03}
    return {
        "left": statistics.median(left),
        "top": statistics.median(top),
        "right": statistics.median(right),
        "bottom": statistics.median(bottom),
    }


def infer_inner(outer: tuple[float, float, float, float], margins: dict[str, float]):
    ox0, oy0, ox1, oy1 = box_to_xyxy(*outer)
    ix0 = ox0 + margins["left"]
    iy0 = oy0 + margins["top"]
    ix1 = ox1 - margins["right"]
    iy1 = oy1 - margins["bottom"]
    cx = min(max((ix0 + ix1) * 0.5, 0.0), 1.0)
    cy = min(max((iy0 + iy1) * 0.5, 0.0), 1.0)
    width = min(max(ix1 - ix0, 0.005), 1.0)
    height = min(max(iy1 - iy0, 0.005), 1.0)
    return cx, cy, width, height


def pseudo_labels(detector, image: np.ndarray, margins, outer_conf: float, inner_conf: float):
    """Return (labels, outer_confidence, inner_from_model) for one image."""
    height, width = image.shape[:2]
    boxes = detector._detect_opencv(image)
    outer = max((box for box in boxes if "outer" in box[0]), key=lambda box: box[1], default=None)
    inner = max((box for box in boxes if "inner" in box[0]), key=lambda box: box[1], default=None)
    if outer is None or outer[1] < outer_conf:
        return None, 0.0, False
    labels = [(0, *box_from_xyxy(outer[2], width, height))]
    if inner is not None and inner[1] >= inner_conf:
        labels.append((1, *box_from_xyxy(inner[2], width, height)))
        return labels, outer[1], True
    labels.append((1, *infer_inner(labels[0][1:], margins)))
    return labels, outer[1], False


def copy_manual_dataset(source: Path, output: Path) -> int:
    copied = 0
    for split in ("train", "val"):
        images = source / "images" / split
        if not images.exists():
            continue
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in images.iterdir():
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            shutil.copyfile(image, output / "images" / split / image.name)
            label = source / "labels" / split / image.with_suffix(".txt").name
            if label.exists():
                shutil.copyfile(label, output / "labels" / split / image.with_suffix(".txt").name)
            copied += 1
    return copied


def add_photo_folder(directory: Path, output: Path, detector, margins,
                     max_side: int, outer_conf: float, inner_conf: float):
    added = 0
    from_model = 0
    skipped = 0
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = read_image(path)
        if image is None:
            skipped += 1
            continue
        labels, _conf, inner_ok = pseudo_labels(detector, image, margins, outer_conf, inner_conf)
        if labels is None:
            skipped += 1
            continue
        name = f"photo-{path.stem}"
        write_image(output / "images" / "train" / f"{name}.jpg", image, max_side)
        write_label(output / "labels" / "train" / f"{name}.txt", labels)
        added += 1
        from_model += int(inner_ok)
    return added, from_model, skipped


def add_session_frames(sessions: Path, output: Path, detector, margins,
                       per_session: int, max_side: int, outer_conf: float, inner_conf: float):
    added = 0
    from_model = 0
    for session in sorted(path for path in sessions.iterdir() if path.is_dir()):
        frames_dir = session / "frames"
        if not frames_dir.is_dir():
            continue
        frames = sorted(path for path in frames_dir.iterdir()
                        if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not frames:
            continue
        step = max(1, len(frames) // per_session)
        for frame in frames[::step][:per_session]:
            image = read_image(frame)
            if image is None:
                continue
            labels, _conf, inner_ok = pseudo_labels(detector, image, margins, outer_conf, inner_conf)
            if labels is None:
                continue
            name = f"sess-{session.name}-{frame.stem}"
            write_image(output / "images" / "train" / f"{name}.jpg", image, max_side)
            write_label(output / "labels" / "train" / f"{name}.txt", labels)
            added += 1
            from_model += int(inner_ok)
    return added, from_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=Path("datasets/maimoller-geometry"))
    parser.add_argument("--source-sessions", type=Path, default=Path("sessions"))
    parser.add_argument("--photo-dir", type=Path, action="append", default=[],
                        help="extra photo folder to auto-label; repeatable")
    parser.add_argument("--model", type=Path, default=Path("models/frame-geometry-yolo11n-v2.onnx"))
    parser.add_argument("--output", type=Path, default=Path("datasets/maimoller-geometry-v3"))
    parser.add_argument("--samples-per-session", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--outer-conf", type=float, default=0.55)
    parser.add_argument("--inner-conf", type=float, default=0.55)
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manual = copy_manual_dataset(args.source_dataset, args.output)
    margins = learn_margins(args.source_dataset)
    print(f"hand-labelled images copied: {manual}")
    print("median inner/outer margins: " +
          ", ".join(f"{key}={value:.4f}" for key, value in margins.items()))

    import sys
    sys.path.insert(0, str(Path.cwd()))
    from pc.process_session import GeometryDetector

    detector = GeometryDetector(args.model)
    for directory in args.photo_dir:
        added, from_model, skipped = add_photo_folder(
            directory, args.output, detector, margins, args.max_side,
            args.outer_conf, args.inner_conf,
        )
        print(f"{directory}: added={added} inner_from_model={from_model} skipped={skipped}")

    added, from_model = add_session_frames(
        args.source_sessions, args.output, detector, margins,
        args.samples_per_session, args.max_side, args.outer_conf, args.inner_conf,
    )
    print(f"session frames: added={added} inner_from_model={from_model}")

    train_images = sorted((args.output / "images" / "train").glob("*"))
    val_images = sorted((args.output / "images" / "val").glob("*"))
    (args.output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 2\n"
        "names: ['outer_buttons', 'inner_screen']\n",
        encoding="utf-8",
    )
    (args.output / "classes.txt").write_text("outer_buttons\ninner_screen\n", encoding="utf-8")
    print(f"dataset ready: train={len(train_images)} val={len(val_images)} -> {args.output}")


if __name__ == "__main__":
    main()
