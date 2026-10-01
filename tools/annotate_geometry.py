#!/usr/bin/env python3
"""Small OpenCV tool for labeling MaiLens geometry and optional key boxes.

The tool intentionally labels only the cabinet geometry:

* ``1`` / ``outer_frame``: the complete visible machine/cabinet body.
* ``2`` / ``inner_screen``: the actual gameplay display rectangle.
* ``3`` / ``button``: one gameplay button region; draw up to eight per frame
  with ``--with-buttons``.

It writes ordinary YOLO detection labels, so the resulting folder can be
used directly to train an Ultralytics model and then exported to ONNX.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("outer_frame", "inner_screen", "button")
COLORS = ((0, 220, 255), (80, 255, 170), (255, 150, 60))  # BGR: yellow, green, orange
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def normalize_box(box: tuple[int, int, int, int], width: int, height: int) -> str:
    """Convert an image-space xyxy box to one YOLO label line."""
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)
    center_x = (x0 + x1) * 0.5 / max(width, 1)
    center_y = (y0 + y1) * 0.5 / max(height, 1)
    return f"{center_x:.6f} {center_y:.6f} {box_width / max(width, 1):.6f} {box_height / max(height, 1):.6f}"


def denormalize_box(values: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert one YOLO label line back to an image-space xyxy box."""
    center_x, center_y, box_width, box_height = values
    pixel_width = box_width * width
    pixel_height = box_height * height
    return (
        int(round((center_x * width) - pixel_width * 0.5)),
        int(round((center_y * height) - pixel_height * 0.5)),
        int(round((center_x * width) + pixel_width * 0.5)),
        int(round((center_y * height) + pixel_height * 0.5)),
    )


@dataclass
class FrameSource:
    """Random-access source for an image list or a sampled video."""

    path: Path
    image_paths: list[Path] | None = None
    every: int = 1
    max_frames: int = 0

    def __post_init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        if self.image_paths is not None:
            self.count = len(self.image_paths)
            self.fps = 0.0
            return
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"无法打开视频：{self.path}")
        source_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.count = max(0, (source_count + self.every - 1) // self.every)
        if self.max_frames > 0:
            self.count = min(self.count, self.max_frames)

    def read(self, index: int) -> tuple[np.ndarray, int]:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        if self.image_paths is not None:
            image = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_COLOR)
            source_frame = index
        else:
            assert self._capture is not None
            source_frame = index * self.every
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = self._capture.read()
            if not ok:
                raise RuntimeError(f"无法读取视频第 {source_frame} 帧：{self.path}")
        if image is None:
            raise RuntimeError(f"无法读取图像：{self.path}")
        return image, source_frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()


def make_source(input_path: Path, every: int, max_frames: int) -> FrameSource:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_dir():
        # A receiver session stores frames in ``session/frames``.  Accepting
        # the session directory directly avoids making the user find it.
        image_root = input_path / "frames" if (input_path / "frames").is_dir() else input_path
        paths = sorted(
            path for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if every > 1:
            paths = paths[::every]
        if max_frames > 0:
            paths = paths[:max_frames]
        if not paths:
            raise RuntimeError(f"目录中没有 JPG/PNG 图像：{image_root}")
        return FrameSource(input_path, image_paths=paths, every=every, max_frames=max_frames)
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return FrameSource(input_path, image_paths=[input_path])
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return FrameSource(input_path, every=max(1, every), max_frames=max_frames)
    raise RuntimeError(f"不支持的输入格式：{input_path.suffix}")


class GeometryAnnotator:
    window_name = "MaiLens geometry annotator"

    def __init__(
        self,
        source: FrameSource,
        output: Path,
        val_every: int,
        *,
        include_buttons: bool = False,
        button_count: int = 8,
    ) -> None:
        self.source = source
        self.output = output.expanduser().resolve()
        self.val_every = max(0, int(val_every))
        self.classes = CLASSES if include_buttons else CLASSES[:2]
        self.include_buttons = include_buttons
        self.button_count = max(1, int(button_count))
        self.index = 0
        self.active_class = 0
        self.boxes: dict[int, list[tuple[int, int, int, int]]] = {}
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.display_scale = 1.0
        self.display_size = (0, 0)
        self.saved: set[int] = set()
        self.dirty = False
        self.frame, self.source_frame = self.source.read(0)
        self.load_existing()

    @property
    def current_base(self) -> str:
        return f"frame-{self.index + 1:06d}"

    def split_name(self) -> str:
        # Holding out every Nth sampled frame gives a reproducible validation
        # set while keeping neighboring frames available for training.
        return "val" if self.val_every and (self.index + 1) % self.val_every == 0 else "train"

    def split_dirs(self) -> tuple[Path, Path]:
        split = self.split_name()
        return self.output / "images" / split, self.output / "labels" / split

    def load_existing(self) -> None:
        for split in ("train", "val"):
            label_path = self.output / "labels" / split / f"{self.current_base}.txt"
            if not label_path.exists():
                continue
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
                image_path = self.output / "images" / split / f"{self.current_base}.jpg"
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                height, width = image.shape[:2]
                for line in lines:
                    fields = line.split()
                    if len(fields) != 5:
                        continue
                    class_id = int(fields[0])
                    if 0 <= class_id < len(self.classes):
                        self.boxes.setdefault(class_id, []).append(
                            denormalize_box([float(value) for value in fields[1:]], width, height)
                        )
                self.saved.add(self.index)
            except (OSError, ValueError):
                continue

    def image_to_display(self, point: tuple[int, int]) -> tuple[int, int]:
        return (int(round(point[0] * self.display_scale)), int(round(point[1] * self.display_scale)))

    def display_to_image(self, point: tuple[int, int]) -> tuple[int, int]:
        width, height = self.frame.shape[1], self.frame.shape[0]
        return (
            max(0, min(width - 1, int(round(point[0] / max(self.display_scale, 1e-6))))),
            max(0, min(height - 1, int(round(point[1] / max(self.display_scale, 1e-6))))),
        )

    def render(self) -> np.ndarray:
        height, width = self.frame.shape[:2]
        max_width, max_height = 1500, 900
        self.display_scale = min(1.0, max_width / max(width, 1), max_height / max(height, 1))
        display = cv2.resize(
            self.frame,
            (max(1, int(round(width * self.display_scale))), max(1, int(round(height * self.display_scale)))),
            interpolation=cv2.INTER_AREA if self.display_scale < 1.0 else cv2.INTER_LINEAR,
        )
        self.display_size = (display.shape[1], display.shape[0])
        for class_id, boxes in self.boxes.items():
            for box_index, box in enumerate(boxes):
                x0, y0 = self.image_to_display((box[0], box[1]))
                x1, y1 = self.image_to_display((box[2], box[3]))
                cv2.rectangle(display, (x0, y0), (x1, y1), COLORS[class_id], 2)
                label = self.classes[class_id]
                if class_id == 2:
                    label = f"button {box_index + 1}/{self.button_count}"
                cv2.putText(display, label, (x0 + 5, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[class_id], 2, cv2.LINE_AA)
        if self.drag_start is not None and self.drag_end is not None:
            start = self.image_to_display(self.drag_start)
            end = self.image_to_display(self.drag_end)
            cv2.rectangle(display, start, end, COLORS[self.active_class], 2)
        help_text = (
            f"{self.index + 1}/{self.source.count}  frame={self.source_frame}  "
            f"[1] outer  [2] inner  "
            f"{'[3] button ' + str(len(self.boxes.get(2, []))) + '/' + str(self.button_count) if self.include_buttons else ''}  "
            f"active={self.classes[self.active_class]}"
        )
        cv2.putText(display, help_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        shortcuts = "drag=draw  s=save  n/space=next  p=previous  x=clear active  r=clear all  q=quit"
        if self.include_buttons:
            shortcuts += "  z=undo button"
        cv2.putText(display, shortcuts, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        return display

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.display_to_image((x, y))
            self.drag_end = self.drag_start
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = self.display_to_image((x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = self.display_to_image((x, y))
            x0, x1 = sorted((self.drag_start[0], self.drag_end[0]))
            y0, y1 = sorted((self.drag_start[1], self.drag_end[1]))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                box = (x0, y0, x1, y1)
                if self.active_class == 2:
                    buttons = self.boxes.setdefault(2, [])
                    if len(buttons) >= self.button_count:
                        print(f"本帧已经有 {self.button_count} 个按键；按 z 撤销最后一个后再补画。")
                    else:
                        buttons.append(box)
                        self.dirty = True
                else:
                    self.boxes[self.active_class] = [box]
                    self.dirty = True
            self.drag_start = None
            self.drag_end = None

    def save(self) -> None:
        image_dir, label_dir = self.split_dirs()
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{self.current_base}.jpg"
        label_path = label_dir / f"{self.current_base}.txt"
        cv2.imwrite(str(image_path), self.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        height, width = self.frame.shape[:2]
        lines = [
            f"{class_id} {normalize_box(box, width, height)}"
            for class_id, boxes in sorted(self.boxes.items())
            for box in boxes
        ]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.saved.add(self.index)
        self.dirty = False
        self.write_dataset_files()
        counts = ", ".join(
            f"{self.classes[class_id]}={len(boxes)}" for class_id, boxes in sorted(self.boxes.items())
        ) or "empty"
        if self.include_buttons and len(self.boxes.get(2, [])) not in (0, self.button_count):
            counts += f"（按键建议 {self.button_count} 个）"
        print(f"已保存 {self.current_base}: {counts}")

    def write_dataset_files(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        val_images = self.output / "images" / "val"
        val_path = "images/val" if val_images.exists() and any(val_images.glob("*.jpg")) else "images/train"
        yaml_text = (
            "path: .\n"
            "train: images/train\n"
            f"val: {val_path}\n"
            f"nc: {len(self.classes)}\n"
            f"names: {list(self.classes)!r}\n"
        )
        (self.output / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
        (self.output / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
        manifest = {
            "classes": list(self.classes),
            "source": str(self.source.path),
            "sampled_frames": self.source.count,
            "saved_frames": len(self.saved),
            "val_every": self.val_every,
        }
        (self.output / "annotation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_index(self, index: int) -> None:
        self.index = max(0, min(self.source.count - 1, index))
        self.frame, self.source_frame = self.source.read(self.index)
        self.boxes = {}
        self.dirty = False
        self.load_existing()

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self.render())
                key = cv2.waitKey(20) & 0xFF
                if key == ord("q") or key == 27:
                    if self.dirty:
                        self.save()
                    break
                if key == ord("1"):
                    self.active_class = 0
                elif key == ord("2"):
                    self.active_class = 1
                elif key == ord("3") and self.include_buttons:
                    self.active_class = 2
                elif key == ord("s"):
                    self.save()
                elif key in (ord("n"), ord(" ")):
                    if self.dirty:
                        self.save()
                    self.load_index(self.index + 1)
                elif key == ord("p"):
                    if self.dirty:
                        self.save()
                    self.load_index(self.index - 1)
                elif key == ord("x"):
                    if self.active_class in self.boxes:
                        self.boxes.pop(self.active_class)
                        self.dirty = True
                elif key == ord("z") and self.active_class == 2:
                    buttons = self.boxes.get(2, [])
                    if buttons:
                        buttons.pop()
                        if not buttons:
                            self.boxes.pop(2, None)
                        self.dirty = True
                elif key == ord("r"):
                    if self.boxes:
                        self.boxes.clear()
                        self.dirty = True
        finally:
            cv2.destroyAllWindows()
            self.source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="视频、单张图片、图片目录或接收会话目录")
    parser.add_argument("--output", type=Path, default=Path("datasets/geometry"), help="YOLO 数据集输出目录")
    parser.add_argument("--every", type=int, default=6, help="视频/目录每隔多少帧取一张，默认 6")
    parser.add_argument("--val-every", type=int, default=10, help="每 N 张放入 val；设 0 表示全部 train")
    parser.add_argument("--max-frames", type=int, default=0, help="最多标注多少张，0 表示不限制")
    parser.add_argument("--with-buttons", action="store_true", help="额外标记谱面按键区域（最多 8 个）")
    parser.add_argument("--button-count", type=int, default=8, help="每帧允许的按键框数量，默认 8；只作上限")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every 必须大于 0")
    if args.val_every < 0:
        parser.error("--val-every 不能为负数")
    if args.max_frames < 0:
        parser.error("--max-frames 不能为负数")
    if args.button_count < 1:
        parser.error("--button-count 必须大于 0")
    source = make_source(args.input, args.every, args.max_frames)
    print(f"共 {source.count} 张待标注图像；输出：{args.output.resolve()}")
    if args.with_buttons:
        print("标注类别：1=机台外框，2=实际内屏，3=每个谱面按键（最多 8 个）。按键类别不参与机台居中。")
    else:
        print("标注类别：1=机台外框，2=实际内屏。不标判定点、手或背景；用 --with-buttons 可添加按键类别。")
    GeometryAnnotator(
        source,
        args.output,
        args.val_every,
        include_buttons=args.with_buttons,
        button_count=args.button_count,
    ).run()


if __name__ == "__main__":
    main()
