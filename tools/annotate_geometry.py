#!/usr/bin/env python3
"""Small OpenCV tool for labeling the two MaiLens detection models.

The ``geometry`` preset labels the cabinet relationship model:

* ``outer_buttons``: one rectangle around the complete outer touch/button ring.
* ``inner_screen``: the circular gameplay screen.

The ``buttons`` preset labels the gameplay-object model:

* ``button``: one box per visible gameplay button/note.
* ``inner_screen``: the screen reference box.

The ``slides`` preset uses ``slide`` in place of ``button`` for slide notes.

It writes ordinary YOLO detection labels, so the resulting folder can be
used directly to train an Ultralytics model and then exported to ONNX.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener
except ImportError:  # Keep non-HEIC use working when the optional codec is absent.
    register_heif_opener = None

if register_heif_opener is not None:
    register_heif_opener()


GEOMETRY_CLASSES = ("outer_buttons", "inner_screen")
BUTTON_CLASSES = ("button", "inner_screen")
SLIDE_CLASSES = ("slide", "inner_screen")
COLORS = ((0, 220, 255), (80, 255, 170), (255, 150, 60))  # BGR: yellow, green, orange
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def read_image(path: Path) -> np.ndarray:
    """Read regular images with OpenCV and iPhone HEIC images through Pillow."""
    if path.suffix.lower() in {".heic", ".heif"}:
        if register_heif_opener is None:
            raise RuntimeError(
                "读取 HEIC 需要 pillow-heif；请运行 .venv\\Scripts\\python.exe -m pip install pillow-heif"
            )
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imread uses the Windows ANSI code page for paths on some builds, so
    # it fails for valid paths containing Chinese characters. pathlib handles
    # Unicode paths; imdecode only receives the image bytes.
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as error:
        raise RuntimeError(f"无法读取图像：{path}") from error
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像：{path}")
    return image


def write_jpeg(path: Path, image: np.ndarray, quality: int = 95) -> None:
    """Write JPEG bytes through pathlib so Unicode output paths also work."""
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError(f"无法编码图像：{path}")
    path.write_bytes(encoded.tobytes())


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
            image = read_image(self.image_paths[index])
            source_frame = index
        else:
            assert self._capture is not None
            source_frame = index * self.every
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = self._capture.read()
            if not ok:
                raise RuntimeError(f"无法读取视频第 {source_frame} 帧：{self.path}")
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
        classes: tuple[str, ...] = GEOMETRY_CLASSES,
        repeated_class: int | None = None,
        max_instances: int = 16,
        prelabel_model: Path | None = None,
        prelabel_detector: object | None = None,
    ) -> None:
        self.source = source
        self.output = output.expanduser().resolve()
        self.val_every = max(0, int(val_every))
        self.classes = tuple(classes)
        if not self.classes:
            raise ValueError("至少需要一个标注类别")
        self.repeated_class = repeated_class if repeated_class in range(len(self.classes)) else None
        self.max_instances = max(1, int(max_instances))
        self.prelabel_detector = prelabel_detector
        if prelabel_model is not None:
            if self.classes != GEOMETRY_CLASSES:
                raise ValueError("几何模型预标注只支持 geometry 类别")
            if self.prelabel_detector is not None:
                raise ValueError("prelabel_model 和 prelabel_detector 只能设置一个")
            project_root = Path(__file__).resolve().parents[1]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from pc.process_session import GeometryDetector
            self.prelabel_detector = GeometryDetector(prelabel_model)
            if not self.prelabel_detector.enabled:
                raise RuntimeError(f"无法加载预标注模型：{prelabel_model}")
        self.index = 0
        self.active_class = 0
        self.boxes: dict[int, list[tuple[int, int, int, int]]] = {}
        self.predicted_classes: set[int] = set()
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.display_scale = 1.0
        self.display_size = (0, 0)
        self.saved: set[int] = set()
        self.dirty = False
        self.frame, self.source_frame = self.source.read(0)
        if not self.load_existing():
            self.apply_model_suggestions()
        resume_index = self.first_unlabeled_index()
        if resume_index is not None and resume_index != self.index:
            self.load_index(resume_index)

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

    def load_existing(self) -> bool:
        loaded = False
        for split in ("train", "val"):
            label_path = self.output / "labels" / split / f"{self.current_base}.txt"
            if not label_path.exists():
                continue
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
                image_path = self.output / "images" / split / f"{self.current_base}.jpg"
                if not image_path.exists():
                    continue
                image = read_image(image_path)
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
                loaded = True
            except (OSError, ValueError):
                continue
        return loaded

    def first_unlabeled_index(self) -> int | None:
        """Resume at the first sampled frame without a saved image/label pair."""
        for index in range(self.source.count):
            base = f"frame-{index + 1:06d}"
            exists = any(
                (self.output / "labels" / split / f"{base}.txt").is_file()
                and (self.output / "images" / split / f"{base}.jpg").is_file()
                for split in ("train", "val")
            )
            if exists:
                self.saved.add(index)
            else:
                return index
        return None

    def apply_model_suggestions(self) -> None:
        """Seed editable boxes from the geometry model when no reviewed label exists."""
        self.boxes = {}
        self.predicted_classes.clear()
        if self.prelabel_detector is None or self.classes != GEOMETRY_CLASSES:
            return
        outer, inner = self.prelabel_detector.detect(self.frame)
        if outer is not None:
            self.boxes[0] = [tuple(int(value) for value in outer)]
            self.predicted_classes.add(0)
        if inner is not None:
            self.boxes[1] = [tuple(int(value) for value in inner)]
            self.predicted_classes.add(1)
        # Keep the existing auto-save-on-next behavior for AI-prefilled frames;
        # dragging a corrected box replaces the proposal before it is saved.
        self.dirty = bool(self.predicted_classes)

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
                if class_id in self.predicted_classes:
                    label += " [AI]"
                if class_id == self.repeated_class:
                    label = f"{label} {box_index + 1}/{self.max_instances}"
                cv2.putText(display, label, (x0 + 5, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[class_id], 2, cv2.LINE_AA)
        if self.drag_start is not None and self.drag_end is not None:
            start = self.image_to_display(self.drag_start)
            end = self.image_to_display(self.drag_end)
            cv2.rectangle(display, start, end, COLORS[self.active_class], 2)
        class_help = "  ".join(
            f"[{class_id + 1}] {name}"
            + (f" {len(self.boxes.get(class_id, []))}/{self.max_instances}" if class_id == self.repeated_class else "")
            for class_id, name in enumerate(self.classes)
        )
        help_text = (
            f"{self.index + 1}/{self.source.count}  frame={self.source_frame}  "
            f"{class_help}  active={self.classes[self.active_class]}"
        )
        cv2.putText(display, help_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        shortcuts = "drag=draw/replace AI box  s=save  n/space=next  p=previous  x=clear active  r=clear all  q=quit"
        if self.repeated_class is not None:
            shortcuts += f"  z=undo {self.classes[self.repeated_class]}"
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
                if self.active_class == self.repeated_class:
                    repeated = self.boxes.setdefault(self.active_class, [])
                    if len(repeated) >= self.max_instances:
                        print(f"本帧已经有 {self.max_instances} 个 {self.classes[self.active_class]}；按 z 撤销最后一个后再补画。")
                    else:
                        repeated.append(box)
                        self.dirty = True
                else:
                    self.boxes[self.active_class] = [box]
                    self.predicted_classes.discard(self.active_class)
                    self.dirty = True
            self.drag_start = None
            self.drag_end = None

    def save(self) -> None:
        image_dir, label_dir = self.split_dirs()
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{self.current_base}.jpg"
        label_path = label_dir / f"{self.current_base}.txt"
        write_jpeg(image_path, self.frame, quality=95)
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
        if self.repeated_class is not None and self.boxes.get(self.repeated_class):
            counts += f"（{self.classes[self.repeated_class]} 可有多个实例）"
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
        (self.output / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
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
        self.predicted_classes.clear()
        self.dirty = False
        if not self.load_existing():
            self.apply_model_suggestions()

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
                if ord("1") <= key <= ord("9"):
                    selected = key - ord("1")
                    if selected < len(self.classes):
                        self.active_class = selected
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
                elif key == ord("z") and self.active_class == self.repeated_class:
                    repeated = self.boxes.get(self.repeated_class, [])
                    if repeated:
                        repeated.pop()
                        if not repeated:
                            self.boxes.pop(self.repeated_class, None)
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
    parser.add_argument(
        "--preset",
        choices=("geometry", "buttons", "slides"),
        default="geometry",
        help="geometry=外圈按键+内屏；buttons=按键+内屏；slides=滑条+内屏",
    )
    parser.add_argument("--every", type=int, default=6, help="视频/目录每隔多少帧取一张，默认 6")
    parser.add_argument("--val-every", type=int, default=10, help="每 N 张放入 val；设 0 表示全部 train")
    parser.add_argument("--max-frames", type=int, default=0, help="最多标注多少张，0 表示不限制")
    parser.add_argument(
        "--model",
        type=Path,
        help="可选几何模型；先画 outer_buttons/inner_screen 预测框，拖动重画即可替换",
    )
    parser.add_argument(
        "--with-buttons",
        dest="legacy_buttons",
        action="store_true",
        help="兼容旧命令，等同于 --preset buttons",
    )
    parser.add_argument(
        "--max-instances",
        "--button-count",
        dest="max_instances",
        type=int,
        default=16,
        help="重复目标每帧的框数量上限，默认 16",
    )
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every 必须大于 0")
    if args.val_every < 0:
        parser.error("--val-every 不能为负数")
    if args.max_frames < 0:
        parser.error("--max-frames 不能为负数")
    if args.max_instances < 1:
        parser.error("--max-instances 必须大于 0")
    if args.legacy_buttons:
        args.preset = "buttons"
    if args.model and args.preset != "geometry":
        parser.error("--model 预标注目前只支持 --preset geometry")
    source = make_source(args.input, args.every, args.max_frames)
    print(f"共 {source.count} 张待标注图像；输出：{args.output.resolve()}")
    presets = {
        "geometry": (GEOMETRY_CLASSES, None),
        "buttons": (BUTTON_CLASSES, 0),
        "slides": (SLIDE_CLASSES, 0),
    }
    classes, repeated_class = presets[args.preset]
    print(f"标注模型：{args.preset}；类别：{', '.join(classes)}")
    if args.model:
        print(f"AI 预标注：{args.model.resolve()}；错误框用鼠标拖出正确框替换，类别框显示 [AI]")
    if repeated_class is not None:
        print(f"第 1 类可以在同一帧重复框选，最多 {args.max_instances} 个；按 z 撤销最后一个。")
    else:
        print("外圈按键只画一个整体框，不要把 8 个外圈按键拆成 8 个框。")
    GeometryAnnotator(
        source,
        args.output,
        args.val_every,
        classes=classes,
        repeated_class=repeated_class,
        max_instances=args.max_instances,
        prelabel_model=args.model,
    ).run()


if __name__ == "__main__":
    main()
