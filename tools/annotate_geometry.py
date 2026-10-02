#!/usr/bin/env python3
"""Small OpenCV tool for labeling the two MaiLens detection models.

The ``geometry`` preset labels the cabinet relationship model:

* ``outer_buttons``: one rectangle around the complete outer touch/button ring.
* ``inner_screen``: the circular gameplay screen.

The ``buttons`` preset labels the gameplay-object model:

* ``button``: one box per visible gameplay button/note.
* ``inner_screen``: the screen reference box.

The ``slides`` preset uses ``slide`` in place of ``button`` for slide notes.

``--gaps`` adds a second layer on top of any preset: the four distances between
the inner screen edge and the outer button ring, one per side.  That is the
blogger's own criterion -- a distortion-free, dead-on picture has the same gap
on all four sides -- so the tool prints every side as it would read if the
average were the real 75 mm, and flags the sides that do not.

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
GAP_COLOR = (255, 255, 255)
GAP_BAD_COLOR = (60, 60, 255)
# Click order for the gap layer: inner edge then outer edge, per side.
GAP_SIDES = ("left", "right", "top", "bottom")
# OpenCV's text renderer only draws ASCII, so the on-screen layer uses these
# and the terminal output uses the Chinese names.
GAP_SIDE_SHORT = {"left": "L", "right": "R", "top": "T", "bottom": "B"}
GAP_SIDE_NAMES = {"left": "左", "right": "右", "top": "上", "bottom": "下"}
INNER_EDGE = "inner"
OUTER_EDGE = "outer"
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
        gaps: bool = False,
        gap_target: float = 75.0,
        gap_tolerance: float = 0.08,
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
        self.gaps_enabled = bool(gaps)
        self.gap_target = float(gap_target)
        self.gap_tolerance = max(float(gap_tolerance), 0.0)
        self.gap_mode = False
        self.gap_marks: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        self.gap_pending: list[tuple[int, int]] = []
        self.predicted_classes: set[int] = set()
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.display_scale = 1.0
        self.display_size = (0, 0)
        self.saved: set[int] = set()
        self.dirty = False
        self.unreadable: list[int] = []
        self.frame, self.source_frame, self.index = self.read_readable(0)
        if not self.load_existing():
            self.apply_model_suggestions()
        self.load_gaps()
        resume_index = self.first_unlabeled_index()
        if resume_index is not None and resume_index != self.index:
            self.load_index(resume_index)

    @property
    def current_base(self) -> str:
        return f"frame-{self.index + 1:06d}"

    # ------------------------------------------------------------- gap layer
    def gap_path(self, index: int | None = None) -> Path:
        base = f"frame-{(self.index if index is None else index) + 1:06d}"
        return self.output / "gaps" / f"{base}.json"

    def gap_values(self) -> dict[str, float]:
        """The four measured gaps in image pixels, by side."""
        out: dict[str, float] = {}
        for side in GAP_SIDES:
            pair = self.gap_marks.get(side)
            if pair is None:
                continue
            inner, outer = pair
            out[side] = float(np.hypot(outer[0] - inner[0], outer[1] - inner[1]))
        return out

    def gap_readout(self) -> dict:
        """Gaps in pixels, plus what each side would read if 75 mm were the mean.

        The real distance between the screen edge and the button ring is a fixed
        75 mm, so a picture with no distortion left has the same *pixel* gap on
        every side.  Scaling the four so their average is the target turns that
        into a number the operator can read directly: four 75s is dead on, and
        anything else names the side that is still pulled.
        """
        values = self.gap_values()
        info: dict = {"px": values, "mm": {}, "mean": None, "spread": None,
                      "missing": [GAP_SIDE_NAMES[s] for s in GAP_SIDES if s not in values]}
        if not values:
            return info
        mean = float(np.mean(list(values.values())))
        info["mean"] = mean
        info["spread"] = float(max(values.values()) - min(values.values())) / max(mean, 1e-6)
        for side, value in values.items():
            info["mm"][side] = self.gap_target * value / max(mean, 1e-6)
        info["ok"] = (
            not info["missing"]
            and info["spread"] <= self.gap_tolerance
        )
        return info

    def load_gaps(self) -> None:
        self.gap_marks = {}
        self.gap_pending = []
        if not self.gaps_enabled:
            return
        path = self.gap_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for side, values in (payload.get("marks") or {}).items():
            if side in GAP_SIDES and len(values) == 2:
                self.gap_marks[side] = (
                    (int(round(values[0][0])), int(round(values[0][1]))),
                    (int(round(values[1][0])), int(round(values[1][1]))),
                )

    def save_gaps(self) -> None:
        if not self.gaps_enabled:
            return
        info = self.gap_readout()
        if not info["px"]:
            return
        path = self.gap_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frame": self.index + 1,
            "source_frame": self.source_frame,
            "marks": {side: [list(pair[0]), list(pair[1])]
                      for side, pair in self.gap_marks.items()},
            "gap_px": info["px"],
            "gap_mm": info["mm"],
            "mean_px": info["mean"],
            "spread": info["spread"],
            "target_mm": self.gap_target,
            "dead_on": bool(info.get("ok")),
            "image": f"images/{self.split_name()}/{self.current_base}.jpg",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def auto_gaps_from_boxes(self) -> bool:
        """Seed the four gaps from the two geometry boxes.

        Only meaningful on a picture that is already upright and dead on --
        which is what the lock delivers.  On a raw fisheye frame the boxes are
        axis-aligned while the machine is not, so the four numbers would be
        measuring the frame rather than the cabinet.
        """
        outer = self.boxes.get(0)
        inner = self.boxes.get(1)
        if not outer or not inner:
            return False
        ox0, oy0, ox1, oy1 = outer[0]
        ix0, iy0, ix1, iy1 = inner[0]
        cx = (ix0 + ix1) * 0.5
        cy = (iy0 + iy1) * 0.5
        self.gap_marks = {
            "left": ((ix0, int(cy)), (ox0, int(cy))),
            "right": ((ix1, int(cy)), (ox1, int(cy))),
            "top": ((int(cx), iy0), (int(cx), oy0)),
            "bottom": ((int(cx), iy1), (int(cx), oy1)),
        }
        self.gap_pending = []
        return True

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

    def put_text(self, display: np.ndarray, text: str, origin: tuple[int, int],
                 colour: tuple[int, int, int], scale: float = 0.52) -> None:
        """Draw a caption that shrinks with the preview.

        cv2 draws text in display pixels, so a down-scaled photo used to carry
        full-size captions that ran into each other.  Everything textual goes
        through here so it stays in proportion with the picture.
        """
        cv2.putText(display, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale * self.text_scale, colour,
                    max(1, int(round(2 * self.text_scale))), cv2.LINE_AA)

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
        self.text_scale = max(0.4, min(1.0, self.display_scale))
        display = cv2.resize(
            self.frame,
            (max(1, int(round(width * self.display_scale))), max(1, int(round(height * self.display_scale)))),
            interpolation=cv2.INTER_AREA if self.display_scale < 1.0 else cv2.INTER_LINEAR,
        )
        self.display_size = (display.shape[1], display.shape[0])
        stroke = max(1, int(round(2 * self.text_scale)))
        for class_id, boxes in self.boxes.items():
            for box_index, box in enumerate(boxes):
                x0, y0 = self.image_to_display((box[0], box[1]))
                x1, y1 = self.image_to_display((box[2], box[3]))
                cv2.rectangle(display, (x0, y0), (x1, y1), COLORS[class_id], stroke)
                label = self.classes[class_id]
                if class_id in self.predicted_classes:
                    label += " [AI]"
                if class_id == self.repeated_class:
                    label = f"{label} {box_index + 1}/{self.max_instances}"
                self.put_text(display, label, (x0 + 5, max(20, y0 - 8)),
                              COLORS[class_id], 0.65)
        if self.drag_start is not None and self.drag_end is not None:
            start = self.image_to_display(self.drag_start)
            end = self.image_to_display(self.drag_end)
            cv2.rectangle(display, start, end, COLORS[self.active_class], stroke)
        gap_line = ""
        if self.gaps_enabled:
            self.draw_gaps(display)
            gap_line = self.gap_caption()
        class_help = "  ".join(
            f"[{class_id + 1}] {name}"
            + (f" {len(self.boxes.get(class_id, []))}/{self.max_instances}" if class_id == self.repeated_class else "")
            for class_id, name in enumerate(self.classes)
        )
        help_text = (
            f"{self.index + 1}/{self.source.count}  frame={self.source_frame}  "
            f"{class_help}  active={self.classes[self.active_class]}"
        )
        shortcuts = "drag=draw/replace  s=save  n/space=next  p=prev  x=clear  r=clear all  q=quit"
        if self.repeated_class is not None:
            shortcuts += f"  z=undo {self.classes[self.repeated_class]}"
        lines = [(help_text, (255, 255, 255), 0.52), (shortcuts, (255, 255, 255), 0.48)]
        if self.gaps_enabled:
            mode = "GAP mode" if self.gap_mode else "box mode"
            hint = (f"g={mode}  then click L R T B, screen edge then button edge  "
                    "a=auto from boxes  z=undo  x=clear")
            lines.append((hint, (200, 255, 200), 0.48))
            if gap_line:
                lines.append((gap_line, GAP_COLOR, 0.52))
        step = max(16, int(round(22 * self.text_scale)))
        y = max(14, int(round(24 * self.text_scale)))
        for text, colour, scale in lines:
            self.put_text(display, text, (12, y), colour, scale)
            y += step
        return display

    def gap_caption(self) -> str:
        """One line: every side in pixels, and what it would read at the target."""
        info = self.gap_readout()
        if not info["px"]:
            next_side = GAP_SIDES[min(len(self.gap_pending) // 2, len(GAP_SIDES) - 1)]
            first = len(self.gap_pending) % 2 == 0
            which = INNER_EDGE if first else OUTER_EDGE
            left = len(GAP_SIDES) * 2 - len(self.gap_pending)
            return (f"gaps: nothing marked yet, next click = "
                    f"{GAP_SIDE_SHORT[next_side]} {which}  ({left} clicks to go)")
        parts = []
        for side in GAP_SIDES:
            if side not in info["px"]:
                parts.append(f"{GAP_SIDE_SHORT[side]} --")
                continue
            parts.append(f"{GAP_SIDE_SHORT[side]} {info['px'][side]:.0f}px/"
                         f"{info['mm'][side]:.1f}")
        verdict = (f"even -> dead on, 4 x {self.gap_target:.0f}" if info.get("ok")
                   else f"uneven {info['spread'] * 100:.1f}%  (want 4 x {self.gap_target:.0f})")
        return "gap " + "  ".join(parts) + "   " + verdict

    def draw_gaps(self, display: np.ndarray) -> None:
        info = self.gap_readout()
        for side, pair in self.gap_marks.items():
            inner = self.image_to_display(pair[0])
            outer = self.image_to_display(pair[1])
            bad = (side not in info["mm"]
                   or abs(info["mm"][side] - self.gap_target) > self.gap_target * self.gap_tolerance)
            color = GAP_BAD_COLOR if bad else GAP_COLOR
            cv2.line(display, inner, outer, color, 3, cv2.LINE_AA)
            cv2.circle(display, inner, 4, color, -1)
            cv2.circle(display, outer, 4, color, -1)
            middle = ((inner[0] + outer[0]) // 2, (inner[1] + outer[1]) // 2)
            label = (f"{GAP_SIDE_SHORT[side]} {info['px'][side]:.0f}px"
                     if side in info["px"] else GAP_SIDE_SHORT[side])
            if side in info["mm"]:
                label += f" (={info['mm'][side]:.1f})"
            self.put_text(display, label, (middle[0] + 6, middle[1] - 6),
                          color, 0.55)
        for index, point in enumerate(self.gap_pending):
            position = self.image_to_display(point)
            cv2.drawMarker(display, position, (255, 120, 255), cv2.MARKER_CROSS,
                           max(10, int(round(18 * self.text_scale))),
                           max(1, int(round(2 * self.text_scale))))
            step = index // 2
            edge = INNER_EDGE if index % 2 == 0 else OUTER_EDGE
            self.put_text(display, f"{GAP_SIDE_SHORT[GAP_SIDES[step]]} {edge}",
                          (position[0] + 8, position[1] + 20), (255, 120, 255), 0.5)

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.gaps_enabled and self.gap_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.add_gap_point(self.display_to_image((x, y)))
            return
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

    def add_gap_point(self, point: tuple[int, int]) -> None:
        """One click of the gap walk: inner edge, then outer edge, four times."""
        if len(self.gap_pending) >= len(GAP_SIDES) * 2:
            self.gap_pending = []
        self.gap_pending.append(point)
        if len(self.gap_pending) % 2 == 0:
            side = GAP_SIDES[len(self.gap_pending) // 2 - 1]
            self.gap_marks[side] = (self.gap_pending[-2], self.gap_pending[-1])
            info = self.gap_readout()
            pixels = info["px"][side]
            print(f"{GAP_SIDE_NAMES[side]} 间距 {pixels:.1f}px")
            if not info["missing"]:
                detail = " ".join(f"{GAP_SIDE_NAMES[s]}={info['mm'][s]:.1f}"
                                  for s in GAP_SIDES)
                verdict = ("四边一致 [OK]" if info.get("ok")
                           else f"还不齐（差 {info['spread'] * 100:.1f}%）")
                print(f"  四个间距都标好了: {detail}  {verdict}")
        self.dirty = True

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
        self.save_gaps()
        counts = ", ".join(
            f"{self.classes[class_id]}={len(boxes)}" for class_id, boxes in sorted(self.boxes.items())
        ) or "empty"
        if self.repeated_class is not None and self.boxes.get(self.repeated_class):
            counts += f"（{self.classes[self.repeated_class]} 可有多个实例）"
        print(f"已保存 {self.current_base}: {counts}")
        if self.gaps_enabled:
            info = self.gap_readout()
            if info["px"]:
                detail = "  ".join(
                    f"{GAP_SIDE_NAMES[s]} {info['px'][s]:.0f}px"
                    f"→{info['mm'][s]:.1f}"
                    for s in GAP_SIDES if s in info["px"]
                )
                verdict = ("四边一致 [OK]" if info.get("ok")
                           else f"还不正（差 {info['spread'] * 100:.1f}%）")
                print(f"    间距 {detail}  {verdict}")

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
        self.frame, self.source_frame, self.index = self.read_readable(index)
        self.boxes = {}
        self.predicted_classes.clear()
        self.dirty = False
        if not self.load_existing():
            self.apply_model_suggestions()
        self.load_gaps()

    def read_readable(self, index: int) -> tuple[np.ndarray, int, int]:
        """Read a frame, stepping past entries OpenCV cannot decode.

        One unreadable file in a folder of screenshots used to end the whole
        session on the spot.  Skipping it keeps the rest of the set usable.
        """
        if self.source.count <= 0:
            raise RuntimeError("输入里没有可读取的帧")
        start = max(0, min(self.source.count - 1, index))
        for step in range(self.source.count):
            candidate = (start + step) % self.source.count
            try:
                frame, source_frame = self.source.read(candidate)
            except (RuntimeError, IndexError) as error:
                if candidate not in self.unreadable:
                    self.unreadable.append(candidate)
                    print(f"跳过无法读取的帧 {candidate}：{error}", flush=True)
                continue
            return frame, source_frame, candidate
        raise RuntimeError("输入里的每一帧都无法读取")

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
                    if self.gaps_enabled and self.gap_mode and self.gap_marks:
                        self.gap_marks.clear()
                        self.gap_pending = []
                        self.dirty = True
                    elif self.active_class in self.boxes:
                        self.boxes.pop(self.active_class)
                        self.dirty = True
                elif key == ord("z"):
                    if self.gaps_enabled and self.gap_mode and self.gap_pending:
                        self.gap_pending.pop()
                        self.dirty = True
                    elif self.active_class == self.repeated_class:
                        repeated = self.boxes.get(self.repeated_class, [])
                        if repeated:
                            repeated.pop()
                            if not repeated:
                                self.boxes.pop(self.repeated_class, None)
                            self.dirty = True
                elif key == ord("g") and self.gaps_enabled:
                    self.gap_mode = not self.gap_mode
                    self.gap_pending = []
                    print("间距模式：依次点 左内屏→左外键、右内屏→右外键、上、下（8 次点击）"
                          if self.gap_mode else "回到框模式")
                elif key == ord("a") and self.gaps_enabled:
                    if self.auto_gaps_from_boxes():
                        info = self.gap_readout()
                        detail = "  ".join(f"{GAP_SIDE_NAMES[s]}={info['mm'][s]:.1f}"
                                           for s in GAP_SIDES)
                        print(f"按两个框自动填入间距：{detail}")
                        self.dirty = True
                    else:
                        print("先画好 outer_buttons 和 inner_screen 两个框，再按 a 自动填间距。")
                elif key == ord("r"):
                    if self.boxes or self.gap_marks:
                        self.boxes.clear()
                        self.gap_marks.clear()
                        self.gap_pending = []
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
    parser.add_argument(
        "--gaps",
        action="store_true",
        help="加一层间距标注：内屏到外键的左/右/上/下四个距离，四边都等于目标值才是正对",
    )
    parser.add_argument(
        "--gap-target",
        type=float,
        default=75.0,
        help="外键与内屏的真实间距，单位随意，默认 75（mm）；四个读数都接近它才算正对",
    )
    parser.add_argument(
        "--gap-tolerance",
        type=float,
        default=0.08,
        help="四边间距允许的相对差，默认 0.08 即 8%",
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
    if args.gap_target <= 0:
        parser.error("--gap-target 必须大于 0")
    if args.gap_tolerance < 0:
        parser.error("--gap-tolerance 不能为负数")
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
    if args.gaps:
        print(f"间距层已打开：按 g 进入间距模式，依次点 "
              + "、".join(f"{GAP_SIDE_NAMES[s]}内屏边→外键边" for s in GAP_SIDES)
              + f"；每边会同时给出像素值和「平均等于 {args.gap_target:g}」时的读数，"
                f"四个读数一致（差 ≤{args.gap_tolerance * 100:.0f}%）就是最正对、最不畸变的画面。")
    GeometryAnnotator(
        source,
        args.output,
        args.val_every,
        classes=classes,
        repeated_class=repeated_class,
        max_instances=args.max_instances,
        prelabel_model=args.model,
        gaps=args.gaps,
        gap_target=args.gap_target,
        gap_tolerance=args.gap_tolerance,
    ).run()


if __name__ == "__main__":
    main()
