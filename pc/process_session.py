#!/usr/bin/env python3
"""Offline PC processor for the MaiLens remote-capture sessions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def quat_to_matrix(q: dict[str, float]) -> np.ndarray:
    """Return a camera-space rotation matrix for x/y/z/w quaternion data."""
    x, y, z, w = (float(q.get(key, 0.0)) for key in ("x", "y", "z", "w"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def rotation_for_row(row: dict, reference: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    pose = row.get("pose") or {}
    quaternion = pose.get("quaternion")
    if not quaternion:
        return np.eye(3, dtype=np.float32), reference
    current = quat_to_matrix(quaternion)
    if reference is None:
        reference = current.copy()
    # Core Motion attitude is expressed in device coordinates, while remap
    # rays are expressed in the rear-camera image coordinates (+X right,
    # +Y down, +Z out through the lens). Convert between those bases before
    # applying q_current^-1 * q_locked. This matches the live iOS renderer.
    camera_to_device = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    relative_device = current.T @ reference
    camera_from_locked = camera_to_device @ relative_device @ camera_to_device
    return camera_from_locked, reference


def make_output_rays(width: int, height: int, crop: float, fov_deg: float) -> np.ndarray:
    """Precompute rectilinear output rays so live processing reuses the grid."""
    virtual_focal = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xx - width * 0.5) / max(virtual_focal * crop, 1.0)
    y = (yy - height * 0.5) / max(virtual_focal * crop, 1.0)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def build_remap(
    width: int,
    height: int,
    rotation: np.ndarray,
    crop: float,
    fov_deg: float,
    k1: float,
    k2: float,
    center_x: float,
    center_y: float,
    output_rays: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a stabilized rectilinear output ray into the raw clip-on fisheye."""
    source_focal = max(width, height) * 772.4089 / 4032.0
    rays = output_rays if output_rays is not None else make_output_rays(width, height, crop, fov_deg)
    # OpenCV's C implementation is substantially faster than a Python-side
    # matrix multiply for the live path. The output ray is a column vector,
    # so this is equivalent to `rays @ rotation.T` for the row-shaped grid.
    source = cv2.transform(rays, rotation)
    radial = cv2.magnitude(source[..., 0], source[..., 1])
    theta = np.arccos(np.clip(source[..., 2], -1.0, 1.0))
    theta2 = theta * theta
    theta_distorted = theta * (1.0 + k1 * theta2 + k2 * theta2 * theta2)
    safe_radial = np.maximum(radial, 1e-6)
    direction_x = source[..., 0] / safe_radial
    direction_y = source[..., 1] / safe_radial
    map_x = (center_x * width + source_focal * direction_x * theta_distorted).astype(np.float32)
    map_y = (center_y * height + source_focal * direction_y * theta_distorted).astype(np.float32)
    return map_x, map_y


class GeometryDetector:
    def __init__(self, model_path: Path | None):
        self.model = None
        self.cv_net = None
        self.backend = "none"
        self.input_size = 640
        self.confidence = 0.30
        self.names = {0: "outer_frame", 1: "inner_screen"}
        if model_path:
            model_path = Path(model_path)
            opencv_error = None
            if model_path.suffix.lower() == ".onnx":
                try:
                    self.cv_net = cv2.dnn.readNetFromONNX(str(model_path))
                    self.backend = "opencv-dnn"
                    return
                except Exception as error:
                    opencv_error = error

            try:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(str(model_path), task="detect")
                self.backend = "ultralytics"
            except (ImportError, ModuleNotFoundError) as error:
                detail = f"；OpenCV DNN 错误：{opencv_error}" if opencv_error else ""
                if model_path.suffix.lower() == ".onnx":
                    raise RuntimeError(f"无法加载 ONNX 检测模型：{model_path}{detail}") from error
                raise RuntimeError("使用非 ONNX 模型前请先安装可用的 ultralytics/torch") from error
            except Exception as error:
                if model_path.suffix.lower() == ".onnx":
                    detail = f"；OpenCV DNN 错误：{opencv_error}" if opencv_error else ""
                    raise RuntimeError(f"无法加载检测模型：{model_path}{detail}") from error
                raise RuntimeError(f"无法加载检测模型：{model_path}") from error

    @property
    def enabled(self) -> bool:
        return self.model is not None or self.cv_net is not None

    def detect(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        if not self.enabled:
            return None, None
        if self.cv_net is not None:
            boxes = self._detect_opencv(frame)
        else:
            result = self.model.predict(frame, imgsz=self.input_size, conf=self.confidence, device="cpu", verbose=False)[0]
            names = result.names
            boxes = []
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls = int(box.cls[0].item())
                label_value = names[cls] if isinstance(names, (list, tuple)) else names.get(cls, cls)
                boxes.append((str(label_value).lower(), float(box.conf[0].item()), tuple(xyxy)))

        return self._pick_geometry_boxes(boxes)

    def _detect_opencv(self, frame: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Run a YOLO export through OpenCV DNN without torch/onnxruntime."""
        height, width = frame.shape[:2]
        scale = min(self.input_size / max(width, 1), self.input_size / max(height, 1))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=True)
        self.cv_net.setInput(blob)
        output = np.squeeze(self.cv_net.forward())
        if output.ndim != 2:
            return []
        # Ultralytics YOLO exports are usually [classes+4, candidates].
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] <= 4:
            return []

        candidates: list[tuple[str, float, tuple[int, int, int, int], int]] = []
        for row in output:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.confidence:
                continue
            center_x, center_y, box_width, box_height = (float(value) for value in row[:4])
            x0 = int(round((center_x - box_width * 0.5 - pad_x) / max(scale, 1e-6)))
            y0 = int(round((center_y - box_height * 0.5 - pad_y) / max(scale, 1e-6)))
            x1 = int(round((center_x + box_width * 0.5 - pad_x) / max(scale, 1e-6)))
            y1 = int(round((center_y + box_height * 0.5 - pad_y) / max(scale, 1e-6)))
            box = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            candidates.append((self.names.get(class_id, str(class_id)), confidence, box, class_id))

        # Keep one stable candidate per class. The final geometry selector
        # chooses the most confident valid outer and inner boxes.
        selected: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for class_id in sorted({candidate[3] for candidate in candidates}):
            class_candidates = [candidate for candidate in candidates if candidate[3] == class_id]
            boxes = [[candidate[2][0], candidate[2][1], candidate[2][2] - candidate[2][0], candidate[2][3] - candidate[2][1]] for candidate in class_candidates]
            scores = [candidate[1] for candidate in class_candidates]
            keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, 0.45)
            for index in np.asarray(keep).reshape(-1).tolist() if len(keep) else []:
                label, confidence, box, _ = class_candidates[int(index)]
                selected.append((label, confidence, box))
        return selected

    @staticmethod
    def _pick_geometry_boxes(boxes: list[tuple[str, float, tuple[int, int, int, int]]]):
        # Normalize labels so both Ultralytics and OpenCV DNN backends share
        # exactly the same outer/inner selection and overlap validation.
        boxes = [(str(label).lower(), float(confidence), tuple(box)) for label, confidence, box in boxes]

        def area(box: tuple[int, int, int, int]) -> int:
            return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

        outer_candidates = [
            (confidence, box)
            for label, confidence, box in boxes
            if any(token in label for token in ("outer", "frame", "cabinet", "machine", "arcade"))
        ]
        inner_candidates = [
            (confidence, box)
            for label, confidence, box in boxes
            if any(token in label for token in ("inner", "screen", "display"))
        ]
        outer = max(outer_candidates, key=lambda item: (item[0], area(item[1])))[1] if outer_candidates else None
        inner = max(inner_candidates, key=lambda item: (item[0], area(item[1])))[1] if inner_candidates else None
        if outer is None and boxes:
            # Keep the detector usable with a one-class machine model while
            # avoiding the old behaviour of treating a second outer box as an
            # inner screen.
            outer = max((box for _, _, box in boxes), key=area)
        if inner is not None and outer is not None:
            ix0, iy0, ix1, iy1 = inner
            ox0, oy0, ox1, oy1 = outer
            intersection = max(0, min(ix1, ox1) - max(ix0, ox0)) * max(0, min(iy1, oy1) - max(iy0, oy0))
            if intersection / max(area(inner), 1) < 0.35:
                inner = None
        return outer, inner


def draw_debug(frame: np.ndarray, outer, inner, text: str) -> np.ndarray:
    result = frame.copy()
    if outer:
        cv2.rectangle(result, outer[:2], outer[2:], (40, 210, 255), 2)
    if inner:
        cv2.rectangle(result, inner[:2], inner[2:], (80, 255, 170), 2)
    cv2.putText(result, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 190), 2, cv2.LINE_AA)
    return result


def transform_box(box, matrix: np.ndarray):
    if box is None:
        return None
    points = np.float32([
        [box[0], box[1]], [box[2], box[1]],
        [box[2], box[3]], [box[0], box[3]],
    ])
    transformed = cv2.transform(points[None, :, :], matrix)[0]
    x0, y0 = transformed.min(axis=0)
    x1, y1 = transformed.max(axis=0)
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))


def apply_geometry_lock(frame: np.ndarray, center: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    """Move the tracked screen center to the output center without a hard snap."""
    height, width = frame.shape[:2]
    zoom = float(max(0.70, min(1.60, zoom)))
    cx, cy = width * 0.5, height * 0.5
    tracked = np.array([center[0] * width, center[1] * height], dtype=np.float32)
    scaled_tracked = np.array([cx, cy], dtype=np.float32) + (tracked - np.array([cx, cy], dtype=np.float32)) * zoom
    translation = np.array([cx, cy], dtype=np.float32) - scaled_tracked
    matrix = np.float32([[zoom, 0.0, (1.0 - zoom) * cx + translation[0]],
                         [0.0, zoom, (1.0 - zoom) * cy + translation[1]]])
    locked = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101)
    return locked, matrix


def update_geometry_lock_state(
    previous_center: np.ndarray,
    previous_zoom: float,
    outer,
    inner,
    width: int,
    height: int,
    lock_fill: float = 0.64,
) -> tuple[np.ndarray, float, str]:
    """Smooth a detected machine target into a center/zoom lock state."""
    center = previous_center.copy()
    zoom = float(previous_zoom)
    target = inner or outer
    source = "none"
    if target:
        center_target = np.array([
            (target[0] + target[2]) / (2 * width),
            (target[1] + target[3]) / (2 * height),
        ], dtype=np.float32)
        center = previous_center * 0.82 + center_target * 0.18
        target_width = max(target[2] - target[0], 1)
        target_height = max(target[3] - target[1], 1)
        lock_fill = float(min(max(lock_fill, 0.35), 0.90))
        target_fill = lock_fill if inner else min(lock_fill + 0.12, 0.90)
        zoom_target = min(target_fill * width / target_width,
                          target_fill * height / target_height)
        zoom_target = max(0.70, min(1.35, zoom_target))
        zoom = previous_zoom * 0.93 + zoom_target * 0.07
        source = "inner_screen" if inner else "outer_frame"
    return center, zoom, source


def process(args: argparse.Namespace) -> Path:
    session = args.session
    metadata_path = session / "capture.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 {metadata_path}")
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("会话没有视频帧")

    first_image = cv2.imread(str(session / rows[0]["frame_path"]))
    if first_image is None:
        raise RuntimeError("无法读取会话第一帧")
    height, width = first_image.shape[:2]
    fps = float(args.fps or (json.loads((session / "session.json").read_text(encoding="utf-8")).get("nominal_fps", 15)))
    output = args.output or session / "processed.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{output}")
    debug_path = session / "debug.jsonl"
    detector = GeometryDetector(args.model)
    reference = None
    previous_center = np.array([0.5, 0.5], dtype=np.float32)
    previous_zoom = 1.0
    previous_outer = None
    previous_inner = None
    try:
        with debug_path.open("w", encoding="utf-8") as debug_file:
            for index, row in enumerate(rows):
                frame = cv2.imread(str(session / row["frame_path"]))
                if frame is None:
                    continue
                rotation, reference = rotation_for_row(row, reference)
                map_x, map_y = build_remap(
                    width, height, rotation, args.crop, args.fov,
                    args.k1, args.k2, args.center_x, args.center_y,
                )
                stabilized = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                if args.model and (
                    index % args.detect_every == 0
                    or (previous_outer is None and previous_inner is None)
                ):
                    previous_outer, previous_inner = detector.detect(stabilized)
                outer, inner = previous_outer, previous_inner
                center, zoom, lock_source = update_geometry_lock_state(
                    previous_center,
                    previous_zoom,
                    outer,
                    inner,
                    width,
                    height,
                    getattr(args, "lock_fill", 0.64),
                )
                if lock_source != "none":
                    previous_center, previous_zoom = center, zoom
                stabilized, geometry_matrix = apply_geometry_lock(stabilized, center, zoom)
                outer = transform_box(outer, geometry_matrix)
                inner = transform_box(inner, geometry_matrix)
                if args.debug:
                    shown = draw_debug(stabilized, outer, inner, f"{index + 1}/{len(rows)}  crop={args.crop:.2f}  zoom={zoom:.2f}")
                    if args.preview:
                        cv2.imshow("MaiLens PC processor", shown)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    frame_to_write = shown
                else:
                    frame_to_write = stabilized
                writer.write(frame_to_write)
                debug_file.write(json.dumps({
                    "frame_id": row.get("frame_id", index + 1),
                    "timestamp": row.get("timestamp"),
                    "sensor_delta_ms": row.get("sensor_delta_ms"),
                    "center": center.tolist(),
                    "zoom": zoom,
                    "detected_outer": outer,
                    "detected_inner": inner,
                    "lock_source": lock_source,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()
    print(f"处理完成：{output}")
    print(f"调试数据：{debug_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--crop", type=float, default=0.74)
    parser.add_argument("--fov", type=float, default=106.4583)
    parser.add_argument("--center-x", type=float, default=0.501753869)
    parser.add_argument("--center-y", type=float, default=0.499423644)
    parser.add_argument("--k1", type=float, default=0.0893163)
    parser.add_argument("--k2", type=float, default=-0.0174637)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--detect-every", type=int, default=3,
                        help="模型每隔多少帧检测一次，默认 3；中间帧沿用平滑结果")
    parser.add_argument("--lock-fill", type=float, default=0.64,
                        help="内屏锁定后占画面短边的比例，默认 0.64")
    args = parser.parse_args()
    if not 0.2 <= args.crop <= 1.0:
        parser.error("--crop 应在 0.2 到 1.0 之间")
    if args.detect_every < 1:
        parser.error("--detect-every 必须大于 0")
    if not 0.35 <= args.lock_fill <= 0.90:
        parser.error("--lock-fill 应在 0.35 到 0.90 之间")
    process(args)


if __name__ == "__main__":
    main()
