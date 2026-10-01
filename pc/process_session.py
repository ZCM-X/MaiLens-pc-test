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
        if model_path:
            try:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(str(model_path))
            except ImportError as error:
                raise RuntimeError("使用 --model 前请先安装 ultralytics") from error

    def detect(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        if self.model is None:
            return None, None
        result = self.model.predict(frame, imgsz=640, conf=0.22, verbose=False)[0]
        boxes = []
        names = result.names
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls = int(box.cls[0].item())
            label_value = names[cls] if isinstance(names, (list, tuple)) else names.get(cls, cls)
            label = str(label_value).lower()
            boxes.append((label, tuple(xyxy)))
        outer = next((box for label, box in boxes if "outer" in label or "frame" in label), None)
        inner = next((box for label, box in boxes if "inner" in label or "screen" in label), None)
        if outer is None and boxes:
            outer = max((box for _, box in boxes), key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
        if inner is None and outer is not None:
            candidates = [box for _, box in boxes if box != outer and box[2] > box[0] and box[3] > box[1]]
            inner = min(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if candidates else None
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
                if args.model and (index % args.detect_every == 0 or previous_inner is None):
                    previous_outer, previous_inner = detector.detect(stabilized)
                outer, inner = previous_outer, previous_inner
                center = previous_center.copy()
                zoom = previous_zoom
                if inner:
                    center_target = np.array([(inner[0] + inner[2]) / (2 * width), (inner[1] + inner[3]) / (2 * height)], dtype=np.float32)
                    center = previous_center * 0.82 + center_target * 0.18
                    inner_width = max(inner[2] - inner[0], 1)
                    zoom_target = max(0.75, min(1.25, 0.30 * width / inner_width))
                    zoom = previous_zoom * 0.93 + zoom_target * 0.07
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
    args = parser.parse_args()
    if not 0.2 <= args.crop <= 1.0:
        parser.error("--crop 应在 0.2 到 1.0 之间")
    if args.detect_every < 1:
        parser.error("--detect-every 必须大于 0")
    process(args)


if __name__ == "__main__":
    main()
