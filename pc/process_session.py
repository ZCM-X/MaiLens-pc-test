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


def map_fisheye_points_to_output(
    points: np.ndarray,
    width: int,
    height: int,
    rotation: np.ndarray,
    crop: float,
    fov_deg: float,
    k1: float,
    k2: float,
    center_x: float,
    center_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw fisheye pixels into the current pose-stabilized output.

    ``build_remap`` maps output pixels back into the raw camera. The detector
    is more reliable on the raw training geometry, so machine boxes are
    mapped in the opposite direction here before the lock is updated.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    source_focal = max(width, height) * 772.4089 / 4032.0
    virtual_focal = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    offset = np.array([center_x * width, center_y * height], dtype=np.float32)
    normalized = (points - offset) / max(source_focal, 1e-6)
    radius = np.linalg.norm(normalized, axis=1)

    # The radial polynomial is monotonic over the forward hemisphere used by
    # the rectilinear output. A lookup table is stable at the lens centre and
    # avoids Newton iterations diverging on the outermost fisheye pixels.
    theta_grid = np.linspace(0.0, math.pi * 0.5 - 1e-4, 2048, dtype=np.float32)
    distorted_grid = theta_grid * (1.0 + k1 * theta_grid ** 2 + k2 * theta_grid ** 4)
    valid = radius <= float(distorted_grid[-1])
    theta = np.interp(
        np.minimum(radius, float(distorted_grid[-1])),
        distorted_grid,
        theta_grid,
    ).astype(np.float32)
    safe_radius = np.maximum(radius, 1e-6)
    sine_scale = np.sin(theta) / safe_radius
    sine_scale[radius < 1e-6] = 1.0
    source_rays = np.stack((
        normalized[:, 0] * sine_scale,
        normalized[:, 1] * sine_scale,
        np.cos(theta),
    ), axis=1)
    output_rays = source_rays @ rotation
    valid &= output_rays[:, 2] > 0.12
    safe_z = np.maximum(output_rays[:, 2], 0.12)
    output = np.stack((
        width * 0.5 + output_rays[:, 0] / safe_z * virtual_focal * crop,
        height * 0.5 + output_rays[:, 1] / safe_z * virtual_focal * crop,
    ), axis=1)
    valid &= (
        (output[:, 0] >= -width * 0.25) & (output[:, 0] <= width * 1.25) &
        (output[:, 1] >= -height * 0.25) & (output[:, 1] <= height * 1.25)
    )
    return output.astype(np.float32), valid


def map_fisheye_box_to_output(
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    rotation: np.ndarray,
    crop: float,
    fov_deg: float,
    k1: float,
    k2: float,
    center_x: float,
    center_y: float,
) -> tuple[int, int, int, int] | None:
    """Convert a raw detector box to output coordinates.

    Fisheye corners can lie outside the forward hemisphere even when the
    machine centre is visible. In that case, estimate the box size from a
    small local Jacobian around its centre instead of using invalid corners.
    """
    if box is None:
        return None
    x0, y0, x1, y1 = (float(value) for value in box)
    center = np.array([[(x0 + x1) * 0.5, (y0 + y1) * 0.5]], dtype=np.float32)
    mapped_center, center_valid = map_fisheye_points_to_output(
        center, width, height, rotation, crop, fov_deg, k1, k2, center_x, center_y,
    )
    if not bool(center_valid[0]) or not np.isfinite(mapped_center[0]).all():
        return None
    output_center = mapped_center[0]
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    mapped_corners, corners_valid = map_fisheye_points_to_output(
        corners, width, height, rotation, crop, fov_deg, k1, k2, center_x, center_y,
    )
    if bool(np.all(corners_valid)) and np.isfinite(mapped_corners).all():
        output_min = mapped_corners.min(axis=0)
        output_max = mapped_corners.max(axis=0)
        return tuple(int(round(value)) for value in (*output_min, *output_max))

    raw_width = max(x1 - x0, 1.0)
    raw_height = max(y1 - y0, 1.0)
    local_width = local_height = None
    for fraction in (0.25, 0.12, 0.06, 0.03):
        half_step = np.array([raw_width * fraction, raw_height * fraction], dtype=np.float32)
        raw_samples = np.array([
            [center[0, 0] - half_step[0], center[0, 1]],
            [center[0, 0] + half_step[0], center[0, 1]],
            [center[0, 0], center[0, 1] - half_step[1]],
            [center[0, 0], center[0, 1] + half_step[1]],
        ], dtype=np.float32)
        mapped_samples, samples_valid = map_fisheye_points_to_output(
            raw_samples, width, height, rotation, crop, fov_deg, k1, k2, center_x, center_y,
        )
        if bool(np.all(samples_valid)) and np.isfinite(mapped_samples).all():
            local_width = float(np.linalg.norm(mapped_samples[1] - mapped_samples[0]) / (2.0 * fraction))
            local_height = float(np.linalg.norm(mapped_samples[3] - mapped_samples[2]) / (2.0 * fraction))
            break
    if local_width is None or local_height is None:
        local_width, local_height = raw_width, raw_height
    local_width = max(8.0, min(width * 2.0, local_width))
    local_height = max(8.0, min(height * 2.0, local_height))
    return (
        int(round(output_center[0] - local_width * 0.5)),
        int(round(output_center[1] - local_height * 0.5)),
        int(round(output_center[0] + local_width * 0.5)),
        int(round(output_center[1] + local_height * 0.5)),
    )


class GeometryDetector:
    def __init__(self, model_path: Path | None):
        self.model = None
        self.cv_net = None
        self.backend = "none"
        self.input_size = 640
        self.confidence = 0.30
        self.names = {0: "outer_buttons", 1: "inner_screen", 2: "button"}
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
            # avoiding the old behaviour of treating a button class as the
            # cabinet when a multi-task model detects only gameplay buttons.
            non_button_boxes = [
                box for label, _, box in boxes
                if not any(token in label for token in ("button", "key", "star", "marker", "slide", "note", "tap"))
            ]
            if non_button_boxes:
                outer = max(non_button_boxes, key=area)
        if inner is not None and outer is not None:
            ix0, iy0, ix1, iy1 = inner
            ox0, oy0, ox1, oy1 = outer
            intersection = max(0, min(ix1, ox1) - max(ix0, ox0)) * max(0, min(iy1, oy1) - max(iy0, oy0))
            if intersection / max(area(inner), 1) < 0.35:
                inner = None
        return outer, inner


def draw_debug(frame: np.ndarray, outer, inner, text: str) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    cv2.drawMarker(
        result,
        (width // 2, height // 2),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(12, min(width, height) // 18),
        thickness=1,
        line_type=cv2.LINE_AA,
    )
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
    snap: bool = False,
) -> tuple[np.ndarray, float, str]:
    """Smooth a detected machine target into a center/zoom lock state."""
    center = previous_center.copy()
    zoom = float(previous_zoom)
    # The cabinet is the lock anchor.  The inner screen is only used to set
    # the crop scale; using its centre as the anchor makes the output drift
    # whenever the bezel is asymmetric or the screen is mounted high/low.
    anchor = outer or inner
    target = inner or outer
    source = "none"
    if anchor and target:
        center_target = np.array([
            (anchor[0] + anchor[2]) / (2 * width),
            (anchor[1] + anchor[3]) / (2 * height),
        ], dtype=np.float32)
        target_width = max(target[2] - target[0], 1)
        target_height = max(target[3] - target[1], 1)
        lock_fill = float(min(max(lock_fill, 0.35), 0.90))
        target_fill = lock_fill if inner else min(lock_fill + 0.12, 0.90)
        zoom_target = min(target_fill * width / target_width,
                          target_fill * height / target_height)
        zoom_target = max(0.70, min(1.35, zoom_target))
        if snap:
            center = center_target
            zoom = zoom_target
        else:
            # The target box is already filtered by GeometryLockTracker. Use
            # its current centre directly so the output does not visibly lag
            # behind a real phone translation; smooth only the scale change.
            center = center_target
            zoom = previous_zoom * 0.85 + zoom_target * 0.15
        source = "inner_screen" if inner else "outer_buttons"
    return center, zoom, source


class GeometryLockTracker:
    """Keep independent outer/inner geometry boxes between detections.

    The outer cabinet is the position anchor while the inner display controls
    the crop size.  Both boxes are carried by the same optical-flow transform
    so a temporary miss of either detector does not move the lock target.
    ``box`` remains as a compatibility alias for the preferred (inner, then
    outer) box used by the live loop to decide when a detector must run.
    """

    def __init__(self, detect_every: int = 4, max_age_frames: int | None = None):
        self.detect_every = max(1, int(detect_every))
        self.max_age_frames = max_age_frames or max(18, self.detect_every * 8)
        self.outer_box: tuple[int, int, int, int] | None = None
        self.inner_box: tuple[int, int, int, int] | None = None
        self.box: tuple[int, int, int, int] | None = None
        self.source = "none"
        self.age_frames = 0
        self.outer_age_frames = 0
        self.inner_age_frames = 0

    def _sync_aliases(self) -> None:
        self.box = self.inner_box or self.outer_box
        if self.inner_box is not None:
            self.source = "inner_screen"
            self.age_frames = self.inner_age_frames
        elif self.outer_box is not None:
            self.source = "outer_buttons"
            self.age_frames = self.outer_age_frames
        else:
            self.source = "none"
            self.age_frames = max(self.outer_age_frames, self.inner_age_frames)

    def reset(self) -> None:
        self.outer_box = None
        self.inner_box = None
        self.box = None
        self.source = "none"
        self.age_frames = 0
        self.outer_age_frames = 0
        self.inner_age_frames = 0

    @staticmethod
    def _center_size(box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)
        size = np.array([max(box[2] - box[0], 1), max(box[3] - box[1], 1)], dtype=np.float32)
        return center, size

    @classmethod
    def _compatible(
        cls,
        previous: tuple[int, int, int, int],
        candidate: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> bool:
        previous_center, previous_size = cls._center_size(previous)
        candidate_center, candidate_size = cls._center_size(candidate)
        normalized_delta = (candidate_center - previous_center) / np.array([width, height], dtype=np.float32)
        if float(np.linalg.norm(normalized_delta)) > 0.30:
            return False
        ratio = candidate_size / np.maximum(previous_size, 1.0)
        return bool(np.all(ratio > 0.42) and np.all(ratio < 2.4))

    @staticmethod
    def _blend_box(
        previous: tuple[int, int, int, int],
        candidate: tuple[int, int, int, int],
        alpha: float,
    ) -> tuple[int, int, int, int]:
        values = np.asarray(previous, dtype=np.float32) * (1.0 - alpha) + np.asarray(candidate, dtype=np.float32) * alpha
        return tuple(int(round(value)) for value in values)

    def update_flow(self, previous_gray: np.ndarray | None, current_gray: np.ndarray | None) -> None:
        """Follow the locked machine with sparse optical flow between detections."""
        if (self.outer_box is None and self.inner_box is None) or previous_gray is None or current_gray is None:
            return
        height, width = current_gray.shape[:2]
        active = [box for box in (self.outer_box, self.inner_box) if box is not None]
        x0 = min(box[0] for box in active)
        y0 = min(box[1] for box in active)
        x1 = max(box[2] for box in active)
        y1 = max(box[3] for box in active)
        pad_x = max(4, int((x1 - x0) * 0.08))
        pad_y = max(4, int((y1 - y0) * 0.08))
        roi = np.zeros_like(previous_gray, dtype=np.uint8)
        roi[max(0, y0 - pad_y):min(height, y1 + pad_y), max(0, x0 - pad_x):min(width, x1 + pad_x)] = 255
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
            mask=roi,
        )
        if points is None or len(points) < 6:
            return
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            return
        valid = status.reshape(-1).astype(bool)
        if errors is not None:
            valid &= errors.reshape(-1) < 30.0
        previous_points = points.reshape(-1, 2)[valid]
        current_points = next_points.reshape(-1, 2)[valid]
        if len(previous_points) < 6:
            return
        median_shift = np.median(current_points - previous_points, axis=0)
        if float(np.linalg.norm(median_shift)) < 1.0:
            # Optical-flow pixel noise on a static target must not slowly
            # walk the lock across the frame.
            return
        matrix, inliers = cv2.estimateAffinePartial2D(
            previous_points, current_points,
            method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if matrix is None or inliers is None or int(inliers.sum()) < 4:
            return
        for attribute in ("outer_box", "inner_box"):
            previous = getattr(self, attribute)
            if previous is None:
                continue
            transformed = transform_box(previous, matrix)
            if transformed is None:
                continue
            tx0, ty0, tx1, ty1 = transformed
            if tx1 <= tx0 or ty1 <= ty0:
                continue
            if self._compatible(previous, transformed, width, height):
                setattr(self, attribute, self._blend_box(previous, transformed, 0.80))
        self._sync_aliases()

    def ingest(
        self,
        outer: tuple[int, int, int, int] | None,
        inner: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> None:
        """Accept each detector result independently after jump checks."""
        for attribute, candidate, age_attribute in (
            ("outer_box", outer, "outer_age_frames"),
            ("inner_box", inner, "inner_age_frames"),
        ):
            previous = getattr(self, attribute)
            if candidate is None:
                age = getattr(self, age_attribute) + self.detect_every
                setattr(self, age_attribute, age)
                if age > self.max_age_frames:
                    setattr(self, attribute, None)
                continue
            if previous is None:
                setattr(self, attribute, tuple(int(value) for value in candidate))
                setattr(self, age_attribute, 0)
                continue
            if self._compatible(previous, candidate, width, height):
                setattr(self, attribute, self._blend_box(previous, candidate, 0.22))
                setattr(self, age_attribute, 0)
            else:
                # A far-away detector result is almost always a false match;
                # keep the last good box and let optical flow bridge the gap.
                setattr(self, age_attribute, getattr(self, age_attribute) + self.detect_every)
                if getattr(self, age_attribute) > self.max_age_frames:
                    setattr(self, attribute, None)
        self._sync_aliases()

    def boxes(self) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        return self.outer_box, self.inner_box


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
    lock_tracker = GeometryLockTracker(args.detect_every)
    previous_gray = None
    previous_lock_source = "none"
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
                current_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                lock_tracker.update_flow(previous_gray, current_gray)
                if args.model and (index % args.detect_every == 0 or lock_tracker.box is None):
                    detected_outer_raw, detected_inner_raw = detector.detect(frame)
                    detected_outer = map_fisheye_box_to_output(
                        detected_outer_raw, width, height, rotation, args.crop, args.fov,
                        args.k1, args.k2, args.center_x, args.center_y,
                    )
                    detected_inner = map_fisheye_box_to_output(
                        detected_inner_raw, width, height, rotation, args.crop, args.fov,
                        args.k1, args.k2, args.center_x, args.center_y,
                    )
                    lock_tracker.ingest(detected_outer, detected_inner, width, height)
                outer, inner = lock_tracker.boxes()
                center, zoom, lock_source = update_geometry_lock_state(
                    previous_center,
                    previous_zoom,
                    outer,
                    inner,
                    width,
                    height,
                    getattr(args, "lock_fill", 0.64),
                    snap=previous_lock_source in ("none", "searching") and lock_tracker.box is not None,
                )
                if lock_source != "none":
                    previous_center, previous_zoom = center, zoom
                    previous_lock_source = lock_source
                stabilized, geometry_matrix = apply_geometry_lock(stabilized, center, zoom)
                outer = transform_box(outer, geometry_matrix)
                inner = transform_box(inner, geometry_matrix)
                previous_gray = current_gray
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
                    "lock_anchor": "outer_buttons" if outer is not None else ("inner_screen" if inner is not None else "none"),
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
    parser.add_argument("--detect-every", type=int, default=4,
                        help="模型每隔多少帧检测一次，默认 4；中间帧使用光流跟踪")
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
