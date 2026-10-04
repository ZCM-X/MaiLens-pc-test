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


# Cached (map_x, map_y) for the fixed fisheye->rectilinear transform. The live
# path builds these once per stream geometry and reuses them for every frame.
_LENS_REMAP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def lens_remap(
    width: int,
    height: int,
    crop: float,
    fov_deg: float,
    k1: float,
    k2: float,
    center_x: float,
    center_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed fisheye-to-rectilinear map, cached per geometry.

    The lens does not move, so this map only depends on the output size and
    the calibration. Caching it lets the live path skip ``build_remap`` on
    every frame and apply the pose as a single perspective warp instead.
    """
    key = (int(width), int(height), float(crop), float(fov_deg), float(k1),
           float(k2), float(center_x), float(center_y))
    cached = _LENS_REMAP_CACHE.get(key)
    if cached is not None:
        return cached
    rays = make_output_rays(width, height, crop, fov_deg)
    result = build_remap(
        width, height, np.eye(3, dtype=np.float32), crop, fov_deg,
        k1, k2, center_x, center_y, output_rays=rays,
    )
    if len(_LENS_REMAP_CACHE) > 8:
        _LENS_REMAP_CACHE.clear()
    _LENS_REMAP_CACHE[key] = result
    return result


def rotation_homography(
    rotation: np.ndarray,
    width: int,
    height: int,
    crop: float,
    fov_deg: float,
) -> np.ndarray:
    """Perspective warp that rotates an already rectified frame by ``rotation``.

    ``build_remap`` rotates the sampling rays inside the fisheye projection.
    For a pinhole output that is exactly equivalent to applying the homography
    ``K @ R @ inv(K)`` to the rectified image, which is far cheaper: one
    constant remap plus one perspective warp instead of a fresh full-frame map
    per frame.
    """
    focal = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5)) * crop
    focal = max(float(focal), 1e-6)
    intrinsic = np.array([
        [focal, 0.0, width * 0.5],
        [0.0, focal, height * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    matrix = intrinsic @ np.asarray(rotation, dtype=np.float64) @ np.linalg.inv(intrinsic)
    return (matrix / matrix[2, 2]).astype(np.float32)


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
        # The small v3 training set produces a real inner-screen prediction
        # below 0.10 on dark/oblique frames.  Geometry pairing and jump checks
        # below provide the safety gate, so a high global confidence threshold
        # would prevent distance compensation from running at all.
        self.confidence = 0.05
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
        # Select a geometrically valid pair rather than taking the most
        # confident box of each class independently.  Two classes can be
        # predicted for the same rectangle, and that pair would otherwise
        # make the distance lock pump or follow the background.
        valid_pairs = []
        for outer_confidence, outer_box in outer_candidates:
            for inner_confidence, inner_box in inner_candidates:
                if plausible_geometry_pair(outer_box, inner_box):
                    score = outer_confidence * inner_confidence
                    valid_pairs.append((score, outer_confidence + inner_confidence,
                                        area(outer_box), outer_box, inner_box))
        if valid_pairs:
            _score, _confidence, _area, outer, inner = max(valid_pairs)
            return outer, inner

        # An explicit outer prediction is still useful as a centre anchor when
        # the inner screen is occluded or temporarily missed.  Keep it only;
        # never reinterpret an inner-screen-only prediction as the cabinet.
        outer = max(outer_candidates, key=lambda item: (item[0], area(item[1])))[1] if outer_candidates else None
        return outer, None


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


def geometry_margins(outer, inner) -> dict[str, int] | None:
    """Return the four pixel gaps between the outer and inner geometry boxes."""
    if outer is None or inner is None:
        return None
    return {
        "left": int(inner[0] - outer[0]),
        "top": int(inner[1] - outer[1]),
        "right": int(outer[2] - inner[2]),
        "bottom": int(outer[3] - inner[3]),
    }


def box_area(box) -> float:
    """Return a non-negative xyxy box area."""
    if box is None:
        return 0.0
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def plausible_geometry_pair(outer, inner) -> bool:
    """Check that the inner screen is really inside the cabinet ring.

    A detector can return two high-confidence boxes for the same rectangle or
    for unrelated background regions.  Such a pair must never drive the lock
    transform: the screen should be mostly contained by a strictly larger
    outer cabinet box with a visible bezel on every side.
    """
    if outer is None or inner is None:
        return False
    ox0, oy0, ox1, oy1 = (float(value) for value in outer)
    ix0, iy0, ix1, iy1 = (float(value) for value in inner)
    ow, oh = ox1 - ox0, oy1 - oy0
    iw, ih = ix1 - ix0, iy1 - iy0
    outer_size = max(ow * oh, 1.0)
    inner_size = max(iw * ih, 1.0)
    if ow <= 1.0 or oh <= 1.0 or iw <= 1.0 or ih <= 1.0:
        return False
    if outer_size <= inner_size * 1.18:
        return False
    overlap_width = max(0.0, min(ox1, ix1) - max(ox0, ix0))
    overlap_height = max(0.0, min(oy1, iy1) - max(oy0, iy0))
    if overlap_width * overlap_height / inner_size < 0.72:
        return False
    # Permit a small mapping error at the edge, but reject an inner box that
    # is actually a background rectangle outside the cabinet.
    margin_x = ow * 0.08
    margin_y = oh * 0.08
    if ix0 < ox0 - margin_x or iy0 < oy0 - margin_y or ix1 > ox1 + margin_x or iy1 > oy1 + margin_y:
        return False
    width_ratio = iw / ow
    height_ratio = ih / oh
    return 0.12 <= width_ratio <= 0.90 and 0.12 <= height_ratio <= 0.90


def soft_geometry_pair(outer, inner) -> bool:
    """Accept a mapped pair whose corners grew slightly during fisheye warp.

    Raw detector boxes are strictly contained. After inverse-fisheye mapping,
    a corner can cross the other box by a few percent even though both boxes
    came from the same machine. This softer gate is only used for that paired
    result; unrelated duplicate boxes still go through the strict gate.
    """
    if outer is None or inner is None:
        return False
    ox0, oy0, ox1, oy1 = (float(value) for value in outer)
    ix0, iy0, ix1, iy1 = (float(value) for value in inner)
    ow, oh = ox1 - ox0, oy1 - oy0
    iw, ih = ix1 - ix0, iy1 - iy0
    if min(ow, oh, iw, ih) <= 4.0:
        return False
    outer_area = ow * oh
    inner_area = iw * ih
    # Inverse-fisheye mapping expands the four inner corners more than the
    # axis-aligned outer rectangle.  A valid raw pair can therefore have an
    # inner mapped box slightly larger than the mapped outer box.  Keep the
    # overlap, centre and per-axis ratio gates below as the false-pair guard;
    # rejecting on area alone made the first valid frame lose inner_screen and
    # delayed plane locking until a later detector refresh.
    if outer_area <= inner_area * 0.72:
        return False
    overlap_width = max(0.0, min(ox1, ix1) - max(ox0, ix0))
    overlap_height = max(0.0, min(oy1, iy1) - max(oy0, iy0))
    if overlap_width * overlap_height / max(inner_area, 1.0) < 0.78:
        return False
    if not (0.45 <= iw / ow <= 1.35 and 0.25 <= ih / oh <= 1.35):
        return False
    outer_center = np.array([(ox0 + ox1) * 0.5, (oy0 + oy1) * 0.5], dtype=np.float32)
    inner_center = np.array([(ix0 + ix1) * 0.5, (iy0 + iy1) * 0.5], dtype=np.float32)
    return float(np.linalg.norm((inner_center - outer_center) / np.array([ow, oh], dtype=np.float32))) < 0.35


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


def _box_points(box: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """Return box corners in clockwise order for homography operations."""
    if box is None:
        return None
    x0, y0, x1, y1 = (float(value) for value in box)
    if x1 - x0 < 4.0 or y1 - y0 < 4.0:
        return None
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray | None:
    """Project 2-D points with a 3x3 homography and reject invalid output."""
    if matrix is None or not np.isfinite(matrix).all():
        return None
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, np.asarray(matrix, dtype=np.float32)).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    return projected


def transform_box_homography(
    box: tuple[int, int, int, int] | None,
    matrix: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Transform an xyxy box through a perspective matrix for debug output."""
    points = _box_points(box)
    if points is None:
        return None
    projected = _project_points(points, matrix)
    if projected is None:
        return None
    x0, y0 = projected.min(axis=0)
    x1, y1 = projected.max(axis=0)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def expand_box(box: tuple[int, int, int, int] | None, scale: float) -> tuple[int, int, int, int] | None:
    """Expand an xyxy box around its centre for the machine-ring overlay."""
    if box is None:
        return None
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    half_width = max((box[2] - box[0]) * float(scale) * 0.5, 1.0)
    half_height = max((box[3] - box[1]) * float(scale) * 0.5, 1.0)
    return (
        int(round(cx - half_width)),
        int(round(cy - half_height)),
        int(round(cx + half_width)),
        int(round(cy + half_height)),
    )


def apply_plane_lock(
    frame: np.ndarray,
    matrix: np.ndarray,
    target_inner_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Composite the locked machine plane over the live background.

    Warping the complete fisheye frame and reflecting its borders duplicates
    walls and tables around the machine. The desired effect is an anchored
    machine foreground, so only a feathered ellipse around the fixed screen
    and button ring is taken from the projective warp; the current background
    remains untouched and can move naturally behind it.
    """
    height, width = frame.shape[:2]
    warped = cv2.warpPerspective(
        frame,
        np.asarray(matrix, dtype=np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid_source = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        np.asarray(matrix, dtype=np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if target_inner_box is None:
        return warped
    # The locked target box is fixed, so the feathered ellipse is identical
    # on every frame of a lock. Building it once keeps the live path at 60 fps
    # instead of rebuilding a blurred full-frame mask per frame.
    alpha = _cached_lock_mask((height, width), target_inner_box)
    alpha = alpha * (valid_source[..., None].astype(np.float32) / 255.0)
    warped_f = warped.astype(np.float32)
    warped_f *= alpha
    frame_f = frame.astype(np.float32)
    frame_f *= (1.0 - alpha)
    warped_f += frame_f
    return np.clip(warped_f, 0.0, 255.0).astype(np.uint8)


_LOCK_MASK_CACHE: dict[tuple, np.ndarray] = {}


def _cached_lock_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Feathered ellipse alpha (h, w, 1) for a fixed locked target box."""
    key = (int(shape[0]), int(shape[1])) + tuple(int(round(v)) for v in box)
    cached = _LOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    height, width = shape
    x0, y0, x1, y1 = box
    center = (int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.5)))
    half_width = max((x1 - x0) * 0.5, 8.0)
    half_height = max((y1 - y0) * 0.5, 8.0)
    # The physical button ring extends beyond the inner display by roughly a
    # third. Cap the mask at the image boundary but keep a broad ellipse so
    # all eight buttons are included in the locked foreground.
    axes = (
        max(8, int(round(min(width * 0.60, half_width * 1.72)))),
        max(8, int(round(min(height * 0.60, half_height * 1.72)))),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255, -1)
    feather = max(9, int(round(min(axes) * 0.06)) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    if len(_LOCK_MASK_CACHE) > 16:
        _LOCK_MASK_CACHE.clear()
    _LOCK_MASK_CACHE[key] = alpha
    return alpha


def _warp_with_valid_source(
    frame: np.ndarray,
    matrix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a frame and return a 0..1 mask for pixels with a source sample."""
    height, width = frame.shape[:2]
    if matrix is None or not np.isfinite(matrix).all():
        return frame.copy(), np.zeros((height, width), dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    warped = cv2.warpPerspective(
        frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    valid = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8), matrix,
        (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped, valid.astype(np.float32) / 255.0


def _soft_ellipse_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int] | None,
    scale: float = 1.0,
) -> np.ndarray:
    """Return a feathered ellipse mask for a fixed output geometry box."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if box is None:
        return mask.astype(np.float32)
    x0, y0, x1, y1 = (float(value) for value in box)
    center = (int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.5)))
    axes = (
        max(8, int(round(abs(x1 - x0) * 0.5 * float(scale)))),
        max(8, int(round(abs(y1 - y0) * 0.5 * float(scale)))),
    )
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255, -1)
    feather = max(3, int(round(min(axes) * 0.06)) * 2 + 1)
    if feather > 3:
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask.astype(np.float32) / 255.0


def apply_multi_plane_lock(
    frame: np.ndarray,
    screen_matrix: np.ndarray | None,
    ring_matrix: np.ndarray | None,
    target_inner_box: tuple[int, int, int, int] | None,
    target_outer_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Composite independent screen and button-ring plane warps.

    A single homography cannot remove parallax between a display and raised
    buttons when the phone moves toward/away from the cabinet.  This function
    keeps the display as the priority layer and applies a second LK/RANSAC
    homography to the annulus around it.  If the ring tracker has not acquired
    enough features, the established single-plane path is used unchanged.
    """
    if screen_matrix is None:
        return frame.copy()
    if ring_matrix is None or target_inner_box is None:
        return apply_plane_lock(frame, screen_matrix, target_inner_box)

    height, width = frame.shape[:2]
    screen_warp, screen_valid = _warp_with_valid_source(frame, screen_matrix)
    ring_warp, ring_valid = _warp_with_valid_source(frame, ring_matrix)
    if target_outer_box is None:
        target_outer_box = expand_box(target_inner_box, 1.72)

    inner_alpha = _soft_ellipse_mask((height, width), target_inner_box, 1.08)
    outer_alpha = _soft_ellipse_mask((height, width), target_outer_box, 1.02)
    # Subtract the screen ellipse so the ring warp cannot overwrite the
    # playable display.  A small overlap is left for feathered compositing.
    ring_alpha = np.clip(outer_alpha - inner_alpha * 0.88, 0.0, 1.0) * ring_valid
    screen_alpha = inner_alpha * screen_valid
    result = frame.astype(np.float32)
    ring_alpha = ring_alpha[..., None]
    screen_alpha = screen_alpha[..., None]
    result = ring_warp.astype(np.float32) * ring_alpha + result * (1.0 - ring_alpha)
    result = screen_warp.astype(np.float32) * screen_alpha + result * (1.0 - screen_alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def _target_plane(box: tuple[int, int, int, int], width: int, height: int, lock_fill: float) -> np.ndarray | None:
    """Create the fixed, centered output rectangle for the first valid lock."""
    points = _box_points(box)
    if points is None:
        return None
    source_width = max(float(box[2] - box[0]), 1.0)
    source_height = max(float(box[3] - box[1]), 1.0)
    fill = float(min(max(lock_fill, 0.35), 0.90))
    scale = min(fill * width / source_width, fill * height / source_height)
    target_width = source_width * scale
    target_height = source_height * scale
    cx, cy = width * 0.5, height * 0.5
    return np.float32([
        [cx - target_width * 0.5, cy - target_height * 0.5],
        [cx + target_width * 0.5, cy - target_height * 0.5],
        [cx + target_width * 0.5, cy + target_height * 0.5],
        [cx - target_width * 0.5, cy + target_height * 0.5],
    ])


def detect_screen_circle(
    gray: np.ndarray,
    box: tuple[int, int, int, int] | None,
) -> tuple[float, float, float] | None:
    """Find the circular playable screen inside a detector reference box."""
    if box is None:
        return None
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = (int(value) for value in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None
    roi = gray[y0:y1, x0:x1]
    blurred = cv2.medianBlur(roi, 5)
    min_radius = max(24, int(min(x1 - x0, y1 - y0) * 0.16))
    max_radius = max(min_radius + 8, int(min(x1 - x0, y1 - y0) * 0.55))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(min(x1 - x0, y1 - y0) * 0.22)),
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    center = np.array([(x1 - x0) * 0.5, (y1 - y0) * 0.5], dtype=np.float32)
    candidates = []
    for raw_x, raw_y, raw_radius in np.asarray(circles[0], dtype=np.float32):
        candidate = np.array([raw_x, raw_y], dtype=np.float32)
        distance = float(np.linalg.norm(candidate - center))
        if distance > max(x1 - x0, y1 - y0) * 0.30:
            continue
        radius = float(raw_radius)
        if not (min_radius <= radius <= max_radius):
            continue
        # Prefer the playable circle (near the detector centre and below the
        # oversized cabinet/body circles Hough often returns).
        score = distance + abs(radius - min(x1 - x0, y1 - y0) * 0.30) * 0.45
        candidates.append((score, raw_x + x0, raw_y + y0, radius))
    if not candidates:
        return None
    _score, cx, cy, radius = min(candidates, key=lambda item: item[0])
    return float(cx), float(cy), float(radius)


def apply_geometry_lock(frame: np.ndarray, center: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    """Move the tracked screen center to the output center without a hard snap."""
    height, width = frame.shape[:2]
    # A close/far movement can require substantially more than the old
    # 0.70..1.60 range.  The source is the fisheye frame, so this is still a
    # crop operation; the bounds only prevent an invalid transform when the
    # target has left the usable lens area.
    zoom = float(max(0.45, min(2.50, zoom)))
    cx, cy = width * 0.5, height * 0.5
    tracked = np.array([center[0] * width, center[1] * height], dtype=np.float32)
    scaled_tracked = np.array([cx, cy], dtype=np.float32) + (tracked - np.array([cx, cy], dtype=np.float32)) * zoom
    translation = np.array([cx, cy], dtype=np.float32) - scaled_tracked
    matrix = np.float32([[zoom, 0.0, (1.0 - zoom) * cx + translation[0]],
                         [0.0, zoom, (1.0 - zoom) * cy + translation[1]]])
    locked = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101)
    return locked, matrix


def geometry_reference(inner, width: int, height: int, lock_fill: float = 0.71) -> tuple[float, float] | tuple[None, None]:
    """Return the first-lock screen size and crop zoom used for distance lock."""
    if inner is None:
        return None, None
    inner_width = max(int(inner[2] - inner[0]), 1)
    inner_height = max(int(inner[3] - inner[1]), 1)
    target_size = math.sqrt(float(inner_width * inner_height))
    lock_fill = float(min(max(lock_fill, 0.35), 0.90))
    reference_zoom = min(lock_fill * width / inner_width,
                         lock_fill * height / inner_height)
    return target_size, max(0.45, min(2.50, float(reference_zoom)))


def update_geometry_lock_state(
    previous_center: np.ndarray,
    previous_zoom: float,
    outer,
    inner,
    width: int,
    height: int,
    lock_fill: float = 0.71,
    snap: bool = False,
    reference_target_size: float | None = None,
    reference_zoom: float | None = None,
) -> tuple[np.ndarray, float, str]:
    """Smooth a detected machine target into a center/zoom lock state.

    ``outer`` controls position.  ``inner`` controls distance compensation:
    its geometric-mean size is compared with the size at the first valid
    lock, and the crop zoom is adjusted in the opposite direction.  When the
    screen is temporarily unavailable, the previous zoom is held rather than
    estimating distance from the less stable outer ring.
    """
    center = previous_center.copy()
    zoom = float(previous_zoom)
    # The visible game plane is the lock anchor.  The outer ring is a
    # validation/ROI signal; its detector box often includes the top display
    # and cabinet body, so anchoring the transform to it makes the actual
    # screen wander when that box expands or contracts.
    anchor = inner or outer
    source = "none"
    if anchor:
        center_target = np.array([
            (anchor[0] + anchor[2]) / (2 * width),
            (anchor[1] + anchor[3]) / (2 * height),
        ], dtype=np.float32)
        if snap:
            center = center_target
        else:
            # The target box is already filtered by GeometryLockTracker. Use
            # its current centre directly so the output does not visibly lag
            # behind a real phone translation.
            center = center_target
        if inner is not None:
            target_width = max(inner[2] - inner[0], 1)
            target_height = max(inner[3] - inner[1], 1)
            target_size = math.sqrt(float(target_width * target_height))
            lock_fill = float(min(max(lock_fill, 0.35), 0.90))
            if reference_target_size is None or reference_zoom is None:
                # Establish the size seen at the first valid lock.  This
                # makes later front/back movement use a stable reference
                # instead of recomputing a new target from detector noise.
                reference_target_size, reference_zoom = geometry_reference(
                    inner, width, height, lock_fill,
                )
            zoom_target = float(reference_zoom) * float(reference_target_size) / max(target_size, 1.0)
            zoom_target = max(0.45, min(2.50, zoom_target))
            if snap:
                zoom = zoom_target
            else:
                # A modest low-pass filter removes detector-size pumping while
                # still following a deliberate change in phone distance.
                if abs(zoom_target - previous_zoom) / max(abs(previous_zoom), 1e-3) > 0.015:
                    zoom = previous_zoom * 0.88 + zoom_target * 0.12
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

    def __init__(self, detect_every: int = 12, max_age_frames: int | None = None):
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
        if matrix is None or inliers is None:
            return
        inlier_mask = inliers.reshape(-1).astype(bool)
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 6 or inlier_count / max(len(previous_points), 1) < 0.45:
            # A few background corners can still produce an affine fit.  Do
            # not let that fit move a locked cabinet.
            return
        predicted = cv2.transform(previous_points[None, :, :], matrix)[0]
        residual = np.linalg.norm(predicted - current_points, axis=1)[inlier_mask]
        if len(residual) == 0 or float(np.median(residual)) > 5.0 or float(np.percentile(residual, 90)) > 9.0:
            return
        a, b = float(matrix[0, 0]), float(matrix[0, 1])
        c, d = float(matrix[1, 0]), float(matrix[1, 1])
        scale_x = math.sqrt(a * a + c * c)
        scale_y = math.sqrt(b * b + d * d)
        rotation_degrees = abs(math.degrees(math.atan2(c, a)))
        if not (0.65 <= scale_x <= 1.55 and 0.65 <= scale_y <= 1.55):
            return
        if abs(scale_x - scale_y) > 0.12 or rotation_degrees > 25.0:
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
        if self.outer_box is not None and self.inner_box is not None and not plausible_geometry_pair(self.outer_box, self.inner_box):
            # Optical flow is allowed to move both boxes, but never allowed to
            # turn them into two overlapping copies of the same rectangle.
            self.inner_box = None
        self._sync_aliases()

    def ingest(
        self,
        outer: tuple[int, int, int, int] | None,
        inner: tuple[int, int, int, int] | None,
        width: int,
        height: int,
        allow_soft_pair: bool = False,
    ) -> None:
        """Accept each detector result independently after jump checks."""
        # Validate the pair before updating state.  An inner-screen-only
        # prediction is not a cabinet anchor; an overlapping pair loses its
        # scale signal but may still leave an explicit outer anchor usable.
        mapped_pair = allow_soft_pair and soft_geometry_pair(outer, inner)
        if outer is None:
            inner = None
        elif inner is not None and not plausible_geometry_pair(outer, inner) and not mapped_pair:
            inner = None

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
        # If a new outer candidate is incompatible with the retained inner
        # box, keep the position anchor and discard the stale scale box.  This
        # prevents a false inner detection from changing distance compensation.
        retained_pair = allow_soft_pair and soft_geometry_pair(self.outer_box, self.inner_box)
        if self.outer_box is not None and self.inner_box is not None and not plausible_geometry_pair(self.outer_box, self.inner_box) and not retained_pair:
            self.inner_box = None
            self.inner_age_frames = self.max_age_frames + self.detect_every
        self._sync_aliases()

    def boxes(self) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        return self.outer_box, self.inner_box


class PlaneLockTracker:
    """Track the screen and button ring as one machine plane.

    The detector supplies an inner-screen box and an outer-buttons box. Good
    features across that combined machine ROI are tracked with LK optical
    flow and a RANSAC homography is estimated for every frame. The same warp
    is applied to the complete image, so the screen and all eight buttons
    move together into one fixed, front-facing target.
    """

    def __init__(
        self,
        detect_every: int = 12,
        max_age_frames: int | None = None,
        feature_region: str = "combined",
        motion_model: str = "homography",
    ):
        self.detect_every = max(1, int(detect_every))
        # Hold a good transform briefly through a missed frame burst, then
        # reacquire instead of freezing an old plane for several seconds.
        self.max_age_frames = max_age_frames or max(18, self.detect_every * 4)
        # ``combined`` is the original behaviour and tracks screen plus
        # bezel.  ``ring`` restricts points to the annulus outside the
        # screen; it is used by DualPlaneLockTracker for the raised buttons.
        self.feature_region = str(feature_region)
        self.motion_model = "similarity" if str(motion_model).lower() == "similarity" else "homography"
        self.reset()

    def reset(self) -> None:
        self.reference_gray: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None
        self.reference_box: tuple[int, int, int, int] | None = None
        self.reference_outer_box: tuple[int, int, int, int] | None = None
        self.current_box: tuple[int, int, int, int] | None = None
        self.current_outer_box: tuple[int, int, int, int] | None = None
        self.reference_to_output: np.ndarray | None = None
        self.current_to_reference = np.eye(3, dtype=np.float32)
        self.points: np.ndarray | None = None
        self.age_frames = 0
        self.inliers = 0
        self.inlier_ratio = 0.0
        self.reprojection_error = 0.0
        self.last_success = False
        self.frame_shape: tuple[int, int] | None = None
        self.frames_since_refresh = 0
        # Last accepted machine quad, used to reject wild homographies.
        self.current_quad: np.ndarray | None = None

    @property
    def locked(self) -> bool:
        return self.reference_gray is not None and self.reference_to_output is not None

    @property
    def output_homography(self) -> np.ndarray | None:
        # A stale projective warp freezes the last good frame while the phone
        # keeps moving. Hold it for only a short grace period, then let the
        # caller use its detector/affine fallback while LK reacquires.
        if not self.locked or self.age_frames > max(3, self.detect_every // 2):
            return None
        matrix = self.reference_to_output @ self.current_to_reference
        return matrix.astype(np.float32) if np.isfinite(matrix).all() else None

    def reacquire_if_stale(
        self,
        gray: np.ndarray,
        box: tuple[int, int, int, int] | None,
        outer_box: tuple[int, int, int, int] | None,
        width: int,
        height: int,
        lock_fill: float,
        fresh_detection: bool,
    ) -> bool:
        """Re-center from a fresh model box after optical flow has gone stale.

        ``output_homography`` stops using a transform after a short grace
        period, but the tracker used to keep its old reference much longer.
        A later flow recovery could therefore resume an accumulated, drifting
        transform.  A fresh, accepted detector result is an absolute anchor:
        initialize from it again so the visible screen returns to the fixed
        output target.
        """
        stale_after = max(3, self.detect_every // 2)
        if (not fresh_detection or not self.locked or self.age_frames <= stale_after
                or box is None):
            return False
        return self.initialize(gray, box, outer_box, width, height, lock_fill)

    def _feature_mask(
        self,
        gray: np.ndarray,
        box: tuple[int, int, int, int] | None,
        outer_box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        feature_box = outer_box or box
        if feature_box is None:
            mask[:, :] = 255
            return mask
        height, width = gray.shape[:2]
        if box is not None and outer_box is not None and self.feature_region == "ring":
            # Track the physical button/bezel ring without allowing display
            # pixels to dominate the homography.  The two ellipses are only a
            # feature-selection mask; the actual warp remains projective.
            x0, y0, x1, y1 = box
            ox0, oy0, ox1, oy1 = outer_box
            outer_cx = int(round((ox0 + ox1) * 0.5))
            outer_cy = int(round((oy0 + oy1) * 0.5))
            inner_cx = int(round((x0 + x1) * 0.5))
            inner_cy = int(round((y0 + y1) * 0.5))
            outer_axes = (
                max(8, int(round((ox1 - ox0) * 0.48))),
                max(8, int(round((oy1 - oy0) * 0.48))),
            )
            inner_axes = (
                max(8, int(round((x1 - x0) * 0.58))),
                max(8, int(round((y1 - y0) * 0.58))),
            )
            cv2.ellipse(mask, (outer_cx, outer_cy), outer_axes, 0.0, 0.0, 360.0, 255, -1)
            # Leave a small overlap at the screen edge so the two warped
            # regions feather without a visible seam.
            inner_cut = np.zeros_like(mask)
            cv2.ellipse(inner_cut, (inner_cx, inner_cy), inner_axes, 0.0, 0.0, 360.0, 255, -1)
            mask = cv2.subtract(mask, inner_cut)
        elif box is not None and outer_box is not None:
            # The detector's outer rectangle also covers cabinet body and
            # background after inverse-fisheye mapping. Restrict features to
            # an ellipse around the screen, which contains the button ring
            # while excluding the wall/table that caused duplicate warps.
            x0, y0, x1, y1 = box
            ox0, oy0, ox1, oy1 = outer_box
            cx = int(round((x0 + x1) * 0.5))
            cy = int(round((y0 + y1) * 0.5))
            inner_half_x = max((x1 - x0) * 0.5, 4.0)
            inner_half_y = max((y1 - y0) * 0.5, 4.0)
            outer_half_x = max((ox1 - ox0) * 0.5, 4.0)
            outer_half_y = max((oy1 - oy0) * 0.5, 4.0)
            axis_x = int(round(max(inner_half_x * 1.15, min(outer_half_x * 1.15, inner_half_x * 1.55))))
            axis_y = int(round(max(inner_half_y * 1.15, min(outer_half_y * 1.15, inner_half_y * 1.55))))
            cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), 0.0, 0.0, 360.0, 255, -1)
        else:
            x0, y0, x1, y1 = feature_box
            inset_x = max(2, int((x1 - x0) * 0.04))
            inset_y = max(2, int((y1 - y0) * 0.04))
            mask[max(0, y0 + inset_y):min(height, y1 - inset_y),
                 max(0, x0 + inset_x):min(width, x1 - inset_x)] = 255
        return mask

    def _find_points(
        self,
        gray: np.ndarray,
        box: tuple[int, int, int, int] | None,
        outer_box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=120,
            qualityLevel=0.008,
            minDistance=5,
            blockSize=7,
            mask=self._feature_mask(gray, box, outer_box),
            useHarrisDetector=False,
        )
        if points is None or len(points) < 8:
            return None
        return points.astype(np.float32)

    def initialize(
        self,
        gray: np.ndarray,
        box: tuple[int, int, int, int] | None,
        outer_box: tuple[int, int, int, int] | None,
        width: int,
        height: int,
        lock_fill: float,
    ) -> bool:
        points = _box_points(box)
        target = _target_plane(box, width, height, lock_fill) if box is not None else None
        if points is None or target is None:
            return False
        reference_to_output = cv2.getPerspectiveTransform(points, target)
        if reference_to_output is None or not np.isfinite(reference_to_output).all():
            return False
        tracked = self._find_points(gray, box, outer_box)
        self.reference_gray = gray.copy()
        self.previous_gray = gray.copy()
        self.reference_box = tuple(int(value) for value in box)
        self.reference_outer_box = tuple(int(value) for value in outer_box) if outer_box is not None else self.reference_box
        self.current_box = tuple(int(value) for value in box)
        self.current_outer_box = tuple(int(value) for value in outer_box) if outer_box is not None else self.current_box
        self.reference_to_output = reference_to_output.astype(np.float32)
        self.current_to_reference = np.eye(3, dtype=np.float32)
        self.current_quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
        self.points = tracked
        self.age_frames = 0
        self.inliers = 0
        self.inlier_ratio = 1.0 if tracked is not None else 0.0
        self.reprojection_error = 0.0
        self.last_success = True
        self.frame_shape = gray.shape[:2]
        self.frames_since_refresh = 0
        return True

    @staticmethod
    def _valid_homography(matrix: np.ndarray, previous_points: np.ndarray, current_points: np.ndarray) -> tuple[bool, int, float, float]:
        if matrix is None or not np.isfinite(matrix).all():
            return False, 0, 0.0, float("inf")
        projected = _project_points(previous_points, matrix)
        if projected is None or len(projected) != len(current_points):
            return False, 0, 0.0, float("inf")
        residual = np.linalg.norm(projected - current_points, axis=1)
        inlier_mask = residual <= 4.0
        inlier_count = int(inlier_mask.sum())
        ratio = inlier_count / max(len(residual), 1)
        median_error = float(np.median(residual[inlier_mask])) if inlier_count else float("inf")
        return inlier_count >= 8 and ratio >= 0.52 and median_error <= 3.0, inlier_count, ratio, median_error

    def _candidate_ok(self, projected: np.ndarray | None) -> bool:
        """Reject transforms that would tilt or mangle the machine plane.

        A hand or arm crossing the screen breaks optical flow, and RANSAC can
        still return a confident-looking homography that squashes the whole
        machine into a leaning plank.  Checking the quad shape and its motion
        against the previous frame keeps that out of the preview.
        """
        if projected is None or len(projected) != 4 or not np.isfinite(projected).all():
            return False
        quad = np.asarray(projected, dtype=np.float32).reshape(4, 2)
        reference = _box_points(self.reference_box)
        if reference is None:
            return False
        reference = np.asarray(reference, dtype=np.float32).reshape(4, 2)
        area = abs(float(cv2.contourArea(quad)))
        reference_area = max(box_area(self.reference_box), 1.0)
        if area <= 0 or not 0.30 <= area / reference_area <= 3.0:
            return False
        cross: list[float] = []
        for index in range(4):
            first = quad[(index + 1) % 4] - quad[index]
            second = quad[(index + 2) % 4] - quad[(index + 1) % 4]
            cross.append(float(first[0] * second[1] - first[1] * second[0]))
        if not (all(value > 0 for value in cross) or all(value < 0 for value in cross)):
            return False
        edges = [float(np.linalg.norm(quad[index] - quad[(index + 1) % 4])) for index in range(4)]
        if min(edges) <= 4.0:
            return False
        # Opposite edges must stay comparable, otherwise the plane reads as a
        # board tilted away from the camera.
        if not 0.45 <= edges[0] / max(edges[2], 1e-6) <= 2.2:
            return False
        if not 0.45 <= edges[1] / max(edges[3], 1e-6) <= 2.2:
            return False
        if self.current_quad is not None:
            diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
            jump = float(np.max(np.linalg.norm(quad - self.current_quad, axis=1)))
            if diagonal > 0 and jump > 0.45 * diagonal:
                return False
        return True

    def update(
        self,
        gray: np.ndarray,
        box: tuple[int, int, int, int] | None,
        outer_box: tuple[int, int, int, int] | None,
        width: int,
        height: int,
        lock_fill: float,
    ) -> bool:
        if self.frame_shape != gray.shape[:2]:
            self.reset()
        if not self.locked:
            return self.initialize(gray, box, outer_box, width, height, lock_fill)
        self.current_box = tuple(int(value) for value in box) if box is not None else self.current_box
        self.current_outer_box = tuple(int(value) for value in outer_box) if outer_box is not None else self.current_outer_box
        success = False
        if self.previous_gray is not None and self.points is not None and len(self.points) >= 8:
            next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                self.points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_points is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                if errors is not None:
                    valid &= errors.reshape(-1) < 30.0
                previous_points = self.points.reshape(-1, 2)[valid]
                current_points = next_points.reshape(-1, 2)[valid]
                if len(previous_points) >= 8:
                    if self.motion_model == "similarity":
                        affine, _inliers = cv2.estimateAffinePartial2D(
                            previous_points,
                            current_points,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                            maxIters=2000,
                            confidence=0.99,
                            refineIters=10,
                        )
                        matrix = (
                            np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
                            if affine is not None
                            else None
                        )
                    else:
                        matrix, _mask = cv2.findHomography(
                            previous_points,
                            current_points,
                            cv2.RANSAC,
                            3.0,
                        )
                    valid_h, count, ratio, error = self._valid_homography(
                        matrix, previous_points, current_points,
                    )
                    if valid_h:
                        try:
                            inverse = np.linalg.inv(matrix).astype(np.float32)
                        except np.linalg.LinAlgError:
                            inverse = None
                        if inverse is not None and np.isfinite(inverse).all():
                            candidate = self.current_to_reference @ inverse
                            ref_corners = _box_points(self.reference_box)
                            projected = _project_points(ref_corners, candidate) if ref_corners is not None else None
                            if self._candidate_ok(projected):
                                self.current_to_reference = candidate
                                self.current_quad = np.asarray(projected, dtype=np.float32).reshape(4, 2)
                                self.points = current_points.reshape(-1, 1, 2).astype(np.float32)
                                self.inliers = count
                                self.inlier_ratio = ratio
                                self.reprojection_error = error
                                self.frames_since_refresh += 1
                                success = True
        if not success:
            self.age_frames += 1
            self.last_success = False
            self.points = None
            if self.age_frames > self.max_age_frames:
                self.reset()
                return False
        else:
            self.age_frames = 0
            self.last_success = True
        self.previous_gray = gray.copy()
        if self.points is None or len(self.points) < 12 or self.frames_since_refresh >= self.detect_every:
            refreshed = self._find_points(gray, self.current_box, self.current_outer_box)
            if refreshed is not None:
                self.points = refreshed
                self.frames_since_refresh = 0
        return self.locked


SESSION_VIDEO_NAMES = ("video.mp4", "phone.mp4", "video.mov", "video.m4v", "raw.mp4")


def load_manifest(session: Path) -> dict:
    """Session manifest, or an empty dict when the session never wrote one."""
    path = session / "session.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_fps(session: Path,
                rows: list[dict],
                override: float | None,
                container_fps: float | None = None) -> float:
    """Nominal frame rate without requiring a manifest.

    A session stopped with Ctrl+C never got its ``session.json``, so fall back
    to the file when it exists and to the frame timestamps when it does not.
    """
    if override:
        return float(override)
    value = load_manifest(session).get("nominal_fps")
    if value:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    timestamps = [float(row["timestamp"]) for row in rows
                  if isinstance(row.get("timestamp"), (int, float))]
    deltas = sorted(later - earlier
                    for earlier, later in zip(timestamps, timestamps[1:])
                    if later > earlier)
    if deltas:
        median = deltas[len(deltas) // 2]
        if median > 0:
            return max(1.0, min(240.0, 1.0 / median))
    if container_fps:
        return max(1.0, min(240.0, float(container_fps)))
    return 60.0


def find_session_video(session: Path, manifest: dict) -> Path:
    """Locate the recorded movie for sessions that kept no JPEG frames."""
    candidates: list[Path] = []
    recorded = manifest.get("video")
    if recorded:
        candidates.append(session / str(recorded))
    candidates.extend(session / name for name in SESSION_VIDEO_NAMES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for suffix in ("*.mp4", "*.mov", "*.m4v"):
        matches = sorted(session.glob(suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"{session} 里既没有 frames/ 也没有录像，无法处理"
    )


def load_pose_track(session: Path) -> list[tuple[float, dict]]:
    """Read the full-rate pose log the phone recorded next to the movie."""
    path = session / "pose.jsonl"
    if not path.exists():
        return []
    track: list[tuple[float, dict]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
        except ValueError:
            continue
        timestamp = sample.get("timestamp")
        if isinstance(timestamp, (int, float)):
            track.append((float(timestamp), sample))
    track.sort(key=lambda item: item[0])
    return track


def pose_for_row(row: dict, track: list[tuple[float, dict]]) -> dict:
    """Pose for one frame: the frame's own sample, else the saved log.

    Phone recordings ship ``pose.jsonl`` at 120 Hz while the frame log only
    carries the sample nearest to each written frame.  Re-matching from the log
    keeps rotation compensation working for frames the log never covered.
    """
    if row.get("pose") or not track:
        return row
    stamp = row.get("timestamp")
    if not isinstance(stamp, (int, float)):
        return row
    sample = min(track, key=lambda item: abs(item[0] - float(stamp)))[1]
    merged = dict(row)
    merged["pose"] = sample
    return merged


class SessionFrames:
    """Frame access for a session, from extracted JPEGs or from the movie.

    ``import_phone_session.py`` still exists for annotation work, but the
    offline stabiliser can read a phone session straight from ``video.mp4`` and
    skip the extra JPEG copy entirely.
    """

    def __init__(self, session: Path, rows: list[dict], manifest: dict):
        self.session = session
        self.rows = rows
        self.uses_frames = any(row.get("frame_path") for row in rows)
        self.video: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.container_fps: float | None = None
        if not self.uses_frames:
            self.video = find_session_video(session, manifest)
            capture = cv2.VideoCapture(str(self.video))
            if not capture.isOpened():
                raise RuntimeError(f"无法打开录像：{self.video}")
            container = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self.container_fps = container if 0.5 < container < 1000 else None
            self.capture = capture

    def first_frame(self):
        if self.uses_frames:
            for row in self.rows:
                path = row.get("frame_path")
                if not path:
                    continue
                image = cv2.imread(str(self.session / path))
                if image is not None:
                    return image
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame

    def __iter__(self):
        if self.uses_frames:
            for index, row in enumerate(self.rows):
                path = row.get("frame_path")
                image = cv2.imread(str(self.session / path)) if path else None
                yield index, image, row
            return
        step = 1.0 / (self.container_fps or 60.0)
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            yield index, frame, self.row_for(index, step)
            index += 1

    def row_for(self, index: int, step: float) -> dict:
        if index < len(self.rows):
            return self.rows[index]
        # The movie kept recording after the log stopped, which happens when a
        # take is killed mid-write.  Extrapolate the clock so pose matching
        # still lands on the right samples.
        stamps = [float(row["timestamp"]) for row in self.rows
                  if isinstance(row.get("timestamp"), (int, float))]
        timestamp = stamps[-1] + (index - len(self.rows) + 1) * step if stamps else None
        return {"frame_id": index + 1, "timestamp": timestamp}

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def process(args: argparse.Namespace) -> Path:
    session = args.session
    metadata_path = session / "capture.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 {metadata_path}")
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("会话没有视频帧")

    manifest = load_manifest(session)
    source = SessionFrames(session, rows, manifest)
    first_image = source.first_frame()
    if first_image is None:
        source.close()
        raise RuntimeError("无法读取会话第一帧：frames/ 和录像都打不开")
    height, width = first_image.shape[:2]
    fps = resolve_fps(session, rows, args.fps, container_fps=source.container_fps)
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
    reference_target_size: float | None = None
    reference_zoom: float | None = None
    lock_tracker = GeometryLockTracker(args.detect_every)
    plane_tracker = PlaneLockTracker(
        args.detect_every,
        motion_model=getattr(args, "plane_model", "homography"),
    )
    previous_gray = None
    previous_lock_source = "none"
    pose_track = load_pose_track(session)
    lens_fix = not getattr(args, "no_fisheye", False)
    max_frames = getattr(args, "max_frames", None)
    plane_smooth = float(getattr(args, "plane_smooth", 0.35))
    plane_full = bool(getattr(args, "plane_full", False))
    plane_smooth_corners: np.ndarray | None = None
    plane_hold_limit = max(0, int(getattr(args, "plane_hold", 12)))
    plane_hold_matrix: np.ndarray | None = None
    plane_hold_frames = 0
    try:
        with debug_path.open("w", encoding="utf-8") as debug_file:
            for index, frame, row in source:
                if frame is None:
                    continue
                if max_frames and index >= max_frames:
                    break
                row = pose_for_row(row, pose_track)
                if lens_fix:
                    rotation, reference = rotation_for_row(row, reference)
                    map_x, map_y = build_remap(
                        width, height, rotation, args.crop, args.fov,
                        args.k1, args.k2, args.center_x, args.center_y,
                    )
                    stabilized = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT101)
                else:
                    # A clip the phone recorded outside this rig has no
                    # calibration, so locking the machine plane straight on the
                    # raw frames beats warping it through the wrong lens model.
                    stabilized = frame
                current_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                if previous_gray is not None and previous_gray.shape != current_gray.shape:
                    previous_gray = None
                    lock_tracker.reset()
                    plane_tracker.reset()
                    previous_center = np.array([0.5, 0.5], dtype=np.float32)
                    previous_zoom = 1.0
                    reference_target_size = None
                    reference_zoom = None
                    previous_lock_source = "searching"
                if not plane_tracker.locked:
                    lock_tracker.update_flow(previous_gray, current_gray)
                fresh_inner_detection = False
                plane_reacquired = False
                if args.model and index % args.detect_every == 0:
                    detected_outer_raw, detected_inner_raw = detector.detect(frame)
                    if lens_fix:
                        detected_outer = map_fisheye_box_to_output(
                            detected_outer_raw, width, height, rotation, args.crop, args.fov,
                            args.k1, args.k2, args.center_x, args.center_y,
                        )
                        detected_inner = map_fisheye_box_to_output(
                            detected_inner_raw, width, height, rotation, args.crop, args.fov,
                            args.k1, args.k2, args.center_x, args.center_y,
                        )
                    else:
                        detected_outer, detected_inner = detected_outer_raw, detected_inner_raw
                    lock_tracker.ingest(
                        detected_outer,
                        detected_inner,
                        width,
                        height,
                        allow_soft_pair=True,
                    )
                    fresh_inner_detection = (
                        detected_inner is not None
                        and lock_tracker.outer_age_frames == 0
                        and lock_tracker.inner_age_frames == 0
                    )
                outer, inner = lock_tracker.boxes()
                plane_locked = plane_tracker.update(
                    current_gray,
                    inner,
                    outer,
                    width,
                    height,
                    getattr(args, "lock_fill", 0.71),
                )
                plane_reacquired = plane_tracker.reacquire_if_stale(
                    current_gray,
                    inner,
                    outer,
                    width,
                    height,
                    getattr(args, "lock_fill", 0.71),
                    fresh_detection=fresh_inner_detection,
                )
                if plane_reacquired:
                    plane_locked = True
                    plane_smooth_corners = None
                    plane_hold_matrix = None
                    plane_hold_frames = 0
                if lock_tracker.box is None:
                    reference_target_size = None
                    reference_zoom = None
                elif inner is not None and (reference_target_size is None or reference_zoom is None):
                    reference_target_size, reference_zoom = geometry_reference(
                        inner, width, height, args.lock_fill,
                    )
                center, zoom, lock_source = update_geometry_lock_state(
                    previous_center,
                    previous_zoom,
                    outer,
                    inner,
                    width,
                    height,
                    getattr(args, "lock_fill", 0.71),
                    snap=previous_lock_source in ("none", "searching") and lock_tracker.box is not None,
                    reference_target_size=reference_target_size,
                    reference_zoom=reference_zoom,
                )
                if lock_source != "none":
                    previous_center, previous_zoom = center, zoom
                    previous_lock_source = lock_source
                plane_matrix = plane_tracker.output_homography if plane_locked else None
                fixed_inner = (
                    transform_box_homography(
                        plane_tracker.reference_box,
                        plane_tracker.reference_to_output,
                    )
                    if plane_tracker.locked else None
                )
                matrix = None
                if plane_matrix is not None and fixed_inner is not None:
                    reference_corners = _box_points(plane_tracker.reference_box)
                    raw_corners = _project_points(reference_corners, plane_matrix)
                    if plane_smooth_corners is None or raw_corners is None:
                        plane_smooth_corners = raw_corners
                    else:
                        # Smooth in corner space instead of averaging a
                        # scale-ambiguous matrix.  This removes the per-frame
                        # projective jitter that showed up as a squashed,
                        # twitching machine patch.
                        plane_smooth_corners = (
                            plane_smooth * raw_corners
                            + (1.0 - plane_smooth) * plane_smooth_corners
                        )
                    if reference_corners is not None and plane_smooth_corners is not None:
                        matrix = cv2.getPerspectiveTransform(
                            reference_corners,
                            plane_smooth_corners.astype(np.float32),
                        )
                    else:
                        matrix = plane_matrix
                    plane_hold_matrix = matrix
                    plane_hold_frames = 0
                    lock_source = "plane_homography"
                elif (plane_hold_matrix is not None and fixed_inner is not None
                      and plane_hold_frames < plane_hold_limit):
                    # A hand crossing the machine breaks optical flow for a few
                    # frames.  Freeze the last good warp instead of snapping to
                    # the detector box, which used to lurch the whole picture.
                    matrix = plane_hold_matrix
                    plane_hold_frames += 1
                    lock_source = "plane_hold"
                else:
                    plane_smooth_corners = None
                    plane_hold_matrix = None
                    plane_hold_frames = 0

                if matrix is not None and fixed_inner is not None:
                    if plane_full:
                        # Warp the complete frame with the same machine-plane
                        # transform.  No feathered seam, and the background moves
                        # with the machine instead of staying live behind it.
                        stabilized = cv2.warpPerspective(
                            stabilized,
                            np.asarray(matrix, dtype=np.float32),
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
                    else:
                        stabilized = apply_plane_lock(stabilized, matrix, fixed_inner)
                    outer = expand_box(fixed_inner, 1.45)
                    inner = fixed_inner
                    center = np.array([0.5, 0.5], dtype=np.float32)
                    zoom = 1.0
                    previous_lock_source = lock_source
                else:
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
                    "geometry_margins": geometry_margins(outer, inner),
                    "lock_source": lock_source,
                    "lock_anchor": "outer_buttons" if outer is not None else ("inner_screen" if inner is not None else "none"),
                    "plane_lock": plane_matrix is not None,
                    "plane_reacquired": plane_reacquired,
                    "plane_age_frames": plane_tracker.age_frames,
                    "plane_inliers": plane_tracker.inliers,
                    "plane_inlier_ratio": plane_tracker.inlier_ratio,
                    "plane_reprojection_error": plane_tracker.reprojection_error,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        writer.release()
        source.close()
        if args.preview:
            cv2.destroyAllWindows()
    print(f"处理完成：{output}")
    print(f"调试数据：{debug_path}")
    return output


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--detect-every", type=int, default=12,
                        help="模型每隔多少帧检测一次，默认 12；中间帧使用光流跟踪")
    parser.add_argument("--lock-fill", type=float, default=0.71,
                        help="内屏锁定后占画面短边的比例，默认 0.71（按近景校准参考）")
    parser.add_argument("--no-fisheye", action="store_true",
                        help="不做鱼眼→直线矫正，直接在原始帧上锁机台（没有标定的素材用这个）")
    parser.add_argument("--max-frames", type=int,
                        help="只处理前 N 帧，用来快速试参数")
    parser.add_argument("--plane-smooth", type=float, default=0.35,
                        help="机台单应矩阵的时间平滑系数，0 不滤波，1 完全冻结，默认 0.35")
    parser.add_argument("--plane-model", choices=("homography", "similarity"),
                        default="homography",
                        help="机台运动模型：homography 允许透视，similarity 只做旋转/等比缩放/平移，后者更不容易显扁")
    parser.add_argument("--plane-full", action="store_true",
                        help="整帧都按机台平面 warp，而不是只在中间合成机台区域")
    parser.add_argument("--plane-hold", type=int, default=12,
                        help="机台被手或手臂挡住时，保持上一帧变换的帧数，默认 12")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.2 <= args.crop <= 1.0:
        parser.error("--crop 应在 0.2 到 1.0 之间")
    if args.detect_every < 1:
        parser.error("--detect-every 必须大于 0")
    if not 0.35 <= args.lock_fill <= 0.90:
        parser.error("--lock-fill 应在 0.35 到 0.90 之间")
    if not 0.0 <= args.plane_smooth <= 1.0:
        parser.error("--plane-smooth 应在 0 到 1 之间")
    if args.plane_hold < 0:
        parser.error("--plane-hold 不能为负数")
    return args


def main() -> None:
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
