#!/usr/bin/env python3
"""Rectify one clip-on-fisheye photo and optionally flatten its screen plane.

The four ``--screen-points`` are cardinal points in the square, inverse-fisheye
image, ordered ``left top right bottom``.  A runtime implementation will get
these points from the detector/ellipse fitter; this command keeps the first
photo experiment reproducible while that detector is being trained.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_oriented(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def inverse_fisheye(
    image: np.ndarray,
    size: int,
    fov: float,
    center_x: float,
    center_y: float,
    k1: float,
    k2: float,
) -> np.ndarray:
    """Render a square rectilinear view from the raw fisheye image."""
    height, width = image.shape[:2]
    virtual_focal = size / (2.0 * math.tan(math.radians(fov) * 0.5))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rays = np.stack(((xx - size * 0.5) / virtual_focal,
                     (yy - size * 0.5) / virtual_focal,
                     np.ones_like(xx)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    source_focal = max(width, height) * 772.4089 / 4032.0
    radial = cv2.magnitude(rays[..., 0], rays[..., 1])
    theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
    theta_distorted = theta * (1.0 + k1 * theta ** 2 + k2 * theta ** 4)
    safe_radial = np.maximum(radial, 1e-6)
    map_x = (center_x * width + source_focal * rays[..., 0] / safe_radial * theta_distorted).astype(np.float32)
    map_y = (center_y * height + source_focal * rays[..., 1] / safe_radial * theta_distorted).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)


def parse_points(value: str) -> np.ndarray:
    try:
        values = [float(item) for item in value.replace(";", " ").split() for item in item.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("点格式应为 x,y x,y x,y x,y") from error
    if len(values) != 8:
        raise argparse.ArgumentTypeError("需要左、上、右、下四个点")
    return np.asarray(values, dtype=np.float32).reshape(4, 2)


def rectify(args: argparse.Namespace) -> Path:
    source = read_oriented(args.input)
    undistorted = inverse_fisheye(
        source, args.work_size, args.fov, args.center_x, args.center_y, args.k1, args.k2,
    )
    output = undistorted
    if args.screen_points is not None:
        center = np.array([args.width * 0.5, args.height * 0.5], dtype=np.float32)
        radius = float(args.radius)
        destination = np.asarray([
            center + [-radius, 0.0], center + [0.0, -radius],
            center + [radius, 0.0], center + [0.0, radius],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(args.screen_points, destination)
        output = cv2.warpPerspective(
            undistorted, matrix, (args.width, args.height),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"无法写入 {args.output}")
    print(f"输入方向校正后：{source.shape[1]}×{source.shape[0]}")
    print(f"输出：{args.output} ({output.shape[1]}×{output.shape[0]})")
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-size", type=int, default=1024,
                        help="中间逆鱼眼画布边长，默认 1024")
    parser.add_argument("--width", type=int, default=625)
    parser.add_argument("--height", type=int, default=559)
    parser.add_argument("--radius", type=float, default=200.0,
                        help="拉正后内屏圆半径，默认 200 像素")
    parser.add_argument("--screen-points", type=parse_points,
                        help="逆鱼眼图中的左上右下四点，例如 '232,577 512,344 796,577 512,811'")
    parser.add_argument("--fov", type=float, default=106.4583)
    parser.add_argument("--center-x", type=float, default=0.501753869)
    parser.add_argument("--center-y", type=float, default=0.499423644)
    parser.add_argument("--k1", type=float, default=0.0893163)
    parser.add_argument("--k2", type=float, default=-0.0174637)
    args = parser.parse_args()
    if args.work_size < 64 or args.width < 64 or args.height < 64:
        parser.error("画布尺寸过小")
    if args.radius <= 0:
        parser.error("--radius 必须大于 0")
    rectify(args)


if __name__ == "__main__":
    main()
