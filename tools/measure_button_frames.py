"""Measure the eight decorative frames around the cabinet.

Each frame is a purple tile on a grey ring.  Blue-minus-green is large on the
tile and near zero both on the ring and on the white highlight that runs along
the tile's outer edge, so working from the outside in over a polar profile
finds both boundaries of every tile without a brightness threshold on a
picture that has a strong lighting gradient.  Both the annotator's red digits
and the game's green effects sit on top of the tiles and break that test for
the wrong reason, so a boundary sample only counts when it is neutral.

The point of measuring them is that all eight are the same part.  Any spread
that is left after removing the picture's own tilt is a real size difference;
anything that follows cos/sin of the angle once round the ring is the camera,
not the cabinet.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Read through pathlib: cv2.imread misses valid Chinese paths on Windows."""
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as error:
        raise SystemExit(f"读不了这张图：{path}") from error
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"读不了这张图：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise SystemExit(f"写不了这张图：{path}")
    path.write_bytes(encoded.tobytes())


def polar_signals(image, centre, angles, radii):
    blue = image[:, :, 0].astype(int)
    green = image[:, :, 1].astype(int)
    red = image[:, :, 2].astype(int)
    height, width = image.shape[:2]
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing="ij")
    radians = np.radians(angle_grid)
    x = np.round(centre[0] + radius_grid * np.cos(radians)).astype(int)
    y = np.round(centre[1] - radius_grid * np.sin(radians)).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    violet = np.full(x.shape, np.nan)
    redness = np.full(x.shape, np.nan)
    violet[inside] = blue[y[inside], x[inside]] - green[y[inside], x[inside]]
    redness[inside] = red[y[inside], x[inside]] - green[y[inside], x[inside]]
    return violet, redness


def frame_edges(violet, redness, radii, low=25, neutral=40):
    """Inner and outer radius of the tile at one angle, or None on a gap."""
    outer = np.nan
    for i, radius in enumerate(radii):
        if radius < 278.0:
            continue
        if radius > 345.0 or not np.isfinite(violet[i]):
            break
        if abs(violet[i]) < low and abs(redness[i]) < neutral:
            outer = radius
            break
    if not np.isfinite(outer):
        return None
    band = [violet[i] for i, radius in enumerate(radii)
            if np.isfinite(violet[i]) and outer - 40.0 < radius < outer - 12.0]
    if not band or float(np.median(band)) < 40.0:
        return None
    for i in range(len(radii) - 1, -1, -1):
        radius = radii[i]
        if radius > outer - 12.0:
            continue
        if radius < 210.0:
            break
        if np.isfinite(violet[i]) and abs(violet[i]) < low:
            return radius, outer
    return None


def split_runs(angles, rows, shortest=8.0):
    """Group the angles where a tile was found into one run per tile."""
    runs, current = [], []
    for j, got in enumerate(rows):
        if got is None:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(j)
    if current:
        runs.append(current)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(angles) - 1:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    return [run for run in runs if len(run) * (angles[1] - angles[0]) > shortest]


def user_label(angle):
    """The operator's numbering: 1 upper-right, clockwise, 8 at the top.

    The frames sit at 22.5 + 45k, so identify the slot first and then read the
    number off that clockwise sequence.
    """
    return ((1 - int(round((angle - 22.5) / 45.0))) % 8) + 1


def measure(image, centre, stride=0.25):
    angles = np.arange(0.0, 360.0, stride)
    radii = np.arange(200.0, 380.0, 0.5)
    return measure_with(image, centre, angles, radii, stride)


def measure_with(image, centre, angles, radii, stride):
    violet, redness = polar_signals(image, centre, angles, radii)
    rows = [frame_edges(violet[j], redness[j], radii) for j in range(len(angles))]
    runs = split_runs(angles, rows)
    out = []
    for run in runs:
        a = np.array([angles[j] for j in run])
        r0 = np.array([rows[j][0] for j in run])
        r1 = np.array([rows[j][1] for j in run])
        if np.ptp(a) > 180.0:
            a = (a + 180.0) % 360.0
        span = float(np.ptp(a)) + stride
        inner, outer = float(np.median(r0)), float(np.median(r1))
        mid = 0.5 * (inner + outer)
        out.append(dict(
            label=user_label(float(a.mean())), angle=float(a.mean()),
            span=span, width=float(np.radians(span) * mid),
            inner=inner, outer=outer, height=outer - inner,
            area=float(np.sum((r1 - r0) * (r0 + r1) / 2.0) * np.radians(stride)),
            samples=len(run)))
    out.sort(key=lambda t: t["angle"])
    return out


def find_centre(image, guess):
    """Seed the centre by search, then settle it on a circle through the frames.

    The radial bounds the measurement uses are absolute, so a guess twenty
    pixels out can lose a frame entirely.  A coarse search over the plausible
    area is cheap enough and removes the need for the operator to click the
    middle of the cabinet first.
    """
    angles = np.arange(0.0, 360.0, 4.0)
    radii = np.arange(200.0, 380.0, 1.0)
    span = int(min(image.shape[:2]) * 0.09)
    best = (None, None)
    for dy in range(-span, span + 1, 6):
        for dx in range(-span, span + 1, 6):
            centre = np.array([guess[0] + dx, guess[1] + dy], float)
            rows = measure_with(image, centre, angles, radii, 4.0)
            if len(rows) != 8:
                continue
            outer = np.array([r["outer"] for r in rows])
            inner = np.array([r["inner"] for r in rows])
            score = float(outer.std() + inner.std())
            if best[0] is None or score < best[0]:
                best = (score, centre)
    if best[1] is None:
        return np.asarray(guess, float)
    centre = best[1]
    for _ in range(6):
        rows = measure(image, centre)
        if len(rows) != 8:
            break
        points = []
        for row in rows:
            mid = np.radians(row["angle"])
            radius = 0.5 * (row["inner"] + row["outer"])
            points.append([centre[0] + radius * np.cos(mid),
                           centre[1] - radius * np.sin(mid)])
        moved = circle_fit(points)
        if np.hypot(*(moved - centre)) < 0.02:
            centre = moved
            break
        centre = moved
    return centre


def fit_tilt(angles, values):
    """Split a set of per-frame sizes into a smooth tilt and what is left."""
    design = np.column_stack([np.ones(len(angles)), np.cos(np.radians(angles)),
                              np.sin(np.radians(angles))])
    weight, *_ = np.linalg.lstsq(design, values, rcond=None)
    left = values - design @ weight
    amplitude = float(np.hypot(weight[1], weight[2]))
    return amplitude, float(np.degrees(np.arctan2(weight[2], weight[1]))), \
        float(left.std())


def circle_fit(points):
    """Algebraic circle fit through the frame centres."""
    p = np.asarray(points, float)
    design = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    weight, *_ = np.linalg.lstsq(design, (p ** 2).sum(axis=1), rcond=None)
    cx, cy = weight[0] / 2.0, weight[1] / 2.0
    return np.array([cx, cy])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="截图或照片")
    parser.add_argument("--output", type=Path, help="把测量结果画成图片")
    parser.add_argument("--centre", type=float, nargs=2, metavar=("X", "Y"),
                        help="机台中心；不给就先按图片中心猜")
    parser.add_argument("--screen-radius", type=float,
                        help="内屏半径（屏幕那圈白点）；给了就一起输出比例")
    args = parser.parse_args()

    image = read_image(args.input)
    guess = np.array(args.centre, float) if args.centre else \
        np.array([image.shape[1] * 0.5, image.shape[0] * 0.5])
    # The ring of frames is the best centre estimate: it is the thing being
    # measured, and it is symmetric once the eight of them agree.
    centre = guess if args.centre else find_centre(image, guess)
    rows = measure(image, centre)
    print(f"centre ({centre[0]:.1f}, {centre[1]:.1f})   {len(rows)} frames")
    if len(rows) != 8:
        print("注意：没有正好找到 8 个装饰框，检查 --centre")
    print(f"{'#':>2} {'angle':>7} {'span':>7} {'width':>7} {'r_in':>7}"
          f" {'r_out':>7} {'height':>7} {'area':>7}")
    for row in rows:
        print(f"{row['label']:>2} {row['angle']:7.1f} {row['span']:6.1f}d"
              f" {row['width']:7.1f} {row['inner']:7.1f} {row['outer']:7.1f}"
              f" {row['height']:7.1f} {row['area']:7.0f}")

    angles = np.array([r["angle"] for r in rows])
    for key in ("width", "height", "area", "inner", "outer"):
        value = np.array([r[key] for r in rows], float)
        amplitude, towards, left = fit_tilt(angles, value)
        print(f"  {key:<6} mean {value.mean():8.2f}  spread"
              f" {100 * (value.max() - value.min()) / value.mean():5.2f}%"
              f"  tilt {100 * 2 * amplitude / value.mean():5.2f}% towards"
              f" {towards:6.1f}deg  leftover {100 * left / value.mean():5.2f}%")
    if args.screen_radius:
        radius = float(args.screen_radius)
        width = float(np.mean([r["width"] for r in rows]))
        height = float(np.mean([r["height"] for r in rows]))
        inner = float(np.mean([r["inner"] for r in rows]))
        outer = float(np.mean([r["outer"] for r in rows]))
        print(f"  相对内屏半径 {radius:.1f}px： 宽 {width / radius:.3f}"
              f"  高 {height / radius:.3f}  内缘 {inner / radius:.3f}"
              f"  外缘 {outer / radius:.3f}  内屏到外缘 {(outer - radius) / radius:.3f}")

    if args.output:
        overlay = image.copy()
        for row in rows:
            span = np.radians(row["span"])
            for radius in (row["inner"], row["outer"]):
                points = []
                for t in np.linspace(-span / 2.0, span / 2.0, 40):
                    angle = np.radians(row["angle"]) + t
                    points.append([centre[0] + radius * np.cos(angle),
                                   centre[1] - radius * np.sin(angle)])
                cv2.polylines(overlay, [np.round(points).astype(np.int32)],
                              False, (0, 255, 0), 1)
            mid = np.radians(row["angle"])
            label_radius = 0.5 * (row["inner"] + row["outer"])
            x = int(centre[0] + label_radius * np.cos(mid))
            y = int(centre[1] - label_radius * np.sin(mid))
            cv2.putText(overlay, str(row["label"]), (x - 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                        cv2.LINE_AA)
        write_image(args.output, overlay)
        print("wrote", args.output)


if __name__ == "__main__":
    main()
