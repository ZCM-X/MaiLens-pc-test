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
import warnings
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


def image_signals(image):
    """Blue-minus-green and red-minus-green for the whole frame, once.

    A polar sweep reads the same three channels over and over, and the centre
    search runs hundreds of sweeps, so converting the frame to int inside each
    one dominated the running time on a full-resolution picture.
    """
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    return blue - green, red - green


def polar_signals(image, centre, angles, radii, signals=None):
    violet_image, redness_image = image_signals(image) if signals is None \
        else signals
    height, width = image.shape[:2]
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing="ij")
    radians = np.radians(angle_grid)
    x = np.round(centre[0] + radius_grid * np.cos(radians)).astype(int)
    y = np.round(centre[1] - radius_grid * np.sin(radians)).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    violet = np.full(x.shape, np.nan)
    redness = np.full(x.shape, np.nan)
    violet[inside] = violet_image[y[inside], x[inside]]
    redness[inside] = redness_image[y[inside], x[inside]]
    return violet, redness


def frame_edges(violet, redness, radii, bounds, low=25, neutral=40):
    """Inner and outer radius of the tile at every angle.

    Returns two arrays; a NaN means that angle fell in a gap between frames.
    Everything is done with whole-array comparisons rather than a Python loop,
    because the sweep needs a fine stride and a plain loop over a million
    samples is far too slow to run interactively.
    """
    outer_lo, outer_hi, inner_lo = bounds
    radial = radii[None, :]
    finite = np.isfinite(violet)
    # Work from the outside in: the tile's outer boundary is the first neutral
    # sample past the middle of the ring, and a boundary only counts as a
    # boundary when the band just inside it is actually violet.
    neutral_ring = finite & (np.abs(violet) < low) & (np.abs(redness) < neutral)
    window = ((radial >= outer_lo) & (radial <= outer_hi))
    candidates = neutral_ring & window
    has_outer = candidates.any(axis=1)
    first = np.argmax(candidates, axis=1)
    outer = np.where(has_outer, radii[first], np.nan)

    inner_span = outer - inner_lo
    band_hi = outer - 0.05 * inner_span
    band_lo = outer - 0.16 * inner_span
    band = np.where(finite & (radial > band_lo[:, None])
                    & (radial < band_hi[:, None]), violet, np.nan)
    with warnings.catch_warnings():
        # A row that found no outer edge is all-NaN and fails the test below
        # anyway; the warning about it is noise on every sweep.
        warnings.simplefilter("ignore", RuntimeWarning)
        violet_band = np.nanmedian(band, axis=1)
    solid = np.isfinite(violet_band) & (violet_band > 40.0)

    inner_limit = np.where(np.isfinite(outer), outer - 0.05 * inner_span, 0.0)
    inward = finite & (np.abs(violet) < low) & (radial < inner_limit[:, None]) \
        & (radial >= inner_lo)
    index = np.where(inward, np.arange(len(radii))[None, :], -1)
    last = index.max(axis=1)
    inner = np.where(last >= 0, radii[np.maximum(last, 0)], np.nan)

    good = solid & np.isfinite(inner)
    return np.where(good, inner, np.nan), np.where(good, outer, np.nan)


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


def violet_mask(image, floor=32, value=70, signals=None):
    violet, _ = image_signals(image) if signals is None else signals
    return (violet > floor) & (image.max(axis=2) > value)


def violet_points(image, most=60000, signals=None):
    """The violet pixels once, as a point cloud, so the search can reuse them."""
    ys, xs = np.nonzero(violet_mask(image, signals=signals))
    if len(xs) < 200:
        return None
    step = max(1, len(xs) // most)
    return np.column_stack([xs[::step], ys[::step]]).astype(float)


def ring_radius(points, centre):
    """Mid radius of the ring, straight off the violet mask.

    Everything in the tool used to be in absolute pixels, which only worked on
    a screenshot at one size.  The tiles themselves set the scale, so the
    median distance from the centre to a violet pixel is the radius to hang
    the rest of the bounds off.
    """
    if points is None or len(points) < 200:
        return None
    radius = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
    return float(np.median(radius))


def measure(image, centre, stride=0.25, radius=None, points=None, signals=None):
    points = violet_points(image, signals=signals) if points is None else points
    radius = radius or ring_radius(points, centre)
    if not radius or radius < 8.0:
        return []
    angles = np.arange(0.0, 360.0, stride)
    step = max(0.25, radius / 600.0)
    radii = np.arange(0.45 * radius, 1.60 * radius, step)
    bounds = (0.92 * radius, 1.36 * radius, 0.45 * radius)
    return measure_with(image, centre, angles, radii, stride, bounds,
                        signals=signals)


def measure_with(image, centre, angles, radii, stride, bounds, signals=None):
    violet, redness = polar_signals(image, centre, angles, radii,
                                    signals=signals)
    inner, outer = frame_edges(violet, redness, radii, bounds)
    rows = [(None if not (np.isfinite(inner[j]) and np.isfinite(outer[j]))
             else (float(inner[j]), float(outer[j])))
            for j in range(len(angles))]
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


def _score(image, centre, points, angles, signals=None):
    radius = ring_radius(points, centre)
    if not radius or radius < 8.0:
        return None, None
    radii = np.arange(0.45 * radius, 1.60 * radius, max(0.5, radius / 200.0))
    bounds = (0.92 * radius, 1.36 * radius, 0.45 * radius)
    rows = measure_with(image, centre, angles, radii, 4.0, bounds,
                        signals=signals)
    if len(rows) != 8:
        return None, None
    outer = np.array([r["outer"] for r in rows])
    inner = np.array([r["inner"] for r in rows])
    return float(outer.std() + inner.std()), radius


def find_centre(image, guess, points=None, signals=None):
    """Seed the centre by search, then settle it on a circle through the frames.

    The radial bounds the measurement uses are absolute, so a guess twenty
    pixels out can lose a frame entirely.  A coarse search over the plausible
    area is cheap enough and removes the need for the operator to click the
    middle of the cabinet first.
    """
    points = violet_points(image, signals=signals) if points is None else points
    if points is None:
        return np.asarray(guess, float)
    signals = image_signals(image) if signals is None else signals
    angles = np.arange(0.0, 360.0, 6.0)
    span = int(min(image.shape[:2]) * 0.09)
    # Each stage refines the last one over three times its own step.  Scanning
    # the whole span again at two pixels would be 68k measurements on a
    # full-resolution photo, which is what made this tool look hung; the coarse
    # grid already covers the span, so the finer stages only have to fill in.
    stages = [(max(4, span // 3), span)]
    middle_step = max(2, span // 12)
    if middle_step < stages[0][0]:
        stages.append((middle_step, middle_step * 3))
    stages.append((2, 6))
    best = (None, None, None)
    for step, reach in stages:
        middle = np.asarray(guess, float) if best[1] is None else best[1]
        for dy in range(-reach, reach + 1, step):
            for dx in range(-reach, reach + 1, step):
                centre = np.array([middle[0] + dx, middle[1] + dy], float)
                score, radius = _score(image, centre, points, angles,
                                       signals=signals)
                if score is not None and (best[0] is None or score < best[0]):
                    best = (score, centre, radius)
    if best[1] is None:
        return np.asarray(guess, float)
    centre = best[1]
    for _ in range(6):
        rows = measure(image, centre, points=points, signals=signals)
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
    signals = image_signals(image)
    points = violet_points(image, signals=signals)
    centre = guess if args.centre else \
        find_centre(image, guess, points, signals=signals)
    rows = measure(image, centre, points=points, signals=signals)
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
