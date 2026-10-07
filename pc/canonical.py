"""Pull the cabinet back to the picture a player sees standing square in front.

The eight decorative frames around the cabinet are one part repeated eight
times, so a head-on picture puts them at the same distance from the screen
centre in all eight directions, and leaves the same gap between the screen and
the ring on every side.  That is what "undistorted" means here, and it is a
measurement rather than a look: build the ratio tile-radius-over-screen-radius
for each of the eight slots, and any variation around the ring is residual
distortion that the screen circle alone cannot see.

The screen is an ellipse in a tilted picture and a circle once the tilt is
gone, so the ellipse fix has to come first (``machine_lock --rectify``).  What
is left after that is a radius that still varies with the direction: the
screen is a circle but the ring around it is not, or the ring is a circle but
the screen is not concentric with it.  Measuring the eight slots gives that
variation directly, and a low-order fit gives a smooth multiplier that walks
each direction back onto the circle.

    r_in = r_out * (1 + ramp(r_out) * (rho(theta) / target - 1))

The ramp is zero at the screen edge and one at the ring, so the screen keeps
its radius exactly, the ring lands on ``target`` in every direction, and the
gap between them comes out equal all the way round.
"""
from __future__ import annotations

import numpy as np

SLOT_COUNT = 8
SLOT_STEP = 360.0 / SLOT_COUNT

#: The operator's numbering: 1 upper-right, clockwise, 8 straight up.
SLOT_ANGLES = np.array([(22.5 + SLOT_STEP * slot) % 360.0
                        for slot in range(SLOT_COUNT)])

#: Tile centre radius over screen radius, on the head-on reference shot.
#:
#: The denominator matters and there are two of them in play.  Measured by
#: hand against the ring of white dots where the screen's bezel sits, the eight
#: frames come out at 269.4 / 197.4 = 1.365.  Measured by this pipeline -- the
#: cyan play field for the screen and the colour centroid of the tile face for
#: the button -- the same picture gives 1.23, and that is the number the
#: correction has to use, because it cancels the offset between the two
#: detectors instead of fighting it.  The tile centroid is a little inside the
#: part's real centre, and the play field is a little inside the bezel.
TILE_RATIO = 1.23

#: Which slots sit on each side of the screen, for the gap report.
SIDE_SLOTS = dict(left=(3, 4), right=(7, 0), top=(1, 2), bottom=(5, 6))

#: How far the correction is allowed to move a pixel, as a fraction.
MAX_FACTOR = 0.18


def user_label(angle: float) -> int:
    """The number the operator wrote next to the slot at ``angle``."""
    return ((1 - int(round((angle - 22.5) / SLOT_STEP))) % SLOT_COUNT) + 1


def slot_ratios(centre, radius: float, points, tolerance: float = 22.5,
                band: float = 0.25):
    """Distance of the tile in each slot from the centre, over the screen radius.

    ``points`` is whatever the button-ring detector found; it often returns one
    or two extra blobs, and on a picture with a lit ring and a busy background
    it can return a dozen.  Two filters keep those out: a blob has to sit at
    roughly the radius the ring is at (``band`` of the median), and then it is
    bucketed into the slot it is nearest to, where a slot with several
    candidates keeps their median.  A slot with no candidate comes back as NaN
    and is left out of the fit.
    """
    centre = np.asarray(centre, dtype=np.float64)
    out = np.full(SLOT_COUNT, np.nan)
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0 or not np.isfinite(radius) or radius <= 0.0:
        return out
    delta = points - centre
    span = np.hypot(delta[:, 0], delta[:, 1]) / radius
    rough = float(np.median(span))
    if rough <= 0.0:
        return out
    on_ring = np.abs(span - rough) <= band * rough
    if not on_ring.any():
        return out
    span, delta = span[on_ring], delta[on_ring]
    angle = np.degrees(np.arctan2(-delta[:, 1], delta[:, 0])) % 360.0
    for slot, target in enumerate(SLOT_ANGLES):
        offset = np.abs((angle - target + 180.0) % 360.0 - 180.0)
        take = offset <= tolerance
        if take.any():
            out[slot] = float(np.median(span[take]))
    return out


#: The button tiles are the only large purple patches in a delivered frame.
PURPLE_LOW = np.array([120, 60, 60], dtype=np.uint8)
PURPLE_HIGH = np.array([170, 255, 255], dtype=np.uint8)


def purple_points(image: np.ndarray) -> np.ndarray:
    """Centroids of the button-sized purple blobs in a delivered frame.

    This lives here rather than in the measuring tool because the live path
    has to find the same eight points on the same pixels; a second copy would
    be free to drift away from the one the acceptance test uses.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_LOW, PURPLE_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    min_area = image.shape[0] * image.shape[1] * 2e-4
    return np.array([centroids[i] for i in range(1, count)
                     if stats[i, cv2.CC_STAT_AREA] >= min_area], dtype=np.float64)


#: The cyan play field.  The same pixels ``pc/machine_lock.py`` calls the
#: screen, and the same ones the trained detector boxes, so a radius taken from
#: here is already in the space ``tools/check_margins.py`` measures in.
SCREEN_LOW = np.array([70, 60, 40], dtype=np.uint8)
SCREEN_HIGH = np.array([115, 255, 255], dtype=np.uint8)

#: Blobs smaller than this fraction of the frame are artwork, not the screen.
MIN_BLOB_FRACTION = 2e-4


def screen_blob(image: np.ndarray, low=SCREEN_LOW, high=SCREEN_HIGH):
    """Centre, mean half-size and box of the cyan play field, or None.

    A blob that reaches the frame border is two regions that merged -- the
    field plus a lit panel, or the field running off the picture -- and its box
    would drag both the centre and the radius with it, so it is refused rather
    than smoothed.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    height, width = image.shape[:2]
    floor = height * width * MIN_BLOB_FRACTION
    best = None
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < floor:
            continue
        if best is None or area > stats[best, cv2.CC_STAT_AREA]:
            best = index
    if best is None:
        return None
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
        return None
    return (np.asarray(centroids[best], dtype=np.float64),
            0.25 * float(w + h),
            (x, y, x + w, y + h))


def _basis(angles: np.ndarray, order: int) -> np.ndarray:
    """Constant, then cos/sin of every harmonic up to ``order``."""
    columns = [np.ones_like(angles)]
    for harmonic in range(1, order + 1):
        columns.append(np.cos(harmonic * angles))
        columns.append(np.sin(harmonic * angles))
    return np.column_stack(columns)


def ring_profile(ratios, order: int = 3):
    """Fit the ring radius as a smooth function of the direction.

    A tilt shows up as one turn around the ring (cos/sin of the angle), a
    squash as two, and lens distortion fills in the rest, so a few harmonics
    describe everything that can be seen with eight samples.  Returns the
    coefficient vector, or None when too few slots were measured to fit one.
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    known = np.isfinite(ratios)
    if known.sum() < 2 * order + 2:
        return None
    design = _basis(np.radians(SLOT_ANGLES[known]), order)
    weight, *_ = np.linalg.lstsq(design, ratios[known], rcond=None)
    return weight


def ring_error(ratios):
    """Spread of the measured ring radii, as a fraction of their mean."""
    known = np.asarray(ratios, dtype=np.float64)
    known = known[np.isfinite(known)]
    if known.size < 3 or known.mean() <= 0.0:
        return None
    return float(known.std() / known.mean())


def _ramp(radius, screen: float, ring: float, shape: str = "smooth"):
    """0 at the screen edge, 1 at the ring, smooth in between."""
    if ring <= screen:
        return np.ones_like(radius)
    unit = np.clip((radius - screen) / (ring - screen), 0.0, 1.0)
    if shape == "linear":
        return unit
    return unit * unit * (3.0 - 2.0 * unit)


def pull_back_maps(width: int, height: int, centre, screen_radius: float,
                   coefficients, strength: float = 1.0, order: int = 3,
                   target: float = TILE_RATIO, coarse: int = 8):
    """Remap arrays that walk the eight slots onto one circle.

    The map is smooth, so it is built on a coarse grid and stretched up; that
    keeps a full-resolution frame well under a millisecond instead of touching
    every pixel in Python.  ``strength`` blends the whole thing back out for
    the frames where the ring was not measured cleanly.
    """
    if coefficients is None or strength <= 0.0:
        return None
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        return None
    centre = np.asarray(centre, dtype=np.float64)
    step = max(1, int(coarse))
    small_w, small_h = max(2, width // step), max(2, height // step)
    scale_x, scale_y = width / small_w, height / small_h
    xs = (np.arange(small_w) + 0.5) * scale_x
    ys = (np.arange(small_h) + 0.5) * scale_y
    dx = xs[None, :] - centre[0]
    dy = ys[:, None] - centre[1]
    radius = np.hypot(dx, dy)
    angle = np.arctan2(-dy, dx)
    fitted = _basis(angle.ravel(), order) @ coefficients
    mean = float(coefficients[0])
    if not np.isfinite(mean) or mean <= 0.0:
        return None
    ratio = np.clip(fitted / target, 1.0 - MAX_FACTOR, 1.0 + MAX_FACTOR)
    ring_radius = target * screen_radius
    blend = _ramp(radius, float(screen_radius), float(ring_radius))
    factor = 1.0 + float(strength) * blend * (ratio.reshape(radius.shape) - 1.0)
    safe = np.maximum(radius, 1e-6)
    map_x = (centre[0] + radius * factor * dx / safe).astype(np.float32)
    map_y = (centre[1] + radius * factor * dy / safe).astype(np.float32)
    if step > 1:
        map_x = cv2_resize(map_x, width, height)
        map_y = cv2_resize(map_y, width, height)
    return map_x, map_y


def cv2_resize(values: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)


def apply(image: np.ndarray, maps) -> np.ndarray:
    """Warp with a map from :func:`pull_back_maps`."""
    import cv2
    if maps is None:
        return image
    map_x, map_y = maps
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT101)


def slot_report(ratios, target: float = TILE_RATIO) -> str:
    """One line per slot, for the run log."""
    parts = []
    for slot, value in enumerate(ratios):
        if not np.isfinite(value):
            parts.append(f"{slot + 1}:--")
        else:
            parts.append(f"{slot + 1}:{100.0 * (value / target - 1.0):+5.1f}%")
    return " ".join(parts)


def side_gaps(ratios, screen_radius: float) -> dict:
    """Distance from the screen edge out to the ring, per side, in pixels."""
    gaps = {}
    for name, slots in SIDE_SLOTS.items():
        values = np.asarray(ratios, dtype=np.float64)[list(slots)]
        known = values[np.isfinite(values)]
        mean = float(known.mean()) if known.size else np.nan
        gaps[name] = mean * screen_radius - screen_radius
    return gaps


def gap_error(gaps: dict) -> tuple:
    """``(|left - right|, |top - bottom|)`` for a :func:`side_gaps` result."""
    return (abs(gaps["left"] - gaps["right"]),
            abs(gaps["top"] - gaps["bottom"]))
