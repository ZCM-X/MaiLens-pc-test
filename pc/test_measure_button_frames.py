import unittest
from unittest import mock

import cv2
import numpy as np

from tools import measure_button_frames as frames
from tools.measure_button_frames import measure, user_label


def synthetic(skip=None):
    """A grey ring with eight identical violet frames on it."""
    # Big enough that the whole ring, plus a margin, stays inside the picture.
    image = np.full((760, 760, 3), (110, 110, 110), dtype=np.uint8)
    centre = np.array([380.0, 390.0])
    for slot in range(8):
        if slot == skip:
            continue
        angle = 22.5 + 45.0 * slot
        arc = np.linspace(np.radians(-10.0), np.radians(10.0), 24)
        outer = [[centre[0] + 310.0 * np.cos(np.radians(angle) + t),
                  centre[1] - 310.0 * np.sin(np.radians(angle) + t)] for t in arc]
        inner = [[centre[0] + 228.0 * np.cos(np.radians(angle) + t),
                  centre[1] - 228.0 * np.sin(np.radians(angle) + t)] for t in arc[::-1]]
        points = outer + inner
        cv2.fillPoly(image, [np.round(points).astype(np.int32)], (170, 100, 110))
    return image, centre


class ButtonFrameTests(unittest.TestCase):
    def test_eight_identical_frames_come_out_the_same_size(self):
        image, centre = synthetic()
        rows = measure(image, centre)
        self.assertEqual(len(rows), 8)
        # The labels must read clockwise from the upper right, with 8 on top.
        self.assertEqual([row["label"] for row in rows], [2, 1, 8, 7, 6, 5, 4, 3])
        for key in ("width", "height", "inner", "outer", "area"):
            value = np.array([row[key] for row in rows], float)
            self.assertLess(value.std() / value.mean(), 0.01, key)
        # The tile as drawn: 20 degrees of arc between 228 and 310 pixels.
        self.assertAlmostEqual(rows[0]["span"], 20.0, delta=0.6)
        self.assertAlmostEqual(rows[0]["inner"], 228.0, delta=2.0)
        self.assertAlmostEqual(rows[0]["outer"], 310.0, delta=2.0)

    def test_label_reads_clockwise_from_the_upper_right(self):
        self.assertEqual([user_label(a) for a in (67.5, 22.5, 337.5, 292.5,
                                                  247.5, 202.5, 157.5, 112.5)],
                         [1, 2, 3, 4, 5, 6, 7, 8])

    def test_a_gap_between_frames_is_not_measured_as_one(self):
        # With one frame missing the ring must report seven, not a fat eighth.
        image, centre = synthetic(skip=3)
        rows = measure(image, centre)
        self.assertLess(len(rows), 8)

    def test_the_centre_search_gives_up_cheaply(self):
        """Violet paint with no ring must not turn into a minute of scanning."""
        image = np.full((600, 600, 3), (110, 110, 110), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (120, 220), (170, 100, 110), -1)
        calls = []
        real = frames._score

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        with mock.patch.object(frames, "_score", counted):
            found = frames.find_centre(image, np.array([300.0, 300.0]))
        self.assertTrue(np.allclose(found, [300.0, 300.0]))
        self.assertLess(len(calls), 400)


if __name__ == "__main__":
    unittest.main()
