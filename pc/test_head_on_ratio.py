"""Reading the head-on tile ratio off a picture, in pipeline units."""
import unittest

import cv2
import numpy as np

from tools.head_on_ratio import measure


def synthetic(ratio: float = 1.3, radius: float = 160.0, tilt: float = 0.0):
    """A cyan play field with eight violet blobs around it."""
    size = 720
    image = np.full((size, size, 3), (60, 60, 60), dtype=np.uint8)
    centre = np.array([size * 0.5, size * 0.5])
    cv2.circle(image, tuple(np.round(centre).astype(int)), int(radius),
               (200, 180, 60), -1)
    for angle in 22.5 + 45.0 * np.arange(8):
        reach = radius * ratio * (1.0 + tilt * np.cos(np.radians(angle - 90.0)))
        radians = np.radians(angle)
        x = centre[0] + reach * np.cos(radians)
        y = centre[1] - reach * np.sin(radians)
        cv2.circle(image, (int(round(x)), int(round(y))), 14, (180, 90, 110), -1)
    return image, centre


class HeadOnRatioTests(unittest.TestCase):
    def test_the_ratio_comes_back_in_screen_radii(self):
        image, _ = synthetic(ratio=1.3)
        found = measure(image)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found["mean"], 1.3, delta=0.03)
        self.assertEqual(found["missing"], 0)
        self.assertLess(found["spread"], 0.02)

    def test_a_tilted_picture_still_reports_the_same_mean(self):
        image, _ = synthetic(ratio=1.25, tilt=0.08)
        found = measure(image)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found["mean"], 1.25, delta=0.03)
        self.assertGreater(found["spread"], 0.01)

    def test_a_picture_with_no_ring_is_refused(self):
        image = np.full((400, 400, 3), (60, 60, 60), dtype=np.uint8)
        self.assertIsNone(measure(image))
