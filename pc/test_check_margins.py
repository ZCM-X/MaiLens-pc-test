"""The margin spec's own knobs: slot mapping, blob finder and shape guard."""
import unittest

import cv2
import numpy as np

from pc import canonical
from tools import check_margins


class SideValueTests(unittest.TestCase):
    def test_the_four_sides_use_the_documented_slots(self):
        ratios = np.arange(1.0, 9.0)
        sides = check_margins.side_values(ratios)
        # SIDE_SLOTS: left=(3,4) right=(7,0) top=(1,2) bottom=(5,6), slots 0-based.
        self.assertAlmostEqual(sides["left"], (4.0 + 5.0) / 2)
        self.assertAlmostEqual(sides["right"], (8.0 + 1.0) / 2)
        self.assertAlmostEqual(sides["top"], (2.0 + 3.0) / 2)
        self.assertAlmostEqual(sides["bottom"], (6.0 + 7.0) / 2)

    def test_a_side_with_no_measured_slot_stays_nan(self):
        ratios = np.full(canonical.SLOT_COUNT, np.nan)
        ratios[0] = 1.3
        ratios[7] = 1.5
        sides = check_margins.side_values(ratios)
        self.assertAlmostEqual(sides["right"], 1.4)
        self.assertTrue(np.isnan(sides["top"]))

    def test_the_table_constants_are_the_spec_numbers(self):
        self.assertAlmostEqual(check_margins.MM_PER_SIDE, 75.0)
        self.assertAlmostEqual(canonical.TILE_RATIO, 1.23)


class RingAspectTests(unittest.TestCase):
    """The guard that stops the spec ratio being satisfied by squashing."""

    def ring(self, x_scale=1.0, y_scale=1.0, radius=100.0):
        angles = np.radians(canonical.SLOT_ANGLES)
        return np.column_stack([np.cos(angles) * radius * x_scale,
                                np.sin(angles) * radius * y_scale])

    def test_an_even_ring_reads_as_a_circle(self):
        self.assertAlmostEqual(check_margins.ring_aspect(self.ring()), 1.0,
                               places=6)

    def test_a_ring_squashed_on_one_axis_reads_its_axis_ratio(self):
        self.assertAlmostEqual(check_margins.ring_aspect(self.ring(y_scale=0.8)),
                               0.8, delta=0.02)

    def test_a_ring_that_is_translated_is_still_a_circle(self):
        shifted = self.ring() + np.array([37.0, -12.0])
        self.assertAlmostEqual(check_margins.ring_aspect(shifted), 1.0, places=6)

    def test_fewer_than_eight_points_is_not_scored(self):
        self.assertTrue(np.isnan(check_margins.ring_aspect(np.zeros((5, 2)))))


class PurplePointTests(unittest.TestCase):
    def test_eight_button_coloured_blobs_come_back_as_points(self):
        image = np.full((400, 400, 3), (60, 60, 60), dtype=np.uint8)
        for angle in canonical.SLOT_ANGLES:
            radians = np.radians(angle)
            x = 200 + 140 * np.cos(radians)
            y = 200 - 140 * np.sin(radians)
            cv2.circle(image, (int(round(x)), int(round(y))), 18,
                       (200, 110, 190), -1)
        points = check_margins.purple_points(image)
        self.assertEqual(len(points), 8)

    def test_a_frame_without_buttons_returns_nothing(self):
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        self.assertEqual(len(check_margins.purple_points(image)), 0)

    def test_the_checker_and_the_live_path_share_one_blob_finder(self):
        # The acceptance number and the correction have to look at the same
        # pixels; a second colour threshold somewhere would let them drift.
        image = np.full((300, 300, 3), (60, 60, 60), dtype=np.uint8)
        for angle in canonical.SLOT_ANGLES:
            radians = np.radians(angle)
            cv2.circle(image, (150 + int(100 * np.cos(radians)),
                               150 - int(100 * np.sin(radians))), 16,
                       (200, 110, 190), -1)
        self.assertTrue(np.array_equal(check_margins.purple_points(image),
                                       canonical.purple_points(image)))
        self.assertFalse(hasattr(check_margins, "PURPLE_LOW"))


if __name__ == "__main__":
    unittest.main()
