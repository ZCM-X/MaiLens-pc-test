"""The lock may never show the machine twice in one frame.

The renderer fills every pixel the warp cannot fetch with the plain rectified
frame, so a transform that reaches past the picture used to leave the machine
pinned in the middle *and* a smeared second copy of it in the corner it came
from.  ``ease_to_source`` eases the transform back until the machine''s own
footprint is covered again, which keeps one viewpoint and one machine.
"""

from __future__ import annotations

import unittest

import numpy as np

from pc.process_session import (
    _warp_with_valid_source,
    ease_to_source,
    source_coverage,
)

WIDTH = 720
HEIGHT = 1280
#: Where the machine sits in the delivered frame before the lock moves it.
FOOTPRINT = (150.0, 480.0, 430.0, 900.0)


def translated(x: float, y: float = 0.0) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def hole_fraction(matrix: np.ndarray) -> float:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    _warped, valid = _warp_with_valid_source(frame, matrix)
    return 1.0 - float(valid.mean())


class PlaneEasingTests(unittest.TestCase):
    def test_a_lock_that_covers_the_machine_is_left_alone(self):
        matrix = translated(-200.0, 60.0)
        self.assertFalse(
            source_coverage(matrix, WIDTH, HEIGHT),
            "这个矩阵本身在画面右边留了洞，正是要靠 footprint 放行的情形",
        )
        eased, fraction = ease_to_source(matrix, WIDTH, HEIGHT, region=FOOTPRINT)
        self.assertEqual(fraction, 1.0)
        np.testing.assert_allclose(eased, matrix, atol=1e-6)

    def test_a_lock_that_would_bury_the_machine_is_eased_back(self):
        # The picture slides far enough left that the hole reaches the machine.
        matrix = translated(-560.0)
        self.assertFalse(source_coverage(matrix, WIDTH, HEIGHT, region=FOOTPRINT))
        eased, fraction = ease_to_source(matrix, WIDTH, HEIGHT, region=FOOTPRINT)
        self.assertLess(fraction, 1.0)
        self.assertTrue(source_coverage(eased, WIDTH, HEIGHT, region=FOOTPRINT))

    def test_the_eased_frame_never_shows_the_machine_twice(self):
        for matrix in (translated(-560.0), translated(-300.0, 700.0)):
            eased, _fraction = ease_to_source(matrix, WIDTH, HEIGHT, region=FOOTPRINT)
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            _warped, valid = _warp_with_valid_source(frame, eased)
            x0, y0, x1, y1 = (int(value) for value in FOOTPRINT)
            self.assertGreaterEqual(
                float(valid[y0:y1, x0:x1].min()), 1.0,
                "机台脚下只要有一块回落到原图，画面上就会多出一台机台",
            )

    def test_a_fold_is_refused_even_when_the_footprint_is_covered(self):
        folded = np.linalg.inv(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.001, -0.001, 1.0]])
        )
        small = (50.0, 50.0, 150.0, 150.0)
        self.assertFalse(source_coverage(folded, WIDTH, HEIGHT, region=small))
        eased, fraction = ease_to_source(folded, WIDTH, HEIGHT, region=small)
        self.assertLess(fraction, 1.0)
        self.assertTrue(source_coverage(eased, WIDTH, HEIGHT, region=small))
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        _warped, valid = _warp_with_valid_source(frame, eased)
        x0, y0, x1, y1 = (int(value) for value in small)
        self.assertGreaterEqual(
            float(valid[y0:y1, x0:x1].min()), 1.0,
            "被保护的区域里不能再出现第二个视角",
        )

    def test_the_easing_does_not_depend_on_the_matrix_scale(self):
        matrix = translated(-520.0)
        eased, fraction = ease_to_source(matrix, WIDTH, HEIGHT, region=FOOTPRINT)
        scaled, scaled_fraction = ease_to_source(3.5 * matrix, WIDTH, HEIGHT, region=FOOTPRINT)
        self.assertAlmostEqual(fraction, scaled_fraction, places=6)
        np.testing.assert_allclose(eased, scaled, atol=1e-5)

    def test_a_footprint_that_runs_off_the_frame_is_clamped_first(self):
        # A cabinet that is already cut off by the capture cannot be shown
        # twice, so the lock must not give up its travel over it.
        clipped = (-400.0, 300.0, 200.0, 900.0)
        matrix = translated(-520.0)
        eased, fraction = ease_to_source(matrix, WIDTH, HEIGHT, region=clipped)
        self.assertTrue(source_coverage(eased, WIDTH, HEIGHT, region=clipped))
        self.assertGreater(fraction, 0.0)

    def test_a_missing_matrix_stays_missing(self):
        eased, fraction = ease_to_source(None, WIDTH, HEIGHT, region=FOOTPRINT)
        self.assertIsNone(eased)
        self.assertEqual(fraction, 0.0)

    def test_a_head_on_lock_still_has_no_hole_at_all(self):
        # Ordinary locked frames magnify the machine into the middle, so the
        # easing must not quietly soften a good take.
        matrix = np.array(
            [[1.4, 0.0, -0.2 * WIDTH], [0.0, 1.4, -0.2 * HEIGHT], [0.0, 0.0, 1.0]]
        )
        eased, fraction = ease_to_source(matrix, WIDTH, HEIGHT, region=FOOTPRINT)
        self.assertEqual(fraction, 1.0)
        self.assertEqual(hole_fraction(matrix), 0.0)


if __name__ == "__main__":
    unittest.main()
