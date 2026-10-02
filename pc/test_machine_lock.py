"""Tests for the two-ring cabinet fit in ``pc.machine_lock``."""
from __future__ import annotations

import math
import unittest

import numpy as np

from pc.machine_lock import (RING_WEIGHT, fit_two_ring, matrix_sqrt, world_lock)

SCREEN_RADIUS = 130.0
RING_RADIUS = 1.24


def cabinet_points(squash: float = 0.90, angle: float = 20.0,
                   centre=(300.0, 420.0), count: int = 72):
    """A concentric screen circle plus eight button centres, then squashed."""
    radians = math.radians(angle)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    linear = np.array([[cos_a, -sin_a], [sin_a, cos_a]]) @ np.diag([1.0, squash])
    centre = np.array(centre)

    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    screen = (unit * SCREEN_RADIUS) @ linear.T + centre
    buttons = (np.column_stack([
        np.cos(np.arange(8) * math.pi / 4.0),
        np.sin(np.arange(8) * math.pi / 4.0)]) * RING_RADIUS * SCREEN_RADIUS) @ linear.T + centre
    return screen, buttons, linear, centre


def circle_radii(points, fit):
    linear = matrix_sqrt(np.array(fit["q"]))
    offset = (np.asarray(points) - np.array(fit["center"])) @ linear.T
    return np.linalg.norm(offset, axis=1)


class TwoRingFitTest(unittest.TestCase):
    def test_recovers_the_squash_of_a_clean_pair(self):
        screen, buttons, linear, _ = cabinet_points()
        fit = fit_two_ring(screen, buttons)
        self.assertIsNotNone(fit)
        radii = circle_radii(screen, fit)
        np.testing.assert_allclose(radii, 1.0, atol=2e-3)
        ring = circle_radii(buttons, fit)
        self.assertLess(ring.std() / ring.mean(), 2e-3)
        self.assertAlmostEqual(ring.mean(), RING_RADIUS, places=2)
        # the recovered shape must undo the squash we put in
        recovered = matrix_sqrt(np.array(fit["q"])) @ linear
        singular = np.linalg.svd(recovered, compute_uv=False)
        self.assertAlmostEqual(singular[0] / singular[1], 1.0, places=2)

    def test_survives_a_hole_in_the_screen_mask(self):
        screen, buttons, _, _ = cabinet_points()
        # the character art eats a quarter of the cyan mask
        keep = screen[:, 1] < screen[:, 1].mean()
        holed = screen[keep]
        fit = fit_two_ring(holed, buttons)
        self.assertIsNotNone(fit)
        radii = circle_radii(holed, fit)
        self.assertLess(radii.std() / radii.mean(), 4e-2)
        ring = circle_radii(buttons, fit)
        self.assertLess(ring.std() / ring.mean(), 3e-2)
        self.assertAlmostEqual(np.mean(radii), 1.0, places=1)

    def test_ignores_a_speck_that_is_not_a_button(self):
        screen, buttons, _, _ = cabinet_points()
        dirty = np.vstack([buttons, [[buttons[:, 0].mean(), buttons[:, 1].mean() + 40.0]]])
        fit = fit_two_ring(screen, dirty)
        self.assertIsNotNone(fit)
        ring = circle_radii(buttons, fit)
        self.assertLess(ring.std() / ring.mean(), 2e-2)

    def test_ring_weight_moves_the_button_ring(self):
        screen, buttons, _, _ = cabinet_points()
        # push the buttons out of true, as the real skirt-visibility bias does
        biased = buttons.copy()
        biased[:, 1] += 0.07 * SCREEN_RADIUS * np.sign(buttons[:, 1] - buttons[:, 1].mean())
        light = fit_two_ring(screen, biased, inner_weight=0.1)
        heavy = fit_two_ring(screen, biased, inner_weight=3.0)
        light_ring = circle_radii(biased, light)
        heavy_ring = circle_radii(biased, heavy)
        self.assertLess(light_ring.std() / light_ring.mean(),
                        heavy_ring.std() / heavy_ring.mean())
        light_screen = circle_radii(screen, light)
        heavy_screen = circle_radii(screen, heavy)
        self.assertLess(heavy_screen.std(), light_screen.std())
        self.assertGreater(RING_WEIGHT, 0.0)

    def test_rejects_a_ring_with_too_few_centres(self):
        screen, buttons, _, _ = cabinet_points()
        self.assertIsNone(fit_two_ring(screen, buttons[:4]))


class WorldLockTest(unittest.TestCase):
    def test_holds_still_inside_the_deadband(self):
        noise = np.random.default_rng(7).normal(0.0, 1.0, 200)
        held = world_lock(100.0 + noise, 4.0, 0.35, 0.10, 3.0, 1.0 / 60.0)
        self.assertLess(np.abs(held - 100.0).max(), 4.0)

    def test_follows_a_real_move(self):
        ramp = np.linspace(0.0, 300.0, 200)
        held = world_lock(ramp, 2.0, 0.35, 0.08, 3.0, 1.0 / 60.0)
        self.assertLess(abs(held[-1] - ramp[-1]), 12.0)
        self.assertGreater(held[-1], 200.0)

    def test_leaves_nan_gaps_to_the_interpolator(self):
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        held = world_lock(values, 0.0, 0.3, 0.1, 3.0, 1.0 / 60.0)
        self.assertTrue(np.isfinite(held).all())


if __name__ == "__main__":
    unittest.main()
