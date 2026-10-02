"""The eight-slot measurement and the pull-back it drives."""
import unittest

import numpy as np

from pc import canonical


def ring_points(centre, radius, angles):
    radians = np.radians(angles)
    return np.column_stack([centre[0] + radius * np.cos(radians),
                            centre[1] - radius * np.sin(radians)])


class SlotTests(unittest.TestCase):
    def test_every_slot_gets_its_own_point(self):
        centre = np.array([320.0, 240.0])
        points = ring_points(centre, 2.0 * 137.0, canonical.SLOT_ANGLES)
        ratios = canonical.slot_ratios(centre, 137.0, points)
        self.assertTrue(np.allclose(ratios, 2.0, atol=1e-9))

    def test_a_stray_blob_is_left_out_instead_of_moving_a_slot(self):
        centre = np.array([320.0, 240.0])
        points = ring_points(centre, 2.0 * 137.0, canonical.SLOT_ANGLES)
        points = np.vstack([points, [centre[0] + 60.0, centre[1] - 12.0]])
        ratios = canonical.slot_ratios(centre, 137.0, points)
        self.assertTrue(np.allclose(ratios, 2.0, atol=1e-9))

    def test_two_blobs_in_one_slot_take_the_median(self):
        centre = np.array([0.0, 0.0])
        points = ring_points(centre, 200.0, canonical.SLOT_ANGLES)
        points = np.vstack([points,
                            ring_points(centre, 220.0, [canonical.SLOT_ANGLES[3]])])
        ratios = canonical.slot_ratios(centre, 100.0, points)
        self.assertAlmostEqual(ratios[3], 2.1, places=9)
        self.assertAlmostEqual(ratios[0], 2.0, places=9)

    def test_the_labels_agree_with_the_annotator(self):
        from tools import measure_button_frames
        for angle in np.arange(0.0, 360.0, 1.0):
            self.assertEqual(canonical.user_label(angle),
                             measure_button_frames.user_label(angle))


class ProfileTests(unittest.TestCase):
    def test_a_tilt_and_a_squash_come_back_out_of_the_eight_samples(self):
        order = 3
        wanted = np.array([2.0, 0.05, -0.03, 0.02, 0.01, -0.004, 0.006])
        design = canonical._basis(np.radians(canonical.SLOT_ANGLES), order)
        ratios = design @ wanted
        found = canonical.ring_profile(ratios, order)
        self.assertTrue(np.allclose(found, wanted, atol=1e-9))

    def test_a_direction_that_was_never_measured_is_not_invented(self):
        ratios = np.full(8, np.nan)
        ratios[1] = 1.4
        self.assertIsNone(canonical.ring_profile(ratios))

    def test_the_spread_is_reported_against_the_mean(self):
        self.assertAlmostEqual(canonical.ring_error([1.0, 1.0, 1.0, 1.0]), 0.0)
        self.assertIsNone(canonical.ring_error([1.0, np.nan]))


class PullBackTests(unittest.TestCase):
    def sample_radius(self, maps, width, height, centre, output_radius, angle):
        """Radius the map reads, bilinear, so pixel rounding is not the answer."""
        radians = np.radians(angle)
        x = centre[0] + output_radius * np.cos(radians)
        y = centre[1] - output_radius * np.sin(radians)
        map_x, map_y = maps

        def bilinear(values):
            # Map values live on the pixel-centre grid, so the query point has
            # to be shifted half a pixel before it is read.
            left, top = int(np.floor(x - 0.5)), int(np.floor(y - 0.5))
            fx, fy = x - 0.5 - left, y - 0.5 - top
            left = min(max(left, 0), width - 2)
            top = min(max(top, 0), height - 2)
            patch = values[top:top + 2, left:left + 2]
            return float(patch[0, 0] * (1 - fx) * (1 - fy)
                         + patch[0, 1] * fx * (1 - fy)
                         + patch[1, 0] * (1 - fx) * fy
                         + patch[1, 1] * fx * fy)

        return float(np.hypot(bilinear(map_x) - centre[0],
                              bilinear(map_y) - centre[1]))

    def test_the_screen_edge_does_not_move(self):
        width = height = 400
        centre = np.array([200.0, 200.0])
        coefficients = np.array([1.40, 0.08, -0.05, 0.0, 0.0, 0.0, 0.0])
        maps = canonical.pull_back_maps(width, height, centre, 60.0,
                                        coefficients, coarse=1)
        for angle in canonical.SLOT_ANGLES:
            # Inside the screen the map is the identity, to the last pixel.
            self.assertAlmostEqual(
                self.sample_radius(maps, width, height, centre, 30.0, angle),
                30.0, delta=0.02)
            # And at the screen edge it has not started moving yet.
            self.assertAlmostEqual(
                self.sample_radius(maps, width, height, centre, 60.0, angle),
                60.0, delta=0.6)

    def test_the_ramp_is_flat_at_both_ends(self):
        radius = np.array([0.0, 59.0, 60.0, 80.0, 82.0, 200.0])
        ramp = canonical._ramp(radius, 60.0, 82.0)
        self.assertTrue(np.allclose(ramp[:3], 0.0))
        self.assertTrue(np.allclose(ramp[4:], 1.0))
        self.assertTrue(np.all((ramp >= 0.0) & (ramp <= 1.0)))

    def test_the_ring_lands_on_the_same_radius_in_every_direction(self):
        width = height = 800
        centre = np.array([400.0, 400.0])
        screen = 100.0
        order = 3
        wanted = np.array([canonical.TILE_RATIO, 0.04, -0.03, 0.02, 0.015,
                           0.0, 0.0])
        design = canonical._basis(np.radians(canonical.SLOT_ANGLES), order)
        ratios = design @ wanted
        maps = canonical.pull_back_maps(width, height, centre, screen, wanted,
                                        coarse=1)
        want = canonical.TILE_RATIO * screen
        for slot, angle in enumerate(canonical.SLOT_ANGLES):
            # The output pixel at the canonical ring radius has to read the
            # input pixel where that tile actually is.
            self.assertAlmostEqual(
                self.sample_radius(maps, width, height, centre, want, angle),
                ratios[slot] * screen, delta=1.0)

    def test_strength_zero_leaves_the_frame_alone(self):
        coefficients = np.array([1.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(canonical.pull_back_maps(200, 200, (100, 100), 40.0,
                                                   coefficients, strength=0.0))

    def test_a_wild_fit_is_clipped_rather_than_wrapping_the_picture(self):
        width = height = 600
        centre = np.array([300.0, 300.0])
        coefficients = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        maps = canonical.pull_back_maps(width, height, centre, 80.0,
                                        coefficients, coarse=1)
        want = canonical.TILE_RATIO * 80.0
        clipped = want * (1.0 - canonical.MAX_FACTOR)
        self.assertAlmostEqual(
            self.sample_radius(maps, width, height, centre, want, 22.5),
            clipped, delta=1.0)
