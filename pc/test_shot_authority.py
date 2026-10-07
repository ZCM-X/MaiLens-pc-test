import unittest

import cv2
import numpy as np

from .process_session import PlaneLockTracker
from .shot_authority import AuthorityGovernor, AuthorityLimits


def travel(shift, zoom=1.0, output_scale=1.0, centre=(160.0, 90.0)):
    return {
        "shift": np.asarray(shift, dtype=np.float64),
        "zoom": float(zoom),
        "output_scale": float(output_scale),
        "target_centre": np.asarray(centre, dtype=np.float64),
    }


class AuthorityGovernorTests(unittest.TestCase):
    def test_a_lock_inside_the_travel_is_untouched(self):
        governor = AuthorityGovernor(AuthorityLimits(max_shift=0.30))
        matrix = np.array([[1.1, 0.0, -12.0], [0.0, 1.1, 7.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        decision = governor.decide(travel((20.0, -10.0)), 320, 180)
        self.assertEqual(decision.mode, "lock")
        self.assertFalse(decision.limited)
        np.testing.assert_allclose(decision.apply(matrix), matrix)
        self.assertEqual(decision.move_box((10, 10, 50, 40)), (10, 10, 50, 40))

    def test_past_the_travel_the_leftover_stays_on_screen(self):
        governor = AuthorityGovernor(AuthorityLimits(max_shift=0.25, min_zoom=1.0, max_zoom=1.0))
        # A pan that asks the crop for a 60 px correction, with only 45 px of
        # travel: 15 px of the move has to stay visible.
        decision = governor.decide(travel((-60.0, 0.0)), 320, 180)
        self.assertEqual(decision.mode, "follow")
        self.assertAlmostEqual(decision.limit, 45.0)
        self.assertAlmostEqual(decision.travelled, 60.0)
        np.testing.assert_allclose(decision.residual_shift, (15.0, 0.0), atol=1e-4)

    def test_the_leftover_keeps_the_direction_the_machine_moved(self):
        governor = AuthorityGovernor(AuthorityLimits(max_shift=0.25, min_zoom=1.0, max_zoom=1.0))
        # The travel is clamped as a distance, not per axis, so a diagonal
        # pan keeps its bearing and only loses length.
        shifted = governor.decide(travel((-60.0, 40.0)), 320, 180)
        self.assertGreater(shifted.residual_shift[0], 0.0)
        self.assertLess(shifted.residual_shift[1], 0.0)
        self.assertAlmostEqual(
            float(np.hypot(*shifted.residual_shift)), shifted.travelled - shifted.limit, places=3,
        )
        bearing = shifted.residual_shift / max(float(np.hypot(*shifted.residual_shift)), 1e-9)
        np.testing.assert_allclose(bearing, (60.0, -40.0) / np.hypot(60.0, 40.0), atol=1e-4)

    def test_the_hand_over_has_no_step(self):
        governor = AuthorityGovernor(AuthorityLimits(max_shift=0.25))
        last = None
        for distance in np.linspace(0.0, 200.0, 401):
            decision = governor.decide(travel((distance, 0.0)), 320, 180)
            residual = float(decision.residual_shift[0])
            self.assertAlmostEqual(distance + residual, min(distance, 45.0), places=4)
            if last is not None:
                self.assertLessEqual(abs(residual - last), 0.5 + 1e-9)
            last = residual

    def test_walking_closer_past_the_range_leaves_the_machine_bigger(self):
        governor = AuthorityGovernor(AuthorityLimits(min_zoom=0.60, max_zoom=1.50))
        decision = governor.decide(travel((0.0, 0.0), zoom=3.0), 320, 180)
        self.assertEqual(decision.mode, "follow")
        self.assertAlmostEqual(decision.residual_zoom, 2.0)
        walked_back = governor.decide(travel((0.0, 0.0), zoom=0.30), 320, 180)
        self.assertEqual(walked_back.mode, "follow")
        self.assertAlmostEqual(walked_back.residual_zoom, 0.5)

    def test_an_unknown_travel_trusts_the_lock(self):
        governor = AuthorityGovernor()
        decision = governor.decide(None, 320, 180)
        self.assertEqual(decision.mode, "lock")
        self.assertFalse(decision.limited)

    def test_the_follow_matrix_lands_the_machine_on_the_residual(self):
        governor = AuthorityGovernor(AuthorityLimits(max_shift=0.25, min_zoom=1.0, max_zoom=1.0))
        decision = governor.decide(travel((0.0, -90.0)), 320, 180)
        centre = np.asarray(decision.target_centre, dtype=np.float32)
        points = np.float32([[centre[0], centre[1]], [centre[0] + 20.0, centre[1]]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(points, decision.follow_matrix()).reshape(-1, 2)
        np.testing.assert_allclose(mapped[0], centre + decision.residual_shift, atol=1e-3)
        np.testing.assert_allclose(mapped[1], centre + decision.residual_shift + (20.0, 0.0), atol=1e-3)

    def test_the_lock_transform_is_composed_not_replaced(self):
        governor = AuthorityGovernor(AuthorityLimits(max_shift=0.25, min_zoom=1.0, max_zoom=1.0))
        decision = governor.decide(travel((0.0, -90.0)), 320, 180)
        lock = np.array([[0.5, 0.0, 120.0], [0.0, 0.5, 60.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        composed = decision.apply(lock)
        point = np.float32([[200.0, 120.0], [80.0, 60.0]]).reshape(-1, 1, 2)
        expected = cv2.perspectiveTransform(point, lock)
        expected = cv2.perspectiveTransform(expected, decision.follow_matrix()).reshape(-1, 2)
        got = cv2.perspectiveTransform(point, composed).reshape(-1, 2)
        np.testing.assert_allclose(got, expected, atol=1e-3)

    def test_the_foreground_mask_rides_with_the_machine(self):
        governor = AuthorityGovernor(AuthorityLimits(max_shift=0.25, min_zoom=1.0, max_zoom=1.0))
        decision = governor.decide(travel((0.0, -90.0)), 320, 180)
        self.assertEqual(decision.move_box((100, 50, 140, 90)), (100, 95, 140, 135))
        self.assertIsNone(decision.move_box(None))


class LockTravelTests(unittest.TestCase):
    """The tracker reports the work the crop is doing, in pixels."""

    def locked_tracker(self, box=(90, 40, 230, 160), width=320, height=200):
        gray = np.random.default_rng(3).integers(0, 256, (height, width), dtype=np.uint8)
        tracker = PlaneLockTracker(detect_every=1)
        self.assertTrue(tracker.initialize(gray, box, None, width, height, 0.71))
        tracker.current_box = tuple(int(value) for value in box)
        return tracker

    def test_a_still_lock_reports_no_travel(self):
        tracker = self.locked_tracker()
        reading = tracker.lock_travel(320, 200)
        self.assertIsNotNone(reading)
        self.assertLess(float(np.hypot(*reading["shift"])), 1e-6)
        self.assertAlmostEqual(reading["zoom"], 1.0, places=6)

    def test_a_pan_reports_the_pixels_the_crop_had_to_take_out(self):
        tracker = self.locked_tracker()
        tracker.current_to_reference = np.array(
            [[1.0, 0.0, -60.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32,
        )
        reading = tracker.lock_travel(320, 200)
        np.testing.assert_allclose(reading["shift"], (-60.0, 20.0), atol=1e-4)
        self.assertGreater(reading["output_scale"], 0.0)
        self.assertGreater(float(np.hypot(*reading["target_centre"])), 0.0)

    def test_zooming_the_crop_reports_the_size_ratio(self):
        tracker = self.locked_tracker()
        # The crop halves the picture at the machine, which means the machine
        # looks twice as big as it did at the lock.
        tracker.current_to_reference = np.array(
            [[0.5, 0.0, 80.0], [0.0, 0.5, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32,
        )
        reading = tracker.lock_travel(320, 200)
        self.assertAlmostEqual(reading["zoom"], 2.0, places=4)

    def test_an_unlocked_tracker_reports_nothing(self):
        tracker = PlaneLockTracker(detect_every=1)
        self.assertIsNone(tracker.lock_travel(320, 200))


if __name__ == "__main__":
    unittest.main()
