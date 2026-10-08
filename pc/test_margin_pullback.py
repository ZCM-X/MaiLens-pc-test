"""The delivered-screen measurement and the equal-margin pull-back."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pc import canonical
from pc.live_processor import LiveProcessor


def button_centres(centre, screen_radius: float, ring: float, squash: float):
    """Where the eight buttons sit, so a test can also paint them out."""
    out = []
    for angle in canonical.SLOT_ANGLES:
        radians = np.radians(angle)
        out.append((centre[0] + ring * screen_radius * np.cos(radians),
                    centre[1] - squash * ring * screen_radius * np.sin(radians)))
    return out


def machine_frame(width: int = 720, height: int = 1280, screen_radius: float = 200.0,
                  ring: float = 1.22, squash: float = 0.80, button_radius: int = 9):
    """A head-on machine with one anisotropic gap between screen and buttons."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    centre = (width * 0.5, height * 0.5)
    cv2.circle(frame, (int(centre[0]), int(centre[1])), int(screen_radius),
               (255, 255, 0), -1, lineType=cv2.LINE_AA)
    for x, y in button_centres(centre, screen_radius, ring, squash):
        cv2.circle(frame, (int(round(x)), int(round(y))), button_radius,
                   (255, 0, 255), -1, lineType=cv2.LINE_AA)
    return frame


def spreads(frame, centre, radius):
    ratios = canonical.slot_ratios(np.asarray(centre, dtype=np.float64), radius,
                                   canonical.purple_points(frame))
    known = ratios[np.isfinite(ratios)]
    return ratios, float(known.std() / known.mean())


class ScreenBlobTests(unittest.TestCase):
    def test_a_centred_screen_is_measured(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 120, (255, 255, 0), -1)
        blob = canonical.screen_blob(frame)
        self.assertIsNotNone(blob)
        np.testing.assert_allclose(blob[0], (320.0, 240.0), atol=2.0)
        self.assertAlmostEqual(blob[1], 120.0, delta=4.0)

    def test_a_screen_running_off_the_frame_is_refused(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 0), 120, (255, 255, 0), -1)
        self.assertIsNone(canonical.screen_blob(frame))

    def test_artwork_is_not_mistaken_for_the_screen(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 3, (255, 255, 0), -1)
        self.assertIsNone(canonical.screen_blob(frame))


class MarginPullBackTests(unittest.TestCase):
    def test_the_ring_is_measured_against_the_delivered_screen(self):
        frame = machine_frame()
        processor = LiveProcessor(ring_round=True)
        measured = processor._measure_ring(frame)
        self.assertIsNotNone(measured)
        centre, radius, _fitted, spread = measured
        np.testing.assert_allclose(centre, (360.0, 640.0), atol=3.0)
        self.assertAlmostEqual(radius, 200.0, delta=6.0)
        self.assertGreater(spread, 0.04, "这副图本来就是不等距的")

    def test_the_pull_back_evens_the_four_sides_out(self):
        frame = machine_frame()
        before, before_spread = spreads(frame, (360.0, 640.0), 200.0)
        self.assertGreater(before_spread, 0.05)
        processor = LiveProcessor(ring_round=True)
        fixed = processor._pull_margins_back(frame)
        after, after_spread = spreads(fixed, (360.0, 640.0), 200.0)
        self.assertLess(after_spread, before_spread * 0.5,
                        f"不等距没有明显改善：{before_spread:.3f} -> {after_spread:.3f}")
        # It evens the ring out; it does not resize the machine.
        self.assertAlmostEqual(float(np.nanmean(after)), float(np.nanmean(before)), delta=0.04)

    def test_a_frame_without_buttons_is_left_alone(self):
        frame = machine_frame()
        for x, y in button_centres((360.0, 640.0), 200.0, 1.22, 0.80):
            cv2.circle(frame, (int(round(x)), int(round(y))), 26, (0, 0, 0), -1)
        processor = LiveProcessor(ring_round=True)
        self.assertIsNone(processor._measure_ring(frame))
        np.testing.assert_array_equal(processor._pull_margins_back(frame), frame)

    def test_the_correction_holds_its_hand_when_the_screen_disappears(self):
        processor = LiveProcessor(ring_round=True, ring_hold=0)
        blank = np.zeros((1280, 720, 3), dtype=np.uint8)
        np.testing.assert_array_equal(processor._pull_margins_back(blank), blank)

    def test_a_frame_whose_screen_left_the_edge_reuses_the_last_reading(self):
        frame = machine_frame()
        processor = LiveProcessor(ring_round=True)
        first = processor._measure_ring(frame)
        self.assertIsNotNone(first)
        # The cabinet runs off the bottom of the picture, so the play field
        # merges with the lit panel below it and cannot be measured on its own.
        cropped = frame.copy()
        cropped[780:, 350:372] = (255, 255, 0)   # the lit panel below, running off the frame
        cropped[-8:, :] = (255, 255, 0)
        self.assertIsNone(canonical.screen_blob(cropped))
        second = processor._measure_ring(cropped)
        self.assertIsNotNone(second, "上一帧的屏幕依然是更好的基准")
        np.testing.assert_allclose(second[1], first[1], rtol=0.02)

    def test_the_correction_can_be_switched_off(self):
        frame = machine_frame()
        processor = LiveProcessor(ring_round=False)
        np.testing.assert_array_equal(processor._pull_margins_back(frame), frame)


if __name__ == "__main__":
    unittest.main()
