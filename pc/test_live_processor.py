import unittest
from unittest.mock import patch

import cv2
import numpy as np

from .live_processor import LiveProcessor


class LiveProcessorTests(unittest.TestCase):
    def test_each_frame_is_processed_without_waiting_for_a_session_file(self):
        processor = LiveProcessor(debug=True)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (60, 30), (260, 150), (80, 220, 150), 3)
        metadata = {
            "frame_id": 1,
            "timestamp": 10.0,
            "pose": {
                "timestamp": 10.0,
                "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
        }
        output, debug = processor.process(frame, metadata)
        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(debug["frame_id"], 1)
        self.assertEqual(debug["sensor_delta_ms"], 0.0)

    def test_machine_lock_expires_after_repeated_missed_detections(self):
        class SequenceDetector:
            enabled = True

            def __init__(self):
                self.calls = 0

            def detect(self, _frame):
                self.calls += 1
                if self.calls == 1:
                    return (30, 35, 210, 155), None
                return None, None

        processor = LiveProcessor(detect_every=1)
        processor.detector = SequenceDetector()
        processor.lock_tracker.max_age_frames = 2
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        metadata = {
            "timestamp": 10.0,
            "pose": {
                "timestamp": 10.0,
                "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
        }

        _output, first = processor.process(frame, metadata)
        self.assertEqual(first["lock_source"], "outer_buttons")
        self.assertIsNotNone(first["detected_outer"])

        for _ in range(3):
            _output, last = processor.process(frame, metadata)
        self.assertEqual(last["lock_source"], "searching")
        self.assertIsNone(last["detected_outer"])
        self.assertGreater(last["detection_age_frames"], processor.lock_tracker.max_age_frames)

    def test_resolution_change_reacquires_without_optical_flow_shape_error(self):
        class SequenceDetector:
            enabled = True

            def detect(self, frame):
                height, width = frame.shape[:2]
                return (20, 20, width - 20, height - 20), None

        processor = LiveProcessor(detect_every=1)
        processor.detector = SequenceDetector()
        metadata = {"timestamp": 10.0, "pose": {"timestamp": 10.0}}
        first, _ = processor.process(np.zeros((180, 320, 3), dtype=np.uint8), metadata)
        second, debug = processor.process(np.zeros((240, 400, 3), dtype=np.uint8), metadata)
        self.assertEqual(first.shape, (180, 320, 3))
        self.assertEqual(second.shape, (240, 400, 3))
        self.assertEqual(debug["lock_source"], "outer_buttons")

    def test_detector_runs_on_fixed_schedule_while_searching(self):
        class CountingDetector:
            enabled = True

            def __init__(self):
                self.calls = 0

            def detect(self, _frame):
                self.calls += 1
                return None, None

        detector = CountingDetector()
        processor = LiveProcessor(detect_every=3)
        processor.detector = detector
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        metadata = {"timestamp": 10.0, "pose": {"timestamp": 10.0}}
        for _ in range(8):
            processor.process(frame, metadata)
        self.assertEqual(detector.calls, 3)

    def test_live_lock_recenters_after_stale_flow_and_fresh_detection(self):
        class FixedDetector:
            enabled = True

            def detect(self, _frame):
                return (30, 15, 290, 170), (95, 45, 225, 135)

        processor = LiveProcessor(detect_every=1)
        processor.detector = FixedDetector()
        frame = np.random.default_rng(21).integers(0, 256, (180, 320, 3), dtype=np.uint8)
        metadata = {"timestamp": 10.0, "pose": {"timestamp": 10.0}}

        with patch("pc.live_processor.map_fisheye_box_to_output", side_effect=lambda box, *_args: box):
            processor.process(frame, metadata)
            processor.plane_tracker.current_to_reference = np.array(
                [[1.0, 0.0, 18.0], [0.0, 1.0, -9.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            processor.plane_tracker.age_frames = 4
            processor.plane_tracker.points = None
            _output, debug = processor.process(frame, metadata)

        self.assertTrue(debug["plane_reacquired"])
        self.assertEqual(processor.plane_tracker.age_frames, 0)
        np.testing.assert_allclose(processor.plane_tracker.current_to_reference, np.eye(3), atol=1e-6)

    def test_the_anchor_does_not_wait_for_the_box_trackers_jump_gate(self):
        class FixedDetector:
            enabled = True
            inner = (95, 45, 225, 135)
            outer = (30, 15, 290, 170)

            def detect(self, _frame):
                return self.outer, self.inner

        detector = FixedDetector()
        processor = LiveProcessor(detect_every=1)
        processor.detector = detector
        frame = np.random.default_rng(7).integers(0, 256, (360, 640, 3), dtype=np.uint8)
        metadata = {"timestamp": 10.0, "pose": {"timestamp": 10.0}}

        with patch("pc.live_processor.map_fisheye_box_to_output", side_effect=lambda box, *_args: box):
            processor.process(frame, metadata)
            self.assertTrue(processor.plane_tracker.locked)
            # Still the machine, but far enough from the last box that the
            # continuity gate keeps the old one.  The absolute anchor is a
            # measurement of where the machine really is, so it must not be
            # dropped at exactly the moment the phone moved a long way.
            detector.inner = (470, 135, 610, 225)
            detector.outer = (400, 100, 640, 260)
            _output, debug = processor.process(frame, metadata)

        self.assertGreater(debug["detection_age_frames"], 0)
        self.assertFalse(np.allclose(processor.plane_tracker.correction_target, np.eye(3)))


if __name__ == "__main__":
    unittest.main()
