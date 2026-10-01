import unittest

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


if __name__ == "__main__":
    unittest.main()
