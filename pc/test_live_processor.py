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


if __name__ == "__main__":
    unittest.main()

