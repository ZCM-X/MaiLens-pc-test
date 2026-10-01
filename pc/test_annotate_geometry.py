import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.annotate_geometry import FrameSource, GeometryAnnotator
from .process_session import GeometryDetector


class AnnotationTests(unittest.TestCase):
    def test_button_boxes_round_trip_as_multiple_yolo_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jpg"
            cv2.imwrite(str(source_path), np.zeros((200, 400, 3), dtype=np.uint8))
            source = FrameSource(source_path, image_paths=[source_path])
            output = root / "dataset"
            try:
                annotator = GeometryAnnotator(source, output, 0, include_buttons=True)
                annotator.boxes = {
                    0: [(20, 20, 380, 180)],
                    1: [(80, 50, 320, 150)],
                    2: [
                        (40 + index * 40, 155, 65 + index * 40, 185)
                        for index in range(8)
                    ],
                }
                annotator.save()

                labels_path = output / "labels" / "train" / "frame-000001.txt"
                lines = labels_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(sum(line.startswith("2 ") for line in lines), 8)
                dataset_yaml = (output / "dataset.yaml").read_text(encoding="utf-8")
                self.assertIn("nc: 3", dataset_yaml)
                self.assertIn("button", dataset_yaml)

                reloaded = GeometryAnnotator(source, output, 0, include_buttons=True)
                self.assertEqual(len(reloaded.boxes[2]), 8)
                self.assertEqual(reloaded.boxes[2][0], (40, 155, 65, 185))
            finally:
                source.close()

    def test_button_only_detections_are_ignored_by_geometry_lock_selector(self):
        outer, inner = GeometryDetector._pick_geometry_boxes([
            ("button", 0.93, (30, 230, 70, 270)),
            ("button", 0.90, (90, 230, 130, 270)),
        ])
        self.assertIsNone(outer)
        self.assertIsNone(inner)

    def test_button_detections_do_not_replace_real_geometry(self):
        outer, inner = GeometryDetector._pick_geometry_boxes([
            ("outer_frame", 0.80, (20, 10, 380, 290)),
            ("inner_screen", 0.90, (80, 40, 320, 230)),
            ("button", 0.99, (100, 210, 140, 250)),
        ])
        self.assertEqual(outer, (20, 10, 380, 290))
        self.assertEqual(inner, (80, 40, 320, 230))


if __name__ == "__main__":
    unittest.main()
