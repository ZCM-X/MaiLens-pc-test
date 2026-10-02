import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.annotate_geometry import (
    BUTTON_CLASSES,
    GEOMETRY_CLASSES,
    FrameSource,
    GeometryAnnotator,
    read_image,
)
from .process_session import GeometryDetector


class AnnotationTests(unittest.TestCase):
    def test_unicode_source_and_output_paths_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "桌面文件" / "训练图"
            source_dir.mkdir(parents=True)
            source_path = source_dir / "棋盘.png"
            original = np.full((120, 200, 3), (20, 100, 220), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", original)
            self.assertTrue(ok)
            source_path.write_bytes(encoded.tobytes())

            source = FrameSource(source_path, image_paths=[source_path])
            try:
                annotator = GeometryAnnotator(
                    source, root / "标注输出", 0, classes=GEOMETRY_CLASSES,
                )
                self.assertEqual(annotator.frame.shape, original.shape)
                annotator.boxes = {0: [(10, 10, 190, 110)]}
                annotator.save()
                saved_path = root / "标注输出" / "images" / "train" / "frame-000001.jpg"
                self.assertTrue(saved_path.is_file())
                self.assertEqual(read_image(saved_path).shape, original.shape)
            finally:
                source.close()

    def test_reopening_review_resumes_at_first_unlabeled_frame(self):
        class FakeDetector:
            enabled = True

            @staticmethod
            def detect(_frame):
                return (10, 10, 190, 190), (40, 40, 160, 160)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_paths = []
            for index in range(2):
                path = root / f"image-{index}.png"
                cv2.imwrite(str(path), np.zeros((200, 200, 3), dtype=np.uint8))
                image_paths.append(path)
            source = FrameSource(root, image_paths=image_paths)
            output = root / "review"
            try:
                first = GeometryAnnotator(source, output, 0, classes=GEOMETRY_CLASSES)
                first.boxes = {0: [(10, 10, 190, 190)], 1: [(40, 40, 160, 160)]}
                first.save()

                resumed = GeometryAnnotator(
                    source,
                    output,
                    0,
                    classes=GEOMETRY_CLASSES,
                    prelabel_detector=FakeDetector(),
                )
                self.assertEqual(resumed.index, 1)
                self.assertEqual(resumed.boxes[0], [(10, 10, 190, 190)])
                self.assertEqual(resumed.boxes[1], [(40, 40, 160, 160)])
                self.assertEqual(resumed.predicted_classes, {0, 1})
            finally:
                source.close()

    def test_model_suggestions_are_visible_and_can_be_replaced_by_dragging(self):
        class FakeDetector:
            enabled = True

            @staticmethod
            def detect(_frame):
                return (20, 20, 380, 180), (100, 40, 300, 160)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jpg"
            cv2.imwrite(str(source_path), np.zeros((200, 400, 3), dtype=np.uint8))
            source = FrameSource(source_path, image_paths=[source_path])
            try:
                annotator = GeometryAnnotator(
                    source,
                    root / "review",
                    0,
                    classes=GEOMETRY_CLASSES,
                    prelabel_detector=FakeDetector(),
                )
                self.assertEqual(annotator.boxes[0], [(20, 20, 380, 180)])
                self.assertEqual(annotator.boxes[1], [(100, 40, 300, 160)])
                self.assertEqual(annotator.predicted_classes, {0, 1})
                self.assertTrue(annotator.dirty)

                annotator.mouse(cv2.EVENT_LBUTTONDOWN, 30, 30, 0, None)
                annotator.mouse(cv2.EVENT_LBUTTONUP, 370, 170, 0, None)
                self.assertEqual(annotator.boxes[0], [(30, 30, 370, 170)])
                self.assertNotIn(0, annotator.predicted_classes)
                self.assertIn(1, annotator.predicted_classes)
            finally:
                source.close()

    def test_button_boxes_round_trip_as_multiple_yolo_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jpg"
            cv2.imwrite(str(source_path), np.zeros((200, 400, 3), dtype=np.uint8))
            source = FrameSource(source_path, image_paths=[source_path])
            output = root / "dataset"
            try:
                annotator = GeometryAnnotator(
                    source, output, 0, classes=BUTTON_CLASSES, repeated_class=0,
                )
                annotator.boxes = {
                    0: [
                        (40 + index * 40, 155, 65 + index * 40, 185)
                        for index in range(8)
                    ],
                    1: [(80, 50, 320, 150)],
                }
                annotator.save()

                labels_path = output / "labels" / "train" / "frame-000001.txt"
                lines = labels_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(sum(line.startswith("0 ") for line in lines), 8)
                dataset_yaml = (output / "dataset.yaml").read_text(encoding="utf-8")
                self.assertIn("nc: 2", dataset_yaml)
                self.assertIn("button", dataset_yaml)

                reloaded = GeometryAnnotator(
                    source, output, 0, classes=BUTTON_CLASSES, repeated_class=0,
                )
                self.assertEqual(len(reloaded.boxes[0]), 8)
                self.assertEqual(reloaded.boxes[0][0], (40, 155, 65, 185))
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
            ("outer_buttons", 0.80, (20, 10, 380, 290)),
            ("inner_screen", 0.90, (80, 40, 320, 230)),
            ("button", 0.99, (100, 210, 140, 250)),
        ])
        self.assertEqual(outer, (20, 10, 380, 290))
        self.assertEqual(inner, (80, 40, 320, 230))


if __name__ == "__main__":
    unittest.main()
