import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .process_session import (
    GeometryDetector,
    GeometryLockTracker,
    PlaneLockTracker,
    apply_geometry_lock,
    build_remap,
    geometry_margins,
    geometry_reference,
    map_fisheye_points_to_output,
    plausible_geometry_pair,
    _project_points,
    process,
    quat_to_matrix,
    resolve_fps,
    rotation_for_row,
    update_geometry_lock_state,
)


class RotationMappingTests(unittest.TestCase):
    def test_relative_rotation_maps_current_camera_back_to_latched_camera(self):
        reference = quat_to_matrix({"x": 0, "y": 0, "z": 0, "w": 1})
        row = {"pose": {"quaternion": {
            "x": 0,
            "y": 0,
            "z": 0.3826834324,
            "w": 0.9238795325,
        }}}
        relative, returned_reference = rotation_for_row(row, reference)
        current = quat_to_matrix(row["pose"]["quaternion"])
        camera_to_device = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        np.testing.assert_allclose(returned_reference, reference)
        np.testing.assert_allclose(
            relative,
            camera_to_device @ current.T @ reference @ camera_to_device,
            atol=1e-6,
        )


class ProcessSessionTests(unittest.TestCase):
    def test_plane_lock_keeps_perspective_screen_in_same_output_position(self):
        width, height = 640, 480
        first = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(first, (120, 80), (520, 400), 80, -1)
        for y in range(100, 400, 20):
            for x in range(140, 520, 20):
                cv2.circle(first, (x, y), 3, 150 + (x + y) % 90, -1)
        source = np.float32([[120, 80], [520, 80], [520, 400], [120, 400]])
        moved = np.float32([[80, 60], [560, 90], [530, 430], [100, 400]])
        motion = cv2.getPerspectiveTransform(source, moved)
        second = cv2.warpPerspective(first, motion, (width, height), borderMode=cv2.BORDER_REFLECT101)

        tracker = PlaneLockTracker(detect_every=3)
        self.assertTrue(tracker.initialize(first, (120, 80, 520, 400), (90, 50, 550, 430), width, height, 0.64))
        self.assertTrue(tracker.update(second, (80, 60, 560, 430), (50, 40, 590, 450), width, height, 0.64))
        projected = _project_points(moved, tracker.output_homography)
        expected = _project_points(source, tracker.reference_to_output)
        self.assertIsNotNone(projected)
        self.assertIsNotNone(expected)
        np.testing.assert_allclose(projected, expected, atol=4.0)
        self.assertGreaterEqual(tracker.inliers, 8)
        self.assertGreaterEqual(tracker.inlier_ratio, 0.52)

    def test_plane_lock_holds_reference_when_flow_quality_fails(self):
        width, height = 320, 240
        first = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(first, (60, 40), (260, 200), 180, 3)
        tracker = PlaneLockTracker(detect_every=2, max_age_frames=3)
        self.assertTrue(tracker.initialize(first, (60, 40, 260, 200), (45, 25, 275, 215), width, height, 0.64))
        before = tracker.output_homography.copy()
        blank = np.zeros_like(first)
        self.assertTrue(tracker.update(blank, (60, 40, 260, 200), (45, 25, 275, 215), width, height, 0.64))
        np.testing.assert_allclose(tracker.output_homography, before, atol=1e-6)
        self.assertFalse(tracker.last_success)

    def test_geometry_margins_are_derived_from_two_boxes(self):
        self.assertEqual(
            geometry_margins((10, 20, 390, 280), (85, 95, 315, 205)),
            {"left": 75, "top": 75, "right": 75, "bottom": 75},
        )

    def test_raw_fisheye_box_mapping_round_trips_the_remap_grid(self):
        width, height = 640, 360
        angle = np.deg2rad(12.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        map_x, map_y = build_remap(
            width, height, rotation, 0.74, 106.4583,
            0.0893163, -0.0174637, 0.501753869, 0.499423644,
        )
        output_point = np.array([width * 0.58, height * 0.44], dtype=np.float32)
        raw_point = np.array([[map_x[int(output_point[1]), int(output_point[0])],
                               map_y[int(output_point[1]), int(output_point[0])]]], dtype=np.float32)
        mapped, valid = map_fisheye_points_to_output(
            raw_point, width, height, rotation, 0.74, 106.4583,
            0.0893163, -0.0174637, 0.501753869, 0.499423644,
        )
        self.assertTrue(bool(valid[0]))
        np.testing.assert_allclose(mapped[0], output_point, atol=2.0)

    def test_geometry_tracker_rejects_a_far_detector_jump(self):
        tracker = GeometryLockTracker(detect_every=3)
        tracker.ingest((80, 60, 280, 220), None, 400, 300)
        tracker.ingest((300, 10, 395, 100), None, 400, 300)
        outer, inner = tracker.boxes()
        self.assertEqual(outer, (80, 60, 280, 220))
        self.assertIsNone(inner)

    def test_geometry_tracker_keeps_outer_and_inner_boxes_when_inner_detector_misses(self):
        tracker = GeometryLockTracker(detect_every=3)
        tracker.ingest((40, 30, 360, 270), (100, 80, 300, 220), 400, 300)
        tracker.ingest((45, 35, 355, 265), None, 400, 300)
        outer, inner = tracker.boxes()
        self.assertIsNotNone(outer)
        self.assertIsNotNone(inner)

    def test_geometry_tracker_follows_translation_between_detector_frames(self):
        width, height = 320, 240
        previous = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(previous, (70, 60), (240, 190), 255, 3)
        for x in range(80, 230, 15):
            cv2.circle(previous, (x, 100 + (x % 30)), 3, 180, -1)
        current = cv2.warpAffine(
            previous, np.float32([[1, 0, 8], [0, 1, 4]]), (width, height),
            borderMode=cv2.BORDER_REFLECT101,
        )
        tracker = GeometryLockTracker(detect_every=3)
        tracker.ingest((70, 60, 240, 190), None, width, height)
        tracker.update_flow(previous, current)
        self.assertIsNotNone(tracker.box)
        self.assertGreaterEqual(tracker.box[0], 74)
        self.assertGreaterEqual(tracker.box[1], 62)

    def test_geometry_transform_maps_target_center_to_output_center(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        target = np.array([0.28, 0.62], dtype=np.float32)
        _locked, matrix = apply_geometry_lock(frame, target, 1.2)
        transformed = cv2.transform(
            np.float32([[[target[0] * frame.shape[1], target[1] * frame.shape[0]]]]),
            matrix,
        )[0, 0]
        np.testing.assert_allclose(transformed, [200.0, 150.0], atol=1e-4)

    def test_geometry_detector_selects_labeled_outer_and_inner_boxes(self):
        outer, inner = GeometryDetector._pick_geometry_boxes([
            ("outer_buttons", 0.90, (20, 20, 380, 280)),
            ("inner_screen", 0.88, (80, 60, 320, 230)),
            ("inner_screen", 0.40, (0, 0, 30, 30)),
        ])
        self.assertEqual(outer, (20, 20, 380, 280))
        self.assertEqual(inner, (80, 60, 320, 230))

    def test_geometry_detector_rejects_overlapping_duplicate_boxes(self):
        outer, inner = GeometryDetector._pick_geometry_boxes([
            ("outer_buttons", 0.94, (20, 20, 380, 280)),
            ("inner_screen", 0.96, (20, 20, 380, 280)),
        ])
        self.assertEqual(outer, (20, 20, 380, 280))
        self.assertIsNone(inner)
        self.assertFalse(plausible_geometry_pair(outer, inner))

    def test_inner_screen_only_never_becomes_outer_anchor(self):
        outer, inner = GeometryDetector._pick_geometry_boxes([
            ("inner_screen", 0.96, (80, 60, 320, 230)),
        ])
        self.assertIsNone(outer)
        self.assertIsNone(inner)

    def test_geometry_lock_uses_outer_frame_when_inner_screen_is_missing(self):
        center, zoom, source = update_geometry_lock_state(
            np.array([0.5, 0.5], dtype=np.float32),
            1.0,
            (40, 40, 240, 240),
            None,
            400,
            300,
        )
        self.assertEqual(source, "outer_buttons")
        self.assertLess(float(center[0]), 0.5)
        self.assertGreater(zoom, 0.70)

    def test_geometry_lock_uses_inner_center_and_inner_size(self):
        center, _zoom, source = update_geometry_lock_state(
            np.array([0.5, 0.5], dtype=np.float32),
            1.0,
            (40, 40, 240, 240),
            (180, 70, 280, 170),
            400,
            300,
        )
        self.assertEqual(source, "inner_screen")
        np.testing.assert_allclose(center, [0.575, 0.4], atol=1e-6)

    def test_geometry_tracker_rejects_far_outer_jump_but_keeps_inner_box(self):
        tracker = GeometryLockTracker(detect_every=3)
        tracker.ingest((40, 30, 360, 270), (100, 80, 300, 220), 400, 300)
        tracker.ingest((300, 10, 395, 100), (105, 84, 305, 224), 400, 300)
        outer, inner = tracker.boxes()
        self.assertEqual(outer, (40, 30, 360, 270))
        self.assertIsNotNone(inner)

    def test_geometry_lock_keeps_state_but_does_not_report_a_stale_box(self):
        previous = np.array([0.42, 0.56], dtype=np.float32)
        center, zoom, source = update_geometry_lock_state(
            previous,
            1.08,
            None,
            None,
            400,
            300,
        )
        np.testing.assert_array_equal(center, previous)
        self.assertEqual(zoom, 1.08)
        self.assertEqual(source, "none")

    def test_geometry_lock_compensates_front_back_motion_in_opposite_direction(self):
        width, height = 400, 300
        outer = (40, 30, 360, 270)
        initial_inner = (120, 90, 280, 210)
        reference_size, reference_zoom = geometry_reference(initial_inner, width, height, 0.64)
        center, first_zoom, source = update_geometry_lock_state(
            np.array([0.5, 0.5], dtype=np.float32),
            1.0,
            outer,
            initial_inner,
            width,
            height,
            0.64,
            snap=True,
            reference_target_size=reference_size,
            reference_zoom=reference_zoom,
        )
        self.assertEqual(source, "inner_screen")
        closer_inner = (80, 60, 320, 240)
        _center, closer_zoom, _source = update_geometry_lock_state(
            center, first_zoom, outer, closer_inner, width, height, 0.64,
            reference_target_size=reference_size, reference_zoom=reference_zoom,
        )
        farther_inner = (150, 112, 250, 188)
        _center, farther_zoom, _source = update_geometry_lock_state(
            center, first_zoom, outer, farther_inner, width, height, 0.64,
            reference_target_size=reference_size, reference_zoom=reference_zoom,
        )
        self.assertLess(closer_zoom, first_zoom)
        self.assertGreater(farther_zoom, first_zoom)

    def test_pose_only_session_produces_processed_video_and_debug_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            rows = []
            for index in range(4):
                image = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.rectangle(image, (70 + index * 2, 30), (250 + index * 2, 150), (80, 220, 150), 3)
                relative = Path("frames") / f"{index + 1:08d}.jpg"
                cv2.imwrite(str(root / relative), image)
                rows.append({
                    "frame_id": index + 1,
                    "timestamp": index / 15,
                    "frame_path": str(relative).replace("\\", "/"),
                    "pose": {
                        "timestamp": index / 15,
                        "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
                    },
                })
            (root / "capture.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            (root / "session.json").write_text(json.dumps({"nominal_fps": 15}), encoding="utf-8")

            output = process(type("Args", (), {
                "session": root,
                "output": root / "processed.mp4",
                "model": None,
                "crop": 0.74,
                "fov": 106.4583,
                "center_x": 0.501753869,
                "center_y": 0.499423644,
                "k1": 0.0893163,
                "k2": -0.0174637,
                "fps": None,
                "debug": True,
                "preview": False,
                "detect_every": 3,
            })())

            self.assertEqual(output, root / "processed.mp4")
            self.assertGreater((root / "processed.mp4").stat().st_size, 0)
            self.assertEqual(len((root / "debug.jsonl").read_text(encoding="utf-8").splitlines()), 4)


class ResolveFpsTests(unittest.TestCase):
    def test_uses_the_manifest_when_it_exists(self):
        with tempfile.TemporaryDirectory() as workspace:
            session = Path(workspace)
            (session / "session.json").write_text(json.dumps({"nominal_fps": 30}), encoding="utf-8")
            self.assertEqual(resolve_fps(session, [], None), 30.0)

    def test_falls_back_to_frame_timestamps_without_a_manifest(self):
        # Ctrl+C stops the receiver before it writes session.json.
        with tempfile.TemporaryDirectory() as workspace:
            session = Path(workspace)
            rows = [{"timestamp": 100.0 + index / 60.0} for index in range(60)]
            self.assertAlmostEqual(resolve_fps(session, rows, None), 60.0, places=3)

    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as workspace:
            session = Path(workspace)
            (session / "session.json").write_text(json.dumps({"nominal_fps": 30}), encoding="utf-8")
            self.assertEqual(resolve_fps(session, [], 24.0), 24.0)

    def test_survives_a_broken_manifest(self):
        with tempfile.TemporaryDirectory() as workspace:
            session = Path(workspace)
            (session / "session.json").write_text("{not json", encoding="utf-8")
            rows = [{"timestamp": 1.0}, {"timestamp": 2.0}]
            self.assertEqual(resolve_fps(session, rows, None), 1.0)


if __name__ == "__main__":
    unittest.main()
