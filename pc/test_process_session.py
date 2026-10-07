import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .process_session import (
    GeometryDetector,
    GeometryLockTracker,
    PlaneCornersSmoother,
    PlaneLockTracker,
    frame_gain,
    apply_geometry_lock,
    apply_plane_lock,
    build_remap,
    geometry_margins,
    geometry_reference,
    load_pose_track,
    map_fisheye_points_to_output,
    parse_args,
    plausible_geometry_pair,
    pose_for_row,
    _project_points,
    process,
    quat_to_matrix,
    resolve_fps,
    rotation_for_row,
    update_geometry_lock_state,
)
from .test_import_phone_session import write_phone_session


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

    def test_stale_plane_lock_reacquires_and_recenters_from_fresh_detection(self):
        width, height = 320, 240
        frame = np.zeros((height, width), dtype=np.uint8)
        for y in range(40, 201, 12):
            for x in range(60, 261, 12):
                cv2.circle(frame, (x, y), 2, 160 + (x + y) % 90, -1)
        original_box = (80, 65, 240, 175)
        fresh_box = (90, 70, 250, 180)
        outer = (45, 30, 275, 210)
        tracker = PlaneLockTracker(detect_every=12)
        self.assertTrue(tracker.initialize(frame, original_box, outer, width, height, 0.64))
        tracker.current_to_reference = np.array(
            [[1.0, 0.0, 24.0], [0.0, 1.0, -13.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tracker.age_frames = 7  # beyond output_homography's grace period
        self.assertTrue(tracker.reacquire_if_stale(
            frame, fresh_box, outer, width, height, 0.64, fresh_detection=True,
        ))
        self.assertEqual(tracker.reference_box, fresh_box)
        self.assertEqual(tracker.age_frames, 0)
        np.testing.assert_allclose(tracker.current_to_reference, np.eye(3), atol=1e-6)

    def test_stale_plane_lock_ignores_unaccepted_detector_result(self):
        width, height = 320, 240
        frame = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(frame, (60, 40), (260, 200), 180, 3)
        box = (80, 65, 240, 175)
        outer = (45, 30, 275, 210)
        tracker = PlaneLockTracker(detect_every=12)
        self.assertTrue(tracker.initialize(frame, box, outer, width, height, 0.64))
        tracker.age_frames = 7
        self.assertFalse(tracker.reacquire_if_stale(
            frame, box, outer, width, height, 0.64, fresh_detection=False,
        ))
        self.assertEqual(tracker.age_frames, 7)

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

    def test_frame_gain_leaves_the_reference_rate_alone(self):
        self.assertAlmostEqual(frame_gain(0.35, None), 0.35)
        self.assertAlmostEqual(frame_gain(0.35, 1.0 / 30.0), 0.35, places=6)

    def test_frame_gain_keeps_the_time_constant_across_rates(self):
        # Two 60 fps steps have to land where one 30 fps step does, otherwise
        # the phone feed passes twice the jitter the render shows.
        fast = frame_gain(0.35, 1.0 / 60.0)
        self.assertLess(fast, 0.35)
        self.assertAlmostEqual((1.0 - fast) ** 2, 0.65, places=3)
        self.assertGreater(frame_gain(0.35, 1.0 / 15.0), 0.35)

    def test_frame_gain_keeps_the_extremes(self):
        self.assertEqual(frame_gain(1.0, 1.0 / 60.0), 1.0)
        self.assertEqual(frame_gain(0.0, 1.0 / 60.0), 0.0)
        self.assertEqual(frame_gain(0.35, 0.0), 0.35)

    def test_the_first_quad_is_adopted_whole(self):
        smoother = PlaneCornersSmoother(0.35)
        quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        np.testing.assert_allclose(smoother.update(quad), quad)

    def test_a_quad_jump_is_blended_instead_of_applied(self):
        smoother = PlaneCornersSmoother(0.35)
        smoother.update(np.zeros((4, 2)))
        np.testing.assert_allclose(smoother.update(np.full((4, 2), 100.0)), 35.0)

    def test_two_fast_quads_match_one_reference_quad(self):
        fast = PlaneCornersSmoother(0.35)
        slow = PlaneCornersSmoother(0.35)
        fast.update(np.zeros((4, 2)))
        slow.update(np.zeros((4, 2)))
        quad = np.full((4, 2), 100.0)
        fast.update(quad, dt=1.0 / 60.0)
        fast.update(quad, dt=1.0 / 60.0)
        slow.update(quad, dt=1.0 / 30.0)
        np.testing.assert_allclose(fast.corners, slow.corners, atol=0.6)

    def test_resetting_the_smoother_forgets_the_old_quad(self):
        smoother = PlaneCornersSmoother(0.35)
        smoother.update(np.full((4, 2), 100.0))
        smoother.reset()
        zeros = np.zeros((4, 2))
        np.testing.assert_allclose(smoother.update(zeros), zeros)

    def test_ingest_blends_a_new_box_by_the_frame_interval(self):
        fast = GeometryLockTracker(detect_every=12)
        slow = GeometryLockTracker(detect_every=12)
        for tracker in (fast, slow):
            tracker.ingest((100, 100, 500, 500), None, 720, 1280)
        candidate = (120, 120, 520, 520)
        fast.ingest(candidate, None, 720, 1280, dt=1.0 / 60.0)
        slow.ingest(candidate, None, 720, 1280, dt=1.0 / 30.0)
        self.assertGreater(fast.outer_box[0], 100)
        self.assertGreater(slow.outer_box[0], fast.outer_box[0])

    def test_full_warp_puts_every_pixel_on_one_viewpoint(self):
        # A uniform translation must move the whole picture: the machine and
        # the room around it share one viewpoint, which is what stops the
        # cabinet from reading as an oval pasted onto a live background.
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[40:80, 60:100] = 255
        matrix = np.float32([[1, 0, 6], [0, 1, 4], [0, 0, 1]])
        locked = apply_plane_lock(frame, matrix, (60, 40, 100, 80), full_warp=True)
        shifted = cv2.warpAffine(frame, matrix[:2], (160, 120))
        np.testing.assert_allclose(locked, shifted, atol=1)

    def test_full_warp_keeps_the_rectified_view_where_the_warp_runs_out(self):
        # Sampling past the source border must not smear an edge inward: those
        # pixels fall back to the plain rectified frame instead.
        frame = np.random.RandomState(4).randint(0, 256, (120, 160, 3)).astype(np.uint8)
        matrix = np.float32([[1, 0, 40], [0, 1, 0], [0, 0, 1]])
        locked = apply_plane_lock(frame, matrix, (60, 40, 100, 80), full_warp=True)
        self.assertFalse(np.array_equal(locked[:, :40], frame[:, :40]))
        np.testing.assert_allclose(locked[:, :24], frame[:, :24], atol=2)

    def test_the_old_patch_path_still_leaves_the_far_background_alone(self):
        frame = np.random.RandomState(7).randint(0, 256, (120, 160, 3)).astype(np.uint8)
        matrix = np.float32([[1, 0, 20], [0, 1, 0], [0, 0, 1]])
        patched = apply_plane_lock(frame, matrix, (60, 40, 100, 80), full_warp=False)
        np.testing.assert_allclose(patched[:, 150:], frame[:, 150:], atol=2)

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

    def test_measured_stamps_beat_a_nominal_fps_of_60(self):
        # The real bug: the receiver writes nominal_fps=60 from its --fps
        # default while the phone pushes ~25 fps, which tagged the output at
        # 60 and made every replay run about 2.4x too fast.
        with tempfile.TemporaryDirectory() as workspace:
            session = Path(workspace)
            (session / "session.json").write_text(json.dumps({"nominal_fps": 60}), encoding="utf-8")
            rows = [{"timestamp": 500.0 + index / 25.0} for index in range(50)]
            self.assertAlmostEqual(resolve_fps(session, rows, None), 25.0, places=3)

    def test_dropped_frames_do_not_speed_up_the_output(self):
        # Every fifth frame never reached the PC.  The clip still spans the
        # same wall-clock time, so tagging it at the nominal 30 fps played
        # everything 1.2x fast and turned each gap into a jump.
        rows = []
        stamp = 700.0
        for index in range(100):
            rows.append({"timestamp": stamp})
            stamp += 1.0 / 30.0 if index % 5 != 4 else 2.0 / 30.0
        with tempfile.TemporaryDirectory() as workspace:
            fps = resolve_fps(Path(workspace), rows, None)
        self.assertAlmostEqual(fps, 99 * 30.0 / 118.0, places=2)

    def test_a_capture_stall_does_not_stretch_the_whole_clip(self):
        # A ten second freeze is a capture failure, not a frame interval.
        rows = [{"timestamp": index / 30.0} for index in range(30)]
        rows.append({"timestamp": 400.0})
        with tempfile.TemporaryDirectory() as workspace:
            fps = resolve_fps(Path(workspace), rows, None)
        self.assertAlmostEqual(fps, 30.0, places=3)

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


class VideoBackedSessionTests(unittest.TestCase):
    def test_processes_a_phone_session_straight_from_its_movie(self):
        with tempfile.TemporaryDirectory() as workspace:
            session = write_phone_session(Path(workspace) / "20261002-150000")
            # A phone session has no frames/ directory at all.
            self.assertFalse((session / "frames").exists())

            output = process(parse_args([str(session)]))

            self.assertEqual(output, session / "processed.mp4")
            self.assertGreater(output.stat().st_size, 0)
            debug = [json.loads(line) for line in
                     (session / "debug.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(debug), 12)
            self.assertIn("lock_source", debug[0])

    def test_movie_keeps_going_when_the_frame_log_stops_early(self):
        with tempfile.TemporaryDirectory() as workspace:
            session = write_phone_session(Path(workspace) / "short-log", frames=12, logged=5)
            output = process(parse_args([str(session)]))

            self.assertGreater(output.stat().st_size, 0)
            debug = [json.loads(line) for line in
                     (session / "debug.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(debug), 12)

    def test_the_render_reports_the_lock_travel_it_is_allowed(self):
        with tempfile.TemporaryDirectory() as workspace:
            session = write_phone_session(Path(workspace) / "authority", frames=6)
            process(parse_args([str(session), "--max-frames", "4"]))

            debug = [json.loads(line) for line in
                     (session / "debug.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(debug), 4)
            for row in debug:
                self.assertIn("lock_mode", row)
                self.assertIn("lock_travel_px", row)
                self.assertIn("lock_travel_limit_px", row)
            # Nobody is locked in a session without a model, and the travel
            # limit is still the one the operator asked for: 0.30 of the
            # short side of the 96x64 synthetic frames.
            self.assertEqual(debug[-1]["lock_mode"], "geometry")
            self.assertEqual(debug[-1]["lock_travel_px"], 0.0)
            self.assertEqual(debug[-1]["lock_travel_limit_px"], 0.30 * 64)

            process(parse_args([str(session), "--max-frames", "4", "--lock-shift", "0.5"]))
            debug = [json.loads(line) for line in
                     (session / "debug.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(debug[-1]["lock_travel_limit_px"], 0.5 * 64)

    def test_no_fisheye_and_max_frames_lock_the_raw_geometry(self):
        # An already-rectified clip must not go through the fisheye model again.
        with tempfile.TemporaryDirectory() as workspace:
            session = write_phone_session(Path(workspace) / "rectified")
            output = process(parse_args([str(session), "--no-fisheye", "--max-frames", "4"]))

            self.assertTrue(output.exists())
            debug = [json.loads(line) for line in
                     (session / "debug.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(debug), 4)


class PlaneSanityTests(unittest.TestCase):
    def test_wild_plane_quads_are_rejected_before_they_reach_the_preview(self):
        tracker = PlaneLockTracker(detect_every=3)
        tracker.reference_box = (100, 100, 300, 300)
        tracker.current_quad = np.float32([[100, 100], [300, 100], [300, 300], [100, 300]])

        # A small, coherent move stays.
        self.assertTrue(tracker._candidate_ok(
            np.float32([[104, 102], [303, 99], [302, 304], [101, 302]])
        ))
        # A hand across the machine produces a leaning plank: opposite edges
        # stop matching, so the plane must not be accepted.
        self.assertFalse(tracker._candidate_ok(
            np.float32([[100, 100], [300, 100], [240, 300], [160, 300]])
        ))
        # Bow-tie quads (self-intersecting) are garbage as well.
        self.assertFalse(tracker._candidate_ok(
            np.float32([[100, 100], [300, 300], [300, 100], [100, 300]])
        ))
        # A sudden jump larger than half the reference diagonal is rejected.
        self.assertFalse(tracker._candidate_ok(
            np.float32([[500, 500], [700, 500], [700, 700], [500, 700]])
        ))
        # Absurd scale changes are rejected too.
        self.assertFalse(tracker._candidate_ok(
            np.float32([[0, 0], [900, 0], [900, 900], [0, 900]])
        ))


class PoseLogTests(unittest.TestCase):
    def test_pose_log_covers_rows_that_carry_no_pose(self):
        with tempfile.TemporaryDirectory() as workspace:
            session = Path(workspace)
            samples = [
                {"timestamp": 101.0, "quaternion": {"x": 0.2, "y": 0, "z": 0, "w": 0.98}},
                {"timestamp": 100.0, "quaternion": {"x": 0.0, "y": 0, "z": 0, "w": 1.0}},
                {"timestamp": 100.5, "quaternion": {"x": 0.1, "y": 0, "z": 0, "w": 0.99}},
            ]
            with (session / "pose.jsonl").open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample) + "\n")
                handle.write("{broken\n")

            track = load_pose_track(session)
            self.assertEqual([stamp for stamp, _ in track], [100.0, 100.5, 101.0])

            row = {"frame_id": 2, "timestamp": 100.52}
            merged = pose_for_row(row, track)
            self.assertAlmostEqual(merged["pose"]["quaternion"]["x"], 0.1)
            self.assertNotIn("pose", row)

            existing = {"frame_id": 1, "timestamp": 100.0, "pose": {"quaternion": {"x": 9.0}}}
            self.assertEqual(pose_for_row(existing, track)["pose"]["quaternion"]["x"], 9.0)


if __name__ == "__main__":
    unittest.main()

class PlaneDriftAnchorTests(unittest.TestCase):
    """The plane transform is accumulated, so it needs an absolute anchor."""

    @staticmethod
    def _tracker(scale: float) -> PlaneLockTracker:
        tracker = PlaneLockTracker(detect_every=12)
        tracker.reference_box = (100, 200, 300, 500)
        tracker.reference_outer_box = (60, 150, 340, 560)
        tracker.reference_to_output = np.eye(3, dtype=np.float32)
        tracker.reference_gray = np.zeros((720, 1280), dtype=np.uint8)
        tracker.points = np.zeros((12, 1, 2), dtype=np.float32)
        tracker.current_to_reference = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return tracker

    @staticmethod
    def _effective(tracker: PlaneLockTracker) -> np.ndarray:
        """The transform that actually lands on the output."""
        return tracker.correction @ tracker.current_to_reference

    def _settle(self, tracker: PlaneLockTracker, box, rounds: int = 400) -> None:
        for _ in range(rounds):
            tracker.reanchor(box)
            tracker.advance_correction()

    def test_drift_is_pulled_back_towards_the_reference_box(self):
        tracker = self._tracker(1.4)
        # The machine really is where the reference box says it is.
        box = (100, 200, 300, 500)
        self.assertTrue(tracker.reanchor(box))
        self._settle(tracker, box)
        self.assertAlmostEqual(float(self._effective(tracker)[0, 0]), 1.0, delta=0.05)

    def test_a_single_reading_never_moves_the_plane_in_one_frame(self):
        tracker = self._tracker(1.4)
        box = (100, 200, 300, 500)
        self.assertTrue(tracker.reanchor(box))
        before = self._effective(tracker).copy()
        tracker.advance_correction()
        after = self._effective(tracker)
        # 35% of a correction that is itself capped well below the whole
        # movement: the visible plane can never be yanked by one reading.
        self.assertLess(abs(float(after[0, 0]) - float(before[0, 0])), 0.16)

    def test_the_payout_is_spread_over_several_frames(self):
        tracker = self._tracker(1.4)
        box = (100, 200, 300, 500)
        tracker.reanchor(box)
        steps = []
        for _ in range(6):
            tracker.advance_correction()
            steps.append(float(self._effective(tracker)[0, 0]))
        self.assertEqual(len(set(round(value, 6) for value in steps)), len(steps))
        self.assertGreater(steps[0] - steps[1], steps[-2] - steps[-1])

    def test_a_single_inflated_box_is_ignored(self):
        tracker = self._tracker(1.0)
        before = tracker.correction.copy()
        # Fisheye boxes near the rim map to inflated quadrilaterals; that must
        # not be read as the machine suddenly shrinking.
        inflated = (0, 0, 1280, 1440)
        self.assertFalse(tracker.reanchor(inflated))
        np.testing.assert_allclose(tracker.correction, before)

    def test_front_back_motion_is_reported_by_the_correction(self):
        tracker = self._tracker(1.0)
        # Twice the reference box means the machine is twice as close as when
        # it was locked, so the transform has to shrink the plane back down.
        self.assertTrue(tracker.reanchor((10, 140, 290, 560)))
        self._settle(tracker, (10, 140, 290, 560))
        self.assertLess(float(self._effective(tracker)[0, 0]), 1.0)

    def test_a_new_reference_drops_the_old_correction(self):
        tracker = self._tracker(1.4)
        tracker.reanchor((100, 200, 300, 500))
        tracker.advance_correction()
        self.assertFalse(np.allclose(tracker.correction, np.eye(3)))
        gray = np.zeros((720, 1280), dtype=np.uint8)
        tracker.initialize(gray, (100, 200, 300, 500), (60, 150, 340, 560), 1280, 720, 0.71)
        np.testing.assert_allclose(tracker.correction, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(tracker.current_to_reference, np.eye(3), atol=1e-6)


