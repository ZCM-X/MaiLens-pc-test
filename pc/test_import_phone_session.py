import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from .import_phone_session import import_session
from .process_session import process


def write_phone_session(directory: Path,
                        frames: int = 12,
                        width: int = 96,
                        height: int = 64,
                        logged: int | None = None,
                        fps: float = 30.0) -> Path:
    """Build the folder layout the iPhone app writes after a take."""
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the test VideoWriter")
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (30 + index * 5, 60, 90)
        cv2.rectangle(frame, (8, 8), (width - 8, height - 8), (200, 200, 200), 2)
        writer.write(frame)
    writer.release()

    rows = []
    for index in range(logged if logged is not None else frames):
        timestamp = 1000.0 + index / fps
        rows.append({
            "frame_id": index + 1,
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "pose": {
                "timestamp": timestamp + 0.001,
                "quaternion": {"x": 0.01 * index, "y": 0.0, "z": 0.0, "w": 1.0},
                "gravity": {"x": 0.0, "y": 0.0, "z": -1.0},
                "rotation_rate": {"x": 0.0, "y": 0.0, "z": 0.0},
                "user_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "sensor_delta_ms": 1.0,
        })
    with (directory / "capture.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    with (directory / "pose.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(frames * 4):
            handle.write(json.dumps({
                "timestamp": 1000.0 + index / (fps * 4),
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "gravity": {"x": 0.0, "y": 0.0, "z": -1.0},
                "rotation_rate": {"x": 0.0, "y": 0.0, "z": 0.0},
                "user_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            }) + "\n")

    (directory / "session.json").write_text(json.dumps({
        "source": "MaiLensRemoteCapture",
        "version": 1,
        "created": "2026-10-02T12:00:00Z",
        "clock": "mach_absolute_time",
        "wall_clock_start": 1000.0,
        "video": "video.mp4",
        "metadata": "capture.jsonl",
        "pose_log": "pose.jsonl",
        "width": width,
        "height": height,
        "nominal_fps": 60,
        "frame_count": frames,
        "pose_sample_count": frames * 4,
        "duration": frames / fps,
        "dropped_frames": 0,
        "wall_clock_duration_s": frames / fps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


class ImportPhoneSessionTests(unittest.TestCase):
    def test_import_extracts_every_frame_and_keeps_the_pose_log(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            phone = write_phone_session(root / "phone" / "20261002-120000")
            output = import_session(phone, output=root / "pc-session", log=lambda *_: None)

            frames = sorted((output / "frames").glob("*.jpg"))
            self.assertEqual(len(frames), 12)
            self.assertTrue((output / "phone.mp4").exists())
            self.assertTrue((output / "pose.jsonl").exists())
            self.assertEqual(len((output / "pose.jsonl").read_text(encoding="utf-8").splitlines()), 48)

            rows = [json.loads(line) for line in
                    (output / "capture.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 12)
            for index, row in enumerate(rows):
                self.assertEqual(row["frame_id"], index + 1)
                self.assertEqual(row["decoded_index"], index)
                self.assertEqual(row["frame_path"], f"frames/{index + 1:08d}.jpg")
                self.assertEqual((row["width"], row["height"]), (96, 64))
                self.assertIsNotNone(row.get("pose"))
                self.assertTrue((output / row["frame_path"]).exists())

            manifest = json.loads((output / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_count"], 12)
            # The phone knows the camera target rate; the container is not trusted.
            self.assertEqual(manifest["nominal_fps"], 60)
            self.assertEqual(manifest["source_frame_count"], 12)

    def test_import_keeps_the_video_length_when_the_log_is_short(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            phone = write_phone_session(root / "phone" / "short-log", frames=10, logged=4)
            output = import_session(phone, output=root / "pc-session", log=lambda *_: None)

            rows = [json.loads(line) for line in
                    (output / "capture.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 10)
            self.assertIsNotNone(rows[0].get("pose"))
            self.assertIsNone(rows[5].get("pose"))
            self.assertEqual(rows[9]["frame_id"], 10)

    def test_imported_session_runs_through_the_offline_processor(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            phone = write_phone_session(root / "phone" / "20261002-130000")
            output = import_session(phone, output=root / "pc-session", log=lambda *_: None)

            processed = process(Namespace(
                session=output,
                output=output / "processed.mp4",
                model=None,
                crop=0.74,
                fov=106.4583,
                center_x=0.501753869,
                center_y=0.499423644,
                k1=0.0893163,
                k2=-0.0174637,
                fps=None,
                debug=False,
                preview=False,
                detect_every=12,
                lock_fill=0.64,
            ))

            self.assertTrue(processed.exists())
            debug_lines = [line for line in
                           (output / "debug.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(debug_lines), 12)
            first = json.loads(debug_lines[0])
            self.assertIn("lock_source", first)


if __name__ == "__main__":
    unittest.main()
