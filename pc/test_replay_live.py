import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .replay_live import (
    NO_KEY,
    PlaybackClock,
    iter_session_frames,
    load_session_rows,
    replay,
    session_fps,
)


class StubProcessor:
    enabled = True

    def __init__(self):
        self.debug = False
        self.processed = 0
        self.metadata: list[dict] = []

    def process(self, frame, metadata):
        self.processed += 1
        self.metadata.append(metadata)
        return frame, {"lock_source": "test"}


def write_session(root: Path, count: int = 5, fps: float = 10.0, with_frames: bool = True) -> Path:
    session = root / "session"
    (session / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        timestamp = 100.0 + index / fps
        row = {
            "frame_id": index,
            "timestamp": timestamp,
            "pose": {"timestamp": timestamp, "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1}},
        }
        if with_frames:
            name = f"{index:08d}.jpg"
            frame = np.full((24, 32, 3), (index * 37) % 255, dtype=np.uint8)
            cv2.imwrite(str(session / "frames" / name), frame)
            row["frame_path"] = f"frames/{name}"
        rows.append(row)
    (session / "capture.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return session


def write_video(session: Path, count: int = 5) -> None:
    writer = cv2.VideoWriter(
        str(session / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    self_check = writer.isOpened()
    for index in range(count):
        writer.write(np.full((24, 32, 3), (index * 37) % 255, dtype=np.uint8))
    writer.release()
    if not self_check:
        raise RuntimeError("cannot write the test video")


class ReplayLiveTests(unittest.TestCase):
    def test_load_session_rows_orders_by_frame_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp))
            rows = [
                {"frame_id": 2, "timestamp": 2.0},
                {"frame_id": 0, "timestamp": 0.0},
                {"frame_id": 1, "timestamp": 1.0},
            ]
            (session / "capture.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            self.assertEqual([row["frame_id"] for row in load_session_rows(session)], [0, 1, 2])

    def test_session_fps_uses_the_recorded_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=6, fps=25.0)
            self.assertAlmostEqual(session_fps(session, load_session_rows(session)), 25.0, places=2)

    def test_session_fps_ignores_a_nominal_fps_the_stamps_contradict(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=6, fps=25.0)
            (session / "session.json").write_text(json.dumps({"nominal_fps": 60}), encoding="utf-8")
            self.assertAlmostEqual(session_fps(session, load_session_rows(session)), 25.0, places=2)

    def test_session_fps_still_uses_the_manifest_without_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=6, fps=25.0)
            (session / "session.json").write_text(json.dumps({"nominal_fps": 60}), encoding="utf-8")
            rows = [{"frame_id": index} for index in range(6)]
            self.assertEqual(session_fps(session, rows), 60.0)

    def test_iter_session_frames_prefers_the_saved_jpegs(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=4)
            frames = list(iter_session_frames(session, load_session_rows(session)))
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[1][1][0, 0, 0], 37)

    def test_iter_session_frames_falls_back_to_the_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=4, with_frames=False)
            write_video(session, count=4)
            frames = list(iter_session_frames(session, load_session_rows(session)))
            self.assertEqual(len(frames), 4)

    def test_replay_runs_every_frame_through_the_live_processor(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=5)
            processor = StubProcessor()
            summary = replay(
                session,
                processor_factory=lambda: processor,
                preview=False,
                realtime=False,
                read_key=lambda: NO_KEY,
            )
            self.assertEqual(processor.processed, 5)
            self.assertEqual(summary["frames"], 5)
            self.assertEqual(summary["lock_sources"], {"test": 5})
            self.assertEqual([row["frame_id"] for row in processor.metadata], [0, 1, 2, 3, 4])

    def test_replay_max_frames_and_start_trim_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=8, fps=10.0)
            processor = StubProcessor()
            summary = replay(
                session,
                processor_factory=lambda: processor,
                preview=False,
                realtime=False,
                start=0.3,
                max_frames=3,
                read_key=lambda: NO_KEY,
            )
            self.assertEqual(summary["frames"], 3)
            self.assertEqual([row["frame_id"] for row in processor.metadata], [3, 4, 5])

    def test_replay_writes_an_output_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=4)
            out = Path(tmp) / "out.mp4"
            replay(
                session,
                processor_factory=StubProcessor,
                preview=False,
                realtime=False,
                output=out,
                read_key=lambda: NO_KEY,
            )
            self.assertTrue(out.exists())
            capture = cv2.VideoCapture(str(out))
            self.assertGreaterEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            capture.release()

    def test_replay_stays_paused_until_it_is_resumed_or_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), count=4)
            calls = {"count": 0}

            def read_key() -> int:
                calls["count"] += 1
                if calls["count"] == 1:
                    return ord(" ")
                if calls["count"] >= 12:
                    return ord("q")
                return NO_KEY

            processor = StubProcessor()
            summary = replay(
                session,
                processor_factory=lambda: processor,
                preview=False,
                realtime=False,
                read_key=read_key,
            )
            self.assertEqual(summary["frames"], 0)
            self.assertEqual(processor.processed, 0)

    def test_playback_clock_follows_the_recorded_timestamps(self):
        now = {"time": 0.0}
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            now["time"] += seconds

        clock = PlaybackClock(speed=1.0, time_fn=lambda: now["time"], sleep_fn=sleep)
        clock.wait(10.0)
        clock.wait(10.5)
        self.assertAlmostEqual(now["time"], 0.5, places=3)
        clock.pause()
        now["time"] += 2.0
        clock.resume()
        clock.wait(11.0)
        self.assertAlmostEqual(now["time"], 3.0, places=3)

    def test_playback_clock_scales_with_speed(self):
        now = {"time": 0.0}

        def sleep(seconds: float) -> None:
            now["time"] += seconds

        clock = PlaybackClock(speed=2.0, time_fn=lambda: now["time"], sleep_fn=sleep)
        clock.wait(0.0)
        clock.wait(4.0)
        self.assertAlmostEqual(now["time"], 2.0, places=3)


if __name__ == "__main__":
    unittest.main()