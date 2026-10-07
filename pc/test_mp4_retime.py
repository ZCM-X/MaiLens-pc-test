"""Tests for the recorded-movie timescale patch."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pc.mp4_retime import load_rows, measured_fps, retime_video, scale_timescale


def write_clip(path: Path, frames: int = 60, fps: float = 30.0, size=(64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("测试环境无法写 mp4")
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def write_log(path: Path, frames: int, rate: float, start: float = 100.0) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(frames):
            handle.write(json.dumps({"timestamp": start + index / rate,
                                     "frame_id": index + 1}) + "\n")


def probe(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    decoded = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    return count, fps, decoded


class Mp4RetimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_stamps_the_real_capture_rate(self) -> None:
        clip = self.root / "raw.mp4"
        log = self.root / "capture.jsonl"
        write_clip(clip, frames=60, fps=60.0)
        write_log(log, frames=60, rate=15.0)
        before_count, before_fps, before_decoded = probe(clip)
        self.assertEqual(before_decoded, 60)
        self.assertAlmostEqual(before_count / before_fps, 1.0, delta=0.1)

        report = retime_video(clip, log, 60.0)

        self.assertEqual(report["status"], "retimed")
        self.assertAlmostEqual(report["measured_fps"], 15.0, delta=0.05)
        count, fps, decoded = probe(clip)
        self.assertEqual(count, before_count)
        self.assertEqual(decoded, 60, "重定时不能丢掉任何一帧")
        self.assertAlmostEqual(count / fps, 4.0, delta=0.1)

    def test_a_clip_at_its_nominal_rate_is_left_alone(self) -> None:
        clip = self.root / "raw.mp4"
        log = self.root / "capture.jsonl"
        write_clip(clip, frames=60, fps=60.0)
        write_log(log, frames=60, rate=60.0)
        before = clip.read_bytes()
        report = retime_video(clip, log, 60.0)
        self.assertEqual(report["status"], "already-accurate")
        self.assertEqual(clip.read_bytes(), before)

    def test_shorter_clip_still_decodes_fully(self) -> None:
        clip = self.root / "raw.mp4"
        log = self.root / "capture.jsonl"
        write_clip(clip, frames=40, fps=60.0)
        write_log(log, frames=40, rate=24.0)
        report = retime_video(clip, log, 60.0)
        self.assertEqual(report["status"], "retimed")
        _count, _fps, decoded = probe(clip)
        self.assertEqual(decoded, 40)

    def test_missing_video_and_unusable_log_are_reported(self) -> None:
        log = self.root / "capture.jsonl"
        write_log(log, frames=60, rate=20.0)
        self.assertEqual(retime_video(self.root / "nope.mp4", log, 60.0)["status"], "missing")

        clip = self.root / "raw.mp4"
        write_clip(clip, frames=60, fps=60.0)
        empty = self.root / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        self.assertEqual(retime_video(clip, empty, 60.0)["status"], "no-timestamps")

    def test_a_corrupt_file_is_left_untouched(self) -> None:
        clip = self.root / "broken.mp4"
        clip.write_bytes(b"definitely not an mp4" * 4)
        before = clip.read_bytes()
        with self.assertRaises(ValueError):
            scale_timescale(clip, 2.0)
        self.assertEqual(clip.read_bytes(), before)
        self.assertFalse((self.root / "broken.mp4.retime").exists())

    def test_a_factor_of_one_reports_without_rewriting(self) -> None:
        clip = self.root / "raw.mp4"
        write_clip(clip, frames=30, fps=30.0)
        before = clip.read_bytes()
        seconds = scale_timescale(clip, 1.0)
        self.assertAlmostEqual(seconds, 1.0, delta=0.05)
        self.assertEqual(clip.read_bytes(), before)

    def test_the_log_reader_skips_damaged_lines(self) -> None:
        log = self.root / "capture.jsonl"
        write_log(log, frames=10, rate=30.0)
        with open(log, "a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        rows = load_rows(log)
        self.assertEqual(len(rows), 10)
        self.assertAlmostEqual(measured_fps(log), 30.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
