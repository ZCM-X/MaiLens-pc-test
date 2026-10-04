#!/usr/bin/env python3
"""Replay a recorded phone session through the live path in real time.

``pc_receiver.py`` only processes frames that arrive from the phone over TCP.
This tool feeds a saved session (``frames/*.jpg`` + ``capture.jsonl``, or
``raw.mp4`` + ``capture.jsonl``) through the same :class:`LiveProcessor`, paced
by the recorded timestamps, so the gimbal and the machine lock can be watched
on the PC before anything is rebuilt for the phone.

Keys in the preview window:

* ``q`` / ``Esc``  stop
* ``Space``        pause / resume
* ``.``            while paused, advance exactly one frame
* ``r``            restart from the first frame
* ``d``            toggle the detection overlay
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np

try:
    from .live_processor import LiveProcessor
except ImportError:  # Running as `python pc/replay_live.py` from the repo root.
    from live_processor import LiveProcessor


VIDEO_NAMES = ("raw.mp4", "video.mp4", "video.mov", "video.m4v")
DEFAULT_MODEL = Path("models/frame-geometry-yolo11n-v5.onnx")
WINDOW = "MaiLens PC replay"
QUIT_KEYS = {ord("q"), 27}
NO_KEY = 255


def load_session_rows(session: Path) -> list[dict]:
    """Return the per-frame metadata of a session, ordered by frame id."""
    path = session / "capture.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{session} 里没有 capture.jsonl")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"{path} 是空的")
    rows.sort(key=lambda row: int(row.get("frame_id", 0)))
    return rows


def session_fps(session: Path, rows: list[dict], default: float = 60.0) -> float:
    """Frame rate of the session, for the output file, the HUD and the pacing.

    Measured first: the manifest's ``nominal_fps`` is the receiver's ``--fps``
    default rather than a measurement, and a real phone push lands at 24-30 fps,
    so replaying at the nominal 60 ran the clip about 2.4x too fast.
    """
    try:
        from .process_session import measured_fps
    except ImportError:  # Running as `python pc/replay_live.py`.
        from process_session import measured_fps
    measured = measured_fps(rows)
    if measured is not None:
        return float(round(measured, 3))
    manifest = session / "session.json"
    if manifest.exists():
        try:
            nominal = json.loads(manifest.read_text(encoding="utf-8")).get("nominal_fps")
        except (json.JSONDecodeError, OSError):
            nominal = None
        if nominal:
            return float(nominal)
    return default


def frame_path(session: Path, row: dict, index: int) -> Path | None:
    relative = row.get("frame_path")
    if relative:
        candidate = session / str(relative).replace("\\", "/")
        if candidate.exists():
            return candidate
    named = session / "frames" / f"{int(row.get('frame_id', index)):08d}.jpg"
    if named.exists():
        return named
    return None


def iter_session_frames(session: Path, rows: Iterable[dict]) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield ``(row, frame)`` pairs, preferring the JPEGs the phone sent."""
    rows = list(rows)
    paths = [frame_path(session, row, index) for index, row in enumerate(rows)]
    if rows and all(path is not None for path in paths):
        for row, path in zip(rows, paths):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"无法解码帧：{path}")
            yield row, frame
        return

    video = next((session / name for name in VIDEO_NAMES if (session / name).exists()), None)
    if video is None:
        raise FileNotFoundError(f"{session} 里既没有 frames/*.jpg 也没有 {'/'.join(VIDEO_NAMES)}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    try:
        for row in rows:
            ok, frame = capture.read()
            if not ok:
                break
            yield row, frame
    finally:
        capture.release()


class PlaybackClock:
    """Turns recorded timestamps into wall-clock pacing for the preview."""

    def __init__(self, *, speed: float = 1.0, realtime: bool = True,
                 time_fn: Callable[[], float] = time.monotonic,
                 sleep_fn: Callable[[float], None] = time.sleep) -> None:
        self.speed = max(float(speed), 0.01)
        self.realtime = bool(realtime)
        self._time = time_fn
        self._sleep = sleep_fn
        self._origin: float | None = None
        self._wall_origin: float | None = None
        self._paused_at: float | None = None

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def reset(self) -> None:
        self._origin = None
        self._wall_origin = None
        self._paused_at = None

    def wait(self, timestamp: float | None) -> None:
        """Sleep until the recorded timestamp is due; the first call anchors."""
        if not self.realtime or timestamp is None:
            return
        if self._origin is None:
            self._origin = float(timestamp)
            self._wall_origin = self._time()
            return
        target = self._wall_origin + (float(timestamp) - self._origin) / self.speed
        while True:
            delay = target - self._time()
            if delay <= 0:
                return
            self._sleep(min(delay, 0.05))

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = self._time()

    def resume(self) -> None:
        """Shift the wall-clock origin so the pause does not cause a catch-up burst."""
        if self._paused_at is not None:
            if self._wall_origin is not None:
                self._wall_origin += self._time() - self._paused_at
            self._paused_at = None


def newest_session(root: Path) -> Path:
    """Newest directory under ``sessions/`` that carries capture.jsonl."""
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "capture.jsonl").exists()]
    if not candidates:
        raise FileNotFoundError(f"{root} 里没有可回放的会话")
    return max(candidates, key=lambda path: (path / "capture.jsonl").stat().st_mtime)


def apply_key(key: int, state: dict, processor) -> None:
    if key in QUIT_KEYS:
        state["quit"] = True
    elif key == ord(" "):
        state["paused"] = not state["paused"]
    elif key == ord("."):
        state["step"] = True
    elif key == ord("r"):
        state["restart"] = True
    elif key == ord("d"):
        processor.debug = not getattr(processor, "debug", False)


def draw_status(frame: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    shown = frame.copy()
    y = 30
    for text in lines:
        cv2.putText(shown, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 190), 2, cv2.LINE_AA)
        y += 28
    return shown


def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    height, width = frame.shape[:2]
    return cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def replay(
    session: Path,
    *,
    processor_factory: Callable[[], object],
    preview: bool = True,
    realtime: bool = True,
    speed: float = 1.0,
    start: float = 0.0,
    max_frames: int | None = None,
    output: Path | None = None,
    processing_scale: float = 1.0,
    clock: PlaybackClock | None = None,
    read_key: Callable[[], int] | None = None,
    report: Callable[[str], None] = print,
) -> dict:
    """Run the session through the live processor and return summary statistics."""
    rows = load_session_rows(session)
    if start > 0:
        origin = float(rows[0].get("timestamp", 0.0))
        rows = [row for row in rows if float(row.get("timestamp", 0.0)) >= origin + start]
    if max_frames is not None:
        rows = rows[:max_frames]
    if not rows:
        raise RuntimeError("没有可回放的帧")

    fps = session_fps(session, rows)
    scale = float(min(max(processing_scale, 0.25), 1.0))
    clock = clock or PlaybackClock(speed=speed, realtime=realtime)
    read_key = read_key or (lambda: cv2.waitKey(1) & 0xFF)

    writer = None
    frame_count = 0
    lock_sources: dict[str, int] = {}
    started = time.monotonic()
    last_report = started
    report_frames = 0
    display_fps = 0.0
    last_shown: np.ndarray | None = None
    last_info: dict = {}

    try:
        while True:
            clock.reset()
            state = {"quit": False, "restart": False, "paused": False, "step": False}
            processor = processor_factory()
            for row, frame in iter_session_frames(session, rows):
                clock.wait(row.get("timestamp"))
                while True:
                    key = read_key()
                    if key is not None and key != NO_KEY:
                        apply_key(key, state, processor)
                    if state["quit"] or state["restart"]:
                        break
                    if state["paused"]:
                        clock.pause()
                        if state["step"]:
                            state["step"] = False
                            break
                        if preview and last_shown is not None:
                            cv2.imshow(WINDOW, draw_status(last_shown, ["PAUSED  space to resume, . to step"]))
                        time.sleep(0.01)
                        continue
                    clock.resume()
                    break
                if state["quit"] or state["restart"]:
                    break

                height, width = frame.shape[:2]
                processed, info = processor.process(scale_frame(frame, scale), row)
                if scale < 0.999:
                    processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_LINEAR)
                last_info = info or {}
                if output is not None and writer is None:
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建输出视频：{output}")
                if writer is not None:
                    writer.write(processed)

                frame_count += 1
                report_frames += 1
                source = str(last_info.get("lock_source", "none"))
                lock_sources[source] = lock_sources.get(source, 0) + 1

                if preview:
                    now = time.monotonic()
                    if now - last_report >= 1.0:
                        display_fps = report_frames / max(now - last_report, 0.001)
                        last_report = now
                        report_frames = 0
                    label = f"REPLAY {speed:g}x  {display_fps:.1f}/{fps:.0f} fps  lock={source}"
                    last_shown = processed
                    cv2.imshow(WINDOW, draw_status(processed, [label]))
                if frame_count % 240 == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    report(f"已回放 {frame_count}/{len(rows)} 帧 · {frame_count / elapsed:.1f} fps · lock={source}")

            if state["quit"] or not state["restart"]:
                break
            last_shown = None
            report("重新开始回放")
    finally:
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started, 0.001)
    return {
        "frames": frame_count,
        "elapsed_seconds": round(elapsed, 3),
        "display_fps": round(frame_count / elapsed, 2),
        "lock_sources": lock_sources,
        "output": str(output) if output else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="?", type=Path,
                        help="会话目录；省略时取 sessions/ 下最新的一次采集")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                        help="显示实时预览窗口，默认开")
    parser.add_argument("--speed", type=float, default=1.0, help="回放倍速，默认 1.0")
    parser.add_argument("--no-realtime", action="store_true", help="不按时间戳等待，尽快跑完")
    parser.add_argument("--start", type=float, default=0.0, help="从会话开始后第几秒开始")
    parser.add_argument("--max-frames", type=int, help="只回放前 N 帧")
    parser.add_argument("--output", type=Path, help="把处理后的画面写成 mp4")
    parser.add_argument("--model", type=Path, help=f"机台检测模型，默认 {DEFAULT_MODEL}")
    parser.add_argument("--machine-lock", action=argparse.BooleanOptionalAction, default=True,
                        help="启用机台锁定，默认开")
    parser.add_argument("--processing-scale", type=float, default=0.5,
                        help="处理分辨率比例，默认 0.5，和接收端实时链路一致；1.0 画质更好但帧率更低")
    parser.add_argument("--detect-every", type=int, default=12,
                        help="模型每隔多少帧检测一次，默认 12")
    parser.add_argument("--lock-fill", type=float, default=0.71)
    parser.add_argument("--debug", action="store_true", help="叠加检测框和锁定状态")
    parser.add_argument("--crop", type=float, default=0.74)
    parser.add_argument("--fov", type=float, default=106.4583)
    parser.add_argument("--center-x", type=float, default=0.501753869)
    parser.add_argument("--center-y", type=float, default=0.499423644)
    parser.add_argument("--k1", type=float, default=0.0893163)
    parser.add_argument("--k2", type=float, default=-0.0174637)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.speed <= 0:
        parser.error("--speed 必须大于 0")
    if args.start < 0:
        parser.error("--start 不能是负数")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames 必须大于 0")
    if not 0.25 <= args.processing_scale <= 1.0:
        parser.error("--processing-scale 应在 0.25 到 1.0 之间")
    if not 0.35 <= args.lock_fill <= 0.90:
        parser.error("--lock-fill 应在 0.35 到 0.90 之间")
    if not 0.2 <= args.crop <= 1.0:
        parser.error("--crop 应在 0.2 到 1.0 之间")
    if args.detect_every < 1:
        parser.error("--detect-every 必须大于 0")
    return args


def main() -> None:
    args = parse_args()
    session = args.session or newest_session(Path("sessions"))
    session = session.resolve()
    print(f"回放会话：{session}")
    model = args.model
    if args.machine_lock and model is None:
        model = DEFAULT_MODEL

    def processor_factory() -> LiveProcessor:
        return LiveProcessor(
            crop=args.crop,
            fov=args.fov,
            center_x=args.center_x,
            center_y=args.center_y,
            k1=args.k1,
            k2=args.k2,
            model=model,
            detect_every=args.detect_every,
            lock_fill=args.lock_fill,
            debug=args.debug,
        )

    try:
        summary = replay(
            session,
            processor_factory=processor_factory,
            preview=args.preview,
            realtime=not args.no_realtime,
            speed=args.speed,
            start=args.start,
            max_frames=args.max_frames,
            output=args.output,
            processing_scale=args.processing_scale,
        )
    except KeyboardInterrupt:
        print("\n回放已停止。")
        return
    print(
        f"回放完成：{summary['frames']} 帧 · {summary['elapsed_seconds']:.1f} 秒 · "
        f"{summary['display_fps']:.1f} fps · 锁定状态 {summary['lock_sources']}"
    )
    if summary["output"]:
        print(f"输出视频：{summary['output']}")


if __name__ == "__main__":
    main()