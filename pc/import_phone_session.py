#!/usr/bin/env python3
"""Turn a phone-recorded MaiLens session into a PC session.

The iOS app records the raw camera into ``video.mp4`` next to the per-frame log
``capture.jsonl`` and the full-rate motion log ``pose.jsonl``.  The offline PC
pipeline reads a directory of JPEG frames, so this tool decodes the movie once
and rewrites the frame log with ``frame_path`` entries.  The result is a
drop-in input for ``pc/process_session.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2

VIDEO_NAMES = ("video.mp4", "video.mov", "video.m4v")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", buffering=1) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_manifest(directory: Path) -> dict:
    path = directory / "session.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def find_video(directory: Path, manifest: dict) -> Path:
    """Prefer the manifest entry, then the names the app writes."""
    candidates = []
    recorded = manifest.get("video")
    if recorded:
        candidates.append(directory / str(recorded))
    candidates.extend(directory / name for name in VIDEO_NAMES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Crash-killed sessions can leave a differently named file behind.
    for suffix in ("*.mp4", "*.mov", "*.m4v"):
        matches = sorted(directory.glob(suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"no recorded video found in {directory} (looked for {', '.join(VIDEO_NAMES)})"
    )


def resolve_output(phone_dir: Path, output: Path | None, sessions_root: Path) -> Path:
    if output is not None:
        return output
    sessions_root.mkdir(parents=True, exist_ok=True)
    base = f"{phone_dir.name}-phone"
    candidate = sessions_root / base
    suffix = 1
    while candidate.exists():
        candidate = sessions_root / f"{base}-{suffix}"
        suffix += 1
    return candidate


def import_session(
    phone_dir: Path,
    output: Path | None = None,
    sessions_root: Path = Path("sessions"),
    fps: float | None = None,
    quality: int = 92,
    copy_video: bool = True,
    log=print,
) -> Path:
    """Decode ``phone_dir/video.mp4`` into ``frames/*.jpg`` and merge the logs."""
    phone_dir = Path(phone_dir)
    if not phone_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {phone_dir}")

    manifest = load_manifest(phone_dir)
    video = find_video(phone_dir, manifest)
    rows = read_jsonl(phone_dir / "capture.jsonl")
    pose_rows = read_jsonl(phone_dir / "pose.jsonl")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}; is FFmpeg available?")

    container_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    nominal_fps = float(
        fps
        or manifest.get("nominal_fps")
        or (container_fps if 0.5 < container_fps < 1000 else 0.0)
        or 60.0
    )

    directory = resolve_output(phone_dir, output, Path(sessions_root))
    frames_dir = directory / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict] = []
    width = height = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            height, width = frame.shape[:2]
            name = f"{index + 1:08d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            row = dict(rows[index]) if index < len(rows) else {}
            row["frame_id"] = row.get("frame_id", index + 1)
            row["frame_path"] = f"frames/{name}"
            row["width"] = width
            row["height"] = height
            row["decoded_index"] = index
            merged.append(row)
            index += 1
    finally:
        capture.release()

    if index == 0:
        raise RuntimeError(f"{video} decoded zero frames")

    write_jsonl(directory / "capture.jsonl", merged)
    if pose_rows:
        shutil.copyfile(phone_dir / "pose.jsonl", directory / "pose.jsonl")

    video_name = None
    if copy_video:
        video_name = "phone.mp4"
        shutil.copyfile(video, directory / video_name)

    output_manifest = dict(manifest)
    output_manifest.update({
        "frame_count": index,
        "nominal_fps": nominal_fps,
        "metadata": "capture.jsonl",
        "pose_log": "pose.jsonl" if pose_rows else None,
        "video": video_name,
        "width": width,
        "height": height,
        "source_video": video.name,
        "source_frame_count": len(rows),
        "source_pose_samples": len(pose_rows),
        "imported_from": str(phone_dir),
        "imported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    (directory / "session.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if len(rows) != index:
        log(f"warning: {len(rows)} logged frames vs {index} decoded frames; kept the video length")
    log(f"imported {index} frames into {directory}")
    log(f"  {width}x{height}, nominal {nominal_fps:g} fps, {len(pose_rows)} pose samples")
    log(f"next: python pc/process_session.py \"{directory}\" --model models/frame-geometry-yolo11n-v2.onnx --output \"{directory}/processed.mp4\"")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="phone session folder copied from the iPhone")
    parser.add_argument("--output", type=Path, help="PC session folder to create")
    parser.add_argument("--sessions-root", type=Path, default=Path("sessions"),
                        help="where to create the session when --output is omitted")
    parser.add_argument("--fps", type=float, help="override the nominal frame rate")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality for the extracted frames")
    parser.add_argument("--no-video", action="store_true", help="do not copy video.mp4 into the session")
    args = parser.parse_args()
    if not 40 <= args.quality <= 100:
        parser.error("--quality should be between 40 and 100")
    import_session(
        args.session,
        output=args.output,
        sessions_root=args.sessions_root,
        fps=args.fps,
        quality=args.quality,
        copy_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
