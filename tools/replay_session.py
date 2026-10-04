#!/usr/bin/env python3
"""Feed a recorded phone session to the live PC receiver, at the phone's pace.

The receiver (``pc/pc_receiver.py``) speaks one TCP protocol and cannot tell a
phone from anything else that speaks it, so a session captured over Wi-Fi can be
played back into ``--process-live --preview`` and the preview reacts exactly as
it would to the phone.  That is how the live path gets exercised on the computer
without rebuilding or re-pointing the app: first prove the geometry on the PC,
then move it to the phone.

Each ``capture.jsonl`` row already carries the frame id, the sensor timestamp
and the pose quaternion, and ``frames/<id>.jpg`` carries the matching JPEG, so
the replay rebuilds the original packets byte for byte.

    python tools/replay_session.py sessions/20261002-035222 --limit 600
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc.protocol import HEADER, MAGIC, TYPE_FRAME, TYPE_HELLO, VERSION  # noqa: E402


def read_rows(session: Path, start: int, limit: int | None) -> list[dict]:
    """capture.jsonl rows, with the frame path resolved and the payload read."""
    capture = session / "capture.jsonl"
    if not capture.exists():
        raise SystemExit(f"找不到 capture.jsonl：{capture}")
    rows: list[dict] = []
    with capture.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    selected = rows[start:None if limit is None else start + limit]
    if not selected:
        raise SystemExit("这个范围里没有帧")
    return selected


def frame_path(session: Path, row: dict) -> Path:
    relative = row.get("frame_path")
    if relative:
        # capture.jsonl stores Windows separators; both separators work here.
        candidate = session / Path(str(relative).replace("\\", "/"))
        if candidate.exists():
            return candidate
    return session / "frames" / f"{int(row['frame_id']):08d}.jpg"


def packet(packet_type: int, metadata: dict, payload: bytes = b"") -> bytes:
    encoded = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    return (HEADER.pack(MAGIC, VERSION, packet_type, len(encoded), len(payload))
            + encoded + payload)


def replay(args: argparse.Namespace) -> None:
    rows = read_rows(args.session, args.start, args.limit)
    first = float(rows[0].get("timestamp") or 0.0)
    print(f"回放 {args.session}：{len(rows)} 帧，速度 {args.speed:g}x，"
          f"发往 {args.host}:{args.port}")

    with socket.create_connection((args.host, args.port), timeout=20) as client:
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = {"client": "MaiLensReplay", "version": VERSION,
                 "source": str(args.session)}
        client.sendall(packet(TYPE_HELLO, hello))

        started = time.monotonic()
        sent = 0
        dropped = 0
        for row in rows:
            path = frame_path(args.session, row)
            try:
                payload = path.read_bytes()
            except OSError:
                dropped += 1
                continue
            metadata = {
                "frame_id": int(row.get("frame_id", sent + 1)),
                "timestamp": row.get("timestamp"),
                "width": row.get("width"),
                "height": row.get("height"),
                "pose": row.get("pose"),
            }
            # Pace on the sensor clock, so the receiver sees the same spacing
            # (and therefore the same pose deltas) the phone produced.
            due = started + (float(row.get("timestamp") or first) - first) / args.speed
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            client.sendall(packet(TYPE_FRAME, metadata, payload))
            sent += 1
            if sent % args.report == 0:
                elapsed = time.monotonic() - started
                print(f"  已发送 {sent}/{len(rows)} 帧，"
                      f"平均 {sent / elapsed:.1f} fps，落后 "
                      f"{max(0.0, elapsed - (float(row.get('timestamp') or first) - first) / args.speed):.2f}s")
        elapsed = time.monotonic() - started
        print(f"回放结束：{sent} 帧，用时 {elapsed:.1f}s（{sent / max(elapsed, 1e-6):.1f} fps）"
              + (f"，跳过 {dropped} 帧找不到的 JPEG" if dropped else ""))
    print("连接已关闭，接收端会结束这个会话。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="sessions/<时间> 或手机导出的会话目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--start", type=int, default=0, help="从第几帧开始（默认 0）")
    parser.add_argument("--limit", type=int, help="最多回放多少帧")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="回放速度倍数，默认 1.0 即原始帧率")
    parser.add_argument("--report", type=int, default=120, help="每多少帧打印一次进度")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed 必须大于 0")
    if args.start < 0:
        parser.error("--start 不能为负")
    if not (0 < args.port < 65536):
        parser.error("--port 必须在 1 到 65535 之间")
    replay(args)


if __name__ == "__main__":
    main()
