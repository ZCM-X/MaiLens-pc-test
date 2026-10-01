#!/usr/bin/env python3
"""Receive same-packet JPEG + IMU samples and save a replayable session."""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from .protocol import TYPE_FRAME, TYPE_HELLO, recv_packet
except ImportError:  # Running as `python pc/pc_receiver.py` from the repo root.
    from protocol import TYPE_FRAME, TYPE_HELLO, recv_packet


def create_session(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = root / name
    suffix = 1
    while directory.exists():
        directory = root / f"{name}-{suffix}"
        suffix += 1
    (directory / "frames").mkdir(parents=True)
    return directory


def client_loop(client: socket.socket, address: tuple[str, int], args: argparse.Namespace) -> Path:
    session = create_session(args.output_dir)
    print(f"\n手机已连接：{address[0]}:{address[1]}")
    print(f"会话目录：{session}")
    print("按 Ctrl+C 停止接收；手机断开后会自动结束当前会话。")

    log_path = session / "capture.jsonl"
    video_writer = None
    frame_count = 0
    last_report = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            while True:
                packet = recv_packet(client)
                if packet is None:
                    break
                packet_type, metadata, payload = packet
                if packet_type == TYPE_HELLO:
                    (session / "client.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"客户端：{metadata.get('client', 'unknown')} / 协议 {metadata.get('version', '?')}")
                    continue
                if packet_type != TYPE_FRAME:
                    print(f"忽略未知数据包类型：{packet_type}")
                    continue
                if not payload:
                    print("忽略空视频帧")
                    continue

                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print("收到无法解码的 JPEG，跳过")
                    continue

                frame_id = int(metadata.get("frame_id", frame_count + 1))
                frame_path = session / "frames" / f"{frame_id:08d}.jpg"
                frame_path.write_bytes(payload)
                height, width = image.shape[:2]
                if video_writer is None:
                    video_path = session / "raw.mp4"
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (width, height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"无法创建录像文件：{video_path}")

                video_writer.write(image)
                pose = metadata.get("pose") or {}
                pose_timestamp = pose.get("timestamp")
                row = {
                    **metadata,
                    "frame_id": frame_id,
                    "width": width,
                    "height": height,
                    "frame_path": str(Path("frames") / frame_path.name),
                    "sensor_delta_ms": (
                        (float(pose_timestamp) - float(metadata["timestamp"])) * 1000
                        if pose_timestamp is not None and metadata.get("timestamp") is not None
                        else None
                    ),
                }
                log_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                frame_count += 1

                if args.preview:
                    shown = image.copy()
                    cv2.putText(shown, f"{frame_count}  {args.fps:.0f} fps", (14, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 190), 2, cv2.LINE_AA)
                    cv2.imshow("MaiLens PC receiver", shown)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                now = time.monotonic()
                if now - last_report >= 2:
                    print(f"已接收 {frame_count} 帧 · 当前 {width}×{height} · 姿态时间差 {row['sensor_delta_ms'] if row['sensor_delta_ms'] is not None else '无'} ms")
                    last_report = now
    finally:
        if video_writer is not None:
            video_writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    manifest = {
        "frame_count": frame_count,
        "nominal_fps": args.fps,
        "video": "raw.mp4",
        "metadata": "capture.jsonl",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (session / "session.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"会话完成：{frame_count} 帧 → {session}")
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认允许局域网连接")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=15.0, help="raw.mp4 标称帧率")
    parser.add_argument("--output-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--preview", action="store_true", help="显示接收画面，按 q 结束")
    args = parser.parse_args()
    if not (0 < args.port < 65536):
        parser.error("--port 必须在 1 到 65535 之间")
    if args.fps <= 0:
        parser.error("--fps 必须大于 0")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"MaiLens PC 接收端监听 {args.host}:{args.port}")
        print("在手机里输入这台电脑的局域网 IPv4 地址；首次运行时允许防火墙访问 Python。")
        try:
            while True:
                client, address = server.accept()
                with client:
                    client.settimeout(20)
                    try:
                        client_loop(client, address, args)
                    except (ConnectionError, TimeoutError, OSError) as error:
                        print(f"连接已结束：{error}")
        except KeyboardInterrupt:
            print("\n接收端已停止。")


if __name__ == "__main__":
    main()
