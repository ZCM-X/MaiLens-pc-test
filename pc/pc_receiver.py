#!/usr/bin/env python3
"""Receive same-packet JPEG + IMU samples, optionally processing each frame live."""

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
    from .live_processor import LiveProcessor
except ImportError:  # Running as `python pc/pc_receiver.py` from the repo root.
    from protocol import TYPE_FRAME, TYPE_HELLO, recv_packet
    from live_processor import LiveProcessor


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


def scale_box(box, scale_x: float, scale_y: float):
    if box is None:
        return None
    return tuple(int(round(value * scale)) for value, scale in zip(box, (scale_x, scale_y, scale_x, scale_y)))


def client_loop(client: socket.socket, address: tuple[str, int], args: argparse.Namespace) -> Path:
    session = create_session(args.output_dir)
    print(f"\n手机已连接：{address[0]}:{address[1]}")
    print(f"会话目录：{session}")
    print("按 Ctrl+C 停止接收；手机断开后会自动结束当前会话。")

    log_path = session / "capture.jsonl"
    processed_log_path = session / "processed.jsonl"
    video_writer = None
    processed_writer = None
    live_processor = None
    if args.process_live:
        live_processor = LiveProcessor(
            crop=args.crop,
            fov=args.fov,
            center_x=args.center_x,
            center_y=args.center_y,
            k1=args.k1,
            k2=args.k2,
            model=args.model,
            detect_every=args.detect_every,
            lock_fill=getattr(args, "lock_fill", 0.64),
            debug=args.debug,
        )
        print("实时处理：鱼眼矫正 + 姿态云台已开启" + (" + 机台检测" if args.model else ""))
    frame_count = 0
    last_report = time.monotonic()
    report_frame_count = 0
    display_fps = 0.0
    processing_scale = float(getattr(args, "processing_scale", 1.0))
    processing_scale = min(max(processing_scale, 0.25), 1.0)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file, \
             processed_log_path.open("w", encoding="utf-8", buffering=1) as processed_log:
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
                processed = image
                processed_row = None
                if live_processor is not None:
                    live_image = image
                    if processing_scale < 0.999:
                        live_image = cv2.resize(
                            image,
                            (max(1, int(round(width * processing_scale))),
                             max(1, int(round(height * processing_scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    processed, processed_row = live_processor.process(live_image, metadata)
                    if processing_scale < 0.999:
                        processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_LINEAR)
                        if processed_row is not None:
                            processed_row["detected_outer"] = scale_box(
                                processed_row.get("detected_outer"),
                                1.0 / processing_scale,
                                1.0 / processing_scale,
                            )
                            processed_row["detected_inner"] = scale_box(
                                processed_row.get("detected_inner"),
                                1.0 / processing_scale,
                                1.0 / processing_scale,
                            )
                    if processed_writer is None:
                        processed_path = session / "processed-live.mp4"
                        processed_writer = cv2.VideoWriter(
                            str(processed_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps,
                            (width, height),
                        )
                        if not processed_writer.isOpened():
                            raise RuntimeError(f"无法创建实时处理录像：{processed_path}")
                    processed_writer.write(processed)
                    processed_log.write(json.dumps(processed_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                frame_count += 1
                report_frame_count += 1

                if args.preview:
                    shown = processed.copy() if live_processor is not None else image.copy()
                    mode = "LIVE" if live_processor is not None else "RAW"
                    cv2.putText(shown, f"{mode}  {display_fps:.1f}/{args.fps:.0f} fps", (14, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 190), 2, cv2.LINE_AA)
                    cv2.imshow("MaiLens PC receiver", shown)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                now = time.monotonic()
                if now - last_report >= 2:
                    display_fps = report_frame_count / max(now - last_report, 0.001)
                    print(f"已接收 {frame_count} 帧 · 实际 {display_fps:.1f}/{args.fps:.0f} fps · 当前 {width}×{height} · 姿态时间差 {row['sensor_delta_ms'] if row['sensor_delta_ms'] is not None else '无'} ms")
                    report_frame_count = 0
                    last_report = now
    finally:
        if video_writer is not None:
            video_writer.release()
        if processed_writer is not None:
            processed_writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    manifest = {
        "frame_count": frame_count,
        "nominal_fps": args.fps,
        "video": "raw.mp4",
        "metadata": "capture.jsonl",
        "live_processed_video": "processed-live.mp4" if live_processor is not None else None,
        "live_processed_metadata": "processed.jsonl" if live_processor is not None else None,
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (session / "session.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"会话完成：{frame_count} 帧 → {session}")
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认允许局域网连接")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=60.0, help="raw.mp4 标称帧率")
    parser.add_argument("--output-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--preview", action="store_true", help="显示接收画面，按 q 结束")
    parser.add_argument("--process-live", action="store_true",
                        help="收到每帧后立即做鱼眼矫正和姿态稳定，并实时预览/保存")
    parser.add_argument("--processing-scale", type=float, default=0.5,
                        help="实时鱼眼处理比例，默认 0.5 以接近 60 fps；原始帧仍保存全分辨率")
    parser.add_argument("--model", type=Path, help="可选外框/内屏模型；与 --process-live 一起使用")
    parser.add_argument("--machine-lock", action="store_true",
                        help="启用机台检测居中；默认使用 models/frame-geometry-yolo11n-v2.onnx")
    parser.add_argument("--lock-fill", type=float, default=0.64,
                        help="内屏锁定后占画面短边的比例，默认 0.64")
    parser.add_argument("--debug", action="store_true", help="实时画面叠加检测框、中心和缩放")
    parser.add_argument("--crop", type=float, default=0.74)
    parser.add_argument("--fov", type=float, default=106.4583)
    parser.add_argument("--center-x", type=float, default=0.501753869)
    parser.add_argument("--center-y", type=float, default=0.499423644)
    parser.add_argument("--k1", type=float, default=0.0893163)
    parser.add_argument("--k2", type=float, default=-0.0174637)
    parser.add_argument("--detect-every", type=int, default=12,
                        help="模型每隔多少帧检测一次，默认 12；中间帧使用光流跟踪")
    args = parser.parse_args()
    if args.machine_lock and args.model is None:
        # v2 has a stable inner-screen head on the current live footage.  v3
        # is kept in the repository for retraining experiments, but it often
        # drops the inner screen and cannot drive the plane lock continuously.
        args.model = Path("models/frame-geometry-yolo11n-v2.onnx")
    if not (0 < args.port < 65536):
        parser.error("--port 必须在 1 到 65535 之间")
    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if not 0.25 <= args.processing_scale <= 1.0:
        parser.error("--processing-scale 应在 0.25 到 1.0 之间")
    if not 0.35 <= args.lock_fill <= 0.90:
        parser.error("--lock-fill 应在 0.35 到 0.90 之间")
    if not 0.2 <= args.crop <= 1.0:
        parser.error("--crop 应在 0.2 到 1.0 之间")
    if args.detect_every < 1:
        parser.error("--detect-every 必须大于 0")
    if args.model and not args.process_live:
        parser.error("--model 需要和 --process-live 一起使用")

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
