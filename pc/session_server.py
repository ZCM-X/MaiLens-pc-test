#!/usr/bin/env python3
"""Receive whole sessions pushed by the phone over the LAN.

The iOS app records raw video plus the pose logs, then sends the folder here on
``--port`` (the frame stream port plus one by default).  Files land in
``--output-dir/<session name>/`` and ``--import`` converts them straight into an
offline session for ``pc/process_session.py``.
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

MAGIC = b"MLSF"
STREAM_MAGIC = b"MLCP"
VERSION = 1
MAX_FIELD = 4096
MAX_FILE = 1 << 40
READ_CHUNK = 1 << 20


class ProtocolError(RuntimeError):
    pass


def safe_component(value: str, fallback: str) -> str:
    """Keep the phone from writing outside the output directory."""
    cleaned = value.replace("\\", "/").split("/")[-1]
    cleaned = cleaned.replace("\x00", "").strip()
    if cleaned in ("", ".", ".."):
        cleaned = fallback
    return cleaned[:128]


class Reader:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.buffer = bytearray()

    def read_exactly(self, count: int) -> bytes:
        while len(self.buffer) < count:
            chunk = self.connection.recv(max(READ_CHUNK, count - len(self.buffer)))
            if not chunk:
                raise ConnectionError("phone closed the connection")
            self.buffer.extend(chunk)
        data = bytes(self.buffer[:count])
        del self.buffer[:count]
        return data

    def read_file(self, path: Path, length: int) -> int:
        """Stream a file body straight to disk instead of buffering it."""
        written = 0
        with path.open("wb") as handle:
            while written < length:
                if self.buffer:
                    take = min(len(self.buffer), length - written)
                    handle.write(bytes(self.buffer[:take]))
                    del self.buffer[:take]
                    written += take
                    continue
                chunk = self.connection.recv(min(READ_CHUNK, length - written))
                if not chunk:
                    raise ConnectionError("phone closed the connection mid-file")
                handle.write(chunk)
                written += len(chunk)
        return written


def receive_session(connection: socket.socket,
                    output_root: Path,
                    log=print) -> tuple[Path, int, int]:
    reader = Reader(connection)
    magic = reader.read_exactly(4)
    if magic != MAGIC:
        if magic == STREAM_MAGIC:
            raise ProtocolError(
                "this is the frame stream (MLCP), so the port in the phone app is set to the "
                "upload port. Set it back to 8765: the app derives the upload port (8765 + 1) "
                "by itself, and this server only accepts finished recordings."
            )
        raise ProtocolError(f"bad magic {magic!r}; is that really the phone uploader?")
    version = reader.read_exactly(1)[0]
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")

    name_length = struct.unpack(">H", reader.read_exactly(2))[0]
    if name_length > MAX_FIELD:
        raise ProtocolError("session name is too long")
    raw_name = reader.read_exactly(name_length).decode("utf-8", "replace")
    name = safe_component(raw_name, datetime.now().strftime("%Y%m%d-%H%M%S"))

    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / name
    suffix = 1
    while directory.exists():
        directory = output_root / f"{name}-{suffix}"
        suffix += 1
    directory.mkdir(parents=True)

    files = 0
    total = 0
    while True:
        kind = reader.read_exactly(1)[0]
        if kind == 2:
            break
        if kind != 1:
            raise ProtocolError(f"unknown record type {kind}")
        file_name_length = struct.unpack(">H", reader.read_exactly(2))[0]
        if file_name_length > MAX_FIELD:
            raise ProtocolError("file name is too long")
        raw_file = reader.read_exactly(file_name_length).decode("utf-8", "replace")
        file_name = safe_component(raw_file, f"file-{files + 1}.bin")
        length = struct.unpack(">Q", reader.read_exactly(8))[0]
        if length > MAX_FILE:
            raise ProtocolError("file is implausibly large")
        written = reader.read_file(directory / file_name, length)
        files += 1
        total += written
        log(f"  {file_name}  {written / 1e6:.1f} MB")

    return directory, files, total


def handle_client(connection: socket.socket,
                  address: tuple[str, int],
                  output_root: Path,
                  log=print) -> Path | None:
    started = time.monotonic()
    log(f"phone connected from {address[0]}:{address[1]}")
    try:
        directory, files, total = receive_session(connection, output_root, log=log)
    except (ProtocolError, OSError) as error:
        log(f"receive failed: {error}")
        try:
            connection.sendall(f"ERR {error}\n".encode("utf-8"))
        except OSError:
            pass
        return None

    log(f"received {files} files / {total / 1e6:.1f} MB in {time.monotonic() - started:.1f}s")
    log(f"session directory: {directory}")
    try:
        connection.sendall(f"OK {files} {total}\n".encode("utf-8"))
    except OSError:
        pass
    return directory


def process_session_folder(directory: Path,
                           model: Path | None = None,
                           debug: bool = False,
                           log=print) -> Path:
    """Run the offline stabiliser straight on the received movie and logs."""
    try:
        from .process_session import parse_args, process
    except ImportError:  # Running as `python pc/session_server.py`.
        from process_session import parse_args, process
    argv = [str(directory)]
    if model:
        argv += ["--model", str(model)]
    if debug:
        argv.append("--debug")
    log(f"processing {directory} ...")
    return process(parse_args(argv))


def post_process(directory: Path,
                 run_import: bool = False,
                 run_process: bool = False,
                 model: Path | None = None,
                 debug: bool = False,
                 sessions_root: Path | None = None,
                 log=print) -> None:
    if run_process:
        try:
            processed = process_session_folder(directory, model=model, debug=debug, log=log)
        except FileNotFoundError as error:
            log(f"processing skipped: {error}")
        else:
            log(f"processed video: {processed}")
        return
    if run_import:
        try:
            from .import_phone_session import import_session
        except ImportError:  # Running as `python pc/session_server.py`.
            from import_phone_session import import_session
        import_session(directory,
                       sessions_root=sessions_root or Path("sessions"),
                       log=log)
        return
    log(f"next: python pc/import_phone_session.py \"{directory}\"")


def create_server(host: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    return server


def serve_forever(server: socket.socket,
                  output_root: Path,
                  run_import: bool = False,
                  run_process: bool = False,
                  model: Path | None = None,
                  debug: bool = False,
                  sessions_root: Path | None = None,
                  background_tasks: bool = True,
                  once: bool = False,
                  log=print) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            client, address = server.accept()
            with client:
                client.settimeout(120)
                directory = handle_client(client, address, output_root, log=log)
            if directory is not None and (run_import or run_process):
                task = lambda directory=directory: post_process(directory,
                                            run_import=run_import,
                                            run_process=run_process,
                                            model=model,
                                            debug=debug,
                                            sessions_root=sessions_root,
                                            log=log)
                if background_tasks:
                    # A long session takes minutes to process; keep accepting.
                    threading.Thread(target=task, daemon=True).start()
                else:
                    task()
            elif directory is not None:
                log(f"next: python pc/import_phone_session.py \"{directory}\"")
            if once:
                return
    except OSError:
        return


def serve(host: str,
          port: int,
          output_root: Path,
          run_import: bool = False,
          run_process: bool = False,
          model: Path | None = None,
          debug: bool = False,
          sessions_root: Path | None = None,
          once: bool = False,
          log=print) -> None:
    server = create_server(host, port)
    with server:
        bound = server.getsockname()[1]
        log(f"MaiLens session upload listening on {host}:{bound}")
        log("This port takes finished recordings only.")
        log(f"Keep the port in the phone app at {max(bound - 1, 1)} for the frame stream;")
        log("the app adds one for uploads, and tapping 发送到电脑 does the rest.")
        try:
            serve_forever(server, output_root,
                          run_import=run_import,
                          run_process=run_process,
                          model=model,
                          debug=debug,
                          sessions_root=sessions_root,
                          once=once,
                          log=log)
        except KeyboardInterrupt:
            log("\nstopped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="address to bind, default: every interface")
    parser.add_argument("--port", type=int, default=8766,
                        help="default 8766, which is the frame stream port plus one")
    parser.add_argument("--output-dir", type=Path, default=Path("phone_sessions"))
    parser.add_argument("--import", dest="run_import", action="store_true",
                        help="run import_phone_session.py as soon as a session arrives")
    parser.add_argument("--process", dest="run_process", action="store_true",
                        help="run process_session.py on the movie as soon as a session arrives")
    parser.add_argument("--model", type=Path, help="detector used by --process")
    parser.add_argument("--debug", action="store_true", help="write debug.jsonl during --process")
    parser.add_argument("--sessions-root", type=Path, default=Path("sessions"),
                        help="where --import writes the offline session")
    parser.add_argument("--once", action="store_true", help="accept a single session and exit")
    args = parser.parse_args()
    if not (0 < args.port < 65536):
        parser.error("--port must be between 1 and 65535")
    serve(args.host, args.port, args.output_dir,
          run_import=args.run_import,
          run_process=args.run_process,
          model=args.model,
          debug=args.debug,
          sessions_root=args.sessions_root,
          once=args.once)


if __name__ == "__main__":
    main()
