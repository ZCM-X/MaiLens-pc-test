import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path

from .session_server import create_server, safe_component, serve_forever
from .test_import_phone_session import write_phone_session


def push_session(connection: socket.socket, name: str, files: dict[str, bytes]) -> str:
    """Speak the same MLSF framing the iOS uploader uses."""
    encoded_name = name.encode("utf-8")
    connection.sendall(b"MLSF" + bytes([1]) + struct.pack(">H", len(encoded_name)) + encoded_name)
    for file_name, payload in files.items():
        encoded = file_name.encode("utf-8")
        connection.sendall(bytes([1])
                           + struct.pack(">H", len(encoded))
                           + encoded
                           + struct.pack(">Q", len(payload)))
        for offset in range(0, len(payload), 64 * 1024):
            connection.sendall(payload[offset:offset + 64 * 1024])
    connection.sendall(bytes([2]))
    reply = b""
    while not reply.endswith(b"\n"):
        chunk = connection.recv(64)
        if not chunk:
            break
        reply += chunk
    return reply.decode("utf-8")


class SessionServerTests(unittest.TestCase):
    def receive(self, name: str, files: dict[str, bytes]) -> tuple[Path, str]:
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        server = create_server("127.0.0.1", 0)
        port = server.getsockname()[1]
        thread = threading.Thread(target=serve_forever,
                                  args=(server, root),
                                  kwargs={"once": True, "log": lambda *_: None},
                                  daemon=True)
        thread.start()
        self.addCleanup(server.close)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=20) as client:
                reply = push_session(client, name, files)
        finally:
            thread.join(timeout=20)
        return root, reply

    def test_receives_a_session_and_answers_ok(self):
        payload = bytes(range(256)) * 8192  # 2 MB, forces several socket reads
        root, reply = self.receive("20261002-153000", {
            "video.mp4": payload,
            "capture.jsonl": b'{"frame_id":1}\n',
            "pose.jsonl": b'{"timestamp":1.0}\n',
            "session.json": b"{}",
        })

        self.assertTrue(reply.startswith("OK 4 "), reply)
        directory = root / "20261002-153000"
        self.assertEqual((directory / "video.mp4").read_bytes(), payload)
        self.assertEqual((directory / "capture.jsonl").read_bytes(), b'{"frame_id":1}\n')
        self.assertEqual(sorted(path.name for path in directory.iterdir()),
                         ["capture.jsonl", "pose.jsonl", "session.json", "video.mp4"])

    def test_session_name_cannot_escape_the_output_directory(self):
        root, reply = self.receive("../../evil", {"session.json": b"{}"})
        self.assertTrue(reply.startswith("OK 1 "), reply)
        self.assertTrue((root / "evil" / "session.json").exists())
        self.assertEqual(sorted(path.name for path in root.iterdir()), ["evil"])

    def test_file_name_cannot_escape_the_session_directory(self):
        root, reply = self.receive("tricky", {"../../../escape.bin": b"payload"})
        self.assertTrue(reply.startswith("OK 1 "), reply)
        self.assertTrue((root / "tricky" / "escape.bin").exists())
        self.assertFalse((root.parent / "escape.bin").exists())

    def test_import_flag_turns_a_pushed_session_into_an_offline_session(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        phone = write_phone_session(root / "source" / "20261002-140000")
        files = {path.name: path.read_bytes() for path in phone.iterdir() if path.is_file()}

        server = create_server("127.0.0.1", 0)
        port = server.getsockname()[1]
        thread = threading.Thread(target=serve_forever,
                                  args=(server, root / "phone_sessions"),
                                  kwargs={"once": True,
                                          "run_import": True,
                                          "background_tasks": False,
                                          "sessions_root": root / "offline",
                                          "log": lambda *_: None},
                                  daemon=True)
        thread.start()
        self.addCleanup(server.close)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=30) as client:
                reply = push_session(client, "20261002-140000", files)
        finally:
            thread.join(timeout=30)

        self.assertTrue(reply.startswith("OK 4 "), reply)
        imported = root / "offline" / "20261002-140000-phone"
        self.assertEqual(len(list((imported / "frames").glob("*.jpg"))), 12)
        self.assertTrue((imported / "capture.jsonl").exists())

    def test_process_flag_runs_the_stabiliser_on_the_received_movie(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        phone = write_phone_session(root / "source" / "20261002-150500")
        files = {path.name: path.read_bytes() for path in phone.iterdir() if path.is_file()}

        server = create_server("127.0.0.1", 0)
        port = server.getsockname()[1]
        thread = threading.Thread(target=serve_forever,
                                  args=(server, root / "phone_sessions"),
                                  kwargs={"once": True,
                                          "run_process": True,
                                          "background_tasks": False,
                                          "log": lambda *_: None},
                                  daemon=True)
        thread.start()
        self.addCleanup(server.close)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=60) as client:
                reply = push_session(client, "20261002-150500", files)
        finally:
            thread.join(timeout=60)

        self.assertTrue(reply.startswith("OK 4 "), reply)
        received = root / "phone_sessions" / "20261002-150500"
        self.assertGreater((received / "processed.mp4").stat().st_size, 0)
        self.assertEqual(len((received / "debug.jsonl").read_text(encoding="utf-8").splitlines()), 12)

    def test_bad_magic_is_answered_with_an_error(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        server = create_server("127.0.0.1", 0)
        port = server.getsockname()[1]
        thread = threading.Thread(target=serve_forever,
                                  args=(server, root),
                                  kwargs={"once": True, "log": lambda *_: None},
                                  daemon=True)
        thread.start()
        self.addCleanup(server.close)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=20) as client:
                client.sendall(b"NOPE" + bytes([1]) + b"\x00\x00")
                reply = b""
                while not reply.endswith(b"\n"):
                    chunk = client.recv(64)
                    if not chunk:
                        break
                    reply += chunk
        finally:
            thread.join(timeout=20)
        self.assertTrue(reply.startswith(b"ERR"), reply)
        self.assertEqual(list(root.iterdir()), [])


class SafeComponentTests(unittest.TestCase):
    def test_strips_directories_and_empty_names(self):
        self.assertEqual(safe_component("../../etc/passwd", "fallback"), "passwd")
        self.assertEqual(safe_component("..", "fallback"), "fallback")
        self.assertEqual(safe_component("", "fallback"), "fallback")
        self.assertEqual(safe_component("a" * 400, "fallback"), "a" * 128)


if __name__ == "__main__":
    unittest.main()
