import json
import socket
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .pc_receiver import client_loop
from .protocol import HEADER, MAGIC, TYPE_FRAME, TYPE_HELLO, VERSION


def packet(packet_type: int, metadata: dict, payload: bytes = b"") -> bytes:
    encoded = json.dumps(metadata).encode("utf-8")
    return HEADER.pack(MAGIC, VERSION, packet_type, len(encoded), len(payload)) + encoded + payload


class LiveReceiverTests(unittest.TestCase):
    def test_receiver_writes_processed_live_video(self):
        with self.subTest(mode="live"):
            import tempfile

            with tempfile.TemporaryDirectory() as temporary:
                left, right = socket.socketpair()
                args = SimpleNamespace(
                    output_dir=Path(temporary),
                    fps=15.0,
                    process_live=True,
                    crop=0.74,
                    fov=106.4583,
                    center_x=0.501753869,
                    center_y=0.499423644,
                    k1=0.0893163,
                    k2=-0.0174637,
                    model=None,
                    detect_every=3,
                    debug=False,
                    preview=False,
                    processing_scale=0.5,
                )
                result = []

                def receive():
                    result.append(client_loop(right, ("local", 1), args))

                worker = threading.Thread(target=receive)
                worker.start()
                left.sendall(packet(TYPE_HELLO, {"client": "test", "version": 1}))
                image = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.rectangle(image, (55, 25), (265, 155), (80, 220, 150), 3)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                for index in range(3):
                    metadata = {
                        "frame_id": index + 1,
                        "timestamp": 20.0 + index / 15.0,
                        "pose": {
                            "timestamp": 20.0 + index / 15.0,
                            "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
                        },
                    }
                    left.sendall(packet(TYPE_FRAME, metadata, encoded.tobytes()))
                left.close()
                worker.join(timeout=10)
                right.close()
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(result), 1)
                session = result[0]
                self.assertTrue((session / "raw.mp4").exists())
                self.assertTrue((session / "processed-live.mp4").exists())
                self.assertEqual(len((session / "processed.jsonl").read_text(encoding="utf-8").splitlines()), 3)
                processed_video = cv2.VideoCapture(str(session / "processed-live.mp4"))
                self.assertEqual(int(processed_video.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
                self.assertEqual(int(processed_video.get(cv2.CAP_PROP_FRAME_HEIGHT)), 180)
                processed_video.release()


if __name__ == "__main__":
    unittest.main()
