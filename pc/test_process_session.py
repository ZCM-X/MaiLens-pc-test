import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .process_session import process


class ProcessSessionTests(unittest.TestCase):
    def test_pose_only_session_produces_processed_video_and_debug_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            rows = []
            for index in range(4):
                image = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.rectangle(image, (70 + index * 2, 30), (250 + index * 2, 150), (80, 220, 150), 3)
                relative = Path("frames") / f"{index + 1:08d}.jpg"
                cv2.imwrite(str(root / relative), image)
                rows.append({
                    "frame_id": index + 1,
                    "timestamp": index / 15,
                    "frame_path": str(relative).replace("\\", "/"),
                    "pose": {
                        "timestamp": index / 15,
                        "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
                    },
                })
            (root / "capture.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            (root / "session.json").write_text(json.dumps({"nominal_fps": 15}), encoding="utf-8")

            output = process(type("Args", (), {
                "session": root,
                "output": root / "processed.mp4",
                "model": None,
                "crop": 0.74,
                "fov": 106.4583,
                "center_x": 0.501753869,
                "center_y": 0.499423644,
                "k1": 0.0893163,
                "k2": -0.0174637,
                "fps": None,
                "debug": True,
                "preview": False,
                "detect_every": 3,
            })())

            self.assertEqual(output, root / "processed.mp4")
            self.assertGreater((root / "processed.mp4").stat().st_size, 0)
            self.assertEqual(len((root / "debug.jsonl").read_text(encoding="utf-8").splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
