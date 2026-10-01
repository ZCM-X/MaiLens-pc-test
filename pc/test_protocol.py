import json
import socket
import struct
import threading
import unittest

from .protocol import HEADER, MAGIC, TYPE_FRAME, VERSION, recv_packet


class ProtocolTests(unittest.TestCase):
    def test_recv_packet_handles_partial_tcp_reads(self):
        left, right = socket.socketpair()
        try:
            metadata = {"frame_id": 7, "timestamp": 12.5, "pose": {"timestamp": 12.49}}
            metadata_bytes = json.dumps(metadata).encode("utf-8")
            payload = b"jpeg-bytes"
            packet = HEADER.pack(MAGIC, VERSION, TYPE_FRAME, len(metadata_bytes), len(payload)) + metadata_bytes + payload

            def send_in_chunks():
                for index in range(0, len(packet), 3):
                    left.sendall(packet[index:index + 3])
                left.close()

            worker = threading.Thread(target=send_in_chunks)
            worker.start()
            packet_type, decoded, decoded_payload = recv_packet(right)
            worker.join(timeout=1)
            self.assertEqual(packet_type, TYPE_FRAME)
            self.assertEqual(decoded, metadata)
            self.assertEqual(decoded_payload, payload)
        finally:
            right.close()


if __name__ == "__main__":
    unittest.main()
