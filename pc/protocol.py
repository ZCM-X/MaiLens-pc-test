"""Wire framing helpers shared by the TCP receiver and protocol tests."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

MAGIC = b"MLCP"
VERSION = 1
TYPE_FRAME = 1
TYPE_HELLO = 2
HEADER = struct.Struct("!4sBBII")
MAX_METADATA = 1_000_000
MAX_PAYLOAD = 16_000_000


def recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if remaining == count:
                return b""
            raise ConnectionError("client disconnected in the middle of a packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_packet(sock: socket.socket) -> tuple[int, dict[str, Any], bytes] | None:
    header = recv_exact(sock, HEADER.size)
    if not header:
        return None
    magic, version, packet_type, metadata_length, payload_length = HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError(f"invalid packet magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    if metadata_length > MAX_METADATA or payload_length > MAX_PAYLOAD:
        raise ValueError("packet length exceeds configured safety limit")
    metadata_bytes = recv_exact(sock, metadata_length)
    payload = recv_exact(sock, payload_length)
    metadata = json.loads(metadata_bytes.decode("utf-8")) if metadata_bytes else {}
    if not isinstance(metadata, dict):
        raise ValueError("packet metadata must be a JSON object")
    return packet_type, metadata, payload

