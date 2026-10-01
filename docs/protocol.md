# MaiLens Remote Capture Protocol v1

The protocol deliberately keeps framing independent from JPEG and JSON. TCP is a byte stream, so the receiver first reads the fixed 14-byte header and then exactly the declared metadata and payload lengths.

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 4 | ASCII `MLCP` |
| 4 | 1 | Version (`1`) |
| 5 | 1 | Type (`1` frame, `2` hello) |
| 6 | 4 | Metadata length, unsigned big-endian |
| 10 | 4 | JPEG payload length, unsigned big-endian |

Frame metadata is UTF-8 JSON. It contains the camera timestamp and the motion sample captured nearest to that frame. Keeping both timestamps is intentional: `timestamp` is the video presentation time, while `pose.timestamp` is the sensor sample time. The PC processor reports their difference instead of silently hiding sensor delay.

`pose` also carries `user_acceleration`, the gravity-free acceleration that the front/back part of the lock needs. PC tools ignore fields they do not know, so older captures stay readable.

## Session upload protocol v1

A recorded session travels over a second TCP connection, on the frame stream port plus one, so a multi-minute upload cannot stall the live preview. Integers are big-endian, and the sender waits for a single reply line at the end.

```text
4 bytes  magic = MLSF
1 byte   version = 1
2 bytes  session name length (N)
N bytes  UTF-8 session name, for example 20261002-153000
```

The header is followed by file records until an end marker:

| Type | Payload |
| ---: | --- |
| 1 | 2-byte file name length, UTF-8 file name, 8-byte payload length, then exactly that many bytes |
| 2 | end of session, no further fields |

The receiver answers `OK <files> <bytes>` or `ERR <reason>`. `pc/session_server.py` writes the files under `phone_sessions/<name>/` and can hand them straight to `pc/import_phone_session.py`.

