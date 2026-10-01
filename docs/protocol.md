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

