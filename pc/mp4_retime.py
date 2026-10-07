"""Rewrite the timescale of a recorded MP4 so it plays at its capture rate.

OpenCV's ``VideoWriter`` stamps every frame with one constant rate known when
the file is opened.  The phone streams at whatever rate the network and the
encoder allow, so a session recorded at a nominal 60 fps really ran at 24-29 fps.
The result, without this pass, is a clip that plays roughly 2x fast: the machine
appears to jump every few frames and the motion reads as stutter even though
every captured frame is present.

Rather than re-encode, patch the two header fields that decide how long the file
is: the media timescale in ``mdhd`` and the movie duration in ``mvhd``/``tkhd``.
Sample count, frame order and every byte of picture data are left alone.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

__all__ = ["measured_fps", "scale_timescale", "retime_video"]

UINT32 = struct.Struct(">I")

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}


def _boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yield ``(type, header_start, payload_start, box_end)`` for sibling boxes."""
    if end is None:
        end = len(data)
    offset = start
    while offset + 8 <= end:
        size = UINT32.unpack_from(data, offset)[0]
        kind = data[offset + 4:offset + 8]
        header = 8
        if size == 1:
            if offset + 16 > end:
                return
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            return
        yield kind, offset, offset + header, offset + size
        offset += size


def _find(data: bytes, want: bytes, start: int, end: int, deep: bool = False):
    """First box of type ``want`` among the siblings, descending if asked."""
    for kind, _hdr, payload, box_end in _boxes(data, start, end):
        if kind == want:
            return payload, box_end
        if deep and kind in CONTAINERS:
            found = _find(data, want, payload, box_end, deep=True)
            if found is not None:
                return found
    return None


def _fill32(data: bytearray, offset: int, value: int) -> None:
    if value > 0xFFFFFFFF:
        raise ValueError(f"值 {value} 超出 32 位字段")
    UINT32.pack_into(data, offset, value)


def _fill64(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into(">Q", data, offset, value)


def _sum_sample_deltas(data: bytes, stts_payload: int, box_end: int) -> int:
    """Total of every ``stts`` entry, i.e. the track duration in media units."""
    if data[stts_payload] != 0:
        raise ValueError("stts version 不为 0，无法解读")
    entries = UINT32.unpack_from(data, stts_payload + 4)[0]
    total = 0
    offset = stts_payload + 8
    for _ in range(entries):
        if offset + 8 > box_end:
            raise ValueError("stts 条目越界")
        count, delta = struct.unpack_from(">II", data, offset)
        total += count * delta
        offset += 8
    return total


def _mdhd_fields(data: bytes, payload: int) -> tuple[int, int, bool]:
    """``(timescale_offset, duration_offset, is_64bit)`` inside one mdhd box."""
    if data[payload] == 1:
        return payload + 20, payload + 24, True
    return payload + 12, payload + 16, False


def _movie_timescale(data: bytes, mvhd_payload: int | None) -> int:
    if mvhd_payload is None:
        return 1000
    return UINT32.unpack_from(data, mvhd_payload + (20 if data[mvhd_payload] == 1 else 12))[0]


def _patch_elst(data: bytearray, payload: int, box_end: int,
                seconds: float, movie_ts: int, factor: float) -> None:
    """Keep the edit list consistent with the new media timescale.

    OpenCV writes one entry whose ``segment_duration`` is the movie length in
    movie units.  A demuxer honours it as a hard stop, so leaving it at the old
    (shorter) value truncates the retimed clip: the session that exposed this
    decoded 92 of its 196 frames.  ``media_time`` is an offset on the stretched
    media timeline, so it scales with the timescale.
    """
    version = data[payload]
    entries = UINT32.unpack_from(bytes(data), payload + 4)[0]
    offset = payload + 8
    for _ in range(entries):
        if version == 1:
            if offset + 16 > box_end:
                return
            segment, media = struct.unpack_from(">Qq", bytes(data), offset)
            if segment != 0xFFFFFFFFFFFFFFFF:
                struct.pack_into(">Q", data, offset, int(round(seconds * movie_ts)))
            struct.pack_into(">q", data, offset + 8, int(round(media * factor)))
            offset += 16
        else:
            if offset + 12 > box_end:
                return
            segment = UINT32.unpack_from(bytes(data), offset)[0]
            media = struct.unpack_from(">i", bytes(data), offset + 4)[0]
            if segment != 0xFFFFFFFF:
                _fill32(data, offset, int(round(seconds * movie_ts)))
            struct.pack_into(">i", data, offset + 4, int(round(media * factor)))
            offset += 12


def _patch_moov(data: bytearray, factor: float) -> float:
    """Scale the media timescale of every track by ``factor``.

    Returns the resulting duration in seconds.  The ``stts`` deltas stay as they
    are: reading the same sample durations against a larger timescale is exactly
    what makes the clip last longer.
    """
    moov = _find(bytes(data), b"moov", 0, len(data))
    if moov is None:
        raise ValueError("文件里没有 moov 盒")
    moov_payload, moov_end = moov

    mvhd_payload = None
    for kind, _hdr, payload, box_end in _boxes(bytes(data), moov_payload, moov_end):
        if kind == b"mvhd":
            mvhd_payload = payload
            break

    seconds = None
    for kind, _hdr, payload, box_end in _boxes(bytes(data), moov_payload, moov_end):
        if kind != b"trak":
            continue
        mdhd = _find(bytes(data), b"mdhd", payload, box_end, deep=True)
        stts = _find(bytes(data), b"stts", payload, box_end, deep=True)
        if mdhd is None or stts is None:
            continue
        ts_off, dur_off, wide = _mdhd_fields(bytes(data), mdhd[0])
        old_ts = UINT32.unpack_from(bytes(data), ts_off)[0]
        units = _sum_sample_deltas(bytes(data), stts[0], stts[1])
        new_ts = max(1, int(round(old_ts * factor)))
        _fill32(data, ts_off, new_ts)
        track_seconds = units / new_ts
        if wide:
            _fill64(data, dur_off, units)
        else:
            _fill32(data, dur_off, units)
        seconds = track_seconds if seconds is None else max(seconds, track_seconds)
        elst = _find(bytes(data), b"elst", payload, box_end, deep=True)
        if elst is not None:
            _patch_elst(data, elst[0], elst[1], track_seconds,
                        _movie_timescale(bytes(data), mvhd_payload), factor)
        tkhd = _find(bytes(data), b"tkhd", payload, box_end)
        if tkhd is not None:
            movie_ts = _movie_timescale(bytes(data), mvhd_payload)
            if data[tkhd[0]] == 1:
                _fill64(data, tkhd[0] + 32, int(round(track_seconds * movie_ts)))
            else:
                _fill32(data, tkhd[0] + 20, int(round(track_seconds * movie_ts)))

    if seconds is None:
        raise ValueError("文件里没有可读的 trak/mdhd")

    if mvhd_payload is not None:
        movie_ts = _movie_timescale(bytes(data), mvhd_payload)
        wide = data[mvhd_payload] == 1
        duration = int(round(seconds * movie_ts))
        if wide:
            _fill64(data, mvhd_payload + 24, duration)
        else:
            _fill32(data, mvhd_payload + 16, duration)
    return seconds


def scale_timescale(path: Path, factor: float) -> float:
    """Stretch (factor > 1) or shorten the media timescale, in place.

    ``factor = measured_fps / nominal_fps`` turns a clip stamped at the nominal
    rate into one whose duration matches the capture: a smaller timescale makes the
    same media units last longer.  Returns the new duration
    in seconds.
    """
    path = Path(path)
    if factor <= 0:
        raise ValueError("factor 必须大于 0")
    data = bytearray(path.read_bytes())
    if len(data) < 16 or data[4:8] not in (b"ftyp", b"moov", b"mdat", b"skip", b"free", b"wide"):
        raise ValueError(f"{path} 不像是 MP4（顶层盒 {data[4:8]!r}）")
    seconds = _patch_moov(data, factor)
    if abs(factor - 1.0) < 1e-6:
        return seconds
    tmp = path.with_suffix(path.suffix + ".retime")
    tmp.write_bytes(bytes(data))
    tmp.replace(path)
    return seconds


def load_rows(capture_jsonl: Path) -> list[dict]:
    """Every frame record in one ``capture.jsonl``, unreadable lines dropped."""
    rows: list[dict] = []
    try:
        with open(capture_jsonl, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def measured_fps(capture_jsonl: Path) -> float | None:
    """Capture rate implied by the frame timestamps of one session log.

    Delegates to the offline processor so a live recording and an offline render
    of the same session agree on its real frame rate: there is one rule for
    rejecting a stalled capture and it lives in ``pc/process_session.py``.
    """
    rows = load_rows(Path(capture_jsonl))
    if len(rows) < 2:
        return None
    try:
        from .process_session import measured_fps as rows_fps
    except ImportError:  # 直接以脚本方式运行时没有包上下文
        from process_session import measured_fps as rows_fps
    return rows_fps(rows)


def retime_video(video: Path, capture_jsonl: Path, nominal_fps: float) -> dict:
    """Match one recorded MP4 to the rate its frames were actually captured at."""
    report = {"video": str(video), "nominal_fps": float(nominal_fps)}
    video = Path(video)
    if not video.exists() or video.stat().st_size == 0:
        report["status"] = "missing"
        return report
    rate = measured_fps(Path(capture_jsonl))
    if rate is None:
        report["status"] = "no-timestamps"
        return report
    factor = rate / nominal_fps
    report["measured_fps"] = rate
    if abs(factor - 1.0) < 0.02:  # 差不到 2%，重定时只会引入误差
        report["status"] = "already-accurate"
        return report
    try:
        seconds = scale_timescale(video, factor)
    except (ValueError, OSError, struct.error) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        return report
    report["status"] = "retimed"
    report["duration_seconds"] = seconds
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="会话目录（内含 capture.jsonl）")
    parser.add_argument("--nominal-fps", type=float, default=60.0)
    parser.add_argument("--video", action="append", default=None,
                        help="要重定时的 mp4，可重复；默认 raw.mp4 与 processed-live.mp4")
    args = parser.parse_args()
    names = args.video or ["raw.mp4", "processed-live.mp4"]
    for name in names:
        print(json.dumps(retime_video(args.session / name, args.session / "capture.jsonl",
                                      args.nominal_fps), ensure_ascii=False))


if __name__ == "__main__":
    main()

