"""Tests for Annex-B video and sidecar conversion input."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.annex_b_input_adapter import AnnexBInputAdapter  # noqa: E402, I001


# Three 16x16 red frames encoded as baseline H.264 Annex-B. Keeping the fixture
# inline makes this test independent of an ffmpeg executable at test time.
_H264_FIXTURE = base64.b64decode(
    "AAAAAWdCwArZHsBEAAADAAQAAAMACDxImSAAAAABaMuDyyAAAAEGBf//adxF6b3m2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjMgcjMwNjAgNWRiNmFhNiAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjEgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0wIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDE6MHgxMTEgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMiBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTAgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz0xIGxvb2thaGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MCB3ZWlnaHRwPTAga2V5aW50PTMga2V5aW50X21pbj0xIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9MyByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAFliIQF/EYoAAyixwABOhjgADYNgAAAAAFBmjgLeoAAAAABQZpUAr6g"
)


def _entry(
    frame_index: int,
    timestamp_ns: int | None,
    *,
    lost_packets: int = 0,
    dropped: str | None = None,
) -> dict:
    return {
        "frame_index": frame_index,
        "capture_timestamp_ns": timestamp_ns,
        "rtp_timestamp": 90_000 + frame_index * 90_000,
        "keyframe": frame_index == 0,
        "lost_packets": lost_packets,
        "session_generation": 1,
        "dropped": dropped,
    }


def _write_fixture(tmp_path: Path, entries: list[dict]) -> Path:
    stream_path = tmp_path / "observation.images.top.h264"
    stream_path.write_bytes(_H264_FIXTURE)
    sidecar = stream_path.with_suffix(".h264.json")
    sidecar.write_text("".join(f"{json.dumps(entry)}\n" for entry in entries), encoding="utf-8")
    return stream_path


def test_filters_dropped_sidecar_entries_before_pairing(tmp_path):
    stream_path = _write_fixture(
        tmp_path,
        [
            _entry(0, 1_000_000_000, lost_packets=1),
            _entry(1, None, dropped="timestamp_unmapped"),
            _entry(2, 3_000_000_000, lost_packets=2),
            _entry(3, 4_000_000_000, lost_packets=3),
        ],
    )

    adapter = AnnexBInputAdapter(stream_path)
    frames = adapter.read_frames("observation.images.top")

    assert [frame.timestamp_ns for frame in frames] == [1_000_000_000, 3_000_000_000, 4_000_000_000]
    assert [frame.lost_packets for frame in frames] == [1, 2, 3]
    assert all(frame.image.shape == (16, 16, 3) for frame in frames)

    report = adapter.integrity_report("observation.images.top")
    assert report.clean is False
    assert report.frame_gaps == [
        {"frame_index": 0, "lost_packets": 1, "reason": "rtp_sequence_gap"},
        {"frame_index": 1, "reason": "timestamp_unmapped"},
        {"frame_index": 2, "lost_packets": 2, "reason": "rtp_sequence_gap"},
        {"frame_index": 3, "lost_packets": 3, "reason": "rtp_sequence_gap"},
    ]


def test_rejects_decoded_and_non_dropped_frame_count_mismatch(tmp_path):
    stream_path = _write_fixture(tmp_path, [_entry(0, 1_000_000_000), _entry(1, 2_000_000_000)])

    with pytest.raises(ValueError, match="2 non-dropped sidecar entries != 3 decoded frames"):
        AnnexBInputAdapter(stream_path).read_frames("observation.images.top")


def test_rejects_missing_sidecar(tmp_path):
    stream_path = tmp_path / "observation.images.top.h264"
    stream_path.write_bytes(_H264_FIXTURE)

    with pytest.raises(FileNotFoundError, match="Sidecar not found"):
        AnnexBInputAdapter(stream_path)


def test_pre_keyframe_entries_are_not_counted_as_frame_gaps(tmp_path):
    """Waiting for the opening keyframe is normal stream entry, not a transport fault.

    h264_stream_recorder treats it that way via the shared ``NON_FAULT_DROP_REASONS``;
    the converter has to agree, or nearly every episode this pipeline records lands in
    the dataset with ``integrity.clean == false``.
    """
    stream_path = _write_fixture(
        tmp_path,
        [
            _entry(0, None, dropped="pre_keyframe"),
            _entry(1, None, dropped="pre_keyframe"),
            _entry(2, 3_000_000_000),
            _entry(3, 4_000_000_000),
            _entry(4, 5_000_000_000),
        ],
    )

    report = AnnexBInputAdapter(stream_path).integrity_report("observation.images.top")

    assert report.clean is True
    assert report.frame_gaps is None


def test_real_faults_are_still_counted_alongside_pre_keyframe(tmp_path):
    """Excluding pre_keyframe must not mask a genuine drop or packet loss."""
    stream_path = _write_fixture(
        tmp_path,
        [
            _entry(0, None, dropped="pre_keyframe"),
            _entry(1, 2_000_000_000, lost_packets=3),
            _entry(2, None, dropped="timestamp_unmapped"),
            _entry(3, 4_000_000_000),
        ],
    )

    report = AnnexBInputAdapter(stream_path).integrity_report("observation.images.top")

    assert report.clean is False
    assert [gap["reason"] for gap in report.frame_gaps] == ["rtp_sequence_gap", "timestamp_unmapped"]
