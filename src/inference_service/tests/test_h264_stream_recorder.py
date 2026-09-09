"""Tests for RTP Annex-B episode recording integrity."""

from __future__ import annotations

import json

import pytest

from inference_service.h264_stream_recorder import H264StreamRecorder
from inference_service.video_recording_coordinator import VideoRecordingCoordinator
from robot_config.observation_transport import NON_FAULT_DROP_REASONS


def _write_frame(
    recorder: H264StreamRecorder,
    frame_index: int,
    *,
    lost_packets: int = 0,
    dropped: str | None = None,
    keyframe: bool | None = None,
) -> None:
    recorder.write_access_unit(
        payload=b"\x65payload",
        capture_timestamp_ns=None if dropped else 1_000_000_000 + frame_index,
        rtp_timestamp=90_000 + frame_index,
        frame_index=frame_index,
        keyframe=frame_index == 0 if keyframe is None else keyframe,
        lost_packets=lost_packets,
        session_generation=1,
        dropped=dropped,
    )


def test_payload_that_already_carries_a_start_code_is_not_prefixed_again(tmp_path):
    """The depacketizer hands over Annex-B, so prefixing again emits an empty NAL.

    `rtp_sender` builds each access unit as `b"".join(START + nal ...)`, so the
    payload already opens with a start code. Writing another one in front of it
    produces a zero-length NAL per access unit: decoders skip it, but the stream
    is malformed and every frame costs four wasted bytes.
    """
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    recorder.start_episode(tmp_path, "observation.images.top")
    payload = b"\x00\x00\x00\x01\x65payload"
    recorder.write_access_unit(
        payload=payload,
        capture_timestamp_ns=1_000_000_000,
        rtp_timestamp=90_000,
        frame_index=0,
        keyframe=True,
        lost_packets=0,
        session_generation=1,
        dropped=None,
    )
    recorder.stop_episode()

    assert (tmp_path / "observation.images.top.h264").read_bytes() == payload


def test_payload_without_a_start_code_still_gets_one(tmp_path):
    """Bare NAL payloads must keep working — the prefix is only skipped, not dropped."""
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0)
    recorder.stop_episode()

    assert (tmp_path / "observation.images.top.h264").read_bytes() == b"\x00\x00\x00\x01\x65payload"


def test_strict_mode_discards_files_on_rtp_sequence_gap(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="strict")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0)
    _write_frame(recorder, 1, lost_packets=2)

    assert recorder.stop_episode() is False
    assert not (tmp_path / "observation.images.top.h264").exists()
    assert not (tmp_path / "observation.images.top.h264.json").exists()


def test_strict_mode_discards_files_on_timestamp_mapping_failure(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="strict")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0)
    _write_frame(recorder, 1, dropped="timestamp_unmapped")

    assert recorder.stop_episode() is False
    assert list(tmp_path.iterdir()) == []


def test_tolerant_mode_preserves_gap_metadata_without_dropped_payload(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0)
    _write_frame(recorder, 1, lost_packets=3)
    _write_frame(recorder, 2, dropped="timestamp_unmapped")

    assert recorder.stop_episode() is True
    stream = (tmp_path / "observation.images.top.h264").read_bytes()
    entries = [json.loads(line) for line in (tmp_path / "observation.images.top.h264.json").read_text().splitlines()]

    assert stream.count(b"\x00\x00\x00\x01") == 2
    assert entries[1]["lost_packets"] == 3
    assert entries[2]["dropped"] == "timestamp_unmapped"
    assert entries[2]["capture_timestamp_ns"] is None


def test_coordinator_strict_failure_discards_every_stream(tmp_path):
    coordinator = VideoRecordingCoordinator()
    clean = H264StreamRecorder(integrity_mode="strict")
    damaged = H264StreamRecorder(integrity_mode="strict")
    coordinator.register_recorder("observation.images.top", clean)
    coordinator.register_recorder("observation.images.wrist", damaged)
    coordinator.start_episode(tmp_path)
    _write_frame(clean, 0)
    _write_frame(damaged, 0, lost_packets=1)

    assert coordinator.stop_episode() is False
    assert list(tmp_path.glob("*.h264")) == []
    assert list(tmp_path.glob("*.h264.json")) == []


def test_episode_holds_payloads_back_until_the_first_clean_keyframe(tmp_path):
    """An episode that opens mid-GOP starts on P-frames it cannot decode.

    Those frames predict from access units the episode never recorded, so
    writing them fills the sidecar with entries no decoder can turn into
    frames. Entering a live stream is normal, so it must not fail the episode
    the way a transport fault does.
    """
    recorder = H264StreamRecorder(integrity_mode="strict")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0, keyframe=False)
    _write_frame(recorder, 1, keyframe=False)
    _write_frame(recorder, 2, keyframe=True)
    _write_frame(recorder, 3, keyframe=False)

    assert recorder.stop_episode() is True
    stream = (tmp_path / "observation.images.top.h264").read_bytes()
    entries = [json.loads(line) for line in (tmp_path / "observation.images.top.h264.json").read_text().splitlines()]

    assert [entry["dropped"] for entry in entries] == ["pre_keyframe", "pre_keyframe", None, None]
    assert stream.count(b"\x00\x00\x00\x01") == 2


@pytest.mark.parametrize("reason", sorted(NON_FAULT_DROP_REASONS))
def test_shared_non_fault_reasons_never_invalidate_an_episode(tmp_path, reason):
    """The recorder classifies ``dropped`` by the set the converter also reads.

    Both sides used to carry their own literal, and the copies disagreeing is exactly
    what marked healthy episodes ``clean=false``. Driving this off the shared set means
    a reason added later is covered here without anyone remembering to come back.
    """
    recorder = H264StreamRecorder(integrity_mode="strict")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0, keyframe=True)
    _write_frame(recorder, 1, dropped=reason)

    assert recorder.stop_episode() is True


def test_a_reason_outside_the_shared_set_still_fails_a_strict_episode(tmp_path):
    """The exclusion must stay narrow -- an unknown reason is a fault until declared one."""
    recorder = H264StreamRecorder(integrity_mode="strict")
    recorder.start_episode(tmp_path, "observation.images.top")
    _write_frame(recorder, 0, keyframe=True)
    _write_frame(recorder, 1, dropped="some_unlisted_reason")

    assert recorder.stop_episode() is False


def test_three_byte_start_code_is_also_recognised(tmp_path):
    """Annex-B allows a three-byte start code; prefixing it would emit an empty NAL.

    No depacketizer in this repo produces that form today, but the guard describes
    what the format permits, so swapping in one that does must not silently bring
    back the zero-length NAL this class exists to avoid.
    """
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    recorder.start_episode(tmp_path, "observation.images.top")
    payload = b"\x00\x00\x01\x65payload"
    recorder.write_access_unit(
        payload=payload,
        capture_timestamp_ns=1_000_000_000,
        rtp_timestamp=90_000,
        frame_index=0,
        keyframe=True,
        lost_packets=0,
        session_generation=1,
        dropped=None,
    )
    recorder.stop_episode()

    assert (tmp_path / "observation.images.top.h264").read_bytes() == payload
