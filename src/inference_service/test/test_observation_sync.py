from __future__ import annotations

import pytest

from inference_service.observation_sync import (
    ObservationSynchronizationError,
    RtpTimestampMapper,
    StreamSelection,
    select_synchronized_streams,
)
from robot_config.contract_utils import StreamBuffer


def test_rtp_timestamp_mapper_maps_90khz_ticks_to_capture_time():
    mapper = _mapper(max_mapping_age_ns=1_000_000_000)
    mapper.update(90_000, 2_000_000_000, 3_000_000_000, session_generation=1)

    assert mapper.map(135_000, now_ns=3_100_000_000, session_generation=1) == 2_500_000_000


def test_rtp_timestamp_mapper_handles_uint32_wraparound():
    mapper = _mapper(max_mapping_age_ns=1_000_000_000)
    mapper.update(0xFFFF_F000, 2_000_000_000, 3_000_000_000, session_generation=1)

    mapped = mapper.map(0x0000_1000, now_ns=3_100_000_000, session_generation=1)

    assert mapped == 2_000_000_000 + round(0x2000 * 1_000_000_000 / 90_000)


def test_rtp_timestamp_mapper_rejects_stale_mapping_and_old_generation():
    mapper = _mapper(max_mapping_age_ns=100)
    mapper.update(100, 1_000, 2_000, session_generation=2)

    with pytest.raises(ObservationSynchronizationError) as stale:
        mapper.map(101, now_ns=2_101, session_generation=2)
    assert stale.value.details["streams"][0]["reason"] == "stale"
    assert stale.value.details["streams"][0]["constraint"] == "timestamp_mapping"
    assert stale.value.details["streams"][0]["stream_id"] == "top"

    with pytest.raises(ObservationSynchronizationError):
        mapper.map(101, now_ns=2_050, session_generation=3)


def test_rtp_timestamp_mapper_resets_mapping_for_new_generation():
    mapper = _mapper(max_mapping_age_ns=100)
    mapper.update(100, 1_000, 2_000, session_generation=1)
    mapper.reset(2)

    assert mapper.ready is False
    with pytest.raises(ObservationSynchronizationError):
        mapper.map(100, now_ns=2_000, session_generation=2)


def test_select_synchronized_streams_returns_values_with_capture_timestamps():
    top = _stream("observation.images.top", "top", 1_000, "top-frame")
    wrist = _stream("observation.images.wrist", "wrist", 1_020, "wrist-frame")

    selected = select_synchronized_streams(
        {top.observation_key: top, wrist.observation_key: wrist},
        1_050,
        now_ns=1_050,
        max_inter_camera_skew_ns=20,
    )

    assert selected[top.observation_key].value == "top-frame"
    assert selected[wrist.observation_key].capture_timestamp_ns == 1_020


def test_select_synchronized_streams_rejects_inter_camera_skew():
    top = _stream("observation.images.top", "top", 1_000, "top-frame")
    wrist = _stream("observation.images.wrist", "wrist", 1_021, "wrist-frame")

    with pytest.raises(ObservationSynchronizationError) as error:
        select_synchronized_streams(
            {top.observation_key: top, wrist.observation_key: wrist},
            1_050,
            now_ns=1_050,
            max_inter_camera_skew_ns=20,
        )

    assert {item["reason"] for item in error.value.details["streams"]} == {"skewed"}
    assert {item["stream_id"] for item in error.value.details["streams"]} == {"top", "wrist"}


def test_select_synchronized_streams_retries_at_shared_capture_tick():
    top = _stream("observation.images.top", "top", 1_000, "top-shared")
    top.buffer.push(1_100, "top-newer", receive_time_ns=1_100)
    wrist = _stream("observation.images.wrist", "wrist", 1_000, "wrist-shared")

    selected = select_synchronized_streams(
        {top.observation_key: top, wrist.observation_key: wrist},
        1_100,
        now_ns=1_100,
        max_inter_camera_skew_ns=20,
    )

    assert selected[top.observation_key].capture_timestamp_ns == 1_000
    assert selected[wrist.observation_key].capture_timestamp_ns == 1_000
    assert selected[top.observation_key].value == "top-shared"


def test_select_synchronized_streams_retries_within_allowed_skew_window():
    top = _stream("observation.images.top", "top", 1_000, "top-old")
    top.buffer.push(1_100, "top-new", receive_time_ns=1_100)
    wrist = _stream("observation.images.wrist", "wrist", 1_040, "wrist-old")
    wrist.buffer.push(1_170, "wrist-new", receive_time_ns=1_170)

    selected = select_synchronized_streams(
        {top.observation_key: top, wrist.observation_key: wrist},
        1_170,
        now_ns=1_170,
        max_inter_camera_skew_ns=50,
    )

    assert selected[top.observation_key].capture_timestamp_ns == 1_000
    assert selected[wrist.observation_key].capture_timestamp_ns == 1_040


def test_select_synchronized_streams_uses_latest_common_tick_after_asymmetric_drop():
    top = _stream("observation.images.top", "top", 1_000, "top-0")
    top.buffer.push(1_100, "top-1", receive_time_ns=1_100)
    top.buffer.push(1_200, "top-2", receive_time_ns=1_200)
    top.buffer.push(1_300, "top-3", receive_time_ns=1_300)
    wrist = _stream("observation.images.wrist", "wrist", 1_000, "wrist-0")
    wrist.buffer.push(1_200, "wrist-2", receive_time_ns=1_200)

    selected = select_synchronized_streams(
        {top.observation_key: top, wrist.observation_key: wrist},
        1_300,
        now_ns=1_300,
        max_inter_camera_skew_ns=20,
    )

    assert selected[top.observation_key].capture_timestamp_ns == 1_200
    assert selected[wrist.observation_key].capture_timestamp_ns == 1_200
    assert selected[top.observation_key].value == "top-2"


@pytest.mark.parametrize(
    ("mapping_ready", "keyframe_ready", "reason"),
    [(False, True, "unmapped"), (True, False, "pre_keyframe")],
)
def test_select_synchronized_streams_reports_readiness_reason(mapping_ready, keyframe_ready, reason):
    stream = _stream(
        "observation.images.top",
        "top",
        1_000,
        "frame",
        mapping_ready=mapping_ready,
        keyframe_ready=keyframe_ready,
    )

    with pytest.raises(ObservationSynchronizationError) as error:
        select_synchronized_streams(
            {stream.observation_key: stream},
            1_000,
            now_ns=1_000,
            max_inter_camera_skew_ns=0,
        )

    issue = error.value.details["streams"][0]
    assert issue["reason"] == reason
    assert issue["observation_key"] == "observation.images.top"
    assert issue["stream_id"] == "top"


def test_select_synchronized_streams_reports_missing_and_stale_streams():
    missing = StreamSelection("observation.images.top", "top", StreamBuffer("hold", 50))
    stale_buffer = StreamBuffer("hold", 50, max_age_ns=10)
    stale_buffer.push(100, "frame", receive_time_ns=100)
    stale = StreamSelection("observation.images.wrist", "wrist", stale_buffer)

    with pytest.raises(ObservationSynchronizationError) as error:
        select_synchronized_streams(
            {missing.observation_key: missing, stale.observation_key: stale},
            120,
            now_ns=120,
            max_inter_camera_skew_ns=20,
        )

    assert {(item["stream_id"], item["reason"]) for item in error.value.details["streams"]} == {
        ("top", "missing"),
        ("wrist", "stale"),
    }


def _stream(
    observation_key: str,
    stream_id: str,
    timestamp_ns: int,
    value: object,
    *,
    mapping_ready: bool = True,
    keyframe_ready: bool = True,
) -> StreamSelection:
    buffer = StreamBuffer("hold", 50, max_age_ns=1_000)
    buffer.push(timestamp_ns, value, receive_time_ns=timestamp_ns)
    return StreamSelection(
        observation_key,
        stream_id,
        buffer,
        timestamp_mapping_ready=mapping_ready,
        keyframe_ready=keyframe_ready,
    )


def _mapper(*, max_mapping_age_ns: int) -> RtpTimestampMapper:
    return RtpTimestampMapper(
        max_mapping_age_ns,
        observation_key="observation.images.top",
        stream_id="top",
    )
