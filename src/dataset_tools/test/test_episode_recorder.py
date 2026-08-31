"""Tests for episode_recorder helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.episode_recorder import (  # noqa: E402
    _ensure_serialized_message,
    _normalize_max_cache_size,
    _resolve_dataset_location,
    _topic_counter_diagnostics,
    _TopicCounter,
)


def test_normalize_max_cache_size_clamps_negative_values():
    assert _normalize_max_cache_size(-1) == 0
    assert _normalize_max_cache_size(1024) == 1024


def test_topic_counter_diagnostics_reports_drop_ratio_sorted_by_topic():
    counts = {
        "/camera/wrist/image_raw": _TopicCounter(seen=10, written=7),
        "/camera/front/image_raw": _TopicCounter(seen=0, written=0),
    }

    diagnostics = _topic_counter_diagnostics(counts)

    assert diagnostics[0] == ("/camera/front/image_raw", 0, 0, 0.0)
    assert diagnostics[1][:3] == ("/camera/wrist/image_raw", 10, 7)
    assert diagnostics[1][3] == pytest.approx(0.3)


def test_ensure_serialized_message_keeps_raw_bytes():
    payload = b"cdr-payload"

    assert _ensure_serialized_message(payload) == payload
    assert _ensure_serialized_message(bytearray(payload)) == payload
    assert _ensure_serialized_message(memoryview(payload)) == payload


def test_dataset_location_falls_back_to_recording_section():
    """RTP recording reaches this node without launch parameters.

    recording_node constructs EpisodeRecorderServer in-process, so the
    launch_builders path that passes bag_base_dir/dataset_name never runs.
    Without this fallback the recorder silently writes to /tmp, which the
    board wipes on reboot.
    """
    base, name = _resolve_dataset_location(
        bag_base_param="",
        dataset_name_param="",
        recording_section={
            "bag_base_dir": "/srv/rosbag/episodes",
            "dataset_name": "lekiwi_rtp_recording",
        },
        contract_robot_type="lekiwi",
        config_stem="lekiwi_navi_rtp_recording_ubuntu",
    )

    assert base == Path("/srv/rosbag/episodes")
    assert name == "lekiwi_rtp_recording"


def test_dataset_location_prefers_explicit_parameters_over_config():
    """launch_builders/recording.py passes both explicitly; it must still win."""
    base, name = _resolve_dataset_location(
        bag_base_param="/explicit/base",
        dataset_name_param="explicit_name",
        recording_section={
            "bag_base_dir": "/srv/rosbag/episodes",
            "dataset_name": "lekiwi_rtp_recording",
        },
        contract_robot_type="lekiwi",
        config_stem="stem",
    )

    assert base == Path("/explicit/base")
    assert name == "explicit_name"


def test_dataset_location_defaults_when_nothing_is_configured():
    base, name = _resolve_dataset_location(
        bag_base_param="",
        dataset_name_param="",
        recording_section={},
        contract_robot_type="lekiwi",
        config_stem="stem",
    )

    assert base == Path("/tmp/episodes")
    assert name == "lekiwi"


def test_dataset_location_sanitizes_the_resolved_name():
    _, name = _resolve_dataset_location(
        bag_base_param="",
        dataset_name_param="",
        recording_section={"dataset_name": "bad name/with slashes"},
        contract_robot_type="lekiwi",
        config_stem="stem",
    )

    assert name == "bad_name_with_slashes"
