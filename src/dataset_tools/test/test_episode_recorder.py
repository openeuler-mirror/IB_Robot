"""Tests for episode_recorder helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.episode_recorder import (  # noqa: E402
    _TopicCounter,
    _ensure_serialized_message,
    _normalize_max_cache_size,
    _topic_counter_diagnostics,
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
