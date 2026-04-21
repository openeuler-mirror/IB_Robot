"""Tests for rerun_viewer image handling helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset_tools.rerun_viewer import (  # noqa: E402
    _default_rerun_memory_limit,
    _downscale_image_for_rerun,
    _image_msg_to_numpy,
    _should_log_sample,
)


def test_image_msg_to_numpy_respects_row_step_padding_for_bgr8():
    width = 3
    height = 4
    channels = 3
    step = 12  # 9 bytes of pixels + 3 bytes of padding per row

    rows = []
    for y in range(height):
        pixels = []
        for x in range(width):
            base = y * 10 + x
            pixels.extend([base + 1, base + 2, base + 3])  # B, G, R
        rows.append(bytes(pixels + [255, 254, 253]))

    msg = SimpleNamespace(
        encoding="bgr8",
        width=width,
        height=height,
        step=step,
        is_bigendian=0,
        data=b"".join(rows),
    )

    arr = _image_msg_to_numpy(msg)

    assert arr is not None
    assert arr.shape == (height, width, channels)
    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr[0, 0], np.array([3, 2, 1], dtype=np.uint8))
    np.testing.assert_array_equal(arr[3, 2], np.array([35, 34, 33], dtype=np.uint8))


def test_image_msg_to_numpy_rejects_invalid_step():
    msg = SimpleNamespace(
        encoding="rgb8",
        width=4,
        height=2,
        step=8,
        is_bigendian=0,
        data=bytes(16),
    )

    assert _image_msg_to_numpy(msg) is None


def test_downscale_image_for_rerun_reduces_long_edge():
    arr = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    reduced = _downscale_image_for_rerun(arr, max_long_edge=4)

    assert reduced.shape == (3, 4, 3)
    assert reduced.dtype == np.uint8


def test_should_log_sample_enforces_min_interval():
    assert _should_log_sample(None, current_timestamp_s=1.0, min_interval_s=0.2)
    assert not _should_log_sample(1.0, current_timestamp_s=1.1, min_interval_s=0.2)
    assert _should_log_sample(1.0, current_timestamp_s=1.2, min_interval_s=0.2)


def test_default_rerun_memory_limit_uses_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LEROBOT_RERUN_MEMORY_LIMIT", "12%")
    assert _default_rerun_memory_limit() == "12%"


def test_default_rerun_memory_limit_falls_back_to_ten_percent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LEROBOT_RERUN_MEMORY_LIMIT", raising=False)
    assert _default_rerun_memory_limit() == "10%"
