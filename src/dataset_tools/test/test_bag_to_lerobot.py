"""Tests for bag_to_lerobot helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.bag_to_lerobot import (  # noqa: E402
    _estimate_stream_rate_hz,
    _resolve_video_codec,
    _selected_indices_for_ticks,
)


def test_resolve_video_codec_prefers_h264_in_auto_mode(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        return DummyCodec(is_encoder=name == "h264")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "h264"


def test_resolve_video_codec_falls_back_to_av1_when_h264_missing(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        if name == "h264":
            raise ValueError("missing")
        return DummyCodec(is_encoder=name == "libsvtav1")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "libsvtav1"


def test_estimate_stream_rate_hz_uses_timestamp_span():
    ts = [0, 33_333_333, 66_666_666, 100_000_000]

    assert abs(_estimate_stream_rate_hz(ts) - 30.0) < 0.05


def test_selected_indices_for_ticks_exposes_hold_duplicates_from_phase_offset():
    ts = np.array([0, 34, 68], dtype=np.int64)
    ticks = np.array([0, 33, 66], dtype=np.int64)

    selected = _selected_indices_for_ticks(
        policy="hold",
        ts_ns=ts,
        ticks_ns=ticks,
        step_ns=33,
        tol_ns=0,
    )

    assert selected.tolist() == [0, 0, 1]
