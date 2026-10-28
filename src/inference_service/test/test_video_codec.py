from __future__ import annotations

import pytest

from inference_service.video_codec import (
    CodecCapabilities,
    EncodedPacket,
    VideoCodecError,
    VideoCodecRegistry,
    VideoFrame,
)


def test_video_values_validate_timestamps_dimensions_and_payloads():
    frame = VideoFrame(object(), 10, 20, 640, 480, "nv12")
    packet = EncodedPacket(b"payload", 90_000, 10, keyframe=True)

    assert frame.color_range == "limited"
    assert packet.keyframe is True
    with pytest.raises(ValueError, match="dimensions"):
        VideoFrame(object(), 10, 20, 0, 480, "nv12")
    with pytest.raises(ValueError, match="payload"):
        EncodedPacket(b"", 0, 10)


def test_explicit_backend_failure_does_not_fall_back():
    registry = VideoCodecRegistry()
    registry.register(
        "ascend",
        priority=100,
        probe=lambda _kind: None,
        encoder_factory=lambda **_options: object(),
    )
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(),
        encoder_factory=lambda **_options: object(),
    )

    with pytest.raises(VideoCodecError) as error:
        registry.resolve("ascend", "encoder")

    assert error.value.code == "backend_unavailable"
    assert error.value.backend == "ascend"


def test_auto_resolution_is_priority_then_name_and_reports_probes():
    registry = VideoCodecRegistry()
    registry.register(
        "nvidia",
        priority=100,
        probe=lambda _kind: CodecCapabilities(hardware_accelerated=True),
        encoder_factory=lambda **_options: "nvidia",
    )
    registry.register(
        "ascend",
        priority=100,
        probe=lambda _kind: CodecCapabilities(hardware_accelerated=True),
        encoder_factory=lambda **_options: "ascend",
    )

    resolved = registry.resolve("auto", "encoder")

    assert resolved.name == "ascend"
    assert resolved.create() == "ascend"
    assert [probe.backend for probe in resolved.probes] == ["ascend"]


def test_auto_resolution_falls_back_to_software_after_isolated_probe_error():
    registry = VideoCodecRegistry()

    def missing_optional_sdk(_kind):
        raise ImportError("optional hardware SDK missing")

    registry.register(
        "ascend",
        priority=100,
        probe=missing_optional_sdk,
        decoder_factory=lambda **_options: "ascend",
    )
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("yuv420p", "nv12")),
        decoder_factory=lambda **_options: "software",
    )

    resolved = registry.resolve("auto", "decoder")

    assert resolved.name == "software"
    assert resolved.create() == "software"
    assert [(probe.backend, probe.available) for probe in resolved.probes] == [
        ("ascend", False),
        ("software", True),
    ]
    assert "ImportError" in resolved.probes[0].reason


def test_auto_resolution_reports_all_failures_when_no_backend_is_available():
    registry = VideoCodecRegistry()
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: None,
        encoder_factory=lambda **_options: object(),
    )

    with pytest.raises(VideoCodecError) as error:
        registry.resolve("auto", "encoder")

    assert error.value.code == "no_backend_available"
    assert error.value.details["probes"] == (("software", "probe reported unavailable"),)


def test_backend_without_requested_codec_direction_is_unavailable():
    registry = VideoCodecRegistry()
    registry.register(
        "encode_only",
        priority=10,
        probe=lambda _kind: CodecCapabilities(),
        encoder_factory=lambda **_options: object(),
    )

    with pytest.raises(VideoCodecError, match="decoder is not implemented"):
        registry.resolve("encode_only", "decoder")
