from __future__ import annotations

import numpy as np
import pytest

from inference_service.software_video_codec import (
    SoftwareH264Decoder,
    SoftwareH264Encoder,
    register_software_backend,
)
from inference_service.video_codec import CodecLifecycleState, VideoCodecError, VideoCodecRegistry, VideoFrame

pytest.importorskip("av")


def test_software_h264_round_trip_preserves_frame_metadata_and_timestamps():
    encoder = _encoder(gop_frames=2)
    decoder = SoftwareH264Decoder()
    decoded = []

    for index in range(4):
        timestamp_ns = 1_000_000_000 + index * 50_000_000
        image = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        packets = encoder.encode(VideoFrame(image, timestamp_ns, timestamp_ns, 64, 48, "rgb24"))
        for packet in packets:
            decoded.extend(decoder.decode(packet))

    assert len(decoded) == 4
    assert [frame.capture_timestamp_ns for frame in decoded] == [
        1_000_000_000,
        1_050_000_000,
        1_100_000_000,
        1_150_000_000,
    ]
    assert all(frame.data.shape == (48, 64, 3) for frame in decoded)
    assert all(frame.pixel_format == "rgb24" for frame in decoded)
    assert all(frame.color_space == "bt709" and frame.color_range == "limited" for frame in decoded)
    assert decoded[0].keyframe is True
    assert np.max(np.abs(decoded[-1].data.astype(np.int16) - 120)) <= 2
    assert encoder.metrics.input_frames == 4
    assert encoder.metrics.output_frames == 4
    assert decoder.metrics.output_frames == 4

    encoder.close()
    decoder.close()
    assert encoder.state is CodecLifecycleState.CLOSED
    assert decoder.state is CodecLifecycleState.CLOSED


def test_software_encoder_rejects_non_monotonic_timestamps():
    encoder = _encoder()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = VideoFrame(image, 1_000, 1_000, 64, 48, "rgb24")
    encoder.encode(frame)

    with pytest.raises(VideoCodecError) as error:
        encoder.encode(frame)

    assert error.value.code == "non_monotonic_timestamp"
    encoder.close()


def test_software_encoder_repeats_headers_at_bounded_gop_keyframes():
    encoder = _encoder(gop_frames=2)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    packets = []
    for index in range(5):
        timestamp_ns = 1_000_000_000 + index * 50_000_000
        packets.extend(encoder.encode(VideoFrame(image, timestamp_ns, timestamp_ns, 64, 48, "rgb24")))

    keyframes = [packet for packet in packets if packet.keyframe]

    assert len(keyframes) == 3
    for packet in keyframes:
        nal_types = _annex_b_nal_types(packet.payload)
        assert 7 in nal_types
        assert 8 in nal_types
        assert 5 in nal_types
    encoder.close()


def test_software_decoder_reports_corrupt_h264_as_recoverable_error():
    from inference_service.video_codec import EncodedPacket

    decoder = SoftwareH264Decoder()

    with pytest.raises(VideoCodecError) as error:
        decoder.decode(EncodedPacket(b"not-h264", 90, 1_000_000))

    assert error.value.code == "decode_failed"
    assert error.value.recoverable is True
    assert decoder.metrics.errors == 1
    decoder.close()


def test_software_codec_reset_allows_new_timestamp_epoch():
    encoder = _encoder()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    encoder.encode(VideoFrame(image, 2_000, 2_000, 64, 48, "rgb24"))

    encoder.reset()
    packets = encoder.encode(VideoFrame(image, 1_000, 1_000, 64, 48, "rgb24"))

    assert packets
    encoder.close()


def test_software_backend_registers_without_eager_optional_import():
    registry = VideoCodecRegistry()
    register_software_backend(registry)

    encoder = registry.resolve("software", "encoder")
    decoder = registry.resolve("auto", "decoder")

    assert encoder.name == "software"
    assert decoder.name == "software"


def _encoder(*, gop_frames: int = 15) -> SoftwareH264Encoder:
    return SoftwareH264Encoder(
        width=64,
        height=48,
        frame_rate_hz=20.0,
        bitrate_bps=200_000,
        gop_frames=gop_frames,
    )


def _annex_b_nal_types(payload: bytes) -> list[int]:
    starts = []
    index = 0
    while index < len(payload) - 3:
        if payload[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append(index + 4)
            index += 4
        elif payload[index : index + 3] == b"\x00\x00\x01":
            starts.append(index + 3)
            index += 3
        else:
            index += 1
    return [payload[start] & 0x1F for start in starts if start < len(payload)]
