from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import inference_service.nvidia_video_codec as nvidia_codec
from inference_service.nvidia_video_codec import (
    NvidiaH264Decoder,
    NvidiaH264Encoder,
    probe_nvidia_codec,
    register_nvidia_backend,
)
from inference_service.software_video_codec import SoftwareH264Decoder
from inference_service.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    VideoCodecError,
    VideoCodecRegistry,
    VideoFrame,
)


class _Packet:
    def __init__(self, payload: bytes, pts: int, *, keyframe: bool) -> None:
        self._payload = payload
        self.pts = pts
        self.is_keyframe = keyframe

    def __bytes__(self) -> bytes:
        return self._payload


class _Codec:
    def __init__(self) -> None:
        self.closed = False
        self.frames = []

    def encode(self, frame):
        if frame is None:
            return []
        self.frames.append(frame)
        return [_Packet(b"\x00\x00\x00\x01\x65nvenc", frame.pts, keyframe=True)]

    def close(self) -> None:
        self.closed = True


class _Frame:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.pts = None
        self.time_base = None


class _Av:
    VideoFrame = SimpleNamespace(from_ndarray=lambda array, format: _Frame(array))


def test_module_import_does_not_eagerly_load_pyav_or_cuda():
    script = (
        "import sys; import inference_service.nvidia_video_codec; "
        "assert 'av' not in sys.modules; assert 'pycuda' not in sys.modules; assert 'cuda.bindings' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)

    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_nvidia_registration_includes_encoder_and_decoder_with_auto_priority(monkeypatch):
    monkeypatch.setattr(
        nvidia_codec,
        "probe_nvidia_codec",
        lambda kind: CodecCapabilities(hardware_accelerated=True),
    )
    registry = VideoCodecRegistry()
    register_nvidia_backend(registry)
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(),
        encoder_factory=lambda **_options: object(),
        decoder_factory=lambda **_options: object(),
    )

    assert registry.resolve("auto", "encoder").name == "nvidia"
    assert registry.resolve("auto", "decoder").name == "nvidia"
    assert registry.resolve("nvidia", "encoder").name == "nvidia"
    assert registry.resolve("nvidia", "decoder").name == "nvidia"


def test_nvidia_encoder_preserves_timestamp_lifecycle_and_creation_options(monkeypatch):
    codecs = []
    creations = []

    def create(_av, **options):
        creations.append(options)
        codec = _Codec()
        codecs.append(codec)
        return codec

    monkeypatch.setattr(nvidia_codec, "_create_nvenc_context", create)
    encoder = NvidiaH264Encoder(
        width=640,
        height=480,
        frame_rate_hz=30,
        bitrate_bps=4_000_000,
        gop_frames=15,
        profile="high",
        av_loader=lambda: _Av,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    packets = encoder.encode(VideoFrame(image, 1_000_000_000, 1_000_000_000, 640, 480, "rgb24"))

    assert creations[0] == {
        "width": 640,
        "height": 480,
        "frame_rate_hz": 30.0,
        "bitrate_bps": 4_000_000,
        "gop_frames": 15,
        "profile": "high",
        "color_range": "limited",
    }
    assert packets[0].rtp_timestamp == 90_000
    assert packets[0].capture_timestamp_ns == 1_000_000_000
    assert packets[0].keyframe is True
    assert encoder.metrics.input_frames == encoder.metrics.output_frames == 1

    encoder.reset()
    assert codecs[0].closed is True
    assert encoder.state is CodecLifecycleState.RUNNING
    encoder.close()
    encoder.close()
    assert codecs[1].closed is True
    assert encoder.state is CodecLifecycleState.CLOSED


def test_nvidia_encoder_rejects_delayed_output(monkeypatch):
    codec = _Codec()
    codec.encode = lambda _frame: []
    monkeypatch.setattr(nvidia_codec, "_create_nvenc_context", lambda _av, **_options: codec)
    encoder = NvidiaH264Encoder(
        width=640,
        height=480,
        frame_rate_hz=30,
        bitrate_bps=4_000_000,
        gop_frames=15,
        av_loader=lambda: _Av,
    )

    with pytest.raises(VideoCodecError) as error:
        encoder.encode(VideoFrame(np.zeros((480, 640, 3), dtype=np.uint8), 1, 1, 640, 480, "rgb24"))

    assert error.value.code == "encode_delay"
    assert encoder.state is CodecLifecycleState.FAILED
    encoder.close()


@pytest.mark.skipif(probe_nvidia_codec("encoder") is None, reason="NVENC hardware is unavailable")
def test_nvenc_hardware_round_trip_repeats_headers_and_decodes_in_software():
    encoder = NvidiaH264Encoder(
        width=640,
        height=480,
        frame_rate_hz=30,
        bitrate_bps=4_000_000,
        gop_frames=2,
    )
    decoder = SoftwareH264Decoder(width=640, height=480, frame_rate_hz=30)
    decoded = []
    keyframe_types = []

    for index in range(5):
        timestamp_ns = 1_000_000_000 + index * 33_333_333
        image = np.full((480, 640, 3), index * 30, dtype=np.uint8)
        packets = encoder.encode(VideoFrame(image, timestamp_ns, timestamp_ns, 640, 480, "rgb24"))
        for packet in packets:
            if packet.keyframe:
                keyframe_types.append(_annex_b_nal_types(packet.payload))
            decoded.extend(decoder.decode(packet))

    assert len(decoded) == 5
    assert [frame.capture_timestamp_ns for frame in decoded] == [
        1_000_000_000 + index * 33_333_333 for index in range(5)
    ]
    assert all({5, 7, 8}.issubset(types) for types in keyframe_types)
    assert all(frame.data.shape == (480, 640, 3) for frame in decoded)
    encoder.close()
    decoder.close()


def _annex_b_nal_types(payload: bytes) -> set[int]:
    types = set()
    index = 0
    while index < len(payload) - 3:
        start_code = 4 if payload[index : index + 4] == b"\x00\x00\x00\x01" else 0
        if not start_code and payload[index : index + 3] == b"\x00\x00\x01":
            start_code = 3
        if start_code:
            types.add(payload[index + start_code] & 0x1F)
            index += start_code
        else:
            index += 1
    return types


@pytest.mark.skipif(probe_nvidia_codec("decoder") is None, reason="CUVID hardware is unavailable")
def test_cuvid_decoder_lifecycle_and_reset():
    decoder = NvidiaH264Decoder(width=640, height=480, output_pixel_format="rgb24")
    assert decoder.state is CodecLifecycleState.RUNNING
    assert decoder.metrics.output_frames == 0

    decoder.reset()
    assert decoder.state is CodecLifecycleState.RUNNING

    decoder.close()
    assert decoder.state is CodecLifecycleState.CLOSED
    decoder.close()  # idempotent


@pytest.mark.skipif(
    probe_nvidia_codec("encoder") is None or probe_nvidia_codec("decoder") is None,
    reason="NVENC or CUVID hardware is unavailable",
)
def test_nvenc_to_cuvid_hardware_round_trip_preserves_keyframes():
    encoder = NvidiaH264Encoder(
        width=640,
        height=480,
        frame_rate_hz=30,
        bitrate_bps=2_000_000,
        gop_frames=10,
    )
    decoder = NvidiaH264Decoder(width=640, height=480, output_pixel_format="rgb24")
    decoded = []

    for index in range(15):
        timestamp_ns = 1_000_000_000 + index * 33_333_333
        image = np.full((480, 640, 3), [index * 10, 128, 255 - index * 10], dtype=np.uint8)
        packets = encoder.encode(VideoFrame(image, timestamp_ns, timestamp_ns, 640, 480, "rgb24"))
        for packet in packets:
            decoded.extend(decoder.decode(packet))

    assert len(decoded) >= 10  # CUVID buffers initial frames
    assert all(frame.data.shape == (480, 640, 3) for frame in decoded)
    assert all(frame.width == 640 and frame.height == 480 for frame in decoded)
    assert any(frame.keyframe for frame in decoded)
    assert encoder.metrics.input_frames == 15
    assert decoder.metrics.output_frames == len(decoded)
    assert decoder.metrics.errors == 0

    encoder.close()
    decoder.close()
