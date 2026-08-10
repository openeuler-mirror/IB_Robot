"""Optional PyAV-backed NVIDIA NVENC H.264 encoder and CUVID decoder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np

from observation_transport.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    CodecMetrics,
    EncodedPacket,
    VideoCodecError,
    VideoCodecRegistry,
    VideoDecoder,
    VideoEncoder,
    VideoFrame,
)

_BACKEND = "nvidia"
_RTP_TIME_BASE = Fraction(1, 90_000)
_BT709 = 1
_COLOR_RANGE_MPEG = 1
_COLOR_RANGE_JPEG = 2


def probe_nvidia_codec(kind: str) -> CodecCapabilities | None:
    """Probe codec presence and a real NVENC/CUVID session without requiring CUDA Python."""
    try:
        import av

        if kind == "encoder":
            av.codec.Codec("h264_nvenc", "w")
            codec = _create_nvenc_context(
                av,
                width=640,
                height=480,
                frame_rate_hz=30,
                bitrate_bps=1_000_000,
                gop_frames=15,
                profile="main",
                color_range="limited",
            )
            _close_codec_context(codec)
        elif kind == "decoder":
            av.codec.Codec("h264_cuvid", "r")
            codec = _create_cuvid_context(av, width=640, height=480, output_pixel_format="rgb24")
            _close_codec_context(codec)
        else:
            return None
    except (ImportError, LookupError, OSError, ValueError):
        return None
    return CodecCapabilities(pixel_formats=("rgb24", "bgr24"), hardware_accelerated=True)


def register_nvidia_backend(registry: VideoCodecRegistry, *, priority: int = 90) -> None:
    registry.register(
        _BACKEND,
        priority=priority,
        probe=probe_nvidia_codec,
        encoder_factory=NvidiaH264Encoder,
        decoder_factory=NvidiaH264Decoder,
    )


class NvidiaH264Encoder(VideoEncoder):
    """Low-latency NVENC adapter that emits one Annex-B access unit per frame."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        frame_rate_hz: float,
        bitrate_bps: int,
        gop_frames: int,
        input_pixel_format: str = "rgb24",
        profile: str = "main",
        color_space: str = "bt709",
        color_range: str = "limited",
        av_loader: Callable[[], Any] | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("NVIDIA H.264 requires positive even dimensions")
        if frame_rate_hz <= 0 or bitrate_bps <= 0 or gop_frames <= 0:
            raise ValueError("frame rate, bitrate, and GOP must be positive")
        if input_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("NVIDIA encoder input_pixel_format must be rgb24 or bgr24")
        if profile not in {"baseline", "main", "high"}:
            raise ValueError("NVIDIA encoder profile must be baseline, main, or high")
        if color_space != "bt709" or color_range not in {"limited", "full"}:
            raise ValueError("NVIDIA H.264 supports BT.709 limited or full range")
        try:
            self._av = av_loader() if av_loader is not None else _load_av()
        except ImportError as exc:
            raise VideoCodecError("backend_unavailable", str(exc), backend=_BACKEND) from exc

        self._width = int(width)
        self._height = int(height)
        self._frame_rate_hz = float(frame_rate_hz)
        self._bitrate_bps = int(bitrate_bps)
        self._gop_frames = int(gop_frames)
        self._input_pixel_format = input_pixel_format
        self._profile = profile
        self._color_range = color_range
        self._state = CodecLifecycleState.CREATED
        self._metrics = CodecMetrics()
        self._last_capture_timestamp_ns = -1
        try:
            self._codec = self._create_codec()
        except Exception as exc:
            self._state = CodecLifecycleState.FAILED
            self._metrics = replace(self._metrics, errors=1)
            raise VideoCodecError("backend_unavailable", str(exc), backend=_BACKEND) from exc
        self._state = CodecLifecycleState.RUNNING

    @property
    def state(self) -> CodecLifecycleState:
        return self._state

    @property
    def metrics(self) -> CodecMetrics:
        return self._metrics

    def encode(self, frame: VideoFrame) -> list[EncodedPacket]:
        self._require_running()
        if frame.capture_timestamp_ns <= self._last_capture_timestamp_ns:
            raise VideoCodecError(
                "non_monotonic_timestamp",
                "NVIDIA encoder requires monotonically increasing capture timestamps",
                backend=_BACKEND,
            )
        array = np.asarray(frame.data)
        if array.shape != (self._height, self._width, 3) or array.dtype != np.uint8:
            raise VideoCodecError(
                "invalid_frame",
                f"expected uint8 HWC frame {(self._height, self._width, 3)}, got {array.shape} {array.dtype}",
                backend=_BACKEND,
            )
        if frame.pixel_format != self._input_pixel_format:
            raise VideoCodecError(
                "invalid_pixel_format",
                f"expected {self._input_pixel_format}, got {frame.pixel_format}",
                backend=_BACKEND,
            )
        if frame.color_space.lower() != "bt709" or frame.color_range.lower() != self._color_range:
            raise VideoCodecError(
                "unsupported_color",
                f"expected BT.709 {self._color_range}-range input",
                backend=_BACKEND,
            )

        try:
            av_frame = self._av.VideoFrame.from_ndarray(array, format=self._input_pixel_format)
            av_frame.pts = _capture_ns_to_rtp_ticks(frame.capture_timestamp_ns)
            av_frame.time_base = _RTP_TIME_BASE
            packets = self._codec.encode(av_frame)
        except Exception as exc:
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            raise VideoCodecError("encode_failed", str(exc), backend=_BACKEND) from exc
        if not packets:
            self._state = CodecLifecycleState.FAILED
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            raise VideoCodecError(
                "encode_delay",
                "NVENC produced no packet despite zero-latency configuration",
                backend=_BACKEND,
            )
        self._last_capture_timestamp_ns = frame.capture_timestamp_ns
        output = [
            EncodedPacket(
                bytes(packet),
                int(packet.pts if packet.pts is not None else av_frame.pts) & 0xFFFF_FFFF,
                frame.capture_timestamp_ns,
                keyframe=bool(packet.is_keyframe),
            )
            for packet in packets
        ]
        self._metrics = replace(
            self._metrics,
            input_frames=self._metrics.input_frames + 1,
            output_frames=self._metrics.output_frames + 1,
            output_packets=self._metrics.output_packets + len(output),
        )
        return output

    def reset(self) -> None:
        self._require_not_closed()
        _close_codec_context(self._codec)
        try:
            self._codec = self._create_codec()
        except Exception as exc:
            self._state = CodecLifecycleState.FAILED
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            raise VideoCodecError("backend_unavailable", str(exc), backend=_BACKEND) from exc
        self._last_capture_timestamp_ns = -1
        self._state = CodecLifecycleState.RUNNING

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        if self._state is CodecLifecycleState.CLOSED:
            return
        try:
            self._codec.encode(None)
        finally:
            _close_codec_context(self._codec)
            self._state = CodecLifecycleState.CLOSED

    def _create_codec(self):
        return _create_nvenc_context(
            self._av,
            width=self._width,
            height=self._height,
            frame_rate_hz=self._frame_rate_hz,
            bitrate_bps=self._bitrate_bps,
            gop_frames=self._gop_frames,
            profile=self._profile,
            color_range=self._color_range,
        )

    def _require_running(self) -> None:
        if self._state is not CodecLifecycleState.RUNNING:
            raise VideoCodecError("invalid_state", f"encoder is {self._state.value}", backend=_BACKEND)

    def _require_not_closed(self) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            raise VideoCodecError("invalid_state", "encoder is closed", backend=_BACKEND)


def _create_nvenc_context(
    av: Any,
    *,
    width: int,
    height: int,
    frame_rate_hz: float,
    bitrate_bps: int,
    gop_frames: int,
    profile: str,
    color_range: str,
):
    codec = av.CodecContext.create("h264_nvenc", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = "nv12"
    codec.bit_rate = bitrate_bps
    codec.time_base = _RTP_TIME_BASE
    codec.framerate = Fraction(frame_rate_hz).limit_denominator(1000)
    codec.gop_size = gop_frames
    codec.max_b_frames = 0
    codec.color_primaries = _BT709
    codec.color_trc = _BT709
    codec.colorspace = _BT709
    codec.color_range = _COLOR_RANGE_JPEG if color_range == "full" else _COLOR_RANGE_MPEG
    codec.options = {
        "preset": "p1",
        "tune": "ull",
        "profile": profile,
        "rc": "cbr",
        "zerolatency": "1",
        "delay": "0",
        "forced-idr": "1",
        "rc-lookahead": "0",
        "spatial-aq": "0",
        "temporal-aq": "0",
    }
    codec.open()
    return codec


def _capture_ns_to_rtp_ticks(capture_timestamp_ns: int) -> int:
    return round(capture_timestamp_ns * 90_000 / 1_000_000_000)


def _load_av():
    import av

    return av


def _close_codec_context(codec: Any) -> None:
    close = getattr(codec, "close", None)
    if close is not None:
        close()


class NvidiaH264Decoder(VideoDecoder):
    """Hardware-accelerated CUVID H.264 decoder."""

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        frame_rate_hz: float | None = None,
        output_pixel_format: str = "rgb24",
        color_space: str = "bt709",
        color_range: str = "limited",
    ) -> None:
        if output_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("nvidia decoder output_pixel_format must be rgb24 or bgr24")
        if width is not None and width <= 0 or height is not None and height <= 0:
            raise ValueError("width and height must be positive")
        if color_space.lower() != "bt709":
            raise ValueError("nvidia decoder only supports bt709 color space")
        if color_range not in {"limited", "full"}:
            raise ValueError("nvidia decoder color_range must be limited or full")
        try:
            av = _load_av()
        except ImportError as exc:
            raise VideoCodecError("backend_unavailable", str(exc), backend=_BACKEND) from exc
        self._av = av
        self._expected_size = (int(width), int(height)) if width is not None and height is not None else None
        self._output_pixel_format = output_pixel_format
        self._color_range = color_range
        self._codec = _create_cuvid_context(
            av,
            width=self._expected_size[0] if self._expected_size else 1920,
            height=self._expected_size[1] if self._expected_size else 1080,
            output_pixel_format=output_pixel_format,
        )
        self._state = CodecLifecycleState.RUNNING
        self._metrics = CodecMetrics()
        self._capture_timestamps: dict[int, int] = {}

    @property
    def state(self) -> CodecLifecycleState:
        return self._state

    @property
    def metrics(self) -> CodecMetrics:
        return self._metrics

    def decode(self, packet: EncodedPacket) -> list[VideoFrame]:
        if self._state is not CodecLifecycleState.RUNNING:
            raise VideoCodecError("invalid_state", f"decoder is {self._state.value}", backend=_BACKEND)
        try:
            av_packet = self._av.Packet(packet.payload)
            av_packet.pts = packet.rtp_timestamp
            av_packet.dts = packet.rtp_timestamp
            av_packet.time_base = _RTP_TIME_BASE
            self._capture_timestamps[packet.rtp_timestamp] = packet.capture_timestamp_ns
            decoded = self._codec.decode(av_packet)
            frames = [self._convert_frame(frame, packet.capture_timestamp_ns) for frame in decoded]
        except Exception as exc:
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            raise VideoCodecError("decode_failed", str(exc), backend=_BACKEND, recoverable=True) from exc
        self._metrics = replace(
            self._metrics,
            output_frames=self._metrics.output_frames + len(frames),
            output_packets=self._metrics.output_packets + 1,
        )
        return frames

    def reset(self) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            raise VideoCodecError("invalid_state", "decoder is closed", backend=_BACKEND)
        _close_codec_context(self._codec)
        self._codec = _create_cuvid_context(
            self._av,
            width=self._expected_size[0] if self._expected_size else 1920,
            height=self._expected_size[1] if self._expected_size else 1080,
            output_pixel_format=self._output_pixel_format,
        )
        self._capture_timestamps.clear()

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        if self._state is CodecLifecycleState.CLOSED:
            return
        try:
            self._codec.decode(None)
        except Exception:
            pass
        finally:
            _close_codec_context(self._codec)
            self._capture_timestamps.clear()
            self._state = CodecLifecycleState.CLOSED

    def _convert_frame(self, frame: Any, fallback_capture_timestamp_ns: int) -> VideoFrame:
        rtp_timestamp = int(frame.pts) if frame.pts is not None else None
        capture_timestamp_ns = self._capture_timestamps.pop(rtp_timestamp, fallback_capture_timestamp_ns)
        array = frame.to_ndarray(format=self._output_pixel_format)
        if self._expected_size and (frame.width, frame.height) != self._expected_size:
            raise VideoCodecError(
                "invalid_dimensions",
                f"decoded frame {(frame.width, frame.height)} does not match {self._expected_size}",
                backend=_BACKEND,
            )
        return VideoFrame(
            np.ascontiguousarray(array),
            capture_timestamp_ns,
            capture_timestamp_ns,
            frame.width,
            frame.height,
            self._output_pixel_format,
            color_space="bt709",
            color_range=self._color_range,
            keyframe=bool(frame.key_frame),
        )


def _create_cuvid_context(av: Any, *, width: int, height: int, output_pixel_format: str):
    """Create a CUVID decoder context for H.264."""
    codec = av.CodecContext.create("h264_cuvid", "r")
    codec.width = width
    codec.height = height
    codec.pix_fmt = output_pixel_format
    # CUVID otherwise buffers several frames before returning the first image.
    # A request-driven robot loop cannot produce those future frames until the
    # first observation is decoded, so force FFmpeg's low-delay path.
    codec.flags |= av.codec.context.Flags.low_delay
    codec.open()
    return codec
