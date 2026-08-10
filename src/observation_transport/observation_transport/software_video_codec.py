"""PyAV-backed low-latency H.264 codec implementation."""

from __future__ import annotations

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

_RTP_TIME_BASE = Fraction(1, 90_000)


def probe_software_codec(kind: str) -> CodecCapabilities | None:
    try:
        import av

        av.codec.Codec("libx264" if kind == "encoder" else "h264", "w" if kind == "encoder" else "r")
    except (ImportError, LookupError):
        return None
    return CodecCapabilities(pixel_formats=("rgb24", "bgr24"))


def register_software_backend(registry: VideoCodecRegistry, *, priority: int = 0) -> None:
    registry.register(
        "software",
        priority=priority,
        probe=probe_software_codec,
        encoder_factory=SoftwareH264Encoder,
        decoder_factory=SoftwareH264Decoder,
    )


class SoftwareH264Encoder(VideoEncoder):
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
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("software H.264 requires positive even dimensions")
        if frame_rate_hz <= 0 or bitrate_bps <= 0 or gop_frames <= 0:
            raise ValueError("frame rate, bitrate, and GOP must be positive")
        if input_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("software H.264 input_pixel_format must be rgb24 or bgr24")
        if profile not in {"baseline", "main", "high"}:
            raise ValueError("software H.264 profile must be baseline, main, or high")
        if color_space != "bt709" or color_range not in {"limited", "full"}:
            raise ValueError("software H.264 supports BT.709 limited or full range")
        try:
            import av
        except ImportError as exc:
            raise VideoCodecError("backend_unavailable", str(exc), backend="software") from exc

        self._av = av
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
        self._codec = self._create_codec()
        self._state = CodecLifecycleState.RUNNING

    def _create_codec(self):
        codec = self._av.CodecContext.create("libx264", "w")
        codec.width = self._width
        codec.height = self._height
        codec.pix_fmt = "yuv420p"
        codec.bit_rate = self._bitrate_bps
        codec.time_base = _RTP_TIME_BASE
        codec.framerate = Fraction(self._frame_rate_hz).limit_denominator(1000)
        codec.gop_size = self._gop_frames
        codec.options = {
            "profile": self._profile,
            "preset": "ultrafast",
            "tune": "zerolatency",
            "x264-params": (
                f"keyint={self._gop_frames}:min-keyint={self._gop_frames}:scenecut=0:bframes=0:repeat-headers=1:"
                f"fullrange={'on' if self._color_range == 'full' else 'off'}:"
                "colorprim=bt709:transfer=bt709:colormatrix=bt709"
            ),
        }
        codec.open()
        return codec

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
                "software encoder requires monotonically increasing capture timestamps",
                backend="software",
                details={
                    "capture_timestamp_ns": frame.capture_timestamp_ns,
                    "previous_timestamp_ns": self._last_capture_timestamp_ns,
                },
            )
        array = np.asarray(frame.data)
        if array.shape != (self._height, self._width, 3) or array.dtype != np.uint8:
            raise VideoCodecError(
                "invalid_frame",
                f"expected uint8 HWC frame {(self._height, self._width, 3)}, got {array.shape} {array.dtype}",
                backend="software",
            )
        if frame.pixel_format != self._input_pixel_format:
            raise VideoCodecError(
                "invalid_pixel_format",
                f"expected {self._input_pixel_format}, got {frame.pixel_format}",
                backend="software",
            )

        try:
            av_frame = self._av.VideoFrame.from_ndarray(array, format=self._input_pixel_format)
            av_frame.pts = _capture_ns_to_rtp_ticks(frame.capture_timestamp_ns)
            av_frame.time_base = _RTP_TIME_BASE
            packets = self._codec.encode(av_frame)
        except Exception as exc:
            self._record_error()
            raise VideoCodecError("encode_failed", str(exc), backend="software") from exc
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
            output_frames=self._metrics.output_frames + int(bool(output)),
            output_packets=self._metrics.output_packets + len(output),
        )
        return output

    def reset(self) -> None:
        self._require_not_closed()
        _close_codec_context(self._codec)
        self._codec = self._create_codec()
        self._last_capture_timestamp_ns = -1

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

    def _record_error(self) -> None:
        self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)

    def _require_running(self) -> None:
        if self._state is not CodecLifecycleState.RUNNING:
            raise VideoCodecError("invalid_state", f"encoder is {self._state.value}", backend="software")

    def _require_not_closed(self) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            raise VideoCodecError("invalid_state", "encoder is closed", backend="software")


class SoftwareH264Decoder(VideoDecoder):
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
            raise ValueError("software decoder output_pixel_format must be rgb24 or bgr24")
        if width is not None and width <= 0 or height is not None and height <= 0:
            raise ValueError("software decoder dimensions must be positive")
        if frame_rate_hz is not None and frame_rate_hz <= 0:
            raise ValueError("software decoder frame rate must be positive")
        if color_space != "bt709" or color_range not in {"limited", "full"}:
            raise ValueError("software H.264 supports BT.709 limited or full range")
        try:
            import av
        except ImportError as exc:
            raise VideoCodecError("backend_unavailable", str(exc), backend="software") from exc
        self._av = av
        self._output_pixel_format = output_pixel_format
        self._expected_size = (width, height) if width is not None and height is not None else None
        self._color_range = color_range
        self._codec = self._create_codec()
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
            raise VideoCodecError("invalid_state", f"decoder is {self._state.value}", backend="software")
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
            raise VideoCodecError("decode_failed", str(exc), backend="software", recoverable=True) from exc
        self._metrics = replace(
            self._metrics,
            output_frames=self._metrics.output_frames + len(frames),
            output_packets=self._metrics.output_packets + 1,
        )
        return frames

    def reset(self) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            raise VideoCodecError("invalid_state", "decoder is closed", backend="software")
        _close_codec_context(self._codec)
        self._codec = self._create_codec()
        self._capture_timestamps.clear()

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        if self._state is CodecLifecycleState.CLOSED:
            return
        try:
            self._codec.decode(None)
        finally:
            _close_codec_context(self._codec)
            self._capture_timestamps.clear()
            self._state = CodecLifecycleState.CLOSED

    def _create_codec(self):
        codec = self._av.CodecContext.create("h264", "r")
        codec.open()
        return codec

    def _convert_frame(self, frame: Any, fallback_capture_timestamp_ns: int) -> VideoFrame:
        rtp_timestamp = int(frame.pts) if frame.pts is not None else None
        capture_timestamp_ns = self._capture_timestamps.pop(rtp_timestamp, fallback_capture_timestamp_ns)
        array = frame.to_ndarray(format=self._output_pixel_format)
        if self._expected_size is not None and (frame.width, frame.height) != self._expected_size:
            raise VideoCodecError(
                "invalid_dimensions",
                f"decoded frame {(frame.width, frame.height)} does not match {self._expected_size}",
                backend="software",
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


def _capture_ns_to_rtp_ticks(capture_timestamp_ns: int) -> int:
    return round(capture_timestamp_ns * 90_000 / 1_000_000_000)


def _close_codec_context(codec: Any) -> None:
    close = getattr(codec, "close", None)
    if close is not None:
        close()
