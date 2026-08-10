"""Public direct-frame ingress contract for IB-Robot observation transport.

This module contains producer-facing protocol and immutable configuration/status
values only.  Codec, RTP packetization, UDP, worker, Benchmark, and Provider
implementations remain behind :func:`create_frame_ingress`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

QueuePolicy = Literal["strict", "latest"]


class FrameAdmissionDisposition(str, Enum):
    """Outcome of one non-blocking frame admission attempt."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class FrameIngressError(RuntimeError):
    """Structured failure raised by the public ingress boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        observation_key: str = "",
        stream_id: str = "",
        recoverable: bool = False,
    ) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("frame ingress error code must be a non-empty string")
        super().__init__(message)
        self.code = code
        self.observation_key = observation_key
        self.stream_id = stream_id
        self.recoverable = bool(recoverable)


DirectFrameProducerError = FrameIngressError


@dataclass(frozen=True, slots=True)
class FrameAdmissionReceipt:
    """Immediate bounded-queue admission result for one canonical frame."""

    disposition: FrameAdmissionDisposition
    admission_id: str
    observation_key: str
    stream_id: str
    capture_timestamp_ns: int
    session_generation: int
    admission_monotonic_ns: int
    queue_depth: int
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, FrameAdmissionDisposition):
            raise TypeError("disposition must be a FrameAdmissionDisposition")
        for value, label in (
            (self.observation_key, "observation_key"),
            (self.stream_id, "stream_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        for value, label in (
            (self.capture_timestamp_ns, "capture_timestamp_ns"),
            (self.session_generation, "session_generation"),
            (self.admission_monotonic_ns, "admission_monotonic_ns"),
            (self.queue_depth, "queue_depth"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative int")
        if self.capture_timestamp_ns == 0 or self.session_generation == 0 or self.admission_monotonic_ns == 0:
            raise ValueError("admission identity and timestamps must be positive")
        if self.disposition is FrameAdmissionDisposition.ACCEPTED:
            if not isinstance(self.admission_id, str) or not self.admission_id.strip():
                raise ValueError("accepted admission requires a transport-owned admission_id")
            if self.reason:
                raise ValueError("accepted admission must not carry a rejection reason")
        else:
            if self.admission_id:
                raise ValueError("rejected admission must not allocate an admission_id")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("rejected admission requires a reason")

    @property
    def accepted(self) -> bool:
        return self.disposition is FrameAdmissionDisposition.ACCEPTED


@dataclass(frozen=True, slots=True)
class FrameStreamSnapshot:
    """Transport-owned asynchronous progress for one observation stream."""

    observation_key: str
    stream_id: str
    ready: bool
    accepted_frames: int = 0
    encoded_frames: int = 0
    sent_frames: int = 0
    sent_packets: int = 0
    received_packets: int = 0
    decoded_frames: int = 0
    dropped_frames: int = 0
    queue_depth: int = 0
    queue_overflow_drops: int = 0
    last_accepted_admission_id: str = ""
    last_encoded_admission_id: str = ""
    last_sent_admission_id: str = ""
    last_dropped_admission_id: str = ""
    last_dropped_capture_timestamp_ns: int = 0
    dropped_capture_history: tuple[tuple[int, str, str], ...] = ()
    last_drop_reason: str = ""
    last_error: str = ""
    worker_alive: bool = False
    worker_failed: bool = False
    worker_last_progress_monotonic_ns: int = 0
    worker_last_progress_kind: str = ""
    recovery_requested: bool = False
    recovery_attempts: int = 0
    encode_latency_ns: int = 0
    send_latency_ns: int = 0

    def __post_init__(self) -> None:
        if not self.observation_key or not self.stream_id:
            raise ValueError("frame stream snapshot requires observation and stream identity")
        counters = (
            self.accepted_frames,
            self.encoded_frames,
            self.sent_frames,
            self.sent_packets,
            self.received_packets,
            self.decoded_frames,
            self.dropped_frames,
            self.queue_depth,
            self.queue_overflow_drops,
            self.last_dropped_capture_timestamp_ns,
            self.worker_last_progress_monotonic_ns,
            self.recovery_attempts,
            self.encode_latency_ns,
            self.send_latency_ns,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters):
            raise ValueError("frame stream counters must be non-negative ints")
        if not isinstance(self.worker_alive, bool) or not isinstance(self.worker_failed, bool):
            raise TypeError("frame stream worker state must use bool values")
        if not isinstance(self.recovery_requested, bool):
            raise TypeError("frame stream recovery_requested must be a bool")
        for item in self.dropped_capture_history:
            if (
                not isinstance(item, tuple)
                or len(item) != 3
                or not isinstance(item[0], int)
                or item[0] <= 0
                or not all(isinstance(value, str) for value in item[1:])
            ):
                raise ValueError("dropped_capture_history must contain (timestamp, admission_id, reason) tuples")


@dataclass(frozen=True, slots=True)
class FrameTransportSnapshot:
    """Read-only snapshot of native ingress session and stream progress."""

    pipeline_id: str
    session_id: str
    session_generation: int
    ready: bool
    streams: tuple[FrameStreamSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("frame transport snapshot requires pipeline_id")
        if self.session_generation < 0:
            raise ValueError("session_generation must be non-negative")
        if self.session_generation == 0 and self.session_id:
            raise ValueError("inactive snapshot cannot carry session_id")
        if self.session_generation > 0 and not self.session_id:
            raise ValueError("active snapshot requires session_id")
        if len({item.observation_key for item in self.streams}) != len(self.streams):
            raise ValueError("frame transport snapshot observation keys must be unique")


@runtime_checkable
class FrameIngress(Protocol):
    """Minimal public interface consumed by direct-frame observation sources."""

    @property
    def observation_keys(self) -> frozenset[str]: ...

    def ready(self) -> bool: ...

    def submit_frame(
        self,
        observation_key: str,
        frame: np.ndarray,
        *,
        capture_timestamp_ns: int,
        receive_timestamp_ns: int,
        pixel_format: str,
    ) -> FrameAdmissionReceipt: ...

    def snapshot(self) -> FrameTransportSnapshot: ...

    def close(self, timeout_s: float = 1.0) -> None: ...


@dataclass(frozen=True, slots=True)
class StreamSessionView:
    """Immutable identity for one active stream transport generation."""

    pipeline_id: str
    session_id: str
    generation: int
    contract_fingerprint: str
    deployment_fingerprint: str
    active: bool = True

    def __post_init__(self) -> None:
        if not self.pipeline_id or not self.contract_fingerprint or not self.deployment_fingerprint:
            raise ValueError("stream session requires pipeline and fingerprint identity")
        if self.active:
            if not self.session_id or self.generation < 1:
                raise ValueError("active stream session requires a session ID and positive generation")
        elif self.session_id or self.generation != 0:
            raise ValueError("inactive stream session cannot carry a live session identity")

    @classmethod
    def inactive(
        cls,
        *,
        pipeline_id: str,
        contract_fingerprint: str,
        deployment_fingerprint: str,
    ) -> StreamSessionView:
        return cls(
            pipeline_id=pipeline_id,
            session_id="",
            generation=0,
            contract_fingerprint=contract_fingerprint,
            deployment_fingerprint=deployment_fingerprint,
            active=False,
        )


@dataclass(frozen=True, slots=True)
class DirectFrameStreamConfig:
    """Materialized native ingress configuration for one image observation."""

    observation_key: str
    stream_id: str
    endpoint_host: str
    endpoint_port: int
    width: int
    height: int
    frame_rate_hz: float
    bitrate_bps: int = 4_000_000
    gop_frames: int = 15
    codec: str = "h264"
    codec_profile: str = "main"
    encoder_backend: str = "software"
    decoder_backend: str = "software"
    pixel_format: str = "nv12"
    input_pixel_format: str = "rgb24"
    color_space: str = "bt709"
    color_range: str = "limited"
    sender_queue_frames: int = 2
    raw_queue_frames: int | None = None
    queue_policy: QueuePolicy = "latest"
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.observation_key or not self.stream_id or not self.endpoint_host:
            raise ValueError("direct-frame stream identity must be non-empty")
        if not 1 <= self.endpoint_port <= 65535:
            raise ValueError("direct-frame endpoint port must be in 1..65535")
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("direct-frame dimensions must be positive and even")
        if self.frame_rate_hz <= 0 or self.bitrate_bps <= 0 or self.gop_frames <= 0:
            raise ValueError("direct-frame rate, bitrate, and GOP must be positive")
        if self.codec != "h264" or self.codec_profile not in {"baseline", "main", "high"}:
            raise ValueError("direct-frame ingress currently supports H.264 baseline/main/high")
        if self.input_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("direct-frame input pixel format must be rgb24 or bgr24")
        if self.sender_queue_frames <= 0:
            raise ValueError("direct-frame sender queue capacity must be positive")
        if self.raw_queue_frames is not None and self.raw_queue_frames <= 0:
            raise ValueError("direct-frame raw queue capacity must be positive")
        if self.queue_policy not in {"strict", "latest"}:
            raise ValueError("direct-frame queue policy must be strict or latest")

    @property
    def effective_raw_queue_frames(self) -> int:
        return self.sender_queue_frames if self.raw_queue_frames is None else self.raw_queue_frames


@dataclass(frozen=True, slots=True)
class DirectFrameStreamDescriptor:
    protocol_version: int
    pipeline_id: str
    session_id: str
    session_generation: int
    observation_key: str
    stream_id: str
    endpoint_host: str
    endpoint_port: int
    ssrc: int
    payload_type: int
    codec: str
    codec_profile: str
    width: int
    height: int
    frame_rate_hz: float
    rtp_clock_rate: int
    pixel_format: str
    color_space: str
    color_range: str
    encoder_backend: str
    contract_fingerprint: str
    deployment_fingerprint: str


@dataclass(frozen=True, slots=True)
class DirectFrameStreamStatus:
    """ROS-wire-compatible sender/receiver stream status value."""

    protocol_version: int
    pipeline_id: str
    session_id: str
    session_generation: int
    observation_key: str
    stream_id: str
    lifecycle_state: str
    ready: bool
    selected_backend: str
    status_origin: str
    timestamp_mapping_valid: bool = False
    mapping_rtp_timestamp: int = 0
    mapping_capture_timestamp_ns: int = 0
    keyframe_ready: bool = False
    encoded_frames: int = 0
    decoded_frames: int = 0
    sent_packets: int = 0
    received_packets: int = 0
    dropped_frames: int = 0
    dropped_packets: int = 0
    lost_packets: int = 0
    sender_queue_depth: int = 0
    receiver_queue_depth: int = 0
    decoded_buffer_depth: int = 0
    reconnect_count: int = 0
    sender_queue_overflow_drops: int = 0
    receiver_queue_overflow_drops: int = 0
    sequence_gap_events: int = 0
    reordered_packets: int = 0
    recovery_keyframes: int = 0
    jitter_ns: int = 0
    encode_start_monotonic_ns: int = 0
    encode_end_monotonic_ns: int = 0
    send_start_monotonic_ns: int = 0
    send_end_monotonic_ns: int = 0
    receive_monotonic_ns: int = 0
    decode_start_monotonic_ns: int = 0
    decode_end_monotonic_ns: int = 0
    last_decoded_capture_timestamp_ns: int = 0
    last_accepted_admission_id: str = ""
    last_encoded_admission_id: str = ""
    last_sent_admission_id: str = ""
    last_dropped_admission_id: str = ""
    last_dropped_capture_timestamp_ns: int = 0
    dropped_capture_history_json: str = "[]"
    last_drop_reason: str = ""
    last_error: str = ""


@dataclass(frozen=True, slots=True)
class PreparedDirectFrame:
    """Compatibility preparation object; commit performs admission only."""

    observation_key: str
    frame: np.ndarray
    capture_timestamp_ns: int
    receive_timestamp_ns: int
    provider_capture_timestamp_ns: int
    pixel_format: str
    session_generation: int
    prepare_start_monotonic_ns: int
    prepare_end_monotonic_ns: int


FrameSubmissionReceipt = FrameAdmissionReceipt


def create_frame_ingress(
    *,
    pipeline_id: str,
    contract_fingerprint: str,
    deployment_fingerprint: str,
    streams: Iterable[DirectFrameStreamConfig],
    codec_registry: Any | None = None,
    sender_factory: Callable[..., Any] | None = None,
    on_control_update: Callable[[], None] | None = None,
    protocol_version: int = 5,
) -> FrameIngress:
    """Create the single native production ingress without exposing internals."""

    from observation_transport.native_frame_ingress import NativeFrameIngress

    options: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "contract_fingerprint": contract_fingerprint,
        "deployment_fingerprint": deployment_fingerprint,
        "streams": streams,
        "codec_registry": codec_registry,
        "on_control_update": on_control_update,
        "protocol_version": protocol_version,
    }
    if sender_factory is not None:
        options["sender_factory"] = sender_factory
    return NativeFrameIngress(**options)


__all__ = [
    "DirectFrameProducerError",
    "DirectFrameStreamConfig",
    "DirectFrameStreamDescriptor",
    "DirectFrameStreamStatus",
    "FrameAdmissionDisposition",
    "FrameAdmissionReceipt",
    "FrameIngress",
    "FrameIngressError",
    "FrameStreamSnapshot",
    "FrameSubmissionReceipt",
    "FrameTransportSnapshot",
    "PreparedDirectFrame",
    "QueuePolicy",
    "StreamSessionView",
    "create_frame_ingress",
]
