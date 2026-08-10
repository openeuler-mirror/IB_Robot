"""DDS control-plane values and fail-closed video stream negotiation."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, fields
from types import MappingProxyType

from inference_service.distributed.types import PROTOCOL_VERSION, StreamReference


class StreamNegotiationError(RuntimeError):
    stage = "handshake"
    recoverable = False

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.details = MappingProxyType(details)


@dataclass(frozen=True, slots=True)
class VideoStreamDescriptor:
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

    def __post_init__(self) -> None:
        required = (
            self.pipeline_id,
            self.session_id,
            self.observation_key,
            self.stream_id,
            self.endpoint_host,
            self.codec,
            self.codec_profile,
            self.pixel_format,
            self.color_space,
            self.color_range,
            self.encoder_backend,
            self.contract_fingerprint,
            self.deployment_fingerprint,
        )
        if any(not value for value in required):
            raise ValueError("video stream descriptor string fields must be non-empty")
        if self.protocol_version < 1 or self.session_generation < 1:
            raise ValueError("descriptor protocol version and session generation must be positive")
        if not 1 <= self.endpoint_port <= 65535 or not 0 <= self.ssrc <= 0xFFFF_FFFF:
            raise ValueError("descriptor endpoint port or SSRC is invalid")
        if not 0 <= self.payload_type <= 127:
            raise ValueError("descriptor payload type must fit in 7 bits")
        if self.width <= 0 or self.height <= 0 or self.rtp_clock_rate <= 0:
            raise ValueError("descriptor dimensions and RTP clock rate must be positive")
        if not math.isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0:
            raise ValueError("descriptor frame rate must be finite and positive")


@dataclass(frozen=True, slots=True)
class VideoStreamRuntimeStatus:
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

    def __post_init__(self) -> None:
        if self.protocol_version < 1 or self.session_generation < 1:
            raise ValueError("stream status protocol version and session generation must be positive")
        if any(
            not value
            for value in (
                self.pipeline_id,
                self.session_id,
                self.observation_key,
                self.stream_id,
                self.lifecycle_state,
                self.selected_backend,
                self.status_origin,
            )
        ):
            raise ValueError("stream status identity, lifecycle, and origin fields must be non-empty")
        if self.status_origin not in {"sender", "receiver"}:
            raise ValueError("stream status origin must be sender or receiver")
        counters = (
            self.encoded_frames,
            self.decoded_frames,
            self.sent_packets,
            self.received_packets,
            self.dropped_frames,
            self.dropped_packets,
            self.lost_packets,
            self.sender_queue_depth,
            self.receiver_queue_depth,
            self.decoded_buffer_depth,
            self.reconnect_count,
            self.sender_queue_overflow_drops,
            self.receiver_queue_overflow_drops,
            self.sequence_gap_events,
            self.reordered_packets,
            self.recovery_keyframes,
            self.jitter_ns,
            self.encode_start_monotonic_ns,
            self.encode_end_monotonic_ns,
            self.send_start_monotonic_ns,
            self.send_end_monotonic_ns,
            self.receive_monotonic_ns,
            self.decode_start_monotonic_ns,
            self.decode_end_monotonic_ns,
            self.last_decoded_capture_timestamp_ns,
            self.last_dropped_capture_timestamp_ns,
        )
        if any(value < 0 for value in counters):
            raise ValueError("stream status counters cannot be negative")
        if self.timestamp_mapping_valid and self.mapping_capture_timestamp_ns <= 0:
            raise ValueError("valid timestamp mapping requires a positive capture timestamp")


VIDEO_DESCRIPTOR_FIELDS = tuple(f.name for f in fields(VideoStreamDescriptor))
VIDEO_STATUS_FIELDS = tuple(
    f.name for f in fields(VideoStreamRuntimeStatus) if f.name != "mapping_capture_timestamp_ns"
)


@dataclass(frozen=True, slots=True)
class VideoStreamDiagnosticSnapshot:
    observation_key: str
    stream_id: str
    mode: str
    configured_encoder_backend: str
    selected_encoder_backend: str
    configured_decoder_backend: str
    selected_decoder_backend: str
    endpoint: tuple[str, int]
    contract_fingerprint: str
    deployment_fingerprint: str
    security: str
    lifecycle_state: str
    ready: bool


@dataclass(frozen=True, slots=True)
class VideoTransportCapabilities:
    protocol_version: int = PROTOCOL_VERSION
    transport_modes: tuple[str, ...] = ("dds", "rtp")
    codecs: tuple[str, ...] = ("h264",)
    decoder_backends: tuple[str, ...] = ("software",)

    def __post_init__(self) -> None:
        if self.protocol_version < 1:
            raise ValueError("capability protocol version must be positive")
        if not self.transport_modes or not self.codecs or not self.decoder_backends:
            raise ValueError("video capabilities must declare transport modes, codecs, and decoder backends")
        if len(set(self.transport_modes)) != len(self.transport_modes):
            raise ValueError("video capabilities contain duplicate transport modes")
        if len(set(self.codecs)) != len(self.codecs) or len(set(self.decoder_backends)) != len(self.decoder_backends):
            raise ValueError("video capabilities contain duplicate codecs or decoder backends")


@dataclass(frozen=True, slots=True)
class VideoStreamRequirement:
    observation_key: str
    stream_id: str
    codec: str = "h264"
    decoder_backend: str = "auto"
    encoder_backend: str = "auto"
    codec_profile: str = "main"
    endpoint_host: str = ""
    endpoint_port: int = 0
    ssrc: int | None = None
    payload_type: int = 96
    width: int = 0
    height: int = 0
    frame_rate_hz: float = 0.0
    rtp_clock_rate: int = 90_000
    pixel_format: str = "nv12"
    color_space: str = "bt709"
    color_range: str = "limited"

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.observation_key,
                self.stream_id,
                self.codec,
                self.decoder_backend,
                self.encoder_backend,
            )
        ):
            raise ValueError("video stream requirements require semantic, stream, codec, and backend identity")
        if self.endpoint_port and not 1 <= self.endpoint_port <= 65535:
            raise ValueError("required endpoint port must be zero or in 1..65535")
        if self.ssrc is not None and not 0 <= self.ssrc <= 0xFFFF_FFFF:
            raise ValueError("required SSRC must fit in uint32")
        if not 0 <= self.payload_type <= 127:
            raise ValueError("required payload type must fit in 7 bits")
        if self.width < 0 or self.height < 0 or self.rtp_clock_rate <= 0:
            raise ValueError("required media dimensions cannot be negative and RTP clock rate must be positive")
        if self.frame_rate_hz < 0 or not math.isfinite(self.frame_rate_hz):
            raise ValueError("required frame rate must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class StreamNegotiationRequirements:
    pipeline_id: str
    session_id: str
    session_generation: int
    contract_fingerprint: str
    deployment_fingerprint: str
    streams: tuple[VideoStreamRequirement, ...] = ()
    transport_mode: str = "dds"

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.pipeline_id,
                self.session_id,
                self.contract_fingerprint,
                self.deployment_fingerprint,
                self.transport_mode,
            )
        ):
            raise ValueError("stream negotiation requires pipeline, session, fingerprint, and transport identity")
        if self.session_generation < 1:
            raise ValueError("stream negotiation session generation must be positive")
        if self.transport_mode not in {"dds", "rtp"}:
            raise ValueError("stream negotiation transport mode must be dds or rtp")
        if self.transport_mode == "dds" and self.streams:
            raise ValueError("DDS stream negotiation cannot declare RTP streams")
        if self.transport_mode == "rtp" and not self.streams:
            raise ValueError("RTP stream negotiation requires at least one stream")
        observation_keys = [stream.observation_key for stream in self.streams]
        stream_ids = [stream.stream_id for stream in self.streams]
        if len(set(observation_keys)) != len(observation_keys) or len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream negotiation requirements contain duplicate observation keys or stream IDs")


def negotiate_video_streams(
    requirements: StreamNegotiationRequirements,
    capabilities: VideoTransportCapabilities,
    descriptors: tuple[VideoStreamDescriptor, ...],
    *,
    validate_deployment_fingerprint: bool = True,
) -> dict[str, VideoStreamDescriptor]:
    """Validate startup capabilities and descriptors without transport fallback."""
    if capabilities.protocol_version != PROTOCOL_VERSION:
        raise StreamNegotiationError(
            "protocol_version_mismatch",
            f"video transport protocol version {capabilities.protocol_version} does not match {PROTOCOL_VERSION}",
            local=PROTOCOL_VERSION,
            remote=capabilities.protocol_version,
        )
    if requirements.transport_mode not in capabilities.transport_modes:
        raise StreamNegotiationError(
            "unsupported_transport_mode",
            f"remote does not support explicit transport mode {requirements.transport_mode!r}",
            transport_mode=requirements.transport_mode,
        )
    if requirements.transport_mode == "dds":
        if requirements.streams or descriptors:
            raise StreamNegotiationError(
                "descriptor_mismatch",
                "DDS compatibility mode cannot negotiate RTP stream descriptors",
            )
        return {}
    if requirements.transport_mode != "rtp":
        raise StreamNegotiationError("unsupported_transport_mode", "unknown observation transport mode")

    by_observation: dict[str, VideoStreamDescriptor] = {}
    stream_ids: set[str] = set()
    for descriptor in descriptors:
        if descriptor.observation_key in by_observation or descriptor.stream_id in stream_ids:
            raise StreamNegotiationError(
                "semantic_key_collision",
                "video descriptors contain duplicate observation keys or stream IDs",
                observation_key=descriptor.observation_key,
                stream_id=descriptor.stream_id,
            )
        by_observation[descriptor.observation_key] = descriptor
        stream_ids.add(descriptor.stream_id)

    required_keys = {stream.observation_key for stream in requirements.streams}
    if len(required_keys) != len(requirements.streams):
        raise StreamNegotiationError("semantic_key_collision", "required streams contain duplicate observation keys")
    if set(by_observation) != required_keys:
        raise StreamNegotiationError(
            "missing_stream",
            "stream descriptors do not exactly match required RTP observations",
            missing=tuple(sorted(required_keys - set(by_observation))),
            unexpected=tuple(sorted(set(by_observation) - required_keys)),
        )

    for required in requirements.streams:
        descriptor = by_observation[required.observation_key]
        checks = (
            (descriptor.protocol_version == PROTOCOL_VERSION, "protocol_version_mismatch", "protocol_version"),
            (descriptor.pipeline_id == requirements.pipeline_id, "descriptor_mismatch", "pipeline_id"),
            (descriptor.session_id == requirements.session_id, "descriptor_mismatch", "session_id"),
            (
                descriptor.session_generation == requirements.session_generation,
                "descriptor_mismatch",
                "session_generation",
            ),
            (descriptor.stream_id == required.stream_id, "descriptor_mismatch", "stream_id"),
            (
                descriptor.contract_fingerprint == requirements.contract_fingerprint,
                "contract_fingerprint_mismatch",
                "contract_fingerprint",
            ),
            (
                not validate_deployment_fingerprint
                or descriptor.deployment_fingerprint == requirements.deployment_fingerprint,
                "deployment_fingerprint_mismatch",
                "deployment_fingerprint",
            ),
            (descriptor.codec == required.codec, "descriptor_mismatch", "codec"),
            (descriptor.codec_profile == required.codec_profile, "descriptor_mismatch", "codec_profile"),
            (
                not required.endpoint_host or descriptor.endpoint_host == required.endpoint_host,
                "descriptor_mismatch",
                "endpoint_host",
            ),
            (
                not required.endpoint_port or descriptor.endpoint_port == required.endpoint_port,
                "descriptor_mismatch",
                "endpoint_port",
            ),
            (required.ssrc is None or descriptor.ssrc == required.ssrc, "descriptor_mismatch", "ssrc"),
            (descriptor.payload_type == required.payload_type, "descriptor_mismatch", "payload_type"),
            (not required.width or descriptor.width == required.width, "descriptor_mismatch", "width"),
            (not required.height or descriptor.height == required.height, "descriptor_mismatch", "height"),
            (
                not required.frame_rate_hz or descriptor.frame_rate_hz == required.frame_rate_hz,
                "descriptor_mismatch",
                "frame_rate_hz",
            ),
            (descriptor.rtp_clock_rate == required.rtp_clock_rate, "descriptor_mismatch", "rtp_clock_rate"),
            (descriptor.pixel_format == required.pixel_format, "descriptor_mismatch", "pixel_format"),
            (descriptor.color_space == required.color_space, "descriptor_mismatch", "color_space"),
            (descriptor.color_range == required.color_range, "descriptor_mismatch", "color_range"),
            (
                required.encoder_backend == "auto" or descriptor.encoder_backend == required.encoder_backend,
                "descriptor_mismatch",
                "encoder_backend",
            ),
        )
        for valid, code, field in checks:
            if not valid:
                raise StreamNegotiationError(
                    code,
                    f"video stream descriptor mismatch for {field}",
                    observation_key=required.observation_key,
                    stream_id=required.stream_id,
                    field=field,
                )
        if required.codec not in capabilities.codecs:
            raise StreamNegotiationError(
                "unsupported_codec",
                f"remote does not support required codec {required.codec!r}",
                stream_id=required.stream_id,
            )
        if required.decoder_backend != "auto" and required.decoder_backend not in capabilities.decoder_backends:
            raise StreamNegotiationError(
                "unsupported_backend",
                f"remote does not support explicit decoder backend {required.decoder_backend!r}",
                stream_id=required.stream_id,
                backend=required.decoder_backend,
            )
    return by_observation


def validate_request_streams(
    references: tuple[StreamReference, ...], negotiated: dict[str, VideoStreamDescriptor]
) -> None:
    if len(references) != len(negotiated):
        raise StreamNegotiationError("missing_stream", "request does not reference every negotiated stream")
    for reference in references:
        descriptor = negotiated.get(reference.observation_key)
        if descriptor is None or descriptor.stream_id != reference.stream_id:
            raise StreamNegotiationError(
                "descriptor_mismatch",
                "request stream reference does not match negotiated descriptor",
                observation_key=reference.observation_key,
                stream_id=reference.stream_id,
            )


class VideoStreamNegotiator:
    """Accumulate late descriptors and gate stream-backed requests."""

    def __init__(
        self,
        requirements: StreamNegotiationRequirements,
        capabilities: VideoTransportCapabilities,
        *,
        validate_deployment_fingerprint: bool = True,
    ) -> None:
        self.requirements = requirements
        self.capabilities = capabilities
        self._validate_deployment_fingerprint = validate_deployment_fingerprint
        self._lock = threading.RLock()
        self._descriptors: dict[str, VideoStreamDescriptor] = {}
        self._negotiated: dict[str, VideoStreamDescriptor] | None = (
            negotiate_video_streams(
                requirements,
                capabilities,
                (),
                validate_deployment_fingerprint=validate_deployment_fingerprint,
            )
            if requirements.transport_mode == "dds"
            else None
        )

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._negotiated is not None

    def observe_descriptor(self, descriptor: VideoStreamDescriptor) -> bool:
        with self._lock:
            if (
                descriptor.pipeline_id != self.requirements.pipeline_id
                or descriptor.session_id != self.requirements.session_id
                or descriptor.session_generation != self.requirements.session_generation
            ):
                raise StreamNegotiationError(
                    "descriptor_mismatch",
                    "video stream descriptor does not match the active session",
                    observation_key=descriptor.observation_key,
                    stream_id=descriptor.stream_id,
                )
            required_keys = {stream.observation_key for stream in self.requirements.streams}
            if descriptor.observation_key not in required_keys:
                raise StreamNegotiationError(
                    "missing_stream",
                    "received an unexpected video stream descriptor",
                    observation_key=descriptor.observation_key,
                    stream_id=descriptor.stream_id,
                )
            existing = self._descriptors.get(descriptor.observation_key)
            if existing is not None and existing != descriptor:
                raise StreamNegotiationError(
                    "descriptor_mismatch",
                    "descriptor changed for an active observation key",
                    observation_key=descriptor.observation_key,
                    stream_id=descriptor.stream_id,
                )
            self._descriptors[descriptor.observation_key] = descriptor
            if set(self._descriptors) != required_keys:
                return False
            self._negotiated = negotiate_video_streams(
                self.requirements,
                self.capabilities,
                tuple(self._descriptors.values()),
                validate_deployment_fingerprint=self._validate_deployment_fingerprint,
            )
            return True

    def validate_request(self, references: tuple[StreamReference, ...]) -> None:
        with self._lock:
            if self._negotiated is None:
                raise StreamNegotiationError(
                    "missing_stream",
                    "video stream descriptors are not ready",
                    missing=tuple(
                        sorted(
                            {stream.observation_key for stream in self.requirements.streams} - set(self._descriptors)
                        )
                    ),
                )
            validate_request_streams(references, self._negotiated)

    def reset(self, session_id: str, session_generation: int) -> None:
        if not session_id or session_generation < 1:
            raise ValueError("stream negotiation reset requires a live session")
        with self._lock:
            self.requirements = StreamNegotiationRequirements(
                pipeline_id=self.requirements.pipeline_id,
                session_id=session_id,
                session_generation=session_generation,
                contract_fingerprint=self.requirements.contract_fingerprint,
                deployment_fingerprint=self.requirements.deployment_fingerprint,
                streams=self.requirements.streams,
                transport_mode=self.requirements.transport_mode,
            )
            self._descriptors.clear()
            self._negotiated = (
                negotiate_video_streams(self.requirements, self.capabilities, ())
                if self.requirements.transport_mode == "dds"
                else None
            )
