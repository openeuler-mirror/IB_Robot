"""Backend-neutral video codec contracts and deterministic backend selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

CodecKind = Literal["encoder", "decoder"]


class CodecLifecycleState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class VideoFrame:
    data: Any
    capture_timestamp_ns: int
    receive_timestamp_ns: int
    width: int
    height: int
    pixel_format: str
    color_space: str = "bt709"
    color_range: str = "limited"
    keyframe: bool = False

    def __post_init__(self) -> None:
        if self.capture_timestamp_ns < 0 or self.receive_timestamp_ns < 0:
            raise ValueError("video frame timestamps cannot be negative")
        if self.width <= 0 or self.height <= 0 or not self.pixel_format:
            raise ValueError("video frame dimensions and pixel format must be valid")


@dataclass(frozen=True, slots=True)
class EncodedPacket:
    payload: bytes
    rtp_timestamp: int
    capture_timestamp_ns: int
    keyframe: bool = False
    end_of_frame: bool = True

    def __post_init__(self) -> None:
        if not self.payload:
            raise ValueError("encoded packet payload cannot be empty")
        if not 0 <= self.rtp_timestamp < 1 << 32:
            raise ValueError("RTP timestamp must fit in uint32")
        if self.capture_timestamp_ns < 0:
            raise ValueError("packet capture timestamp cannot be negative")


@dataclass(frozen=True, slots=True)
class CodecCapabilities:
    codecs: tuple[str, ...] = ("h264",)
    pixel_formats: tuple[str, ...] = ("nv12",)
    hardware_accelerated: bool = False

    def __post_init__(self) -> None:
        if not self.codecs or not self.pixel_formats:
            raise ValueError("codec capabilities must declare codecs and pixel formats")


@dataclass(frozen=True, slots=True)
class CodecMetrics:
    input_frames: int = 0
    output_frames: int = 0
    output_packets: int = 0
    dropped_frames: int = 0
    errors: int = 0
    decoder_backlog_depth: int = 0
    decoder_output_age_ns: int = 0
    dropped_stale_decoder_frames: int = 0
    metadata_fifo_depth: int = 0
    input_frame_rate_hz: float = 0.0
    output_frame_rate_hz: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.input_frames,
                self.output_frames,
                self.output_packets,
                self.dropped_frames,
                self.errors,
                self.decoder_backlog_depth,
                self.decoder_output_age_ns,
                self.dropped_stale_decoder_frames,
                self.metadata_fifo_depth,
                self.input_frame_rate_hz,
                self.output_frame_rate_hz,
            )
            < 0
        ):
            raise ValueError("codec metrics cannot be negative")


class VideoCodecError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        backend: str,
        recoverable: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if not code or not backend:
            raise ValueError("video codec errors require code and backend")
        super().__init__(message)
        self.code = code
        self.backend = backend
        self.recoverable = recoverable
        self.details = MappingProxyType(dict(details or {}))


class VideoEncoder(ABC):
    @property
    @abstractmethod
    def state(self) -> CodecLifecycleState: ...

    @property
    @abstractmethod
    def metrics(self) -> CodecMetrics: ...

    @abstractmethod
    def encode(self, frame: VideoFrame) -> list[EncodedPacket]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def close(self, timeout_s: float = 1.0) -> None: ...

    def discard_pending_output(self) -> None:
        """Drop encoded output still buffered inside the codec.

        Called at session rollover so access units of the retired session
        that are still in flight through a pipelined codec (e.g. the Ascend
        DVPP pipeline delay) do not surface as the first output of the new
        session. Backends without internal output buffering keep the no-op.
        """
        return


class VideoDecoder(ABC):
    @property
    @abstractmethod
    def state(self) -> CodecLifecycleState: ...

    @property
    @abstractmethod
    def metrics(self) -> CodecMetrics: ...

    @abstractmethod
    def decode(self, packet: EncodedPacket) -> list[VideoFrame]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def close(self, timeout_s: float = 1.0) -> None: ...


@dataclass(frozen=True, slots=True)
class BackendProbeResult:
    backend: str
    kind: CodecKind
    available: bool
    capabilities: CodecCapabilities | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedCodecBackend:
    name: str
    kind: CodecKind
    capabilities: CodecCapabilities
    probes: tuple[BackendProbeResult, ...]
    factory: Callable[..., VideoEncoder | VideoDecoder] = field(repr=False, compare=False)

    def create(self, **options: object) -> VideoEncoder | VideoDecoder:
        return self.factory(**options)


@dataclass(frozen=True, slots=True)
class _BackendRegistration:
    name: str
    priority: int
    probe: Callable[[CodecKind], CodecCapabilities | None]
    encoder_factory: Callable[..., VideoEncoder] | None
    decoder_factory: Callable[..., VideoDecoder] | None


class VideoCodecRegistry:
    """Resolve explicit or auto codec policy without importing optional SDKs."""

    def __init__(self) -> None:
        self._backends: dict[str, _BackendRegistration] = {}

    def register(
        self,
        name: str,
        *,
        priority: int,
        probe: Callable[[CodecKind], CodecCapabilities | None],
        encoder_factory: Callable[..., VideoEncoder] | None = None,
        decoder_factory: Callable[..., VideoDecoder] | None = None,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized or normalized == "auto":
            raise ValueError("codec backend name must be non-empty and cannot be auto")
        if normalized in self._backends:
            raise ValueError(f"codec backend {normalized!r} is already registered")
        if encoder_factory is None and decoder_factory is None:
            raise ValueError("codec backend must provide an encoder or decoder factory")
        self._backends[normalized] = _BackendRegistration(
            normalized,
            int(priority),
            probe,
            encoder_factory,
            decoder_factory,
        )

    def resolve(self, policy: str, kind: CodecKind) -> ResolvedCodecBackend:
        if kind not in {"encoder", "decoder"}:
            raise ValueError(f"unsupported codec kind {kind!r}")
        normalized = policy.strip().lower()
        if normalized != "auto":
            registration = self._backends.get(normalized)
            if registration is None:
                raise VideoCodecError(
                    "unknown_backend",
                    f"video codec backend {normalized!r} is not registered",
                    backend=normalized,
                )
            result = self._probe(registration, kind)
            if not result.available:
                raise VideoCodecError(
                    "backend_unavailable",
                    f"video codec backend {normalized!r} is unavailable: {result.reason}",
                    backend=normalized,
                    details={"kind": kind, "reason": result.reason},
                )
            return self._resolved(registration, kind, result, (result,))

        probes: list[BackendProbeResult] = []
        registrations = sorted(self._backends.values(), key=lambda item: (-item.priority, item.name))
        for registration in registrations:
            result = self._probe(registration, kind)
            probes.append(result)
            if result.available:
                return self._resolved(registration, kind, result, tuple(probes))
        raise VideoCodecError(
            "no_backend_available",
            f"no {kind} backend is available",
            backend="auto",
            details={"probes": tuple((probe.backend, probe.reason) for probe in probes)},
        )

    @staticmethod
    def _probe(registration: _BackendRegistration, kind: CodecKind) -> BackendProbeResult:
        factory = registration.encoder_factory if kind == "encoder" else registration.decoder_factory
        if factory is None:
            return BackendProbeResult(registration.name, kind, False, reason=f"{kind} is not implemented")
        try:
            capabilities = registration.probe(kind)
        except Exception as exc:
            return BackendProbeResult(
                registration.name,
                kind,
                False,
                reason=f"{type(exc).__name__}: {exc}",
            )
        if capabilities is None:
            return BackendProbeResult(registration.name, kind, False, reason="probe reported unavailable")
        return BackendProbeResult(registration.name, kind, True, capabilities=capabilities)

    @staticmethod
    def _resolved(
        registration: _BackendRegistration,
        kind: CodecKind,
        result: BackendProbeResult,
        probes: tuple[BackendProbeResult, ...],
    ) -> ResolvedCodecBackend:
        factory = registration.encoder_factory if kind == "encoder" else registration.decoder_factory
        assert factory is not None
        assert result.capabilities is not None
        return ResolvedCodecBackend(registration.name, kind, result.capabilities, probes, factory)


def create_default_video_codec_registry() -> VideoCodecRegistry:
    """Create the platform-neutral registry without eagerly loading optional runtimes."""
    from observation_transport.ascend_ffmpeg_video_codec import register_ascend_backend
    from observation_transport.nvidia_video_codec import register_nvidia_backend
    from observation_transport.software_video_codec import register_software_backend

    registry = VideoCodecRegistry()
    register_ascend_backend(registry)
    register_nvidia_backend(registry)
    register_software_backend(registry)
    return registry
