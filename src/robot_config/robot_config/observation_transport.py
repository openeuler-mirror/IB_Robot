"""Typed observation video transport configuration and validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import Any

_STREAM_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
VIDEO_CODEC_BACKENDS = frozenset({"auto", "software", "ascend", "nvidia", "vaapi", "v4l2m2m", "rkmpp"})
_PROFILES = {"baseline", "main", "high"}
_COLOR_RANGES = {"limited", "full"}


@dataclass(frozen=True, slots=True)
class RtpEndpointSpec:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class H264Spec:
    profile: str = "main"
    bitrate_bps: int = 4_000_000
    gop_frames: int = 15


@dataclass(frozen=True, slots=True)
class VideoMediaSpec:
    width: int | None = None
    height: int | None = None
    frame_rate_hz: float | None = None
    pixel_format: str = "nv12"
    color_space: str = "bt709"
    color_range: str = "limited"


@dataclass(frozen=True, slots=True)
class VideoBufferSpec:
    sender_queue_frames: int = 2
    receiver_queue_packets: int = 256
    decoded_frame_capacity: int = 32
    retention_ms: int = 1000


@dataclass(frozen=True, slots=True)
class VideoReadinessSpec:
    keyframe_timeout_ms: int = 3000
    timestamp_mapping_max_age_ms: int = 1000
    max_inter_camera_skew_ms: int = 50


@dataclass(frozen=True, slots=True)
class RecordingSpec:
    integrity_mode: str = "strict"


@dataclass(frozen=True, slots=True)
class ObservationTransportSpec:
    mode: str = "dds"
    stream_id: str | None = None
    endpoint: RtpEndpointSpec | None = None
    codec: str = "h264"
    h264: H264Spec | None = None
    encoder_backend: str = "auto"
    decoder_backend: str = "auto"
    media: VideoMediaSpec | None = None
    buffer: VideoBufferSpec | None = None
    readiness: VideoReadinessSpec | None = None
    recording: RecordingSpec | None = None
    security: str = "none"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _check_fields(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unsupported fields: {', '.join(unknown)}")


def parse_observation_transport(value: Any) -> ObservationTransportSpec | None:
    """Parse an optional transport declaration without changing its mode."""
    if value is None:
        return None
    data = _mapping(value, "transport")
    _check_fields(
        data,
        {
            "mode",
            "stream_id",
            "endpoint",
            "codec",
            "h264",
            "encoder_backend",
            "decoder_backend",
            "media",
            "buffer",
            "readiness",
            "recording",
            "security",
        },
        "transport",
    )
    mode = str(data.get("mode", "dds")).lower()
    if mode == "dds":
        rtp_fields = sorted(set(data) - {"mode"})
        if rtp_fields:
            raise ValueError(f"transport mode dds cannot define RTP-specific fields: {', '.join(rtp_fields)}")
    endpoint = None
    if data.get("endpoint") is not None:
        item = _mapping(data["endpoint"], "transport.endpoint")
        _check_fields(item, {"host", "port"}, "transport.endpoint")
        endpoint = RtpEndpointSpec(host=str(item.get("host", "")).strip(), port=int(item.get("port", 0)))
    h264 = None
    if data.get("h264") is not None:
        item = _mapping(data["h264"], "transport.h264")
        _check_fields(item, {"profile", "bitrate_bps", "gop_frames"}, "transport.h264")
        h264 = H264Spec(
            profile=str(item.get("profile", "main")).lower(),
            bitrate_bps=int(item.get("bitrate_bps", 4_000_000)),
            gop_frames=int(item.get("gop_frames", 15)),
        )
    media = None
    if data.get("media") is not None:
        item = _mapping(data["media"], "transport.media")
        _check_fields(
            item,
            {"width", "height", "frame_rate_hz", "pixel_format", "color_space", "color_range"},
            "transport.media",
        )
        media = VideoMediaSpec(
            width=int(item["width"]) if item.get("width") is not None else None,
            height=int(item["height"]) if item.get("height") is not None else None,
            frame_rate_hz=float(item["frame_rate_hz"]) if item.get("frame_rate_hz") is not None else None,
            pixel_format=str(item.get("pixel_format", "nv12")).lower(),
            color_space=str(item.get("color_space", "bt709")).lower(),
            color_range=str(item.get("color_range", "limited")).lower(),
        )
    buffer = None
    if data.get("buffer") is not None:
        item = _mapping(data["buffer"], "transport.buffer")
        _check_fields(
            item,
            {"sender_queue_frames", "receiver_queue_packets", "decoded_frame_capacity", "retention_ms"},
            "transport.buffer",
        )
        buffer = VideoBufferSpec(
            sender_queue_frames=int(item.get("sender_queue_frames", 2)),
            receiver_queue_packets=int(item.get("receiver_queue_packets", 256)),
            decoded_frame_capacity=int(item.get("decoded_frame_capacity", 32)),
            retention_ms=int(item.get("retention_ms", 1000)),
        )
    readiness = None
    if data.get("readiness") is not None:
        item = _mapping(data["readiness"], "transport.readiness")
        _check_fields(
            item,
            {"keyframe_timeout_ms", "timestamp_mapping_max_age_ms", "max_inter_camera_skew_ms"},
            "transport.readiness",
        )
        readiness = VideoReadinessSpec(
            keyframe_timeout_ms=int(item.get("keyframe_timeout_ms", 3000)),
            timestamp_mapping_max_age_ms=int(item.get("timestamp_mapping_max_age_ms", 1000)),
            max_inter_camera_skew_ms=int(item.get("max_inter_camera_skew_ms", 50)),
        )
    recording = None
    if data.get("recording") is not None:
        item = _mapping(data["recording"], "transport.recording")
        _check_fields(item, {"integrity_mode"}, "transport.recording")
        recording = RecordingSpec(integrity_mode=str(item.get("integrity_mode", "strict")).lower())
    return ObservationTransportSpec(
        mode=mode,
        stream_id=str(data["stream_id"]).strip() if data.get("stream_id") is not None else None,
        endpoint=endpoint,
        codec=str(data.get("codec", "h264")).lower(),
        h264=h264,
        encoder_backend=str(data.get("encoder_backend", "auto")).lower(),
        decoder_backend=str(data.get("decoder_backend", "auto")).lower(),
        media=media,
        buffer=buffer,
        readiness=readiness,
        recording=recording,
        security=str(data.get("security", "none")).lower(),
    )


def effective_observation_transport(value: ObservationTransportSpec | None) -> ObservationTransportSpec:
    return value or ObservationTransportSpec()


def observation_transport_to_dict(value: ObservationTransportSpec) -> dict[str, Any]:
    if value.mode == "dds":
        return {"mode": "dds"}
    payload = asdict(value)
    if value.recording is None:
        payload.pop("recording")
    return payload


def resolve_observation_transport(
    value: ObservationTransportSpec | None,
    *,
    image: Mapping[str, Any] | None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_fps: float | None = None,
) -> ObservationTransportSpec | None:
    if value is None or value.mode != "rtp":
        return value
    resize = (image or {}).get("resize")
    height = int(resize[0]) if resize and len(resize) == 2 else camera_height
    width = int(resize[1]) if resize and len(resize) == 2 else camera_width
    media = value.media or VideoMediaSpec()
    return replace(
        value,
        h264=value.h264 or H264Spec(),
        media=replace(
            media,
            width=media.width if media.width is not None else width,
            height=media.height if media.height is not None else height,
            frame_rate_hz=media.frame_rate_hz if media.frame_rate_hz is not None else camera_fps,
        ),
        buffer=value.buffer or VideoBufferSpec(),
        readiness=value.readiness or VideoReadinessSpec(),
    )


def validate_observation_transports(
    observations: Sequence[Any],
    *,
    distributed_enabled: bool | None = None,
) -> list[str]:
    errors: list[str] = []
    stream_ids: dict[str, str] = {}
    endpoints: dict[tuple[str, int], str] = {}
    integrity_modes: set[str] = set()
    for obs in observations:
        key = str(getattr(obs, "key", "?"))
        ros_type = str(getattr(obs, "type", "") or "")
        image = getattr(obs, "image", None)
        value = getattr(obs, "transport", None)
        if value is None:
            continue
        if value.mode not in {"dds", "rtp"}:
            errors.append(f"Observation '{key}' transport.mode must be one of: dds, rtp")
            continue
        if value.mode == "dds":
            if any(
                (
                    value.stream_id,
                    value.endpoint,
                    value.h264,
                    value.media,
                    value.buffer,
                    value.readiness,
                    value.recording,
                )
            ):
                errors.append(f"Observation '{key}' DDS transport cannot define RTP-specific fields")
            continue
        if distributed_enabled is False:
            errors.append(f"Observation '{key}' RTP transport requires a distributed inference pipeline")
        if ros_type != "sensor_msgs/msg/Image":
            errors.append(f"Observation '{key}' RTP transport requires sensor_msgs/msg/Image")
        encoding = str((image or {}).get("encoding", "")).lower()
        if encoding not in {"rgb8", "bgr8"}:
            errors.append(f"Observation '{key}' RTP transport requires image.encoding rgb8 or bgr8")
        if not value.stream_id or not _STREAM_ID_RE.fullmatch(value.stream_id):
            errors.append(f"Observation '{key}' transport.stream_id is invalid")
        elif value.stream_id in stream_ids:
            errors.append(f"Observation '{key}' duplicates stream_id '{value.stream_id}'")
        else:
            stream_ids[value.stream_id] = key
        if value.endpoint is None or not value.endpoint.host or not 1 <= value.endpoint.port <= 65535:
            errors.append(f"Observation '{key}' transport.endpoint must have a host and port in 1..65535")
        elif (value.endpoint.host, value.endpoint.port) in endpoints:
            errors.append(f"Observation '{key}' duplicates RTP endpoint {value.endpoint.host}:{value.endpoint.port}")
        else:
            endpoints[(value.endpoint.host, value.endpoint.port)] = key
        if value.codec != "h264":
            errors.append(f"Observation '{key}' transport.codec currently must be h264")
        h264 = value.h264
        if h264 is None or h264.profile not in _PROFILES or h264.bitrate_bps <= 0 or h264.gop_frames <= 0:
            errors.append(f"Observation '{key}' has invalid H.264 profile, bitrate, or GOP")
        media = value.media
        if (
            media is None
            or media.width is None
            or media.height is None
            or media.frame_rate_hz is None
            or media.width <= 0
            or media.height <= 0
            or media.width % 2
            or media.height % 2
            or not math.isfinite(media.frame_rate_hz)
            or media.frame_rate_hz <= 0
        ):
            errors.append(f"Observation '{key}' transport.media requires positive even dimensions and frame rate")
        elif media.pixel_format != "nv12" or media.color_space != "bt709" or media.color_range not in _COLOR_RANGES:
            errors.append(f"Observation '{key}' has unsupported transport media format or color metadata")
        if value.encoder_backend not in VIDEO_CODEC_BACKENDS or value.decoder_backend not in VIDEO_CODEC_BACKENDS:
            errors.append(f"Observation '{key}' has unsupported video codec backend")
        if value.security != "none":
            errors.append(f"Observation '{key}' transport.security currently must be none")
        integrity_mode = value.recording.integrity_mode if value.recording is not None else "strict"
        if integrity_mode not in {"strict", "tolerant"}:
            errors.append(f"Observation '{key}' transport.recording.integrity_mode must be strict or tolerant")
        else:
            integrity_modes.add(integrity_mode)
        if value.buffer is None or min(asdict(value.buffer).values()) <= 0:
            errors.append(f"Observation '{key}' transport.buffer values must be positive")
        if value.readiness is None:
            errors.append(f"Observation '{key}' transport.readiness is required")
        elif (
            value.readiness.keyframe_timeout_ms <= 0
            or value.readiness.timestamp_mapping_max_age_ms <= 0
            or value.readiness.max_inter_camera_skew_ms < 0
        ):
            errors.append(f"Observation '{key}' transport.readiness values are invalid")
    if len(integrity_modes) > 1:
        errors.append("integrity_mode must be uniform across episode (found both strict and tolerant)")
    return errors


def require_valid_observation_transports(
    observations: Sequence[Any],
    *,
    distributed_enabled: bool | None = None,
) -> None:
    """Raise one deterministic error for an invalid observation transport contract."""
    errors = validate_observation_transports(observations, distributed_enabled=distributed_enabled)
    if errors:
        raise ValueError("Invalid observation transport configuration:\n- " + "\n- ".join(errors))


def robot_config_has_distributed_pipeline(robot_config: Mapping[str, Any]) -> bool:
    """Return whether any configured inference pipeline explicitly uses distributed execution."""
    control_modes = robot_config.get("control_modes", {})
    if not isinstance(control_modes, Mapping):
        return False
    for mode in control_modes.values():
        if not isinstance(mode, Mapping):
            continue
        inference = mode.get("inference", {})
        if not isinstance(inference, Mapping):
            continue
        pipelines = inference.get("pipelines", {})
        if isinstance(pipelines, Mapping) and any(
            isinstance(pipeline, Mapping) and pipeline.get("execution_mode") == "distributed"
            for pipeline in pipelines.values()
        ):
            return True
    return False


def validate_robot_config_observation_transports(robot_config: Mapping[str, Any]) -> list[str]:
    """Validate transport declarations directly from a raw robot configuration."""
    cameras = {
        item.get("name"): item
        for item in robot_config.get("peripherals", []) or []
        if isinstance(item, Mapping) and item.get("type") == "camera"
    }
    observations = []
    contract = robot_config.get("contract", {})
    raw_observations = contract.get("observations", []) if isinstance(contract, Mapping) else []
    for item in raw_observations or []:
        if not isinstance(item, Mapping):
            continue
        camera = cameras.get(item.get("peripheral"))
        image = item.get("image")
        if camera is not None and not image:
            image = {
                "resize": [camera.get("height", 480), camera.get("width", 640)],
                "encoding": camera.get("pixel_format", "bgr8"),
            }
        transport = resolve_observation_transport(
            parse_observation_transport(item.get("transport")),
            image=image,
            camera_width=camera.get("width") if camera else None,
            camera_height=camera.get("height") if camera else None,
            camera_fps=camera.get("fps") if camera else None,
        )
        observations.append(
            SimpleNamespace(
                key=item.get("key", "?"),
                type=item.get("type") or ("sensor_msgs/msg/Image" if item.get("peripheral") else ""),
                image=image,
                transport=transport,
            )
        )
    return validate_observation_transports(
        observations,
        distributed_enabled=robot_config_has_distributed_pipeline(robot_config),
    )
