"""Typed observation video transport configuration and validation."""

from __future__ import annotations

import copy
import hashlib
import json
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

# ``dropped`` reasons that describe normal stream entry rather than a transport fault.
# The recorder consults this when it computes ``has_gap``; the offline converter consults
# it when it counts ``integrity.frame_gaps``. The two must classify a reason identically
# -- when they disagree, a healthy episode is silently marked ``clean=false``, which is
# why the set lives here rather than as a literal on either side.
NON_FAULT_DROP_REASONS = frozenset({"pre_keyframe"})


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


# Benchmark transport projection support. The compiler below is generic: it
# accepts a mode and an optional RTP profile, then uses the same parser,
# resolver, validator and fingerprinting used by all robot-config consumers.
# Benchmark code only selects the profile and delegates here.
_COMPILE_MODE_VALUES = frozenset({"dds", "rtp"})
_COMPILE_RTP_FIELDS = frozenset(
    {
        "endpoint_host",
        "base_port",
        "port_stride",
        "codec",
        "encoder_backend",
        "decoder_backend",
        "h264",
        "media",
        "buffer",
        "readiness",
        "security",
        "streams",
    }
)
_COMPILE_STREAM_FIELDS = frozenset(
    {
        "stream_id",
        "endpoint",
        "codec",
        "encoder_backend",
        "decoder_backend",
        "h264",
        "media",
        "buffer",
        "readiness",
        "security",
    }
)


class ObservationTransportCompileError(ValueError):
    """Raised when a generic observation transport profile cannot be compiled."""


def materialize_observation_transports(
    image_observations: list[dict[str, Any]],
    mode: str,
    *,
    rtp: Mapping[str, Any] | None = None,
    rate_hz: float = 20.0,
) -> None:
    """Compile one transport selection into raw image observations.

    Product-level callers may own a mode selector, but all transport schema,
    defaults, parsing, resolution, validation and fingerprinting stay in this
    generic module.
    """
    if mode not in _COMPILE_MODE_VALUES:
        raise ObservationTransportCompileError("transport mode must be exactly 'dds' or 'rtp'")
    if not image_observations:
        raise ObservationTransportCompileError("at least one image observation is required")
    if not isinstance(rate_hz, int | float) or isinstance(rate_hz, bool) or not math.isfinite(float(rate_hz)):
        raise ObservationTransportCompileError("contract.rate_hz must be a finite positive number")
    if float(rate_hz) <= 0:
        raise ObservationTransportCompileError("contract.rate_hz must be a finite positive number")
    if any("transport" in item for item in image_observations):
        raise ObservationTransportCompileError("transport declarations must be absent before materialization")

    if mode == "dds":
        if rtp is not None:
            raise ObservationTransportCompileError("DDS transport cannot define an RTP profile")
        for item in image_observations:
            item["transport"] = {"mode": "dds"}
        return

    profile = _compile_mapping(rtp or {}, "rtp profile")
    _compile_check_fields(profile, _COMPILE_RTP_FIELDS, "rtp profile")
    global_endpoint_host = _compile_exact_string(profile.get("endpoint_host", "127.0.0.1"), "rtp.endpoint_host")
    base_port = _compile_exact_int(profile.get("base_port", 55000), "rtp.base_port", minimum=1, maximum=65535)
    port_stride = _compile_exact_int(profile.get("port_stride", 2), "rtp.port_stride", minimum=1, maximum=65535)
    global_codec = _compile_exact_string(profile.get("codec", "h264"), "rtp.codec").lower()
    global_encoder = _compile_exact_string(profile.get("encoder_backend", "auto"), "rtp.encoder_backend").lower()
    global_decoder = _compile_exact_string(profile.get("decoder_backend", "auto"), "rtp.decoder_backend").lower()
    global_security = _compile_exact_string(profile.get("security", "none"), "rtp.security").lower()
    global_h264 = _compile_profile_section(profile.get("h264"), "rtp.h264", {"profile", "bitrate_bps", "gop_frames"})
    global_media = _compile_profile_section(
        profile.get("media"),
        "rtp.media",
        {"width", "height", "frame_rate_hz", "pixel_format", "color_space", "color_range"},
    )
    global_buffer = _compile_profile_section(
        profile.get("buffer"),
        "rtp.buffer",
        {"sender_queue_frames", "receiver_queue_packets", "decoded_frame_capacity", "retention_ms"},
    )
    global_readiness = _compile_profile_section(
        profile.get("readiness"),
        "rtp.readiness",
        {"keyframe_timeout_ms", "timestamp_mapping_max_age_ms", "max_inter_camera_skew_ms"},
    )
    streams = _compile_mapping(profile.get("streams", {}), "rtp.streams")

    known_keys = {str(item.get("key")) for item in image_observations}
    unknown = sorted(set(streams) - known_keys)
    if unknown:
        raise ObservationTransportCompileError(f"RTP stream overrides reference unknown image observations: {unknown}")

    for index, observation in enumerate(image_observations):
        key = _compile_exact_string(observation.get("key"), f"image observation {index}.key")
        image = _compile_mapping(observation.get("image"), f"contract observation {key}.image")
        resize = image.get("resize")
        if not isinstance(resize, list | tuple) or len(resize) != 2:
            raise ObservationTransportCompileError(
                f"RTP image observation {key!r} requires image.resize [height, width]"
            )
        image_height = _compile_exact_int(resize[0], f"{key}.image.resize[0]", minimum=1)
        image_width = _compile_exact_int(resize[1], f"{key}.image.resize[1]", minimum=1)
        override = _compile_mapping(streams.get(key, {}), f"rtp.streams.{key}")
        _compile_check_fields(override, _COMPILE_STREAM_FIELDS, f"rtp.streams.{key}")
        endpoint_override = _compile_profile_section(
            override.get("endpoint"), f"rtp.streams.{key}.endpoint", {"host", "port"}
        )
        stream_h264 = _compile_merged_section(
            global_h264, override.get("h264"), f"rtp.streams.{key}.h264", {"profile", "bitrate_bps", "gop_frames"}
        )
        stream_media = _compile_merged_section(
            global_media,
            override.get("media"),
            f"rtp.streams.{key}.media",
            {"width", "height", "frame_rate_hz", "pixel_format", "color_space", "color_range"},
        )
        stream_buffer = _compile_merged_section(
            global_buffer,
            override.get("buffer"),
            f"rtp.streams.{key}.buffer",
            {"sender_queue_frames", "receiver_queue_packets", "decoded_frame_capacity", "retention_ms"},
        )
        stream_readiness = _compile_merged_section(
            global_readiness,
            override.get("readiness"),
            f"rtp.streams.{key}.readiness",
            {"keyframe_timeout_ms", "timestamp_mapping_max_age_ms", "max_inter_camera_skew_ms"},
        )
        stream_id = _compile_exact_string(
            override.get("stream_id", _compile_default_stream_id(key)), f"rtp.streams.{key}.stream_id"
        )
        endpoint_host = _compile_exact_string(
            endpoint_override.get("host", global_endpoint_host), f"rtp.streams.{key}.endpoint.host"
        )
        endpoint_port = _compile_exact_int(
            endpoint_override.get("port", base_port + index * port_stride),
            f"rtp.streams.{key}.endpoint.port",
            minimum=1,
            maximum=65535,
        )

        raw_transport = {
            "mode": "rtp",
            "stream_id": stream_id,
            "endpoint": {"host": endpoint_host, "port": endpoint_port},
            "codec": override.get("codec", global_codec),
            "encoder_backend": override.get("encoder_backend", global_encoder),
            "decoder_backend": override.get("decoder_backend", global_decoder),
            "h264": stream_h264,
            "media": stream_media,
            "buffer": stream_buffer,
            "readiness": stream_readiness,
            "security": override.get("security", global_security),
        }
        try:
            parsed = parse_observation_transport(raw_transport)
            resolved = resolve_observation_transport(parsed, image=image, camera_fps=float(rate_hz))
        except (TypeError, ValueError) as exc:
            raise ObservationTransportCompileError(f"RTP stream {key!r}: {exc}") from exc
        assert resolved is not None
        if resolved.mode != "rtp" or resolved.media is None:
            raise ObservationTransportCompileError(f"RTP stream {key!r} did not resolve to RTP media")
        if (resolved.media.height, resolved.media.width) != (image_height, image_width):
            raise ObservationTransportCompileError(
                f"RTP stream {key!r} media dimensions {resolved.media.height}x{resolved.media.width} "
                f"must match image.resize {image_height}x{image_width}"
            )
        observation["transport"] = observation_transport_to_dict(resolved)


def observation_transport_fingerprint(
    robot_config: Mapping[str, Any],
    mode: str,
    pipeline_id: str,
) -> str:
    """Hash the final contract and selected pipeline topology deterministically."""
    contract = copy.deepcopy(robot_config.get("contract", {}))
    control_mode = str(robot_config.get("default_control_mode", "model_inference"))
    control_modes = robot_config.get("control_modes", {})
    mode_config = control_modes.get(control_mode, {}) if isinstance(control_modes, Mapping) else {}
    inference = mode_config.get("inference", {}) if isinstance(mode_config, Mapping) else {}
    pipelines = inference.get("pipelines", {}) if isinstance(inference, Mapping) else {}
    pipeline = pipelines.get(pipeline_id, {}) if isinstance(pipelines, Mapping) else {}
    payload = {
        "schema_version": 1,
        "mode": mode,
        "pipeline_id": pipeline_id,
        "execution_mode": pipeline.get("execution_mode") if isinstance(pipeline, Mapping) else None,
        "contract": contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _compile_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservationTransportCompileError(f"{path} must be a mapping")
    return value


def _compile_check_fields(value: Mapping[str, Any], allowed: set[str] | frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ObservationTransportCompileError(f"{path} contains unsupported fields: {', '.join(unknown)}")


def _compile_profile_section(value: Any, path: str, allowed: set[str]) -> dict[str, Any]:
    if value is None:
        return {}
    result = dict(_compile_mapping(value, path))
    _compile_check_fields(result, allowed, path)
    return result


def _compile_merged_section(base: Mapping[str, Any], value: Any, path: str, allowed: set[str]) -> dict[str, Any]:
    result = dict(base)
    result.update(_compile_profile_section(value, path, allowed))
    return result


def _compile_exact_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ObservationTransportCompileError(f"{path} must be an exact non-empty string")
    return value


def _compile_exact_int(value: Any, path: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObservationTransportCompileError(f"{path} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ObservationTransportCompileError(f"{path} must be in {bounds}")
    return value


def _compile_default_stream_id(key: str) -> str:
    token = re.sub(r"[^a-z0-9_-]+", "_", key.lower()).strip("_-")
    if not token or not token[0].isalpha():
        token = f"stream_{token}"
    if len(token) <= 63:
        return token
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{token[:54]}_{suffix}"
