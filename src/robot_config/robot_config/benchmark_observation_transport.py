"""Materialize the Benchmark observation transport user choice into effective SSOT."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from robot_config.observation_transport import VIDEO_CODEC_BACKENDS

_MODE_VALUES = frozenset({"dds", "rtp"})
_GLOBAL_FIELDS = frozenset({"mode", "rtp", "effective_fingerprint"})
_RTP_FIELDS = frozenset(
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
_STREAM_FIELDS = frozenset(
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
_STREAM_TOKEN = re.compile(r"[^a-z0-9_-]+")


class BenchmarkObservationTransportError(ValueError):
    """Raised when the one-field Benchmark transport selection is invalid."""


def benchmark_observation_transport_mode(robot_config: Mapping[str, Any]) -> str | None:
    benchmark = robot_config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    evaluation = benchmark.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    config = evaluation.get("observation_transport")
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise BenchmarkObservationTransportError("benchmark.evaluation.observation_transport must be a mapping")
    mode = config.get("mode")
    if not isinstance(mode, str) or mode.strip() != mode or mode.lower() not in _MODE_VALUES:
        raise BenchmarkObservationTransportError(
            "benchmark.evaluation.observation_transport.mode must be exactly 'dds' or 'rtp'"
        )
    return mode.lower()


def materialize_benchmark_observation_transport(robot_config: dict[str, Any]) -> dict[str, Any]:
    """Apply the Benchmark-wide mode to image observations and selected topology.

    Configurations without ``benchmark.evaluation.observation_transport`` keep
    the legacy per-observation behavior unchanged. When the new field exists it
    becomes the sole transport authority for Benchmark image observations.
    """
    mode = benchmark_observation_transport_mode(robot_config)
    if mode is None:
        return robot_config

    benchmark = _mapping(robot_config.get("benchmark"), "benchmark")
    evaluation = _mapping(benchmark.get("evaluation"), "benchmark.evaluation")
    selection = _mapping(
        evaluation.get("observation_transport"),
        "benchmark.evaluation.observation_transport",
    )
    _check_fields(selection, _GLOBAL_FIELDS, "benchmark.evaluation.observation_transport")

    contract = _mapping(robot_config.get("contract"), "contract")
    observations = contract.get("observations")
    if not isinstance(observations, list):
        raise BenchmarkObservationTransportError("robot.contract.observations must be a list")
    image_observations = [
        item for item in observations if isinstance(item, dict) and str(item.get("type", "")) == "sensor_msgs/msg/Image"
    ]
    if not image_observations:
        raise BenchmarkObservationTransportError(
            "benchmark observation transport selection requires at least one sensor_msgs/msg/Image observation"
        )

    pipeline_id, pipeline = _selected_pipeline(robot_config)
    execution_mode = pipeline.get("execution_mode")
    if execution_mode not in {"monolithic", "distributed"}:
        raise BenchmarkObservationTransportError(
            "selected Benchmark inference pipeline execution_mode must be 'monolithic' or 'distributed'"
        )
    if mode == "rtp" and execution_mode != "distributed":
        raise BenchmarkObservationTransportError(
            "benchmark RTP routes require a distributed inference pipeline; "
            "configure the maintained Benchmark pipeline once, then users only switch observation_transport.mode"
        )
    materialized_fingerprint = selection.get("effective_fingerprint")
    if materialized_fingerprint is not None:
        if not isinstance(materialized_fingerprint, str) or len(materialized_fingerprint) != 64:
            raise BenchmarkObservationTransportError(
                "benchmark.evaluation.observation_transport.effective_fingerprint is invalid"
            )
        if mode == "rtp" and pipeline.get("execution_mode") != "distributed":
            raise BenchmarkObservationTransportError(
                "materialized Benchmark RTP transport requires a distributed inference pipeline"
            )
        if any(
            not isinstance(item.get("transport"), Mapping) or str(item["transport"].get("mode", "")).lower() != mode
            for item in image_observations
        ):
            raise BenchmarkObservationTransportError(
                "materialized Benchmark image transports do not match the selected mode"
            )
        expected = _effective_fingerprint(robot_config, mode, pipeline_id)
        if materialized_fingerprint != expected:
            raise BenchmarkObservationTransportError(
                "materialized Benchmark observation transport fingerprint does not match the effective contract/topology"
            )
        return robot_config

    for item in image_observations:
        if "transport" in item:
            raise BenchmarkObservationTransportError(
                "benchmark.evaluation.observation_transport is the transport authority; "
                f"remove contract observation transport from {item.get('key')!r} and use rtp.streams overrides"
            )

    if mode == "dds":
        if selection.get("rtp") is not None:
            raise BenchmarkObservationTransportError(
                "benchmark.evaluation.observation_transport.rtp is valid only when mode is 'rtp'"
            )
        for item in image_observations:
            item["transport"] = {"mode": "dds"}
    else:
        _reject_scheduled_rtp(robot_config)
        rtp = selection.get("rtp", {})
        rtp = _mapping(rtp, "benchmark.evaluation.observation_transport.rtp")
        _check_fields(rtp, _RTP_FIELDS, "benchmark.evaluation.observation_transport.rtp")
        _materialize_rtp(robot_config, image_observations, rtp)

    fingerprint = _effective_fingerprint(robot_config, mode, pipeline_id)
    selection["mode"] = mode
    selection["effective_fingerprint"] = fingerprint
    return robot_config


def _materialize_rtp(
    robot_config: Mapping[str, Any],
    image_observations: list[dict[str, Any]],
    rtp: Mapping[str, Any],
) -> None:
    host = _exact_string(rtp.get("endpoint_host", "127.0.0.1"), "rtp.endpoint_host")
    base_port = _exact_int(rtp.get("base_port", 55000), "rtp.base_port", minimum=1, maximum=65535)
    stride = _exact_int(rtp.get("port_stride", 2), "rtp.port_stride", minimum=1, maximum=65535)
    codec = _exact_string(rtp.get("codec", "h264"), "rtp.codec").lower()
    if codec != "h264":
        raise BenchmarkObservationTransportError("benchmark RTP currently supports codec 'h264' only")
    encoder = _codec_backend(rtp.get("encoder_backend", "auto"), "rtp.encoder_backend")
    decoder = _codec_backend(rtp.get("decoder_backend", "auto"), "rtp.decoder_backend")
    security = _exact_string(rtp.get("security", "none"), "rtp.security").lower()
    if security != "none":
        raise BenchmarkObservationTransportError("benchmark RTP security currently must be 'none'")

    h264 = _section(rtp.get("h264"), "rtp.h264", {"profile", "bitrate_bps", "gop_frames"})
    media = _section(
        rtp.get("media"),
        "rtp.media",
        {"width", "height", "frame_rate_hz", "pixel_format", "color_space", "color_range"},
    )
    buffer = _section(
        rtp.get("buffer"),
        "rtp.buffer",
        {"sender_queue_frames", "receiver_queue_packets", "decoded_frame_capacity", "retention_ms"},
    )
    readiness = _section(
        rtp.get("readiness"),
        "rtp.readiness",
        {"keyframe_timeout_ms", "timestamp_mapping_max_age_ms", "max_inter_camera_skew_ms"},
    )
    streams = _mapping(rtp.get("streams", {}), "benchmark.evaluation.observation_transport.rtp.streams")
    known_keys = {str(item.get("key")) for item in image_observations}
    unknown = sorted(set(streams) - known_keys)
    if unknown:
        raise BenchmarkObservationTransportError(
            f"RTP stream overrides reference unknown image observations: {unknown}"
        )

    rate_hz = _positive_number(robot_config.get("contract", {}).get("rate_hz", 20.0), "contract.rate_hz")
    used_stream_ids: set[str] = set()
    used_endpoints: set[tuple[str, int]] = set()
    for index, observation in enumerate(image_observations):
        key = _exact_string(observation.get("key"), f"contract.observations[{index}].key")
        override = _mapping(streams.get(key, {}), f"rtp.streams.{key}")
        _check_fields(override, _STREAM_FIELDS, f"rtp.streams.{key}")
        image = _mapping(observation.get("image"), f"contract observation {key}.image")
        resize = image.get("resize")
        if not isinstance(resize, list | tuple) or len(resize) != 2:
            raise BenchmarkObservationTransportError(
                f"RTP image observation {key!r} requires image.resize [height, width]"
            )
        height = _exact_int(resize[0], f"{key}.image.resize[0]", minimum=1)
        width = _exact_int(resize[1], f"{key}.image.resize[1]", minimum=1)

        stream_id = _exact_string(override.get("stream_id", _default_stream_id(key)), f"rtp.streams.{key}.stream_id")
        endpoint_override = _section(override.get("endpoint"), f"rtp.streams.{key}.endpoint", {"host", "port"})
        endpoint_host = _exact_string(endpoint_override.get("host", host), f"rtp.streams.{key}.endpoint.host")
        default_port = base_port + index * stride
        endpoint_port = _exact_int(
            endpoint_override.get("port", default_port),
            f"rtp.streams.{key}.endpoint.port",
            minimum=1,
            maximum=65535,
        )
        if stream_id in used_stream_ids:
            raise BenchmarkObservationTransportError(f"duplicate RTP stream_id {stream_id!r}")
        if (endpoint_host, endpoint_port) in used_endpoints:
            raise BenchmarkObservationTransportError(f"duplicate RTP endpoint {endpoint_host}:{endpoint_port}")
        used_stream_ids.add(stream_id)
        used_endpoints.add((endpoint_host, endpoint_port))

        stream_h264 = _merged_section(
            h264, override.get("h264"), f"rtp.streams.{key}.h264", {"profile", "bitrate_bps", "gop_frames"}
        )
        stream_media = _merged_section(
            media,
            override.get("media"),
            f"rtp.streams.{key}.media",
            {"width", "height", "frame_rate_hz", "pixel_format", "color_space", "color_range"},
        )
        stream_buffer = _merged_section(
            buffer,
            override.get("buffer"),
            f"rtp.streams.{key}.buffer",
            {"sender_queue_frames", "receiver_queue_packets", "decoded_frame_capacity", "retention_ms"},
        )
        stream_readiness = _merged_section(
            readiness,
            override.get("readiness"),
            f"rtp.streams.{key}.readiness",
            {"keyframe_timeout_ms", "timestamp_mapping_max_age_ms", "max_inter_camera_skew_ms"},
        )
        stream_encoder = _codec_backend(override.get("encoder_backend", encoder), f"rtp.streams.{key}.encoder_backend")
        stream_decoder = _codec_backend(override.get("decoder_backend", decoder), f"rtp.streams.{key}.decoder_backend")
        stream_codec = _exact_string(override.get("codec", codec), f"rtp.streams.{key}.codec").lower()
        if stream_codec != "h264":
            raise BenchmarkObservationTransportError(f"RTP stream {key!r} currently supports codec 'h264' only")
        stream_security = _exact_string(override.get("security", security), f"rtp.streams.{key}.security").lower()
        if stream_security != "none":
            raise BenchmarkObservationTransportError(f"RTP stream {key!r} security currently must be 'none'")

        observation["transport"] = {
            "mode": "rtp",
            "stream_id": stream_id,
            "endpoint": {"host": endpoint_host, "port": endpoint_port},
            "codec": stream_codec,
            "encoder_backend": stream_encoder,
            "decoder_backend": stream_decoder,
            "h264": {
                "profile": str(stream_h264.get("profile", "main")).lower(),
                "bitrate_bps": _exact_int(
                    stream_h264.get("bitrate_bps", 4_000_000), f"{key}.h264.bitrate_bps", minimum=1
                ),
                "gop_frames": _exact_int(stream_h264.get("gop_frames", 15), f"{key}.h264.gop_frames", minimum=1),
            },
            "media": {
                "width": _exact_int(stream_media.get("width", width), f"{key}.media.width", minimum=1),
                "height": _exact_int(stream_media.get("height", height), f"{key}.media.height", minimum=1),
                "frame_rate_hz": _positive_number(
                    stream_media.get("frame_rate_hz", rate_hz), f"{key}.media.frame_rate_hz"
                ),
                "pixel_format": str(stream_media.get("pixel_format", "nv12")).lower(),
                "color_space": str(stream_media.get("color_space", "bt709")).lower(),
                "color_range": str(stream_media.get("color_range", "limited")).lower(),
            },
            "buffer": {
                "sender_queue_frames": _exact_int(
                    stream_buffer.get("sender_queue_frames", 2), f"{key}.buffer.sender_queue_frames", minimum=1
                ),
                "receiver_queue_packets": _exact_int(
                    stream_buffer.get("receiver_queue_packets", 256), f"{key}.buffer.receiver_queue_packets", minimum=1
                ),
                "decoded_frame_capacity": _exact_int(
                    stream_buffer.get("decoded_frame_capacity", 32), f"{key}.buffer.decoded_frame_capacity", minimum=1
                ),
                "retention_ms": _exact_int(
                    stream_buffer.get("retention_ms", 5000), f"{key}.buffer.retention_ms", minimum=1
                ),
            },
            "readiness": {
                "keyframe_timeout_ms": _exact_int(
                    stream_readiness.get("keyframe_timeout_ms", 5000), f"{key}.readiness.keyframe_timeout_ms", minimum=1
                ),
                "timestamp_mapping_max_age_ms": _exact_int(
                    stream_readiness.get("timestamp_mapping_max_age_ms", 5000),
                    f"{key}.readiness.timestamp_mapping_max_age_ms",
                    minimum=1,
                ),
                "max_inter_camera_skew_ms": _exact_int(
                    stream_readiness.get("max_inter_camera_skew_ms", 100),
                    f"{key}.readiness.max_inter_camera_skew_ms",
                    minimum=0,
                ),
            },
            "security": stream_security,
        }


def _selected_pipeline(robot_config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    mode_name = str(robot_config.get("default_control_mode", "model_inference"))
    control_modes = _mapping(robot_config.get("control_modes"), "control_modes")
    mode = _mapping(control_modes.get(mode_name), f"control_modes.{mode_name}")
    inference = _mapping(mode.get("inference"), f"control_modes.{mode_name}.inference")
    pipelines = _mapping(inference.get("pipelines"), f"control_modes.{mode_name}.inference.pipelines")
    executor = _mapping(mode.get("executor", {}), f"control_modes.{mode_name}.executor")
    pipeline_id = executor.get("inference_pipeline")
    if pipeline_id is None:
        if len(pipelines) != 1:
            raise BenchmarkObservationTransportError(
                "benchmark observation transport requires executor.inference_pipeline when multiple pipelines exist"
            )
        pipeline_id = next(iter(pipelines))
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise BenchmarkObservationTransportError("executor.inference_pipeline must select one pipeline")
    pipeline = pipelines.get(pipeline_id)
    if not isinstance(pipeline, dict):
        raise BenchmarkObservationTransportError(f"executor selects unknown inference pipeline {pipeline_id!r}")
    return pipeline_id, pipeline


def _reject_scheduled_rtp(robot_config: Mapping[str, Any]) -> None:
    mode_name = str(robot_config.get("default_control_mode", "model_inference"))
    mode = _mapping(
        _mapping(robot_config.get("control_modes"), "control_modes").get(mode_name), f"control_modes.{mode_name}"
    )
    inference = _mapping(mode.get("inference"), f"control_modes.{mode_name}.inference")
    scheduler = inference.get("scheduler", {})
    if isinstance(scheduler, Mapping) and scheduler.get("enable") is True:
        raise BenchmarkObservationTransportError("benchmark RTP is incompatible with scheduled monolithic inference")


def _effective_fingerprint(robot_config: Mapping[str, Any], mode: str, pipeline_id: str) -> str:
    contract = copy.deepcopy(robot_config.get("contract", {}))
    control_mode = str(robot_config.get("default_control_mode", "model_inference"))
    pipeline = (
        robot_config.get("control_modes", {})
        .get(control_mode, {})
        .get("inference", {})
        .get("pipelines", {})
        .get(pipeline_id, {})
    )
    payload = {
        "schema_version": 1,
        "mode": mode,
        "pipeline_id": pipeline_id,
        "execution_mode": pipeline.get("execution_mode") if isinstance(pipeline, Mapping) else None,
        "contract": contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _default_stream_id(key: str) -> str:
    token = _STREAM_TOKEN.sub("_", key.lower()).strip("_-")
    if not token or not token[0].isalpha():
        token = f"stream_{token}"
    if len(token) <= 63:
        return token
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{token[:54]}_{suffix}"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkObservationTransportError(f"{path} must be a mapping")
    return value


def _check_fields(value: Mapping[str, Any], allowed: frozenset[str] | set[str], path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise BenchmarkObservationTransportError(f"{path} contains unsupported fields: {', '.join(unknown)}")


def _section(value: Any, path: str, allowed: set[str]) -> dict[str, Any]:
    if value is None:
        return {}
    result = dict(_mapping(value, path))
    _check_fields(result, allowed, path)
    return result


def _merged_section(base: Mapping[str, Any], value: Any, path: str, allowed: set[str]) -> dict[str, Any]:
    result = dict(base)
    result.update(_section(value, path, allowed))
    return result


def _exact_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BenchmarkObservationTransportError(f"{path} must be an exact non-empty string")
    return value


def _exact_int(value: Any, path: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkObservationTransportError(f"{path} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise BenchmarkObservationTransportError(f"{path} must be in {bounds}")
    return value


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
        raise BenchmarkObservationTransportError(f"{path} must be a positive number")
    return float(value)


def _codec_backend(value: Any, path: str) -> str:
    """Validate one native IB-Robot codec policy without selecting a backend here."""
    backend = _exact_string(value, path).lower()
    if backend not in VIDEO_CODEC_BACKENDS:
        supported = ", ".join(sorted(VIDEO_CODEC_BACKENDS))
        raise BenchmarkObservationTransportError(f"{path} must be one of: {supported}")
    return backend
