"""Benchmark mode projection onto the generic observation transport compiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robot_config.observation_transport import (
    ObservationTransportCompileError,
    materialize_observation_transports,
    observation_transport_fingerprint,
)

_MODE_VALUES = frozenset({"dds", "rtp"})
_GLOBAL_FIELDS = frozenset({"mode", "rtp", "effective_fingerprint"})


class BenchmarkObservationTransportError(ValueError):
    """Raised when Benchmark transport selection cannot be projected."""


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
    """Project Benchmark's mode choice through the generic transport compiler."""
    mode = benchmark_observation_transport_mode(robot_config)
    if mode is None:
        return robot_config

    benchmark = _mapping(robot_config.get("benchmark"), "benchmark")
    evaluation = _mapping(benchmark.get("evaluation"), "benchmark.evaluation")
    selection = _mapping(evaluation.get("observation_transport"), "benchmark.evaluation.observation_transport")
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
            "benchmark RTP routes require a distributed inference pipeline; configure the maintained Benchmark pipeline once, then users only switch observation_transport.mode"
        )

    materialized_fingerprint = selection.get("effective_fingerprint")
    if materialized_fingerprint is not None:
        if not isinstance(materialized_fingerprint, str) or len(materialized_fingerprint) != 64:
            raise BenchmarkObservationTransportError(
                "benchmark.evaluation.observation_transport.effective_fingerprint is invalid"
            )
        if any(
            not isinstance(item.get("transport"), Mapping) or str(item["transport"].get("mode", "")).lower() != mode
            for item in image_observations
        ):
            raise BenchmarkObservationTransportError(
                "materialized Benchmark image transports do not match the selected mode"
            )
        expected = observation_transport_fingerprint(robot_config, mode, pipeline_id)
        if materialized_fingerprint != expected:
            raise BenchmarkObservationTransportError(
                "materialized Benchmark observation transport fingerprint does not match the effective contract/topology"
            )
        return robot_config

    rate_hz = contract.get("rate_hz", 20.0)
    try:
        materialize_observation_transports(
            image_observations,
            mode,
            rtp=selection.get("rtp"),
            rate_hz=float(rate_hz),
        )
    except (ObservationTransportCompileError, TypeError, ValueError) as exc:
        raise BenchmarkObservationTransportError(str(exc)) from exc

    selection["mode"] = mode
    selection["effective_fingerprint"] = observation_transport_fingerprint(robot_config, mode, pipeline_id)
    return robot_config


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


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkObservationTransportError(f"{path} must be a mapping")
    return value


def _check_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise BenchmarkObservationTransportError(f"{path} contains unsupported fields: {', '.join(unknown)}")
