"""Benchmark launch builder for production evaluation.

The builder remains provider-agnostic: it never imports LIBERO, robosuite,
MuJoCo, ``benchmark_runtime``, or ``benchmark_libero`` Python modules.

For production evaluation it owns the launch-time run identity boundary. It
allocates one safe, non-overwriting run ID, writes an ignored materialized
robot YAML containing that ID, and passes the materialized path to every
runtime role. Native and canonical artifact owners therefore read the same
final identity from the same SSOT snapshot.
"""

from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from launch_ros.actions import Node

from robot_config.benchmark_endpoints import (
    BenchmarkEndpointError,
    resolve_benchmark_endpoints,
)
from robot_config.contract_utils import contract_fingerprint, iter_specs
from robot_config.inference_config import (
    InferenceConfigError,
    parse_inference_config,
)
from robot_config.loader import build_contract_from_robot_config_dict
from robot_config.logger_utils import get_colored_logger
from robot_config.observation_transport import effective_observation_transport

logger = get_colored_logger("robot_config.benchmark")


class BenchmarkLaunchError(RuntimeError):
    """Raised when the benchmark nodes cannot be built from SSOT."""


def _require_non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkLaunchError(f"{label} must be a non-empty string in the SSOT")
    return value


_SAFE_RUN_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _require_safe_run_component(value: Any, label: str) -> str:
    component = _require_non_empty(value, label)
    if component != component.strip():
        raise BenchmarkLaunchError(f"{label} must not contain surrounding whitespace")
    if component in {".", ".."} or not _SAFE_RUN_COMPONENT.fullmatch(component):
        raise BenchmarkLaunchError(
            f"{label} must be one safe path component containing only letters, digits, '.', '_' or '-'"
        )
    return component


def _workspace_root(config_path: str) -> Path:
    resolved = Path(config_path).expanduser().resolve()
    repository = next((parent for parent in resolved.parents if (parent / ".git").exists()), None)
    if repository is not None:
        return repository
    workspace_value = os.environ.get("WORKSPACE")
    if workspace_value:
        workspace = Path(workspace_value).expanduser()
        if workspace.is_absolute() and (workspace / ".git").exists():
            return workspace.resolve()
    return resolved.parent


def _resolve_output_root(evaluation: dict[str, Any], config_path: str) -> Path:
    output = evaluation.get("output", {})
    if not isinstance(output, dict):
        raise BenchmarkLaunchError("benchmark.evaluation.output must be a mapping when present")
    root_value = output.get("root", "outputs/benchmark")
    root_text = _require_non_empty(root_value, "benchmark.evaluation.output.root")
    output_root = Path(root_text).expanduser()
    if not output_root.is_absolute():
        output_root = _workspace_root(config_path) / output_root
    return output_root.resolve()


def _runtime_config_path(config_path: str, run_id: str) -> Path:
    return _workspace_root(config_path) / "tmp" / "benchmark_run_configs" / f"{run_id}.yaml"


def _write_materialized_config(robot_config: dict[str, Any], run_id: str, destination: Path) -> None:
    snapshot = copy.deepcopy(robot_config)
    snapshot.pop("_config_path", None)
    benchmark = snapshot.get("benchmark")
    if not isinstance(benchmark, dict):
        raise BenchmarkLaunchError("benchmark section must be a mapping")
    evaluation = benchmark.get("evaluation")
    if not isinstance(evaluation, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping")
    evaluation["run_id"] = run_id
    encoded = yaml.safe_dump({"robot": snapshot}, sort_keys=False, allow_unicode=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise
    except OSError as exc:
        raise BenchmarkLaunchError(f"cannot materialize benchmark run config at {destination}: {exc}") from exc


def _prepare_run_identity(
    robot_config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> str:
    """Allocate one run ID and materialize it into a launch-owned YAML.

    Automatic IDs use UTC ``timestamp_suite_run-name``. If the same-second
    base is already represented by either an output directory or a runtime
    config reservation, ``_01``, ``_02``, ... are tried in order. Explicit
    IDs fail closed on either condition and are never suffixed.
    """
    benchmark = robot_config.get("benchmark")
    if not isinstance(benchmark, dict):
        raise BenchmarkLaunchError("benchmark section must be a mapping")
    evaluation = benchmark.get("evaluation")
    if not isinstance(evaluation, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping")
    config_path = _require_non_empty(robot_config.get("_config_path"), "robot._config_path")
    output_root = _resolve_output_root(evaluation, config_path)

    explicit = evaluation.get("run_id")
    if explicit is not None:
        run_id = _require_safe_run_component(explicit, "benchmark.evaluation.run_id")
        destination = _runtime_config_path(config_path, run_id)
        if (output_root / run_id).exists() or destination.exists():
            raise BenchmarkLaunchError(f"benchmark run_id already exists and will not be overwritten: {run_id}")
        try:
            _write_materialized_config(robot_config, run_id, destination)
        except FileExistsError as exc:
            raise BenchmarkLaunchError(
                f"benchmark run_id already exists and will not be overwritten: {run_id}"
            ) from exc
    else:
        suite = _require_safe_run_component(evaluation.get("suite"), "benchmark.evaluation.suite")
        output = evaluation.get("output", {})
        if not isinstance(output, dict):
            raise BenchmarkLaunchError("benchmark.evaluation.output must be a mapping when present")
        run_name = _require_safe_run_component(
            output.get("run_name", "benchmark_evaluation"),
            "benchmark.evaluation.output.run_name",
        )
        timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        base = f"{timestamp}_{suite}_{run_name}"
        suffix = 0
        while True:
            run_id = base if suffix == 0 else f"{base}_{suffix:02d}"
            destination = _runtime_config_path(config_path, run_id)
            if (output_root / run_id).exists() or destination.exists():
                suffix += 1
                continue
            try:
                _write_materialized_config(robot_config, run_id, destination)
            except FileExistsError:
                suffix += 1
                continue
            break

    evaluation["run_id"] = run_id
    robot_config["_config_path"] = str(destination)
    return run_id


def _resolve_benchmark_endpoints(robot_config: dict) -> tuple[str, str, str, str, str, str]:
    """Resolve all benchmark endpoint names from the SSOT.

    Returns (namespace, reset_service, step_service, plan_service,
    finalize_service, status_topic). The plan and finalize services are
    derived from the benchmark namespace for plan and finalization services.
    GetBenchmarkPlan and FinalizeBenchmarkScope wires.
    """
    try:
        endpoints = resolve_benchmark_endpoints(robot_config)
    except BenchmarkEndpointError as exc:
        raise BenchmarkLaunchError(f"benchmark endpoint resolution failed: {exc}") from exc
    namespace = endpoints.namespace
    plan_service = f"{namespace}/get_plan"
    finalize_service = f"{namespace}/finalize"
    return (
        namespace,
        endpoints.reset_service,
        endpoints.step_service,
        plan_service,
        finalize_service,
        endpoints.status_topic,
    )


def _resolve_evaluator_endpoints(robot_config: dict) -> dict[str, str]:
    """Resolve the evaluator's target endpoints from the SSOT.

    The evaluator orchestrates:
      ResetBenchmark -> PreparePolicyEpisode -> RunPolicy -> StepBenchmark

    PreparePolicyEpisode and RunPolicy are served by the **action_dispatcher**
    node, NOT the inference_service. The inference_service serves DispatchInfer
    and health; the action_dispatcher wraps them behind its own episode gate.

    Endpoint derivation:
      - action_dispatcher node name is "action_dispatcher" (set by the
        execution launch builder). The node creates ``~/prepare_episode``
        and ``~/run_policy`` when executor_type=benchmark and
        scheduler_mode=wait_for_feedback, which expand to:
          /action_dispatcher/prepare_episode
          /action_dispatcher/run_policy
      - inference_health_topic comes from the inference pipeline transport
        config (the only endpoint that IS from the inference_service).

    Returns a dict with the evaluator's endpoint parameter names and
    resolved values.
    """
    # action_dispatcher endpoints — derived from the node name "action_dispatcher"
    # (set in execution.py generate_action_dispatcher_node). The node creates
    # ~/prepare_episode and ~/run_policy in benchmark + wait_for_feedback mode.
    action_dispatcher_name = "action_dispatcher"
    prepare_policy_episode = f"/{action_dispatcher_name}/prepare_episode"
    run_policy_action = f"/{action_dispatcher_name}/run_policy"

    # inference_health_topic — the only endpoint from the inference_service.
    control_mode = str(robot_config.get("default_control_mode", "model_inference"))
    try:
        inference = parse_inference_config(robot_config, control_mode)
    except InferenceConfigError:
        inference_health = ""
    else:
        if inference.enabled and inference.pipelines:
            pipeline = next(iter(inference.pipelines.values()))
            inference_health = pipeline.transport.health_topic
        else:
            inference_health = ""

    return {
        "prepare_policy_episode_service": prepare_policy_episode,
        "run_policy_action_server": run_policy_action,
        "inference_health_topic": inference_health,
    }


def _selected_benchmark_pipeline(robot_config: dict):
    control_mode = str(robot_config.get("default_control_mode", "model_inference"))
    inference = parse_inference_config(robot_config, control_mode)
    if not inference.enabled or not inference.pipelines:
        raise BenchmarkLaunchError("benchmark RTP requires an enabled inference pipeline")
    executor = robot_config.get("control_modes", {}).get(control_mode, {}).get("executor", {}) or {}
    selected = executor.get("inference_pipeline")
    if selected is None:
        if len(inference.pipelines) != 1:
            raise BenchmarkLaunchError("benchmark executor must select one inference pipeline")
        return next(iter(inference.pipelines.values()))
    try:
        return inference.pipelines[selected]
    except KeyError as exc:
        raise BenchmarkLaunchError(f"benchmark executor selects unknown pipeline {selected!r}") from exc


def _frame_ingress_parameters(robot_config: dict, benchmark: dict) -> dict[str, Any]:
    contract = build_contract_from_robot_config_dict(robot_config)
    streams = []
    for spec in iter_specs(contract):
        if spec.is_action:
            continue
        transport = effective_observation_transport(spec.transport)
        if transport.mode != "rtp":
            continue
        if transport.stream_id is None or transport.endpoint is None:
            raise BenchmarkLaunchError(f"RTP observation {spec.key!r} lacks stream identity or endpoint")
        if transport.media is None or transport.h264 is None or transport.buffer is None:
            raise BenchmarkLaunchError(f"RTP observation {spec.key!r} lacks resolved media/codec/buffer settings")
        streams.append(
            {
                "observation_key": spec.key,
                "stream_id": transport.stream_id,
                "endpoint_host": transport.endpoint.host,
                "endpoint_port": transport.endpoint.port,
                "width": transport.media.width,
                "height": transport.media.height,
                "frame_rate_hz": transport.media.frame_rate_hz,
                "bitrate_bps": transport.h264.bitrate_bps,
                "gop_frames": transport.h264.gop_frames,
                "codec": transport.codec,
                "codec_profile": transport.h264.profile,
                "encoder_backend": transport.encoder_backend,
                "decoder_backend": transport.decoder_backend,
                "pixel_format": transport.media.pixel_format,
                "input_pixel_format": "rgb24",
                "color_space": transport.media.color_space,
                "color_range": transport.media.color_range,
                "sender_queue_frames": transport.buffer.sender_queue_frames,
                "raw_queue_frames": transport.buffer.sender_queue_frames,
                "queue_policy": "strict",
            }
        )
    enabled = bool(streams)
    pipeline = _selected_benchmark_pipeline(robot_config) if enabled else None
    if pipeline is not None and pipeline.execution_mode != "distributed":
        raise BenchmarkLaunchError("benchmark RTP routes require execution_mode=distributed")
    evaluation = benchmark.get("evaluation", {}) or {}
    timeout = (evaluation.get("timeouts", {}) or {}).get("startup_timeout_sec", 120.0)
    transport = pipeline.transport if pipeline is not None else None
    ingress_contract = (
        {
            "schema_version": 1,
            "pipeline_id": pipeline.pipeline_id,
            "contract_fingerprint": contract_fingerprint(contract),
            "deployment_fingerprint": pipeline.validated_manifest.fingerprint,
            "heartbeat_topic": transport.heartbeat_topic,
            "descriptor_topic": transport.video_descriptor_topic,
            "status_topic": transport.video_status_topic,
            "streams": streams,
        }
        if pipeline is not None and transport is not None
        else {}
    )
    return {
        "frame_ingress_enabled": enabled,
        "frame_ingress_contract_json": json.dumps(ingress_contract, sort_keys=True, separators=(",", ":")),
        "frame_ingress_startup_timeout_sec": float(timeout),
    }


def _build_environment_node(
    robot_config: dict,
    benchmark: dict,
    reset_service: str,
    step_service: str,
    plan_service: str,
    finalize_service: str,
) -> Node:
    """Build the production benchmark environment node."""
    return Node(
        package="benchmark_runtime",
        executable="benchmark_environment_node",
        name="benchmark_environment",
        parameters=[
            {
                "robot_config_path": str(robot_config["_config_path"]),
                "benchmark_type": _require_non_empty(benchmark.get("type"), "benchmark.type"),
                "adapter_name": _require_non_empty(benchmark.get("adapter"), "benchmark.adapter"),
                "reset_service": reset_service,
                "step_service": step_service,
                "plan_service": plan_service,
                "finalize_service": finalize_service,
                **_frame_ingress_parameters(robot_config, benchmark),
            }
        ],
        output="screen",
    )


def _build_evaluator_node(
    robot_config: dict,
    benchmark: dict,
    endpoint_params: dict[str, str] | None = None,
) -> Node:
    """Build the production benchmark evaluator node."""
    evaluation_section = benchmark.get("evaluation", {})
    if not isinstance(evaluation_section, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping when present")
    evaluation_enabled = bool(evaluation_section.get("enabled", False))

    params: dict[str, Any] = {
        "robot_config_path": str(robot_config["_config_path"]),
        "evaluation_enabled": evaluation_enabled,
    }
    # Endpoint and run identity parameters are supplied by the caller.
    if endpoint_params:
        params.update(endpoint_params)

    return Node(
        package="benchmark_runtime",
        executable="benchmark_evaluator_node",
        name="benchmark_evaluator",
        parameters=[params],
        output="screen",
    )


def generate_benchmark_nodes(robot_config: dict) -> list[Node]:
    """Build production benchmark nodes from the robot configuration.

    The builder references benchmark packages by ROS package/executable name
    and keeps provider imports outside ``robot_config``. When evaluation is
    enabled it launches both the environment and evaluator; otherwise it
    launches only the environment.
    """

    _require_non_empty(robot_config.get("_config_path"), "robot._config_path")

    benchmark = robot_config.get("benchmark", {})
    if not isinstance(benchmark, dict):
        raise BenchmarkLaunchError("benchmark section must be a mapping")
    _require_non_empty(benchmark.get("type"), "benchmark.type")
    _require_non_empty(benchmark.get("adapter"), "benchmark.adapter")
    evaluation_section = benchmark.get("evaluation", {})
    if not isinstance(evaluation_section, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping when present")
    evaluation_enabled = bool(evaluation_section.get("enabled", False))

    (
        namespace,
        reset_service,
        step_service,
        plan_service,
        finalize_service,
        status_topic,
    ) = _resolve_benchmark_endpoints(robot_config)

    evaluator_action_endpoints = _resolve_evaluator_endpoints(robot_config)
    run_id = _prepare_run_identity(robot_config) if evaluation_enabled else ""

    environment_node = _build_environment_node(
        robot_config,
        benchmark,
        reset_service=reset_service,
        step_service=step_service,
        plan_service=plan_service,
        finalize_service=finalize_service,
    )

    if not evaluation_enabled:
        logger.info("benchmark evaluation disabled: launching environment node only")
        return [environment_node]

    endpoint_params = {
        "benchmark_namespace": namespace,
        "benchmark_reset_service": reset_service,
        "benchmark_step_service": step_service,
        "benchmark_plan_service": plan_service,
        "benchmark_finalize_service": finalize_service,
        "benchmark_status_topic": status_topic,
        "prepare_policy_episode_service": evaluator_action_endpoints["prepare_policy_episode_service"],
        "run_policy_action_server": evaluator_action_endpoints["run_policy_action_server"],
        "inference_health_topic": evaluator_action_endpoints["inference_health_topic"],
        "run_id": run_id,
    }
    evaluator_node = _build_evaluator_node(
        robot_config,
        benchmark,
        endpoint_params=endpoint_params,
    )
    logger.info("benchmark evaluation enabled: launching environment and evaluator nodes")
    return [environment_node, evaluator_node]
