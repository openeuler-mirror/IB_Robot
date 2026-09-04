"""Provider-agnostic serial benchmark evaluator ROS node.

The evaluator orchestrates the configured reset, policy, step, and finalization
contracts, writes canonical reports, and emits a durable terminal marker.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from diagnostic_msgs.msg import DiagnosticStatus
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from benchmark_runtime._strict_json import StrictJSONError, loads_strict, require_json_array, require_json_object
from benchmark_runtime.canonical_models import ArtifactStatus, CanonicalEpisodeRecord
from benchmark_runtime.canonical_report import CanonicalReportWriter, reduce_episode_records
from benchmark_runtime.disk_preflight import check_disk_preflight, estimate_disk_usage
from benchmark_runtime.evaluator_classifier import EpisodeOutcome, RunPolicyResultData, classify_run_policy
from benchmark_runtime.evaluator_contract import validate_contract_compatibility
from benchmark_runtime.evaluator_readiness import ReadinessBarrier, ReadinessError, parse_health_values
from benchmark_runtime.evaluator_supervisor import (
    FinalizeCommand,
    PrepareCommand,
    ResetCommand,
    RunPolicyCommand,
    SerialSupervisor,
    SupervisorError,
)
from benchmark_runtime.finalization import finalization_payload_to_wire
from benchmark_runtime.io_descriptor import (
    descriptor_from_robot_contract,
    descriptor_from_robot_model,
    validate_io_compatibility,
)
from benchmark_runtime.plan import BenchmarkPlanJSONError, benchmark_plan_from_json

READY_TAG = "[IBROBOT_BENCHMARK][EVALUATOR_NODE_READY]"
PRODUCTION_READY_TAG = "[IBROBOT_BENCHMARK][EVALUATOR_PRODUCTION_READY]"
COMMAND_TAG = "[IBROBOT_BENCHMARK][EVALUATOR_COMMAND]"
RESULT_TAG = "[IBROBOT_BENCHMARK][EVALUATOR_RESULT]"
EVALUATION_COMPLETE_TAG = "[IBROBOT_BENCHMARK][EVALUATION_COMPLETE]"
EVALUATION_FINISHED_TAG = "[IBROBOT_BENCHMARK][EVALUATION_FINISHED]"
ERROR_TAG = "[IBROBOT_BENCHMARK][EVALUATOR_ERROR]"


class ScaffoldValidationError(RuntimeError):
    """Raised when the evaluator scaffold cannot satisfy its READY contract."""


class EvaluatorStartupError(RuntimeError):
    """Raised when production evaluator parameters/contracts are invalid."""


def _require_absolute_endpoint(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise EvaluatorStartupError(f"{name} must be an absolute ROS path, got {value!r}")
    return value


def _load_robot_mapping(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        raise EvaluatorStartupError(f"robot_config_path does not exist: {path_str}")
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise EvaluatorStartupError("robot config must be a mapping")
    robot = raw.get("robot", raw)
    if not isinstance(robot, dict):
        raise EvaluatorStartupError("robot config robot section must be a mapping")
    robot["_config_path"] = str(path.resolve())
    return robot


def _validate_robot_model_contract(robot_config_path: str) -> None:
    """Validate robot/model semantics with the Phase 2 canonical descriptors."""
    robot = _load_robot_mapping(robot_config_path)
    robot_descriptor = descriptor_from_robot_contract(robot)
    model_descriptor = descriptor_from_robot_model(robot)
    # The evaluator owns robot/model readiness.  The environment node adds the
    # provider adapter as the third side before it creates reset services.
    validate_io_compatibility(
        adapter=robot_descriptor,
        robot=robot_descriptor,
        model=model_descriptor,
    )
    benchmark = robot.get("benchmark", {})
    control_mode = benchmark.get("control_mode") if isinstance(benchmark, dict) else None
    validate_contract_compatibility(
        observations=tuple((item.key, item.kind, item.shape) for item in model_descriptor.observations),
        actions=tuple((item.key, item.shape) for item in model_descriptor.actions),
        control_mode=str(control_mode or robot.get("default_control_mode", "model_inference")),
        prompt_passthrough=True,
    )


def _artifact_image_size(robot: dict[str, Any], *, video_enabled: bool) -> tuple[int, int] | None:
    contract = robot.get("contract", {})
    observations = contract.get("observations", []) if isinstance(contract, dict) else []
    if not isinstance(observations, list):
        raise EvaluatorStartupError("robot.contract.observations must be a list")
    dimensions: list[tuple[int, int]] = []
    for item in observations:
        if not isinstance(item, dict) or item.get("type") != "sensor_msgs/msg/Image":
            continue
        image = item.get("image")
        resize = image.get("resize") if isinstance(image, dict) else None
        if not isinstance(resize, list | tuple) or len(resize) != 2:
            raise EvaluatorStartupError(f"image observation {item.get('key')!r} must declare image.resize [H, W]")
        height, width = resize
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (height, width)):
            raise EvaluatorStartupError(f"image observation {item.get('key')!r} has invalid image.resize")
        dimensions.append((width, height))
    if dimensions:
        return (max(width for width, _ in dimensions), max(height for _, height in dimensions))
    if video_enabled:
        raise EvaluatorStartupError("video artifacts require at least one image observation in robot.contract")
    return None


def _package_identity(distribution_name: str) -> dict[str, str | None]:
    """Collect package identity without importing heavy provider modules."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "path": None}
    return {
        "version": distribution.version,
        "path": str(Path(distribution.locate_file("")).resolve()),
    }


def _git_identity(workspace_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "submodules": []}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        raw = subprocess.check_output(
            ["git", "submodule", "status"],
            cwd=workspace_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        result["submodules"] = [line.strip() for line in raw.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _runtime_manifest_identity(robot_config_path: str) -> dict[str, Any]:
    robot = _load_robot_mapping(robot_config_path)
    config_path = Path(robot_config_path).resolve()
    workspace_root = next((parent for parent in config_path.parents if (parent / ".git").exists()), config_path.parent)
    mode_name = str(robot.get("default_control_mode", "model_inference"))
    modes = robot.get("control_modes", {})
    mode = modes.get(mode_name, {}) if isinstance(modes, dict) else {}
    inference = mode.get("inference", {}) if isinstance(mode, dict) else {}
    pipelines = inference.get("pipelines", {}) if isinstance(inference, dict) else {}
    executor = mode.get("executor", {}) if isinstance(mode, dict) else {}
    selected = executor.get("inference_pipeline") if isinstance(executor, dict) else None
    if selected is None and isinstance(pipelines, dict) and len(pipelines) == 1:
        selected = next(iter(pipelines))
    pipeline_id = selected if isinstance(selected, str) and selected in pipelines else None
    pipeline = pipelines.get(pipeline_id, {}) if pipeline_id is not None else {}
    model_path = pipeline.get("model_path") if isinstance(pipeline, dict) else None
    deployment = pipeline.get("deployment") if isinstance(pipeline, dict) else None
    bundle_path: Path | None = None
    inference_manifest: dict[str, Any] = {}
    if isinstance(model_path, str) and model_path.strip():
        bundle_path = Path(model_path)
        if not bundle_path.is_absolute():
            bundle_path = workspace_root / bundle_path
        manifest_path = bundle_path / "inference_manifest.json"
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    inference_manifest = loaded
            except (OSError, json.JSONDecodeError):
                pass
    benchmark = robot.get("benchmark", {})
    evaluation = benchmark.get("evaluation", {}) if isinstance(benchmark, dict) else {}
    observation_transport = evaluation.get("observation_transport", {}) if isinstance(evaluation, dict) else {}
    provider_packages = benchmark.get("identity_packages", []) if isinstance(benchmark, dict) else []
    if not isinstance(provider_packages, list) or any(
        not isinstance(name, str) or not name for name in provider_packages
    ):
        provider_packages = []
    package_names = tuple(dict.fromkeys(("torch", "torchvision", *provider_packages)))
    packages = {name: _package_identity(name) for name in package_names}
    deployment_payload = (
        inference_manifest.get("deployments", {}).get(deployment, {})
        if isinstance(inference_manifest.get("deployments"), dict)
        else {}
    )
    return {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": packages,
        "git": _git_identity(workspace_root),
        "inference": {
            "pipeline_id": pipeline_id,
            "bundle_path": str(bundle_path.resolve()) if bundle_path is not None else None,
            "deployment": deployment,
            "bundle": inference_manifest.get("bundle"),
            "deployment_identity": deployment_payload,
        },
        "environment_contract": robot.get("contract", {}),
        "observation_transport": {
            "mode": observation_transport.get("mode") if isinstance(observation_transport, dict) else None,
            "effective_fingerprint": (
                observation_transport.get("effective_fingerprint") if isinstance(observation_transport, dict) else None
            ),
            "execution_mode": pipeline.get("execution_mode") if isinstance(pipeline, dict) else None,
        },
    }


def _configured_startup_timeout_sec(robot_config_path: str) -> float:
    """Read the single startup budget from the robot SSOT before ROS waiting.

    The evaluator must own one monotonic deadline even when the plan service is
    absent or never replies.  The returned provider plan is later required to
    carry this exact timeout, so this is not a second configuration authority.
    """
    robot = _load_robot_mapping(robot_config_path)
    benchmark = robot.get("benchmark")
    if not isinstance(benchmark, dict):
        raise EvaluatorStartupError("robot benchmark section must be a mapping")
    evaluation = benchmark.get("evaluation")
    if not isinstance(evaluation, dict):
        raise EvaluatorStartupError("benchmark.evaluation must be a mapping")
    timeouts = evaluation.get("timeouts")
    if not isinstance(timeouts, dict):
        raise EvaluatorStartupError("benchmark.evaluation.timeouts must be a mapping")
    value = timeouts.get("startup_timeout_sec")
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise EvaluatorStartupError("evaluation.timeouts.startup_timeout_sec must be a number")
    value = float(value)
    if not value > 0.0:
        raise EvaluatorStartupError("evaluation.timeouts.startup_timeout_sec must be positive")
    return value


class BenchmarkEvaluatorNode(Node):
    """Provider-agnostic serial benchmark evaluator."""

    def __init__(
        self,
        parameter_overrides: list[Parameter] | None = None,
        *,
        monotonic_fn=time.monotonic,
    ) -> None:
        super().__init__("benchmark_evaluator", parameter_overrides=parameter_overrides)
        self._monotonic = monotonic_fn
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("evaluation_enabled", False)
        self.declare_parameter("scaffold_mode", False)
        self.declare_parameter("benchmark_plan_service", "")
        self.declare_parameter("benchmark_reset_service", "")
        self.declare_parameter("benchmark_finalize_service", "")
        self.declare_parameter("prepare_policy_episode_service", "")
        self.declare_parameter("run_policy_action_server", "")
        self.declare_parameter("inference_health_topic", "")
        self.declare_parameter("run_id", "")

        self._plan_client: Any = None
        self._reset_client: Any = None
        self._finalize_client: Any = None
        self._prepare_client: Any = None
        self._run_action_client: Any = None
        self._health_subscription: Any = None
        self._timer: Any = None
        self._latest_health: tuple[int, dict[str, str]] | None = None
        self._startup_start_monotonic = 0.0
        self._readiness: ReadinessBarrier | None = None
        self._supervisor: SerialSupervisor | None = None
        self._plan_future: Any = None
        self._pending_future: Any = None
        self._pending_kind = ""
        self._goal_handle: Any = None
        self._failed = False
        self._production_ready = False
        self._completion_logged = False
        self._terminal_requested = False
        self._terminal_exit_code = 0
        self._run_id = ""
        self._last_readiness_log_monotonic = float("-inf")
        self._canonical_writer: CanonicalReportWriter | None = None
        self._canonical_manifest: dict[str, Any] | None = None
        self._canonical_output_root: Path | None = None
        self._canonical_config: dict[str, Any] | None = None
        self._canonical_reset_metadata: dict[str, Any] | None = None
        self._canonical_episode_started_monotonic: float | None = None
        self._canonical_outcome: EpisodeOutcome | None = None
        self._canonical_finalized = False
        self._canonical_final_status: str | None = None
        self._canonical_run_partial = False
        self._canonical_episode_payload: Any = None
        self._canonical_scope_artifacts: list[ArtifactStatus] = []
        self._pending_scope = ""
        self._robot_config_path = ""

    def read_parameters(self) -> dict[str, Any]:
        names = (
            "robot_config_path",
            "evaluation_enabled",
            "scaffold_mode",
            "benchmark_plan_service",
            "benchmark_reset_service",
            "benchmark_finalize_service",
            "prepare_policy_episode_service",
            "run_policy_action_server",
            "inference_health_topic",
            "run_id",
        )
        return {name: self.get_parameter(name).value for name in names}

    def announce_ready(self) -> dict[str, Any]:
        """Validate and announce the no-business-handles scaffold mode."""
        params = self.read_parameters()
        robot_config_path = params["robot_config_path"]
        if not isinstance(robot_config_path, str) or not robot_config_path.strip():
            self.get_logger().error("robot_config_path must be a non-empty string")
            raise ScaffoldValidationError("robot_config_path must be a non-empty string")
        if params["scaffold_mode"] is not True:
            self.get_logger().error("scaffold_mode must be true for announce_ready()")
            raise ScaffoldValidationError("scaffold_mode must be true")
        enabled = "true" if bool(params["evaluation_enabled"]) else "false"
        self.get_logger().info(f"{READY_TAG} enabled={enabled} scaffold=true")
        return params

    def start_production(self, *, monotonic_now: float | None = None) -> None:
        params = self.read_parameters()
        if params["scaffold_mode"] is True:
            raise EvaluatorStartupError("production evaluator requires scaffold_mode=false")
        if not params["evaluation_enabled"]:
            raise EvaluatorStartupError("production evaluator requires evaluation_enabled=true")
        robot_config_path = params["robot_config_path"]
        if not isinstance(robot_config_path, str) or not robot_config_path.strip():
            raise EvaluatorStartupError("robot_config_path must be a non-empty string")
        endpoint_names = (
            "benchmark_plan_service",
            "benchmark_reset_service",
            "benchmark_finalize_service",
            "prepare_policy_episode_service",
            "run_policy_action_server",
            "inference_health_topic",
        )
        endpoints = {name: _require_absolute_endpoint(params[name], name) for name in endpoint_names}
        run_id = params["run_id"]
        if run_id is not None and not isinstance(run_id, str):
            raise EvaluatorStartupError("run_id must be a string when provided")
        startup_timeout_sec = _configured_startup_timeout_sec(robot_config_path)
        self._startup_start_monotonic = self._monotonic() if monotonic_now is None else monotonic_now
        self._readiness = ReadinessBarrier(self._startup_start_monotonic, startup_timeout_sec)
        _validate_robot_model_contract(robot_config_path)
        self._readiness.set_contract_compatible()

        from ibrobot_msgs.action import RunPolicy  # noqa: PLC0415
        from ibrobot_msgs.srv import (  # noqa: PLC0415
            FinalizeBenchmarkScope,
            GetBenchmarkPlan,
            PreparePolicyEpisode,
            ResetBenchmark,
        )

        self._plan_client = self.create_client(GetBenchmarkPlan, endpoints["benchmark_plan_service"])
        self._reset_client = self.create_client(ResetBenchmark, endpoints["benchmark_reset_service"])
        self._finalize_client = self.create_client(FinalizeBenchmarkScope, endpoints["benchmark_finalize_service"])
        self._prepare_client = self.create_client(PreparePolicyEpisode, endpoints["prepare_policy_episode_service"])
        self._run_action_client = ActionClient(self, RunPolicy, endpoints["run_policy_action_server"])
        self._health_subscription = self.create_subscription(
            DiagnosticStatus, endpoints["inference_health_topic"], self._health_callback, 10
        )
        self._timer = self.create_timer(0.05, self._drive)
        self._run_id = run_id.strip() if isinstance(run_id, str) else ""
        self._robot_config_path = robot_config_path

    def _set_readiness_plan(self, plan: Any) -> None:
        """Accept resolved plans at the canonical readiness gate."""
        assert self._readiness is not None
        self._readiness.set_plan(plan, allow_artifacts=bool(plan.artifacts.write_canonical))

    def _evaluation_config(self) -> dict[str, Any]:
        robot = _load_robot_mapping(self._robot_config_path)
        evaluation = robot.get("benchmark", {}).get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise EvaluatorStartupError("benchmark.evaluation must be a mapping")
        return evaluation

    def _resolve_canonical_identity(self, plan: Any, parameter_run_id: Any) -> None:
        evaluation = self._evaluation_config()
        output = evaluation.get("output", {})
        if not isinstance(output, dict):
            raise EvaluatorStartupError("benchmark.evaluation.output must be a mapping")
        configured = evaluation.get("run_id") or parameter_run_id or output.get("run_name")
        if not isinstance(configured, str) or not configured.strip():
            configured = f"{plan.suite}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        configured = configured.strip()
        if configured in {".", ".."} or Path(configured).name != configured:
            raise EvaluatorStartupError("evaluation run_id must be a single path component")
        self._run_id = configured

    def _start_canonical_run_if_enabled(self, plan: Any) -> None:
        if not plan.artifacts.write_canonical:
            return
        evaluation = self._evaluation_config()
        output = evaluation.get("output", {})
        if not isinstance(output, dict):
            raise EvaluatorStartupError("benchmark.evaluation.output must be a mapping")
        root_value = output.get("root", "outputs/benchmark")
        if not isinstance(root_value, str) or not root_value.strip():
            raise EvaluatorStartupError("evaluation.output.root must be a non-empty string")
        output_root = Path(root_value).expanduser()
        if not output_root.is_absolute():
            config_path = Path(self._robot_config_path).resolve()
            workspace = next(
                (parent for parent in config_path.parents if (parent / ".git").exists()), config_path.parent
            )
            output_root = workspace / output_root
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        robot = _load_robot_mapping(self._robot_config_path)
        video = evaluation.get("video", {})
        if not isinstance(video, dict):
            raise EvaluatorStartupError("benchmark.evaluation.video must be a mapping")
        artifact_config = {"video": video, "save_sim_states": evaluation.get("save_sim_states", False)}
        image_size = _artifact_image_size(robot, video_enabled=bool(video.get("enabled", False)))
        estimate = estimate_disk_usage(
            plan,
            artifact_config,
            image_size,
            max_steps=plan.max_steps,
        )
        checked = check_disk_preflight(output_root, estimate)
        if not checked.passed:
            raise EvaluatorStartupError(
                f"insufficient disk space at {output_root}: free={checked.free_bytes} required={checked.required_bytes}"
            )
        run_root = output_root / self._run_id
        manifest = {
            "schema_version": 1,
            "artifact_layout_version": 1,
            "run_id": self._run_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "suite": plan.suite,
            "benchmark_type": plan.benchmark_type,
            "adapter": plan.adapter,
            "output_root": str(output_root),
            "run_root": str(run_root),
            "robot_config_path": str(Path(self._robot_config_path).resolve()),
            "runtime_identity": _runtime_manifest_identity(self._robot_config_path),
            "plan": {
                "requested_task_order_index": plan.requested_task_order_index,
                "effective_task_order_index": plan.effective_task_order_index,
                "selected_task_ids": list(plan.selected_task_ids),
                "planned_episodes": plan.planned_episodes,
                "episodes_per_task": plan.episodes_per_task,
                "seed": plan.seed,
                "init_state_policy": {
                    "use_init_state_id": plan.init_state_policy.use_init_state_id,
                    "selection": plan.init_state_policy.selection,
                },
                "max_steps": plan.max_steps,
                "lane_count": plan.lane_count,
                "timeouts": {
                    "startup_timeout_sec": plan.timeouts.startup_timeout_sec,
                    "max_duration_sec": plan.timeouts.max_duration_sec,
                },
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "prompt": task.prompt,
                        "init_state_count": task.init_state_count,
                        "metadata": dict(task.metadata),
                    }
                    for task in plan.tasks
                ],
            },
            "artifacts": {
                "save_sim_states": plan.artifacts.save_sim_states,
                "video_enabled": plan.artifacts.video_enabled,
                "write_native": plan.artifacts.write_native,
                "write_canonical": plan.artifacts.write_canonical,
                "video": video,
            },
            "disk_preflight": checked.to_dict(),
        }
        writer = CanonicalReportWriter(run_root)
        writer.start_run(manifest)
        self._canonical_writer = writer
        self._canonical_manifest = manifest
        self._canonical_output_root = output_root
        self._canonical_config = {"video": video, "save_sim_states": bool(evaluation.get("save_sim_states", False))}
        self.get_logger().info(
            f"[IBROBOT_BENCHMARK][CANONICAL_READY] run_id={self._run_id} "
            f"run_root={run_root} free_bytes={checked.free_bytes} required_bytes={checked.required_bytes}"
        )

    def _planned_tasks(self) -> dict[int, dict[str, Any]]:
        assert self._supervisor is not None
        return {
            task.task_id: {"planned_episodes": self._supervisor.plan.episodes_per_task, "task_name": task.name}
            for task in self._supervisor.plan.tasks
            if task.task_id in self._supervisor.plan.selected_task_ids
        }

    def _canonical_artifacts(self, response: Any, scope: str) -> tuple[ArtifactStatus, ...]:
        raw_refs = getattr(response, "artifact_refs_json", "[]") or "[]"
        refs = require_json_array(loads_strict(str(raw_refs)), "artifact_refs_json")
        status_raw = getattr(response, "artifact_status_json", "")
        entries: list[Any] = []
        failures: list[Any] = []
        if status_raw:
            parsed = loads_strict(str(status_raw))
            if isinstance(parsed, dict):
                entries = parsed.get("artifacts", [])
                failures = parsed.get("failures", [])
            elif isinstance(parsed, list):
                entries = parsed
        artifacts: list[ArtifactStatus] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            if "ref" not in item and "path" in item:
                item["ref"] = item.pop("path")
            item.setdefault("scope", scope)
            item.setdefault("required", True)
            item.setdefault("status", "available")
            item.setdefault("kind", Path(str(item.get("ref", "artifact"))).suffix.lstrip(".") or "artifact")
            if item.get("status") in {"failed", "missing"}:
                item.setdefault("error", "artifact unavailable")
            artifacts.append(ArtifactStatus(**item))
        for index, failure in enumerate(failures):
            if not isinstance(failure, dict):
                continue
            stage = str(failure.get("stage", "artifact"))
            error = str(failure.get("error", "artifact unavailable"))
            artifacts.append(
                ArtifactStatus(
                    ref=str(failure.get("ref", f"missing://{scope}/{stage}/{index}")),
                    kind=str(failure.get("kind", stage)),
                    scope=scope,
                    required=True,
                    status="failed",
                    error=error,
                    metadata={key: value for key, value in failure.items() if key not in {"ref", "kind", "error"}},
                )
            )
        known = {artifact.ref for artifact in artifacts}
        for ref in refs:
            if not isinstance(ref, str) or not ref.strip() or ref in known:
                continue
            artifacts.append(
                ArtifactStatus(
                    ref=ref,
                    kind=Path(ref).suffix.lstrip(".") or "artifact",
                    scope=scope,
                    required=True,
                    status="available",
                )
            )
        return tuple(artifacts)

    def _record_canonical_scope_artifacts(self, response: Any, scope: str) -> tuple[ArtifactStatus, ...]:
        artifacts = self._canonical_artifacts(response, scope)
        if scope != "episode":
            known = {(item.ref, item.status, item.error) for item in self._canonical_scope_artifacts}
            for artifact in artifacts:
                key = (artifact.ref, artifact.status, artifact.error)
                if key not in known:
                    self._canonical_scope_artifacts.append(artifact)
                    known.add(key)
            if any(item.failed for item in artifacts):
                self._canonical_run_partial = True
        return artifacts

    def _with_scope_failures(self, summary: Any) -> Any:
        failure_items = [
            *summary.artifact_failures,
            *[item for item in self._canonical_scope_artifacts if item.failed],
        ]
        failures = tuple(
            {(item.ref, item.kind, item.scope, item.status, item.error): item for item in failure_items}.values()
        )
        metadata = dict(summary.metadata)
        metadata["artifact_index"] = [item.to_dict() for item in self._canonical_scope_artifacts]
        metadata.setdefault("cleanup_pending", True)
        writer = self._canonical_writer
        unresolved_refs = {item.ref for item in self._canonical_scope_artifacts if item.failed}
        failed_records: set[tuple[int, int]] = set()
        failed_by_task: dict[int, int] = {}
        if writer is not None and unresolved_refs:
            for record in writer.records:
                if any(item.ref in unresolved_refs for item in record.artifacts):
                    failed_records.add((record.task_id, record.episode_index))
            for task_id, _ in failed_records:
                failed_by_task[task_id] = failed_by_task.get(task_id, 0) + 1
        counters = replace(
            summary.counters,
            artifact_failure_episodes=max(summary.counters.artifact_failure_episodes, len(failed_records)),
        )
        tasks = tuple(
            replace(
                task,
                artifact_failure_episodes=max(task.artifact_failure_episodes, failed_by_task.get(task.task_id, 0)),
                status=("partial" if failed_by_task.get(task.task_id, 0) else task.status),
            )
            for task in summary.tasks
        )
        return replace(
            summary,
            counters=counters,
            tasks=tasks,
            artifact_failures=failures,
            metadata=metadata,
        )

    def _resolve_pending_artifact_completeness(self) -> None:
        writer = self._canonical_writer
        if writer is None:
            return
        available = {item.ref for item in self._canonical_scope_artifacts if item.status == "available"}
        known_failures = {item.ref for item in self._canonical_scope_artifacts if item.failed}
        for record in writer.records:
            for artifact in record.artifacts:
                if not artifact.required or artifact.status != "pending":
                    continue
                if artifact.ref in available or artifact.ref in known_failures:
                    continue
                failure = ArtifactStatus(
                    ref=artifact.ref,
                    kind=artifact.kind,
                    scope=artifact.scope,
                    required=True,
                    status="missing",
                    error="required artifact was not available after run finalization",
                    metadata=artifact.metadata,
                )
                self._canonical_scope_artifacts.append(failure)
                known_failures.add(failure.ref)
                self._canonical_run_partial = True

    def _append_canonical_episode(self, response: Any) -> None:
        if self._canonical_writer is None:
            return
        assert self._supervisor is not None
        payload = self._canonical_episode_payload
        if payload is None:
            raise SupervisorError("episode finalization payload is unavailable")
        result = dict(payload.result)
        metadata = self._canonical_reset_metadata or {}
        has_success = bool(result.get("has_success", False))
        started = self._canonical_episode_started_monotonic
        duration = 0.0 if started is None else max(0.0, self._monotonic() - started)
        task = self._supervisor.plan.tasks[payload.identity.task_id]
        record = CanonicalEpisodeRecord(
            run_id=self._run_id,
            suite=payload.identity.suite or self._supervisor.plan.suite,
            task_id=payload.identity.task_id,
            task_name=task.name,
            episode_index=payload.identity.episode_index,
            episode_id=payload.identity.episode_id,
            seed=self._supervisor.plan.seed,
            requested_init_state_id=metadata.get("requested_init_state_id"),
            actual_init_state_id=metadata.get("actual_init_state_id"),
            executed_steps=int(result.get("published_actions", 0)),
            duration_s=duration,
            terminated=bool(result.get("terminated", False)),
            truncated=bool(result.get("truncated", False)),
            termination_reason=payload.termination_reason,
            native_result_available=has_success,
            is_success=bool(result.get("success", False)) if has_success else None,
            infrastructure_error=payload.error_category or None,
            artifacts=self._record_canonical_scope_artifacts(response, "episode"),
            metadata={
                "prompt": task.prompt,
                "reset_observation_delivery": metadata.get("observation_delivery"),
                "reward": result.get("reward") if result.get("has_reward") else None,
                "standard_metrics": loads_strict(result.get("standard_metrics_json", "{}")),
                "native_metrics": loads_strict(result.get("native_metrics_json", "{}")),
                "info": loads_strict(result.get("info_json", "{}")),
                "classifier": {
                    "completed": bool(self._canonical_outcome.completed) if self._canonical_outcome else False,
                    "partial": bool(self._canonical_outcome.partial) if self._canonical_outcome else True,
                    "error_category": self._canonical_outcome.error_category
                    if self._canonical_outcome
                    else payload.error_category,
                },
            },
        )
        self._canonical_writer.append_episode(record)
        self._canonical_reset_metadata = None
        self._canonical_episode_started_monotonic = None
        self._canonical_episode_payload = None
        self._canonical_run_partial |= bool(
            record.infrastructure_error or any(item.status in {"failed", "missing"} for item in record.artifacts)
        )

    def _write_canonical_summary(self, *, run_status: str) -> None:
        if self._canonical_writer is None:
            return
        assert self._supervisor is not None
        summary = reduce_episode_records(
            self._canonical_writer.records,
            run_id=self._run_id,
            suite=self._supervisor.plan.suite,
            planned_episodes=self._supervisor.plan.planned_episodes,
            planned_tasks=self._planned_tasks(),
            run_status=run_status,
        )
        self._canonical_writer.write_summary(self._with_scope_failures(summary))

    def _canonical_plan_complete(self) -> bool:
        writer = self._canonical_writer
        if writer is None or self._supervisor is None:
            return False
        expected = {
            (task_id, episode_index)
            for task_id in self._supervisor.plan.selected_task_ids
            for episode_index in range(self._supervisor.plan.episodes_per_task)
        }
        actual = {(record.task_id, record.episode_index) for record in writer.records}
        return actual == expected

    def _finalize_canonical_run(self) -> None:
        if self._canonical_writer is None or self._canonical_finalized:
            return
        assert self._supervisor is not None
        self._resolve_pending_artifact_completeness()
        self._canonical_run_partial |= not self._canonical_plan_complete()
        status = "partial" if self._canonical_run_partial else "complete"
        summary = reduce_episode_records(
            self._canonical_writer.records,
            run_id=self._run_id,
            suite=self._supervisor.plan.suite,
            planned_episodes=self._supervisor.plan.planned_episodes,
            planned_tasks=self._planned_tasks(),
            run_status=status,
        )
        summary = self._with_scope_failures(summary)
        final_status = summary.run_status
        manifest = dict(self._canonical_manifest or {})
        manifest.update({"status": final_status, "finished_at": datetime.now(timezone.utc).isoformat()})
        self._canonical_writer.finalize(summary, manifest=manifest)
        self._canonical_manifest = manifest
        self._canonical_final_status = final_status
        self._canonical_finalized = True

    def _abort_canonical(self, message: str) -> None:
        writer = self._canonical_writer
        if writer is None or self._canonical_finalized:
            return
        try:
            assert self._supervisor is not None
            records = writer.records
            status = "partial" if records else "failed"
            summary = reduce_episode_records(
                records,
                run_id=self._run_id,
                suite=self._supervisor.plan.suite,
                planned_episodes=self._supervisor.plan.planned_episodes,
                planned_tasks=self._planned_tasks(),
                run_status=status,
                metadata={"error": message},
            )
            manifest = dict(self._canonical_manifest or {})
            manifest.update({"status": status, "error": message, "finished_at": datetime.now(timezone.utc).isoformat()})
            writer.abort(summary, manifest=manifest)
            self._canonical_manifest = manifest
            self._canonical_final_status = summary.run_status
            self._canonical_finalized = True
        except Exception as exc:
            self.get_logger().error(f"{ERROR_TAG} canonical abort failed: {exc}")

    def _health_callback(self, message: DiagnosticStatus) -> None:
        try:
            values = parse_health_values((item.key, item.value) for item in message.values)
        except ReadinessError as exc:
            self._fail(str(exc))
            return
        level = message.level[0] if isinstance(message.level, bytes | bytearray) else int(message.level)
        self._latest_health = (level, values)

    def _drive(self) -> None:
        if self._failed:
            return
        if self.evaluation_completed:
            self._announce_completed_once()
            return
        try:
            self._drive_plan_and_readiness()
            if self._readiness is not None and self._readiness.ready:
                self._drive_transaction()
            if self.evaluation_completed:
                self._announce_completed_once()
        except Exception as exc:
            self._fail(str(exc))

    def _drive_plan_and_readiness(self) -> None:
        from ibrobot_msgs.srv import GetBenchmarkPlan  # noqa: PLC0415

        if self._readiness is None:  # pragma: no cover - startup invariant
            raise EvaluatorStartupError("readiness barrier was not initialized")
        if self._production_ready:
            return
        self._readiness.remaining(self._monotonic())

        plan_service_ready = self._plan_client.service_is_ready()
        if self._supervisor is None:
            if self._plan_future is None:
                if plan_service_ready:
                    self._plan_future = self._plan_client.call_async(GetBenchmarkPlan.Request())
                self._log_readiness_snapshot(plan_service_ready=plan_service_ready)
                return
            if not self._plan_future.done():
                self._log_readiness_snapshot(plan_service_ready=plan_service_ready)
                return
            response = self._plan_future.result()
            if response is None or not response.success:
                raise EvaluatorStartupError(f"GetBenchmarkPlan failed: {getattr(response, 'message', '')}")
            try:
                plan = benchmark_plan_from_json(response.plan_json)
            except BenchmarkPlanJSONError as exc:
                raise EvaluatorStartupError(f"malformed benchmark plan: {exc}") from exc
            self._set_readiness_plan(plan)
            self._resolve_canonical_identity(plan, self._run_id)
            self._supervisor = SerialSupervisor(plan, run_id=self._run_id, ready=False)
            self._start_canonical_run_if_enabled(plan)

        self._readiness.set_service_available("reset", self._reset_client.service_is_ready())
        self._readiness.set_service_available("finalize", self._finalize_client.service_is_ready())
        self._readiness.set_service_available("prepare", self._prepare_client.service_is_ready())
        self._readiness.set_action_available(self._run_action_client.server_is_ready())
        if self._latest_health is not None:
            level, values = self._latest_health
            if level == 0 and values.get("state") == "ready" and values.get("backend_state") == "ready":
                self._readiness.set_health(level=level, values=values)
        if self._readiness.ready and not self._production_ready:
            assert self._supervisor is not None
            self._supervisor.mark_ready()
            self._production_ready = True
            self.get_logger().info(f"{PRODUCTION_READY_TAG} run_id={self._run_id}")
        elif not self._readiness.ready:
            self._log_readiness_snapshot(plan_service_ready=plan_service_ready)

    def _log_readiness_snapshot(self, *, plan_service_ready: bool) -> None:
        """Emit low-rate typed readiness evidence for real diagnostic review."""
        now = self._monotonic()
        if now - self._last_readiness_log_monotonic < 1.0:
            return
        self._last_readiness_log_monotonic = now
        health_level = self._latest_health[0] if self._latest_health is not None else None
        health_values = self._latest_health[1] if self._latest_health is not None else {}
        self.get_logger().info(
            "[IBROBOT_BENCHMARK][EVALUATOR_READINESS] "
            f"plan_service={plan_service_ready} plan_future={self._plan_future is not None} "
            f"plan_done={bool(self._plan_future is not None and self._plan_future.done())} "
            f"plan={bool(self._readiness is not None and self._readiness.plan_ready)} "
            f"contract={bool(self._readiness is not None and self._readiness.contract_ready)} "
            f"health={bool(self._readiness is not None and self._readiness.health_ready)} "
            f"services={bool(self._readiness is not None and self._readiness.services_ready)} "
            f"action={bool(self._readiness is not None and self._readiness.action_ready)} "
            f"reset={bool(self._reset_client and self._reset_client.service_is_ready())} "
            f"finalize={bool(self._finalize_client and self._finalize_client.service_is_ready())} "
            f"prepare={bool(self._prepare_client and self._prepare_client.service_is_ready())} "
            f"run_action={bool(self._run_action_client and self._run_action_client.server_is_ready())} "
            f"health_level={health_level} health_state={health_values.get('state')} "
            f"backend_state={health_values.get('backend_state')}"
        )

    def _drive_transaction(self) -> None:
        if self._pending_future is not None:
            if self._pending_future.done():
                self._complete_pending()
            return
        assert self._readiness is not None and self._supervisor is not None
        command = self._supervisor.next_command(self._supervisor.plan.timeouts.startup_timeout_sec)
        if command is None:
            return
        if isinstance(command, ResetCommand):
            self.get_logger().info(
                f"{COMMAND_TAG} scope=reset suite={command.suite} task={command.task_id} "
                f"episode_index={command.init_state_id} seed={command.seed}"
            )
            self._pending_kind = "reset"
            self._pending_scope = ""
            self._pending_future = self._reset_client.call_async(self._build_reset_request(command))
        elif isinstance(command, PrepareCommand):
            from ibrobot_msgs.srv import PreparePolicyEpisode  # noqa: PLC0415

            self.get_logger().info(
                f"{COMMAND_TAG} scope=prepare task={self._supervisor.current_task_id} "
                f"episode_index={self._supervisor.episode_index} episode={self._supervisor.episode_id}"
            )
            self._pending_kind = "prepare"
            self._pending_scope = ""
            self._pending_future = self._prepare_client.call_async(PreparePolicyEpisode.Request())
        elif isinstance(command, RunPolicyCommand):
            self.get_logger().info(
                f"{COMMAND_TAG} scope=run task={self._supervisor.current_task_id} "
                f"episode_index={self._supervisor.episode_index} episode={command.episode_id} "
                f"preparation={command.preparation_id} max_actions={command.max_actions}"
            )
            self._pending_kind = "run_goal"
            self._pending_scope = ""
            self._pending_future = self._run_action_client.send_goal_async(self._build_run_goal(command))
        elif isinstance(command, FinalizeCommand):
            identity = command.payload.identity
            self.get_logger().info(
                f"{COMMAND_TAG} scope=finalize_{command.scope} task={identity.task_id} "
                f"episode_index={identity.episode_index} episode={identity.episode_id} "
                f"partial={command.partial}"
            )
            self._pending_kind = "finalize"
            self._pending_scope = command.scope
            if command.scope == "episode":
                self._canonical_episode_payload = command.payload
            self._pending_future = self._finalize_client.call_async(self._build_finalize_request(command))

    def _complete_pending(self) -> None:
        assert self._supervisor is not None
        future = self._pending_future
        kind = self._pending_kind
        pending_scope = self._pending_scope
        self._pending_future = None
        self._pending_kind = ""
        self._pending_scope = ""
        try:
            response = future.result()
        except Exception as exc:
            self._handle_pending_exception(kind, exc)
            return
        if kind == "reset":
            if response is None:
                self._supervisor.accept_reset(False, 0, 0, "", 0, "ResetBenchmark transport returned no response")
                return
            if not response.success:
                self._canonical_run_partial = True
                self.get_logger().warn(f"{RESULT_TAG} scope=reset success=false message={response.message}")
                self._supervisor.accept_reset(False, 0, 0, "", 0, str(response.message))
                return
            metadata = json.loads(response.metadata_json)
            self._validate_reset_metadata(metadata)
            self._canonical_reset_metadata = dict(metadata)
            self._canonical_episode_started_monotonic = self._monotonic()
            timestamp_ns = int(response.obs_timestamp.sec) * 1_000_000_000 + int(response.obs_timestamp.nanosec)
            self._supervisor.accept_reset(
                bool(response.success),
                int(response.episode_id),
                int(response.step_id),
                response.task_prompt,
                timestamp_ns,
            )
            self.get_logger().info(
                f"{RESULT_TAG} scope=reset success=true task={self._supervisor.current_task_id} "
                f"episode_index={self._supervisor.episode_index} episode={response.episode_id} "
                f"step={response.step_id} requested_init={metadata.get('requested_init_state_id')} "
                f"actual_init={metadata.get('actual_init_state_id')} seed={metadata.get('seed')} "
                f"env_generation={metadata.get('task_environment_generation')} "
                f"env_reused={metadata.get('task_environment_reused')} "
                f"env_close_count={metadata.get('task_environment_close_count')}"
            )
        elif kind == "prepare":
            if response is None:
                self._supervisor.accept_prepare(False, 0, "PreparePolicyEpisode transport returned no response")
                return
            if not response.success:
                self._canonical_run_partial = True
            self._supervisor.accept_prepare(bool(response.success), int(response.preparation_id), str(response.message))
            self.get_logger().info(
                f"{RESULT_TAG} scope=prepare success={bool(response.success)} preparation={response.preparation_id}"
            )
        elif kind == "run_goal":
            if response is None or not response.accepted:
                self._accept_action_fault("execution_rejected", "RunPolicy goal rejected")
                return
            self._goal_handle = response
            self._pending_kind = "run_result"
            self._pending_future = response.get_result_async()
        elif kind == "run_result":
            if response is None:
                self._accept_action_fault("execution_uncertain", "RunPolicy result missing")
                return
            try:
                status = self._action_status_name(int(response.status))
                _, data = self._extract_run_result(int(response.status), response.result)
                self._validate_run_result_data(data)
            except (TypeError, ValueError, AttributeError, StrictJSONError, SupervisorError) as exc:
                self._accept_action_fault("identity_mismatch", f"invalid RunPolicy result: {exc}")
                return
            outcome = classify_run_policy(status, data)
            self._canonical_outcome = outcome
            self._canonical_run_partial |= bool(outcome.partial or outcome.infrastructure_error)
            self._supervisor.accept_run(outcome)
            self.get_logger().info(
                f"{RESULT_TAG} scope=run status={status} episode={data.episode_id} "
                f"termination={outcome.termination_reason} has_success={data.has_success} "
                f"native_success={data.success} partial={outcome.partial}"
            )
        else:
            if response is None:
                raise SupervisorError("FinalizeBenchmarkScope transport returned no response")
            if not response.success:
                self._abort_canonical(f"finalize_{pending_scope or 'scope'} failed")
                raise SupervisorError(str(getattr(response, "message", "finalization failed")))
            scope = str(response.committed_scope)
            if scope == "episode":
                self._append_canonical_episode(response)
            elif scope == "task":
                self._record_canonical_scope_artifacts(response, "task")
                self._write_canonical_summary(run_status="running")
            elif scope == "run":
                self._record_canonical_scope_artifacts(response, "run")
                self._finalize_canonical_run()
            self._supervisor.accept_finalize(True, scope)
            self.get_logger().info(
                f"{RESULT_TAG} scope=finalize_{scope or 'unknown'} "
                f"success=true already_committed={bool(response.already_committed)}"
            )
            if scope == "run":
                self._announce_completed_once()

    def _handle_pending_exception(self, kind: str, exc: Exception) -> None:
        """Convert asynchronous transport failures into serial partial paths.

        Finalization is idempotent, so a transport exception leaves the
        supervisor in the same finalization phase and the next timer tick
        retries the identical payload.  Failures before a successful reset do
        not invent an episode identity; failures after reset finalize it.
        """
        assert self._supervisor is not None
        message = f"{kind} transport exception: {exc}"
        self._canonical_run_partial = True
        if kind == "reset":
            self._supervisor.accept_reset(False, 0, 0, "", 0, message)
        elif kind == "prepare":
            self._supervisor.accept_prepare(False, 0, message)
        elif kind in {"run_goal", "run_result"}:
            reason = "execution_rejected" if kind == "run_goal" else "execution_uncertain"
            self._accept_action_fault(reason, message)
        elif kind == "finalize":
            self.get_logger().warn(f"{RESULT_TAG} scope=finalize transport_retry=true message={message}")
        else:  # pragma: no cover - internal invariant
            raise SupervisorError(f"unknown pending future kind {kind!r}")

    def _validate_reset_metadata(self, metadata: Any) -> None:
        if not isinstance(metadata, dict):
            raise SupervisorError("reset metadata must be a JSON object")
        assert self._supervisor is not None
        task_id = self._supervisor.current_task_id
        episode_index = self._supervisor.episode_index
        required = {
            "suite": self._supervisor.plan.suite,
            "task_id": task_id,
            "seed": self._supervisor.plan.seed,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise SupervisorError(
                    f"reset metadata {key} mismatch: got {metadata.get(key)!r}, expected {expected!r}"
                )
        policy = self._supervisor.plan.init_state_policy
        if not policy.use_init_state_id:
            for key in ("requested_init_state_id", "actual_init_state_id", "init_state_count"):
                if key in metadata and metadata[key] is not None:
                    raise SupervisorError(f"reset metadata {key} is forbidden when initial-state IDs are disabled")
            return
        requested = metadata.get("requested_init_state_id")
        count = metadata.get("init_state_count")
        actual = metadata.get("actual_init_state_id")
        if requested != episode_index:
            raise SupervisorError("reset metadata requested initial-state identity mismatch")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or actual != episode_index % count:
            raise SupervisorError("reset metadata actual initial-state identity mismatch")

    def _accept_action_fault(self, reason: str, message: str) -> None:
        assert self._supervisor is not None
        data = RunPolicyResultData(
            False,
            message,
            False,
            False,
            reason,
            self._supervisor.episode_id,
            False,
            0,
            0,
            False,
            0.0,
            False,
            False,
            "{}",
            "{}",
            "{}",
            0.0,
        )
        self._supervisor.accept_run(classify_run_policy("aborted", data))

    def _validate_run_result_data(self, data: RunPolicyResultData) -> None:
        assert self._supervisor is not None
        if data.episode_id != self._supervisor.episode_id:
            raise SupervisorError(
                f"RunPolicy episode identity mismatch: got {data.episode_id}, expected {self._supervisor.episode_id}"
            )
        for name, payload in (
            ("standard_metrics_json", data.standard_metrics_json),
            ("native_metrics_json", data.native_metrics_json),
            ("info_json", data.info_json),
        ):
            require_json_object(loads_strict(payload), name)

    @staticmethod
    def _build_reset_request(command: ResetCommand):
        from ibrobot_msgs.srv import ResetBenchmark  # noqa: PLC0415

        request = ResetBenchmark.Request()
        request.suite = command.suite
        request.task_id = command.task_id
        request.seed = command.seed
        request.init_state_id = command.init_state_id
        request.use_init_state_id = command.use_init_state_id
        return request

    @staticmethod
    def _build_run_goal(command: RunPolicyCommand):
        from ibrobot_msgs.action import RunPolicy  # noqa: PLC0415

        goal = RunPolicy.Goal()
        goal.prompt = command.prompt
        goal.preparation_id = command.preparation_id
        goal.episode_id = command.episode_id
        goal.initial_step_id = command.initial_step_id
        goal.initial_observation_timestamp.sec = command.initial_observation_timestamp_ns // 1_000_000_000
        goal.initial_observation_timestamp.nanosec = command.initial_observation_timestamp_ns % 1_000_000_000
        goal.max_actions = command.max_actions
        goal.max_duration_sec = command.max_duration_sec
        goal.startup_timeout_sec = command.startup_timeout_sec
        return goal

    @staticmethod
    def _build_finalize_request(command: FinalizeCommand):
        from ibrobot_msgs.srv import FinalizeBenchmarkScope  # noqa: PLC0415

        wire = finalization_payload_to_wire(command.payload)
        request = FinalizeBenchmarkScope.Request()
        request.scope = wire.scope
        request.identity_json = wire.identity_json
        request.result_json = wire.result_json
        request.termination_reason = wire.termination_reason
        request.error_category = wire.error_category
        request.partial = wire.partial
        request.artifact_refs_json = wire.artifact_refs_json
        return request

    @staticmethod
    def _action_status_name(status: int) -> str:
        if status == GoalStatus.STATUS_SUCCEEDED:
            return "succeeded"
        if status == GoalStatus.STATUS_CANCELED:
            return "canceled"
        return "aborted"

    @classmethod
    def _extract_run_result(cls, status: int, result: Any) -> tuple[str, RunPolicyResultData]:
        status_name = cls._action_status_name(status)
        return status_name, RunPolicyResultData(
            protocol_success=status_name == "succeeded",
            message=str(result.message),
            has_success=bool(result.has_success),
            success=bool(result.success),
            termination_reason=str(result.termination_reason),
            episode_id=int(result.episode_id),
            has_final_step=bool(result.has_final_step),
            final_step_id=int(result.final_step_id),
            published_actions=int(result.published_actions),
            has_reward=bool(result.has_reward),
            reward=float(result.reward),
            terminated=bool(result.terminated),
            truncated=bool(result.truncated),
            standard_metrics_json=str(result.standard_metrics_json),
            native_metrics_json=str(result.native_metrics_json),
            info_json=str(result.info_json),
            round_trip_latency_ms=float(result.round_trip_latency_ms),
        )

    def _fail(self, message: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._abort_canonical(message)
        self._terminal_exit_code = 1
        self._terminal_requested = True
        self.get_logger().error(f"{ERROR_TAG} {message}")

    @property
    def evaluation_completed(self) -> bool:
        return self._supervisor is not None and self._supervisor.completed

    def _announce_completed_once(self) -> None:
        """Emit the terminal marker only after the evaluator result is durable."""
        if self._completion_logged:
            return
        writer = self._canonical_writer
        status = self._canonical_final_status
        if writer is not None:
            if not self._canonical_finalized or status is None:
                return
            run_root = writer.run_root.resolve()
            summary_path = writer.summary_path.resolve()
            artifact_message = f"run_root={run_root} summary={summary_path}"
        else:
            # Canonical output is optional. Native/provider finalization still
            # makes the evaluator's terminal state durable when it is disabled.
            status = "complete"
            artifact_message = "canonical_output=disabled"
        tag = EVALUATION_COMPLETE_TAG if status == "complete" else EVALUATION_FINISHED_TAG
        self._completion_logged = True
        self.get_logger().info(f"{tag} status={status} run_id={self._run_id} {artifact_message}")
        # The launch layer observes this clean process exit and shuts down the
        # Benchmark evaluation graph after the durable terminal marker.
        self._terminal_requested = True
        self._terminal_exit_code = 0

    def destroy_node(self) -> bool:
        """Persist an interrupted canonical run without changing completed output."""
        if not self._completion_logged and not self._canonical_finalized:
            self._abort_canonical("evaluation interrupted before durable final marker")
        return super().destroy_node()


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = BenchmarkEvaluatorNode()
    executor = SingleThreadedExecutor()
    exit_code = 0
    try:
        if bool(node.get_parameter("scaffold_mode").value):
            node.announce_ready()
        else:
            node.start_production()
        executor.add_node(node)
        while rclpy.ok() and not node._terminal_requested:
            try:
                executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                break
        exit_code = node._terminal_exit_code
    except (ScaffoldValidationError, EvaluatorStartupError):
        exit_code = 1
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        # ROS SIGINT may mark the context not-ok before spin returns. Canonical
        # interruption durability still belongs to destroy_node(), so invoke it
        # unconditionally and only guard the subsequent global shutdown call.
        with contextlib.suppress(Exception):
            executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
