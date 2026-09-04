"""Provider-agnostic benchmark environment ROS node.

The node loads the configured adapter through the benchmark plugin registry,
creates the reset, step, plan, and finalization services, publishes observations
from the robot contract, and owns the exactly-once episode/step identity.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from benchmark_runtime.action_codec import ActionDecodeError, decode_step_action_generic
from benchmark_runtime.adapter import BenchmarkAdapter, PreNativeValidationError
from benchmark_runtime.capability_negotiation import (
    BenchmarkCapabilityAgreement,
    CapabilityNegotiationError,
    negotiate_provider_capabilities,
)
from benchmark_runtime.direct_frame_transport import FrameIngressObservationSink
from benchmark_runtime.disk_preflight import DiskPreflightError, ensure_disk_preflight, estimate_disk_usage
from benchmark_runtime.finalization import FinalizationJSONError, finalization_payload_from_wire
from benchmark_runtime.finalization_ledger import FinalizationLedger, FinalizationLedgerError
from benchmark_runtime.identity import EnvironmentRuntime
from benchmark_runtime.io_descriptor import (
    BenchmarkIODescriptor,
    IODescriptorError,
    coerce_observation_batch,
    descriptor_from_robot_contract,
    descriptor_from_robot_model,
    validate_action_payloads,
    validate_io_compatibility,
    validate_observation_batch,
)
from benchmark_runtime.metrics_json import JSONSerializationError, to_metrics_json
from benchmark_runtime.models import EpisodeResult, ResetRequest, RunManifest, RunSummary
from benchmark_runtime.observation_publisher import (
    build_dds_routes,
    build_publishers,
    resolve_publisher_specs,
    stamp_to_builtin_time,
)
from benchmark_runtime.observation_router import (
    DeliveryContext,
    ObservationCommitError,
    ObservationPrepareError,
    ObservationRouter,
    ObservationRouterError,
)
from benchmark_runtime.plan import BenchmarkPlan, benchmark_plan_to_json
from benchmark_runtime.registry import (
    BenchmarkPlugin,
    BenchmarkRegistryError,
    load_plugin,
)
from observation_transport import create_managed_frame_ingress

READY_TAG = "[IBROBOT_BENCHMARK][ENVIRONMENT_NODE_READY]"
PRODUCTION_READY_TAG = "[IBROBOT_BENCHMARK][ENVIRONMENT_NODE_PRODUCTION_READY]"
STARTUP_ERROR_TAG = "[IBROBOT_BENCHMARK][ENVIRONMENT_NODE_STARTUP_ERROR]"
RESET_TAG = "[IBROBOT_BENCHMARK][RESET_RESULT]"
STEP_TAG = "[IBROBOT_BENCHMARK][STEP_RESULT]"
CLOSE_TAG = "[IBROBOT_BENCHMARK][ENVIRONMENT_NODE_CLOSE]"


class ScaffoldValidationError(RuntimeError):
    """Raised when a scaffold node cannot satisfy its READY-only contract."""


class EnvironmentStartupError(RuntimeError):
    """Raised when the production environment node cannot start up."""


def _load_robot_config_dict(path_str: str) -> dict[str, Any]:
    """Load the robot YAML file as a plain dict (the unpacked ``robot`` section).

    Mirrors the canonical loader but stays local: this module must not import
    ``robot_config`` (dependency direction: ``robot_config`` -> ``benchmark_runtime``
    never holds; the environment node loads the YAML directly using PyYAML).
    """
    path = Path(path_str)
    if not path.is_file():
        raise EnvironmentStartupError(f"robot_config_path does not exist: {path_str}")
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise EnvironmentStartupError("PyYAML is required to load the robot config") from exc
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise EnvironmentStartupError(f"robot config file is not a mapping: {path_str}")
    if "robot" in raw and isinstance(raw["robot"], dict):
        robot = raw["robot"]
    else:
        robot = raw
    if not isinstance(robot, dict):
        raise EnvironmentStartupError(f"robot config file is malformed: {path_str}")
    robot.setdefault("_config_path", str(path.resolve()))
    return robot


def _require_benchmark_section(robot_config: dict[str, Any]) -> dict[str, Any]:
    benchmark = robot_config.get("benchmark")
    if not isinstance(benchmark, dict):
        raise EnvironmentStartupError("robot_config is missing the 'benchmark' section required by production mode")
    return benchmark


def _require_endpoints_from_params(params: dict[str, Any], *, require_plan: bool) -> tuple[str, str, str, str]:
    """Read the resolved endpoint names from ROS parameters.

    The launch builder (in ``robot_config.launch_builders.benchmark``) is the
    only caller that should set these parameters; it uses the pure-Python
    ``robot_config.benchmark_endpoints.resolve_benchmark_endpoints`` resolver
    so that ``benchmark_runtime`` itself never imports ``robot_config`` (the
    dependency direction ``robot_config -> benchmark_runtime`` must hold).
    """
    reset_service = params["reset_service"]
    step_service = params["step_service"]
    if not isinstance(reset_service, str) or not reset_service.startswith("/"):
        raise EnvironmentStartupError(f"reset_service must be an absolute ROS path, got {reset_service!r}")
    if not isinstance(step_service, str) or not step_service.startswith("/"):
        raise EnvironmentStartupError(f"step_service must be an absolute ROS path, got {step_service!r}")
    plan_service = params["plan_service"]
    finalize_service = params["finalize_service"]
    if require_plan:
        if not isinstance(plan_service, str) or not plan_service.startswith("/"):
            raise EnvironmentStartupError(f"plan_service must be an absolute ROS path, got {plan_service!r}")
        if not isinstance(finalize_service, str) or not finalize_service.startswith("/"):
            raise EnvironmentStartupError(f"finalize_service must be an absolute ROS path, got {finalize_service!r}")
    return (reset_service, step_service, str(plan_service), str(finalize_service))


def _attach_delivery_receipt(payload_json: str, receipt: Any) -> str:
    """Attach one already-committed router receipt to validated JSON."""
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):  # pragma: no cover - to_metrics_json invariant
        raise JSONSerializationError("delivery receipt base payload must be a JSON object")
    if "observation_delivery" in payload:
        raise JSONSerializationError("reserved metadata key 'observation_delivery' is already present")
    payload["observation_delivery"] = receipt.to_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _attach_step_performance(
    payload_json: str,
    *,
    provider_step_start_monotonic_ns: int,
    provider_step_end_monotonic_ns: int,
    observation_capture_monotonic_ns: int,
    delivery_receipt: Any,
) -> str:
    """Attach provider/router/transport timing without conflating clock domains."""
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):  # pragma: no cover - strict JSON invariant
        raise JSONSerializationError("performance base payload must be a JSON object")
    if "benchmark_performance" in payload:
        raise JSONSerializationError("reserved info key 'benchmark_performance' is already present")
    payload["benchmark_performance"] = {
        "schema_version": 1,
        "clock_domains": {
            "stage_timestamps": "monotonic",
            "observation_delivery_timestamp": "ros",
            "rtp_media_timestamp": "rtp_90khz",
        },
        "environment": {
            "provider_step_start_monotonic_ns": provider_step_start_monotonic_ns,
            "provider_step_end_monotonic_ns": provider_step_end_monotonic_ns,
            "provider_step_latency_ns": provider_step_end_monotonic_ns - provider_step_start_monotonic_ns,
            "observation_capture_monotonic_ns": observation_capture_monotonic_ns,
        },
        "observation_delivery": delivery_receipt.to_dict(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class BenchmarkEnvironmentNode(Node):
    """Production benchmark environment node."""

    def __init__(self, parameter_overrides: list[Parameter] | None = None) -> None:
        super().__init__("benchmark_environment", parameter_overrides=parameter_overrides)
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("benchmark_type", "")
        self.declare_parameter("adapter_name", "")
        self.declare_parameter("scaffold_mode", False)
        # Production-only parameters (filled in by the launch builder).
        self.declare_parameter("reset_service", "")
        self.declare_parameter("step_service", "")
        self.declare_parameter("plan_service", "")
        self.declare_parameter("finalize_service", "")
        self.declare_parameter("frame_ingress_enabled", False)
        self.declare_parameter("frame_ingress_contract_json", "{}")
        self.declare_parameter("frame_ingress_startup_timeout_sec", 120.0)

        # Production runtime state. Allocated eagerly so failure paths can
        # call close() without checking for None. Note: attribute names must
        # NOT collide with rclpy.Node's reserved list attributes
        # (``_publishers``, ``_subscriptions``, ``_clients``, ``_services``,
        # ``_timers``). We prefix with ``_obs_`` / ``_srv_`` to be safe.
        self._runtime = EnvironmentRuntime()
        self._adapter: BenchmarkAdapter | None = None
        self._plugin: BenchmarkPlugin | None = None
        self._reset_srv_handle: Any = None
        self._step_srv_handle: Any = None
        self._plan_srv_handle: Any = None
        self._finalize_srv_handle: Any = None
        self._plan: BenchmarkPlan | None = None
        self._plan_json = ""
        self._ledger: FinalizationLedger | None = None
        self._native_reporter: Any = None
        self._native_artifacts_enabled = False
        self._native_config_cache: dict[str, Any] = {}
        self._native_run_id = ""
        self._native_run_root: Path | None = None
        self._native_failures: list[dict[str, Any]] = []
        self._native_task_identity: tuple[str, int] | None = None
        self._native_episode_context: dict[str, Any] | None = None
        self._native_finalize_cache: dict[tuple[str, str, int, int | None], tuple[list[Any], str]] = {}
        self._obs_publishers: dict[str, Any] = {}
        self._obs_publisher_specs: list = []
        self._observation_router: ObservationRouter | None = None
        self._frame_ingress: Any = None
        self._io_descriptor: BenchmarkIODescriptor | None = None
        self._capability_agreement: BenchmarkCapabilityAgreement | None = None
        self._observation_transaction_id: str | None = None
        self._close_lock = threading.Lock()
        self._closed = False
        self._production_ready = False

    def read_parameters(self) -> dict[str, Any]:
        return {
            "robot_config_path": self.get_parameter("robot_config_path").value,
            "benchmark_type": self.get_parameter("benchmark_type").value,
            "adapter_name": self.get_parameter("adapter_name").value,
            "scaffold_mode": self.get_parameter("scaffold_mode").value,
            "reset_service": self.get_parameter("reset_service").value,
            "step_service": self.get_parameter("step_service").value,
            "plan_service": self.get_parameter("plan_service").value,
            "finalize_service": self.get_parameter("finalize_service").value,
            "frame_ingress_enabled": self.get_parameter("frame_ingress_enabled").value,
            "frame_ingress_contract_json": self.get_parameter("frame_ingress_contract_json").value,
            "frame_ingress_startup_timeout_sec": self.get_parameter("frame_ingress_startup_timeout_sec").value,
        }

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
        self.get_logger().info(
            f"{READY_TAG} type={params['benchmark_type']} adapter={params['adapter_name']} scaffold=true"
        )
        return params

    @staticmethod
    def _evaluation_run_id(benchmark_section: dict[str, Any]) -> str:
        evaluation = benchmark_section.get("evaluation", {})
        explicit = evaluation.get("run_id") if isinstance(evaluation, dict) else None
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit.strip():
                raise EnvironmentStartupError("evaluation.run_id must be a non-empty string")
            explicit = explicit.strip()
            if explicit in {".", ".."} or Path(explicit).name != explicit:
                raise EnvironmentStartupError("evaluation.run_id must be a single path component")
            return explicit
        instance = str(benchmark_section.get("instance_id", "benchmark"))
        suite = str(evaluation.get("suite", "suite"))
        seed = int(evaluation.get("seed", 0))
        tasks = evaluation.get("tasks")
        task_component = "all" if tasks is None else "-".join(str(item) for item in tasks)
        return f"benchmark-{instance}-{suite}-tasks-{task_component}-seed-{seed}"

    @staticmethod
    def _native_status_json(refs: list[Any], failures: list[dict[str, Any]]) -> str:
        payload = {
            "artifacts": [
                {
                    "path": artifact.path,
                    "kind": artifact.kind,
                    "scope": artifact.scope,
                    "status": "available",
                    "metadata": dict(artifact.metadata),
                }
                for artifact in refs
            ],
            "failures": failures,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _record_native_failure(
        self,
        stage: str,
        exc: Exception,
        *,
        task_id: int | None = None,
        episode_index: int | None = None,
    ) -> None:
        context = self._native_episode_context or {}
        if task_id is None:
            task_id = context.get("task_id")
        if episode_index is None:
            episode_index = context.get("episode_index")
        failure: dict[str, Any] = {
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if task_id is not None:
            failure["task_id"] = int(task_id)
        if episode_index is not None:
            failure["episode_index"] = int(episode_index)
        self._native_failures.append(failure)

    def _native_failures_for(
        self, *, scope: str, task_id: int | None, episode_index: int | None
    ) -> list[dict[str, Any]]:
        if scope == "run":
            return [dict(item) for item in self._native_failures]
        result: list[dict[str, Any]] = []
        for failure in self._native_failures:
            failure_task = failure.get("task_id")
            failure_episode = failure.get("episode_index")
            if task_id is not None and failure_task != task_id:
                continue
            if scope == "episode" and failure_episode != episode_index:
                continue
            result.append(dict(failure))
        return result

    def _reporter_status(self, *, scope: str, task_id: int | None, episode_index: int | None) -> dict[str, Any]:
        reporter = self._native_reporter
        if reporter is None:
            return {"artifacts": [], "failures": []}
        try:
            status = reporter.artifact_status(scope=scope, task_id=task_id, episode_index=episode_index)
        except Exception as exc:
            self._record_native_failure("artifact_status", exc, task_id=task_id, episode_index=episode_index)
            return {"artifacts": [], "failures": []}
        return status if isinstance(status, dict) else {"artifacts": [], "failures": []}

    @staticmethod
    def _native_config(benchmark_section: dict[str, Any]) -> dict[str, Any]:
        evaluation = benchmark_section.get("evaluation", {})
        if not isinstance(evaluation, dict):
            return {}
        video = evaluation.get("video", {})
        output = evaluation.get("output", {})
        if not isinstance(video, dict):
            video = {}
        if not isinstance(output, dict):
            output = {}
        return {
            "video": {
                "enabled": bool(video.get("enabled", False)),
                "fps": int(video.get("fps", 30)),
                "camera_name": str(video.get("camera_name", "")),
                "single_video": bool(video.get("single_video", False)),
            },
            "save_sim_states": bool(evaluation.get("save_sim_states", False)),
            "write_native": bool(output.get("write_native", False)),
            "write_canonical": bool(output.get("write_canonical", False)),
            "output_root": str(output.get("root", "outputs/benchmark")),
        }

    def _start_native_reporting(self, benchmark_section: dict[str, Any]) -> None:
        config = self._native_config(benchmark_section)
        self._native_config_cache = config
        self._native_artifacts_enabled = any(
            (config["video"]["enabled"], config["save_sim_states"], config["write_native"])
        )
        if not self._native_artifacts_enabled:
            return
        if self._plugin is None or self._plan is None:
            raise EnvironmentStartupError("native reporting requires a resolved plugin and plan")
        output_root = Path(config["output_root"]).expanduser()
        if not output_root.is_absolute():
            config_path = Path(str(self.get_parameter("robot_config_path").value)).resolve()
            workspace_root = next(
                (parent for parent in config_path.parents if (parent / ".git").exists()),
                config_path.parent,
            )
            output_root = workspace_root / output_root
        output_root = output_root.resolve()
        self._native_run_id = self._evaluation_run_id(benchmark_section)
        self._native_run_root = output_root / self._native_run_id
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            image_specs = [spec for spec in self._obs_publisher_specs if spec.image_width > 0]
            image_size = (
                None
                if not image_specs
                else (max(spec.image_width for spec in image_specs), max(spec.image_height for spec in image_specs))
            )
            if config["video"]["enabled"] and image_size is None:
                raise ValueError("video artifacts require at least one image observation in robot.contract")
            if config["video"]["enabled"] and not config["video"]["camera_name"]:
                raise ValueError("benchmark.evaluation.video.camera_name is required when video is enabled")
            estimate = estimate_disk_usage(
                self._plan,
                config,
                image_size=image_size,
            )
            ensure_disk_preflight(output_root, estimate)
            self._native_run_root.mkdir(exist_ok=False)
        except (DiskPreflightError, ValueError, OSError) as exc:
            raise EnvironmentStartupError(f"native disk preflight failed: {exc}") from exc
        try:
            self._native_reporter = self._plugin.create_native_reporter()
            manifest = RunManifest(
                run_id=self._native_run_id,
                benchmark_type=self._plan.benchmark_type,
                status="running",
                output_ref=str(self._native_run_root),
                metadata={
                    "suite": self._plan.suite,
                    "plan": json.loads(self._plan_json),
                    "native_reporter": config,
                    "disk_preflight": estimate.to_dict(),
                },
            )
            self._native_reporter.on_run_started(manifest)
        except Exception as exc:
            raise EnvironmentStartupError(f"native reporter startup failed: {exc}") from exc

    def _native_task_start(self, task_id: int) -> None:
        if self._native_reporter is None or self._plan is None:
            return
        identity = (self._plan.suite, task_id)
        if self._native_task_identity == identity:
            return
        task = next(item for item in self._plan.tasks if item.task_id == task_id)
        config = self._native_config_cache
        try:
            self._native_reporter.on_task_started(
                suite=self._plan.suite,
                task_id=task.task_id,
                task_name=task.name,
                prompt=task.prompt,
                planned_episodes=self._plan.episodes_per_task,
                video_enabled=config["video"]["enabled"],
                fps=config["video"]["fps"],
                camera_name=config["video"]["camera_name"],
                single_video=config["video"]["single_video"],
                save_sim_states=config["save_sim_states"],
            )
            self._native_task_identity = identity
        except Exception as exc:
            self._record_native_failure("task_start", exc, task_id=task_id)

    def _native_payload(self, method_name: str, fallback: dict[str, Any]) -> dict[str, Any]:
        adapter = self._adapter
        if adapter is None:
            return fallback
        try:
            payload = getattr(adapter, method_name)()
        except Exception as exc:
            self._record_native_failure(method_name, exc)
            return fallback
        return payload if isinstance(payload, dict) else fallback

    def _native_run_summary(self, *, status: str) -> RunSummary:
        return RunSummary(
            run_id=self._native_run_id or "disabled",
            benchmark_type=self._plan.benchmark_type if self._plan else "benchmark",
            status=status,
            output_ref=str(self._native_run_root or ""),
            metadata={"artifact_failures": self._native_failures},
        )

    def start_production(self) -> None:
        """Start the production environment and close cleanly on failure."""
        try:
            self._start_production_internal()
        except Exception as exc:
            self.get_logger().error(f"{STARTUP_ERROR_TAG} {exc}")
            try:
                self.close()
            except Exception as close_exc:  # pragma: no cover - best effort
                self.get_logger().error(f"close during startup failure raised: {close_exc}")
            raise EnvironmentStartupError(str(exc)) from exc

    def _start_production_internal(self) -> None:
        params = self.read_parameters()
        robot_config_path = params["robot_config_path"]
        adapter_name = params["adapter_name"]
        if params["scaffold_mode"] is True:
            raise EnvironmentStartupError("start_production() requires scaffold_mode=false")
        if not isinstance(robot_config_path, str) or not robot_config_path.strip():
            raise EnvironmentStartupError("robot_config_path must be a non-empty string")
        if not isinstance(adapter_name, str) or not adapter_name.strip():
            raise EnvironmentStartupError("adapter_name must be a non-empty string")

        robot_config = _load_robot_config_dict(robot_config_path)
        benchmark_section = _require_benchmark_section(robot_config)

        # 1. Load the exact adapter plugin via the registry. No direct import
        # of LIBERO/robosuite/MuJoCo at this layer.
        try:
            self._plugin = load_plugin(adapter_name)
        except BenchmarkRegistryError as exc:
            raise EnvironmentStartupError(f"failed to load adapter plugin '{adapter_name}': {exc}") from exc

        # 2. Validate the SSOT benchmark mapping into a BenchmarkEnvironmentConfig.
        try:
            env_config = self._plugin.validate_environment_config(benchmark_section)
        except Exception as exc:
            raise EnvironmentStartupError(
                f"plugin '{adapter_name}' rejected the SSOT benchmark mapping: {exc}"
            ) from exc

        # 3. Resolve service endpoints from the ROS launch parameters. The
        # launch builder has already used the pure-Python
        # ``robot_config.benchmark_endpoints`` resolver to populate them, so
        # this layer never imports ``robot_config``.
        evaluation = benchmark_section.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise EnvironmentStartupError("benchmark.evaluation must be a mapping when present")
        evaluation_enabled = bool(evaluation.get("enabled", False))
        reset_service, step_service, plan_service, finalize_service = _require_endpoints_from_params(
            params, require_plan=evaluation_enabled
        )

        # 4. Resolve the observation Contract and create publishers.
        # Canonical observation contract: ``robot.contract`` (top-level) is the only observation SSOT,
        # consumed by both the canonical robot loader /
        # ``inference_service`` ``PipelinePolicyNode`` and this environment
        # node. The duplicate ``benchmark.contract`` was removed.
        contract_section = robot_config.get("contract", {})
        if not isinstance(contract_section, dict):
            raise EnvironmentStartupError("robot.contract must be a mapping when present")
        contract_observations = contract_section.get("observations", [])
        self._obs_publisher_specs = resolve_publisher_specs(contract_observations)
        self._obs_publishers = build_publishers(self, self._obs_publisher_specs)
        sinks_by_key = {sink.key: sink for sink in build_dds_routes(self._obs_publishers, self._obs_publisher_specs)}
        rtp_specs = [spec for spec in self._obs_publisher_specs if spec.transport_mode == "rtp"]
        if rtp_specs:
            if not bool(params["frame_ingress_enabled"]):
                raise EnvironmentStartupError("RTP observation routes require the IB-Robot frame ingress")
            self._frame_ingress = create_managed_frame_ingress(
                self,
                str(params["frame_ingress_contract_json"]),
                startup_timeout_s=float(params["frame_ingress_startup_timeout_sec"]),
            )
            for spec in rtp_specs:
                sinks_by_key[spec.key] = FrameIngressObservationSink(
                    self._frame_ingress,
                    spec.key,
                    optional=spec.optional,
                    logger=self.get_logger(),
                )
        elif bool(params["frame_ingress_enabled"]):
            raise EnvironmentStartupError("frame ingress was enabled but robot.contract has no RTP routes")
        self._observation_router = ObservationRouter(
            tuple(sinks_by_key[spec.key] for spec in self._obs_publisher_specs)
        )

        # 5. Create the adapter and configure it. Heavy native imports
        # (LIBERO, MuJoCo) happen inside the adapter's configure(). On
        # failure, leave the adapter reference in place so ``close()`` can
        # release it exactly once.
        try:
            self._adapter = self._plugin.create_adapter()
            self._adapter.configure(env_config)
            try:
                adapter_descriptor = self._adapter.get_io_descriptor()
            except NotImplementedError:
                adapter_descriptor = None
            if evaluation_enabled and adapter_descriptor is None:
                raise IODescriptorError(
                    "evaluation-enabled adapters must declare a benchmark I/O descriptor before the first reset"
                )
            if evaluation_enabled and adapter_descriptor is not None:
                robot_descriptor = descriptor_from_robot_contract(robot_config)
                model_descriptor = descriptor_from_robot_model(robot_config)
                validate_io_compatibility(
                    adapter=adapter_descriptor,
                    robot=robot_descriptor,
                    model=model_descriptor,
                )
            self._io_descriptor = adapter_descriptor
            if evaluation_enabled:
                self._plan = self._adapter.get_plan()
                try:
                    self._capability_agreement = negotiate_provider_capabilities(
                        self._adapter.capabilities,
                        self._plan,
                    )
                except CapabilityNegotiationError as exc:
                    raise EnvironmentStartupError(str(exc)) from exc
                self._plan_json = benchmark_plan_to_json(self._plan)
                self._start_native_reporting(benchmark_section)
                callback = None if self._native_reporter is not None else self._adapter.on_task_finalized
                self._ledger = FinalizationLedger(self._plan, on_task_committed=callback)
        except Exception as exc:
            raise EnvironmentStartupError(f"adapter configure failed: {exc}") from exc

        # 6. Create the Reset/Step services.
        from ibrobot_msgs.srv import (  # noqa: PLC0415
            FinalizeBenchmarkScope,
            GetBenchmarkPlan,
            ResetBenchmark,
            StepBenchmark,
        )

        self._reset_srv_handle = self.create_service(
            ResetBenchmark,
            reset_service,
            self._handle_reset,
        )
        self._step_srv_handle = self.create_service(
            StepBenchmark,
            step_service,
            self._handle_step,
        )
        if evaluation_enabled:
            self._plan_srv_handle = self.create_service(GetBenchmarkPlan, plan_service, self._handle_get_plan)
            self._finalize_srv_handle = self.create_service(
                FinalizeBenchmarkScope, finalize_service, self._handle_finalize
            )

        self._production_ready = True
        self.get_logger().info(
            f"{PRODUCTION_READY_TAG} adapter={adapter_name} "
            f"reset={reset_service} step={step_service} plan={plan_service} finalize={finalize_service} "
            f"publishers={len(self._obs_publishers)}"
        )

    # ------------------------------------------------------------------ #
    # Plan/finalization services (plan/finalization)
    # ------------------------------------------------------------------ #

    def _handle_get_plan(self, request, response) -> Any:
        del request
        if self._plan is None or not self._plan_json:
            response.success = False
            response.message = "resolved benchmark plan is unavailable"
            response.plan_json = ""
            return response
        response.success = True
        response.message = "ok"
        response.plan_json = self._plan_json
        return response

    @staticmethod
    def _set_finalize_failure(response: Any, message: str) -> Any:
        response.success = False
        response.message = message
        response.committed_scope = ""
        response.artifact_refs_json = "[]"
        response.artifact_status_json = '{"artifacts":[],"failures":[]}'
        response.already_committed = False
        return response

    @staticmethod
    def _native_finalize_key(payload: Any) -> tuple[str, str, int, int | None]:
        identity = payload.identity
        return (payload.scope, identity.run_id, int(identity.task_id or 0), identity.episode_index)

    def _handle_finalize(self, request, response) -> Any:
        ledger = self._ledger
        if ledger is None:
            return self._set_finalize_failure(response, "finalization ledger is unavailable")
        try:
            payload = finalization_payload_from_wire(
                scope=request.scope,
                identity_json=request.identity_json,
                result_json=request.result_json,
                termination_reason=request.termination_reason,
                error_category=request.error_category,
                partial=request.partial,
                artifact_refs_json=request.artifact_refs_json,
            )
            result = ledger.commit(payload)
        except (FinalizationJSONError, FinalizationLedgerError, NotImplementedError, RuntimeError) as exc:
            return self._set_finalize_failure(response, str(exc))

        refs: list[Any] = []
        native_key = self._native_finalize_key(payload)
        cached = self._native_finalize_cache.get(native_key)
        status_json = '{"artifacts":[],"failures":[]}'
        if cached is not None:
            refs, status_json = cached
        if self._native_reporter is not None and cached is None:
            try:
                if payload.scope == "episode":
                    context = self._native_episode_context or {}
                    task_id = payload.identity.task_id or 0
                    task = (
                        next((item for item in self._plan.tasks if item.task_id == task_id), None)
                        if self._plan
                        else None
                    )
                    native_result = EpisodeResult(
                        benchmark_type=self._plan.benchmark_type if self._plan else "benchmark",
                        suite=payload.identity.suite or "",
                        task_id=task_id,
                        task_name=task.name if task else f"task_{task_id:03d}",
                        episode_index=payload.identity.episode_index or 0,
                        seed=int(context.get("seed", self._plan.seed if self._plan else 0)),
                        init_state_id=context.get("init_state_id"),
                        success=(bool(payload.result.get("success")) if payload.result.get("has_success") else None),
                        reward_sum=context.get("reward_sum"),
                        reward_max=context.get("reward_max"),
                        steps=int(context.get("steps", 0)),
                        duration_s=max(0.0, time.monotonic() - float(context.get("started", time.monotonic()))),
                        termination_reason=payload.termination_reason,
                        error_category=payload.error_category,
                    )
                    refs = self._native_reporter.finalize_episode(native_result)
                    self._native_episode_context = None
                elif payload.scope == "task":
                    refs = self._native_reporter.finalize_task()
                elif payload.scope == "run":
                    refs = self._native_reporter.finalize(
                        self._native_run_summary(status="partial" if payload.partial else "complete")
                    )
            except Exception as exc:
                self._record_native_failure(
                    f"{payload.scope}_finalize",
                    exc,
                    task_id=payload.identity.task_id,
                    episode_index=payload.identity.episode_index,
                )

            # The reporter is finalized before task-scoped native resources are
            # released. This keeps task stats/video/state ownership in the
            # adapter process and preserves the serial evaluation lifecycle boundary.
            if payload.scope == "task":
                try:
                    self._adapter.on_task_finalized(payload.identity.suite or "", payload.identity.task_id or 0)
                except Exception as exc:
                    return self._set_finalize_failure(response, f"task environment close failed: {exc}")
                self._native_task_identity = None
            reporter_status = self._reporter_status(
                scope=payload.scope,
                task_id=payload.identity.task_id,
                episode_index=payload.identity.episode_index,
            )
            combined_failures = [
                *self._native_failures_for(
                    scope=payload.scope,
                    task_id=payload.identity.task_id,
                    episode_index=payload.identity.episode_index,
                ),
                *reporter_status.get("failures", []),
            ]
            status_json = json.dumps(
                {"artifacts": reporter_status.get("artifacts", []), "failures": combined_failures},
                sort_keys=True,
                separators=(",", ":"),
            )
            self._native_finalize_cache[native_key] = (list(refs), status_json)

        response.success = True
        response.message = "ok"
        response.committed_scope = result.committed_scope
        response.artifact_refs_json = json.dumps([item.path for item in refs], separators=(",", ":"))
        response.artifact_status_json = status_json
        response.already_committed = result.already_committed
        return response

    # ------------------------------------------------------------------ #
    # Reset service callback
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_reset_failure(response: Any, message: str) -> Any:
        response.success = False
        response.message = message
        response.obs_timestamp.sec = 0
        response.obs_timestamp.nanosec = 0
        response.episode_id = 0
        response.step_id = 0
        response.task_prompt = ""
        response.metadata_json = "{}"
        return response

    def _handle_reset(self, request, response) -> Any:
        from ibrobot_msgs.srv import ResetBenchmark  # noqa: PLC0415

        assert isinstance(response, ResetBenchmark.Response)
        plan = self._plan
        if (
            plan is not None
            and any((plan.artifacts.save_sim_states, plan.artifacts.video_enabled, plan.artifacts.write_native))
            and self._native_reporter is None
        ):
            return self._set_reset_failure(response, "artifact-enabled plan has no initialized native reporter")
        ledger = self._ledger
        if ledger is not None:
            try:
                ledger.authorize_reset(str(request.suite), int(request.task_id), int(request.init_state_id))
            except (TypeError, ValueError, FinalizationLedgerError) as exc:
                return self._set_reset_failure(response, f"reset ordering rejected: {exc}")
        with self._runtime.lock:
            # begin_reset invalidates the previous episode immediately, even
            # if this reset later fails. The previous identity never survives.
            self._runtime.begin_reset()

        # Single-lane serialization is provided by the lock in begin_reset.
        # Adapter.reset is called outside the runtime lock so a slow reset
        # does not block identity snapshots, but the runtime's reset-in-progress
        # flag prevents a concurrent step from authorizing.
        adapter = self._adapter
        if adapter is None:  # pragma: no cover - defensive
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            response.success = False
            response.message = "adapter is not configured"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.episode_id = 0
            response.step_id = 0
            response.task_prompt = ""
            response.metadata_json = "{}"
            return response

        try:
            reset_request = ResetRequest(
                suite=str(request.suite),
                task_id=int(request.task_id),
                seed=int(request.seed),
                use_init_state_id=bool(request.use_init_state_id),
                init_state_id=int(request.init_state_id),
            )
        except (TypeError, ValueError) as exc:
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            response.success = False
            response.message = f"invalid reset request: {exc}"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.episode_id = 0
            response.step_id = 0
            response.task_prompt = ""
            response.metadata_json = "{}"
            return response

        self._native_task_start(reset_request.task_id)
        try:
            reset_result = adapter.reset(reset_request)
        except Exception as exc:
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            self.get_logger().error(f"{RESET_TAG} adapter.reset raised: {exc}")
            response.success = False
            response.message = f"adapter.reset failed: {exc}"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.episode_id = 0
            response.step_id = 0
            response.task_prompt = ""
            response.metadata_json = "{}"
            return response

        try:
            observation_batch = coerce_observation_batch(
                reset_result.observations,
                episode_transaction_id=None,
                sequence_id=0,
            )
            if self._io_descriptor is not None:
                validate_observation_batch(observation_batch, self._io_descriptor)
            reset_result = replace(reset_result, observations=observation_batch)
        except (IODescriptorError, TypeError, ValueError) as exc:
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            self._observation_transaction_id = None
            self.get_logger().error(f"{RESET_TAG} canonical observation rejected: {exc}")
            return self._set_reset_failure(response, f"canonical observation rejected: {exc}")

        # Observation transaction: build/validate ALL observation messages and serialize metadata
        # BEFORE publishing or committing identity. A validation failure on
        # any observation must NOT produce a partial publication and must NOT
        # commit the new identity.
        stamp = self.get_clock().now()
        sec, nanosec = stamp_to_builtin_time(stamp)

        router = self._observation_router
        if router is None:  # pragma: no cover - defensive
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            return self._set_reset_failure(response, "observation router is not configured")
        try:
            prepared_batch = router.prepare(
                observation_batch,
                DeliveryContext(timestamp_sec=sec, timestamp_nanosec=nanosec),
            )
        except (ObservationPrepareError, ObservationRouterError, TypeError, ValueError) as exc:
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            self.get_logger().error(f"{RESET_TAG} observation route prepare failed: {exc}")
            return self._set_reset_failure(response, f"observation validation failed during route prepare: {exc}")

        try:
            metadata_json = to_metrics_json(reset_result.metadata)
        except JSONSerializationError as exc:
            prepared_batch.cancel()
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            self.get_logger().error(f"{RESET_TAG} metadata serialization failed: {exc}")
            return self._set_reset_failure(response, f"metadata serialization failed: {exc}")

        try:
            delivery_receipt = prepared_batch.commit()
        except ObservationCommitError as exc:
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            receipt = exc.receipt
            self.get_logger().error(
                f"{RESET_TAG} observation transport failure route={receipt.failure_route} "
                f"committed={list(receipt.committed_keys)} error={receipt.error}"
            )
            return self._set_reset_failure(
                response,
                f"observation publish transport failure (episode invalidated): {receipt.error}",
            )
        try:
            metadata_json = _attach_delivery_receipt(metadata_json, delivery_receipt)
        except (JSONSerializationError, TypeError, ValueError) as exc:
            self._runtime.commit_reset_failure()
            if ledger is not None:
                ledger.abort_reset()
            self.get_logger().error(f"{RESET_TAG} delivery receipt serialization failed: {exc}")
            return self._set_reset_failure(
                response, f"delivery receipt serialization failed (episode invalidated): {exc}"
            )

        # All validation, serialization and publication succeeded. Commit the
        # new identity LAST so that a public episode ID is only visible after
        # the full transaction completed successfully.
        episode_id, step_id = self._runtime.commit_reset_success()
        self._observation_transaction_id = observation_batch.episode_transaction_id
        if ledger is not None:
            ledger.record_reset(episode_id)
        if self._native_reporter is not None:
            payload = self._native_payload(
                "get_native_reset_payload",
                {"observations": reset_result.observations},
            )
            self._native_episode_context = {
                "suite": reset_request.suite,
                "task_id": reset_request.task_id,
                "episode_id": episode_id,
                "episode_index": reset_request.init_state_id,
                "init_state_id": reset_result.metadata.get("actual_init_state_id", reset_request.init_state_id),
                "started": time.monotonic(),
                "seed": reset_request.seed,
                "steps": 0,
                "reward_sum": 0.0,
                "reward_max": None,
            }
            try:
                reset_kwargs = {
                    "episode_index": reset_request.init_state_id,
                    "init_state_id": self._native_episode_context["init_state_id"],
                    "native_env": payload.get("native_env"),
                }
                if "sim_state" in payload:
                    reset_kwargs["sim_state"] = payload["sim_state"]
                self._native_reporter.on_episode_reset(
                    payload.get("observations", reset_result.observations),
                    **reset_kwargs,
                )
            except Exception as exc:
                self._record_native_failure("episode_reset", exc)

        response.success = True
        response.message = "ok"
        response.obs_timestamp.sec = sec
        response.obs_timestamp.nanosec = nanosec
        response.episode_id = int(episode_id)
        response.step_id = int(step_id)
        response.task_prompt = str(reset_result.task_prompt)
        response.metadata_json = metadata_json
        self.get_logger().info(
            f"{RESET_TAG} success episode={episode_id} step={step_id} prompt_len={len(response.task_prompt)} "
            f"transaction={delivery_receipt.episode_transaction_id} sequence={delivery_receipt.sequence_id} "
            f"routes={list(delivery_receipt.committed_keys)}"
        )
        return response

    # ------------------------------------------------------------------ #
    # Step service callback
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_step_failure_from_result(response: Any, request: Any, result: Any, message: str) -> Any:
        response.success = False
        response.message = message
        response.obs_timestamp.sec = 0
        response.obs_timestamp.nanosec = 0
        response.step_id = int(request.expected_step_id)
        response.has_reward = bool(result.reward is not None)
        response.reward = float(result.reward) if result.reward is not None else 0.0
        response.terminated = bool(result.terminated)
        response.truncated = bool(result.truncated)
        response.has_is_success = bool(result.is_success is not None)
        response.is_success = bool(result.is_success) if result.is_success is not None else False
        response.standard_metrics_json = "{}"
        response.native_metrics_json = "{}"
        response.info_json = "{}"
        return response

    def _handle_step(self, request, response) -> Any:
        from ibrobot_msgs.srv import StepBenchmark  # noqa: PLC0415

        assert isinstance(response, StepBenchmark.Response)

        # 1. Authorize the step against the runtime identity. On rejection,
        # no adapter call, no observation publication, no identity advance.
        ok, message = self._runtime.begin_step(
            int(request.episode_id),
            int(request.expected_step_id),
        )
        if not ok:
            response.success = False
            response.message = message
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.step_id = int(request.expected_step_id)
            response.has_reward = False
            response.reward = 0.0
            response.terminated = False
            response.truncated = False
            response.has_is_success = False
            response.is_success = False
            response.standard_metrics_json = "{}"
            response.native_metrics_json = "{}"
            response.info_json = "{}"
            self.get_logger().warn(
                f"{STEP_TAG} rejected episode={request.episode_id} expected_step={request.expected_step_id}: {message}"
            )
            return response

        # 2. Generic decode of the VariantsList. Pre-native validation: the generic runtime
        # only decodes the wire format into a dict of numpy arrays; it does
        # NOT impose LIBERO-specific dimension/range/gripper semantics. The
        # concrete adapter validates those in its step() and raises
        # ``PreNativeValidationError`` for pre-native validation failures.
        try:
            action_dict = decode_step_action_generic(request.action)
        except ActionDecodeError as exc:
            self._runtime.abort_step()
            response.success = False
            response.message = f"action decode rejected: {exc}"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.step_id = int(request.expected_step_id)
            response.has_reward = False
            response.reward = 0.0
            response.terminated = False
            response.truncated = False
            response.has_is_success = False
            response.is_success = False
            response.standard_metrics_json = "{}"
            response.native_metrics_json = "{}"
            response.info_json = "{}"
            self.get_logger().warn(f"{STEP_TAG} action decode rejected: {exc}")
            return response

        try:
            if self._io_descriptor is not None:
                validate_action_payloads(action_dict, self._io_descriptor)
        except (IODescriptorError, TypeError, ValueError) as exc:
            self._runtime.abort_step()
            response.success = False
            response.message = f"action descriptor rejected: {exc}"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.step_id = int(request.expected_step_id)
            response.has_reward = False
            response.reward = 0.0
            response.terminated = False
            response.truncated = False
            response.has_is_success = False
            response.is_success = False
            response.standard_metrics_json = "{}"
            response.native_metrics_json = "{}"
            response.info_json = "{}"
            self.get_logger().warn(f"{STEP_TAG} action descriptor rejected: {exc}")
            return response

        # 3. Call the adapter. Pre-native validation: the adapter validates the action
        # dimension/range/gripper BEFORE entering the native env.step(). If
        # pre-native validation fails, the adapter raises
        # ``PreNativeValidationError`` and the runtime releases the identity
        # slot (abort_step) without poisoning, allowing retry. Any OTHER
        # exception means the native step was entered: poison, no retry.
        adapter = self._adapter
        if adapter is None:  # pragma: no cover - defensive
            self._runtime.poison_after_native_failure()
            response.success = False
            response.message = "adapter is not configured"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.step_id = int(request.expected_step_id)
            response.has_reward = False
            response.reward = 0.0
            response.terminated = False
            response.truncated = False
            response.has_is_success = False
            response.is_success = False
            response.standard_metrics_json = "{}"
            response.native_metrics_json = "{}"
            response.info_json = "{}"
            return response

        provider_step_start_monotonic_ns = time.monotonic_ns()
        try:
            step_result = adapter.step(action_dict)
        except PreNativeValidationError as exc:
            # Pre-native validation failure: release the identity slot
            # without poisoning. The same step can be retried.
            self._runtime.abort_step()
            self.get_logger().warn(f"{STEP_TAG} pre-native action validation rejected: {exc}")
            response.success = False
            response.message = f"action validation rejected: {exc}"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.step_id = int(request.expected_step_id)
            response.has_reward = False
            response.reward = 0.0
            response.terminated = False
            response.truncated = False
            response.has_is_success = False
            response.is_success = False
            response.standard_metrics_json = "{}"
            response.native_metrics_json = "{}"
            response.info_json = "{}"
            return response
        except Exception as exc:
            # Native step entry failure: poison, no retry, no advance.
            self._runtime.poison_after_native_failure()
            self.get_logger().error(f"{STEP_TAG} adapter.step raised: {exc}")
            response.success = False
            response.message = f"adapter.step failed (episode poisoned): {exc}"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.step_id = int(request.expected_step_id)
            response.has_reward = False
            response.reward = 0.0
            response.terminated = False
            response.truncated = False
            response.has_is_success = False
            response.is_success = False
            response.standard_metrics_json = "{}"
            response.native_metrics_json = "{}"
            response.info_json = "{}"
            return response
        provider_step_end_monotonic_ns = time.monotonic_ns()

        try:
            if self._observation_transaction_id is None:
                raise IODescriptorError("step observation has no active Episode transaction; reset required")
            observation_batch = coerce_observation_batch(
                step_result.observations,
                episode_transaction_id=self._observation_transaction_id,
                sequence_id=int(request.expected_step_id) + 1,
            )
            if self._io_descriptor is not None:
                validate_observation_batch(observation_batch, self._io_descriptor)
            step_result = replace(step_result, observations=observation_batch)
        except (IODescriptorError, TypeError, ValueError) as exc:
            self._runtime.poison_after_native_failure()
            self.get_logger().error(f"{STEP_TAG} canonical observation rejected: {exc}")
            response.success = False
            response.message = f"canonical observation rejected (episode poisoned): {exc}"
            response.obs_timestamp.sec = 0
            response.obs_timestamp.nanosec = 0
            response.step_id = int(request.expected_step_id)
            response.has_reward = bool(step_result.reward is not None)
            response.reward = float(step_result.reward) if step_result.reward is not None else 0.0
            response.terminated = bool(step_result.terminated)
            response.truncated = bool(step_result.truncated)
            response.has_is_success = bool(step_result.is_success is not None)
            response.is_success = bool(step_result.is_success) if step_result.is_success is not None else False
            response.standard_metrics_json = "{}"
            response.native_metrics_json = "{}"
            response.info_json = "{}"
            return response

        # Observation transaction: native step has been entered. Build/validate ALL observation
        # messages and serialize ALL JSON BEFORE publishing or committing
        # identity. Any failure from here poisons the episode (we cannot
        # undo a native step) and does NOT advance the public expected_step_id.
        stamp = self.get_clock().now()
        sec, nanosec = stamp_to_builtin_time(stamp)

        router = self._observation_router
        if router is None:  # pragma: no cover - defensive
            self._runtime.poison_after_native_failure()
            return self._set_step_failure_from_result(
                response, request, step_result, "observation router is not configured"
            )
        try:
            prepared_batch = router.prepare(
                observation_batch,
                DeliveryContext(timestamp_sec=sec, timestamp_nanosec=nanosec),
            )
        except (ObservationPrepareError, ObservationRouterError, TypeError, ValueError) as exc:
            self._runtime.poison_after_native_failure()
            self.get_logger().error(f"{STEP_TAG} observation route prepare failed: {exc}")
            return self._set_step_failure_from_result(
                response,
                request,
                step_result,
                f"observation validation failed during route prepare (episode poisoned): {exc}",
            )

        try:
            standard_metrics_json = to_metrics_json(step_result.standard_metrics)
            native_metrics_json = to_metrics_json(step_result.native_metrics)
            info_json = to_metrics_json(step_result.info)
        except JSONSerializationError as exc:
            prepared_batch.cancel()
            self._runtime.poison_after_native_failure()
            self.get_logger().error(f"{STEP_TAG} metrics/info serialization failed: {exc}")
            return self._set_step_failure_from_result(
                response, request, step_result, f"metrics serialization failed (episode poisoned): {exc}"
            )

        try:
            delivery_receipt = prepared_batch.commit()
        except ObservationCommitError as exc:
            self._runtime.poison_after_native_failure()
            receipt = exc.receipt
            self.get_logger().error(
                f"{STEP_TAG} observation transport failure route={receipt.failure_route} "
                f"committed={list(receipt.committed_keys)} error={receipt.error}"
            )
            return self._set_step_failure_from_result(
                response,
                request,
                step_result,
                f"observation publish transport failure (episode poisoned): {receipt.error}",
            )
        try:
            info_json = _attach_delivery_receipt(info_json, delivery_receipt)
            info_json = _attach_step_performance(
                info_json,
                provider_step_start_monotonic_ns=provider_step_start_monotonic_ns,
                provider_step_end_monotonic_ns=provider_step_end_monotonic_ns,
                observation_capture_monotonic_ns=observation_batch.capture_timestamp_ns,
                delivery_receipt=delivery_receipt,
            )
        except (JSONSerializationError, TypeError, ValueError) as exc:
            self._runtime.poison_after_native_failure()
            self.get_logger().error(f"{STEP_TAG} delivery receipt serialization failed: {exc}")
            return self._set_step_failure_from_result(
                response,
                request,
                step_result,
                f"delivery receipt serialization failed (episode poisoned): {exc}",
            )

        # All validation, serialization and publication succeeded. Commit the
        # new step ID LAST so that a public step ID is only visible after the
        # full transaction completed successfully.
        committed_step = self._runtime.commit_step_success()
        if self._native_reporter is not None and self._native_episode_context is not None:
            context = self._native_episode_context
            context["steps"] = committed_step + 1
            if step_result.reward is not None:
                context["reward_sum"] += float(step_result.reward)
                context["reward_max"] = (
                    float(step_result.reward)
                    if context["reward_max"] is None
                    else max(context["reward_max"], float(step_result.reward))
                )
            payload = self._native_payload(
                "get_native_step_payload",
                {"observations": step_result.observations},
            )
            try:
                step_kwargs = {
                    "episode_index": context["episode_index"],
                    "result": step_result,
                    "done": step_result.terminated or step_result.truncated,
                    "native_env": payload.get("native_env"),
                }
                if "sim_state" in payload:
                    step_kwargs["sim_state"] = payload["sim_state"]
                self._native_reporter.on_native_step(
                    payload.get("observations", step_result.observations),
                    **step_kwargs,
                )
            except Exception as exc:
                self._record_native_failure("native_step", exc)

        response.success = True
        response.message = "ok"
        response.obs_timestamp.sec = sec
        response.obs_timestamp.nanosec = nanosec
        response.step_id = int(committed_step)
        response.has_reward = bool(step_result.reward is not None)
        response.reward = float(step_result.reward) if step_result.reward is not None else 0.0
        response.terminated = bool(step_result.terminated)
        response.truncated = bool(step_result.truncated)
        response.has_is_success = bool(step_result.is_success is not None)
        response.is_success = bool(step_result.is_success) if step_result.is_success is not None else False
        response.standard_metrics_json = standard_metrics_json
        response.native_metrics_json = native_metrics_json
        response.info_json = info_json
        self.get_logger().info(
            f"{STEP_TAG} success episode={request.episode_id} step={committed_step} "
            f"terminated={response.terminated} truncated={response.truncated} "
            f"has_success={response.has_is_success} transaction={delivery_receipt.episode_transaction_id} "
            f"sequence={delivery_receipt.sequence_id} routes={list(delivery_receipt.committed_keys)}"
        )
        return response

    # ------------------------------------------------------------------ #
    # Close / shutdown
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Release all production resources. Idempotent.

        Closes the adapter exactly once. Destroys services and publishers.
        Repeated calls are no-ops. Safe to call from any thread.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._production_ready = False
            adapter = self._adapter
            self._adapter = None
            plugin = self._plugin
            self._plugin = None

            # Destroy services/publishers first so no new callbacks fire while
            # the adapter is being closed.
            for handle in (
                self._reset_srv_handle,
                self._step_srv_handle,
                self._plan_srv_handle,
                self._finalize_srv_handle,
            ):
                if handle is not None:
                    with contextlib.suppress(Exception):
                        self.destroy_service(handle)
            self._reset_srv_handle = None
            self._step_srv_handle = None
            self._plan_srv_handle = None
            self._finalize_srv_handle = None
            if self._observation_router is not None:
                self._observation_router.close()
                self._observation_router = None
            if self._frame_ingress is not None:
                try:
                    self._frame_ingress.close()
                except Exception as exc:
                    self.get_logger().error(f"{CLOSE_TAG} frame ingress close raised: {exc}")
                self._frame_ingress = None
            for pub in list(self._obs_publishers.values()):
                with contextlib.suppress(Exception):
                    self.destroy_publisher(pub)
            self._obs_publishers = {}
            self._obs_publisher_specs = []

            # Flush native artifacts before releasing a still-active native env.
            if self._native_reporter is not None:
                with contextlib.suppress(Exception):
                    self._native_reporter.finalize(self._native_run_summary(status="partial"))
                self._native_reporter = None

            # Close the adapter exactly once.
            if adapter is not None:
                try:
                    adapter.close()
                    self.get_logger().info(f"{CLOSE_TAG} adapter closed plugin={plugin.name if plugin else 'unknown'}")
                except Exception as exc:  # pragma: no cover - best effort
                    self.get_logger().error(f"{CLOSE_TAG} adapter.close raised: {exc}")

    @property
    def production_ready(self) -> bool:
        return self._production_ready


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = BenchmarkEnvironmentNode()
    try:
        if bool(node.get_parameter("scaffold_mode").value):
            node.announce_ready()
        else:
            try:
                node.start_production()
            except EnvironmentStartupError:
                # start_production already closed partial resources and logged the
                # startup error; tear down ROS resources and exit non-zero.
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                sys.exit(1)
    except (ScaffoldValidationError, EnvironmentStartupError):
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)
    except Exception:
        # Any unexpected exception during startup: attempt a clean close and
        # re-raise so the process exits non-zero.
        with contextlib.suppress(Exception):
            node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        raise

    # Benchmark providers such as LIBERO own native simulator state and are
    # intentionally single-lane.  Keep ROS service/control callbacks on that
    # same lane; RTP encode/send remains asynchronous in FrameIngress workers.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
