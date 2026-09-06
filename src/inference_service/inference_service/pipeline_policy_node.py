"""ROS 2 node exposing one validated unified inference pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import rclpy.action
import torch
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from ibrobot_msgs.action import (
    CloseInferenceSession,
    DispatchInfer,
    OpenInferenceSession,
    ScheduledDispatchInfer,
)
from ibrobot_msgs.msg import (
    DistributedInferenceRequest,
    DistributedInferenceResult,
    InferenceOutcome,
    InferencePipelineStatus,
    InferenceServingStatus,
    InferenceWorkCapacity,
    VariantsList,
    VideoStreamDescriptor,
    VideoStreamStatus,
)
from inference_manifest import ValidatedManifest, load_inference_manifest, load_inference_manifest_metadata
from inference_service.backends import InferenceRequest
from inference_service.device_video_streams import DeviceVideoStreamManager
from inference_service.distributed import (
    DistributedProtocolError,
    DistributedResult,
    EdgeProcessorRuntime,
    EdgeSession,
    Operation,
    StreamReference,
    StructuredError,
    build_pipeline_identity,
    structured_error_from_exception,
)
from inference_service.distributed.ros_protocol import (
    error_to_message,
    request_to_message,
    result_from_message,
    status_from_message,
    status_to_message,
    video_descriptor_to_message,
    video_status_from_message,
    video_status_to_message,
)
from inference_service.pipeline import InferencePipelineManager, create_pipeline_manager
from inference_service.runtime_composition import (
    build_policy_runtime_dependencies,
    require_runtime_dependencies,
)
from inference_service.scheduler.action_idempotency import (
    ResolutionErrorCodes,
    execute_resolved_action,
)
from inference_service.scheduler.goal_slots import GOAL_CONTEXTS_PER_ENDPOINT, GoalSlotPool
from inference_service.scheduler.idempotency import (
    IdempotencyError,
    canonical_fingerprint,
    validate_uuid4,
)
from inference_service.scheduler.ledger import (
    IdempotencyLedger,
    LedgerAction,
    LedgerError,
    close_key,
    dispatch_key,
    open_key,
)
from inference_service.scheduler.time_domains import monotonic_expiry_to_ros_ns
from inference_service.scheduler.wire_bounds import set_scheduled_error, utf8_size
from inference_service.unified_runtime import RegistrySet, RuntimeProviders
from observation_transport.frame_ingress import StreamSessionView
from robot_config.contract_utils import (
    SpecView,
    StreamBuffer,
    contract_fingerprint,
    decode_value,
    iter_specs,
    qos_profile_from_dict,
    stamp_from_header_ns,
)
from robot_config.inference_runtime_options import effective_latency_runtime_options
from robot_config.utils import (
    build_joint_conversion_table,
    build_joint_conversion_table_from_urdf,
    parse_bool,
    resolve_calibration_source_specs_from_config,
    resolve_gripper_joints_from_config,
    resolve_joint_names_from_config,
    resolve_lerobot_norm_mode,
)
from tensormsg.converter import TensorMsgConverter

_CLOUD_HANDSHAKE_WARNING_DELAY_S = 5.0
_CLOUD_HANDSHAKE_WARNING_THROTTLE_S = 10.0


@dataclass(frozen=True)
class PipelineNodeConfig:
    pipeline_id: str
    model_path: str
    deployment: str
    execution_mode: str
    request_timeout: float
    default_task: str
    runtime_options_json: str
    robot_config_path: str
    use_sim: bool
    action_server: str
    reset_service: str
    health_topic: str
    action_topic: str
    request_topic: str
    result_topic: str
    heartbeat_topic: str
    video_descriptor_topic: str = ""
    video_status_topic: str = ""
    external_video_producer: bool = False
    # Empty when scheduler is disabled.
    scheduled_open_session: str = ""
    scheduled_dispatch: str = ""
    scheduled_close_session: str = ""
    scheduled_serving_status: str = ""
    runtime_policy_json: str = ""
    runtime_policy_fingerprint: str = ""
    hardware_resource_id: str = ""
    session_idle_timeout_ns: int = 0
    max_prompt_bytes: int = 4096
    max_error_message_bytes: int = 1024
    max_error_details_bytes: int = 8192
    public_capacity_json: str = ""
    max_session_records: int = 1
    terminal_result_cache_entries: int = 1
    max_duplicate_waiters_per_request: int = 1
    terminal_session_retention_ns: int = 1

    @property
    def scheduler_enabled(self) -> bool:
        """Derive runtime mode from the SSOT-generated policy, never a second switch."""

        return bool(self.runtime_policy_json)


@dataclass
class _SubState:
    spec: SpecView
    max_age_ns: int
    step_ns: int
    history_window_ns: int
    buffer: StreamBuffer = field(init=False)

    def __post_init__(self) -> None:
        policy = str(getattr(self.spec, "resample_policy", "hold")).lower()
        tolerance_ns = max(0, int(getattr(self.spec, "asof_tol_ms", 0))) * 1_000_000
        self.buffer = StreamBuffer(
            policy,
            self.step_ns,
            tolerance_ns,
            max_age_ns=self.max_age_ns,
            retention_ns=self.history_window_ns,
        )

    @property
    def history(self) -> list[tuple[int, int, object]]:
        return self.buffer.history


@dataclass
class _PendingOperation:
    event: threading.Event
    operation: Operation
    session_id: str = ""
    session_generation: int = 0
    deployment_fingerprint: str = ""
    result: DistributedResult | None = None
    error: StructuredError | None = None


@dataclass
class _RoundTripProgress:
    published: bool = False
    response_received: bool = False
    response_success: bool = False
    backend_ready: bool = False


class ObservationNotReadyError(RuntimeError):
    code = "observation_not_ready"
    recoverable = True
    stage = "observation"

    def __init__(self, observations: list[dict[str, object]]) -> None:
        summary = ", ".join(f"{item['key']} ({item['topic']}): {item['reason']}" for item in observations)
        super().__init__(f"required policy observations are not ready: {summary}")
        self.details = {"observations": observations}


class PipelineBusyError(RuntimeError):
    code = "pipeline_busy"
    recoverable = True
    stage = "admission"


class RequestCanceledError(RuntimeError):
    code = "request_canceled"
    recoverable = True
    stage = "cancel"

    def __init__(self, message: str, *, operation_started: bool = False) -> None:
        super().__init__(message)
        self.operation_started = operation_started
        self.outcome_known = True


class DeadlineExceededError(RuntimeError):
    code = "deadline_exceeded"
    recoverable = True
    stage = "deadline"


class _ExternalVideoProducerView:
    """Request-side session view for an out-of-process frame producer.

    This object never owns a sender, descriptor publisher, or stream-status
    publisher. It only keeps request stream references aligned with the active
    distributed inference session.
    """

    def __init__(
        self,
        observation_specs: list[SpecView],
        *,
        pipeline_id: str,
        contract_fingerprint: str,
        deployment_fingerprint: str,
    ) -> None:
        from robot_config.observation_transport import effective_observation_transport

        self._references = tuple(
            StreamReference(spec.key, transport.stream_id or "")
            for spec in observation_specs
            if (transport := effective_observation_transport(spec.transport)).mode == "rtp"
        )
        self._session = StreamSessionView.inactive(
            pipeline_id=pipeline_id,
            contract_fingerprint=contract_fingerprint,
            deployment_fingerprint=deployment_fingerprint,
        )
        self._latest_sent_capture_ns: dict[str, int] = {}
        self._status_lock = threading.Lock()

    @property
    def observation_keys(self) -> frozenset[str]:
        return frozenset(reference.observation_key for reference in self._references)

    @property
    def stream_references(self) -> tuple[StreamReference, ...]:
        return self._references

    @property
    def session(self) -> StreamSessionView:
        return self._session

    def bind_session(self, session: StreamSessionView) -> bool:
        changed = self._session != session
        self._session = session
        if changed:
            with self._status_lock:
                self._latest_sent_capture_ns.clear()
        return changed

    def clear_session(self) -> None:
        self._session = StreamSessionView.inactive(
            pipeline_id=self._session.pipeline_id,
            contract_fingerprint=self._session.contract_fingerprint,
            deployment_fingerprint=self._session.deployment_fingerprint,
        )
        with self._status_lock:
            self._latest_sent_capture_ns.clear()

    def latest_sent_capture_ns(self, observation_key: str) -> int:
        with self._status_lock:
            return self._latest_sent_capture_ns.get(observation_key, 0)

    def observe_sender_status(self, status: object) -> bool:
        """Record the sender's post-send capture mapping for external RTP input."""
        if (
            getattr(status, "status_origin", "") != "sender"
            or getattr(status, "pipeline_id", "") != self._session.pipeline_id
            or not self._session.active
            or getattr(status, "session_id", "") != self._session.session_id
            or int(getattr(status, "session_generation", 0)) != self._session.generation
            or getattr(status, "observation_key", "") not in self.observation_keys
            or not bool(getattr(status, "timestamp_mapping_valid", False))
            or int(getattr(status, "encoded_frames", 0)) <= 0
            or int(getattr(status, "sent_packets", 0)) <= 0
        ):
            return False
        capture_ns = int(getattr(status, "mapping_capture_timestamp_ns", 0))
        if capture_ns <= 0:
            return False
        key = status.observation_key
        with self._status_lock:
            if capture_ns < self._latest_sent_capture_ns.get(key, 0):
                return False
            self._latest_sent_capture_ns[key] = capture_ns
        return True

    def reset(self) -> None:
        with self._status_lock:
            self._latest_sent_capture_ns.clear()

    def close(self) -> None:
        self.clear_session()

    def descriptors(self) -> tuple[()]:
        return ()

    def statuses(self) -> tuple[()]:
        return ()

    def diagnostic_snapshots(self) -> tuple[()]:
        return ()

    def sender_diagnostics(self) -> tuple[()]:
        """External producers own sender diagnostics and publish their own status."""
        return ()


def _execution_horizon_from_metadata(metadata: object, chunk_size: int) -> int:
    """Validate the optional policy-selected prefix before publishing a result."""
    value = metadata.get("execution_horizon", 0) if isinstance(metadata, Mapping) else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > chunk_size:
        raise ValueError(f"execution_horizon={value!r} is outside chunk_size={chunk_size}")
    return value


class PipelinePolicyNode(Node):
    """Own exactly one pipeline process and its pipeline-scoped ROS interfaces."""

    def __init__(
        self,
        config: PipelineNodeConfig,
        *,
        node_name: str,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        super().__init__(node_name)
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._registry_set = registry_set
        self._providers = providers
        if config.execution_mode not in {"monolithic", "distributed"}:
            raise RuntimeError(f"unsupported pipeline execution mode {config.execution_mode!r}")
        if config.scheduler_enabled and config.execution_mode != "monolithic":
            raise RuntimeError("scheduled inference supports only monolithic pipelines")

        self._config = config
        self._manager: InferencePipelineManager | None = None
        self._edge_runtime: EdgeProcessorRuntime | None = None
        self._edge_session: EdgeSession | None = None
        self._video_stream_manager: DeviceVideoStreamManager | None = None
        manifest_loader = (
            load_inference_manifest if config.execution_mode == "monolithic" else load_inference_manifest_metadata
        )
        self._manifest: ValidatedManifest = manifest_loader(config.model_path, config.deployment)
        self._n_obs_steps = self._manifest.policy.n_obs_steps
        self._contract = None
        self._frequency = 10.0
        self._obs_specs: list[SpecView] = []
        self._state_specs: list[SpecView] = []
        self._subs: dict[str, _SubState] = {}
        self._joint_rad_limits: list[tuple[float, float, float, float]] = []
        self._last_inference_time: float | None = None
        self._inference_count = 0
        self._last_error = ""
        self._remote_state = "unavailable"
        self._distributed_started_monotonic = time.monotonic()
        self._last_cloud_status_received_monotonic: float | None = None
        # Preserve the legacy operation lock unchanged. Scheduled Dispatch uses
        # separate capacity slots validated against backend capabilities.
        self._operation_lock = threading.Lock()
        self._reset_pending = threading.Event()
        self._observation_lock = threading.RLock()
        self._observation_epoch = 0
        self._observation_reset_cutoff_ns = 0
        self._pending_lock = threading.RLock()
        self._pending: dict[str, _PendingOperation] = {}
        self._goal_state_lock = threading.Lock()
        self._cancel_requested_goals: set[int] = set()
        self._cancel_confirmed_goals: set[int] = set()
        self._completed_goals: set[int] = set()
        self._goal_request_ids: dict[int, str] = {}

        self._load_contract(config.robot_config_path)
        runtime_policy: dict[str, object] = {}
        if config.scheduler_enabled:
            self._pipeline_compatibility_fingerprint = self._build_pipeline_compatibility_fingerprint()
            if not config.runtime_policy_fingerprint:
                raise RuntimeError("scheduled runtime policy JSON and fingerprint must be provided together")
            expected = hashlib.sha256(config.runtime_policy_json.encode("utf-8")).hexdigest()
            if expected != config.runtime_policy_fingerprint:
                raise RuntimeError("scheduled runtime policy JSON/fingerprint mismatch")
            try:
                runtime_policy = json.loads(config.runtime_policy_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"scheduled runtime policy JSON is invalid: {exc}") from exc
            if not isinstance(runtime_policy, dict):
                raise RuntimeError("scheduled runtime policy JSON must decode to an object")
            self._validate_runtime_policy(runtime_policy)
        self._setup_observation_subscriptions()
        try:
            runtime_options = json.loads(config.runtime_options_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"pipeline {config.pipeline_id!r} runtime_options_json is invalid: {exc}") from exc
        if not isinstance(runtime_options, dict):
            raise RuntimeError(f"pipeline {config.pipeline_id!r} runtime_options_json must decode to an object")
        if config.execution_mode == "monolithic":
            self._manager = create_pipeline_manager(
                config.pipeline_id,
                self._manifest,
                request_timeout=config.request_timeout,
                default_task=config.default_task or None,
                runtime_options=runtime_options,
                priority_scheduling=config.scheduler_enabled,
                registry_set=self._registry_set,
                providers=self._providers,
            )
        else:
            for name in ("request_topic", "result_topic", "heartbeat_topic"):
                if not getattr(config, name):
                    raise RuntimeError(f"distributed pipeline requires {name}")
            self._edge_runtime = EdgeProcessorRuntime(
                config.pipeline_id,
                self._manifest,
                default_task=config.default_task or None,
            )
            self._edge_runtime.load()
            self._edge_session = EdgeSession(build_pipeline_identity(config.pipeline_id, self._manifest))
            self._edge_session.start()
            if config.external_video_producer:
                self._video_stream_manager = _ExternalVideoProducerView(
                    self._obs_specs,
                    pipeline_id=config.pipeline_id,
                    contract_fingerprint=contract_fingerprint(self._contract),
                    deployment_fingerprint=self._manifest.fingerprint,
                )
            else:
                self._video_stream_manager = DeviceVideoStreamManager(
                    pipeline_id=config.pipeline_id,
                    contract_fingerprint=contract_fingerprint(self._contract),
                    deployment_fingerprint=self._manifest.fingerprint,
                    observation_specs=self._obs_specs,
                )

        self._action_pub = None
        if not config.scheduler_enabled:
            self._action_pub = self.create_publisher(VariantsList, config.action_topic, 10)
        self._health_pub = self.create_publisher(DiagnosticStatus, config.health_topic, 10)
        if config.execution_mode == "distributed":
            status_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            descriptor_qos = QoSProfile(
                depth=max(1, len(self._video_stream_manager.diagnostic_snapshots())),
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._request_pub = self.create_publisher(DistributedInferenceRequest, config.request_topic, 10)
            self.create_subscription(
                DistributedInferenceResult,
                config.result_topic,
                self._distributed_result_callback,
                10,
                callback_group=ReentrantCallbackGroup(),
            )
            self._status_pub = self.create_publisher(InferencePipelineStatus, config.heartbeat_topic, status_qos)
            self._video_descriptor_pub = None
            self._video_status_pub = None
            self._video_status_timer = None
            if config.external_video_producer:
                self.create_subscription(
                    VideoStreamStatus,
                    config.video_status_topic,
                    self._external_video_status_callback,
                    10,
                    callback_group=ReentrantCallbackGroup(),
                )
            else:
                self._video_descriptor_pub = self.create_publisher(
                    VideoStreamDescriptor, config.video_descriptor_topic, descriptor_qos
                )
                self._video_status_pub = self.create_publisher(VideoStreamStatus, config.video_status_topic, 10)
                self._video_status_timer = self.create_timer(0.25, self._publish_video_stream_control)
            self.create_subscription(
                InferencePipelineStatus,
                config.heartbeat_topic,
                self._cloud_status_callback,
                status_qos,
                callback_group=ReentrantCallbackGroup(),
            )
            self._status_timer = self.create_timer(0.5, self._publish_distributed_status)
            self._heartbeat_timer = self.create_timer(0.25, self._check_heartbeat)
        self._action_server = None
        self._reset_server = None
        self._reset_callback_group = MutuallyExclusiveCallbackGroup()
        if not config.scheduler_enabled:
            self._action_server = rclpy.action.ActionServer(
                self,
                DispatchInfer,
                config.action_server,
                execute_callback=self._dispatch_infer_callback,
                goal_callback=lambda _request: rclpy.action.GoalResponse.ACCEPT,
                cancel_callback=self._cancel_callback,
                callback_group=(
                    MutuallyExclusiveCallbackGroup()
                    if config.execution_mode == "monolithic"
                    else ReentrantCallbackGroup()
                ),
            )
            self._reset_server = self.create_service(
                Trigger,
                config.reset_service,
                self._reset_callback,
                callback_group=self._reset_callback_group,
            )
        self._health_timer = self.create_timer(1.0, self._publish_health)
        self._log_video_stream_diagnostics()

        # Scheduled-path action servers and serving status.
        # Only registered when scheduler_enabled; the legacy DispatchInfer/reset
        # above remain for the disabled branch.
        if config.scheduler_enabled:
            self._scheduled_operation_slots: threading.BoundedSemaphore | None = None
            self._scheduled_operation_capacity = 0
            self._session_controller = None
            self._pipeline_ledger = None
            self._serving_status_pub = None
            self._serving_status_timer = None
            self._boot_id = str(uuid.uuid4())
            self._serving_sequence = 0
            self._setup_scheduled_path()

        startup = (
            f"Unified pipeline started: id={config.pipeline_id}, mode={config.execution_mode}, "
            f"bundle={self._manifest.manifest.bundle.name}, "
            f"deployment={config.deployment}, backend={self._manifest.deployment.backend}, "
        )
        if config.scheduler_enabled:
            startup += "path=scheduled"
        else:
            startup += f"action={config.action_server}, reset={config.reset_service}"
        self.get_logger().info(startup)

    def _validate_runtime_policy(self, policy: dict[str, object]) -> None:
        expected_scalars = {
            "pipeline_id": self._config.pipeline_id,
            "execution_mode": self._config.execution_mode,
            "hardware_resource_id": self._config.hardware_resource_id,
            "deployment_fingerprint": self._manifest.fingerprint,
        }
        for field_name, expected in expected_scalars.items():
            if policy.get(field_name) != expected:
                raise RuntimeError(
                    f"scheduled runtime policy {field_name} mismatch: {policy.get(field_name)!r} != {expected!r}"
                )
        try:
            configured_capacity = json.loads(self._config.public_capacity_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"public_capacity_json is invalid: {exc}") from exc
        if policy.get("public_capacity") != configured_capacity:
            raise RuntimeError("scheduled runtime policy public_capacity mismatch")
        transport = policy.get("transport")
        if not isinstance(transport, dict):
            raise RuntimeError("scheduled runtime policy transport must be an object")
        expected_transport = {
            "open_session": self._config.scheduled_open_session,
            "dispatch": self._config.scheduled_dispatch,
            "close_session": self._config.scheduled_close_session,
            "serving_status": self._config.scheduled_serving_status,
            "health_topic": self._config.health_topic,
        }
        for field_name, expected in expected_transport.items():
            if transport.get(field_name) != expected:
                raise RuntimeError(f"scheduled runtime policy transport.{field_name} mismatch")
        # The runtime policy carries the normalized effective runtime options
        # declared in the SSOT; latency profiles are calibrated against that
        # execution configuration. A node parameter override that changes the
        # actually-executed options (e.g. enabling AutoHorizon attention
        # collection via runtime_options_json alone) would silently decouple
        # measurement identity from execution, so it fails startup instead.
        try:
            actual_options = json.loads(self._config.runtime_options_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} runtime_options_json is invalid: {exc}") from exc
        if not isinstance(actual_options, dict):
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} runtime_options_json must decode to an object")
        if policy.get("runtime_options") != effective_latency_runtime_options(actual_options):
            raise RuntimeError(
                "scheduled runtime policy runtime_options mismatch: the node would execute "
                "different runtime options than its SSOT-declared policy; update "
                "control_modes.<mode>.inference.pipelines.<id>.runtime_options in the robot "
                "configuration instead of overriding the runtime_options_json parameter"
            )

    def _load_contract(self, robot_config_path: str) -> None:
        config_path = Path(robot_config_path)
        if not config_path.is_file():
            raise RuntimeError(f"robot config file not found: {robot_config_path}")

        from robot_config.loader import build_contract_from_robot_config_dict, load_robot_config_dict

        robot_config = load_robot_config_dict(config_path)
        self._contract = build_contract_from_robot_config_dict(robot_config)
        self._frequency = float(self._contract.rate_hz)

        required_inputs = set(self._manifest.policy.input_features)
        all_observations = [spec for spec in iter_specs(self._contract) if not spec.is_action]
        self._obs_specs = [spec for spec in all_observations if spec.key in required_inputs]
        missing_inputs = sorted(required_inputs - {spec.key for spec in self._obs_specs})
        if missing_inputs:
            raise RuntimeError(
                f"pipeline {self._config.pipeline_id!r} robot contract does not provide required observations: "
                f"{missing_inputs}"
            )
        self._state_specs = [spec for spec in self._obs_specs if spec.key == "observation.state"]
        self._topic_to_qos = {observation.topic: observation.qos for observation in self._contract.observations}

        calibration_sources = resolve_calibration_source_specs_from_config(robot_config)
        joint_names = resolve_joint_names_from_config(robot_config)
        gripper_joints = resolve_gripper_joints_from_config(robot_config) or ["6"]
        norm_mode = resolve_lerobot_norm_mode(robot_config)
        if parse_bool(self._config.use_sim, default=False) and joint_names:
            from robot_config.launch_builders.description import generate_robot_description

            description = generate_robot_description(robot_config, True)
            if description is None:
                raise RuntimeError("failed to generate robot_description for simulated joint conversion")
            robot_description, _ = description
            self._joint_rad_limits = build_joint_conversion_table_from_urdf(
                robot_description,
                joint_names,
                gripper_joints,
                norm_mode=norm_mode,
            )
        elif calibration_sources and joint_names:
            self._joint_rad_limits = build_joint_conversion_table(
                calibration_sources,
                joint_names,
                gripper_joints,
                norm_mode=norm_mode,
            )

        velocity_joints = robot_config.get("ros2_control", {}).get("velocity_joints", [])
        base_velocity_max = robot_config.get("ros2_control", {}).get("base_vel_max_rad", 0)
        if velocity_joints and base_velocity_max > 0:
            self._joint_rad_limits.extend(
                (-float(base_velocity_max), float(base_velocity_max), 200.0, -100.0) for _ in velocity_joints
            )

    def _setup_observation_subscriptions(self) -> None:
        from rosidl_runtime_py.utilities import get_message

        for spec in self._obs_specs:
            key = self._subscription_key(spec)
            max_age_ms = int(spec.max_age_ms)
            step_ns = int(1e9 / self._frequency)
            asof_tolerance_ns = max(0, int(getattr(spec, "asof_tol_ms", 0))) * 1_000_000
            policy = str(getattr(spec, "resample_policy", "hold")).lower()
            alignment_window_ns = asof_tolerance_ns if policy == "asof" else step_ns if policy == "drop" else 0
            observation_history_ns = (self._n_obs_steps - 1) * step_ns
            self._subs[key] = _SubState(
                spec=spec,
                max_age_ns=max_age_ms * 1_000_000,
                step_ns=step_ns,
                history_window_ns=(
                    max(
                        step_ns * 2,
                        max_age_ms * 1_000_000 + step_ns,
                        alignment_window_ns + step_ns,
                    )
                    + observation_history_ns
                ),
            )
            message_type = get_message(spec.ros_type)
            qos = qos_profile_from_dict(self._topic_to_qos.get(spec.topic, {})) or QoSProfile(depth=10)
            self.create_subscription(
                message_type,
                spec.topic,
                lambda message, current_spec=spec: self._observation_callback(message, current_spec),
                qos,
                callback_group=ReentrantCallbackGroup(),
            )

    def _build_pipeline_compatibility_fingerprint(self) -> str:
        observation_specs = {self._subscription_key(spec): asdict(spec) for spec in self._obs_specs}
        action_specs = [asdict(spec) for spec in iter_specs(self._contract) if spec.is_action]
        policy = self._manifest.policy
        payload = {
            "rate_hz": self._frequency,
            "n_obs_steps": self._n_obs_steps,
            "observations": observation_specs,
            "actions": action_specs,
            "policy_type": policy.policy_type,
            "input_features": {
                key: feature.model_dump(mode="json") for key, feature in sorted(policy.input_features.items())
            },
            "output_features": {
                key: feature.model_dump(mode="json") for key, feature in sorted(policy.output_features.items())
            },
            "max_action_dimension": policy.max_action_dimension,
            "default_task": self._config.default_task,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _subscription_key(self, spec: SpecView) -> str:
        if spec.key == "observation.state" and len(self._state_specs) > 1:
            return f"{spec.key}_{spec.topic.replace('/', '_')}"
        return spec.key

    def _observation_callback(self, message: object, spec: SpecView) -> None:
        with self._observation_lock:
            epoch = self._observation_epoch
        receive_time = self.get_clock().now().nanoseconds
        timestamp = stamp_from_header_ns(message) if spec.stamp_src == "header" else None
        with self._observation_lock:
            if epoch != self._observation_epoch:
                return
            reset_cutoff_ns = self._observation_reset_cutoff_ns
            if reset_cutoff_ns > 0 and receive_time < reset_cutoff_ns:
                self._observation_reset_cutoff_ns = 0
                reset_cutoff_ns = 0
            if timestamp is not None and timestamp <= reset_cutoff_ns:
                return
            stored = PipelinePolicyNode._store_observation_locked(
                self, self._subscription_key(spec), epoch, int(timestamp or receive_time), receive_time, message
            )
        manager = getattr(self, "_video_stream_manager", None)
        if stored and manager is not None and spec.key in manager.observation_keys:
            try:
                manager.submit_ros_image(
                    spec.key,
                    message,
                    capture_timestamp_ns=int(timestamp or receive_time),
                    receive_timestamp_ns=receive_time,
                )
            except Exception as exc:
                self._last_error = f"video stream {spec.key} failed: {exc}"
                self.get_logger().error(self._last_error, throttle_duration_sec=1.0)

    def _store_observation(
        self,
        key: str,
        epoch: int,
        timestamp_ns: int,
        value: object,
        *,
        receive_time_ns: int | None = None,
    ) -> bool:
        with self._observation_lock:
            return PipelinePolicyNode._store_observation_locked(
                self,
                key,
                epoch,
                timestamp_ns,
                receive_time_ns if receive_time_ns is not None else timestamp_ns,
                value,
            )

    @staticmethod
    def _store_observation_locked(
        node: object, key: str, epoch: int, timestamp_ns: int, receive_time_ns: int, value: object
    ) -> bool:
        if epoch != node._observation_epoch:
            return False
        state = node._subs[key]
        return PipelinePolicyNode._buffer_for_state(state).push(
            timestamp_ns,
            value,
            receive_time_ns=receive_time_ns,
        )

    @staticmethod
    def _buffer_for_state(state: object) -> StreamBuffer:
        buffer = getattr(state, "buffer", None)
        if buffer is not None:
            return buffer
        spec = state.spec
        buffer = StreamBuffer(
            str(getattr(spec, "resample_policy", "hold")).lower(),
            state.step_ns,
            max(0, int(getattr(spec, "asof_tol_ms", 0))) * 1_000_000,
            max_age_ns=state.max_age_ns,
            retention_ns=state.history_window_ns,
        )
        buffer.history.extend(state.history)
        state.buffer = buffer
        state.history = buffer.history
        return buffer

    @staticmethod
    def _rtp_video_send_issue(node: object, state: _SubState, now_ns: int) -> dict[str, object] | None:
        """Gate RTP-video freshness on what the sender actually put on the wire.

        The local subscription buffer says what the device received; the
        compute side can only see frames that were encoded and sent.  Frames
        lost to encode failures, queue overflow, or session rollovers must
        therefore read as "not fresh" here, mirroring the compute side's
        snapshot instead of the local buffer.
        """
        manager = getattr(node, "_video_stream_manager", None)
        sent_ns = manager.latest_sent_capture_ns(state.spec.key) if manager is not None else 0
        if sent_ns <= 0:
            return {"key": state.spec.key, "topic": state.spec.topic, "reason": "video_not_sent"}
        age_ns = now_ns - sent_ns
        if state.max_age_ns > 0 and age_ns > state.max_age_ns:
            return {
                "key": state.spec.key,
                "topic": state.spec.topic,
                "reason": "video_send_stale",
                "age_ms": age_ns / 1_000_000,
                "max_age_ms": state.max_age_ns / 1_000_000,
            }
        return None

    @staticmethod
    def _sample_observation(
        state: _SubState,
        sample_time_ns: int,
        now_ns: int | None = None,
        *,
        check_live_age: bool = True,
    ) -> tuple[Any | None, dict[str, object] | None]:
        value, issue = PipelinePolicyNode._buffer_for_state(state).select(
            sample_time_ns,
            now_ns=now_ns,
            check_live_age=check_live_age,
        )
        if issue is not None:
            issue = {"key": state.spec.key, "topic": state.spec.topic, **issue}
        return value, issue

    @staticmethod
    def _sample_observation_history(
        state: _SubState, sample_times_ns: list[int], now_ns: int
    ) -> tuple[list[Any] | None, dict[str, object] | None]:
        if not state.history:
            _, issue = PipelinePolicyNode._sample_observation(state, sample_times_ns[-1], now_ns)
            return None, issue
        if state.history[0][0] > sample_times_ns[-1]:
            _, issue = PipelinePolicyNode._sample_observation(state, sample_times_ns[-1], now_ns)
            return None, issue

        first_timestamp_ns, _, first_value = state.history[0]
        values: list[Any] = []
        for index, sample_time_ns in enumerate(sample_times_ns):
            if sample_time_ns < first_timestamp_ns:
                values.append(first_value)
                continue
            value, issue = PipelinePolicyNode._sample_observation(
                state,
                sample_time_ns,
                now_ns,
                check_live_age=index == len(sample_times_ns) - 1,
            )
            if issue is not None:
                return None, issue
            values.append(value)
        return values, None

    def _sample_observations(
        self, sample_time_ns: int, *, rtp_video_keys: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Sample subscribed observations at ``sample_time_ns``.

        Keys in ``rtp_video_keys`` (distributed RTP video streams) are never
        consumed from the local buffer -- the policy samples their decoded
        frames on the compute side -- so their freshness gate reads the
        sender's last-sent capture timestamp instead of the buffer entry.
        A fresh buffer with nothing on the wire correctly fails closed.
        """
        selected: dict[str, object] = {}
        issues: list[dict[str, object]] = []
        now_ns = self.get_clock().now().nanoseconds if hasattr(self, "get_clock") else sample_time_ns
        step_ns = next(iter(self._subs.values())).step_ns if self._subs else int(1e9 / self._frequency)
        sample_times_ns = [sample_time_ns - step_ns * offset for offset in reversed(range(self._n_obs_steps))]
        with self._observation_lock:
            for key, state in self._subs.items():
                if state.spec.key in rtp_video_keys:
                    selected[key] = None
                    issue = PipelinePolicyNode._rtp_video_send_issue(self, state, now_ns)
                    if issue is not None:
                        issues.append(issue)
                    continue
                if self._n_obs_steps == 1:
                    value, issue = self._sample_observation(state, sample_time_ns, now_ns)
                else:
                    value, issue = self._sample_observation_history(state, sample_times_ns, now_ns)
                if issue is not None:
                    issues.append(issue)
                selected[key] = value

        if issues:
            raise ObservationNotReadyError(issues)

        sampled: dict[str, Any] = {}
        decode_issues: list[dict[str, object]] = []
        for key, message in selected.items():
            state = self._subs[key]
            if state.spec.key in rtp_video_keys:
                continue
            messages = message if self._n_obs_steps > 1 else [message]
            values = [decode_value(state.spec.ros_type, item, state.spec) for item in messages]
            if any(value is None for value in values):
                decode_issues.append({"key": state.spec.key, "topic": state.spec.topic, "reason": "decode_failed"})
                sampled[key] = None
            else:
                sampled[key] = np.ascontiguousarray(np.stack(values)[None, ...]) if self._n_obs_steps > 1 else values[0]
        if decode_issues:
            raise ObservationNotReadyError(decode_issues)

        observations: dict[str, Any] = {}
        if len(self._state_specs) > 1:
            state_parts = [sampled[self._subscription_key(spec)] for spec in self._state_specs]
            observations["observation.state"] = np.concatenate(state_parts, axis=-1)

        for key, value in sampled.items():
            if key.startswith("observation.state_") and len(self._state_specs) > 1:
                continue
            observations[key] = value
        return observations

    def _clear_observation_buffers(self, reset_time_ns: int | None = None) -> None:
        with self._observation_lock:
            self._observation_epoch += 1
            if reset_time_ns is not None:
                self._observation_reset_cutoff_ns = max(self._observation_reset_cutoff_ns, reset_time_ns)
            for state in self._subs.values():
                PipelinePolicyNode._buffer_for_state(state).reset()
        video_stream_manager = getattr(self, "_video_stream_manager", None)
        if video_stream_manager is not None:
            video_stream_manager.reset()

    def _rad_to_lerobot(self, state: np.ndarray) -> np.ndarray:
        if not self._joint_rad_limits:
            return np.ascontiguousarray(state, dtype=np.float32)
        converted = state.astype(np.float64).copy()
        for index, (minimum, maximum, span, offset) in enumerate(self._joint_rad_limits):
            if index < converted.shape[-1]:
                converted[..., index] = (state[..., index] - minimum) / (maximum - minimum) * span + offset
        return converted.astype(np.float32)

    def _lerobot_to_rad(self, action: object) -> np.ndarray:
        candidate = action
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        if not self._joint_rad_limits:
            # closed-loop inference: when no joint conversion is configured, the action
            # must still be returned as contiguous float32 so the
            # StepBenchmark wire contract (and the LIBERO adapter's
            # validate_libero_action_dict) accepts it. The previous code
            # converted to float64 before this early return, producing a
            # float64 payload that the adapter rejects.
            return np.ascontiguousarray(candidate, dtype=np.float32)
        converted = np.asarray(candidate).astype(np.float64).copy()
        for index, (minimum, maximum, span, offset) in enumerate(self._joint_rad_limits):
            if index < converted.shape[-1]:
                converted[..., index] = (converted[..., index] - offset) / span * (maximum - minimum) + minimum
        return converted.astype(np.float32)

    @staticmethod
    def _to_policy_inputs(observations: dict[str, Any]) -> dict[str, Any]:
        return {
            key: torch.from_numpy(np.ascontiguousarray(value)) if isinstance(value, np.ndarray) else value
            for key, value in observations.items()
        }

    def _dispatch_infer_callback(self, goal_handle: object) -> DispatchInfer.Result:
        request = goal_handle.request
        request_id = request.inference_id or f"{self._config.pipeline_id}-{time.monotonic_ns()}"
        self._goal_request_ids[id(goal_handle)] = request_id
        sample_time = request.obs_timestamp.sec * 1_000_000_000 + request.obs_timestamp.nanosec
        if sample_time <= 0:
            sample_time = self.get_clock().now().nanoseconds
        total_start = time.perf_counter()
        request_start_monotonic_ns = time.monotonic_ns()

        try:
            if self._goal_cancel_requested(goal_handle):
                raise RequestCanceledError(f"inference request {request_id!r} was canceled before admission")
            self._acquire_operation()
            try:
                if self._goal_cancel_requested(goal_handle):
                    raise RequestCanceledError(f"inference request {request_id!r} was canceled before execution")
                return self._execute_inference_request(
                    goal_handle,
                    request,
                    request_id,
                    sample_time,
                    total_start,
                    request_start_monotonic_ns,
                )
            finally:
                self._operation_lock.release()
        except Exception as exc:
            self._last_error = str(exc)
            if bool(getattr(exc, "recoverable", False)):
                self.get_logger().warning(str(exc), throttle_duration_sec=1.0)
            else:
                self.get_logger().error(f"pipeline inference failed: {exc}\n{traceback.format_exc()}")
            error = (
                exc.error
                if isinstance(exc, DistributedProtocolError)
                else structured_error_from_exception(exc, str(getattr(exc, "stage", "pipeline")))
            )
            if self._goal_cancel_confirmed(goal_handle) and error.code != "request_canceled":
                error = StructuredError(
                    code="request_canceled",
                    message=f"inference request {request_id!r} was canceled; remote cancellation status: {exc}",
                    stage="cancel",
                    recoverable=True,
                    details={"cause_code": error.code, "cause_message": error.message},
                )
            response = DispatchInfer.Result()
            response.action_chunk = VariantsList()
            response.chunk_size = 0
            response.success = False
            response.message = str(exc)
            response.inference_latency_ms = (time.perf_counter() - total_start) * 1000.0
            response.pipeline_id = self._config.pipeline_id
            response.request_id = request_id
            response.deployment_fingerprint = self._manifest.fingerprint
            response.backend_latency_ms = 0.0
            response.performance_json = json.dumps(
                {
                    "clock_domain": "monotonic",
                    "execution_mode": self._config.execution_mode,
                    "request_start_monotonic_ns": request_start_monotonic_ns,
                    "result_monotonic_ns": time.monotonic_ns(),
                    "error_stage": error.stage,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            response.error = error_to_message(error)
            if error.code == "request_canceled":
                self._finish_canceled_goal(goal_handle)
            else:
                goal_handle.abort()
            return response
        finally:
            with self._goal_state_lock:
                self._cancel_requested_goals.discard(id(goal_handle))
                self._cancel_confirmed_goals.discard(id(goal_handle))
                self._completed_goals.discard(id(goal_handle))
            self._goal_request_ids.pop(id(goal_handle), None)

    def _acquire_operation(self) -> None:
        if self._reset_pending.is_set():
            raise PipelineBusyError(f"pipeline {self._config.pipeline_id!r} is waiting to reset")
        if not self._operation_lock.acquire(blocking=False):
            raise PipelineBusyError(f"pipeline {self._config.pipeline_id!r} is already processing another operation")
        if self._reset_pending.is_set():
            self._operation_lock.release()
            raise PipelineBusyError(f"pipeline {self._config.pipeline_id!r} is waiting to reset")

    def _goal_cancel_requested(self, goal_handle: object) -> bool:
        return bool(goal_handle.is_cancel_requested)

    def _goal_cancel_confirmed(self, goal_handle: object) -> bool:
        with self._goal_state_lock:
            return id(goal_handle) in self._cancel_confirmed_goals

    def _mark_goal_cancel_confirmed(self, goal_handle: object) -> None:
        with self._goal_state_lock:
            self._cancel_confirmed_goals.add(id(goal_handle))

    def _finish_canceled_goal(self, goal_handle: object) -> None:
        deadline = time.monotonic() + min(0.1, self._config.request_timeout)
        while not goal_handle.is_cancel_requested and time.monotonic() < deadline:
            time.sleep(0.001)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()

    def _commit_action(
        self,
        goal_handle: object,
        request_id: str,
        raw_action: object,
        deadline: datetime | None = None,
    ) -> VariantsList:
        action = self._lerobot_to_rad(raw_action)
        action_message = TensorMsgConverter.to_variant({"action": action})
        with self._goal_state_lock:
            if deadline is not None and datetime.now(timezone.utc) >= deadline:
                expired = True
                canceled = False
            elif id(goal_handle) in self._cancel_requested_goals or goal_handle.is_cancel_requested:
                expired = False
                canceled = True
            else:
                expired = False
                canceled = False
                if self._action_pub is None:
                    raise RuntimeError("legacy action publisher is unavailable on the scheduled path")
                self._action_pub.publish(action_message)
                goal_handle.succeed()
                self._completed_goals.add(id(goal_handle))
        if expired:
            raise DeadlineExceededError(f"inference request {request_id!r} expired before action publication")
        if canceled:
            self._fail_distributed_after_late_cancel(request_id)
            raise RequestCanceledError(f"inference request {request_id!r} was canceled before action publication")
        return action_message

    def _fail_distributed_after_late_cancel(self, request_id: str) -> None:
        if self._config.execution_mode != "distributed":
            return
        error = StructuredError(
            code="cancellation_after_remote_completion",
            message=f"inference request {request_id!r} was canceled after remote execution completed",
            stage="cancel",
            details={"request_id": request_id},
        )
        update = self._require_edge_session().fail(error)
        self._complete_invalidated(update.invalidated_request_ids, update.error)

    def _execute_inference_request(
        self,
        goal_handle: object,
        request: object,
        request_id: str,
        sample_time: int,
        total_start: float,
        request_start_monotonic_ns: int,
    ) -> DispatchInfer.Result:
        deadline = self._goal_deadline(request.deadline)
        if deadline is None:
            deadline = datetime.now(timezone.utc) + timedelta(seconds=self._config.request_timeout)
        self._raise_if_deadline_expired(deadline, request_id)
        video_keys = (
            self._video_stream_manager.observation_keys
            if self._config.execution_mode == "distributed" and self._video_stream_manager is not None
            else frozenset()
        )
        observations = self._sample_observations(sample_time, rtp_video_keys=video_keys)
        self._raise_if_deadline_expired(deadline, request_id)
        if "observation.state" in observations:
            observations["observation.state"] = self._rad_to_lerobot(observations["observation.state"])
        service_performance: dict[str, object]
        if self._config.execution_mode == "monolithic":
            observations = self._to_policy_inputs(observations)
            inference_start_monotonic_ns = time.monotonic_ns()
            result = self._require_manager().infer(
                self._config.pipeline_id,
                InferenceRequest(
                    request_id=request_id,
                    inputs=observations,
                    prompt=request.prompt if request.prompt else None,
                    deadline=deadline,
                    priority=0,
                ),
            )
            raw_action = result.action
            chunk_size = result.actual_chunk_size
            inference_end_monotonic_ns = time.monotonic_ns()
            execution_horizon = _execution_horizon_from_metadata(result.metadata, chunk_size)
            backend_latency_ms = result.backend_latency_ms
            total_latency_ms = result.total_latency_ms
            service_performance = {
                "clock_domain": "monotonic",
                "execution_mode": "monolithic",
                "request_start_monotonic_ns": request_start_monotonic_ns,
                "inference_start_monotonic_ns": inference_start_monotonic_ns,
                "inference_end_monotonic_ns": inference_end_monotonic_ns,
                "transport": {"mode": "dds", "streams": []},
            }
        else:
            edge_runtime = self._require_edge_runtime()
            try:
                canonical_inputs = edge_runtime.preprocess(
                    observations,
                    prompt=request.prompt if request.prompt else None,
                )
                video_manager = self._video_stream_manager
                stream_references: tuple[StreamReference, ...] = ()
                if video_manager is not None:
                    stream_references = video_manager.stream_references
                    canonical_inputs = {
                        key: value
                        for key, value in canonical_inputs.items()
                        if key not in video_manager.observation_keys
                    }
                self._raise_if_deadline_expired(deadline, request_id)
                distributed_result = self._round_trip(
                    Operation.INFER,
                    request_id,
                    inputs=dict(canonical_inputs),
                    prompt=request.prompt if request.prompt else None,
                    deadline=deadline,
                    goal_handle=goal_handle,
                    observation_timestamp_ns=sample_time if stream_references else 0,
                    stream_references=stream_references,
                )
                self._raise_if_deadline_expired(deadline, request_id)
                if self._goal_cancel_requested(goal_handle):
                    self._fail_distributed_after_late_cancel(request_id)
                    raise RequestCanceledError(f"inference request {request_id!r} was canceled before postprocessing")
                raw_action = edge_runtime.postprocess(
                    distributed_result.action,
                    actual_chunk_size=distributed_result.actual_chunk_size,
                )
                self._raise_if_deadline_expired(deadline, request_id)
                chunk_size = distributed_result.actual_chunk_size
                execution_horizon = int(distributed_result.execution_horizon)
                backend_latency_ms = distributed_result.backend_latency_ms
                total_latency_ms = (time.perf_counter() - total_start) * 1000.0
                service_performance = dict(distributed_result.performance)
                service_performance["execution_mode"] = "distributed"
            except Exception as exc:
                self._fail_distributed_after_deadline(request_id, exc)
                raise

        try:
            action_message = self._commit_action(goal_handle, request_id, raw_action, deadline)
        except Exception as exc:
            self._fail_distributed_after_deadline(request_id, exc)
            raise

        response = DispatchInfer.Result()
        response.action_chunk = action_message
        response.chunk_size = chunk_size
        response.execution_horizon = execution_horizon
        response.success = True
        response.message = "OK"
        response.inference_latency_ms = total_latency_ms
        response.pipeline_id = self._config.pipeline_id
        response.request_id = request_id
        response.deployment_fingerprint = self._manifest.fingerprint
        response.backend_latency_ms = backend_latency_ms
        service_performance["pipeline_result_monotonic_ns"] = time.monotonic_ns()
        response.performance_json = json.dumps(service_performance, sort_keys=True, separators=(",", ":"))
        response.error = error_to_message(None)
        self._last_inference_time = time.time()
        self._inference_count += 1
        self._last_error = ""
        self.get_logger().info(
            f"Inference succeeded: request={request_id}, mode={self._config.execution_mode}, "
            f"chunk={chunk_size}, latency={total_latency_ms:.1f}ms, backend={backend_latency_ms:.1f}ms"
        )
        return response

    def _reset_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=self._config.request_timeout)
        return PipelinePolicyNode._reset_with_deadline(self, response, deadline)

    def _reset_with_deadline(self, response: Trigger.Response, deadline: datetime) -> Trigger.Response:
        self._reset_pending.set()
        acquired = self._operation_lock.acquire(
            timeout=max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
        )
        if not acquired:
            self._reset_pending.clear()
            response.success = False
            response.message = f"pipeline {self._config.pipeline_id!r} reset timed out waiting for active operation"
            self._last_error = response.message
            return response
        try:
            reset_error: Exception | None = None
            terminal_reset_failure = False
            clear_observations = False
            if self._config.execution_mode == "monolithic":
                try:
                    self._require_manager().reset(self._config.pipeline_id, deadline)
                except Exception as exc:
                    reset_error = exc
                    # Only clear the observation buffers when the pipeline is
                    # broken after a failed reset. On success the pipeline
                    # state is already reset by ``manager.reset()``; keeping
                    # the observation history intact allows the first
                    # inference of a freshly-prepared episode to sample the
                    # reset observation at the exact reset timestamp. The
                    # prepare-flow reset (benchmark episode controller PreparePolicyEpisode) runs
                    # between ResetBenchmark and RunPolicy, so clearing the
                    # buffers here would discard the reset observations the
                    # first inference needs.
                    clear_observations = not self._require_manager().health(self._config.pipeline_id).ready
                else:
                    clear_observations = False
            else:
                reset_error, terminal_reset_failure = self._reset_distributed_pipeline(deadline)
                # PreparePolicyEpisode runs after ResetBenchmark has delivered
                # the reset observation. A successful policy reset must retain
                # both DDS state history and externally streamed frames. Only a
                # terminal/unknown distributed failure invalidates that data.
                clear_observations = terminal_reset_failure
            if clear_observations:
                self._clear_observation_buffers(self.get_clock().now().nanoseconds)

            if reset_error is not None:
                if terminal_reset_failure:
                    error = (
                        reset_error.error
                        if isinstance(reset_error, DistributedProtocolError)
                        else structured_error_from_exception(reset_error, "reset")
                    )
                    update = self._require_edge_session().fail(error)
                    self._complete_invalidated(update.invalidated_request_ids, update.error)
                response.success = False
                response.message = str(reset_error)
                self._last_error = str(reset_error)
            else:
                response.success = True
                response.message = f"pipeline {self._config.pipeline_id} reset"
                self._last_error = ""
            return response
        finally:
            self._reset_pending.clear()
            self._operation_lock.release()

    def _reset_distributed_pipeline(self, deadline: datetime | None = None) -> tuple[Exception | None, bool]:
        if not self._require_edge_session().reset_supported:
            return (
                DistributedProtocolError(
                    StructuredError(
                        code="unsupported_capability",
                        message="cloud pipeline does not support reset",
                        stage="reset",
                    )
                ),
                False,
            )

        deadline = deadline or datetime.now(timezone.utc) + timedelta(seconds=self._config.request_timeout)
        request_id = f"{self._config.pipeline_id}-reset-{time.monotonic_ns()}"
        progress = _RoundTripProgress()
        try:
            self._round_trip(
                Operation.RESET,
                request_id,
                deadline=deadline,
                progress=progress,
            )
        except Exception as exc:
            terminal = progress.published and (not progress.response_received or not progress.backend_ready)
            return exc, terminal
        try:
            self._require_edge_runtime().reset(deadline)
        except Exception as exc:
            return exc, True
        return None, False

    def _publish_health(self) -> None:
        try:
            if self._config.execution_mode == "monolithic":
                diagnostics = self._require_manager().health(self._config.pipeline_id)
                backend_health = diagnostics.backend_health
                level = DiagnosticStatus.OK if diagnostics.ready else DiagnosticStatus.WARN
                message = backend_health.message or diagnostics.state.value
                values = {
                    **diagnostics.metadata,
                    "backend_state": backend_health.state.value,
                    "backend_reason": backend_health.reason_code or "",
                    "backend_failure_count": backend_health.failure_count,
                }
            else:
                session = self._require_edge_session()
                level = DiagnosticStatus.OK if session.ready else DiagnosticStatus.WARN
                message = self._remote_state if session.ready else session.state.value
                session_id, generation = session.session
                values = {
                    "pipeline_id": self._config.pipeline_id,
                    "bundle": self._manifest.manifest.bundle.name,
                    "deployment": self._config.deployment,
                    "deployment_fingerprint": self._manifest.fingerprint,
                    "backend": self._manifest.deployment.backend,
                    "state": session.state.value,
                    "backend_state": self._remote_state,
                    "remote_state": self._remote_state,
                    "external_video_producer": bool(getattr(self._config, "external_video_producer", False)),
                    "session_id": session_id,
                    "session_generation": generation,
                }
                video_manager = self._video_stream_manager
                if video_manager is not None:
                    snapshots = video_manager.diagnostic_snapshots()
                    values["video_stream.count"] = len(snapshots)
                    for snapshot in snapshots:
                        values[f"video_stream.{snapshot.observation_key}"] = self._format_video_stream_diagnostic(
                            snapshot
                        )
            values.update(
                {
                    "inference_count": self._inference_count,
                    "last_inference_time": self._last_inference_time or "",
                }
            )
        except Exception as exc:
            level = DiagnosticStatus.ERROR
            message = str(exc)
            values = {"pipeline_id": self._config.pipeline_id, "state": "unavailable"}

        health = DiagnosticStatus()
        health.level = level
        health.name = f"inference/{self._config.pipeline_id}"
        health.message = message if level == DiagnosticStatus.ERROR else self._last_error or message
        health.hardware_id = self._manifest.fingerprint
        health.values = [KeyValue(key=str(key), value=str(value)) for key, value in values.items()]
        self._health_pub.publish(health)

    # ==================================================================
    # Scheduled path: product session and serving status.
    # ==================================================================

    def _set_scheduled_error(
        self,
        error,
        *,
        code: object,
        message: object = "",
        recoverable: bool = False,
        stage: object = "",
        details: dict[str, object] | None = None,
    ) -> None:
        set_scheduled_error(
            error,
            code=code,
            message=message,
            recoverable=recoverable,
            stage=stage,
            details=details,
            max_message_bytes=self._config.max_error_message_bytes,
            max_details_bytes=self._config.max_error_details_bytes,
        )

    def _scheduled_deadline(self, deadline_message: object) -> datetime:
        deadline = self._goal_deadline(deadline_message)
        if deadline is None:
            raise IdempotencyError("scheduled pipeline requires an absolute deadline from Global")
        return deadline

    def _setup_scheduled_path(self) -> None:
        """Register pipeline-scoped scheduled action servers and serving status.

        Uses ProductSessionController for session/generation/fencing and the
        existing InferencePipeline.infer() for whole-graph execution.
        """
        from inference_service.scheduler.session_controller import (
            ProductSessionController,
            WorkClassCapacity,
        )
        from inference_service.scheduler.work_classes import WorkClass, work_class_name

        # parse public_capacity from config JSON
        caps: dict[WorkClass, WorkClassCapacity] = {}
        try:
            cap_data = json.loads(self._config.public_capacity_json) if self._config.public_capacity_json else {}
        except json.JSONDecodeError:
            cap_data = {}
        for work_class in WorkClass:
            wc_name = work_class_name(work_class)
            if wc_name in cap_data:
                caps[work_class] = WorkClassCapacity(
                    work_class,
                    int(cap_data[wc_name].get("max_in_flight", 1)),
                )
        # Always include session_control and action_generation.
        if WorkClass.SESSION_CONTROL not in caps:
            caps[WorkClass.SESSION_CONTROL] = WorkClassCapacity(WorkClass.SESSION_CONTROL, 1)
        if WorkClass.ACTION_GENERATION not in caps:
            caps[WorkClass.ACTION_GENERATION] = WorkClassCapacity(WorkClass.ACTION_GENERATION, 1)

        action_capacity = caps[WorkClass.ACTION_GENERATION].max_in_flight
        backend_capacity = self._require_manager().capabilities(self._config.pipeline_id).max_in_flight_per_instance
        if action_capacity > backend_capacity:
            raise RuntimeError(
                "configured action_generation capacity exceeds backend max_in_flight_per_instance: "
                f"{action_capacity} > {backend_capacity}"
            )
        self._scheduled_operation_slots = threading.BoundedSemaphore(action_capacity)
        self._scheduled_operation_capacity = action_capacity

        self._session_controller = ProductSessionController(
            boot_id=self._boot_id,
            capacities=caps,
            session_idle_timeout_ns=self._config.session_idle_timeout_ns or 30_000_000_000,
            now_ns=time.monotonic_ns,
        )
        self._session_controller.mark_ready()
        self._pipeline_ledger = IdempotencyLedger(
            max_session_records=self._config.max_session_records,
            max_duplicate_waiters_per_request=self._config.max_duplicate_waiters_per_request,
            terminal_session_retention_ns=self._config.terminal_session_retention_ns,
            now_ns=time.monotonic_ns,
            max_entries=self._config.max_session_records * (self._config.terminal_result_cache_entries + 4),
            max_terminal_entries_per_session=self._config.terminal_result_cache_entries,
        )

        # Serving status publisher.
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._serving_status_pub = self.create_publisher(
            InferenceServingStatus, self._config.scheduled_serving_status, status_qos
        )
        self._serving_status_timer = self.create_timer(0.5, self._publish_serving_status)

        # scheduled action servers (reentrant group — independent from legacy)
        group = ReentrantCallbackGroup()
        self._scheduled_lifecycle_goal_slots = GoalSlotPool(("open", "close"))
        self._scheduled_dispatch_goal_slots = GoalSlotPool(
            ("dispatch",),
            capacity=max(GOAL_CONTEXTS_PER_ENDPOINT, action_capacity),
        )
        self._scheduled_open_server = rclpy.action.ActionServer(
            self,
            OpenInferenceSession,
            self._config.scheduled_open_session,
            execute_callback=lambda goal_handle: self._scheduled_lifecycle_goal_slots.run(
                "open", self._scheduled_open_callback, goal_handle
            ),
            goal_callback=lambda _request: (
                rclpy.action.GoalResponse.ACCEPT
                if self._scheduled_lifecycle_goal_slots.try_acquire("open")
                else rclpy.action.GoalResponse.REJECT
            ),
            callback_group=group,
        )
        self._scheduled_dispatch_server = rclpy.action.ActionServer(
            self,
            ScheduledDispatchInfer,
            self._config.scheduled_dispatch,
            execute_callback=lambda goal_handle: self._scheduled_dispatch_goal_slots.run(
                "dispatch", self._scheduled_dispatch_callback, goal_handle
            ),
            goal_callback=lambda _request: (
                rclpy.action.GoalResponse.ACCEPT
                if self._scheduled_dispatch_goal_slots.try_acquire("dispatch")
                else rclpy.action.GoalResponse.REJECT
            ),
            cancel_callback=self._scheduled_cancel_callback,
            callback_group=group,
        )
        self._scheduled_close_server = rclpy.action.ActionServer(
            self,
            CloseInferenceSession,
            self._config.scheduled_close_session,
            execute_callback=lambda goal_handle: self._scheduled_lifecycle_goal_slots.run(
                "close", self._scheduled_close_callback, goal_handle
            ),
            goal_callback=lambda _request: (
                rclpy.action.GoalResponse.ACCEPT
                if self._scheduled_lifecycle_goal_slots.try_acquire("close")
                else rclpy.action.GoalResponse.REJECT
            ),
            callback_group=group,
        )

    def _publish_serving_status(self) -> None:
        """Publish state, generation, fingerprints,
        hardware identity, capacities."""
        if self._session_controller is None or self._serving_status_pub is None:
            return
        snap = self._session_controller.snapshot()
        msg = InferenceServingStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pipeline_id = self._config.pipeline_id
        msg.boot_id = self._boot_id
        self._serving_sequence += 1
        msg.sequence = self._serving_sequence
        msg.state = int(snap.state)
        msg.product_session_id = snap.product_session_id
        msg.product_session_generation = snap.product_session_generation
        lease_expires_at_ros_ns = monotonic_expiry_to_ros_ns(
            snap.lease_expires_at_ns,
            monotonic_now_ns=time.monotonic_ns(),
            ros_now_ns=self.get_clock().now().nanoseconds,
        )
        msg.lease_expires_at.sec, msg.lease_expires_at.nanosec = divmod(lease_expires_at_ros_ns, 1_000_000_000)
        msg.deployment_fingerprint = self._manifest.fingerprint
        msg.runtime_policy_fingerprint = self._config.runtime_policy_fingerprint
        msg.pipeline_compatibility_fingerprint = self._pipeline_compatibility_fingerprint
        msg.configured_hardware_resource_id = self._config.hardware_resource_id
        runtime_resource_id = self._runtime_hardware_resource_id()
        msg.runtime_hardware_resource_id = runtime_resource_id
        msg.hardware_priority_levels = self._runtime_hardware_priority_levels()
        from inference_service.scheduler.session_controller import ServingState
        from inference_service.scheduler.work_classes import WorkClass, work_class_name

        for work_class in WorkClass:
            try:
                in_flight = self._session_controller.capacity_count(work_class)
            except KeyError:
                continue
            configured = json.loads(self._config.public_capacity_json)[work_class_name(work_class)]
            capacity = InferenceWorkCapacity()
            capacity.work_class = int(work_class)
            capacity.max_in_flight = int(configured["max_in_flight"])
            capacity.current_in_flight = in_flight
            state_accepting = (
                snap.state == ServingState.IDLE
                if work_class == WorkClass.SESSION_CONTROL
                else snap.state == ServingState.ACTIVE
            )
            capacity.accepting_requests = state_accepting and in_flight < capacity.max_in_flight
            msg.capacities.append(capacity)
        if not runtime_resource_id:
            msg.error.code = "runtime_hardware_identity_unavailable"
            msg.error.message = "backend did not report a runtime hardware resource identity"
            msg.error.recoverable = False
        self._serving_status_pub.publish(msg)

    def _runtime_hardware_resource_id(self) -> str:
        capabilities = self._require_manager().capabilities(self._config.pipeline_id)
        return capabilities.hardware_resource_id or ""

    def _runtime_hardware_priority_levels(self) -> int:
        mapping = self._require_manager().capabilities(self._config.pipeline_id).priority_mapping
        return mapping.generic_level_count if mapping is not None else 1

    def _scheduled_open_callback(self, goal_handle) -> OpenInferenceSession.Result:
        goal = goal_handle.request
        return self._execute_pipeline_idempotent(
            goal_handle=goal_handle,
            action=LedgerAction.OPEN,
            key=open_key(goal.session_id),
            payload={"deadline": goal.deadline},
            deadline=goal.deadline,
            execute=self._scheduled_open_once,
        )

    def _scheduled_dispatch_callback(self, goal_handle) -> ScheduledDispatchInfer.Result:
        goal = goal_handle.request
        return self._execute_pipeline_idempotent(
            goal_handle=goal_handle,
            action=LedgerAction.DISPATCH,
            key=dispatch_key(goal.session_id, goal.session_generation, goal.request_id),
            payload={
                "session_generation": goal.session_generation,
                "obs_timestamp": goal.obs_timestamp,
                "prompt": goal.prompt,
                "priority": goal.priority,
                "target_pipeline_id": goal.target_pipeline_id,
                "fallback_chain": list(goal.fallback_chain),
                "deadline": goal.deadline,
            },
            deadline=goal.deadline,
            execute=self._scheduled_dispatch_once,
            request_id=goal.request_id,
        )

    def _scheduled_close_callback(self, goal_handle) -> CloseInferenceSession.Result:
        goal = goal_handle.request
        return self._execute_pipeline_idempotent(
            goal_handle=goal_handle,
            action=LedgerAction.CLOSE,
            key=close_key(goal.session_id, goal.session_generation),
            payload={"session_generation": goal.session_generation},
            deadline=goal.deadline,
            execute=self._scheduled_close_once,
        )

    def _execute_pipeline_idempotent(
        self,
        *,
        goal_handle,
        action: LedgerAction,
        key: tuple,
        payload: dict[str, object],
        deadline,
        execute,
        request_id: str = "",
    ):
        goal = goal_handle.request
        ledger = self._pipeline_ledger
        if ledger is None:
            return self._pipeline_idempotency_failure(
                goal_handle, action, goal, request_id, "pipeline_ledger_unavailable", "", InferenceOutcome.UNKNOWN
            )
        try:
            validate_uuid4(goal.session_id, field="session_id")
            if request_id:
                validate_uuid4(request_id, field="request_id")
            original_deadline_ns = int(deadline.sec) * 1_000_000_000 + int(deadline.nanosec)
            if original_deadline_ns <= 0:
                raise IdempotencyError("scheduled pipeline requires an absolute deadline from Global")
            resolution = ledger.resolve(
                action=action,
                key=key,
                payload_fingerprint=canonical_fingerprint(payload),
                effective_deadline_utc_ns=original_deadline_ns,
            )
        except (IdempotencyError, LedgerError, TypeError, ValueError) as exc:
            return self._pipeline_idempotency_failure(
                goal_handle, action, goal, request_id, "request_conflict", str(exc), InferenceOutcome.NOT_STARTED
            )
        return execute_resolved_action(
            ledger,
            resolution,
            key=key,
            goal_handle=goal_handle,
            execute=lambda _entry: execute(goal_handle),
            failure=lambda code, message, outcome: self._pipeline_idempotency_failure(
                goal_handle, action, goal, request_id, code, message, outcome
            ),
            error_codes=ResolutionErrorCodes(
                ledger_full="pipeline_ledger_full",
                ledger_error="pipeline_ledger_error",
                internal_error="pipeline_internal_error",
            ),
            not_started_outcome=InferenceOutcome.NOT_STARTED,
            unknown_outcome=InferenceOutcome.UNKNOWN,
            log_internal_error=lambda exc: self.get_logger().exception(
                f"unhandled pipeline {action.value} failure: {exc}"
            ),
        )

    def _pipeline_idempotency_failure(
        self, goal_handle, action, goal, request_id: str, code: str, message: str, outcome: int
    ):
        if action is LedgerAction.OPEN:
            result = OpenInferenceSession.Result()
            result.session_id = goal.session_id
        elif action is LedgerAction.DISPATCH:
            result = ScheduledDispatchInfer.Result()
            result.request_id = request_id
            result.session_id = goal.session_id
            result.session_generation = goal.session_generation
            result.pipeline_id = self._config.pipeline_id
        else:
            result = CloseInferenceSession.Result()
            result.session_id = goal.session_id
            result.pipeline_id = self._config.pipeline_id
        result.outcome.value = outcome
        self._set_scheduled_error(
            result.error,
            code=code,
            message=message,
            recoverable=outcome == InferenceOutcome.NOT_STARTED,
            stage="admission",
        )
        goal_handle.abort()
        return result

    def _scheduled_open_once(self, goal_handle) -> OpenInferenceSession.Result:
        """Perform atomic session admission and the reset barrier."""
        goal = goal_handle.request
        result = OpenInferenceSession.Result()
        result.session_id = goal.session_id
        ctrl = self._session_controller
        if ctrl is None:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code="session_not_configured", stage="open")
            return result
        deadline = self._scheduled_deadline(goal.deadline)
        try:
            self._raise_if_deadline_expired(deadline, goal.session_id)
        except DeadlineExceededError as exc:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code=exc.code, message=exc, recoverable=True, stage=exc.stage)
            return result
        try:
            validate_uuid4(goal.session_id, field="session_id")
        except IdempotencyError as exc:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(
                result.error, code="invalid_session_id", message=exc, recoverable=True, stage="open"
            )
            return result
        open_res = ctrl.begin_open(goal.session_id)
        if not open_res.accepted:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code=open_res.code, recoverable=True, stage="open")
            return result
        result.session_generation = open_res.fence_generation
        # Run the whole-graph reset barrier.
        try:
            reset_response = self._reset_with_deadline(Trigger.Response(), deadline)
            if not reset_response.success:
                raise RuntimeError(reset_response.message)
            ctrl.finish_open(success=True)
            self._publish_serving_status()
            result.success = True
            result.actual_pipeline_id = self._config.pipeline_id
            result.session_generation = open_res.fence_generation
            result.deployment_fingerprint = self._manifest.fingerprint
            result.runtime_policy_fingerprint = self._config.runtime_policy_fingerprint
            snapshot = ctrl.snapshot()
            lease_expires_at_ros_ns = monotonic_expiry_to_ros_ns(
                snapshot.lease_expires_at_ns,
                monotonic_now_ns=time.monotonic_ns(),
                ros_now_ns=self.get_clock().now().nanoseconds,
            )
            result.lease_expires_at.sec, result.lease_expires_at.nanosec = divmod(
                lease_expires_at_ros_ns, 1_000_000_000
            )
            result.outcome.value = InferenceOutcome.COMPLETED
            goal_handle.succeed()
        except Exception as exc:
            with contextlib.suppress(Exception):
                ctrl.finish_open(success=False)
            self._publish_serving_status()
            result.success = False
            result.outcome.value = InferenceOutcome.COMPLETED
            self._set_scheduled_error(result.error, code="session_reset_failed", message=exc, stage="open")
            goal_handle.abort()
        return result

    def _scheduled_dispatch_once(self, goal_handle) -> ScheduledDispatchInfer.Result:
        """Dispatch scheduled inference within an active session."""
        goal = goal_handle.request
        operation_acquired = False
        side_effect_started = False
        result = ScheduledDispatchInfer.Result()
        result.request_id = goal.request_id
        result.session_id = goal.session_id
        result.session_generation = goal.session_generation
        result.pipeline_id = self._config.pipeline_id
        result.deployment_fingerprint = self._manifest.fingerprint
        result.runtime_policy_fingerprint = self._config.runtime_policy_fingerprint
        ctrl = self._session_controller
        if ctrl is None:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code="session_not_configured", stage="dispatch")
            return result
        from inference_service.scheduler.work_classes import WorkClass

        deadline = self._scheduled_deadline(goal.deadline)
        try:
            self._raise_if_deadline_expired(deadline, goal.request_id)
        except DeadlineExceededError as exc:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code=exc.code, message=exc, recoverable=True, stage=exc.stage)
            return result
        if utf8_size(goal.prompt) > self._config.max_prompt_bytes:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(
                result.error,
                code="prompt_too_large",
                message=f"prompt exceeds max_prompt_bytes={self._config.max_prompt_bytes}",
                recoverable=True,
                stage="admission",
            )
            return result
        if goal.target_pipeline_id not in ("", self._config.pipeline_id) or goal.fallback_chain:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(
                result.error,
                code="invalid_pipeline_scoped_dispatch",
                recoverable=True,
                stage="admission",
            )
            return result
        if goal.priority < 0:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code="invalid_priority", recoverable=True, stage="admission")
            return result
        priority_mapping = self._require_manager().capabilities(self._config.pipeline_id).priority_mapping
        priority_levels = priority_mapping.generic_level_count if priority_mapping is not None else 1
        try:
            if priority_mapping is None:
                if goal.priority != 0:
                    raise ValueError("single-priority backend")
            else:
                priority_mapping.map_generic(goal.priority)
        except ValueError:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(
                result.error,
                code="unsupported_priority",
                message=(
                    f"backend for pipeline {self._config.pipeline_id!r} supports priorities "
                    f"[0, {priority_levels - 1}], got {goal.priority}"
                ),
                recoverable=False,
                stage="admission",
            )
            return result
        admission = ctrl.admit(
            WorkClass.ACTION_GENERATION,
            generation=goal.session_generation,
            session_id=goal.session_id,
        )
        if not admission.accepted:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(
                result.error,
                code=admission.code,
                recoverable=admission.code == "no_session_capacity",
                stage="admission",
            )
            return result
        try:
            if self._goal_cancel_requested(goal_handle):
                raise RequestCanceledError(f"scheduled dispatch {goal.request_id!r} was canceled before admission")
            if self._scheduled_operation_slots is None or not self._scheduled_operation_slots.acquire(blocking=False):
                raise PipelineBusyError(f"pipeline {self._config.pipeline_id!r} has no scheduled execution capacity")
            operation_acquired = True
            # capture observation snapshot (zero obs_timestamp -> use now)
            sample_time = goal.obs_timestamp.sec * 1_000_000_000 + goal.obs_timestamp.nanosec
            if sample_time <= 0:
                sample_time = self.get_clock().now().nanoseconds
            observations = self._sample_observations(sample_time)
            if self._goal_cancel_requested(goal_handle):
                raise RequestCanceledError(f"scheduled dispatch {goal.request_id!r} was canceled before execution")
            ctrl.record_product_activity()
            self._raise_if_deadline_expired(deadline, goal.request_id)
            if "observation.state" in observations:
                observations["observation.state"] = self._rad_to_lerobot(observations["observation.state"])
            request = InferenceRequest(
                request_id=goal.request_id,
                inputs=self._to_policy_inputs(observations),
                prompt=goal.prompt if goal.prompt else None,
                deadline=deadline,
                priority=goal.priority,
                metadata={
                    "product_session_id": goal.session_id,
                    "product_session_generation": goal.session_generation,
                },
            )
            backend_result = self._require_manager().infer(
                self._config.pipeline_id,
                request,
            )
            side_effect_started = True
            raw_action = backend_result.action
            chunk_size = backend_result.actual_chunk_size
            backend_latency_ms = backend_result.backend_latency_ms
            total_latency_ms = backend_result.total_latency_ms
            if self._goal_cancel_requested(goal_handle):
                raise RequestCanceledError(
                    f"scheduled dispatch {goal.request_id!r} was canceled before publication",
                    operation_started=side_effect_started,
                )
            if ctrl.is_stale_generation(goal.session_generation):
                raise RuntimeError("dispatch completion crossed a session generation fence")
            action = self._lerobot_to_rad(raw_action)
            result.action_chunk = TensorMsgConverter.to_variant({"action": action})
            result.chunk_size = chunk_size
            result.execution_horizon = _execution_horizon_from_metadata(backend_result.metadata, chunk_size)
            result.success = True
            result.inference_latency_ms = total_latency_ms
            result.backend_latency_ms = backend_latency_ms
            result.outcome.value = InferenceOutcome.COMPLETED
            self._last_inference_time = time.time()
            self._inference_count += 1
            self._last_error = ""
            self.get_logger().info(
                f"Inference succeeded: request={goal.request_id}, mode=scheduled, "
                f"chunk={chunk_size}, latency={total_latency_ms:.1f}ms, backend={backend_latency_ms:.1f}ms"
            )
            goal_handle.succeed()
        except Exception as exc:
            self._last_error = str(exc)
            result.success = False
            if not bool(getattr(exc, "outcome_known", True)):
                result.outcome.value = InferenceOutcome.UNKNOWN
            elif bool(getattr(exc, "operation_started", side_effect_started)):
                result.outcome.value = InferenceOutcome.COMPLETED
            else:
                result.outcome.value = InferenceOutcome.NOT_STARTED
            if result.outcome.value == InferenceOutcome.UNKNOWN:
                ctrl.mark_failed_quarantine()
            self._set_scheduled_error(
                result.error,
                code=getattr(exc, "code", "dispatch_failed"),
                message=exc,
                recoverable=(
                    result.outcome.value == InferenceOutcome.NOT_STARTED and bool(getattr(exc, "recoverable", True))
                ),
                stage=getattr(exc, "stage", "dispatch"),
                details=getattr(exc, "details", None),
            )
            if getattr(exc, "code", "") == "request_canceled":
                self._finish_canceled_goal(goal_handle)
            else:
                goal_handle.abort()
        finally:
            if operation_acquired:
                assert self._scheduled_operation_slots is not None
                self._scheduled_operation_slots.release()
            ctrl.release_in_flight(WorkClass.ACTION_GENERATION)
        return result

    def _scheduled_close_once(self, goal_handle) -> CloseInferenceSession.Result:
        """Close the session through the drain barrier."""
        goal = goal_handle.request
        result = CloseInferenceSession.Result()
        result.session_id = goal.session_id
        result.pipeline_id = self._config.pipeline_id
        ctrl = self._session_controller
        if ctrl is None:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code="session_not_configured", stage="close")
            return result
        deadline = self._scheduled_deadline(goal.deadline)
        try:
            self._raise_if_deadline_expired(deadline, goal.session_id)
        except DeadlineExceededError as exc:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code=exc.code, message=exc, recoverable=True, stage=exc.stage)
            return result
        snapshot = ctrl.snapshot()
        if snapshot.product_session_id != goal.session_id:
            goal_handle.abort()
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code="session_mismatch", recoverable=True, stage="close")
            return result
        close_res = ctrl.begin_close(generation=goal.session_generation)
        if not close_res.accepted and close_res.code == "cleanup_not_needed":
            goal_handle.succeed()
            result.success = True
            result.closed_session_generation = 0
            result.drained_generation = 0
            result.outcome.value = InferenceOutcome.COMPLETED
            return result
        if not close_res.accepted:
            goal_handle.abort()
            result.success = False
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_scheduled_error(result.error, code=close_res.code, recoverable=True, stage="close")
            return result
        result.closed_session_generation = close_res.closed_generation
        result.drained_generation = close_res.drain_generation
        # Reuse the whole-graph reset path for the Close drain barrier.
        drained_slots = 0
        try:
            drained_slots = self._acquire_scheduled_drain_slots(deadline)
            reset_response = self._reset_with_deadline(Trigger.Response(), deadline)
            if not reset_response.success:
                raise RuntimeError(reset_response.message)
            ctrl.finish_close(success=True)
            self._publish_serving_status()
            result.success = True
            result.outcome.value = InferenceOutcome.COMPLETED
            goal_handle.succeed()
        except Exception as exc:
            with contextlib.suppress(Exception):
                ctrl.finish_close(success=False)
            self._publish_serving_status()
            result.success = False
            result.outcome.value = InferenceOutcome.COMPLETED
            self._set_scheduled_error(result.error, code="close_drain_failed", message=exc, stage="close")
            goal_handle.abort()
        finally:
            self._release_scheduled_drain_slots(drained_slots)
        return result

    def _acquire_scheduled_drain_slots(self, deadline: datetime) -> int:
        slots = self._scheduled_operation_slots
        capacity = self._scheduled_operation_capacity
        if slots is None or capacity <= 0:
            raise RuntimeError("scheduled execution slots are unavailable")
        acquired = 0
        try:
            while acquired < capacity:
                remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                if remaining <= 0 or not slots.acquire(timeout=max(0.0, remaining)):
                    raise RuntimeError("scheduled Close timed out draining active operations")
                acquired += 1
            return acquired
        except Exception:
            self._release_scheduled_drain_slots(acquired)
            raise

    def _release_scheduled_drain_slots(self, count: int) -> None:
        slots = self._scheduled_operation_slots
        if slots is None:
            return
        for _ in range(count):
            slots.release()

    @staticmethod
    def _format_video_stream_diagnostic(snapshot: object) -> str:
        return json.dumps(
            {
                "stream_id": snapshot.stream_id,
                "mode": snapshot.mode,
                "configured_encoder_backend": snapshot.configured_encoder_backend,
                "selected_encoder_backend": snapshot.selected_encoder_backend,
                "configured_decoder_backend": snapshot.configured_decoder_backend,
                "selected_decoder_backend": snapshot.selected_decoder_backend,
                "endpoint": f"{snapshot.endpoint[0]}:{snapshot.endpoint[1]}",
                "contract_fingerprint": snapshot.contract_fingerprint,
                "deployment_fingerprint": snapshot.deployment_fingerprint,
                "security": snapshot.security,
                "lifecycle_state": snapshot.lifecycle_state,
                "ready": snapshot.ready,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _log_video_stream_diagnostics(self) -> None:
        manager = self._video_stream_manager
        if manager is None:
            return
        for snapshot in manager.diagnostic_snapshots():
            self.get_logger().info(
                f"Video stream startup: observation={snapshot.observation_key}, "
                f"diagnostic={self._format_video_stream_diagnostic(snapshot)}"
            )

    def _external_video_status_callback(self, message: VideoStreamStatus) -> None:
        manager = self._video_stream_manager
        if manager is None or not isinstance(manager, _ExternalVideoProducerView):
            return
        try:
            manager.observe_sender_status(video_status_from_message(message))
        except Exception as exc:
            self._last_error = f"invalid external video status: {exc}"
            self.get_logger().error(self._last_error)

    def _cloud_status_callback(self, message: InferencePipelineStatus) -> None:
        if message.role != InferencePipelineStatus.ROLE_CLOUD:
            return
        try:
            status = status_from_message(message)
            self._last_cloud_status_received_monotonic = time.monotonic()
            was_ready = self._require_edge_session().ready
            update = self._require_edge_session().observe_cloud(status)
            self._remote_state = status.runtime_state
            if not status.ready:
                self.get_logger().warning(
                    "Cloud pipeline discovered but is not ready: "
                    f"runtime_state={status.runtime_state!r}; waiting for cloud backend readiness "
                    "before binding the distributed session",
                    throttle_duration_sec=_CLOUD_HANDSHAKE_WARNING_THROTTLE_S,
                )
            self._complete_invalidated(update.invalidated_request_ids, update.error)
            video_manager = self._video_stream_manager
            if video_manager is not None:
                session_id, session_generation = self._require_edge_session().session
                if session_id:
                    session = StreamSessionView(
                        pipeline_id=self._config.pipeline_id,
                        session_id=session_id,
                        generation=session_generation,
                        contract_fingerprint=contract_fingerprint(self._contract),
                        deployment_fingerprint=self._manifest.fingerprint,
                    )
                    if video_manager.bind_session(session):
                        self._publish_video_stream_control()
                else:
                    video_manager.clear_session()
            if not was_ready and self._require_edge_session().ready:
                session_id, session_generation = self._require_edge_session().session
                self.get_logger().info(
                    f"Cloud handshake established: session_id={session_id}, session_generation={session_generation}"
                )
        except Exception as exc:
            self._last_error = f"invalid cloud status: {exc}"
            self.get_logger().error(self._last_error)

    def _publish_distributed_status(self) -> None:
        try:
            message = status_to_message(
                self._require_edge_session().local_status(),
                stamp=self.get_clock().now().to_msg(),
            )
            self._status_pub.publish(message)
        except Exception as exc:
            self._last_error = f"failed to publish distributed status: {exc}"

    def _check_heartbeat(self) -> None:
        session = self._require_edge_session()
        if not session.ready and self._last_cloud_status_received_monotonic is None:
            elapsed = time.monotonic() - self._distributed_started_monotonic
            if elapsed >= _CLOUD_HANDSHAKE_WARNING_DELAY_S:
                self.get_logger().warning(
                    "Distributed pipeline is running locally but no cloud heartbeat has been received "
                    f"for {elapsed:.1f}s; start pure_inference_node and verify the heartbeat topic "
                    f"{self._config.heartbeat_topic!r} and ROS 2 discovery settings",
                    throttle_duration_sec=_CLOUD_HANDSHAKE_WARNING_THROTTLE_S,
                )
        update = session.expire_heartbeat()
        self._complete_invalidated(update.invalidated_request_ids, update.error)
        if update.error is not None:
            if self._video_stream_manager is not None:
                self._video_stream_manager.clear_session()
            if update.error.code == "heartbeat_expired":
                self.get_logger().warning(
                    "Cloud heartbeat expired; distributed session was cleared and is waiting for handshake recovery"
                )

    def _publish_video_stream_control(self) -> None:
        manager = self._video_stream_manager
        descriptor_publisher = getattr(self, "_video_descriptor_pub", None)
        status_publisher = getattr(self, "_video_status_pub", None)
        if manager is None or descriptor_publisher is None or status_publisher is None:
            return
        stamp = self.get_clock().now().to_msg()
        try:
            for descriptor in manager.descriptors():
                descriptor_publisher.publish(video_descriptor_to_message(descriptor, stamp=stamp))
            for status in manager.statuses():
                status_publisher.publish(video_status_to_message(status, stamp=stamp))
            diagnostics = manager.sender_diagnostics()
            if diagnostics:
                self.get_logger().info(
                    "Video sender runtime: " + json.dumps(diagnostics, sort_keys=True, separators=(",", ":")),
                    throttle_duration_sec=2.0,
                )
        except Exception as exc:
            self._last_error = f"failed to publish video stream control status: {exc}"
            self.get_logger().error(self._last_error, throttle_duration_sec=1.0)

    def _distributed_result_callback(self, message: DistributedInferenceResult) -> None:
        try:
            result = result_from_message(message)
            update = self._require_edge_session().accept_result(result)
        except Exception as exc:
            with self._pending_lock:
                pending = self._pending.get(message.request_id)
                matches_pending = pending is not None and self._matches_pending_envelope(message, pending)
            if not matches_pending:
                self.get_logger().warning(
                    f"discarded malformed stale or mismatched response {message.request_id!r}: {exc}"
                )
                return
            error = StructuredError(
                code="decode_failed",
                message=str(exc) or type(exc).__name__,
                stage="decode",
            )
            with self._pending_lock:
                pending = self._pending.get(message.request_id)
                if pending is not None:
                    pending.error = error
                    pending.event.set()
            self.get_logger().error(f"invalid distributed result: {exc}")
            return
        if update.error is not None and update.error.code == "stale_response":
            self.get_logger().warning(update.error.message)
            return
        with self._pending_lock:
            pending = self._pending.get(result.request_id)
            if pending is not None:
                if update.error is not None:
                    pending.error = update.error
                else:
                    pending.result = result
                    pending.error = result.error
                pending.event.set()
            if update.canceled_request_id:
                canceled = self._pending.get(update.canceled_request_id)
                if canceled is not None:
                    canceled.error = StructuredError(
                        code="request_canceled",
                        message=f"distributed request {update.canceled_request_id!r} was canceled",
                        stage="cancel",
                    )
                    canceled.event.set()
        self._complete_invalidated(update.invalidated_request_ids, update.error)

    def _round_trip(
        self,
        operation: Operation,
        request_id: str,
        *,
        inputs: dict[str, object] | None = None,
        prompt: str | None = None,
        deadline: datetime | None = None,
        target_request_id: str = "",
        goal_handle: object | None = None,
        progress: _RoundTripProgress | None = None,
        observation_timestamp_ns: int = 0,
        stream_references: tuple[StreamReference, ...] = (),
    ) -> DistributedResult:
        pending = _PendingOperation(event=threading.Event(), operation=operation)
        with self._pending_lock:
            if request_id in self._pending:
                raise RuntimeError(f"duplicate pending distributed request {request_id!r}")
            self._pending[request_id] = pending

        cancel_sent = False
        try:

            def publish_request(current) -> None:
                with self._pending_lock:
                    pending.session_id = current.session_id
                    pending.session_generation = current.session_generation
                    pending.deployment_fingerprint = current.deployment_fingerprint
                self._request_pub.publish(request_to_message(current))
                if progress is not None:
                    progress.published = True

            request = self._require_edge_session().dispatch_request(
                operation,
                request_id,
                publish_request,
                inputs=inputs,
                prompt=prompt,
                deadline=deadline,
                target_request_id=target_request_id,
                observation_timestamp_ns=observation_timestamp_ns,
                stream_references=stream_references,
            )
            while not pending.event.wait(timeout=0.05):
                if request.deadline is not None and datetime.now(timezone.utc) >= request.deadline:
                    raise DistributedProtocolError(
                        StructuredError(
                            code="deadline_exceeded",
                            message=f"distributed request {request.request_id!r} timed out",
                            stage="transport",
                        )
                    )
                if goal_handle is not None and self._goal_cancel_requested(goal_handle) and not cancel_sent:
                    cancel_sent = True
                    try:
                        self._route_cancel(request.request_id)
                    except Exception as exc:
                        error = StructuredError(
                            code="cancellation_outcome_unknown",
                            message=f"remote cancellation outcome is unknown: {exc}",
                            stage="cancel",
                            details={"request_id": request.request_id},
                        )
                        update = self._require_edge_session().fail(error)
                        self._complete_invalidated(update.invalidated_request_ids, update.error)
                        raise DistributedProtocolError(error) from exc
                    self._mark_goal_cancel_confirmed(goal_handle)
            if request.deadline is not None and datetime.now(timezone.utc) >= request.deadline:
                raise DistributedProtocolError(
                    StructuredError(
                        code="deadline_exceeded",
                        message=f"distributed request {request.request_id!r} timed out",
                        stage="transport",
                    )
                )
            if pending.result is not None and progress is not None:
                progress.response_received = True
                progress.response_success = pending.result.success
                progress.backend_ready = pending.result.backend_ready
            if pending.error is not None:
                raise DistributedProtocolError(pending.error)
            if pending.result is None:
                raise RuntimeError(f"distributed request {request.request_id!r} completed without a result")
            if not pending.result.success:
                assert pending.result.error is not None
                raise DistributedProtocolError(pending.result.error)
            return pending.result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._require_edge_session().abandon_request(request_id)

    def _matches_pending_envelope(
        self,
        message: DistributedInferenceResult,
        pending: _PendingOperation,
    ) -> bool:
        return (
            message.pipeline_id == self._config.pipeline_id
            and message.session_id == pending.session_id
            and message.session_generation == pending.session_generation
            and message.deployment_fingerprint == pending.deployment_fingerprint
            and message.operation in {int(pending.operation), int(Operation.UNKNOWN)}
        )

    def _route_cancel(self, target_request_id: str) -> None:
        session = self._require_edge_session()
        if not session.cancellation_supported:
            raise DistributedProtocolError(
                StructuredError(
                    code="unsupported_capability",
                    message="cloud backend does not support cancellation",
                    stage="cancel",
                )
            )
        request_id = f"{target_request_id}-cancel-{time.monotonic_ns()}"
        self._round_trip(
            Operation.CANCEL,
            request_id,
            target_request_id=target_request_id,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=self._config.request_timeout),
        )

    def _scheduled_cancel_callback(self, goal_handle: object):
        request = goal_handle.request
        request_id = getattr(request, "request_id", "")
        if not request_id:
            return rclpy.action.CancelResponse.REJECT
        manager = self._require_manager()
        if not manager.capabilities(self._config.pipeline_id).supports_cancellation:
            return rclpy.action.CancelResponse.REJECT
        with contextlib.suppress(Exception):
            manager.cancel(
                self._config.pipeline_id,
                request_id,
                datetime.now(timezone.utc) + timedelta(seconds=self._config.request_timeout),
            )
        # Before admission there is no runtime work item to signal. The execute
        # callback still observes is_cancel_requested before starting work.
        return rclpy.action.CancelResponse.ACCEPT

    def _cancel_callback(self, goal_handle: object):
        if self._config.execution_mode == "distributed" and self._require_edge_session().cancellation_supported:
            with self._goal_state_lock:
                if id(goal_handle) in self._completed_goals:
                    return rclpy.action.CancelResponse.REJECT
                self._cancel_requested_goals.add(id(goal_handle))
                return rclpy.action.CancelResponse.ACCEPT
        return rclpy.action.CancelResponse.REJECT

    def _complete_invalidated(
        self,
        request_ids: tuple[str, ...],
        error: StructuredError | None,
    ) -> None:
        if not request_ids:
            return
        failure = error or StructuredError(
            code="session_invalidated",
            message="distributed session was invalidated",
            stage="transport",
            recoverable=True,
        )
        with self._pending_lock:
            for request_id in request_ids:
                pending = self._pending.get(request_id)
                if pending is None:
                    continue
                pending.error = failure
                pending.event.set()

    def _goal_deadline(self, deadline_message: object) -> datetime | None:
        seconds = int(deadline_message.sec)
        nanoseconds = int(deadline_message.nanosec)
        if seconds == 0 and nanoseconds == 0:
            return None
        return datetime.fromtimestamp(seconds + nanoseconds / 1_000_000_000, tz=timezone.utc)

    @staticmethod
    def _raise_if_deadline_expired(deadline: datetime, request_id: str) -> None:
        if datetime.now(timezone.utc) >= deadline:
            raise DeadlineExceededError(f"inference request {request_id!r} exceeded its deadline")

    def _fail_distributed_after_deadline(self, request_id: str, exc: Exception) -> None:
        if self._config.execution_mode != "distributed":
            return
        error = (
            exc.error if isinstance(exc, DistributedProtocolError) else structured_error_from_exception(exc, "deadline")
        )
        if error.code != "deadline_exceeded":
            return
        failure = StructuredError(
            code="deadline_exceeded",
            message=f"distributed inference {request_id!r} exceeded its deadline after edge processing began",
            stage="deadline",
            details={"request_id": request_id, "cause": error.message},
        )
        update = self._require_edge_session().fail(failure)
        self._complete_invalidated(update.invalidated_request_ids, update.error)

    def _require_manager(self) -> InferencePipelineManager:
        if self._manager is None:
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} manager is closed")
        return self._manager

    def _require_edge_runtime(self) -> EdgeProcessorRuntime:
        if self._edge_runtime is None:
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} edge runtime is closed")
        return self._edge_runtime

    def _require_edge_session(self) -> EdgeSession:
        if self._edge_session is None:
            raise RuntimeError(f"pipeline {self._config.pipeline_id!r} distributed session is closed")
        return self._edge_session

    def destroy_node(self) -> None:
        video_stream_manager = self._video_stream_manager
        self._video_stream_manager = None
        if video_stream_manager is not None:
            try:
                video_stream_manager.close()
            except Exception as exc:
                self.get_logger().error(f"video stream shutdown failed: {exc}")
        manager = self._manager
        self._manager = None
        if manager is not None:
            try:
                manager.close()
            except Exception as exc:
                self.get_logger().error(f"pipeline shutdown failed: {exc}")
        edge_session = self._edge_session
        self._edge_session = None
        if edge_session is not None:
            pending = edge_session.close()
            self._complete_invalidated(pending, None)
        edge_runtime = self._edge_runtime
        self._edge_runtime = None
        if edge_runtime is not None:
            edge_runtime.close()
        super().destroy_node()


def _read_config() -> tuple[PipelineNodeConfig, str]:
    reader = Node("_pipeline_policy_param_reader")
    defaults: dict[str, object] = {
        "pipeline_id": "policy",
        "model_path": "",
        "deployment": "cpu",
        "execution_mode": "monolithic",
        "request_timeout": 5.0,
        "default_task": "",
        "runtime_options_json": "{}",
        "robot_config_path": "",
        "use_sim": False,
        "node_name": "inference_policy",
        "action_server": "/inference/policy/dispatch",
        "reset_service": "/inference/policy/reset",
        "health_topic": "/inference/policy/health",
        "action_topic": "/actions/policy",
        "request_topic": "",
        "result_topic": "",
        "heartbeat_topic": "",
        "scheduled_open_session": "",
        "scheduled_dispatch": "",
        "scheduled_close_session": "",
        "scheduled_serving_status": "",
        "runtime_policy_json": "",
        "runtime_policy_fingerprint": "",
        "hardware_resource_id": "",
        "session_idle_timeout_ns": 0,
        "max_prompt_bytes": 4096,
        "max_error_message_bytes": 1024,
        "max_error_details_bytes": 8192,
        "public_capacity_json": "",
        "max_session_records": 1,
        "terminal_result_cache_entries": 1,
        "max_duplicate_waiters_per_request": 1,
        "terminal_session_retention_ns": 1,
        "video_descriptor_topic": "/inference/policy/video/descriptors",
        "video_status_topic": "/inference/policy/video/status",
        "external_video_producer": False,
    }
    for name, default in defaults.items():
        reader.declare_parameter(name, default)
    values = {name: reader.get_parameter(name).value for name in defaults}
    reader.destroy_node()
    node_name = str(values.pop("node_name"))
    return PipelineNodeConfig(**values), node_name


def _pipeline_executor_threads(config: PipelineNodeConfig) -> int:
    if not config.scheduler_enabled:
        return 4
    try:
        capacity = json.loads(config.public_capacity_json)["action_generation"]["max_in_flight"]
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        capacity = 1
    return max(8, int(capacity) + 4)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PipelinePolicyNode | None = None
    dependencies = None
    try:
        config, node_name = _read_config()
        dependencies = build_policy_runtime_dependencies()
        node = PipelinePolicyNode(
            config,
            node_name=node_name,
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
        executor = MultiThreadedExecutor(num_threads=_pipeline_executor_threads(config))
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if dependencies is not None:
            dependencies.providers.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
