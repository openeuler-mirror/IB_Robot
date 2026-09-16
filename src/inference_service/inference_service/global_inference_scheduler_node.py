"""ROS wrapper for the global inference session scheduler."""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass

import rclpy
import rclpy.action
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from ibrobot_msgs.action import (
    CloseInferenceSession,
    ClosePipelineBinding,
    DispatchPipelineBinding,
    OpenInferenceSession,
    OpenPipelineBinding,
    ScheduledDispatchInfer,
)
from ibrobot_msgs.msg import InferenceOutcome, InferenceServingStatus
from inference_service.runtime_composition import (
    build_model_service_runtime_dependencies,
    require_runtime_dependencies,
)
from inference_service.scheduler.action_idempotency import (
    ResolutionErrorCodes,
    execute_resolved_action,
)
from inference_service.scheduler.deadline_reservations import DeadlineReservation, DeadlineReservationTable
from inference_service.scheduler.global_scheduler_core import (
    BindingDecision,
    GlobalSchedulerCore,
    GlobalSessionState,
    PipelineCandidate,
    SchedulerError,
)
from inference_service.scheduler.goal_slots import GoalSlotPool
from inference_service.scheduler.idempotency import (
    IdempotencyError,
    canonical_fingerprint,
    resolve_entry_deadline_ns,
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
from inference_service.scheduler.operations import (
    Certainty,
    OperationIdentity,
    OperationKind,
    OperationRegistry,
)
from inference_service.scheduler.profiles import ProfileError, ProfileRegistry
from inference_service.scheduler.result_identity import result_identity_error
from inference_service.scheduler.time_domains import monotonic_expiry_to_ros_ns
from inference_service.scheduler.wire_bounds import set_scheduled_error, utf8_size
from inference_service.scheduler.work_classes import WorkClass, work_class_name
from inference_service.unified_runtime import RegistrySet, RuntimeProviders

_SESSION_OPEN_CLOSURE = "session_open"
_FULL_INFER_CLOSURE = "full_infer"


@dataclass
class _ObservedStatus:
    message: InferenceServingStatus
    received_monotonic_ns: int
    invalid_reason: str = ""


@dataclass
class _DownstreamCall:
    certainty: str
    result: object | None = None
    reason: str = ""


class _MaintenanceGoalHandle:
    """Stand-in goal handle for timer-driven Close sweeps."""

    is_cancel_requested = False

    def __init__(self, request):
        self.request = request

    def succeed(self):
        return None

    def abort(self):
        return None

    def canceled(self):
        return None


class GlobalInferenceSchedulerNode(Node):
    """Own logical sessions, per-request routing, and scheduled public endpoints."""

    def __init__(
        self,
        *,
        parameter_overrides=None,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        super().__init__("global_inference_scheduler", parameter_overrides=parameter_overrides)
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        # This node is a transport/control-plane proxy. It never resolves a
        # Session or runtime itself; downstream pipeline nodes own construction.
        del registry_set, providers
        self._load_parameters()
        self._candidates = self._build_candidates()
        self._candidate_by_id = {candidate.pipeline_id: candidate for candidate in self._candidates}
        self._core = GlobalSchedulerCore(
            candidates=self._candidates,
            max_session_records=self._max_session_records,
            max_product_requests_per_session=self._max_product_requests,
            terminal_session_retention_ns=self._terminal_retention_ns,
            session_idle_timeout_ns=self._session_idle_timeout_ns,
            max_fallback_pipelines=self._max_fallback_pipelines,
            now_ns=time.monotonic_ns,
        )
        self._ingress_ledger = IdempotencyLedger(
            max_session_records=self._max_session_records,
            max_duplicate_waiters_per_request=self._max_duplicate_waiters,
            terminal_session_retention_ns=self._terminal_retention_ns,
            now_ns=time.monotonic_ns,
            max_entries=self._max_session_records * (self._terminal_result_cache_entries + 4),
            max_terminal_entries_per_session=self._terminal_result_cache_entries,
        )
        self._status_lock = threading.RLock()
        self._serving_status: dict[str, _ObservedStatus] = {}
        self._trusted_status_cursors: dict[str, tuple[str, int]] = {}
        self._close_retry_exhausted: set[str] = set()
        self._deadline_reservations = DeadlineReservationTable(policy=self._global_policy)
        self._downstream_operations = OperationRegistry(
            max_records=self._max_session_records * (self._terminal_result_cache_entries + 4),
            max_waiters_per_operation=self._max_duplicate_waiters,
        )
        self._client_group = ReentrantCallbackGroup()
        self._pipeline_clients = self._create_pipeline_clients()
        self._profile_registries, self._profile_errors = self._load_profile_registries()
        self._subscribe_serving_status()

        self._readiness_srv = self.create_service(
            Trigger,
            self._readiness_endpoint,
            self._readiness_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        server_group = ReentrantCallbackGroup()
        self._lifecycle_goal_slots = GoalSlotPool(("open", "close"))
        # Keep dispatch bounded while preventing lower-priority traffic from
        # consuming the configured priority-0 reserve.
        self._dispatch_goal_slots = GoalSlotPool(
            ("dispatch",),
            capacity=self._dispatch_goal_contexts,
            protected_capacity=(self._dispatch_goal_contexts - self._lower_priority_dispatch_goal_contexts),
        )
        self._open_server = rclpy.action.ActionServer(
            self,
            OpenInferenceSession,
            self._open_endpoint_name,
            execute_callback=lambda goal_handle: self._lifecycle_goal_slots.run(
                "open", self._open_endpoint, goal_handle
            ),
            goal_callback=lambda _request: (
                rclpy.action.GoalResponse.ACCEPT
                if self._lifecycle_goal_slots.try_acquire("open")
                else rclpy.action.GoalResponse.REJECT
            ),
            callback_group=server_group,
        )
        self._dispatch_server = rclpy.action.ActionServer(
            self,
            ScheduledDispatchInfer,
            self._dispatch_endpoint_name,
            execute_callback=lambda goal_handle: self._dispatch_goal_slots.run(
                "dispatch",
                self._dispatch_endpoint,
                goal_handle,
                protected=goal_handle.request.priority == 0,
            ),
            goal_callback=lambda request: (
                rclpy.action.GoalResponse.ACCEPT
                if self._dispatch_goal_slots.try_acquire("dispatch", protected=request.priority == 0)
                else rclpy.action.GoalResponse.REJECT
            ),
            cancel_callback=self._cancel_scheduled_endpoint,
            callback_group=server_group,
        )
        self._close_server = rclpy.action.ActionServer(
            self,
            CloseInferenceSession,
            self._close_endpoint_name,
            execute_callback=lambda goal_handle: self._lifecycle_goal_slots.run(
                "close", self._close_endpoint, goal_handle
            ),
            goal_callback=lambda _request: (
                rclpy.action.GoalResponse.ACCEPT
                if self._lifecycle_goal_slots.try_acquire("close")
                else rclpy.action.GoalResponse.REJECT
            ),
            callback_group=server_group,
        )
        self._idle_timer = self.create_timer(1.0, self._idle_sweep, callback_group=ReentrantCallbackGroup())
        self.get_logger().info("GlobalInferenceScheduler started; waiting for verified serving status")

    def _load_parameters(self) -> None:
        string_parameters = (
            ("readiness_endpoint", "/inference/scheduler/ready"),
            ("open_session_endpoint", "/inference/session/open"),
            ("dispatch_endpoint", "/inference/dispatch"),
            ("close_session_endpoint", "/inference/session/close"),
            ("default_target_pipeline_id", ""),
            ("pipelines_json", "[]"),
            ("global_policy", "fifo"),
        )
        for name, default in string_parameters:
            self.declare_parameter(name, default)
        integer_parameters = (
            "default_open_timeout_ns",
            "default_request_timeout_ns",
            "status_stale_timeout_ns",
            "clock_skew_tolerance_ns",
            "goal_acceptance_timeout_ns",
            "session_idle_timeout_ns",
            "terminal_session_retention_ns",
            "max_duplicate_waiters_per_request",
            "max_product_requests_per_session",
            "terminal_result_cache_entries",
            "max_session_records",
            "max_fallback_pipelines",
            "profile_min_samples",
            "profile_max_age_days",
            "goal_acceptance_safety_margin_ms",
            "dispatch_safety_margin_ms",
            "max_prompt_bytes",
            "max_error_message_bytes",
            "max_error_details_bytes",
            "default_priority",
        )
        for name in integer_parameters:
            self.declare_parameter(name, 0)
        self.declare_parameter("dispatch_goal_contexts", 4)
        self.declare_parameter("lower_priority_dispatch_goal_contexts", 2)
        self.declare_parameter("priority_zero_deadline_admission_enabled", False)
        self.declare_parameter("priority_zero_deadline_admission_safety_margin_ms", 5)
        self._readiness_endpoint = str(self.get_parameter("readiness_endpoint").value)
        self._open_endpoint_name = str(self.get_parameter("open_session_endpoint").value)
        self._dispatch_endpoint_name = str(self.get_parameter("dispatch_endpoint").value)
        self._close_endpoint_name = str(self.get_parameter("close_session_endpoint").value)
        self._default_target_pipeline_id = str(self.get_parameter("default_target_pipeline_id").value)
        self._pipelines_json = str(self.get_parameter("pipelines_json").value)
        self._global_policy = str(self.get_parameter("global_policy").value)
        self._priority_zero_deadline_admission_enabled = bool(
            self.get_parameter("priority_zero_deadline_admission_enabled").value
        )
        self._priority_zero_deadline_admission_safety_margin_ms = int(
            self.get_parameter("priority_zero_deadline_admission_safety_margin_ms").value
        )
        self._default_open_timeout_ns = int(self.get_parameter("default_open_timeout_ns").value)
        self._default_request_timeout_ns = int(self.get_parameter("default_request_timeout_ns").value)
        self._status_stale_timeout_ns = int(self.get_parameter("status_stale_timeout_ns").value)
        self._clock_skew_tolerance_ns = int(self.get_parameter("clock_skew_tolerance_ns").value)
        self._goal_acceptance_timeout_ns = int(self.get_parameter("goal_acceptance_timeout_ns").value)
        self._session_idle_timeout_ns = int(self.get_parameter("session_idle_timeout_ns").value)
        self._terminal_retention_ns = int(self.get_parameter("terminal_session_retention_ns").value)
        self._max_duplicate_waiters = int(self.get_parameter("max_duplicate_waiters_per_request").value)
        self._max_product_requests = int(self.get_parameter("max_product_requests_per_session").value)
        self._max_session_records = int(self.get_parameter("max_session_records").value)
        self._terminal_result_cache_entries = int(self.get_parameter("terminal_result_cache_entries").value)
        self._max_fallback_pipelines = int(self.get_parameter("max_fallback_pipelines").value)
        self._dispatch_safety_margin_ms = int(self.get_parameter("dispatch_safety_margin_ms").value)
        self._dispatch_goal_contexts = int(self.get_parameter("dispatch_goal_contexts").value)
        self._lower_priority_dispatch_goal_contexts = int(
            self.get_parameter("lower_priority_dispatch_goal_contexts").value
        )
        self._profile_min_samples = int(self.get_parameter("profile_min_samples").value)
        self._profile_max_age_days = int(self.get_parameter("profile_max_age_days").value)
        self._goal_acceptance_safety_margin_ms = int(self.get_parameter("goal_acceptance_safety_margin_ms").value)
        self._max_prompt_bytes = int(self.get_parameter("max_prompt_bytes").value)
        self._max_error_message_bytes = int(self.get_parameter("max_error_message_bytes").value)
        self._max_error_details_bytes = int(self.get_parameter("max_error_details_bytes").value)
        self._default_priority = int(self.get_parameter("default_priority").value)

    def _set_error(
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
            max_message_bytes=self._max_error_message_bytes,
            max_details_bytes=self._max_error_details_bytes,
        )

    def _build_candidates(self) -> list[PipelineCandidate]:
        candidates: list[PipelineCandidate] = []
        for entry in json.loads(self._pipelines_json):
            interface = str(entry.get("interface", "policy")).strip()
            if interface != "policy":
                raise SchedulerError(
                    "global scheduler accepts policy pipelines only",
                    code="distributed_tensor_model_unsupported",
                )
            capacities = {name: int(value["max_in_flight"]) for name, value in entry.get("public_capacity", {}).items()}
            candidates.append(
                PipelineCandidate(
                    pipeline_id=entry["pipeline_id"],
                    compatibility_group=entry.get("compatibility_group", ""),
                    hardware_resource_id=entry.get("hardware_resource_id", ""),
                    hardware_profile_fingerprint=entry.get("hardware_profile_fingerprint", ""),
                    deployment_fingerprint=entry.get("deployment_fingerprint", ""),
                    runtime_policy_fingerprint=entry.get("runtime_policy_fingerprint", ""),
                    profile_compatibility_fingerprint=entry.get("profile_compatibility_fingerprint", ""),
                    endpoint_open=entry.get("open_session", ""),
                    endpoint_dispatch=entry.get("dispatch", ""),
                    endpoint_close=entry.get("close_session", ""),
                    endpoint_serving_status=entry.get("serving_status", ""),
                    profile_path=entry.get("profile_path", ""),
                    required=bool(entry.get("required", True)),
                    public_capacity=capacities,
                )
            )
        return candidates

    def _create_pipeline_clients(self) -> dict[str, dict[str, object]]:
        return {
            candidate.pipeline_id: {
                "open": rclpy.action.ActionClient(
                    self, OpenPipelineBinding, candidate.endpoint_open, callback_group=self._client_group
                ),
                "dispatch": rclpy.action.ActionClient(
                    self, DispatchPipelineBinding, candidate.endpoint_dispatch, callback_group=self._client_group
                ),
                "close": rclpy.action.ActionClient(
                    self, ClosePipelineBinding, candidate.endpoint_close, callback_group=self._client_group
                ),
            }
            for candidate in self._candidates
        }

    def _load_profile_registries(self) -> tuple[dict[str, ProfileRegistry], dict[str, str]]:
        registries: dict[str, ProfileRegistry] = {}
        errors: dict[str, str] = {}
        for candidate in self._candidates:
            if not candidate.profile_path:
                errors[candidate.pipeline_id] = "priority_zero_profile_unavailable:profile_path_missing"
                continue
            registry = ProfileRegistry(
                profile_path=candidate.profile_path,
                profile_min_samples=self._profile_min_samples,
                profile_max_age_days=self._profile_max_age_days,
                deployment_fingerprint=candidate.deployment_fingerprint,
                hardware_fingerprint=candidate.hardware_profile_fingerprint,
                profile_compatibility_fingerprint=candidate.profile_compatibility_fingerprint,
                now_ns=time.time_ns,
            )
            try:
                registry.load()
            except (OSError, TypeError, ValueError, ProfileError) as exc:
                errors[candidate.pipeline_id] = str(exc)
            registries[candidate.pipeline_id] = registry
        return registries, errors

    def _subscribe_serving_status(self) -> None:
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for candidate in self._candidates:
            self.create_subscription(
                InferenceServingStatus,
                candidate.endpoint_serving_status,
                self._make_status_callback(candidate.pipeline_id),
                qos,
                callback_group=ReentrantCallbackGroup(),
            )

    def _make_status_callback(self, pipeline_id: str):
        def _callback(message: InferenceServingStatus) -> None:
            with self._status_lock:
                candidate = self._candidate_by_id[pipeline_id]
                reason = self._status_message_reason(candidate, message)
                previous_cursor = self._trusted_status_cursors.get(pipeline_id)
                if (
                    not reason
                    and previous_cursor is not None
                    and message.boot_id == previous_cursor[0]
                    and message.sequence <= previous_cursor[1]
                ):
                    reason = "non_monotonic_sequence"
                self._serving_status[pipeline_id] = _ObservedStatus(message, time.monotonic_ns(), reason)
                if reason:
                    return
                if previous_cursor is not None and message.boot_id != previous_cursor[0]:
                    self._core.reconcile_pipeline_boot(pipeline_id, boot_id=message.boot_id)
                    self._deadline_reservations.reconcile_pipeline(pipeline_id)
                    self._downstream_operations.fence_boot(previous_cursor[0])
                self._trusted_status_cursors[pipeline_id] = (message.boot_id, int(message.sequence))

        return _callback

    def _status_message_reason(
        self,
        candidate: PipelineCandidate,
        message: InferenceServingStatus,
        *,
        require_active: bool = False,
        required_priority: int | None = None,
    ) -> str:
        if message.pipeline_id != candidate.pipeline_id:
            return "pipeline_id_mismatch"
        try:
            validate_uuid4(message.boot_id, field="boot_id")
        except IdempotencyError:
            return "invalid_boot_or_sequence"
        if message.sequence < 1:
            return "invalid_boot_or_sequence"
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        now_ros_ns = self.get_clock().now().nanoseconds
        if stamp_ns <= 0 or stamp_ns > now_ros_ns + self._clock_skew_tolerance_ns:
            return "invalid_status_timestamp"
        if now_ros_ns - stamp_ns > self._status_stale_timeout_ns + self._clock_skew_tolerance_ns:
            return "stale_status_timestamp"
        allowed_states = (
            {InferenceServingStatus.ACTIVE}
            if require_active
            else {
                InferenceServingStatus.IDLE,
                InferenceServingStatus.ACTIVE,
            }
        )
        if message.state not in allowed_states:
            return f"state_{message.state}"
        if message.error.code:
            return f"pipeline_error:{message.error.code}"
        if message.deployment_fingerprint != candidate.deployment_fingerprint:
            return "deployment_fingerprint_mismatch"
        if message.runtime_policy_fingerprint != candidate.runtime_policy_fingerprint:
            return "runtime_policy_fingerprint_mismatch"
        if message.configured_hardware_resource_id != candidate.hardware_resource_id:
            return "configured_hardware_resource_mismatch"
        if message.runtime_hardware_resource_id != candidate.hardware_resource_id:
            return "runtime_hardware_resource_mismatch"
        if message.hardware_priority_levels not in {1, 8}:
            return "hardware_priority_levels_invalid"
        schema_version = int(message.scheduling_capability_schema_version)
        if schema_version != 1:
            return "scheduling_capability_schema_mismatch"
        if message.stage_policy not in {"SEQUENTIAL", "INDEPENDENT"}:
            return "stage_policy_invalid"
        if message.stage_policy == "INDEPENDENT" and message.supports_priority_zero_deadline_admission:
            # Independent stages cannot honor deadline-driven priority-0
            # ordering; a publisher claiming both is inconsistent.
            return "priority_zero_deadline_admission_capability_invalid"
        if message.max_supported_public_priority < 0 or message.max_supported_public_priority > 7:
            return "max_supported_public_priority_invalid"
        if required_priority is not None and required_priority >= message.hardware_priority_levels:
            return "unsupported_default_priority"
        if required_priority is not None and required_priority > message.max_supported_public_priority:
            return "unsupported_public_priority"
        expected = candidate.public_capacity or {}
        actual: dict[str, int] = {}
        for capacity in message.capacities:
            try:
                name = work_class_name(int(capacity.work_class))
            except ValueError:
                return "invalid_capacity"
            if name in actual:
                return "invalid_capacity"
            actual[name] = int(capacity.max_in_flight)
        if actual != expected:
            return "capacity_mismatch"
        return ""

    def _status_reason(
        self,
        candidate: PipelineCandidate,
        *,
        require_active: bool = False,
        required_priority: int | None = None,
    ) -> str:
        with self._status_lock:
            observed = self._serving_status.get(candidate.pipeline_id)
            if observed is None:
                return "missing"
            message = observed.message
            if observed.invalid_reason:
                return observed.invalid_reason
            if time.monotonic_ns() - observed.received_monotonic_ns > self._status_stale_timeout_ns:
                return "stale"
            return self._status_message_reason(
                candidate,
                message,
                require_active=require_active,
                required_priority=required_priority,
            )

    def _readiness_callback(self, _request, response: Trigger.Response) -> Trigger.Response:
        failures = {
            candidate.pipeline_id: reason
            for candidate in self._candidates
            if candidate.required and (reason := self._status_reason(candidate))
        }
        if self._default_priority > 0:
            target = self._candidate_by_id.get(self._default_target_pipeline_id)
            if target is None:
                failures[self._default_target_pipeline_id or "default_target"] = "unknown_default_target_pipeline"
            elif reason := self._status_reason(target, required_priority=self._default_priority):
                failures[target.pipeline_id] = reason
        compatibility: dict[str, str] = {}
        if not failures:
            with self._status_lock:
                for candidate in self._candidates:
                    if not candidate.required:
                        continue
                    fingerprint = self._serving_status[candidate.pipeline_id].message.pipeline_compatibility_fingerprint
                    previous = compatibility.setdefault(candidate.compatibility_group, fingerprint)
                    if not fingerprint or previous != fingerprint:
                        failures[candidate.pipeline_id] = "pipeline_compatibility_mismatch"
        response.success = not failures
        response.message = "" if response.success else json.dumps(failures, sort_keys=True)
        return response

    def _compatibility_reason(self, target: PipelineCandidate, candidate: PipelineCandidate) -> str:
        if target.compatibility_group != candidate.compatibility_group:
            return "pipeline_compatibility_mismatch"
        if target.pipeline_id == candidate.pipeline_id:
            return ""
        if self._status_reason(target):
            # The configured compatibility group is the static substitution
            # boundary. A healthy fallback may serve while the primary is
            # unavailable; its own status/profile checks remain mandatory.
            return ""
        with self._status_lock:
            target_status = self._serving_status.get(target.pipeline_id)
            candidate_status = self._serving_status.get(candidate.pipeline_id)
            if candidate_status is None:
                return "missing_compatibility_status"
            expected = target_status.message.pipeline_compatibility_fingerprint
            actual = candidate_status.message.pipeline_compatibility_fingerprint
            if not expected or actual != expected:
                return "pipeline_compatibility_mismatch"
            return ""

    def _capacity_accepting(self, pipeline_id: str, work_class: int) -> bool:
        """Return the latest advertised admission state for a work class.

        This is an early filter only. The pipeline-local session controller is
        still authoritative because serving status is necessarily sampled.
        """

        with self._status_lock:
            observed = self._serving_status.get(pipeline_id)
            if observed is None:
                return False
            matches = [capacity for capacity in observed.message.capacities if int(capacity.work_class) == work_class]
            return len(matches) == 1 and bool(matches[0].accepting_requests)

    def _call_downstream(
        self,
        client,
        goal,
        *,
        operation_kind: OperationKind,
        deadline_monotonic_ns: int,
        upstream_goal_handle=None,
        late_acceptance_callback=None,
    ) -> _DownstreamCall:
        if upstream_goal_handle is not None and upstream_goal_handle.is_cancel_requested:
            return _DownstreamCall("not_started", reason="request_canceled")
        remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return _DownstreamCall("not_started", reason="deadline_exceeded")
        wait_timeout_ns = min(100_000_000, self._goal_acceptance_timeout_ns, remaining_ns)
        if not client.wait_for_server(timeout_sec=wait_timeout_ns / 1_000_000_000):
            reason = "deadline_exceeded" if deadline_monotonic_ns <= time.monotonic_ns() else "downstream_unavailable"
            return _DownstreamCall("not_started", reason=reason)
        if deadline_monotonic_ns <= time.monotonic_ns():
            return _DownstreamCall("not_started", reason="deadline_exceeded")
        operation_id = str(getattr(goal, "operation_id", ""))
        if not operation_id:
            return _DownstreamCall("unknown", reason="downstream_operation_id_missing")
        waiter_id = str(uuid.uuid4())
        downstream_operations = self._downstream_operations
        idempotency_key = (operation_kind.value, operation_id)
        # A cached UNKNOWN outcome must never be replayed (P0-2): fence the
        # quarantined record and re-send under a fresh operation id. Known
        # outcomes (COMPLETED/NOT_STARTED) remain idempotent replays.
        cached = downstream_operations.find(idempotency_key)
        if cached is not None and cached.terminal and cached.certainty is Certainty.UNKNOWN:
            if not downstream_operations.fence_operation(cached.operation_id):
                return _DownstreamCall("unknown", reason="downstream_unknown_retry_waiters", result=cached.result)
            operation_id = str(uuid.uuid4())
            goal.operation_id = operation_id
            idempotency_key = (operation_kind.value, operation_id)
        try:
            operation, created = downstream_operations.create_or_get(
                kind=operation_kind,
                idempotency_key=idempotency_key,
                identity=OperationIdentity(
                    str(goal.session_id),
                    max(1, int(getattr(goal, "logical_generation", 1))),
                    str(getattr(goal, "binding_id", "")),
                    max(1, int(getattr(goal, "binding_incarnation", 1))),
                    str(getattr(goal, "expected_boot_id", "")),
                ),
                deadline_mono_ns=deadline_monotonic_ns,
                waiter_id=waiter_id,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return _DownstreamCall("unknown", reason=f"downstream_operation_identity_invalid:{exc}")
        if not created:
            if operation.wait(max(0.0, (deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000)):
                call = self._operation_result(operation)
                downstream_operations.detach_waiter(operation.operation_id, waiter_id)
                return call
            downstream_operations.detach_waiter(operation.operation_id, waiter_id)
            return _DownstreamCall("unknown", reason="downstream_result_timeout")
        operation.claim_send()
        send_done = threading.Event()
        holder: dict[str, object] = {}
        acceptance_timed_out = False
        late_cleanup_sent = False
        acceptance_state_lock = threading.Lock()
        result_done = threading.Event()

        def _result(future) -> None:
            try:
                holder["result"] = future.result().result
            except Exception as exc:  # noqa: BLE001
                holder["result_error"] = exc
                downstream_operations.finish(operation.operation_id, certainty=Certainty.UNKNOWN, error=str(exc))
            else:
                result = holder["result"]
                outcome = int(result.outcome.value)
                certainty = {
                    InferenceOutcome.NOT_STARTED: Certainty.NOT_STARTED,
                    InferenceOutcome.COMPLETED: Certainty.COMPLETED,
                }.get(outcome, Certainty.UNKNOWN)
                downstream_operations.finish(
                    operation.operation_id,
                    certainty=certainty,
                    result=result,
                    error=str(result.error.code),
                )
            result_done.set()

        def _run_late_cleanup() -> None:
            nonlocal late_cleanup_sent
            with acceptance_state_lock:
                late_goal_handle = holder.get("goal_handle")
                if (
                    not acceptance_timed_out
                    or late_cleanup_sent
                    or late_acceptance_callback is None
                    or late_goal_handle is None
                    or not getattr(late_goal_handle, "accepted", False)
                ):
                    return
                late_cleanup_sent = True
            with suppress(Exception):
                late_acceptance_callback(late_goal_handle)

        def _sent(future) -> None:
            try:
                holder["goal_handle"] = future.result()
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
            _run_late_cleanup()
            goal_handle = holder.get("goal_handle")
            if "error" in holder:
                downstream_operations.finish(
                    operation.operation_id, certainty=Certainty.UNKNOWN, error=str(holder["error"])
                )
            elif goal_handle is None or not getattr(goal_handle, "accepted", False):
                downstream_operations.finish(
                    operation.operation_id, certainty=Certainty.NOT_STARTED, error="goal_rejected"
                )
            else:
                try:
                    goal_handle.get_result_async().add_done_callback(_result)
                except Exception as exc:  # noqa: BLE001
                    holder["result_error"] = exc
                    downstream_operations.finish(operation.operation_id, certainty=Certainty.UNKNOWN, error=str(exc))
                    result_done.set()
            send_done.set()

        try:
            client.send_goal_async(goal).add_done_callback(_sent)
        except Exception as exc:  # noqa: BLE001
            downstream_operations.finish(operation.operation_id, certainty=Certainty.NOT_STARTED, error=str(exc))
            downstream_operations.detach_waiter(operation.operation_id, waiter_id)
            return _DownstreamCall("not_started", reason=str(exc))
        acceptance_timeout = min(
            self._goal_acceptance_timeout_ns,
            max(0, deadline_monotonic_ns - time.monotonic_ns()),
        )
        cancel_requested = False
        acceptance_deadline_ns = time.monotonic_ns() + acceptance_timeout
        while not send_done.wait(min(0.05, max(0, acceptance_deadline_ns - time.monotonic_ns()) / 1_000_000_000)):
            cancel_requested = cancel_requested or bool(
                upstream_goal_handle is not None and upstream_goal_handle.is_cancel_requested
            )
            if time.monotonic_ns() >= acceptance_deadline_ns:
                with acceptance_state_lock:
                    acceptance_timed_out = True
                _run_late_cleanup()
                downstream_operations.detach_waiter(operation.operation_id, waiter_id)
                return _DownstreamCall("unknown", reason="goal_acceptance_timeout")
        if "error" in holder:
            downstream_operations.finish(
                operation.operation_id, certainty=Certainty.UNKNOWN, error=str(holder["error"])
            )
            downstream_operations.detach_waiter(operation.operation_id, waiter_id)
            return _DownstreamCall("unknown", reason=str(holder["error"]))
        goal_handle = holder.get("goal_handle")
        if goal_handle is None or not goal_handle.accepted:
            downstream_operations.detach_waiter(operation.operation_id, waiter_id)
            return _DownstreamCall("not_started", reason="goal_rejected")

        cancel_sent = False

        def request_downstream_cancel() -> None:
            nonlocal cancel_sent
            if cancel_sent:
                return
            cancel_sent = True
            with suppress(Exception):
                goal_handle.cancel_goal_async()

        if cancel_requested or (upstream_goal_handle is not None and upstream_goal_handle.is_cancel_requested):
            request_downstream_cancel()

        while not result_done.wait(0.05):
            if upstream_goal_handle is not None and upstream_goal_handle.is_cancel_requested:
                request_downstream_cancel()
            if time.monotonic_ns() >= deadline_monotonic_ns:
                downstream_operations.detach_waiter(operation.operation_id, waiter_id)
                return _DownstreamCall("unknown", reason="downstream_result_timeout")
        if "result_error" in holder:
            downstream_operations.detach_waiter(operation.operation_id, waiter_id)
            return _DownstreamCall("unknown", reason=str(holder["result_error"]))
        call = self._operation_result(operation)
        downstream_operations.detach_waiter(operation.operation_id, waiter_id)
        return call

    @staticmethod
    def _operation_result(operation) -> _DownstreamCall:
        result = operation.result
        if operation.certainty is Certainty.NOT_STARTED:
            return _DownstreamCall("not_started", result=result, reason=operation.error)
        if operation.certainty is Certainty.COMPLETED:
            return _DownstreamCall("completed", result=result)
        return _DownstreamCall("unknown", result=result, reason=operation.error or "downstream_outcome_unknown")

    @staticmethod
    def _validate_downstream_result(action: str, goal, result, candidate: PipelineCandidate) -> str:
        expected: dict[str, object] = {
            "session_id": goal.session_id,
            "logical_generation": goal.logical_generation,
            "binding_id": goal.binding_id,
            "binding_incarnation": goal.binding_incarnation,
            "operation_id": goal.operation_id,
            "boot_id": goal.expected_boot_id,
        }
        if action == "open":
            if result.success:
                expected.update(
                    pipeline_id=candidate.pipeline_id,
                    deployment_fingerprint=candidate.deployment_fingerprint,
                    runtime_policy_fingerprint=candidate.runtime_policy_fingerprint,
                )
                if int(result.pipeline_generation) <= 0:
                    return "open_pipeline_generation_invalid"
        elif action == "dispatch":
            expected.update(
                request_id=goal.request_id,
                pipeline_id=candidate.pipeline_id,
                deployment_fingerprint=candidate.deployment_fingerprint,
                runtime_policy_fingerprint=candidate.runtime_policy_fingerprint,
            )
            if result.success:
                chunk_size = int(result.chunk_size)
                execution_horizon = int(getattr(result, "execution_horizon", 0))
                if chunk_size < 1 or execution_horizon < 0 or execution_horizon > chunk_size:
                    return "execution_horizon_invalid"
            if int(goal.expected_pipeline_generation) > 0:
                expected["pipeline_generation"] = goal.expected_pipeline_generation
        elif action == "close":
            expected["pipeline_id"] = candidate.pipeline_id
            if int(result.outcome.value) == InferenceOutcome.COMPLETED and goal.expected_pipeline_generation > 0:
                expected["closed_pipeline_generation"] = goal.expected_pipeline_generation
        reason = result_identity_error(
            action,
            result,
            expected,
        )
        if reason:
            return reason
        if action == "close" and result.success:
            closed = int(getattr(result, "closed_pipeline_generation", -1))
            drained = int(getattr(result, "drained_generation", -1))
            cleanup_not_needed = goal.expected_pipeline_generation == 0 and closed == 0 and drained == 0
            if not cleanup_not_needed and (closed <= 0 or drained <= closed):
                return "close_drained_generation_invalid"
        return ""

    def _deadline_monotonic_ns(self, deadline, default_ns: int) -> int:
        utc_ns = int(deadline.sec) * 1_000_000_000 + int(deadline.nanosec)
        if utc_ns == 0:
            return time.monotonic_ns() + default_ns
        return time.monotonic_ns() + max(0, utc_ns - time.time_ns())

    @staticmethod
    def _copy_deadline(source, target) -> None:
        target.sec = source.sec
        target.nanosec = source.nanosec

    @staticmethod
    def _set_absolute_deadline(goal, deadline_utc_ns: int) -> None:
        goal.deadline.sec, goal.deadline.nanosec = divmod(int(deadline_utc_ns), 1_000_000_000)

    def _finish_upstream(self, goal_handle, result):
        if result.success:
            goal_handle.succeed()
        elif goal_handle.is_cancel_requested and getattr(result.error, "code", "") == "request_canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    @staticmethod
    def _cancel_scheduled_endpoint(goal_handle):
        request = goal_handle.request
        if getattr(request, "request_id", ""):
            return rclpy.action.CancelResponse.ACCEPT
        return rclpy.action.CancelResponse.REJECT

    def _open_endpoint(self, goal_handle) -> OpenInferenceSession.Result:
        goal = goal_handle.request
        return self._execute_idempotent(
            goal_handle=goal_handle,
            action=LedgerAction.OPEN,
            key=open_key(goal.session_id),
            payload={},
            deadline=goal.deadline,
            is_open=True,
            execute=self._open_once,
        )

    def _dispatch_endpoint(self, goal_handle) -> ScheduledDispatchInfer.Result:
        goal = goal_handle.request
        payload = {
            "session_generation": goal.session_generation,
            "obs_timestamp": goal.obs_timestamp,
            "prompt": goal.prompt,
            "priority": goal.priority,
            "target_pipeline_id": goal.target_pipeline_id,
        }
        if goal.priority == 0:
            payload["fallback_chain"] = list(goal.fallback_chain)
            payload["deadline"] = goal.deadline
        return self._execute_idempotent(
            goal_handle=goal_handle,
            action=LedgerAction.DISPATCH,
            key=dispatch_key(goal.session_id, goal.session_generation, goal.request_id),
            payload=payload,
            deadline=goal.deadline,
            is_open=False,
            execute=self._dispatch_once,
            request_id=goal.request_id,
        )

    def _close_endpoint(self, goal_handle) -> CloseInferenceSession.Result:
        goal = goal_handle.request
        return self._execute_idempotent(
            goal_handle=goal_handle,
            action=LedgerAction.CLOSE,
            key=close_key(goal.session_id, goal.session_generation),
            payload={"session_generation": goal.session_generation},
            deadline=goal.deadline,
            is_open=False,
            execute=self._close_once,
        )

    def _execute_idempotent(
        self,
        *,
        goal_handle,
        action: LedgerAction,
        key: tuple,
        payload: dict[str, object],
        deadline,
        is_open: bool,
        execute,
        request_id: str = "",
    ):
        goal = goal_handle.request
        if action is LedgerAction.DISPATCH and goal_handle.is_cancel_requested:
            return self._idempotency_failure(
                goal_handle,
                action,
                goal,
                request_id,
                "request_canceled",
                "request was canceled before scheduling",
                InferenceOutcome.NOT_STARTED,
            )
        if action is LedgerAction.DISPATCH and utf8_size(goal.prompt) > self._max_prompt_bytes:
            return self._idempotency_failure(
                goal_handle,
                action,
                goal,
                request_id,
                "prompt_too_large",
                f"prompt exceeds max_prompt_bytes={self._max_prompt_bytes}",
                InferenceOutcome.NOT_STARTED,
            )
        try:
            validate_uuid4(goal.session_id, field="session_id")
            if request_id:
                validate_uuid4(request_id, field="request_id")
            original_deadline_ns = int(deadline.sec) * 1_000_000_000 + int(deadline.nanosec)
            if original_deadline_ns < 0:
                raise IdempotencyError("deadline cannot be negative")
            if action is LedgerAction.DISPATCH:
                if goal.session_generation <= 0:
                    raise IdempotencyError("session_generation must be positive")
                if goal.priority < 0:
                    raise IdempotencyError("priority cannot be negative")
            effective_request_deadline_ns = (
                0 if action is LedgerAction.DISPATCH and goal.priority > 0 else original_deadline_ns
            )
            resolution = self._ingress_ledger.resolve(
                action=action,
                key=key,
                payload_fingerprint=canonical_fingerprint(payload),
                effective_deadline_utc_ns=resolve_entry_deadline_ns(
                    effective_request_deadline_ns,
                    is_open=is_open,
                    default_open_timeout_ns=self._default_open_timeout_ns,
                    default_request_timeout_ns=self._default_request_timeout_ns,
                ),
            )
        except (IdempotencyError, LedgerError, TypeError, ValueError) as exc:
            return self._idempotency_failure(
                goal_handle, action, goal, request_id, "request_conflict", str(exc), InferenceOutcome.NOT_STARTED
            )
        return execute_resolved_action(
            self._ingress_ledger,
            resolution,
            key=key,
            goal_handle=goal_handle,
            execute=lambda entry: execute(goal_handle, entry),
            failure=lambda code, message, outcome: self._idempotency_failure(
                goal_handle, action, goal, request_id, code, message, outcome
            ),
            error_codes=ResolutionErrorCodes(
                ledger_full="scheduler_ledger_full",
                ledger_error="scheduler_ledger_error",
                internal_error="scheduler_internal_error",
            ),
            not_started_outcome=InferenceOutcome.NOT_STARTED,
            unknown_outcome=InferenceOutcome.UNKNOWN,
            log_internal_error=lambda exc: self.get_logger().exception(
                f"unhandled {action.value} endpoint failure: {exc}"
            ),
            prepare_entry=lambda entry: self._set_absolute_deadline(goal, entry.effective_deadline_utc_ns),
        )

    def _idempotency_failure(self, goal_handle, action, goal, request_id: str, code: str, message: str, outcome: int):
        if action is LedgerAction.OPEN:
            result = OpenInferenceSession.Result()
            result.session_id = goal.session_id
        elif action is LedgerAction.DISPATCH:
            result = ScheduledDispatchInfer.Result()
            result.request_id = request_id
            result.session_id = goal.session_id
            result.session_generation = goal.session_generation
        else:
            result = CloseInferenceSession.Result()
            result.session_id = goal.session_id
        result.outcome.value = outcome
        self._set_error(
            result.error,
            code=code,
            message=message,
            recoverable=outcome == InferenceOutcome.NOT_STARTED,
            stage="admission",
        )
        if code == "request_canceled" and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _open_once(self, goal_handle, _ledger_entry) -> OpenInferenceSession.Result:
        goal = goal_handle.request
        deadline_utc_ns = int(goal.deadline.sec) * 1_000_000_000 + int(goal.deadline.nanosec)
        if deadline_utc_ns <= time.time_ns():
            return self._open_failure(
                goal_handle,
                goal.session_id,
                "deadline_exceeded",
                "",
                InferenceOutcome.NOT_STARTED,
            )
        try:
            decision = self._core.open_session(session_id=goal.session_id)
        except SchedulerError as exc:
            return self._open_failure(
                goal_handle,
                goal.session_id,
                str(exc),
                "",
                InferenceOutcome.NOT_STARTED,
            )
        result = OpenInferenceSession.Result()
        result.success = True
        result.session_id = goal.session_id
        result.session_generation = decision.session_generation
        lease_expires_at_ros_ns = monotonic_expiry_to_ros_ns(
            decision.lease_expires_at_ns,
            monotonic_now_ns=time.monotonic_ns(),
            ros_now_ns=self.get_clock().now().nanoseconds,
        )
        result.lease_expires_at.sec, result.lease_expires_at.nanosec = divmod(lease_expires_at_ros_ns, 1_000_000_000)
        result.outcome.value = InferenceOutcome.COMPLETED
        return self._finish_upstream(goal_handle, result)

    def _open_failure(self, goal_handle, session_id: str, code: str, message: str, outcome: int):
        result = OpenInferenceSession.Result()
        result.session_id = session_id
        result.outcome.value = outcome
        self._set_error(
            result.error,
            code=code or "open_failed",
            message=message,
            recoverable=outcome == InferenceOutcome.NOT_STARTED,
            stage="open",
        )
        goal_handle.abort()
        return result

    def _dispatch_once(self, goal_handle, _ledger_entry) -> ScheduledDispatchInfer.Result:
        goal = goal_handle.request
        global_policy = getattr(self, "_global_policy", "fifo")
        profile_admission = global_policy == "fifo" and self._priority_zero_deadline_admission_enabled
        priority_zero_scheduling = goal.priority == 0 and (global_policy == "edf" or profile_admission)
        # Deadline-driven priority-0 routing is a sequential-only contract;
        # both EDF ordering and FIFO profile admission require it.
        if goal.fallback_chain and goal.priority == 0 and global_policy == "fifo" and not profile_admission:
            return self._dispatch_not_started(
                goal_handle,
                goal,
                goal.target_pipeline_id,
                "fallback_not_supported_for_policy",
                "",
                recoverable=False,
            )
        try:
            plan = self._core.resolve_dispatch_plan(
                session_id=goal.session_id,
                session_generation=goal.session_generation,
                target_pipeline_id=goal.target_pipeline_id,
                fallback_chain=list(goal.fallback_chain),
                priority=goal.priority,
                request_id=goal.request_id,
            )
        except SchedulerError as exc:
            return self._dispatch_not_started(goal_handle, goal, "", str(exc), "")
        try:
            target = self._candidate_by_id[goal.target_pipeline_id]
            last_pipeline_id = goal.target_pipeline_id
            last_reason = "no_feasible_deadline" if goal.priority == 0 else "downstream_not_started"
            last_detail = ""
            last_recoverable = True
            deadline_monotonic_ns = self._deadline_monotonic_ns(goal.deadline, self._default_request_timeout_ns)
            for pipeline_id in plan.candidate_ids:
                reservation: DeadlineReservation | None = None
                last_pipeline_id = pipeline_id
                candidate = self._candidate_by_id[pipeline_id]
                reason = self._status_reason(candidate, required_priority=goal.priority)
                if not reason:
                    reason = self._compatibility_reason(target, candidate)
                if reason:
                    last_reason = reason
                    last_recoverable = True
                    continue
                try:
                    observed = getattr(self, "_serving_status", {}).get(pipeline_id)
                    expected_boot_id = observed.message.boot_id if observed is not None else None
                    binding = self._core.prepare_dispatch_candidate(
                        session_id=goal.session_id,
                        session_generation=goal.session_generation,
                        pipeline_id=pipeline_id,
                        expected_boot_id=expected_boot_id,
                    )
                except SchedulerError as exc:
                    return self._dispatch_not_started(goal_handle, goal, pipeline_id, str(exc), "")
                if binding.pipeline_id is None:
                    last_reason = getattr(binding, "reason", "pipeline_unavailable")
                    last_recoverable = True
                    continue
                if getattr(binding, "reason", "") == "binding_open_in_progress":
                    wait_for_open = getattr(self._core, "wait_for_binding_open", None)
                    if not callable(wait_for_open) or not wait_for_open(
                        session_id=goal.session_id,
                        pipeline_id=pipeline_id,
                        deadline_monotonic_ns=deadline_monotonic_ns,
                    ):
                        last_reason = "binding_open_timeout"
                        last_detail = "another request is still opening the pipeline binding"
                        last_recoverable = True
                        continue
                    observed = getattr(self, "_serving_status", {}).get(pipeline_id)
                    expected_boot_id = observed.message.boot_id if observed is not None else None
                    try:
                        binding = self._core.prepare_dispatch_candidate(
                            session_id=goal.session_id,
                            session_generation=goal.session_generation,
                            pipeline_id=pipeline_id,
                            expected_boot_id=expected_boot_id,
                        )
                    except SchedulerError as exc:
                        return self._dispatch_not_started(goal_handle, goal, pipeline_id, str(exc), "")
                    if binding.pipeline_id is None:
                        last_reason = getattr(binding, "reason", "pipeline_unavailable")
                        last_recoverable = True
                        continue
                    if getattr(binding, "reason", "") == "binding_open_in_progress":
                        last_reason = "binding_open_in_progress"
                        last_detail = "pipeline binding did not settle"
                        last_recoverable = True
                        continue
                required_work_class = WorkClass.SESSION_CONTROL if binding.needs_open else WorkClass.ACTION_GENERATION
                if not self._capacity_accepting(pipeline_id, required_work_class):
                    if binding.needs_open:
                        self._core.release_dispatch_candidate(
                            session_id=goal.session_id,
                            pipeline_id=pipeline_id,
                            not_started=True,
                        )
                    last_reason = "pipeline_busy"
                    last_recoverable = True
                    continue
                if priority_zero_scheduling:
                    observed = getattr(self, "_serving_status", {}).get(pipeline_id)
                    status = observed.message if observed is not None else None
                    if status is not None and not status.supports_priority_zero_deadline_admission:
                        if binding.needs_open:
                            self._core.release_dispatch_candidate(
                                session_id=goal.session_id, pipeline_id=pipeline_id, not_started=True
                            )
                        last_reason = "priority_zero_deadline_admission_not_supported"
                        last_recoverable = True
                        continue
                    reservation, detail = self._reserve_priority_zero(
                        pipeline_id=pipeline_id,
                        hardware_resource_id=candidate.hardware_resource_id,
                        deadline_monotonic_ns=deadline_monotonic_ns,
                        requires_open=binding.needs_open,
                        prompt_bytes=utf8_size(goal.prompt),
                        session_id=goal.session_id,
                        binding_id=binding.binding_id,
                        binding_incarnation=binding.binding_incarnation,
                        expected_boot_id=binding.expected_boot_id,
                    )
                    if reservation is None:
                        if binding.needs_open:
                            self._core.release_dispatch_candidate(
                                session_id=goal.session_id,
                                pipeline_id=pipeline_id,
                                not_started=True,
                            )
                        last_reason = "no_feasible_deadline"
                        last_detail = detail
                        last_recoverable = False
                        continue
                    turn_result = self._deadline_reservations.wait_for_turn(
                        reservation,
                        deadline_ns=deadline_monotonic_ns,
                        cancel_requested=lambda: bool(goal_handle.is_cancel_requested),
                    )
                    if turn_result != "ready":
                        self._release_reservation(reservation)
                        reservation = None
                        if binding.needs_open:
                            self._core.release_dispatch_candidate(
                                session_id=goal.session_id,
                                pipeline_id=pipeline_id,
                                not_started=True,
                            )
                        if turn_result == "request_canceled":
                            return self._dispatch_not_started(
                                goal_handle,
                                goal,
                                pipeline_id,
                                "request_canceled",
                                "",
                                recoverable=True,
                            )
                        last_reason = "no_feasible_deadline"
                        last_detail = "resource_turn_exceeds_deadline"
                        last_recoverable = False
                        continue
                try:
                    pipeline_generation = binding.pipeline_generation
                    if binding.needs_open:
                        open_call = self._open_dispatch_binding(
                            goal_handle,
                            goal,
                            candidate,
                            binding=binding,
                            deadline_monotonic_ns=deadline_monotonic_ns,
                        )
                        if open_call.certainty == "not_started":
                            self._release_reservation(reservation)
                            reservation = None
                            self._core.release_dispatch_candidate(
                                session_id=goal.session_id,
                                pipeline_id=pipeline_id,
                                not_started=True,
                            )
                            last_reason = open_call.reason or "downstream_not_started"
                            last_recoverable = self._call_recoverable(open_call)
                            continue
                        if open_call.certainty == "unknown" or open_call.result is None:
                            self._mark_reservation_unknown(reservation, expected_boot_id=expected_boot_id)
                            reservation = None
                            self._core.mark_session_failed(goal.session_id, pipeline_id=pipeline_id)
                            return self._dispatch_unknown(
                                goal_handle,
                                goal,
                                pipeline_id,
                                open_call.reason or "binding_open_outcome_unknown",
                            )
                        if not open_call.result.success:
                            self._release_reservation(reservation)
                            reservation = None
                            self._core.mark_session_failed(goal.session_id, pipeline_id=pipeline_id)
                            return self._dispatch_completed_failure(
                                goal_handle,
                                goal,
                                pipeline_id,
                                open_call.result.error.code or "binding_open_failed",
                                open_call.result.error.message,
                            )
                        opened_generation = int(
                            getattr(
                                open_call.result,
                                "pipeline_generation",
                                getattr(open_call.result, "session_generation", 0),
                            )
                        )
                        close_raced = self._core.record_binding_open_success(
                            session_id=goal.session_id,
                            pipeline_id=pipeline_id,
                            pipeline_generation=opened_generation,
                            hardware_resource_id=candidate.hardware_resource_id,
                            binding_id=getattr(open_call.result, "binding_id", None),
                            binding_incarnation=getattr(open_call.result, "binding_incarnation", None),
                            boot_id=getattr(open_call.result, "boot_id", None),
                        )
                        if close_raced:
                            self._release_reservation(reservation)
                            reservation = None
                            return self._dispatch_not_started(
                                goal_handle, goal, pipeline_id, "session_closing", "", recoverable=False
                            )
                        pipeline_generation = opened_generation
                    call, _downstream_goal = self._dispatch_bound_pipeline(
                        goal_handle,
                        goal,
                        candidate,
                        pipeline_generation=pipeline_generation,
                        deadline_monotonic_ns=deadline_monotonic_ns,
                    )
                    if call.certainty == "not_started":
                        self._release_reservation(reservation)
                        reservation = None
                        last_reason = call.reason or "downstream_not_started"
                        last_recoverable = self._call_recoverable(call)
                        continue
                    if call.certainty == "unknown" or call.result is None:
                        self._mark_reservation_unknown(reservation, expected_boot_id=expected_boot_id)
                        reservation = None
                        self._core.mark_session_failed(goal.session_id, pipeline_id=pipeline_id)
                        return self._dispatch_unknown(goal_handle, goal, pipeline_id, call.reason)
                    self._release_reservation(reservation)
                    reservation = None
                    downstream_result = call.result
                    result = ScheduledDispatchInfer.Result()
                    # Field-for-field copy from the private DispatchPipelineBinding
                    # result. Keep this list in sync with the public
                    # ScheduledDispatchInfer.result field set —
                    # test_scheduler_node_guards.py::test_public_dispatch_result_fields_cover_private_result
                    # fails when the two message definitions drift apart.
                    for field in (
                        "action_chunk",
                        "chunk_size",
                        "execution_horizon",
                        "success",
                        "inference_latency_ms",
                        "backend_latency_ms",
                        "request_id",
                        "session_id",
                        "pipeline_id",
                        "deployment_fingerprint",
                        "runtime_policy_fingerprint",
                        "outcome",
                        "error",
                    ):
                        setattr(result, field, getattr(downstream_result, field))
                    result.session_generation = goal.session_generation
                    return self._finish_upstream(goal_handle, result)
                except Exception:
                    self._mark_reservation_unknown(reservation, expected_boot_id=expected_boot_id)
                    raise
            return self._dispatch_not_started(
                goal_handle,
                goal,
                last_pipeline_id,
                last_reason,
                last_detail,
                recoverable=last_recoverable,
            )
        finally:
            self._core.record_request_terminal(
                goal.session_id, request_id=goal.request_id, session_generation=goal.session_generation
            )

    def _priority_zero_estimate_ms(
        self,
        *,
        pipeline_id: str,
        requires_open: bool,
        prompt_bytes: int,
    ) -> tuple[float | None, str]:
        registry = self._profile_registries.get(pipeline_id)
        if registry is None:
            return None, self._profile_errors.get(
                pipeline_id,
                "priority_zero_profile_unavailable:profile_path_missing",
            )
        with self._status_lock:
            observed = self._serving_status.get(pipeline_id)
            input_contract_fingerprint = (
                observed.message.pipeline_compatibility_fingerprint if observed is not None else ""
            )
        if not input_contract_fingerprint:
            return None, "priority_zero_profile_unavailable:input_contract_fingerprint_missing"
        phases = [
            (
                WorkClass.ACTION_GENERATION,
                _FULL_INFER_CLOSURE,
                input_contract_fingerprint,
                prompt_bytes,
                self._dispatch_safety_margin_ms,
            )
        ]
        if requires_open:
            phases.insert(0, (WorkClass.SESSION_CONTROL, _SESSION_OPEN_CLOSURE, "", 0, 0))
        estimate_ms = 0.0
        for work_class, closure_key, contract_fingerprint, covered_prompt_bytes, phase_margin_ms in phases:
            profile_p99_ms = registry.closure_p99_ms(
                work_class=work_class,
                closure_key=closure_key,
                hardware_priority=0,
                input_contract_fingerprint=contract_fingerprint,
                prompt_bytes=covered_prompt_bytes,
            )
            acceptance_p999_ms = registry.goal_acceptance_p999_ms(
                work_class=work_class,
                closure_key=closure_key,
                hardware_priority=0,
                input_contract_fingerprint=contract_fingerprint,
                prompt_bytes=covered_prompt_bytes,
            )
            if profile_p99_ms is None or acceptance_p999_ms is None:
                reason = self._profile_errors.get(
                    pipeline_id,
                    f"priority_zero_profile_unavailable:{closure_key}",
                )
                return None, reason
            estimate_ms += (
                profile_p99_ms + acceptance_p999_ms + self._goal_acceptance_safety_margin_ms + phase_margin_ms
            )
        estimate_ms += getattr(self, "_priority_zero_deadline_admission_safety_margin_ms", 0)
        return estimate_ms, ""

    def _reserve_priority_zero(
        self,
        *,
        pipeline_id: str,
        hardware_resource_id: str,
        deadline_monotonic_ns: int,
        requires_open: bool,
        prompt_bytes: int,
        session_id: str | None = None,
        binding_id: str | None = None,
        binding_incarnation: int | None = None,
        expected_boot_id: str | None = None,
    ) -> tuple[DeadlineReservation | None, str]:
        policy = getattr(self, "_global_policy", "fifo")
        profile_admission = policy == "fifo" and self._priority_zero_deadline_admission_enabled
        owner = {
            "session_id": session_id,
            "binding_id": binding_id,
            "binding_incarnation": binding_incarnation,
            "expected_boot_id": expected_boot_id,
        }
        if not profile_admission:
            reservation = self._deadline_reservations.try_reserve(
                pipeline_id=pipeline_id,
                hardware_resource_id=hardware_resource_id,
                now_ns=time.monotonic_ns(),
                deadline_ns=deadline_monotonic_ns,
                **owner,
            )
            return reservation, "" if reservation is not None else "resource_queue_unavailable_or_expired"
        estimate_ms, reason = self._priority_zero_estimate_ms(
            pipeline_id=pipeline_id,
            requires_open=requires_open,
            prompt_bytes=prompt_bytes,
        )
        if estimate_ms is None:
            return None, reason
        estimate_ns = int(estimate_ms * 1_000_000)
        reservation = self._deadline_reservations.try_reserve(
            pipeline_id=pipeline_id,
            hardware_resource_id=hardware_resource_id,
            now_ns=time.monotonic_ns(),
            deadline_ns=deadline_monotonic_ns,
            estimate_ns=estimate_ns,
            **owner,
        )
        return reservation, "" if reservation is not None else "resource_reservation_exceeds_deadline"

    def _release_reservation(self, reservation: DeadlineReservation | None) -> None:
        if reservation is not None:
            self._deadline_reservations.release(reservation)

    def _mark_reservation_unknown(
        self, reservation: DeadlineReservation | None, *, expected_boot_id: str | None
    ) -> None:
        if reservation is not None:
            # Serialize with boot reconciliation: a fenced request must not
            # reintroduce UNKNOWN ownership after the new process is trusted.
            with self._status_lock:
                cursor = self._trusted_status_cursors.get(reservation.pipeline_id)
                if expected_boot_id is not None and cursor is not None and cursor[0] != expected_boot_id:
                    self._deadline_reservations.release(reservation)
                else:
                    self._deadline_reservations.mark_unknown(reservation)

    @staticmethod
    def _call_recoverable(call: _DownstreamCall) -> bool:
        return bool(call.result is None or getattr(call.result.error, "recoverable", False))

    def _open_dispatch_binding(
        self,
        goal_handle,
        goal,
        candidate: PipelineCandidate,
        *,
        binding: BindingDecision,
        deadline_monotonic_ns: int,
    ) -> _DownstreamCall:
        downstream_goal = OpenPipelineBinding.Goal()
        downstream_goal.session_id = goal.session_id
        downstream_goal.logical_generation = goal.session_generation
        downstream_goal.binding_id = binding.binding_id
        downstream_goal.binding_incarnation = binding.binding_incarnation
        downstream_goal.operation_id = str(uuid.uuid4())
        downstream_goal.expected_boot_id = binding.expected_boot_id
        self._copy_deadline(goal.deadline, downstream_goal.deadline)
        call = self._call_downstream(
            self._pipeline_clients[candidate.pipeline_id]["open"],
            downstream_goal,
            operation_kind=OperationKind.OPEN,
            deadline_monotonic_ns=deadline_monotonic_ns,
            upstream_goal_handle=goal_handle,
            late_acceptance_callback=lambda late_goal_handle: self._cleanup_late_open(
                candidate.pipeline_id, downstream_goal, late_goal_handle
            ),
        )
        if call.result is not None:
            reason = self._validate_downstream_result("open", downstream_goal, call.result, candidate)
            if reason:
                return _DownstreamCall("unknown", result=call.result, reason=reason)
        return call

    def _cleanup_late_open(self, pipeline_id: str, open_goal, late_goal_handle) -> None:
        """Cancel a late-accepted Open and issue generation-0 cleanup."""
        with suppress(Exception):
            late_goal_handle.cancel_goal_async()

        def _send_close(_open_result_future) -> None:
            close_goal = ClosePipelineBinding.Goal()
            close_goal.session_id = open_goal.session_id
            close_goal.logical_generation = 0
            close_goal.binding_id = open_goal.binding_id
            close_goal.binding_incarnation = open_goal.binding_incarnation
            close_goal.operation_id = str(uuid.uuid4())
            close_goal.expected_boot_id = open_goal.expected_boot_id
            close_goal.expected_pipeline_generation = 0
            self._set_absolute_deadline(close_goal, time.time_ns() + self._default_request_timeout_ns)

            def _close_goal_sent(future) -> None:
                try:
                    close_goal_handle = future.result()
                    if close_goal_handle is None or not close_goal_handle.accepted:
                        return

                    def _close_result_done(result_future) -> None:
                        try:
                            result = result_future.result().result
                        except Exception:  # noqa: BLE001
                            return
                        if result is not None and result.success:
                            candidate = self._candidate_by_id.get(pipeline_id)
                            identity_reason = (
                                self._validate_downstream_result("close", close_goal, result, candidate)
                                if candidate is not None
                                else "late_close_pipeline_missing"
                            )
                            if identity_reason:
                                return
                            self._record_drained_binding(pipeline_id, close_goal)
                            self._core.record_close_complete(open_goal.session_id, success=True)

                    close_goal_handle.get_result_async().add_done_callback(_close_result_done)
                except Exception:  # noqa: BLE001
                    return

            with suppress(Exception):
                self._pipeline_clients[pipeline_id]["close"].send_goal_async(close_goal).add_done_callback(
                    _close_goal_sent
                )

        with suppress(Exception):
            late_goal_handle.get_result_async().add_done_callback(_send_close)

    def _dispatch_bound_pipeline(
        self,
        goal_handle,
        goal,
        candidate: PipelineCandidate,
        *,
        pipeline_generation: int,
        deadline_monotonic_ns: int,
    ) -> tuple[_DownstreamCall, DispatchPipelineBinding.Goal]:
        binding = self._core.session_record(goal.session_id).bindings[candidate.pipeline_id]
        downstream_goal = DispatchPipelineBinding.Goal()
        downstream_goal.obs_timestamp.sec = goal.obs_timestamp.sec
        downstream_goal.obs_timestamp.nanosec = goal.obs_timestamp.nanosec
        downstream_goal.prompt = goal.prompt
        downstream_goal.request_id = goal.request_id
        downstream_goal.session_id = goal.session_id
        downstream_goal.logical_generation = goal.session_generation
        downstream_goal.binding_id = binding.binding_id
        downstream_goal.binding_incarnation = binding.binding_incarnation
        downstream_goal.operation_id = str(uuid.uuid4())
        downstream_goal.expected_boot_id = binding.expected_boot_id or ""
        downstream_goal.expected_pipeline_generation = pipeline_generation
        self._copy_deadline(goal.deadline, downstream_goal.deadline)
        downstream_goal.priority = goal.priority
        call = self._call_downstream(
            self._pipeline_clients[candidate.pipeline_id]["dispatch"],
            downstream_goal,
            operation_kind=OperationKind.DISPATCH,
            deadline_monotonic_ns=deadline_monotonic_ns,
            upstream_goal_handle=goal_handle,
        )
        if call.result is not None:
            reason = self._validate_downstream_result("dispatch", downstream_goal, call.result, candidate)
            if reason:
                call = _DownstreamCall("unknown", result=call.result, reason=reason)
        return call, downstream_goal

    def _dispatch_completed_failure(self, goal_handle, goal, pipeline_id: str, code: str, message: str):
        result = ScheduledDispatchInfer.Result()
        result.request_id = goal.request_id
        result.session_id = goal.session_id
        result.session_generation = goal.session_generation
        result.pipeline_id = pipeline_id
        result.outcome.value = InferenceOutcome.COMPLETED
        self._set_error(result.error, code=code, message=message, stage="dispatch")
        goal_handle.abort()
        return result

    def _dispatch_not_started(
        self,
        goal_handle,
        goal,
        pipeline_id: str,
        code: str,
        message: str,
        *,
        recoverable: bool = True,
    ):
        result = ScheduledDispatchInfer.Result()
        result.request_id = goal.request_id
        result.session_id = goal.session_id
        result.session_generation = goal.session_generation
        result.pipeline_id = pipeline_id
        result.outcome.value = InferenceOutcome.NOT_STARTED
        self._set_error(result.error, code=code, message=message, recoverable=recoverable, stage="dispatch")
        if code == "request_canceled" and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _dispatch_unknown(self, goal_handle, goal, pipeline_id: str, reason: str):
        result = ScheduledDispatchInfer.Result()
        result.request_id = goal.request_id
        result.session_id = goal.session_id
        result.session_generation = goal.session_generation
        result.pipeline_id = pipeline_id
        result.outcome.value = InferenceOutcome.UNKNOWN
        self._set_error(result.error, code=reason or "dispatch_outcome_unknown", stage="dispatch")
        goal_handle.abort()
        return result

    def _close_once(self, goal_handle, _ledger_entry) -> CloseInferenceSession.Result:
        goal = goal_handle.request
        deadline_monotonic_ns = self._deadline_monotonic_ns(goal.deadline, self._default_request_timeout_ns)
        try:
            self._core.begin_close(session_id=goal.session_id, session_generation=goal.session_generation)
        except SchedulerError as exc:
            result = CloseInferenceSession.Result()
            result.session_id = goal.session_id
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_error(result.error, code=str(exc), recoverable=True, stage="close")
            goal_handle.abort()
            return result
        record = self._core.session_record(goal.session_id)
        logical_generation = record.session_generation if record is not None else goal.session_generation
        if not self._core.wait_for_bindings_to_settle(goal.session_id, deadline_monotonic_ns):
            self._core.mark_session_failed(goal.session_id)
            result = CloseInferenceSession.Result()
            result.session_id = goal.session_id
            result.outcome.value = InferenceOutcome.UNKNOWN
            self._set_error(result.error, code="close_waiting_for_open_timeout", stage="close")
            goal_handle.abort()
            return result

        close_unknown = False
        close_unknown_code = ""
        close_unknown_pipeline = ""
        close_failure = ""
        close_failure_pipeline = ""
        close_not_started = ""
        close_not_started_pipeline = ""
        close_recoverable = True
        for binding in self._core.close_bindings(goal.session_id):
            downstream_goal = ClosePipelineBinding.Goal()
            downstream_goal.session_id = goal.session_id
            downstream_goal.logical_generation = logical_generation
            downstream_goal.binding_id = getattr(binding, "binding_id", "")
            downstream_goal.binding_incarnation = getattr(binding, "binding_incarnation", 1)
            downstream_goal.operation_id = getattr(binding, "close_operation_id", None) or str(uuid.uuid4())
            downstream_goal.expected_boot_id = getattr(binding, "expected_boot_id", None) or ""
            downstream_goal.expected_pipeline_generation = binding.pipeline_generation
            self._copy_deadline(goal.deadline, downstream_goal.deadline)
            call = self._call_downstream(
                self._pipeline_clients[binding.pipeline_id]["close"],
                downstream_goal,
                operation_kind=OperationKind.CLOSE,
                deadline_monotonic_ns=deadline_monotonic_ns,
            )
            candidate = self._candidate_by_id[binding.pipeline_id]
            if call.result is not None:
                identity_reason = self._validate_downstream_result("close", downstream_goal, call.result, candidate)
                if identity_reason:
                    call = _DownstreamCall("unknown", result=call.result, reason=identity_reason)
            if call.certainty == "unknown":
                self._core.mark_session_failed(goal.session_id, pipeline_id=binding.pipeline_id)
                close_unknown = True
                close_unknown_code = call.reason or "close_outcome_unknown"
                close_unknown_pipeline = binding.pipeline_id
                continue
            if call.certainty == "not_started":
                close_not_started = call.reason or "downstream_not_started"
                close_not_started_pipeline = binding.pipeline_id
                close_recoverable = close_recoverable and self._call_recoverable(call)
                continue
            if not call.result.success:
                self._core.mark_session_failed(goal.session_id, pipeline_id=binding.pipeline_id)
                close_failure = call.result.error.code or "close_failed"
                close_failure_pipeline = binding.pipeline_id
                continue
            self._record_drained_binding(binding.pipeline_id, downstream_goal)

        if close_unknown:
            self._core.record_close_complete(goal.session_id, success=False, unresolved=True)
            result = CloseInferenceSession.Result()
            result.session_id = goal.session_id
            result.pipeline_id = close_unknown_pipeline
            result.closed_session_generation = logical_generation
            result.outcome.value = InferenceOutcome.UNKNOWN
            self._set_error(result.error, code=close_unknown_code, stage="close")
            goal_handle.abort()
            return result
        if close_failure:
            self._core.record_close_complete(goal.session_id, success=False, unresolved=True)
            return self._close_completed_failure(
                goal_handle,
                goal.session_id,
                close_failure_pipeline,
                logical_generation,
                close_failure,
            )
        if close_not_started:
            self._core.record_close_not_started(goal.session_id)
            result = CloseInferenceSession.Result()
            result.session_id = goal.session_id
            result.pipeline_id = close_not_started_pipeline
            result.outcome.value = InferenceOutcome.NOT_STARTED
            self._set_error(result.error, code=close_not_started, recoverable=close_recoverable, stage="close")
            goal_handle.abort()
            return result
        drained_generation = self._core.record_close_complete(goal.session_id, success=True)
        result = CloseInferenceSession.Result()
        result.success = True
        result.session_id = goal.session_id
        result.closed_session_generation = logical_generation
        result.drained_generation = drained_generation
        result.outcome.value = InferenceOutcome.COMPLETED
        return self._finish_upstream(goal_handle, result)

    def _record_drained_binding(self, pipeline_id: str, goal) -> None:
        self._downstream_operations.fence_binding(
            session_id=goal.session_id,
            binding_id=goal.binding_id,
            binding_incarnation=goal.binding_incarnation,
            expected_boot_id=goal.expected_boot_id,
        )
        self._deadline_reservations.reconcile_binding(
            pipeline_id=pipeline_id,
            session_id=goal.session_id,
            binding_id=goal.binding_id,
            binding_incarnation=goal.binding_incarnation,
            expected_boot_id=goal.expected_boot_id,
        )
        self._core.record_binding_close_success(goal.session_id, pipeline_id)

    def _close_completed_failure(
        self,
        goal_handle,
        session_id: str,
        pipeline_id: str,
        logical_generation: int,
        code: str,
    ) -> CloseInferenceSession.Result:
        result = CloseInferenceSession.Result()
        result.session_id = session_id
        result.pipeline_id = pipeline_id
        result.closed_session_generation = logical_generation
        result.outcome.value = InferenceOutcome.COMPLETED
        self._set_error(result.error, code=code, stage="close")
        goal_handle.abort()
        return result

    def _idle_sweep(self) -> None:
        for session_id in self._core.expired_sessions():
            record = self._core.session_record(session_id)
            if record is None:
                continue
            goal = CloseInferenceSession.Goal()
            goal.session_id = session_id
            goal.session_generation = record.session_generation
            deadline_ns = time.time_ns() + self._default_request_timeout_ns
            self._set_absolute_deadline(goal, deadline_ns)
            self._close_retry_exhausted.discard(session_id)
            try:
                self._close_once(_MaintenanceGoalHandle(goal), None)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"idle Close failed for {session_id}: {exc}")
        # Aged QUARANTINED/CLOSING sessions get exactly one bounded Close
        # retry. Exhaustion never supplies evidence that execution drained.
        quarantine_age_ns = max(self._session_idle_timeout_ns, self._default_request_timeout_ns)
        aged = self._core.aged_quarantined_sessions(quarantine_age_ns)
        if self._close_retry_exhausted:
            still_stuck = set(aged)
            for session_id in tuple(self._close_retry_exhausted):
                record = self._core.session_record(session_id)
                if session_id not in still_stuck and (
                    record is None or record.state not in (GlobalSessionState.QUARANTINED, GlobalSessionState.CLOSING)
                ):
                    self._close_retry_exhausted.discard(session_id)
        for session_id in aged:
            record = self._core.session_record(session_id)
            if record is None or record.in_flight_requests > 0:
                continue
            if session_id in self._close_retry_exhausted:
                continue
            goal = CloseInferenceSession.Goal()
            goal.session_id = session_id
            goal.session_generation = record.session_generation
            deadline_ns = time.time_ns() + self._default_request_timeout_ns
            self._set_absolute_deadline(goal, deadline_ns)
            try:
                retry_result = self._close_once(_MaintenanceGoalHandle(goal), None)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"quarantined Close retry failed for {session_id}: {exc}")
                continue
            if (
                retry_result.outcome.value == InferenceOutcome.NOT_STARTED
                and retry_result.error.code == "close_in_progress"
            ):
                # Another Close is concurrently running (reentrant sweep or an
                # external action): this attempt never dispatched, so the
                # bounded retry slot stays available for the next sweep.
                continue
            self._close_retry_exhausted.add(session_id)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    dependencies = build_model_service_runtime_dependencies()
    node = GlobalInferenceSchedulerNode(
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        dependencies.providers.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
