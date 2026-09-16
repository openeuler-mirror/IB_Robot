"""ScheduledActionDispatcher for the scheduler-enabled control-plane path.

Implements the WAITING_READY/STOPPED/OPENING/ACTIVE/CLOSING/FAILED
state machine that owns the single product session, calls the global Scheduler's
Open/ScheduledDispatchInfer/Close, runs the live control loop with zero
obs_timestamp, performs result dedup by (session_id, generation, request_id),
and exposes the stable /action_dispatcher/{start_evaluate, stop_evaluate,
get_status, restart_session} services.

Shares action-chunk normalization, TopicExecutor and temporal smoothing with
the legacy action_dispatcher_node; it does NOT share ROS action/session state.
Safe-stop is scheduled-path behavior so the disabled legacy path remains
unchanged. When scheduler.enable=true this executable is launched instead of
action_dispatcher_node; the two never coexist.

The dispatcher consumes whole-graph action chunks from the scheduled pipeline
entrypoint and does not own model execution.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import uuid
from enum import Enum

import numpy as np
import rclpy
import rclpy.action
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty, Trigger

from action_dispatch.action_chunk import validate_action_chunk, validate_execution_horizon
from action_dispatch.active_plan import ActivePlan, PlanSource
from action_dispatch.chunk_planning import create_chunk_planner
from action_dispatch.contract_binding import resolve_joint_feedback
from action_dispatch.executors.topic import capture_event, capture_time, capture_traces, execution_trace
from action_dispatch.policy_admission import PolicyAdmission
from action_dispatch.safe_stop import (
    JointSnapshot,
    SafeStopError,
    build_safe_stop_plan,
    construct_safety_command,
    validate_joint_state,
)
from action_dispatch.schedulers.continuous import should_replenish_plan
from action_dispatch.temporal_smoother import TemporalSmootherManager
from action_dispatch.topic_executor import TopicExecutor
from ibrobot_msgs.action import (
    CloseInferenceSession,
    OpenInferenceSession,
    ScheduledDispatchInfer,
)
from ibrobot_tracing import get_trace_emitter
from robot_config.dispatch_strategies import (
    DispatchStrategyError,
    resolve_dispatch_strategies,
)

trace = get_trace_emitter("ib_trace.dispatch", component_id="action_dispatcher")


def _goal_flow_id(goal_handle, fallback: str) -> str:
    try:
        goal_id = bytes(goal_handle.goal_id.uuid)
    except (AttributeError, TypeError, ValueError):
        return fallback
    return f"{fallback}:{goal_id.hex()}" if any(goal_id) else fallback


class DispatcherState(str, Enum):
    WAITING_READY = "waiting_ready"
    STOPPED = "stopped"
    OPENING = "opening"
    ACTIVE = "active"
    CLOSING = "closing"
    FAILED = "failed"


class ScheduledActionDispatcherNode(Node):
    """Scheduled-path product dispatcher. Node name `/action_dispatcher`."""

    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__("action_dispatcher", parameter_overrides=parameter_overrides)
        self._load_parameters()
        self._joint_state_qos = rclpy.qos.qos_profile_sensor_data
        self._blending_update = None
        self.add_on_set_parameters_callback(self._validate_blending_update)
        self._load_contract_and_plan()
        self._state_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._state: DispatcherState = DispatcherState.WAITING_READY
        self._session_id: str = ""
        self._session_generation: int = 0
        # received-result set: (session_id, generation, request_id)
        self._received_results: set[tuple] = set()
        self._last_action: np.ndarray | None = None
        self._joint_snapshot = JointSnapshot()
        self._inflight_request_id: str = ""
        self._inflight_deadline_utc_ns = 0
        self._inflight_observation_time_ns = 0
        self._inflight_goal_handle = None
        self._plan_length_at_inference_start = 0
        self._trace_action_steps = os.environ.get("IB_TRACE_ACTION_STEPS", "first").lower()
        self._startup_started_ns = time.monotonic_ns()
        self._failure_handling = False
        self._pending_failure: tuple[str | None, str, int, str] | None = None
        self._shutdown_started = False
        self._pending_open_session_id = ""
        self._pending_open_completion: threading.Event | None = None
        self._close_after_open = False
        self._close_in_progress = False
        self._close_completion: threading.Event | None = None
        self._close_success = False
        self._close_identity: tuple[str, int] | None = None

        self._executor = TopicExecutor(self, {"action_specs": self._action_specs})
        if not self._executor.initialize():
            raise RuntimeError("failed to initialize TopicExecutor")
        self._smoother = TemporalSmootherManager(
            enabled=self._smoothing_enabled,
            temporal_ensemble_coeff=self._ensemble_coeff,
            chunk_size=self._chunk_size,
            device=self._smoothing_device,
        )
        # Strategy instances are constructed once at init (matching the
        # scheduler/executor registry wiring) so stateful follow-up
        # strategies can hold cross-request state; per-tick code selects
        # instances, never types. The internal smoothing flag is derived from
        # the authoritative blending strategy.
        self._chunk_planner = create_chunk_planner(self._chunking_strategy)
        # Scheduled toggles retain the queue but discard the disabled smoother.
        # Each retained store therefore owns its own source and watermark.
        self._queue_plan = ActivePlan(capacity=self._queue_size, watermark=self._watermark, overflow="fail_closed")
        self._smoothed_plan = ActivePlan(capacity=self._queue_size, watermark=self._watermark, smoother=self._smoother)
        self._control_group = MutuallyExclusiveCallbackGroup()
        self._client_group = ReentrantCallbackGroup()

        # Contract-validating JointState callback.
        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._joint_cb,
            self._joint_state_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        # Services: stable names shared with legacy dispatcher (never coexist).
        self._start_srv = self.create_service(
            Trigger, "~/start_evaluate", self._start_cb, callback_group=ReentrantCallbackGroup()
        )
        self._stop_srv = self.create_service(
            Trigger, "~/stop_evaluate", self._stop_cb, callback_group=ReentrantCallbackGroup()
        )
        self._get_status_srv = self.create_service(
            Trigger, "~/get_status", self._get_status_cb, callback_group=ReentrantCallbackGroup()
        )
        self._toggle_smoothing_srv = self.create_service(
            Empty, "~/toggle_smoothing", self._toggle_smoothing_cb, callback_group=ReentrantCallbackGroup()
        )
        self._restart_srv = self.create_service(
            Trigger, "~/restart_session", self._restart_cb, callback_group=ReentrantCallbackGroup()
        )
        self._queue_size_pub = self.create_publisher(Int32, "~/queue_size", 10)
        self._smoothing_pub = self.create_publisher(Bool, "~/smoothing_enabled", 10)

        # Product callers only touch the Global Scheduler action endpoints.
        self._open_client = rclpy.action.ActionClient(
            self, OpenInferenceSession, self._open_endpoint, callback_group=self._client_group
        )
        self._dispatch_client = rclpy.action.ActionClient(
            self, ScheduledDispatchInfer, self._dispatch_endpoint, callback_group=self._client_group
        )
        self._close_client = rclpy.action.ActionClient(
            self, CloseInferenceSession, self._close_endpoint, callback_group=self._client_group
        )
        self._readiness_client = self.create_client(
            Trigger, self._readiness_endpoint, callback_group=self._client_group
        )
        self._runtime_admission = None
        if self._robot_config.runtime.get("provider"):
            self._runtime_admission = PolicyAdmission(
                self, self._robot_config_path, self._robot_config.runtime, self._state_lock, self._runtime_revoked
            )

        # Live dispatches use zero obs_timestamp.
        self._control_timer = self.create_timer(
            1.0 / self._control_hz, self._control_loop, callback_group=self._control_group
        )
        # Readiness polling before Open.
        self._readiness_timer = self.create_timer(0.5, self._readiness_poll, callback_group=ReentrantCallbackGroup())
        self._failure_timer = self.create_timer(
            0.01,
            self._drain_pending_failure,
            callback_group=MutuallyExclusiveCallbackGroup(),
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        # Both modes wait for verified Scheduler readiness. Navigation mode then
        # enters STOPPED; normal mode opens immediately.

    # ------------------------------------------------------------------
    # parameters + contract
    # ------------------------------------------------------------------

    def _load_parameters(self) -> None:
        if "temporal_smoothing_enabled" in self._parameter_overrides:
            raise ValueError("temporal_smoothing_enabled was removed; use blending_strategy=none|temporal_ensemble")
        for name, default in (
            ("robot_config_path", ""),
            ("joint_state_topic", "/joint_states"),
            ("queue_size", 100),
            ("watermark_threshold", 20),
            ("control_frequency", 100.0),
            ("chunk_size", 100),
            ("temporal_ensemble_coeff", 0.01),
            ("smoothing_device", ""),
            ("navigation_mode", False),
        ):
            self.declare_parameter(name, default)
        for name, default in (
            ("scheduler_readiness_endpoint", "/inference/scheduler/ready"),
            ("open_session_endpoint", "/inference/session/open"),
            ("dispatch_endpoint", "/inference/dispatch"),
            ("close_session_endpoint", "/inference/session/close"),
            ("inference_pipeline", ""),
            ("inference_fallback_chain", "[]"),
            ("inference_retry_json", "{}"),
            ("inference_prompt", ""),
            # Chunk-planning and action-blending strategy selection. SSOT
            # ``dispatch.chunking``/``dispatch.blending`` are passed through by
            # launch builders. Standalone fusion defaults to none.
            ("chunking_strategy", ""),
            ("blending_strategy", "none"),
            ("executor_type", "topic"),
            ("scheduler_mode", "continuous"),
        ):
            self.declare_parameter(name, default)
        for name in (
            "startup_readiness_timeout_ns",
            "default_open_timeout_ns",
            "default_request_timeout_ns",
            "inference_priority",
        ):
            self.declare_parameter(name, 0)
        self._robot_config_path = str(self.get_parameter("robot_config_path").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._queue_size = int(self.get_parameter("queue_size").value)
        self._watermark = int(self.get_parameter("watermark_threshold").value)
        self._control_hz = float(self.get_parameter("control_frequency").value)
        self._chunk_size = int(self.get_parameter("chunk_size").value)
        self._ensemble_coeff = float(self.get_parameter("temporal_ensemble_coeff").value)
        self._smoothing_device = str(self.get_parameter("smoothing_device").value) or None
        self._navigation_mode = bool(self.get_parameter("navigation_mode").value)
        self._readiness_endpoint = str(self.get_parameter("scheduler_readiness_endpoint").value)
        self._open_endpoint = str(self.get_parameter("open_session_endpoint").value)
        self._dispatch_endpoint = str(self.get_parameter("dispatch_endpoint").value)
        self._close_endpoint = str(self.get_parameter("close_session_endpoint").value)
        self._inference_pipeline = str(self.get_parameter("inference_pipeline").value)
        self._inference_fallback_chain = json.loads(str(self.get_parameter("inference_fallback_chain").value))
        self._inference_priority = int(self.get_parameter("inference_priority").value)
        self._inference_prompt = str(self.get_parameter("inference_prompt").value)
        retry = json.loads(str(self.get_parameter("inference_retry_json").value))
        self._retry_max_attempts = int(retry.get("max_not_started_attempts", 3))
        self._retry_initial_ms = int(retry.get("initial_backoff_ms", 50))
        self._retry_max_ms = int(retry.get("max_backoff_ms", 500))
        self._strategy_selection = resolve_dispatch_strategies(
            executor_type=self.get_parameter("executor_type").value,
            scheduler_mode=self.get_parameter("scheduler_mode").value,
            chunking=self.get_parameter("chunking_strategy").value,
            blending=self.get_parameter("blending_strategy").value,
            entrypoint="scheduled",
        )
        self._chunking_strategy = self._strategy_selection.chunking
        self._smoothing_enabled = self._strategy_selection.blending == "temporal_ensemble"
        self._startup_readiness_timeout_ns = int(self.get_parameter("startup_readiness_timeout_ns").value)
        self._default_open_timeout_ns = int(self.get_parameter("default_open_timeout_ns").value)
        self._default_request_timeout_ns = int(self.get_parameter("default_request_timeout_ns").value)
        if self._queue_size <= 0 or self._chunk_size <= 0:
            raise ValueError("queue_size and chunk_size must be positive")
        if self._chunk_size > self._queue_size:
            raise ValueError("chunk_size must not exceed queue_size")

    def _load_contract_and_plan(self) -> None:
        from robot_config import load_robot_config
        from robot_config.contract_utils import iter_specs

        rc = load_robot_config(self._robot_config_path)
        self._robot_config = rc
        contract = rc.to_contract()
        self._action_specs = [s for s in iter_specs(contract) if s.is_action]
        joint_order = list(rc.joints.get("all") or []) if hasattr(rc, "joints") and rc.joints else []
        self._safe_stop_plan = build_safe_stop_plan(action_specs=self._action_specs, joint_order=joint_order)
        self._joint_state_topic, self._joint_state_qos, self._joint_max_age_ns = resolve_joint_feedback(
            contract, self._joint_state_topic, self._joint_state_qos, rc.runtime
        )

    # ------------------------------------------------------------------
    # Contract-validating joint callback; snapshot under lock.
    # ------------------------------------------------------------------

    def _joint_cb(self, msg: JointState) -> None:
        joint_order = self._safe_stop_plan.joint_order
        if not joint_order:
            return
        snap = validate_joint_state(
            joint_names=list(msg.name), positions=list(msg.position), expected_joint_order=joint_order
        )
        with self._state_lock:
            if snap.valid:
                snap.received_monotonic_ns = time.monotonic_ns()
                self._joint_snapshot = snap

    # ------------------------------------------------------------------
    # Poll readiness before Open.
    # ------------------------------------------------------------------

    def _readiness_poll(self) -> None:
        with self._state_lock:
            if self._state != DispatcherState.WAITING_READY:
                self._readiness_timer.cancel()
                return
            if time.monotonic_ns() - self._startup_started_ns >= self._startup_readiness_timeout_ns:
                self.get_logger().fatal("scheduler readiness deadline expired; terminating scheduled topology")
                self._readiness_timer.cancel()
                raise RuntimeError("scheduler startup readiness timeout")
        if not self._readiness_client.wait_for_service(timeout_sec=0.1):
            return
        future = self._readiness_client.call_async(Trigger.Request())

        def _cb(fut) -> None:
            try:
                resp = fut.result()
            except Exception:  # noqa: BLE001
                return
            if resp is None or not resp.success:
                return
            with self._state_lock:
                if self._state != DispatcherState.WAITING_READY:
                    return
                if self._navigation_mode or getattr(self, "_runtime_admission", None) is not None:
                    self._set_state(DispatcherState.STOPPED)
                else:
                    self._open_new_session()

        future.add_done_callback(_cb)

    # ------------------------------------------------------------------
    # State transitions.
    # ------------------------------------------------------------------

    def _set_state(self, state: DispatcherState) -> None:
        with self._state_lock:
            self._state = state
            self.get_logger().info(f"state -> {state.value}")

    def _open_new_session(
        self,
        *,
        session_id: str | None = None,
        completion: threading.Event | None = None,
    ) -> None:
        """Acquire the runtime before opening a fresh scheduler session."""
        admission = getattr(self, "_runtime_admission", None)
        if admission is None:
            self._open_admitted_session(session_id=session_id, completion=completion)
            return
        with self._state_lock:
            if self._state not in (DispatcherState.WAITING_READY, DispatcherState.STOPPED):
                if completion is not None:
                    completion.set()
                return

        def admitted(success, message):
            if success:
                self._open_admitted_session(session_id=session_id, completion=completion)
            else:
                self.get_logger().warning(f"policy admission rejected: {message}")
                if completion is not None:
                    completion.set()

        admission.acquire(admitted)

    def _runtime_revoked(self):
        with self._state_lock:
            self._cancel_inflight_dispatch()
            self._clear_inflight_locked()
            self._clear_plans_locked()
            self._last_action = None
            self._executor.invalidate_pending()
            # Preserve scheduler identity for explicit Close/restart reconciliation.
            self._close_after_open = True
            self._state = DispatcherState.FAILED if self._session_id else DispatcherState.STOPPED

    def _open_admitted_session(self, *, session_id=None, completion=None):
        with self._state_lock:
            if self._state not in (DispatcherState.WAITING_READY, DispatcherState.STOPPED):
                return
            self._state = DispatcherState.OPENING
            session_id = session_id or str(uuid.uuid4())
            self._session_id = session_id
            self._pending_open_session_id = session_id
            self._pending_open_completion = threading.Event()
            self._close_after_open = False
            self._received_results.clear()
            self._clear_inflight_locked()
            self._clear_plans_locked()
        goal = OpenInferenceSession.Goal()
        goal.session_id = session_id
        open_deadline_ns = time.time_ns() + self._default_open_timeout_ns
        goal.deadline.sec, goal.deadline.nanosec = divmod(open_deadline_ns, 1_000_000_000)
        if not self._open_client.wait_for_server(timeout_sec=0.1):
            self._handle_open_not_started(session_id, completion)
            return
        send_future = self._open_client.send_goal_async(goal)

        def _goal_response_cb(result_future) -> None:
            try:
                goal_handle = result_future.result()
                if goal_handle is None or not goal_handle.accepted:
                    self._handle_open_not_started(session_id, completion)
                    return
                goal_handle.get_result_async().add_done_callback(
                    lambda future: self._open_result_callback(future, session_id, completion)
                )
            except Exception:  # noqa: BLE001
                if self._tracks_pending_open(session_id):
                    self._complete_pending_open(session_id, completion)
                    self._fail_and_close("Open outcome unknown", session_id=session_id)
                else:
                    self._complete_pending_open(session_id, completion)

        send_future.add_done_callback(_goal_response_cb)

    def _open_result_callback(self, future, session_id: str, completion: threading.Event | None) -> None:
        try:
            result = future.result().result
        except Exception:  # noqa: BLE001
            if self._tracks_pending_open(session_id):
                self._complete_pending_open(session_id, completion)
                self._fail_and_close("Open result outcome unknown", session_id=session_id)
            else:
                self._complete_pending_open(session_id, completion)
            return
        if not self._tracks_pending_open(session_id):
            self._complete_pending_open(session_id, completion)
            return
        if result.session_id != session_id:
            self._complete_pending_open(session_id, completion)
            self._fail_and_close("Open result identity mismatch", session_id=session_id)
            return
        if not result.success and result.outcome.value == 1 and result.error.recoverable:
            self._handle_open_not_started(session_id, completion)
            return
        close_after_open, failure_reason = self._on_open_result(result, session_id)
        self._complete_pending_open(session_id, completion)
        if failure_reason:
            self._fail_and_close(failure_reason, session_id=session_id)
        elif close_after_open:
            self._begin_close_session(expected_identity=(session_id, int(result.session_generation)))

    def _tracks_pending_open(self, session_id: str) -> bool:
        with self._state_lock:
            return self._pending_open_session_id == session_id and self._session_id == session_id

    def _complete_pending_open(self, session_id: str, completion: threading.Event | None) -> None:
        with self._state_lock:
            if self._pending_open_session_id == session_id:
                self._pending_open_session_id = ""
                if self._pending_open_completion is not None:
                    self._pending_open_completion.set()
        if completion is not None:
            completion.set()

    def _handle_open_not_started(self, session_id: str, completion: threading.Event | None) -> None:
        with self._state_lock:
            if self._pending_open_session_id == session_id and self._session_id == session_id:
                if self._state == DispatcherState.OPENING:
                    self._state = DispatcherState.STOPPED
                self._session_id = ""
                self._session_generation = 0
                self._close_after_open = False
        self._complete_pending_open(session_id, completion)

    def _on_open_result(self, result, session_id: str) -> tuple[bool, str]:
        failure_reason = ""
        close_after_open = False
        with self._state_lock:
            if self._pending_open_session_id != session_id or self._session_id != session_id:
                return False, ""
            if result.success:
                if int(result.session_generation) <= 0:
                    failure_reason = "Open result identity mismatch"
                else:
                    self._session_generation = int(result.session_generation)
                    close_after_open = self._close_after_open or self._state != DispatcherState.OPENING
                    if not close_after_open:
                        self._state = DispatcherState.ACTIVE
                        self.get_logger().info(f"session active gen={self._session_generation}")
            elif result.outcome.value == 1 and result.error.recoverable:
                # Recoverable NOT_STARTED returns to STOPPED for the next start.
                if self._state == DispatcherState.OPENING:
                    self._state = DispatcherState.STOPPED
                self._session_id = ""
                self._session_generation = 0
                self._close_after_open = False
            else:
                if int(result.session_generation) > 0:
                    self._session_generation = int(result.session_generation)
                failure_reason = result.error.code or "Open failed after downstream acceptance"
        return close_after_open, failure_reason

    def _start_cb(self, _req, resp: Trigger.Response) -> Trigger.Response:
        return self._run_lifecycle_service(self._start_cb_locked, _req, resp)

    def _run_lifecycle_service(self, callback, request, response: Trigger.Response) -> Trigger.Response:
        # A contending service must leave a worker available for Open/Close.
        with capture_traces(trace.enabled):
            if not self._lifecycle_lock.acquire(blocking=False):
                response.success = False
                response.message = "lifecycle operation in progress"
                return response
            try:
                return callback(request, response)
            finally:
                self._lifecycle_lock.release()

    def _start_cb_locked(self, _req, resp: Trigger.Response) -> Trigger.Response:
        with self._state_lock:
            state = self._state
        if state == DispatcherState.ACTIVE:
            admission = getattr(self, "_runtime_admission", None)
            if admission is not None:
                completion = threading.Event()

                def admitted(success, message):
                    resp.success, resp.message = success, message
                    completion.set()

                admission.acquire(admitted, own_active_session=True)
                if not completion.wait(2.0):
                    admission.cancel()
                return resp
            resp.success = True
            return resp
        if state == DispatcherState.STOPPED:
            completion = threading.Event()
            self._open_new_session(completion=completion)
            if not completion.wait(self._default_open_timeout_ns / 1_000_000_000):
                admission = getattr(self, "_runtime_admission", None)
                if admission is not None:
                    admission.cancel()
            with self._state_lock:
                resp.success = self._state == DispatcherState.ACTIVE
                resp.message = "" if resp.success else f"Open did not complete ({self._state.value})"
            return resp
        if state == DispatcherState.FAILED:
            resp.success = False
            resp.message = "FAILED: must Close/lease-reconcile before start"
            return resp
        resp.success = False
        resp.message = f"cannot start from {state.value}"
        return resp

    def _stop_cb(self, _req, resp: Trigger.Response) -> Trigger.Response:
        return self._run_lifecycle_service(self._stop_cb_locked, _req, resp)

    def _stop_cb_locked(self, _req, resp: Trigger.Response) -> Trigger.Response:
        # Safe-stop first, then Close; both succeed -> STOPPED.
        safe_ok = self._safe_stop()
        close_ok = self._close_session_sync()
        admission = getattr(self, "_runtime_admission", None)
        if admission is not None:
            safe_ok = admission.release() and safe_ok
        with self._state_lock:
            self._state = DispatcherState.STOPPED if (safe_ok and close_ok) else DispatcherState.FAILED
        resp.success = safe_ok and close_ok
        if not resp.success:
            resp.message = "safe_stop or Close failed; FAILED"
        return resp

    def _restart_cb(self, _req, resp: Trigger.Response) -> Trigger.Response:
        return self._run_lifecycle_service(self._restart_cb_locked, _req, resp)

    def _restart_cb_locked(self, _req, resp: Trigger.Response) -> Trigger.Response:
        # Safe-stop -> Close -> clear local -> Open new UUID.
        safe_ok = self._safe_stop()
        close_ok = self._close_session_sync()
        admission = getattr(self, "_runtime_admission", None)
        if admission is not None:
            safe_ok = admission.release() and safe_ok
        with self._state_lock:
            self._received_results.clear()
            self._clear_plans_locked()
            self._last_action = None
        if not safe_ok:
            # A safe-stop failure still closes, but must not open a new session.
            self._set_state(DispatcherState.FAILED)
            resp.success = False
            resp.message = "safe_stop failed; Close attempted, no new session"
            return resp
        if not close_ok:
            self._set_state(DispatcherState.FAILED)
            resp.success = False
            resp.message = "Close failed; no new session"
            return resp
        self._set_state(DispatcherState.STOPPED)
        completion = threading.Event()
        self._open_new_session(completion=completion)
        if not completion.wait(self._default_open_timeout_ns / 1_000_000_000) and admission is not None:
            admission.cancel()
        with self._state_lock:
            resp.success = self._state == DispatcherState.ACTIVE
            if not resp.success:
                resp.message = f"Open did not complete ({self._state.value})"
        return resp

    def _get_status_cb(self, _req, resp: Trigger.Response) -> Trigger.Response:
        with self._state_lock:
            resp.success = True
            resp.message = self._state.value
        return resp

    def _validate_blending_update(self, parameters):
        for parameter in parameters:
            if parameter.name == "blending_strategy":
                if parameter is not self._blending_update:
                    return SetParametersResult(successful=False, reason="Use ~/toggle_smoothing at runtime")
                # Single-use object identity, not a flag that concurrent writes can inherit.
                self._blending_update = None
        return SetParametersResult(successful=True)

    def _toggle_smoothing_cb(self, _request, response: Empty.Response) -> Empty.Response:
        with self._state_lock:
            enabled = not self._smoothing_enabled
            current = self._strategy_selection
            try:
                selection = resolve_dispatch_strategies(
                    executor_type=current.executor_type,
                    scheduler_mode=current.scheduler_mode,
                    chunking=current.chunking,
                    blending="temporal_ensemble" if enabled else "none",
                    entrypoint="scheduled",
                )
            except DispatchStrategyError as exc:
                self.get_logger().error(f"Temporal smoothing toggle rejected: {exc}")
                return response
            from rclpy.parameter import Parameter

            parameter = Parameter("blending_strategy", value=selection.blending)
            self._blending_update = parameter
            try:
                result = self.set_parameters_atomically([parameter])
            finally:
                self._blending_update = None
            if not result.successful:
                self.get_logger().error(f"Temporal smoothing toggle rejected: {result.reason}")
                return response
            old_remaining = self._active_plan.snapshot().remaining
            self._strategy_selection = selection
            self._smoothing_enabled = selection.blending == "temporal_ensemble"
            self._smoothed_plan.set_smoothing_enabled(self._smoothing_enabled)
            if not self._smoothing_enabled:
                self._smoothed_plan.clear()
            if self._inflight_request_id:
                # Keep request-relative consumption unchanged when switching
                # between retained stores of different lengths.
                self._plan_length_at_inference_start += self._active_plan.snapshot().remaining - old_remaining
            enabled = self._smoothing_enabled
        self._smoothing_pub.publish(Bool(data=enabled))
        self.get_logger().info(f"Temporal smoothing {'ENABLED' if enabled else 'DISABLED'}")
        return response

    # ------------------------------------------------------------------
    # Control loop with zero obs_timestamp dispatch.
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        with self._state_lock:
            # One-way lifecycle gate: closed sessions block inference requests
            # and action submission until the lifecycle reopens dispatch.
            if self._state is not DispatcherState.ACTIVE:
                return
            snapshot = self._active_plan.snapshot()
            q_size = snapshot.remaining
            replenish = should_replenish_plan(
                q_size, snapshot.watermark, inference_in_progress=bool(self._inflight_request_id)
            )
        self._queue_size_pub.publish(Int32(data=q_size))
        # Watermark replenishment via the shared continuous per-tick rule. The
        # scheduled path has no policy-reset gate, so that flag stays False.
        # An adaptive-horizon plan carries its own replenishment watermark
        # (0: re-request once the prefix is consumed) inside the ActivePlan
        # snapshot instead of prefetching.
        if replenish:
            self._request_dispatch()
        # execute one action per tick
        self._execute_next_action()

    def _request_dispatch(self, *, attempt: int = 0, replace_request_id: str = "") -> None:
        with self._state_lock:
            if self._state != DispatcherState.ACTIVE:
                if replace_request_id and self._inflight_request_id == replace_request_id:
                    self._clear_inflight_locked()
                return
            if replace_request_id and self._inflight_request_id != replace_request_id:
                return
            if not replace_request_id and self._inflight_request_id:
                return
            request_id = str(uuid.uuid4())
            if replace_request_id:
                deadline_utc_ns = self._inflight_deadline_utc_ns
                observation_time_ns = self._inflight_observation_time_ns
            else:
                deadline_utc_ns = time.time_ns() + self._default_request_timeout_ns
                self._inflight_deadline_utc_ns = deadline_utc_ns
                observation_time_ns = self.get_clock().now().nanoseconds
                self._inflight_observation_time_ns = observation_time_ns
            if deadline_utc_ns <= time.time_ns():
                request_id = self._inflight_request_id
                expired = True
            else:
                expired = False
                self._inflight_request_id = request_id
                self._inflight_goal_handle = None
                if not replace_request_id:
                    self._plan_length_at_inference_start = self._current_plan_length_locked()
            sid = self._session_id
            gen = self._session_generation
            if trace.enabled:
                snapshot = self._active_plan.snapshot()
        if expired:
            self._fail_current_request(request_id, sid, gen, "scheduled dispatch deadline expired before retry")
            return
        goal = ScheduledDispatchInfer.Goal()
        goal.request_id = request_id
        goal.session_id = sid
        goal.session_generation = gen
        goal.target_pipeline_id = self._inference_pipeline
        goal.fallback_chain = list(self._inference_fallback_chain) if self._inference_priority == 0 else []
        goal.priority = self._inference_priority
        goal.prompt = self._inference_prompt
        goal.obs_timestamp.sec, goal.obs_timestamp.nanosec = divmod(observation_time_ns, 1_000_000_000)
        goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_utc_ns, 1_000_000_000)
        if trace.enabled:
            trace.event(
                "dispatch_request",
                origin="built-in",
                trace_id=request_id,
                component_id="action_dispatcher.request",
                queue_size=snapshot.remaining,
                watermark=snapshot.watermark,
                pipeline_id=self._inference_pipeline,
                priority=self._inference_priority,
                attempt=attempt + 1,
            )
        if not self._dispatch_client.wait_for_server(timeout_sec=0.1):
            self._retry_dispatch_not_started(request_id, attempt)
            return
        send_timestamp_ns = time.time_ns() if trace.enabled else None
        send_future = self._dispatch_client.send_goal_async(goal)

        def _goal_response_cb(result_future) -> None:
            try:
                gh = result_future.result()
            except Exception:  # noqa: BLE001
                self._on_dispatch_unknown(request_id, sid, gen)
                return
            if trace.enabled:
                trace.flow_send(
                    "scheduled_dispatch_to_scheduler",
                    _goal_flow_id(gh, request_id),
                    timestamp_ns=send_timestamp_ns,
                    trace_id=request_id,
                    component_id="action_dispatcher.request",
                    pipeline_id=self._inference_pipeline,
                    session_id=sid,
                    session_generation=gen,
                    priority=self._inference_priority,
                    attempt=attempt + 1,
                )
            try:
                if gh is None or not gh.accepted:
                    self.get_logger().error("scheduled dispatch goal rejected; endpoint goal slots are full")
                    self._dispatch_rejected(request_id)
                    return
                with self._state_lock:
                    is_current = (
                        self._state == DispatcherState.ACTIVE
                        and self._inflight_request_id == request_id
                        and self._session_id == sid
                        and self._session_generation == gen
                    )
                    if is_current:
                        self._inflight_goal_handle = gh
                if not is_current:
                    gh.cancel_goal_async()
                    return
                gh.get_result_async().add_done_callback(
                    lambda future: self._dispatch_result_callback(future, request_id, sid, gen, attempt)
                )
            except Exception:  # noqa: BLE001
                self._on_dispatch_unknown(request_id, sid, gen)

        send_future.add_done_callback(_goal_response_cb)

    def _dispatch_result_callback(
        self, future, request_id: str, session_id: str, session_generation: int, attempt: int
    ) -> None:
        with self._state_lock:
            goal_handle = self._inflight_goal_handle if trace.enabled else None
            if self._inflight_request_id == request_id:
                self._inflight_goal_handle = None
        try:
            result = future.result().result
        except Exception:  # noqa: BLE001
            self._on_dispatch_unknown(request_id, session_id, session_generation)
            return
        if not self._request_is_current(request_id, session_id, session_generation):
            return
        if (
            result.request_id != request_id
            or result.session_id != session_id
            or int(result.session_generation) != session_generation
        ):
            self._fail_current_request(request_id, session_id, session_generation, "scheduled result identity mismatch")
            return
        if trace.enabled:
            trace.flow_receive(
                "scheduler_result_to_dispatcher",
                _goal_flow_id(goal_handle, request_id),
                trace_id=request_id,
                component_id="action_dispatcher.decode",
                pipeline_id=result.pipeline_id,
                session_id=session_id,
                session_generation=session_generation,
                priority=self._inference_priority,
            )
        if not result.success and result.outcome.value == 1 and result.error.recoverable:
            self._retry_dispatch_not_started(request_id, attempt, result.error.code)
            return
        self._on_dispatch_result(result)

    def _request_is_current(self, request_id: str, session_id: str, session_generation: int) -> bool:
        with self._state_lock:
            return (
                self._state == DispatcherState.ACTIVE
                and self._inflight_request_id == request_id
                and self._session_id == session_id
                and self._session_generation == session_generation
            )

    def _dispatch_rejected(self, request_id: str) -> None:
        with self._state_lock:
            sid, gen = self._session_id, self._session_generation
        self._fail_current_request(
            request_id, sid, gen, "scheduled dispatch rejected because endpoint goal slots are full"
        )

    def _retry_dispatch_not_started(self, request_id: str, attempt: int, code: str = "") -> None:
        non_retryable = {
            "no_feasible_deadline",
            "no_session_capacity",
            "pipeline_busy",
            "request_canceled",
            "session_closing",
            "unsupported_priority",
            "hardware_priority_unavailable",
        }
        if code in non_retryable:
            with self._state_lock:
                sid, gen = self._session_id, self._session_generation
            self._fail_current_request(request_id, sid, gen, f"scheduled dispatch rejected: {code}")
            return
        if attempt >= self._retry_max_attempts:
            with self._state_lock:
                sid, gen = self._session_id, self._session_generation
            self._fail_current_request(request_id, sid, gen, "scheduled dispatch retries exhausted")
            return
        delay_ms = min(self._retry_initial_ms * (2**attempt), self._retry_max_ms)
        timer = None

        def _retry() -> None:
            timer.cancel()
            self._request_dispatch(attempt=attempt + 1, replace_request_id=request_id)

        timer = self.create_timer(max(0.001, delay_ms / 1000.0), _retry, callback_group=ReentrantCallbackGroup())

    def _on_dispatch_result(self, result) -> None:
        with capture_traces(trace.enabled):
            return self._on_dispatch_result_observed(result)

    def _on_dispatch_result_observed(self, result) -> None:
        source = PlanSource(
            result.request_id, session_id=result.session_id, session_generation=int(result.session_generation)
        )
        with self._state_lock:
            key = (result.session_id, int(result.session_generation), result.request_id)
            if key in self._received_results or not self._request_is_current(
                source.request_id, source.session_id, source.session_generation
            ):
                return
            result_timestamp_ns = time.time_ns() if trace.enabled else None
        if trace.enabled:
            trace.event(
                "dispatch_result",
                timestamp_ns=result_timestamp_ns,
                origin="built-in",
                trace_id=source.request_id,
                component_id="action_dispatcher.decode",
                success=result.success,
                pipeline_id=result.pipeline_id,
                policy_total_ms=result.inference_latency_ms,
                chunk_size=result.chunk_size,
            )
        failure_reason = ""
        if result.success:
            # Observation context construction and teardown cannot be classified
            # as an invalid business chunk by the existing error handler.
            with trace.trace_context(source.request_id, component_id="action_dispatcher.decode"):
                try:
                    self._enqueue_chunk(
                        result.action_chunk,
                        reported_chunk_size=int(result.chunk_size),
                        source=source,
                        execution_horizon=int(result.execution_horizon),
                    )
                except (TypeError, ValueError, RuntimeError, MemoryError) as exc:
                    self.get_logger().error(f"invalid scheduled action chunk: {exc}")
                    failure_reason = "invalid scheduled action chunk"
        else:
            failure_reason = result.error.code or "scheduled dispatch failed"
        if failure_reason:
            self._fail_current_request(source.request_id, source.session_id, source.session_generation, failure_reason)

    def _on_dispatch_unknown(self, request_id: str, session_id: str, session_generation: int) -> None:
        with self._state_lock:
            if self._inflight_request_id == request_id:
                self._inflight_goal_handle = None
        self._fail_current_request(
            request_id, session_id, session_generation, f"scheduled dispatch outcome unknown: {request_id}"
        )

    def _fail_current_request(self, request_id, session_id, session_generation, reason):
        if not self._request_is_current(request_id, session_id, session_generation):
            return
        self._try_failure(request_id, session_id, session_generation, reason)

    def _try_failure(self, request_id, session_id, session_generation, reason):
        # Never wait for a lifecycle owner that may need this worker for Close.
        with capture_traces(trace.enabled):
            acquired = self._lifecycle_lock.acquire(blocking=False)
            try:
                with self._state_lock:
                    if (self._session_id, self._session_generation) != (session_id, session_generation):
                        return
                    if request_id is not None and not self._request_is_current(
                        request_id, session_id, session_generation
                    ):
                        return
                    if not acquired:
                        # Only the current identity can occupy this bounded mailbox.
                        self._pending_failure = (request_id, session_id, session_generation, reason)
                        return
                    if request_id is not None:
                        self._state = DispatcherState.CLOSING
                self._fail_and_close_locked(reason)
            finally:
                if acquired:
                    self._lifecycle_lock.release()

    def _drain_pending_failure(self):
        with self._state_lock:
            failure = self._pending_failure
            self._pending_failure = None
        if failure is not None:
            self._try_failure(*failure)

    def _fail_and_close(self, reason: str, *, session_id: str | None = None) -> None:
        with self._state_lock:
            sid, gen = self._session_id, self._session_generation
            if session_id is not None and sid != session_id:
                return
        self._try_failure(None, sid, gen, reason)

    def _fail_and_close_locked(self, reason: str) -> None:
        with self._state_lock:
            if self._failure_handling or self._state == DispatcherState.FAILED:
                return
            self._failure_handling = True
        try:
            self.get_logger().error(reason)
            self._safe_stop()
            self._close_session_sync()
            admission = getattr(self, "_runtime_admission", None)
            if admission is not None:
                admission.release()
            self._set_state(DispatcherState.FAILED)
        finally:
            with self._state_lock:
                self._failure_handling = False

    def _enqueue_chunk(
        self, action_chunk_msg, *, reported_chunk_size: int, source: PlanSource, execution_horizon: int = 0
    ) -> None:
        from tensormsg.converter import TensorMsgConverter

        decode_start = capture_time(monotonic=True) if trace.enabled else None
        decoded_ok = False
        try:
            decoded = TensorMsgConverter.from_variant(action_chunk_msg)
            action = decoded.get("action") if isinstance(decoded, dict) else None
            if action is None:
                raise ValueError("scheduled action chunk is missing 'action'")
            action_np = validate_action_chunk(
                action,
                expected_action_dimension=self._safe_stop_plan.total_positions,
                reported_chunk_size=reported_chunk_size,
            ).array
            decoded_ok = True
        finally:
            if trace.enabled:
                decode_stop = capture_time(monotonic=True)
                capture_event(
                    trace,
                    "dispatch_decode",
                    origin="built-in",
                    trace_id=source.request_id,
                    component_id="action_dispatcher.decode",
                    status="ok" if decoded_ok else "error",
                    decode_ms=(decode_stop - decode_start) * 1000
                    if decode_stop is not None and decode_start is not None
                    else None,
                )
        decode_end_ns = capture_time() if trace.enabled else None
        with self._state_lock:
            key = (source.session_id, source.session_generation, source.request_id)
            if key in self._received_results or not self._request_is_current(
                source.request_id, source.session_id, source.session_generation
            ):
                return
            refill_start_ns = capture_time() if trace.enabled else None
            actions_executed = max(0, self._plan_length_at_inference_start - self._current_plan_length_locked())
            # The result-level execution prefix is chunk-planning input: the
            # auto_horizon strategy truncates the executable plan and switches
            # to consume-then-replan replenishment (ActivePlan applies the
            # candidate's start/stop/watermark transactionally); full_chunk
            # ignores it.
            chunk_plan = self._chunk_planner.plan(
                action_np,
                actions_executed=actions_executed,
                execution_horizon=validate_execution_horizon(execution_horizon, len(action_np)),
            )
            snapshot = self._active_plan.accept(
                chunk_plan, source, action_dimension=self._safe_stop_plan.total_positions
            )
            self._received_results.add(key)
            self._clear_inflight_locked()
            refill_end_ns = capture_time() if trace.enabled else None
        if trace.enabled:
            with capture_traces(trace.enabled):
                capture_event(
                    trace,
                    "flow_send",
                    edge_id="decode_to_queue",
                    flow_id=source.request_id,
                    timestamp_ns=decode_end_ns,
                    trace_id=source.request_id,
                    component_id="action_dispatcher.decode",
                )
                capture_event(
                    trace,
                    "flow_receive",
                    edge_id="decode_to_queue",
                    flow_id=source.request_id,
                    timestamp_ns=refill_start_ns,
                    trace_id=source.request_id,
                    component_id="action_dispatcher.queue",
                )
                capture_event(
                    trace,
                    "queue_refill",
                    trace_id=source.request_id,
                    component_id="action_dispatcher.queue",
                    timestamp_ns=refill_end_ns,
                    origin="built-in",
                    new=chunk_plan.stop - chunk_plan.start,
                    skipped=actions_executed,
                    after=snapshot.remaining,
                    watermark=snapshot.watermark,
                )
                if snapshot.remaining:
                    capture_event(
                        trace,
                        "flow_send",
                        edge_id="queue_to_execute",
                        flow_id=source.request_id,
                        trace_id=source.request_id,
                        component_id="action_dispatcher.queue",
                        timestamp_ns=refill_end_ns,
                    )

    @property
    def _active_plan(self) -> ActivePlan:
        return self._smoothed_plan if self._smoothing_enabled else self._queue_plan

    def _clear_plans_locked(self) -> None:
        self._queue_plan.clear()
        self._smoothed_plan.clear()

    def _current_plan_length_locked(self) -> int:
        return self._active_plan.snapshot().remaining

    def _clear_inflight_locked(self) -> None:
        self._inflight_request_id = ""
        self._inflight_deadline_utc_ns = 0
        self._inflight_observation_time_ns = 0
        self._inflight_goal_handle = None

    def _execute_next_action(self) -> None:
        sample_interval = 0
        if trace.enabled and self._trace_action_steps.startswith("sample:"):
            try:
                sample_interval = max(1, int(self._trace_action_steps.partition(":")[2]))
            except ValueError:
                sample_interval = 0
        with capture_traces(trace.enabled), self._state_lock:
            # One-way lifecycle gate: only an ACTIVE session permits submission.
            if self._state is not DispatcherState.ACTIVE:
                return
            if trace.enabled:
                snapshot = self._active_plan.snapshot()
            blended = self._active_plan.take_action(last_action=self._last_action)
            action = blended.action
            if action is None:
                return
            self._last_action = np.asarray(action, dtype=float).reshape(-1)
            trace_step = trace.enabled and (
                blended.source != "hold"
                and snapshot.source is not None
                and (
                    snapshot.consumed == 0
                    or self._trace_action_steps == "all"
                    or (sample_interval > 0 and snapshot.consumed % sample_interval == 0)
                )
            )
            if trace_step:
                execute_timestamp_ns = time.time_ns()
                execute_start = time.perf_counter()
            # Serialize publication with safe-stop's freeze and safety output.
            if trace_step:
                with execution_trace(
                    snapshot.source.request_id,
                    snapshot.consumed,
                    snapshot.next_position if snapshot.next_position is not None else -1,
                    snapshot.remaining,
                ):
                    self._executor.execute(self._last_action)
            else:
                self._executor.execute(self._last_action)
            if trace_step:
                publish_ms = (time.perf_counter() - execute_start) * 1000.0
                publish_end_ns = time.time_ns()
        if trace_step:
            with trace.trace_context(snapshot.source.request_id, component_id="action_dispatcher.execute"):
                if snapshot.consumed == 0:
                    trace.flow_receive(
                        "queue_to_execute", snapshot.source.request_id, timestamp_ns=execute_timestamp_ns
                    )
                trace.event(
                    "first_action_execute" if snapshot.consumed == 0 else "action_execute",
                    timestamp_ns=execute_timestamp_ns,
                    origin="built-in",
                    consumed_index=snapshot.consumed,
                    execute_index=snapshot.next_position if snapshot.next_position is not None else -1,
                    source=blended.source,
                    queue_before=snapshot.remaining,
                    publish_ms=publish_ms,
                    publish_end_ns=publish_end_ns,
                )

    # ------------------------------------------------------------------
    # Contract-based safe-stop command construction.
    # ------------------------------------------------------------------

    def _safe_stop(self) -> bool:
        admission = getattr(self, "_runtime_admission", None)
        if admission is not None:
            with self._state_lock:
                admission.cancel()
                self._cancel_inflight_dispatch()
                self._clear_plans_locked()
                self._clear_inflight_locked()
                self._last_action = None
                self._state = DispatcherState.CLOSING
            return True
        self._cancel_inflight_dispatch()
        with self._state_lock:
            # freeze: stop control timer output + new dispatch
            if self._state not in (DispatcherState.STOPPED, DispatcherState.FAILED):
                self._state = DispatcherState.CLOSING
            self._clear_plans_locked()
            self._clear_inflight_locked()
            last = self._last_action
            snap = self._joint_snapshot
            if (
                snap.valid
                and self._joint_max_age_ns > 0
                and time.monotonic_ns() - snap.received_monotonic_ns > self._joint_max_age_ns
            ):
                snap = JointSnapshot(valid=False)
        try:
            commands = construct_safety_command(plan=self._safe_stop_plan, last_action=last, joint_snapshot=snap)
        except SafeStopError as exc:
            self.get_logger().fatal(f"safe_stop failed: {exc}")
            return False
        # Publish zeros channels first, then hold channels.
        for channel, cmd in zip(self._safe_stop_plan.channels, commands, strict=True):
            if channel.safety_behavior != "zeros":
                continue
            try:
                self._executor.execute_channel(channel.topic, np.array(cmd, dtype=float))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"safe_stop publish zeros failed: {exc}")
                return False
        for channel, cmd in zip(self._safe_stop_plan.channels, commands, strict=True):
            if channel.safety_behavior != "hold":
                continue
            try:
                self._executor.execute_channel(channel.topic, np.array(cmd, dtype=float))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"safe_stop publish hold failed: {exc}")
                return False
        with self._state_lock:
            self._last_action = None
        return True

    def _cancel_inflight_dispatch(self) -> None:
        with self._state_lock:
            goal_handle = self._inflight_goal_handle
            self._inflight_goal_handle = None
        if goal_handle is None:
            return
        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001 - Close remains the drain barrier.
            self.get_logger().warning(f"failed to request in-flight dispatch cancellation: {exc}")

    @staticmethod
    def _wait_for_event(event: threading.Event, timeout_ns: int, spin_once=None) -> bool:
        deadline_ns = time.monotonic_ns() + timeout_ns
        while not event.is_set():
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return False
            wait_sec = min(0.05, remaining_ns / 1_000_000_000)
            if spin_once is None:
                event.wait(wait_sec)
            else:
                spin_once(timeout_sec=wait_sec)
        return True

    def _begin_close_session(self, *, expected_identity: tuple[str, int] | None = None) -> threading.Event | None:
        with self._state_lock:
            if expected_identity is not None and (self._session_id, self._session_generation) != expected_identity:
                return None
            self._clear_plans_locked()
            self._clear_inflight_locked()
            sid = self._session_id
            gen = self._session_generation
            if not sid:
                return None
            if self._close_in_progress:
                return self._close_completion
            self._state = DispatcherState.CLOSING
            self._close_in_progress = True
            self._close_success = False
            self._close_identity = (sid, gen)
            self._close_completion = threading.Event()
            completion = self._close_completion
        goal = CloseInferenceSession.Goal()
        goal.session_id = sid
        goal.session_generation = gen
        close_deadline_ns = time.time_ns() + self._default_request_timeout_ns
        goal.deadline.sec, goal.deadline.nanosec = divmod(close_deadline_ns, 1_000_000_000)
        if not self._close_client.wait_for_server(timeout_sec=0.1):
            self._finish_close_session(sid, gen, success=False)
            return completion

        def _result_done(future) -> None:
            try:
                result = future.result().result
                success = bool(
                    result is not None
                    and result.success
                    and result.session_id == sid
                    and (gen == 0 or int(result.closed_session_generation) == gen)
                )
            except Exception:  # noqa: BLE001
                success = False
            self._finish_close_session(sid, gen, success=success)

        def _goal_done(future) -> None:
            try:
                goal_handle = future.result()
                if goal_handle is None or not goal_handle.accepted:
                    self._finish_close_session(sid, gen, success=False)
                    return
                goal_handle.get_result_async().add_done_callback(_result_done)
            except Exception:  # noqa: BLE001
                self._finish_close_session(sid, gen, success=False)

        try:
            self._close_client.send_goal_async(goal).add_done_callback(_goal_done)
        except Exception:  # noqa: BLE001
            self._finish_close_session(sid, gen, success=False)
        return completion

    def _finish_close_session(self, sid: str, gen: int, *, success: bool) -> None:
        with self._state_lock:
            if self._close_identity != (sid, gen) or self._close_completion is None:
                return
            self._close_success = success
            self._close_in_progress = False
            self._close_after_open = False
            if success and self._session_id == sid and (gen == 0 or self._session_generation == gen):
                self._session_id = ""
                self._session_generation = 0
            self._close_completion.set()

    def _close_session_sync(self, *, spin_once=None) -> bool:
        with self._state_lock:
            self._clear_plans_locked()
            self._clear_inflight_locked()
            sid = self._session_id
            pending_open = (
                self._pending_open_completion
                if sid and self._pending_open_session_id == sid and self._session_generation == 0
                else None
            )
            if pending_open is not None:
                self._close_after_open = True
                self._state = DispatcherState.CLOSING
        if pending_open is not None and not self._wait_for_event(
            pending_open, self._default_open_timeout_ns, spin_once=spin_once
        ):
            return False
        completion = self._begin_close_session()
        if completion is None:
            return True
        if not self._wait_for_event(completion, self._default_request_timeout_ns, spin_once=spin_once):
            return False
        with self._state_lock:
            return self._close_success

    def shutdown_cleanup(self, *, spin_once=None) -> bool:
        with capture_traces(trace.enabled), self._lifecycle_lock:
            return self._shutdown_cleanup_locked(spin_once=spin_once)

    def _shutdown_cleanup_locked(self, *, spin_once=None) -> bool:
        if self._shutdown_started:
            return True
        self._shutdown_started = True
        try:
            safe_ok = self._safe_stop()
            close_ok = self._close_session_sync(spin_once=spin_once)
            admission = getattr(self, "_runtime_admission", None)
            if admission is not None:
                safe_ok = admission.release(spin_once=spin_once) and safe_ok
            if not safe_ok or not close_ok:
                self.get_logger().error("scheduled dispatcher shutdown cleanup did not complete")
            return safe_ok and close_ok
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"scheduled dispatcher shutdown cleanup failed: {exc}")
            return False

    def destroy_node(self):
        self.shutdown_cleanup()
        return super().destroy_node()


def main(argv=None) -> None:
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    node = ScheduledActionDispatcherNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    shutdown_requested = threading.Event()

    def _request_shutdown(_signum, _frame) -> None:
        shutdown_requested.set()

    previous_handlers = {signum: signal.signal(signum, _request_shutdown) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        while not shutdown_requested.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        shutdown_requested.set()
    finally:
        node.shutdown_cleanup(spin_once=executor.spin_once)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
