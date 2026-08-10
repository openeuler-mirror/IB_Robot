#!/usr/bin/env python3
"""
Minimal Action Dispatcher Node.

Maintains a queue of actions and triggers inference when low.
Publishes actions to ros2_control via TopicExecutor at a fixed frequency.

Supports cross-frame temporal smoothing for action chunks.
"""

import collections
import json

# Business tracepoints via Python logging.
# When lttngust is imported, these are auto-captured by LTTng as
# python:logging events — no wrapper package needed.
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
import rclpy
import rclpy.action
import rclpy.time
import torch
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty, Trigger

from ibrobot_msgs.action import DispatchInfer, RunPolicy
from ibrobot_msgs.srv import PreparePolicyEpisode
from robot_config.contract_utils import iter_specs
from robot_config.tracing_utils import create_trace_logger
from tensormsg.converter import TensorMsgConverter

from .action_chunk import normalize_action_chunk
from .episode import (
    TERMINATION_CANCELED,
    TERMINATION_EXECUTION_FAILED,
    TERMINATION_EXECUTION_REJECTED,
    TERMINATION_EXECUTION_UNCERTAIN,
    TERMINATION_EXTERNAL_RESET,
    TERMINATION_IDENTITY_MISMATCH,
    TERMINATION_INFERENCE_FAILED,
    TERMINATION_OBSERVATION_STARTUP_TIMEOUT,
    EpisodeGoalSpec,
    EpisodeResult,
    EpisodeStateMachine,
    StepCompletionData,
)
from .executors.completion import ExecutionContext
from .executors.registry import create_executor
from .schedulers.base import (
    ActionDecision,
    CompletionDecision,
    SchedulerSnapshot,
)
from .schedulers.registry import create_scheduler
from .temporal_smoother import (
    TemporalSmootherManager,
)

_trace = create_trace_logger("ib_trace.dispatch")


@dataclass
class GoalExecutionContext:
    """Per-goal execution context with independent Event and frozen result.

    Created in handle_accepted_callback when the goal handle is bound.
    The execute callback waits on this context's own Event (not a shared
    Event). The terminal path freezes the result and sets this context's
    Event. Old episodes cannot touch new episodes' contexts.
    """

    goal_uuid: bytes
    generation: int
    done_event: threading.Event
    result: EpisodeResult | None = None
    termination_reason: str | None = None
    terminal_status: str | None = None  # "succeeded" / "aborted" / "canceled"


@dataclass
class PrepareContext:
    """Per-prepare barrier context with independent Event.

    Created in prepare callback. The reset done callback captures this context
    object (not just an integer generation) and operates on it under the
    dispatcher lock. A late callback from a previous prepare cannot modify
    the current prepare's context.
    """

    generation: int
    done_event: threading.Event
    success: bool = False
    message: str = ""
    closed: bool = False  # timeout or superseded


# Preserve the private helper used by the legacy dispatcher and its callers.
_normalize_action_chunk = normalize_action_chunk


class ActionDispatcherNode(Node):
    """
    Simplified action dispatcher.
    - Queue: collections.deque (when smoothing disabled) or TemporalSmoother (when enabled)
    - Trigger: Simple watermark check
    - Execution: TopicExecutor (100Hz streaming)

    Cross-frame smoothing can be enabled via parameters to ensure smooth
    transitions between consecutive action chunks.
    """

    def __init__(self, **kwargs):
        super().__init__("action_dispatcher", **kwargs)
        self.get_logger().info("Initializing Action Dispatcher")

        # 1. Parameters
        self.declare_parameter("queue_size", 100)
        self.declare_parameter("watermark_threshold", 20)
        self.declare_parameter("control_frequency", 100.0)
        self.declare_parameter("inference_action_server", "/inference/policy/dispatch")
        self.declare_parameter("inference_reset_service", "/inference/policy/reset")
        self.declare_parameter("inference_prompt", "")
        self.declare_parameter("policy_reset_timeout_sec", 2.0)
        # Safety net: if an inference goal never completes (server hiccup, dropped
        # response, or a goal abandoned across a stop/start), abandon it after this
        # many seconds so the control loop can request a fresh one instead of
        # wedging forever with _inference_in_progress stuck True.
        self.declare_parameter("inference_timeout_sec", 10.0)
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("navigation_mode", False)
        # Last-mile executor selection. SSOT ``executor.type`` is passed through
        # verbatim by the launch builder; the action_dispatch registry fail-fasts
        # on unknown values. Default is the only production executor in executor registry contract.
        self.declare_parameter("executor_type", "topic")
        # Dispatch scheduler selection. SSOT ``dispatch.scheduler`` is passed
        # through by launch builders; the executor contract only declares the
        # parameter so the dispatcher can fail-fast on unknown values without
        # modifying robot_config/YAML. Default ``continuous`` keeps all
        # existing IB-Robot behaviour byte-for-byte.
        self.declare_parameter("scheduler_mode", "continuous")
        # Execution timeout for ``wait_for_feedback`` mode only. Monotonic
        # nanoseconds, never mixed with ROS/sim observation time. Continuous
        # mode does not use this value.
        self.declare_parameter("execution_timeout_sec", 30.0)
        # production benchmark wiring: internal resolved benchmark step service endpoint. This is
        # NOT a customer SSOT path; the robot_config launch builder calls the
        # benchmark endpoint resolver endpoint resolver and passes the resolved ``/benchmark/<id>/step``
        # string here. Empty string is the legacy default for topic/continuous
        # configs; ``BenchmarkStepExecutor`` reads it from its factory config.
        self.declare_parameter("benchmark_step_service", "")

        # Temporal smoothing parameters
        self.declare_parameter("temporal_smoothing_enabled", False)
        self.declare_parameter("temporal_ensemble_coeff", 0.01)
        self.declare_parameter("chunk_size", 100)
        self.declare_parameter("smoothing_device", "")

        self._queue_limit = self.get_parameter("queue_size").value
        self._watermark = self.get_parameter("watermark_threshold").value
        self._control_hz = self.get_parameter("control_frequency").value
        self._server_name = self.get_parameter("inference_action_server").value

        # Smoothing config
        self._smoothing_enabled = self.get_parameter("temporal_smoothing_enabled").value
        self._temporal_ensemble_coeff = self.get_parameter("temporal_ensemble_coeff").value
        self._chunk_size = self.get_parameter("chunk_size").value
        smoothing_device = self.get_parameter("smoothing_device").value
        if smoothing_device == "":
            smoothing_device = None

        # 2. State & Queue
        self._navigation_mode = self.get_parameter("navigation_mode").value
        self._queue = collections.deque(maxlen=self._queue_limit)
        self._last_action: np.ndarray | None = None
        self._inference_in_progress = False
        self._inflight_request_id = ""
        self._policy_reset_in_progress = False
        self._policy_reset_started_at = 0.0
        self._policy_reset_timeout_s = float(self.get_parameter("policy_reset_timeout_sec").value)
        self._inference_started_at = 0.0
        self._inference_timeout_s = float(self.get_parameter("inference_timeout_sec").value)
        self._request_generation = 0
        # In navigation mode, start in stopped state; otherwise run immediately.
        self._is_running = not self._navigation_mode

        # Track actions executed during inference for temporal alignment
        self._plan_length_at_inference_start: int = 0

        # 3. Initialize Temporal Smoother (if enabled)
        self._smoother: TemporalSmootherManager | None = None
        if self._smoothing_enabled:
            self._smoother = TemporalSmootherManager(
                enabled=True,
                chunk_size=self._chunk_size,
                temporal_ensemble_coeff=self._temporal_ensemble_coeff,
                device=smoothing_device,
            )
            self.get_logger().info(
                f"Temporal smoothing ENABLED: coeff={self._temporal_ensemble_coeff}, chunk_size={self._chunk_size}"
            )
        else:
            self.get_logger().info("Temporal smoothing DISABLED (using simple queue)")

        # 4. Load Contract (Essential for TopicExecutor mapping)
        robot_config_path = self.get_parameter("robot_config_path").value
        self._action_specs = []
        if robot_config_path:
            try:
                from robot_config.loader import load_robot_config

                robot_cfg = load_robot_config(robot_config_path)
                self._contract = robot_cfg.to_contract()
                self._action_specs = [s for s in iter_specs(self._contract) if s.is_action]
                self.get_logger().info(f"Loaded {len(self._action_specs)} action specs from robot_config")
            except Exception as e:
                self.get_logger().error(f"Failed to load contract from {robot_config_path}: {e}")
        else:
            self.get_logger().warn("No robot_config_path provided! TopicExecutor will use defaults.")

        # 4b. Detect base action spec for navigation mode stop command.
        # Base spec: 3 names with first index >= 6 (e.g. action.6, action.7, action.8).
        self._base_act_spec = None
        for sv in self._action_specs:
            if sv.names and len(sv.names) == 3:
                first_idx = int(sv.names[0].split(".")[-1])
                if first_idx >= 6:
                    self._base_act_spec = sv
                    break
        if self._base_act_spec:
            self.get_logger().info(f"Detected base action spec: {[n for n in self._base_act_spec.names]}")

        # 5. Executor (Topic-based, selected via registry)
        executor_type_str = self.get_parameter("executor_type").value
        self.get_logger().info(f"[IBROBOT_EXECUTOR][SELECTED] type={executor_type_str}")
        # production benchmark wiring: read scheduler_mode early so the generic pairing guard can
        # fail-fast on illegal combinations before any executor/scheduler is
        # constructed. The guard only knows the stable generic values
        # ``topic``/``benchmark``/``continuous``/``wait_for_feedback``; it
        # does NOT alias, lowercase or fall back. Legacy ``action`` and other
        # unknown strings pass through to the registries which fail-fast.
        scheduler_mode_str = self.get_parameter("scheduler_mode").value
        self._validate_executor_scheduler_pairing(executor_type_str, scheduler_mode_str)
        # create_executor raises ExecutorNotFoundError for unknown types; letting
        # it propagate fail-fasts the node at init without falling back.
        # production benchmark wiring: factory config is a unified dict containing ``action_specs``
        # and the resolved ``step_service`` endpoint string. TopicExecutor
        # ignores the extra key; BenchmarkStepExecutor reads ``step_service``.
        self._executor = create_executor(
            executor_type_str,
            self,
            {
                "action_specs": self._action_specs,
                "step_service": self.get_parameter("benchmark_step_service").value,
            },
        )
        if not self._executor.initialize():
            raise RuntimeError(f"Failed to initialize executor type={executor_type_str}")
        self.get_logger().info(f"[IBROBOT_EXECUTOR][READY] type={executor_type_str}")

        # 5b. Dispatch scheduler (continuous by default; wait_for_feedback for
        # future benchmark step executors). completion-aware executor contract does not modify robot_config
        # builder/YAML, so all existing launches keep using continuous.
        self._scheduler_mode = scheduler_mode_str
        self._execution_timeout_sec = float(self.get_parameter("execution_timeout_sec").value)
        self.get_logger().info(f"[IBROBOT_SCHEDULER][SELECTED] mode={self._scheduler_mode}")
        # create_scheduler raises SchedulerNotFoundError for unknown modes;
        # letting it propagate fail-fasts the node at init without falling back.
        self._scheduler = create_scheduler(
            self._scheduler_mode,
            {
                "watermark": self._watermark,
                "execution_timeout_sec": self._execution_timeout_sec,
            },
        )
        self.get_logger().info(f"[IBROBOT_SCHEDULER][READY] mode={self._scheduler_mode}")

        # completion-aware executor contract wait-for-feedback reservation state. Only used when
        # scheduler_mode == "wait_for_feedback"; continuous never touches this.
        self._reservation_context: ExecutionContext | None = None
        self._reservation_plan_generation: int = 0
        self._plan_generation: int = 0

        # benchmark episode controller: episode state machine and benchmark-only goal gate.
        # The episode state machine is created unconditionally (it is pure
        # Python and harmless when unused), but the ROS handles
        # (~/prepare_episode, ~/run_policy) are only created when
        # executor_type=benchmark and scheduler_mode=wait_for_feedback.
        self._is_benchmark = executor_type_str == "benchmark" and scheduler_mode_str == "wait_for_feedback"
        self._episode = EpisodeStateMachine()
        # Per-goal concurrency handling: per-goal execution contexts with independent Events.
        # Keyed by goal UUID (bytes). The execute callback gets its context by
        # UUID and waits on the context's own Event. No shared Event.
        self._goal_contexts: dict[bytes, GoalExecutionContext] = {}
        self._active_goal_handle = None
        # Per-goal concurrency handling: dispatcher lock for protecting goal context dict
        # and prepare context during terminal/cancel/abort operations.
        self._dispatch_lock = threading.RLock()
        # Per-goal concurrency handling: per-prepare context (replaces shared event/success/message).
        self._active_prepare_context: PrepareContext | None = None
        # Observation readiness retry deadline (monotonic ns). Set on goal
        # bind and on each step commit; checked when inference returns
        # observation_not_ready + recoverable.
        self._readiness_retry_deadline_ns = 0
        # benchmark episode controller: benchmark mode starts with gate closed. No inference, no
        # step submit until a valid PreparePolicyEpisode + RunPolicy goal
        # opens the gate. Continuous/topic mode keeps the legacy
        # ``not navigation_mode`` default.
        if self._is_benchmark:
            self._is_running = False

        # 6. Communication
        self._infer_client = rclpy.action.ActionClient(self, DispatchInfer, self._server_name)
        self._policy_reset_client = self.create_client(
            Trigger,
            self.get_parameter("inference_reset_service").value,
        )

        # Subscriptions
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._joint_sub = self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._joint_cb,
            qos,
        )

        # Publishers
        self._queue_size_pub = self.create_publisher(Int32, "~/queue_size", 10)
        self._smoothing_enabled_pub = self.create_publisher(Bool, "~/smoothing_enabled", 10)

        # 7. Timers
        self._cb_group = MutuallyExclusiveCallbackGroup()
        self._timer = self.create_timer(1.0 / self._control_hz, self._control_loop, callback_group=self._cb_group)

        # Services
        self._reset_srv = self.create_service(Empty, "~/reset", self._reset_cb)
        self._toggle_smoothing_srv = self.create_service(Empty, "~/toggle_smoothing", self._toggle_smoothing_cb)

        # Control services — usable in any mode (nav or sim inference).
        self._start_nav_srv = self.create_service(Trigger, "~/start_evaluate", self._start_nav_cb)
        self._stop_nav_srv = self.create_service(Trigger, "~/stop_evaluate", self._stop_nav_cb)
        self._get_status_srv = self.create_service(Trigger, "~/get_status", self._get_status_cb)

        # benchmark episode controller: benchmark-only episode handles. Only created when
        # executor_type=benchmark and scheduler_mode=wait_for_feedback.
        # Topic/continuous nodes never expose these; all legacy surface is
        # preserved. Uses a dedicated ReentrantCallbackGroup so the
        # RunPolicy execute callback can block on the episode-done event
        # without blocking the control timer or the policy reset Future
        # callback.
        if self._is_benchmark:
            self._episode_cb_group = ReentrantCallbackGroup()
            self._prepare_episode_srv = self.create_service(
                PreparePolicyEpisode,
                "~/prepare_episode",
                self._prepare_episode_cb,
                callback_group=self._episode_cb_group,
            )
            self._run_policy_server = ActionServer(
                self,
                RunPolicy,
                "~/run_policy",
                execute_callback=self._run_policy_execute_cb,
                goal_callback=self._run_policy_goal_cb,
                handle_accepted_callback=self._run_policy_handle_accepted_cb,
                cancel_callback=self._run_policy_cancel_cb,
                callback_group=self._episode_cb_group,
            )
        else:
            self._episode_cb_group = None
            self._prepare_episode_srv = None
            self._run_policy_server = None

        mode_label = "NAV" if self._navigation_mode else "NORMAL"
        self.get_logger().info(
            f"Dispatcher ready [{mode_label}]. Hz: {self._control_hz}, "
            f"Watermark: {self._watermark}, "
            f"Smoothing: {'ON' if self._smoothing_enabled else 'OFF'}"
        )
        self.get_logger().info(f"Waiting for inference server: {self._server_name}")

        # Periodic stats tracking
        self._dispatch_count = 0
        self._total_inference_latency_ms = 0.0
        self._last_stats_time = time.monotonic()
        self._stats_interval_s = 5.0
        self._hold_count = 0
        self._consecutive_failures = 0
        self._last_stats_dispatch_count = 0
        self._active_request_id = ""
        self._actions_executed_from_active_request = 0
        self._last_queue_refill_monotonic_ns = 0
        self._last_stall_log_ns = time.monotonic_ns()
        self._benchmark_inference_timings: dict[str, dict[str, object]] = {}

    @staticmethod
    def _validate_executor_scheduler_pairing(executor_type: str, scheduler_mode: str) -> None:
        """production benchmark wiring generic executor/scheduler pairing guard.

        Legal combinations:
        - ``topic`` + ``continuous``
        - ``benchmark`` + ``wait_for_feedback``

        Illegal:
        - ``benchmark`` + ``continuous`` (benchmark requires step feedback)
        - ``wait_for_feedback`` + any executor other than ``benchmark``

        Unknown executor/scheduler strings are NOT aliased, case-folded or
        fallback-corrected here; they pass through so the executor/scheduler
        registries can fail-fast with their own clear error messages. The
        legacy ``action`` string is preserved verbatim (it is not silently
        rewritten to ``topic``).

        Raises:
            ValueError: if a known special-pair rule is violated.
        """
        # benchmark requires wait_for_feedback; benchmark + continuous illegal.
        if executor_type == "benchmark" and scheduler_mode != "wait_for_feedback":
            raise ValueError(
                f"executor type 'benchmark' requires scheduler_mode 'wait_for_feedback'; "
                f"got scheduler_mode={scheduler_mode!r}"
            )
        # wait_for_feedback requires benchmark; any non-benchmark executor illegal.
        if scheduler_mode == "wait_for_feedback" and executor_type != "benchmark":
            raise ValueError(
                f"scheduler_mode 'wait_for_feedback' requires executor type 'benchmark'; "
                f"got executor_type={executor_type!r}"
            )

    def _joint_cb(self, msg):
        """Optional: could use current state for safety or initialization."""
        pass

    def _get_plan_length(self) -> int:
        """Get current plan length (works for both modes)."""
        if self._smoother is not None:
            return self._smoother.plan_length
        return len(self._queue)

    def _control_loop(self):
        # benchmark episode controller: benchmark/wait_for_feedback uses the episode gate (opened by
        # RunPolicy goal, closed on terminal), NOT the legacy ``_is_running``
        # flag. Routing this path BEFORE the ``_is_running`` check keeps the
        # continuous body below byte-for-byte identical to executor registry contract/completion-aware executor contract.
        if self._scheduler_mode == "wait_for_feedback":
            self._control_loop_wait_for_feedback()
            return

        if not self._is_running:
            return

        self._expire_policy_reset_if_needed()
        self._expire_inference_if_needed()

        q_size = self._get_plan_length()
        self._queue_size_pub.publish(Int32(data=q_size))
        self._smoothing_enabled_pub.publish(Bool(data=self._smoothing_enabled))

        # A. Trigger Inference if queue is low
        if q_size <= self._watermark and not self._inference_in_progress and not self._policy_reset_in_progress:
            self._request_inference()

        # Diagnostic: surface a running-but-not-dispatching loop (stays quiet in
        # normal operation; only fires when the queue stays empty for >2s).
        if q_size == 0:
            now_ns = time.monotonic_ns()
            stalled_ms = (
                (now_ns - self._last_queue_refill_monotonic_ns) / 1e6 if self._last_queue_refill_monotonic_ns else 1e9
            )
            if stalled_ms > 2000.0 and (now_ns - self._last_stall_log_ns) / 1e6 > 2000.0:
                self._last_stall_log_ns = now_ns
                self.get_logger().warn(
                    f"[diag] running but queue empty: in_progress={self._inference_in_progress} "
                    f"policy_reset={self._policy_reset_in_progress} "
                    f"server_ready={self._infer_client.server_is_ready()} "
                    f"watermark={self._watermark} gen={self._request_generation}"
                )

        # B. Get Action
        action = None
        action_source = "empty"
        if q_size > 0:
            if self._smoother is not None:
                action_tensor = self._smoother.get_next_action()
                if isinstance(action_tensor, torch.Tensor):
                    action = action_tensor.detach().cpu().numpy()
                else:
                    action = action_tensor
                action_source = "smoother"
            else:
                action = self._queue.popleft()
                action_source = "queue"
            self._last_action = action
        elif self._last_action is not None:
            action = self._last_action
            self._hold_count += 1
            action_source = "hold"

        # C. Execute
        if action is not None:
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy()
            else:
                action_np = np.array(action)
            execute_index = self._actions_executed_from_active_request
            execute_start = time.perf_counter()
            self._executor.execute(
                action_np,
                {
                    "request_id": self._active_request_id,
                    "execute_index": execute_index,
                    "queue_size": q_size,
                },
            )
            publish_ms = (time.perf_counter() - execute_start) * 1000.0
            queue_after = self._get_plan_length()
            since_refill_ms = (
                max(
                    0.0,
                    (time.monotonic_ns() - self._last_queue_refill_monotonic_ns) / 1_000_000,
                )
                if self._last_queue_refill_monotonic_ns
                else -1.0
            )
            _trace.info(
                "[action_execute] request_id=%s index=%d source=%s "
                "queue_before=%d queue_after=%d since_refill_ms=%.2f publish_ms=%.2f",
                self._active_request_id,
                execute_index,
                action_source,
                q_size,
                queue_after,
                since_refill_ms,
                publish_ms,
            )
            if action_source != "hold":
                self._actions_executed_from_active_request += 1

        # D. Periodic stats (only when new inferences arrived)
        now = time.monotonic()
        if now - self._last_stats_time >= self._stats_interval_s:
            new_inferences = self._dispatch_count - self._last_stats_dispatch_count
            if new_inferences > 0:
                avg_lat = self._total_inference_latency_ms / self._dispatch_count
                self.get_logger().info(
                    f"[stats] inferences={self._dispatch_count}, "
                    f"avg_latency={avg_lat:.1f}ms, "
                    f"queue={q_size}, hold={self._hold_count}"
                )
            self._last_stats_dispatch_count = self._dispatch_count
            self._hold_count = 0
            self._last_stats_time = now

    def _request_inference(self, timestamp_ns: int | None = None):
        """Send async goal to inference service.

        ``timestamp_ns`` is the optional environment observation timestamp
        (positive integer nanoseconds) used by ``wait_for_feedback`` mode so
        the next inference goal carries the completion timestamp rather than
        the dispatcher wall clock. Continuous mode always passes ``None`` and
        keeps using ``get_clock().now()`` exactly as before.
        """
        if not self._infer_client.wait_for_server(timeout_sec=0.1):
            return

        self._inference_in_progress = True
        self._inference_started_at = time.monotonic()
        self._plan_length_at_inference_start = self._get_plan_length()
        self._current_request_id = uuid.uuid4().hex[:8]
        self._inflight_request_id = self._current_request_id
        self._benchmark_inference_timings[self._current_request_id] = {
            "request_start_monotonic_ns": time.monotonic_ns(),
        }

        goal = DispatchInfer.Goal()
        if timestamp_ns is not None and timestamp_ns > 0:
            goal.obs_timestamp = rclpy.time.Time(nanoseconds=timestamp_ns).to_msg()
        else:
            goal.obs_timestamp = self.get_clock().now().to_msg()
        # benchmark episode controller correctness handling: use the RunPolicy goal's prompt when an episode
        # is active. Continuous/topic mode keeps using the inference_prompt
        # parameter for backward compatibility.
        episode_prompt = self._episode.get_prompt()
        if episode_prompt is not None:
            goal.prompt = episode_prompt
        else:
            goal.prompt = self.get_parameter("inference_prompt").value
        goal.inference_id = self._current_request_id

        _trace.info(
            "[dispatch_request] request_id=%s queue_size=%d watermark=%d",
            self._current_request_id,
            self._plan_length_at_inference_start,
            self._watermark,
        )
        self.get_logger().debug(
            f"Requesting inference @ {goal.obs_timestamp.sec}, "
            f"plan_length_at_start: {self._plan_length_at_inference_start}"
        )

        send_goal_future = self._infer_client.send_goal_async(goal)
        request_generation = self._request_generation
        # Canonical lock ordering: capture the episode goal generation at request time so the
        # result callback uses the generation that was active when this
        # inference was dispatched, not the global generation at callback time.
        episode_goal_generation = self._episode.goal_generation
        send_goal_future.add_done_callback(
            lambda future, req_id=self._current_request_id, gen=request_generation, epg=episode_goal_generation: (
                self._goal_response_cb(future, req_id, gen, epg)
            )
        )

    def _control_loop_wait_for_feedback(self) -> None:
        """completion-aware executor contract/benchmark episode controller wait-for-feedback control loop.

        benchmark episode controller: the episode gate (``EpisodeStateMachine.is_gate_open``) controls
        whether inference/submit may proceed. When closed (no valid RunPolicy
        goal or after terminal), only stale completions are drained; no new
        inference or step submission occurs.

        The continuous legacy block stays byte-for-byte identical to executor registry contract.
        Uses peek/submit/drain/reservation so the logical cursor only advances
        after a matching completion; never pops speculatively, never holds,
        never auto-retries on timeout.
        """
        self._expire_policy_reset_if_needed()
        self._expire_inference_if_needed()

        # 1. Drain delayed completions and apply transitions (only when gate
        # is open). When the gate is closed, stale completions are silently
        # discarded so old callbacks from a previous episode do not pollute
        # the scheduler.
        if self._episode.is_gate_open:
            for completion in self._executor.drain_completions():
                transition = self._scheduler.on_completion(completion)
                self._apply_completion_transition_wait_for_feedback(transition, completion)
        else:
            self._executor.drain_completions()

        # 2. Check execution timeout (monotonic nanoseconds) — only when
        # gate is open.
        if self._episode.is_gate_open:
            transition = self._scheduler.on_tick(time.monotonic_ns())
            if transition.decision is CompletionDecision.FAIL_CLOSED:
                self._apply_completion_transition_wait_for_feedback(transition, None)

        # Fail-closed: stop this tick after draining; do not pop, retry or hold.
        if self._scheduler.fault_status is not None:
            return

        # benchmark episode controller: episode gate closed — no new inference or step submit.
        if not self._episode.is_gate_open:
            return

        # benchmark episode controller: check for RunPolicy cancel request. Cancel closes the gate,
        # invalidates inference/step, clears local state, and signals the
        # execute callback to return CANCELED.
        if self._active_goal_handle is not None and self._active_goal_handle.is_cancel_requested:
            self._cancel_episode()
            return

        # benchmark episode controller correctness handling: check startup_timeout deadline on every tick
        # during STARTING. Uses generation-aware API so stale episodes don't
        # trigger false timeouts.
        gen = self._episode.goal_generation
        startup_reason = self._episode.try_check_startup_deadline(gen, time.monotonic_ns())
        if startup_reason is not None:
            self._abort_episode(startup_reason)
            return

        # benchmark episode controller: check max_duration deadline before starting new work.
        if self._scheduler.inflight_correlation_id is None:
            deadline_reason = self._episode.try_check_episode_deadline(gen, time.monotonic_ns())
            if deadline_reason is not None:
                self._close_episode_terminal(deadline_reason)
                return

        # Re-read plan length AFTER drain/commit: a matching completion pops
        # the queue, so the pre-drain q_size is stale. The scheduler snapshot
        # must reflect the real current plan length.
        q_size = self._get_plan_length()
        self._queue_size_pub.publish(Int32(data=q_size))
        self._smoothing_enabled_pub.publish(Bool(data=self._smoothing_enabled))

        snapshot = SchedulerSnapshot(
            plan_length=q_size,
            watermark=self._watermark,
            inference_in_progress=self._inference_in_progress,
            policy_reset_in_progress=self._policy_reset_in_progress,
            has_last_action=self._last_action is not None,
        )

        # 3. Inference decision. Wait-for-feedback never overlaps inference and
        # execution: the scheduler gates both. If the scheduler says we should
        # request inference this tick, we MUST stop after the inference attempt
        # — even if the inference server is not ready and
        # _inference_in_progress stays False, we must NOT submit an old-plan
        # action. The next tick will retry inference.
        if (
            self._scheduler.should_request_inference(snapshot)
            and not self._inference_in_progress
            and not self._policy_reset_in_progress
        ):
            # benchmark episode controller: use the episode's authoritative inference timestamp when
            # the gate is open (first inference uses goal's initial timestamp;
            # subsequent uses last committed step's response timestamp).
            episode_ts = self._episode.get_inference_timestamp()
            ts_ns = episode_ts if episode_ts is not None else self._scheduler.observation_timestamp_for_inference()
            self._request_inference(timestamp_ns=ts_ns)
            return

        # 4. Action decision. Only submit if the scheduler permits and there is
        # no inference in-progress (mutual exclusion).
        decision = self._scheduler.choose_action(snapshot)
        if (
            decision is ActionDecision.TAKE_NEXT
            and self._scheduler.can_accept_next
            and not self._inference_in_progress
            and not self._policy_reset_in_progress
        ):
            self._submit_next_action_wait_for_feedback()
        # WAIT: do nothing. Wait-for-feedback never holds the last action.

    def _submit_next_action_wait_for_feedback(self) -> None:
        """Peek the next action, build a context, submit exactly once."""
        # Peek (do not pop) so the logical plan length stays unchanged until
        # a matching completion arrives.
        if self._smoother is not None:
            peeked = self._smoother.peek_next_action()
            if peeked is None:
                return
            if isinstance(peeked, torch.Tensor):
                action_np = peeked.detach().cpu().numpy()
            else:
                action_np = np.array(peeked)
        else:
            if not self._queue:
                return
            action_np = np.array(self._queue[0])

        correlation_id = uuid.uuid4().hex[:8]
        # benchmark episode controller: inject environment-owned episode_id and expected_step_id from
        # the episode state machine into every ExecutionContext. These are
        # environment-reset-owned; the dispatcher never invents them.
        # Only injected when a RunPolicy goal is active (goal handle set);
        # when no goal is active, identity is None so the scheduler skips
        # its identity checks (preserves completion-aware executor contract/executor lifecycle isolation behavior).
        has_goal = self._active_goal_handle is not None
        context = ExecutionContext(
            correlation_id=correlation_id,
            episode_id=self._episode.episode_id if has_goal else None,
            expected_step_id=self._episode.expected_step_id if has_goal else None,
            metadata={
                "request_id": self._active_request_id,
                "execute_index": self._actions_executed_from_active_request,
                "queue_size": self._get_plan_length(),
                "action_reservation_monotonic_ns": time.monotonic_ns(),
            },
        )

        receipt = self._executor.submit(action_np, context)
        submitted_ns = time.monotonic_ns()
        self._scheduler.on_submission(context, receipt, submitted_ns)

        # Record the reservation so a later matching completion can validate
        # that the plan has not been replaced underneath us.
        self._reservation_context = context
        self._reservation_plan_generation = self._plan_generation

        # If the executor returns an immediate completion (e.g. TopicExecutor),
        # process it exactly once here. The executor's drain must not return it
        # again; TopicExecutor.drain_completions always returns ().
        if receipt.immediate_completion is not None:
            transition = self._scheduler.on_completion(receipt.immediate_completion)
            self._apply_completion_transition_wait_for_feedback(transition, receipt.immediate_completion)

    def _apply_completion_transition_wait_for_feedback(
        self,
        transition,
        completion,
    ) -> None:
        """Apply a scheduler transition: commit exactly one action, or fail-closed.

        executor lifecycle isolation: when the fail-closed is caused by a timeout (``completion is
        None``, meaning the Future is still pending and no completion arrived),
        the executor's local pending is invalidated exactly once so late
        callbacks cannot pollute the next generation. Completion-based
        fail-closed (FAILED/UNCERTAIN/missing_timestamp/etc.) already cleared
        the executor pending in the callback, so invalidate is NOT called again.

        benchmark episode controller: on COMMIT, the step is committed to the episode state machine
        (advancing expected_step_id and updating the last snapshot), one
        RunPolicy feedback is published with full typed flags and raw JSON,
        and terminal conditions are evaluated. On FAIL_CLOSED, the episode is
        aborted with the mapped termination reason.
        """
        from .schedulers.base import SchedulerTransition as _Transition

        if not isinstance(transition, _Transition):
            return

        if transition.decision is CompletionDecision.COMMIT:
            # Generation mismatch means the plan was replaced between
            # submission and completion; fail-closed via the scheduler API
            # (not by mutating private fields). The fault persists until
            # reset; no pop, no retry.
            if self._reservation_plan_generation != self._plan_generation:
                self.get_logger().error(
                    "wait_for_feedback commit rejected: plan generation mismatch "
                    f"(reservation={self._reservation_plan_generation}, "
                    f"current={self._plan_generation})"
                )
                self._scheduler.mark_fault("plan_generation_mismatch")
                self._reservation_context = None
                # benchmark episode controller: abort the episode only when a RunPolicy goal is active.
                if self._active_goal_handle is not None and self._episode.is_gate_open:
                    self._abort_episode(TERMINATION_EXECUTION_UNCERTAIN)
                return
            # Atomic commit handling: commit + action consume is one atomic dispatcher transaction.
            # _dispatch_lock is acquired BEFORE try_commit_step and held
            # through pop+cursor+feedback+terminal. This prevents reset/cancel
            # from inserting between commit and pop.
            #
            # Lock order (canonical): _dispatch_lock -> EpisodeStateMachine._lock
            # (try_commit_step acquires _episode._lock internally).
            # publish_feedback is non-blocking and allowed inside _dispatch_lock.
            # No ROS Future/Event/service wait inside _dispatch_lock.
            has_goal = self._active_goal_handle is not None and self._episode.is_gate_open
            queue_before = self._get_plan_length()

            if has_goal and completion is not None:
                step_data = self._build_step_completion_data(completion, queue_before)
                gen = self._episode.goal_generation

                # Benchmark response validation: verify reserved action exists BEFORE try_commit_step
                # to prevent "episode committed but action not consumed" state.
                # Atomic commit handling: entire commit+consume under _dispatch_lock.
                with self._dispatch_lock:
                    # Pre-check: if queue/smoother is empty, fail-closed
                    # BEFORE calling try_commit_step (no half-committed state).
                    if self._smoother is not None:
                        if self._smoother.plan_length == 0:
                            self.get_logger().error("[IBROBOT_EPISODE][COMMIT_NO_ACTION] smoother empty before commit")
                            self._abort_episode(TERMINATION_EXECUTION_UNCERTAIN)
                            return
                    else:
                        if len(self._queue) == 0:
                            self.get_logger().error("[IBROBOT_EPISODE][COMMIT_NO_ACTION] queue empty before commit")
                            self._abort_episode(TERMINATION_EXECUTION_UNCERTAIN)
                            return

                    outcome = self._episode.try_commit_step(gen, step_data, time.monotonic_ns())

                    if outcome.identity_mismatch:
                        self.get_logger().error("[IBROBOT_EPISODE][IDENTITY_MISMATCH]")
                        # Abort inside the lock; _abort_episode acquires
                        # _dispatch_lock (RLock, reentrant) then episode._lock.
                        self._abort_episode(TERMINATION_IDENTITY_MISMATCH)
                        return

                    if not outcome.applied:
                        # Stale or gate closed: no pop, no feedback, no count,
                        # no cursor update.
                        self._reservation_context = None
                        return

                    # applied=True: consume the reserved action.
                    # Queue/smoother non-emptiness verified above.
                    if self._smoother is not None:
                        action = self._smoother.get_next_action()
                        if isinstance(action, torch.Tensor):
                            action_np = action.detach().cpu().numpy()
                        else:
                            action_np = np.array(action)
                    else:
                        action_np = self._queue.popleft()
                    queue_after = self._get_plan_length()
                    _trace.info(
                        "[action_execute] request_id=%s index=%d source=commit queue_before=%d queue_after=%d",
                        self._active_request_id,
                        self._actions_executed_from_active_request,
                        queue_before,
                        queue_after,
                    )
                    self._last_action = action_np
                    self._actions_executed_from_active_request += 1
                    self._reservation_context = None

                    # Publish feedback (non-blocking, allowed inside lock).
                    self._publish_step_feedback(step_data, queue_after)

                    if step_data.observation_timestamp_ns is not None and step_data.observation_timestamp_ns > 0:
                        self._readiness_retry_deadline_ns = time.monotonic_ns() + int(
                            self._episode.startup_timeout_sec * 1_000_000_000
                        )

                    # Terminal close inside the lock (RLock reentrant).
                    if outcome.terminal_reason is not None:
                        self._close_episode_terminal(outcome.terminal_reason)
            else:
                # No active goal (executor lifecycle isolation legacy path): pop as before.
                if self._smoother is not None:
                    action = self._smoother.get_next_action()
                    if isinstance(action, torch.Tensor):
                        action_np = action.detach().cpu().numpy()
                    else:
                        action_np = np.array(action)
                else:
                    action_np = self._queue.popleft()
                queue_after = self._get_plan_length()
                _trace.info(
                    "[action_execute] request_id=%s index=%d source=commit queue_before=%d queue_after=%d",
                    self._active_request_id,
                    self._actions_executed_from_active_request,
                    queue_before,
                    queue_after,
                )
                self._last_action = action_np
                self._actions_executed_from_active_request += 1
                self._reservation_context = None
            return

        if transition.decision is CompletionDecision.FAIL_CLOSED:
            self._reservation_context = None
            # executor lifecycle isolation: only timeout (completion is None, Future still pending)
            # requires executor invalidation. Completion-based fail-closed
            # (FAILED/UNCERTAIN/missing_timestamp/etc.) already cleared
            # pending in the executor callback; calling invalidate again
            # would be redundant and would bump the generation unnecessarily.
            if completion is None:
                self._executor.invalidate_pending()
            self.get_logger().error(
                f"wait_for_feedback fail-closed: {transition.message} (fault={transition.fault_status})"
            )
            # benchmark episode controller: abort the episode with the mapped termination reason.
            if self._active_goal_handle is not None and self._episode.is_gate_open:
                reason = self._map_scheduler_fault_to_termination(transition.fault_status, completion)
                self._abort_episode(reason)
            return

        # IGNORE: reservation stays in place (if any), nothing to do.

    def _goal_response_cb(self, future, request_id: str, request_generation: int, episode_goal_generation: int):
        if request_generation != self._request_generation:
            self._complete_inflight_request(request_id)
            self.get_logger().debug(f"Ignoring stale inference goal response: {request_id}")
            return
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Inference goal REJECTED")
            self._complete_inflight_request(request_id)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, req_id=request_id, gen=request_generation, epg=episode_goal_generation: self._result_cb(
                future, req_id, gen, epg
            )
        )

    def _result_cb(self, future, request_id: str, request_generation: int, episode_goal_generation: int):
        if request_generation != self._request_generation:
            self._complete_inflight_request(request_id)
            self.get_logger().debug(f"Ignoring stale inference result: {request_id}")
            return
        self._complete_inflight_request(request_id)
        result = future.result().result
        req_id = request_id
        result_monotonic_ns = time.monotonic_ns()
        inference_timing = self._benchmark_inference_timings.setdefault(req_id, {})
        inference_timing.update(
            {
                "clock_domain": "monotonic",
                "result_monotonic_ns": result_monotonic_ns,
                "total_latency_ms": float(result.inference_latency_ms),
                "backend_latency_ms": float(result.backend_latency_ms),
            }
        )
        try:
            service_performance = json.loads(str(getattr(result, "performance_json", "") or "{}"))
            if not isinstance(service_performance, dict):
                raise ValueError("inference performance_json must contain an object")
            inference_timing["service"] = service_performance
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            inference_timing["performance_error"] = str(exc)
        if not result.success:
            # benchmark episode controller: in benchmark episode mode, handle observation_not_ready
            # (retry with same timestamp within deadline) and abort on other
            # failures. Continuous mode keeps the existing auto-recovery.
            # Benchmark response validation: unified benchmark validity check BEFORE any decode/refill.
            # Gate closed (episode terminal) + same generation must still discard.
            if self._is_benchmark:
                if self._active_goal_handle is None:
                    self.get_logger().info(
                        f"[IBROBOT_EPISODE][STALE_INFERENCE_FAILURE] no active goal — discarding (request_id={req_id})"
                    )
                    return
                if not self._episode.is_gate_open:
                    self.get_logger().info(
                        f"[IBROBOT_EPISODE][STALE_INFERENCE_FAILURE] gate closed — discarding (request_id={req_id})"
                    )
                    return
                if episode_goal_generation != self._episode.goal_generation:
                    self.get_logger().info(
                        f"[IBROBOT_EPISODE][STALE_INFERENCE_FAILURE] gen={episode_goal_generation} "
                        f"current={self._episode.goal_generation} — discarding"
                    )
                    return
                error = getattr(result, "error", None)
                if (
                    error is not None
                    and getattr(error, "code", "") == "observation_not_ready"
                    and getattr(error, "recoverable", False)
                ):
                    # Retry with the same timestamp within the readiness
                    # deadline. Do NOT change timestamp, send action, or
                    # busy-loop; the control loop re-requests on next tick.
                    if time.monotonic_ns() < self._readiness_retry_deadline_ns:
                        self.get_logger().debug(
                            f"[IBROBOT_EPISODE][OBSERVATION_NOT_READY] retrying "
                            f"with same timestamp (request_id={req_id})"
                        )
                        return
                    self.get_logger().warn(
                        f"[IBROBOT_EPISODE][OBSERVATION_STARTUP_TIMEOUT] deadline exceeded (request_id={req_id})"
                    )
                    self._abort_episode(TERMINATION_OBSERVATION_STARTUP_TIMEOUT)
                    return
                # Other inference failure: abort the active episode.
                self.get_logger().warn(f"[IBROBOT_EPISODE][INFERENCE_FAILED] {result.message} (request_id={req_id})")
                self._abort_episode(TERMINATION_INFERENCE_FAILED)
                return
            self._consecutive_failures += 1
            _trace.info(
                "[dispatch_result] request_id=%s success=False",
                req_id,
            )
            if self._consecutive_failures == 1:
                self.get_logger().warn(f"Inference failed: {result.message}")
            else:
                self.get_logger().debug(f"Inference failed (#{self._consecutive_failures}): {result.message}")
            return

        if self._consecutive_failures > 0:
            self.get_logger().info(f"Inference recovered (after {self._consecutive_failures} failures)")
            self._consecutive_failures = 0

        # Atomic response handling: unified benchmark validity check + decode + refill under
        # _dispatch_lock. This prevents reset/cancel/abort from inserting
        # between the validity check and the queue refill.
        # _dispatch_lock is NOT held while waiting for ROS Futures/Events;
        # this callback is invoked by a completed Future, so we're already
        # past the wait. The decode (TensorMsgConverter) and queue/smoother
        # update are pure-Python and safe inside the lock.
        # Continuous/topic mode keeps the existing behavior without lock.
        if self._is_benchmark:
            with self._dispatch_lock:
                if self._active_goal_handle is None:
                    self.get_logger().info(
                        f"[IBROBOT_EPISODE][STALE_INFERENCE] no active goal — discarding (request_id={req_id})"
                    )
                    return
                if not self._episode.is_gate_open:
                    self.get_logger().info(
                        f"[IBROBOT_EPISODE][STALE_INFERENCE] gate closed — discarding (request_id={req_id})"
                    )
                    return
                if episode_goal_generation != self._episode.goal_generation:
                    self.get_logger().info(
                        f"[IBROBOT_EPISODE][STALE_INFERENCE] gen={episode_goal_generation} "
                        f"current={self._episode.goal_generation} — discarding"
                    )
                    return
                applied = self._episode.try_on_inference_success(episode_goal_generation)
                if not applied:
                    self.get_logger().info(
                        f"[IBROBOT_EPISODE][STALE_INFERENCE] try_on_inference_success returned False "
                        f"gen={episode_goal_generation} — discarding"
                    )
                    return

                # All checks passed; decode and refill under the same lock.
                self._dispatch_count += 1
                self._total_inference_latency_ms += result.inference_latency_ms

                decode_start = time.perf_counter()
                batch = TensorMsgConverter.from_variant(result.action_chunk)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                _trace.info(
                    "[dispatch_decode] request_id=%s chunk_size=%d decode_ms=%.2f",
                    req_id,
                    result.chunk_size,
                    decode_ms,
                )
                if "action" in batch:
                    action_chunk_tensor, action_chunk_np = _normalize_action_chunk(batch["action"])
                    current_plan_length = self._get_plan_length()
                    actions_executed = max(0, self._plan_length_at_inference_start - current_plan_length)
                    _trace.info(
                        "[dispatch_result] request_id=%s success=True latency_ms=%.2f chunk_size=%d",
                        req_id,
                        result.inference_latency_ms,
                        len(action_chunk_np),
                    )
                    if self._smoother is not None:
                        new_length = self._smoother.update(action_chunk_tensor, actions_executed)
                        self._active_request_id = req_id
                        self._actions_executed_from_active_request = 0
                        self._last_queue_refill_monotonic_ns = time.monotonic_ns()
                        self._plan_generation += 1
                        _trace.info(
                            "[queue_refill] request_id=%s new=%d skipped=%d after=%d",
                            req_id,
                            len(action_chunk_np),
                            actions_executed,
                            new_length,
                        )
                        self.get_logger().debug(
                            f"Smoothed update: {len(action_chunk_np)} new, skipped {actions_executed}, plan={new_length}"
                        )
                    else:
                        relevant_actions = action_chunk_np[actions_executed:]
                        self._queue.clear()
                        self._queue.extend(relevant_actions)
                        self._active_request_id = req_id
                        self._actions_executed_from_active_request = 0
                        self._last_queue_refill_monotonic_ns = time.monotonic_ns()
                        self._plan_generation += 1
                        _trace.info(
                            "[queue_refill] request_id=%s new=%d skipped=%d after=%d",
                            req_id,
                            len(relevant_actions),
                            actions_executed,
                            len(self._queue),
                        )
                        self.get_logger().debug(
                            f"Queue update: {len(relevant_actions)} actions "
                            f"(skipped {actions_executed}), total={len(self._queue)}"
                        )

                    if self._dispatch_count == 1:
                        self.get_logger().info(
                            f"✓ First inference received: "
                            f"chunk={len(action_chunk_np)}, "
                            f"latency={result.inference_latency_ms:.1f}ms, "
                            f"queue={self._get_plan_length()}"
                        )
                    self.get_logger().info(
                        "[IBROBOT_BENCHMARK][INFERENCE_TIMING] "
                        f"request={req_id} start_mono_ns={inference_timing.get('request_start_monotonic_ns', 0)} "
                        f"result_mono_ns={result_monotonic_ns} "
                        f"total_latency_ms={result.inference_latency_ms:.6f} "
                        f"backend_latency_ms={result.backend_latency_ms:.6f}"
                    )
            return

        # Continuous/topic mode: existing behavior without lock.
        self._dispatch_count += 1
        self._total_inference_latency_ms += result.inference_latency_ms

        decode_start = time.perf_counter()
        batch = TensorMsgConverter.from_variant(result.action_chunk)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        _trace.info(
            "[dispatch_decode] request_id=%s chunk_size=%d decode_ms=%.2f",
            req_id,
            result.chunk_size,
            decode_ms,
        )
        if "action" in batch:
            action_chunk_tensor, action_chunk_np = _normalize_action_chunk(batch["action"])
            current_plan_length = self._get_plan_length()
            actions_executed = max(0, self._plan_length_at_inference_start - current_plan_length)
            _trace.info(
                "[dispatch_result] request_id=%s success=True latency_ms=%.2f chunk_size=%d",
                req_id,
                result.inference_latency_ms,
                len(action_chunk_np),
            )
            if self._smoother is not None:
                new_length = self._smoother.update(action_chunk_tensor, actions_executed)
                self._active_request_id = req_id
                self._actions_executed_from_active_request = 0
                self._last_queue_refill_monotonic_ns = time.monotonic_ns()
                self._plan_generation += 1
                _trace.info(
                    "[queue_refill] request_id=%s new=%d skipped=%d after=%d",
                    req_id,
                    len(action_chunk_np),
                    actions_executed,
                    new_length,
                )
                self.get_logger().debug(
                    f"Smoothed update: {len(action_chunk_np)} new, skipped {actions_executed}, plan={new_length}"
                )
            else:
                relevant_actions = action_chunk_np[actions_executed:]
                self._queue.clear()
                self._queue.extend(relevant_actions)
                self._active_request_id = req_id
                self._actions_executed_from_active_request = 0
                self._last_queue_refill_monotonic_ns = time.monotonic_ns()
                self._plan_generation += 1
                _trace.info(
                    "[queue_refill] request_id=%s new=%d skipped=%d after=%d",
                    req_id,
                    len(relevant_actions),
                    actions_executed,
                    len(self._queue),
                )
                self.get_logger().debug(
                    f"Queue update: {len(relevant_actions)} actions "
                    f"(skipped {actions_executed}), total={len(self._queue)}"
                )

            if self._dispatch_count == 1:
                self.get_logger().info(
                    f"✓ First inference received: "
                    f"chunk={len(action_chunk_np)}, "
                    f"latency={result.inference_latency_ms:.1f}ms, "
                    f"queue={self._get_plan_length()}"
                )

    def _reset_cb(self, request, response):
        self.get_logger().info("Resetting dispatcher state")
        # benchmark episode controller: if a benchmark episode goal is active, a legacy ~/reset call is
        # an external reset. Canonical lock ordering: unified lock order _dispatch_lock -> episode._lock.
        if self._is_benchmark and self._episode.is_active_goal:
            gen = self._episode.goal_generation
            with self._dispatch_lock:
                close_result = self._episode.try_close_faulted_with_result(gen, TERMINATION_EXTERNAL_RESET)
                if close_result.applied:
                    self._clear_episode_local_state(invalidate_executor=True, invalidate_inference=True)
                    self._is_running = False
                    self._freeze_and_signal_context_unlocked(
                        gen, close_result.result, TERMINATION_EXTERNAL_RESET, "aborted"
                    )
            if close_result.applied:
                self.get_logger().warn("[IBROBOT_EPISODE][FAULTED] reason=external_reset")
            return response
        # executor lifecycle isolation: isolate executor generation first so late callbacks from the
        # old generation cannot pollute the fresh pipeline. This is a LOCAL
        # reset of dispatcher/executor state; it is NOT an environment episode
        # reset barrier and does NOT claim the remote action was cancelled.
        self._executor.invalidate_pending()
        self._queue.clear()
        if self._smoother is not None:
            self._smoother.reset()
        self._request_generation += 1
        self._inference_in_progress = False
        self._inflight_request_id = ""
        self._inference_started_at = 0.0
        self._request_policy_reset()
        self._plan_length_at_inference_start = 0
        self._last_action = None
        self._active_request_id = ""
        self._actions_executed_from_active_request = 0
        self._last_queue_refill_monotonic_ns = 0
        self._current_request_id = ""
        # Reset scheduler and reservation so old completions become stale.
        # plan_generation bump is in addition to request_generation so any
        # in-flight reservation whose plan was cleared by this reset can be
        # detected at commit time.
        self._scheduler.reset()
        self._reservation_context = None
        self._reservation_plan_generation = 0
        self._plan_generation += 1
        return response

    def _complete_inflight_request(self, request_id: str):
        if request_id != self._inflight_request_id:
            return
        self._inference_in_progress = False
        self._inflight_request_id = ""
        self._inference_started_at = 0.0

    def _expire_inference_if_needed(self):
        """Abandon a stuck inference request so the dispatcher can self-recover.

        Without this, a single inference goal that never returns (server hiccup,
        a goal abandoned during a stop/start, or a lost response) leaves
        ``_inference_in_progress`` True forever and the control loop stops
        requesting new inferences — the dispatcher goes silent.
        """
        if not self._inference_in_progress:
            return
        if self._inference_started_at <= 0.0:
            return
        if time.monotonic() - self._inference_started_at <= self._inference_timeout_s:
            return
        self.get_logger().warn(
            f"Inference request '{self._inflight_request_id}' timed out after "
            f"{self._inference_timeout_s:.1f}s; abandoning to recover dispatch loop"
        )
        # Invalidate the stuck goal's eventual response, then clear the flag so
        # the next control loop tick issues a fresh inference request.
        self._request_generation += 1
        self._inference_in_progress = False
        self._inflight_request_id = ""
        self._inference_started_at = 0.0

    def _expire_policy_reset_if_needed(self):
        if not self._policy_reset_in_progress:
            return
        if self._policy_reset_started_at <= 0.0:
            return
        if time.monotonic() - self._policy_reset_started_at <= self._policy_reset_timeout_s:
            return
        self._policy_reset_in_progress = False
        self._policy_reset_started_at = 0.0
        self.get_logger().warn("Policy reset timed out; continuing with dispatcher reset complete")

    def _request_policy_reset(self):
        """Reset policy-local runtime state for a new episode boundary."""
        service_name = self.get_parameter("inference_reset_service").value
        if not service_name:
            self._policy_reset_in_progress = False
            self._policy_reset_started_at = 0.0
            return

        if not self._policy_reset_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(f"Policy reset service unavailable: {service_name}")
            self._policy_reset_in_progress = False
            self._policy_reset_started_at = 0.0
            return

        self._policy_reset_in_progress = True
        self._policy_reset_started_at = time.monotonic()
        try:
            future = self._policy_reset_client.call_async(Trigger.Request())
            future.add_done_callback(self._policy_reset_done_cb)
        except Exception as e:
            self._policy_reset_in_progress = False
            self._policy_reset_started_at = 0.0
            self.get_logger().warn(f"Policy reset request failed: {e}")

    def _policy_reset_done_cb(self, future):
        self._policy_reset_in_progress = False
        self._policy_reset_started_at = 0.0
        try:
            result = future.result()
            if result is None:
                self.get_logger().warn("Policy reset returned no response")
                return
            if not result.success:
                self.get_logger().warn(f"Policy reset failed: {result.message}")
                return
            self.get_logger().info("Policy runtime state reset")
        except Exception as e:
            self.get_logger().warn(f"Policy reset request failed: {e}")

    def _toggle_smoothing_cb(self, request, response):
        """Toggle smoothing on/off at runtime (requires smoother to be initialized)."""
        if self._smoother is None:
            self.get_logger().warn("Cannot toggle smoothing: smoother not initialized")
            return response

        self._smoothing_enabled = not self._smoothing_enabled
        self._smoother._config.enabled = self._smoothing_enabled
        self._smoother._smoother.config.enabled = self._smoothing_enabled

        self.get_logger().info(f"Temporal smoothing {'ENABLED' if self._smoothing_enabled else 'DISABLED'}")
        return response

    def _stop_base(self):
        """Send zero-velocity command to base controller via TopicExecutor."""
        if self._base_act_spec is None:
            self.get_logger().warn("No base action spec found, cannot stop base")
            return

        from std_msgs.msg import Float64MultiArray

        for topic, info in self._executor._publishers.items():
            if info["spec"] is self._base_act_spec:
                msg = Float64MultiArray()
                msg.data = [0.0, 0.0, 0.0]
                info["pub"].publish(msg)
                self.get_logger().info(f"Published zero base command to {topic}")
                break

    def _start_nav_cb(self, request, response):
        """Start or resume dispatcher evaluation (idempotent clean restart).

        In model_inference mode the dispatcher auto-runs, so a control panel
        cannot rely on a strict not-running precondition. Treat start as
        "ensure a fresh inference cycle is running": invalidate any in-flight
        request and clear the flags that could otherwise wedge the control loop,
        so inference reliably resumes whether or not it was already running.

        benchmark episode controller: benchmark mode rejects start_evaluate — it cannot bypass the
        episode gate. The caller must use PreparePolicyEpisode + RunPolicy.
        """
        if self._is_benchmark:
            response.success = False
            response.message = "benchmark mode requires RunPolicy goal; start_evaluate cannot bypass gate"
            return response
        was_running = self._is_running
        self._request_generation += 1
        self._inference_in_progress = False
        self._inflight_request_id = ""
        self._inference_started_at = 0.0
        self._policy_reset_in_progress = False
        self._policy_reset_started_at = 0.0
        self._is_running = True
        response.success = True
        response.message = "Evaluate resumed" if was_running else "Evaluate started"
        self.get_logger().info(response.message)
        return response

    def _stop_nav_cb(self, request, response):
        """Stop or pause dispatcher evaluation (idempotent).

        Always succeeds and clears any in-flight inference so a later start is
        never blocked by a goal that completes (or hangs) after we stopped.

        benchmark episode controller: benchmark mode rejects stop_evaluate — it is not a pause/resume
        for benchmark episodes. The caller must use RunPolicy action cancel.
        """
        if self._is_benchmark:
            response.success = False
            response.message = "benchmark mode does not support stop_evaluate; use RunPolicy cancel"
            return response
        was_running = self._is_running
        self._is_running = False
        self._inference_in_progress = False
        self._inflight_request_id = ""
        self._inference_started_at = 0.0
        if self._navigation_mode:
            self._stop_base()
        response.success = True
        response.message = "Evaluate stopped" if was_running else "Evaluate already stopped"
        self.get_logger().info(response.message)
        return response

    def _get_status_cb(self, request, response):
        """Get status service callback."""
        if self._is_benchmark:
            response.message = "running" if self._episode.is_gate_open else "stopped"
        elif self._is_running:
            response.message = "running"
        else:
            response.message = "stopped"
        response.success = True
        return response

    # ==================================================================
    # benchmark episode controller: Episode gate, preparation barrier, RunPolicy goal gate
    # ==================================================================

    # Mapping from scheduler fault_status to episode termination reason.
    _FAULT_TO_TERMINATION: dict[str, str] = {
        "timeout_uncertain": TERMINATION_EXECUTION_UNCERTAIN,
        "failed": TERMINATION_EXECUTION_FAILED,
        "uncertain": TERMINATION_EXECUTION_UNCERTAIN,
        "missing_timestamp": TERMINATION_EXECUTION_UNCERTAIN,
        "episode_id_mismatch": TERMINATION_IDENTITY_MISMATCH,
        "step_id_mismatch": TERMINATION_IDENTITY_MISMATCH,
        "rejected": TERMINATION_EXECUTION_REJECTED,
        "double_submission": TERMINATION_EXECUTION_UNCERTAIN,
        "plan_generation_mismatch": TERMINATION_EXECUTION_UNCERTAIN,
        "submission_correlation_mismatch": TERMINATION_EXECUTION_UNCERTAIN,
    }

    # ------------------------------------------------------------------
    # PreparePolicyEpisode service callback (strict preparation barrier)
    # ------------------------------------------------------------------

    def _prepare_episode_cb(self, request, response):
        """Strict preparation barrier: close -> invalidate -> clear -> reset -> token.

        Ordering is test-locked:
        1. Atomically check can_prepare + begin_preparing (under lock).
        2. Invalidate executor pending/completions.
        3. Invalidate old inference generation (best-effort, generation is authoritative).
        4. Clear queue/smoother/last action/reservation/scheduler.
        5. Strict policy reset request with per-prepare generation identity.
        6. On success: complete_prepare -> return token >= 1.
        7. On failure/timeout/exception: fail_prepare -> return false/id=0.

        benchmark episode controller correctness handling: uses ``try_begin_preparing`` (atomic check-and-act)
        to prevent a concurrent goal callback from consuming the token between
        can_prepare() and begin_preparing(). Uses per-prepare generation for
        the policy reset callback so a late callback from prepare N cannot
        complete prepare N+1.
        """
        # 1. Atomically check + transition to PREPARING.
        if not self._episode.try_begin_preparing():
            response.success = False
            response.message = f"cannot prepare: episode phase={self._episode.phase.value}"
            response.preparation_id = 0
            return response

        self.get_logger().info("[IBROBOT_EPISODE][PREPARING] barrier started")

        # 2. Invalidate executor pending/completions.
        self._executor.invalidate_pending()

        # 3. Invalidate old inference generation (best-effort cancel).
        self._request_generation += 1
        self._inference_in_progress = False
        self._inflight_request_id = ""
        self._inference_started_at = 0.0

        # 4. Clear queue/smoother/last action/reservation/scheduler.
        self._queue.clear()
        if self._smoother is not None:
            self._smoother.reset()
        self._reservation_context = None
        self._reservation_plan_generation = 0
        self._plan_generation += 1
        self._scheduler.reset()
        self._last_action = None
        self._active_request_id = ""
        self._actions_executed_from_active_request = 0
        self._last_queue_refill_monotonic_ns = 0

        # 5. Strict policy reset request with per-prepare context.
        # Per-goal concurrency handling: creates a PrepareContext with its own Event.
        # The reset done callback captures the context object (not just a
        # generation integer) and operates on it under _dispatch_lock.
        if not hasattr(self, "_prepare_counter"):
            self._prepare_counter = 0
        with self._dispatch_lock:
            self._prepare_counter += 1
            prep_ctx = PrepareContext(
                generation=self._prepare_counter,
                done_event=threading.Event(),
            )
            self._active_prepare_context = prep_ctx

        self._request_policy_reset_for_prepare(prep_ctx)

        # 6. Bounded wait on this context's own Event.
        timeout = self._policy_reset_timeout_s
        if not prep_ctx.done_event.wait(timeout=timeout):
            # Timeout: mark context as closed under lock.
            with self._dispatch_lock:
                if self._active_prepare_context is prep_ctx:
                    prep_ctx.closed = True
                    self._active_prepare_context = None
                else:
                    # A new prepare has already started; this one is stale.
                    prep_ctx.closed = True
            self._episode.fail_prepare()
            self.get_logger().warn(
                f"[IBROBOT_EPISODE][PREPARE_FAILED] policy reset timeout (gen={prep_ctx.generation})"
            )
            response.success = False
            response.message = "policy reset timeout"
            response.preparation_id = 0
            return response

        # 7. Check reset result from this context.
        if prep_ctx.closed:
            # Context was closed (timeout or superseded by a newer prepare).
            self._episode.fail_prepare()
            self.get_logger().warn(f"[IBROBOT_EPISODE][PREPARE_STALE] gen={prep_ctx.generation} closed")
            response.success = False
            response.message = "prepare superseded or timed out"
            response.preparation_id = 0
            return response
        if not prep_ctx.success:
            self._episode.fail_prepare()
            self.get_logger().warn(f"[IBROBOT_EPISODE][PREPARE_FAILED] {prep_ctx.message} (gen={prep_ctx.generation})")
            response.success = False
            response.message = prep_ctx.message
            response.preparation_id = 0
            return response

        # 8. Complete preparation, issue new token.
        token = self._episode.complete_prepare()
        self.get_logger().info(f"[IBROBOT_EPISODE][PREPARED] preparation_id={token}")
        response.success = True
        response.message = "prepared"
        response.preparation_id = token
        return response

    def _request_policy_reset_for_prepare(self, prep_ctx: PrepareContext):
        """Request a strict policy reset for the prepare barrier.

        Per-goal concurrency handling: captures the PrepareContext object (not just an
        integer generation). The done callback operates on the context under
        _dispatch_lock. A late callback from a previous prepare cannot modify
        the current prepare's context.

        Does NOT use ``spin_until_future_complete`` or nested spin.
        """
        service_name = self.get_parameter("inference_reset_service").value
        if not service_name:
            self._complete_prepare_context(prep_ctx, False, "no reset service configured")
            return

        if not self._policy_reset_client.wait_for_service(timeout_sec=0.2):
            self._complete_prepare_context(prep_ctx, False, f"policy reset service unavailable: {service_name}")
            return

        try:
            future = self._policy_reset_client.call_async(Trigger.Request())
            future.add_done_callback(lambda f, ctx=prep_ctx: self._prepare_policy_reset_done_cb(f, ctx))
        except Exception as exc:
            self._complete_prepare_context(prep_ctx, False, f"policy reset request failed: {exc}")

    def _complete_prepare_context(self, prep_ctx: PrepareContext, success: bool, message: str) -> None:
        """Complete a PrepareContext under _dispatch_lock.

        Per-goal concurrency handling: checks that the context is still the active prepare
        and not already closed before writing success/message and setting the
        Event. Late callbacks are logged as stale and ignored.
        """
        with self._dispatch_lock:
            if prep_ctx.closed:
                self.get_logger().info(
                    f"[IBROBOT_EPISODE][PREPARE_RESET_STALE] gen={prep_ctx.generation} "
                    "context already closed — late callback ignored"
                )
                return
            if self._active_prepare_context is not prep_ctx:
                self.get_logger().info(
                    f"[IBROBOT_EPISODE][PREPARE_RESET_STALE] gen={prep_ctx.generation} "
                    "not active prepare — late callback ignored"
                )
                return
            prep_ctx.success = success
            prep_ctx.message = message
            prep_ctx.done_event.set()

    def _prepare_policy_reset_done_cb(self, future, prep_ctx: PrepareContext):
        """Done callback for the prepare-flow policy reset Future.

        Per-goal concurrency handling: captures the PrepareContext object. All operations
        on the context are under _dispatch_lock. A late callback from a
        previous prepare is logged as stale and cannot modify the current
        prepare's success/message/Event.
        """
        try:
            result = future.result()
            if result is None:
                self._complete_prepare_context(prep_ctx, False, "policy reset returned no response")
            elif not result.success:
                self._complete_prepare_context(prep_ctx, False, result.message)
            else:
                self._complete_prepare_context(prep_ctx, True, "")
        except Exception as exc:
            self._complete_prepare_context(prep_ctx, False, f"policy reset exception: {exc}")

    # ------------------------------------------------------------------
    # RunPolicy ActionServer callbacks
    # ------------------------------------------------------------------

    def _run_policy_goal_cb(self, goal_request):
        """Goal callback: atomically validate + consume token.

        Per-goal concurrency handling: sets phase to ACCEPTED_PENDING (NOT STARTING).
        The gate remains closed until handle_accepted_callback binds the
        goal handle and calls bind_goal. This prevents the control loop
        from doing work before the goal handle is bound.

        Second concurrent goal gets REJECT (token already consumed).
        PreparePolicyEpisode is also rejected (phase is ACCEPTED_PENDING).
        """
        spec = self._build_goal_spec(goal_request)
        error = self._episode.try_accept_goal(spec, time.monotonic_ns())
        if error is not None:
            self.get_logger().warn(f"[IBROBOT_EPISODE][GOAL_REJECTED] {error}")
            return GoalResponse.REJECT

        self.get_logger().info(
            f"[IBROBOT_EPISODE][GOAL_ACCEPTED] preparation_id={spec.preparation_id} "
            f"episode_id={spec.episode_id} max_actions={spec.max_actions} "
            f"gen={self._episode.goal_generation} (pending bind)"
        )
        return GoalResponse.ACCEPT

    def _run_policy_handle_accepted_cb(self, goal_handle):
        """Handle accepted callback: atomically bind + create context + open gate.

        Write-once result handling correctness handling: the entire bind-and-install sequence is under
        _dispatch_lock, preventing reset/abort/cancel from inserting between
        try_bind_goal_for_handle and context registration:

        1. Acquire _dispatch_lock
        2. try_bind_goal_for_handle (ACCEPTED_PENDING -> STARTING)
        3. Create GoalExecutionContext
        4. Register in _goal_contexts[goal_uuid]
        5. Set _active_goal_handle
        6. Release _dispatch_lock
        7. Set scheduler timestamp / readiness deadline (non-critical)
        8. goal_handle.execute()

        If bind fails (superseded), abort the goal handle without leaving
        any context residue.
        """
        goal_uuid = bytes(goal_handle.goal_id.uuid)
        generation = self._episode.goal_generation

        with self._dispatch_lock:
            # Atomically bind + create context + register.
            if not self._episode.try_bind_goal_for_handle(generation, goal_uuid, time.monotonic_ns()):
                self.get_logger().warn(f"[IBROBOT_EPISODE][HANDLE_ACCEPTED] gen={generation} superseded, aborting goal")
                goal_handle.abort()
                return

            ctx = GoalExecutionContext(
                goal_uuid=goal_uuid,
                generation=generation,
                done_event=threading.Event(),
            )
            self._goal_contexts[goal_uuid] = ctx
            self._active_goal_handle = goal_handle

        # Set scheduler timestamp and readiness deadline (outside lock; non-critical).
        spec_ts = self._episode.get_inference_timestamp()
        if spec_ts is not None:
            self._scheduler.set_observation_timestamp(spec_ts)
        self._readiness_retry_deadline_ns = self._episode.readiness_deadline_ns

        self._is_running = True

        self.get_logger().info(f"[IBROBOT_EPISODE][BOUND] gen={generation} uuid={goal_uuid[:4]!r} gate=open")

        goal_handle.execute()

    def _run_policy_cancel_cb(self, goal_handle):
        """Cancel callback: always accept cancel."""
        self.get_logger().info("[IBROBOT_EPISODE][CANCEL_REQUESTED]")
        return CancelResponse.ACCEPT

    def _run_policy_execute_cb(self, goal_handle):
        """Execute callback: wait on this goal's own context Event.

        Per-goal concurrency handling: gets the per-goal GoalExecutionContext by UUID,
        waits on the context's own Event (not a shared Event), and returns
        the frozen result from the context. Does NOT read global state
        machine state after waking. Does NOT clear shared Event.

        The Event is created unset in handle_accepted_callback; this method
        does NOT call Event.clear() (preventing lost-wakeup).
        """
        goal_uuid = bytes(goal_handle.goal_id.uuid)

        # Get this goal's own context.
        with self._dispatch_lock:
            ctx = self._goal_contexts.get(goal_uuid)
        if ctx is None:
            # Context not found; abort.
            self.get_logger().error(f"[IBROBOT_EPISODE][EXECUTE] no context for uuid={goal_uuid[:4]!r}")
            goal_handle.abort()
            return RunPolicy.Result()

        # Wait on this goal's own Event. No clear() — the Event was created
        # unset in handle_accepted_callback and is only set by the terminal
        # path when this goal's generation matches.
        ctx.done_event.wait()

        # Return the frozen result from this goal's context. Do NOT read
        # global state machine state — the context was frozen at terminal time.
        result = self._build_run_policy_result_from_context(ctx)
        reason = ctx.termination_reason or ""

        # Clean up this goal's context only (not other goals' contexts).
        with self._dispatch_lock:
            # Only clear the active goal handle if this goal is still active.
            if self._active_goal_handle is goal_handle:
                self._active_goal_handle = None
            self._goal_contexts.pop(goal_uuid, None)

        # Set terminal status on the goal handle.
        if ctx.terminal_status == "canceled":
            self.get_logger().info(f"[IBROBOT_EPISODE][CANCELED] reason={reason}")
            goal_handle.canceled()
        elif ctx.terminal_status == "succeeded":
            self.get_logger().info(f"[IBROBOT_EPISODE][SUCCEEDED] reason={reason}")
            goal_handle.succeed()
        else:
            self.get_logger().warn(f"[IBROBOT_EPISODE][ABORTED] reason={reason}")
            goal_handle.abort()

        return result

    # ------------------------------------------------------------------
    # Episode terminal / cancel / abort
    # ------------------------------------------------------------------
    # Canonical lock order (Canonical lock ordering): _dispatch_lock -> EpisodeStateMachine._lock
    # No ROS Future/Event wait while holding _dispatch_lock.
    # terminal/bind/reset all acquire _dispatch_lock first, then call
    # try_close_*_with_result (which acquires _episode._lock internally).

    def _close_episode_terminal(self, reason: str) -> None:
        """Close terminal with atomic result snapshot.

        Canonical lock ordering lock order: _dispatch_lock -> episode._lock (via try_close_*_with_result).
        """
        gen = self._episode.goal_generation
        with self._dispatch_lock:
            close_result = self._episode.try_close_terminal_with_result(gen, reason)
            if not close_result.applied:
                return
            self._clear_episode_local_state(invalidate_executor=False, invalidate_inference=False)
            self._is_running = False
            self._freeze_and_signal_context_unlocked(gen, close_result.result, reason, "succeeded")
        self.get_logger().info(f"[IBROBOT_EPISODE][TERMINAL] reason={reason}")

    def _abort_episode(self, reason: str) -> None:
        """Abort with atomic result snapshot.

        Canonical lock ordering lock order: _dispatch_lock -> episode._lock.
        """
        gen = self._episode.goal_generation
        with self._dispatch_lock:
            close_result = self._episode.try_close_faulted_with_result(gen, reason)
            if not close_result.applied:
                return
            self._clear_episode_local_state(invalidate_executor=True, invalidate_inference=True)
            self._is_running = False
            self._freeze_and_signal_context_unlocked(gen, close_result.result, reason, "aborted")
        self.get_logger().warn(f"[IBROBOT_EPISODE][FAULTED] reason={reason}")

    def _cancel_episode(self) -> None:
        """Cancel with atomic result snapshot.

        Canonical lock ordering lock order: _dispatch_lock -> episode._lock.
        """
        gen = self._episode.goal_generation
        with self._dispatch_lock:
            close_result = self._episode.try_cancel_with_result(gen)
            if not close_result.applied:
                return
            self._clear_episode_local_state(invalidate_executor=True, invalidate_inference=True)
            self._is_running = False
            self._freeze_and_signal_context_unlocked(gen, close_result.result, TERMINATION_CANCELED, "canceled")
        self.get_logger().info("[IBROBOT_EPISODE][CANCELED]")

    def _freeze_and_signal_context(self, generation: int, result: EpisodeResult, reason: str, status: str) -> None:
        """Public entry: acquire _dispatch_lock then call _unlocked version."""
        with self._dispatch_lock:
            self._freeze_and_signal_context_unlocked(generation, result, reason, status)

    def _freeze_and_signal_context_unlocked(
        self, generation: int, result: EpisodeResult, reason: str, status: str
    ) -> None:
        """Freeze result in context (caller holds _dispatch_lock).

        Write-once result handling: receives already-frozen EpisodeResult. Write-once: refuses
        overwrite if ctx.result is already set.
        """
        ctx = None
        for c in self._goal_contexts.values():
            if c.generation == generation:
                ctx = c
                break
        if ctx is None:
            self.get_logger().warn(f"[IBROBOT_EPISODE][FREEZE] no context for gen={generation}")
            return
        if ctx.result is not None:
            self.get_logger().warn(f"[IBROBOT_EPISODE][FREEZE] gen={generation} already frozen")
            return
        ctx.result = result
        ctx.termination_reason = reason
        ctx.terminal_status = status
        ctx.done_event.set()

    def _clear_episode_local_state(self, *, invalidate_executor: bool, invalidate_inference: bool) -> None:
        """Clear local dispatcher state after episode end.

        - ``invalidate_executor``: call ``executor.invalidate_pending()``.
        - ``invalidate_inference``: bump ``_request_generation`` and clear
          inference flags so any in-flight goal's response is stale.
        """
        if invalidate_executor:
            self._executor.invalidate_pending()
        if invalidate_inference:
            self._request_generation += 1
            self._inference_in_progress = False
            self._inflight_request_id = ""
            self._inference_started_at = 0.0
        self._queue.clear()
        if self._smoother is not None:
            self._smoother.reset()
        self._reservation_context = None
        self._reservation_plan_generation = 0
        self._plan_generation += 1
        self._scheduler.reset()
        self._last_action = None
        self._active_request_id = ""
        self._actions_executed_from_active_request = 0
        self._last_queue_refill_monotonic_ns = 0

    # ------------------------------------------------------------------
    # Helpers: goal spec, result, step data, feedback, fault mapping
    # ------------------------------------------------------------------

    def _build_goal_spec(self, goal) -> EpisodeGoalSpec:
        """Convert a RunPolicy.Goal into an EpisodeGoalSpec."""
        ts = goal.initial_observation_timestamp
        initial_ts_ns = int(ts.sec) * 1_000_000_000 + int(ts.nanosec)
        return EpisodeGoalSpec(
            preparation_id=int(goal.preparation_id),
            episode_id=int(goal.episode_id),
            initial_step_id=int(goal.initial_step_id),
            initial_observation_timestamp_ns=initial_ts_ns,
            max_actions=int(goal.max_actions),
            max_duration_sec=float(goal.max_duration_sec),
            startup_timeout_sec=float(goal.startup_timeout_sec),
            prompt=str(goal.prompt),
        )

    def _build_run_policy_result(self):
        """Build a RunPolicy.Result from the episode state machine (legacy)."""
        from ibrobot_msgs.action import RunPolicy as _RunPolicy

        ep_result = self._episode.build_result()
        result = _RunPolicy.Result()
        result.success = ep_result.success
        result.message = ep_result.termination_reason
        result.has_success = ep_result.has_success
        result.termination_reason = ep_result.termination_reason
        result.episode_id = ep_result.episode_id
        result.has_final_step = ep_result.has_final_step
        result.final_step_id = ep_result.final_step_id
        result.published_actions = ep_result.published_actions
        result.has_reward = ep_result.has_reward
        result.reward = ep_result.reward
        result.terminated = ep_result.terminated
        result.truncated = ep_result.truncated
        result.standard_metrics_json = ep_result.standard_metrics_json
        result.native_metrics_json = ep_result.native_metrics_json
        result.info_json = ep_result.info_json
        result.round_trip_latency_ms = ep_result.round_trip_latency_ms
        return result

    def _build_run_policy_result_from_context(self, ctx: GoalExecutionContext):
        """Build a RunPolicy.Result from the frozen GoalExecutionContext.

        Per-goal concurrency handling: returns the frozen result from the context, not
        the current state machine state. This ensures old execute callbacks
        return their own episode's result, not a subsequent episode's.
        """
        from ibrobot_msgs.action import RunPolicy as _RunPolicy

        result = _RunPolicy.Result()
        if ctx.result is not None:
            ep = ctx.result
            result.success = ep.success
            result.has_success = ep.has_success
            result.termination_reason = ep.termination_reason
            result.episode_id = ep.episode_id
            result.has_final_step = ep.has_final_step
            result.final_step_id = ep.final_step_id
            result.published_actions = ep.published_actions
            result.has_reward = ep.has_reward
            result.reward = ep.reward
            result.terminated = ep.terminated
            result.truncated = ep.truncated
            result.standard_metrics_json = ep.standard_metrics_json
            result.native_metrics_json = ep.native_metrics_json
            result.info_json = ep.info_json
            result.round_trip_latency_ms = ep.round_trip_latency_ms
            result.message = ep.termination_reason
        return result

    def _build_step_completion_data(self, completion, queue_depth: int) -> StepCompletionData:
        """Extract StepCompletionData from an ExecutionCompletion.

        Raw JSON strings are preserved byte-for-byte; no parsing/merging.
        The ``queue_depth`` parameter is for feedback metadata but is not
        stored in the immutable StepCompletionData.
        """
        details = completion.details
        info_json = str(details.get("info_json", "{}"))
        try:
            info = json.loads(info_json)
        except (TypeError, json.JSONDecodeError):
            info = {}
        if isinstance(info, dict):
            performance = info.setdefault("benchmark_performance", {"schema_version": 1})
            if isinstance(performance, dict):
                reservation = self._reservation_context
                metadata = dict(reservation.metadata) if reservation is not None else {}
                request_id = str(metadata.get("request_id", ""))
                performance["inference"] = dict(self._benchmark_inference_timings.get(request_id, {}))
                performance["action"] = {
                    "reservation_monotonic_ns": int(metadata.get("action_reservation_monotonic_ns", 0)),
                    "submit_monotonic_ns": int(details.get("transport.submit_monotonic_ns", 0)),
                    "completion_monotonic_ns": int(details.get("transport.completion_monotonic_ns", 0)),
                    "dispatcher_commit_monotonic_ns": time.monotonic_ns(),
                    "step_service_round_trip_latency_ms": float(details.get("transport.round_trip_latency_ms", 0.0)),
                }
                info_json = json.dumps(info, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return StepCompletionData(
            episode_id=completion.episode_id,
            step_id=completion.step_id,
            observation_timestamp_ns=completion.observation_timestamp_ns,
            has_reward=bool(details.get("has_reward", False)),
            reward=float(details.get("reward", 0.0)),
            terminated=bool(details.get("terminated", False)),
            truncated=bool(details.get("truncated", False)),
            has_success=bool(details.get("has_is_success", False)),
            success=bool(details.get("is_success", False)),
            standard_metrics_json=str(details.get("standard_metrics_json", "{}")),
            native_metrics_json=str(details.get("native_metrics_json", "{}")),
            info_json=info_json,
            round_trip_latency_ms=float(details.get("transport.round_trip_latency_ms", 0.0)),
            message=str(completion.message),
        )

    def _publish_step_feedback(self, step_data: StepCompletionData, queue_depth: int) -> None:
        """Publish exactly one RunPolicy feedback per matching COMMIT step.

        Carries full typed flags, raw JSON (byte-for-byte) and transport
        round-trip latency. No aggregation, no file IO.
        """
        if self._active_goal_handle is None:
            return
        from ibrobot_msgs.action import RunPolicy as _RunPolicy

        feedback = _RunPolicy.Feedback()
        feedback.published_actions = self._episode.committed_count
        feedback.queue_depth = queue_depth
        feedback.status = "committed"
        feedback.episode_id = step_data.episode_id if step_data.episode_id is not None else 0
        feedback.step_id = step_data.step_id if step_data.step_id is not None else 0
        if step_data.observation_timestamp_ns is not None and step_data.observation_timestamp_ns > 0:
            ts_ns = step_data.observation_timestamp_ns
            feedback.observation_timestamp.sec = int(ts_ns // 1_000_000_000)
            feedback.observation_timestamp.nanosec = int(ts_ns % 1_000_000_000)
        feedback.has_reward = step_data.has_reward
        feedback.reward = step_data.reward
        feedback.terminated = step_data.terminated
        feedback.truncated = step_data.truncated
        feedback.has_success = step_data.has_success
        feedback.success = step_data.success
        feedback.standard_metrics_json = step_data.standard_metrics_json
        feedback.native_metrics_json = step_data.native_metrics_json
        feedback.info_json = step_data.info_json
        feedback.round_trip_latency_ms = step_data.round_trip_latency_ms
        self._active_goal_handle.publish_feedback(feedback)

    @classmethod
    def _map_scheduler_fault_to_termination(cls, fault_status, completion) -> str:
        """Map a scheduler fault_status to an episode termination reason."""
        if fault_status is not None:
            return cls._FAULT_TO_TERMINATION.get(fault_status, TERMINATION_EXECUTION_UNCERTAIN)
        return TERMINATION_EXECUTION_UNCERTAIN


def main(args=None):
    rclpy.init(args=args)
    node = ActionDispatcherNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
