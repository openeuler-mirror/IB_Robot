#!/usr/bin/env python3
"""
Minimal Action Dispatcher Node.

Maintains a queue of actions and triggers inference when low.
Publishes actions to ros2_control via TopicExecutor at a fixed frequency.

Supports cross-frame temporal smoothing for action chunks.
"""

import collections

# Business tracepoints via Python logging.
# When lttngust is imported, these are auto-captured by LTTng as
# python:logging events — no wrapper package needed.
import time
import uuid

import numpy as np
import rclpy
import rclpy.action
import torch
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty, Trigger

from ibrobot_msgs.action import DispatchInfer
from robot_config.contract_utils import iter_specs
from robot_config.tracing_utils import create_trace_logger
from tensormsg.converter import TensorMsgConverter

from .temporal_smoother import (
    TemporalSmootherManager,
)
from .topic_executor import TopicExecutor

_trace = create_trace_logger("ib_trace.dispatch")


class ActionDispatcherNode(Node):
    """
    Simplified action dispatcher.
    - Queue: collections.deque (when smoothing disabled) or TemporalSmoother (when enabled)
    - Trigger: Simple watermark check
    - Execution: TopicExecutor (100Hz streaming)

    Cross-frame smoothing can be enabled via parameters to ensure smooth
    transitions between consecutive action chunks.
    """

    def __init__(self):
        super().__init__("action_dispatcher")
        self.get_logger().info("Initializing Action Dispatcher")

        # 1. Parameters
        self.declare_parameter("queue_size", 100)
        self.declare_parameter("watermark_threshold", 20)
        self.declare_parameter("control_frequency", 100.0)
        self.declare_parameter("inference_action_server", "/act_inference_node/DispatchInfer")
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("navigation_mode", False)

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

        # 5. Executor (Topic-based)
        self._executor = TopicExecutor(self, {"action_specs": self._action_specs})
        if not self._executor.initialize():
            raise RuntimeError("Failed to initialize TopicExecutor")

        # 6. Communication
        self._infer_client = rclpy.action.ActionClient(self, DispatchInfer, self._server_name)

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

        # Navigation mode services (only useful when navigation_mode=True)
        self._start_nav_srv = self.create_service(Trigger, "~/start_evaluate", self._start_nav_cb)
        self._stop_nav_srv = self.create_service(Trigger, "~/stop_evaluate", self._stop_nav_cb)
        self._get_status_srv = self.create_service(Trigger, "~/get_status", self._get_status_cb)

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

    def _joint_cb(self, msg):
        """Optional: could use current state for safety or initialization."""
        pass

    def _get_plan_length(self) -> int:
        """Get current plan length (works for both modes)."""
        if self._smoother is not None:
            return self._smoother.plan_length
        return len(self._queue)

    def _control_loop(self):
        if not self._is_running:
            return

        q_size = self._get_plan_length()
        self._queue_size_pub.publish(Int32(data=q_size))
        self._smoothing_enabled_pub.publish(Bool(data=self._smoothing_enabled))

        # A. Trigger Inference if queue is low
        if q_size <= self._watermark and not self._inference_in_progress:
            self._request_inference()

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

    def _request_inference(self):
        """Send async goal to inference service."""
        if not self._infer_client.wait_for_server(timeout_sec=0.1):
            return

        self._inference_in_progress = True
        self._plan_length_at_inference_start = self._get_plan_length()
        self._current_request_id = uuid.uuid4().hex[:8]

        goal = DispatchInfer.Goal()
        goal.obs_timestamp = self.get_clock().now().to_msg()
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
        send_goal_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Inference goal REJECTED")
            self._inference_in_progress = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._inference_in_progress = False
        result = future.result().result
        req_id = getattr(self, "_current_request_id", "")
        if not result.success:
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

        self._dispatch_count += 1
        self._total_inference_latency_ms += result.inference_latency_ms

        # Decode VariantsList to Numpy/Tensor
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
            action_chunk = batch["action"]

            # Convert to Numpy if it's a Torch Tensor
            if hasattr(action_chunk, "detach"):
                action_chunk_tensor = action_chunk
                action_chunk_np = action_chunk.detach().cpu().numpy()
            else:
                action_chunk_tensor = torch.from_numpy(action_chunk)
                action_chunk_np = action_chunk

            # Reshape to (N, action_dim)
            if action_chunk_np.ndim == 1:
                action_chunk_np = action_chunk_np.reshape(1, -1)
                action_chunk_tensor = action_chunk_tensor.reshape(1, -1)

            # Calculate actions executed during inference
            current_plan_length = self._get_plan_length()
            actions_executed = max(0, self._plan_length_at_inference_start - current_plan_length)

            _trace.info(
                "[dispatch_result] request_id=%s success=True latency_ms=%.2f chunk_size=%d",
                req_id,
                result.inference_latency_ms,
                len(action_chunk_np),
            )

            if self._smoother is not None:
                # Use smoother for cross-frame smoothing
                new_length = self._smoother.update(action_chunk_tensor, actions_executed)
                self._active_request_id = req_id
                self._actions_executed_from_active_request = 0
                self._last_queue_refill_monotonic_ns = time.monotonic_ns()
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
                # Simple queue mode: align and replace
                relevant_actions = action_chunk_np[actions_executed:]
                self._queue.clear()
                self._queue.extend(relevant_actions)
                self._active_request_id = req_id
                self._actions_executed_from_active_request = 0
                self._last_queue_refill_monotonic_ns = time.monotonic_ns()
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
        self._queue.clear()
        if self._smoother is not None:
            self._smoother.reset()
        self._inference_in_progress = False
        self._plan_length_at_inference_start = 0
        self._last_action = None
        self._active_request_id = ""
        self._actions_executed_from_active_request = 0
        self._last_queue_refill_monotonic_ns = 0
        self._current_request_id = ""
        return response

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
        """Start evaluate service callback."""
        if not self._navigation_mode:
            response.success = False
            response.message = "Navigation mode is not enabled"
            return response

        if self._is_running:
            response.success = False
            response.message = "Evaluate is already running"
            return response

        self._is_running = True
        self.get_logger().info("Evaluate started")
        response.success = True
        response.message = "Evaluate started"
        return response

    def _stop_nav_cb(self, request, response):
        """Stop evaluate service callback."""
        if not self._navigation_mode:
            response.success = False
            response.message = "Navigation mode is not enabled"
            return response

        if not self._is_running:
            response.success = False
            response.message = "Evaluate is already stopped"
            return response

        self._is_running = False
        self._stop_base()
        self.get_logger().info("Evaluate stopped")
        response.success = True
        response.message = "Evaluate stopped"
        return response

    def _get_status_cb(self, request, response):
        """Get status service callback."""
        if self._is_running:
            response.message = "running"
        else:
            response.message = "stopped"
        response.success = True
        return response


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
