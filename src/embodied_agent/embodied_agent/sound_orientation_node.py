"""ROS adapter for fixed-trigger sound orientation."""

from __future__ import annotations

import math
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from unique_identifier_msgs.msg import UUID

from embodied_common.dispatch_binding import new_binding
from embodied_common.skill_request import skill_goal_uuid
from ibrobot_msgs.action import SkillCommand
from ibrobot_msgs.msg import SpeechDirection
from ibrobot_msgs.srv import GetSkillGatewayStatus

from .sound_orientation_policy import (
    DecisionKind,
    DirectionSample,
    GatewaySnapshot,
    OrientationPolicyConfig,
    PolicyDecision,
    SoundOrientationPolicy,
)

_UNKNOWN_TERMINAL_CODES = {
    "GATEWAY_FINALIZATION_FAILED",
    "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
    "SKILL_CANCEL_TIMEOUT",
}


class SoundOrientationNode(Node):
    """Trigger one guarded ``nav_turn`` from an exact ASR phrase."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("sound_orientation_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("trigger_phrases", ["转向我"])
        self.declare_parameter("direction_topic", "/voice/speech_direction")
        self.declare_parameter("command_topic", "/voice_command")
        self.declare_parameter("gateway_status_service", "/embodied/get_skill_gateway_status")
        self.declare_parameter("skill_action_name", "/embodied/execute_skill")
        self.declare_parameter("skill_name", "nav_turn")
        self.declare_parameter("direction_frame", "base_link")
        self.declare_parameter("deadband_deg", 15.0)
        self.declare_parameter("max_direction_age_sec", 1.3)
        self.declare_parameter("direction_wait_sec", 0.5)
        self.declare_parameter("cooldown_sec", 1.5)
        self.declare_parameter("max_turn_deg", 180.0)
        self.declare_parameter("turn_timeout_sec", 10.0)
        self.declare_parameter("action_acceptance_timeout_sec", 2.0)
        self.declare_parameter("status_retry_sec", 0.5)
        self.declare_parameter("reset_status_max_age_sec", 2.0)
        self.declare_parameter("debug_tracing", False)

        trigger_phrases = tuple(self.get_parameter("trigger_phrases").value)
        self._skill_name = str(self.get_parameter("skill_name").value).strip()
        if self._skill_name != "nav_turn":
            raise ValueError("sound orientation currently supports only skill_name=nav_turn")
        self._turn_timeout_sec = float(self.get_parameter("turn_timeout_sec").value)
        self._action_acceptance_timeout_sec = float(self.get_parameter("action_acceptance_timeout_sec").value)
        self._status_retry_sec = float(self.get_parameter("status_retry_sec").value)
        self._reset_status_max_age_sec = float(self.get_parameter("reset_status_max_age_sec").value)
        if not math.isfinite(self._turn_timeout_sec) or self._turn_timeout_sec <= 0.0:
            raise ValueError("turn_timeout_sec must be finite and positive")
        if not math.isfinite(self._action_acceptance_timeout_sec) or self._action_acceptance_timeout_sec <= 0.0:
            raise ValueError("action_acceptance_timeout_sec must be finite and positive")
        if not math.isfinite(self._status_retry_sec) or self._status_retry_sec <= 0.0:
            raise ValueError("status_retry_sec must be finite and positive")
        if not math.isfinite(self._reset_status_max_age_sec) or self._reset_status_max_age_sec <= 0.0:
            raise ValueError("reset_status_max_age_sec must be finite and positive")

        self._policy = SoundOrientationPolicy(
            OrientationPolicyConfig(
                trigger_phrases=trigger_phrases,
                direction_frame=str(self.get_parameter("direction_frame").value),
                deadband_deg=float(self.get_parameter("deadband_deg").value),
                max_direction_age_sec=float(self.get_parameter("max_direction_age_sec").value),
                direction_wait_sec=float(self.get_parameter("direction_wait_sec").value),
                cooldown_sec=float(self.get_parameter("cooldown_sec").value),
                max_turn_deg=float(self.get_parameter("max_turn_deg").value),
            )
        )
        self._debug = bool(self.get_parameter("debug_tracing").value)
        self._lock = threading.RLock()
        self._status_future = None
        self._status_request_generation = 0
        self._status_deadline = 0.0
        self._status_needed = False
        self._next_status_retry = 0.0
        self._active_goal_handle = None
        self._active_task_id = ""
        self._goal_send_future = None
        self._goal_result_future = None
        self._goal_generation = 0
        self._goal_acceptance_deadline = 0.0
        self._goal_result_deadline = 0.0
        self._last_gateway_snapshot = None
        self._last_status_monotonic = 0.0
        self._fault_entered_monotonic = 0.0

        callback_group = ReentrantCallbackGroup()
        self._status_client = self.create_client(
            GetSkillGatewayStatus,
            str(self.get_parameter("gateway_status_service").value),
            callback_group=callback_group,
        )
        self._skill_client = ActionClient(
            self,
            SkillCommand,
            str(self.get_parameter("skill_action_name").value),
            callback_group=callback_group,
        )
        self._direction_sub = self.create_subscription(
            SpeechDirection,
            str(self.get_parameter("direction_topic").value),
            self._direction_callback,
            1,
            callback_group=callback_group,
        )
        self._command_sub = self.create_subscription(
            String,
            str(self.get_parameter("command_topic").value),
            self._command_callback,
            10,
            callback_group=callback_group,
        )
        self._timer = self.create_timer(0.05, self._timer_callback, callback_group=callback_group)
        self._reset_service = self.create_service(Trigger, "~/reset_fault", self._reset_fault_callback)

        self.get_logger().info(
            f"sound orientation ready: triggers={list(self._policy.config.trigger_phrases)}, skill={self._skill_name}"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1_000_000_000

    def _direction_callback(self, msg: SpeechDirection) -> None:
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1_000_000_000
        sample = DirectionSample(
            seq_id=int(msg.seq_id),
            stamp_sec=stamp_sec,
            azimuth_rad=float(msg.azimuth_rad),
            frame_id=str(msg.header.frame_id),
        )
        with self._lock:
            if self._policy.state.value == "fault_unknown":
                return
            decision = self._policy.handle_direction(
                sample,
                now_sec=self._now_sec(),
            )
            self._consume_decision(decision)
            if decision.kind is DecisionKind.WAITING:
                self._status_needed = True

    def _command_callback(self, msg) -> None:
        with self._lock:
            if self._policy.state.value == "fault_unknown":
                return
            decision = self._policy.handle_text(
                msg.data or "",
                now_sec=self._now_sec(),
            )
            self._consume_decision(decision)
            if decision.kind is DecisionKind.WAITING:
                self._status_needed = True

    def _timer_callback(self) -> None:
        with self._lock:
            now = self._now_sec()
            tick_decision = self._policy.tick(now_sec=now)
            self._consume_decision(tick_decision)
            if tick_decision.reason == "DIRECTION_WAIT_TIMEOUT":
                self._status_needed = False
            if (
                (self._status_needed or self._policy.state.value == "fault_unknown")
                and self._status_future is None
                and time.monotonic() >= self._next_status_retry
            ):
                self._request_status()
            self._watchdog_tick(now)

    def _request_status(self) -> None:
        if not self._status_client.service_is_ready():
            self._next_status_retry = time.monotonic() + self._status_retry_sec
            return
        request = GetSkillGatewayStatus.Request()
        request.schema_version = 1
        try:
            self._status_request_generation += 1
            generation = self._status_request_generation
            self._status_future = self._status_client.call_async(request)
            self._status_deadline = time.monotonic() + self._action_acceptance_timeout_sec
            self._status_future.add_done_callback(lambda future: self._status_done(future, generation))
        except Exception as exc:
            self._status_future = None
            self._next_status_retry = time.monotonic() + self._status_retry_sec
            self.get_logger().warning(f"sound orientation status request failed: {exc}")

    def _status_done(self, future, generation: int) -> None:
        with self._lock:
            if generation != self._status_request_generation:
                return
            self._status_future = None
            self._status_deadline = 0.0
            try:
                status = future.result()
            except Exception as exc:
                self._next_status_retry = time.monotonic() + self._status_retry_sec
                self.get_logger().warning(f"sound orientation status response failed: {exc}")
                return

            gateway = self._gateway_snapshot(status)
            self._last_gateway_snapshot = gateway
            self._last_status_monotonic = time.monotonic()
            decision = self._policy.try_dispatch(now_sec=self._now_sec(), gateway=gateway)
            self._consume_decision(decision)
            self._status_needed = decision.kind is DecisionKind.WAITING
            if decision.kind is DecisionKind.WAITING and decision.reason == "WAITING_FOR_DIRECTION":
                self._status_needed = False
            if decision.kind is DecisionKind.DISPATCH:
                self._send_turn_goal(decision, status)

    def _gateway_snapshot(self, status) -> GatewaySnapshot:
        capability = next(
            (item for item in status.capabilities if item.name == self._skill_name),
            None,
        )
        return GatewaySnapshot(
            control_plane_ready=bool(status.control_plane_ready),
            motion_authorized=bool(status.motion_authorized),
            busy=bool(status.busy),
            capability_ready=bool(capability and capability.ready),
            active_control_mode=str(status.active_control_mode),
            required_control_mode=str(capability.required_control_mode) if capability else "base_navigation",
            # The Gateway owns mode switching. This field only avoids rejecting
            # a valid hybrid runtime during the client's early preflight.
            control_mode_switching_enabled=True,
        )

    def _send_turn_goal(self, decision: PolicyDecision, status) -> None:
        request = decision.request
        if request is None or self._active_goal_handle is not None:
            self._finish_known_action("NO_ACTIVE_OR_DUPLICATE_REQUEST")
            return
        if not self._skill_client.server_is_ready():
            self._finish_known_action("SKILL_ACTION_NOT_READY")
            return

        task_id = f"sound-orientation/{request.direction_event_key[0]}/{int(request.direction_event_key[1] * 1e9)}"
        timeout_sec = min(self._turn_timeout_sec, float(status.task_budget_sec))
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            self._finish_known_action("INVALID_TASK_BUDGET")
            return

        goal = SkillCommand.Goal()
        goal.schema_version = 2
        goal.dispatch_binding = new_binding(task_id=task_id)
        goal.dispatch_binding.expected_registry_epoch = str(status.registry_epoch)
        goal.dispatch_binding.expected_registry_generation = int(status.registry_generation)
        goal.dispatch_binding.expected_registry_digest = str(status.registry_digest)
        self._set_task_budget(goal.dispatch_binding, timeout_sec=float(status.task_budget_sec))
        goal.skill_name = self._skill_name
        goal.direction = request.direction
        goal.degree = float(request.degree)
        goal.timeout_sec = timeout_sec

        try:
            goal_uuid = UUID(uuid=list(skill_goal_uuid(task_id).bytes))
            send_future = self._skill_client.send_goal_async(goal, goal_uuid=goal_uuid)
            self._active_task_id = task_id
            self._goal_generation += 1
            generation = self._goal_generation
            self._goal_send_future = send_future
            self._goal_acceptance_deadline = time.monotonic() + min(
                self._action_acceptance_timeout_sec,
                self._turn_timeout_sec,
            )
            send_future.add_done_callback(lambda future: self._goal_response_done(future, generation))
        except Exception as exc:
            self._active_task_id = ""
            self._policy.complete_action(now_sec=self._now_sec(), terminal_known=False)
            self.get_logger().error(f"sound orientation goal submission failed: {exc}")

    def _goal_response_done(self, future, generation: int) -> None:
        with self._lock:
            if generation != self._goal_generation:
                return
            self._goal_send_future = None
            self._goal_acceptance_deadline = 0.0
            try:
                goal_handle = future.result()
            except Exception as exc:
                self._fail_unknown("SKILL_ACTION_ACCEPTANCE_FAILED")
                self.get_logger().error(f"sound orientation goal acceptance failed: {exc}")
                return
            if goal_handle is None or not goal_handle.accepted:
                self._active_task_id = ""
                self._finish_known_action("SKILL_GOAL_REJECTED")
                return
            self._active_goal_handle = goal_handle
            self._policy.mark_action_submitted()
            try:
                result_future = goal_handle.get_result_async()
            except Exception as exc:
                self._fail_unknown("SKILL_RESULT_SUBSCRIPTION_FAILED", goal_handle)
                self.get_logger().error(f"sound orientation result subscription failed: {exc}")
                return
            self._goal_result_future = result_future
            self._goal_result_deadline = time.monotonic() + self._turn_timeout_sec
            result_future.add_done_callback(lambda future: self._result_done(future, generation))

    def _result_done(self, future, generation: int) -> None:
        with self._lock:
            if generation != self._goal_generation:
                return
            self._goal_result_future = None
            self._goal_result_deadline = 0.0
            try:
                action_result = future.result()
            except Exception as exc:
                self._fail_unknown("SKILL_ACTION_RESULT_FAILED", self._active_goal_handle)
                self.get_logger().error(f"sound orientation terminal result is unknown: {exc}")
                return
            status = int(getattr(action_result, "status", GoalStatus.STATUS_UNKNOWN))
            terminal_known = status in {
                GoalStatus.STATUS_SUCCEEDED,
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED,
            }
            result = getattr(action_result, "result", None)
            error_code = str(getattr(result, "error_code", "")) if result is not None else ""
            success = bool(getattr(result, "success", False)) if result is not None else False
            if result is not None:
                self.get_logger().info(
                    f"sound orientation finished: success={bool(result.success)} error_code={str(result.error_code)}"
                )
            self._active_goal_handle = None
            self._active_task_id = ""
            # Cancellation is not proof that the base reached stable zero
            # velocity. Only succeeded/aborted with a result is a known terminal.
            self._policy.complete_action(
                now_sec=self._now_sec(),
                terminal_known=(
                    terminal_known
                    and result is not None
                    and error_code not in _UNKNOWN_TERMINAL_CODES
                    and (
                        (status == GoalStatus.STATUS_SUCCEEDED and success and not error_code)
                        or (status == GoalStatus.STATUS_ABORTED and not success and bool(error_code))
                    )
                ),
            )

    def _watchdog_tick(self, now_sec: float) -> None:
        now_monotonic = time.monotonic()
        if self._status_future is not None and now_monotonic >= self._status_deadline:
            self._status_request_generation += 1
            self._status_future = None
            self._status_deadline = 0.0
            self._status_needed = False
            self._policy.drop_pending()
            self.get_logger().warning("sound orientation Gateway status request timed out")
        if self._goal_send_future is not None and now_monotonic >= self._goal_acceptance_deadline:
            self._goal_generation += 1
            self._goal_send_future = None
            self._goal_acceptance_deadline = 0.0
            self._fail_unknown("SKILL_ACTION_ACCEPTANCE_TIMEOUT")
        if self._goal_result_future is not None and now_monotonic >= self._goal_result_deadline:
            goal_handle = self._active_goal_handle
            self._goal_generation += 1
            self._goal_result_future = None
            self._goal_result_deadline = 0.0
            self._fail_unknown("SKILL_ACTION_RESULT_TIMEOUT", goal_handle)

    def _fail_unknown(self, reason: str, goal_handle=None) -> None:
        """Stop automatic dispatch when action state or physical stop is unknown."""

        self._active_goal_handle = None
        self._active_task_id = ""
        self._goal_send_future = None
        self._goal_result_future = None
        self._goal_acceptance_deadline = 0.0
        self._goal_result_deadline = 0.0
        self._last_gateway_snapshot = None
        self._last_status_monotonic = 0.0
        self._fault_entered_monotonic = time.monotonic()
        self._status_needed = True
        self._next_status_retry = 0.0
        self._policy.complete_action(now_sec=self._now_sec(), terminal_known=False)
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().error(f"sound orientation best-effort cancel failed: {exc}")
        self.get_logger().error(f"sound orientation entered FAULT_UNKNOWN: {reason}")

    def _reset_fault_callback(self, _request, response):
        with self._lock:
            if self._policy.state.value != "fault_unknown":
                response.success = True
                response.message = "sound orientation is not in fault_unknown"
                return response
            if self._active_goal_handle is not None or self._goal_send_future is not None:
                response.success = False
                response.message = "active sound orientation action is still unresolved"
                return response
            if (
                self._last_gateway_snapshot is None
                or self._last_status_monotonic <= self._fault_entered_monotonic
                or time.monotonic() - self._last_status_monotonic > self._reset_status_max_age_sec
                or self._last_gateway_snapshot.busy
            ):
                response.success = False
                response.message = "fresh Gateway status with busy=false is required before reset"
                return response
            self._policy.reset_fault()
            response.success = True
            response.message = "sound orientation fault reset"
            return response

    def _finish_known_action(self, reason: str) -> None:
        self._policy.complete_action(now_sec=self._now_sec(), terminal_known=True)
        self.get_logger().warning(f"sound orientation request dropped: {reason}")

    def _consume_decision(self, decision: PolicyDecision) -> None:
        if self._debug and decision.reason not in {"NO_TRANSITION", "DIRECTION_CACHED", "NON_EXACT_TRIGGER"}:
            self.get_logger().debug(f"sound orientation decision={decision.kind.value} reason={decision.reason}")

    def _set_task_budget(self, binding, *, timeout_sec: float) -> None:
        started = self._now_sec()
        deadline = started + timeout_sec
        binding.task_budget.schema_version = 1
        for target, value in ((binding.task_budget.started_at, started), (binding.task_budget.deadline, deadline)):
            target.sec = int(value)
            target.nanosec = int((value - int(value)) * 1_000_000_000)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = SoundOrientationNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown(timeout_sec=0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
