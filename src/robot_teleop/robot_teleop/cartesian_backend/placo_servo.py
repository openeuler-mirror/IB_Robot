"""SO101 Placo Servo client backend.

A thin device-side adapter for the ``placo_servo`` solver mode. From the
caller's perspective this behaves *exactly* like the sibling backends::

    backend.servo(linear=(vx, vy, vz), angular=(wx, wy, wz))

so the upstream device code in :mod:`robot_teleop.devices.xbox_controller`
and :mod:`robot_teleop.phone.phone_device` stays solver-agnostic.

Internally the backend publishes the twist to two **private** topics consumed
by ``so101_placo_servo_node``:

* ``linear_cmd_base``  — :class:`geometry_msgs.msg.Vector3Stamped`,
  ``frame_id = base_link``, carries ``linear`` (already base-frame per the
  device convention).
* ``angular_cmd_base`` — :class:`geometry_msgs.msg.Vector3Stamped`,
  ``frame_id = base_link``, carries ``angular`` **converted tool→base** via
  :class:`ToolAngularAdapter`.

``placo_servo`` controls a true Cartesian orientation: the QP orientation task
expects a base-frame angular velocity, so the raw tool-frame stick angular must
be rotated into the base frame before publishing.
"""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.task import Future
from std_msgs.msg import Empty
from std_srvs.srv import Trigger

from ibrobot_msgs.action import ArmReturnHome

from .base import CartesianBackend
from .frame_adapter import ToolAngularAdapter

Vec3 = tuple[float, float, float]

_DEFAULT_LINEAR_TOPIC = "/so101_placo_servo_node/linear_cmd_base"
_DEFAULT_ANGULAR_TOPIC = "/so101_placo_servo_node/angular_cmd_base"
_DEFAULT_POSE_TOPIC = "/so101_placo_servo_node/pose_cmd_base"
_DEFAULT_START_SRV = "/so101_placo_servo_node/start"
_DEFAULT_STOP_SRV = "/so101_placo_servo_node/stop"
_DEFAULT_HOME_ACTION = "/so101_placo_servo_node/return_home"
_DEFAULT_COMMAND_LEASE_TOPIC = "/so101_placo_servo_node/command_lease"


# Physical max speeds that map to a user scale value of 1.0.
_MAX_LINEAR_SPEED_MPS: float = 1.0  # 1.0 m/s at scale=1.0
_MAX_ANGULAR_SPEED_RPS: float = 6.0  # 6.0 rad/s at scale=1.0


class PlacoServoBackend(CartesianBackend):
    """Publish base-frame linear/angular commands to ``so101_placo_servo_node``."""

    def __init__(
        self,
        node,
        tf_buffer,
        base_link: str,
        tool_frame: str,
        linear_speed: float = 0.3,
        angular_speed: float = 0.67,
        linear_topic: str = _DEFAULT_LINEAR_TOPIC,
        angular_topic: str = _DEFAULT_ANGULAR_TOPIC,
        pose_topic: str = _DEFAULT_POSE_TOPIC,
        start_srv: str = _DEFAULT_START_SRV,
        stop_srv: str = _DEFAULT_STOP_SRV,
        home_action: str = _DEFAULT_HOME_ACTION,
        command_lease_topic: str = _DEFAULT_COMMAND_LEASE_TOPIC,
        input_mode: str = "velocity",
        stale_threshold_s: float = 0.2,
        managed_teleop: bool = False,
        runtime_status_topic: str = "/runtime/status",
        **_unused,  # tolerate extra kwargs for signature parity with siblings
    ) -> None:
        self._node = node
        self._managed = managed_teleop
        self._base = base_link
        self._input_mode = str(input_mode).lower()
        if self._input_mode not in ("velocity", "pose"):
            raise ValueError(f"Placo input_mode must be 'velocity' or 'pose', got {input_mode!r}")
        # tool→base converter for the angular channel (true Cartesian posture).
        self._adapter = ToolAngularAdapter(
            node=node,
            tf_buffer=tf_buffer,
            base_link=base_link,
            tool_frame=tool_frame,
            stale_threshold_s=stale_threshold_s,
        )
        # linear_speed / angular_speed are 0.0~1.0 fractions of the physical
        # maximums; convert to actual m/s, rad/s before publishing.
        self._linear_speed = float(linear_speed) * _MAX_LINEAR_SPEED_MPS
        self._angular_speed = float(angular_speed) * _MAX_ANGULAR_SPEED_RPS

        self._linear_pub = node.create_publisher(Vector3Stamped, linear_topic, 10)
        self._angular_pub = node.create_publisher(Vector3Stamped, angular_topic, 10)
        self._pose_pub = node.create_publisher(PoseStamped, pose_topic, 10)
        self._start_cli = node.create_client(Trigger, start_srv)
        self._stop_cli = node.create_client(Trigger, stop_srv)
        self._home_client = ActionClient(node, ArmReturnHome, home_action)
        self._lease_pub = node.create_publisher(Empty, command_lease_topic, 10)
        self._requested_enabled = False
        self._active_enabled = False
        self._start_request_inflight = False
        self._start_retry_timer = None
        self._enable_result = None
        self._enable_timeout_timer = None
        self._stop_pending = False
        self._stop_request_inflight = False
        self._stop_retry_timer = None
        self._home_pending = False
        self._home_result: bool | None = None
        self._home_goal_handle = None
        self._home_goal_generation: int | None = None
        self._lifecycle_generation = 0
        self._runtime_stop_latched = False
        if self._managed:
            from ibrobot_msgs.msg import RuntimeStatus

            node.create_subscription(RuntimeStatus, runtime_status_topic, self._on_runtime_status, 1)

    def _on_runtime_status(self, status):
        invalid = status.stop_latched or status.active_mode != "stream" or status.lifecycle != "ACTIVE"
        stop_edge = status.stop_latched and not self._runtime_stop_latched
        self._runtime_stop_latched = status.stop_latched
        if stop_edge:
            if self._requested_enabled or self._active_enabled or self._home_pending:
                self.disable()
            else:
                self._lifecycle_generation += 1
            return
        if invalid and (self._active_enabled or self._home_pending):
            self._lifecycle_generation += 1
            self._requested_enabled = False
            self._active_enabled = False
            self._home_pending = False
            self._home_result = False
            self._cancel_start_retry()
            self._finish_enable(False, "runtime admission lost")
            if self._home_goal_handle is not None:
                self._home_goal_handle.cancel_goal_async()
                self._home_goal_handle = None

    # ------------------------------------------------------------------ enable
    def enable_with_result(self, timeout_s: float) -> Future:
        """Return final admission, retaining enable()'s queued-bool device API."""
        result = Future()
        if self._requested_enabled or self._start_request_inflight or self._stop_pending or self._home_pending:
            result.set_result(Trigger.Response(success=False, message="admission busy or stop pending"))
            return result
        self._enable_result = result
        self._enable_timeout_timer = self._node.create_timer(
            timeout_s, self._enable_timed_out, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )
        if not self.enable():
            self._finish_enable(False, "start service unavailable or admission refused")
        return result

    def _finish_enable(self, success: bool, message: str) -> None:
        result = self._enable_result
        if self._enable_timeout_timer is not None:
            self._node.destroy_timer(self._enable_timeout_timer)
            self._enable_timeout_timer = None
        self._enable_result = None
        if result is not None and not result.done():
            result.set_result(Trigger.Response(success=success, message=message))

    def _enable_timed_out(self) -> None:
        self._finish_enable(False, "runtime admission timed out; stop requested")
        self.disable()

    def enable(self) -> bool:
        if self._stop_pending or getattr(self, "_runtime_stop_latched", False):
            return False
        # A start requested before the previous release must not be reused as a
        # new grip. Wait for its response; a late success will issue another
        # stop, after which the normal enable path can safely re-latch.
        if self._start_request_inflight and not self._requested_enabled:
            return False
        if self._requested_enabled:
            return True
        self._requested_enabled = True
        if not self._start_cli.service_is_ready():
            self._node.get_logger().warn(
                "so101_placo_servo_node start service not ready; will retry every 0.5 s",
            )
            self._schedule_start_retry()
            return self._requested_enabled
        self._send_start_request()
        return self._requested_enabled

    def _schedule_start_retry(self) -> None:
        if self._managed:
            self._requested_enabled = False
            return
        if self._start_retry_timer is not None:
            return
        self._start_retry_timer = self._node.create_timer(0.5, self._retry_start)

    def _retry_start(self) -> None:
        if not self._requested_enabled:
            self._cancel_start_retry()
            return
        if self._start_cli.service_is_ready():
            self._node.get_logger().info("so101_placo_servo_node start service now ready; requesting start")
            self._send_start_request()

    def _send_start_request(self) -> None:
        if self._start_request_inflight or not self._requested_enabled or self._stop_pending:
            return
        self._cancel_start_retry()
        try:
            future = self._start_cli.call_async(Trigger.Request())
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f"so101_placo_servo_node start request failed: {exc}")
            self._finish_enable(False, f"start request failed: {exc}")
            if self._requested_enabled:
                self._schedule_start_retry()
            return
        self._start_request_inflight = True
        generation = self._lifecycle_generation
        future.add_done_callback(lambda completed: self._on_start_response(completed, generation))

    def _on_start_response(self, future: Future, generation: int | None = None) -> None:
        self._start_request_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            if generation is not None and generation != self._lifecycle_generation:
                return
            self._active_enabled = False
            self._finish_enable(False, f"start request failed: {exc}")
            self._node.get_logger().error(f"so101_placo_servo_node start request failed: {exc}")
            if self._requested_enabled:
                self._schedule_start_retry()
            return

        if generation is not None and generation != self._lifecycle_generation:
            if response.success:
                self.disable()
            self._node.get_logger().info("Ignoring stale Placo start response after lifecycle transition")
            return

        if response.success:
            if not self._requested_enabled:
                self._active_enabled = False
                self.disable()
                self._node.get_logger().info("Placo start completed after release; stop requested")
                return
            self._active_enabled = True
            self._cancel_start_retry()
            self._node.get_logger().info(response.message or "so101_placo_servo_node enabled")
            self._finish_enable(True, response.message or "runtime admitted")
            return

        self._active_enabled = False
        self._finish_enable(False, response.message or "runtime rejected admission")
        self._node.get_logger().error(response.message or "so101_placo_servo_node rejected start request")
        if self._requested_enabled:
            self._schedule_start_retry()

    def _cancel_start_retry(self) -> None:
        if self._start_retry_timer is None:
            return
        self._start_retry_timer.cancel()
        self._start_retry_timer.destroy()
        self._start_retry_timer = None

    def disable(self) -> bool:
        self._lifecycle_generation += 1
        self._finish_enable(False, "admission canceled by stop")
        self._requested_enabled = False
        self._cancel_start_retry()
        self._active_enabled = False
        self._home_pending = False
        self._home_result = False
        if self._home_goal_handle is not None:
            self._home_goal_handle.cancel_goal_async()
            self._home_goal_handle = None
            self._home_goal_generation = None
        self._stop_pending = True
        self._try_stop()
        return True

    def _try_stop(self) -> None:
        if not self._stop_pending or self._stop_request_inflight:
            return
        if not self._stop_cli.service_is_ready():
            self._schedule_stop_retry()
            return
        try:
            future = self._stop_cli.call_async(Trigger.Request())
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f"so101_placo_servo_node stop request failed: {exc}")
            self._schedule_stop_retry()
            return
        self._stop_request_inflight = True
        generation = self._lifecycle_generation
        future.add_done_callback(lambda completed: self._on_stop_response(completed, generation))

    def _on_stop_response(self, future: Future, generation: int | None = None) -> None:
        self._stop_request_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f"so101_placo_servo_node stop request failed: {exc}")
            self._schedule_stop_retry()
            return
        if response.success:
            # An older HOLD cannot acknowledge a stop requested after a late start.
            if generation is not None and generation != self._lifecycle_generation:
                self._try_stop()
                return
            self._stop_pending = False
            self._cancel_stop_retry()
            self._node.get_logger().info(response.message or "so101_placo_servo_node stopped")
            return
        self._node.get_logger().error(response.message or "so101_placo_servo_node rejected stop request")
        self._schedule_stop_retry()

    def _schedule_stop_retry(self) -> None:
        if not self._stop_pending or self._stop_retry_timer is not None:
            return
        self._stop_retry_timer = self._node.create_timer(0.1, self._retry_stop)

    def _retry_stop(self) -> None:
        if not self._stop_pending:
            self._cancel_stop_retry()
            return
        self._try_stop()

    def _cancel_stop_retry(self) -> None:
        if self._stop_retry_timer is None:
            return
        self._stop_retry_timer.cancel()
        self._stop_retry_timer.destroy()
        self._stop_retry_timer = None

    # ------------------------------------------------------------------ servo
    def servo(self, linear: Vec3, angular: Vec3) -> None:
        if not self._active_enabled:
            return
        if self._input_mode != "velocity":
            raise RuntimeError("Placo backend is configured for relative-pose input")
        # Convert tool-frame angular → base frame (true Cartesian posture).
        v_base, w_base = self._adapter.convert(linear, angular)
        stamp = self._node.get_clock().now().to_msg()

        lin_msg = Vector3Stamped()
        lin_msg.header.stamp = stamp
        lin_msg.header.frame_id = self._base
        lin_msg.vector.x = float(v_base[0]) * self._linear_speed
        lin_msg.vector.y = float(v_base[1]) * self._linear_speed
        lin_msg.vector.z = float(v_base[2]) * self._linear_speed
        self._linear_pub.publish(lin_msg)

        ang_msg = Vector3Stamped()
        ang_msg.header.stamp = stamp
        ang_msg.header.frame_id = self._base  # base-frame angular (converted)
        ang_msg.vector.x = float(w_base[0]) * self._angular_speed
        ang_msg.vector.y = float(w_base[1]) * self._angular_speed
        ang_msg.vector.z = float(w_base[2]) * self._angular_speed
        self._angular_pub.publish(ang_msg)

    def servo_pose(self, position: Vec3, orientation: tuple[float, float, float, float]) -> None:
        if not self._active_enabled:
            return
        if self._input_mode != "pose":
            raise RuntimeError("Placo backend is configured for velocity input")
        msg = PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._base
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self._pose_pub.publish(msg)

    def home(self) -> bool:
        if self._managed and not self._active_enabled:
            return False
        if self._stop_pending or self._home_pending or not self._home_client.server_is_ready():
            return False
        self._lifecycle_generation += 1
        self._requested_enabled = False
        self._cancel_start_retry()
        self._active_enabled = False
        self._home_pending = True
        self._home_result = None
        generation = self._lifecycle_generation
        goal = ArmReturnHome.Goal()
        goal.target_name = "home"
        try:
            future = self._home_client.send_goal_async(goal)
        except Exception as exc:  # noqa: BLE001
            self._home_pending = False
            self._node.get_logger().error(f"ArmReturnHome goal dispatch failed: {exc}")
            return False
        future.add_done_callback(lambda completed: self._on_home_goal_response(completed, generation))
        return True

    def _on_home_goal_response(self, future: Future, generation: int | None = None) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            if generation is not None and generation != self._lifecycle_generation:
                return
            self._node.get_logger().error(f"ArmReturnHome goal request failed: {exc}")
            self._home_pending = False
            self._home_result = False
            return
        if generation is not None and generation != self._lifecycle_generation:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            self._node.get_logger().info("Canceling stale ArmReturnHome goal after lifecycle transition")
            return
        if not goal_handle.accepted:
            self._home_pending = False
            self._home_result = False
            self._node.get_logger().error("ArmReturnHome goal was rejected")
            return
        self._home_goal_handle = goal_handle
        self._home_goal_generation = generation
        goal_handle.get_result_async().add_done_callback(lambda completed: self._on_home_result(completed, generation))

    def _on_home_result(self, future: Future, generation: int | None = None) -> None:
        if generation is not None and generation != self._lifecycle_generation:
            return
        self._home_goal_handle = None
        self._home_goal_generation = None
        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f"ArmReturnHome result failed: {exc}")
            self._home_pending = False
            self._home_result = False
            return
        self._home_pending = False
        self._home_result = bool(result.success)
        log = self._node.get_logger().info if result.success else self._node.get_logger().error
        log(result.message or result.error_code or "ArmReturnHome completed")

    def consume_home_result(self) -> bool | None:
        result = self._home_result
        self._home_result = None
        return result

    def keepalive(self) -> None:
        self._lease_pub.publish(Empty())

    @property
    def is_enabled(self) -> bool:
        return self._active_enabled

    @property
    def stop_pending(self) -> bool:
        """Whether a requested stop still awaits a successful service response."""
        return self._stop_pending

    @property
    def max_linear_speed(self) -> float:
        return self._linear_speed

    @property
    def max_angular_speed(self) -> float:
        return self._angular_speed
