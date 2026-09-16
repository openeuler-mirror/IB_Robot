"""
TeleopNode - Main ROS 2 node for teleoperation control

This node bridges teleoperation devices to robot controllers,
providing zero-latency control with safety filtering.
"""

import math
import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.task import Future
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Trigger

from .base_teleop import BaseTeleopDevice
from .device_factory import device_factory
from .safety_filter import SafetyFilter
from .teleop_groups import resolve_node_publish_groups


def connect_device_or_raise(device: BaseTeleopDevice) -> None:
    """Connect a teleoperation device and fail node startup when its transport is unavailable."""
    if not device.connect():
        raise RuntimeError("Teleoperation device connection failed")


class TeleopNode(Node):
    """
    Main teleoperation control node.

    This node:
    1. Loads teleoperation device from configuration
    2. Reads joint targets from device at high frequency
    3. Applies safety filtering (joint limits)
    4. Publishes commands to robot controllers

    Publishers:
        - arm_command_topic (Float64MultiArray, default /arm_position_controller/commands)
        - gripper_command_topic (Float64MultiArray, default /gripper_position_controller/commands)
        - /diagnostics (DiagnosticArray)

    Subscribers:
        - estop_topic (Bool) - Configured emergency stop signal

    Parameters:
        - control_frequency (double): Control loop frequency in Hz (default: 50.0)
        - device_config (dict): Teleoperation device configuration
        - joint_limits (dict): Joint limits for safety filter
        - arm_command_topic (string): Arm controller command topic
        - gripper_command_topic (string): Gripper controller command topic
        - estop_topic (string): Emergency-stop Bool topic
    """

    def __init__(self):
        """Initialize teleop node."""
        super().__init__("robot_teleop_node")

        # Declare parameters
        self.declare_parameter("control_frequency", 50.0)
        self.declare_parameter("latency_warn_s", 0.0)
        self.declare_parameter("diagnostics_period_s", 1.0)
        self.declare_parameter("rearm_timeout_s", 5.0)
        self.declare_parameter("device_config", "")
        self.declare_parameter("joint_limits", "")
        self.declare_parameter("publish_groups", "")
        self.declare_parameter("arm_joint_names", ["1", "2", "3", "4", "5"])
        self.declare_parameter("gripper_joint_names", ["6"])

        self.declare_parameter("arm_command_topic", "/arm_position_controller/commands")
        self.declare_parameter("gripper_command_topic", "/gripper_position_controller/commands")
        self.declare_parameter("estop_topic", "/emergency_stop")
        self.declare_parameter("managed_joint_topic", "")

        # Get parameters
        self.control_frequency = self.get_parameter("control_frequency").value
        self.latency_warn_s = self.get_parameter("latency_warn_s").value
        self.diagnostics_period_s = self.get_parameter("diagnostics_period_s").value
        self.rearm_timeout_s = self.get_parameter("rearm_timeout_s").value
        for name, value in (
            ("control_frequency", self.control_frequency),
            ("diagnostics_period_s", self.diagnostics_period_s),
            ("rearm_timeout_s", self.rearm_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.latency_warn_s) or self.latency_warn_s < 0:
            raise ValueError("latency_warn_s must be finite and non-negative")
        if self.latency_warn_s == 0:
            self.latency_warn_s = 1.0 / self.control_frequency
        device_config_str = self.get_parameter("device_config").value
        joint_limits_str = self.get_parameter("joint_limits").value
        publish_groups_value = self.get_parameter("publish_groups").value
        self.arm_joint_names = self.get_parameter("arm_joint_names").value
        self.gripper_joint_names = self.get_parameter("gripper_joint_names").value
        self.arm_command_topic = self.get_parameter("arm_command_topic").value
        self.gripper_command_topic = self.get_parameter("gripper_command_topic").value
        self.estop_topic = self.get_parameter("estop_topic").value

        # Parse JSON parameters if provided as strings
        import json

        device_config = (
            json.loads(device_config_str) if isinstance(device_config_str, str) and device_config_str else {}
        )
        joint_limits = json.loads(joint_limits_str) if isinstance(joint_limits_str, str) and joint_limits_str else {}
        self._managed = bool(device_config.get("managed_teleop", False))
        self._managed_backend = None
        self._managed_config = device_config
        self._rearm_pending = False
        self._rearm_lock = threading.Lock()
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

        # Initialize device
        self.device: BaseTeleopDevice | None = None
        self._device_lock = threading.Lock()
        self._device_disconnected = False

        try:
            self.device = device_factory(device_config, node=self)
            self.get_logger().info(f"Created device: {device_config.get('type', 'unknown')}")

            # Connect to device
            connect_device_or_raise(self.device)
            self.get_logger().info("Device connected successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to create/connect device: {e}")
            raise

        if self._managed:
            self._managed_backend = getattr(self.device, "servo_client", None)
            if self._managed_backend is None:
                from tf2_ros import Buffer

                from .cartesian_backend import make_cartesian_backend

                self._managed_backend = make_cartesian_backend(
                    "runtime",
                    node=self,
                    tf_buffer=Buffer(),
                    base_link=device_config["base_link_name"],
                    tool_frame=device_config["tool_frame"],
                    managed_teleop=True,
                    **device_config["cartesian_backend_config"],
                )
            self._managed_publisher = self.create_publisher(
                JointState, self.get_parameter("managed_joint_topic").value, 1
            )
            # Awaiting a Future must not occupy the source/status/client callback group.
            self.create_service(Trigger, "~/rearm", self._rearm_managed, callback_group=ReentrantCallbackGroup())
            self.create_service(Trigger, "~/home", self._home_managed)

        # Initialize safety filter.
        # Let the device override the static YAML gripper limits with values
        # derived from the follower calibration (so swapping or re-calibrating
        # the follower needs no YAML edit). Devices that don't implement this
        # return an empty dict and the YAML limits stand unchanged.
        #
        # Without this the gripper, now emitting radians, gets clipped back to
        # the legacy [0.0, 1.0] range and only closes halfway again.
        if self.device is not None:
            for joint_name, limits in self.device.get_gripper_limits().items():
                joint_limits[joint_name] = limits
                self.get_logger().info(
                    f"Overriding '{joint_name}' safety limits from device: "
                    f"[{limits['min']:.3f}, {limits['max']:.3f}] rad"
                )
        # Gripper ratio → radian mapping. Leader/phone inputs emit an opening
        # ratio in [0, 1]; the follower joint speaks calibrated radians. The
        # mapping endpoints come from the runtime public description for both
        # the managed and the standalone path, so no deployment copies them.
        self._ratio_input = device_config.get("type") in ("leader_arm", "leader_topic", "phone")
        self._gripper_mapping = None
        self._resolve_gripper_mapping(device_config, joint_limits)
        self.safety_filter = SafetyFilter(joint_limits)

        # Publishers
        self.publish_groups = resolve_node_publish_groups(
            publish_groups_value,
            arm_joint_names=self.arm_joint_names,
            gripper_joint_names=self.gripper_joint_names,
            arm_command_topic=self.arm_command_topic,
            gripper_command_topic=self.gripper_command_topic,
        )
        self.command_publishers = (
            {}
            if self._managed
            else {
                group.name: self.create_publisher(Float64MultiArray, group.topic, 10) for group in self.publish_groups
            }
        )

        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        # Emergency stop
        self.estop_active = False
        self._estop_state_lock = threading.Lock()
        self._estop_stop_pending = False
        self._estop_release_pending = False
        self.estop_sub = self.create_subscription(
            Bool,
            self.estop_topic,
            self.estop_callback,
            10,
        )

        # Control loop timer
        timer_period = 1.0 / self.control_frequency  # seconds
        self.timer = self.create_timer(
            timer_period, self.control_loop_callback, callback_group=MutuallyExclusiveCallbackGroup()
        )

        # Diagnostics
        self.loop_count = 0
        self.last_loop_time = time.monotonic()
        self.avg_loop_time = 0.0
        self.max_loop_time = 0.0

        self.get_logger().info(f"TeleopNode initialized at {self.control_frequency} Hz")

    def control_loop_callback(self):
        """
        Main control loop - called at control_frequency.

        Reads device, applies safety, publishes commands.
        """
        loop_start = time.perf_counter()

        # Skip if emergency stop active. Dispatching the device stop here also
        # retries a request that arrived while another control cycle owned the
        # device lock.
        if self._estop_is_active():
            self._try_dispatch_estop()
            return

        # Read from device
        estop_seen_after_lock = False
        with self._device_lock:
            if self._estop_is_active():
                estop_seen_after_lock = True
            elif self.device is None or not self.device.is_connected:
                if self._managed and self._managed_backend.is_enabled:
                    self.get_logger().warn("Disabling managed teleop: device unavailable or disconnected")
                    self._managed_backend.disable()
                return
            else:
                try:
                    joint_targets = self.device.get_joint_targets()
                except Exception as e:
                    self.get_logger().error(f"Device read failed: {e}")
                    if self._managed:
                        self._managed_backend.disable()
                    return

        if estop_seen_after_lock:
            self._try_dispatch_estop()
            return

        # A multi-threaded executor may deliver E-stop while get_joint_targets
        # is running. Never publish the command computed by that in-flight cycle.
        if self._estop_is_active():
            self._try_dispatch_estop()
            return

        if self._managed:
            backend = self._managed_backend
            if not joint_targets:
                if backend.is_enabled:
                    reason = getattr(self.device, "last_invalid_reason", "")
                    self.get_logger().warn(
                        "Disabling managed teleop: device returned no joint targets"
                        + (f"; last_invalid_reason={reason}" if reason else "")
                    )
                    backend.disable()
                return
            if backend._home_pending:
                backend.keepalive()
                return
            if not backend.is_enabled:
                return
            # Arm radians and gripper ratio have distinct input quantities.
            if self._managed_config["type"] in ("leader_topic", "phone") and self._gripper_mapping is None:
                self.get_logger().warn("Disabling managed teleop: gripper ratio mapping unavailable")
                backend.disable()
                return
            backend.keepalive()

        # Gripper ratio → radians, shared by the managed and standalone paths.
        # Ratio-emitting devices must never publish their raw [0,1] value as
        # radians: without endpoints the target is dropped (fail-closed).
        if self._ratio_input and self.gripper_joint_names:
            name = self.gripper_joint_names[0]
            if name in joint_targets:
                if self._gripper_mapping is None:
                    joint_targets.pop(name, None)
                else:
                    from .public_target import map_gripper_ratio

                    try:
                        joint_targets[name] = map_gripper_ratio(joint_targets[name], *self._gripper_mapping)
                    except ValueError as exc:
                        if self._managed:
                            self.get_logger().warn(
                                f"Disabling managed teleop: invalid gripper ratio for {name!r}: "
                                f"{joint_targets[name]!r}; {exc}"
                            )
                            self._managed_backend.disable()
                            return
                        joint_targets.pop(name, None)

        # Apply safety filter
        safe_targets = self.safety_filter.apply_limits(joint_targets)

        if not safe_targets:
            if self._managed and joint_targets:
                self._managed_backend.disable()
            return

        self._publish_targets(safe_targets)

        # Update diagnostics
        loop_time = time.perf_counter() - loop_start
        self._update_diagnostics(loop_time)

    def _resolve_gripper_mapping(self, device_config: dict, joint_limits: dict) -> None:
        """Resolve gripper ratio endpoints and fill public safety limits.

        Endpoint priority: device config (managed builder) → runtime public
        description (/runtime/get_status) → the device's own follower stroke
        (provider-less legacy deployments). Without endpoints, ratio devices
        drop the gripper target (fail-closed) instead of publishing raw [0,1]
        as radians. The runtime public limits also serve as the safety-filter
        authority for non-ratio devices that were launched without limits.
        """
        if not self.gripper_joint_names:
            return
        gripper_name = self.gripper_joint_names[0]
        closed = device_config.get("gripper_closed")
        opened = device_config.get("gripper_open")
        fetched = None
        if not joint_limits:
            fetched = self._fetch_runtime_gripper_model()
            if fetched is not None:
                for joint_name, bounds in fetched[2].items():
                    joint_limits.setdefault(joint_name, bounds)
        if self._ratio_input:
            if closed is None or opened is None:
                if fetched is None:
                    fetched = self._fetch_runtime_gripper_model()
                if fetched is not None:
                    closed, opened = fetched[0], fetched[1]
            if (closed is None or opened is None) and self.device is not None:
                stroke = self.device.get_gripper_stroke()
                if stroke is not None:
                    closed, opened = stroke
                    joint_limits.setdefault(gripper_name, {"min": stroke[0], "max": stroke[1]})
            if closed is not None and opened is not None and gripper_name in joint_limits:
                self._gripper_mapping = (float(closed), float(opened), joint_limits[gripper_name])
                self.get_logger().info(
                    f"Gripper ratio mapping: [0,1] -> [{float(closed):.3f}, {float(opened):.3f}] rad"
                )
            else:
                self.get_logger().error(
                    "Gripper ratio endpoints unavailable; gripper targets are dropped (fail-closed)"
                )

    def _fetch_runtime_gripper_model(self):
        """One-shot public-description fetch: (closed, open, joint_limits) or None.

        The runtime is already a hard dependency of any hardware teleop session
        (the follower must be up), so startup pulls the calibrated gripper
        conversion endpoints and physical limits from `/runtime/get_status`
        instead of carrying deployment snapshots.
        """
        import json

        from ibrobot_msgs.srv import GetRuntimeStatus

        client = self.create_client(GetRuntimeStatus, "/runtime/get_status")
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/runtime/get_status unavailable; cannot resolve gripper endpoints")
            return None
        future = client.call_async(GetRuntimeStatus.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None:
            self.get_logger().error("/runtime/get_status timed out; cannot resolve gripper endpoints")
            return None
        try:
            description = json.loads(future.result().status.interface_description_json)
            model = description["model"]
            conversion = model["joint_conversions"]["modes"]["range_m100_100"][self.gripper_joint_names[0]]
            closed, opened = float(conversion["min"]), float(conversion["max"])
            if not closed < opened:
                raise ValueError("degenerate gripper conversion range")
            limits = {name: dict(bounds) for name, bounds in model["joint_limits"].items()}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Public model gripper endpoints unusable: {exc}")
            return None
        return closed, opened, limits

    def _publish_targets(self, safe_targets: dict[str, float]) -> None:
        """Publish each complete command group in its configured joint order."""
        if getattr(self, "_managed", False):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [name for name in self.arm_joint_names + self.gripper_joint_names if name in safe_targets]
            msg.position = [safe_targets[name] for name in msg.name]
            self._managed_publisher.publish(msg)
            return
        for group in self.publish_groups:
            if not all(name in safe_targets for name in group.joint_names):
                continue
            msg = Float64MultiArray()
            msg.data = [safe_targets[name] for name in group.joint_names]
            self.command_publishers[group.name].publish(msg)

    async def _rearm_managed(self, _request, response):
        backend = self._managed_backend
        with self._rearm_lock:
            if self._rearm_pending:
                response.message = "rearm already pending"
                return response
            self._rearm_pending = True
        timer = None
        try:
            if self._estop_is_active() or backend.stop_pending:
                response.message = "stop pending; explicitly clear runtime stop before rearm"
                return response
            if backend.is_enabled or backend._requested_enabled or backend._home_pending:
                response.message = "session already active or admission pending"
                return response
            deadline = time.monotonic() + self.rearm_timeout_s
            generation = backend._lifecycle_generation
            if self._managed_config["type"] == "leader_topic":
                # Clear before admission, so the first empty cycle cannot stop a new session.
                with self._device_lock:
                    self.device.prepare_rearm()
                fresh = Future()

                def check_source():
                    if fresh.done():
                        return
                    if (
                        self._estop_is_active()
                        or backend._lifecycle_generation != generation
                        or time.monotonic() >= deadline
                    ):
                        fresh.set_result(False)
                    elif self.device.is_connected and self.device.get_joint_targets():
                        fresh.set_result(True)

                timer = self.create_timer(0.01, check_source, clock=self._steady_clock)
                if not await fresh:
                    response.message = "fresh leader input wait canceled or timed out"
                    return response
                self.destroy_timer(timer)
                timer = None
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._estop_is_active() or backend._lifecycle_generation != generation:
                response.message = "rearm canceled or timed out before admission"
                return response
            result = await backend.enable_with_result(remaining)
            response.success = result.success
            response.message = result.message
            if response.success:
                fresh_source = self._managed_config["type"] != "leader_topic" or (
                    self.device.is_connected and bool(self.device.get_joint_targets())
                )
                if self._estop_is_active() or not backend.is_enabled or not fresh_source:
                    backend.disable()
                    response.success = False
                    response.message = "admission lost or leader input stale; stop requested"
            return response
        finally:
            if timer is not None:
                self.destroy_timer(timer)
            with self._rearm_lock:
                self._rearm_pending = False

    def _home_managed(self, _request, response):
        response.success = (
            not self._estop_is_active() and self._managed_backend.is_enabled and self._managed_backend.home()
        )
        response.message = "HOME requested" if response.success else "HOME requires an admitted fresh session"
        return response

    def estop_callback(self, msg):
        """Latch or explicitly release the emergency-stop gate."""
        if msg.data:
            with self._estop_state_lock:
                newly_active = not self.estop_active
                self.estop_active = True
                self._estop_release_pending = False
                if newly_active:
                    self._estop_stop_pending = True
            self._try_dispatch_estop()
            if newly_active:
                self.get_logger().warn("Emergency stop activated")
            return

        with self._estop_state_lock:
            if not self.estop_active:
                return
            self._estop_release_pending = True
            released = False

        if not released:
            self._try_dispatch_estop()
            released = not self._estop_is_active()
        if released:
            self.get_logger().warn("Emergency stop released; WebPhone requires deadman release and re-press")
        else:
            self.get_logger().warn("Emergency stop release deferred until the device stop is dispatched")

    def _estop_is_active(self) -> bool:
        with self._estop_state_lock:
            return self.estop_active

    def _try_dispatch_estop(self) -> bool:
        """Dispatch a pending device stop without blocking on the control-loop lock."""
        with self._estop_state_lock:
            if not self._estop_stop_pending and not self._estop_release_pending:
                return True

        if not self._device_lock.acquire(blocking=False):
            return False
        try:
            with self._estop_state_lock:
                stop_pending = self._estop_stop_pending
                release_pending = self._estop_release_pending
            if stop_pending:
                try:
                    if self.device is not None:
                        self.device.emergency_stop()
                    if getattr(self, "_managed_backend", None) is not None:
                        self._managed_backend.disable()
                except Exception as exc:  # noqa: BLE001 - retry on the next control cycle
                    self.get_logger().error(f"Emergency stop dispatch failed: {exc}")
                    return False
                with self._estop_state_lock:
                    self._estop_stop_pending = False

            if release_pending:
                try:
                    if self.device is not None:
                        release = getattr(self.device, "emergency_stop_released", None)
                        if callable(release):
                            release()
                except Exception as exc:  # noqa: BLE001 - retry on the next control cycle
                    self.get_logger().error(f"Emergency stop release dispatch failed: {exc}")
                    return False
                with self._estop_state_lock:
                    self.estop_active = False
                    self._estop_release_pending = False
            return True
        finally:
            self._device_lock.release()

    def _update_diagnostics(self, loop_time: float):
        """Update diagnostic statistics."""
        self.loop_count += 1

        # Update timing stats
        if self.loop_count == 1:
            self.avg_loop_time = loop_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_loop_time = alpha * loop_time + (1 - alpha) * self.avg_loop_time

        self.max_loop_time = max(self.max_loop_time, loop_time)

        now = time.monotonic()
        if now - self.last_loop_time >= self.diagnostics_period_s:
            self.last_loop_time = now
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()

            status = DiagnosticStatus()
            status.name = "robot_teleop"
            status.level = DiagnosticStatus.OK if self.avg_loop_time <= self.latency_warn_s else DiagnosticStatus.WARN
            status.message = (
                f"Loop wall time: avg={self.avg_loop_time * 1000:.2f}ms, "
                f"max={self.max_loop_time * 1000:.2f}ms, budget={self.latency_warn_s * 1000:.2f}ms"
            )

            diag_msg.status.append(status)
            self.diag_pub.publish(diag_msg)

            # Log warning if latency high
            if self.avg_loop_time > self.latency_warn_s:
                self.get_logger().warn(
                    f"High loop wall time: {self.avg_loop_time * 1000:.2f}ms > {self.latency_warn_s * 1000:.2f}ms"
                )

    def disconnect_device(self) -> None:
        """Request device shutdown while the ROS context is still alive."""
        with self._device_lock:
            if getattr(self, "_managed_backend", None) is not None and self._managed_backend.is_enabled:
                self._managed_backend.disable()
            if self.device is not None and not self._device_disconnected:
                try:
                    self.device.disconnect()
                    self._device_disconnected = True
                    self.get_logger().info("Device disconnected")
                except Exception as e:
                    self.get_logger().error(f"Error disconnecting device: {e}")

    def device_shutdown_complete(self) -> bool:
        """Return whether any asynchronous device stop has been acknowledged."""
        with self._device_lock:
            if getattr(self, "_managed_backend", None) is not None and self._managed_backend.stop_pending:
                return False
            if self.device is None:
                return True
            return bool(getattr(self.device, "shutdown_complete", True))

    def destroy_node(self):
        """Clean up resources on node shutdown."""
        self.get_logger().info("Shutting down TeleopNode...")
        self.disconnect_device()

        super().destroy_node()


def main(args=None):
    """Entry point for teleop_node."""
    # Keep the ROS context alive while Python handles SIGINT. The finally block
    # can then spin until an asynchronous Cartesian stop is acknowledged.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None

    try:
        node = TeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"TeleopNode failed: {e}")
        raise
    finally:
        if node is not None:
            node.disconnect_device()
            deadline = time.monotonic() + 0.5
            while rclpy.ok() and not node.device_shutdown_complete() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            if not node.device_shutdown_complete():
                node.get_logger().error(
                    "Device stop was not acknowledged before shutdown; verify the Placo node is stopped"
                )
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
