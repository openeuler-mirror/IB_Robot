"""
TeleopNode - Main ROS 2 node for teleoperation control

This node bridges teleoperation devices to robot controllers,
providing zero-latency control with safety filtering.
"""

import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Float64MultiArray

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
        self.declare_parameter("device_config", "")
        self.declare_parameter("joint_limits", "")
        self.declare_parameter("publish_groups", "")
        self.declare_parameter("arm_joint_names", ["1", "2", "3", "4", "5"])
        self.declare_parameter("gripper_joint_names", ["6"])

        self.declare_parameter("arm_command_topic", "/arm_position_controller/commands")
        self.declare_parameter("gripper_command_topic", "/gripper_position_controller/commands")
        self.declare_parameter("estop_topic", "/emergency_stop")

        # Get parameters
        self.control_frequency = self.get_parameter("control_frequency").value
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
        self.safety_filter = SafetyFilter(joint_limits)

        # Publishers
        self.publish_groups = resolve_node_publish_groups(
            publish_groups_value,
            arm_joint_names=self.arm_joint_names,
            gripper_joint_names=self.gripper_joint_names,
            arm_command_topic=self.arm_command_topic,
            gripper_command_topic=self.gripper_command_topic,
        )
        self.command_publishers = {
            group.name: self.create_publisher(Float64MultiArray, group.topic, 10) for group in self.publish_groups
        }

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
        self.last_loop_time = time.time()
        self.avg_loop_time = 0.0
        self.max_loop_time = 0.0

        self.get_logger().info(f"TeleopNode initialized at {self.control_frequency} Hz")

    def control_loop_callback(self):
        """
        Main control loop - called at control_frequency.

        Reads device, applies safety, publishes commands.
        """
        loop_start = time.time()

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
                return
            else:
                try:
                    joint_targets = self.device.get_joint_targets()
                except Exception as e:
                    self.get_logger().error(f"Device read failed: {e}")
                    return

        if estop_seen_after_lock:
            self._try_dispatch_estop()
            return

        # A multi-threaded executor may deliver E-stop while get_joint_targets
        # is running. Never publish the command computed by that in-flight cycle.
        if self._estop_is_active():
            self._try_dispatch_estop()
            return

        # Apply safety filter
        safe_targets = self.safety_filter.apply_limits(joint_targets)

        if not safe_targets:
            return

        self._publish_targets(safe_targets)

        # Update diagnostics
        loop_time = time.time() - loop_start
        self._update_diagnostics(loop_time)

    def _publish_targets(self, safe_targets: dict[str, float]) -> None:
        """Publish each complete command group in its configured joint order."""
        for group in self.publish_groups:
            if not all(name in safe_targets for name in group.joint_names):
                continue
            msg = Float64MultiArray()
            msg.data = [safe_targets[name] for name in group.joint_names]
            self.command_publishers[group.name].publish(msg)

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

        # Publish diagnostics every 50 cycles
        if self.loop_count % 50 == 0:
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()

            status = DiagnosticStatus()
            status.name = "robot_teleop"
            status.level = DiagnosticStatus.OK if self.avg_loop_time < 0.005 else DiagnosticStatus.WARN
            status.message = f"Loop time: avg={self.avg_loop_time * 1000:.2f}ms, max={self.max_loop_time * 1000:.2f}ms"

            diag_msg.status.append(status)
            self.diag_pub.publish(diag_msg)

            # Log warning if latency high
            if self.avg_loop_time > 0.005:  # 5ms threshold
                self.get_logger().warn(f"High latency detected: {self.avg_loop_time * 1000:.2f}ms > 5ms")

    def disconnect_device(self) -> None:
        """Request device shutdown while the ROS context is still alive."""
        with self._device_lock:
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
