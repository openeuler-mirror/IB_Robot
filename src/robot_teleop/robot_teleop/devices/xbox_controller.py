"""
XboxTeleopDevice - Xbox controller implementation for robot teleoperation.

Refined Logic: Full-State Reporting with Direction-Snap.
Ensures zero jumps and instant response even at limits.
"""

import math
import os
import sys
import threading
import time
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState, Joy
from tf2_ros import Buffer, TransformListener

from ..base_teleop import BaseTeleopDevice
from ..cartesian_backend import make_cartesian_backend


class XboxTeleopDevice(BaseTeleopDevice):
    def __init__(self, config: dict, node=None):
        super().__init__(config, node=node)

        # 1. Device parameters
        params = config.get("control_params", {})
        self.deadzone = params.get("deadzone", 0.1)
        self.joint_velocity_gain = params.get("joint_velocity_gain", 1.5)
        self.cartesian_linear_speed = params.get("cartesian_linear_speed", 1.0)
        self.cartesian_angular_speed = params.get("cartesian_angular_speed", 1.0)
        self.long_press_duration = params.get("long_press_duration", 0.5)
        self.gripper_jog_speed = params.get("gripper_jog_speed", 8.0)

        # 2. Configuration
        self.gripper_joint_names = config.get("gripper_joint_names", ["6"])
        self.arm_joint_names = config.get("arm_joint_names", ["1", "2", "3", "4", "5"])
        self.joint_limits = config.get("joint_limits", {})

        # Injected by robot_config.launch_builders.teleop based on
        # robot.teleoperation.cartesian.{solver,tool_frame} SSOT fields.
        self.cartesian_solver = config.get("cartesian_solver", "placo_servo")
        self.tool_frame = config.get("tool_frame", "gripper")
        self.base_link = config.get("base_link_name", "base")

        # 3. Mappings
        mapping_name = config.get("mapping_config", "xbox_mapping")
        self.axis_mapping = self._load_mapping(mapping_name)
        self.button_mapping = config.get(
            "button_mapping",
            {"arm_enable": 0, "arm_disable": 1, "home_position": 2, "preset_position": 3, "mode_switch": 4},
        )

        # 4. State Management
        self._mode = config.get("default_mode", "joint")
        self._is_control_enabled = False
        self._lb_press_time = None
        self._last_commanded_positions = {}  # The Source of Truth for commands
        self._current_joint_states = {}  # The Source of Truth for physical robot
        self._latest_joy: Joy | None = None
        self._latest_joy_at = 0.0
        self._latest_state_at = 0.0
        self._managed = bool(config.get("managed_teleop", False))
        self._input_stale_s = float(config.get("input_stale_s", 0.2))
        self._previous_buttons = []
        self._trigger_initialized = False
        self._first_state_received = False

        self._state_lock = threading.Lock()
        self._last_debug_time = 0

        # Gripper
        self.GRIPPER_OPEN_POS = float(config.get("gripper_open", 1.0))
        self.GRIPPER_CLOSED_POS = float(config.get("gripper_closed", 0.0))
        self._current_gripper_pos = self.GRIPPER_OPEN_POS

        self.servo_client = None
        self._joy_sub = None
        self._joint_state_sub = None

        self.logger.info(f"XboxTeleopDevice initialized in {self._mode} mode")

    def _print_usage(self, level="full"):
        """
        Print dynamic usage instructions.
        level: 'full' (on enable/switch), 'compact' (on boot), 'status' (on disable).
        """
        mode_str = self._mode.upper()
        status_str = "ENABLED" if self._is_control_enabled else "DISABLED (Press [A] to enable)"

        if level == "status":
            sys.stdout.write(f"\n>>> XBOX STATUS: {status_str} <<<\n")
            sys.stdout.flush()
            return

        if level == "compact":
            banner = f"""
================================================================================
XBOX TELEOP READY
================================================================================
Available Modes: JOINT, CARTESIAN
Current Mode:    {mode_str}
Controls:        [A] to ENABLE, [B] to DISABLE
================================================================================
"""
            sys.stdout.write(banner)
            sys.stdout.flush()
            return

        # Full mode detailed help
        if self._mode == "joint":
            mode_help = """JOINT MODE (Incremental):
  - Left Stick L/R:  Joint 1
  - Left Stick U/D:  Joint 2
  - Right Stick L/R: Joint 3
  - Right Stick U/D: Joint 4
  - D-Pad Left/Right: Joint 5"""
        else:
            mode_help = """CARTESIAN MODE (Cartesian Backend):
  - Left Stick L/R:  Linear X (Left/Right)
  - Left Stick U/D:  Linear Y (Forward/Backward)
  - Right Stick U/D: Linear Z (Up/Down)
  - Right Stick L/R: Angular Y (Pitch / Tilt)
  - D-Pad Left/Right: Angular Z (Wrist Roll)"""

        banner = f"""
================================================================================
XBOX CONTROLLER TELEOP STATUS UPDATE
================================================================================
STATUS:
  - Mode:    {mode_str}
  - Control: {status_str}

CONTROLS:
  - [A] Button:      ENABLE arm control
  - [B] Button:      DISABLE arm control
  - [LB] Long Press: TOGGLE mode (Joint <-> Cartesian)
  - [X] Button:      Go to HOME position
  - [Y] Button:      Go to PRESET position
  - [LT]/[RT]:       GRIPPER Close/Open

{mode_help}
================================================================================
"""
        sys.stdout.write(banner)
        sys.stdout.flush()

    def _load_mapping(self, name: str) -> dict[str, Any]:
        try:
            share_dir = get_package_share_directory("robot_teleop")
            mapping_file = os.path.join(share_dir, "config", f"{name}.yaml")
            if os.path.exists(mapping_file):
                with open(mapping_file) as f:
                    data = yaml.safe_load(f)
                    return data.get("axis_mapping", {})
            return {}
        except Exception:
            return {}

    def connect(self) -> bool:
        if self._node is None:
            return False
        try:
            self._joy_sub = self._node.create_subscription(
                Joy, self._config.get("source_topic", "/joy"), self._process_joy_event, 1
            )
            self._joint_state_sub = self._node.create_subscription(
                JointState, self._config.get("joint_state_topic", "/joint_states"), self._joint_state_callback, 1
            )

            # TF buffer for tool_frame angular-velocity conversion.
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)

            self.servo_client = make_cartesian_backend(
                solver=self.cartesian_solver,
                node=self._node,
                tf_buffer=self._tf_buffer,
                base_link=self.base_link,
                tool_frame=self.tool_frame,
                linear_speed=self.cartesian_linear_speed,
                angular_speed=self.cartesian_angular_speed,
                managed_teleop=self._managed,
                **self._config.get("cartesian_backend_config", {}),
            )
            self.logger.info(
                f"XboxTeleopDevice: cartesian solver={self.cartesian_solver}, tool_frame={self.tool_frame}"
            )
            self._is_connected = True
            self._print_usage(level="compact")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def get_joint_targets(self) -> dict[str, float]:
        """Main interface called by TeleopNode."""
        with self._state_lock:
            if self._managed and (
                time.monotonic() - self._latest_joy_at > self._input_stale_s
                or time.monotonic() - self._latest_state_at > self._input_stale_s
            ):
                self.emergency_stop()
                return {}
            if not self._is_control_enabled or self._latest_joy is None:
                return {}
            if self._managed and self.servo_client._home_pending:
                self.servo_client.keepalive()
                return {}

            # Update gripper
            self._handle_gripper()

            if self._mode == "joint":
                # Compute deltas and update self._last_commanded_positions
                targets = self._compute_joint_deltas()

                # MANDATORY: Return ALL joints to prevent TeleopNode from jumping to 0.0
                full_targets = self._last_commanded_positions.copy()
                full_targets.update(targets)
                full_targets[self.gripper_joint_names[0]] = self._current_gripper_pos
                return full_targets

            elif self._mode == "cartesian":
                self._handle_cartesian_servo()
                # Return ONLY gripper to avoid conflict with Servo's arm control
                # TeleopNode will only publish gripper commands as arm joints are missing
                return {self.gripper_joint_names[0]: self._current_gripper_pos}
        return {}

    def _compute_joint_deltas(self) -> dict[str, float]:
        """Update persistent commands with Reverse-Snap and Full-time sync for inactive joints."""
        joy = self._latest_joy
        mapping = self.axis_mapping.get("joint_mode", {})
        dt = 0.02

        active_joints: set[str] = set()
        for j_name, axis_idx in mapping.items():
            if axis_idx < len(joy.axes) and abs(joy.axes[axis_idx]) > self.deadzone:
                active_joints.add(j_name)

        if not active_joints:
            return {}

        # 2. ACTIVE INTEGRATOR
        targets = {}
        for j_name in active_joints:
            axis_idx = mapping[j_name]
            axis_val = joy.axes[axis_idx]
            delta = axis_val * self.joint_velocity_gain * dt

            actual = self._current_joint_states.get(j_name, 0.0)
            prev_cmd = self._last_commanded_positions.get(j_name, actual)

            # REVERSE-SNAP
            lead = prev_cmd - actual
            if (delta > 0.001 and lead < -0.01) or (delta < -0.001 and lead > 0.01):
                prev_cmd = actual

            new_pos = prev_cmd + delta

            # Lead Clamp (0.5 rad)
            new_pos = max(actual - 0.5, min(actual + 0.5, new_pos))

            # Physical Limits
            if j_name in self.joint_limits:
                lim = self.joint_limits[j_name]
                new_pos = max(lim["min"], min(lim["max"], new_pos))

            self._last_commanded_positions[j_name] = new_pos
            targets[j_name] = new_pos

        return targets

    def _handle_cartesian_servo(self):
        if not self.servo_client:
            return
        joy = self._latest_joy
        mapping = self.axis_mapping.get("cartesian_mode", {})

        def get_val(key, default_idx):
            idx = mapping.get(key, default_idx)
            if idx < len(joy.axes):
                val = joy.axes[idx]
                return val if abs(val) > self.deadzone else 0.0
            return 0.0

        # Linux /joy reports left-stick Y as negative when pushed up. Normalize
        # it here so Xbox Cartesian controls match the SO-101 operator
        # intuition: left stick X = left/right, left stick Y = forward/backward.
        # This device mapping convention is shared by all Cartesian backends.
        linear = (get_val("linear_x", 1), -get_val("linear_y", 0), get_val("linear_z", 4))
        angular = (0.0, get_val("angular_y", 6), get_val("angular_z", 3))
        self.servo_client.servo(linear=linear, angular=angular)

    def _handle_gripper(self):
        joy = self._latest_joy
        mapping = self.axis_mapping.get("gripper", {})
        open_idx, close_idx = mapping.get("open", 5), mapping.get("close", 2)
        if open_idx >= len(joy.axes) or close_idx >= len(joy.axes):
            return

        rt_raw, lt_raw = joy.axes[open_idx], joy.axes[close_idx]
        if not self._trigger_initialized:
            if rt_raw != 0.0 or lt_raw != 0.0:
                self._trigger_initialized = True
            else:
                return

        rt_val, lt_val = (rt_raw - 1.0) / -2.0, (lt_raw - 1.0) / -2.0
        if abs(rt_val) < 0.1 and abs(lt_val) < 0.1:
            self._current_gripper_pos = self._current_joint_states.get(
                self.gripper_joint_names[0], self._current_gripper_pos
            )
            return

        target_pos = self._current_gripper_pos
        if rt_val > 0.1:
            target_pos += self.gripper_jog_speed * rt_val * 0.02
        elif lt_val > 0.1:
            target_pos -= self.gripper_jog_speed * lt_val * 0.02

        target_pos = max(self.GRIPPER_CLOSED_POS, min(self.GRIPPER_OPEN_POS, target_pos))
        self._current_gripper_pos = target_pos

    def _process_joy_event(self, msg: Joy):
        with self._state_lock:
            if self._managed:
                now = self._node.get_clock().now().nanoseconds * 1e-9
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if not 0 <= now - stamp <= self._input_stale_s or not all(math.isfinite(v) for v in msg.axes):
                    return
            self._latest_joy = msg
            self._latest_joy_at = time.monotonic()
            if not self._previous_buttons:
                self._previous_buttons = list(msg.buttons)
                return
            self._previous_buttons += [0] * max(0, len(msg.buttons) - len(self._previous_buttons))
            a_idx, b_idx = self.button_mapping.get("arm_enable", 0), self.button_mapping.get("arm_disable", 1)

            if a_idx < len(msg.buttons) and msg.buttons[a_idx] and not self._previous_buttons[a_idx]:
                self._is_control_enabled = not self._managed or (
                    self._first_state_received and time.monotonic() - self._latest_state_at <= self._input_stale_s
                )
                self._sync_targets_to_actual("Enabled")
                if self._is_control_enabled and (self._managed or self._mode == "cartesian") and self.servo_client:
                    self.servo_client.enable()
                self._print_usage(level="full")

            if b_idx < len(msg.buttons) and msg.buttons[b_idx] and not self._previous_buttons[b_idx]:
                self._is_control_enabled = False
                if self.servo_client and self.servo_client.is_enabled:
                    self.servo_client.disable()
                self._print_usage(level="status")

            x_idx, y_idx = self.button_mapping.get("home_position", 2), self.button_mapping.get("preset_position", 3)
            if x_idx < len(msg.buttons) and msg.buttons[x_idx] and not self._previous_buttons[x_idx]:
                self._go_to_fixed_position("home")
            if y_idx < len(msg.buttons) and msg.buttons[y_idx] and not self._previous_buttons[y_idx]:
                self._go_to_fixed_position("preset")
            self._handle_mode_switch(msg)
            self._previous_buttons = list(msg.buttons)

    def _sync_targets_to_actual(self, reason: str = "Unknown"):
        self._last_commanded_positions = self._current_joint_states.copy()
        if self.gripper_joint_names[0] in self._current_joint_states:
            self._current_gripper_pos = self._current_joint_states[self.gripper_joint_names[0]]
        sys.stdout.write(f"\n>>> SYNC: Targets aligned to Robot State ({reason}) <<<\n")
        sys.stdout.flush()

    def _go_to_fixed_position(self, target_key: str):
        if self._managed:
            if target_key == "home" and self._is_control_enabled and self.servo_client.is_enabled:
                self.servo_client.home()
                self._last_commanded_positions.clear()
            return
        if self._mode == "cartesian":
            self._toggle_mode()
        target_values = {
            "home": {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0},
            "preset": {"1": -0.0076, "2": 0.0537, "3": -3.1186, "4": 0.0, "5": -0.0015},
        }
        for joint_name, pos in target_values.get(target_key, {}).items():
            self._last_commanded_positions[joint_name] = pos
        sys.stdout.write(f"\n>>> Moving to {target_key.upper()} position <<<\n")
        sys.stdout.flush()

    def _joint_state_callback(self, msg: JointState):
        with self._state_lock:
            if self._managed and not all(name in msg.name for name in self.arm_joint_names + self.gripper_joint_names):
                return
            if self._managed:
                now = self._node.get_clock().now().nanoseconds * 1e-9
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if (
                    len(msg.name) != len(msg.position)
                    or not all(math.isfinite(v) for v in msg.position)
                    or not 0 <= now - stamp <= self._input_stale_s
                ):
                    return
            self._latest_state_at = time.monotonic()
            for name, pos in zip(msg.name, msg.position, strict=False):
                self._current_joint_states[name] = pos
            if not self._first_state_received:
                self._sync_targets_to_actual("Boot")
                self._first_state_received = True

    def _handle_mode_switch(self, msg: Joy):
        lb_idx = self.button_mapping.get("mode_switch", 4)
        if lb_idx >= len(msg.buttons):
            return
        lb_state, prev_lb_state = msg.buttons[lb_idx], self._previous_buttons[lb_idx] if self._previous_buttons else 0
        if lb_state == 1 and prev_lb_state == 0:
            self._lb_press_time = time.time()
        if lb_state == 0 and prev_lb_state == 1:
            if self._lb_press_time and (time.time() - self._lb_press_time) >= self.long_press_duration:
                self._toggle_mode()
            self._lb_press_time = None

    def _toggle_mode(self):
        old_mode = self._mode
        self._mode = "cartesian" if old_mode == "joint" else "joint"
        if self._managed:
            self.emergency_stop()
            return
        if self._mode == "joint":
            if self.servo_client and self.servo_client.is_enabled:
                self.servo_client.disable()
            self._sync_targets_to_actual("Mode Switch")
        else:
            if self._is_control_enabled and self.servo_client:
                self.servo_client.enable()
        self._print_usage(level="full")

    def disconnect(self):
        if self.servo_client and self.servo_client.is_enabled:
            self.servo_client.disable()
        self._is_connected = False

    def emergency_stop(self):
        self._is_control_enabled = False
        self._last_commanded_positions.clear()
        if self.servo_client and (self.servo_client.is_enabled or self.servo_client._requested_enabled):
            self.servo_client.disable()
