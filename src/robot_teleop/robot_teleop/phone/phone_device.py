"""
Phone teleoperation device implementation.

Uses browser-based WebXR or optical-flow input for relative-pose Cartesian
phone teleoperation.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener

from ..base_teleop import BaseTeleopDevice
from ..cartesian_backend import make_cartesian_backend
from ..vr_rotation import compute_base_rotation_delta, remap_base_rotation
from .config_phone import PhoneConfig
from .web_phone import WebPhone

logger = logging.getLogger(__name__)

_MAX_WEBPHONE_STOP_REQUEST_LATENCY_S = 0.22

# Proper rotation from the calibrated Web control frame to the robot base frame.
# Translation and rotation must share this basis so one browser pose remains one
# rigid-body transform all the way to the Placo base-frame pose contract.
_WEB_WORLD_TO_BACKEND = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def compute_webphone_pose_delta(
    current_position: np.ndarray,
    current_rotation: Rotation,
    clutch_position: np.ndarray,
    clutch_rotation: Rotation,
    *,
    position_scale: float,
    angular_scale: float,
    user_scale: float,
) -> tuple[np.ndarray, Rotation]:
    """Map a browser viewer-pose delta into the Placo base-frame contract."""
    relative_position = (
        (_WEB_WORLD_TO_BACKEND @ (np.asarray(current_position, dtype=float) - np.asarray(clutch_position, dtype=float)))
        * position_scale
        * user_scale
    )
    control_delta = compute_base_rotation_delta(current_rotation, clutch_rotation)
    # WebXR or the optical virtual pose supplies an active viewer-to-world
    # orientation. Preserve that active relative-rotation direction when
    # re-expressing it in the robot base frame; Placo then left-multiplies the
    # result onto the clutch-baseline tool attitude.
    base_delta = remap_base_rotation(control_delta, _WEB_WORLD_TO_BACKEND)
    scaled_rotvec = base_delta.as_rotvec() * angular_scale * user_scale
    return relative_position, Rotation.from_rotvec(scaled_rotvec)


@dataclass
class _CartesianCommand:
    """Internal per-cycle phone command for the Cartesian backend and gripper."""

    gripper_pos: float = 0.0
    go_home: bool = False
    enabled: bool = False
    pose_position: np.ndarray | None = None
    pose_rotation: Rotation | None = None
    user_scale: float = 1.0


class PhoneDevice(BaseTeleopDevice):
    """
    Phone-based teleoperation device.

    Uses built-in WebPhone with WebXR AR or optical-flow input.

    Parses sensor data from the phone, drives the selected Cartesian backend for arm
    control via relative-pose commands, and returns only the gripper
    target to TeleopNode for direct publishing.

    Go-Home is delegated to the Placo backend, which reports measured joint
    completion while PhoneDevice keeps a backend-independent deadman gate.
    """

    def __init__(self, config: dict, node=None):
        super().__init__(config, node=node)
        if node is not None and hasattr(node, "get_logger"):
            # Route device lifecycle and WebPhone access URLs through ROS logs.
            self.logger = node.get_logger()

        phone_config_data = config.get("phone_config", {})
        if isinstance(phone_config_data, dict):
            self.phone_config = PhoneConfig.from_dict(phone_config_data)
        else:
            self.phone_config = PhoneConfig()

        self._phone_impl: WebPhone | None = None
        self._last_gripper_pos: float = 0.0
        self._pose_clutch_pos: np.ndarray | None = None
        self._pose_clutch_rot: Rotation | None = None
        self._pose_sent_pos = np.zeros(3)
        self._pose_sent_rot = Rotation.identity()
        self._pose_filtered_rotvec = np.zeros(3)
        self._go_home_prev = False

        # _state_lock protects only shared state; ROS calls happen outside the lock
        self._state_lock = threading.Lock()
        self.servo_client = None
        self._joint_state_sub = None
        self._current_joint_states: dict[str, float] = {}
        self._first_state_received = False
        self._going_home = False
        self._servo_enabled = False
        self._deadman_release_required = False
        self._last_command_failure_reason = "phone command unavailable"

        # Injected by teleop.py launch builder
        self.arm_joint_names = config.get("arm_joint_names", ["1", "2", "3", "4", "5"])
        self.gripper_joint_names = config.get("gripper_joint_names", ["6"])
        self._cartesian_solver = str(config.get("cartesian_solver", "placo_servo"))
        if self._cartesian_solver not in ("placo_servo", "runtime"):
            raise ValueError("PhoneDevice requires cartesian_solver=placo_servo")
        raw_control_frequency = config.get("control_frequency", 50.0)
        if isinstance(raw_control_frequency, bool):
            raise ValueError("PhoneDevice control_frequency must be finite and positive")
        control_frequency = float(raw_control_frequency)
        if not np.isfinite(control_frequency) or control_frequency <= 0.0:
            raise ValueError("PhoneDevice control_frequency must be finite and positive")
        self._control_dt = 1.0 / control_frequency
        if self.phone_config.web.command_stale_s + self._control_dt > _MAX_WEBPHONE_STOP_REQUEST_LATENCY_S + 1e-9:
            raise ValueError(
                "WebPhone command_stale_s plus one control period must not exceed "
                f"{_MAX_WEBPHONE_STOP_REQUEST_LATENCY_S:.2f}s before a stop request is issued"
            )

    def connect(self) -> bool:
        """Connect to phone hardware and initialise Cartesian backend."""
        if self._node is None:
            self.logger.error("PhoneDevice requires a ROS node reference (node=None)")
            return False

        try:
            self._phone_impl = WebPhone(self.phone_config)

            if not self._phone_impl.connect():
                return False

            base_link = self._config.get("base_link_name", "base")
            solver = self._cartesian_solver
            tool_frame = self._config.get("tool_frame", "gripper")
            control_params = self._config.get("control_params", {}) or {}
            backend_config = self._config.get("cartesian_backend_config", {}) or {}
            if not isinstance(backend_config, dict):
                raise ValueError("cartesian_backend_config must be a mapping")
            linear_speed = float(control_params.get("cartesian_linear_speed", 1.0))
            angular_speed = float(control_params.get("cartesian_angular_speed", 1.0))

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)

            self.servo_client = make_cartesian_backend(
                solver=solver,
                node=self._node,
                tf_buffer=self._tf_buffer,
                base_link=base_link,
                tool_frame=tool_frame,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                input_mode=self.phone_config.cartesian_input_mode,
                managed_teleop=self._config.get("managed_teleop", False),
                **backend_config,
            )
            self.logger.info(
                f"PhoneDevice: cartesian solver={solver}, input_mode={self.phone_config.cartesian_input_mode}, "
                f"tool_frame={tool_frame}, "
                f"base={base_link}, linear_speed={linear_speed}, angular_speed={angular_speed}"
            )

            from sensor_msgs.msg import JointState

            self._joint_state_sub = self._node.create_subscription(
                JointState, self._config.get("joint_state_topic", "/joint_states"), self._joint_state_callback, 10
            )

            self._is_connected = True
            self.logger.info("Phone device connected. Servo will be enabled on first control cycle.")
            self.logger.warning(
                "WebPhone has no user authentication and is only supported on a trusted internal LAN. "
                "Do not expose its HTTP/WebSocket ports through public forwarding, cloud tunnels, "
                "guest Wi-Fi, or untrusted VPNs; restrict access with the host or network firewall."
            )
            for url in self._phone_impl.access_urls:
                self.logger.info(f"WebPhone page: {url}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect phone device: {e}")
            if self._phone_impl is not None:
                self._phone_impl.disconnect()
                self._phone_impl = None
            self._is_connected = False
            return False

    def get_joint_targets(self) -> dict[str, float]:
        """
        Drive the selected Cartesian backend for arm control; return gripper target.

        During go_home Placo owns joint-space motion and measured arrival
        detection through the shared ArmReturnHome action.
        """
        self._consume_transport_stop()

        # Read shared state under lock; ROS calls happen outside
        with self._state_lock:
            going_home = self._going_home
            first_state_rcvd = self._first_state_received
            servo_enabled = self._servo_enabled

        if going_home:
            if self._config.get("managed_teleop", False):
                cmd = self._get_cmd_internal()
                if cmd is None or not cmd.enabled or self._validate_cartesian_command(cmd) is not None:
                    self._disable_motion(force=True)
                    self._require_deadman_release("HOME input lost", request_transport_stop=False)
                else:
                    self.servo_client.keepalive()
            if (
                not self._config.get("managed_teleop", False)
                and self.servo_client is not None
                and not getattr(self.servo_client, "stop_pending", False)
            ):
                self.servo_client.keepalive()
            return self._update_home_state()

        cmd = self._get_cmd_internal()
        if cmd is None:
            backend_active = bool(
                self.servo_client and (servo_enabled or getattr(self.servo_client, "is_enabled", False))
            )
            if backend_active:
                self._fail_closed_on_invalid_command(self._last_command_failure_reason)
            return {}
        invalid_reason = self._validate_cartesian_command(cmd)
        if invalid_reason is not None:
            backend_active = bool(
                self.servo_client and (servo_enabled or getattr(self.servo_client, "is_enabled", False))
            )
            if backend_active:
                self._fail_closed_on_invalid_command(invalid_reason)
            return {}
        if self.servo_client is not None and not getattr(self.servo_client, "stop_pending", False):
            self.servo_client.keepalive()

        if cmd.go_home:
            if self.servo_client is None or not self.servo_client.home():
                self.logger.warning("Go-Home ignored because the Placo ArmReturnHome action is not ready")
                return {self.gripper_joint_names[0]: cmd.gripper_pos}
            with self._state_lock:
                self._going_home = True
                self._servo_enabled = False
                self._clear_pose_state_locked()
            if not self._config.get("managed_teleop", False):
                self._require_deadman_release("go-home requested", request_transport_stop=False)
            return {self.gripper_joint_names[0]: cmd.gripper_pos}

        if not cmd.enabled:
            self._disable_motion()
            return {self.gripper_joint_names[0]: cmd.gripper_pos}

        if not servo_enabled:
            if first_state_rcvd:
                requested = bool(self.servo_client and self.servo_client.enable())
                with self._state_lock:
                    self._servo_enabled = requested
                    self._clear_pose_state_locked()
                if requested:
                    self.logger.info("Servo start requested: waiting for Placo to latch the EE baseline.")
            else:
                return {}
            return {self.gripper_joint_names[0]: cmd.gripper_pos}

        if self.servo_client and not self.servo_client.is_enabled:
            if self._config.get("managed_teleop", False) and not self.servo_client._requested_enabled:
                self._servo_enabled = False
                self._require_deadman_release("runtime admission lost", request_transport_stop=False)
            return {self.gripper_joint_names[0]: cmd.gripper_pos}

        if self._pose_clutch_pos is None or self._pose_clutch_rot is None:
            self._pose_clutch_pos = cmd.pose_position.copy()
            self._pose_clutch_rot = cmd.pose_rotation
            self._pose_sent_pos = np.zeros(3)
            self._pose_sent_rot = Rotation.identity()
            self._pose_filtered_rotvec = np.zeros(3)
            self.servo_client.servo_pose(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
            self.logger.info("Phone pose clutch latched after Placo start confirmation.")
            return {self.gripper_joint_names[0]: cmd.gripper_pos}

        target_position, target_rotation = compute_webphone_pose_delta(
            cmd.pose_position,
            cmd.pose_rotation,
            self._pose_clutch_pos,
            self._pose_clutch_rot,
            position_scale=self.phone_config.position_scale,
            angular_scale=self.phone_config.angular_scale,
            user_scale=cmd.user_scale,
        )
        bounds = self.phone_config.end_effector_bounds
        target_position = np.clip(target_position, bounds["min"], bounds["max"])
        target_rotation = self._filter_pose_rotation(target_rotation)
        position, rotation = self._limit_pose_step(target_position, target_rotation)
        quaternion = rotation.as_quat()
        self.servo_client.servo_pose(
            position=tuple(float(v) for v in position),
            orientation=tuple(float(v) for v in quaternion),
        )

        return {self.gripper_joint_names[0]: cmd.gripper_pos}

    def _update_home_state(self) -> dict[str, float]:
        """Hold the gate until Placo reports measured Home completion."""
        result = self.servo_client.consume_home_result() if self.servo_client else False
        if result is False:
            self._disable_motion(force=True)
            with self._state_lock:
                self._going_home = False
            # The terminal Home path is already disabled on the Placo side. One
            # explicit stop above covers action rejection/transport failure;
            # requesting another transport stop would only delay the next grip.
            self._require_deadman_release("go-home rejected", request_transport_stop=False)
            self.logger.error("Placo rejected Go-Home; phone motion remains released")
            return {self.gripper_joint_names[0]: self._last_gripper_pos}
        if result is True:
            with self._state_lock:
                self._going_home = False
                self._servo_enabled = False
                self._clear_pose_state_locked()
            if self._config.get("managed_teleop", False):
                self._require_deadman_release("HOME completed", request_transport_stop=False)
            # The release latch was set when Home was requested and commands are
            # not consumed while Home owns the backend. A real release observed
            # after this point clears it; do not re-latch or stop Placo again.
        return {self.gripper_joint_names[0]: self._last_gripper_pos}

    def _require_deadman_release(self, reason: str, *, request_transport_stop: bool) -> None:
        with self._state_lock:
            self._deadman_release_required = True
            self._clear_pose_state_locked()
        if self._phone_impl is not None:
            self._phone_impl.require_release(reason, request_stop=request_transport_stop)

    def _clear_pose_state_locked(self) -> None:
        self._pose_clutch_pos = None
        self._pose_clutch_rot = None
        self._pose_sent_pos = np.zeros(3)
        self._pose_sent_rot = Rotation.identity()
        self._pose_filtered_rotvec = np.zeros(3)

    def _filter_pose_rotation(self, target_rotation: Rotation) -> Rotation:
        """Apply reachable-axis masking, deadzone, and low-pass filtering."""
        rotvec = target_rotation.as_rotvec() * self.phone_config.orientation_axis_mask
        rotvec[np.abs(rotvec) < self.phone_config.orientation_deadzone_rad] = 0.0
        alpha = self.phone_config.orientation_filter_alpha
        self._pose_filtered_rotvec = alpha * rotvec + (1.0 - alpha) * self._pose_filtered_rotvec
        return Rotation.from_rotvec(self._pose_filtered_rotvec)

    def _disable_motion(self, *, force: bool = False) -> None:
        # Safety stop must be able to bypass local active flags: during
        # asynchronous home those flags are false while Placo is still moving.
        should_disable = bool(self.servo_client and (force or self._servo_enabled or self.servo_client.is_enabled))
        if should_disable:
            self.servo_client.disable()
        with self._state_lock:
            self._servo_enabled = False
            self._clear_pose_state_locked()

    def _limit_pose_step(self, target_position: np.ndarray, target_rotation: Rotation) -> tuple[np.ndarray, Rotation]:
        position_step = np.asarray(target_position, dtype=float) - self._pose_sent_pos
        position_step = self._limit_vector_norm(position_step, self.phone_config.max_ee_step_m)
        self._pose_sent_pos = self._pose_sent_pos + position_step

        rotation_step = target_rotation * self._pose_sent_rot.inv()
        limited_rotvec = self._limit_vector_norm(rotation_step.as_rotvec(), self.phone_config.max_angular_step_rad)
        self._pose_sent_rot = Rotation.from_rotvec(limited_rotvec) * self._pose_sent_rot
        return self._pose_sent_pos.copy(), self._pose_sent_rot

    def _consume_transport_stop(self) -> None:
        if self._phone_impl is None:
            return
        reason = self._phone_impl.consume_stop_request()
        if reason is None:
            return
        self._disable_motion(force=True)
        with self._state_lock:
            self._going_home = False
            self._deadman_release_required = True
        self.logger.warning(f"WebPhone safety stop: {reason}; release and re-press deadman to resume")

    def _joint_state_callback(self, msg) -> None:
        """Open the start gate only after all selected arm joints are observed."""
        measured = dict(zip(msg.name, msg.position, strict=False))
        with self._state_lock:
            self._current_joint_states.update(measured)
            self._first_state_received = all(name in self._current_joint_states for name in self.arm_joint_names)

    def _get_cmd_internal(self) -> _CartesianCommand | None:
        """Read phone hardware and compute Cartesian command (ROS-free)."""
        if not self._is_connected or self._phone_impl is None:
            self._last_command_failure_reason = "phone transport is not connected"
            return None
        try:
            action = self._phone_impl.get_action()
            if not action:
                self._last_command_failure_reason = "phone input returned no action"
                return None
            command = self._compute_cartesian_command(action)
            if command is None:
                self._last_command_failure_reason = "phone action is incomplete or invalid"
                return None
            self._last_command_failure_reason = ""
            return command
        except Exception as e:
            self._last_command_failure_reason = f"phone input processing failed: {e}"
            self.logger.error(f"Failed to get Cartesian command from phone: {e}")
            return None

    def _validate_cartesian_command(self, command: _CartesianCommand) -> str | None:
        """Return a fail-closed reason when a per-cycle command is incomplete or non-finite."""
        try:
            if not np.isfinite(command.gripper_pos):
                return "phone command contains a non-finite gripper target"
            if command.go_home or not command.enabled:
                return None
            if command.pose_position is None or command.pose_rotation is None:
                return "phone pose command is incomplete"
            position = np.asarray(command.pose_position, dtype=float)
            quaternion = np.asarray(command.pose_rotation.as_quat(), dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
                return "phone pose command contains invalid values"
            return None
        except (AttributeError, TypeError, ValueError):
            return "phone command has an invalid structure"

    def _fail_closed_on_invalid_command(self, reason: str) -> None:
        """Stop an active backend when the current control cycle has no safe command."""
        self._disable_motion(force=True)
        self._require_deadman_release(reason, request_transport_stop=False)
        self.logger.error(f"Phone command failed closed: {reason}; release and re-press deadman to resume")

    def _compute_cartesian_command(self, action: dict[str, Any]) -> _CartesianCommand | None:
        """Map a normalized phone action to unitless Cartesian backend commands."""
        enabled = bool(action.get("phone.enabled", False))
        position = action.get("phone.pos")
        rotation = action.get("phone.rot")
        raw_inputs = action.get("phone.raw_inputs", {})
        if position is None or rotation is None:
            return None

        button_a = float(raw_inputs.get("reservedButtonA", 0.0))
        button_b = float(raw_inputs.get("reservedButtonB", 0.0))
        gripper_velocity = button_a - button_b
        go_home_requested = bool(raw_inputs.get("goHome", False)) or (bool(button_a) and bool(button_b))
        go_home = go_home_requested and not self._go_home_prev
        self._go_home_prev = go_home_requested

        with self._state_lock:
            release_required = self._deadman_release_required
            if release_required and not enabled:
                self._deadman_release_required = False
        if release_required:
            enabled = False
            go_home = False

        self._last_gripper_pos = float(
            np.clip(
                self._last_gripper_pos + gripper_velocity * self.phone_config.gripper_speed_factor * self._control_dt,
                self.phone_config.gripper_range[0],
                self.phone_config.gripper_range[1],
            )
        )

        if not enabled:
            return _CartesianCommand(
                gripper_pos=self._last_gripper_pos,
                go_home=go_home,
                enabled=False,
            )

        scale = float(np.clip(self._finite_float(raw_inputs.get("scale", 1.0), 1.0), 0.1, 4.0))
        return _CartesianCommand(
            gripper_pos=self._last_gripper_pos,
            go_home=go_home,
            enabled=True,
            pose_position=np.asarray(position, dtype=float).copy(),
            pose_rotation=rotation,
            user_scale=scale,
        )

    @staticmethod
    def _limit_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if not np.isfinite(max_norm) or max_norm <= 0.0:
            return np.zeros_like(vector)
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm <= 1e-12:
            return vector
        return vector * (max_norm / norm)

    @staticmethod
    def _finite_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if np.isfinite(parsed) else default

    def disconnect(self) -> None:
        """Disconnect from phone device and disable Servo."""
        self._disable_motion(force=True)
        with self._state_lock:
            self._going_home = False
            self._deadman_release_required = True
        if self._phone_impl is not None:
            self._phone_impl.disconnect()
            self._phone_impl = None
        self._is_connected = False

    @property
    def shutdown_complete(self) -> bool:
        """Return whether the downstream stop request has been acknowledged."""
        return not bool(self.servo_client and getattr(self.servo_client, "stop_pending", False))

    def emergency_stop(self) -> None:
        """Disable Cartesian motion and latch WebPhone re-grip recovery."""
        self._disable_motion(force=True)
        with self._state_lock:
            self._going_home = False
        self._require_deadman_release("emergency stop", request_transport_stop=False)
