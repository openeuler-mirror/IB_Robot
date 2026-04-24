"""
Phone teleoperation device implementation.

Supports iOS (via HEBI Mobile I/O) and Android (via WebXR) for
6-DoF pose-based robot teleoperation.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np

from ..base_teleop import BaseTeleopDevice
from scipy.spatial.transform import Rotation
from .config_phone import PhoneConfig, PhoneOS

logger = logging.getLogger(__name__)


@dataclass
class _CartesianCommand:
    """Internal per-cycle Cartesian command from phone (linear/angular displacement + gripper)."""
    linear: "np.ndarray"
    angular: "np.ndarray"
    gripper_pos: float = 0.0
    go_home: bool = False


class BasePhone:
    """Base class for phone teleoperation with calibration support."""

    _enabled: bool = False
    _calib_pos: Optional[np.ndarray] = None
    _calib_rot_inv: Optional[Rotation] = None

    def _reapply_position_calibration(self, pos: np.ndarray) -> None:
        """Reapply position calibration (called on enable rising edge)."""
        self._calib_pos = pos.copy()

    @property
    def is_calibrated(self) -> bool:
        """Check if the phone has been calibrated."""
        return (self._calib_pos is not None) and (self._calib_rot_inv is not None)

    def get_action_features(self) -> Dict[str, type]:
        """Get the action features provided by this phone."""
        return {
            "phone.pos": np.ndarray,
            "phone.rot": Rotation,
            "phone.raw_inputs": dict,
            "phone.enabled": bool,
        }


class IOSPhone(BasePhone):
    """
    iOS phone teleoperation via HEBI Mobile I/O app.

    Uses HEBI SDK to communicate with the Mobile I/O app for
    ARKit-based 6-DoF pose tracking.
    """

    def __init__(self, config: PhoneConfig):
        self.config = config
        self._group = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._group is not None

    def connect(self) -> bool:
        """Connect to iOS device via HEBI SDK."""
        try:
            import hebi

            logger.info("Connecting to iPhone, make sure to open the HEBI Mobile I/O app.")
            lookup = hebi.Lookup()
            time.sleep(2.0)

            group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
            if group is None:
                raise RuntimeError(
                    "Mobile I/O not found — check name/family settings in the app."
                )

            self._group = group
            self._is_connected = True
            logger.info(f"Connected to HEBI group with {group.size} module(s).")

            self.calibrate()
            return True

        except ImportError:
            logger.error("=" * 60)
            logger.error("MISSING DEPENDENCY: 'hebi' package not installed!")
            logger.error("")
            logger.error("This package is required for iOS phone teleoperation.")
            logger.error("")
            logger.error("Install with:")
            logger.error("    pip install hebi")
            logger.error("")
            logger.error("Or install all phone dependencies:")
            logger.error("    pip install -r $(ros2 pkg prefix robot_teleop)/share/robot_teleop/requirements.txt")
            logger.error("=" * 60)
            return False
        except Exception as e:
            logger.error(f"Failed to connect to iOS device: {e}")
            self._is_connected = False
            return False

    def calibrate(self) -> None:
        """Perform calibration by capturing reference pose."""
        print(
            "Hold the phone so that: top edge points forward in same direction "
            "as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Press and hold B1 in the HEBI Mobile I/O app to capture this pose...\n")

        position, rotation = self._wait_for_capture_trigger()
        self._calib_pos = position.copy()
        self._calib_rot_inv = rotation.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> Tuple[np.ndarray, Rotation]:
        """Wait for B1 button press to capture calibration pose."""
        while True:
            has_pose, position, rotation, fb_pose = self._read_current_pose()
            if not has_pose:
                time.sleep(0.01)
                continue

            io = getattr(fb_pose, "io", None)
            button_b = getattr(io, "b", None) if io is not None else None
            button_b1_pressed = False
            if button_b is not None:
                button_b1_pressed = bool(button_b.get_int(1))

            if button_b1_pressed:
                return position, rotation

            time.sleep(0.01)

    def _read_current_pose(
        self,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Rotation], Any]:
        """Read the current 6-DoF pose from the iOS device."""
        if self._group is None:
            return False, None, None, None

        fbk = self._group.get_next_feedback()
        pose = fbk[0]
        ar_pos = getattr(pose, "ar_position", None)
        ar_quat = getattr(pose, "ar_orientation", None)

        if ar_pos is None or ar_quat is None:
            return False, None, None, None

        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))
        rot = Rotation.from_quat(quat_xyzw)
        pos = ar_pos - rot.apply(self.config.camera_offset)

        return True, pos, rot, pose

    def get_action(self) -> Dict[str, Any]:
        """Get the current phone action (pose and inputs)."""
        has_pose, raw_position, raw_rotation, fb_pose = self._read_current_pose()

        if not has_pose or not self.is_calibrated:
            return {}

        raw_inputs: Dict[str, Any] = {}
        io = getattr(fb_pose, "io", None)
        if io is not None:
            bank_a, bank_b = io.a, io.b
            if bank_a:
                for ch in range(1, 9):
                    if bank_a.has_float(ch):
                        raw_inputs[f"a{ch}"] = float(bank_a.get_float(ch))
            if bank_b:
                for ch in range(1, 9):
                    if bank_b.has_int(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_int(ch))
                    elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_bool(ch))

        enable = bool(raw_inputs.get("b1", 0))

        if enable and not self._enabled:
            self._reapply_position_calibration(raw_position)

        pos_cal = self._calib_rot_inv.apply(raw_position - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rotation

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    def disconnect(self) -> None:
        """Disconnect from iOS device."""
        self._group = None
        self._is_connected = False


class AndroidPhone(BasePhone):
    """
    Android phone teleoperation via WebXR.

    Uses the teleop package to receive pose data from a WebXR session
    in the phone's browser.
    """

    def __init__(self, config: PhoneConfig):
        self.config = config
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._android_lock = threading.Lock()
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._teleop is not None

    def connect(self) -> bool:
        """Connect to Android device via teleop package."""
        try:
            from teleop import Teleop

            logger.info("Starting teleop stream for Android...")
            self._teleop = Teleop()
            self._teleop.subscribe(self._android_callback)
            self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
            self._teleop_thread.start()
            self._is_connected = True
            logger.info("Connected, teleop stream started.")

            self.calibrate()
            return True

        except ImportError:
            logger.error("=" * 60)
            logger.error("MISSING DEPENDENCY: 'teleop' package not installed!")
            logger.error("")
            logger.error("This package is required for Android phone teleoperation.")
            logger.error("")
            logger.error("Install with:")
            logger.error("    pip install teleop")
            logger.error("")
            logger.error("Or install all phone dependencies:")
            logger.error("    pip install -r $(ros2 pkg prefix robot_teleop)/share/robot_teleop/requirements.txt")
            logger.error("=" * 60)
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Android device: {e}")
            self._is_connected = False
            return False

    def calibrate(self) -> None:
        """Perform calibration by capturing reference pose."""
        print(
            "Hold the phone so that: top edge points forward in same direction "
            "as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Touch and move on the WebXR page to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> Tuple[np.ndarray, Rotation]:
        """Wait for touch move event to capture calibration pose."""
        while True:
            with self._android_lock:
                msg = self._latest_message or {}

            if bool(msg.get("move", False)):
                ok, pos, rot, _pose = self._read_current_pose()
                if ok:
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(
        self,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Rotation], Any]:
        """Read the latest pose from the Android device."""
        with self._android_lock:
            if self._latest_pose is None:
                return False, None, None, None
            p = self._latest_pose.copy()
            pose = self._latest_pose

        rot = Rotation.from_matrix(p[:3, :3])
        pos = p[:3, 3] - rot.apply(self.config.camera_offset)

        return True, pos, rot, pose

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        """Callback for receiving pose data from Android device."""
        with self._android_lock:
            self._latest_pose = pose
            self._latest_message = message

    def get_action(self) -> Dict[str, Any]:
        """Get the current phone action (pose and inputs)."""
        ok, raw_pos, raw_rot, pose = self._read_current_pose()

        if not ok or not self.is_calibrated:
            return {}

        raw_inputs: Dict[str, Any] = {}
        msg = self._latest_message or {}
        raw_inputs["move"] = bool(msg.get("move", False))
        raw_inputs["scale"] = float(msg.get("scale", 1.0))
        raw_inputs["reservedButtonA"] = bool(msg.get("reservedButtonA", False))
        raw_inputs["reservedButtonB"] = bool(msg.get("reservedButtonB", False))

        enable = bool(raw_inputs.get("move", False))

        if enable and not self._enabled:
            self._reapply_position_calibration(raw_pos)

        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    def disconnect(self) -> None:
        """Disconnect from Android device."""
        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
        self._latest_pose = None
        self._is_connected = False


class PhoneDevice(BaseTeleopDevice):
    """
    Phone-based teleoperation device.

    Supports iOS (via HEBI Mobile I/O) and Android (via WebXR).

    Parses sensor data from the phone, drives MoveIt Servo for arm Cartesian
    control via servo_client.servo(), and returns only the gripper target to
    TeleopNode for direct publishing.

    go_home mode switches to joint-position control: returns full arm+gripper
    targets until the arm reaches home, then re-enables Servo.
    """

    def __init__(self, config: dict, node=None):
        super().__init__(config, node=node)

        phone_config_data = config.get("phone_config", {})
        if isinstance(phone_config_data, dict):
            self.phone_config = PhoneConfig.from_dict(phone_config_data)
        else:
            self.phone_config = PhoneConfig()

        self._phone_impl: Optional[BasePhone] = None
        self._last_gripper_pos: float = 0.0
        self._prev_pos: Optional[np.ndarray] = None
        self._prev_rot: Optional[Rotation] = None

        # _state_lock protects only shared state; ROS calls happen outside the lock
        self._state_lock = threading.Lock()
        self.servo_client = None
        self._joint_state_sub = None
        self._current_joint_states: Dict[str, float] = {}
        self._first_state_received = False
        self._going_home = False
        self._servo_enabled = False
        self._home_start_time: Optional[float] = None

        # Injected by teleop.py launch builder
        self.arm_joint_names = config.get('arm_joint_names', ['1', '2', '3', '4', '5'])
        self.gripper_joint_names = config.get('gripper_joint_names', ['6'])
        self.home_joint_positions = config.get('home_joint_positions', [0.0, 0.0, 0.0, 0.0, 0.0])
        self._control_dt = 1.0 / config.get('control_frequency', 30.0)

    def connect(self) -> bool:
        """Connect to phone hardware and initialise MoveIt Servo client."""
        if self._node is None:
            self.logger.error("PhoneDevice requires a ROS node reference (node=None)")
            return False

        try:
            if self.phone_config.phone_os == PhoneOS.IOS:
                self._phone_impl = IOSPhone(self.phone_config)
            elif self.phone_config.phone_os == PhoneOS.ANDROID:
                self._phone_impl = AndroidPhone(self.phone_config)
            else:
                raise ValueError(f"Invalid phone_os: {self.phone_config.phone_os}")

            if not self._phone_impl.connect():
                return False

            from pymoveit2.moveit2_servo import MoveIt2Servo
            frame_id = self._config.get('base_link_name', 'base')
            self.servo_client = MoveIt2Servo(
                node=self._node, frame_id=frame_id,
                linear_speed=1.0, angular_speed=1.0,
                enable_at_init=False)

            from sensor_msgs.msg import JointState
            self._joint_state_sub = self._node.create_subscription(
                JointState, '/joint_states', self._joint_state_callback, 10)

            self._is_connected = True
            self.logger.info("Phone device connected. Servo will be enabled on first control cycle.")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect phone device: {e}")
            self._is_connected = False
            return False

    def get_joint_targets(self) -> Dict[str, float]:
        """
        Drive MoveIt Servo for arm Cartesian control; return gripper target.

        During go_home: returns full arm+gripper targets for joint-position
        control and re-enables Servo once arm reaches home position.
        """
        # Read shared state under lock; ROS calls happen outside
        with self._state_lock:
            going_home = self._going_home
            home_start_time = self._home_start_time
            first_state_rcvd = self._first_state_received
            current_joints = dict(self._current_joint_states)
            servo_enabled = self._servo_enabled

        if going_home:
            return self._compute_home_targets(
                current_joints, first_state_rcvd, home_start_time)

        cmd = self._get_cmd_internal()
        if cmd is None:
            return {}

        if not servo_enabled:
            if first_state_rcvd and not cmd.go_home:
                with self._state_lock:
                    self._servo_enabled = True
                self.servo_client.enable()
                self.logger.info("Servo enabled: joint_states ready and phone control active.")
            else:
                return {}

        if self.servo_client and not self.servo_client.is_enabled:
            return {}

        if cmd.go_home:
            with self._state_lock:
                self._going_home = True
                self._home_start_time = self._node.get_clock().now().nanoseconds * 1e-9
            if self.servo_client and self.servo_client.is_enabled:
                self.servo_client.disable()          # ROS call, outside lock
            return self._compute_home_targets(
                current_joints, first_state_rcvd,
                self._node.get_clock().now().nanoseconds * 1e-9)

        # Cartesian control: normalize displacement to [-1, 1] for MoveIt Servo unitless mode.
        # cmd.linear is already clamped to max_ee_step_m, so dividing gives values in [-1, 1].
        max_lin = self.phone_config.max_ee_step_m
        max_ang = self.phone_config.max_angular_step_rad
        linear = tuple(float(v) / max_lin for v in cmd.linear)
        angular = tuple(float(v) / max_ang for v in cmd.angular)
        self.servo_client.servo(linear=linear, angular=angular)  # ROS call, outside lock

        return {self.gripper_joint_names[0]: cmd.gripper_pos}

    def _compute_home_targets(
        self,
        current_joints: Dict[str, float],
        first_state_rcvd: bool,
        home_start_time: Optional[float],
    ) -> Dict[str, float]:
        """Return home joint targets; clear _going_home flag once arm arrives."""
        targets = dict(zip(self.arm_joint_names, self.home_joint_positions))
        targets[self.gripper_joint_names[0]] = self._last_gripper_pos
        now = self._node.get_clock().now().nanoseconds * 1e-9
        elapsed = now - (home_start_time or now)
        # Arrival check: received joint_states + ≥0.5 s elapsed + error < 0.05 rad
        if first_state_rcvd and elapsed >= 0.5:
            actual = [current_joints.get(n) for n in self.arm_joint_names]
            if None not in actual:
                if all(abs(a - h) < 0.05 for a, h in zip(actual, self.home_joint_positions)):
                    with self._state_lock:
                        self._going_home = False
                        self._home_start_time = None
                        self._servo_enabled = True
                    self.servo_client.enable()       # ROS call, outside lock
        return targets

    def _joint_state_callback(self, msg) -> None:
        """Update joint state cache (only writes shared state, never blocks)."""
        with self._state_lock:
            for name, pos in zip(msg.name, msg.position):
                self._current_joint_states[name] = pos
            self._first_state_received = True

    def _get_cmd_internal(self) -> Optional[_CartesianCommand]:
        """Read phone hardware and compute Cartesian command (ROS-free)."""
        if not self._is_connected or self._phone_impl is None:
            return None
        try:
            action = self._phone_impl.get_action()
            if not action:
                return None
            return self._compute_cartesian_command(action)
        except Exception as e:
            self.logger.error(f"Failed to get Cartesian command from phone: {e}")
            return None

    def _compute_cartesian_command(self, action: Dict[str, Any]) -> Optional[_CartesianCommand]:
        """Map raw phone action to a _CartesianCommand (pure math, no ROS)."""
        enabled = action.get("phone.enabled", False)
        pos = action.get("phone.pos")
        rot = action.get("phone.rot")
        raw_inputs = action.get("phone.raw_inputs", {})

        if pos is None or rot is None:
            return None

        if self.phone_config.phone_os == PhoneOS.IOS:
            gripper_vel = float(raw_inputs.get("a3", 0.0))
            go_home = bool(raw_inputs.get("b2", 0))
        else:
            a = float(raw_inputs.get("reservedButtonA", 0.0))
            b = float(raw_inputs.get("reservedButtonB", 0.0))
            gripper_vel = a - b
            go_home = (
                bool(raw_inputs.get("reservedButtonA", 0))
                and bool(raw_inputs.get("reservedButtonB", 0))
            )

        if not enabled:
            self._prev_pos = None
            self._prev_rot = None
            return _CartesianCommand(
                linear=np.zeros(3),
                angular=np.zeros(3),
                gripper_pos=self._last_gripper_pos,
                go_home=go_home,
            )

        self._last_gripper_pos = float(np.clip(
            self._last_gripper_pos + gripper_vel * self.phone_config.gripper_speed_factor,
            self.phone_config.gripper_range[0],
            self.phone_config.gripper_range[1],
        ))

        # First frame after enable: record baseline, output zero
        if self._prev_pos is None or self._prev_rot is None:
            self._prev_pos = pos.copy()
            self._prev_rot = rot
            return _CartesianCommand(
                linear=np.zeros(3),
                angular=np.zeros(3),
                gripper_pos=self._last_gripper_pos,
                go_home=go_home,
            )

        # Differential: compute change since last frame
        delta_pos = pos - self._prev_pos
        delta_rot = self._prev_rot.inv() * rot

        self._prev_pos = pos.copy()
        self._prev_rot = rot

        linear = delta_pos.astype(float) * self.phone_config.linear_scale

        # Clamp to max single-step magnitude
        norm = float(np.linalg.norm(linear))
        if norm > self.phone_config.max_ee_step_m and norm > 0:
            linear = linear * (self.phone_config.max_ee_step_m / norm)

        # Differential rotation → angular displacement in base frame.
        # delta_rot is in body frame (phone frame); rotate it to base frame
        # so MoveIt Servo receives axis directions in the fixed robot coordinate system.
        angular = rot.apply(delta_rot.as_rotvec())

        # Clamp to max single-step angular magnitude
        angular_norm = float(np.linalg.norm(angular))
        if angular_norm > self.phone_config.max_angular_step_rad and angular_norm > 0:
            angular = angular * (self.phone_config.max_angular_step_rad / angular_norm)

        return _CartesianCommand(
            linear=linear,
            angular=angular,
            gripper_pos=self._last_gripper_pos,
            go_home=go_home,
        )

    def disconnect(self) -> None:
        """Disconnect from phone device and disable Servo."""
        if self.servo_client and self.servo_client.is_enabled:
            self.servo_client.disable()
        with self._state_lock:
            self._servo_enabled = False
            self._going_home = False
        if self._phone_impl is not None:
            self._phone_impl.disconnect()
            self._phone_impl = None
        self._is_connected = False
