#!/usr/bin/env python3
"""SO101 Placo Servo node.

In-process Placo QP differential-IK Cartesian teleop backend for the SO-101
5-DOF arm. ``placo_servo`` is the SO-101 Cartesian teleop solver selected by
``robot.teleoperation.cartesian.solver == 'placo_servo'``.

Why an independent node:

* **Does not block the 50 Hz teleop loop** — Placo IK runs on this node's own
  timer, isolated from ``teleop_node``'s 5 ms latency budget.

Design (Jacobian + QP velocity-level differential IK, see
``so101_placo_kinematics.SO101PlacoDiffIK``):

* **Command-side reference, not measured-pose ratchet.** The node maintains a
  virtual Cartesian reference and advances it by ``v * dt``. The QP is seeded
  from the last published command, not from hardware ``/joint_states`` every
  tick: real servos sag/lag under gravity, and feeding that measured lag into a
  5-DOF position-only QP makes the nullspace drift. Measured joints initialise
  enable/reset only; the command path sends the same ideal trajectory to
  hardware that simulation computes.
* **Position primary, orientation soft.** A Placo ``PositionTask`` constrains
  target position and an optional low-weight ``OrientationTask`` follows angular
  joystick input without letting orientation dominate the under-actuated arm.
* **Hard limits in the QP.** Joint + velocity limits are enforced
  inside the QP (``enable_joint_limits`` / ``enable_velocity_limits``), so the
  unreachable velocity component is projected onto the feasible set (correct
  energy decomposition) and the arm yields at the reachable edge. In Cartesian
  mode the arm command chain bypasses ``TeleopNode``'s ``SafetyFilter``, so this
  in-solver clamp is the authoritative joint-limit guard; the node also clamps
  the published command to the configured limits as defense in depth.
* **Hardware-safe seed.** Every step seeds from the command-side joint state
  (last published command). ``/joint_states`` remains a diagnostic/initial latch
  source, not the hot-path seed.
* **Same control point path for gripper & tcp.** ``target_frame`` is
  ``ik_link_name`` (== ``moveit.ee_link``); tcp is a fixed child of gripper, and
  Placo's frame Jacobian propagates the 95 mm lever arm exactly — no special
  case.

Smoothness: a fixed-rate timer (default 50 Hz), in-QP velocity limit as the real
speed ceiling, and an optional first-order low-pass on the joint command.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray
from std_srvs.srv import Trigger

from ibrobot_msgs.action import ArmReturnHome
from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode, StopRuntime

# The radian-native Placo wrapper lives next to this node (installed into the
# same lib/<pkg> directory). Make the script dir importable so both the
# installed node and a direct source-tree run resolve it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from so101_placo_kinematics import SO101PlacoDiffIK  # noqa: E402


def _require_placo(logger) -> None:
    try:
        import placo  # noqa: F401
    except ImportError as exc:
        logger.fatal(
            "placo is required for solver=placo_servo. Run ./scripts/setup.sh "
            "to install LeRobot with the kinematics extra, or install the "
            "matching LeRobot extra manually with: python3 -m pip install -e "
            "'libs/lerobot[kinematics]'"
        )
        raise RuntimeError("placo is required for solver=placo_servo") from exc


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _rotation_delta(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix for a base-frame angular delta vector."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = np.asarray(rotvec, dtype=np.float64) / theta
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix (3x3) from a quaternion (x, y, z, w).

    Normalises defensively; a zero/degenerate quaternion falls back to identity
    so a malformed pose command cannot inject NaNs into the QP.
    """
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass
class _HomeActionRequest:
    """Cross-callback handoff for one ArmReturnHome action transaction."""

    goal_handle: object
    done: threading.Event
    outcome: str | None = None
    error_code: str = ""
    message: str = ""
    max_joint_error_rad: float = float("inf")
    cancel_requested: bool = False


@dataclass(frozen=True)
class _HomePreemption:
    """Terminal error reserved before the action execute callback starts."""

    outcome: str
    error_code: str
    message: str


class SO101PlacoServoNode(Node):
    """In-process Placo QP differential-IK Cartesian servo for SO-101."""

    # ------------------------------------------------------------------ init
    def __init__(self) -> None:
        super().__init__("so101_placo_servo_node")

        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.action_group = ReentrantCallbackGroup()

        # ---- parameters ----
        self.declare_parameter("planning_frame", "base")
        self.declare_parameter("ik_link_name", "gripper")  # target frame in URDF
        self.declare_parameter("arm_joint_names", ["1", "2", "3", "4", "5"])

        # Set true only for diagnostics. Normal SO-101 Cartesian teleop uses
        # position primary + a low-weight orientation task.
        self.declare_parameter("position_only", False)

        # Differential-IK damping: regularization weight giving DLS-like smooth
        # yielding at singularities / reachable boundaries.
        self.declare_parameter("diffik_damping", 1e-3)

        # Orientation tracking: small soft weight, lower than position.
        # Ignored when position_only=true.
        self.declare_parameter("orientation_weight", 0.01)

        # Per-joint velocity ceiling (rad/s) for the QP velocity-limit
        # constraint. The SO-101 URDF declares 10 rad/s (too fast for hand
        # teleop); this is the real speed ceiling of the servo.
        self.declare_parameter("max_joint_speed", 2.0)

        # Smoothness knob: optional first-order low-pass on q_des. The
        # real speed ceiling is the in-QP velocity limit, not a per-tick clip.
        self.declare_parameter("output_lowpass_alpha", 0.0)  # 0 = off; (0,1] = on

        # Arm joint limits (rad) — self-owned safety. Required: the
        # node refuses to start without them so a misconfig fails loudly.
        # dynamic_typing avoids the "declare name only" deprecation while still
        # treating absence as "not set" (validated below).
        from rcl_interfaces.msg import ParameterDescriptor

        _dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("joint_limits_lower", descriptor=_dyn)
        self.declare_parameter("joint_limits_upper", descriptor=_dyn)

        # Input mode. "velocity" (default): integrate a base-frame twist into
        # the command reference (xbox/phone path). "pose": consume a relative
        # clutch pose delta in the base frame (VR passthrough) — the command
        # reference is SET (not integrated), so a held hand pose holds the arm
        # with zero drift. PoseStamped.position is added to the measured EE
        # position latched at enable (ee0); orientation is a base-frame rotation
        # delta that left-multiplies the latched EE attitude. Clutch baseline
        # (ee0) and FK live here, not on the VR node.
        self.declare_parameter("input_mode", "velocity")
        self.declare_parameter("pose_cmd_topic", "/so101_placo_servo_node/pose_cmd_base")

        # Topics & services consumed by PlacoServoBackend.
        self.declare_parameter("linear_cmd_topic", "/so101_placo_servo_node/linear_cmd_base")
        self.declare_parameter("angular_cmd_topic", "/so101_placo_servo_node/angular_cmd_base")
        self.declare_parameter("start_service", "/so101_placo_servo_node/start")
        self.declare_parameter("stop_service", "/so101_placo_servo_node/stop")
        self.declare_parameter("home_action", "/so101_placo_servo_node/return_home")
        self.declare_parameter("command_lease_topic", "/so101_placo_servo_node/command_lease")
        self.declare_parameter("estop_topic", "/emergency_stop")
        # Teleop Home is a deterministic joint-space safe return. The launch
        # builder injects ros2_control.reset_positions in arm_joint_names order.
        self.declare_parameter("home_joint_positions", descriptor=_dyn)
        self.declare_parameter("home_joint_tolerance_rad", 0.05)
        self.declare_parameter("home_max_joint_speed", 1.0)
        self.declare_parameter("home_joint_state_stale_s", 0.2)
        self.declare_parameter("home_stable_duration_s", 0.2)
        self.declare_parameter("home_timeout_s", 10.0)
        self.declare_parameter("command_out_topic", "/arm_position_controller/commands")

        # Timing.
        self.declare_parameter("control_period", 0.02)  # 50 Hz solve/publish
        self.declare_parameter("incoming_command_timeout", 0.5)
        self.declare_parameter("target_reset_timeout_s", 2.0)
        self.declare_parameter("tf_stale_threshold_s", 0.2)
        # Zero preserves the existing VR/Xbox behavior. Phone launch enables the
        # lease from its existing command_stale_s setting.
        self.declare_parameter("command_lease_timeout_s", 0.0)
        # Runtime provider (design D11): claim the runtime's stream mode on
        # start and release it (idle) on stop through SetRuntimeMode. Empty
        # disables the integration (in-config ros2_control bring-up).
        self.declare_parameter("runtime_mode_service", "")
        self.declare_parameter("runtime_stop_service", "")
        self.declare_parameter("runtime_status_topic", "")
        self.declare_parameter("managed_teleop", False)
        self.declare_parameter("joint_intent_topic", "")
        self.declare_parameter("gripper_joint_names", ["6"])
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("gripper_command_out_topic", "")
        self.declare_parameter("gripper_limits_lower", descriptor=_dyn)
        self.declare_parameter("gripper_limits_upper", descriptor=_dyn)
        self.declare_parameter("future_tolerance_s", 0.05)
        self.declare_parameter("runtime_stream_mode", "stream")
        self.declare_parameter("runtime_idle_mode", "idle")

        # Optional pre-expanded URDF path; if absent the node expands the
        # so101 xacro in-memory at runtime (decision: runtime expansion).
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description", "")

        # ---- snapshot ----
        self.planning_frame = self.get_parameter("planning_frame").value
        self.ik_link_name = self.get_parameter("ik_link_name").value
        self.arm_joint_names: list[str] = list(self.get_parameter("arm_joint_names").value)
        self.position_only = bool(self.get_parameter("position_only").value)
        self.diffik_damping = float(self.get_parameter("diffik_damping").value)
        self.orientation_weight = float(self.get_parameter("orientation_weight").value)
        self.max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self.output_lowpass_alpha = _clamp(float(self.get_parameter("output_lowpass_alpha").value), 0.0, 1.0)

        def _opt_param(name):
            try:
                value = self.get_parameter(name).value
            except Exception:  # noqa: BLE001 — treat unset as None
                return None
            return value

        jl_lower = _opt_param("joint_limits_lower")
        jl_upper = _opt_param("joint_limits_upper")
        if jl_lower is None or jl_upper is None:
            raise RuntimeError(
                "joint_limits_lower / joint_limits_upper are required. "
                "Load so101_placo_servo.yaml (set moveit.so101_placo_servo_config_path)."
            )
        self.joint_lo = np.array(jl_lower, dtype=np.float64)
        self.joint_hi = np.array(jl_upper, dtype=np.float64)
        if len(self.joint_lo) != len(self.arm_joint_names) or len(self.joint_hi) != len(self.arm_joint_names):
            raise RuntimeError(
                f"joint_limits_lower/upper length must match arm_joint_names ({len(self.arm_joint_names)})"
            )

        self.linear_topic = self.get_parameter("linear_cmd_topic").value
        self.angular_topic = self.get_parameter("angular_cmd_topic").value
        self.pose_topic = self.get_parameter("pose_cmd_topic").value
        self.input_mode = str(self.get_parameter("input_mode").value).lower()
        if self.input_mode not in ("velocity", "pose", "auto"):
            raise RuntimeError(f"input_mode must be velocity, pose or auto, got {self.input_mode!r}")
        self.start_srv_name = self.get_parameter("start_service").value
        self.stop_srv_name = self.get_parameter("stop_service").value
        self.home_action_name = self.get_parameter("home_action").value
        self.command_lease_topic = self.get_parameter("command_lease_topic").value
        self.estop_topic = self.get_parameter("estop_topic").value
        home_joint_positions = _opt_param("home_joint_positions")
        self._home_q = np.asarray(home_joint_positions, dtype=np.float64) if home_joint_positions is not None else None
        self._home_enabled = bool(
            self._home_q is not None
            and self._home_q.shape == (len(self.arm_joint_names),)
            and np.all(np.isfinite(self._home_q))
        )
        if self._home_enabled and (np.any(self._home_q < self.joint_lo) or np.any(self._home_q > self.joint_hi)):
            raise RuntimeError("home_joint_positions must stay within joint_limits_lower/upper")
        self.home_joint_tolerance_rad = float(self.get_parameter("home_joint_tolerance_rad").value)
        self.home_max_joint_speed = float(self.get_parameter("home_max_joint_speed").value)
        self.home_joint_state_stale_s = float(self.get_parameter("home_joint_state_stale_s").value)
        self.home_stable_duration_s = float(self.get_parameter("home_stable_duration_s").value)
        self.home_timeout_s = float(self.get_parameter("home_timeout_s").value)
        home_limits = (
            self.home_joint_tolerance_rad,
            self.home_max_joint_speed,
            self.home_joint_state_stale_s,
            self.home_stable_duration_s,
            self.home_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in home_limits):
            raise RuntimeError(
                "Home joint tolerance, speed, stale threshold, stable duration, and timeout must be positive"
            )
        if self.home_max_joint_speed > self.max_joint_speed:
            raise RuntimeError("home_max_joint_speed must not exceed max_joint_speed")
        self.cmd_out_topic = self.get_parameter("command_out_topic").value
        self.control_period = float(self.get_parameter("control_period").value)
        self.input_timeout = float(self.get_parameter("incoming_command_timeout").value)
        self.target_reset_timeout = float(self.get_parameter("target_reset_timeout_s").value)
        self.tf_stale_threshold_s = float(self.get_parameter("tf_stale_threshold_s").value)
        self.command_lease_timeout_s = float(self.get_parameter("command_lease_timeout_s").value)
        if not np.isfinite(self.command_lease_timeout_s) or self.command_lease_timeout_s < 0.0:
            raise RuntimeError("command_lease_timeout_s must be finite and non-negative")
        self.runtime_mode_service = str(self.get_parameter("runtime_mode_service").value).strip()
        self.managed_teleop = bool(self.get_parameter("managed_teleop").value)
        self.runtime_stop_service = str(self.get_parameter("runtime_stop_service").value).strip()
        self.runtime_status_topic = str(self.get_parameter("runtime_status_topic").value).strip()
        self.joint_intent_topic = str(self.get_parameter("joint_intent_topic").value).strip()
        self.gripper_joint_names = list(self.get_parameter("gripper_joint_names").value)
        self.gripper_out_topic = str(self.get_parameter("gripper_command_out_topic").value).strip()
        self.future_tolerance_s = float(self.get_parameter("future_tolerance_s").value)
        self.gripper_lo = np.asarray(_opt_param("gripper_limits_lower") or [], dtype=np.float64)
        self.gripper_hi = np.asarray(_opt_param("gripper_limits_upper") or [], dtype=np.float64)
        self.runtime_stream_mode = str(self.get_parameter("runtime_stream_mode").value).strip()
        self.runtime_idle_mode = str(self.get_parameter("runtime_idle_mode").value).strip()
        urdf_path = self.get_parameter("urdf_path").value or None
        urdf_xml = self.get_parameter("robot_description").value or None
        if self.managed_teleop:
            if not all(
                (
                    self.runtime_mode_service,
                    self.runtime_stop_service,
                    self.runtime_status_topic,
                    self.joint_intent_topic,
                    self.gripper_out_topic,
                    urdf_xml,
                )
            ):
                raise RuntimeError(
                    "managed teleop requires runtime endpoints, joint/gripper topics and robot_description"
                )
            names = self.arm_joint_names + self.gripper_joint_names
            if not self.gripper_joint_names or len(set(names)) != len(names):
                raise RuntimeError("managed joint names must be nonempty and disjoint")
            for lo, hi, count in (
                (self.gripper_lo, self.gripper_hi, len(self.gripper_joint_names)),
                (self.joint_lo, self.joint_hi, len(self.arm_joint_names)),
            ):
                if lo.shape != (count,) or hi.shape != (count,) or not np.all(np.isfinite([lo, hi])) or np.any(lo > hi):
                    raise RuntimeError("managed joint limits must be finite ordered vectors")
            if not np.isfinite(self.input_timeout) or self.input_timeout <= 0:
                raise RuntimeError("incoming_command_timeout must be positive")
            if not np.isfinite(self.future_tolerance_s) or self.future_tolerance_s < 0:
                raise RuntimeError("future_tolerance_s must be nonnegative")
            if self.command_lease_timeout_s == 0:
                self.command_lease_timeout_s = self.input_timeout

        # ---- Placo differential-IK (in-process, radian-native, IB-Robot URDF) ----
        _require_placo(self.get_logger())
        self.diffik = SO101PlacoDiffIK(
            urdf_path=None if urdf_xml else urdf_path,
            urdf_xml=urdf_xml,
            target_frame=self.ik_link_name,
            arm_joint_names=self.arm_joint_names,
            damping=self.diffik_damping,
            control_period=self.control_period,
            max_joint_speed=self.max_joint_speed,
            orientation_weight=0.0 if self.position_only else self.orientation_weight,
        )

        # ---- state ----
        self._enabled: bool = False
        self._estop_active = False
        # Command-side IK state: hardware /joint_states are used to initialise
        # and reset, but NOT as the per-tick IK seed. Real servos lag/sag under
        # gravity; if that measured lag is fed back into a 5-DOF position-only
        # QP, the under-constrained nullspace drifts (seen on hardware as J4
        # snapping while simulation stayed correct). Instead, _last_cmd is the
        # seed and _p_ref is the held Cartesian reference. This sends the same
        # ideal command trajectory to hardware that simulation computes, while
        # still letting enable snap cleanly to the real measured state.
        self._p_ref: np.ndarray | None = None  # command-side EE reference (base frame, m)
        self._r_ref: np.ndarray | None = None  # command-side EE orientation (base frame, 3x3)
        self._last_cmd: np.ndarray | None = None  # last published joint command
        self._latest_linear: Vector3Stamped | None = None
        self._latest_linear_stamp: float = 0.0
        self._latest_angular: Vector3Stamped | None = None
        self._latest_angular_stamp: float = 0.0
        self._accept_velocity_commands: bool = False
        # Pose-mode input: the latest pose command carries a RELATIVE clutch
        # increment in the base frame. ``position`` is added to the EE position
        # latched at enable; ``orientation`` is a base-frame rotation delta that
        # left-multiplies the EE attitude latched at enable.
        self._latest_pose = None  # PoseStamped | None
        self._latest_pose_stamp: float = 0.0
        # Gate for pose-topic acceptance. stop/home close it and only the next
        # start re-opens it. A pose message that was already in the DDS queue when
        # a home/stop service call ran therefore cannot land in _on_pose afterward
        # and revive a stale relative displacement onto the freshly-latched
        # baseline. The pose topic and service have no cross-entity ordering
        # guarantee, so clearing _latest_pose alone does not cover an in-flight
        # message; this gate does.
        self._accept_pose_commands: bool = False
        self._ee0_p: np.ndarray | None = None  # measured EE position at enable (base, m)
        self._ee0_R: np.ndarray | None = None  # measured EE rotation at enable (base, 3x3)
        self._latest_js: JointState | None = None
        self._latest_js_received_at = 0.0
        self._joint_state_generation = 0
        self._last_input_time: float = 0.0
        self._last_lease_time: float = self._now()
        self._home_active = False
        self._home_started_at = 0.0
        self._home_stable_since: float | None = None
        self._home_last_joint_state_generation = 0
        self._home_request_lock = threading.Lock()
        self._home_goal_reserved = False
        self._home_preemption: _HomePreemption | None = None
        self._pending_home_request: _HomeActionRequest | None = None
        self._active_home_request: _HomeActionRequest | None = None
        self._managed_owner = False
        self._stop_latched = False
        self._runtime_status = None
        self._managed_mode = None
        self._joint_intent = None
        self._joint_intent_stamp = 0.0
        self._last_managed_input = 0.0
        self._managed_start_stamp = float("inf")
        self._stop_pending = False
        self._stop_confirmed = False
        self._stop_future = None
        self._stop_attempt_at = 0.0
        self._status_received_at = 0.0

        # ---- diagnostics counters ----
        self._recovery_count: int = 0
        self._dropped_frame_count: int = 0
        self._solve_count: int = 0

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- ROS I/O ----
        # Velocity-mode inputs (xbox/phone). Always subscribed so a mode switch
        # via param reconfigure does not require re-wiring; ignored in pose mode.
        self.create_subscription(Vector3Stamped, self.linear_topic, self._on_linear, 10, callback_group=self.cb_group)
        self.create_subscription(Vector3Stamped, self.angular_topic, self._on_angular, 10, callback_group=self.cb_group)
        # Pose-mode input (VR passthrough). Only subscribed in pose mode.
        if self.input_mode in ("pose", "auto"):
            self.create_subscription(PoseStamped, self.pose_topic, self._on_pose, 10, callback_group=self.cb_group)
        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            1,
            callback_group=self.cb_group,
        )
        self.create_subscription(Bool, self.estop_topic, self._on_estop, 10, callback_group=self.cb_group)
        if self.managed_teleop and self.joint_intent_topic:
            self.create_subscription(
                JointState, self.joint_intent_topic, self._on_joint_intent, 10, callback_group=self.cb_group
            )
        self._runtime_status_sub = (
            self.create_subscription(
                RuntimeStatus, self.runtime_status_topic, self._on_runtime_status, 10, callback_group=self.cb_group
            )
            if self.managed_teleop and self.runtime_status_topic
            else None
        )
        if self.command_lease_timeout_s > 0.0:
            self.create_subscription(
                Empty,
                self.command_lease_topic,
                self._on_command_lease,
                10,
                callback_group=self.cb_group,
            )
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.cmd_out_topic, 10)
        self.create_service(Trigger, self.start_srv_name, self._on_start_srv, callback_group=self.cb_group)
        self.create_service(Trigger, self.stop_srv_name, self._on_stop_srv, callback_group=self.cb_group)
        self._runtime_mode_client = (
            self.create_client(SetRuntimeMode, self.runtime_mode_service, callback_group=self.action_group)
            if self.runtime_mode_service
            else None
        )
        self._runtime_stop_client = (
            self.create_client(StopRuntime, self.runtime_stop_service, callback_group=self.action_group)
            if self.managed_teleop and self.runtime_stop_service
            else None
        )
        self.gripper_pub = (
            self.create_publisher(Float64MultiArray, self.gripper_out_topic, 10) if self.gripper_out_topic else None
        )
        self._home_action_server = ActionServer(
            self,
            ArmReturnHome,
            self.home_action_name,
            execute_callback=self._execute_home_action,
            goal_callback=self._home_goal_callback,
            cancel_callback=self._home_cancel_callback,
            callback_group=self.action_group,
        )

        # ---- timer (fixed-rate solve+publish for smoothness) ----
        self.create_timer(self.control_period, self._on_control_tick, callback_group=self.cb_group)
        if self.managed_teleop:
            self.create_timer(0.05, self._retry_runtime_stop, callback_group=self.cb_group)

        self.get_logger().info(
            f"so101_placo_servo_node up: frame={self.planning_frame} ik_link={self.ik_link_name} "
            f"mode={self.input_mode} "
            f"joints={self.arm_joint_names} rate={1.0 / self.control_period:.0f}Hz "
            f"diffik(damping={self.diffik_damping}, max_speed={self.max_joint_speed}rad/s, "
            f"orientation_weight={0.0 if self.position_only else self.orientation_weight}) "
            f"{'POSITION-ONLY ' if self.position_only else ''}"
            f"lowpass={self.output_lowpass_alpha}"
        )

    # ------------------------------------------------------------------ subs
    def _on_linear(self, msg: Vector3Stamped) -> None:
        if self.managed_teleop and not self._accept_managed_cartesian(msg, "velocity"):
            return
        if not self._accept_velocity_commands:
            return
        self._latest_linear = msg
        self._latest_linear_stamp = self._now()
        self._last_input_time = self._latest_linear_stamp

    def _on_angular(self, msg: Vector3Stamped) -> None:
        if self.managed_teleop and not self._accept_managed_cartesian(msg, "velocity"):
            return
        if not self._accept_velocity_commands:
            return
        self._latest_angular = msg
        self._latest_angular_stamp = self._now()
        self._last_input_time = self._latest_angular_stamp

    def _on_pose(self, msg: PoseStamped) -> None:
        if self.managed_teleop and not self._accept_managed_cartesian(msg, "pose"):
            return
        # Reject pose commands while the gate is closed (between a stop/home and
        # the next start re-latch). This drops any message that was already
        # queued in DDS when the service ran, which would otherwise overwrite the
        # freshly-latched baseline with a stale relative displacement.
        if not self._accept_pose_commands:
            return
        self._latest_pose = msg
        self._latest_pose_stamp = self._now()
        self._last_input_time = self._latest_pose_stamp

    def _on_joint_intent(self, msg: JointState) -> None:
        if (
            not self._enabled
            or self._home_active
            or not self._managed_owner
            or not self._valid_managed_stamp(msg.header.stamp)
        ):
            return
        if (
            len(msg.name) != len(msg.position)
            or len(set(msg.name)) != len(msg.name)
            or not np.all(np.isfinite(msg.position))
        ):
            return
        names = dict(zip(msg.name, msg.position, strict=False))
        full = set(names) == set(self.arm_joint_names + self.gripper_joint_names)
        if not full and set(names) != set(self.gripper_joint_names):
            return
        if full:
            if self._managed_mode not in (None, "joint"):
                return
            self._managed_mode = "joint"
            self._joint_intent = msg
            self._joint_intent_stamp = self._stamp(msg.header.stamp)
            self._last_managed_input = self._now()
        if self.gripper_pub and all(n in names for n in self.gripper_joint_names):
            values = np.clip([names[n] for n in self.gripper_joint_names], self.gripper_lo, self.gripper_hi)
            out = Float64MultiArray()
            out.data = [float(v) for v in values]
            self.gripper_pub.publish(out)

    def _accept_managed_cartesian(self, msg, mode: str) -> bool:
        if (
            not self._enabled
            or not self._managed_owner
            or self._home_active
            or self._managed_mode not in (None, mode)
            or self.input_mode not in ("auto", mode)
            or not self._valid_managed_stamp(msg.header.stamp)
            or msg.header.frame_id != self.planning_frame
        ):
            return False
        if mode == "pose":
            p, q = msg.pose.position, msg.pose.orientation
            values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
            if np.linalg.norm(values[3:]) < 1e-8:
                return False
        else:
            values = [msg.vector.x, msg.vector.y, msg.vector.z]
        if not np.all(np.isfinite(values)):
            return False
        self._managed_mode = mode
        self._last_managed_input = self._now()
        return True

    def _on_runtime_status(self, msg: RuntimeStatus) -> None:
        if self._managed_owner and hasattr(msg, "stamp") and self._stamp(msg.stamp) < self._managed_start_stamp:
            return
        self._runtime_status = msg
        self._status_received_at = self._now()
        self._stop_latched = msg.stop_latched
        if (
            self._stop_pending
            and msg.stop_latched
            and msg.stop_policy == "HOLD"
            and msg.active_mode == self.runtime_idle_mode
        ):
            self._stop_confirmed = True
            if self._stop_future is None:
                self._stop_pending = False
        if self._managed_owner and (
            msg.active_mode != self.runtime_stream_mode or msg.lifecycle != "ACTIVE" or msg.stop_latched
        ):
            self._managed_stop("runtime stream ownership lost")

    @staticmethod
    def _stamp(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _valid_managed_stamp(self, stamp) -> bool:
        value = self._stamp(stamp)
        now = self.get_clock().now().nanoseconds * 1e-9
        return (
            self._managed_owner
            and value > self._managed_start_stamp
            and value <= now + self.future_tolerance_s
            and now - value <= self.input_timeout
        )

    def _managed_stop(self, reason: str) -> None:
        self._managed_owner = False
        self._preempt_home("STOP_REQUESTED", reason)
        self._home_active = False
        self._disable_motion_state()
        self._managed_mode = None
        self._joint_intent = None
        self._managed_start_stamp = float("inf")
        if not self._stop_pending:
            self._stop_confirmed = False
        self._stop_pending = True
        self._stop_latched = True
        self._retry_runtime_stop()

    def _retry_runtime_stop(self) -> None:
        if not self._stop_pending:
            return
        now = self._now()
        if self._stop_future is not None:
            if self._stop_future.done():
                try:
                    response = self._stop_future.result()
                    if response is not None and response.success:
                        self._stop_confirmed = True
                        self._stop_pending = False
                        self._stop_future = None
                        return
                except Exception:
                    pass
                self._stop_future = None
            else:
                # Cancelling a client future does not cancel the remote request.
                # Never queue another HOLD behind a slow stop and a later rearm.
                return
        if self._runtime_stop_client is not None and self._runtime_stop_client.service_is_ready():
            request = StopRuntime.Request()
            request.policy = "HOLD"
            self._stop_attempt_at = now
            try:
                self._stop_future = self._runtime_stop_client.call_async(request)
            except Exception:
                self._stop_future = None

    def _on_joint_state(self, msg: JointState) -> None:
        if self.managed_teleop:
            now = self.get_clock().now().nanoseconds * 1e-9
            if (
                len(msg.name) != len(msg.position)
                or len(set(msg.name)) != len(msg.name)
                or not set(self.arm_joint_names + self.gripper_joint_names).issubset(msg.name)
                or not np.all(np.isfinite(msg.position))
                or not 0 <= now - self._stamp(msg.header.stamp) <= self.home_joint_state_stale_s
            ):
                return
        self._latest_js = msg
        self._latest_js_received_at = self._now()
        self._joint_state_generation += 1

    def _on_command_lease(self, _msg: Empty) -> None:
        if self.managed_teleop and not self._managed_owner:
            return
        self._last_lease_time = self._now()

    def _on_estop(self, msg: Bool) -> None:
        active = bool(msg.data)
        if active:
            self._estop_active = True
            if self.managed_teleop:
                self._managed_stop("emergency stop")
            self._preempt_home("EMERGENCY_STOP", "ArmReturnHome aborted by emergency stop")
            self._disable_motion_state()
            self.get_logger().error("emergency stop active; Placo motion disabled")
            return
        if self._estop_active:
            self.get_logger().info("emergency stop released; a new start is still required")
        self._estop_active = False

    def _home_goal_callback(self, goal_request: ArmReturnHome.Goal) -> GoalResponse:
        if self.managed_teleop and (
            not self._managed_owner
            or not self._enabled
            or self._stop_latched
            or self._now() - self._last_lease_time > self.command_lease_timeout_s
        ):
            return GoalResponse.REJECT
        if self._estop_active or goal_request.target_name not in ("", "home") or not self._home_enabled:
            return GoalResponse.REJECT
        with self._home_request_lock:
            if (
                self._home_goal_reserved
                or self._pending_home_request is not None
                or self._active_home_request is not None
            ):
                return GoalResponse.REJECT
            self._home_goal_reserved = True
            self._home_preemption = None
        return GoalResponse.ACCEPT

    def _home_cancel_callback(self, goal_handle) -> CancelResponse:
        with self._home_request_lock:
            for request in (self._pending_home_request, self._active_home_request):
                if request is not None and request.goal_handle is goal_handle:
                    request.cancel_requested = True
                    return CancelResponse.ACCEPT
            if self._home_goal_reserved:
                if self._home_preemption is None:
                    self._home_preemption = _HomePreemption(
                        "canceled",
                        "CANCELED",
                        "ArmReturnHome canceled by client before execution",
                    )
                return CancelResponse.ACCEPT
        return CancelResponse.REJECT

    def _execute_home_action(self, goal_handle) -> ArmReturnHome.Result:
        request = _HomeActionRequest(goal_handle=goal_handle, done=threading.Event())
        with self._home_request_lock:
            preemption = self._home_preemption
            self._home_preemption = None
            if preemption is None:
                self._pending_home_request = request
            else:
                self._home_goal_reserved = False

        if preemption is not None:
            self._complete_home_request(request, preemption.outcome, preemption.error_code, preemption.message)

        while rclpy.ok() and not request.done.wait(timeout=0.05):
            if goal_handle.is_cancel_requested:
                request.cancel_requested = True

        if not request.done.is_set():
            self._complete_home_request(
                request,
                "aborted",
                "ROS_SHUTDOWN",
                "ArmReturnHome interrupted by ROS shutdown",
            )

        result = ArmReturnHome.Result()
        result.success = request.outcome == "succeeded"
        result.error_code = request.error_code
        result.message = request.message
        if request.outcome == "succeeded":
            goal_handle.succeed()
        elif request.outcome == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _process_home_action(self, now: float) -> None:
        with self._home_request_lock:
            request = self._active_home_request
            if request is None and self._pending_home_request is not None:
                request = self._pending_home_request
                self._pending_home_request = None
                self._active_home_request = request
        if request is None:
            return
        if request.cancel_requested:
            self._finish_home("canceled", "CANCELED", "ArmReturnHome canceled by client")
            return
        if not self._home_active:
            self._begin_home(now)

    def _begin_home(self, now: float) -> None:
        if self.managed_teleop and (
            not self._managed_owner or not self._enabled or now - self._last_lease_time > self.command_lease_timeout_s
        ):
            self._managed_stop("HOME lease or ownership unavailable")
            return
        q = self._measured_arm_joints()
        joint_state_fresh = (
            self._latest_js_received_at > 0.0 and now - self._latest_js_received_at <= self.home_joint_state_stale_s
        )
        if q is None or not np.all(np.isfinite(q)) or not joint_state_fresh:
            self._finish_home("aborted", "JOINT_STATE_UNAVAILABLE", "ArmReturnHome requires fresh finite joint states")
            return

        self._enabled = True
        self._accept_velocity_commands = False
        self._accept_pose_commands = False
        self._latest_linear = None
        self._latest_angular = None
        self._latest_pose = None
        self._p_ref = None
        self._r_ref = None
        self._ee0_p = None
        self._ee0_R = None
        self._last_cmd = q.copy()
        if not self.managed_teleop:
            self._last_lease_time = now
        self._home_active = True
        self._home_started_at = now
        self._home_stable_since = None
        self._home_last_joint_state_generation = self._joint_state_generation
        self.get_logger().info("ArmReturnHome accepted: moving to ros2_control.reset_positions")

    def _finish_home(self, outcome: str, error_code: str, message: str, max_error: float = float("inf")) -> None:
        self._home_active = False
        self._home_stable_since = None
        self._disable_motion_state()
        if self.managed_teleop:
            self._managed_owner = False
            self._stop_pending = True
            self._stop_confirmed = False
            self._stop_latched = True
            self._retry_runtime_stop()
        with self._home_request_lock:
            request = self._active_home_request
            self._active_home_request = None
            self._home_goal_reserved = False
            self._home_preemption = None
        if request is None:
            return
        self._complete_home_request(request, outcome, error_code, message, max_error)

    @staticmethod
    def _complete_home_request(
        request: _HomeActionRequest,
        outcome: str,
        error_code: str,
        message: str,
        max_error: float = float("inf"),
    ) -> None:
        request.outcome = outcome
        request.error_code = error_code
        request.message = message
        request.max_joint_error_rad = max_error
        request.done.set()

    def _preempt_home(self, error_code: str, message: str) -> None:
        """Make stop/shutdown win even before the action execute callback exists."""
        with self._home_request_lock:
            request = self._active_home_request or self._pending_home_request
            if request is None:
                if self._home_goal_reserved:
                    self._home_preemption = _HomePreemption("aborted", error_code, message)
                return
            self._active_home_request = None
            self._pending_home_request = None
            self._home_goal_reserved = False
            self._home_preemption = None

        self._home_active = False
        self._home_stable_since = None
        self._disable_motion_state()
        self._complete_home_request(request, "aborted", error_code, message)

    def _disable_motion_state(self) -> None:
        self._enabled = False
        self._accept_velocity_commands = False
        self._accept_pose_commands = False
        self._latest_linear = None
        self._latest_angular = None
        self._latest_pose = None
        self._p_ref = None
        self._r_ref = None
        self._ee0_p = None
        self._ee0_R = None
        self._last_cmd = None

    # ------------------------------------------------------------------ srvs
    def _on_start_srv(self, _req, resp: Trigger.Response) -> Trigger.Response:
        if self.managed_teleop:
            status = self._runtime_status
            if (
                self._managed_owner
                or self._stop_pending
                or self._stop_latched
                or status is None
                or status.lifecycle != "ACTIVE"
                or status.active_mode != self.runtime_idle_mode
                or status.stop_latched
                or self._now() - self._status_received_at > 2.0
            ):
                resp.success = False
                resp.message = "enable refused: runtime must be ACTIVE/idle, unlatched and unowned"
                return resp
            js = self._latest_js
            if (
                js is None
                or len(js.name) != len(js.position)
                or len(set(js.name)) != len(js.name)
                or not set(self.arm_joint_names + self.gripper_joint_names).issubset(js.name)
                or not np.all(np.isfinite(js.position))
                or self._now() - self._latest_js_received_at > self.home_joint_state_stale_s
            ):
                resp.success = False
                resp.message = "enable refused: fresh complete finite joint state required"
                return resp
        if self._estop_active:
            resp.success = False
            resp.message = "enable refused: emergency stop is active"
            return resp
        with self._home_request_lock:
            home_reserved = self._home_goal_reserved
        if self._home_active or home_reserved:
            resp.success = False
            resp.message = "enable refused: ArmReturnHome has priority"
            return resp
        # Latch command-side state to the current measured pose on enable so the
        # first hold tick targets where the arm actually is (no jump). After
        # that, the hot path seeds from _last_cmd to avoid feeding hardware
        # sag/lag into the position-only QP nullspace.
        q = self._measured_arm_joints()
        if q is None:
            resp.success = False
            resp.message = "enable refused: no /joint_states seed available yet"
            self.get_logger().error(resp.message)
            return resp
        ok, reason = self._request_runtime_mode(self.runtime_stream_mode)
        if not ok:
            if self.managed_teleop:
                self._managed_stop("stream claim failed or timed out")
            resp.success = False
            resp.message = f"enable refused: {reason}"
            self.get_logger().error(resp.message)
            return resp
        if self.managed_teleop and self._now() - self._latest_js_received_at > self.home_joint_state_stale_s:
            self._managed_stop("joint feedback expired during stream admission")
            resp.success = False
            resp.message = "enable refused: fresh reference expired during mode transition"
            return resp
        self._p_ref = self.diffik.ee_position(q)
        self._r_ref = self.diffik.ee_rotation(q)
        # Clutch baseline for pose mode: incoming relative displacements are
        # added to this measured EE position latched at enable, and relative
        # rotations are composed onto this measured EE attitude.
        self._ee0_p = self._p_ref.copy()
        self._ee0_R = self._r_ref.copy()
        self._last_cmd = q.copy()
        # Clear any stale pose command from the previous grip cycle. On trigger
        # release the VR node stops publishing (arm holds), so _latest_pose is
        # frozen at the last relative displacement of the PREVIOUS grip. Without
        # this reset, re-enabling (re-grip) would add that stale displacement
        # onto the freshly-latched _ee0_p before the first new pose arrives,
        # jerking the arm hard — especially after moving the hand between grips
        # (e.g. shifting seat/headset). Dropping it forces pose_stale=True until
        # a fresh command lands, so the arm holds the new baseline.
        self._latest_pose = None
        now = self._now()
        if self.managed_teleop:
            self._managed_owner = True
            self._managed_mode = None
            self._joint_intent = None
            self._managed_start_stamp = self.get_clock().now().nanoseconds * 1e-9
            self._last_managed_input = now
            self._stop_confirmed = False
        self._latest_pose_stamp = now
        self._enabled = True
        self._last_input_time = now
        # Topic and service callbacks have no cross-entity delivery ordering.
        # Refresh here so a freshly accepted Phone start cannot lose its lease
        # before the first keepalive message is dispatched by DDS.
        self._last_lease_time = now
        # Open the pose gate LAST: drop the stale cache, then accept fresh
        # commands. Any pose queued before this instant was rejected by the gate.
        self._latest_linear = None
        self._latest_angular = None
        if self.managed_teleop:
            self._latest_linear_stamp = now
            self._latest_angular_stamp = now
        self._accept_velocity_commands = True
        self._accept_pose_commands = True
        resp.success = True
        resp.message = "so101_placo_servo_node enabled"
        self.get_logger().info(resp.message)
        return resp

    def _on_stop_srv(self, _req, resp: Trigger.Response) -> Trigger.Response:
        if self.managed_teleop:
            if not self._stop_confirmed:
                self._managed_stop("teleop stop requested")
                self._retry_runtime_stop()
            resp.success = self._stop_confirmed
            resp.message = "runtime HOLD confirmed" if resp.success else "runtime HOLD pending; retry stop"
            return resp
        self._preempt_home("STOP_REQUESTED", "ArmReturnHome aborted by stop request")
        self._disable_motion_state()
        ok, reason = self._request_runtime_mode(self.runtime_idle_mode)
        if not ok:
            self.get_logger().warning(f"runtime mode release failed: {reason}")
        resp.success = True
        resp.message = "so101_placo_servo_node disabled"
        self.get_logger().info(resp.message)
        return resp

    def _request_runtime_mode(self, mode: str) -> tuple[bool, str]:
        """Ask the runtime facade for ``mode``; a no-op without a runtime provider."""
        client = getattr(self, "_runtime_mode_client", None)
        if client is None or not mode:
            return True, ""
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f"runtime mode service {self.runtime_mode_service} unavailable"
        request = SetRuntimeMode.Request()
        request.mode = mode
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(5.0):
            return False, f"runtime mode request {mode!r} timed out"
        response = future.result()
        if response is None or not response.success:
            return False, f"runtime rejected mode {mode!r}: {response.message if response else 'no response'}"
        return True, ""

    # ------------------------------------------------------------------ tick
    def _on_control_tick(self) -> None:
        """50 Hz: closed-loop QP step toward the command-side reference, publish.

        Steps:
            1. fetch linear twist (zero if stale / idle) — already in base frame
            2. read measured joints for enable/reset/diagnostics, but seed the
               QP from the command-side joint state (last published command)
            3. advance the command-side reference position by v*dt (only while
               the user commands motion; a zero hold leaves it fixed)
            4. one Placo PositionTask QP step toward that absolute reference;
               joint + velocity limits enforced inside the QP
            5. finite-check, defense-in-depth limit clamp, optional low-pass
            6. publish
        """
        now = self._now()
        if self.managed_teleop and self._managed_owner:
            home_pending = self._home_active or self._home_goal_reserved
            expired = (
                now - self._last_lease_time > self.command_lease_timeout_s
                if home_pending
                else now - self._last_managed_input > self.input_timeout
            )
            if not home_pending:
                expired = expired or now - self._last_lease_time > self.command_lease_timeout_s
                if self._managed_mode == "velocity":
                    expired = expired or any(
                        now - stamp > self.input_timeout
                        for stamp in (self._latest_linear_stamp, self._latest_angular_stamp)
                    )
            expired = expired or now - self._latest_js_received_at > self.home_joint_state_stale_s
            if expired or now - self._status_received_at > 2.0:
                self._managed_stop("teleop command or runtime status expired")
                return
        if self._estop_active:
            self._disable_motion_state()
            return
        self._process_home_action(now)
        if not self._enabled:
            return

        if (
            not self.managed_teleop
            and self.command_lease_timeout_s > 0.0
            and now - self._last_lease_time > self.command_lease_timeout_s
        ):
            if self._home_active:
                self._finish_home("aborted", "COMMAND_LEASE_EXPIRED", "ArmReturnHome command lease expired")
            else:
                self._disable_motion_state()
            self.get_logger().error("command lease expired; so101_placo_servo_node disabled")
            return
        if self._home_timed_out(now):
            return
        dt = self.control_period

        # Hardware /joint_states are not a stable IK seed under gravity: the
        # servos can lag the command, and feeding that lag back into a
        # position-only 5-DOF QP lets the under-constrained wrist/nullspace drift
        # (observed on hardware as J4 snapping upward while sim stayed correct).
        # Use measured joints only to initialise/reset the command-side state
        # and for diagnostics. The actual IK seed is last_cmd, i.e. the ideal
        # command trajectory — same as simulation and what the user requested
        # (“even sending the sim ideal values to hardware is OK”).
        q_measured = self._measured_arm_joints()
        if q_measured is None:
            self.get_logger().warn("no /joint_states yet; skipping solve", throttle_duration_sec=2.0)
            self._dropped_frame_count += 1
            return
        q_cmd_seed = self._last_cmd.copy() if self._last_cmd is not None else q_measured.copy()

        if self._home_active:
            if now - self._latest_js_received_at > self.home_joint_state_stale_s:
                self._finish_home("aborted", "JOINT_STATE_STALE", "ArmReturnHome joint-state feedback became stale")
                return
            if not np.all(np.isfinite(q_measured)):
                self._finish_home("aborted", "JOINT_STATE_INVALID", "ArmReturnHome joint-state feedback is non-finite")
                return
            max_step = self.home_max_joint_speed * dt
            q_des = q_cmd_seed + np.clip(self._home_q - q_cmd_seed, -max_step, max_step)
            q_des = np.clip(q_des, self.joint_lo, self.joint_hi)
            out = Float64MultiArray()
            out.data = [float(value) for value in q_des]
            self.cmd_pub.publish(out)
            self._last_cmd = q_des
            self._solve_count += 1
            self._update_home_progress(q_measured, now)
            return

        if self.managed_teleop and self._managed_mode == "joint":
            if self._joint_intent is not None:
                values = dict(zip(self._joint_intent.name, self._joint_intent.position, strict=True))
                target = np.clip([values[n] for n in self.arm_joint_names], self.joint_lo, self.joint_hi)
                step = self.max_joint_speed * dt
                q_des = q_cmd_seed + np.clip(target - q_cmd_seed, -step, step)
                out = Float64MultiArray()
                out.data = [float(v) for v in np.clip(q_des, self.joint_lo, self.joint_hi)]
                self.cmd_pub.publish(out)
                self._last_cmd = np.asarray(out.data)
            return
        if self.managed_teleop and self._managed_mode is None:
            return

        idle = (now - self._last_input_time) > self.target_reset_timeout

        if self.input_mode == "pose" or (self.managed_teleop and self._managed_mode == "pose"):
            # === Pose passthrough (VR) ===
            # The command reference is SET from a relative clutch command, not
            # integrated. Incoming PoseStamped.position is a relative EE
            # displacement (hand delta * scale) added to the clutch baseline
            # (_ee0_p, the measured EE latched at enable). Its orientation is a
            # base-frame rotation delta that left-multiplies the latched EE
            # attitude. Because the reference is set (not accumulated), a held
            # hand pose holds the arm with zero drift and a stopped hand stops the
            # arm immediately — no residual integration.
            pose_stale = self._latest_pose is None or (now - self._latest_pose_stamp) > self.input_timeout
            if self._p_ref is None or self._ee0_p is None or self._ee0_R is None:
                self._p_ref = self.diffik.ee_position(q_measured)
                self._r_ref = self.diffik.ee_rotation(q_measured)
                self._ee0_p = self._p_ref.copy()
                self._ee0_R = self._r_ref.copy()
                q_cmd_seed = q_measured.copy()
            elif not pose_stale:
                # Fresh pose command: position is a relative displacement added
                # to the clutch baseline; orientation is a relative rotation
                # composed onto the baseline EE attitude. rel_R comes from VR as
                # an increment in the BASE frame (same frame as the position
                # delta: ΔR_base = R_current * R_clutch^-1), so it must
                # LEFT-multiply: r_ref = rel_R @ ee0_R. Using ee0_R @ rel_R would
                # treat rel_R as a tool/body-frame increment, which couples a
                # single-axis wrist turn into a compound EE motion. At the press
                # instant rel_R = identity so the arm holds. The VR node
                # (vr_teleop._control_so101_pose) produces rel_R under exactly
                # this base-frame contract.
                p = self._latest_pose.pose.position
                q_ = self._latest_pose.pose.orientation
                self._p_ref = self._ee0_p + np.array([p.x, p.y, p.z], dtype=np.float64)
                if not self.position_only:
                    rel_R = _quat_to_matrix(q_.x, q_.y, q_.z, q_.w)
                    self._r_ref = rel_R @ self._ee0_R
            # else: stale/idle -> hold last _p_ref/_r_ref unchanged (arm holds).
        else:
            # === Velocity integration (xbox/phone) ===
            # --- linear velocity (base frame, m/s) ---
            # Hold to zero when input is stale or after a long idle.
            if idle or self._latest_linear is None or (now - self._latest_linear_stamp) > self.input_timeout:
                v = np.zeros(3)
            else:
                lv = self._latest_linear.vector
                v = np.array([lv.x, lv.y, lv.z], dtype=np.float64)

            # --- angular velocity (base frame, rad/s) ---
            # Integrate a command-side orientation reference. The backend has
            # already converted tool-frame stick semantics into base-frame angular
            # velocity for placo_servo.
            if (
                self.position_only
                or idle
                or self._latest_angular is None
                or (now - self._latest_angular_stamp) > self.input_timeout
            ):
                w = np.zeros(3)
            else:
                av = self._latest_angular.vector
                w = np.array([av.x, av.y, av.z], dtype=np.float64)

            # === Command-side reference (the gravity-ratchet fix) ===
            # _p_ref is the EE position the arm is commanded to hold. It is advanced
            # by v*dt ONLY when the user commands motion; a zero hold leaves it
            # fixed. On a real arm the measured joints sag under gravity, so the
            # measured EE drifts away from _p_ref — and because the QP targets the
            # FIXED _p_ref (not "v*dt from the sagging measured pose"), it actively
            # corrects the sag instead of welding it in (the hardware bug).
            #
            # Do NOT re-snap _p_ref just because the sticks are idle: that would
            # reintroduce the hardware gravity ratchet after target_reset_timeout
            # (the reference would repeatedly accept the sagging measured pose).
            # Re-snap only when the reference is missing (startup/recovery); use the
            # stop/start service if a human physically moved or blocked the arm and
            # wants to accept the new pose as the command baseline.
            if self._p_ref is None:
                self._p_ref = self.diffik.ee_position(q_measured)
                self._r_ref = self.diffik.ee_rotation(q_measured)
                q_cmd_seed = q_measured.copy()
            else:
                self._p_ref = self._p_ref + v * dt
                if not self.position_only:
                    if self._r_ref is None:
                        self._r_ref = self.diffik.ee_rotation(q_measured)
                    self._r_ref = _rotation_delta(w * dt) @ self._r_ref

        # === QP step toward the absolute reference (Jacobian + QP) ===
        # Seeded from the command-side joints; target is the command-side _p_ref.
        # Joint + velocity limits are enforced INSIDE the QP, so a large
        # reference error converges over a few ticks rather than snapping, and
        # an unreachable reference component is projected onto the feasible set.
        try:
            if self.position_only or self._r_ref is None:
                q_des = self.diffik.solve_to_position(q_cmd_seed, self._p_ref, dt)
            else:
                q_des = self.diffik.solve_to_pose(q_cmd_seed, self._p_ref, self._r_ref, dt)
        except Exception as exc:  # noqa: BLE001
            # The QP is built to stay feasible; a raised exception is
            # unexpected. Skip this tick rather than publish a bad command; the
            # next tick re-seeds from the command-side state again.
            self.get_logger().warn(f"diffik step raised; skipping tick: {exc}", throttle_duration_sec=1.0)
            self._recovery_count += 1
            return

        if not np.all(np.isfinite(q_des)):
            self.get_logger().warn("diffik produced non-finite joints; skipping tick", throttle_duration_sec=1.0)
            self._recovery_count += 1
            return

        # === Defense-in-depth joint-limit clamp ===
        # The QP already enforces limits; this is a redundant guard because the
        # Cartesian arm chain bypasses TeleopNode's SafetyFilter.
        q_des = np.clip(q_des, self.joint_lo, self.joint_hi)

        # === Optional output low-pass for de-jitter ===
        if self.output_lowpass_alpha > 0.0 and self._last_cmd is not None:
            a = self.output_lowpass_alpha
            q_des = a * q_des + (1.0 - a) * self._last_cmd

        out = Float64MultiArray()
        out.data = [float(x) for x in q_des]
        self.cmd_pub.publish(out)
        self._last_cmd = q_des
        self._solve_count += 1
        self._update_home_progress(q_measured, now)

    # ------------------------------------------------------------------ helpers
    def _now(self) -> float:
        return time.monotonic()

    def _update_home_progress(self, q_measured: np.ndarray, now: float) -> None:
        if not self._home_active or self._home_q is None:
            return
        if self._joint_state_generation <= self._home_last_joint_state_generation:
            return
        self._home_last_joint_state_generation = self._joint_state_generation

        max_error = float(np.max(np.abs(q_measured - self._home_q)))
        with self._home_request_lock:
            request = self._active_home_request
        if request is not None:
            request.max_joint_error_rad = max_error
            feedback = ArmReturnHome.Feedback()
            feedback.state = "moving"
            feedback.max_joint_error_rad = max_error
            request.goal_handle.publish_feedback(feedback)
        within_tolerance = max_error <= self.home_joint_tolerance_rad
        if not within_tolerance:
            self._home_stable_since = None
            return
        if self._home_stable_since is None:
            self._home_stable_since = now
            return
        if now - self._home_stable_since < self.home_stable_duration_s:
            return

        self._finish_home("succeeded", "", "ArmReturnHome reached reset_positions", max_error)
        self.get_logger().info(f"ArmReturnHome reached: max_joint_error={max_error:.4f}rad")

    def _home_timed_out(self, now: float) -> bool:
        """Abort Home independently of joint-state and IK availability."""
        if not self._home_active or now - self._home_started_at <= self.home_timeout_s:
            return False

        message = "ArmReturnHome timed out"
        max_error = float("inf")
        q_measured = self._measured_arm_joints()
        if (
            q_measured is not None
            and self._home_q is not None
            and q_measured.shape == self._home_q.shape
            and np.all(np.isfinite(q_measured))
        ):
            errors = np.abs(q_measured - self._home_q)
            worst_index = int(np.argmax(errors))
            max_error = float(errors[worst_index])
            joint_name = self.arm_joint_names[worst_index]
            message = (
                f"ArmReturnHome timed out: joint {joint_name!r} error={max_error:.4f}rad "
                f"(measured={q_measured[worst_index]:.4f}, target={self._home_q[worst_index]:.4f}, "
                f"tolerance={self.home_joint_tolerance_rad:.4f})"
            )

        self._finish_home("aborted", "TIMEOUT", message, max_error)
        self.get_logger().error(f"{message}; servo disabled")
        return True

    def _measured_arm_joints(self) -> np.ndarray | None:
        """Return measured arm joints (rad) ordered by ``arm_joint_names``."""
        if self._latest_js is None:
            return None
        measured = dict(zip(self._latest_js.name, self._latest_js.position, strict=False))
        try:
            return np.array([measured[n] for n in self.arm_joint_names], dtype=np.float64)
        except KeyError:
            return None

    def _tf_age_ok(self) -> bool:
        """Best-effort TF freshness check (diagnostics; not on the hot path)."""
        try:
            tr = self.tf_buffer.lookup_transform(self.planning_frame, self.ik_link_name, Time(), Duration(seconds=0))
        except Exception:  # noqa: BLE001
            return False
        now_s = self.get_clock().now().nanoseconds * 1e-9
        tf_s = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
        return (now_s - tf_s) <= self.tf_stale_threshold_s

    def destroy_node(self) -> bool:
        self.prepare_shutdown()
        with self._home_request_lock:
            self._home_goal_reserved = False
            self._home_preemption = None
        if hasattr(self, "_home_action_server"):
            self._home_action_server.destroy()
        if hasattr(self, "diffik"):
            self.diffik.close()
        return super().destroy_node()

    def prepare_shutdown(self) -> None:
        """Release a blocking Action execute callback before executor shutdown."""
        self._preempt_home("ROS_SHUTDOWN", "ArmReturnHome interrupted by ROS shutdown")
        if self.managed_teleop:
            self._managed_stop("ROS shutdown")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SO101PlacoServoNode()
    # All callbacks are short (<5 ms) and the solve tick must never overlap
    # itself.
    # One worker waits for the long-running ArmReturnHome action while the motion
    # callback group remains serialized on the other worker.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # ArmReturnHome execute_callback waits on a transaction event. Signal it
        # before executor.shutdown(wait_for_threads=True), otherwise shutdown can
        # wait forever for the callback that destroy_node would release later.
        node.prepare_shutdown()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
