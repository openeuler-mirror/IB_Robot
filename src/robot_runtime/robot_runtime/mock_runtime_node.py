"""Mock runtime: in-memory reference implementation of the runtime contract.

Implements every robot-runtime-contract and robot-motion-services requirement
without hardware or a planning framework, so the conformance suite has an
executable baseline and the core-only independence gate has a provider.

Kinematics are a deterministic planar serial chain (every joint rotates about
Z, equal link lengths): FK is closed-form, IK is cyclic coordinate descent
from the seed, orientation is projected to yaw (the mock's "reduced-DOF
strategy"), and the residual orientation error is reported.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Any

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import (
    ComputeFk,
    ComputeIk,
    GetRuntimeStatus,
    MoveToConfiguration,
    MoveToPose,
    SetRuntimeMode,
    StopRuntime,
)
from robot_runtime.contract import (
    CMD_VEL_TOPIC,
    COMPUTE_FK_SERVICE,
    COMPUTE_IK_SERVICE,
    GET_STATUS_SERVICE,
    IDLE_MODE,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_DEGRADED,
    MOVE_TO_JOINT_SERVICE,
    MOVE_TO_POSE_SERVICE,
    NAVIGATION_ACK_TOPIC,
    NAVIGATION_ENABLE_SERVICE,
    ODOM_TOPIC,
    OUTCOME_CANCELLED,
    OUTCOME_PLANNING_FAILED,
    OUTCOME_REJECTED,
    OUTCOME_SUCCEEDED,
    SET_MODE_SERVICE,
    STATUS_TOPIC,
    STOP_POLICIES,
    STOP_SERVICE,
    STOP_TORQUE_OFF,
)
from robot_runtime.interface_description import build_description
from robot_runtime.interface_monitor import InterfaceMonitor
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.peripherals import load_peripherals_file, merge_peripherals, perception_capabilities
from robot_runtime.profile import load_profile, validate_profile
from robot_runtime.state import RuntimeState
from robot_runtime.synthetic_perception import SyntheticStreams

LINK_LENGTH_M = 0.1
EE_LINK = "ee"
BASE_FRAME = "base"
MAX_JOINT_SPEED = 2.0  # rad/s, simulated actuator limit
GOAL_TOLERANCE = 0.01  # rad


def default_profile(base: bool = True) -> dict[str, Any]:
    joints = [f"joint_{i}" for i in range(1, 7)]
    capabilities: dict[str, Any] = {
        "joint.state": {"joint_count": len(joints), "rate_hz": 50.0},
        "joint.position_stream": {"joint_count": len(joints), "rate_hz": 50.0},
        "joint.trajectory": {"joint_count": len(joints)},
        "gripper.1d": {"count": 1},
        "motion.fk": {},
        "motion.ik": {"endpoints": [COMPUTE_IK_SERVICE], "default_orientation_tolerance": 0.5},
        "motion.move_to_joint": {},
        "motion.move_to_pose": {},
        "runtime.stop": {"cancel_bound_s": 0.5, "idle_bound_s": 1.0, "torque_off_bound_s": 2.0},
    }
    modes: dict[str, Any] = {
        "initial": IDLE_MODE,
        IDLE_MODE: {"controllers": [], "transitions": ["stream", "trajectory"]},
        "stream": {
            "controllers": ["arm_position_controller"],
            "allows_stream": True,
            "transitions": [IDLE_MODE, "trajectory"],
        },
        "trajectory": {
            "controllers": ["arm_trajectory_controller"],
            "allows_trajectory": True,
            "transitions": [IDLE_MODE, "stream"],
        },
    }
    if base:
        capabilities |= {
            "base.cmd_vel": {"max_vx": 0.5, "max_vy": 0.5, "max_wz": 1.0, "staleness_s": 0.3},
            "base.odom": {},
            "base.navigation_gate": {},
        }
        modes["base_navigation"] = {
            "controllers": ["base_velocity_controller"],
            "allows_base": True,
            "transitions": [IDLE_MODE],
        }
        modes[IDLE_MODE]["transitions"].append("base_navigation")
    return {
        "runtime": {"name": "mock_runtime", "version": "0.1.0"},
        "simulated": True,
        "modes": modes,
        "capabilities": capabilities,
        "joints": joints,
        "arm_joints": joints[:5],
        "command_channels": [
            {"channel": "arm_stream", "topic": "/arm_position_controller/commands", "joints": joints[:5]},
            {"channel": "gripper_stream", "topic": "/gripper_position_controller/commands", "joints": joints[5:]},
        ],
        "trajectory_actions": ["/arm_trajectory_controller/follow_joint_trajectory"],
        "joint_state_topic": "/joint_states",
        "control_rate": 50.0,
    }


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _quat_angle_to_yaw_only(q, yaw: float) -> float:
    """Angle between quaternion q and the pure-yaw quaternion for ``yaw``."""
    half = yaw / 2.0
    qy = (0.0, 0.0, math.sin(half), math.cos(half))
    dot = abs(q.x * qy[0] + q.y * qy[1] + q.z * qy[2] + q.w * qy[3])
    return 2.0 * math.acos(min(1.0, dot))


class MockRuntime(Node):
    def __init__(self):
        super().__init__("mock_runtime")
        self.declare_parameter("profile", "")
        self.declare_parameter("base_profile", True)
        self.declare_parameter("peripherals", "")
        path = str(self.get_parameter("profile").value)
        if path:
            self._profile = load_profile(path)
        else:
            self._profile = default_profile(bool(self.get_parameter("base_profile").value))
            validate_profile(self._profile, "<mock default>")
        peripherals_path = str(self.get_parameter("peripherals").value or "").strip()
        self.declare_parameter("instance_id", str(self._profile["runtime"].get("instance_id", "")))
        if self.get_parameter("instance_id").value:
            self._profile["runtime"]["instance_id"] = str(self.get_parameter("instance_id").value)
        fragment = load_peripherals_file(peripherals_path) if peripherals_path else {}
        peripherals = merge_peripherals(self._profile.get("peripherals"), fragment.get("peripherals"))
        capabilities = dict(self._profile["capabilities"])
        capabilities.update(perception_capabilities(peripherals))
        capabilities.setdefault("runtime.status", {})
        description = build_description(self._profile, peripherals, simulated=True)
        self._interface_monitor = InterfaceMonitor(self, description)
        self._synthetic_streams = SyntheticStreams(self, description)
        self._modes = ModeModel(ModeModelConfig.from_profile(self._profile))
        self._state = RuntimeState(
            str(self._profile["runtime"]["name"]),
            str(self._profile["runtime"]["version"]),
            capabilities,
            self._modes,
            on_change=self._publish_status,
            interface_description=description,
            interface_states=self._interface_monitor.states,
        )
        self._lock = threading.RLock()
        self._joints: list[str] = [str(j) for j in self._profile["joints"]]
        self._arm_joints: list[str] = [str(j) for j in self._profile.get("arm_joints", self._joints)]
        self._positions = dict.fromkeys(self._joints, 0.0)
        self._targets = dict.fromkeys(self._joints, 0.0)
        self._torque = True
        self._read_fail_until = 0.0
        self._rate = float(self._profile.get("control_rate", 50.0))
        self._motion_generation = 0  # bumped by stop to cancel in-flight moves
        self._active_goal = None

        cb = ReentrantCallbackGroup()
        self._status_pub = self.create_publisher(RuntimeStatus, STATUS_TOPIC, 10)
        self._joint_pub = self.create_publisher(JointState, str(self._profile["joint_state_topic"]), 10)
        self.create_service(SetRuntimeMode, SET_MODE_SERVICE, self._on_set_mode, callback_group=cb)
        self.create_service(GetRuntimeStatus, GET_STATUS_SERVICE, self._on_get_status, callback_group=cb)
        self.create_service(StopRuntime, STOP_SERVICE, self._on_stop, callback_group=cb)
        self.create_service(
            SetBool, "/mock_runtime/inject_read_failure", self._on_inject_read_failure, callback_group=cb
        )

        for entry in self._profile["command_channels"]:
            if str(entry.get("type", "float64_array")).lower() == "twist":
                continue  # The base channel is served by _on_cmd_vel below.
            joints = [str(j) for j in entry.get("joints", self._joints)]
            self.create_subscription(
                Float64MultiArray,
                str(entry["topic"]),
                lambda msg, ch=str(entry["channel"]), js=joints: self._on_stream(ch, js, msg),
                10,
                callback_group=cb,
            )
        self._traj_servers = [
            ActionServer(
                self,
                FollowJointTrajectory,
                str(name),
                execute_callback=self._on_trajectory,
                goal_callback=self._on_trajectory_goal,
                cancel_callback=lambda _h: CancelResponse.ACCEPT,
                callback_group=cb,
            )
            for name in self._profile["trajectory_actions"]
        ]

        if self._state.has_capability("motion.fk"):
            self.create_service(ComputeFk, COMPUTE_FK_SERVICE, self._on_fk, callback_group=cb)
        if self._state.has_capability("motion.ik"):
            for endpoint in self._state.capabilities["motion.ik"].get("endpoints", [COMPUTE_IK_SERVICE]):
                self.create_service(ComputeIk, str(endpoint), self._on_ik, callback_group=cb)
        if self._state.has_capability("motion.move_to_joint"):
            self.create_service(MoveToConfiguration, MOVE_TO_JOINT_SERVICE, self._on_move_to_joint, callback_group=cb)
        if self._state.has_capability("motion.move_to_pose"):
            self.create_service(MoveToPose, MOVE_TO_POSE_SERVICE, self._on_move_to_pose, callback_group=cb)

        self._base = self._state.has_capability("base.cmd_vel")
        if self._base:
            params = self._state.capabilities["base.cmd_vel"]
            self._base_limits = (float(params["max_vx"]), float(params["max_vy"]), float(params["max_wz"]))
            self._base_staleness = float(params.get("staleness_s", 0.3))
            self._base_pose = [0.0, 0.0, 0.0]
            self._base_cmd = [0.0, 0.0, 0.0]
            self._base_cmd_stamp = 0.0
            self._nav_enabled = False
            self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
            self._tf = TransformBroadcaster(self)
            self.create_subscription(Twist, CMD_VEL_TOPIC, self._on_cmd_vel, 10, callback_group=cb)
            self._ack_pub = self.create_publisher(Bool, NAVIGATION_ACK_TOPIC, 10)
            self.create_service(SetBool, NAVIGATION_ENABLE_SERVICE, self._on_navigation_enable, callback_group=cb)
            self.create_timer(0.2, lambda: self._ack_pub.publish(Bool(data=self._nav_enabled)), callback_group=cb)

        timer_cb = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / self._rate, self._tick, callback_group=timer_cb)
        self.create_timer(1.0, self._publish_status, callback_group=cb)
        self._state.set_lifecycle(LIFECYCLE_ACTIVE)

    # --- simulation tick ------------------------------------------------------------

    def _tick(self) -> None:
        dt = 1.0 / self._rate
        now = time.monotonic()
        with self._lock:
            if self._torque:
                step = MAX_JOINT_SPEED * dt
                for j in self._joints:
                    delta = self._targets[j] - self._positions[j]
                    self._positions[j] += max(-step, min(step, delta))
            if self._base:
                if self._base_cmd_stamp and now - self._base_cmd_stamp > self._base_staleness:
                    self._base_cmd = [0.0, 0.0, 0.0]
                vx, vy, wz = self._base_cmd
                theta = self._base_pose[2]
                self._base_pose[0] += (vx * math.cos(theta) - vy * math.sin(theta)) * dt
                self._base_pose[1] += (vx * math.sin(theta) + vy * math.cos(theta)) * dt
                self._base_pose[2] = _wrap(theta + wz * dt)
            if now < self._read_fail_until:
                if self._state.lifecycle == LIFECYCLE_ACTIVE:
                    self._state.add_fault("simulated read failure")
                    self._state.set_lifecycle(LIFECYCLE_DEGRADED)
                return
            if self._state.lifecycle == LIFECYCLE_DEGRADED and not self._state.stop_latched:
                self._state.clear_faults()
                self._state.set_lifecycle(LIFECYCLE_ACTIVE)
            positions = dict(self._positions)
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = list(self._joints)
        js.position = [positions[j] for j in self._joints]
        js.velocity = [0.0] * len(self._joints)
        self._joint_pub.publish(js)
        if self._base:
            self._publish_odom(stamp)

    def _publish_odom(self, stamp) -> None:
        x, y, theta = self._base_pose
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(theta / 2.0)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = self._base_cmd
        self._odom_pub.publish(odom)
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation = odom.pose.pose.orientation
        self._tf.sendTransform(tf)

    def _publish_status(self) -> None:
        self._status_pub.publish(self._state.to_msg(self.get_clock().now().to_msg()))

    # --- streaming and trajectory channels ------------------------------------------

    def _on_stream(self, channel: str, joints: list[str], msg: Float64MultiArray) -> None:
        if self._state.stop_latched or not self._modes.spec().allows_stream:
            self._modes.note_rejected(channel)
            return
        with self._lock:
            for j, value in zip(joints, msg.data, strict=False):
                self._targets[j] = float(value)

    def _on_trajectory_goal(self, request) -> GoalResponse:
        unknown = [j for j in request.trajectory.joint_names if j not in self._joints]
        if unknown:
            self.get_logger().warning(f"trajectory rejected: unknown joints {unknown}")
            return GoalResponse.REJECT
        if self._state.stop_latched or not self._modes.spec().allows_trajectory:
            self.get_logger().warning(
                f"trajectory rejected: mode {self._modes.mode!r} / latched={self._state.stop_latched}"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_trajectory(self, goal_handle):
        result = FollowJointTrajectory.Result()
        trajectory = goal_handle.request.trajectory
        names = list(trajectory.joint_names)
        generation = self._motion_generation
        with self._lock:
            self._active_goal = goal_handle
        try:
            elapsed = 0.0
            for point in trajectory.points:
                t_point = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                while elapsed < t_point:
                    if goal_handle.is_cancel_requested or generation != self._motion_generation:
                        with self._lock:
                            for j in self._joints:
                                self._targets[j] = self._positions[j]
                        goal_handle.canceled()
                        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                        result.error_string = "cancelled"
                        return result
                    time.sleep(1.0 / self._rate)
                    elapsed += 1.0 / self._rate
                with self._lock:
                    for j, p in zip(names, point.positions, strict=False):
                        self._targets[j] = float(p)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self._lock:
                    done = all(abs(self._positions[j] - self._targets[j]) <= GOAL_TOLERANCE for j in names)
                if done:
                    goal_handle.succeed()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    return result
                if goal_handle.is_cancel_requested or generation != self._motion_generation:
                    goal_handle.canceled()
                    result.error_string = "cancelled"
                    return result
                time.sleep(1.0 / self._rate)
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            return result
        finally:
            with self._lock:
                self._active_goal = None

    # --- status / mode / stop ---------------------------------------------------------

    def _on_get_status(self, _request, response):
        response.status = self._state.to_msg(self.get_clock().now().to_msg())
        return response

    def _on_set_mode(self, request, response):
        target = str(request.mode)
        epoch = self._state.stop_epoch
        if self._state.stop_latched and target != IDLE_MODE:
            response.success = False
            response.message = f"stop latched ({self._state.stop_policy}); request {IDLE_MODE!r} to clear it"
            response.valid_transitions = [IDLE_MODE]
            return response
        decision = self._modes.can_switch(target)
        if not decision.allowed:
            response.success = False
            response.message = decision.reason
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        self._modes.commit(target)
        self._state.set_active_controllers(self._modes.spec().controllers)
        with self._lock:
            unchanged = self._state.clear_stop_if_unchanged(epoch)
            if unchanged:
                self._torque = True
        if not unchanged:
            self._modes.commit(IDLE_MODE)
            self._state.set_active_controllers(self._modes.spec(IDLE_MODE).controllers)
            self._state.add_fault(f"mode {target!r} aborted: stop engaged during the controller switch")
            response.success = False
            response.message = (
                f"stop engaged while switching to {target!r}; runtime returned to {IDLE_MODE!r}, "
                f"request {IDLE_MODE!r} to clear the latch"
            )
            response.valid_transitions = [IDLE_MODE]
            self._publish_status()
            return response
        response.success = True
        response.message = f"mode {target!r} active"
        response.valid_transitions = sorted(self._modes.valid_transitions())
        self._publish_status()
        return response

    def _on_stop(self, request, response):
        policy = str(request.policy or STOP_POLICIES[0])
        if policy not in STOP_POLICIES:
            response.success = False
            response.message = f"unknown stop policy {policy!r}"
            response.cancel_latency_s = response.idle_latency_s = response.torque_off_latency_s = -1.0
            return response
        t0 = time.monotonic()
        with self._lock:
            self._state.engage_stop(policy)
            self._motion_generation += 1
            for j in self._joints:
                self._targets[j] = self._positions[j]
        response.cancel_latency_s = time.monotonic() - t0
        self._modes.commit(IDLE_MODE)
        self._state.set_active_controllers(self._modes.spec(IDLE_MODE).controllers)
        response.idle_latency_s = time.monotonic() - t0
        response.torque_off_latency_s = -1.0
        if policy == STOP_TORQUE_OFF:
            with self._lock:
                self._torque = False
            response.torque_off_latency_s = time.monotonic() - t0
        response.success = True
        response.message = f"stop ({policy}) engaged"
        self._publish_status()
        return response

    def _on_inject_read_failure(self, request, response):
        self._read_fail_until = time.monotonic() + (1.0 if request.data else 0.0)
        response.success = True
        response.message = "read failure injected" if request.data else "cleared"
        return response

    # --- motion services -----------------------------------------------------------------

    def _fk(self, angles: list[float]) -> tuple[float, float, float]:
        x = y = 0.0
        yaw = 0.0
        for angle in angles:
            yaw += angle
            x += LINK_LENGTH_M * math.cos(yaw)
            y += LINK_LENGTH_M * math.sin(yaw)
        return x, y, _wrap(yaw)

    def _joint_map(self, js: JointState) -> dict[str, float]:
        return {str(n): float(p) for n, p in zip(js.name, js.position, strict=False)}

    def _on_fk(self, request, response):
        provided = self._joint_map(request.joint_state)
        missing = [j for j in self._arm_joints if j not in provided]
        if missing:
            response.success = False
            response.message = f"joint state missing declared arm joints: {missing}"
            return response
        unknown = [link for link in request.link_names if link != EE_LINK]
        if unknown:
            response.success = False
            response.message = f"unknown link names: {unknown} (declared: [{EE_LINK}])"
            return response
        x, y, yaw = self._fk([provided[j] for j in self._arm_joints])
        for _link in request.link_names:
            pose = PoseStamped()
            pose.header.frame_id = BASE_FRAME
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            response.poses.append(pose)
        response.success = True
        return response

    def _solve_ik(
        self, target_x: float, target_y: float, seed: list[float], iterations: int = 300
    ) -> list[float] | None:
        n = len(seed)
        if math.hypot(target_x, target_y) > n * LINK_LENGTH_M + 1e-9:
            return None
        angles = list(seed)
        for _ in range(iterations):
            for i in range(n - 1, -1, -1):
                jx, jy, _ = self._fk(angles[:i])
                ex, ey, _ = self._fk(angles)
                a_e = math.atan2(ey - jy, ex - jx)
                a_t = math.atan2(target_y - jy, target_x - jx)
                angles[i] = _wrap(angles[i] + (a_t - a_e))
            ex, ey, _ = self._fk(angles)
            if math.hypot(ex - target_x, ey - target_y) < 1e-4:
                return angles
        ex, ey, _ = self._fk(angles)
        return angles if math.hypot(ex - target_x, ey - target_y) < 1e-3 else None

    def _on_ik(self, request, response):
        seed_map = self._joint_map(request.seed)
        seed = [seed_map.get(j, 0.0) for j in self._arm_joints]
        tolerance = float(request.orientation_tolerance) or float(
            self._state.capabilities["motion.ik"].get("default_orientation_tolerance", 0.5)
        )
        target = request.target.pose
        solution = self._solve_ik(target.position.x, target.position.y, seed)
        if solution is None:
            response.success = False
            response.code = "NO_IK_SOLUTION"
            response.message = "target position unreachable for the planar chain"
            return response
        # Reduced-DOF strategy: project the requested orientation to yaw and report the residual.
        _, _, yaw = self._fk(solution)
        error = _quat_angle_to_yaw_only(target.orientation, yaw)
        if error > tolerance:
            response.success = False
            response.code = "ORIENTATION_TOLERANCE"
            response.message = f"orientation error {error:.3f} rad exceeds tolerance {tolerance:.3f}"
            response.orientation_error = error
            return response
        response.success = True
        response.code = "SUCCESS"
        response.orientation_error = error
        response.solution.name = list(self._arm_joints)
        response.solution.position = solution
        return response

    def _execute_to(self, targets: dict[str, float], velocity_scaling: float) -> tuple[str, str, float]:
        if self._state.stop_latched:
            return OUTCOME_REJECTED, "runtime is STOPPED (stop latched)", 0.0
        if not self._modes.spec().allows_trajectory:
            return OUTCOME_REJECTED, f"mode {self._modes.mode!r} does not permit trajectory execution", 0.0
        generation = self._motion_generation
        t0 = time.monotonic()
        with self._lock:
            self._targets.update(targets)
        deadline = t0 + 10.0
        while time.monotonic() < deadline:
            if generation != self._motion_generation:
                return OUTCOME_CANCELLED, "cancelled by stop", time.monotonic() - t0
            with self._lock:
                done = all(abs(self._positions[j] - v) <= GOAL_TOLERANCE for j, v in targets.items())
            if done:
                return OUTCOME_SUCCEEDED, "reached", time.monotonic() - t0
            time.sleep(1.0 / self._rate)
        return "EXECUTION_FAILED", "goal tolerance not reached in time", time.monotonic() - t0

    def _on_move_to_joint(self, request, response):
        targets = {j: v for j, v in self._joint_map(request.target_joint_state).items() if j in self._joints}
        if not targets:
            response.success, response.outcome = False, OUTCOME_PLANNING_FAILED
            response.message = "target joint state names no declared joint"
            return response
        response.outcome, response.message, response.execution_time_s = self._execute_to(
            targets, request.velocity_scaling
        )
        response.success = response.outcome == OUTCOME_SUCCEEDED
        return response

    def _on_move_to_pose(self, request, response):
        with self._lock:
            seed = [self._positions[j] for j in self._arm_joints]
        solution = self._solve_ik(request.target_pose.position.x, request.target_pose.position.y, seed)
        if solution is None:
            response.success, response.outcome = False, OUTCOME_PLANNING_FAILED
            response.message = "target pose unreachable"
            return response
        targets = dict(zip(self._arm_joints, solution, strict=True))
        response.outcome, response.message, response.execution_time_s = self._execute_to(
            targets, request.velocity_scaling
        )
        response.success = response.outcome == OUTCOME_SUCCEEDED
        with self._lock:
            x, y, yaw = self._fk([self._positions[j] for j in self._arm_joints])
        response.position_error_m = math.hypot(x - request.target_pose.position.x, y - request.target_pose.position.y)
        response.orientation_error_rad = _quat_angle_to_yaw_only(request.target_pose.orientation, yaw)
        return response

    # --- base ------------------------------------------------------------------------------

    def _on_cmd_vel(self, msg: Twist) -> None:
        if self._state.stop_latched or not self._modes.spec().allows_base or not self._nav_enabled:
            self._modes.note_rejected("base_cmd_vel")
            return
        mx, my, mw = self._base_limits
        with self._lock:
            self._base_cmd = [
                max(-mx, min(mx, msg.linear.x)),
                max(-my, min(my, msg.linear.y)),
                max(-mw, min(mw, msg.angular.z)),
            ]
            self._base_cmd_stamp = time.monotonic()

    def _on_navigation_enable(self, request, response):
        with self._lock:
            self._base_cmd = [0.0, 0.0, 0.0]
            self._base_cmd_stamp = 0.0
            self._nav_enabled = bool(request.data)
        self._ack_pub.publish(Bool(data=self._nav_enabled))
        response.success = True
        response.message = "navigation enabled" if self._nav_enabled else "navigation disabled"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MockRuntime()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
