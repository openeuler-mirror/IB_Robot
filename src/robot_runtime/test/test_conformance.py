"""Conformance suite: the executable definition of the runtime contract.

Covers the robot-runtime-contract, robot-motion-services, and
robot-runtime-conformance-suite specs. Scoped by the capabilities the
target declares in RuntimeStatus; a runtime is conformant when it passes
every test for its declared set.

Targets:
- default: the mock runtime, spun in-process (headless CI baseline).
- ``CONFORMANCE_LAUNCH="<command>"``: an external command that starts the
  runtime under test through its own launch entry (e.g.
  ``ros2 launch so101_robot runtime.launch.py profile:=... simulated:=true``).
  ``CONFORMANCE_PROFILE`` must then point at that runtime's profile so the
  suite knows its joints, command channels, and trajectory endpoints.

Every assertion message names the requirement under test and the observed
deviation so failures are diagnosable without reading this file.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import threading
import time

import pytest
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint

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
from robot_runtime import contract as C
from robot_runtime.profile import load_profile

TARGET_LAUNCH = os.environ.get("CONFORMANCE_LAUNCH", "").strip()
TARGET_PROFILE = os.environ.get("CONFORMANCE_PROFILE", "").strip()
STARTUP_TIMEOUT_S = float(os.environ.get("CONFORMANCE_STARTUP_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Harness(Node):
    def __init__(self, profile: dict):
        super().__init__("runtime_conformance_harness")
        self.profile = profile
        self.joints = [str(j) for j in profile["joints"]]
        self.arm_joints = [str(j) for j in profile.get("arm_joints", self.joints)]
        self.joint_states: list[tuple[float, JointState]] = []
        self.statuses: list[RuntimeStatus] = []
        self.odom: list[Odometry] = []
        self.create_subscription(JointState, str(profile["joint_state_topic"]), self._on_js, 50)
        self.create_subscription(RuntimeStatus, C.STATUS_TOPIC, self.statuses.append, 50)
        self.create_subscription(Odometry, C.ODOM_TOPIC, self.odom.append, 50)
        self._svc = {
            "set_mode": self.create_client(SetRuntimeMode, C.SET_MODE_SERVICE),
            "get_status": self.create_client(GetRuntimeStatus, C.GET_STATUS_SERVICE),
            "stop": self.create_client(StopRuntime, C.STOP_SERVICE),
            "fk": self.create_client(ComputeFk, C.COMPUTE_FK_SERVICE),
            "ik": self.create_client(ComputeIk, C.COMPUTE_IK_SERVICE),
            "move_joint": self.create_client(MoveToConfiguration, C.MOVE_TO_JOINT_SERVICE),
            "move_pose": self.create_client(MoveToPose, C.MOVE_TO_POSE_SERVICE),
            "nav_enable": self.create_client(SetBool, C.NAVIGATION_ENABLE_SERVICE),
            "inject_read_failure": self.create_client(SetBool, "/mock_runtime/inject_read_failure"),
        }
        self._cmd_pubs = {
            str(e["channel"]): (
                self.create_publisher(Float64MultiArray, str(e["topic"]), 10),
                [str(j) for j in e.get("joints", self.joints)],
            )
            for e in profile["command_channels"]
            if str(e.get("type", "float64_array")).lower() != "twist"
        }
        self._cmd_vel_pub = self.create_publisher(Twist, C.CMD_VEL_TOPIC, 10)
        self._traj_clients = [
            ActionClient(self, FollowJointTrajectory, str(name)) for name in profile["trajectory_actions"]
        ]

    def _on_js(self, msg):
        self.joint_states.append((time.monotonic(), msg))

    # --- spinning ---------------------------------------------------------------

    def spin_for(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    def spin_until(self, predicate, timeout: float, what: str):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.02)
        raise AssertionError(f"timed out after {timeout}s waiting for {what}")

    def call(self, name: str, request, timeout: float = 10.0):
        client = self._svc[name]
        assert client.wait_for_service(timeout_sec=5.0), f"service {client.srv_name} unavailable"
        future = client.call_async(request)
        self.spin_until(future.done, timeout, f"{client.srv_name} response")
        return future.result()

    def call_async(self, name: str, request):
        client = self._svc[name]
        assert client.wait_for_service(timeout_sec=5.0), f"service {client.srv_name} unavailable"
        return client.call_async(request)

    def has_service(self, name: str) -> bool:
        return self._svc[name].wait_for_service(timeout_sec=0.5)

    # --- contract helpers -------------------------------------------------------------

    def status(self) -> RuntimeStatus:
        return self.call("get_status", GetRuntimeStatus.Request()).status

    def capabilities(self) -> set[str]:
        return set(self.status().capabilities)

    def params(self, capability: str) -> dict:
        return json.loads(self.status().capabilities_json).get(capability, {})

    def set_mode(self, mode: str):
        request = SetRuntimeMode.Request()
        request.mode = mode
        return self.call("set_mode", request)

    def stop(self, policy: str):
        request = StopRuntime.Request()
        request.policy = policy
        return self.call("stop", request)

    def reset_to_idle(self):
        response = self.set_mode(C.IDLE_MODE)
        assert response.success, f"could not reset to idle: {response.message}"

    def positions(self) -> dict[str, float]:
        self.joint_states.clear()
        self.spin_until(lambda: len(self.joint_states) >= 2, 5.0, "fresh joint state")
        msg = self.joint_states[-1][1]
        return dict(zip(msg.name, msg.position, strict=False))

    def stream(self, channel: str, values: list[float]):
        pub, _joints = self._cmd_pubs[channel]
        pub.publish(Float64MultiArray(data=[float(v) for v in values]))

    def stream_channel(self) -> tuple[str, list[str]]:
        channel = next(iter(self._cmd_pubs))
        return channel, self._cmd_pubs[channel][1]

    def send_trajectory(self, joint_names: list[str], positions: list[float], duration_s: float):
        client = self._traj_clients[0]
        assert client.wait_for_server(timeout_sec=5.0), "trajectory action server unavailable"
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start.sec = int(duration_s)
        point.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1e9)
        goal.trajectory.points = [point]
        future = client.send_goal_async(goal)
        self.spin_until(future.done, 5.0, "trajectory goal response")
        return future.result()

    def await_result(self, goal_handle, timeout: float = 15.0):
        future = goal_handle.get_result_async()
        self.spin_until(future.done, timeout, "trajectory result")
        return future.result()

    def cmd_vel(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self._cmd_vel_pub.publish(msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def profile() -> dict:
    if TARGET_LAUNCH:
        assert TARGET_PROFILE, "CONFORMANCE_PROFILE is required when CONFORMANCE_LAUNCH is set"
        return load_profile(TARGET_PROFILE)
    from robot_runtime.mock_runtime_node import default_profile

    return default_profile(base=True)


@pytest.fixture(scope="session")
def target(profile):
    """Start the runtime under test: mock in-process, or the external launch command."""
    rclpy.init()
    process = None
    executor = None
    thread = None
    node = None
    if TARGET_LAUNCH:
        process = subprocess.Popen(shlex.split(TARGET_LAUNCH), start_new_session=True)
    else:
        from robot_runtime.mock_runtime_node import MockRuntime

        node = MockRuntime()
        executor = MultiThreadedExecutor(num_threads=8)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
    yield node
    if executor is not None:
        executor.shutdown(timeout_sec=2.0)
    if process is not None:
        import signal

        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    rclpy.shutdown()


@pytest.fixture(scope="session")
def harness(target, profile) -> Harness:
    node = Harness(profile)
    node.spin_until(
        lambda: any(s.lifecycle == C.LIFECYCLE_ACTIVE for s in node.statuses),
        STARTUP_TIMEOUT_S,
        f"runtime lifecycle ACTIVE on {C.STATUS_TOPIC}",
    )
    yield node
    node.destroy_node()


@pytest.fixture(autouse=True)
def _clean_state(harness: Harness):
    """Every test starts from idle with the stop latch cleared."""
    harness.reset_to_idle()
    harness.joint_states.clear()
    harness.statuses.clear()
    yield


def arm_targets(harness: Harness, before: dict[str, float], deltas: dict[str, float]) -> tuple[list[str], list[float]]:
    """Complete arm joint target (trajectory controllers and motion servers may reject partial goals)."""
    names = list(harness.arm_joints)
    return names, [before[j] + deltas.get(j, 0.0) for j in names]


def requires(harness: Harness, *capabilities: str):
    missing = [c for c in capabilities if c not in harness.capabilities()]
    if missing:
        pytest.skip(f"runtime does not declare {missing}")


def _mode_named(harness: Harness, flag: str, default: str) -> str:
    """A declared mode carrying ``flag``, or the conventional name.

    Runtimes name their modes; only the semantics are contractual. A runtime
    that declares no mode with the requested semantics still returns the
    conventional name, so the caller's own ``requires`` gate decides whether
    the test runs at all.
    """
    for name, spec in (harness.profile.get("modes") or {}).items():
        if name != "initial" and isinstance(spec, dict) and spec.get(flag):
            return str(name)
    return default


def _trajectory_mode(harness: Harness) -> str:
    return _mode_named(harness, "allows_trajectory", "trajectory")


def _stream_mode(harness: Harness) -> str:
    return _mode_named(harness, "allows_stream", "stream")


def _any_other_mode(harness: Harness) -> str:
    """Any declared mode other than idle, for tests about switching itself."""
    for name in harness.status().declared_modes:
        if name != C.IDLE_MODE:
            return str(name)
    raise AssertionError("runtime declares no mode other than idle")


# ---------------------------------------------------------------------------
# Joint state publication
# ---------------------------------------------------------------------------


def test_joint_state_rate_and_naming(harness: Harness):
    requires(harness, "joint.state")
    rate = float(harness.params("joint.state").get("rate_hz", 0.0))
    assert rate > 0, "joint.state must declare rate_hz"
    harness.joint_states.clear()
    harness.spin_for(2.0)
    count = len(harness.joint_states)
    assert count >= 0.9 * rate * 2.0, (
        f"[Joint state publication / rate] observed {count} messages in 2 s, expected >= 90% of {rate} Hz ({0.9 * rate * 2.0:.0f})"
    )
    for _, msg in harness.joint_states:
        missing = set(harness.joints) - set(msg.name)
        assert not missing, f"[Joint state publication / naming] message missing declared joints {sorted(missing)}"


def test_read_failure_is_not_masked(harness: Harness):
    requires(harness, "joint.state")
    if not harness.has_service("inject_read_failure"):
        pytest.skip("target exposes no read-failure injection")
    harness.call("inject_read_failure", SetBool.Request(data=True))
    harness.joint_states.clear()
    harness.spin_for(0.6)
    stale = [t for t, _ in harness.joint_states if t > time.monotonic() - 0.4]
    assert not stale, "[Joint state publication / read failure] joint state published as fresh during a read failure"
    assert harness.status().lifecycle == C.LIFECYCLE_DEGRADED, (
        "[Joint state publication / read failure] lifecycle not DEGRADED"
    )
    harness.call("inject_read_failure", SetBool.Request(data=False))
    harness.spin_until(lambda: harness.status().lifecycle == C.LIFECYCLE_ACTIVE, 5.0, "recovery to ACTIVE")


# ---------------------------------------------------------------------------
# Streaming channel
# ---------------------------------------------------------------------------


def test_stream_commands_honored_in_stream_mode(harness: Harness):
    requires(harness, "joint.position_stream")
    channel, joints = harness.stream_channel()
    assert harness.set_mode(_stream_mode(harness)).success
    before = harness.positions()
    target = [before[j] + 0.3 for j in joints]
    for _ in range(5):
        harness.stream(channel, target)
        harness.spin_for(0.1)
    harness.spin_for(0.5)
    after = harness.positions()
    moved = [j for j in joints if abs(after[j] - before[j]) > 0.1]
    assert moved, (
        f"[Streaming command channel / honored] joints did not move toward the command in stream mode: {after}"
    )


def test_stream_commands_rejected_outside_stream_mode(harness: Harness):
    requires(harness, "joint.position_stream")
    channel, joints = harness.stream_channel()
    before_status = harness.status()
    counts = dict(zip(before_status.rejected_channels, before_status.rejected_counts, strict=False))
    before = harness.positions()
    for _ in range(5):
        harness.stream(channel, [before[j] + 0.5 for j in joints])
        harness.spin_for(0.05)
    harness.spin_for(0.4)
    after = harness.positions()
    assert all(abs(after[j] - before[j]) < 0.02 for j in joints), (
        "[Streaming command channel / rejected outside stream mode] joints moved in idle mode"
    )
    status = harness.status()
    new_counts = dict(zip(status.rejected_channels, status.rejected_counts, strict=False))
    assert new_counts.get(channel, 0) > counts.get(channel, 0), (
        f"[Streaming command channel / rejected outside stream mode] rejected counter for {channel!r} did not increase: {new_counts}"
    )


def test_stream_stops_joints_hold(harness: Harness):
    requires(harness, "joint.position_stream")
    channel, joints = harness.stream_channel()
    assert harness.set_mode(_stream_mode(harness)).success
    before = harness.positions()
    target = [before[j] + 0.2 for j in joints]
    for _ in range(10):
        harness.stream(channel, target)
        harness.spin_for(0.05)
    harness.spin_for(0.5)
    held = harness.positions()
    harness.spin_for(1.0)
    later = harness.positions()
    assert all(abs(later[j] - held[j]) < 0.02 for j in joints), (
        f"[Streaming command channel / hold] joints drifted after streaming stopped: {held} -> {later}"
    )


# ---------------------------------------------------------------------------
# Trajectory channel
# ---------------------------------------------------------------------------


def test_trajectory_reaches_goal_tolerance(harness: Harness):
    requires(harness, "joint.trajectory")
    assert harness.set_mode(_trajectory_mode(harness)).success
    before = harness.positions()
    moved = harness.arm_joints[:2]
    joints, targets = arm_targets(harness, before, dict.fromkeys(moved, 0.25))
    handle = harness.send_trajectory(joints, targets, 0.5)
    assert handle.accepted, "[Trajectory execution / reaches goal] valid goal was rejected"
    result = harness.await_result(handle)
    assert result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL, (
        f"[Trajectory execution / reaches goal] error_code {result.result.error_code}: {result.result.error_string}"
    )
    after = harness.positions()
    for j, t in zip(joints, targets, strict=True):
        assert abs(after[j] - t) < 0.05, (
            f"[Trajectory execution / reaches goal] joint {j} at {after[j]:.3f}, goal {t:.3f}"
        )
    assert any(abs(after[j] - before[j]) > 0.1 for j in moved), (
        "[Trajectory execution / reaches goal] joints did not move"
    )


def test_trajectory_unknown_joints_rejected(harness: Harness):
    requires(harness, "joint.trajectory")
    assert harness.set_mode(_trajectory_mode(harness)).success
    handle = harness.send_trajectory(["definitely_not_a_joint"], [0.1], 0.5)
    assert not handle.accepted, "[Trajectory execution / unknown joints] goal naming an undeclared joint was accepted"


def test_trajectory_cancel_holds_position(harness: Harness):
    requires(harness, "joint.trajectory")
    assert harness.set_mode(_trajectory_mode(harness)).success
    before = harness.positions()
    moved = harness.arm_joints[0]
    joints, targets = arm_targets(harness, before, {moved: 1.5})
    handle = harness.send_trajectory(joints, targets, 2.0)
    assert handle.accepted
    harness.spin_for(0.4)
    cancel = handle.cancel_goal_async()
    harness.spin_until(cancel.done, 5.0, "cancel response")
    harness.spin_for(0.3)
    at_cancel = harness.positions()
    harness.spin_for(0.8)
    later = harness.positions()
    assert abs(later[moved] - at_cancel[moved]) < 0.05, (
        f"[Trajectory execution / cancel holds] joint moved after cancel: {at_cancel[moved]:.3f} -> {later[moved]:.3f}"
    )


# ---------------------------------------------------------------------------
# Status, mode, stop
# ---------------------------------------------------------------------------


def test_status_published_on_mode_change_and_query_parity(harness: Harness):
    harness.statuses.clear()
    # Any declared mode exercises publication-on-change; this is not a
    # trajectory test, so a runtime without a trajectory mode still runs it.
    target = _any_other_mode(harness)
    assert harness.set_mode(target).success
    harness.spin_until(
        lambda: any(s.active_mode == target for s in harness.statuses),
        1.5,
        "[Runtime status / on change] status carrying the new mode within one period",
    )
    queried = harness.status()
    assert queried.active_mode == target, "[Runtime status / query parity] queried mode differs"
    assert queried.lifecycle == C.LIFECYCLE_ACTIVE
    assert queried.runtime_name and queried.runtime_version, "[Runtime status] identity fields empty"
    assert set(queried.declared_modes) >= {C.IDLE_MODE}, "[Runtime status] declared modes missing idle"


def test_mode_invalid_transition_lists_alternatives(harness: Harness):
    response = harness.set_mode("no_such_mode")
    assert not response.success, "[Mode service / invalid] undeclared mode accepted"
    assert response.valid_transitions, "[Mode service / invalid] response lists no valid transitions"
    assert harness.status().active_mode == C.IDLE_MODE, "[Mode service / invalid] mode changed on rejected request"


@pytest.mark.parametrize("policy", [C.STOP_HOLD, C.STOP_TORQUE_OFF])
def test_stop_guarantees_within_declared_bounds(harness: Harness, policy: str):
    requires(harness, "runtime.stop", "joint.trajectory")
    bounds = harness.params("runtime.stop")
    assert harness.set_mode(_trajectory_mode(harness)).success
    before = harness.positions()
    joint = harness.arm_joints[0]
    joints, targets = arm_targets(harness, before, {joint: 1.5})
    handle = harness.send_trajectory(joints, targets, 3.0)
    assert handle.accepted
    result_future = handle.get_result_async()
    harness.spin_for(0.3)
    response = harness.stop(policy)
    assert response.success, f"[Stop service / {policy}] {response.message}"
    harness.spin_until(result_future.done, 5.0, "trajectory result after stop")
    assert result_future.result().status in (4, 5, 6), (
        f"[Stop service / {policy}] trajectory not terminated (status {result_future.result().status})"
    )
    for key, measured in (("cancel_bound_s", response.cancel_latency_s), ("idle_bound_s", response.idle_latency_s)):
        assert measured >= 0, f"[Stop service / {policy}] {key} not measured"
        assert measured <= float(bounds[key]), (
            f"[Stop service / {policy}] {key.removesuffix('_bound_s')} took {measured:.3f}s, declared bound {bounds[key]}s"
        )
    if policy == C.STOP_TORQUE_OFF:
        assert 0 <= response.torque_off_latency_s <= float(bounds["torque_off_bound_s"]), (
            f"[Stop service / TORQUE_OFF] torque-off took {response.torque_off_latency_s:.3f}s, bound {bounds['torque_off_bound_s']}s"
        )
    status = harness.status()
    assert status.lifecycle == C.LIFECYCLE_STOPPED and status.stop_latched and status.stop_policy == policy, (
        f"[Stop service / {policy}] status lifecycle={status.lifecycle} latched={status.stop_latched} policy={status.stop_policy!r}"
    )
    assert status.active_mode == C.IDLE_MODE, f"[Stop service / {policy}] mode is {status.active_mode!r}, expected idle"
    held = harness.positions()
    harness.spin_for(0.5)
    later = harness.positions()
    assert abs(later[joint] - held[joint]) < 0.05, f"[Stop service / {policy}] joint moved after stop"


def test_stop_latches_until_cleared(harness: Harness):
    requires(harness, "runtime.stop")
    assert harness.stop(C.STOP_HOLD).success
    rejected = harness.set_mode(_any_other_mode(harness))
    assert not rejected.success and "latch" in rejected.message.lower(), (
        f"[Stop service / latch] mode request accepted while stopped: {rejected.message}"
    )
    assert rejected.valid_transitions == [C.IDLE_MODE]
    cleared = harness.set_mode(C.IDLE_MODE)
    assert cleared.success, f"[Stop service / latch] idle request did not clear the latch: {cleared.message}"
    status = harness.status()
    assert not status.stop_latched and status.lifecycle == C.LIFECYCLE_ACTIVE


@pytest.mark.parametrize("policy", [C.STOP_HOLD, C.STOP_TORQUE_OFF])
def test_stop_fences_in_flight_mode_switch(harness: Harness, target, monkeypatch, policy):
    if target is None:
        pytest.skip("requires in-process mode-switch pause hook; facade has a slow controller_manager regression")
    entered = threading.Event()
    release = threading.Event()
    commit = target._modes.commit

    def slow_commit(mode):
        if mode == _stream_mode(harness):
            entered.set()
            assert release.wait(5.0), "test never released the mode switch"
        commit(mode)

    monkeypatch.setattr(target._modes, "commit", slow_commit)
    pending = harness.call_async("set_mode", SetRuntimeMode.Request(mode=_stream_mode(harness)))
    try:
        assert entered.wait(3.0), "mode switch never reached the pause hook"
        response = harness.stop(policy)
        assert response.success, response.message
        assert not release.is_set(), "stop waited for the in-flight mode switch"
    finally:
        release.set()
    harness.spin_until(pending.done, 5.0, "fenced mode switch response")
    response = pending.result()
    assert not response.success and response.valid_transitions == [C.IDLE_MODE]
    status = harness.status()
    assert status.stop_latched and status.lifecycle == C.LIFECYCLE_STOPPED
    assert status.active_mode == C.IDLE_MODE and not status.active_controllers
    assert any("stop engaged during the controller switch" in fault for fault in status.faults)
    if policy == C.STOP_TORQUE_OFF:
        assert not target._torque


# ---------------------------------------------------------------------------
# Motion services
# ---------------------------------------------------------------------------


def _pose(x: float, y: float, z: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "base"
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
    cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
    pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = (
        cy * cr,
        cy * sr,
        sy * sr,
        sy * cr,
    )
    return pose


def _joint_state(names: list[str], positions: list[float]) -> JointState:
    js = JointState()
    js.name = list(names)
    js.position = [float(p) for p in positions]
    return js


def _reachable_pose(harness: Harness) -> PoseStamped:
    """A pose the runtime's FK says is reachable: FK of a mild configuration."""
    request = ComputeFk.Request()
    request.joint_state = _joint_state(harness.arm_joints, [0.2] * len(harness.arm_joints))
    request.link_names = [harness.params("motion.fk").get("ee_link", "ee")]
    response = harness.call("fk", request)
    assert response.success, response.message
    return response.poses[0]


def test_fk_pose_for_declared_link(harness: Harness):
    requires(harness, "motion.fk")
    request = ComputeFk.Request()
    request.joint_state = _joint_state(harness.arm_joints, [0.0] * len(harness.arm_joints))
    request.link_names = [harness.params("motion.fk").get("ee_link", "ee")]
    response = harness.call("fk", request)
    assert response.success and len(response.poses) == 1, f"[FK / declared link] {response.message}"
    assert response.poses[0].header.frame_id, "[FK / declared link] pose has no frame"


def test_fk_incomplete_joint_state_rejected(harness: Harness):
    requires(harness, "motion.fk")
    request = ComputeFk.Request()
    request.joint_state = _joint_state(harness.arm_joints[:1], [0.0])
    request.link_names = [harness.params("motion.fk").get("ee_link", "ee")]
    response = harness.call("fk", request)
    assert not response.success and harness.arm_joints[1] in response.message, (
        f"[FK / incomplete joint state] not rejected by name: {response.message}"
    )


def test_ik_reachable_pose_returns_solution(harness: Harness):
    requires(harness, "motion.ik", "motion.fk")
    request = ComputeIk.Request()
    request.target = _reachable_pose(harness)
    request.seed = _joint_state(harness.arm_joints, [0.0] * len(harness.arm_joints))
    request.orientation_tolerance = 0.5
    response = harness.call("ik", request)
    assert response.success, f"[IK / reachable] code={response.code}: {response.message}"
    assert set(response.solution.name) >= set(harness.arm_joints), "[IK / reachable] solution missing arm joints"
    assert response.orientation_error <= 0.5, (
        f"[IK / reachable] orientation error {response.orientation_error:.3f} > tolerance"
    )


def test_ik_unreachable_pose_returns_code(harness: Harness):
    requires(harness, "motion.ik")
    request = ComputeIk.Request()
    request.target = _pose(50.0, 50.0)
    request.seed = _joint_state(harness.arm_joints, [0.0] * len(harness.arm_joints))
    response = harness.call("ik", request)
    assert not response.success and response.code, (
        f"[IK / unreachable] success={response.success} code={response.code!r}"
    )
    assert not response.solution.name, "[IK / unreachable] joint state returned on failure"


def test_ik_reduced_dof_orientation_handled_by_runtime(harness: Harness):
    requires(harness, "motion.ik", "motion.fk")
    reachable = _reachable_pose(harness)
    tilted = _pose(reachable.pose.position.x, reachable.pose.position.y, reachable.pose.position.z, yaw=0.0, roll=0.3)
    request = ComputeIk.Request()
    request.target = tilted
    request.seed = _joint_state(harness.arm_joints, [0.0] * len(harness.arm_joints))
    request.orientation_tolerance = 1.0
    response = harness.call("ik", request)
    assert response.success, (
        f"[IK / reduced-DOF] runtime did not apply its own orientation strategy: {response.message}"
    )
    assert response.orientation_error > 0.0, "[IK / reduced-DOF] residual orientation error not reported"


def test_ik_endpoints_declared_in_status_answer(harness: Harness):
    requires(harness, "motion.ik")
    endpoints = harness.params("motion.ik").get("endpoints", [])
    assert endpoints, "[IK endpoints] motion.ik declares no endpoints"
    for endpoint in endpoints:
        client = harness.create_client(ComputeIk, str(endpoint))
        assert client.wait_for_service(timeout_sec=5.0), f"[IK endpoints] declared endpoint {endpoint} unavailable"
        harness.destroy_client(client)


def test_ik_available_in_idle(harness: Harness):
    requires(harness, "motion.ik")
    assert harness.status().active_mode == C.IDLE_MODE
    request = ComputeIk.Request()
    request.target = _pose(50.0, 50.0)
    request.seed = _joint_state(harness.arm_joints, [0.0] * len(harness.arm_joints))
    response = harness.call("ik", request)
    assert response.code, "[Motion gating / IK in idle] IK not answered in idle"


def test_move_to_joint_reaches_configuration(harness: Harness):
    requires(harness, "motion.move_to_joint")
    assert harness.set_mode(_trajectory_mode(harness)).success
    before = harness.positions()
    moved = harness.arm_joints[:2]
    joints, targets = arm_targets(harness, before, dict.fromkeys(moved, 0.2))
    request = MoveToConfiguration.Request()
    request.target_joint_state = _joint_state(joints, targets)
    response = harness.call("move_joint", request, timeout=20.0)
    assert response.success and response.outcome == C.OUTCOME_SUCCEEDED, (
        f"[Move-to-joint / reaches] outcome={response.outcome}: {response.message}"
    )
    assert response.execution_time_s > 0
    after = harness.positions()
    for j in moved:
        assert abs(after[j] - (before[j] + 0.2)) < 0.05, f"[Move-to-joint / reaches] joint {j} at {after[j]:.3f}"


def test_move_rejected_while_stopped(harness: Harness):
    requires(harness, "motion.move_to_joint", "runtime.stop")
    assert harness.stop(C.STOP_HOLD).success
    request = MoveToConfiguration.Request()
    request.target_joint_state = _joint_state(harness.arm_joints, [0.3] * len(harness.arm_joints))
    response = harness.call("move_joint", request)
    assert not response.success and response.outcome == C.OUTCOME_REJECTED, (
        f"[Motion gating / stopped] outcome={response.outcome}: {response.message}"
    )
    assert "stop" in response.message.lower()


def test_stop_cancels_in_flight_move(harness: Harness):
    requires(harness, "motion.move_to_joint", "runtime.stop")
    assert harness.set_mode(_trajectory_mode(harness)).success
    before = harness.positions()
    joints, targets = arm_targets(harness, before, {harness.arm_joints[0]: 1.5})
    request = MoveToConfiguration.Request()
    request.target_joint_state = _joint_state(joints, targets)
    move = harness.call_async("move_joint", request)
    harness.spin_for(0.3)
    assert harness.stop(C.STOP_HOLD).success
    harness.spin_until(move.done, 10.0, "move response after stop")
    assert move.result().outcome == C.OUTCOME_CANCELLED, (
        f"[Move-to-joint / cancelled by stop] outcome={move.result().outcome}"
    )


def test_move_to_pose_reaches_within_reported_error(harness: Harness):
    requires(harness, "motion.move_to_pose", "motion.fk")
    assert harness.set_mode(_trajectory_mode(harness)).success
    request = MoveToPose.Request()
    request.target_pose = _reachable_pose(harness).pose
    response = harness.call("move_pose", request, timeout=20.0)
    assert response.success and response.outcome == C.OUTCOME_SUCCEEDED, (
        f"[Move-to-pose / reaches] {response.outcome}: {response.message}"
    )
    assert response.position_error_m < 0.02, (
        f"[Move-to-pose / reaches] position error {response.position_error_m:.4f} m"
    )


# ---------------------------------------------------------------------------
# Base channels
# ---------------------------------------------------------------------------


def _base_mode(harness: Harness) -> str:
    return _mode_named(harness, "allows_base", "base_navigation")


def test_base_velocity_honored_and_odometry_integrates(harness: Harness):
    requires(harness, "base.cmd_vel", "base.odom")
    assert harness.set_mode(_base_mode(harness)).success
    if "base.navigation_gate" in harness.capabilities():
        assert harness.call("nav_enable", SetBool.Request(data=True)).success
    harness.odom.clear()
    harness.spin_for(0.3)
    x0 = harness.odom[-1].pose.pose.position.x
    for _ in range(10):
        harness.cmd_vel(0.2, 0.0, 0.0)
        harness.spin_for(0.1)
    x1 = harness.odom[-1].pose.pose.position.x
    assert x1 - x0 > 0.1, f"[Base velocity / honored] odometry advanced {x1 - x0:.3f} m under 0.2 m/s for 1 s"


def test_base_stale_velocity_stops(harness: Harness):
    requires(harness, "base.cmd_vel", "base.odom")
    staleness = float(harness.params("base.cmd_vel").get("staleness_s", 0.3))
    assert harness.set_mode(_base_mode(harness)).success
    if "base.navigation_gate" in harness.capabilities():
        assert harness.call("nav_enable", SetBool.Request(data=True)).success
    for _ in range(5):
        harness.cmd_vel(0.2, 0.0, 0.0)
        harness.spin_for(0.1)
    harness.spin_for(staleness + 0.3)
    x0 = harness.odom[-1].pose.pose.position.x
    harness.spin_for(0.5)
    x1 = harness.odom[-1].pose.pose.position.x
    assert abs(x1 - x0) < 0.01, f"[Base velocity / stale] base kept moving {x1 - x0:.3f} m after commands ceased"


def test_navigation_gate_enable_clears_and_disable_rejects(harness: Harness):
    requires(harness, "base.cmd_vel", "base.navigation_gate")
    assert harness.set_mode(_base_mode(harness)).success
    acks: list[bool] = []
    sub = harness.create_subscription(
        __import__("std_msgs.msg", fromlist=["Bool"]).Bool, C.NAVIGATION_ACK_TOPIC, lambda m: acks.append(m.data), 10
    )
    assert harness.call("nav_enable", SetBool.Request(data=True)).success
    harness.spin_until(lambda: True in acks, 2.0, "[Navigation gating / enable] acknowledgment heartbeat")
    assert harness.call("nav_enable", SetBool.Request(data=False)).success
    before = harness.status()
    counts = dict(zip(before.rejected_channels, before.rejected_counts, strict=False))
    for _ in range(5):
        harness.cmd_vel(0.2, 0.0, 0.0)
        harness.spin_for(0.05)
    after = harness.status()
    new_counts = dict(zip(after.rejected_channels, after.rejected_counts, strict=False))
    assert sum(new_counts.values()) > sum(counts.values()), (
        "[Navigation gating / disable] velocity commands not rejected and counted"
    )
    harness.destroy_subscription(sub)


# ---------------------------------------------------------------------------
# Perception surface (D12): the runtime exposes its sensor topics in both
# real and simulated transport; the conformance suite only asserts what the
# runtime declares through RuntimeStatus.
# ---------------------------------------------------------------------------


def _wait_for_message(harness: Harness, message_type, topic: str, timeout_s: float = 10.0):
    """Wait for one message on a declared perception topic; returns it or None."""
    received = {}

    def _on_message(msg):
        received.setdefault("msg", msg)

    from rclpy.qos import qos_profile_sensor_data

    sub = harness.create_subscription(message_type, topic, _on_message, qos_profile_sensor_data)
    try:
        deadline = time.time() + timeout_s
        while not received and time.time() < deadline:
            rclpy.spin_once(harness, timeout_sec=0.2)
        return received.get("msg")
    finally:
        harness.destroy_subscription(sub)


def test_camera_topics_declared_in_status_publish(harness: Harness):
    requires(harness, "perception.camera")
    _assert_perception_interfaces(harness, "perception.camera")


def test_lidar_topic_declared_in_status_publishes(harness: Harness):
    requires(harness, "perception.lidar")
    _assert_perception_interfaces(harness, "perception.lidar")


def _assert_perception_interfaces(harness, capability):
    from rosidl_runtime_py.utilities import get_message

    from robot_runtime.interface_description import validate_description

    description = json.loads(harness.status().interface_description_json)
    validate_description(description)
    interfaces = {key: spec for key, spec in description["interfaces"].items() if spec["capability"] == capability}
    assert interfaces, f"[Perception] {capability} has no public interfaces"
    for name, spec in interfaces.items():
        msg = _wait_for_message(harness, get_message(spec["message_type"]), spec["endpoint"])
        assert msg is not None, f"[Perception] {name} ({spec['message_type']}): no message on {spec['endpoint']}"
        if spec.get("frame_id"):
            assert msg.header.frame_id == spec["frame_id"], f"[Perception] {name} frame mismatch"
        if spec["message_type"] == "sensor_msgs/msg/Image":
            configured = spec["configured_profile"]
            assert (msg.width, msg.height) == (configured["width"], configured["height"]), name
            assert len(msg.data) == msg.height * msg.step and msg.step > 0, name
            if configured["encoding"]:
                assert msg.encoding == configured["encoding"], name
