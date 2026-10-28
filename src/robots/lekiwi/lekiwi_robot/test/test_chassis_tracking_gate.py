"""Chassis tracking gate on the physics platforms (lekiwi-gazebo-runtime-platform D3,
lekiwi-mujoco-runtime-platform D3/D3.2).

The provider-alone runtime is launched on the real simulator and driven
through the same command path as the physical robot (/cmd_vel -> base_node ->
base_velocity_controller -> the platform's velocity interfaces). The ground
truth comes from the shared test-support stream (``open_ground_truth``):
Gazebo bridges the world pose off the ROS graph; MuJoCo subscribes the
simulator-private floating-base topic (D4/D9). Every window is measured in
simulation time through the pose stream's own stamps.

Gate (all windows in simulation time; wall clock only bounds the test
timeout), with the step commands at 50 % of the published limits:

- ``vx`` and ``vy`` steps held 3 s, measured over the last 2.5 s: mean planar
  speed within 20 % of the command, mean heading within 15°, mean |yaw rate|
  <= 0.1 rad/s;
- ``wz`` step held 3 s, measured over the last 2.5 s: mean yaw rate within
  20 % of the command, mean translational speed <= 0.05 m/s;
- stop: after the stream stops, planar speed <= 0.02 m/s and |yaw rate| <=
  0.05 rad/s within 1 s of simulation time.

On MuJoCo the gate additionally exercises the published envelope (D3.2):
full-limit, over-limit and combined-axis commands assert both the wheel
odometry's reported twist against the ground-truth motion and the
ground-truth speed at or below the published limits (+5 % tolerance).

A failing threshold fails the test with the measured numbers in the message
(the tasks' STOP condition). A platform runs only when its code capabilities
and its simulator environment are present; ``IBROBOT_REQUIRE_SIM_PLATFORMS``
turns any skip of a listed platform into a failure (the acceptance runs).
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import rclpy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode
from robot_runtime.profile import PHYSICS_SIM_PLATFORMS, load_profile
from robot_runtime.testing.sim_environment import environment_missing, skip_or_fail
from robot_runtime.testing.sim_ground_truth import (
    RUN_MARKER_ENV,
    open_ground_truth,
    terminate_marked_processes,
)

PACKAGE = "lekiwi_robot"
MODEL_NAME = "lekiwi_robot"  # the runtime's model name in the simulator
WORLD_NAME = "empty"  # the provider's standalone world

# The profile's published limits (capabilities.base.cmd_vel) halved per D3.
VX_STEP = 0.26 / 2
VY_STEP = 0.23 / 2
WZ_STEP = 1.5 / 2

LAUNCH_STARTUP_TIMEOUT = 180.0  # wall seconds
COMMAND_RATE_HZ = 20.0

_LIDAR_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_lidar.yaml"


def _environment_missing(sim_platform: str) -> str | None:
    """The shared environment probe (robot_runtime.testing.sim_environment).

    Kept as a local name for the call sites; the Gazebo probe accepts EITHER
    stack (``ign gazebo`` / ``gz sim``) and the MuJoCo probe names every
    missing piece. The decision layer is the shared require-or-skip rule.
    """
    return environment_missing(sim_platform)


def _gate_skip_or_fail(sim_platform: str, reason: str) -> None:
    """Skip — unless acceptance runs required this platform: then fail."""
    skip_or_fail(sim_platform, reason, context="chassis tracking gate")


def _platform_capabilities_missing(sim_platform: str) -> list[str]:
    """The gate's code capabilities: ground truth and the profile platform."""
    from robot_runtime.testing.sim_ground_truth import UnsupportedGroundTruth

    missing = []
    try:
        open_ground_truth(sim_platform, world=WORLD_NAME, model=MODEL_NAME)
    except UnsupportedGroundTruth as error:
        missing.append(f"ground truth: {error}")
    supported = [str(entry) for entry in load_profile(_LIDAR_PROFILE).get("sim_platforms") or []]
    if sim_platform not in supported:
        missing.append(f"runtime profile: {sim_platform!r} is not in sim_platforms {supported}")
    return missing


class GateNode(Node):
    def __init__(self):
        super().__init__("lekiwi_chassis_gate")
        self.status: RuntimeStatus | None = None
        self.odom: list[Odometry] = []
        self.create_subscription(RuntimeStatus, "/runtime_status", self._status, 10)
        self.create_subscription(Odometry, "/odom", self._odom, 20)

    def _status(self, msg: RuntimeStatus) -> None:
        self.status = msg

    def _odom(self, msg: Odometry) -> None:
        self.odom.append(msg)


def _measure(window: list) -> dict:
    """Per-sample planar speed / heading and yaw rate over a pose window."""
    speeds, headings, yaw_rates = [], [], []
    for previous, current in zip(window, window[1:], strict=False):
        dt = current.stamp_sec - previous.stamp_sec
        if dt <= 1e-6:
            continue
        dx, dy = current.x - previous.x, current.y - previous.y
        speeds.append(math.hypot(dx, dy) / dt)
        if math.hypot(dx, dy) > 1e-9:
            headings.append(math.atan2(dy, dx))
        dyaw = math.atan2(math.sin(current.yaw - previous.yaw), math.cos(current.yaw - previous.yaw))
        yaw_rates.append(dyaw / dt)
    return {
        "mean_planar_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "mean_abs_yaw_rate": sum(abs(rate) for rate in yaw_rates) / len(yaw_rates) if yaw_rates else 0.0,
        "mean_heading": sum(headings) / len(headings) if headings else None,
    }


def _measure_body(window: list) -> tuple[float, float, float]:
    """Mean body-frame twist (vx, vy, wz) over a ground-truth pose window.

    The ground-truth pose is world-frame; each sample pair's displacement is
    rotated by the midpoint yaw into the body frame, so the result compares
    directly with the wheel odometry's body twist (D3.2).
    """
    vxs, vys, wzs = [], [], []
    for previous, current in zip(window, window[1:], strict=False):
        dt = current.stamp_sec - previous.stamp_sec
        if dt <= 1e-6:
            continue
        dx, dy = current.x - previous.x, current.y - previous.y
        yaw = 0.5 * (previous.yaw + current.yaw)
        cos, sin = math.cos(yaw), math.sin(yaw)
        vxs.append((cos * dx + sin * dy) / dt)
        vys.append((-sin * dx + cos * dy) / dt)
        dyaw = math.atan2(math.sin(current.yaw - previous.yaw), math.cos(current.yaw - previous.yaw))
        wzs.append(dyaw / dt)
    return (
        sum(vxs) / len(vxs) if vxs else 0.0,
        sum(vys) / len(vys) if vys else 0.0,
        sum(wzs) / len(wzs) if wzs else 0.0,
    )


def _angle_within(measured: float, expected: float, tolerance: float) -> bool:
    return abs(math.atan2(math.sin(measured - expected), math.cos(measured - expected))) <= tolerance


@pytest.mark.parametrize("sim_platform", PHYSICS_SIM_PLATFORMS)
def test_chassis_tracking_gate(tmp_path, sim_platform):
    missing = _platform_capabilities_missing(sim_platform)
    if missing:
        _gate_skip_or_fail(
            sim_platform, f"chassis tracking gate: platform {sim_platform!r} lacks capabilities: {missing}"
        )
    environment_missing = _environment_missing(sim_platform)
    if environment_missing:
        _gate_skip_or_fail(
            sim_platform, f"chassis tracking gate: platform {sim_platform!r} cannot run here: {environment_missing}"
        )
    partition = f"lekiwi_gate_{uuid.uuid4().hex[:8]}"
    environment = dict(os.environ)
    # the neutral per-run marker (review round 1): every platform's launch
    # children carry it, so the teardown sweep below also catches MuJoCo
    # descendants that escape the launch process group (the Gazebo-only
    # partition marker never matched them)
    environment[RUN_MARKER_ENV] = partition
    if sim_platform == "gazebo":
        # both partition variables: Fortress (ign, gz-transport11) reads only
        # IGN_PARTITION, newer gz reads GZ_PARTITION — set both to isolate
        # this run's gz bus on either stack
        environment.update(
            {
                "GZ_PARTITION": partition,
                "IGN_PARTITION": partition,
                "IBROBOT_GAZEBO_HEADLESS": "1",
            }
        )
    else:
        environment["IBROBOT_MUJOCO_HEADLESS"] = "1"
    environment["RCUTILS_COLORIZED_OUTPUT"] = "0"
    launch_log = tmp_path / "launch.log"
    launch_arguments = [
        "ros2",
        "launch",
        PACKAGE,
        "runtime.launch.py",
        "profile:=lekiwi_lidar",
        f"sim_platform:={sim_platform}",
    ]
    if sim_platform == "mujoco":
        launch_arguments.append(f"mujoco_scene:={WORLD_NAME}")
    launch = subprocess.Popen(
        launch_arguments,
        stdout=launch_log.open("w"),
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    measurements: dict[str, dict] = {}
    ground_truth = None
    node = None
    try:
        rclpy.init()
        node = GateNode()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        deadline = time.monotonic() + LAUNCH_STARTUP_TIMEOUT
        while time.monotonic() < deadline and (
            node.status is None or node.status.lifecycle != "ACTIVE" or node.status.active_mode != "idle"
        ):
            executor.spin_once(timeout_sec=0.2)
            if launch.poll() is not None:
                pytest.fail(f"the runtime launch exited early ({launch.returncode}); see {launch_log}")
        assert node.status is not None and node.status.lifecycle == "ACTIVE"
        assert node.status.active_mode == "idle"

        # the shared test-support stream (D4/D9): nothing ground-truth is
        # advertised on the ROS graph. The launch env (with BOTH partition
        # variables — Fortress reads only IGN_PARTITION) is passed so the
        # stream joins the same isolated bus as the simulator it observes.
        ground_truth = open_ground_truth(sim_platform, world=WORLD_NAME, model=MODEL_NAME, env=environment).start()
        # the pose stream is the simulation clock for every window below
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and ground_truth.latest_time() is None:
            time.sleep(0.1)
        assert ground_truth.latest_time() is not None, "no world-pose samples arrived"

        # admission through the contract names: mode, then gate. The mode
        # request is retried: under parallel test load the controller
        # spawners may not have finished loading when the first switch
        # arrives and controller_manager rejects it (a timing race, not a
        # tracking failure).
        mode_client = node.create_client(SetRuntimeMode, "/runtime/set_mode")
        assert mode_client.wait_for_service(timeout_sec=15.0)
        request = SetRuntimeMode.Request()
        request.mode = "base_navigation"
        admitted = False
        for _ in range(10):
            future = mode_client.call_async(request)
            executor.spin_until_future_complete(future, timeout_sec=15.0)
            result = future.result()
            if result is not None and result.success:
                admitted = True
                break
            time.sleep(1.0)
        assert admitted, f"base_navigation mode was not admitted: {result and result.message}"
        gate_client = node.create_client(SetBool, "/motion_mode/set_navigation_enabled")
        assert gate_client.wait_for_service(timeout_sec=15.0)
        request = SetBool.Request()
        request.data = True
        future = gate_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and node.status.active_mode != "base_navigation":
            executor.spin_once(timeout_sec=0.2)
        assert node.status.active_mode == "base_navigation"

        command = node.create_publisher(Twist, "/cmd_vel", 10)

        def hold(twist: Twist, sim_seconds: float) -> tuple[float, float]:
            """Publish at COMMAND_RATE_HZ until sim time advanced by sim_seconds."""
            start = ground_truth.latest_time()
            assert start is not None
            deadline = time.monotonic() + sim_seconds * 20.0 + 30.0
            while True:
                executor.spin_once(timeout_sec=0.01)
                command.publish(twist)
                now = ground_truth.latest_time()
                assert now is not None
                if now - start >= sim_seconds or time.monotonic() > deadline:
                    return start, now
                time.sleep(1.0 / COMMAND_RATE_HZ)

        def step_gate(label: str, twist: Twist, expect_translation: bool, expected_heading: float | None) -> None:
            hold_start, hold_end = hold(twist, 3.0)
            window = ground_truth.window(hold_end - 2.5, hold_end - 0.05)
            assert len(window) > 10, f"{label}: too few ground-truth samples ({len(window)})"
            result = _measure(window)
            measurements[label] = {**result, "samples": len(window)}
            print(f"[gate] {label}: {measurements[label]}", flush=True)
            if expect_translation:
                command_speed = math.hypot(twist.linear.x, twist.linear.y)
                assert result["mean_planar_speed"] == pytest.approx(command_speed, rel=0.2), (
                    f"{label}: mean planar speed {result['mean_planar_speed']:.4f} m/s outside 20 % of "
                    f"{command_speed:.4f} m/s"
                )
                assert result["mean_heading"] is not None and _angle_within(
                    result["mean_heading"], expected_heading, math.radians(15)
                ), (
                    f"{label}: mean heading "
                    f"{math.degrees(result['mean_heading']):.1f}° outside 15° of "
                    f"{math.degrees(expected_heading):.1f}°"
                )
                assert result["mean_abs_yaw_rate"] <= 0.1, (
                    f"{label}: mean |yaw rate| {result['mean_abs_yaw_rate']:.4f} rad/s exceeds 0.1"
                )
            else:
                assert result["mean_abs_yaw_rate"] == pytest.approx(abs(twist.angular.z), rel=0.2), (
                    f"{label}: mean yaw rate {result['mean_abs_yaw_rate']:.4f} rad/s outside 20 % of "
                    f"{abs(twist.angular.z):.4f} rad/s"
                )
                assert result["mean_planar_speed"] <= 0.05, (
                    f"{label}: mean translational speed {result['mean_planar_speed']:.4f} m/s exceeds 0.05"
                )

        # translational steps: expected heading follows the robot's yaw at the
        # step start (the world-frame direction of the commanded body axis)
        start_yaw = ground_truth.latest().yaw
        step_gate("vx", Twist(linear=Vector3(x=VX_STEP)), True, start_yaw)
        start_yaw = ground_truth.latest().yaw
        step_gate("vy", Twist(linear=Vector3(y=VY_STEP)), True, start_yaw + math.pi / 2)
        step_gate("wz", Twist(angular=Vector3(z=WZ_STEP)), False, None)

        # lekiwi-mujoco-runtime-platform D3.2: the published envelope governs
        # the executed twist on MuJoCo — full-limit, over-limit and
        # combined-axis commands assert the wheel odometry's reported twist
        # against the ground truth and the per-axis ground-truth speed at or
        # below the published limits (+5 % tolerance).
        if sim_platform == "mujoco":
            envelope = {"vx": 0.26, "vy": 0.23, "wz": 1.5}

            def _stamp(msg):
                return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            def envelope_step(label: str, twist: Twist) -> None:
                hold_start, hold_end = hold(twist, 3.0)
                window = ground_truth.window(hold_end - 2.5, hold_end - 0.05)
                assert len(window) > 10, f"{label}: too few ground-truth samples ({len(window)})"
                gt_vx, gt_vy, gt_wz = _measure_body(window)
                odom_window = [msg for msg in node.odom if hold_start < _stamp(msg) <= hold_end]
                assert len(odom_window) > 10, f"{label}: too few /odom messages ({len(odom_window)})"
                odom_vx = sum(msg.twist.twist.linear.x for msg in odom_window) / len(odom_window)
                odom_vy = sum(msg.twist.twist.linear.y for msg in odom_window) / len(odom_window)
                odom_wz = sum(msg.twist.twist.angular.z for msg in odom_window) / len(odom_window)
                measurements[label] = {
                    "gt": (gt_vx, gt_vy, gt_wz),
                    "odom": (odom_vx, odom_vy, odom_wz),
                    "samples": len(window),
                }
                print(f"[gate] {label}: {measurements[label]}", flush=True)
                # /odom twist against the ground-truth twist (20 %)
                for name, reported, measured_gt in (
                    ("vx", odom_vx, gt_vx),
                    ("vy", odom_vy, gt_vy),
                    ("wz", odom_wz, gt_wz),
                ):
                    assert reported == pytest.approx(measured_gt, rel=0.2, abs=0.02), (
                        f"{label}: /odom {name} {reported:.4f} vs ground truth {measured_gt:.4f}"
                    )
                # the ground-truth per-axis speed at or below the limit (+5 %)
                for name, measured_gt, limit in (
                    ("vx", gt_vx, envelope["vx"]),
                    ("vy", gt_vy, envelope["vy"]),
                    ("wz", gt_wz, envelope["wz"]),
                ):
                    assert abs(measured_gt) <= limit * 1.05, (
                        f"{label}: ground-truth {name} {measured_gt:.4f} exceeds the published limit {limit} (+5 %)"
                    )

            envelope_step("full_vx", Twist(linear=Vector3(x=envelope["vx"])))
            envelope_step("over_vx", Twist(linear=Vector3(x=2.0 * envelope["vx"])))
            envelope_step(
                "combined",
                Twist(
                    linear=Vector3(x=envelope["vx"], y=envelope["vy"]),
                    angular=Vector3(z=envelope["wz"]),
                ),
            )

        # stop: after the stream stops the base must settle within 1 s (sim)
        stop_start = ground_truth.latest_time()
        assert stop_start is not None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
            now = ground_truth.latest_time()
            assert now is not None
            if now - stop_start >= 1.0:
                break
        settle = ground_truth.window(stop_start + 0.7, stop_start + 1.0)
        result = _measure(settle)
        measurements["stop"] = {**result, "samples": len(settle)}
        print(f"[gate] stop: {measurements['stop']}", flush=True)
        assert result["mean_planar_speed"] <= 0.02, (
            f"stop: planar speed {result['mean_planar_speed']:.4f} m/s exceeds 0.02 within 1 s"
        )
        assert result["mean_abs_yaw_rate"] <= 0.05, (
            f"stop: |yaw rate| {result['mean_abs_yaw_rate']:.4f} rad/s exceeds 0.05 within 1 s"
        )
    finally:
        if ground_truth is not None:
            ground_truth.close()
        # destroy the node BEFORE the shutdown: an undestroyed node's DDS
        # entities linger in discovery after the context closes and the
        # next test in this process (same domain) then sees this test's
        # /cmd_vel publisher on its own graph check
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Review round 2: the launch may already be gone (an early fail-fast
        # exit), when os.getpgid raises ProcessLookupError — the kill sequence
        # must not abort the teardown, and the marker sweep must be
        # UNCONDITIONAL (it is what catches process-group escapees).
        try:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(launch.pid), signal.SIGINT)
            launch.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(launch.pid), signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                launch.wait(timeout=5)
        finally:
            terminate_marked_processes({RUN_MARKER_ENV: partition, "GZ_PARTITION": partition})
        print(f"[gate] measurements: {measurements}", flush=True)
