"""Chassis tracking gate on Gazebo (openspec lekiwi-gazebo-runtime-platform D3).

The provider-alone runtime is launched on the real Gazebo platform and driven
through the same command path as the physical robot (/cmd_vel -> base_node ->
base_velocity_controller -> gz velocity interfaces). The test bridges the
simulator's world pose itself (``/world/<w>/pose/info`` via ``ign topic -e``):
nothing ground-truth is advertised on the ROS graph (D4), and every window is
measured in simulation time through the pose stream's own stamps.

Gate (all windows in simulation time; wall clock only bounds the test
timeout), with the step commands at 50 % of the published limits:

- ``vx`` and ``vy`` steps held 3 s, measured over the last 2.5 s: mean planar
  speed within 20 % of the command, mean heading within 15°, mean |yaw rate|
  <= 0.1 rad/s;
- ``wz`` step held 3 s, measured over the last 2.5 s: mean yaw rate within
  20 % of the command, mean translational speed <= 0.05 m/s;
- stop: after the stream stops, planar speed <= 0.02 m/s and |yaw rate| <=
  0.05 rad/s within 1 s of simulation time.

A failing threshold fails the test with the measured numbers in the message
(the tasks' STOP condition).
"""

from __future__ import annotations

import math
import os
import subprocess
import time
import uuid

import pytest
import rclpy
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode
from robot_runtime.testing.sim_ground_truth import open_ground_truth, terminate_partition_processes

PACKAGE = "lekiwi_robot"
MODEL_NAME = "lekiwi_robot"  # the runtime's model name in the simulator
WORLD_NAME = "empty"  # the provider's standalone world

# The profile's published limits (capabilities.base.cmd_vel) halved per D3.
VX_STEP = 0.26 / 2
VY_STEP = 0.23 / 2
WZ_STEP = 1.5 / 2

LAUNCH_STARTUP_TIMEOUT = 180.0  # wall seconds
COMMAND_RATE_HZ = 20.0


def _gazebo_available() -> bool:
    try:
        return subprocess.run(["ign", "gazebo", "--version"], capture_output=True, timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _gazebo_available(), reason="Gazebo (ign gazebo) is not available")


class GateNode(Node):
    def __init__(self):
        super().__init__("lekiwi_chassis_gate")
        self.status: RuntimeStatus | None = None
        self.create_subscription(RuntimeStatus, "/runtime_status", self._status, 10)

    def _status(self, msg: RuntimeStatus) -> None:
        self.status = msg


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


def _angle_within(measured: float, expected: float, tolerance: float) -> bool:
    return abs(math.atan2(math.sin(measured - expected), math.cos(measured - expected))) <= tolerance


def test_chassis_tracking_gate(tmp_path):
    partition = f"lekiwi_gate_{uuid.uuid4().hex[:8]}"
    environment = dict(os.environ)
    # both partition variables: Fortress (ign, gz-transport11) reads only
    # IGN_PARTITION, newer gz reads GZ_PARTITION — set both to isolate this
    # run's gz bus on either stack
    environment.update(
        {
            "GZ_PARTITION": partition,
            "IGN_PARTITION": partition,
            "IBROBOT_GAZEBO_HEADLESS": "1",
            "RCUTILS_COLORIZED_OUTPUT": "0",
        }
    )
    launch_log = tmp_path / "launch.log"
    launch = subprocess.Popen(
        [
            "ros2",
            "launch",
            PACKAGE,
            "runtime.launch.py",
            "profile:=lekiwi_lidar",
            "sim_platform:=gazebo",
        ],
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

        # the shared test-support bridge (D4): nothing ground-truth on the
        # graph. The launch env (with BOTH partition variables — Fortress
        # reads only IGN_PARTITION) is passed so the echo joins the same
        # isolated gz bus as the simulator it observes.
        ground_truth = open_ground_truth("gazebo", world=WORLD_NAME, model=MODEL_NAME, env=environment).start()
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
        os.killpg(os.getpgid(launch.pid), 15)
        try:
            launch.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(launch.pid), 9)
        terminate_partition_processes(partition)
        print(f"[gate] measurements: {measurements}", flush=True)
