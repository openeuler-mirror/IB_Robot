"""MuJoCo chassis stop gate (lekiwi-mujoco-runtime-platform D3).

One provider-alone MuJoCo run drives the base through the real command path
and exercises every stop path on ground truth (D3's stop tests):

- (a) an ``idle`` mode request, (b) stop HOLD, (c) stop TORQUE_OFF, (d) the
  navigation gate disable, (e) the ``/cmd_vel`` stream stopping — moving at
  50 % first, then planar speed <= 0.02 m/s and |yaw rate| <= 0.05 rad/s
  within 1 s of simulation time, the wheel odometry's twist at zero, and no
  motion after re-entering ``base_navigation`` without new commands (the
  never-released MuJoCo velocity controller is safe because base_node is the
  gate, D3.1);

- the command-to-motion latency at the default real-time factor (D3.4/D12);

- pause/resume: with the simulation paused nothing moves and no simulation
  time passes; on resume the stale-command path keeps the base stopped;

- the reset chain (D3.3): moving -> pause (the sim clock freezes with a
  cached NONZERO plugin command; the command stream stops and wall-clock
  staleness engages) -> ``reset_world`` -> resume: the relay's reset epoch
  zeroes the plugin's cached command and the base stays at rest at the
  restored spawn pose. A plain un-paused reset only adds base_node's
  documented wall-clock staleness residue (D3.4), which would mask the
  relay's contribution.

A second, throttled launch (``sim_speed_factor`` 0.25 through a staged
profile copy) measures the command-to-motion latency at RTF ~ 0.25.

Skips follow the shared rules: missing environment skips with a reason
unless ``IBROBOT_REQUIRE_SIM_PLATFORMS`` lists ``mujoco`` (then it fails).
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import random
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import rclpy
import yaml
from geometry_msgs.msg import Twist, Vector3
from mujoco_test_support import require_mujoco
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode, StopRuntime
from robot_runtime.profile import load_profile
from robot_runtime.testing.sim_ground_truth import (
    RUN_MARKER_ENV,
    open_ground_truth,
    terminate_marked_processes,
)

# mujoco_ros2_control_msgs (SetPause/ResetWorld) is imported lazily inside the
# tests: the package exists only where the simulator is installed, and a
# module-level import would break collection of this whole file on openEuler.

PACKAGE = "lekiwi_robot"
MODEL_NAME = "lekiwi_robot"
SCENE = "empty"
PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_lidar.yaml"
VX_STEP = 0.26 / 2  # 50 % of the published limit, as in the tracking gate

LAUNCH_STARTUP_TIMEOUT = 120.0
COMMAND_RATE_HZ = 20.0


class StopProbe(Node):
    def __init__(self):
        super().__init__("mujoco_stop_gate_probe")
        self.status: RuntimeStatus | None = None
        self.odom: list[Odometry] = []
        self.create_subscription(RuntimeStatus, "/runtime_status", self._on_status, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)

    def _on_status(self, msg: RuntimeStatus) -> None:
        self.status = msg

    def _on_odom(self, msg: Odometry) -> None:
        self.odom.append(msg)

    def last_odom_twist(self) -> tuple[float, float, float]:
        msg = self.odom[-1]
        return (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z)


def _measure(window: list) -> dict:
    """Per-sample planar speed and yaw rate over a ground-truth pose window."""
    speeds, yaw_rates = [], []
    for previous, current in zip(window, window[1:], strict=False):
        dt = current.stamp_sec - previous.stamp_sec
        if dt <= 1e-6:
            continue
        speeds.append(math.hypot(current.x - previous.x, current.y - previous.y) / dt)
        dyaw = math.atan2(math.sin(current.yaw - previous.yaw), math.cos(current.yaw - previous.yaw))
        yaw_rates.append(dyaw / dt)
    return {
        "mean_planar_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "mean_abs_yaw_rate": sum(abs(rate) for rate in yaw_rates) / len(yaw_rates) if yaw_rates else 0.0,
    }


def _launch(environment: dict, log_path: Path, profile_path: Path | None = None):
    arguments = [
        "ros2",
        "launch",
        PACKAGE,
        "runtime.launch.py",
        f"profile:={profile_path or PROFILE}",
        "sim_platform:=mujoco",
        f"mujoco_scene:={SCENE}",
    ]
    return subprocess.Popen(
        arguments,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )


def _terminate(launch) -> None:
    # Review round 2: the launch may already be gone, when os.getpgid
    # raises ProcessLookupError — the kill sequence must not abort the
    # caller's teardown (the marker sweep in the caller's finally is what
    # catches process-group escapees, and it must always run).
    try:
        if launch.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(launch.pid), signal.SIGINT)
            launch.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(launch.pid), signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            launch.wait(timeout=10)


def _fresh_environment() -> tuple[dict, str, str]:
    """The launch environment on a fresh domain, plus this run's sweep marker.

    The neutral ``IBROBOT_TEST_RUN`` marker (review round 1) is inherited by
    every launch child, so the teardown sweep catches MuJoCo descendants that
    escape the launch process group — the Gazebo-only partition sweep never
    matched this package's MuJoCo launches.
    """
    domain = str(random.randint(150, 229))
    run = f"lekiwi_stop_{uuid.uuid4().hex[:8]}"
    environment = dict(os.environ)
    environment.update({"ROS_DOMAIN_ID": domain, "IBROBOT_MUJOCO_HEADLESS": "1", RUN_MARKER_ENV: run})
    # the probe joins the launch's domain (rclpy reads ROS_DOMAIN_ID at init)
    os.environ["ROS_DOMAIN_ID"] = domain
    return environment, domain, run


def test_mujoco_stop_paths_latency_pause_and_reset_chain(tmp_path):
    require_mujoco()
    from mujoco_ros2_control_msgs.srv import ResetWorld, SetPause

    environment, _domain, run = _fresh_environment()
    launch_log = tmp_path / "launch.log"
    launch = _launch(environment, launch_log)
    probe = None
    executor = None
    measurements: dict[str, dict] = {}
    ground_truth = None
    try:
        rclpy.init()
        probe = StopProbe()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(probe)

        # --- readiness ----------------------------------------------------------
        deadline = time.monotonic() + LAUNCH_STARTUP_TIMEOUT
        while time.monotonic() < deadline and (probe.status is None or probe.status.lifecycle != "ACTIVE"):
            executor.spin_once(timeout_sec=0.2)
            if launch.poll() is not None:
                pytest.fail(f"the launch exited early ({launch.returncode}); see {launch_log}")
        assert probe.status is not None and probe.status.lifecycle == "ACTIVE", f"see {launch_log}"
        ground_truth = open_ground_truth("mujoco", world=SCENE, model=MODEL_NAME, env=environment).start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and ground_truth.latest_time() is None:
            executor.spin_once(timeout_sec=0.1)
        assert ground_truth.latest_time() is not None, "no floating-base state arrived"

        def admit() -> None:
            """base_navigation + the navigation gate open (retried)."""
            mode_client = probe.create_client(SetRuntimeMode, "/runtime/set_mode")
            assert mode_client.wait_for_service(timeout_sec=15.0)
            request = SetRuntimeMode.Request()
            request.mode = "base_navigation"
            for _ in range(10):
                future = mode_client.call_async(request)
                executor.spin_until_future_complete(future, timeout_sec=15.0)
                if future.result() is not None and future.result().success:
                    break
                time.sleep(1.0)
            assert future.result() is not None and future.result().success, "base_navigation not admitted"
            gate_client = probe.create_client(SetBool, "/motion_mode/set_navigation_enabled")
            assert gate_client.wait_for_service(timeout_sec=15.0)
            gate_request = SetBool.Request()
            gate_request.data = True
            future = gate_client.call_async(gate_request)
            executor.spin_until_future_complete(future, timeout_sec=15.0)
            assert future.result() is not None and future.result().success
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and (
                probe.status is None or probe.status.active_mode != "base_navigation"
            ):
                executor.spin_once(timeout_sec=0.1)
            assert probe.status is not None and probe.status.active_mode == "base_navigation"

        def clear_stop() -> None:
            """Clear a latched stop through the documented idle request."""
            mode_client = probe.create_client(SetRuntimeMode, "/runtime/set_mode")
            request = SetRuntimeMode.Request()
            request.mode = "idle"
            for _ in range(10):
                future = mode_client.call_async(request)
                executor.spin_until_future_complete(future, timeout_sec=15.0)
                if future.result() is not None and future.result().success:
                    return
                time.sleep(1.0)
            pytest.fail("the idle request did not clear the stop latch")

        command = probe.create_publisher(Twist, "/cmd_vel", 10)
        moving = Twist(linear=Vector3(x=VX_STEP))

        def drive(sim_seconds: float) -> None:
            start = ground_truth.latest_time()
            deadline = time.monotonic() + sim_seconds * 20.0 + 30.0
            while True:
                executor.spin_once(timeout_sec=0.01)
                command.publish(moving)
                now = ground_truth.latest_time()
                if now - start >= sim_seconds or time.monotonic() > deadline:
                    return
                time.sleep(1.0 / COMMAND_RATE_HZ)

        def hold(sim_seconds: float) -> None:
            """No commands; advance sim time (the stream-stop path)."""
            start = ground_truth.latest_time()
            deadline = time.monotonic() + sim_seconds * 20.0 + 30.0
            while True:
                executor.spin_once(timeout_sec=0.01)
                now = ground_truth.latest_time()
                if now - start >= sim_seconds or time.monotonic() > deadline:
                    return
                time.sleep(0.02)

        def assert_stopped(label: str, stop_start: float) -> None:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
                now = ground_truth.latest_time()
                if now - stop_start >= 1.0:
                    break
            settle = ground_truth.window(stop_start + 0.7, stop_start + 1.0)
            result = _measure(settle)
            measurements[label] = {**result, "samples": len(settle)}
            print(f"[stop-gate] {label}: {measurements[label]}", flush=True)
            assert result["mean_planar_speed"] <= 0.02, (
                f"{label}: planar speed {result['mean_planar_speed']:.4f} m/s exceeds 0.02 within 1 s"
            )
            assert result["mean_abs_yaw_rate"] <= 0.05, (
                f"{label}: |yaw rate| {result['mean_abs_yaw_rate']:.4f} rad/s exceeds 0.05 within 1 s"
            )
            twist = probe.last_odom_twist()
            assert max(abs(value) for value in twist) <= 0.02, f"{label}: /odom twist not zero: {twist}"

        def assert_no_resume(label: str) -> None:
            """Re-entering base_navigation without commands must not move."""
            admit()
            start = ground_truth.latest_time()
            start_pose = ground_truth.latest()
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
                if ground_truth.latest_time() - start >= 2.0:
                    break
            displacement = math.hypot(
                ground_truth.latest().x - start_pose.x,
                ground_truth.latest().y - start_pose.y,
            )
            measurements[f"{label}_no_resume"] = {"displacement": displacement}
            assert displacement < 0.01, f"{label}: re-entering base_navigation moved the base {displacement:.4f} m"

        # --- latency at the default real-time factor (D3.4/D12) ----------------
        admit()
        probe.odom.clear()
        latency_start = time.monotonic()
        moving_detected = None
        start_pose = ground_truth.latest()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
            command.publish(moving)
            pose = ground_truth.latest()
            if moving_detected is None and math.hypot(pose.x - start_pose.x, pose.y - start_pose.y) > 0.02:
                moving_detected = time.monotonic() - latency_start
                break
            time.sleep(1.0 / COMMAND_RATE_HZ)
        assert moving_detected is not None, "the base never moved under the command"
        measurements["latency_default"] = {"command_to_motion_s": moving_detected}
        print(f"[stop-gate] latency (default RTF): {moving_detected:.3f} s", flush=True)
        # stop the motion again for a clean baseline
        hold(1.5)

        # --- (e) the /cmd_vel stream stops --------------------------------------
        drive(1.5)
        stop_start = ground_truth.latest_time()
        hold(1.5)
        assert_stopped("stream_stop", stop_start)
        assert_no_resume("stream_stop")

        # --- (a) an idle mode request -------------------------------------------
        admit()
        drive(1.5)
        stop_start = ground_truth.latest_time()
        mode_client = probe.create_client(SetRuntimeMode, "/runtime/set_mode")
        request = SetRuntimeMode.Request()
        request.mode = "idle"
        future = mode_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success
        hold(1.5)
        assert_stopped("idle_request", stop_start)
        assert_no_resume("idle_request")

        # --- (b) stop HOLD --------------------------------------------------------
        admit()
        drive(1.5)
        stop_start = ground_truth.latest_time()
        stop_client = probe.create_client(StopRuntime, "/runtime/stop")
        request = StopRuntime.Request()
        request.policy = "HOLD"
        future = stop_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success, future.result() and future.result().message
        hold(1.5)
        assert_stopped("stop_hold", stop_start)
        clear_stop()
        assert_no_resume("stop_hold")

        # --- (c) stop TORQUE_OFF --------------------------------------------------
        admit()
        drive(1.5)
        stop_start = ground_truth.latest_time()
        request = StopRuntime.Request()
        request.policy = "TORQUE_OFF"
        future = stop_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success, future.result() and future.result().message
        hold(1.5)
        assert_stopped("stop_torque_off", stop_start)
        clear_stop()
        assert_no_resume("stop_torque_off")

        # --- (d) the navigation gate disable --------------------------------------
        admit()
        drive(1.5)
        stop_start = ground_truth.latest_time()
        gate_client = probe.create_client(SetBool, "/motion_mode/set_navigation_enabled")
        request = SetBool.Request()
        request.data = False
        future = gate_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success
        hold(1.5)
        assert_stopped("gate_disable", stop_start)
        assert_no_resume("gate_disable")

        # --- pause / resume (D3.4) -------------------------------------------------
        admit()
        drive(1.5)
        pause_client = probe.create_client(SetPause, "/mujoco_ros2_control_node/set_pause")
        assert pause_client.wait_for_service(timeout_sec=15.0)
        request = SetPause.Request()
        request.paused = True
        future = pause_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success
        # Drain the pre-pause backlog before measuring the freeze: odometry
        # already in flight when the acknowledgment returned still arrives
        # afterwards (deeper under load — the flake of review round 1's final
        # run), and only a stamp arriving AFTER the drain, with nothing left
        # in flight, proves the simulator is still stepping.
        drain_deadline = time.monotonic() + 0.5
        while time.monotonic() < drain_deadline:
            executor.spin_once(timeout_sec=0.05)
        paused_time = ground_truth.latest_time()
        paused_wall = time.monotonic()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        frozen = ground_truth.latest_time() == paused_time
        measurements["pause"] = {
            "frozen_sim_time": frozen,
            "wall_seconds_paused": time.monotonic() - paused_wall,
        }
        assert frozen, "simulation time advanced while paused"
        request.paused = False
        future = pause_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and ground_truth.latest_time() == paused_time:
            executor.spin_once(timeout_sec=0.05)
        assert ground_truth.latest_time() > paused_time, "simulation time did not resume"
        # the command stream stopped while paused (staleness is wall clock):
        # the first control tick after resume zeroes the target
        stop_start = ground_truth.latest_time()
        hold(2.0)
        assert_stopped("pause_resume", stop_start)

        # --- the reset chain (D3.3) -------------------------------------------------
        # The design's scenario is "moving -> reset_world -> queued odometry
        # -> no fresh feedback: the base stays at rest". Constructed with a
        # pause around the reset: while moving the plugin holds a cached
        # NONZERO command; pausing freezes the sim clock (so base_node's
        # staleness cannot publish its zero — the control timer is on sim
        # time), the command stream stops long enough for staleness to
        # engage on the wall clock, then reset_world restores the initial
        # state (including the clock) and the resume re-runs everything on
        # the new epoch. Without the relay's reset epoch the cached command
        # would read "fresh" on the backwards-jumped clock and drive the
        # base; with it the zero stream overwrites the cache and the base
        # stays at rest at the restored spawn pose.
        admit()
        drive(1.5)
        pause_client = probe.create_client(SetPause, "/mujoco_ros2_control_node/set_pause")
        assert pause_client.wait_for_service(timeout_sec=15.0)
        request = SetPause.Request()
        request.paused = True
        future = pause_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success
        time.sleep(0.8)  # wall: the command staleness engages while frozen
        reset_client = probe.create_client(ResetWorld, "/mujoco_ros2_control_node/reset_world")
        assert reset_client.wait_for_service(timeout_sec=15.0)
        future = reset_client.call_async(ResetWorld.Request())
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success, future.result() and future.result().message
        request.paused = False
        future = pause_client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success
        # the world (and the clock) jumped back: wait for the stream to
        # settle on the new epoch near the restored spawn pose
        deadline = time.monotonic() + 15.0
        reset_pose = None
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            pose = ground_truth.latest()
            if pose is not None and math.hypot(pose.x, pose.y) < 0.05:
                reset_pose = pose
                break
            reset_pose = pose
        assert reset_pose is not None, "no ground-truth pose after the reset"
        assert math.hypot(reset_pose.x, reset_pose.y) < 0.05, (
            f"the reset did not restore the spawn pose: {reset_pose.x:.3f}, {reset_pose.y:.3f}"
        )
        settle_start = ground_truth.latest_time()
        start_pose = ground_truth.latest()
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
            if ground_truth.latest_time() - settle_start >= 2.0:
                break
        pose = ground_truth.latest()
        displacement = math.hypot(pose.x - start_pose.x, pose.y - start_pose.y)
        measurements["reset_chain"] = {"displacement_after_reset": displacement}
        print(f"[stop-gate] reset chain: displacement {displacement:.4f} m", flush=True)
        assert displacement < 0.01, (
            f"the base moved {displacement:.4f} m after reset_world (the relay's zero stream failed)"
        )
    finally:
        if ground_truth is not None:
            ground_truth.close()
        if executor is not None:
            executor.shutdown()
        if probe is not None:
            probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _terminate(launch)
        terminate_marked_processes({RUN_MARKER_ENV: run})
        print(f"[stop-gate] measurements: {measurements}", flush=True)


def test_mujoco_command_latency_throttled(tmp_path):
    """D3.4/D12: the command-to-motion latency at RTF ~ 0.25 (sim_speed_factor)."""
    require_mujoco()
    profile = load_profile(PROFILE)
    throttled = copy.deepcopy(profile)
    hardware = dict(throttled.get("hardware") or {})
    xacro_args = dict(hardware.get("xacro_args") or {})
    xacro_args["mujoco_sim_speed_factor"] = "0.25"
    hardware["xacro_args"] = xacro_args
    throttled["hardware"] = hardware
    staged = tmp_path / "lekiwi_lidar_throttled.yaml"
    staged.write_text(yaml.safe_dump(throttled, default_flow_style=False, sort_keys=False), encoding="utf-8")
    environment, _domain, run = _fresh_environment()
    launch_log = tmp_path / "launch.log"
    launch = _launch(environment, launch_log, profile_path=staged)
    probe = None
    executor = None
    ground_truth = None
    try:
        rclpy.init()
        probe = StopProbe()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(probe)
        deadline = time.monotonic() + LAUNCH_STARTUP_TIMEOUT
        while time.monotonic() < deadline and (probe.status is None or probe.status.lifecycle != "ACTIVE"):
            executor.spin_once(timeout_sec=0.2)
            if launch.poll() is not None:
                pytest.fail(f"the throttled launch exited early ({launch.returncode}); see {launch_log}")
        assert probe.status is not None and probe.status.lifecycle == "ACTIVE", f"see {launch_log}"
        ground_truth = open_ground_truth("mujoco", world=SCENE, model=MODEL_NAME, env=environment).start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and ground_truth.latest_time() is None:
            executor.spin_once(timeout_sec=0.1)
        assert ground_truth.latest_time() is not None, "no floating-base state arrived"

        # the throttled real-time factor: sim seconds per wall second ~ 0.25
        wall_start = time.monotonic()
        sim_start = ground_truth.latest_time()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        rtf = (ground_truth.latest_time() - sim_start) / (time.monotonic() - wall_start)

        # admission, then the command-to-motion latency on the throttled clock
        mode_client = probe.create_client(SetRuntimeMode, "/runtime/set_mode")
        assert mode_client.wait_for_service(timeout_sec=15.0)
        request = SetRuntimeMode.Request()
        request.mode = "base_navigation"
        for _ in range(10):
            future = mode_client.call_async(request)
            executor.spin_until_future_complete(future, timeout_sec=15.0)
            if future.result() is not None and future.result().success:
                break
            time.sleep(1.0)
        assert future.result() is not None and future.result().success
        gate_client = probe.create_client(SetBool, "/motion_mode/set_navigation_enabled")
        assert gate_client.wait_for_service(timeout_sec=15.0)
        gate_request = SetBool.Request()
        gate_request.data = True
        future = gate_client.call_async(gate_request)
        executor.spin_until_future_complete(future, timeout_sec=15.0)
        assert future.result() is not None and future.result().success

        command = probe.create_publisher(Twist, "/cmd_vel", 10)
        moving = Twist(linear=Vector3(x=VX_STEP))
        latency_start = time.monotonic()
        moving_detected = None
        start_pose = ground_truth.latest()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
            command.publish(moving)
            pose = ground_truth.latest()
            if moving_detected is None and math.hypot(pose.x - start_pose.x, pose.y - start_pose.y) > 0.02:
                moving_detected = time.monotonic() - latency_start
                break
            time.sleep(1.0 / COMMAND_RATE_HZ)
        assert moving_detected is not None, "the base never moved under the throttled command"
        print(f"[stop-gate] latency (throttled RTF {rtf:.2f}): {moving_detected:.3f} s", flush=True)
        # no committed threshold in the design (measured evidence, D12); a
        # broken chain would take the whole budget
        assert moving_detected < 5.0
    finally:
        if ground_truth is not None:
            ground_truth.close()
        if executor is not None:
            executor.shutdown()
        if probe is not None:
            probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _terminate(launch)
        terminate_marked_processes({RUN_MARKER_ENV: run})
