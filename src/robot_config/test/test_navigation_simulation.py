"""Gazebo navigation E2E on the provider path (openspec lekiwi-gazebo-runtime-platform D10).

The production ``lekiwi_lidar`` navigation stage (FAST-LIO, the scan
converter, AMCL, the collision monitor, the navigation admission) runs
through generic launch on the provider's Gazebo platform in the shipped
``nav_room`` scene — staged from the ``lekiwi_navi_sim`` overlay. No test-only
odometry publishers, velocity relays or static map-to-odometry transforms:
the test bridges the simulator's world pose itself for its assertions.

Usage:
    colcon test --packages-select robot_config --pytest-args -k "test_navigation_simulation"
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy as QoSReliability
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

from ibrobot_msgs.msg import RuntimeStatus
from robot_runtime.testing.sim_ground_truth import open_ground_truth, terminate_partition_processes
from test.navigation_sim_support import load_staged_navigation_config, stage_navigation_config

POSITION_TOLERANCE = 0.25
GAZEBO_STARTUP_TIMEOUT = 240.0
NAV2_SETTLE_TIME = 5.0
NAV_GOAL_TIMEOUT = 180.0
STOP_PLANAR_LIMIT = 0.02
STOP_YAW_RATE_LIMIT = 0.05
NAV_TEST_PROFILE_ENV = "NAV_TEST_PROFILE"
NAV2_TEST_PARAMS_FILE_ENV = "NAV2_TEST_PARAMS_FILE"

WORLD_NAME = "demo"  # the scene template's world name
MODEL_NAME = "lekiwi_robot"  # the runtime's model name in the simulator

SPAWN_X = -1.5
SPAWN_Y = -1.5
POINT_A_X = 1.0
POINT_A_Y = 1.0


# ---------------------------------------------------------------------------
# Gazebo-free: the staged overlay resolves to the intended navigation stage
# ---------------------------------------------------------------------------


def test_staged_navigation_overlay_resolves_the_production_stage(tmp_path):
    config = load_staged_navigation_config(stage_navigation_config(tmp_path))

    # the provider path with the production navigation stage selected
    assert config["runtime"]["provider"] == "lekiwi_robot"
    assert config["runtime"]["profile"] == "lekiwi_lidar"
    assert config["nav_stage"] == "navigation"

    # the platform and the scene with its spawn (inside the stage)
    assert config["simulation"]["platform"] == "gazebo"
    assert config["simulation"]["scene"] == "nav_room"
    assert config["simulation"]["robot_spawn"] == {"x": SPAWN_X, "y": SPAWN_Y, "z": 0.05, "yaw": 0.0}

    # the navigation stage stays production: FAST-LIO, admission, collision
    # monitor — only the map/params/localisation policy are simulated
    navigation = config["navigation"]
    assert navigation["fast_lio"]["enabled"] is True
    assert navigation["open_gate_on_startup"] is True
    assert navigation["nav2_bringup"]["use_amcl"] is True
    assert navigation["nav2_bringup"]["auto_global_localization"] is False
    assert navigation["nav2_bringup"]["dyn_avoid_enabled"] is True
    assert navigation["nav2_bringup"]["map_file"].endswith("maps/sim_map.yaml")
    assert navigation["nav2_bringup"]["params_file"].endswith("nav2_params_sim.yaml")
    # no static map -> odom is composed anywhere in the config
    for static_tf in navigation.get("static_tfs", []):
        assert static_tf.get("child_frame") != "odom" or static_tf.get("parent_frame") != "map"

    # voice/embodied stay off inside the stage
    assert (config.get("voice_asr") or {}).get("enabled") is False
    assert (config.get("embodied") or {}).get("enabled") is False

    # the Nav2 simulation parameters: AMCL starts at the spawn pose and is
    # the single map -> odom publisher
    from robot_config.loader import resolve_ros_path

    params = yaml.safe_load(
        Path(resolve_ros_path(navigation["nav2_bringup"]["params_file"])).read_text(encoding="utf-8")
    )
    amcl = params["amcl"]["ros__parameters"]
    assert amcl["set_initial_pose"] is True
    assert amcl["initial_pose"]["x"] == SPAWN_X and amcl["initial_pose"]["y"] == SPAWN_Y
    assert amcl["tf_broadcast"] is True
    assert params["bt_navigator"]["ros__parameters"]["use_sim_time"] is True


# ---------------------------------------------------------------------------
# The live Gazebo run
# ---------------------------------------------------------------------------


def _gazebo_available() -> bool:
    for cmd in (["ign", "gazebo", "--version"], ["gz", "sim", "--version"]):
        try:
            if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _is_openeuler() -> bool:
    try:
        with open("/etc/os-release", encoding="utf-8") as os_release:
            return any(line.strip().lower() == "id=openeuler" for line in os_release)
    except OSError:
        return False


def _select_nav2_test_profile():
    requested = os.environ.get(NAV_TEST_PROFILE_ENV, "").strip().lower()
    if requested:
        if requested not in {"full", "minimal"}:
            pytest.skip(f"Unsupported {NAV_TEST_PROFILE_ENV}={requested!r}; expected 'full' or 'minimal'")
        return requested
    if _is_openeuler():
        pytest.skip(
            "openEuler minimal E2E profile: Gazebo navigation simulation is off; "
            f"set {NAV_TEST_PROFILE_ENV}=full to run it"
        )
    return "full"


class NavigationProbe(Node):
    def __init__(self):
        super().__init__("robot_config_navigation_probe", automatically_declare_parameters_from_overrides=False)
        # the navigation world runs on the simulation clock; goal stamps must
        # carry sim time or Nav2's transforms reject them as stale
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
        self.status: RuntimeStatus | None = None
        self.gate_enabled: Bool | None = None
        self.tf_frames: list[TFMessage] = []
        self.tf_static_frames: list[TFMessage] = []
        self.scans = []
        self.filtered_odometry = []
        self.create_subscription(RuntimeStatus, "/runtime_status", self._status, 10)
        self.create_subscription(Bool, "/motion_mode/base_navigation_enabled", self._gate, 10)
        self.create_subscription(TFMessage, "/tf", self.tf_frames.append, 50)
        self.create_subscription(TFMessage, "/tf_static", self.tf_static_frames.append, 10)
        # the derived scan is a best-effort sensor stream (the advertised QoS);
        # a reliable subscription is incompatible and receives nothing
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scans.append,
            QoSProfile(depth=10, reliability=QoSReliability.BEST_EFFORT),
        )
        self.create_subscription(Odometry, "/odometry/filtered", self.filtered_odometry.append, 10)
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def _status(self, msg: RuntimeStatus) -> None:
        self.status = msg

    def _gate(self, msg: Bool) -> None:
        self.gate_enabled = msg


def _wait_for(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    print(f"[navigation-sim] TIMEOUT waiting for {what}", flush=True)
    return False


def _send_nav_goal(probe: NavigationProbe, x, y, theta, timeout=NAV_GOAL_TIMEOUT):
    client = probe.nav_to_pose_client
    if not client.wait_for_server(timeout_sec=10.0):
        return None
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "map"
    # zero stamp: the pose is valid at the latest common transform time, so
    # small sim-clock skews between the probe and Nav2's TF buffers cannot
    # reject the goal as past/future extrapolation
    goal.pose.header.stamp = Time()
    goal.pose.pose.position.x = x
    goal.pose.pose.position.y = y
    goal.pose.pose.orientation.z = math.sin(theta / 2.0)
    goal.pose.pose.orientation.w = math.cos(theta / 2.0)
    future = client.send_goal_async(goal)
    deadline = time.monotonic() + 10.0
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not future.done():
        return None
    goal_handle = future.result()
    if goal_handle is None or not goal_handle.accepted:
        return None
    result_future = goal_handle.get_result_async()
    deadline = time.monotonic() + timeout
    while not result_future.done() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not result_future.done():
        goal_handle.cancel_goal()
        return None
    return result_future.result().status


@pytest.fixture(scope="module")
def gazebo_env():
    if _select_nav2_test_profile() == "minimal":
        pytest.skip(
            "Gazebo navigation simulation is disabled by the minimal E2E profile; "
            f"set {NAV_TEST_PROFILE_ENV}=full to run it."
        )
    if not _gazebo_available():
        pytest.skip("Gazebo not available; skipping navigation simulation tests")

    partition = f"lekiwi_nav_{uuid.uuid4().hex[:8]}"
    params_override = os.environ.get(NAV2_TEST_PARAMS_FILE_ENV, "").strip()
    if params_override and not Path(params_override).expanduser().is_file():
        pytest.fail(f"{NAV2_TEST_PARAMS_FILE_ENV} does not exist: {params_override}")

    stage_dir = Path(tempfile.mkdtemp(prefix="robot_config_nav_stage_"))
    config_path = stage_navigation_config(stage_dir, nav2_params_override=params_override)
    log_dir = Path(tempfile.mkdtemp(prefix="robot_config_nav_sim_"))
    print(f"[navigation-sim] staged config: {config_path}; logs: {log_dir}", flush=True)

    environment = dict(os.environ)
    # both partition variables: Fortress (ign, gz-transport11) reads only
    # IGN_PARTITION, newer gz reads GZ_PARTITION — set both to isolate this
    # run's gz bus on either stack
    environment.update(
        {
            "IBROBOT_GAZEBO_HEADLESS": "1",
            "GZ_PARTITION": partition,
            "IGN_PARTITION": partition,
            "RCUTILS_COLORIZED_OUTPUT": "0",
        }
    )

    launch_log = (log_dir / "robot_launch.log").open("w", encoding="utf-8")
    launch = subprocess.Popen(
        [
            "ros2",
            "launch",
            "robot_config",
            "robot.launch.py",
            f"config_path:={config_path}",
            "use_sim:=true",
            "sim_platform:=gazebo",
            "nav_stage:=navigation",
        ],
        stdout=launch_log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )

    ground_truth = None
    probe: NavigationProbe | None = None
    executor = None
    try:
        rclpy.init()
        probe = NavigationProbe()
        executor = MultiThreadedExecutor()
        executor.add_node(probe)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        def runtime_ready():
            return (
                probe.status is not None
                and probe.status.lifecycle == "ACTIVE"
                and probe.status.active_mode == "base_navigation"
                and probe.gate_enabled is not None
                and probe.gate_enabled.data is True
            )

        assert _wait_for(
            runtime_ready, GAZEBO_STARTUP_TIMEOUT, "runtime ACTIVE in base_navigation with the gate open (admission)"
        ), "navigation admission did not complete; see the launch log"
        assert _wait_for(probe.nav_to_pose_client.wait_for_server, GAZEBO_STARTUP_TIMEOUT, "the Nav2 action server")

        # the full perception chain before any goal: FAST-LIO odometry and
        # the derived scan flowing on the simulation clock (AMCL's first
        # map -> odom alone is only its initial pose, published before
        # FAST-LIO is ready — the controller then aborts on a frozen
        # transform)
        assert _wait_for(
            lambda: len(probe.filtered_odometry) >= 5, GAZEBO_STARTUP_TIMEOUT, "FAST-LIO odometry on /odometry/filtered"
        ), f"FAST-LIO produced {len(probe.filtered_odometry)} odometry messages; see the launch log"
        assert _wait_for(lambda: len(probe.scans) >= 5, GAZEBO_STARTUP_TIMEOUT, "the derived scan on /scan")

        # the launch env (with BOTH partition variables — Fortress reads only
        # IGN_PARTITION) is passed so the echo joins the same isolated gz bus
        # as the simulator it observes
        ground_truth = open_ground_truth("gazebo", world=WORLD_NAME, model=MODEL_NAME, env=environment).start()
        assert _wait_for(ground_truth.position, 30.0, "world-pose samples"), f"/world/{WORLD_NAME}/pose/info"
        time.sleep(NAV2_SETTLE_TIME)

        yield {"probe": probe, "ground_truth": ground_truth, "log_dir": log_dir, "launch": launch}
    finally:
        if ground_truth is not None:
            ground_truth.close()
        if probe is not None:
            probe.destroy_node()
        if executor is not None:
            executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(launch.pid), signal.SIGINT)
        try:
            launch.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(launch.pid), signal.SIGKILL)
            launch.wait(timeout=5)
        terminate_partition_processes(partition)
        launch_log.close()


def _assert_ground_truth_at(ground_truth, x, y, what):
    position = ground_truth.position()
    assert position is not None, f"{what}: no ground-truth position"
    distance = math.hypot(position[0] - x, position[1] - y)
    print(f"[navigation-sim] {what}: ground truth {position} vs ({x}, {y}) -> {distance:.3f} m", flush=True)
    assert distance < POSITION_TOLERANCE, f"{what}: ground truth {distance:.3f} m from the goal exceeds 0.25 m"


def test_navigation_simulation(gazebo_env):
    """Reach a goal, there-and-back, and a cancel-based stop (D10).

    Goals go directly to Nav2's ``navigate_to_pose`` action; the stop cancels
    an active goal and asserts the D3 stop thresholds on the test's own
    ground-truth bridge; the map -> odom transform has exactly one publisher
    (AMCL) and never a static one.
    """
    probe = gazebo_env["probe"]
    ground_truth = gazebo_env["ground_truth"]

    # exactly one map -> odom publisher, the map localizer (AMCL): no static
    # transform is composed and only AMCL joins the runtime's own TF sources
    publishers = {info.node_name for info in probe.get_publishers_info_by_topic("/tf")}
    print(f"[navigation-sim] /tf publishers: {sorted(publishers)}", flush=True)
    assert "amcl" in publishers
    static_map_odom = [
        transform
        for message in probe.tf_static_frames
        for transform in message.transforms
        if transform.header.frame_id == "map" and transform.child_frame_id == "odom"
    ]
    assert not static_map_odom, "no static map -> odom may be composed in simulation"

    def _map_odom_published():
        return any(
            transform.header.frame_id == "map" and transform.child_frame_id == "odom"
            for message in probe.tf_frames
            for transform in message.transforms
        )

    # AMCL's first update needs the full chain (FAST-LIO odometry, the
    # derived scan): wait for it rather than asserting a cold graph
    assert _wait_for(_map_odom_published, 90.0, "AMCL's first map -> odom transform"), (
        "AMCL must publish the map -> odom transform"
    )

    # reach A
    status = _send_nav_goal(probe, POINT_A_X, POINT_A_Y, 0.0)
    assert status == GoalStatus.STATUS_SUCCEEDED, f"navigation to point A failed with status {status}"
    _assert_ground_truth_at(ground_truth, POINT_A_X, POINT_A_Y, "point A")

    # there-and-back: back to the spawn origin
    status = _send_nav_goal(probe, SPAWN_X, SPAWN_Y, 0.0)
    assert status == GoalStatus.STATUS_SUCCEEDED, f"navigation back to the origin failed with status {status}"
    _assert_ground_truth_at(ground_truth, SPAWN_X, SPAWN_Y, "origin")

    # stop: cancel an active goal, then the D3 thresholds within 1 s (sim)
    client = probe.nav_to_pose_client
    assert client.wait_for_server(timeout_sec=10.0)
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "map"
    goal.pose.header.stamp = Time()
    goal.pose.pose.position.x = POINT_A_X
    goal.pose.pose.position.y = POINT_A_Y
    future = client.send_goal_async(goal)
    deadline = time.monotonic() + 10.0
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    goal_handle = future.result()
    assert goal_handle is not None and goal_handle.accepted, "the stop-case goal was not accepted"

    def _moving():
        now = ground_truth.latest_time()
        if now is None:
            return False
        window = ground_truth.window(now - 0.5, now)
        speeds = [
            math.hypot(current.x - previous.x, current.y - previous.y) / (current.stamp_sec - previous.stamp_sec)
            for previous, current in zip(window, window[1:], strict=False)
            if current.stamp_sec - previous.stamp_sec > 1e-6
        ]
        return bool(speeds) and max(speeds) > 0.02

    assert _wait_for(_moving, 60.0, "the robot to start moving towards the stop-case goal")

    stop_start = ground_truth.latest_time()
    goal_handle.cancel_goal()
    assert _wait_for(lambda: ground_truth.latest_time() - stop_start >= 1.0, 30.0, "1 s of simulation time")

    settle = ground_truth.window(stop_start + 0.7, stop_start + 1.0)
    speeds, yaw_rates = [], []
    for previous, current in zip(settle, settle[1:], strict=False):
        dt = current.stamp_sec - previous.stamp_sec
        if dt <= 1e-6:
            continue
        speeds.append(math.hypot(current.x - previous.x, current.y - previous.y) / dt)
        yaw_rates.append(
            abs(math.atan2(math.sin(current.yaw - previous.yaw), math.cos(current.yaw - previous.yaw))) / dt
        )
    planar = max(speeds, default=0.0)
    yaw = max(yaw_rates, default=0.0)
    print(f"[navigation-sim] stop: max planar {planar:.4f} m/s, max |yaw rate| {yaw:.4f} rad/s", flush=True)
    assert planar <= STOP_PLANAR_LIMIT, f"stop: planar speed {planar:.4f} m/s exceeds 0.02 within 1 s"
    assert yaw <= STOP_YAW_RATE_LIMIT, f"stop: yaw rate {yaw:.4f} rad/s exceeds 0.05 within 1 s"
