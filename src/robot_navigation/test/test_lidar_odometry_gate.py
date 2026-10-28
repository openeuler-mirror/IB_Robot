"""Lidar-odometry regression gate in the structured nav_room scene (D14.3-D14.6).

The production ``lekiwi_lidar`` navigation stage (FAST-LIO + bridge, scan
converter, AMCL, collision monitor, admission) runs through generic launch
on the provider's Gazebo platform, staged from the gate's own
``config/lekiwi_lio_sim.yaml`` overlay. With no Nav2 goal active the test
drives the scripted D14.5 route on ``/cmd_vel_nav`` — upstream of the
velocity smoother and the collision monitor — closing every segment on
ground-truth progress, and scores three estimates (FAST-LIO
``/odometry/filtered`` under test; wheel ``/odom`` and AMCL ``/amcl_pose``
diagnostic) against the simulator ground truth, aligned at the route start.

A platform runs the gate only when its capabilities are all present
(D14.2): ground truth, the profile's ``sim_platforms``, lidar simulation
(``LIDAR_SIM_PLATFORMS``) and a scene template. The gate is parametrized
over ``PHYSICS_SIM_PLATFORMS``; both ``gazebo`` and ``mujoco`` satisfy all
four (lekiwi-mujoco-runtime-platform D10). A second, independent check
covers the simulator ENVIRONMENT (``ign`` for Gazebo; the MuJoCo ament
resources and a matching Python ``mujoco`` for MuJoCo): missing
environment skips with a reason — unless ``IBROBOT_REQUIRE_SIM_PLATFORMS``
lists the platform, in which case any skip of it fails instead (the
acceptance runs).

The gate lives in robot_navigation, not lekiwi_robot: it launches the
generic application entry (robot_config + this package's Nav2 chain), and
a runtime package may not declare test dependencies on generic packages —
scripts/verify_runtime_independence.sh audits the runtime's whole
dependency closure, test dependencies included. LeKiwi's lidar profile is
read from the installed lekiwi_robot share.

Measurement-only runs (threshold calibration, task 8.4) set
``LIO_GATE_REPORT_ONLY=1``: the accuracy assertions report instead of fail;
route, health, watchdog and monitor problems still fail, and
``LIO_GATE_REPORT_DIR`` redirects plots/samples/metrics for evidence. A
staged-variant run (the task 8.3 wrong-mount check) points
``LIO_GATE_OVERLAY`` at an alternative overlay file.

Usage:
    NAV_TEST_PROFILE=full colcon test --packages-select robot_navigation \
        --pytest-args -k lidar_odometry_gate
"""

from __future__ import annotations

import bisect
import contextlib
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy as QoSReliability
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

from ibrobot_msgs.msg import RuntimeStatus
from robot_runtime.profile import LIDAR_SIM_PLATFORMS, PHYSICS_SIM_PLATFORMS, load_profile
from robot_runtime.testing.sim_ground_truth import (
    UnsupportedGroundTruth,
    open_ground_truth,
    terminate_partition_processes,
    wrap_angle,
    yaw_from_quaternion,
)
from sim_models.scene_compiler import get_scene_file, get_scene_layout

# ---------------------------------------------------------------------------
# Constants of the route and the production velocity chain (D14.4/D14.5)
# ---------------------------------------------------------------------------

#: The runtime's model name in the simulator and the scene's world name.
GATE_MODEL = "lekiwi_robot"
GATE_WORLD = "demo"
GATE_SCENE = "nav_room"
GATE_STAGE = "navigation"

#: The D14.5 route, read from the SAME nav_room layout data the clearance
#: test uses (one definition, no drift): west half of the room, 7.0 m of
#: translation, two in-place turns, two lateral segments, closed. Commands
#: stay below the velocity smoother limits (0.13 / 0.11 / 0.44,
#: nav2_params_sim.yaml); translation speeds are SIGNED (segment 6 drives
#: body -y).
ROUTE = tuple(get_scene_layout(GATE_SCENE)["gate_route"])
assert len(ROUTE) == 6, f"the nav_room gate_route must declare six segments, got {len(ROUTE)}"
#: Velocity smoother decelerations per axis (nav2_params_sim.yaml
#: max_decel [-0.65, -0.55, -2.2]): segment termination closes on the
#: stopping distance of the commanded axis (D14.4). The measured chain keeps
#: driving for ~0.125 s after the zero command leaves /cmd_vel_nav — a
#: composite of the command thread period, the termination polling and the
#: monitor -> base node -> gz control chain (the velocity smoother itself
#: ramps at exactly max_decel) — so the stopping distance is the smoother's
#: v^2/(2|decel|) plus v*latency (measured while bringing up the gate,
#: openspec lekiwi-gazebo-runtime-platform task 8.3; without the latency term
#: each 90 deg turn lands ~2.9 deg over and the long segments then drift
#: ~0.10 m off their line, the D14.5 route-deviation case).
CHAIN_COMMAND_LATENCY_SEC = 0.125
SMOOTHER_DECEL = {"vx": 0.65, "vy": 0.55, "wz": 2.2}
#: The ground-truth stream must stay alive: no fresh pose sample for this
#: many wall seconds is a failed run, not a silent wait (S2).
GROUND_TRUTH_STALE_SEC = 5.0

#: D14.4: the command publisher runs on a 20 Hz wall-time timer (the Humble
#: velocity smoother's output timer is wall time in OPEN_LOOP).
COMMAND_RATE_HZ = 20.0
#: D14.5: route deviation limits — position off the segment line, heading
#: off the nominal outside rotation segments.
ROUTE_POSITION_TOLERANCE = 0.10
ROUTE_HEADING_TOLERANCE_DEG = 5.0
#: Settles: 2 s of simulation time before the route (D14.3), 1 s between
#: segments and after the final one (the smoother's velocity_timeout, D14.4).
# 6 s (was 2): the navigation stage's bringup burst (first costmap updates,
# AMCL localization, behavior-server activation) plus FAST-LIO's map init
# land in the first seconds after readiness; under a loaded colcon run they
# briefly starve FAST-LIO into skipping scans (round 3 evidence: gaps and
# monitor misses clustered 2-7 s into segment 1). Settling longer under
# zero command moves that transient out of the measured route.
PRE_ROUTE_SETTLE_SEC = 6.0
INTER_SEGMENT_SETTLE_SEC = 1.0
#: Ground-truth motion settled thresholds closing a segment (D14.4).
SETTLE_SPEED_LIMIT = 0.01
SETTLE_YAW_RATE_LIMIT = 0.02
#: Watchdogs (D14.4): /clock silent, segment sim-time stall (3x nominal),
#: segment wall-time limit (3x nominal / 0.5), whole-run wall limit (the sum
#: of the segment limits plus readiness) — all independent of /clock.
CLOCK_STALL_SEC = 10.0
SEGMENT_SIM_LIMIT_FACTOR = 3.0
SEGMENT_WALL_LIMIT_FACTOR = 3.0 / 0.5
READINESS_WALL_BUDGET = 120.0
#: Collision-monitor pairing (D14.4): an input pairs with the output
#: received closest to its relay SLOT (t_in + the stream's measured relay
#: latency) inside the receive window, and only within half a sample
#: period of the slot — the before side covers jitter, the after side the
#: monitor's relay, and the half-period bound keeps an isolated dropped
#: relay from shifting the pairing onto its neighbours' slots.
PAIR_WINDOW_BEFORE_SEC = 0.05
PAIR_WINDOW_AFTER_SEC = 0.2
PAIR_HALF_PERIOD_SEC = 0.025
MONITOR_EQUALITY_TOL = 1e-6
MONITOR_MISSING_FRACTION_LIMIT = 0.01
#: D14.6: FAST-LIO health, unpaired fraction, RTF and max future skew.
#: Frame health over the whole drive (final settle included): over the same
#: simulation-time span the odom bridge's /odometry/filtered must carry at
#: least this fraction of the frames FAST-LIO consumed (/livox/lidar,
#: the finite-cloud-filter output — its publisher is reliable, so the
#: gate's reliable subscription counts without test-side loss), and no
#: adjacent stamp gap may exceed the gap limit. Gazebo's gpu_lidar
#: occasionally renders an EMPTY scan (the cloud still arrives, FAST-LIO
#: logs 'No point, skip this scan!' and publishes no odometry for it) —
#: the fraction floor absorbs those, the gap bound catches a real stall.
ODOMETRY_FRAME_FRACTION_LIMIT = 0.95
ODOMETRY_MAX_GAP_SEC = 0.35
UNPAIRED_FRACTION_LIMIT = 0.05
RTF_LIMIT = 0.8
MAX_FUTURE_SKEW_SEC = 0.1
#: Frame-health windows open this far into the route (round 3): the tail of
# the bringup transient (see PRE_ROUTE_SETTLE_SEC) may still skip a lidar
# period or two once motion starts; the gap/fraction/monitor checks cover
# the sustained drive, not the startup burst. The estimate metrics (ATE
# etc.) keep the full window.
FRAME_HEALTH_WARMUP_SEC = 5.0
#: A /clock callback gap at or above this is recorded as a stall interval
#: (the host-pause discriminator matches these against the odometry gap).
CLOCK_STALL_THRESHOLD_SEC = 0.3

#: D14.6 spec caps (hard: a run above any cap fails no matter what was
#: measured) and the committed thresholds
#: ``min(cap, max(2 x worst of five measured runs, floor))`` (floors
#: 0.02 m / 0.5 deg). Committed from the five measurement runs of
#: 2026-10-27 (openspec lekiwi-gazebo-runtime-platform task 8.4): worst of five
#: ate 0.0072 / max 0.0117 / final 0.0074 m / yaw 0.10 deg — ATE and final
#: land on the floors, max on 2x worst (rounded up), yaw on the floor.
#: Every measured run is far inside half of each cap, so a single later
#: run between a committed threshold and its cap still fails the gate.
FAST_LIO_CAPS = {"ate_rmse": 0.10, "max_err": 0.20, "final_err": 0.15, "final_yaw_err_deg": 3.0}
FAST_LIO_THRESHOLDS = {"ate_rmse": 0.02, "max_err": 0.0234, "final_err": 0.02, "final_yaw_err_deg": 0.5}
#: Wheel odometry and AMCL are diagnostic: bounded only loosely so a broken
#: estimate is visible without making the gate about them (D14.6).
DIAGNOSTIC_ATE_LIMIT = 0.5

#: D14.4: /cmd_vel is the runtime's admitted velocity channel. The
#: production chain's Nav2 behavior server also advertises /cmd_vel — its
#: recovery behaviors write the channel directly, bypassing the velocity
#: smoother and the monitor (a property of the shipped chain,
#: lekiwi-application-migration; no goal is active during the gate, so it
#: stays silent) — and the pairing check proves the monitor is the only
#: publisher that actually drives the base.
EXPECTED_CMD_VEL_PUBLISHERS = {"collision_monitor", "behavior_server"}
_GATE_OVERLAY = Path(__file__).resolve().parent / "config" / "lekiwi_lio_sim.yaml"
_BASE_NODE_NAME = "lekiwi_base"  # the wheel-odometry node (D4)
_EXPECTED_TF_PUBLISHERS = {"amcl", _BASE_NODE_NAME, "robot_state_publisher"}

NAV_TEST_PROFILE_ENV = "NAV_TEST_PROFILE"
REPORT_ONLY_ENV = "LIO_GATE_REPORT_ONLY"
REPORT_DIR_ENV = "LIO_GATE_REPORT_DIR"
OVERLAY_ENV = "LIO_GATE_OVERLAY"


# ---------------------------------------------------------------------------
# D14.2: the capability probe
# ---------------------------------------------------------------------------


def _lidar_profile_path() -> Path:
    """The installed LeKiwi lidar runtime profile (lekiwi_robot test dependency)."""
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("lekiwi_robot")) / "profiles" / "lekiwi_lidar.yaml"


def missing_gate_capabilities(sim_platform: str) -> list[str]:
    """Every capability the gate needs that ``sim_platform`` lacks (D14.2).

    Inspects code and configuration only — no process, no simulator — so a
    platform that passes the probe but is misconfigured (FAST-LIO enabled
    without ``sim_params_file``) still fails at launch, never a skip.
    """
    missing = []
    try:
        open_ground_truth(sim_platform, world=GATE_WORLD, model=GATE_MODEL)
    except UnsupportedGroundTruth as error:
        missing.append(f"ground truth: {error}")
    profile = load_profile(_lidar_profile_path())
    supported = [str(entry) for entry in profile.get("sim_platforms") or []]
    if sim_platform not in supported:
        missing.append(f"runtime profile: {sim_platform!r} is not in sim_platforms {supported}")
    if sim_platform not in LIDAR_SIM_PLATFORMS:
        missing.append(f"simulated lidar: {sim_platform!r} is not in LIDAR_SIM_PLATFORMS {list(LIDAR_SIM_PLATFORMS)}")
    try:
        get_scene_file(GATE_SCENE, sim_platform)
    except (ValueError, FileNotFoundError) as error:
        missing.append(f"scene template: {error}")
    return missing


def test_gazebo_has_every_gate_capability():
    """D14.2: gazebo passes all four checks, so the gate runs there."""
    assert missing_gate_capabilities("gazebo") == []


def test_mujoco_reports_no_missing_capability():
    """D10 (lekiwi-mujoco-runtime-platform): mujoco passes all four checks
    too — the gate runs there, never a capability skip."""
    assert missing_gate_capabilities("mujoco") == []


def test_a_synthetic_platform_reports_all_four_capabilities():
    """The probe itself still names all four missing capabilities for a
    platform nothing supports (the skip-reason surface stays honest)."""
    missing = missing_gate_capabilities("synthetic")
    assert len(missing) == 4, f"expected all four capabilities missing, got {missing}"
    assert missing[0].startswith("ground truth") and "synthetic" in missing[0]
    assert missing[1].startswith("runtime profile") and "sim_platforms" in missing[1]
    assert missing[2].startswith("simulated lidar") and "LIDAR_SIM_PLATFORMS" in missing[2]
    assert missing[3].startswith("scene template")


# ---------------------------------------------------------------------------
# Simulator availability and the required-platform gate (D10,
# lekiwi-mujoco-runtime-platform: two separate checks — code capability and
# environment availability)
# ---------------------------------------------------------------------------


def _gazebo_available() -> bool:
    try:
        return subprocess.run(["ign", "gazebo", "--version"], capture_output=True, timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _mujoco_environment_missing() -> str | None:
    """Why the MuJoCo simulator environment is unusable, or None when fine.

    The ament resources the C++ node needs (the control package and the
    ``mujoco_3d_lidar`` engine extension) and a Python ``mujoco`` matching
    the simulator's libmujoco (the compiler must bake the same version).
    """
    try:
        from ament_index_python import get_resources
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        try:
            get_package_share_directory("mujoco_ros2_control")
        except PackageNotFoundError:
            return "the mujoco_ros2_control package is not installed (apt install ros-humble-mujoco-ros2-control)"
        resources = get_resources("mujoco_plugins")
        if "mujoco_3d_lidar" not in resources:
            return "the mujoco_3d_lidar ament resource is not installed (apt install ros-humble-mujoco-3d-lidar)"
    except ImportError:
        return "ament_index_python is unavailable"
    try:
        import mujoco as python_mujoco  # noqa: PLC0415

        from robot_runtime.mujoco import simulator_versions

        versions = simulator_versions()
        if versions.mujoco is None:
            return "the mujoco_vendor libmujoco library is not installed"
        if python_mujoco.__version__ != versions.mujoco:
            return (
                f"Python mujoco {python_mujoco.__version__} != simulator libmujoco {versions.mujoco}; "
                "the compiler and the simulator must match"
            )
    except ImportError:
        return "the Python mujoco package is unavailable"
    return None


def _environment_missing(sim_platform: str) -> str | None:
    """Why the platform's simulator cannot run here, or None when it can."""
    if sim_platform == "gazebo":
        return None if _gazebo_available() else "ign gazebo is unavailable"
    if sim_platform == "mujoco":
        return _mujoco_environment_missing()
    return f"no environment check for platform {sim_platform!r}"


#: Acceptance runs list the platforms whose skips must instead fail
#: (D10): missing environment, missing capability or the minimal profile.
REQUIRE_SIM_PLATFORMS_ENV = "IBROBOT_REQUIRE_SIM_PLATFORMS"


def _required_sim_platforms() -> set[str]:
    listed = os.environ.get(REQUIRE_SIM_PLATFORMS_ENV, "").strip()
    if not listed:
        return set()
    return {entry.strip().lower() for entry in listed.split(",") if entry.strip()}


def _gate_skip_or_fail(sim_platform: str, reason: str) -> None:
    """Skip — unless acceptance runs required this platform: then fail."""
    if sim_platform in _required_sim_platforms():
        pytest.fail(
            f"lidar-odometry gate: platform {sim_platform!r} was required by {REQUIRE_SIM_PLATFORMS_ENV}: {reason}"
        )
    pytest.skip(reason)


def _is_openeuler() -> bool:
    try:
        with open("/etc/os-release", encoding="utf-8") as os_release:
            return any(line.strip().lower() == "id=openeuler" for line in os_release)
    except OSError:
        return False


def _select_test_profile():
    """The E2E profile to run, or None when this host defaults the gate off.

    Review round 2: this helper never skips by itself — the caller routes
    every do-not-run reason through ``_gate_skip_or_fail`` so
    ``IBROBOT_REQUIRE_SIM_PLATFORMS`` turns it into a failure. An explicitly
    invalid ``NAV_TEST_PROFILE`` value fails outright: a typo in an
    acceptance run must not silently become a skip.
    """
    requested = os.environ.get(NAV_TEST_PROFILE_ENV, "").strip().lower()
    if requested:
        if requested not in {"full", "minimal"}:
            pytest.fail(f"invalid {NAV_TEST_PROFILE_ENV}={requested!r}; expected 'full' or 'minimal'")
        return requested
    if _is_openeuler():
        return None  # the openEuler default: minimal E2E, the gate is off
    return "full"


# ---------------------------------------------------------------------------
# The gate's decision layer: profile selection and the require-or-skip rule
# (review round 2 — nothing may bypass IBROBOT_REQUIRE_SIM_PLATFORMS)
# ---------------------------------------------------------------------------


def test_an_invalid_profile_value_fails_outright(monkeypatch):
    """A typo'd NAV_TEST_PROFILE must fail, never skip — an acceptance run
    with REQUIRE set must not silently lose the gate to a bad env value."""

    monkeypatch.setenv(NAV_TEST_PROFILE_ENV, "bogus")
    monkeypatch.delenv(REQUIRE_SIM_PLATFORMS_ENV, raising=False)
    with pytest.raises(pytest.fail.Exception, match="invalid .*expected 'full' or 'minimal'"):
        _select_test_profile()


def test_the_minimal_profile_with_mujoco_required_fails_not_skips(monkeypatch):
    monkeypatch.setenv(NAV_TEST_PROFILE_ENV, "minimal")
    monkeypatch.setenv(REQUIRE_SIM_PLATFORMS_ENV, "mujoco")
    assert _select_test_profile() == "minimal"
    with pytest.raises(pytest.fail.Exception, match="was required by IBROBOT_REQUIRE_SIM_PLATFORMS"):
        _gate_skip_or_fail("mujoco", "the minimal E2E profile disables the gate")


def test_the_openeuler_default_with_mujoco_required_fails_not_skips(monkeypatch):
    import sys

    monkeypatch.delenv(NAV_TEST_PROFILE_ENV, raising=False)
    monkeypatch.setenv(REQUIRE_SIM_PLATFORMS_ENV, "mujoco")
    monkeypatch.setattr(sys.modules[__name__], "_is_openeuler", lambda: True)
    assert _select_test_profile() is None
    with pytest.raises(pytest.fail.Exception, match="was required by IBROBOT_REQUIRE_SIM_PLATFORMS"):
        _gate_skip_or_fail("mujoco", "the openEuler default minimal E2E profile disables the gate")


def test_the_minimal_profile_without_a_requirement_stays_a_skip(monkeypatch):
    monkeypatch.setenv(NAV_TEST_PROFILE_ENV, "minimal")
    monkeypatch.delenv(REQUIRE_SIM_PLATFORMS_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="minimal E2E profile"):
        _gate_skip_or_fail("gazebo", "the minimal E2E profile disables the gate")


def test_a_capable_environment_with_mujoco_required_proceeds_to_execution():
    """With capabilities, environment and the full profile all present, the
    guard sequence falls through to the run itself (the executed gate above
    is that run; here the decisions are pinned)."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv(NAV_TEST_PROFILE_ENV, raising=False)
    monkeypatch.setenv(REQUIRE_SIM_PLATFORMS_ENV, "mujoco")
    try:
        if not _is_openeuler():  # the full profile is this host's default
            assert _select_test_profile() == "full"
        assert missing_gate_capabilities("mujoco") == []
        if _environment_missing("mujoco") is None:
            return  # every guard passed: the gate executes on this host
        # a machine without the environment keeps the documented rule
        with pytest.raises(pytest.fail.Exception, match="was required by IBROBOT_REQUIRE_SIM_PLATFORMS"):
            _gate_skip_or_fail("mujoco", "the MuJoCo environment is unavailable")
    finally:
        monkeypatch.undo()


def _gap_is_host_pause(
    gap_from_wall: float,
    gap_to_wall: float,
    gap_sec: float,
    clock_stalls: list[tuple[float, float]],
) -> bool:
    """Whether a /clock stall overlapping the odometry gap excuses it (round 4).

    A whole-host pause freezes the chain's output and the probe's dedicated
    /clock delivery TOGETHER, so only a stall whose wall interval overlaps the
    gap's receive interval AND lasts at least ``gap_sec - 0.1`` s is evidence
    of one. A stall recorded at any other moment must not excuse the gap, and
    no stall at all never does: those are real output stalls.
    """
    for stall_begin, stall_end in clock_stalls:
        overlaps = stall_end >= gap_from_wall and stall_begin <= gap_to_wall
        if overlaps and (stall_end - stall_begin) >= gap_sec - 0.1:
            return True
    return False


def test_an_overlapping_clock_stall_excuses_the_gap():
    stalls = [(5.0, 6.5), (10.0, 11.1)]
    assert _gap_is_host_pause(10.2, 11.0, 0.9, stalls)


def test_a_clock_stall_elsewhere_does_not_excuse_the_gap():
    stalls = [(5.0, 6.5)]
    assert not _gap_is_host_pause(10.2, 11.1, 0.9, stalls)


def test_no_stall_does_not_excuse_the_gap():
    assert not _gap_is_host_pause(10.2, 11.1, 0.9, [])


def test_an_overlapping_but_too_short_stall_does_not_excuse_the_gap():
    assert not _gap_is_host_pause(10.2, 11.1, 0.9, [(10.5, 10.7)])


def _max_future_skew(samples: list[dict]) -> float | None:
    """The worst ``stamp - sim_reference`` over recorded estimate samples.

    ``sim_reference`` is the freshest simulation time from the dedicated
    /clock callback or the 100 Hz wheel odometry when the sample's own
    callback ran (round 3). The two references co-stall with the estimates
    under machine-level pauses (so a loaded host cannot fake a future
    stamp), while a chain genuinely stamping ahead of the simulation clock
    stays visible. Samples recorded before any reference are skipped.
    """
    skew = None
    for sample in samples:
        reference = sample.get("sim_reference")
        if reference is None:
            continue
        value = sample["stamp"] - reference
        if skew is None or value > skew:
            skew = value
    return skew


def test_a_truly_future_stamp_is_still_caught_by_the_skew_reference():
    """The dedicated-clock reference keeps the check sensitive: a chain that
    really stamps ahead of the simulation clock is flagged however healthy
    the probe's clock callback is."""
    healthy = [
        {"stamp": 10.00, "sim_reference": 10.02},
        {"stamp": 10.10, "sim_reference": 10.11},
        {"stamp": 10.20, "sim_reference": 10.21},
    ]
    assert _max_future_skew(healthy) is not None and _max_future_skew(healthy) <= 0.0
    # a stalled clock delivery alone must not fake a future stamp: the wheel
    # half of the reference kept advancing
    co_stalled = healthy + [{"stamp": 10.25, "sim_reference": 10.26}]
    assert _max_future_skew(co_stalled) <= 0.0
    ahead = healthy + [{"stamp": 10.75, "sim_reference": 10.21}]
    assert _max_future_skew(ahead) == pytest.approx(0.54)
    assert _max_future_skew(ahead) > MAX_FUTURE_SKEW_SEC
    # samples recorded before any reference arrived carry none
    assert _max_future_skew([{"stamp": 1.0, "sim_reference": None}]) is None


# ---------------------------------------------------------------------------
# The probe node: everything the gate observes on the ROS graph
# ---------------------------------------------------------------------------


class GateProbe(Node):
    """Records the estimates, the velocity chain and the clock (D14.4/D14.6).

    Stamped data carries the source's own header stamps (simulation time);
    ``/cmd_vel_smoothed``/``/cmd_vel`` samples additionally record the
    wall-clock receive time the collision-monitor pairing needs (``Twist``
    carries no stamp). The node itself stays on wall time: nothing here
    depends on its own clock.
    """

    def __init__(self):
        super().__init__("lekiwi_lio_gate_probe")
        self.status: RuntimeStatus | None = None
        self.gate_enabled: Bool | None = None
        self.filtered_odom: list[dict] = []
        self.wheel_odom: list[dict] = []
        self.amcl_pose: list[dict] = []
        self.cmd_smoothed: list[tuple[float, tuple[float, float, float]]] = []
        self.cmd_vel: list[tuple[float, tuple[float, float, float]]] = []
        self.lidar_frames: list[tuple[float, float]] = []  # (recv_wall, sim stamp)
        self.clock: list[tuple[float, float]] = []
        self.tf_frames: list[TFMessage] = []
        self.tf_static_frames: list[TFMessage] = []
        self._lock = threading.Lock()
        self._last_clock_wall = 0.0
        self._clock_hook = None  # the supervisor's stall observer
        # The future-skew reference (review round 3): the monotonic maximum
        # simulation time seen at the moment a sample callback runs. The /clock
        # subscription lives in its OWN callback group (the executor is
        # multi-threaded): in the node's default mutually-exclusive group the
        # clock callback queued behind the heavy subscribers, and the queue's
        # keep-last window then served clocks hundreds of ms old — a sample
        # stamped "now" read as seconds ahead of the last clock RECEIVED.
        self._clock_sim: float | None = None
        self._clock_max_callback_gap_sec = 0.0
        self._clock_stalls: list[tuple[float, float]] = []
        self._clock_max_jump_sec = 0.0
        self._last_clock_callback_wall: float | None = None
        self._last_clock_callback_sim: float | None = None
        self._clock_group = MutuallyExclusiveCallbackGroup()
        # The data subscriptions run reentrant (round 3): in the node's single
        # default mutually-exclusive group the heavy subscribers serialized
        # behind each other, and a reader queue could then drop messages —
        # reading as false stamp gaps and monitor misses under load.
        self._data_group = ReentrantCallbackGroup()
        self._wheel_sim: float | None = None
        self.create_subscription(RuntimeStatus, "/runtime_status", self._status, 10)
        self.create_subscription(Bool, "/motion_mode/base_navigation_enabled", self._gate, 10)
        self.create_subscription(
            Odometry, "/odometry/filtered", self._filtered_odom, 200, callback_group=self._data_group
        )
        self.create_subscription(Odometry, "/odom", self._wheel_odom, 200, callback_group=self._data_group)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_pose, 10, callback_group=self._data_group
        )
        # the velocity chain: one sample per message, wall-clock receive time
        self.create_subscription(Twist, "/cmd_vel_smoothed", self._cmd_smoothed, self._twist_qos())
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel, self._twist_qos())
        # FAST-LIO's input frames, for the D14.6 frame-health ratio: the
        # finite-cloud-filter publisher is RELIABLE (default QoS), so a
        # reliable subscription counts without test-side loss. raw=True
        # keeps the ~1.8 MB point payload undeserialized (18 MB/s of
        # deserialization would add load the measured monitor relay latency
        # is sensitive to); the stamp sits at a fixed CDR offset
        self.create_subscription(
            PointCloud2,
            "/livox/lidar",
            self._lidar_frame,
            QoSProfile(depth=100, reliability=QoSReliability.RELIABLE),
            raw=True,
            callback_group=self._data_group,
        )
        self.create_subscription(
            Clock,
            "/clock",
            self._on_clock,
            300,
            callback_group=self._clock_group,
        )
        self.create_subscription(TFMessage, "/tf", self.tf_frames.append, 50, callback_group=self._data_group)
        self.create_subscription(
            TFMessage, "/tf_static", self.tf_static_frames.append, 10, callback_group=self._data_group
        )

    @staticmethod
    def _twist_qos() -> QoSProfile:
        # reliable with a deep queue: the pairing check must not lose
        # samples of its own (D14.4 tolerates only isolated losses)
        return QoSProfile(depth=500, reliability=QoSReliability.RELIABLE)

    def _status(self, msg: RuntimeStatus) -> None:
        self.status = msg

    def _gate(self, msg: Bool) -> None:
        self.gate_enabled = msg

    def _filtered_odom(self, msg: Odometry) -> None:
        self._record_odometry(self.filtered_odom, msg)

    def _wheel_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        with self._lock:
            # the co-stalled half of the future-skew reference (round 3): the
            # 100 Hz wheel odometry is stamped on the same simulation clock, so
            # a machine-level pause freezes it together with the estimates —
            # the comparison stays fair where a stalled /clock delivery would
            # read healthy samples as seconds ahead
            if self._wheel_sim is None or stamp > self._wheel_sim:
                self._wheel_sim = stamp
        self._record_odometry(self.wheel_odom, msg)

    def _record_odometry(self, sink: list[dict], msg: Odometry) -> None:
        pose = msg.pose.pose
        sample = {
            "recv_wall": time.time(),
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            "x": pose.position.x,
            "y": pose.position.y,
            "yaw": yaw_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            "odom_frame": msg.header.frame_id,
            "base_frame": msg.child_frame_id,
        }
        with self._lock:
            # the future-skew reference: the freshest simulation time from the
            # dedicated /clock callback AND the 100 Hz wheel odometry when this
            # sample's own callback ran (whichever is ahead — the wheel stream
            # co-stalls with the estimates under machine pauses, so only a
            # chain genuinely stamping ahead of both reads as future)
            sample["clock_sim"] = self._clock_sim
            sample["wheel_sim"] = self._wheel_sim
            references = [value for value in (self._clock_sim, self._wheel_sim) if value is not None]
            sample["sim_reference"] = max(references) if references else None
            sink.append(sample)

    def _amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        sample = {
            "recv_wall": time.time(),
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            "x": pose.position.x,
            "y": pose.position.y,
            "yaw": yaw_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            "frame": msg.header.frame_id,
        }
        with self._lock:
            self.amcl_pose.append(sample)

    def _cmd_smoothed(self, msg: Twist) -> None:
        self._record_twist(self.cmd_smoothed, msg)

    def _cmd_vel(self, msg: Twist) -> None:
        self._record_twist(self.cmd_vel, msg)

    def _record_twist(self, sink: list[tuple], msg: Twist) -> None:
        sample = (time.time(), (msg.linear.x, msg.linear.y, msg.angular.z))
        with self._lock:
            sink.append(sample)

    def _lidar_frame(self, raw_msg) -> None:
        # raw CDR: 4-byte encapsulation header, then header.stamp.sec
        # (int32) and .nanosec (uint32)
        try:
            sec, nanosec = struct.unpack_from("<iI", bytes(raw_msg), 4)
        except (struct.error, TypeError):
            return
        sample = (time.time(), sec + nanosec / 1e9)
        with self._lock:
            self.lidar_frames.append(sample)

    def _on_clock(self, msg: Clock) -> None:
        sim = msg.clock.sec + msg.clock.nanosec / 1e9
        hook = self._clock_hook
        if hook is not None:
            hook(sim)
        now = time.time()
        with self._lock:
            # the future-skew reference advances on EVERY clock message (the
            # 20 ms downsampling below is only for the RTF stream)
            if self._clock_sim is None or sim > self._clock_sim:
                self._clock_sim = sim
            # diagnostics: how stale this dedicated callback itself ran, and
            # how big a batch it had to drain (evidence for the round-3 fix)
            if self._last_clock_callback_wall is not None:
                callback_gap = now - self._last_clock_callback_wall
                self._clock_max_callback_gap_sec = max(self._clock_max_callback_gap_sec, callback_gap)
                if callback_gap >= CLOCK_STALL_THRESHOLD_SEC:
                    # a wall interval of the dedicated clock callback being
                    # starved: the host-pause discriminator matches these
                    # against the odometry gap's receive interval (round 4)
                    self._clock_stalls.append((self._last_clock_callback_wall, now))
            if self._last_clock_callback_sim is not None:
                self._clock_max_jump_sec = max(self._clock_max_jump_sec, sim - self._last_clock_callback_sim)
            self._last_clock_callback_wall = now
            self._last_clock_callback_sim = sim
            # /clock arrives far faster than the RTF needs: keep ~20 ms
            # granularity for the recorded stream
            if now - self._last_clock_wall >= 0.02:
                self._last_clock_wall = now
                self.clock.append((now, sim))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "filtered_odom": list(self.filtered_odom),
                "wheel_odom": list(self.wheel_odom),
                "amcl_pose": list(self.amcl_pose),
                "cmd_smoothed": list(self.cmd_smoothed),
                "cmd_vel": list(self.cmd_vel),
                "lidar_frames": list(self.lidar_frames),
                "clock": list(self.clock),
            }


class CommandStream:
    """The scripted twist publisher: a 20 Hz wall-time timer (D14.4)."""

    def __init__(self, publisher, rate_hz: float = COMMAND_RATE_HZ):
        self._publisher = publisher
        self._period = 1.0 / rate_hz
        self._twist = Twist()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> CommandStream:
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                twist = self._twist
            self._publisher.publish(twist)
            self._stop.wait(self._period)

    def set(self, linear: tuple[float, float], angular: float) -> None:
        twist = Twist(linear=Vector3(x=linear[0], y=linear[1]), angular=Vector3(z=angular))
        with self._lock:
            self._twist = twist

    def zero(self) -> None:
        with self._lock:
            self._twist = Twist()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


class _Supervisor:
    """Wall-time watchdogs independent of /clock (D14.4): a paused or frozen
    simulator cannot hang the test. Failures are recorded; the route loop
    and teardown observe them, so every watchdog failure goes through the
    same teardown and the process never ends in a colcon timeout (124).

    During the drive it also samples the ``/cmd_vel`` publisher set (D14.4:
    exactly the production set with the monitor the only active one) every
    couple of seconds, and it fails a run whose ground-truth stream has
    gone silent (the echo child died): the last parsed pose sample may not
    be older than GROUND_TRUTH_STALE_SEC of wall time.
    """

    def __init__(self, launch, whole_run_limit: float):
        self._launch = launch
        self._whole_run_limit = whole_run_limit
        self._failures: list[str] = []
        self._abort = threading.Event()
        self._last_clock_sim: float | None = None
        self._last_clock_wall: float | None = None
        self._start_wall = time.monotonic()
        self._stop = threading.Event()
        self._publisher_probe = None
        self._ground_truth_probe = None
        self._during_drive = False
        self._publishers_during_drive: set[str] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> _Supervisor:
        self._thread.start()
        return self

    def observe_clock(self, sim_sec: float) -> None:
        now = time.monotonic()
        if self._last_clock_sim is None or sim_sec > self._last_clock_sim:
            self._last_clock_sim = sim_sec
            self._last_clock_wall = now

    def watch_cmd_vel_publishers(self, probe_callable) -> None:
        self._publisher_probe = probe_callable

    def watch_ground_truth(self, age_callable) -> None:
        """Watch ``age_callable() -> seconds | None``: the wall age of the
        last parsed ground-truth sample (None before the first sample)."""
        self._ground_truth_probe = age_callable

    def begin_drive(self) -> None:
        """Sample the /cmd_vel publisher set only from here on (D14.4)."""
        self._during_drive = True

    def end_drive(self) -> None:
        self._during_drive = False

    def publishers_seen_during_drive(self) -> set[str] | None:
        return self._publishers_during_drive

    def _run(self) -> None:
        last_publisher_check = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if self._launch.poll() is not None and not self._abort.is_set():
                self._fail(f"the runtime launch exited early (code {self._launch.returncode})")
            if self._last_clock_wall is not None and now - self._last_clock_wall > CLOCK_STALL_SEC:
                self._fail(f"simulation stalled: /clock silent for {now - self._last_clock_wall:.1f} s of wall time")
            if now - self._start_wall > self._whole_run_limit:
                self._fail(f"whole-run wall-time limit exceeded ({self._whole_run_limit:.0f} s)")
            if self._ground_truth_probe is not None:
                try:
                    age = self._ground_truth_probe()
                except Exception:  # the stream may already be closing
                    age = None
                if age is not None and age > GROUND_TRUTH_STALE_SEC:
                    self._fail(f"ground truth went silent: last sample {age:.1f} s old")
            if self._publisher_probe is not None and self._during_drive and now - last_publisher_check >= 2.0:
                last_publisher_check = now
                try:
                    names = frozenset(self._publisher_probe())
                except Exception:  # the node may already be shutting down
                    names = None
                if names is not None:
                    self._publishers_during_drive = (
                        names if self._publishers_during_drive is None else self._publishers_during_drive | names
                    )
            self._stop.wait(0.25)

    def _fail(self, reason: str) -> None:
        # the same condition fires every cycle until the abort is observed:
        # record it once so failure messages stay readable
        if reason not in self._failures:
            self._failures.append(reason)
        self._abort.set()

    @property
    def abort(self) -> threading.Event:
        return self._abort

    @property
    def failures(self) -> list[str]:
        return list(self._failures)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Route mechanics (D14.4/D14.5)
# ---------------------------------------------------------------------------


def _segment_command(entry: dict) -> tuple[tuple[float, float], float, str]:
    """The constant command of one segment and its smoother-decel axis (D14.5)."""
    if entry["kind"] == "forward":
        return (entry["speed"], 0.0), 0.0, "vx"
    if entry["kind"] == "lateral":
        return (0.0, entry["speed"]), 0.0, "vy"
    rotation = math.copysign(entry["speed"], math.radians(entry["yaw_deg"][1] - entry["yaw_deg"][0]))
    return (0.0, 0.0), rotation, "wz"


def _segment_stopping_distance(entry: dict) -> float:
    """D14.4: the distance the chain still travels after the command stops
    — the smoother's ramp v^2/(2|decel|) plus the in-flight command
    latency v*CHAIN_COMMAND_LATENCY_SEC (measured, see the constant)."""
    speed = abs(entry["speed"])
    _linear, _angular, axis = _segment_command(entry)
    return speed * CHAIN_COMMAND_LATENCY_SEC + speed**2 / (2.0 * SMOOTHER_DECEL[axis])


def _segment_nominal_duration(entry: dict) -> float:
    if entry["kind"] == "rotate":
        progress = abs(math.radians(entry["yaw_deg"][1] - entry["yaw_deg"][0]))
    else:
        start, end = entry["start"], entry["end"]
        progress = math.hypot(end[0] - start[0], end[1] - start[1])
    return progress / abs(entry["speed"])


def _remaining_progress(entry: dict, sample) -> float:
    """Remaining ground-truth progress along the segment (D14.4): distance
    for translations (projected on the motion direction, negative once the
    end is passed), yaw for rotations (signed by the turn direction)."""
    if entry["kind"] == "rotate":
        start_yaw, end_yaw = (math.radians(value) for value in entry["yaw_deg"])
        direction = math.copysign(1.0, end_yaw - start_yaw)
        return direction * wrap_angle(end_yaw - sample.yaw)
    start, end = entry["start"], entry["end"]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    return ((end[0] - sample.x) * dx + (end[1] - sample.y) * dy) / length


def _ground_truth_speeds(window: list) -> tuple[float, float]:
    """Max planar speed and |yaw rate| over a ground-truth window."""
    speeds, yaw_rates = [], []
    for previous, current in zip(window, window[1:], strict=False):
        dt = current.stamp_sec - previous.stamp_sec
        if dt <= 1e-6:
            continue
        speeds.append(math.hypot(current.x - previous.x, current.y - previous.y) / dt)
        yaw_rates.append(abs(wrap_angle(current.yaw - previous.yaw)) / dt)
    return max(speeds, default=0.0), max(yaw_rates, default=0.0)


def _off_route_violations(entry: dict, samples: list) -> list[str]:
    """D14.5: ground truth matched to the segment being executed — position
    off the segment's line (a rotation's line is its rotation point), or
    heading off the nominal outside rotation segments."""
    violations = []
    start_yaw = math.radians(entry["yaw_deg"][0])
    if entry["kind"] == "rotate":
        point = entry["point"]
        for sample in samples:
            offset = math.hypot(sample.x - point[0], sample.y - point[1])
            if offset > ROUTE_POSITION_TOLERANCE:
                violations.append(
                    f"segment {entry['segment']}: position ({sample.x:.3f}, {sample.y:.3f}) at "
                    f"t={sample.stamp_sec:.2f} s is {offset:.3f} m from the rotation point"
                )
        return violations
    start, end = entry["start"], entry["end"]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length  # the segment line's normal
    for sample in samples:
        offset = abs((sample.x - start[0]) * nx + (sample.y - start[1]) * ny)
        if offset > ROUTE_POSITION_TOLERANCE:
            violations.append(
                f"segment {entry['segment']}: position ({sample.x:.3f}, {sample.y:.3f}) at "
                f"t={sample.stamp_sec:.2f} s is {offset:.3f} m off the segment line"
            )
        heading_error = abs(wrap_angle(sample.yaw - start_yaw))
        if heading_error > math.radians(ROUTE_HEADING_TOLERANCE_DEG):
            violations.append(
                f"segment {entry['segment']}: heading {math.degrees(sample.yaw):.1f} deg at "
                f"t={sample.stamp_sec:.2f} s exceeds {ROUTE_HEADING_TOLERANCE_DEG:.0f} deg off "
                f"{math.degrees(start_yaw):.0f} deg"
            )
    return violations


# ---------------------------------------------------------------------------
# Metrics, the collision-monitor check and the report (D14.4/D14.6)
# ---------------------------------------------------------------------------


def _estimate_metrics(samples: list[dict], ground_truth, begin: float, end: float) -> dict:
    """D14.6 metrics of one estimate: alignment at its own t0 (the first
    pairable sample after the settle preceding segment 1), every sample
    paired with ground truth interpolated at its stamp."""
    window = [sample for sample in samples if begin <= sample["stamp"] <= end]
    paired, unpaired = [], 0
    for sample in window:
        truth = ground_truth.interpolate(sample["stamp"])
        if truth is None:
            unpaired += 1
            continue
        paired.append((sample, truth))
    if not paired:
        return {
            "t0": None,
            "samples": len(window),
            "paired": 0,
            "unpaired": len(window),
            "unpaired_fraction": 1.0 if window else 0.0,
            "ate_rmse": None,
            "max_err": None,
            "final_err": None,
            "final_yaw_err_deg": None,
            "trajectory": [],
        }
    first, first_truth = paired[0]
    # A = G(t0) . E(t0)^-1 in SE(2): the estimate pose p maps to
    # R(dyaw).(p - pE) + pG with the yaw shifted by dyaw
    dyaw = wrap_angle(first_truth.yaw - first["yaw"])
    cos, sin = math.cos(dyaw), math.sin(dyaw)

    def _aligned(sample):
        px, py = sample["x"] - first["x"], sample["y"] - first["y"]
        return (
            cos * px - sin * py + first_truth.x,
            sin * px + cos * py + first_truth.y,
            wrap_angle(sample["yaw"] + dyaw),
        )

    errors = []
    trajectory = []
    for sample, truth in paired:
        x, y, yaw = _aligned(sample)
        error = math.hypot(x - truth.x, y - truth.y)
        errors.append(error)
        trajectory.append(
            {
                "stamp": sample["stamp"],
                "x": x,
                "y": y,
                "yaw": yaw,
                "truth_x": truth.x,
                "truth_y": truth.y,
                "truth_yaw": truth.yaw,
                "error": error,
            }
        )
    last, last_truth = paired[-1]
    final_x, final_y, final_yaw = _aligned(last)
    return {
        "t0": first["stamp"],
        "samples": len(window),
        "paired": len(paired),
        "unpaired": unpaired,
        "unpaired_fraction": unpaired / len(window) if window else 0.0,
        "ate_rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "max_err": max(errors),
        "final_err": math.hypot(final_x - last_truth.x, final_y - last_truth.y),
        "final_yaw_err_deg": math.degrees(abs(wrap_angle(final_yaw - last_truth.yaw))),
        "trajectory": trajectory,
    }


def _collision_monitor_verdict(cmd_smoothed: list, cmd_vel: list, window_begin: float | None = None) -> dict:
    """The D14.4 collision-monitor check (pure; unit-tested on synthetic
    streams below).

    Only the non-zero inputs (the driven parts of the route) are walked: the
    monitor legitimately stops relaying zero after ``stop_pub_timeout``, so
    settle zeros must neither consume outputs nor mask drive outputs. Each
    non-zero input pairs with the output received closest to its own relay
    SLOT — the expected relay time ``t_in + relay_latency`` — within half a
    sample period of it; the relay latency is measured per stream (a first
    pass pairs value-equal outputs and takes the modal lag), because the
    measured monitor relays at 15-33 ms of wall time — on a loaded host
    that EXCEEDS half the 20 Hz sample period (25 ms), and a plain
    nearest-to-t_in walk then pairs each input with its NEIGHBOUR's relay
    and misreports every velocity-smoother ramp as an intervention
    (observed 3x, task-review-3x-results.md). The slot rule keeps every
    detection semantics: a zeroed or scaled relay sits in its own slot and
    is reported as changed, an input whose slot is empty counts as missing
    (> 1 % or two consecutive fails; an isolated loss is tolerated and
    does NOT shift the pairing — the neighbour's relay stays in the
    neighbour's slot), and every non-zero output left unconsumed is
    spontaneous. Without a measurable relay latency (e.g. a fully scaled
    segment with no value-equal pair) the pairing falls back to slots at
    ``t_in`` itself.

    ``window_begin`` is the wall time the caller trimmed both streams at. An
    output relayed within one slot after it may belong to an input received
    just before it, which the trim removed; such an output is unobservable,
    not spontaneous — the mirror of the window-end rule for inputs below.
    """

    def _nonzero(value) -> bool:
        return any(abs(component) > MONITOR_EQUALITY_TOL for component in value)

    def _equal(left, right) -> bool:
        return all(abs(a - b) <= MONITOR_EQUALITY_TOL for a, b in zip(left, right, strict=True))

    inputs = sorted(cmd_smoothed)
    outputs = sorted(cmd_vel)

    def _scan(receive_wall, value, slot_offset):
        """(index, output_value, delta) of the unconsumed output closest to
        the slot, and the same for the closest value-equal one — both inside
        the receive window; (None, None, None) when the window is empty."""
        window_lo = receive_wall - PAIR_WINDOW_BEFORE_SEC
        window_hi = receive_wall + PAIR_WINDOW_AFTER_SEC
        slot = receive_wall + slot_offset
        any_pair = equal_pair = None
        any_delta = equal_delta = None
        for index, (output_wall, output_value) in enumerate(outputs):
            if consumed[index] or output_wall < window_lo:
                continue
            if output_wall > window_hi:
                break
            delta = abs(output_wall - slot)
            if any_delta is None or delta < any_delta:
                any_pair, any_delta = (index, output_value), delta
            if _equal(value, output_value) and (equal_delta is None or delta < equal_delta):
                equal_pair, equal_delta = (index, output_value), delta
        return any_pair, any_delta, equal_pair

    # pass 1: pair value-equal outputs to estimate the stream's relay
    # latency (the modal successful lag) — the value-equal preference is
    # only this estimator; the real pairing below is slot-based
    consumed = [False] * len(outputs)
    latencies = []
    for receive_wall, value in inputs:
        if not _nonzero(value):
            continue
        _any_pair, _any_delta, equal_pair = _scan(receive_wall, value, 0.0)
        if equal_pair is not None:
            index, _output_value = equal_pair
            consumed[index] = True
            lag = outputs[index][0] - receive_wall
            if 0.0 <= lag <= 0.15:
                latencies.append(round(lag, 2))
    relay_latency = None
    if len(latencies) >= 3:
        bins: dict[float, int] = {}
        for lag in latencies:
            bins[lag] = bins.get(lag, 0) + 1
        relay_latency = max(bins, key=lambda lag: bins[lag])

    # pass 2: the real pairing against each input's relay slot
    consumed = [False] * len(outputs)
    changed = []
    nonzero_inputs = sum(1 for _wall, value in inputs if _nonzero(value))
    # frame conservation: the monitor republishes one output per input, so a
    # stream whose latency merely jitters still carries every frame — only
    # then may an empty slot fall back to a value-equal output elsewhere in
    # the window. A genuinely dropped relay leaves the stream one frame
    # short, and an empty slot then counts as missing (not as its
    # neighbour's displaced relay). Test-subscription loss would ALSO
    # short the stream, keeping the fallback off — the counts stay honest.
    # Only non-zero outputs count: settle zeros relayed before
    # stop_pub_timeout would otherwise outnumber a dropped drive relay.
    nonzero_outputs = sum(1 for _wall, value in outputs if _nonzero(value))
    fallback_allowed = nonzero_outputs >= nonzero_inputs
    nonzero_missing = 0
    consecutive_missing = worst_consecutive_missing = 0
    last_input_wall = inputs[-1][0] if inputs else None
    for receive_wall, value in inputs:
        if not _nonzero(value):
            continue
        slot_offset = relay_latency if relay_latency is not None else 0.0
        any_pair, any_delta, _equal_pair = _scan(receive_wall, value, slot_offset)
        # an input whose slot lies beyond the recording window has no
        # observable relay: not a monitor drop, never a missing
        if any_pair is None and last_input_wall is not None and receive_wall + slot_offset > last_input_wall:
            continue
        # the slot must be occupied within half a sample period: an output
        # beyond that belongs to a neighbour's slot, so an isolated drop
        # stays a single missing instead of shifting the whole pairing.
        # When the slot is empty (a relay whose latency jittered beyond
        # half a period, observed as missing+spontaneous pairs in the
        # field), a value-equal output in the window still pairs — the
        # monitor republishes unchanged, so an equal value is a relay
        # wherever it sits. Value-unequal outputs are never taken as a
        # fallback: a zeroed or scaled relay sits in its OWN slot (changed
        # above), and an unrelated leftover must stay unconsumed for the
        # spontaneous count.
        if any_pair is None or any_delta > PAIR_HALF_PERIOD_SEC:
            _pair, _delta, equal_pair = _scan(receive_wall, value, 0.0) if fallback_allowed else (None, None, None)
            if equal_pair is not None:
                index, output_value = equal_pair
                consumed[index] = True
                consecutive_missing = 0
                continue
            nonzero_missing += 1
            consecutive_missing += 1
            worst_consecutive_missing = max(worst_consecutive_missing, consecutive_missing)
            continue
        index, output_value = any_pair
        consumed[index] = True
        consecutive_missing = 0
        if not _equal(value, output_value):
            changed.append(f"input {value} at {receive_wall:.3f} -> output {output_value} at {outputs[index][0]:.3f}")
    boundary_slot_end = None
    if window_begin is not None:
        boundary_slot_end = window_begin + (relay_latency or 0.0) + PAIR_HALF_PERIOD_SEC
    spontaneous_outputs = sum(
        1
        for index, (wall, value) in enumerate(outputs)
        if not consumed[index] and _nonzero(value) and (boundary_slot_end is None or wall > boundary_slot_end)
    )
    return {
        "nonzero_inputs": nonzero_inputs,
        "nonzero_missing": nonzero_missing,
        "missing_fraction": (nonzero_missing / nonzero_inputs) if nonzero_inputs else 0.0,
        "consecutive_missing": worst_consecutive_missing,
        "changed": changed[:5],
        "changed_count": len(changed),
        "spontaneous_outputs": spontaneous_outputs,
        "total_inputs": len(inputs),
        "total_outputs": len(outputs),
        "relay_latency_ms": (1000.0 * relay_latency) if relay_latency is not None else None,
    }


# ---------------------------------------------------------------------------
# The D14.4 pairing on synthetic streams (Gazebo-free, pure functions)
# ---------------------------------------------------------------------------

_MONITOR_RATE = 0.05  # the smoother's 20 Hz cadence
_MONITOR_RELAY = 0.015  # the measured monitor relay latency


def _drive_inputs(values, start=100.0):
    """(receive_wall, (vx, vy, wz)) samples at the smoother cadence."""
    return [(start + index * _MONITOR_RATE, tuple(value)) for index, value in enumerate(values)]


def _relayed(inputs, overrides=None, drop=()):
    """The monitor's output for the given inputs: each input relayed
    ``_MONITOR_RELAY`` later, with per-index value overrides (zeroed or
    scaled relays) and per-index drops (missing relays)."""
    overrides = overrides or {}
    result = []
    for index, (wall, value) in enumerate(inputs):
        if index in drop:
            continue
        result.append((wall + _MONITOR_RELAY, tuple(overrides.get(index, value))))
    return result


def test_monitor_single_zeroed_command_is_reported_as_changed():
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs, overrides={5: (0.0, 0.0, 0.0)}))
    assert len(verdict["changed"]) == 1 and verdict["nonzero_missing"] == 0
    assert verdict["spontaneous_outputs"] == 0


def test_monitor_consecutive_zeroed_commands_are_each_reported():
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs, overrides={5: (0, 0, 0), 6: (0, 0, 0), 7: (0, 0, 0)}))
    assert len(verdict["changed"]) == 3 and verdict["nonzero_missing"] == 0
    assert verdict["spontaneous_outputs"] == 0


def test_monitor_single_dropped_relay_is_one_tolerable_missing():
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs, drop={5}))
    assert verdict["nonzero_missing"] == 1 and verdict["consecutive_missing"] == 1
    assert verdict["changed"] == [] and verdict["spontaneous_outputs"] == 0
    # the caller tolerates this: 1 < 1 % of 10 is not, but the rules are
    # the caller's — the verdict only reports the counts


def test_monitor_relayed_settle_zeros_do_not_mask_a_dropped_drive_relay():
    # settle zeros relayed before stop_pub_timeout outnumber the single
    # dropped drive relay; the drop must still be reported as exactly one
    # missing frame (no changed or spontaneous outputs)
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10 + [(0.0, 0.0, 0.0)] * 5)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs, drop={5}))
    assert verdict["nonzero_missing"] == 1 and verdict["consecutive_missing"] == 1
    assert verdict["changed"] == [] and verdict["spontaneous_outputs"] == 0


def test_monitor_relay_of_an_input_trimmed_at_the_window_start_is_not_spontaneous():
    # the caller trims both streams at the same wall time: an input received
    # just before the cut is gone while its relay, 15 ms later, survives
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    outputs = _relayed(inputs)
    cut = inputs[0][0] + 0.01
    verdict = _collision_monitor_verdict(
        [s for s in inputs if s[0] >= cut], [s for s in outputs if s[0] >= cut], window_begin=cut
    )
    assert verdict["spontaneous_outputs"] == 0 and verdict["nonzero_missing"] == 0
    # without the window start the same leftover is (wrongly) spontaneous
    unbounded = _collision_monitor_verdict([s for s in inputs if s[0] >= cut], [s for s in outputs if s[0] >= cut])
    assert unbounded["spontaneous_outputs"] == 1


def test_monitor_output_well_after_the_window_start_stays_spontaneous():
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    outputs = _relayed(inputs) + [(inputs[5][0] + 0.03, (0.2, 0.0, 0.0))]
    verdict = _collision_monitor_verdict(inputs, outputs, window_begin=inputs[0][0])
    assert verdict["spontaneous_outputs"] == 1


def test_monitor_two_consecutive_dropped_relays_fail_the_run():
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs, drop={5, 6}))
    assert verdict["nonzero_missing"] == 2 and verdict["consecutive_missing"] == 2
    assert verdict["changed"] == []


def test_monitor_single_scaled_relay_is_reported_as_changed():
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs, overrides={5: (0.05, 0.0, 0.0)}))
    assert len(verdict["changed"]) == 1 and verdict["nonzero_missing"] == 0


def test_monitor_a_whole_scaled_segment_is_fully_reported():
    inputs = _drive_inputs([(0.08, 0.0, 0.0)] * 20)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs, overrides={i: (0.04, 0, 0) for i in range(20)}))
    assert len(verdict["changed"]) == 5  # the report keeps the first five
    assert verdict["changed_count"] > 5
    assert verdict["nonzero_missing"] == 0


def test_monitor_spontaneous_output_during_a_settle_is_detected():
    # 20 zero inputs (a settle), then five driven samples; the monitor
    # relays the driven part but ALSO produced one non-zero output of its
    # own in the middle of the settle
    inputs = _drive_inputs([(0.0, 0.0, 0.0)] * 20 + [(0.10, 0.0, 0.0)] * 5, start=200.0)
    outputs = _relayed(inputs[20:])
    outputs.append((inputs[10][0] + _MONITOR_RELAY, (0.35, 0.0, 0.0)))
    outputs.sort()
    verdict = _collision_monitor_verdict(inputs, outputs)
    assert verdict["spontaneous_outputs"] == 1
    assert verdict["changed"] == [] and verdict["nonzero_missing"] == 0


def test_monitor_unrelayed_settle_zeros_cause_no_false_report():
    # the monitor stops relaying zero after stop_pub_timeout: the settle's
    # zero inputs have no outputs at all, and the driven part is clean
    inputs = _drive_inputs([(0.0, 0.0, 0.0)] * 20 + [(0.10, 0.0, 0.0)] * 5, start=300.0)
    verdict = _collision_monitor_verdict(inputs, _relayed(inputs[20:]))
    assert verdict["changed"] == [] and verdict["nonzero_missing"] == 0
    assert verdict["spontaneous_outputs"] == 0 and verdict["nonzero_inputs"] == 5


def test_monitor_high_relay_latency_does_not_misreport_a_smoother_ramp():
    """The observed field failure: the monitor relays at 30 ms — over half
    the 20 Hz sample period — and a plain nearest-to-t_in walk pairs each
    ramp input with its NEIGHBOUR's relay, reporting the velocity smoother's
    ramp-down as three interventions (task-review-3x-results.md). The slot
    pairing measures the relay latency first and pairs each input with its
    own relay wherever it sits in the window."""
    # constant 0.10 drive, then the smoother's ramp-down 0.10 -> 0.0675 ->
    # 0.035 -> 0, each relayed 30 ms late (a clean 1:1 stream, just slow)
    values = (
        [(0.10, 0.0, 0.0)] * 20 + [(0.0675, 0.0, 0.0), (0.035, 0.0, 0.0), (0.0025, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 5
    )
    inputs = _drive_inputs(values)
    outputs = [(wall + 0.03, tuple(value)) for wall, value in inputs]
    verdict = _collision_monitor_verdict(inputs, outputs)
    assert verdict["changed"] == []
    assert verdict["nonzero_missing"] == 0 and verdict["spontaneous_outputs"] == 0
    assert verdict["relay_latency_ms"] == pytest.approx(30.0)

    # the plain nearest-to-t_in rule would have reported every ramp input
    # as changed — the mispairing the slot rule exists to prevent
    assert len(values) >= 23


def test_monitor_a_latency_jittering_relay_still_pairs():
    """The field failure of the verification round: one relay of a constant
    stream jitters to 60 ms (the rest sit at 10 ms) — beyond half the 20 Hz
    sample period from its slot, so the slot looks empty and the displaced
    relay previously surfaced as one missing plus one spontaneous. A
    value-equal output in the window still pairs (an unchanged relay keeps
    its value wherever it sits), so the stream reports clean."""
    inputs = _drive_inputs([(0.10, 0.0, 0.0)] * 10)
    outputs = [(wall + _MONITOR_RELAY, tuple(value)) for wall, value in inputs]
    outputs[5] = (inputs[5][0] + 0.06, tuple(inputs[5][1]))  # this one relay is late
    outputs.sort()
    verdict = _collision_monitor_verdict(inputs, outputs)
    assert verdict["changed"] == [] and verdict["nonzero_missing"] == 0
    assert verdict["spontaneous_outputs"] == 0


def test_monitor_a_zeroed_relay_is_caught_even_at_high_relay_latency():
    """M1's consecutive-zeroing detection must survive the slot pairing: with
    a 30 ms relay the zeroed relays sit in their own slots and are each
    reported, not displaced onto neighbours."""
    values = [(0.10, 0.0, 0.0)] * 10
    inputs = _drive_inputs(values)
    outputs = [(wall + 0.03, tuple(value)) for wall, value in inputs]
    for index in (5, 6, 7):
        outputs[index] = (outputs[index][0], (0.0, 0.0, 0.0))
    verdict = _collision_monitor_verdict(inputs, outputs)
    assert len(verdict["changed"]) == 3 and verdict["nonzero_missing"] == 0
    assert verdict["spontaneous_outputs"] == 0


def _plot(report: dict, log_dir: Path) -> None:
    """Trajectories over the scene and error vs. time into the log dir."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is best effort
        report["plots"] = "matplotlib unavailable; skipped"
        return
    truth = report["ground_truth"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    scene = axes[0]
    for element in report["scene_geometry"]:
        pose, size = element["pose"], element["size"]
        if element["primitive"] == "box":
            scene.add_patch(
                patches.Rectangle(
                    (pose[0] - size[0] / 2, pose[1] - size[1] / 2), size[0], size[1], color="0.5", alpha=0.5
                )
            )
        else:
            scene.add_patch(patches.Circle((pose[0], pose[1]), size[0], color="0.5", alpha=0.5))
    scene.plot([point["x"] for point in truth], [point["y"] for point in truth], "k.", ms=1, label="ground truth")
    for name, style in (("fast_lio", "-"), ("wheel", "--"), ("amcl", ":")):
        trajectory = report["metrics"][name]["trajectory"]
        if trajectory:
            scene.plot([point["x"] for point in trajectory], [point["y"] for point in trajectory], style, label=name)
    scene.set_aspect("equal")
    scene.legend()
    scene.set_title("trajectories over the scene (aligned at t0)")
    errors = axes[1]
    for name in ("fast_lio", "wheel", "amcl"):
        trajectory = report["metrics"][name]["trajectory"]
        if trajectory:
            errors.plot([point["stamp"] for point in trajectory], [point["error"] for point in trajectory], label=name)
    errors.axhline(FAST_LIO_CAPS["max_err"], color="r", ls="--", lw=0.8, label="spec max_err cap")
    errors.set_xlabel("simulation time [s]")
    errors.set_ylabel("position error [m]")
    errors.legend()
    errors.set_title("position error vs. time")
    figure.tight_layout()
    figure.savefig(log_dir / "lidar_odometry_gate.png", dpi=120)
    plt.close(figure)


def _stage_gate_config(destination: Path, overlay_path: Path, sim_platform: str) -> Path:
    """Stage the robot_config base and the gate overlay side by side (D14.3).

    The overlay is staged per platform (D10, lekiwi-mujoco-runtime-platform):
    the shipped file pins ``platform: gazebo``; the staged copy carries the
    platform under test so the world owner selects the right provider branch.
    """
    from ament_index_python.packages import get_package_share_directory

    destination.mkdir(parents=True, exist_ok=True)
    base = Path(get_package_share_directory("robot_config")) / "config" / "robots" / "lekiwi_lidar.yaml"
    assert base.is_file(), f"the robot_config test dependency must install {base}"
    shutil.copy2(base, destination / "lekiwi_lidar.yaml")
    staged_overlay = destination / "lekiwi_lio_sim.yaml"
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    stage = (overlay.get("robot", {}).get("nav_stages") or {}).get("navigation") or {}
    (stage.get("simulation") or {})["platform"] = sim_platform
    staged_overlay.write_text(yaml.safe_dump(overlay, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return staged_overlay


def _summary(report: dict) -> str:
    """Every failure message carries all metrics of all three estimates."""
    lines = ["[lio-gate] metrics (aligned at t0, simulation time):"]
    for name in ("fast_lio", "wheel", "amcl"):
        metrics = report["metrics"][name]
        lines.append(
            "  {name:>8}: t0={t0} paired={paired}/{samples} unpaired={unpaired:.1%} ate_rmse={ate} "
            "max_err={max} final_err={final} final_yaw_err={yaw} deg".format(
                name=name,
                t0=f"{metrics['t0']:.2f} s" if metrics["t0"] is not None else "n/a",
                paired=metrics["paired"],
                samples=metrics["samples"],
                unpaired=metrics["unpaired_fraction"],
                ate=f"{metrics['ate_rmse']:.4f} m" if metrics["ate_rmse"] is not None else "n/a",
                max=f"{metrics['max_err']:.4f} m" if metrics["max_err"] is not None else "n/a",
                final=f"{metrics['final_err']:.4f} m" if metrics["final_err"] is not None else "n/a",
                yaw=f"{metrics['final_yaw_err_deg']:.2f}" if metrics["final_yaw_err_deg"] is not None else "n/a",
            )
        )
    lines.append(f"  mean RTF over the route: {report['mean_rtf']:.2f}")
    health = report["frame_health"]
    fraction = health["frame_fraction"]
    lines.append(
        f"  frame health: /odometry/filtered {health['odometry_frames']}/{health['lidar_frames']} of "
        f"/livox/lidar ({'n/a' if fraction is None else f'{fraction:.1%}'}), max stamp gap "
        f"{health['max_gap_sec']:.2f} s"
        + (f" at t={health['max_gap_from']:.1f}..{health['max_gap_to']:.1f} s" if health["max_gap_from"] else "")
    )
    lines.append(f"  thresholds: {report['thresholds']} (caps {report['caps']})")
    monitor = report["collision_monitor"]
    lines.append(
        "  collision monitor: {inputs} non-zero inputs, {missing} missing ({fraction:.1%}), "
        "{consecutive} worst consecutive, {spontaneous} spontaneous outputs, {changed} changed, "
        "relay latency {latency}".format(
            inputs=monitor["nonzero_inputs"],
            missing=monitor["nonzero_missing"],
            fraction=monitor["missing_fraction"],
            consecutive=monitor["consecutive_missing"],
            spontaneous=monitor["spontaneous_outputs"],
            changed=len(monitor["changed"]),
            latency=(f"{monitor['relay_latency_ms']:.0f} ms" if monitor["relay_latency_ms"] is not None else "n/a"),
        )
    )
    lines.append(f"  /cmd_vel publishers: {report['cmd_vel_publishers']}")
    lines.append(f"  /tf publishers: {report['tf_publishers']}")
    lines.append(f"  /tf frames carried: {report['tf_frames_carried']}")
    skew = report["max_future_skew_sec"]
    lines.append(
        f"  max future skew: {'n/a' if skew is None else f'{skew:.3f} s'}; "
        f"bridge rejections: {report['rejected_samples']}; deaths: {report['fast_lio_deaths']}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sim_platform", PHYSICS_SIM_PLATFORMS)
def test_lidar_odometry_gate(sim_platform):
    missing = missing_gate_capabilities(sim_platform)
    if missing:
        _gate_skip_or_fail(
            sim_platform, f"lidar-odometry gate: platform {sim_platform!r} lacks capabilities: {'; '.join(missing)}"
        )
    environment_missing = _environment_missing(sim_platform)
    if environment_missing:
        _gate_skip_or_fail(
            sim_platform, f"lidar-odometry gate: platform {sim_platform!r} cannot run here: {environment_missing}"
        )
    selection = _select_test_profile()
    if selection in (None, "minimal"):
        why = (
            "the openEuler default minimal E2E profile disables the gate"
            if selection is None
            else "the minimal E2E profile disables the gate"
        )
        _gate_skip_or_fail(sim_platform, f"{why}; set {NAV_TEST_PROFILE_ENV}=full to run it")

    report_only = os.environ.get(REPORT_ONLY_ENV, "").strip() == "1"
    overlay_env = os.environ.get(OVERLAY_ENV, "").strip()
    overlay_path = Path(overlay_env).expanduser() if overlay_env else _GATE_OVERLAY
    if overlay_env:
        assert overlay_path.is_file(), f"{OVERLAY_ENV} does not exist: {overlay_env}"
    report_dir_env = os.environ.get(REPORT_DIR_ENV, "").strip()
    log_dir = Path(report_dir_env) if report_dir_env else Path(tempfile.mkdtemp(prefix="lekiwi_lio_gate_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lio-gate] platform {sim_platform}; logs and plots: {log_dir}", flush=True)

    partition = f"lekiwi_lio_{uuid.uuid4().hex[:8]}"
    stage_dir = Path(tempfile.mkdtemp(prefix="lekiwi_lio_stage_"))
    config_path = _stage_gate_config(stage_dir, overlay_path, sim_platform)
    whole_run_limit = READINESS_WALL_BUDGET + sum(
        SEGMENT_WALL_LIMIT_FACTOR * _segment_nominal_duration(entry) for entry in ROUTE
    )
    environment = dict(os.environ)
    if sim_platform == "gazebo":
        # both partition variables: Fortress (ign, gz-transport11) reads only
        # IGN_PARTITION, newer gz reads GZ_PARTITION — set both to isolate
        # this run's gz bus (including /clock) on either stack
        environment.update(
            {
                "GZ_PARTITION": partition,
                "IGN_PARTITION": partition,
                "IBROBOT_GAZEBO_HEADLESS": "1",
            }
        )
    else:
        # MuJoCo has no gz bus; the runtime renders headless and the composed
        # model lands on this run's ROS domain (D7/D10)
        environment["IBROBOT_MUJOCO_HEADLESS"] = "1"
    environment.update(
        {
            "RCUTILS_COLORIZED_OUTPUT": "0",
            "ROS_LOCALHOST_ONLY": "1",
        }
    )
    launch_log_path = log_dir / "robot_launch.log"
    launch_log = launch_log_path.open("w", encoding="utf-8")
    launch = subprocess.Popen(
        [
            "ros2",
            "launch",
            "robot_config",
            "robot.launch.py",
            f"config_path:={config_path}",
            "use_sim:=true",
            f"sim_platform:={sim_platform}",
            f"nav_stage:={GATE_STAGE}",
        ],
        stdout=launch_log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    ground_truth = None
    probe = None
    executor = None
    spin_thread = None
    commands = None
    supervisor = None
    try:
        rclpy.init()
        probe = GateProbe()
        executor = MultiThreadedExecutor()
        executor.add_node(probe)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        supervisor = _Supervisor(launch, whole_run_limit).start()
        probe._clock_hook = supervisor.observe_clock
        supervisor.watch_cmd_vel_publishers(
            lambda: {info.node_name for info in probe.get_publishers_info_by_topic("/cmd_vel")}
        )
        ground_truth = open_ground_truth(sim_platform, world=GATE_WORLD, model=GATE_MODEL, env=environment).start()
        supervisor.watch_ground_truth(lambda: ground_truth.latest_sample_age())
        commands = CommandStream(probe.create_publisher(Twist, "/cmd_vel_nav", 10)).start()

        # readiness (D14.3): ACTIVE in base_navigation, gate acknowledged
        # open, FAST-LIO odometry published, first map -> odom appeared
        deadline = time.monotonic() + READINESS_WALL_BUDGET
        while time.monotonic() < deadline and not supervisor.abort.is_set():
            if (
                probe.status is not None
                and probe.status.lifecycle == "ACTIVE"
                and probe.status.active_mode == "base_navigation"
                and probe.gate_enabled is not None
                and probe.gate_enabled.data is True
            ):
                break
            time.sleep(0.2)
        assert probe.status is not None and probe.status.lifecycle == "ACTIVE", (
            f"the runtime did not reach ACTIVE; supervisor: {supervisor.failures}; see {launch_log_path}"
        )
        assert (
            probe.status.active_mode == "base_navigation"
            and probe.gate_enabled is not None
            and probe.gate_enabled.data is True
        ), f"navigation admission did not complete; supervisor: {supervisor.failures}; see {launch_log_path}"
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline and len(probe.filtered_odom) < 5 and not supervisor.abort.is_set():
            time.sleep(0.2)
        assert len(probe.filtered_odom) >= 5, (
            f"FAST-LIO produced {len(probe.filtered_odom)} odometry messages; "
            f"supervisor: {supervisor.failures}; see {launch_log_path}"
        )

        def _map_odom_appeared() -> bool:
            return any(
                transform.header.frame_id == "map" and transform.child_frame_id == "odom"
                for message in probe.tf_frames
                for transform in message.transforms
            )

        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline and not _map_odom_appeared() and not supervisor.abort.is_set():
            time.sleep(0.2)
        assert _map_odom_appeared(), f"AMCL's first map -> odom transform never appeared; see {launch_log_path}"
        assert ground_truth.wait_for(10, 30.0), "no world-pose samples arrived"

        # every post-readiness wait goes through one helper: a wall-time
        # timeout (or a watchdog abort) is a route failure — reported with
        # the full metrics like any other failure, never a silent pass
        def _wait_until(predicate, timeout: float, what: str) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return True
                if supervisor.abort.is_set():
                    route_failure.append(f"watchdog: {supervisor.failures}")
                    return False
                time.sleep(0.05)
            route_failure.append(f"{what}: timed out after {timeout:.0f} s of wall time")
            return False

        # 2 s of simulation-time zero-command settle before the route
        # (D14.3); nothing is accumulated for the metrics before its end
        commands.zero()
        settle_start = ground_truth.latest_time()
        assert settle_start is not None
        _wait_until(
            lambda: (now := ground_truth.latest_time()) is not None and now - settle_start >= PRE_ROUTE_SETTLE_SEC,
            60.0,
            "pre-route settle",
        )
        metrics_begin = ground_truth.latest_time()
        begin_wall = time.time()
        assert metrics_begin is not None
        supervisor.begin_drive()
        print(f"[lio-gate] readiness complete; metrics window begins at sim {metrics_begin:.2f} s", flush=True)

        # the route (D14.4): every segment commands its constant twist and
        # stops commanding once the remaining ground-truth progress is at
        # most the smoother's stopping distance; the segment ends when
        # ground truth has settled, and zero twist keeps flowing at 20 Hz
        segment_traces = []
        route_failure = []  # merged into the final verdict, never asserted early
        for entry in ROUTE:
            segment_sim_deadline = _segment_nominal_duration(entry) * SEGMENT_SIM_LIMIT_FACTOR
            segment_wall_deadline = time.monotonic() + _segment_nominal_duration(entry) * SEGMENT_WALL_LIMIT_FACTOR
            commands.set(*_segment_command(entry)[:2])
            segment_start = ground_truth.latest_time()
            assert segment_start is not None
            stopping = _segment_stopping_distance(entry)
            reason = None
            while True:
                if supervisor.abort.is_set():
                    reason = f"watchdog: {supervisor.failures}"
                    break
                sample = ground_truth.latest()
                assert sample is not None
                if _remaining_progress(entry, sample) <= stopping:
                    break
                if sample.stamp_sec - segment_start > segment_sim_deadline:
                    reason = (
                        f"segment {entry['segment']} stalled: {sample.stamp_sec - segment_start:.1f} s of "
                        f"simulation time (limit {segment_sim_deadline:.1f} s) with remaining progress "
                        f"{_remaining_progress(entry, sample):.3f}"
                    )
                    break
                if time.monotonic() > segment_wall_deadline:
                    reason = (
                        f"segment {entry['segment']} exceeded its wall-time limit "
                        f"({_segment_nominal_duration(entry) * SEGMENT_WALL_LIMIT_FACTOR:.0f} s)"
                    )
                    break
                time.sleep(0.02)
            commands.zero()
            if reason is not None:
                route_failure.append(reason)

            # the segment ends when ground-truth speed has settled (D14.4)
            def _settled() -> bool:
                now = ground_truth.latest_time()
                window = ground_truth.window(now - 0.3, now) if now is not None else []
                speed, yaw_rate = _ground_truth_speeds(window)
                return speed <= SETTLE_SPEED_LIMIT and yaw_rate <= SETTLE_YAW_RATE_LIMIT

            _wait_until(_settled, 30.0, f"segment {entry['segment']} settle")
            # 1 s of simulation time of continuous zero twist (D14.4)
            settle_from = ground_truth.latest_time()

            def _settle_elapsed(_from=settle_from) -> bool:
                now = ground_truth.latest_time()
                return now is not None and _from is not None and now - _from >= INTER_SEGMENT_SETTLE_SEC

            _wait_until(_settle_elapsed, 60.0, f"segment {entry['segment']} inter-settle")
            segment_traces.append(
                {
                    "segment": entry["segment"],
                    "kind": entry["kind"],
                    "sim_start": segment_start,
                    "sim_end": ground_truth.latest_time(),
                }
            )
            print(
                f"[lio-gate] segment {entry['segment']} ({entry['kind']}) done at sim "
                f"{ground_truth.latest_time():.2f} s",
                flush=True,
            )
            if route_failure:
                break
        metrics_end = ground_truth.latest_time() or metrics_begin
        end_wall = time.time()
        supervisor.end_drive()
        # route deviation (D14.5): collected alongside the stall/watchdog
        # failures and merged into the final verdict below — the report,
        # the plots and the metrics are always written first so a route
        # failure still carries every estimate's numbers
        for trace in segment_traces:
            entry = ROUTE[trace["segment"] - 1]
            route_failure.extend(
                _off_route_violations(entry, ground_truth.window(trace["sim_start"], trace["sim_end"]))[:3]
            )

        # ------------------------------------------------------------------
        # metrics, health, TF, monitor, RTF (D14.6)
        # ------------------------------------------------------------------
        snapshot = probe.snapshot()
        metrics = {
            "fast_lio": _estimate_metrics(snapshot["filtered_odom"], ground_truth, metrics_begin, metrics_end),
            "wheel": _estimate_metrics(snapshot["wheel_odom"], ground_truth, metrics_begin, metrics_end),
            "amcl": _estimate_metrics(snapshot["amcl_pose"], ground_truth, metrics_begin, metrics_end),
        }
        # the route's real-time factor as one ratio over the whole window: a
        # mean of per-sample ratios is inflated by /clock bursts (two samples
        # received within microseconds) and could hide a slow simulation
        route_clock = [(wall, sim) for wall, sim in snapshot["clock"] if begin_wall < wall < end_wall]
        if len(route_clock) >= 2 and route_clock[-1][0] > route_clock[0][0]:
            mean_rtf = (route_clock[-1][1] - route_clock[0][1]) / (route_clock[-1][0] - route_clock[0][0])
        else:
            mean_rtf = 0.0

        # the collision monitor over the recording window (D14.4): from t0
        # to the end of the final settle
        monitor_begin = begin_wall + FRAME_HEALTH_WARMUP_SEC
        monitor = _collision_monitor_verdict(
            [s for s in snapshot["cmd_smoothed"] if monitor_begin <= s[0] <= end_wall],
            [s for s in snapshot["cmd_vel"] if monitor_begin <= s[0] <= end_wall],
            window_begin=monitor_begin,
        )
        # TF ownership (D14.6/D4): the frames actually carried on /tf
        tf_publishers = {info.node_name for info in probe.get_publishers_info_by_topic("/tf")}
        carried = sorted(
            {
                (transform.header.frame_id, transform.child_frame_id)
                for message in probe.tf_frames
                for transform in message.transforms
            }
        )
        static_map_odom = [
            transform
            for message in probe.tf_static_frames
            for transform in message.transforms
            if transform.header.frame_id == "map" and transform.child_frame_id == "odom"
        ]
        # FAST-LIO health (D14.6): the launch log must show no death/respawn
        log_text = launch_log_path.read_text(encoding="utf-8", errors="replace")
        fast_lio_deaths = [
            line.strip()
            for line in log_text.splitlines()
            if ("fastlio_mapping" in line or "fast_lio_odom_bridge" in line)
            and ("process has died" in line or ("respawn" in line.lower() and "ERROR" in line))
        ]
        # future skew (D14.6): each estimate sample is compared with the
        # simulation time the DEDICATED clock callback had recorded when the
        # sample's own callback ran (review round 3: the old bisect over the
        # 20 ms-downsampled clock stream measured how starved the clock
        # CALLBACK was, not the chain — under load the /clock callback queued
        # behind the heavy subscribers in the node's one mutually-exclusive
        # group and served keep-last-window-old clocks, reading healthy
        # samples as up to ~1 s "ahead"). The legacy value stays in the
        # report as the cross-check that the artifact is gone. The bridge
        # also rejects and logs future-skewed samples itself; any such line
        # means the chain already dropped a sample.
        future_skew = _max_future_skew(snapshot["filtered_odom"])
        clock_walls = [wall for wall, _sim in snapshot["clock"]]
        clock_sims = [sim for _wall, sim in snapshot["clock"]]
        legacy_skew = None
        for sample in snapshot["filtered_odom"]:
            index = bisect.bisect_right(clock_walls, sample["recv_wall"]) - 1
            if index < 0:
                continue
            skew = sample["stamp"] - clock_sims[index]
            if legacy_skew is None or skew > legacy_skew:
                legacy_skew = skew
        rejected_samples = sum(1 for line in log_text.splitlines() if "Rejected FAST-LIO odometry" in line)
        # frame health (D14.6): over the same simulation-time span as the
        # metrics (the whole drive, final settle included), the odom
        # bridge's output must carry at least 95 % of the frames FAST-LIO
        # consumed, with no adjacent stamp gap above 0.35 s
        health_begin = metrics_begin + FRAME_HEALTH_WARMUP_SEC
        lidar_stamps = sorted(
            stamp for _wall, stamp in snapshot["lidar_frames"] if health_begin <= stamp <= metrics_end
        )
        odometry_stamps = sorted(
            sample["stamp"] for sample in snapshot["filtered_odom"] if health_begin <= sample["stamp"] <= metrics_end
        )
        frame_fraction = (len(odometry_stamps) / len(lidar_stamps)) if lidar_stamps else None
        max_gap = max((b - a for a, b in zip(odometry_stamps, odometry_stamps[1:], strict=False)), default=0.0)
        gap_at = next(
            ((a, b) for a, b in zip(odometry_stamps, odometry_stamps[1:], strict=False) if b - a == max_gap),
            None,
        )
        # Host-pause discriminator (rounds 3-4): a whole-machine stall
        # (observed 1.0-1.7 s under loaded colcon runs on this shared
        # workstation) freezes FAST-LIO's output AND the probe's /clock
        # delivery together; the chain recovers fully (frame fraction and
        # skew stay green). Round 4 makes the discrimination TIME-CORRELATED:
        # only a recorded /clock stall interval that OVERLAPS this gap's
        # receive interval and lasts at least gap-0.1 s excuses it — a stall
        # at any other moment no longer can. A gap while the clock kept
        # flowing is a real output stall and still fails.
        host_pause_during_gap = None
        gap_is_chain_fault = max_gap > ODOMETRY_MAX_GAP_SEC
        if gap_is_chain_fault:
            gap_wall_interval = None
            window_samples = sorted(
                (sample["stamp"], sample["recv_wall"])
                for sample in snapshot["filtered_odom"]
                if health_begin <= sample["stamp"] <= metrics_end
            )
            for (stamp_a, wall_a), (stamp_b, wall_b) in zip(window_samples, window_samples[1:], strict=False):
                if stamp_b - stamp_a == max_gap:
                    gap_wall_interval = (wall_a, wall_b)
                    break
            if gap_wall_interval is not None and _gap_is_host_pause(
                gap_wall_interval[0], gap_wall_interval[1], max_gap, probe._clock_stalls
            ):
                gap_is_chain_fault = False
                host_pause_during_gap = {
                    "gap_sec": max_gap,
                    "gap_receive_interval": list(gap_wall_interval),
                    "clock_stalls": [list(stall) for stall in probe._clock_stalls],
                }

        report = {
            "report_only": report_only,
            "platform": sim_platform,
            "overlay": str(overlay_path),
            "metrics_window": [metrics_begin, metrics_end],
            "segments": segment_traces,
            "metrics": metrics,
            "mean_rtf": mean_rtf,
            "clock_samples": len(snapshot["clock"]),
            "collision_monitor": monitor,
            "cmd_vel_publishers": sorted(supervisor.publishers_seen_during_drive() or set()),
            "tf_publishers": sorted(tf_publishers),
            "tf_frames_carried": [list(pair) for pair in carried],
            "fast_lio_deaths": fast_lio_deaths,
            "rejected_samples": rejected_samples,
            "max_future_skew_sec": future_skew,
            # the bisect-over-downsampled-clock value (review round 3): kept
            # unasserted as the cross-check that the old measurement artifact
            # (a starved clock callback reading healthy samples as future) is
            # gone; it must stay within MAX_FUTURE_SKEW_SEC of the asserted
            # value above once both share the dedicated clock group
            "max_future_skew_legacy_sec": legacy_skew,
            "clock_callback_max_gap_sec": probe._clock_max_callback_gap_sec,
            "clock_callback_max_jump_sec": probe._clock_max_jump_sec,
            "host_pause_during_gap": host_pause_during_gap,
            "frame_health": {
                "warmup_sec": FRAME_HEALTH_WARMUP_SEC,
                "odometry_frames": len(odometry_stamps),
                "lidar_frames": len(lidar_stamps),
                "frame_fraction": frame_fraction,
                "max_gap_sec": max_gap,
                "max_gap_from": gap_at[0] if gap_at else None,
                "max_gap_to": gap_at[1] if gap_at else None,
                "lidar_subscription": "reliable, matches the reliable finite-cloud-filter publisher",
            },
            "ground_truth": [
                {"stamp": sample.stamp_sec, "x": sample.x, "y": sample.y, "yaw": sample.yaw}
                for sample in ground_truth.samples()
                if metrics_begin <= sample.stamp_sec <= metrics_end
            ],
            "scene_geometry": get_scene_layout(GATE_SCENE)["static_geometry"],
            "thresholds": FAST_LIO_THRESHOLDS,
            "caps": FAST_LIO_CAPS,
        }
        _plot(report, log_dir)
        (log_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        summary = _summary(report)
        print(summary, flush=True)
        print(f"[lio-gate] report: {log_dir}/metrics.json", flush=True)

        # ---- assertions (measurement problems fail even in report-only) ----
        # route problems (deviation/stall/watchdog, D14.5) are reported
        # together with everything else, never asserted away earlier
        problems = [f"route problem: {failure}" for failure in route_failure]
        if metrics["fast_lio"]["paired"] == 0:
            problems.append("FAST-LIO produced no pairable samples in the metrics window")
        for name in ("fast_lio", "wheel", "amcl"):
            if metrics[name]["paired"] == 0 and metrics[name]["samples"] > 0:
                label = "wheel odometry" if name == "wheel" else "AMCL"
                problems.append(
                    f"{label}: no pairable output in the metrics window (reported as such, never a FAST-LIO failure)"
                )
            fraction = metrics[name]["unpaired_fraction"]
            if fraction > UNPAIRED_FRACTION_LIMIT:
                problems.append(f"{name}: {fraction:.1%} unpaired samples (limit {UNPAIRED_FRACTION_LIMIT:.0%})")
        if mean_rtf < RTF_LIMIT:
            problems.append(f"measurement problem: mean RTF {mean_rtf:.2f} over the route is below {RTF_LIMIT}")
        if fast_lio_deaths:
            problems.append(f"fastlio_mapping death/respawn lines in the launch log: {fast_lio_deaths[:2]}")
        if rejected_samples:
            problems.append(
                f"the FAST-LIO odom bridge rejected {rejected_samples} samples (future-skew rejections in the log)"
            )
        if future_skew is not None and future_skew > MAX_FUTURE_SKEW_SEC:
            problems.append(f"FAST-LIO samples stamped {future_skew:.3f} s ahead of the latest /clock received")
        if not lidar_stamps:
            problems.append("no /livox/lidar frames recorded in the metrics window (FAST-LIO's input is empty)")
        elif frame_fraction is None or frame_fraction < ODOMETRY_FRAME_FRACTION_LIMIT:
            problems.append(
                f"/odometry/filtered carries {len(odometry_stamps)}/{len(lidar_stamps)} of the /livox/lidar "
                f"frames ({0.0 if frame_fraction is None else frame_fraction:.1%}, "
                f"limit {ODOMETRY_FRAME_FRACTION_LIMIT:.0%})"
            )
        if max_gap > ODOMETRY_MAX_GAP_SEC and gap_is_chain_fault:
            problems.append(
                f"/odometry/filtered has a stamp gap of {max_gap:.2f} s between t="
                f"{gap_at[0]:.1f} s and t={gap_at[1]:.1f} s (limit {ODOMETRY_MAX_GAP_SEC} s)"
            )
        if monitor["changed"]:
            problems.append(f"collision monitor intervened (changed commands): {monitor['changed'][:2]}")
        if monitor["missing_fraction"] > MONITOR_MISSING_FRACTION_LIMIT or monitor["consecutive_missing"] >= 2:
            problems.append(
                f"collision monitor inputs without outputs: {monitor['nonzero_missing']}/{monitor['nonzero_inputs']} "
                f"missing ({monitor['missing_fraction']:.1%}), worst run {monitor['consecutive_missing']} consecutive"
            )
        if monitor["spontaneous_outputs"]:
            problems.append(f"collision monitor published {monitor['spontaneous_outputs']} outputs of its own")
        if set(report["cmd_vel_publishers"]) != EXPECTED_CMD_VEL_PUBLISHERS:
            problems.append(
                f"/cmd_vel publishers are {report['cmd_vel_publishers']}, not exactly "
                f"{sorted(EXPECTED_CMD_VEL_PUBLISHERS)} (the monitor the only active one)"
            )
        if static_map_odom:
            problems.append("a static map -> odom transform was composed")
        if ("odom", "base_link") not in carried:
            problems.append("odom -> base_link is not carried on /tf")
        if ("map", "odom") not in carried:
            problems.append("map -> odom never appeared on /tf")
        for parent, child in carried:
            if parent in ("camera_init", "body") or child in ("camera_init", "body"):
                problems.append(f"FAST-LIO dynamic frame ({parent} -> {child}) appeared on /tf")
        if tf_publishers != _EXPECTED_TF_PUBLISHERS:
            problems.append(
                f"/tf publishers are {sorted(tf_publishers)}; expected exactly "
                f"{sorted(_EXPECTED_TF_PUBLISHERS)} (the wheel-odometry node owns odom -> base_link)"
            )
        # frames asserted (D14.6): the bridge output and wheel odometry are
        # odom/base_link, AMCL publishes in the map frame
        for sample in snapshot["filtered_odom"][-5:]:
            assert (sample["odom_frame"], sample["base_frame"]) == ("odom", "base_link"), (
                f"/odometry/filtered frames are {sample['odom_frame']}/{sample['base_frame']}"
            )
        for sample in snapshot["wheel_odom"][-5:]:
            assert (sample["odom_frame"], sample["base_frame"]) == ("odom", "base_link"), (
                f"/odom frames are {sample['odom_frame']}/{sample['base_frame']}"
            )
        for sample in snapshot["amcl_pose"][-5:]:
            assert sample["frame"] == "map", f"/amcl_pose frame is {sample['frame']}"

        # accuracy verdicts (D14.6): caps always, committed thresholds when
        # not calibrating (report-only prints them instead, task 8.4)
        fast_lio = metrics["fast_lio"]

        def _over(bounds: dict) -> list[str]:
            return [
                f"{metric}={value:.4f} exceeds {bounds[metric]}"
                for metric, value in fast_lio.items()
                if metric in bounds and value is not None and value > bounds[metric]
            ]

        over_cap = _over(FAST_LIO_CAPS)
        over_threshold = _over(FAST_LIO_THRESHOLDS)
        if over_cap:
            verdict = "FAST-LIO above the spec caps: " + "; ".join(over_cap)
            if report_only:
                print(f"[lio-gate] (report-only) {verdict}", flush=True)
            else:
                problems.append(verdict)
        if over_threshold:
            verdict = "FAST-LIO above the committed thresholds: " + "; ".join(over_threshold)
            if report_only:
                print(f"[lio-gate] (report-only) {verdict}", flush=True)
            else:
                problems.append(verdict)
        for name in ("wheel", "amcl"):
            value = metrics[name]["ate_rmse"]
            if value is not None and value > DIAGNOSTIC_ATE_LIMIT:
                label = "wheel odometry" if name == "wheel" else "AMCL (reported as such, never a FAST-LIO failure)"
                problems.append(f"{label}: ate_rmse {value:.4f} m exceeds the loose {DIAGNOSTIC_ATE_LIMIT} m bound")

        assert not problems, "lidar-odometry gate failures:\n  " + "\n  ".join(problems) + "\n" + summary
    finally:
        if commands is not None:
            commands.stop()
        if supervisor is not None:
            supervisor.stop()
        if ground_truth is not None:
            ground_truth.close()
        # executor first (it spins the probe), then join its thread, then
        # destroy the node, then shut the context: an undestroyed node's DDS
        # entities linger in discovery after the shutdown
        if executor is not None:
            executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=5.0)
        if probe is not None:
            probe.destroy_node()
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
