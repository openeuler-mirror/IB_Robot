"""Navigation E2E on the provider path, both physics platforms (D10/D15).

The production ``lekiwi_lidar`` navigation stage (FAST-LIO, the scan
converter, AMCL, the collision monitor, the navigation admission) runs
through generic launch on the provider's simulator platform in the shipped
``nav_room`` scene — staged from the ``lekiwi_navi_sim`` overlay, whose
staged copy carries the platform under test (the shipped overlay pins
``gazebo``; the D14 gate staging pattern). No test-only odometry
publishers, velocity relays or static map-to-odometry transforms: the test
bridges the simulator's world pose itself for its assertions.

The test is parametrized over ``PHYSICS_SIM_PLATFORMS`` (gazebo, mujoco):
goals, thresholds and assertions are identical on both platforms
(lekiwi-mujoco-runtime-platform D15). A platform runs only when its
capabilities and its simulator ENVIRONMENT are present; every do-not-run
reason goes through the required-platform rule —
``IBROBOT_REQUIRE_SIM_PLATFORMS`` listing the platform turns the skip into a
failure, so acceptance runs cannot silently lose a platform. The openEuler
``NAV_TEST_PROFILE`` minimal/full policy is kept.

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
import xml.etree.ElementTree as ET
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
from rclpy.qos import DurabilityPolicy as QoSDurability
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy as QoSReliability
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

from ibrobot_msgs.msg import RuntimeStatus
from robot_runtime.profile import LIDAR_SIM_PLATFORMS, PHYSICS_SIM_PLATFORMS
from robot_runtime.testing.sim_environment import (
    REQUIRED_SIM_PLATFORMS_ENV,
    environment_missing,
    skip_or_fail,
)
from robot_runtime.testing.sim_ground_truth import (
    RUN_MARKER_ENV,
    UnsupportedGroundTruth,
    leftover_domain_processes,
    open_ground_truth,
    terminate_marked_processes,
)
from sim_models.scene_compiler import get_scene_file
from test.navigation_sim_support import load_staged_navigation_config, stage_navigation_config

POSITION_TOLERANCE = 0.25
SIM_STARTUP_TIMEOUT = 240.0
NAV2_SETTLE_TIME = 5.0
NAV_GOAL_TIMEOUT = 180.0
STOP_PLANAR_LIMIT = 0.02
STOP_YAW_RATE_LIMIT = 0.05
NAV_TEST_PROFILE_ENV = "NAV_TEST_PROFILE"
NAV2_TEST_PARAMS_FILE_ENV = "NAV2_TEST_PARAMS_FILE"

WORLD_NAME = "demo"  # the scene template's world name
MODEL_NAME = "lekiwi_robot"  # the runtime's model name in the simulator
SCENE_NAME = "nav_room"

SPAWN_X = -1.5
SPAWN_Y = -1.5
POINT_A_X = 1.0
POINT_A_Y = 1.0

#: The complete /tf publisher set of the navigation stage (the D14 gate's
#: own expectation), as fully-qualified node identities: AMCL (map -> odom),
#: the base node (odom -> base_link) and robot_state_publisher (the URDF's
#: joint pairs). Exactly this set, each with exactly ONE /tf endpoint — any
#: fourth source, a duplicate of a listed node, or a same-name node under
#: another namespace fails the ownership claim below (review round 2).
_EXPECTED_TF_PUBLISHERS = {"/amcl", "/lekiwi_base", "/robot_state_publisher"}

#: The only dynamic pairs the navigation stage may carry beyond the URDF's
#: own joint pairs: the base node's single transform (odom -> base_link,
#: pinned by lekiwi_robot's test_base_tf_contract.py) and AMCL's map -> odom.
_NON_URDF_DYNAMIC_PAIRS = {("odom", "base_link"), ("map", "odom")}


def _publisher_identities(infos) -> list[str]:
    """The fully-qualified node identity of every /tf publisher ENDPOINT.

    Review round 2: ``info.node_name`` alone is not an identity — a second
    ``/amcl``, or an ``/amcl`` under another namespace, would collapse into
    ``"amcl"`` and pass. The namespace is joined onto the name, and one
    entry per endpoint keeps multiplicity visible: a node with two /tf
    endpoints is a duplicate even when its name matches the expectation.
    """
    identities = []
    for info in infos:
        namespace = (info.node_namespace or "").rstrip("/")
        name = info.node_name or ""
        identities.append(f"{namespace}/{name}" if namespace else f"/{name}")
    return identities


def _urdf_frame_pairs(urdf_text: str) -> tuple[set[str], set[tuple[str, str]], list[str]]:
    """(link names, joint (parent, child) pairs, problems) of a URDF string.

    Only KINEMATIC joints are collected: the same ``<joint>`` tag also
    appears inside ``<transmission>`` and ``<ros2_control>`` as a bare
    reference with no ``<parent>``/``<child>`` (the real LeKiwi description
    carries three of each), and those are skipped. Missing
    ``name``/``link`` attributes on a joint that DOES claim kinematics are
    reported as problems instead of raising: a malformed description must
    fail the ownership check with a named reason, not a KeyError (review
    round 1: the first version of this parse crashed on live legs — once on
    an XPath construct ElementTree does not support, once on the
    transmission references).
    """
    try:
        urdf = ET.fromstring(urdf_text)
    except ET.ParseError as exc:
        return set(), set(), [f"the live /robot_description is not parseable XML: {exc}"]
    links: set[str] = set()
    joints: set[tuple[str, str]] = set()
    problems: list[str] = []
    for link in urdf.iter("link"):
        name = link.get("name")
        if name:
            links.add(name)
    for joint in urdf.iter("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None and child is None:
            # a <joint> element without kinematics: the joint REFERENCES inside
            # <transmission> and <ros2_control> (e.g. <joint name="7"
            # role="wheel"/>) name the same joints for the control stack and
            # carry no parent/child — not kinematic joints, out of scope here
            continue
        name = joint.get("name", "<unnamed>")
        parent_name = parent.get("link") if parent is not None else None
        child_name = child.get("link") if child is not None else None
        if parent_name and child_name:
            joints.add((parent_name, child_name))
        else:
            problems.append(f"URDF joint {name!r} has no parent/child link name")
    return links, joints, problems


def _map_odom_ownership_problems(
    urdf_text: str,
    dynamic_pairs: set[tuple[str, str]],
    static_pairs: set[tuple[str, str]],
    tf_publisher_endpoints: list[str],
) -> list[str]:
    """Why AMCL is NOT provably the sole publisher of map -> odom ([] when it is).

    Pure function of the observed graph data, unit-tested without a
    simulator (review round 1: the attribution logic must be exercised on
    every machine, not only where the simulators are installed):

    1. the /tf publisher ENDPOINTS are exactly the expected
       fully-qualified identities, each with exactly one endpoint — any
       fourth source, a duplicate of a listed node (two endpoints, same
       identity) or a same-name node under another namespace is named
       (review round 2);
    2. robot_state_publisher cannot carry map -> odom: ``map`` is not a
       URDF link and (map, odom) is not a URDF joint pair of the live
       /robot_description;
    3. map -> odom is never composed statically;
    4. no unexpected dynamic pair appears beyond the URDF's joint pairs,
       the base node's odom -> base_link and AMCL's own map -> odom.

    With all four true and map -> odom carried dynamically (the caller
    waits for it), AMCL is the only node that can be publishing the pair.
    The caller re-runs this after its goals: a broadcaster that joins the
    graph late is caught at the end of the observation window.
    """
    problems: list[str] = []
    counts: dict[str, int] = {}
    for identity in tf_publisher_endpoints:
        counts[identity] = counts.get(identity, 0) + 1
    duplicates = {identity: count for identity, count in counts.items() if count > 1}
    if duplicates:
        problems.append(
            f"/tf publisher endpoints with unexpected multiplicity: "
            f"{sorted(f'{identity} x{count}' for identity, count in duplicates.items())}; expected exactly one each"
        )
    observed = set(counts)
    if observed != _EXPECTED_TF_PUBLISHERS:
        problems.append(
            f"/tf publishers are {sorted(observed)}; expected exactly "
            f"{sorted(_EXPECTED_TF_PUBLISHERS)} (AMCL must be the sole map -> odom publisher)"
        )
    links, joint_pairs, parse_problems = _urdf_frame_pairs(urdf_text)
    problems.extend(parse_problems)
    if "map" in links:
        problems.append("map is a URDF link: robot_state_publisher could carry map -> odom")
    if ("map", "odom") in joint_pairs:
        problems.append("map -> odom is a URDF joint pair: robot_state_publisher could carry map -> odom")
    if ("map", "odom") in static_pairs:
        problems.append("a static map -> odom transform was composed")
    unexpected_pairs = dynamic_pairs - joint_pairs - _NON_URDF_DYNAMIC_PAIRS
    if unexpected_pairs:
        problems.append(f"unexpected dynamic TF pairs on /tf: {sorted(unexpected_pairs)}")
    return problems


# ---------------------------------------------------------------------------
# Gazebo-free: the staged overlay resolves to the intended navigation stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sim_platform", PHYSICS_SIM_PLATFORMS)
def test_staged_navigation_overlay_resolves_the_production_stage(tmp_path, sim_platform):
    config = load_staged_navigation_config(stage_navigation_config(tmp_path, sim_platform=sim_platform))

    # the provider path with the production navigation stage selected
    assert config["runtime"]["provider"] == "lekiwi_robot"
    assert config["runtime"]["profile"] == "lekiwi_lidar"
    assert config["nav_stage"] == "navigation"

    # the platform under test and the scene with its spawn (inside the stage)
    assert config["simulation"]["platform"] == sim_platform
    assert config["simulation"]["scene"] == SCENE_NAME
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


def test_shipped_navigation_overlay_pins_gazebo():
    """D15: the shipped overlay stays Gazebo-pinned — only the staged copy
    carries the platform under test (the D14 gate staging pattern)."""
    from test.navigation_sim_support import _package_dir

    shipped = yaml.safe_load((_package_dir() / "test" / "config" / "lekiwi_navi_sim.yaml").read_text(encoding="utf-8"))
    assert shipped["robot"]["nav_stages"]["navigation"]["simulation"]["platform"] == "gazebo"


def test_staging_fails_loudly_without_a_simulation_section(tmp_path, monkeypatch):
    """An overlay without ``navigation.simulation`` must fail staging, not
    silently stage a platform nobody set (the loud-staging rule, D15)."""
    import test.navigation_sim_support as support

    package = tmp_path / "pkg"
    (package / "config" / "robots").mkdir(parents=True)
    (package / "config" / "robots" / "lekiwi_lidar.yaml").write_text("robot: {}\n", encoding="utf-8")
    (package / "test" / "config").mkdir(parents=True)
    (package / "test" / "config" / "lekiwi_navi_sim.yaml").write_text(
        "robot:\n  base_config: lekiwi_lidar\n  nav_stages:\n    navigation: {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(support, "_package_dir", lambda: package)
    with pytest.raises(ValueError, match="navigation.simulation"):
        support.stage_navigation_config(tmp_path / "out", sim_platform="mujoco")


# ---------------------------------------------------------------------------
# D15: the capability probe (the D14.2 lidar-odometry gate pattern)
# ---------------------------------------------------------------------------


def missing_navigation_capabilities(sim_platform: str) -> list[str]:
    """Every capability the navigation e2e needs that ``sim_platform`` lacks.

    Inspects code and configuration only — no process, no simulator — so a
    platform that passes the probe but is misconfigured still fails at
    launch, never a skip. Mirrors the D14.2 gate probe; the profile check is
    the robot_config surface (the staged overlay resolves with the platform
    selected on the provider path) because the provider's runtime profile
    file lives in lekiwi_robot, which robot_config must not import —
    lekiwi_robot's own gates pin the profile's ``sim_platforms``.
    """
    missing = []
    try:
        open_ground_truth(sim_platform, world=WORLD_NAME, model=MODEL_NAME)
    except UnsupportedGroundTruth as error:
        missing.append(f"ground truth: {error}")
    if sim_platform not in LIDAR_SIM_PLATFORMS:
        missing.append(f"simulated lidar: {sim_platform!r} is not in LIDAR_SIM_PLATFORMS {list(LIDAR_SIM_PLATFORMS)}")
    try:
        get_scene_file(SCENE_NAME, sim_platform)
    except (ValueError, FileNotFoundError) as error:
        missing.append(f"scene template: {error}")
    if sim_platform not in PHYSICS_SIM_PLATFORMS:
        missing.append(f"sim platform: {sim_platform!r} is not in PHYSICS_SIM_PLATFORMS {list(PHYSICS_SIM_PLATFORMS)}")
    else:
        with tempfile.TemporaryDirectory(prefix="robot_config_nav_probe_") as probe_dir:
            try:
                config = load_staged_navigation_config(
                    stage_navigation_config(Path(probe_dir), sim_platform=sim_platform)
                )
                resolved = str((config.get("simulation") or {}).get("platform", ""))
                if resolved != sim_platform:
                    missing.append(f"staged overlay: resolved platform {resolved!r} is not {sim_platform!r}")
            except Exception as error:  # noqa: BLE001  (any loader failure is one missing capability)
                missing.append(f"staged overlay: does not resolve for {sim_platform!r}: {error}")
    return missing


@pytest.mark.parametrize("sim_platform", PHYSICS_SIM_PLATFORMS)
def test_no_missing_navigation_capability(sim_platform):
    """D15: both physics platforms pass every capability check, so the e2e
    runs there, never a capability skip."""
    assert missing_navigation_capabilities(sim_platform) == []


def test_a_synthetic_platform_reports_missing_capabilities():
    """The probe itself still names every missing capability for a platform
    nothing supports (the skip-reason surface stays honest)."""
    missing = missing_navigation_capabilities("synthetic")
    assert len(missing) == 4, f"expected all four capabilities missing, got {missing}"
    assert missing[0].startswith("ground truth") and "synthetic" in missing[0]
    assert missing[1].startswith("simulated lidar") and "LIDAR_SIM_PLATFORMS" in missing[1]
    assert missing[2].startswith("scene template")
    assert missing[3].startswith("sim platform") and "PHYSICS_SIM_PLATFORMS" in missing[3]


# ---------------------------------------------------------------------------
# The profile policy and the required-platform rule (D15: nothing may bypass
# IBROBOT_REQUIRE_SIM_PLATFORMS; an invalid profile value fails outright)
# ---------------------------------------------------------------------------


def _is_openeuler() -> bool:
    try:
        with open("/etc/os-release", encoding="utf-8") as os_release:
            return any(line.strip().lower() == "id=openeuler" for line in os_release)
    except OSError:
        return False


def _select_test_profile():
    """The E2E profile to run, or None when this host defaults the e2e off.

    This helper never skips by itself — the caller routes every do-not-run
    reason through ``_nav_skip_or_fail`` so ``IBROBOT_REQUIRE_SIM_PLATFORMS``
    turns it into a failure. An explicitly invalid ``NAV_TEST_PROFILE`` value
    fails outright: a typo in an acceptance run must not silently become a
    skip (the D14 gate's review-round-2 rule).
    """
    requested = os.environ.get(NAV_TEST_PROFILE_ENV, "").strip().lower()
    if requested:
        if requested not in {"full", "minimal"}:
            pytest.fail(f"invalid {NAV_TEST_PROFILE_ENV}={requested!r}; expected 'full' or 'minimal'")
        return requested
    if _is_openeuler():
        return None  # the openEuler default: minimal E2E, the e2e is off
    return "full"


def _nav_skip_or_fail(sim_platform: str, reason: str) -> None:
    """Skip — unless acceptance runs required this platform: then fail."""
    skip_or_fail(sim_platform, reason, context="navigation simulation")


def _validate_requested_profile() -> None:
    """Fail outright on an invalid explicitly-set NAV_TEST_PROFILE value.

    Review round 1: the fixture validated the value only AFTER the
    capability and environment checks, so on a host whose simulator or
    environment is unavailable a typo'd ``NAV_TEST_PROFILE`` silently
    became a skip — the acceptance run then lost the e2e without a red
    result. Validation runs BEFORE any skip decision can fire; the
    effective-profile selection (including the openEuler default) stays
    where it was, last in the decision sequence.
    """
    requested = os.environ.get(NAV_TEST_PROFILE_ENV, "").strip().lower()
    if requested and requested not in {"full", "minimal"}:
        pytest.fail(f"invalid {NAV_TEST_PROFILE_ENV}={requested!r}; expected 'full' or 'minimal'")


def _decide_navigation_run(sim_platform: str) -> None:
    """The whole do-not-run decision sequence for one platform, in order.

    Review round 1: an invalid profile value fails FIRST (before any skip
    can fire — a typo must never degrade into a skip on a host where the
    environment is missing), then the D14 order stays: capabilities,
    environment, effective profile. Every do-not-run reason routes through
    ``_nav_skip_or_fail`` so ``IBROBOT_REQUIRE_SIM_PLATFORMS`` turns it
    into a failure.
    """
    _validate_requested_profile()
    missing = missing_navigation_capabilities(sim_platform)
    if missing:
        _nav_skip_or_fail(sim_platform, f"platform {sim_platform!r} lacks capabilities: {'; '.join(missing)}")
    environment_unusable = environment_missing(sim_platform)
    if environment_unusable:
        _nav_skip_or_fail(sim_platform, f"the {sim_platform} simulator cannot run here: {environment_unusable}")
    selection = _select_test_profile()
    if selection in (None, "minimal"):
        why = (
            "the openEuler default minimal E2E profile disables navigation simulation"
            if selection is None
            else "the minimal E2E profile disables navigation simulation"
        )
        _nav_skip_or_fail(sim_platform, f"{why}; set {NAV_TEST_PROFILE_ENV}=full to run it")


def test_an_invalid_profile_value_fails_outright(monkeypatch):
    """A typo'd NAV_TEST_PROFILE must fail, never skip — an acceptance run
    with REQUIRE set must not silently lose the e2e to a bad env value."""
    monkeypatch.setenv(NAV_TEST_PROFILE_ENV, "bogus")
    monkeypatch.delenv(REQUIRED_SIM_PLATFORMS_ENV, raising=False)
    with pytest.raises(pytest.fail.Exception, match="invalid .*expected 'full' or 'minimal'"):
        _select_test_profile()


def test_an_invalid_profile_fails_even_when_the_run_would_skip(monkeypatch):
    """Review round 1: the invalid value fails BEFORE the capability and
    environment skips — on a host without the simulator a typo'd
    ``NAV_TEST_PROFILE`` must surface as a red result, not vanish into the
    environment skip. A synthetic platform name makes every later check a
    skip candidate (missing capabilities, no environment check), so only
    the early validation can produce this failure."""
    monkeypatch.setenv(NAV_TEST_PROFILE_ENV, "bogus")
    monkeypatch.delenv(REQUIRED_SIM_PLATFORMS_ENV, raising=False)
    with pytest.raises(pytest.fail.Exception, match="invalid .*expected 'full' or 'minimal'"):
        _decide_navigation_run("a_platform_without_capabilities_or_environment")


# ---------------------------------------------------------------------------
# map -> odom ownership attribution (D15, review round 1): pure logic, no
# simulator — the same class of live-only bug this round caught (an
# untestable inline check that crashed on its first execution) must not be
# reintroduced.
# ---------------------------------------------------------------------------

_MINIMAL_URDF = """
<robot name="lekiwi">
  <link name="base_link"/>
  <link name="left_wheel_link"/>
  <joint name="base_left" type="fixed">
    <parent link="base_link"/>
    <child link="left_wheel_link"/>
  </joint>
  <transmission name="wheel_7_transmission">
    <plugin>transmission_interface/SimpleTransmission</plugin>
    <joint name="7" role="wheel"/>
    <actuator name="wheel_7" role="wheel"/>
  </transmission>
  <ros2_control name="RobotSystem" type="system">
    <joint name="7">
      <command_interface name="velocity"/>
    </joint>
  </ros2_control>
</robot>
"""


def _live_like_graph():
    """The observed data of a healthy navigation stage: URDF joint pairs,
    the base node's odom -> base_link, AMCL's map -> odom, the exact three
    fully-qualified publishers, one /tf endpoint each, nothing static."""
    return {
        "urdf_text": _MINIMAL_URDF,
        "dynamic_pairs": {("base_link", "left_wheel_link"), ("odom", "base_link"), ("map", "odom")},
        "static_pairs": {("base_link", "back_wheel_link")},
        "tf_publisher_endpoints": sorted(_EXPECTED_TF_PUBLISHERS),
    }


def test_amcl_is_the_sole_map_odom_publisher_on_the_healthy_graph():
    assert _map_odom_ownership_problems(**_live_like_graph()) == []


def test_a_second_endpoint_with_the_same_identity_fails_the_ownership():
    """A duplicate /amcl (two /tf endpoints, one identity) is not
    indistinguishable from the real AMCL — multiplicity must reject it
    (review round 2)."""
    data = _live_like_graph()
    data["tf_publisher_endpoints"] = sorted(data["tf_publisher_endpoints"]) + ["/amcl"]
    problems = _map_odom_ownership_problems(**data)
    assert any("/amcl x2" in problem for problem in problems), problems
    assert any("unexpected multiplicity" in problem for problem in problems), problems


def test_a_same_name_node_under_another_namespace_fails_the_ownership():
    """/other/amcl collapses into "amcl" when only node_name is compared —
    the fully-qualified identity must name it (review round 2)."""
    data = _live_like_graph()
    data["tf_publisher_endpoints"] = sorted(_EXPECTED_TF_PUBLISHERS) + ["/other/amcl"]
    problems = _map_odom_ownership_problems(**data)
    assert any("'/other/amcl'" in problem for problem in problems), problems


def test_a_late_map_odom_broadcaster_is_caught_by_the_post_goal_recheck():
    """The pre-goal graph is healthy; a broadcaster that joins DURING the
    goals is invisible before and must fail the ownership at the end of
    the observation window — the caller's re-check pattern."""
    before = _live_like_graph()
    assert _map_odom_ownership_problems(**before) == []
    after = _live_like_graph()
    after["tf_publisher_endpoints"] = sorted(_EXPECTED_TF_PUBLISHERS) + ["/fake_map_odom_publisher"]
    problems = _map_odom_ownership_problems(**after)
    assert any("'/fake_map_odom_publisher'" in problem for problem in problems), problems


def test_an_unexpected_dynamic_pair_fails_the_ownership():
    data = _live_like_graph()
    data["dynamic_pairs"] = set(data["dynamic_pairs"]) | {("map", "base_link")}
    problems = _map_odom_ownership_problems(**data)
    assert any("map', 'base_link" in problem for problem in problems), problems


def test_a_static_map_odom_composition_fails_the_ownership():
    data = _live_like_graph()
    data["static_pairs"] = set(data["static_pairs"]) | {("map", "odom")}
    assert any("static map -> odom" in problem for problem in _map_odom_ownership_problems(**data))


def test_a_urdf_that_could_carry_map_odom_fails_the_ownership():
    data = _live_like_graph()
    data["urdf_text"] = _MINIMAL_URDF.replace(
        "</robot>",
        '  <link name="odom"/>\n'
        '  <link name="map"/>\n'
        '  <joint name="map_odom" type="fixed">\n'
        '    <parent link="map"/>\n'
        '    <child link="odom"/>\n'
        "  </joint>\n"
        "</robot>",
    )
    problems = _map_odom_ownership_problems(**data)
    assert any("map is a URDF link" in problem for problem in problems), problems
    assert any("map -> odom is a URDF joint pair" in problem for problem in problems), problems


def test_a_malformed_or_unparseable_description_fails_with_a_named_problem():
    data = _live_like_graph()
    data["urdf_text"] = "<robot name="
    problems = _map_odom_ownership_problems(**data)
    assert any("not parseable XML" in problem for problem in problems), problems
    broken_joint = _MINIMAL_URDF.replace('<parent link="base_link"/>', "<parent/>")
    data["urdf_text"] = broken_joint
    assert any("no parent/child link name" in problem for problem in _map_odom_ownership_problems(**data))


def test_the_minimal_profile_with_a_platform_required_fails_not_skips(monkeypatch):
    monkeypatch.setenv(NAV_TEST_PROFILE_ENV, "minimal")
    monkeypatch.setenv(REQUIRED_SIM_PLATFORMS_ENV, "mujoco")
    assert _select_test_profile() == "minimal"
    with pytest.raises(pytest.fail.Exception, match="was required by IBROBOT_REQUIRE_SIM_PLATFORMS"):
        _nav_skip_or_fail("mujoco", "the minimal E2E profile disables navigation simulation")


def test_the_minimal_profile_without_a_requirement_stays_a_skip(monkeypatch):
    monkeypatch.setenv(NAV_TEST_PROFILE_ENV, "minimal")
    monkeypatch.delenv(REQUIRED_SIM_PLATFORMS_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="minimal E2E profile"):
        _nav_skip_or_fail("gazebo", "the minimal E2E profile disables navigation simulation")


# ---------------------------------------------------------------------------
# The live run
# ---------------------------------------------------------------------------


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
        self.robot_description: str | None = None
        self.scans = []
        self.filtered_odometry = []
        self.create_subscription(RuntimeStatus, "/runtime_status", self._status, 10)
        self.create_subscription(Bool, "/motion_mode/base_navigation_enabled", self._gate, 10)
        self.create_subscription(TFMessage, "/tf", self.tf_frames.append, 50)
        self.create_subscription(TFMessage, "/tf_static", self.tf_static_frames.append, 10)
        # the description robot_state_publisher latched at startup (TRANSIENT
        # LOCAL): map->odom ownership is eliminated against the URDF's own
        # joint pairs, not against a hardcoded frame list
        self.create_subscription(
            String,
            "/robot_description",
            lambda message: setattr(self, "robot_description", message.data),
            QoSProfile(depth=1, reliability=QoSReliability.RELIABLE, durability=QoSDurability.TRANSIENT_LOCAL),
        )
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
        # RTF evidence only (never asserted): one ratio over the whole run
        # window, the D14 gate's measure — a mean of per-sample ratios is
        # inflated by /clock bursts
        self.clock: list[tuple[float, float]] = []  # (wall receive time, sim time)
        self.create_subscription(Clock, "/clock", self._on_clock, 100)

    def _status(self, msg: RuntimeStatus) -> None:
        self.status = msg

    def _gate(self, msg: Bool) -> None:
        self.gate_enabled = msg

    def _on_clock(self, msg: Clock) -> None:
        self.clock.append((time.time(), msg.clock.sec + msg.clock.nanosec / 1e9))

    def mean_rtf(self) -> float | None:
        """Simulation seconds per wall second over the recorded window."""
        if len(self.clock) < 2 or self.clock[-1][0] <= self.clock[0][0]:
            return None
        return (self.clock[-1][1] - self.clock[0][1]) / (self.clock[-1][0] - self.clock[0][0])


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


@pytest.fixture(params=PHYSICS_SIM_PLATFORMS)
def simulation_env(request):
    sim_platform = request.param
    # the D14 decision order, with the profile value validated FIRST
    # (review round 1): capabilities, environment, profile — every
    # do-not-run reason goes through the required-platform rule
    _decide_navigation_run(sim_platform)

    params_override = os.environ.get(NAV2_TEST_PARAMS_FILE_ENV, "").strip()
    if params_override and not Path(params_override).expanduser().is_file():
        pytest.fail(f"{NAV2_TEST_PARAMS_FILE_ENV} does not exist: {params_override}")

    stage_dir = Path(tempfile.mkdtemp(prefix="robot_config_nav_stage_"))
    config_path = stage_navigation_config(stage_dir, nav2_params_override=params_override, sim_platform=sim_platform)
    log_dir = Path(tempfile.mkdtemp(prefix=f"robot_config_nav_sim_{sim_platform}_"))
    print(f"[navigation-sim] platform {sim_platform}; staged config: {config_path}; logs: {log_dir}", flush=True)

    partition = f"lekiwi_nav_{uuid.uuid4().hex[:8]}"
    environment = dict(os.environ)
    # the neutral per-run marker (review round 1): every platform's launch
    # children carry it, so the teardown sweep below also catches MuJoCo
    # descendants that escape the launch process group (the Gazebo-only
    # partition marker never matched them)
    environment[RUN_MARKER_ENV] = partition
    if sim_platform == "gazebo":
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
    else:
        # MuJoCo has no gz bus; the runtime renders headless and the composed
        # model lands on this run's ROS domain (the D14 gate's D7/D10 pattern)
        environment.update(
            {
                "IBROBOT_MUJOCO_HEADLESS": "1",
                "ROS_LOCALHOST_ONLY": "1",
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
            f"sim_platform:={sim_platform}",
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
            runtime_ready, SIM_STARTUP_TIMEOUT, "runtime ACTIVE in base_navigation with the gate open (admission)"
        ), (
            "navigation admission did not complete; see the launch log; last status: "
            f"lifecycle={getattr(probe.status, 'lifecycle', None)!r} "
            f"active_mode={getattr(probe.status, 'active_mode', None)!r} "
            f"active_controllers={list(getattr(probe.status, 'active_controllers', []) or [])} "
            f"faults={list(getattr(probe.status, 'faults', []) or [])}"
        )
        assert _wait_for(probe.nav_to_pose_client.wait_for_server, SIM_STARTUP_TIMEOUT, "the Nav2 action server")

        # the full perception chain before any goal: FAST-LIO odometry and
        # the derived scan flowing on the simulation clock (AMCL's first
        # map -> odom alone is only its initial pose, published before
        # FAST-LIO is ready — the controller then aborts on a frozen
        # transform)
        assert _wait_for(
            lambda: len(probe.filtered_odometry) >= 5, SIM_STARTUP_TIMEOUT, "FAST-LIO odometry on /odometry/filtered"
        ), f"FAST-LIO produced {len(probe.filtered_odometry)} odometry messages; see the launch log"
        assert _wait_for(lambda: len(probe.scans) >= 5, SIM_STARTUP_TIMEOUT, "the derived scan on /scan")

        # the launch env is passed so a Gazebo ground-truth echo joins the
        # same isolated gz bus as the simulator it observes (on MuJoCo the
        # stream reads the env's ROS domain instead; D9)
        ground_truth = open_ground_truth(sim_platform, world=WORLD_NAME, model=MODEL_NAME, env=environment).start()
        assert _wait_for(ground_truth.position, 30.0, "world-pose samples"), f"ground truth of {MODEL_NAME}"
        time.sleep(NAV2_SETTLE_TIME)

        yield {"probe": probe, "ground_truth": ground_truth, "log_dir": log_dir, "launch": launch}
    finally:
        if probe is not None:
            rtf = probe.mean_rtf()
            if rtf is not None:
                print(f"[navigation-sim] mean RTF over the run: {rtf:.2f}", flush=True)
        if ground_truth is not None:
            ground_truth.close()
        if probe is not None:
            probe.destroy_node()
        if executor is not None:
            executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        # Review round 2: the kill sequence must not abort the teardown — the
        # launch may already be gone (ProcessLookupError) and a SIGKILLed
        # D-state process can outlive the wait; the marker sweep is
        # UNCONDITIONAL (it is what catches process-group escapees).
        try:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(launch.pid), signal.SIGINT)
            launch.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(launch.pid), signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                launch.wait(timeout=5)
        finally:
            terminate_marked_processes({RUN_MARKER_ENV: partition, "GZ_PARTITION": partition})
            # passive leftover scan (review, run-2 startup stall): after the
            # sweep, anything still on this domain is a leak the sweep cannot see
            # or a foreign process — printed so the NEXT leg's health is
            # attributable to a clean graph
            domain_leftovers = leftover_domain_processes(os.environ.get("ROS_DOMAIN_ID", ""))
            if domain_leftovers:
                print(
                    "[navigation-sim] WARNING: processes still alive on ROS domain "
                    f"{os.environ.get('ROS_DOMAIN_ID')!r} after the sweep:\n  " + "\n  ".join(domain_leftovers),
                    flush=True,
                )
        launch_log.close()


def _assert_ground_truth_at(ground_truth, x, y, what):
    position = ground_truth.position()
    assert position is not None, f"{what}: no ground-truth position"
    distance = math.hypot(position[0] - x, position[1] - y)
    print(f"[navigation-sim] {what}: ground truth {position} vs ({x}, {y}) -> {distance:.3f} m", flush=True)
    assert distance < POSITION_TOLERANCE, f"{what}: ground truth {distance:.3f} m from the goal exceeds 0.25 m"


def test_navigation_simulation(simulation_env):
    """Reach a goal, there-and-back, and a cancel-based stop (D10/D15).

    Goals go directly to Nav2's ``navigate_to_pose`` action; the stop cancels
    an active goal and asserts the D3 stop thresholds on the test's own
    ground-truth bridge; the map -> odom transform has exactly one publisher
    (AMCL) and never a static one. Identical on every physics platform.
    """
    probe = simulation_env["probe"]
    ground_truth = simulation_env["ground_truth"]

    # map -> odom ownership (D15): AMCL is the SOLE publisher of the pair —
    # attributed, not assumed. Method (review rounds 1-2): (1) the /tf
    # publisher ENDPOINTS are exactly the three expected fully-qualified
    # node identities, each with one endpoint — a duplicate of a listed
    # node or a same-name node under another namespace fails (the identity
    # keeps namespace and multiplicity, not just the bare name); (2) the
    # remaining two are eliminated by what each can carry, from data rather
    # than name heuristics — robot_state_publisher's whole output is the
    # URDF's joint pairs (checked against the live latched
    # /robot_description: ``map`` is not a URDF link and map -> odom is not
    # a joint pair), and the base node's TF sender emits exactly one
    # odom -> base_link transform per odometry update (pinned by
    # lekiwi_robot's test_base_tf_contract.py); (3) no unexpected dynamic
    # pair appears anywhere on /tf. With map -> odom carried dynamically
    # (asserted below) and never composed statically, AMCL is the only
    # node that can be publishing it. The ownership is RE-CHECKED after the
    # goals (end of the observation window): a broadcaster joining the
    # graph late is caught there.
    publishers = _publisher_identities(probe.get_publishers_info_by_topic("/tf"))
    print(f"[navigation-sim] /tf publisher endpoints: {sorted(publishers)}", flush=True)
    assert _wait_for(lambda: probe.robot_description is not None, 30.0, "the latched /robot_description"), (
        "the latched /robot_description never arrived"
    )
    carried_pairs = {
        (transform.header.frame_id, transform.child_frame_id)
        for message in probe.tf_frames
        for transform in message.transforms
    }
    static_pairs = {
        (transform.header.frame_id, transform.child_frame_id)
        for message in probe.tf_static_frames
        for transform in message.transforms
    }
    problems = _map_odom_ownership_problems(probe.robot_description, carried_pairs, static_pairs, publishers)
    assert not problems, "map -> odom ownership problems:\n  " + "\n  ".join(problems)

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
    started = time.monotonic()
    status = _send_nav_goal(probe, POINT_A_X, POINT_A_Y, 0.0)
    assert status == GoalStatus.STATUS_SUCCEEDED, f"navigation to point A failed with status {status}"
    _assert_ground_truth_at(ground_truth, POINT_A_X, POINT_A_Y, "point A")
    print(f"[navigation-sim] point A: goal completed in {time.monotonic() - started:.1f} s of wall time", flush=True)

    # there-and-back: back to the spawn origin
    started = time.monotonic()
    status = _send_nav_goal(probe, SPAWN_X, SPAWN_Y, 0.0)
    assert status == GoalStatus.STATUS_SUCCEEDED, f"navigation back to the origin failed with status {status}"
    _assert_ground_truth_at(ground_truth, SPAWN_X, SPAWN_Y, "origin")
    print(f"[navigation-sim] origin: goal completed in {time.monotonic() - started:.1f} s of wall time", flush=True)

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

    # the ownership re-check at the END of the observation window (review
    # round 2): a map -> odom broadcaster that joined the graph during the
    # goals was invisible to the pre-goal check and must fail here — the
    # fully-qualified endpoint identities and the accumulated dynamic pairs
    # both cover the whole drive
    late_publishers = _publisher_identities(probe.get_publishers_info_by_topic("/tf"))
    print(f"[navigation-sim] /tf publisher endpoints after goals: {sorted(late_publishers)}", flush=True)
    late_pairs = {
        (transform.header.frame_id, transform.child_frame_id)
        for message in probe.tf_frames
        for transform in message.transforms
    }
    late_static_pairs = {
        (transform.header.frame_id, transform.child_frame_id)
        for message in probe.tf_static_frames
        for transform in message.transforms
    }
    late_problems = _map_odom_ownership_problems(
        probe.robot_description, late_pairs, late_static_pairs, late_publishers
    )
    assert not late_problems, "map -> odom ownership problems after the goals:\n  " + "\n  ".join(late_problems)
