# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chassis geometry comes from the profile only, and published limits stay realizable.

Regressions for two defects the first LeKiwi runtime shipped with: geometry keys
silently fell back to LeKiwi constants, and both profiles advertised
``max_vx``/``max_vy`` of 0.5 m/s -- about twice what the wheels can reach at
``max_wheel_radps``, so a consumer planning to the published limit got a base
that silently scaled its commands down.
"""

import copy
from pathlib import Path

import pytest

from lekiwi_robot.base_profile import (
    REQUIRED_BASE_KEYS,
    arm_motion_model,
    base_node_parameters,
    geometry_from_base,
    lidar_odometry_owns_tf,
    realizable_limits,
)
from robot_runtime.launch_support import RuntimeProfileError
from robot_runtime.profile import load_profile

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
SHIPPED = sorted(PROFILES.glob("*.yaml"))


def lidar_profile() -> dict:
    return load_profile(PROFILES / "lekiwi_lidar.yaml")


@pytest.mark.parametrize("path", SHIPPED, ids=[p.stem for p in SHIPPED])
def test_shipped_profiles_resolve_and_publish_realizable_limits(path):
    profile = load_profile(path)
    params = base_node_parameters(profile, str(path))
    assert params["wheel_joints"] == ["7", "8", "9"]
    assert params["navigation_enabled_on_startup"] is False
    bounds = realizable_limits(geometry_from_base(profile["base"]))
    declared = profile["capabilities"]["base.cmd_vel"]
    for axis, bound in bounds.items():
        assert declared[axis] <= bound, f"{path.name}: {axis}={declared[axis]} exceeds {bound:.4f}"


@pytest.mark.parametrize("path", SHIPPED, ids=[p.stem for p in SHIPPED])
def test_shipped_profiles_leave_motion_hardware_feedback_topic_empty(path):
    # LeKiwiSystemHardware publishes no joint-current heartbeat; only
    # So101SystemHardware publishes /so101_follower/joint_currents. A non-empty
    # motion_hardware_feedback_topic makes the motion server's post-motion
    # barrier wait for a heartbeat that never arrives, so every physical
    # MoveToPose/MoveToConfiguration would time out and report EXECUTION_FAILED.
    # LeKiwi read failures already surface as controller errors, so a shipped
    # LeKiwi profile must not name a feedback topic its closure does not publish.
    profile = load_profile(path)
    assert not (profile.get("motion") or {}).get("motion_hardware_feedback_topic")


def test_realizable_limits_follow_the_mount_geometry():
    # 4.602 rad/s x 0.05 m = 0.2301 m/s of wheel rim speed. Forward motion
    # loads the two front wheels at cos(30 deg), sideways motion loads the back
    # wheel fully, rotation loads every wheel by base_radius.
    bounds = realizable_limits(geometry_from_base(lidar_profile()["base"]))
    assert bounds["max_vx"] == pytest.approx(0.2301 / 0.8660254, rel=1e-6)
    assert bounds["max_vy"] == pytest.approx(0.2301, rel=1e-6)
    assert bounds["max_wz"] == pytest.approx(0.2301 / 0.125, rel=1e-6)


@pytest.mark.parametrize("key", REQUIRED_BASE_KEYS)
def test_missing_geometry_key_fails_naming_it(key):
    profile = lidar_profile()
    del profile["base"][key]
    with pytest.raises(RuntimeProfileError, match=rf"'base\.{key}'"):
        base_node_parameters(profile, "lekiwi_lidar.yaml")


def test_missing_staleness_fails_naming_it():
    profile = lidar_profile()
    del profile["capabilities"]["base.cmd_vel"]["staleness_s"]
    with pytest.raises(RuntimeProfileError, match=r"'capabilities\.base\.cmd_vel\.staleness_s'"):
        base_node_parameters(profile, "lekiwi_lidar.yaml")


@pytest.mark.parametrize(("axis", "value"), [("max_vx", 0.5), ("max_vy", 0.5), ("max_wz", 2.0)])
def test_limit_beyond_the_envelope_is_rejected_naming_the_axis(axis, value):
    profile = copy.deepcopy(lidar_profile())
    profile["capabilities"]["base.cmd_vel"][axis] = value
    with pytest.raises(RuntimeProfileError, match=rf"base\.cmd_vel\.{axis}=.*exceeds"):
        base_node_parameters(profile, "lekiwi_lidar.yaml")


@pytest.mark.parametrize(
    ("fast_lio", "simulated", "owns"),
    [
        ({"enabled": True, "publish_tf": True}, False, True),
        ({"enabled": True}, False, True),  # FAST-LIO publishes TF by default
        ({"enabled": True, "publish_tf": False}, False, False),
        ({"enabled": False, "publish_tf": True}, False, False),
        ({"enabled": True, "publish_tf": True}, True, False),  # not launched in simulation
        ({}, False, False),
    ],
)
def test_one_odom_to_base_link_publisher(fast_lio, simulated, owns):
    # Wheel odometry publishes the TF exactly when lidar odometry does not.
    assert lidar_odometry_owns_tf(fast_lio, simulated) is owns


def test_arm_motion_model_is_the_so101_arm_without_the_chassis():
    # so101_motion's SRDF names the robot "so101" and MoveIt only receives the
    # arm broadcaster's joints 1-6. Handing move_group the whole LeKiwi left
    # wheels 7-9 without state, and FK, IK and planning all failed.
    import xml.etree.ElementTree as ET

    path = PROFILES / "lekiwi_nav_grasp.yaml"
    root = ET.fromstring(arm_motion_model(load_profile(path), str(path), simulated=True))
    movable = {joint.get("name") for joint in root.findall("joint") if joint.get("type") != "fixed"}
    assert root.get("name") == "so101"
    assert movable == {"1", "2", "3", "4", "5", "6"}


def test_arm_motion_model_is_required_for_a_motion_profile():
    profile = load_profile(PROFILES / "lekiwi_nav_grasp.yaml")
    del profile["motion"]["model_xacro"]
    with pytest.raises(RuntimeProfileError, match=r"'motion\.model_xacro'"):
        arm_motion_model(profile, "lekiwi_nav_grasp.yaml", simulated=True)
