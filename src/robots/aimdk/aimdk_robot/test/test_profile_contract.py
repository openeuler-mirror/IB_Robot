"""Profile-level contract checks for the X2 runtime profile.

These run without ROS, without the vendor SDK and without a robot: they assert
that the shipped profile is a valid runtime profile, that it projects the
humanoid the way `design.md` D2/D3/D5/D7 decided, and that the description it
produces is accepted by the contract layer as-is.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aimdk_robot import projection

from robot_runtime.capabilities import all_capabilities
from robot_runtime.interface_description import build_description, validate_description
from robot_runtime.profile import load_profile

PROFILE_PATH = Path(__file__).resolve().parents[1] / "profiles" / "x2_ultra.yaml"

# The arm command order the vendor's UpperBodyCommandArray.arm_pos declares
# (docs Interface/control_mod/upper_body_control.html): left 7 then right 7.
VENDOR_ARM_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]


@pytest.fixture(scope="module")
def profile() -> dict:
    return load_profile(PROFILE_PATH)


def _channel(profile: dict, name: str) -> dict:
    for channel in profile["command_channels"]:
        if channel["channel"] == name:
            return channel
    raise AssertionError(f"profile declares no command channel {name!r}")


def test_profile_loads_and_capabilities_are_registered(profile):
    assert set(profile["capabilities"]) <= all_capabilities()


def test_arm_channel_uses_the_vendor_command_order(profile):
    assert _channel(profile, "arm_stream")["joints"] == VENDOR_ARM_ORDER
    assert profile["arm_joints"] == VENDOR_ARM_ORDER


def test_public_joints_exclude_vendor_controlled_body(profile):
    """Head, waist and legs are not public joint-group members (design D2)."""
    public = set(profile["joints"])
    assert public == set(profile["arm_joints"]) | set(profile["gripper_joints"])
    body = {"head_yaw_joint", "waist_yaw_joint", "left_knee_joint", "right_hip_pitch_joint"}
    assert not public & body


def test_head_is_a_separate_mode_not_a_concurrent_channel(profile):
    """The vendor ignores head targets outside HEAD_ONLY, so the modes are disjoint."""
    head = _channel(profile, "head_stream")
    arm = _channel(profile, "arm_stream")
    assert head["modes"] == ["head"]
    assert not set(head["modes"]) & set(arm["modes"])
    assert set(head["joints"]).isdisjoint(profile["joints"])


def test_channel_modes_exist_and_allow_streaming(profile):
    modes = profile["modes"]
    for channel in profile["command_channels"]:
        for mode in channel.get("modes", []):
            assert mode in modes, f"channel {channel['channel']} names unknown mode {mode}"
            assert modes[mode].get("allows_stream"), f"mode {mode} must allow streaming"


def test_no_capability_the_vendor_tier_cannot_serve(profile):
    """No trajectory execution, no FK/IK/move-to, no wheel odometry (design D3/D6)."""
    declared = set(profile["capabilities"])
    assert not declared & {
        "joint.trajectory",
        "motion.fk",
        "motion.ik",
        "motion.move_to_joint",
        "motion.move_to_pose",
        "base.odom",
        "nav.goal",
        "body.whole_stream",
        "hand.multi_joint",
        "hand.gesture",
    }
    assert profile["trajectory_actions"] == []


def test_no_ros2_control_stack_is_configured(profile):
    assert "ros2_control" not in profile
    assert profile["hardware_components"] == []
    assert all(not mode.get("controllers") for name, mode in profile["modes"].items() if name != "initial")


def test_stop_policy_is_posture_conditional(profile):
    assert profile["stop_default_policy"] == "HOLD"
    assert profile["vendor"]["stop"]["torque_off_policy"] in ("soft_estop", "zero_torque_when_seated")


def test_vendor_section_covers_every_mode(profile):
    modes = {name for name in profile["modes"] if name != "initial"}
    assert set(profile["vendor"]["mc_actions"]) == modes
    source = profile["vendor"]["input_source"]
    # Documented SDK band for secondary development (MC_control.html).
    assert 20 <= source["priority"] <= 39
    assert source["name"]


def test_description_builds_and_publishes_no_model(profile):
    description = build_description(profile, simulated=True)
    validate_description(description)
    assert "model" not in description
    assert not [name for name in description["interfaces"] if name.startswith("motion.arm")]


def test_description_declares_vendor_sensor_endpoints(profile):
    interfaces = build_description(profile, simulated=True)["interfaces"]
    cameras = {name: spec for name, spec in interfaces.items() if spec["capability"] == "perception.camera"}
    assert cameras
    for spec in cameras.values():
        assert spec["endpoint"].startswith("/aima/hal/sensor/"), spec["endpoint"]
    # Geometry is deliberately undeclared: the vendor documents that it varies
    # across revisions and must be read from CameraInfo. The contract requires
    # image interfaces to state this explicitly rather than omit it.
    images = [spec for spec in cameras.values() if spec["message_type"] == "sensor_msgs/msg/Image"]
    assert images
    for spec in images:
        assert spec["configured_profile"] is None
        assert spec["supported_profiles"] is None
        assert spec["camera_info_topic"] in {entry["endpoint"] for entry in interfaces.values()}


def test_commandable_channels_are_bridge_owned_topics(profile):
    interfaces = build_description(profile, simulated=True)["interfaces"]
    assert interfaces["joint.arm_stream"]["endpoint"] == "/aimdk/arm/commands"
    assert interfaces["base.cmd_vel"]["endpoint"] == "/cmd_vel"
    assert interfaces["base.cmd_vel"]["limits"] == {"max_vx": 1.0, "max_vy": 1.0, "max_wz": 1.0}


def test_only_firmware_verified_sensor_endpoints_are_declared(profile):
    """The declared set is a promise: endpoints the firmware does not serve stay out.

    Each absence here was established against a real X2 (firmware >= v1.1.0,
    2026-09-22) or the vendor's own changelog; see the profile comments. A
    future edit that re-adds one from the documentation alone would declare an
    interface consumers cannot use.
    """
    capabilities = set(profile["capabilities"])
    endpoints = {spec.get("endpoint") for spec in (profile.get("interfaces") or {}).values()}

    # Documented point cloud endpoint absent on the verified firmware.
    assert "perception.lidar" not in capabilities
    assert "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud" not in endpoints
    # ... but the lidar's own IMU is a plain Imu topic and stays declared.
    assert "/aima/hal/sensor/lidar_chest_front/imu" in endpoints
    assert "/aima/hal/sensor/lidar_chest_front/imu" in profile["capabilities"]["perception.imu"]["topics"]

    # GNSS is an installed option, not part of the platform.
    assert "perception.gnss" not in capabilities
    assert "/aima/hal/sensor/gnss" not in endpoints

    # Relocalization needs the optional vendor SLAM component.
    assert "localization.map" not in capabilities
    assert "/localization/relocalize" not in endpoints
    # Lidar odometry is served, so the pose interface stays.
    assert "localization.pose" in capabilities

    # Withdrawn from the open interface set (v0.8.1) / undocumented type.
    assert not [e for e in endpoints if e and "rgb_head_front_center" in e]
    assert not [e for e in endpoints if e and e.endswith("/h265")]


def test_vendor_endpoint_qos_matches_the_documented_profiles(profile):
    """QoS is part of the endpoint contract; a mismatch drops messages silently.

    The platform subscribes RELIABLE on audio playback, so publishing
    BEST_EFFORT there delivers nothing (observed on the robot). Its state topics
    publish TRANSIENT_LOCAL, so subscribing TRANSIENT_LOCAL picks up the current
    value on subscribe instead of waiting for the next sample.
    """
    qos = profile["vendor"]["qos"]
    assert qos["audio_playback"]["reliability"] == "reliable"
    assert qos["audio_capture"]["reliability"] == "reliable"
    for key in ("mc_state", "system_state", "joint_state", "hand_state", "power_state"):
        assert qos[key]["durability"] == "transient_local", key
    # The vendor's own command topics are BEST_EFFORT + VOLATILE.
    for key in ("upper_body_command", "locomotion_command"):
        assert qos[key] == {"reliability": "best_effort", "durability": "volatile"}, key
    for key, entry in qos.items():
        assert entry["reliability"] in projection.QOS_RELIABILITIES, key
        assert entry.get("durability", "volatile") in projection.QOS_DURABILITIES, key


def test_audio_playback_declares_the_focus_it_has_to_hold(profile):
    """Raw playback is focus-gated: without focus it is accepted and not heard."""
    audio = profile["vendor"]["audio"]
    priority, weight, release_idle_s, buffer_chunks = projection.audio_focus_request(audio)
    assert priority in projection.AUDIO_FOCUS_PRIORITIES
    assert weight in projection.AUDIO_FOCUS_WEIGHTS
    assert release_idle_s > 0.0
    assert buffer_chunks >= 1
    services = profile["vendor"]["services"]
    assert services["request_audio_focus"].endswith("RequestAudioFocus")
    assert services["abandon_audio_focus"].endswith("AbandonAudioFocus")
    # Preemption is announced on a latched topic, so the current holder is known
    # on subscribe rather than after the next change.
    assert profile["vendor"]["topics"]["audio_focus_response"] == "/aima/hal/audio/focus_response"
    assert profile["vendor"]["qos"]["audio_focus_response"] == {
        "reliability": "reliable",
        "durability": "transient_local",
    }


def test_every_vendor_action_named_by_the_profile_is_one_the_firmware_registers(profile):
    """The McAction IDL enum is wider than the firmware's registered set.

    `SetMcAction` matches on `action_desc` against the platform's configuration
    and answers code 3 ("not registered") for a name that exists only in the
    enum. The three names this profile used to carry — JOINT_FREEZE,
    SOFT_EMERGENCY_STOP, ZERO_TORQUE_DEFAULT — were all of that kind, so every
    stop was refused by the platform.
    """
    vendor = profile["vendor"]
    for mode, action in vendor["mc_actions"].items():
        assert action in projection.VENDOR_MOTION_MODES, f"mode {mode} -> unregistered action {action}"

    stop = vendor["stop"]
    for key in ("hold_action", "soft_estop_action", "zero_torque_action"):
        action = stop.get(key, "")
        if action:
            assert action in projection.VENDOR_MOTION_MODES, f"stop.{key} -> unregistered action {action}"

    # Both stop policies must resolve without raising, for every body pose the
    # platform can report.
    for pose in projection.BODY_POSE_NAMES:
        plan = projection.resolve_stop_plan("HOLD", body_pose=pose, stop_config=stop)
        assert plan.latch and plan.release_arbitration


def test_torque_off_maps_onto_a_mode_the_platform_always_permits(profile):
    """Damping and zero torque are the two modes permitted under safety protection."""
    stop = profile["vendor"]["stop"]
    standing = projection.resolve_stop_plan("TORQUE_OFF", body_pose=projection.BODY_POSE_STAND, stop_config=stop)
    assert standing.vendor_action in ("DAMPING_DEFAULT", "PASSIVE_DEFAULT")
    assert standing.downgraded_from == "TORQUE_OFF", "TORQUE_OFF on a balancing biped must be reported as downgraded"


def test_declared_preset_motions_are_pairs_the_platform_maps_to_an_animation(profile):
    """`area` is half of a (motion, area) key, not a body region.

    Since v0.8.0 the vendor maps the pair to one specific animation, so neither
    enumeration is a guide to validity: McPresetMotion lists values that appear
    in no documented pair, and the documented table lists pairs (3017/11) that
    appear in no enumeration. A profile inventing a combination would be
    accepted and animate nothing.
    """
    from aimdk_robot.named_motion import CONTROL_AREAS

    presets = profile["vendor"]["preset_motions"]
    assert presets, "the profile declares motion.named, so it must map the names"
    for name, entry in presets.items():
        motion = int(entry["motion"])
        target = str(entry.get("target", ""))
        area = int(entry.get("area", CONTROL_AREAS[target]))
        assert (motion, area) in projection.PRESET_MOTION_PAIRS, f"{name}: ({motion}, {area}) is not a documented pair"

    # Every name the capability advertises is actually mapped.
    advertised = set(profile["capabilities"]["motion.named"]["names"])
    assert advertised <= set(presets) | set(profile["vendor"]["postures"])
