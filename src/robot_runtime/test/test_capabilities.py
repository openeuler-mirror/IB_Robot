"""Unit tests for the capability vocabulary and reconciliation."""

from robot_runtime.capabilities import (
    all_capabilities,
    is_valid,
    requires_subset_of_discovered,
    reserved_names,
    validate_capability_set,
)


def test_all_capabilities_non_empty():
    caps = all_capabilities()
    assert len(caps) >= 10
    assert "joint.state" in caps
    assert "base.cmd_vel" in caps


def test_reserved_names_are_valid():
    for name in reserved_names():
        assert is_valid(name), f"reserved name {name!r} not in vocabulary"


def test_validate_capability_set_rejects_unknown():
    invalid = validate_capability_set(["joint.state", "not.a.capability"])
    assert invalid == ["not.a.capability"]


def test_reconciliation_passes_when_subset():
    required = ["joint.state", "joint.trajectory"]
    discovered = ["joint.state", "joint.trajectory", "gripper.1d"]
    assert requires_subset_of_discovered(required, discovered) == []


def test_reconciliation_fails_with_named_missing():
    required = ["joint.state", "base.cmd_vel", "nav.goal"]
    discovered = ["joint.state", "gripper.1d"]
    missing = requires_subset_of_discovered(required, discovered)
    assert missing == ["base.cmd_vel", "nav.goal"]


def test_reconciliation_empty_required_always_passes():
    assert requires_subset_of_discovered([], []) == []
    assert requires_subset_of_discovered([], ["anything"]) == []


def test_motion_and_stop_capabilities_registered():
    caps = all_capabilities()
    for name in (
        "motion.fk",
        "motion.ik",
        "motion.move_to_joint",
        "motion.move_to_pose",
        "runtime.stop",
        "base.navigation_gate",
    ):
        assert name in caps, f"{name} missing from the vocabulary"


def test_dropped_priority_stream_is_reserved_not_removed():
    assert "joint.priority_stream" in all_capabilities()
    assert "joint.priority_stream" in reserved_names()


def test_parameter_schema_for_stop_bounds():
    from robot_runtime.capabilities import parameter_schema

    schema = parameter_schema("runtime.stop")
    assert set(schema) == {"cancel_bound_s", "idle_bound_s", "torque_off_bound_s"}
    assert parameter_schema("motion.fk") == {}


def test_registry_v3_vendor_surfaces_registered():
    """v3 adds the surfaces a vendor-owned runtime republishes (aimdk-runtime-migration)."""
    caps = all_capabilities()
    for name in (
        "perception.imu",
        "perception.touch",
        "perception.gnss",
        "localization.pose",
        "localization.map",
        "power.state",
        "diagnostics.codes",
        "interaction.tts",
        "interaction.audio_capture",
        "interaction.audio_playback",
        "interaction.expression",
        "interaction.led",
        "motion.posture",
    ):
        assert name in caps, f"{name} missing from the vocabulary"


def test_v2_vocabulary_is_preserved():
    """Extension-only: every v2 name survives the v3 extension."""
    caps = all_capabilities()
    for name in (
        "joint.state",
        "joint.position_stream",
        "joint.trajectory",
        "gripper.1d",
        "hand.multi_joint",
        "hand.gesture",
        "base.cmd_vel",
        "base.odom",
        "base.navigation_gate",
        "nav.goal",
        "motion.fk",
        "motion.ik",
        "motion.move_to_joint",
        "motion.move_to_pose",
        "motion.named",
        "perception.camera",
        "perception.lidar",
        "runtime.stop",
        "runtime.status",
        "body.whole_stream",
        "joint.priority_stream",
    ):
        assert name in caps, f"v2 name {name} was removed"


def test_localization_pose_is_not_base_odom():
    """A map-frame pose is a distinct capability: it jumps on relocalization."""
    from robot_runtime.capabilities import parameter_schema

    assert set(parameter_schema("localization.pose")) == {"topics", "reference_frame"}
    assert set(parameter_schema("base.odom")) == set()
