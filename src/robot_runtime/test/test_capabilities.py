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
