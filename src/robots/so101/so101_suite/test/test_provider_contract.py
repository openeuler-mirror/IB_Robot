"""Verify so101_suite satisfies the generic pipeline's provider contracts."""

from so101_suite import gripper_geometry, wrist_guard

from manipulation_execution.providers import (
    GRASP_GEOMETRY_FUNCTIONS,
    WRIST_GUARD_FUNCTIONS,
    load_provider,
)


def test_gripper_geometry_satisfies_grasp_geometry_provider_contract():
    module = load_provider(
        "target_geometry.grasp_geometry_provider",
        "so101_suite.gripper_geometry",
        GRASP_GEOMETRY_FUNCTIONS,
    )

    assert module is gripper_geometry


def test_wrist_guard_satisfies_wrist_guard_provider_contract():
    module = load_provider(
        "target_gripper.ik_orientation_guard.wrist_guard_provider",
        "so101_suite.wrist_guard",
        WRIST_GUARD_FUNCTIONS,
    )

    assert module is wrist_guard
