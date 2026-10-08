"""The explicit arm-surface opt-out for runtimes that own their own motion."""

from pathlib import Path

import pytest

from robot_config.loader import (
    load_robot_config,
    robot_requires_arm_surface,
    validate_arm_surface_config,
    validate_config,
)

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"


def _runtime_robot(**embodied):
    return {
        "runtime": {"provider": "vendor_runtime"},
        "capabilities": {"requires": ["joint.state", "motion.named", "runtime.stop"]},
        "embodied": {"enabled": True, **embodied},
    }


@pytest.mark.parametrize(
    "robot_config",
    [
        {},
        {"embodied": {"enabled": True}},
        {"embodied": "not-a-mapping"},
        # A runtime robot that declares no arm capability still keeps the arm
        # surface: opting out is never inferred from an omission.
        {"runtime": {"provider": "lekiwi_robot"}, "embodied": {"enabled": True}},
        _runtime_robot(),
    ],
)
def test_arm_surface_is_required_unless_explicitly_disabled(robot_config):
    assert robot_requires_arm_surface(robot_config) is True


def test_explicit_opt_out_disables_the_arm_surface():
    assert robot_requires_arm_surface(_runtime_robot(arm_surface=False)) is False


def test_opt_out_is_valid_for_a_motion_owning_runtime():
    assert validate_arm_surface_config(_runtime_robot(arm_surface=False)) == []
    assert validate_arm_surface_config(_runtime_robot(arm_surface=True)) == []
    assert validate_arm_surface_config(_runtime_robot()) == []


def test_opt_out_must_be_boolean():
    assert validate_arm_surface_config(_runtime_robot(arm_surface="false")) == [
        "embodied.arm_surface must be a boolean"
    ]


def test_opt_out_requires_a_runtime_provider():
    robot_config = {"embodied": {"enabled": True, "arm_surface": False}}
    errors = validate_arm_surface_config(robot_config)
    assert any("requires a runtime.provider" in error for error in errors)


@pytest.mark.parametrize("capability", ["motion.move_to_joint", "joint.trajectory", "motion.ik"])
def test_opt_out_contradicting_required_arm_capabilities_is_rejected(capability):
    robot_config = _runtime_robot(arm_surface=False)
    robot_config["capabilities"]["requires"].append(capability)
    errors = validate_arm_surface_config(robot_config)
    assert any(capability in error for error in errors)


def test_arm_named_poses_are_still_required_by_default():
    config = load_robot_config(ROBOTS / "so101_single_arm_legacy.yaml")
    config.embodied.enabled = True
    config.embodied.named_poses = {"home": {}}

    errors = validate_config(config)

    assert any("missing required pose(s): observe_table, zero" in error for error in errors)


def test_arm_named_poses_are_not_required_without_an_arm_surface():
    config = load_robot_config(ROBOTS / "so101_single_arm_legacy.yaml")
    config.embodied.enabled = True
    config.embodied.arm_surface = False
    config.embodied.named_poses = {}
    config.embodied.default_place_name = "tray_right"

    errors = validate_config(config)

    assert not any("named_poses" in error for error in errors)
    assert not any("default_place_name" in error for error in errors)
