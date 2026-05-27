"""Tests for LeRobot joint conversion helpers."""

import pytest

from robot_config.utils import build_joint_conversion_table_from_calibration


def test_degrees_norm_mode_keeps_gripper_range_0_100_semantics():
    """Test degrees mode preserves RANGE_0_100 for configured gripper joints."""
    calibration = {
        "1": {"range_min": 1024, "range_max": 3072},
        "6": {"range_min": 1200, "range_max": 2200},
    }

    table = build_joint_conversion_table_from_calibration(
        calibration,
        joint_names=["1", "6"],
        gripper_joints=["6"],
        norm_mode="degrees",
    )

    arm_entry, gripper_entry = table

    assert arm_entry[2] == pytest.approx(2048 * 360.0 / 4095.0)
    assert arm_entry[3] == pytest.approx(-1024 * 360.0 / 4095.0)
    assert gripper_entry[2:] == (100.0, 0.0)


def test_degrees_norm_mode_maps_gripper_actions_to_calibration_ends():
    """Test gripper action 0/100 map to the calibrated closed/open endpoints."""
    calibration = {"6": {"range_min": 1200, "range_max": 2200}}
    (rad_min, rad_max, span, offset) = build_joint_conversion_table_from_calibration(
        calibration,
        joint_names=["6"],
        gripper_joints=["6"],
        norm_mode="degrees",
    )[0]

    action_zero_rad = (0.0 - offset) / span * (rad_max - rad_min) + rad_min
    action_full_rad = (100.0 - offset) / span * (rad_max - rad_min) + rad_min

    assert action_zero_rad == pytest.approx(rad_min)
    assert action_full_rad == pytest.approx(rad_max)
