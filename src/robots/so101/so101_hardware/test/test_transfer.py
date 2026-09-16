"""Tests for calibration transfer helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_hardware.calibration.transfer import (  # noqa: E402
    compute_migrated_limits,
    migrate_calibration_data,
)
from so101_hardware.calibration.validation import (  # noqa: E402
    collect_validation_errors,
    validate_calibration_data,
)


def test_compute_migrated_limits_applies_special_joint_offset():
    range_min, range_max = compute_migrated_limits(
        homing_offset=10,
        range_min=1000,
        range_max=3000,
        drive_mode=0,
        joint_name="2",
    )

    assert range_min == 14
    assert range_max == 2014


def test_compute_migrated_limits_flips_drive_mode_and_clamps_to_zero():
    range_min, range_max = compute_migrated_limits(
        homing_offset=300,
        range_min=200,
        range_max=600,
        drive_mode=1,
        joint_name="1",
    )

    assert range_min == 100
    assert range_max == 500


def test_migrate_calibration_data_updates_all_except_gripper():
    template = {
        "1": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 200, "range_max": 600},
        "2": {"id": 2, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000},
        "3": {"id": 3, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000},
        "4": {"id": 4, "drive_mode": 0, "homing_offset": 0, "range_min": 200, "range_max": 600},
        "5": {"id": 5, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
        "6": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 100},
    }
    legacy = {
        "homing_offset": [-100, 10, 20, -100, 0, 0],
        "drive_mode": [0, 0, 0, 0, 0, 0],
    }

    migrated = migrate_calibration_data(template, legacy)

    assert migrated["1"]["range_min"] == 0
    assert migrated["1"]["range_max"] == 200
    assert migrated["2"]["range_min"] == 14
    assert migrated["2"]["range_max"] == 2014
    assert migrated["3"]["range_min"] == 4
    assert migrated["3"]["range_max"] == 2004
    assert migrated["6"]["range_min"] == 0
    assert migrated["6"]["range_max"] == 100


def test_validate_calibration_data_rejects_invalid_ranges():
    invalid = {
        "1": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 500, "range_max": 100},
    }

    errors = collect_validation_errors(invalid)

    assert errors
    assert not validate_calibration_data(invalid)
    assert "range_min" in errors[0]
