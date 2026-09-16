"""Calibration data validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from so101_hardware.calibration.constants import MAX_STEP, MIN_STEP

REQUIRED_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def _entry_value(entry: Any, field: str):
    if isinstance(entry, Mapping):
        return entry.get(field)
    return getattr(entry, field, None)


def collect_validation_errors(calibration_data) -> list[str]:
    """Collect validation errors for calibration data."""
    if not isinstance(calibration_data, Mapping) or not calibration_data:
        return ["Calibration data must be a non-empty mapping."]

    errors: list[str] = []
    seen_ids: set[int] = set()

    for joint_name, entry in calibration_data.items():
        missing = [field for field in REQUIRED_FIELDS if _entry_value(entry, field) is None]
        if missing:
            errors.append(f"{joint_name}: missing required fields: {', '.join(missing)}")
            continue

        motor_id = int(_entry_value(entry, "id"))
        drive_mode = int(_entry_value(entry, "drive_mode"))
        range_min = int(_entry_value(entry, "range_min"))
        range_max = int(_entry_value(entry, "range_max"))

        if motor_id in seen_ids:
            errors.append(f"{joint_name}: duplicate motor id {motor_id}")
        seen_ids.add(motor_id)

        if drive_mode not in (0, 1):
            errors.append(f"{joint_name}: drive_mode must be 0 or 1")

        if range_min < MIN_STEP or range_min > MAX_STEP:
            errors.append(f"{joint_name}: range_min {range_min} is outside allowed range [{MIN_STEP}, {MAX_STEP}]")
        if range_max < MIN_STEP or range_max > MAX_STEP:
            errors.append(f"{joint_name}: range_max {range_max} is outside allowed range [{MIN_STEP}, {MAX_STEP}]")
        if range_min > range_max:
            errors.append(f"{joint_name}: range_min {range_min} must be <= range_max {range_max}")

    return errors


def validate_calibration_data(calibration_data) -> bool:
    """Validate calibration data integrity."""
    return not collect_validation_errors(calibration_data)
