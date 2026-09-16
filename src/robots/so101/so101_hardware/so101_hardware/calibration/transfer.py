"""Helpers for migrating legacy SO-101 calibration files."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from so101_hardware.calibration.validation import collect_validation_errors

JOINT_ORDER = ["1", "2", "3", "4", "5", "6"]
NAMED_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
JOINT_ALIASES = dict(zip(JOINT_ORDER, NAMED_JOINT_ORDER, strict=False))
JOINT_ALIASES.update({value: key for key, value in JOINT_ALIASES.items()})
SPECIAL_RANGE_JOINTS = {"2", "3", "shoulder_lift", "elbow_flex"}
SKIP_JOINTS = {"6", "gripper"}
DEFAULT_OUTPUT_PREFIX = "finetune_"


def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Mapping, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def compute_migrated_limits(
    homing_offset: int,
    range_min: int,
    range_max: int,
    drive_mode: int | bool,
    joint_name: str,
) -> tuple[int, int]:
    """Compute migrated min/max limits using the legacy compatibility formula."""
    if joint_name in SPECIAL_RANGE_JOINTS:
        bias = -(2 * homing_offset + range_min + range_max - 2048) / 2
    else:
        bias = -(2 * homing_offset + range_min + range_max) / 2

    migrated_min = int(range_min + bias)
    migrated_max = int(range_max + bias)

    if drive_mode:
        preserved_max = migrated_max
        migrated_max = -migrated_min
        migrated_min = -preserved_max

    if migrated_min < 0:
        migrated_max += migrated_min
        migrated_min = 0

    return migrated_min, migrated_max


def _candidate_joint_names(joint_name: str, index: int) -> list[str]:
    candidates = [joint_name]

    alias = JOINT_ALIASES.get(joint_name)
    if alias:
        candidates.append(alias)

    if index < len(JOINT_ORDER):
        candidates.extend([JOINT_ORDER[index], NAMED_JOINT_ORDER[index]])

    return list(dict.fromkeys(candidates))


def _legacy_joint_meta(
    legacy_data: Mapping,
    joint_names: Iterable[str],
) -> dict[str, dict[str, int]]:
    joint_names = list(joint_names)

    if "homing_offset" in legacy_data and "drive_mode" in legacy_data:
        homing_offsets = legacy_data["homing_offset"]
        drive_modes = legacy_data["drive_mode"]
        if len(homing_offsets) < len(joint_names) or len(drive_modes) < len(joint_names):
            raise ValueError("Legacy calibration arrays are shorter than target joint list.")

        return {
            joint_name: {
                "homing_offset": int(homing_offsets[index]),
                "drive_mode": int(drive_modes[index]),
            }
            for index, joint_name in enumerate(joint_names)
        }

    resolved: dict[str, dict[str, int]] = {}
    for index, joint_name in enumerate(joint_names):
        for candidate in _candidate_joint_names(joint_name, index):
            entry = legacy_data.get(candidate)
            if not isinstance(entry, Mapping):
                continue
            if "homing_offset" in entry and "drive_mode" in entry:
                resolved[joint_name] = {
                    "homing_offset": int(entry["homing_offset"]),
                    "drive_mode": int(entry["drive_mode"]),
                }
                break

        if joint_name not in resolved:
            raise ValueError(
                f"Legacy calibration data missing joint metadata for '{joint_name}'.",
            )

    return resolved


def migrate_calibration_data(template_data: Mapping, legacy_data: Mapping) -> dict:
    """Migrate a calibration template using legacy homing and drive-mode metadata."""
    migrated = copy.deepcopy(template_data)
    legacy_meta = _legacy_joint_meta(legacy_data, migrated.keys())

    for _index, (joint_name, joint_data) in enumerate(migrated.items()):
        if joint_name in SKIP_JOINTS or JOINT_ALIASES.get(joint_name) in SKIP_JOINTS:
            continue

        meta = legacy_meta[joint_name]
        new_min, new_max = compute_migrated_limits(
            homing_offset=meta["homing_offset"],
            range_min=int(joint_data["range_min"]),
            range_max=int(joint_data["range_max"]),
            drive_mode=meta["drive_mode"],
            joint_name=joint_name,
        )
        joint_data["range_min"] = new_min
        joint_data["range_max"] = new_max

    errors = collect_validation_errors(migrated)
    if errors:
        raise ValueError("Migrated calibration data is invalid:\n- " + "\n- ".join(errors))

    return migrated


def migrate_calibration_file(
    template_path: str | Path,
    legacy_path: str | Path,
    output_path: str | Path,
) -> dict:
    template_data = read_json(template_path)
    legacy_data = read_json(legacy_path)
    migrated = migrate_calibration_data(template_data, legacy_data)
    write_json(migrated, output_path)
    return migrated
