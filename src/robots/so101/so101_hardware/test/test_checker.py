"""Tests for calibration checker helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_hardware.calibration.checker import (  # noqa: E402
    apply_joint_updates,
    build_default_check_steps,
    default_lerobot_calibration_dir,
    resolve_checker_calibration_file,
)
from so101_hardware.calibration.interactive import load_calibration  # noqa: E402


def test_build_default_check_steps_contains_expected_sequence():
    steps = build_default_check_steps()

    assert len(steps) == 9
    assert steps[0].prompt.startswith("<<按回车机械臂到30度")
    assert steps[0].updates == {
        "1": 90,
        "2": -60,
        "3": 60,
        "4": 60,
    }
    assert steps[3].prompt.startswith("<<按回车机械臂到0位")
    assert steps[4].updates == {"5": 90}
    assert steps[5].updates == {"5": -90}
    assert steps[-1].updates == {
        "1": 0,
        "2": -95,
        "3": 85,
        "4": 65,
        "5": 0,
        "6": 0,
    }


def test_apply_joint_updates_returns_new_action_dict():
    action = {"1": 0, "2": 10, "3": 20}

    updated = apply_joint_updates(action, {"2": 30, "4": 40})

    assert updated == {"1": 0, "2": 30, "3": 20, "4": 40}
    assert action == {"1": 0, "2": 10, "3": 20}


def test_load_calibration_accepts_legacy_named_joint_keys(tmp_path: Path):
    calib_file = tmp_path / "single_follower_arm.json"
    calib_file.write_text(
        json.dumps(
            {
                "shoulder_pan": {
                    "id": 1,
                    "drive_mode": 0,
                    "homing_offset": 10,
                    "range_min": 100,
                    "range_max": 200,
                },
                "shoulder_lift": {
                    "id": 2,
                    "drive_mode": 0,
                    "homing_offset": 20,
                    "range_min": 200,
                    "range_max": 300,
                },
                "elbow_flex": {
                    "id": 3,
                    "drive_mode": 0,
                    "homing_offset": 30,
                    "range_min": 300,
                    "range_max": 400,
                },
                "wrist_flex": {
                    "id": 4,
                    "drive_mode": 0,
                    "homing_offset": 40,
                    "range_min": 400,
                    "range_max": 500,
                },
                "wrist_roll": {
                    "id": 5,
                    "drive_mode": 0,
                    "homing_offset": 50,
                    "range_min": 500,
                    "range_max": 600,
                },
                "gripper": {
                    "id": 6,
                    "drive_mode": 0,
                    "homing_offset": 60,
                    "range_min": 0,
                    "range_max": 100,
                },
            },
        ),
        encoding="utf-8",
    )

    calibration = load_calibration(calib_file, ["1", "2", "3", "4", "5", "6"])

    assert calibration["1"].id == 1
    assert calibration["5"].range_max == 600
    assert calibration["6"].range_min == 0


def test_resolve_checker_calibration_file_prefers_legacy_robot_id_path(
    monkeypatch,
    tmp_path: Path,
):
    legacy_root = tmp_path / "legacy"
    legacy_file = legacy_root / "robots" / "so101_follower" / "single_follower_arm.json"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(legacy_root))

    resolved = resolve_checker_calibration_file(robot_id="single_follower_arm")

    assert resolved == legacy_file


def test_resolve_checker_calibration_file_honors_explicit_legacy_dir(tmp_path: Path):
    calibration_dir = tmp_path / "custom_calibration"

    resolved = resolve_checker_calibration_file(
        robot_id="single_follower_arm",
        calibration_dir=calibration_dir,
    )

    assert resolved == calibration_dir / "single_follower_arm.json"


def test_default_lerobot_calibration_dir_uses_environment(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "hf_calibration"))

    resolved = default_lerobot_calibration_dir()

    assert resolved == tmp_path / "hf_calibration" / "robots" / "so101_follower"
