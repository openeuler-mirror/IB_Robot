#!/usr/bin/env python3
"""Interactive calibration module for SO-101 arm motors."""

from __future__ import annotations

import json
import pathlib
import sys

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech.feetech import OperatingMode

JOINT_ALIASES = {
    "1": "shoulder_pan",
    "2": "shoulder_lift",
    "3": "elbow_flex",
    "4": "wrist_flex",
    "5": "wrist_roll",
    "6": "gripper",
}
JOINT_ALIASES.update({value: key for key, value in JOINT_ALIASES.items()})


def run_interactive_calibration(bus, joint_names: list[str], logger=None) -> dict[str, MotorCalibration] | None:
    """Run interactive calibration to capture homing offsets and ranges.

    Args:
        bus: FeetechMotorsBus instance
        joint_names: List of joint names to calibrate
        logger: Optional ROS logger for output

    Returns:
        dict mapping joint names to MotorCalibration objects, or None on failure
    """
    log = logger.info if logger else print
    log_error = logger.error if logger else print

    log("\n--- Starting arm calibration ---")
    log("!!! Note: This process requires user interaction !!!")

    try:
        bus.disable_torque()
        for motor_name in joint_names:
            bus.write("Operating_Mode", motor_name, OperatingMode.POSITION.value)
        log("All motors set to position mode.")

        # Capture homing offsets
        input(">>> (1/2) Move arm to mid-position of range, then press ENTER ...")
        sys.stdout.flush()
        homing_offsets = bus.set_half_turn_homings()
        log(f"Captured homing offsets: {homing_offsets}")

        # Capture ranges
        full_turn_motor = "5"
        unknown_range_motors = [n for n in joint_names if n != full_turn_motor]

        print(
            f"\n>>> (2/2) Move all joints (except '{full_turn_motor}') through full range, then press ENTER to stop recording..."
        )
        range_mins, range_maxes = bus.record_ranges_of_motion(unknown_range_motors)

        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        log(f"Recorded range mins: {range_mins}")
        log(f"Recorded range maxes: {range_maxes}")

        # Build calibration data
        calibration_data = {}
        for motor_name, m in bus.motors.items():
            calibration_data[motor_name] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )

        mins_text = ", ".join(f"{key}: {value.range_min}" for key, value in calibration_data.items())
        maxes_text = ", ".join(f"{key}: {value.range_max}" for key, value in calibration_data.items())
        log(f"Calibration complete: mins={{{mins_text}}}")
        log(f"Calibration complete: maxes={{{maxes_text}}}")

        return calibration_data

    except Exception as e:
        log_error(f"Calibration failed: {e}")
        import traceback

        log_error(traceback.format_exc())
        return None


def save_calibration(data: dict[str, MotorCalibration], path: pathlib.Path, logger=None):
    """Save calibration data to JSON file.

    Args:
        data: dict mapping joint names to MotorCalibration objects
        path: Path to save calibration file
        logger: Optional ROS logger for output
    """
    log = logger.info if logger else print
    log_error = logger.error if logger else print

    data_to_save = {
        name: {
            "id": cal.id,
            "drive_mode": cal.drive_mode,
            "homing_offset": cal.homing_offset,
            "range_min": cal.range_min,
            "range_max": cal.range_max,
        }
        for name, cal in data.items()
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        log(f"Calibration saved to: {path}")
    except Exception as e:
        log_error(f"Failed to save calibration: {e}")
        raise


def load_calibration(path: pathlib.Path, joint_names: list[str], logger=None) -> dict[str, MotorCalibration]:
    """Load calibration data from JSON file.

    Args:
        path: Path to calibration file
        joint_names: Expected joint names
        logger: Optional ROS logger for output

    Returns:
        dict mapping joint names to MotorCalibration objects

    Raises:
        FileNotFoundError: If calibration file doesn't exist
        ValueError: If calibration file is invalid or incomplete
    """
    log = logger.info if logger else print

    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path) as f:
        loaded_data = json.load(f)

    calibration_data = {}
    for joint_name in joint_names:
        candidate_names = [joint_name]
        alias = JOINT_ALIASES.get(joint_name)
        if alias:
            candidate_names.append(alias)

        for candidate_name in candidate_names:
            data_dict = loaded_data.get(candidate_name)
            if data_dict is None:
                continue
            calibration_data[joint_name] = MotorCalibration(**data_dict)
            break

    if not all(j in calibration_data for j in joint_names):
        raise ValueError("Calibration file is invalid or incomplete")

    log(f"Loaded calibration from: {path}")
    return calibration_data
