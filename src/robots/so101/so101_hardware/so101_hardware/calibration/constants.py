"""Shared calibration constants for SO-101 hardware."""

import pathlib

# Default calibration paths
CALIB_DIR = pathlib.Path.home() / ".calibrate"
LEADER_CALIB_FILE = CALIB_DIR / "so101_leader_calibrate.json"
FOLLOWER_CALIB_FILE = CALIB_DIR / "so101_follower_calibrate.json"

# Motor configurations
MOTOR_COUNT = 6
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Joint 5 is full-turn (wrist roll)
FULL_TURN_MOTOR_ID = 5

# Motor step ranges
MIN_STEP = 0
MAX_STEP = 4095

# Default serial port
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"

# Default publishing rates
DEFAULT_LEADER_PUBLISH_RATE = 50.0  # Hz
DEFAULT_CONTROL_RATE = 100.0  # Hz

# STS3215 Present_Current raw unit conversion: 1 LSB = 6.5 mA.
CURRENT_RAW_TO_AMPERE = 0.0065

# Default motor configurations
DEFAULT_MOTOR_CONFIGS: dict[str, dict] = {
    "1": {"model": "sts3215", "mode": "RANGE_M100_100"},
    "2": {"model": "sts3215", "mode": "RANGE_M100_100"},
    "3": {"model": "sts3215", "mode": "RANGE_M100_100"},
    "4": {"model": "sts3215", "mode": "RANGE_M100_100"},
    "5": {"model": "sts3215", "mode": "RANGE_0_100"},  # Full-turn
    "6": {"model": "sts3215", "mode": "RANGE_0_100"},
}
