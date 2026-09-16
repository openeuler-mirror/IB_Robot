#!/usr/bin/env python3
"""Interactive SO-101 follower arm calibration checker."""

from __future__ import annotations

import argparse
from pathlib import Path

from so101_hardware.calibration.checker import (
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEGACY_ROBOT_TYPE,
    ArmCalibrationChecker,
    resolve_checker_calibration_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check follower arm calibration with fixed poses",
    )
    parser.add_argument("--port", default=None, help="Follower arm serial port")
    parser.add_argument(
        "--robot.port",
        dest="port",
        default=None,
        help="Compatible alias for follower arm serial port",
    )
    parser.add_argument(
        "--calib-file",
        dest="calib_file",
        default=None,
        help="Path to follower calibration file",
    )
    parser.add_argument(
        "--calib_file",
        dest="calib_file",
        default=None,
        help="Compatible alias for calibration file",
    )
    parser.add_argument(
        "--robot.type",
        dest="robot_type",
        default="so101_follower",
        help="Compatible robot type argument",
    )
    parser.add_argument(
        "--robot.id",
        dest="robot_id",
        default="single_follower_arm",
        help="Compatible robot id argument",
    )
    parser.add_argument(
        "--robot.calibration_dir",
        dest="robot_calibration_dir",
        default=None,
        help="Compatible legacy calibration directory",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    if parsed.robot_type != "so101_follower":
        parser.error(f"Unsupported robot type: {parsed.robot_type}")

    calib_file = resolve_checker_calibration_file(
        calib_file=parsed.calib_file,
        robot_id=parsed.robot_id,
        calibration_dir=parsed.robot_calibration_dir,
        robot_type=parsed.robot_type or DEFAULT_LEGACY_ROBOT_TYPE,
    )

    checker = ArmCalibrationChecker(
        port=parsed.port or DEFAULT_FOLLOWER_PORT,
        calib_file=Path(calib_file).expanduser(),
    )

    try:
        checker.connect()
        checker.run()
    finally:
        checker.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
