#!/usr/bin/env python3
"""Unified calibration tool for SO-101 Leader and Follower arms."""

import argparse
import pathlib

import rclpy
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from rclpy.node import Node

from so101_hardware.calibration.constants import (
    DEFAULT_SERIAL_PORT,
    FOLLOWER_CALIB_FILE,
    LEADER_CALIB_FILE,
)
from so101_hardware.calibration.interactive import run_interactive_calibration, save_calibration

# Arm-specific configurations
ARM_CONFIGS = {
    "follower": {
        "calib_path": FOLLOWER_CALIB_FILE,
        "default_port": DEFAULT_SERIAL_PORT,
        "joints": {
            "1": {"id": 1, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "2": {"id": 2, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "3": {"id": 3, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "4": {"id": 4, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "5": {"id": 5, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "6": {"id": 6, "model": "sts3215", "mode": MotorNormMode.RANGE_0_100},
        },
    },
    "leader": {
        "calib_path": LEADER_CALIB_FILE,
        "default_port": DEFAULT_SERIAL_PORT,
        "joints": {
            "1": {"id": 1, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "2": {"id": 2, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "3": {"id": 3, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "4": {"id": 4, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "5": {"id": 5, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "6": {"id": 6, "model": "sts3215", "mode": MotorNormMode.RANGE_0_100},
        },
    },
}


class UnifiedCalibrator(Node):
    def __init__(self, arm_type: str, port: str, calib_path_override: str = None):
        super().__init__(f"so101_{arm_type}_calibrator")
        self.arm_type = arm_type
        self.port = port

        config = ARM_CONFIGS[arm_type]
        self.calib_path = (
            pathlib.Path(calib_path_override).expanduser() if calib_path_override else config["calib_path"]
        )
        self.joints_config = config["joints"]

    def run(self):
        self.get_logger().info(f"Starting calibration for {self.arm_type.upper()} arm on {self.port}")

        # Build motor objects
        motors = {name: Motor(cfg["id"], cfg["model"], cfg["mode"]) for name, cfg in self.joints_config.items()}
        joint_names = list(self.joints_config.keys())

        # Connect to bus
        bus = FeetechMotorsBus(self.port, motors)
        self.get_logger().info(f"Connecting to motor bus on {self.port}...")
        try:
            bus.connect()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to motor bus: {e}")
            return

        try:
            # Run calibration
            calibration_data = run_interactive_calibration(bus, joint_names, self.get_logger())

            if calibration_data is None:
                self.get_logger().error("Calibration failed.")
                return

            # Write to motor firmware
            self.get_logger().info("Writing calibration to motor firmware...")
            bus.write_calibration(calibration_data)

            # Save to file
            save_calibration(calibration_data, self.calib_path, self.get_logger())

            self.get_logger().info(f"✅ {self.arm_type.upper()} arm calibration completed successfully!")
        finally:
            bus.disconnect()
            self.get_logger().info("Disconnected from motor bus.")


def main():
    parser = argparse.ArgumentParser(description="Calibrate SO-101 Leader or Follower arm")
    parser.add_argument(
        "--arm", choices=["leader", "follower"], default="follower", help="Which arm to calibrate (default: follower)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port (default: /dev/ttyACM0 for follower, /dev/ttyACM1 for leader)",
    )
    parser.add_argument(
        "--calib-file", type=str, default=None, help="Custom path for the calibration JSON file (overrides default)"
    )

    args = parser.parse_args()

    # Use default port if not specified
    port = args.port if args.port else ARM_CONFIGS[args.arm]["default_port"]

    rclpy.init()
    calibrator = UnifiedCalibrator(args.arm, port, args.calib_file)
    try:
        calibrator.run()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
