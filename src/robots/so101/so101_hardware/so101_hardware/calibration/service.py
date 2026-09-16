#!/usr/bin/env python3
"""ROS2 Service wrapper for SO-101 calibration tool."""

import rclpy
from rclpy.node import Node
from scripts.calibrate_arm import ARM_CONFIGS, UnifiedCalibrator
from std_srvs.srv import Trigger


class CalibrationService(Node):
    def __init__(self):
        super().__init__("calibration_service")

        self.declare_parameter("arm_type", "follower")
        self.declare_parameter("port", "")

        self.arm_type = self.get_parameter("arm_type").value
        port_param = self.get_parameter("port").value

        if self.arm_type not in ARM_CONFIGS:
            self.get_logger().error(f"Invalid arm_type: {self.arm_type}")
            raise ValueError(f"Invalid arm_type: {self.arm_type}")

        self.port = port_param if port_param else ARM_CONFIGS[self.arm_type]["default_port"]

        self.srv = self.create_service(Trigger, f"/calibrate_{self.arm_type}", self.calibrate_callback)

        self.get_logger().info(f"Calibration service ready for {self.arm_type} arm on {self.port}")

    def calibrate_callback(self, request, response):
        self.get_logger().info(f"Starting calibration for {self.arm_type} arm via service...")

        try:
            calibrator = UnifiedCalibrator(self.arm_type, self.port)
            calibrator.run()
            response.success = True
            response.message = f"{self.arm_type.upper()} arm calibration completed successfully"
        except Exception as e:
            self.get_logger().error(f"Calibration failed: {str(e)}")
            response.success = False
            response.message = f"Calibration failed: {str(e)}"

        return response


def main():
    rclpy.init()
    service = CalibrationService()
    rclpy.spin(service)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
