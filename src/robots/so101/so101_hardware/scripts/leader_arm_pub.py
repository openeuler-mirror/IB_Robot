#!/usr/bin/env python3
"""Publish provisioned read-only leader input, not follower joint commands.

JointState.position contains radians except the explicitly named gripper, which
is an opening ratio. The frame marker identifies this input-only unit contract.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from so101_hardware.leader_input import LeaderInput


class LeaderArmPublisher(Node):
    def __init__(self):
        super().__init__("so101_leader_input")
        self.declare_parameter("port", "")
        self.declare_parameter("baudrate", 1000000)
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("calibration_version", 0)
        self.declare_parameter("joint_order", ["1", "2", "3", "4", "5", "6"])
        self.declare_parameter("gripper_joint", "6")
        self.declare_parameter("source_topic", "/inputs/so101_leader/state")
        self.declare_parameter("publish_rate", 50.0)
        config = {
            "port": str(self.get_parameter("port").value),
            "baudrate": int(self.get_parameter("baudrate").value),
            "calibration_file": str(self.get_parameter("calibration_file").value),
            "calibration_version": int(self.get_parameter("calibration_version").value),
            "joint_order": [str(name) for name in self.get_parameter("joint_order").value],
            "gripper_joint": str(self.get_parameter("gripper_joint").value),
            "source_topic": str(self.get_parameter("source_topic").value),
        }
        frequency = float(self.get_parameter("publish_rate").value)
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("publish_rate must be finite and positive")
        if not config["port"] or not config["source_topic"]:
            raise ValueError("port and source_topic are required")
        self.source = LeaderInput(config)
        self.source.connect()
        self.publisher = self.create_publisher(JointState, config["source_topic"], 1)
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

    def timer_callback(self):
        values = self.source.read()
        if values is None:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "leader_radians_gripper_ratio_v1"
        message.name = self.source.names
        message.position = [values[name] for name in self.source.names]
        self.publisher.publish(message)

    def destroy_node(self):
        self.source.disconnect()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LeaderArmPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
