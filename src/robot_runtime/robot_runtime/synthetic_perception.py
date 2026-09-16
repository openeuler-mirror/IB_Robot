"""Simulated sensor streams with the configured wire type, dimensions and rate."""

from __future__ import annotations

import json
import math
import struct

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointField

from robot_runtime.interface_description import validate_description


class SyntheticStreams:
    """Shared by standalone synthetic_perception and the in-process mock runtime."""

    def __init__(self, node, description):
        validate_description(description)
        self._timers = []
        interfaces = description["interfaces"]
        for name, spec in interfaces.items():
            if spec["kind"] != "topic" or spec["direction"] != "publish":
                continue
            if not (spec["capability"].startswith("perception.") or name.startswith("localization.")):
                continue
            message_type = spec["message_type"]
            message = get_message(message_type)()
            frame = spec.get("frame_id") or name.replace(".", "_")
            rate = spec.get("rate_hz") or 10.0
            if message_type == "sensor_msgs/msg/Image":
                profile = spec["configured_profile"]
                if profile is None:
                    raise ValueError(f"interface {name}: a configured image profile is required for simulation")
                rate = profile["fps"] or 30.0
                encoding = profile["encoding"] or "rgb8"
                bytes_per_pixel = {"rgb8": 3, "bgr8": 3, "mono8": 1, "16UC1": 2, "32FC1": 4}.get(encoding)
                if bytes_per_pixel is None:
                    raise ValueError(f"cannot simulate image encoding {encoding!r} for {name}")
                message.width, message.height, message.encoding = profile["width"], profile["height"], encoding
                message.step = message.width * bytes_per_pixel
                message.data = bytes(message.step * message.height)
            elif message_type == "sensor_msgs/msg/CameraInfo":
                image = next(
                    (entry for entry in interfaces.values() if entry.get("camera_info_topic") == spec["endpoint"]), None
                )
                profile = image["configured_profile"] if image else {"width": 1, "height": 1}
                message.width, message.height = profile["width"], profile["height"]
                message.distortion_model = "plumb_bob"
                message.k = [1.0, 0.0, message.width / 2.0, 0.0, 1.0, message.height / 2.0, 0.0, 0.0, 1.0]
            elif message_type == "sensor_msgs/msg/PointCloud2":
                message.width, message.height = 1, 1
                message.fields = [
                    PointField(name=axis, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                    for i, axis in enumerate(("x", "y", "z"))
                ]
                message.point_step = message.row_step = 12
                message.is_dense = True
                message.data = struct.pack("<fff", 1.0, 0.0, 0.0)
            elif message_type == "livox_ros_driver2/msg/CustomMsg":
                point = get_message("livox_ros_driver2/msg/CustomPoint")()
                point.x = 1.0
                message.points = [point]
                message.point_num = 1
            elif message_type == "sensor_msgs/msg/LaserScan":
                message.angle_min, message.angle_max, message.angle_increment = -math.pi, math.pi, math.pi / 2
                message.range_min, message.range_max = 0.1, 8.0
                message.ranges = [1.0] * 5
            elif message_type == "sensor_msgs/msg/Imu":
                message.orientation.w = 1.0
            elif message_type == "nav_msgs/msg/Odometry":
                message.child_frame_id = "base_link"
                message.pose.pose.orientation.w = 1.0
            qos = spec["qos"]
            publisher = node.create_publisher(
                type(message),
                spec["endpoint"],
                QoSProfile(
                    depth=qos["depth"],
                    reliability=ReliabilityPolicy.RELIABLE
                    if qos["reliability"] == "reliable"
                    else ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL
                    if qos["durability"] == "transient_local"
                    else DurabilityPolicy.VOLATILE,
                ),
            )

            def publish(pub=publisher, msg=message, frame_id=frame):
                # Not every sensor message is stamped. A plain array is a
                # legitimate declared interface, and assuming a header here
                # raised AttributeError inside the timer — which took this
                # publisher down and with it every other synthetic topic it
                # serves, so an unrelated camera test was what failed.
                header = getattr(msg, "header", None)
                if header is not None:
                    header.stamp = node.get_clock().now().to_msg()
                    header.frame_id = frame_id
                pub.publish(msg)

            self._timers.append(node.create_timer(1.0 / float(rate), publish))


class SyntheticPerception(Node):
    def __init__(self):
        super().__init__("synthetic_perception")
        self.declare_parameter("description_json", "")
        self._streams = SyntheticStreams(self, json.loads(self.get_parameter("description_json").value))


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticPerception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
