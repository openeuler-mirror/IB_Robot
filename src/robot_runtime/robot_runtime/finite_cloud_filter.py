"""Drop non-finite points from a simulated lidar cloud (D5).

Gazebo reports gpu_lidar beams without a hit as ``±inf``; FAST-LIO's generic
handler keeps them, so the provider bridges the raw cloud on an internal
topic and this robot-agnostic node republishes the finite points on the
advertised endpoint with the same header, fields and point layout. Runs on
simulation time as part of the Gazebo embodiment.
"""

from __future__ import annotations

import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


def filter_finite_points(message: PointCloud2) -> PointCloud2:
    """A copy of ``message`` holding only the points with finite x/y/z.

    The header, field layout, point step and endianness are preserved; a
    cloud that is already finite is returned as-is (no copy, no resharding).
    """
    count = message.width * message.height
    if count == 0:
        return message
    offsets = {field.name: field.offset for field in message.fields}
    if not {"x", "y", "z"} <= offsets.keys():
        return message
    points = np.frombuffer(bytes(message.data), dtype=np.uint8, count=count * message.point_step).reshape(
        count, message.point_step
    )
    finite = np.ones(count, dtype=bool)
    for axis in ("x", "y", "z"):
        column = np.ascontiguousarray(points[:, offsets[axis] : offsets[axis] + 4])
        finite &= np.isfinite(column.view(np.float32).reshape(-1))
    if finite.all():
        return message
    filtered = PointCloud2()
    filtered.header = message.header
    filtered.fields = message.fields
    filtered.point_step = message.point_step
    filtered.is_bigendian = message.is_bigendian
    filtered.is_dense = message.is_dense
    filtered.width = int(finite.sum())
    filtered.height = 1
    filtered.data = points[finite].tobytes()
    return filtered


class FiniteCloudFilter(Node):
    def __init__(self):
        super().__init__("finite_cloud_filter")
        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", "")
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        if not input_topic or not output_topic:
            raise ValueError("finite_cloud_filter requires the input_topic and output_topic parameters")
        self._publisher = self.create_publisher(PointCloud2, output_topic, 10)
        self._subscription = self.create_subscription(PointCloud2, input_topic, self._on_cloud, 10)

    def _on_cloud(self, message: PointCloud2) -> None:
        # a stateless relay: the simulation-time stamps of the source cloud
        # are preserved verbatim in the republished header
        self._publisher.publish(filter_finite_points(message))


def main(args=None):
    rclpy.init(args=args)
    node = FiniteCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
