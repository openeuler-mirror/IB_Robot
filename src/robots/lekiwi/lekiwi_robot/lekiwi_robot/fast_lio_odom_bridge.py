"""Normalize official FAST-LIO odometry for the robot navigation frame contract.

Canonical home since D12 (sensor processing belongs to the robot runtime):
composed by the lekiwi runtime launch. robot_navigation keeps a legacy copy
for provider-less configurations until those migrate to a runtime provider.
"""

from copy import deepcopy

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from lekiwi_robot.fast_lio_odom_contract import (
    base_pose_from_body_pose,
    invert_transform,
    project_planar_pose,
    transform_twist,
    validate_odometry_sample,
    validate_odometry_timestamp,
)


class FastLioOdomBridge(Node):
    """Convert official FAST-LIO camera_init/body odometry to odom/base_link."""

    def __init__(self) -> None:
        super().__init__("fast_lio_odom_bridge")
        self.declare_parameter("source_topic", "/fast_lio/odometry_raw")
        self.declare_parameter("output_topic", "/odometry/filtered")
        self.declare_parameter("source_odom_frame", "camera_init")
        self.declare_parameter("source_body_frame", "body")
        self.declare_parameter("output_odom_frame", "odom")
        self.declare_parameter("output_base_frame", "base_link")
        self.declare_parameter("body_to_base_translation", [0.0, 0.0, 0.0])
        self.declare_parameter("body_to_base_rotation", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("planar_output", False)
        self.declare_parameter("max_future_skew_sec", 0.1)

        self._source_odom_frame = str(self.get_parameter("source_odom_frame").value)
        self._source_body_frame = str(self.get_parameter("source_body_frame").value)
        self._output_odom_frame = str(self.get_parameter("output_odom_frame").value)
        self._output_base_frame = str(self.get_parameter("output_base_frame").value)
        self._translation = tuple(float(value) for value in self.get_parameter("body_to_base_translation").value)
        self._rotation = tuple(float(value) for value in self.get_parameter("body_to_base_rotation").value)
        if len(self._translation) != 3 or len(self._rotation) != 4:
            raise ValueError("FAST-LIO body-to-base transform requires 3 translation and 4 quaternion values")
        # rejects non-finite or zero-norm transforms up front
        invert_transform(self._translation, self._rotation)

        source_topic = str(self.get_parameter("source_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._planar_output = bool(self.get_parameter("planar_output").value)
        self._max_future_skew_sec = float(self.get_parameter("max_future_skew_sec").value)
        self._publisher = self.create_publisher(Odometry, output_topic, 10)
        self._broadcaster = TransformBroadcaster(self)
        self._subscription = self.create_subscription(Odometry, source_topic, self._on_odometry, 10)
        self._last_error = ""

    def _on_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        error = validate_odometry_sample(
            message.header.frame_id,
            message.child_frame_id,
            (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
            self._source_odom_frame,
            self._source_body_frame,
        )
        if error:
            if error != self._last_error:
                self.get_logger().error(f"Rejected FAST-LIO odometry: {error}")
                self._last_error = error
            return

        stamp_error = validate_odometry_timestamp(
            message.header.stamp.sec,
            message.header.stamp.nanosec,
            now_sec=self.get_clock().now().nanoseconds / 1_000_000_000.0,
            max_future_skew_sec=self._max_future_skew_sec,
        )
        if stamp_error:
            if stamp_error != self._last_error:
                self.get_logger().error(f"Rejected FAST-LIO odometry: {stamp_error}")
                self._last_error = stamp_error
            return

        self._last_error = ""
        output = deepcopy(message)
        # the base pose is the body pose conjugated by the mount (a yawed
        # mount must not yaw the odometry frame with it)
        position, orientation = base_pose_from_body_pose(
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            self._translation,
            self._rotation,
        )
        if self._planar_output:
            position, orientation = project_planar_pose(position, orientation)
        output.header.frame_id = self._output_odom_frame
        output.child_frame_id = self._output_base_frame
        output.pose.pose.position.x, output.pose.pose.position.y, output.pose.pose.position.z = position
        (
            output.pose.pose.orientation.x,
            output.pose.pose.orientation.y,
            output.pose.pose.orientation.z,
            output.pose.pose.orientation.w,
        ) = orientation
        linear, angular = transform_twist(
            (message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z),
            (message.twist.twist.angular.x, message.twist.twist.angular.y, message.twist.twist.angular.z),
            self._translation,
            self._rotation,
        )
        if self._planar_output:
            linear = (linear[0], linear[1], 0.0)
            angular = (0.0, 0.0, angular[2])
        output.twist.twist.linear.x, output.twist.twist.linear.y, output.twist.twist.linear.z = linear
        output.twist.twist.angular.x, output.twist.twist.angular.y, output.twist.twist.angular.z = angular
        output.twist.covariance[0] = -1.0
        self._publisher.publish(output)

        if self._publish_tf:
            transform = TransformStamped()
            transform.header = output.header
            transform.child_frame_id = output.child_frame_id
            transform.transform.translation.x = output.pose.pose.position.x
            transform.transform.translation.y = output.pose.pose.position.y
            transform.transform.translation.z = output.pose.pose.position.z
            transform.transform.rotation = output.pose.pose.orientation
            self._broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FastLioOdomBridge()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
