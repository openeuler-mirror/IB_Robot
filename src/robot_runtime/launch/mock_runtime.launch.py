"""Launch the mock runtime (conformance reference implementation)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="robot_runtime",
                executable="mock_runtime",
                name="mock_runtime",
                output="screen",
            ),
        ]
    )
