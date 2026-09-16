#!/usr/bin/env python3
"""
Standalone launch file for robot_teleop testing

Launches teleop node with leader arm device for testing without full robot_config integration.

Usage:
    ros2 launch robot_teleop teleop_device.launch.py \
        port:=/dev/ttyACM1 \
        calib_file:=/path/to/calib.json \
        control_frequency:=50.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for teleop device testing."""

    # Declare launch arguments
    port_arg = DeclareLaunchArgument(
        "port", default_value="/dev/ttyACM1", description="Serial port for leader arm device"
    )

    calib_file_arg = DeclareLaunchArgument(
        "calib_file", default_value="", description="Path to calibration file (optional)"
    )

    control_frequency_arg = DeclareLaunchArgument(
        "control_frequency", default_value="50.0", description="Control loop frequency in Hz"
    )

    # Device configuration (constructed from launch args)
    device_config = {
        "type": "leader_arm",
        "name": "so101_leader",
        "port": LaunchConfiguration("port"),
        "calib_file": LaunchConfiguration("calib_file"),
    }

    # Joint limits and gripper ratio endpoints are resolved from the runtime
    # public description (/runtime/get_status) by teleop_node at startup —
    # no deployment snapshots here, so recalibration never invalidates them.

    # Teleop node
    teleop_node = Node(
        package="robot_teleop",
        executable="teleop_node",
        name="robot_teleop_node",
        output="screen",
        parameters=[
            {
                "control_frequency": LaunchConfiguration("control_frequency"),
                "device_config": device_config,
            }
        ],
        remappings=[
            # Add any topic remappings if needed
        ],
    )

    return LaunchDescription(
        [
            # Launch arguments
            port_arg,
            calib_file_arg,
            control_frequency_arg,
            # Log startup info
            LogInfo(msg=["Starting robot_teleop with leader arm on port: ", LaunchConfiguration("port")]),
            LogInfo(msg=["Control frequency: ", LaunchConfiguration("control_frequency"), " Hz"]),
            # Nodes
            teleop_node,
        ]
    )
