"""Standalone launch for hardware_mock (debugging only).

Most users should invoke this via robot_config's robot.launch.py with
``use_mock:=true`` so the rest of the stack (inference, action_dispatch,
recording) is wired up. This launch file only spawns the mock node.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    cfg_name = context.launch_configurations.get("robot_config", "so101_single_arm")
    override = context.launch_configurations.get("config_path", "")
    if override:
        config_path = override
    else:
        share = get_package_share_directory("robot_config")
        config_path = str(Path(share) / "config" / "robots" / f"{cfg_name}.yaml")

    return [
        Node(
            package="hardware_mock",
            executable="contract_mock",
            name="contract_mock",
            output="screen",
            parameters=[{"robot_config_path": config_path, "use_sim_time": False}],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config",
                default_value="so101_single_arm",
                description="Robot configuration name (without .yaml)",
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Optional explicit path to the robot YAML",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
