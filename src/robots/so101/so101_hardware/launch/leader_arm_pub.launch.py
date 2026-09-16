"""Launch provisioned read-only leader input independently of the follower."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _source(context):
    values = {
        name: LaunchConfiguration(name).perform(context)
        for name in ("port", "calib_file", "publish_rate", "calibration_version", "source_topic")
    }
    if int(values["calibration_version"]) != 1:
        raise ValueError("provision leader first, then explicitly pass calibration_version:=1")
    return [
        Node(
            package="so101_hardware",
            executable="leader_arm_pub",
            name="so101_leader_input",
            output="screen",
            parameters=[
                {
                    "port": values["port"],
                    "calibration_file": values["calib_file"],
                    "calibration_version": int(values["calibration_version"]),
                    "publish_rate": float(values["publish_rate"]),
                    "source_topic": values["source_topic"],
                }
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("port", description="Operator leader serial port"),
            DeclareLaunchArgument("calib_file", description="Provisioned leader calibration JSON"),
            DeclareLaunchArgument(
                "calibration_version", default_value="0", description="Explicit provisioning acceptance"
            ),
            DeclareLaunchArgument("publish_rate", default_value="50.0"),
            DeclareLaunchArgument("source_topic", default_value="/inputs/so101_leader/state"),
            OpaqueFunction(function=_source),
        ]
    )
