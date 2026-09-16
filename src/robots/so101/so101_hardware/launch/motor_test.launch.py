"""Motor diagnostics launch file for SO-101 robot arm."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    """Generate launch description for motor diagnostics."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "port", default_value="/dev/ttyACM0", description="Serial port for motor communication"
            ),
            # TODO: Add motor diagnostic tools when implemented
        ]
    )
