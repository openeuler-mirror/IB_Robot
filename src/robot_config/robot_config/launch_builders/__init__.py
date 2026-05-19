"""Launch builder modules for robot_config.

This package contains modules for building ROS2 launch components:
- description.py: URDF building (xacro processing + camera injection)
- control.py: ros2_control nodes, controller spawners
- perception.py: Camera drivers, TF publishers
- simulation.py: Gazebo and simulation nodes
- execution.py: Action dispatcher and inference nodes
"""

from robot_config.launch_builders.control import (
    generate_controller_spawners,
    generate_ros2_control_nodes,
    validate_joint_config,
)
from robot_config.launch_builders.description import (
    generate_robot_description,
)
from robot_config.launch_builders.hardware_mock import (
    generate_hardware_mock_nodes,
    mock_mode_skips_subsystem,
    validate_mock_mode,
)
from robot_config.launch_builders.moveit import generate_moveit_nodes
from robot_config.launch_builders.navigation import (
    generate_navigation_nodes,
)
from robot_config.launch_builders.perception import (
    generate_camera_nodes,
    generate_lidar_nodes,
    generate_tf_nodes,
    generate_virtual_camera_relays,
)
from robot_config.launch_builders.simulation import generate_gazebo_nodes
from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

__all__ = [
    # Description
    "generate_robot_description",
    # Control
    "generate_ros2_control_nodes",
    "generate_controller_spawners",
    "validate_joint_config",
    # Perception
    "generate_camera_nodes",
    "generate_lidar_nodes",
    "generate_tf_nodes",
    "generate_virtual_camera_relays",
    # Simulation
    "generate_gazebo_nodes",
    # MoveIt
    "generate_moveit_nodes",
    # Voice ASR
    "generate_voice_asr_nodes",
    # Navigation
    "generate_navigation_nodes",
    # Hardware mock
    "generate_hardware_mock_nodes",
    "mock_mode_skips_subsystem",
    "validate_mock_mode",
]
