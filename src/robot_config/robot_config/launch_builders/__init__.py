"""Launch builder modules for robot_config.

This package contains modules for building ROS2 launch components:
- description.py: URDF building (xacro processing + camera injection)
- control.py: ros2_control nodes, controller spawners
- perception.py: Camera drivers, TF publishers
- simulation.py: Gazebo and simulation nodes
- execution.py: Action dispatcher and inference nodes
"""

from robot_config.launch_builders.benchmark import generate_benchmark_nodes
from robot_config.launch_builders.control import (
    generate_controller_spawners,
    generate_ros2_control_nodes,
    validate_joint_config,
    validate_runtime_resources,
)
from robot_config.launch_builders.description import (
    generate_robot_description,
)
from robot_config.launch_builders.hand_sources import generate_hand_source_nodes
from robot_config.launch_builders.hardware_mock import (
    generate_hardware_mock_nodes,
    mock_mode_skips_subsystem,
)
from robot_config.launch_builders.navigation import (
    generate_navigation_nodes,
)
from robot_config.launch_builders.perception import (
    generate_camera_nodes,
    generate_lidar_nodes,
    generate_tf_nodes,
    generate_virtual_camera_relays,
)
from robot_config.launch_builders.perception_models import generate_perception_model_nodes
from robot_config.launch_builders.runtime import generate_runtime_provider_actions, runtime_provider
from robot_config.launch_builders.simulation import generate_gazebo_nodes


def __getattr__(name: str):
    if name == "generate_audio_io_actions":
        from robot_config.launch_builders.audio_io import generate_audio_io_actions

        return generate_audio_io_actions
    if name == "generate_voice_asr_nodes":
        from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

        return generate_voice_asr_nodes
    if name == "generate_voice_tts_nodes":
        from robot_config.launch_builders.voice_tts import generate_voice_tts_nodes

        return generate_voice_tts_nodes
    if name == "generate_speech_direction_actions":
        from robot_config.launch_builders.speech_direction import generate_speech_direction_actions

        return generate_speech_direction_actions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Description
    "generate_robot_description",
    # Control
    "generate_ros2_control_nodes",
    "generate_controller_spawners",
    "generate_hand_source_nodes",
    "validate_runtime_resources",
    "validate_joint_config",
    # Perception
    "generate_camera_nodes",
    "generate_lidar_nodes",
    "generate_tf_nodes",
    "generate_virtual_camera_relays",
    "generate_perception_model_nodes",
    # Simulation
    "generate_gazebo_nodes",
    # MoveIt
    "generate_runtime_provider_actions",
    "runtime_provider",
    # Voice ASR
    "generate_voice_asr_nodes",
    "generate_audio_io_actions",
    "generate_voice_tts_nodes",
    "generate_speech_direction_actions",
    # Navigation
    "generate_navigation_nodes",
    # Hardware mock
    "generate_hardware_mock_nodes",
    "mock_mode_skips_subsystem",
    # Benchmark
    "generate_benchmark_nodes",
]
