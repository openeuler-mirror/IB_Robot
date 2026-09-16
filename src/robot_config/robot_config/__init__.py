"""robot_config package - Unified robot configuration system."""

from robot_config.config import (
    CameraConfig,
    ContractExtensionConfig,
    PeripheralConfig,
    RobotConfig,
    Ros2ControlConfig,
)
from robot_config.inference_config import (
    ControlModeInferenceConfig,
    InferenceConfigError,
    InferencePipelineConfig,
    InferenceTransportConfig,
    parse_inference_config,
)
from robot_config.observation_transport import (
    H264Spec,
    ObservationTransportSpec,
    RtpEndpointSpec,
    VideoBufferSpec,
    VideoMediaSpec,
    VideoReadinessSpec,
)
from robot_config.perception_runtime_config import (
    PerceptionModelServiceConfig,
    PerceptionRuntimeConfig,
    PerceptionRuntimeConfigError,
    parse_perception_runtime_config,
)

_LAZY_EXPORTS = {
    "bind_robot_interfaces": ("robot_config.interface_binding", "bind_robot_interfaces"),
    "build_contract_from_robot_config_dict": (
        "robot_config.loader",
        "build_contract_from_robot_config_dict",
    ),
    "load_robot_config": ("robot_config.loader", "load_robot_config"),
    "load_robot_config_dict": ("robot_config.loader", "load_robot_config_dict"),
    "load_robot_section": ("robot_config.loader", "load_robot_section"),
    "resolve_robot_config_path": ("robot_config.config_path", "resolve_robot_config_path"),
    "validate_config": ("robot_config.loader", "validate_config"),
    "parse_bool": ("robot_config.utils", "parse_bool"),
    "resolve_ros_path": ("robot_config.utils", "resolve_ros_path"),
    "generate_ros2_control_nodes": (
        "robot_config.launch_builders",
        "generate_ros2_control_nodes",
    ),
    "generate_camera_nodes": (
        "robot_config.launch_builders",
        "generate_camera_nodes",
    ),
    "generate_tf_nodes": (
        "robot_config.launch_builders",
        "generate_tf_nodes",
    ),
    "generate_virtual_camera_relays": (
        "robot_config.launch_builders",
        "generate_virtual_camera_relays",
    ),
    "generate_gazebo_nodes": (
        "robot_config.launch_builders",
        "generate_gazebo_nodes",
    ),
    "validate_joint_config": (
        "robot_config.launch_builders",
        "validate_joint_config",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    # Config classes
    "RobotConfig",
    "Ros2ControlConfig",
    "PeripheralConfig",
    "ContractExtensionConfig",
    "CameraConfig",
    "ObservationTransportSpec",
    "RtpEndpointSpec",
    "H264Spec",
    "VideoMediaSpec",
    "VideoBufferSpec",
    "VideoReadinessSpec",
    # Loaders
    "bind_robot_interfaces",
    "load_robot_config",
    "load_robot_config_dict",
    "load_robot_section",
    "resolve_robot_config_path",
    "build_contract_from_robot_config_dict",
    "validate_config",
    "ControlModeInferenceConfig",
    "InferenceConfigError",
    "InferencePipelineConfig",
    "InferenceTransportConfig",
    "parse_inference_config",
    "PerceptionModelServiceConfig",
    "PerceptionRuntimeConfig",
    "PerceptionRuntimeConfigError",
    "parse_perception_runtime_config",
    # Launch builders
    "generate_ros2_control_nodes",
    "generate_camera_nodes",
    "generate_tf_nodes",
    "generate_virtual_camera_relays",
    "generate_gazebo_nodes",
    "validate_joint_config",
    # Utilities
    "resolve_ros_path",
    "parse_bool",
]
