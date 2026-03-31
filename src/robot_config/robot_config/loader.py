"""Configuration loader and validator for robot_config."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import logging
import yaml

from robot_config.config import (
    RobotConfig,
    Ros2ControlConfig,
    CameraConfig,
    PeripheralConfig,
    ContractExtensionConfig,
    ContractObservation,
    ContractAction,
    VoiceASRConfig,
)

from .utils import resolve_ros_path

logger = logging.getLogger(__name__)


def load_camera_config(data: Dict[str, Any]) -> CameraConfig:
    """Load camera configuration from dict.

    Example:
    ```yaml
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame
      camera_info_url: file:///path/to/calibration.yaml
    ```
    """
    driver = data.get("driver", "opencv")

    # Handle different index/port naming conventions
    index_or_port = data.get("index", data.get("port", data.get("serial_number", 0)))
    if driver == "realsense" and "serial_number" in data:
        index_or_port = data["serial_number"]

    return CameraConfig(
        name=data["name"],
        driver=driver,
        index_or_port=index_or_port,
        width=data.get("width", 640),
        height=data.get("height", 480),
        fps=data.get("fps", 30),
        frame_id=data.get("frame_id", f"camera_{data['name']}_frame"),
        optical_frame_id=data.get("optical_frame_id", f"camera_{data['name']}_optical_frame"),
        camera_info_url=data.get("camera_info_url"),
        pixel_format=data.get("pixel_format", "bgr8"),
        depth_width=data.get("depth_width"),
        depth_height=data.get("depth_height"),
        depth_fps=data.get("depth_fps"),
    )


def load_ros2_control_config(data: Dict[str, Any], config_dir: Optional[Path] = None) -> Ros2ControlConfig:
    """Load ros2_control configuration from dict.

    Example:
    ```yaml
    ros2_control:
      hardware_plugin: so101_hardware/SO101SystemHardware
      port: /dev/ttyACM0
      calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
      reset_positions: {1: 0.0, 2: 0.0}
      urdf_path: $(find robot_description)/urdf/lerobot/so101/so101.urdf.xacro
    ```
    """
    params = {}
    for key, value in data.items():
        if key not in ["hardware_plugin", "urdf_path"]:
            # Resolve paths in parameters
            if isinstance(value, str):
                    value = resolve_ros_path(value)
            params[key] = value

    return Ros2ControlConfig(
        hardware_plugin=data.get("hardware_plugin", ""),
        params=params,
        urdf_path=resolve_ros_path(data.get("urdf_path")),
    )


def load_contract_config(data: Dict[str, Any]) -> ContractExtensionConfig:
    """Load contract extension configuration from dict.

    Example:
    ```yaml
    contract:
      base_contract: $(find robot_config)/config/contracts/act_grab_pan.yaml
      observations:
        - key: observation.images.top
          topic: /camera/top
          peripheral: top
      actions:
        - key: action
          publish:
            topic: /joint_commands
            type: sensor_msgs/msg/JointState
    ```
    """
    observations = []
    for obs_data in data.get("observations", []):
        observations.append(
            ContractObservation(
                key=obs_data["key"],
                topic=obs_data.get("topic"),
                type=obs_data.get("type"),
                peripheral=obs_data.get("peripheral"),
                selector=obs_data.get("selector"),
                image=obs_data.get("image"),
                align=obs_data.get("align"),
                qos=obs_data.get("qos"),
            )
        )

    actions = []
    for action_data in data.get("actions", []):
        actions.append(
            ContractAction(
                key=action_data["key"],
                publish=action_data.get("publish", {}),
                selector=action_data.get("selector"),
                from_tensor=action_data.get("from_tensor"),
                safety_behavior=action_data.get("safety_behavior", "zeros"),
            )
        )

    return ContractExtensionConfig(
        base_contract=data.get("base_contract"),
        observations=observations,
        actions=actions,
        rate_hz=data.get("rate_hz", 20.0),
        max_duration_s=data.get("max_duration_s", 30.0),
    )


def load_voice_asr_config(data: Dict[str, Any]) -> VoiceASRConfig:
    """Load voice ASR configuration from dict."""
    model_path = data.get("model_path", "")
    tokens_path = data.get("tokens_path", "")

    return VoiceASRConfig(
        enabled=data.get("enabled", False),
        active_mode=data.get("active_mode", "manual"),
        language=data.get("language", "zh"),
        model_path=resolve_ros_path(model_path) if model_path else "",
        tokens_path=resolve_ros_path(tokens_path) if tokens_path else "",
        provider=data.get("provider", "cpu"),
        model_type=data.get("model_type", "auto"),
        max_recording_duration=data.get("max_recording_duration", 10.0),
        vad_sensitivity=data.get("vad_sensitivity", 0.5),
        publish_partial=data.get("publish_partial", True),
        output_topic=data.get("output_topic", "/voice_command"),
        sample_rate=data.get("sample_rate", 16000),
        chunk_size=data.get("chunk_size", 512),
        buffer_seconds=data.get("buffer_seconds", 5.0),
        device_index=data.get("device_index", -1),
    )


def load_robot_config(config_path: Union[str, Path]) -> RobotConfig:
    """Load robot configuration from YAML file.

    Args:
        config_path: Path to robot configuration YAML file

    Returns:
        RobotConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    config_dir = config_path.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Robot configuration not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if "robot" not in data:
        raise ValueError(f"Invalid robot config: missing 'robot' section in {config_path}")

    robot_data = data["robot"]

    # Load required fields
    name = robot_data.get("name")
    if not name:
        raise ValueError(f"Invalid robot config: missing 'name' in {config_path}")

    robot_type = robot_data.get("robot_type", robot_data.get("type", name))
    type_ = robot_data.get("type", name)

    # Load ros2_control config
    ros2_control_data = robot_data.get("ros2_control", {})
    ros2_control = load_ros2_control_config(ros2_control_data, config_dir)

    # Load peripherals (cameras)
    peripherals = []
    for periph_data in robot_data.get("peripherals", []):
        if periph_data.get("type") == "camera":
            peripherals.append(load_camera_config(periph_data))
        else:
            # Generic peripheral
            peripherals.append(
                PeripheralConfig(
                    type=periph_data["type"],
                    name=periph_data["name"],
                    driver=periph_data.get("driver", "generic"),
                    params=periph_data.get("params", {}),
                    frame_id=periph_data.get("frame_id"),
                )
            )

    # Load contract config
    contract_data = robot_data.get("contract", {})
    contract = load_contract_config(contract_data)

    voice_asr = load_voice_asr_config(robot_data.get("voice_asr", {}))

    return RobotConfig(
        name=name,
        type=type_,
        robot_type=robot_type,
        ros2_control=ros2_control,
        peripherals=peripherals,
        contract=contract,
        voice_asr=voice_asr,
    )


def validate_config(config: RobotConfig) -> List[str]:
    """Validate robot configuration.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate ros2_control config
    if not config.ros2_control.hardware_plugin:
        errors.append("ros2_control.hardware_plugin is required")

    # Check calib_file exists if specified
    if "calib_file" in config.ros2_control.params:
        calib_file = config.ros2_control.params["calib_file"]
        if calib_file and not Path(calib_file).exists():
            errors.append(f"Calibration file not found: {calib_file}")

    # Validate cameras
    camera_names = set()
    for cam in config.peripherals:
        if cam.name in camera_names:
            errors.append(f"Duplicate camera name: {cam.name}")
        camera_names.add(cam.name)

        # Validate camera parameters
        if cam.width <= 0 or cam.height <= 0:
            errors.append(f"Invalid camera dimensions for {cam.name}: {cam.width}x{cam.height}")
        if cam.fps <= 0:
            errors.append(f"Invalid FPS for {cam.name}: {cam.fps}")

        # Validate calibration file if specified
        if cam.camera_info_url and cam.camera_info_url.startswith("file://"):
            calib_path = Path(cam.camera_info_url.replace("file://", ""))
            if not calib_path.exists():
                errors.append(f"Camera calibration file not found: {calib_path}")

    # Validate contract-peripheral references
    for obs in config.contract.observations:
        if obs.peripheral:
            if obs.peripheral not in camera_names:
                errors.append(
                    f"Observation '{obs.key}' references undefined peripheral: {obs.peripheral}"
                )

    if config.voice_asr.enabled and not config.voice_asr.model_path:
        errors.append("voice_asr.model_path is required when voice_asr.enabled is true")

    return errors


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """Validate a robot configuration file.

    Returns:
        True if valid, False otherwise
    """
    try:
        config = load_robot_config(config_path)
        errors = validate_config(config)

        if errors:
            logger.error(f"Configuration errors in {config_path}:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info(f"Configuration {config_path} is valid")
        return True

    except Exception as e:
        logger.error(f"Failed to validate {config_path}: {e}")
        return False
