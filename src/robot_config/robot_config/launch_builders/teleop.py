"""Teleoperation node generation for robot_config.

This module provides utilities to generate teleoperation nodes
for integration with the robot_config launch system.
"""

import os
import re
import json
from pathlib import Path
from launch_ros.actions import Node
from robot_config.logger_utils import get_colored_logger
from typing import Dict, List, Any

from robot_config.utils import resolve_ros_path, prepare_lerobot_env

logger = get_colored_logger('robot_config.teleop')


def generate_teleop_nodes(robot_config: dict, robot_description_dict: dict = None) -> List[Node]:
    """
    Generate teleoperation nodes based on robot configuration.

    This function creates ROS 2 nodes for teleoperation based on the
    robot configuration YAML. It integrates with the robot_config launch system.

    Args:
        robot_config: Robot configuration dictionary loaded from YAML
        robot_description_dict: Dictionary containing robot_description (URDF)

    Returns:
        List of Node actions for teleoperation

    Example:
        >>> from robot_config.launch_builders.teleop import generate_teleop_nodes
        >>> config = load_robot_config('so101_single_arm')
        >>> nodes = generate_teleop_nodes(config, {'robot_description': '...'})
        >>> ld.add_action(nodes[0])

    Configuration Format:
        ```yaml
        robot:
          teleoperation:
            enabled: true
            active_device: "so101_leader"
            devices:
              - name: "so101_leader"
                type: "leader_arm"
                port: "/dev/ttyUSB0"
                calib_file: "~/.calibrate/so101_leader_calibrate.json"
            safety:
              joint_limits:
                "1": {"min": -3.14, "max": 3.14}
        ```
    """
    nodes = []

    # Get teleoperation config
    teleop_config = robot_config.get('teleoperation', {})
    if not teleop_config.get('enabled', False):
        logger.info("Teleoperation not enabled, skipping")
        return nodes

    # Get active device
    active_device_name = teleop_config.get('active_device', '')
    if not active_device_name:
        logger.warning("No active_device specified")
        return nodes

    # Find device config
    device_config = None
    for device in teleop_config.get('devices', []):
        if device.get('name') == active_device_name:
            device_config = device
            break

    if not device_config:
        logger.error(f"Active device '{active_device_name}' not found")
        return nodes

    # Get joint limits from safety config
    safety_config = teleop_config.get('safety', {})
    joint_limits = safety_config.get('joint_limits', {})

    # Get joint names from robot config
    joints_config = robot_config.get('joints', {})
    arm_joint_names = joints_config.get('arm', [])
    gripper_joint_names = joints_config.get('gripper', [])

    # Build device config for node parameter
    device_param = {
        'type': device_config.get('type', ''),
        'name': device_config.get('name', ''),
    }

    # Add optional device parameters
    if 'port' in device_config:
        device_param['port'] = device_config['port']
    if 'calib_file' in device_config:
        # Expand environment variables in calib_file path
        calib_file_raw = device_config['calib_file']
        calib_file_expanded = resolve_ros_path(device_config['calib_file'])
        logger.debug(f"calib_file_raw: {calib_file_raw}")
        logger.debug(f"calib_file_expanded: {calib_file_expanded}")
        device_param['calib_file'] = calib_file_expanded
        if not Path(calib_file_expanded).exists():
            logger.error("=" * 60)
            logger.error("Leader arm calibration file not found!")
            logger.error(f"  Resolved path: {calib_file_expanded}")
            logger.error(f"  Raw path:      {calib_file_raw}")
            logger.error(
                f"  HOME=$HOME -> {os.environ.get('HOME', '(unset)')}"
            )
            calib_port = device_config.get("port", "/dev/ttyACM0")
            logger.error("")
            logger.error("  Please run calibration first:")
            logger.error(
                "    ros2 run so101_hardware calibrate_arm --arm leader --port " + calib_port
            )
            logger.error("=" * 60)
            raise RuntimeError(
                f"Calibration file not found: {calib_file_expanded}. "
                f"Run: ros2 run so101_hardware calibrate_arm --arm leader --port " + calib_port
            )
    if 'joint_mapping' in device_config:
        device_param['joint_mapping'] = device_config['joint_mapping']

    # Add joint limits for proper scaling
    if joint_limits:
        device_param['joint_limits'] = joint_limits

    # Add any extra device-specific parameters
    known_keys = {
        'name', 'type', 'port', 'calib_file', 'joint_mapping', 'phone_config',
        'group_name', 'base_link_name', 'ee_frame_name', 'target_frame_name', 'ik_timeout',
    }
    for key, value in device_config.items():
        if key not in known_keys:
            device_param[key] = value

    if 'phone_config' in device_config:
        device_param['phone_config'] = device_config['phone_config']

    for moveit_key in ('group_name', 'base_link_name', 'ee_frame_name', 'target_frame_name', 'ik_timeout'):
        if moveit_key in device_config:
            device_param[moveit_key] = device_config[moveit_key]

    device_type = device_config.get('type', '')

    control_frequency = device_config.get('control_frequency', 50.0)

    # Prepare lerobot environment
    env = prepare_lerobot_env()

    # Convert dicts to JSON strings for ROS 2 parameter passing
    device_param_json = json.dumps(device_param)
    joint_limits_json = json.dumps(joint_limits)
    logger.debug(f"device_param_json: {device_param_json}")
    logger.debug(f"device_param dict: {device_param}")

    moveit_config = robot_config.get('moveit', {})

    # For phone devices: inject extra params into device_config so PhoneDevice
    # can read arm/gripper joint names, home positions, and servo frame at runtime.
    if device_type == 'phone':
        base_link_name = device_param.get('base_link_name', moveit_config.get('base_link', 'base_link'))
        reset_positions = robot_config.get('ros2_control', {}).get('reset_positions', {})
        home_positions_list = [reset_positions.get(n, 0.0) for n in arm_joint_names]

        device_param_ext = dict(device_param)
        device_param_ext.update({
            'arm_joint_names':      arm_joint_names,
            'gripper_joint_names':  gripper_joint_names,
            'home_joint_positions': home_positions_list,
            'base_link_name':       base_link_name,
            'control_frequency':    50.0,
        })
        device_param_json = json.dumps(device_param_ext)
        control_frequency = 50.0

    teleop_node = Node(
        package='robot_teleop',
        executable='teleop_node',
        name='robot_teleop_node',
        output='screen',
        env=env,
        parameters=[{
            'control_frequency':   control_frequency,
            'device_config':       device_param_json,
            'joint_limits':        joint_limits_json,
            'arm_joint_names':     arm_joint_names,
            'gripper_joint_names': gripper_joint_names,
        }],
    )
    nodes.append(teleop_node)
    logger.info(
        f"Generated teleop_node for device: {active_device_name} "
        f"(type: {device_type})"
    )

    # Add joy_node to read physical joystick and publish to /joy
    if device_config.get('type') == 'xbox_controller':
        input_dev = device_config.get('input_device', '/dev/input/js0')
        joy_node = Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'device_id': 0,
                'device_name': '',
                'deadzone': 0.1,
                'autorepeat_rate': 20.0,
                'sticky_buttons': False,
            }],
            output='screen'
        )
        nodes.append(joy_node)
        logger.info(f"Added joy_node for input device: {input_dev}")

    # Add MoveIt Servo node for Xbox controller and phone (both use Cartesian Servo control)
    if device_config.get('type') in ('xbox_controller', 'phone'):
        servo_node = _create_servo_node(robot_config, device_config, robot_description_dict)
        nodes.append(servo_node)
        logger.info("Generated servo_node for Cartesian control")

    return nodes


def _create_servo_node(robot_config: dict, device_config: dict, robot_description_dict: dict = None) -> Node:
    """Create MoveIt Servo node."""
    from robot_config.utils import resolve_ros_path
    import yaml

    # 1. Load servo parameters
    servo_config_name = device_config.get('servo_config', 'so101_servo')
    servo_params_path = resolve_ros_path(f"$(find robot_moveit)/config/{servo_config_name}.yaml")

    with open(servo_params_path, 'r') as f:
        servo_params = yaml.safe_load(f)

    # 2. Build MoveIt configuration manually for robustness
    # MoveItConfigsBuilder can be finicky with relative paths in different environments
    robot_type = robot_config.get('type', 'so101')

    moveit_params = {}
    try:
        # Load SRDF (Semantic Robot Description Format)
        srdf_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/{robot_type}.srdf")
        if os.path.exists(srdf_path):
            with open(srdf_path, 'r') as f:
                moveit_params["robot_description_semantic"] = f.read()
            logger.info(f"Loaded SRDF from {srdf_path}")
        else:
            logger.warning(f"SRDF not found at {srdf_path}")

        # Load Kinematics
        kinematics_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/kinematics.yaml")
        if os.path.exists(kinematics_path):
            with open(kinematics_path, 'r') as f:
                moveit_params["robot_description_kinematics"] = yaml.safe_load(f)
            logger.info(f"Loaded kinematics from {kinematics_path}")

        # Load Joint Limits
        joint_limits_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/joint_limits.yaml")
        if os.path.exists(joint_limits_path):
            with open(joint_limits_path, 'r') as f:
                joint_limits_data = yaml.safe_load(f)
                moveit_params.update(joint_limits_data)
                # MoveIt 2 nodes also look for joint_limits under robot_description_planning
                moveit_params["robot_description_planning"] = joint_limits_data
            logger.info(f"Loaded joint limits from {joint_limits_path}")
    except Exception as e:
        logger.warning(f"Failed to manually load MoveIt configs: {e}")

    # Merge robot_description_dict to ensure robot_description
    # and use_sim_time are always present (from description layer)
    if robot_description_dict:
        moveit_params.update(robot_description_dict)
        # If use_sim_time is true, also tell moveit_servo to use gazebo if configured
        if robot_description_dict.get("use_sim_time"):
            servo_params["use_gazebo"] = True

    # Create the servo node
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        output='screen',
        parameters=[
            {"moveit_servo": servo_params},
            moveit_params,
        ],
    )
    return servo_node


def validate_teleop_config(teleop_config: Dict[str, Any]) -> List[str]:
    """
    Validate teleoperation configuration.

    Args:
        teleop_config: Teleoperation configuration dictionary

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> errors = validate_teleop_config(config['teleoperation'])
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    errors = []

    if not teleop_config.get('enabled', False):
        return errors  # Not enabled, skip validation

    # Check active device
    active_device = teleop_config.get('active_device')
    if not active_device:
        errors.append("active_device must be specified when teleoperation is enabled")
        return errors

    # Check devices list
    devices = teleop_config.get('devices', [])
    if not devices:
        errors.append("devices list is empty")
        return errors

    # Find active device in list
    device_found = False
    for device in devices:
        if device.get('name') == active_device:
            device_found = True

            # Validate device type
            if not device.get('type'):
                errors.append(f"Device '{active_device}': missing 'type' field")

            # Type-specific validation
            if device.get('type') == 'leader_arm':
                if not device.get('port'):
                    errors.append(f"Device '{active_device}': leader_arm requires 'port' field")

            if device.get('type') == 'phone':
                phone_config = device.get('phone_config', {})
                if not phone_config:
                    errors.append(f"Device '{active_device}': phone requires 'phone_config' field")

            break

    if not device_found:
        errors.append(f"active_device '{active_device}' not found in devices list")

    # Validate safety config
    safety = teleop_config.get('safety', {})
    joint_limits = safety.get('joint_limits', {})

    if not joint_limits:
        errors.append("safety.joint_limits not specified (recommended for safe operation)")
    else:
        # Validate joint limit format
        for joint_name, limits in joint_limits.items():
            if 'min' not in limits or 'max' not in limits:
                errors.append(f"Joint '{joint_name}': limits must have 'min' and 'max' fields")
            elif limits['min'] >= limits['max']:
                errors.append(f"Joint '{joint_name}': min must be less than max")

    return errors


def get_recording_topics(robot_config: dict) -> List[str]:
    """
    Get list of topics to record for teleoperation sessions.

    Args:
        robot_config: Robot configuration dictionary

    Returns:
        List of topic names for rosbag recording

    Example:
        >>> topics = get_recording_topics(config)
        >>> cmd = ['ros2', 'bag', 'record'] + topics
    """
    topics = []

    # Always record joint states
    topics.append('/joint_states')

    # Add controller command topics
    topics.append('/arm_position_controller/commands')
    topics.append('/gripper_position_controller/commands')

    # Add teleop diagnostics
    topics.append('/diagnostics')

    # Add camera topics from peripherals
    peripherals = robot_config.get('peripherals', [])
    for peripheral in peripherals:
        if peripheral.get('type') == 'camera':
            name = peripheral.get('name', 'camera')
            # Add common camera topics
            topics.append(f'/camera/{name}/image_raw')
            topics.append(f'/camera/{name}/camera_info')

    return topics
