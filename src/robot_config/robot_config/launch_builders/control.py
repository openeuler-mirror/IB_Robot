"""Control system launch builders.

This module handles:
- ros2_control node generation
- Controller spawner creation
- Joint configuration validation (delegates to utils.py)

URDF building (xacro processing + camera injection) is in description.py.
"""


import os
import tempfile
from pathlib import Path

import yaml
from robot_config.logger_utils import get_colored_logger
from launch_ros.actions import Node

from robot_config.utils import resolve_ros_path, parse_bool, validate_joint_config
from robot_config.launch_builders.description import generate_robot_description

logger = get_colored_logger("robot_config.control")


def generate_controller_spawners(controller_names, use_sim=True, controller_manager_name="controller_manager"):
    """Generate controller spawner nodes.

    Args:
        controller_names: List of controller names to spawn
        use_sim: Simulation mode (affects timeout and use_sim_time)
        controller_manager_name: Name of controller manager service

    Returns:
        List of Node actions for controller spawners
    """
    is_sim = parse_bool(use_sim, default=True)

    if not controller_names:
        return []

    timeout = 60 if is_sim else 10
    switch_timeout = 30 if is_sim else 5

    return [Node(
        package="controller_manager",
        executable="spawner",
        name=f"spawner_{controller_names[0]}_group",
        parameters=[
            {"use_sim_time": is_sim}
        ],
        arguments=[
            *controller_names,
            "--controller-manager",
            controller_manager_name,
            "--controller-manager-timeout",
            str(timeout),
            "--switch-timeout",
            str(switch_timeout),
            "--activate-as-group",
        ],
        output="screen",
    )]


def generate_ros2_control_nodes(robot_config, use_sim, auto_start_controllers='true'):
    """Generate ros2_control nodes from configuration.

    Args:
        robot_config: Robot configuration dict
        use_sim: Simulation mode flag (string or bool)
        auto_start_controllers: Whether to automatically start controllers (string or bool)

    Returns:
        Tuple: (nodes, controller_names, deferred_sim_spawners, robot_description)
        In Gazebo simulation, controller spawners are returned in
        ``deferred_sim_spawners`` (not included in ``nodes``) so launch can
        start them after ``ros_gz_sim create`` exits.
    """
    is_sim = parse_bool(use_sim, default=False)
    is_auto_start = parse_bool(auto_start_controllers, default=True)

    nodes = []
    deferred_sim_spawners = []
    ros2_control_config = robot_config.get("ros2_control")

    if not ros2_control_config:
        logger.warning("No ros2_control configuration found")
        return nodes, [], deferred_sim_spawners, {}

    logger.info("Creating ros2_control nodes")

    # Pre-flight check: calibration file must exist for real hardware
    if not is_sim:
        calib_file_raw = ros2_control_config.get("calib_file", "")
        if calib_file_raw:
            calib_file_resolved = resolve_ros_path(calib_file_raw)
            if not Path(calib_file_resolved).exists():
                logger.error("Calibration file not found!")
                logger.error(f"  Resolved path: {calib_file_resolved}")
                logger.error(f"  Raw path:      {calib_file_raw}")
                logger.error(
                    f"  HOME=$HOME -> {os.environ.get('HOME', '(unset)')}"
                )
                logger.error("")
                logger.error("  Please run calibration first:")
                calib_port = ros2_control_config.get("port", "/dev/ttyACM0")
                logger.error(
                    "    ros2 run so101_hardware calibrate_arm --arm follower --port " + calib_port
                )
                raise RuntimeError(
                    f"Calibration file not found: {calib_file_resolved}. "
                    f"Run: ros2 run so101_hardware calibrate_arm --arm follower --port " + calib_port
                )

    # Validate joint configuration
    validate_joint_config(robot_config)

    # Build URDF (xacro processing + camera injection) via description layer
    _desc_result = generate_robot_description(robot_config, use_sim)
    if _desc_result is None:
        return nodes, [], deferred_sim_spawners, {}

    robot_description_str, robot_description = _desc_result

    # Robot State Publisher
    nodes.append(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    ))

    # Get control mode configuration
    control_mode_name = robot_config.get("default_control_mode", "model_inference")
    control_modes = robot_config.get("control_modes", {})

    if control_modes:
        if control_mode_name not in control_modes:
            available_modes = list(control_modes.keys())
            logger.error(f"Control mode '{control_mode_name}' not found")
            logger.info(f"Available modes: {available_modes}")
            if available_modes:
                control_mode_name = available_modes[0]

        if control_mode_name:
            mode_config = control_modes[control_mode_name]
            controller_names = mode_config.get("controllers", [])
            mode_description = mode_config.get("description", "No description")
            logger.info(f"Using control mode: {control_mode_name}")
            logger.info(f"  Description: {mode_description}")
            logger.info(f"  Controllers: {controller_names}")
        else:
            controller_names = []
    else:
        controller_names = ros2_control_config.get("controllers", [])

    controllers_config = resolve_ros_path(ros2_control_config.get("controllers_config"))

    if not is_sim:
        # Real hardware mode
        logger.info("Real hardware mode")

        if controllers_config and Path(controllers_config).exists():
            logger.info(f"Controllers config: {controllers_config}")

            # Write robot_description to a temp YAML under the 'controller_manager'
            # node name.  ros2_control_node internally creates a node called
            # 'controller_manager', but launch writes dict params under the
            # executable name ('ros2_control_node') — a namespace mismatch.
            # Using a file with the correct key avoids the mismatch WITHOUT
            # setting name= on the Node (which would add a global __node
            # remapping that breaks child controller nodes).
            cm_params_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False,
                prefix='cm_robot_desc_',
            )
            yaml.dump(
                {'controller_manager': {'ros__parameters': {
                    'robot_description': robot_description_str,
                }}},
                cm_params_file,
                default_flow_style=False,
            )
            cm_params_file.close()
            logger.info(f"Controller manager params: {cm_params_file.name}")

            nodes.append(Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[cm_params_file.name, controllers_config],
                remappings=[
                    ("~/robot_description", "/robot_description"),
                ],
                output="screen",
            ))

            if is_auto_start and controller_names:
                spawners = generate_controller_spawners(controller_names, use_sim=False)
                nodes.extend(spawners)
    else:
        # Simulation mode
        # gz_ros2_control plugin provides controller_manager, but spawners
        # must wait until the Gazebo entity is fully created and the plugin
        # has initialized the hardware interface.
        logger.info("Simulation mode: Gazebo provides controller_manager")
        logger.info(f"Controllers to spawn (deferred until after gz spawn): {controller_names}")

        if is_auto_start and controller_names:
            deferred_sim_spawners = generate_controller_spawners(controller_names, use_sim=True)
            logger.info(
                f"Deferring {len(deferred_sim_spawners)} controller spawners "
                "(handled by caller)"
            )

    return nodes, controller_names, deferred_sim_spawners, robot_description
