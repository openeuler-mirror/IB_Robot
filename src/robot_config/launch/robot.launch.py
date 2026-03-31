"""Main robot launch file for robot_config.

This launch file loads robot configuration from YAML and dynamically generates:
- ros2_control hardware interface and controllers
- Robot state publisher
- Camera drivers (usb_cam, realsense2_camera)
- Static TF publishers for camera frames
- Voice ASR node (optional, configured from robot.voice_asr)
- Inference service and action dispatcher (optional, auto-detected)
- MoveIt motion planning (optional, auto-detected)

Controllers are automatically spawned in both simulation and real hardware modes:
- Simulation mode: Uses Gazebo's gz_ros2_control plugin for controller_manager
- Hardware mode: Starts ros2_control_node for controller_manager

Expected ROS interfaces (depends on ``control_mode`` and options):
- ``control_mode:=moveit_planning`` (and MoveIt enabled): planning/move_group topics such as ``/planning_scene``; not started for ``model_inference`` or ``teleop`` alone.
- Gazebo sim + cameras: bridged topics ``/camera/{top,wrist,front}/image_raw`` and ``.../camera_info`` (names from YAML ``peripherals[].name``), not raw Ignition link paths.
- After controller spawners succeed: ``/arm_position_controller/commands``, ``/gripper_position_controller/commands``, ``/joint_states``, etc.

**CRITICAL**: This workspace uses ROS_DOMAIN_ID=<ID> to avoid conflicts with other ROS 2 systems.
Always set this before launching:
```bash
export ROS_DOMAIN_ID=<ID>
```

Usage:
    # Basic simulation
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

    # Model inference mode (auto-detected)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true control_mode:=model_inference

    # Teleop mode (human teleoperation)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop record:=true

    # Teleop mode with episodic recording (episode-by-episode)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop record:=true record_mode:=episodic

    # MoveIt planning mode (auto-detected, with RViz)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true

    # MoveIt mode without RViz (headless)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true moveit_display:=false

    # Real hardware
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=false

    # Override auto-detection
    ros2 launch robot_config robot.launch.py control_mode:=model_inference with_inference:=true use_sim:=true

    # Distributed inference (two machines, set same ROS_DOMAIN_ID on both):
    #   Machine A (sim/robot): launch edge node only
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true control_mode:=model_inference execution_mode:=distributed
    #   Machine B (GPU): launch cloud inference node
    ros2 launch inference_service cloud_inference.launch.py policy_path:=/path/to/model device:=cuda

    # Distributed inference (single-machine testing):
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true control_mode:=model_inference execution_mode:=distributed cloud_local:=true

**Cleanup**: If you encounter "Controller already loaded" errors, run:
```bash
./scripts/cleanup_ros.sh
```

Launch Arguments:
    robot_config: Robot configuration name (default: test_cam)
    config_path: Optional full path to robot config file
    use_sim: Use simulation mode (default: false)
    auto_start_controllers: Automatically spawn controllers (default: true, set to false for debugging)
    control_mode: Override control mode from YAML (teleop, model_inference, or moveit_planning). If empty, uses default_control_mode from config file
    with_inference: Enable inference pipeline. If empty, auto-detects from control mode config
    cloud_local: In distributed mode, also launch cloud node locally (default: false)
    execution_mode: Override execution mode from YAML ('monolithic' or 'distributed'). If empty, uses YAML value
    with_moveit: Enable MoveIt motion planning. If empty, auto-detects from control mode name
    moveit_display: Launch RViz for MoveIt visualization (default: true, only used if MoveIt is enabled)
    record: Enable automatic rosbag recording (default: false, auto-discovers topics from config)
    record_mode: Recording mode - 'continuous' (default, all-in-one bag) or 'episodic' (triggered episode-by-episode, requires manual record_cli in separate terminal)
"""

import os
import yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

# Import utility functions
from robot_config.utils import resolve_ros_path, parse_bool
from robot_config.logger_utils import get_colored_logger

# Import node generators from launch_builders modules
from robot_config.launch_builders.control import generate_ros2_control_nodes
from robot_config.launch_builders.perception import generate_camera_nodes, generate_tf_nodes
from robot_config.launch_builders.sim_backend import get_sim_backend
from robot_config.launch_builders.execution import generate_execution_nodes
from robot_config.launch_builders.teleop import generate_teleop_nodes
from robot_config.launch_builders.recording import generate_recording_nodes
from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

logger = get_colored_logger("robot_config.launch")


def load_robot_config(robot_config_name, config_path_override=None):
    """Load robot configuration from YAML file.

    Args:
        robot_config_name: Robot configuration name
        config_path_override: Optional full path to config file

    Returns:
        Robot configuration dict
    """
    # Get package share directory
    try:
        robot_config_share = get_package_share_directory("robot_config")
    except:
        robot_config_share = str(Path(__file__).parent.parent)

    # Determine config file path
    if config_path_override:
        config_path = Path(config_path_override)
    else:
        config_path = Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml"

    logger.info(f"Loading config from: {config_path}")
    logger.info(f"Config exists: {config_path.exists()}")

    # Load YAML
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    robot_config = data.get("robot", {})
    logger.info(f"Loaded robot: {robot_config.get('name', 'UNKNOWN')}")
    logger.info(f"Peripherals: {len(robot_config.get('peripherals', []))}")

    return robot_config


def _start_actions_on_success(start_actions, success_message: str, failure_reason: str):
    """Run launch actions only when the target process exits successfully."""
    frozen_actions = tuple(start_actions)

    def _handler(event, _context):
        if event.returncode == 0:
            logger.info(success_message)
            return list(frozen_actions)

        logger.error(f"{failure_reason} (returncode={event.returncode})")
        return [EmitEvent(event=Shutdown(reason=failure_reason))]

    return _handler


def _resolve_controller_startup_timeout(robot_config: dict, use_sim: bool) -> float:
    """Resolve controller startup timeout from robot YAML."""
    configured_timeout = robot_config.get("controller_startup_timeout")
    if configured_timeout is None:
        return 120.0 if use_sim else 30.0

    timeout_value = configured_timeout
    if isinstance(configured_timeout, dict):
        profile_key = "sim" if use_sim else "hardware"
        timeout_value = configured_timeout.get(profile_key)
        if timeout_value is None:
            timeout_value = configured_timeout.get("default")

    if timeout_value is None:
        raise ValueError(
            "robot.controller_startup_timeout must define a value for the active launch profile."
        )

    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "robot.controller_startup_timeout must be a number or a mapping of numbers."
        ) from exc

    if timeout <= 0.0:
        raise ValueError("robot.controller_startup_timeout must be greater than zero.")

    return timeout


def _create_controller_ready_waiter(robot_config: dict, controller_names, use_sim: bool):
    """Create a probe that exits when the required controllers are active."""
    timeout = _resolve_controller_startup_timeout(robot_config, use_sim)
    return Node(
        package="robot_config",
        executable="wait_for_controllers",
        name="wait_for_active_controllers",
        parameters=[{"use_sim_time": use_sim}],
        arguments=[
            *controller_names,
            "--controller-manager",
            "controller_manager",
            "--timeout",
            str(timeout),
        ],
        output="screen",
    )


def launch_setup(context, *args, **kwargs):
    """Launch setup function that generates all nodes.

    This is the "orchestrator" that:
    1. Loads and normalizes all parameters
    2. Calls each builder module to generate nodes
    3. Returns the combined actions list

    Args:
        context: Launch context

    Returns:
        List of launch actions
    """
    actions = []
    controller_dependent_actions = []

    # ========== 1. Get and normalize launch parameters ==========
    robot_config_name = context.launch_configurations.get('robot_config', 'test_cam')
    config_path_override = context.launch_configurations.get('config_path', '')
    use_sim_str = context.launch_configurations.get('use_sim', 'false')
    auto_start_controllers = context.launch_configurations.get('auto_start_controllers', 'true')
    control_mode_override = context.launch_configurations.get('control_mode', '')

    # Normalize use_sim to boolean
    use_sim = parse_bool(use_sim_str, default=False)

    logger.info(f"========== Launch Parameters ==========")
    logger.info(f"robot_config: {robot_config_name}")
    logger.info(
        f"config_path: {config_path_override if config_path_override else '(none)'}"
    )
    logger.info(f"use_sim: {use_sim} (from '{use_sim_str}')")
    logger.info(f"auto_start_controllers: {auto_start_controllers}")
    logger.info(
        f"control_mode: {control_mode_override if control_mode_override else '(from config)'}"
    )

    # ========== 2. Load robot configuration ==========
    try:
        robot_config = load_robot_config(
            robot_config_name,
            config_path_override if config_path_override else None
        )
    except Exception as e:
        logger.error(f"loading config: {e}")
        raise

    # Store config path for downstream modules (e.g., recording)
    if config_path_override:
        robot_config['_config_path'] = config_path_override
    else:
        try:
            robot_config_share = get_package_share_directory("robot_config")
        except:
            robot_config_share = str(Path(__file__).parent.parent)
        robot_config['_config_path'] = str(Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml")

    # ========== 3. Apply control mode override ==========
    if control_mode_override:
        robot_config['default_control_mode'] = control_mode_override
    
    active_control_mode = robot_config.get('default_control_mode', 'model_inference')
    logger.info(f"Active control mode: {active_control_mode}")

    # Determine with_inference flag globally
    with_inference_str = context.launch_configurations.get('with_inference', '')
    if with_inference_str != '':
        with_inference = parse_bool(with_inference_str, default=False)
    else:
        control_mode_config = robot_config.get('control_modes', {}).get(active_control_mode, {})
        with_inference = control_mode_config.get('inference', {}).get('enabled', False)

    # Force disable inference in teleop mode if not explicitly overridden
    if active_control_mode == 'teleop' and with_inference_str == '':
        with_inference = False
        logger.info("Teleop mode: forcing with_inference=False")

    logger.info(f"Final with_inference={with_inference}")

    # Determine cloud_local flag for distributed mode
    cloud_local_str = context.launch_configurations.get('cloud_local', 'false')
    cloud_local = parse_bool(cloud_local_str, default=False)

    # CLI override for execution_mode (overrides YAML if provided)
    execution_mode_override = context.launch_configurations.get('execution_mode', '')
    if execution_mode_override:
        control_modes = robot_config.get("control_modes", {})
        mode_cfg = control_modes.get(active_control_mode, {})
        inf_cfg = mode_cfg.get("inference", {})
        inf_cfg["execution_mode"] = execution_mode_override
        logger.info(f"CLI override: execution_mode={execution_mode_override}")

    # ========== 4. Generate Control System Nodes ==========
    logger.info(f"========== Generating Control Nodes ==========")
    deferred_sim_spawners = []
    controller_names = []
    robot_description = {}
    try:
        control_nodes, controller_names, deferred_sim_spawners, robot_description = generate_ros2_control_nodes(
            robot_config, use_sim, auto_start_controllers
        )
        actions.extend(control_nodes)
        logger.info(f"Added {len(control_nodes)} control nodes")
    except Exception as e:
        logger.error(f"generating control nodes: {e}")
        raise

    controller_ready_waiter = None
    if parse_bool(auto_start_controllers, default=True) and controller_names:
        controller_ready_waiter = _create_controller_ready_waiter(
            robot_config,
            controller_names,
            use_sim,
        )

    # ========== 5. Generate Simulation Nodes (only in simulation mode) ==========
    gz_create_entity = None
    if use_sim:
        logger.info(f"========== Generating Simulation Nodes ==========")
        sim_platform = robot_config.get('simulation', {}).get('platform', 'gazebo')
        logger.info(f"Sim platform: {sim_platform}")
        try:
            sim_adapter = get_sim_backend(sim_platform)
            sim_nodes, gz_create_entity = sim_adapter.start_backend(robot_config)
            sim_nodes += sim_adapter.spawn_peripheral_bridges(
                robot_config.get("peripherals", [])
            )
            actions.extend(sim_nodes)
            logger.info(f"Added {len(sim_nodes)} simulation nodes ({sim_platform})")
        except NotImplementedError:
            logger.warning(
                f"sim platform '{sim_platform}' not implemented yet, "
                f"skipping simulation nodes (set simulation.platform: gazebo to use Gazebo)"
            )
        except Exception as e:
            logger.error(f"generating simulation nodes: {e}")
            raise

    if deferred_sim_spawners:
        startup_sequence = list(deferred_sim_spawners)
        if controller_ready_waiter is not None:
            startup_sequence.append(controller_ready_waiter)
        if use_sim and gz_create_entity is not None:
            logger.info(
                "Scheduling controller startup after ros_gz_sim create exits"
            )
            actions.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=gz_create_entity,
                        on_exit=_start_actions_on_success(
                            startup_sequence,
                            success_message="Robot entity created; starting controller startup sequence.",
                            failure_reason="Robot entity creation failed; aborting launch.",
                        ),
                    )
                )
            )
        else:
            actions.extend(startup_sequence)
    elif controller_ready_waiter is not None:
        actions.append(controller_ready_waiter)

    # ========== 6. Generate Perception Nodes ==========
    logger.info(f"========== Generating Perception Nodes ==========")
    try:
        # Camera nodes (Physical drivers)
        camera_nodes = generate_camera_nodes(robot_config, use_sim)
        actions.extend(camera_nodes)
        logger.info(f"Added {len(camera_nodes)} camera nodes")

        # Virtual camera relay nodes (Topic tools)
        from robot_config.launch_builders.perception import generate_virtual_camera_relays
        virtual_nodes = generate_virtual_camera_relays(robot_config)
        actions.extend(virtual_nodes)
        if virtual_nodes:
            logger.info(f"Added {len(virtual_nodes)} virtual camera relays")

        # Static TF publishers
        tf_nodes = generate_tf_nodes(robot_config, use_sim)
        actions.extend(tf_nodes)
        logger.info(f"Added {len(tf_nodes)} TF nodes")
    except Exception as e:
        logger.error(f"generating perception nodes: {e}")
        raise

    # ========== 7. Generate Teleop Nodes (if in teleop mode) ==========
    logger.info(f"========== Checking Teleop Mode ==========")
    try:
        # Check if teleop mode is enabled
        _teleop_modes = ('teleop',)
        if active_control_mode in _teleop_modes:
            logger.info(f"TELEOP MODE DETECTED ({active_control_mode})")

            # Check if teleoperation is configured
            teleop_config = robot_config.get('teleoperation', {})
            if not teleop_config.get('enabled', False):
                logger.info(
                    f"WARNING: Teleop mode requested but teleoperation config not found"
                )
            else:
                # Generate teleop nodes
                teleop_nodes = generate_teleop_nodes(robot_config, robot_description)

                if controller_ready_waiter is not None:
                    logger.info("Deferring teleop nodes until required controllers are active...")
                    controller_dependent_actions.extend(teleop_nodes)
                else:
                    logger.info(
                        "No controller readiness probe active, launching teleop immediately"
                    )
                    actions.extend(teleop_nodes)

                logger.info(f"Prepared {len(teleop_nodes)} teleop nodes")
        else:
            logger.info(f"Skipping teleop nodes (mode is {active_control_mode})")
    except Exception as e:
        logger.error(f"checking teleop mode: {e}")
        raise

    # ========== 8. Generate Voice ASR Nodes ==========
    logger.info(f"========== Checking Voice ASR ==========")
    try:
        voice_asr_nodes = generate_voice_asr_nodes(robot_config)
        actions.extend(voice_asr_nodes)
        if voice_asr_nodes:
            logger.info(f"Added {len(voice_asr_nodes)} voice ASR node(s)")
    except Exception as e:
        logger.error(f"generating voice ASR nodes: {e}")
        raise

    # ========== 9. Generate Execution Nodes ==========
    logger.info(f"========== Generating Execution Nodes ==========")
    try:
        if with_inference:
            execution_nodes = generate_execution_nodes(
                robot_config, active_control_mode, use_sim, cloud_local=cloud_local)
            if controller_ready_waiter is not None:
                logger.info("Deferring execution nodes until required controllers are active...")
                controller_dependent_actions.extend(execution_nodes)
            else:
                actions.extend(execution_nodes)
            logger.info(f"Prepared {len(execution_nodes)} execution nodes")
        else:
            logger.info(f"Skipping execution nodes")
    except Exception as e:
        logger.error(f"generating execution nodes: {e}")
        raise

    # ========== 10. Generate MoveIt Nodes ==========
    try:
        # Determine with_moveit flag
        with_moveit_str = context.launch_configurations.get('with_moveit', '')
        moveit_display = parse_bool(context.launch_configurations.get('moveit_display', 'true'), default=True)

        if with_moveit_str != '':
            with_moveit = parse_bool(with_moveit_str, default=False)
        else:
            with_moveit = 'moveit' in active_control_mode.lower()

        logger.info(f"with_moveit={with_moveit}")

        if with_moveit:
            from robot_config.launch_builders.moveit import generate_moveit_nodes
            moveit_nodes = generate_moveit_nodes(robot_config, active_control_mode, use_sim, moveit_display)

            if controller_ready_waiter is not None:
                logger.info("Deferring MoveIt nodes until required controllers are active...")
                controller_dependent_actions.extend(moveit_nodes)
            else:
                logger.info(
                    "No controller readiness probe active, launching MoveIt immediately"
                )
                actions.extend(moveit_nodes)
        else:
            logger.info(f"Skipping MoveIt nodes")
    except Exception as e:
        logger.error(f"generating MoveIt nodes: {e}")
        logger.info(f"Continuing without MoveIt...")

    # ========== 11. Automatic Recording (if record:=true) ==========
    try:
        record_str = context.launch_configurations.get('record', 'false')
        record_enabled = parse_bool(record_str, default=False)

        if record_enabled:
            # Get recording mode (continuous or episodic)
            record_mode = context.launch_configurations.get('record_mode', 'continuous')

            logger.info(
                f"========== Setting up Recording (mode: {record_mode}) =========="
            )

            # Generate recording nodes using the recording builder
            recording_nodes = generate_recording_nodes(robot_config, active_control_mode, record_mode)
            actions.extend(recording_nodes)
            logger.info(f"Added {len(recording_nodes)} recording node(s)")
        else:
            logger.info(f"Recording disabled (record:={record_str})")
    except Exception as e:
        logger.error(f"setting up recording: {e}")
        logger.info(f"Continuing without recording...")

    if controller_dependent_actions:
        if controller_ready_waiter is not None:
            logger.info(
                f"Controller readiness barrier armed for "
                f"{len(controller_dependent_actions)} control-dependent action(s)"
            )
            actions.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=controller_ready_waiter,
                        on_exit=_start_actions_on_success(
                            controller_dependent_actions,
                            success_message="Required controllers are active; starting control-dependent nodes.",
                            failure_reason="Controller readiness probe failed; aborting launch.",
                        ),
                    )
                )
            )
        else:
            actions.extend(controller_dependent_actions)

    logger.info(f"========== Total nodes to launch: {len(actions)} ==========")

    return actions


def generate_launch_description():
    """Generate launch description for robot system."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_config",
            default_value="so101_single_arm",
            description="Robot configuration name (without .yaml extension)",
        ),
        DeclareLaunchArgument(
            "config_path",
            default_value="",
            description="Optional: Full path to robot config file (overrides robot_config)",
        ),
        DeclareLaunchArgument(
            "use_sim",
            default_value="false",
            description="Use simulation mode (skip camera nodes)",
        ),
        DeclareLaunchArgument(
            "auto_start_controllers",
            default_value="true",
            description="Automatically spawn controllers (set to false for debugging)",
        ),
        DeclareLaunchArgument(
            "control_mode",
            default_value="",
            description="Override control mode from YAML (teleop, model_inference, or moveit_planning). If empty, uses default_control_mode from config file",
        ),
        DeclareLaunchArgument(
            "with_inference",
            default_value="",
            description="Enable full execution pipeline (inference + dispatcher). If empty, auto-detects from control mode config",
        ),
        DeclareLaunchArgument(
            "cloud_local",
            default_value="false",
            description="In distributed mode, also launch the cloud inference node locally (for single-machine testing). Default: false (cloud node runs on separate GPU machine)",
        ),
        DeclareLaunchArgument(
            "execution_mode",
            default_value="monolithic",
            description="Override inference execution mode from YAML ('monolithic' or 'distributed'). If empty, uses value from robot config YAML.",
        ),
        DeclareLaunchArgument(
            "with_moveit",
            default_value="",
            description="Enable MoveIt motion planning. If empty, auto-detects from control mode config",
        ),
        DeclareLaunchArgument(
            "moveit_display",
            default_value="true",
            description="Launch RViz for MoveIt visualization (only used if MoveIt is enabled)",
        ),
        DeclareLaunchArgument(
            "record",
            default_value="false",
            description="Enable automatic rosbag recording (auto-discovers topics from config)",
        ),
        DeclareLaunchArgument(
            "record_mode",
            default_value="continuous",
            description="Recording mode: 'continuous' (all-in-one bag) or 'episodic' (triggered episode-by-episode via episode_recorder)",
        ),
        OpaqueFunction(function=launch_setup),
    ])
