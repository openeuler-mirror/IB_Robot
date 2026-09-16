"""Control system launch builders.

This module handles:
- ros2_control node generation
- Controller spawner creation
- Joint configuration validation (delegates to utils.py)

URDF building (xacro processing + camera injection) is in description.py.
"""

import math
import os
import tempfile
from pathlib import Path

import yaml
from launch.actions import EmitEvent
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.launch_builders.description import generate_robot_description
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import (
    parse_bool,
    resolve_calibration_paths_from_config,
    resolve_ros_path,
    validate_joint_config,
)

logger = get_colored_logger("robot_config.control")


def _shutdown_when_process_exits(node_name: str):
    """Stop the complete launch when a required hardware process exits."""
    return EmitEvent(event=Shutdown(reason=f"required process {node_name!r} exited"))


def runtime_component_enabled(config: dict, *, use_sim: bool = False, control_mode: str | None = None) -> bool:
    """Return whether one external runtime component belongs in this launch."""
    if not parse_bool(config.get("enabled", True), default=True):
        return False

    active_control_modes = config.get("active_control_modes")
    if active_control_modes is not None:
        if not isinstance(active_control_modes, list | tuple) or not all(
            isinstance(mode, str) and mode.strip() for mode in active_control_modes
        ):
            raise ValueError("active_control_modes must be a list of non-empty strings")
        if control_mode is not None and control_mode not in active_control_modes:
            return False

    # Real external devices must never be pulled into a simulation launch merely
    # because the physical robot configuration declares them. Explicit mock
    # components remain available for simulation and offline validation.
    return not use_sim or parse_bool(config.get("mock", False), default=False)


def validate_runtime_resources(
    robot_config: dict,
    *,
    use_sim: bool = False,
    control_mode: str | None = None,
) -> None:
    """Reject duplicate real serial/exclusive resources before starting any node."""
    owners: dict[str, list[str]] = {}

    def add(owner: str, resource, *, resolve_path: bool = False) -> None:
        if resource is None:
            return
        value = str(resource).strip()
        if not value:
            return
        if resolve_path:
            value = str(resolve_ros_path(value) or "").strip()
            if not value:
                raise ValueError(f"{owner} resolved to an empty hardware resource")
            if value.startswith("/dev/"):
                value = os.path.realpath(value)
        resource_owners = owners.setdefault(value, [])
        if owner not in resource_owners:
            resource_owners.append(owner)

    def add_exclusive(owner: str, config: dict, *, default=None) -> None:
        resources = config.get("exclusive_resources", default or [])
        if isinstance(resources, str):
            resources = [resources]
        if not isinstance(resources, list | tuple):
            raise ValueError(f"{owner} exclusive_resources must be a list of strings")
        for resource in resources:
            add(owner, resource, resolve_path=str(resource).strip().startswith("/dev/"))

    if not use_sim:
        ros2_control = robot_config.get("ros2_control", {}) or {}
        if isinstance(ros2_control, dict):
            add("ros2_control", ros2_control.get("port"), resolve_path=True)

    teleoperation = robot_config.get("teleoperation", {}) or {}
    if (
        isinstance(teleoperation, dict)
        and parse_bool(teleoperation.get("enabled", False), default=False)
        and control_mode in (None, "teleop")
    ):
        devices = [device for device in teleoperation.get("devices", []) if isinstance(device, dict)]
        active_names = teleoperation.get("active_devices")
        if active_names:
            active_names = {str(name) for name in active_names}
            devices = [device for device in devices if str(device.get("name", "")) in active_names]
        else:
            active_name = str(teleoperation.get("active_device", ""))
            devices = [device for device in devices if str(device.get("name", "")) == active_name]
        for device in devices:
            owner = f"teleoperation.{device.get('name', device.get('type', 'device'))}"
            if str(device.get("port", "")).strip().startswith("/dev/"):
                add(owner, device.get("port"), resolve_path=True)
            add_exclusive(owner, device)

    for name, config in _named_configs(robot_config.get("auxiliary_actuators", {}) or {}, "auxiliary_actuators"):
        if not runtime_component_enabled(config, use_sim=use_sim, control_mode=control_mode) or parse_bool(
            config.get("mock", False), default=False
        ):
            continue
        owner = f"auxiliary_actuators.{name}"
        add(owner, config.get("port"), resolve_path=True)
        add_exclusive(owner, config)

    for name, config in _named_configs(robot_config.get("hand_sources", {}) or {}, "hand_sources"):
        if not runtime_component_enabled(config, use_sim=use_sim, control_mode=control_mode) or parse_bool(
            config.get("mock", False), default=False
        ):
            continue
        owner = f"hand_sources.{name}"
        default_resources = ["mhandpro_sdk"] if str(config.get("type", "")).strip() == "mhandpro" else []
        add_exclusive(owner, config, default=default_resources)

    conflicts = {resource: names for resource, names in owners.items() if len(names) > 1}
    if conflicts:
        details = "; ".join(f"{resource}: {', '.join(names)}" for resource, names in conflicts.items())
        raise ValueError(f"real hardware resources are used more than once: {details}")


def _named_configs(raw_configs, label: str) -> list[tuple[str, dict]]:
    if isinstance(raw_configs, dict):
        configs = []
        for name, value in raw_configs.items():
            if not isinstance(value, dict):
                raise ValueError(f"{label}.{name} must be an object")
            configs.append((str(name), value))
        return configs
    if isinstance(raw_configs, list):
        configs = []
        for index, value in enumerate(raw_configs):
            if not isinstance(value, dict):
                raise ValueError(f"{label}[{index}] must be an object")
            name = str(value.get("name", "")).strip()
            if not name:
                raise ValueError(f"{label}[{index}].name must be non-empty")
            configs.append((name, value))
        return configs
    raise ValueError(f"{label} must be an object or list")


def _resolve_command_limits(
    robot_config: dict, joint_names: list[str], *, control_mode: str | None
) -> tuple[list, list]:
    """Resolve actuator command limits from the same safety SSOT used by teleop."""
    root_safety = robot_config.get("safety", {}) or {}
    teleoperation = robot_config.get("teleoperation", {}) or {}
    if not isinstance(root_safety, dict) or not isinstance(teleoperation, dict):
        raise ValueError("Robot safety and teleoperation configuration must be objects")
    teleop_safety = teleoperation.get("safety", root_safety) or {}
    safety = teleop_safety if control_mode in (None, "teleop") else root_safety
    if not isinstance(safety, dict):
        raise ValueError("Actuator safety configuration must be an object")
    joint_limits = safety.get("joint_limits", {}) or {}
    if not isinstance(joint_limits, dict):
        raise ValueError("Actuator safety joint_limits must be an object")

    lower_limits = []
    upper_limits = []
    for joint_name in joint_names:
        limits = joint_limits.get(joint_name)
        if limits is None:
            if joint_limits:
                raise ValueError(f"Missing safety limits for actuator joint {joint_name!r}")
            return [], []
        if not isinstance(limits, dict) or "min" not in limits or "max" not in limits:
            raise ValueError(f"Missing min/max safety limits for actuator joint {joint_name!r}")
        lower = float(limits["min"])
        upper = float(limits["max"])
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(f"Invalid safety limits for actuator joint {joint_name!r}")
        lower_limits.append(lower)
        upper_limits.append(upper)
    return lower_limits, upper_limits


def generate_auxiliary_actuator_nodes(
    robot_config: dict,
    *,
    use_sim: bool = False,
    control_mode: str | None = None,
) -> list[Node]:
    """Generate target-hand-independent, non-ros2_control actuator nodes."""
    raw_configs = robot_config.get("auxiliary_actuators", {}) or {}
    if isinstance(raw_configs, dict):
        actuator_configs = []
        for name, value in raw_configs.items():
            if not isinstance(value, dict):
                raise ValueError(f"auxiliary_actuators.{name} must be an object")
            actuator_configs.append({"name": name, **value})
    elif isinstance(raw_configs, list):
        actuator_configs = raw_configs
    else:
        raise ValueError("auxiliary_actuators must be an object or list")

    nodes: list[Node] = []
    names: set[str] = set()
    command_topics: set[str] = set()
    state_topics: set[str] = set()
    real_ports: set[str] = set()
    for index, config in enumerate(actuator_configs):
        if not isinstance(config, dict):
            raise ValueError(f"auxiliary_actuators[{index}] must be an object")
        if not runtime_component_enabled(config, use_sim=use_sim, control_mode=control_mode):
            continue

        name = str(config.get("name", "")).strip()
        actuator_type = str(config.get("type", "")).strip()
        if not name or name in names:
            raise ValueError(f"auxiliary_actuators[{index}].name must be non-empty and unique")
        driver = config.get("driver", {}) or {}
        if not isinstance(driver, dict):
            raise ValueError(f"Auxiliary actuator {name!r} driver must be an object")
        package = str(driver.get("package", "aero_hand_hardware" if actuator_type == "aero_hand" else "")).strip()
        executable = str(driver.get("executable", "aero_hand_node" if actuator_type == "aero_hand" else "")).strip()
        if not actuator_type or not package or not executable:
            raise ValueError(f"Auxiliary actuator {name!r} requires type plus driver.package and driver.executable")

        mock = parse_bool(config.get("mock", False), default=False)
        port_raw = str(config.get("port", "")).strip()
        port = str(resolve_ros_path(port_raw) or "").strip()
        joint_names = [str(joint) for joint in config.get("joint_names", [])]
        command_topic = str(config.get("command_topic", "")).strip()
        state_topic = str(config.get("joint_state_topic", "")).strip()
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError(f"Auxiliary actuator {name!r} must define non-empty unique joint_names")
        if not command_topic or command_topic in command_topics:
            raise ValueError(f"Auxiliary actuator {name!r} command_topic must be non-empty and unique")
        if not state_topic or state_topic in state_topics:
            raise ValueError(f"Auxiliary actuator {name!r} joint_state_topic must be non-empty and unique")
        exclusive_resources = list(
            dict.fromkeys(
                str(resolve_ros_path(str(value).strip()) or "").strip()
                for value in config.get("exclusive_resources", [])
            )
        )
        if any(not value for value in exclusive_resources):
            raise ValueError(f"Auxiliary actuator {name!r} exclusive_resources must be non-empty strings")
        if not mock and port and port not in exclusive_resources:
            exclusive_resources.append(port)
        if not mock:
            if actuator_type == "aero_hand" and not port:
                raise ValueError(f"Auxiliary actuator {name!r} requires port when mock is false")
            for resource in exclusive_resources:
                if resource in real_ports:
                    label = "serial port" if resource == port else "exclusive resource"
                    raise ValueError(f"Auxiliary actuator {label} is used more than once: {resource}")
                real_ports.add(resource)

        names.add(name)
        command_topics.add(command_topic)
        state_topics.add(state_topic)
        parameters = dict(config.get("parameters", {}) or {})
        parameters.update(
            {
                "mock": mock,
                "joint_names": joint_names,
                "command_topic": command_topic,
                "joint_state_topic": state_topic,
            }
        )
        if port:
            parameters["port"] = port
        if actuator_type == "aero_hand":
            estop_topic = str(config.get("estop_topic", "/emergency_stop")).strip()
            estop_behavior = str(config.get("estop_behavior", "hold")).strip().lower()
            raw_safe_pose = config.get("safe_pose", [])
            if not isinstance(raw_safe_pose, list):
                raise ValueError(f"Auxiliary actuator {name!r} safe_pose must be a list")
            safe_pose = [float(value) for value in raw_safe_pose]
            if not estop_topic:
                raise ValueError(f"Auxiliary actuator {name!r} estop_topic must be non-empty")
            if estop_behavior not in ("hold", "safe_pose"):
                raise ValueError(f"Auxiliary actuator {name!r} estop_behavior must be hold or safe_pose")
            if estop_behavior == "safe_pose" and (
                len(safe_pose) != 7 or not all(math.isfinite(value) for value in safe_pose)
            ):
                raise ValueError(f"Auxiliary actuator {name!r} safe_pose must contain seven finite radians")
            command_lower_limits, command_upper_limits = _resolve_command_limits(
                robot_config,
                joint_names,
                control_mode=control_mode,
            )
            if not mock and not command_lower_limits:
                raise ValueError(f"Auxiliary actuator {name!r} requires safety joint_limits for every hardware joint")
            parameters.update(
                {
                    "baudrate": int(config.get("baudrate", 921600)),
                    "command_frequency": float(config.get("command_frequency", 50.0)),
                    "state_frequency": float(config.get("state_frequency", 20.0)),
                    "command_timeout": float(config.get("command_timeout", 0.25)),
                    "estop_topic": estop_topic,
                    "estop_behavior": estop_behavior,
                }
            )
            if command_lower_limits:
                parameters["command_lower_limits"] = command_lower_limits
                parameters["command_upper_limits"] = command_upper_limits
            # ROS 2 cannot infer a type for an empty array parameter and
            # launch_ros converts [] to an invalid empty tuple. The node's
            # declared default is already [] for hold behavior; only pass a
            # configured safe pose when it contains values.
            if safe_pose:
                parameters["safe_pose"] = safe_pose
        nodes.append(
            Node(
                package=package,
                executable=executable,
                name=name,
                output="screen",
                parameters=[parameters],
                on_exit=_shutdown_when_process_exits(name) if actuator_type == "aero_hand" else None,
            )
        )
    return nodes


def generate_controller_spawners(
    controller_names,
    use_sim=True,
    controller_manager_name="controller_manager",
    controller_manager_timeout=None,
    inactive_controller_names=(),
):
    """Generate one timeout-aware controller group spawner.

    Args:
        controller_names: Controller names to load, configure, and activate
        use_sim: Simulation mode (affects timeout and use_sim_time)
        controller_manager_name: Name of controller manager service
        controller_manager_timeout: Timeout for manager discovery and service calls
        inactive_controller_names: Controller names to load/configure but leave inactive

    Returns:
        Empty list or a single Node action for atomic controller group activation
    """
    is_sim = parse_bool(use_sim, default=True)

    controller_names = list(controller_names)
    inactive_controller_names = list(inactive_controller_names)
    if not controller_names and not inactive_controller_names:
        return []
    if len(set(controller_names)) != len(controller_names):
        raise ValueError("Active controller names must be unique")
    if len(set(inactive_controller_names)) != len(inactive_controller_names):
        raise ValueError("Inactive controller names must be unique")
    overlap = set(controller_names) & set(inactive_controller_names)
    if overlap:
        raise ValueError(f"Controllers cannot be both active and inactive: {sorted(overlap)}")

    timeout = float(controller_manager_timeout) if controller_manager_timeout is not None else (60 if is_sim else 10)
    if timeout <= 0.0:
        raise ValueError("controller_manager_timeout must be greater than zero")
    service_call_timeout = min(timeout, 10.0)
    switch_timeout = timeout
    arguments = [
        *controller_names,
        "--controller-manager",
        controller_manager_name,
        "--controller-manager-timeout",
        str(timeout),
        "--service-call-timeout",
        str(service_call_timeout),
        "--switch-timeout",
        str(switch_timeout),
    ]
    for controller_name in inactive_controller_names:
        arguments.extend(["--inactive-controller", controller_name])

    return [
        Node(
            package="robot_config",
            executable="controller_spawner",
            name="spawner_controller_group",
            parameters=[{"use_sim_time": is_sim}],
            arguments=arguments,
            output="screen",
        )
    ]


def _calibration_tool_command(robot_config: dict, arm: str, port: str) -> str:
    """Build the calibration tool command from the robot configuration.

    The command is derived from ``robot.calibration.tool_command`` (a format
    string with ``{arm}`` and ``{port}`` placeholders) instead of hardcoding
    a specific robot package. When the key is absent, the proven default
    tool is used.
    """
    template = (robot_config.get("calibration") or {}).get(
        "tool_command"
    ) or "ros2 run so101_hardware calibrate_arm --arm {arm} --port {port}"
    return template.format(arm=arm, port=port)


def generate_ros2_control_nodes(
    robot_config,
    use_sim,
    auto_start_controllers="true",
    controller_startup_timeout=None,
):
    """Generate ros2_control nodes from configuration.

    Args:
        robot_config: Robot configuration dict
        use_sim: Simulation mode flag (string or bool)
        auto_start_controllers: Whether to automatically start controllers (string or bool)
        controller_startup_timeout: Timeout used by controller-manager spawners

    Returns:
        Tuple: (nodes, controller_names, deferred_spawners, robot_description)
        Controller spawners are returned in ``deferred_spawners`` so launch can
        gate them on hardware or simulator readiness.
    """
    is_sim = parse_bool(use_sim, default=False)
    is_auto_start = parse_bool(auto_start_controllers, default=True)

    nodes = []
    deferred_spawners = []
    ros2_control_config = robot_config.get("ros2_control")

    if not ros2_control_config:
        logger.warning("No ros2_control configuration found")
        return nodes, [], deferred_spawners, {}

    logger.info("Creating ros2_control nodes")

    # Pre-flight check: calibration files must exist for real hardware.
    if not is_sim:
        try:
            calibration_paths = resolve_calibration_paths_from_config(robot_config)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid ros2_control calibration configuration: {exc}") from exc
        if not calibration_paths:
            logger.warning(
                "No calibration files configured for real hardware mode; calibrated joint conversion is unavailable"
            )
        for calib_file_resolved in calibration_paths:
            if not Path(calib_file_resolved).exists():
                logger.error("Calibration file not found!")
                logger.error(f"  Resolved path: {calib_file_resolved}")
                logger.error(f"  HOME=$HOME -> {os.environ.get('HOME', '(unset)')}")
                logger.error("")
                logger.error("  Please run calibration first:")
                calib_port = ros2_control_config.get("port", "/dev/ttyACM0")
                calib_command = _calibration_tool_command(robot_config, "follower", calib_port)
                logger.error(f"    {calib_command}")
                raise RuntimeError(f"Calibration file not found: {calib_file_resolved}. Run: {calib_command}")

    # Validate joint configuration
    validate_joint_config(robot_config)

    # Build URDF (xacro processing + camera injection) via description layer
    _desc_result = generate_robot_description(robot_config, use_sim)
    if _desc_result is None:
        return nodes, [], deferred_spawners, {}

    robot_description_str, robot_description = _desc_result

    # Robot State Publisher
    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        )
    )

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
            if is_sim and mode_config.get("sim_controllers") is not None:
                controller_names = mode_config.get("sim_controllers", [])
                inactive_controller_names = mode_config.get("sim_inactive_controllers", [])
            elif not is_sim and mode_config.get("hardware_controllers") is not None:
                controller_names = mode_config.get("hardware_controllers", [])
                inactive_controller_names = mode_config.get("hardware_inactive_controllers", [])
            else:
                controller_names = mode_config.get("controllers", [])
                inactive_controller_names = mode_config.get("inactive_controllers", [])
            mode_description = mode_config.get("description", "No description")
            logger.info(f"Using control mode: {control_mode_name}")
            logger.info(f"  Description: {mode_description}")
            logger.info(f"  Controllers: {controller_names}")
            logger.info(f"  Inactive controllers: {inactive_controller_names}")
        else:
            controller_names = []
            inactive_controller_names = []
    else:
        controller_names = ros2_control_config.get("controllers", [])
        inactive_controller_names = []

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
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
                prefix="cm_robot_desc_",
            ) as cm_params_file:
                yaml.dump(
                    {
                        "controller_manager": {
                            "ros__parameters": {
                                "robot_description": robot_description_str,
                            }
                        }
                    },
                    cm_params_file,
                    default_flow_style=False,
                )
            logger.info(f"Controller manager params: {cm_params_file.name}")

            nodes.append(
                Node(
                    package="controller_manager",
                    executable="ros2_control_node",
                    parameters=[cm_params_file.name, controllers_config],
                    remappings=[
                        ("~/robot_description", "/robot_description"),
                    ],
                    output="screen",
                )
            )

            if is_auto_start and (controller_names or inactive_controller_names):
                deferred_spawners = generate_controller_spawners(
                    controller_names,
                    use_sim=False,
                    controller_manager_timeout=controller_startup_timeout,
                    inactive_controller_names=inactive_controller_names,
                )
                logger.info(f"Deferring {len(deferred_spawners)} controller spawners until hardware is active")
    else:
        # Simulation mode
        # gz_ros2_control plugin provides controller_manager, but spawners
        # must wait until the Gazebo entity is fully created and the plugin
        # has initialized the hardware interface.
        logger.info("Simulation mode: Gazebo provides controller_manager")
        logger.info(f"Controllers to spawn (deferred until after gz spawn): {controller_names}")

        if is_auto_start and (controller_names or inactive_controller_names):
            deferred_spawners = generate_controller_spawners(
                controller_names,
                use_sim=True,
                controller_manager_timeout=controller_startup_timeout,
                inactive_controller_names=inactive_controller_names,
            )
            logger.info(f"Deferring {len(deferred_spawners)} controller spawners (handled by caller)")

    return nodes, controller_names, deferred_spawners, robot_description
