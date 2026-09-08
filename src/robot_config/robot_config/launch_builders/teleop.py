"""Teleoperation node generation for robot_config.

This module provides utilities to generate teleoperation nodes
for integration with the robot_config launch system.
"""

import json
import math
import os
import re
from pathlib import Path

from launch.actions import EmitEvent
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import (
    prepare_lerobot_env,
    resolve_calibration_source_specs_from_config,
    resolve_ros_path,
)
from robot_teleop.teleop_groups import resolve_target_publish_groups

logger = get_colored_logger("robot_config.teleop")

# Opt-in flag that derives follower_calib_file from the ros2_control calibration
# source, so the follower calibration path is configured in exactly one place.
_MAP_GRIPPER_FLAG = "map_gripper_to_follower_stroke"


def _shutdown_when_process_exits(node_name: str):
    """Stop the complete launch when the teleoperation node exits."""
    return EmitEvent(event=Shutdown(reason=f"required process {node_name!r} exited"))


_PLACO_ENDPOINT_DEFAULTS = {
    "linear_cmd_topic": "/so101_placo_servo_node/linear_cmd_base",
    "angular_cmd_topic": "/so101_placo_servo_node/angular_cmd_base",
    "pose_cmd_topic": "/so101_placo_servo_node/pose_cmd_base",
    "start_service": "/so101_placo_servo_node/start",
    "stop_service": "/so101_placo_servo_node/stop",
    "home_action": "/so101_placo_servo_node/return_home",
    "command_lease_topic": "/so101_placo_servo_node/command_lease",
    "estop_topic": "/emergency_stop",
    "command_out_topic": "/arm_position_controller/commands",
}


def _is_finite_number(value: object) -> bool:
    """Return whether a YAML value is a finite real number, excluding booleans."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def generate_teleop_nodes(robot_config: dict, robot_description_dict: dict = None) -> list[Node]:
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
    teleop_config = robot_config.get("teleoperation", {})
    if not teleop_config.get("enabled", False):
        logger.info("Teleoperation not enabled, skipping")
        return nodes

    validation_errors = validate_teleop_config(
        teleop_config,
        robot_config.get("auxiliary_actuators", {}),
        robot_config.get("joints", {}),
        robot_config.get("safety", {}),
    )
    if validation_errors:
        raise ConfigError("Invalid teleoperation configuration: " + "; ".join(validation_errors))

    active_device_names = teleop_config.get("active_devices")
    if active_device_names:
        device_configs = {device.get("name"): device for device in teleop_config.get("devices", [])}
        for active_device_name in active_device_names:
            device_config = device_configs.get(active_device_name)
            if not device_config:
                logger.error(f"Active device '{active_device_name}' not found")
                continue
            nodes.extend(_generate_device_nodes(robot_config, device_config, robot_description_dict))
        return nodes

    active_device_name = teleop_config.get("active_device", "")
    if not active_device_name:
        logger.warning("No active_device specified")
        return nodes

    # Find device config
    device_config = None
    for device in teleop_config.get("devices", []):
        if device.get("name") == active_device_name:
            device_config = device
            break

    if not device_config:
        logger.error(f"Active device '{active_device_name}' not found")
        return nodes

    return _generate_device_nodes(robot_config, device_config, robot_description_dict)


def _resolve_follower_calib_file(robot_config: dict, device_config: dict, device_type: str) -> str | None:
    """Resolve the follower calibration that maps the leader gripper into radians.

    The leader reports its gripper as a 0~1 percentage while the follower runs a
    radian position controller, so the follower's calibrated stroke is needed to
    convert between them.  ``map_gripper_to_follower_stroke`` derives that file
    from the single ``ros2_control`` calibration source, keeping the path in
    exactly one place: re-calibrating the follower then needs no teleoperation
    edit and cannot silently go out of sync.  An explicit ``follower_calib_file``
    remains supported for layouts the derivation cannot express and is expanded
    the same way.

    Returns:
        Resolved absolute path, or None when neither key is configured (the
        device then keeps the legacy 0~1 gripper behavior).
    """
    derive = bool(device_config.get(_MAP_GRIPPER_FLAG, False))
    explicit = str(device_config.get("follower_calib_file", "") or "").strip()
    device_name = device_config.get("name", "")

    if not derive and not explicit:
        return None

    if device_type != "leader_arm":
        raise ConfigError(
            f"Device '{device_name}': {_MAP_GRIPPER_FLAG}/follower_calib_file only apply to "
            f"leader_arm devices, but this device is '{device_type}'"
        )

    if derive and explicit:
        raise ConfigError(
            f"Device '{device_name}': {_MAP_GRIPPER_FLAG} derives the follower calibration from "
            "ros2_control, so it cannot be combined with an explicit follower_calib_file; "
            "keep one of the two"
        )

    if derive:
        try:
            specs = resolve_calibration_source_specs_from_config(robot_config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Device '{device_name}': cannot derive the follower calibration from ros2_control: {exc}"
            ) from exc
        if len(specs) != 1:
            namespaces = ", ".join(spec.namespace for spec in specs) or "(none)"
            raise ConfigError(
                f"Device '{device_name}': {_MAP_GRIPPER_FLAG} needs exactly one ros2_control "
                f"calibration source, found: {namespaces}; set follower_calib_file explicitly instead"
            )
        resolved = specs[0].resolved_path
    else:
        resolved = resolve_ros_path(explicit)

    # Mirror the calib_file pre-flight check: a missing file used to degrade
    # silently to the 0~1 command, i.e. the half-closing gripper this mapping
    # exists to fix. Fail at launch instead.
    if not Path(resolved).exists():
        logger.error("=" * 60)
        logger.error("Follower calibration file not found!")
        logger.error(f"  Resolved path: {resolved}")
        logger.error(f"  Source:        {'ros2_control (derived)' if derive else 'follower_calib_file'}")
        logger.error(f"  HOME=$HOME -> {os.environ.get('HOME', '(unset)')}")
        logger.error("")
        logger.error("  Please run calibration first:")
        follower_port = (robot_config.get("ros2_control", {}) or {}).get("port", "/dev/ttyACM0")
        logger.error(f"    ros2 run so101_hardware calibrate_arm --arm follower --port {follower_port}")
        logger.error("=" * 60)
        raise RuntimeError(
            f"Follower calibration file not found: {resolved}. "
            f"Run: ros2 run so101_hardware calibrate_arm --arm follower --port {follower_port}"
        )

    return resolved


def _generate_device_nodes(robot_config: dict, device_config: dict, robot_description_dict: dict = None) -> list[Node]:
    nodes = []
    teleop_config = robot_config.get("teleoperation", {})
    device_type = device_config.get("type", "")

    # Get joint limits from safety config
    safety_config = teleop_config.get("safety", robot_config.get("safety", {}))
    joint_limits = safety_config.get("joint_limits", {})

    # Get joint names from robot config
    joints_config = robot_config.get("joints", {})
    target_config = device_config.get("target", {}) or {}
    publish_groups = resolve_target_publish_groups(
        target_config,
        joints_config,
        robot_config.get("auxiliary_actuators", {}),
    )
    groups_by_name = {group.name: group for group in publish_groups}
    arm_joint_names = list(groups_by_name["arm"].joint_names) if "arm" in groups_by_name else []
    gripper_joint_names = list(groups_by_name["gripper"].joint_names) if "gripper" in groups_by_name else []

    # Build device config for node parameter
    device_param = {
        "type": device_config.get("type", ""),
        "name": device_config.get("name", ""),
    }

    # Add optional device parameters
    if "port" in device_config:
        device_param["port"] = device_config["port"]
    if "calib_file" in device_config:
        # Expand environment variables in calib_file path
        calib_file_raw = device_config["calib_file"]
        calib_file_expanded = resolve_ros_path(device_config["calib_file"])
        logger.debug(f"calib_file_raw: {calib_file_raw}")
        logger.debug(f"calib_file_expanded: {calib_file_expanded}")
        device_param["calib_file"] = calib_file_expanded
        if not Path(calib_file_expanded).exists():
            logger.error("=" * 60)
            logger.error(f"{device_type} calibration file not found!")
            logger.error(f"  Resolved path: {calib_file_expanded}")
            logger.error(f"  Raw path:      {calib_file_raw}")
            logger.error(f"  HOME=$HOME -> {os.environ.get('HOME', '(unset)')}")
            logger.error("")
            logger.error("  Please run calibration first:")
            if device_type == "hand_retarget":
                side = device_config.get("side", "right")
                calibration_command = (
                    f'ros2 run robot_teleop calibrate_glove --side {side} --lib-path "$MHANDPRO_SDK_LIB"'
                )
            else:
                calib_port = device_config.get("port", "/dev/ttyACM0")
                calibration_command = "ros2 run so101_hardware calibrate_arm --arm leader --port " + calib_port
            logger.error("    " + calibration_command)
            logger.error("=" * 60)
            raise RuntimeError(f"Calibration file not found: {calib_file_expanded}. Run: {calibration_command}")
    follower_calib_file = _resolve_follower_calib_file(robot_config, device_config, device_type)
    if follower_calib_file:
        device_param["follower_calib_file"] = follower_calib_file
    if "joint_mapping" in device_config:
        device_param["joint_mapping"] = device_config["joint_mapping"]

    device_param["arm_joint_names"] = arm_joint_names
    device_param["gripper_joint_names"] = gripper_joint_names
    device_param["joint_names"] = [joint for group in publish_groups for joint in group.joint_names]
    device_param["publish_groups"] = [group.to_dict() for group in publish_groups]

    # Add joint limits for proper scaling
    if joint_limits:
        device_param["joint_limits"] = joint_limits

    # Add any extra device-specific parameters
    known_keys = {
        "name",
        "type",
        "port",
        "lib_path",
        "calib_file",
        # Handled above; leaving these out would let the passthrough loop below
        # overwrite the resolved path with the raw, unexpanded config value.
        "follower_calib_file",
        _MAP_GRIPPER_FLAG,
        "joint_mapping",
        "phone_config",
        "group_name",
        "base_link_name",
        "ee_frame_name",
        "target_frame_name",
        "ik_timeout",
        "target",
    }
    for key, value in device_config.items():
        if key not in known_keys:
            device_param[key] = value

    if "phone_config" in device_config:
        device_param["phone_config"] = device_config["phone_config"]

    for moveit_key in ("group_name", "base_link_name", "ee_frame_name", "target_frame_name", "ik_timeout"):
        if moveit_key in device_config:
            device_param[moveit_key] = device_config[moveit_key]

    # ----- Cartesian solver selection -----
    # Cartesian solver selection lives in robot.teleoperation.cartesian.solver.
    # The default tool frame follows robot.moveit.ee_link so arm/gripper and
    # arm_tcp/tcp remain a single MoveIt-owned SSOT; cartesian.tool_frame is
    # only an explicit override.
    cart_cfg = teleop_config.get("cartesian", {}) or {}
    cart_solver = cart_cfg.get("solver", "placo_servo")
    if cart_solver not in ("moveit_servo", "placo_servo"):
        raise ValueError(f"teleoperation.cartesian.solver must be 'moveit_servo' or 'placo_servo', got {cart_solver!r}")
    moveit_cfg = robot_config.get("moveit", {}) or {}
    cart_tool_frame = (
        cart_cfg.get("tool_frame") or device_config.get("ee_frame_name") or moveit_cfg.get("ee_link") or "gripper"
    )
    _validate_tool_frame(cart_tool_frame, robot_config)
    device_param["cartesian_solver"] = cart_solver
    device_param["tool_frame"] = cart_tool_frame
    device_param["base_link_name"] = device_param.get("base_link_name", moveit_cfg.get("base_link", "base"))

    placo_cfg = cart_cfg.get("placo_servo", {}) or {}
    if cart_solver == "placo_servo":
        placo_linear_speed = placo_cfg.get("linear_speed", 0.3)
        placo_angular_speed = placo_cfg.get("angular_speed", 0.7)
        cp = device_param.setdefault("control_params", {})
        cp["cartesian_linear_speed"] = placo_linear_speed
        cp["cartesian_angular_speed"] = placo_angular_speed
        logger.info(
            f"placo_servo speed override: linear={placo_linear_speed} (0~1), angular={placo_angular_speed} (0~1)"
        )

    logger.info(f"Cartesian solver={cart_solver}, tool_frame={cart_tool_frame}")

    device_type = device_config.get("type", "")

    control_frequency = device_config.get("control_frequency", 50.0)
    phone_input_mode = None
    if device_type == "phone":
        phone_cfg = device_config.get("phone_config", {}) or {}
        phone_backend = str(phone_cfg.get("backend", "webphone")).lower()
        if phone_backend != "webphone":
            raise ValueError("phone_config.backend must be 'webphone'")
        phone_input_mode = "pose"
        legacy_input_mode = phone_cfg.get("input_mode")
        if legacy_input_mode is not None and str(legacy_input_mode).lower() != phone_input_mode:
            raise ValueError(
                f"phone_config.input_mode={legacy_input_mode!r} is no longer supported for "
                "WebPhone; phone teleoperation uses the fixed 'pose' contract"
            )
        if cart_solver != "placo_servo":
            raise ValueError("Phone teleoperation requires teleoperation.cartesian.solver=placo_servo")

    if device_type == "joy_teleop":
        return _create_joy_teleop_nodes(device_config)

    if device_type == "vr_teleop":
        return _generate_vr_teleop_nodes(
            robot_config,
            device_config,
            robot_description_dict,
            cart_solver=cart_solver,
            base_link_name=device_param["base_link_name"],
            tool_frame=cart_tool_frame,
        )

    # Prepare lerobot environment
    env = prepare_lerobot_env()

    # Resolve the Placo runtime contract once, then inject the same topic/service
    # names into both the device-side backend and the downstream solver node.
    resolved_placo_params = None
    if cart_solver == "placo_servo" and device_type == "phone":
        position_only_override = placo_cfg.get("position_only")
        if "position_only" in phone_cfg:
            logger.warning(
                "phone_config.position_only is deprecated; use teleoperation.cartesian.placo_servo.position_only"
            )
            position_only_override = phone_cfg["position_only"]
        resolved_placo_params = _resolve_so101_placo_servo_params(
            robot_config,
            arm_joint_names=arm_joint_names,
            command_out_topic=groups_by_name["arm"].topic if "arm" in groups_by_name else None,
            input_mode=phone_input_mode,
            position_only=position_only_override,
            require_complete_contract=True,
        )
        resolved_placo_params["command_lease_timeout_s"] = float(
            (phone_cfg.get("web", {}) or {}).get("command_stale_s", 0.18)
        )
        device_param["cartesian_backend_config"] = _placo_backend_config(resolved_placo_params)

    # Convert dicts to JSON strings for ROS 2 parameter passing
    device_param_json = json.dumps(device_param)
    joint_limits_json = json.dumps(joint_limits)
    publish_groups_json = json.dumps([group.to_dict() for group in publish_groups])
    logger.debug(f"device_param_json: {device_param_json}")
    logger.debug(f"device_param dict: {device_param}")

    moveit_config = robot_config.get("moveit", {})

    if device_type == "phone":
        base_link_name = device_param.get("base_link_name", moveit_config.get("base_link", "base_link"))

        device_param_ext = dict(device_param)
        device_param_ext.update(
            {
                "base_link_name": base_link_name,
                "control_frequency": control_frequency,
            }
        )
        device_param_json = json.dumps(device_param_ext)

    node_name = "robot_teleop_node"
    if device_config.get("name"):
        node_name = f"robot_teleop_{re.sub(r'[^A-Za-z0-9_]+', '_', device_config['name']).strip('_')}"

    teleop_parameters = {
        "control_frequency": control_frequency,
        "device_config": device_param_json,
        "joint_limits": joint_limits_json,
        "publish_groups": publish_groups_json,
        "arm_command_topic": groups_by_name.get("arm").topic
        if "arm" in groups_by_name
        else "/arm_position_controller/commands",
        "gripper_command_topic": (
            groups_by_name["gripper"].topic if "gripper" in groups_by_name else "/gripper_position_controller/commands"
        ),
        "estop_topic": safety_config.get("estop_topic", "/emergency_stop"),
    }
    # Empty arrays are not valid untyped ROS 2 parameters. Hand retargeting
    # uses explicit publish_groups, so omit unused legacy arrays.
    if arm_joint_names:
        teleop_parameters["arm_joint_names"] = arm_joint_names
    if gripper_joint_names:
        teleop_parameters["gripper_joint_names"] = gripper_joint_names

    teleop_node = Node(
        package="robot_teleop",
        executable="teleop_node",
        name=node_name,
        output="screen",
        env=env,
        parameters=[teleop_parameters],
        on_exit=_shutdown_when_process_exits(node_name),
    )
    nodes.append(teleop_node)
    logger.info(f"Generated teleop_node for device: {device_config.get('name', '')} (type: {device_type})")

    # Add joy_node to read physical joystick and publish to /joy
    if device_config.get("type") == "xbox_controller":
        input_dev = device_config.get("input_device", "/dev/input/js0")
        joy_node = Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            parameters=[
                {
                    "device_id": 0,
                    "device_name": "",
                    "deadzone": 0.1,
                    "autorepeat_rate": 20.0,
                    "sticky_buttons": False,
                }
            ],
            output="screen",
        )
        nodes.append(joy_node)
        logger.info(f"Added joy_node for input device: {input_dev}")

    # Add the selected Cartesian backend node for Xbox controller and phone.
    if device_config.get("type") in ("xbox_controller", "phone"):
        if cart_solver == "moveit_servo":
            servo_node = _create_servo_node(robot_config, device_config, robot_description_dict)
            nodes.append(servo_node)
            logger.info("Generated servo_node for Cartesian servo (MoveIt Servo) control")
        elif cart_solver == "placo_servo":
            placo_node = _create_so101_placo_servo_node(
                robot_config,
                device_config,
                robot_description_dict,
                input_mode=phone_input_mode if device_type == "phone" else None,
                resolved_params=resolved_placo_params,
            )
            nodes.append(placo_node)
            logger.info("Generated so101_placo_servo_node for SO101 Placo QP Cartesian control")

    return nodes


def _create_joy_teleop_nodes(device_config: dict) -> list[Node]:
    """Create joy_node + joy_teleop launch actions for mobile-base teleoperation."""
    nodes = []

    input_dev = device_config.get("input_device", "/dev/input/js0")
    joy_params = {
        "dev": input_dev,
        "deadzone": float(device_config.get("deadzone", 0.1)),
        "autorepeat_rate": float(device_config.get("autorepeat_rate", 20.0)),
        "sticky_buttons": bool(device_config.get("sticky_buttons", False)),
    }
    nodes.append(
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            parameters=[joy_params],
            output="screen",
        )
    )

    config_path = device_config.get("config_path")
    if not config_path:
        raise ValueError("joy_teleop device requires 'config_path'")

    nodes.append(
        Node(
            package="joy_teleop",
            executable="joy_teleop",
            name="joy_teleop",
            parameters=[resolve_ros_path(config_path)],
            output="screen",
        )
    )
    print(f"[teleop_builder] Generated joy_teleop stack using {config_path}")
    return nodes


def _generate_vr_teleop_nodes(
    robot_config: dict,
    device_config: dict,
    robot_description_dict: dict = None,
    *,
    cart_solver: str = "placo_servo",
    base_link_name: str = "base",
    tool_frame: str = "gripper",
) -> list[Node]:
    """Launch the standalone VR teleop server + (for so101) the placo servo.

    The VR server (``vr_teleop``) is a self-contained rclpy node — it
    does NOT go through ``teleop_node`` / device_factory. Its so101 output
    profile drives the placo cartesian servo, so downstream topic/service names
    default to ``so101_placo_servo_node`` and this builder also spawns that
    servo node when ``output_profile == 'so101'``.
    """
    nodes = []
    vr_cfg = device_config.get("vr_config", {}) or {}
    output_profile = str(vr_cfg.get("output_profile", "humanoid"))
    # so101 input mode: "velocity" (differential twist → placo integrates) or
    # "pose" (absolute EE pose passthrough → placo sets its reference). This
    # single source of truth in vr_config drives BOTH the VR node and the placo
    # servo node (the placo node's input_mode is overridden below to match).
    so101_input_mode = str(vr_cfg.get("so101_input_mode", "velocity"))
    so101_pose_topic = vr_cfg.get("so101_pose_topic", "/so101_placo_servo_node/pose_cmd_base")

    env = prepare_lerobot_env()

    # control_frequency is declared at the device layer (sibling of vr_config)
    # to match every other teleop device; accept it there as the default and
    # still allow a vr_config override. Reading only vr_config.control_frequency
    # silently ignored the device-layer value.
    control_frequency = float(vr_cfg.get("control_frequency", device_config.get("control_frequency", 50.0)))

    # Resolve the Placo start/stop services and Home action ONCE so
    # both the VR node and the placo node below are wired to the same names.
    # A user override of any of these in vr_config must reach BOTH sides.
    so101_start_service = vr_cfg.get("so101_start_service", "/so101_placo_servo_node/start")
    so101_stop_service = vr_cfg.get("so101_stop_service", "/so101_placo_servo_node/stop")
    so101_home_action = vr_cfg.get("so101_home_action", "/so101_placo_servo_node/return_home")
    safety_cfg = (robot_config.get("teleoperation", {}) or {}).get("safety", robot_config.get("safety", {}) or {})

    # Downstream placo wiring (overridable via vr_config). base/tool frames come
    # from the shared cartesian SSOT so the tool→base angular conversion inside
    # the VR node uses the same frames as the xbox/phone placo path.
    vr_params = {
        "host": vr_cfg.get("host", "0.0.0.0"),
        "port": int(vr_cfg.get("port", 8889)),
        "control_frequency": control_frequency,
        "output_profile": output_profile,
        "controller_side": vr_cfg.get("controller_side", "right"),
        "base_link_name": vr_cfg.get("base_link_name", base_link_name),
        "tool_frame": vr_cfg.get("tool_frame", tool_frame),
        "so101_linear_topic": vr_cfg.get("so101_linear_topic", "/so101_placo_servo_node/linear_cmd_base"),
        "so101_angular_topic": vr_cfg.get("so101_angular_topic", "/so101_placo_servo_node/angular_cmd_base"),
        "so101_gripper_topic": vr_cfg.get("so101_gripper_topic", "/gripper_position_controller/commands"),
        "so101_start_service": so101_start_service,
        "so101_stop_service": so101_stop_service,
        "so101_home_action": so101_home_action,
        "estop_topic": safety_cfg.get("estop_topic", "/emergency_stop"),
        "so101_input_mode": so101_input_mode,
        "so101_pose_topic": so101_pose_topic,
    }
    # Forward any remaining tuning knobs verbatim (speed scales, deadzones,
    # gripper open/closed, tf_stale_threshold_s, position_scale, etc.).
    _passthrough = {
        "linear_speed_scale",
        "angular_speed_scale",
        "max_linear_speed",
        "max_angular_speed",
        "velocity_ema_alpha",
        "linear_deadzone",
        "angular_deadzone",
        "so101_gripper_open",
        "so101_gripper_closed",
        "tf_stale_threshold_s",
        "position_scale",
        "so101_position_only",
        "so101_command_stale_s",
    }
    for key in _passthrough:
        if key in vr_cfg:
            vr_params[key] = vr_cfg[key]

    nodes.append(
        Node(
            package="robot_teleop",
            executable="vr_teleop",
            name="vr_teleop",
            output="screen",
            env=env,
            parameters=[vr_params],
        )
    )
    logger.info(f"Generated vr_teleop (output_profile={output_profile})")

    # so101 profile drives the cartesian servo; spawn placo alongside.
    if output_profile == "so101":
        if cart_solver != "placo_servo":
            raise ValueError(
                f"vr_teleop output_profile=so101 requires cartesian.solver='placo_servo', got {cart_solver!r}"
            )
        placo_node = _create_so101_placo_servo_node(
            robot_config,
            device_config,
            robot_description_dict,
            input_mode=so101_input_mode,
            pose_cmd_topic=so101_pose_topic,
            start_service=so101_start_service,
            stop_service=so101_stop_service,
            home_action=so101_home_action,
        )
        nodes.append(placo_node)
        logger.info(f"Generated so101_placo_servo_node for VR so101 Cartesian control (input_mode={so101_input_mode})")

    return nodes


def _create_servo_node(robot_config: dict, device_config: dict, robot_description_dict: dict = None) -> Node:
    """Create MoveIt Servo node."""
    import yaml

    from robot_config.utils import resolve_ros_path

    # 1. Load servo parameters
    servo_config_name = device_config.get("servo_config", "so101_servo")
    servo_params_path = resolve_ros_path(f"$(find robot_moveit)/config/{servo_config_name}.yaml")

    with open(servo_params_path) as f:
        servo_params = yaml.safe_load(f)

    # 2. Build MoveIt configuration manually for robustness
    # MoveItConfigsBuilder can be finicky with relative paths in different environments
    robot_type = robot_config.get("type", "so101")

    moveit_params = {}
    try:
        # Load SRDF (Semantic Robot Description Format)
        srdf_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/{robot_type}.srdf")
        if os.path.exists(srdf_path):
            with open(srdf_path) as f:
                moveit_params["robot_description_semantic"] = f.read()
            logger.info(f"Loaded SRDF from {srdf_path}")
        else:
            logger.warning(f"SRDF not found at {srdf_path}")

        # Load Kinematics
        kinematics_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/kinematics.yaml")
        if os.path.exists(kinematics_path):
            with open(kinematics_path) as f:
                moveit_params["robot_description_kinematics"] = yaml.safe_load(f)
            logger.info(f"Loaded kinematics from {kinematics_path}")

        # Load Joint Limits
        joint_limits_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/joint_limits.yaml")
        if os.path.exists(joint_limits_path):
            with open(joint_limits_path) as f:
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
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            {"moveit_servo": servo_params},
            moveit_params,
        ],
    )
    return servo_node


# ---------------------------------------------------------------------------
# Cartesian helpers: tool_frame validation + placo_servo launcher
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when the SSOT YAML is internally inconsistent."""


def _validate_tool_frame(tool_frame: str, robot_config: dict) -> None:
    """Confirm ``tool_frame`` resolves at launch time.

    Accepts the configured ``base_link`` / ``ee_link`` and otherwise lets URDF
    links through to runtime TF validation. Launch-time code does not enumerate
    URDF links, so the Servo node is the final authority for custom link names.
    """
    if not tool_frame:
        raise ConfigError("tool_frame is empty")
    moveit_cfg = robot_config.get("moveit", {}) or {}
    base_link = moveit_cfg.get("base_link", "base")
    if tool_frame == base_link:
        return
    if tool_frame == moveit_cfg.get("ee_link"):
        return
    # Assume the user supplied a URDF link name; runtime TF will validate it
    # and the Servo node will fail fast if the link does not exist.
    logger.warning(
        f"tool_frame={tool_frame!r} is neither base_link nor ee_link; "
        f"assuming it is a URDF link. Runtime TF will validate."
    )


def _placo_backend_config(params: dict) -> dict[str, str]:
    """Translate resolved solver parameter names to backend constructor keys."""
    return {
        "linear_topic": params["linear_cmd_topic"],
        "angular_topic": params["angular_cmd_topic"],
        "pose_topic": params["pose_cmd_topic"],
        "start_srv": params["start_service"],
        "stop_srv": params["stop_service"],
        "home_action": params["home_action"],
        "command_lease_topic": params["command_lease_topic"],
    }


def _resolve_so101_placo_servo_params(
    robot_config: dict,
    *,
    arm_joint_names: list[str] | None = None,
    command_out_topic: str | None = None,
    input_mode: str | None = None,
    linear_cmd_topic: str | None = None,
    angular_cmd_topic: str | None = None,
    pose_cmd_topic: str | None = None,
    start_service: str | None = None,
    stop_service: str | None = None,
    home_action: str | None = None,
    command_lease_topic: str | None = None,
    position_only: bool | None = None,
    require_complete_contract: bool = False,
) -> dict:
    """Resolve Placo parameters, optionally requiring Phone's full endpoint contract."""
    import yaml as _yaml

    moveit_cfg = robot_config.get("moveit", {}) or {}
    yaml_ref = moveit_cfg.get("so101_placo_servo_config_path")
    if not yaml_ref:
        raise ConfigError(
            "solver=placo_servo requires moveit.so101_placo_servo_config_path "
            "(e.g. $(find robot_moveit)/config/so101_placo_servo.yaml)"
        )
    yaml_path = resolve_ros_path(yaml_ref)
    with open(yaml_path) as f:
        params = _yaml.safe_load(f) or {}
    missing_endpoints = [key for key in _PLACO_ENDPOINT_DEFAULTS if not params.get(key)]
    if require_complete_contract and missing_endpoints:
        raise ConfigError(
            "Placo solver config must define the complete topic/service contract; missing: "
            + ", ".join(missing_endpoints)
        )

    if arm_joint_names is None:
        joints_cfg = robot_config.get("joints", {}) or {}
        arm_joint_names = joints_cfg.get("arm", [])
    if not arm_joint_names:
        raise ConfigError(
            "solver=placo_servo requires the selected target to list arm "
            "joint names (used to order the position command output)"
        )
    params["arm_joint_names"] = list(arm_joint_names)
    if command_out_topic is not None:
        params["command_out_topic"] = command_out_topic

    # Drop-in TCP support: the IK tip frame is the SSOT moveit.ee_link
    # (gripper | tcp). Inject it so selecting tcp re-targets placo's frame
    # task without editing the solver YAML. Defaults to the YAML value (gripper)
    # when ee_link is absent, so the gripper path is byte-for-byte unchanged.
    ee_link = moveit_cfg.get("ee_link")
    if ee_link:
        params["ik_link_name"] = ee_link

    if input_mode is not None:
        params["input_mode"] = input_mode
    if linear_cmd_topic is not None:
        params["linear_cmd_topic"] = linear_cmd_topic
    if angular_cmd_topic is not None:
        params["angular_cmd_topic"] = angular_cmd_topic
    if pose_cmd_topic is not None:
        params["pose_cmd_topic"] = pose_cmd_topic

    if start_service is not None:
        params["start_service"] = start_service
    if stop_service is not None:
        params["stop_service"] = stop_service
    if home_action is not None:
        params["home_action"] = home_action
    if command_lease_topic is not None:
        params["command_lease_topic"] = command_lease_topic
    if position_only is not None:
        params["position_only"] = position_only
    teleop_safety = (robot_config.get("teleoperation", {}) or {}).get("safety", robot_config.get("safety", {}) or {})
    params["estop_topic"] = teleop_safety.get("estop_topic", "/emergency_stop")
    invalid_endpoints = [
        key
        for key in _PLACO_ENDPOINT_DEFAULTS
        if key in params and (not isinstance(params[key], str) or not params[key].strip())
    ]
    if invalid_endpoints:
        raise ConfigError("Placo topic/service names must be non-empty strings: " + ", ".join(invalid_endpoints))

    reset_positions = robot_config.get("ros2_control", {}).get("reset_positions", {}) or {}
    if not isinstance(reset_positions, dict):
        raise ConfigError("ros2_control.reset_positions must be a mapping")
    missing_home_joints = [name for name in arm_joint_names if name not in reset_positions]
    if missing_home_joints:
        raise ConfigError(
            "ros2_control.reset_positions is missing arm joint target(s) required by Placo ArmReturnHome; "
            f"missing: {', '.join(missing_home_joints)}"
        )
    home_joint_positions = [reset_positions[name] for name in arm_joint_names]
    invalid_home_joints = [
        name
        for name, position in zip(arm_joint_names, home_joint_positions, strict=True)
        if not _is_finite_number(position)
    ]
    if invalid_home_joints:
        raise ConfigError(
            "ros2_control.reset_positions must contain a finite number for every Placo ArmReturnHome arm joint; "
            f"invalid: {', '.join(invalid_home_joints)}"
        )
    params["home_joint_positions"] = [float(position) for position in home_joint_positions]
    return params


def _create_so101_placo_servo_node(
    robot_config: dict,
    device_config: dict,
    robot_description_dict: dict = None,
    *,
    input_mode: str = None,
    pose_cmd_topic: str = None,
    start_service: str = None,
    stop_service: str = None,
    home_action: str = None,
    position_only: bool | None = None,
    resolved_params: dict | None = None,
) -> Node:
    """Launch ``so101_placo_servo_node`` from a resolved shared contract."""
    target_config = device_config.get("target", {}) or {}
    joints_config = robot_config.get("joints", {}) or {}
    selected_arm_joint_names = target_config.get("arm_joint_names", joints_config.get("arm", []))
    params = (
        dict(resolved_params)
        if resolved_params is not None
        else _resolve_so101_placo_servo_params(
            robot_config,
            arm_joint_names=selected_arm_joint_names,
            command_out_topic=target_config.get("arm_command_topic"),
            input_mode=input_mode,
            linear_cmd_topic=None,
            angular_cmd_topic=None,
            pose_cmd_topic=pose_cmd_topic,
            start_service=start_service,
            stop_service=stop_service,
            home_action=home_action,
            command_lease_topic=None,
            position_only=position_only,
        )
    )

    # The node expands the so101 xacro in-memory at runtime via the
    # robot_description package share dir, so robot_description_dict is not
    # required for kinematics; it is still forwarded for parity / use_sim_time.
    extra = {}
    if robot_description_dict:
        extra.update(robot_description_dict)

    return Node(
        package="robot_moveit",
        executable="so101_placo_servo_node.py",
        name="so101_placo_servo_node",
        output="screen",
        parameters=[params, extra],
    )


def validate_teleop_config(
    teleop_config: dict[str, object],
    auxiliary_actuators: dict | None = None,
    joints: dict | None = None,
    root_safety: dict | None = None,
) -> list[str]:
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

    if not teleop_config.get("enabled", False):
        return errors  # Not enabled, skip validation

    devices = teleop_config.get("devices", [])
    if not devices:
        errors.append("devices list is empty")
        return errors

    active_devices = teleop_config.get("active_devices")
    active_device = teleop_config.get("active_device")
    if active_devices:
        if not isinstance(active_devices, list):
            errors.append("active_devices must be a list when specified")
            return errors
        active_device_names = active_devices
    elif active_device:
        active_device_names = [active_device]
    else:
        errors.append("active_device or active_devices must be specified when teleoperation is enabled")
        return errors
    if not all(isinstance(name, str) and name for name in active_device_names):
        errors.append("active_device names must be non-empty strings")
        return errors

    devices_by_name = {device.get("name"): device for device in devices}
    if len(set(active_device_names)) != len(active_device_names):
        errors.append("active_devices must not contain duplicate device names")
    cartesian_devices = []
    command_topic_owners: dict[str, list[tuple[str, str]]] = {}
    for name in active_device_names:
        device = devices_by_name.get(name) or {}
        device_type = device.get("type")
        if (
            device_type in ("phone", "xbox_controller")
            or device_type == "vr_teleop"
            and (device.get("vr_config", {}) or {}).get("output_profile", "humanoid") == "so101"
        ):
            cartesian_devices.append(name)
        commands_so101 = device_type in ("leader_arm", "phone", "xbox_controller") or (
            device_type == "vr_teleop"
            and (device.get("vr_config", {}) or {}).get("output_profile", "humanoid") == "so101"
        )
        target = device.get("target", {}) or {}
        if not isinstance(target, dict):
            target = {}
        if device_type == "vr_teleop":
            if target:
                errors.append(f"Device '{name}': vr_teleop uses vr_config outputs and cannot define target")
            if commands_so101:
                vr_config = device.get("vr_config", {}) or {}
                command_groups = [
                    ("arm", "/arm_position_controller/commands"),
                    ("gripper", vr_config.get("so101_gripper_topic", "/gripper_position_controller/commands")),
                ]
            else:
                command_groups = []
        else:
            try:
                groups = resolve_target_publish_groups(
                    target,
                    joints or {"arm": ["1", "2", "3", "4", "5"], "gripper": ["6"]},
                    auxiliary_actuators or {},
                )
            except ValueError as exc:
                errors.append(f"Device '{name}': {exc}")
                groups = []
            command_groups = [(group.name, group.topic) for group in groups]
            if device_type == "hand_retarget" and "publish_groups" not in target and "actuator" not in target:
                errors.append(
                    f"Device '{name}': hand retarget requires explicit target.publish_groups or target.actuator"
                )
        for group_name, topic in command_groups:
            if not isinstance(topic, str) or not topic.strip():
                errors.append(f"Device '{name}': command topics must be non-empty strings")
                continue
            command_topic_owners.setdefault(topic, []).append((name, group_name))
    if len(cartesian_devices) > 1:
        errors.append(
            f"only one active SO-101 Cartesian device is currently supported; selected: {', '.join(cartesian_devices)}"
        )
    for topic, ownership in command_topic_owners.items():
        if len(ownership) > 1:
            owners = [owner for owner, _group in ownership]
            groups = {group for _owner, group in ownership}
            role = next(iter(groups)) if len(groups) == 1 and groups <= {"arm", "gripper"} else ""
            label = f"{role} command topic" if role else "command topic"
            errors.append(f"active devices share {label} {topic!r}: {', '.join(owners)}")

    cartesian = teleop_config.get("cartesian", {}) or {}
    placo_config = cartesian.get("placo_servo", {}) or {}
    if "position_only" in placo_config and not isinstance(placo_config["position_only"], bool):
        errors.append("teleoperation.cartesian.placo_servo.position_only must be a boolean")
    requires_joint_limits = False
    for active_device_name in active_device_names:
        device = devices_by_name.get(active_device_name)
        if not device:
            errors.append(f"active device '{active_device_name}' not found in devices list")
            continue

        # Validate device type
        device_type = device.get("type")
        if not device_type:
            errors.append(f"Device '{active_device_name}': missing 'type' field")
        elif device_type == "mhandpro_glove":
            errors.append(
                f"Device '{active_device_name}': mhandpro_glove has been removed; "
                "use hand_sources.mhandpro with a hand_retarget device"
            )
        elif device_type not in ("joy_teleop", "vr_teleop"):
            requires_joint_limits = True

        # Type-specific validation
        if device_type == "leader_arm" and not device.get("port"):
            errors.append(f"Device '{active_device_name}': leader_arm requires 'port' field")

        if device_type == "hand_retarget":
            side = device.get("side", "right")
            if side not in ("left", "right"):
                errors.append(f"Device '{active_device_name}': hand_retarget side must be left or right")
            if not device.get("source_topic"):
                errors.append(f"Device '{active_device_name}': hand_retarget requires source_topic")
            retargeter = device.get("retargeter", {}) or {}
            if not isinstance(retargeter, dict) or not retargeter.get("type"):
                errors.append(f"Device '{active_device_name}': hand_retarget requires retargeter.type")
            elif retargeter.get("type") == "aero_compact" and not device.get("calib_file"):
                errors.append(f"Device '{active_device_name}': aero_compact hand_retarget requires calib_file")

        if device_type == "phone":
            phone_config = device.get("phone_config", {})
            if not phone_config:
                errors.append(f"Device '{active_device_name}': phone requires 'phone_config' field")
                continue
            backend = str(phone_config.get("backend", "webphone")).lower()
            if backend != "webphone":
                errors.append(f"Device '{active_device_name}': phone backend must be 'webphone'")
            control_frequency = device.get("control_frequency", 50.0)
            control_frequency_valid = _is_finite_number(control_frequency) and control_frequency > 0
            if not control_frequency_valid:
                errors.append(f"Device '{active_device_name}': control_frequency must be finite and positive")

            input_mode = "pose"
            legacy_input_mode = phone_config.get("input_mode")
            if "position_only" in phone_config and not isinstance(phone_config["position_only"], bool):
                errors.append(f"Device '{active_device_name}': legacy phone_config.position_only must be a boolean")
            if legacy_input_mode is not None and str(legacy_input_mode).lower() != input_mode:
                errors.append(
                    f"Device '{active_device_name}': phone_config.input_mode={legacy_input_mode!r} is not supported "
                    f"for WebPhone; expected {input_mode!r}"
                )
            if cartesian.get("solver", "placo_servo") != "placo_servo":
                errors.append(f"Device '{active_device_name}': phone teleoperation requires solver=placo_servo")
            bounds = phone_config.get(
                "end_effector_bounds",
                {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]},
            )
            bounds_min = bounds.get("min") if isinstance(bounds, dict) else None
            bounds_max = bounds.get("max") if isinstance(bounds, dict) else None
            bounds_are_valid = (
                isinstance(bounds_min, list | tuple)
                and isinstance(bounds_max, list | tuple)
                and len(bounds_min) == 3
                and len(bounds_max) == 3
                and all(_is_finite_number(value) for value in (*bounds_min, *bounds_max))
                and all(minimum < 0.0 < maximum for minimum, maximum in zip(bounds_min, bounds_max, strict=True))
            )
            if not bounds_are_valid:
                errors.append(
                    f"Device '{active_device_name}': phone end_effector_bounds must contain zero "
                    "strictly inside every axis (min < 0 < max)"
                )
            web = phone_config.get("web", {})
            if not isinstance(web, dict):
                errors.append(f"Device '{active_device_name}': phone_config.web must be a mapping")
                continue
            ar_enabled = web.get("ar_enabled", True)
            optical_fallback = phone_config.get("optical_flow_fallback_enabled", True)
            if not isinstance(ar_enabled, bool):
                errors.append(f"Device '{active_device_name}': phone_config.web.ar_enabled must be a boolean")
            if not isinstance(optical_fallback, bool):
                errors.append(
                    f"Device '{active_device_name}': phone_config.optical_flow_fallback_enabled must be a boolean"
                )
            if ar_enabled is False and optical_fallback is False:
                errors.append(f"Device '{active_device_name}': WebPhone requires WebXR AR or optical-flow fallback")
            http_port = web.get("http_port", 8765)
            websocket_port = web.get("websocket_port", 8766)
            for field, port in (("http_port", http_port), ("websocket_port", websocket_port)):
                if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
                    errors.append(f"Device '{active_device_name}': phone_config.web.{field} is invalid")
            if http_port == websocket_port:
                errors.append(f"Device '{active_device_name}': WebPhone ports must differ")
            stale_s = web.get("command_stale_s", 0.18)
            if not _is_finite_number(stale_s) or stale_s <= 0:
                errors.append(
                    f"Device '{active_device_name}': phone_config.web.command_stale_s must be finite and positive"
                )
            elif control_frequency_valid:
                stop_latency_s = float(stale_s) + 1.0 / float(control_frequency)
                if stop_latency_s > 0.22 + 1e-9:
                    errors.append(
                        f"Device '{active_device_name}': WebPhone stop-request latency bound "
                        f"command_stale_s + one control period is {stop_latency_s:.3f}s, exceeding 0.220s"
                    )
            tls = web.get("tls", {})
            if not isinstance(tls, dict):
                errors.append(f"Device '{active_device_name}': phone_config.web.tls must be a mapping")
            elif tls.get("enabled", True):
                cert_file = tls.get("cert_file")
                key_file = tls.get("key_file")
                if bool(cert_file) != bool(key_file):
                    errors.append(f"Device '{active_device_name}': WebPhone TLS cert_file/key_file must be paired")

    # Validate safety config
    safety = teleop_config.get("safety", root_safety or {})
    joint_limits = safety.get("joint_limits", {})
    estop_topic = safety.get("estop_topic", "/emergency_stop")
    if not isinstance(estop_topic, str) or not estop_topic.strip():
        errors.append("safety.estop_topic must be a non-empty string")

    if requires_joint_limits and not joint_limits:
        errors.append("safety.joint_limits must be specified for teleoperation")
    elif joint_limits:
        # Validate joint limit format
        for joint_name, limits in joint_limits.items():
            if "min" not in limits or "max" not in limits:
                errors.append(f"Joint '{joint_name}': limits must have 'min' and 'max' fields")
            elif limits["min"] >= limits["max"]:
                errors.append(f"Joint '{joint_name}': min must be less than max")

    return errors


def get_recording_topics(robot_config: dict) -> list[str]:
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
    topics.append("/joint_states")

    # Add controller command topics
    topics.append("/arm_position_controller/commands")
    topics.append("/gripper_position_controller/commands")

    # Add teleop diagnostics
    topics.append("/diagnostics")

    # Add camera topics from peripherals
    peripherals = robot_config.get("peripherals", [])
    for peripheral in peripherals:
        if peripheral.get("type") == "camera":
            name = peripheral.get("name", "camera")
            # Add common camera topics
            topics.append(f"/camera/{name}/image_raw")
            topics.append(f"/camera/{name}/camera_info")

    return topics
