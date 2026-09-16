"""Runtime-owned single-arm teleoperation executor composition."""

import yaml
from launch_ros.actions import Node

from robot_runtime.path_subst import resolve_path


def generate_teleoperation_nodes(profile, descriptor, robot_description_dict):
    config = profile.get("teleoperation", {})
    if not config.get("enabled", False):
        return []
    model = descriptor["model"]
    group = config["target_group"]
    arm = model["joint_groups"][group]
    gripper = model["joint_groups"]["gripper"]
    if group != "arm" or len(gripper) != 1:
        raise ValueError("runtime teleoperation supports one named SO101 arm and gripper")
    interfaces = descriptor["interfaces"]
    prefix = f"motion.{group}."
    with open(resolve_path(config["servo_config"]), encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    for parameter, interface in {
        "pose_cmd_topic": "pose",
        "linear_cmd_topic": "linear",
        "angular_cmd_topic": "angular",
        "start_service": "start",
        "stop_service": "stop",
        "home_action": "home",
        "command_lease_topic": "lease",
        "joint_intent_topic": "joints",
    }.items():
        params[parameter] = interfaces[prefix + interface]["endpoint"]
    channels = profile["command_channels"]
    arm_channel = [item for item in channels if item.get("joints") == arm]
    gripper_channel = [item for item in channels if item.get("joints") == gripper]
    if len(arm_channel) != 1 or len(gripper_channel) != 1:
        raise ValueError("runtime teleoperation output channels are ambiguous")
    params.update(
        {
            "managed_teleop": True,
            "input_mode": "auto",
            "arm_joint_names": arm,
            "gripper_joint_names": gripper,
            "command_out_topic": arm_channel[0]["topic"],
            "gripper_command_out_topic": gripper_channel[0]["topic"],
            "joint_limits_lower": [model["joint_limits"][name]["min"] for name in arm],
            "joint_limits_upper": [model["joint_limits"][name]["max"] for name in arm],
            "gripper_limits_lower": [model["joint_limits"][name]["min"] for name in gripper],
            "gripper_limits_upper": [model["joint_limits"][name]["max"] for name in gripper],
            "home_joint_positions": [model["home_positions"][name] for name in arm],
            "planning_frame": model["frames"]["base_link"],
            "ik_link_name": model["frames"]["ee_link"],
            "incoming_command_timeout": config["command_stale_s"],
            "command_lease_timeout_s": config["command_stale_s"],
            "runtime_mode_service": interfaces["runtime.set_mode"]["endpoint"],
            "runtime_stop_service": interfaces["runtime.stop"]["endpoint"],
            "runtime_status_topic": interfaces["runtime.status"]["endpoint"],
            "joint_state_topic": interfaces["joint.state"]["endpoint"],
        }
    )
    params.update(robot_description_dict)
    nodes = [
        Node(
            package="so101_motion",
            executable="so101_placo_servo_node.py",
            name="so101_placo_servo_node",
            output="screen",
            parameters=[params],
        )
    ]
    source = config.get("leader_source")
    if source:
        nodes.append(
            Node(
                package="so101_hardware",
                executable="leader_arm_pub",
                name="teleop_input_source",
                output="screen",
                parameters=[
                    {
                        "port": str(source["port"]),
                        "baudrate": int(source["baudrate"]),
                        "calibration_file": resolve_path(source["calibration_file"]),
                        "calibration_version": int(source["calibration_version"]),
                        "joint_order": list(profile["joints"]),
                        "gripper_joint": gripper[0] if gripper else "6",
                        "source_topic": "/inputs/so101_leader/state",
                        "publish_rate": float(source["publish_rate"]),
                    }
                ],
            )
        )
    return nodes
