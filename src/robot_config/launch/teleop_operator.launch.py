"""Teleoperation-only launch entry for an operator workstation.

`robot.launch.py` always brings up ros2_control outside simulation, so it can
only serve a host that owns the follower hardware. That is the wrong shape for
a mobile base: the LeKiwi cart carries its onboard computer while the leader
arm and the gamepad are wired to the operator's workstation, so the leader can
never share a host with the follower.

This entry starts just the teleoperation stack from the same robot config the
robot host launches, mirroring how `cloud_inference.launch.py` reads that
config on the recorder host. Both ends therefore read one file and their
contract fingerprints match by construction, without duplicating the contract
across two configs that would have to be kept in sync by hand.

Usage:

    # robot host: follower, base, cameras, RTP senders
    ros2 launch robot_config robot.launch.py \\
        robot_config:=lekiwi_rtp_distributed control_mode:=teleop

    # operator workstation: leader arm and gamepad
    ros2 launch robot_config teleop_operator.launch.py \\
        robot_config:=lekiwi_rtp_distributed

The operator host needs a local copy of the follower calibration, because it is
the host that converts the leader's 0~1 gripper percentage into the follower's
radian stroke.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from robot_config.launch_builders.teleop import generate_teleop_nodes
from robot_config.loader import load_robot_config_dict
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.teleop_operator")


def _resolve_config_path(robot_config_name: str, config_path_override: str) -> Path:
    if config_path_override:
        return Path(config_path_override)
    try:
        share = get_package_share_directory("robot_config")
    except Exception:
        share = str(Path(__file__).parent.parent)
    return Path(share) / "config" / "robots" / f"{robot_config_name}.yaml"


def launch_setup(context, *args, **kwargs):
    robot_config_name = context.launch_configurations.get("robot_config", "")
    config_path_override = context.launch_configurations.get("config_path", "")
    config_path = _resolve_config_path(robot_config_name, config_path_override)

    logger.info("========== Teleoperation Operator Host ==========")
    logger.info(f"Loading config from: {config_path}")
    robot_config = load_robot_config_dict(config_path)
    robot_config["_config_path"] = str(config_path)

    teleop_config = robot_config.get("teleoperation", {}) or {}
    if not teleop_config.get("devices"):
        raise RuntimeError(
            f"{config_path} declares no teleoperation devices, so this host has nothing to drive. "
            "Add the leader arm and any other operator hardware under robot.teleoperation.devices."
        )

    # `teleoperation.enabled` answers whether the host running robot.launch.py
    # also owns the operator hardware, which is false whenever the leader lives
    # on a separate workstation. Launching this file is that host saying it does
    # own it, so the flag is not consulted here.
    if not teleop_config.get("enabled", False):
        logger.info("teleoperation.enabled is false in the config; enabling it for this operator host")
        teleop_config = {**teleop_config, "enabled": True}
        robot_config["teleoperation"] = teleop_config

    nodes = generate_teleop_nodes(robot_config, {})
    logger.info(f"Started {len(nodes)} teleoperation nodes; ros2_control stays on the robot host")
    return nodes


def generate_launch_description():
    """Generate the launch description for a teleoperation-only host."""
    return LaunchDescription(
        [
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
            OpaqueFunction(function=launch_setup),
        ]
    )
