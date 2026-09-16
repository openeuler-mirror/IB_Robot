"""CmdVel bridge builder for navigation.

Generates the cmd_vel_bridge_node that translates ROS cmd_vel to wheel commands.
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.cmd_vel")


def generate_cmd_vel_nodes(
    nav_config: dict[str, Any],
    motion_mode_config: dict[str, Any] | None = None,
    ekf_enabled: bool = False,
    use_sim: bool = False,
    robot_config: dict[str, Any] | None = None,
) -> list:
    """Generate cmd_vel bridge nodes.

    Args:
        nav_config: navigation section from robot_config YAML
        motion_mode_config: shared arm/base command-authorization configuration
        ekf_enabled: whether EKF is enabled (disables bridge TF publishing)
        use_sim: simulation mode flag
        robot_config: full robot config (to check runtime.provider)

    Returns:
        List of Node actions
    """
    nodes = []

    bridge_config = nav_config.get("cmd_vel_bridge", {})
    if not bridge_config.get("enabled", False):
        return nodes

    # Runtime provider path: the robot runtime's base node owns cmd_vel
    # projection, odometry and navigation gating (lekiwi_robot/base_node);
    # the generic bridge is only for in-config ros2_control bring-up.
    if str(((robot_config or {}).get("runtime") or {}).get("provider", "") or "").strip():
        logger.info("runtime.provider configured: cmd_vel bridge skipped (the runtime's base node owns it)")
        return nodes

    # Use publish_tf from config; default to False when EKF is enabled
    bridge_publish_tf = bridge_config.get("publish_tf", not ekf_enabled)
    motion_mode_config = motion_mode_config or {}
    motion_mode_enabled = bool(motion_mode_config.get("enabled", False))
    if motion_mode_enabled and "navigation_enabled_on_startup" not in motion_mode_config:
        raise ValueError("robot.motion_mode.navigation_enabled_on_startup is required when motion_mode is enabled")
    bridge_publish_odom = bridge_config.get("publish_odom", True)
    cmd_vel_topic = bridge_config.get("cmd_vel_topic", "/cmd_vel")

    bridge_params = {
        "wheel_radius": bridge_config.get("wheel_radius", 0.05),
        "base_radius": bridge_config.get("base_radius", 0.125),
        "max_radps": bridge_config.get("max_radps", 4.602),
        "odom_frame": bridge_config.get("odom_frame", "odom"),
        "base_frame": bridge_config.get("base_frame", "base_link"),
        "publish_tf": bridge_publish_tf,
        "publish_odom": bridge_publish_odom,
        "control_frequency": bridge_config.get("control_frequency", 50.0),
        "cmd_timeout": bridge_config.get("cmd_timeout", 0.5),
        "cmd_vel_topic": cmd_vel_topic,
        "joint_states_topic": bridge_config.get("joint_states_topic", "/joint_states"),
        "odom_topic": bridge_config.get("odom_topic", "/odom"),
        "use_sim_time": use_sim,
        "motion_mode_enabled": motion_mode_enabled,
        "navigation_enabled_on_startup": motion_mode_config.get("navigation_enabled_on_startup", False),
        "navigation_enabled_topic": motion_mode_config.get(
            "navigation_enabled_topic", "motion_mode/navigation_enabled"
        ),
        "navigation_mode_ack_topic": motion_mode_config.get(
            "navigation_mode_ack_topic", "motion_mode/base_navigation_enabled"
        ),
    }

    nodes.append(
        Node(
            package="robot_navigation",
            executable="cmd_vel_bridge_node",
            name="cmd_vel_bridge",
            output="screen",
            parameters=[bridge_params],
        )
    )
    logger.info(f"Added cmd_vel_bridge node (publish_tf: {bridge_publish_tf}, publish_odom: {bridge_publish_odom})")

    return nodes
