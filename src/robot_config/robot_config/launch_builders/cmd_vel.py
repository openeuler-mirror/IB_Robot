"""CmdVel bridge builder for navigation.

Generates the cmd_vel_bridge_node that translates ROS cmd_vel to wheel commands.
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.cmd_vel")


def generate_cmd_vel_nodes(
    nav_config: dict[str, Any],
    ekf_enabled: bool = False,
    use_sim: bool = False,
) -> list:
    """Generate cmd_vel bridge nodes.

    Args:
        nav_config: navigation section from robot_config YAML
        ekf_enabled: whether EKF is enabled (disables bridge TF publishing)
        use_sim: simulation mode flag

    Returns:
        List of Node actions
    """
    nodes = []

    bridge_config = nav_config.get("cmd_vel_bridge", {})
    if not bridge_config.get("enabled", False) or use_sim:
        return nodes

    # Use publish_tf from config; default to False when EKF is enabled
    bridge_publish_tf = bridge_config.get("publish_tf", not ekf_enabled)

    bridge_params = {
        "wheel_radius": bridge_config.get("wheel_radius", 0.05),
        "base_radius": bridge_config.get("base_radius", 0.125),
        "max_radps": bridge_config.get("max_radps", 4.602),
        "odom_frame": bridge_config.get("odom_frame", "odom"),
        "base_frame": bridge_config.get("base_frame", "base_link"),
        "publish_tf": bridge_publish_tf,
        "control_frequency": bridge_config.get("control_frequency", 50.0),
        "cmd_timeout": bridge_config.get("cmd_timeout", 0.5),
        "cmd_vel_topic": bridge_config.get("cmd_vel_topic", "/cmd_vel"),
        "joint_states_topic": bridge_config.get("joint_states_topic", "/joint_states"),
        "odom_topic": bridge_config.get("odom_topic", "/odom"),
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
    logger.info(f"Added cmd_vel_bridge node (publish_tf: {bridge_publish_tf})")

    return nodes
