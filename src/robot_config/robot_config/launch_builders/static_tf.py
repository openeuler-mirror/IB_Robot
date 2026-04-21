"""Static TF publisher builder for navigation.

Generates static_transform_publisher nodes from navigation config.
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.static_tf")


def generate_static_tf_nodes(
    nav_config: dict[str, Any],
    extra_tfs: list[dict[str, Any]] = None,
    use_sim: bool = False,
) -> list:
    """Generate static TF publisher nodes.

    Args:
        nav_config: navigation section from robot_config YAML
        extra_tfs: additional TF configs to append (e.g. camera TF from localization)
        use_sim: simulation mode flag

    Returns:
        List of Node actions
    """
    nodes = []

    static_tfs = nav_config.get("static_tfs", [])
    if extra_tfs:
        static_tfs = static_tfs + extra_tfs

    for tf_config in static_tfs:
        tf_name = tf_config.get("name", "static_tf")
        parent = tf_config.get("parent_frame", "")
        child = tf_config.get("child_frame", "")
        translation = tf_config.get("translation", [0, 0, 0])
        rotation = tf_config.get("rotation", [0, 0, 0])

        args = [
            str(translation[0]),
            str(translation[1]),
            str(translation[2]),
            str(rotation[0]),
            str(rotation[1]),
            str(rotation[2]),
            parent,
            child,
        ]

        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=tf_name,
                parameters=[{"use_sim_time": use_sim}],
                arguments=args,
            )
        )
        logger.info(f"Added static TF: {parent} -> {child} ({tf_name})")

    return nodes
