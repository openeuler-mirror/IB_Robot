"""Localization builder for navigation.

Generates EKF + RTAB-Map stack nodes:
- RTAB-Map SLAM
- EKF sensor fusion
"""

import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.localization")


def generate_localization_nodes(
    nav_config: dict[str, Any],
    use_sim: bool = False,
) -> list:
    """Generate localization nodes (EKF + RTAB-Map stack).

    Args:
        nav_config: navigation section from robot_config YAML
        use_sim: simulation mode flag

    Returns:
        List of Node actions
    """
    nodes = []

    ekf_rtabmap_config = nav_config.get("ekf_rtabmap", {})
    ekf_rtabmap_enabled = ekf_rtabmap_config.get("enabled", False)

    if not (ekf_rtabmap_enabled and not use_sim):
        return nodes

    try:
        # RTAB-Map
        rtabmap_config = ekf_rtabmap_config.get("rtabmap", {})
        rtabmap_dir = get_package_share_directory("rtabmap_launch")
        rtabmap_args = {
            "use_sim_time": "false",
            "localization": str(rtabmap_config.get("localization", True)).lower(),
        }
        for key in [
            "rgb_topic",
            "depth_topic",
            "camera_info_topic",
            "database_path",
            "frame_id",
            "odom_frame_id",
            "rtabmap_args",
            "approx_sync",
            "queue_size",
            "log_level",
        ]:
            val = rtabmap_config.get(key)
            if val is not None:
                rtabmap_args[key] = str(val)

        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(rtabmap_dir, "launch", "rtabmap.launch.py")),
                launch_arguments=rtabmap_args.items(),
            )
        )
        logger.info(f"Added RTAB-Map launch (localization: {rtabmap_args.get('localization')})")

        # EKF node
        ekf_node_config = ekf_rtabmap_config.get("ekf", {})
        ekf_config_file = ekf_node_config.get("config_file", "")
        if not ekf_config_file:
            try:
                robot_navigation_share = get_package_share_directory("robot_navigation")
                ekf_config_file = os.path.join(robot_navigation_share, "config", "ekf.yaml")
            except Exception:
                ekf_config_file = ""

        if ekf_config_file:
            nodes.append(
                Node(
                    package="robot_localization",
                    executable="ekf_node",
                    name="ekf_filter_node",
                    output="screen",
                    parameters=[ekf_config_file],
                )
            )
            logger.info(f"Added EKF node (config: {ekf_config_file})")

        logger.info("EKF + RTAB-Map stack enabled")

    except Exception as e:
        logger.warning(f"Could not add EKF+RTAB-Map nodes: {e}")

    return nodes
