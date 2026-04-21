"""Nav2 bringup builder for navigation.

Generates Nav2 stack IncludeLaunchDescription from navigation config.
"""

import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.nav2")


def generate_nav2_nodes(
    nav_config: dict[str, Any],
    use_sim: bool = False,
) -> list:
    """Generate Nav2 bringup nodes.

    Args:
        nav_config: navigation section from robot_config YAML
        use_sim: simulation mode flag

    Returns:
        List of launch actions
    """
    nodes = []

    nav2_config = nav_config.get("nav2_bringup", {})
    if not nav2_config.get("enabled", False):
        return nodes

    try:
        nav2_bringup_dir = get_package_share_directory("nav2_bringup")
        nav2_launch_file = os.path.join(nav2_bringup_dir, "launch", "bringup_launch.py")

        # Resolve map file
        map_file = nav2_config.get("map_file", "")
        if not map_file:
            map_file = "~/workspace/map/rtabmap.yaml"

        # Resolve params file
        params_file = nav2_config.get("params_file", "")
        if not params_file:
            try:
                robot_navigation_share = get_package_share_directory("robot_navigation")
                params_file = os.path.join(robot_navigation_share, "config", "nav2_params.yaml")
            except Exception:
                params_file = ""

        launch_args = {
            "use_sim_time": str(use_sim).lower(),
            "map": map_file,
            "namespace": "",
            "use_namespace": "false",
            "use_composition": "False",
        }
        if params_file:
            launch_args["params_file"] = params_file

        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_launch_file),
                launch_arguments=launch_args.items(),
            )
        )
        logger.info(f"Added Nav2 bringup (map: {map_file}, sim: {use_sim})")

    except Exception as e:
        logger.warning(f"Could not add Nav2 bringup: {e}")

    return nodes
