"""Voice ASR + Nav2 goal client builder for navigation.

Generates:
- nav2_goal_client: Nav2 goal action client
- funasr_client_node: FunASR speech-to-text client for voice-controlled navigation
"""

import json
import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.voice_nav")


def generate_voice_nav_nodes(nav_config: dict[str, Any], use_sim: bool = False) -> list:
    """Generate voice-controlled navigation nodes.

    Args:
        nav_config: navigation section from robot_config YAML
        use_sim: simulation mode flag

    Returns:
        List of Node actions
    """
    nodes = []

    robot_navigation_config = nav_config.get("robot_navigation", {})
    if not robot_navigation_config.get("enabled", True):
        return nodes

    nav2_client_params = {
        "use_sim_time": use_sim,
        "enable_voice_control": robot_navigation_config.get("enable_voice_control", True),
        "global_frame": robot_navigation_config.get("global_frame", "map"),
        "enable_feedback": robot_navigation_config.get("enable_feedback", True),
        "trigger_evaluation": robot_navigation_config.get("trigger_evaluation", False),
    }

    nodes.append(
        Node(
            package="robot_navigation",
            executable="nav2_goal_client",
            name="nav2_goal_client",
            output="screen",
            parameters=[nav2_client_params],
        )
    )
    logger.info(
        f"Added nav2_goal_client node "
        f"(voice_control: {nav2_client_params['enable_voice_control']}, "
        f"use_sim_time: {use_sim})"
    )

    # FunASR client (when voice control enabled)
    if robot_navigation_config.get("enable_voice_control", True):
        funasr_node = _create_funasr_client_node(nav_config, robot_navigation_config)
        if funasr_node:
            nodes.append(funasr_node)

    return nodes


def _create_funasr_client_node(
    nav_config: dict[str, Any],
    robot_navigation_config: dict[str, Any],
) -> Node:
    """Create FunASR client node for voice recognition."""
    # Resolve keywords file path
    keywords_file = robot_navigation_config.get("keywords_file", "")
    if keywords_file:
        keywords_file = resolve_ros_path(keywords_file)
    else:
        try:
            robot_navigation_share = get_package_share_directory("robot_navigation")
            keywords_file = os.path.join(robot_navigation_share, "config", "keywords.json")
        except Exception:
            keywords_file = ""

    # Load keywords JSON
    keywords_json = "{}"
    if keywords_file:
        try:
            with open(keywords_file, encoding="utf-8") as f:
                keywords_json = f.read()
        except Exception as e:
            logger.warning(f"Failed to read keywords file: {e}")

    # Extract destinations from config
    destinations = robot_navigation_config.get("destinations", {})
    destinations_json = json.dumps(destinations) if destinations else "{}"

    # FunASR connection parameters
    funasr_config = nav_config.get("funasr", {})
    funasr_params = {
        "host": funasr_config.get("host", "127.0.0.1"),
        "port": funasr_config.get("port", "10095"),
        "mode": funasr_config.get("mode", "2pass"),
        "hotword_msg": funasr_config.get("hotword_msg", ""),
        "keywords_json": keywords_json,
        "destinations_json": destinations_json,
        "global_frame": robot_navigation_config.get("global_frame", "map"),
        "topic_text": "/voice_asr/text",
        "topic_status": "/voice_asr/status",
        "topic_keyword_matched": "/voice_asr/keyword_matched",
        "topic_nav_stop": "/voice_asr/nav_stop",
    }

    node = Node(
        package="robot_navigation",
        executable="funasr_client_node",
        name="funasr_client_node",
        output="screen",
        parameters=[funasr_params],
    )
    logger.info(
        f"Added funasr_client_node "
        f"(host: {funasr_params['host']}:{funasr_params['port']}, "
        f"mode: {funasr_params['mode']}, "
        f"{len(destinations)} destinations)"
    )
    return node
