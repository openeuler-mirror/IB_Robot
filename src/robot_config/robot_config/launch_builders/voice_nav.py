"""Voice ASR + Nav2 goal client builder for navigation.

Generates:
- nav2_goal_client: Nav2 goal action client
- voice_control: bridges voice_asr_service (sherpa-onnx) to nav2_goal_client
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

    # Voice control (when voice control enabled)
    if robot_navigation_config.get("enable_voice_control", True):
        vc_node = _create_voice_control_node(nav_config, robot_navigation_config)
        if vc_node:
            nodes.append(vc_node)

    return nodes


def _resolve_keywords_json(robot_navigation_config: dict[str, Any]) -> str:
    """Load keywords JSON from config file or parameter."""
    keywords_file = robot_navigation_config.get("keywords_file", "")
    if keywords_file:
        keywords_file = resolve_ros_path(keywords_file)
    else:
        try:
            robot_navigation_share = get_package_share_directory("robot_navigation")
            keywords_file = os.path.join(robot_navigation_share, "config", "keywords.json")
        except Exception:
            keywords_file = ""

    keywords_json = "{}"
    if keywords_file:
        try:
            with open(keywords_file, encoding="utf-8") as f:
                keywords_json = f.read()
        except Exception as e:
            logger.warning(f"Failed to read keywords file: {e}")

    return keywords_json


def _create_voice_control_node(
    nav_config: dict[str, Any],
    robot_navigation_config: dict[str, Any],
) -> Node:
    """Create voice_control node for keyword matching and navigation bridging."""
    keywords_json = _resolve_keywords_json(robot_navigation_config)

    # Extract destinations from config
    destinations = robot_navigation_config.get("destinations", {})
    destinations_json = json.dumps(destinations) if destinations else "{}"

    # Get voice_asr config for output topic
    voice_asr_config = nav_config.get("voice_asr", {})
    topic_text = voice_asr_config.get("output_topic", "/voice_command")

    bridge_params = {
        "topic_text": topic_text,
        "topic_keyword_matched": "/voice_asr/keyword_matched",
        "topic_nav_stop": "/voice_asr/nav_stop",
        "keywords_json": keywords_json,
        "destinations_json": destinations_json,
    }

    node = Node(
        package="robot_navigation",
        executable="voice_control",
        name="voice_control",
        output="screen",
        parameters=[bridge_params],
    )
    logger.info(f"Added voice_control (sub: {topic_text}, {len(destinations)} destinations)")
    return node
