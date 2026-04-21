import json
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_node(context, *args, **kwargs):
    """Generate node with evaluated parameters."""
    host = LaunchConfiguration("host").perform(context)
    port = LaunchConfiguration("port").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    hotword_msg = LaunchConfiguration("hotword_msg").perform(context)
    keywords_file = LaunchConfiguration("keywords_file").perform(context)
    global_frame = LaunchConfiguration("global_frame").perform(context)
    robot_config_file = LaunchConfiguration("robot_config_file").perform(context)

    # Read keywords from file if specified
    keywords_json = "{}"
    if keywords_file and keywords_file != "":
        try:
            with open(keywords_file) as f:
                keywords_json = f.read()
        except Exception as e:
            print(f"Warning: Failed to read keywords file: {e}")

    # Read destinations from robot config YAML
    destinations_json = "{}"
    if robot_config_file and robot_config_file != "":
        try:
            with open(robot_config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            nav_config = config.get("robot", {}).get("navigation", {}).get("robot_navigation", {})
            destinations = nav_config.get("destinations", {})
            if destinations:
                destinations_json = json.dumps(destinations)
        except Exception as e:
            print(f"Warning: Failed to read destinations from config: {e}")

    return [
        Node(
            package="robot_navigation",
            executable="funasr_client_node",
            name="funasr_client_node",
            output="screen",
            parameters=[
                {
                    "host": host,
                    "port": port,
                    "mode": mode,
                    "hotword_msg": hotword_msg,
                    "keywords_json": keywords_json,
                    "destinations_json": destinations_json,
                    "global_frame": global_frame,
                    "topic_text": "/voice_asr/text",
                    "topic_status": "/voice_asr/status",
                    "topic_keyword_matched": "/voice_asr/keyword_matched",
                    "topic_nav_stop": "/voice_asr/nav_stop",
                }
            ],
        )
    ]


def generate_launch_description():
    # Get package directories
    pkg_dir = get_package_share_directory("robot_navigation")
    default_keywords_file = os.path.join(pkg_dir, "config", "keywords.json")

    # Default robot config file from robot_config package
    try:
        robot_config_dir = get_package_share_directory("robot_config")
        default_robot_config_file = os.path.join(robot_config_dir, "config", "robots", "lekiwi_navi.yaml")
    except Exception:
        default_robot_config_file = ""

    # Declare launch arguments
    host_arg = DeclareLaunchArgument("host", default_value="127.0.0.1", description="FunASR server host")
    port_arg = DeclareLaunchArgument("port", default_value="10095", description="FunASR server port")
    mode_arg = DeclareLaunchArgument(
        "mode", default_value="2pass", description="Recognition mode (2pass, online, offline)"
    )
    hotword_arg = DeclareLaunchArgument("hotword_msg", default_value="", description="Hotword message for FunASR")
    keywords_file_arg = DeclareLaunchArgument(
        "keywords_file", default_value=default_keywords_file, description="Path to keywords JSON config file"
    )
    global_frame_arg = DeclareLaunchArgument(
        "global_frame", default_value="map", description="Global frame for navigation"
    )
    robot_config_arg = DeclareLaunchArgument(
        "robot_config_file",
        default_value=default_robot_config_file,
        description="Path to robot config YAML file for destinations",
    )

    # Create node using OpaqueFunction to evaluate LaunchConfiguration
    funasr_node = OpaqueFunction(function=generate_node)

    return LaunchDescription(
        [
            host_arg,
            port_arg,
            mode_arg,
            hotword_arg,
            keywords_file_arg,
            global_frame_arg,
            robot_config_arg,
            funasr_node,
        ]
    )
