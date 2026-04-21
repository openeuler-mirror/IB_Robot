"""Navigation launch builder for robot_config.

Supports two navigation architectures selected by config structure:

1. **Sim-style** (``navigation.modes`` key present):
   Lightweight EKF-only node launched from a mode config path.
   Used by simulation robots (so101, etc.).

2. **Real-hardware stack** (``navigation.ekf_rtabmap`` or ``navigation.nav2_bringup``):
   Full navigation stack with sub-builders:
   - static_tf: static TF publishers
   - localization: RTAB-Map + EKF
   - nav2: Nav2 bringup
   - cmd_vel: cmd_vel bridge
   - voice_nav: Nav2 goal client + FunASR
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool, resolve_ros_path

logger = get_colored_logger("robot_config.navigation")


def generate_navigation_nodes(
    robot_config: dict[str, Any],
    use_sim: bool = False,
    navigation_mode: str = "",
    force_enable: bool = False,
) -> list:
    """Generate navigation nodes from robot_config.

    Dispatches to the appropriate sub-builder based on config structure.

    Args:
        robot_config: Robot configuration dictionary
        use_sim: Whether simulation mode is enabled
        navigation_mode: Override navigation mode (sim-style only)
        force_enable: Force enable navigation even if not configured

    Returns:
        List of navigation nodes and launch actions
    """
    navigation_config = robot_config.get("navigation", {})

    # Determine which config architecture is in use
    if "modes" in navigation_config:
        return _generate_sim_navigation(navigation_config, use_sim, navigation_mode, force_enable)

    return _generate_real_navigation(robot_config, navigation_config, use_sim)


# ---------------------------------------------------------------------------
# Sim-style navigation (lightweight EKF)
# ---------------------------------------------------------------------------


def _generate_sim_navigation(
    navigation_config: dict,
    use_sim: bool,
    navigation_mode: str,
    force_enable: bool,
) -> list:
    """Sim-style navigation: single EKF node from mode config path."""
    if not force_enable and not navigation_config.get("enabled", False):
        return []

    resolved_mode = navigation_mode or navigation_config.get("default_mode", "")
    mode_configs = navigation_config.get("modes", {})
    if not resolved_mode:
        raise ValueError("robot.navigation.default_mode must be set when navigation is enabled")
    if resolved_mode not in mode_configs:
        raise ValueError(f"Unknown navigation mode '{resolved_mode}'. Available: {list(mode_configs.keys())}")

    mode_config = mode_configs[resolved_mode]
    config_path = mode_config.get("config")
    if not config_path:
        raise ValueError(f"Navigation mode '{resolved_mode}' is missing a config path")

    node_parameters = [resolve_ros_path(config_path)]
    if parse_bool(use_sim, default=False):
        node_parameters.append({"use_sim_time": True})

    return [
        Node(
            package=navigation_config.get("package", "robot_localization"),
            executable=navigation_config.get("executable", "ekf_node"),
            name=navigation_config.get("node_name", "ekf_node"),
            output="screen",
            parameters=node_parameters,
        )
    ]


# ---------------------------------------------------------------------------
# Real-hardware navigation (full stack)
# ---------------------------------------------------------------------------


def _generate_real_navigation(
    robot_config: dict[str, Any],
    navigation_config: dict,
    use_sim: bool,
) -> list:
    """Real-hardware navigation: full stack with sub-builders."""
    if not navigation_config.get("enabled", False):
        logger.info("Navigation disabled, skipping")
        return []

    logger.info("Navigation enabled, generating nodes...")
    nodes = []

    # Lazy imports for sub-builders (only needed in real-hardware mode)
    from robot_config.launch_builders.cmd_vel import generate_cmd_vel_nodes
    from robot_config.launch_builders.localization import generate_localization_nodes
    from robot_config.launch_builders.nav2 import generate_nav2_nodes
    from robot_config.launch_builders.static_tf import generate_static_tf_nodes
    from robot_config.launch_builders.voice_nav import generate_voice_nav_nodes

    # 1. Static TF publishers (base frames)
    nodes.extend(generate_static_tf_nodes(navigation_config, use_sim=use_sim))

    # 2. Localization (RTAB-Map + EKF)
    nodes.extend(generate_localization_nodes(navigation_config, use_sim=use_sim))

    # 3. Nav2 bringup
    nodes.extend(generate_nav2_nodes(navigation_config, use_sim=use_sim))

    # 4. CmdVel bridge (needs to know if EKF is running)
    ekf_rtabmap_config = navigation_config.get("ekf_rtabmap", {})
    ekf_enabled = ekf_rtabmap_config.get("enabled", False) and not use_sim
    nodes.extend(generate_cmd_vel_nodes(navigation_config, ekf_enabled=ekf_enabled, use_sim=use_sim))

    # 5. Voice-controlled navigation (Nav2 goal client + FunASR)
    nodes.extend(generate_voice_nav_nodes(navigation_config, use_sim=use_sim))

    # 6. Robot evaluate (inference evaluation mode)
    evaluate_config = navigation_config.get("robot_evaluate", {})
    if evaluate_config.get("enabled", False):
        evaluate_params = {
            "robot_id": evaluate_config.get("robot_id", "lekiwi"),
            "remote_ip": evaluate_config.get("remote_ip", "192.168.1.99"),
            "hf_model_id": evaluate_config.get("hf_model_id", ""),
            "hf_dataset_id": evaluate_config.get("hf_dataset_id", ""),
            "model_type": evaluate_config.get("model_type", "act"),
            "enable_stable_mode": evaluate_config.get("enable_stable_mode", False),
        }
        nodes.append(
            Node(
                package="robot_evaluate",
                executable="robot_evaluate_node",
                name="robot_evaluate_node",
                output="screen",
                parameters=[evaluate_params],
            )
        )
        logger.info(f"Added robot_evaluate_node (model: {evaluate_params['hf_model_id']})")

    # 7. RViz2 visualization
    rviz_config = navigation_config.get("rviz", {})
    if rviz_config.get("enabled", False):
        nodes.extend(_generate_rviz_nodes(rviz_config, use_sim=use_sim))

    logger.info(f"Total navigation nodes: {len(nodes)}")
    return nodes


def _generate_rviz_nodes(rviz_config: dict[str, Any], use_sim: bool = False) -> list:
    """Generate RViz2 visualization node."""
    import os

    from ament_index_python.packages import get_package_share_directory

    rviz_config_file = rviz_config.get("config_file", "")
    if not rviz_config_file:
        try:
            robot_navigation_share = get_package_share_directory("robot_navigation")
            rviz_config_file = os.path.join(robot_navigation_share, "config", "config.rviz")
        except Exception:
            rviz_config_file = ""

    if not rviz_config_file:
        return []

    node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": use_sim}],
        output="screen",
    )
    logger.info(f"Added rviz2 (config: {rviz_config_file})")
    return [node]
