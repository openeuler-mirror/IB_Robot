"""Navigation launch builder for robot_config.

This module is ``robot_navigation``'s contribution to the launch-builder
registry (openspec change ``migrate-business-launch-builders``): the
``navigation`` entry is declared as an entry point in the
``ibrobot.launch_builders`` group (see ``setup.py``), so the navigation
stack's owning package registers its own launch wiring without any
robot_config change. The content is a verbatim migration of the former
``robot_config/launch_builders/navigation.py``; the only mechanical changes
are the module paths (self-import, and the sibling sub-builders nav2 /
localization / navigation_command / voice_nav, which moved into this
package too). The shared helpers stay in robot_config and keep being
imported from there (``fast_lio`` / ``static_tf`` / the runtime provider
lookup).

Supports two navigation architectures selected by config structure:

1. **Sim-style** (``navigation.modes`` key present):
   Lightweight EKF-only node launched from a mode config path.
   Used by simulation robots (so101, etc.).

2. **Real-hardware stack** (``navigation.ekf_rtabmap`` or ``navigation.nav2_bringup``):
   Full navigation stack with sub-builders:
   - static_tf: static TF publishers
   - localization: RTAB-Map + EKF
   - nav2: Nav2 bringup (collision monitor output lands on the admitted
     /cmd_vel channel; the velocity smoother is dropped while it runs)
   - admission: launch-time mode/gate admission for provider stages that
     declare navigation.open_gate_on_startup
   - voice_nav: Nav2 goal client + FunASR
"""

from typing import Any

from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.launch_registry.builder_support import resolve_with_navigation
from robot_config.launch_registry.launch_options import ArgumentSpec
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
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

    if "nav_stage" in robot_config:
        _validate_global_localization_provider(robot_config["nav_stage"], navigation_config)
    _validate_navigation_velocity_contract(robot_config, navigation_config)

    logger.info("Navigation enabled, generating nodes...")
    nodes = []

    # Under a runtime provider the lidar odometry chain belongs to the robot
    # runtime: launch_builders.runtime hands the stage's fast_lio section to
    # the provider, which launches FAST-LIO and its odom bridge. Launching them
    # here too would run two FAST-LIO instances and two odom -> base_link
    # publishers.
    from robot_config.launch_builders.runtime import runtime_provider

    provider = runtime_provider(robot_config)
    if provider:
        logger.info("runtime.provider configured: FAST-LIO left to the robot runtime")
    else:
        from robot_config.launch_builders.fast_lio import generate_fast_lio_nodes

        nodes.extend(generate_fast_lio_nodes(navigation_config, use_sim=use_sim))

    # Lazy imports for sub-builders (only needed in real-hardware mode)
    from robot_config.launch_builders.static_tf import generate_static_tf_nodes
    from robot_navigation.launch_builders.localization import generate_localization_nodes
    from robot_navigation.launch_builders.nav2 import generate_nav2_nodes
    from robot_navigation.launch_builders.navigation_command import generate_navigation_command_nodes
    from robot_navigation.launch_builders.voice_nav import generate_voice_nav_nodes

    # 1. Static TF publishers (base frames)
    nodes.extend(generate_static_tf_nodes(navigation_config, use_sim=use_sim))

    # 2. Localization (RTAB-Map + EKF)
    nodes.extend(generate_localization_nodes(navigation_config, use_sim=use_sim))

    # 3. Nav2 bringup (the collision monitor routes onto the runtime channel)
    nodes.extend(generate_nav2_nodes(navigation_config, use_sim=use_sim))
    nodes.extend(generate_navigation_command_nodes(navigation_config, use_sim=use_sim))

    # 4. Launch-time navigation admission (openspec
    #    lekiwi-application-migration A3/A5): a stage declaring
    #    navigation.open_gate_on_startup may drive the base from boot. The
    #    runtime boots idle with the gate closed, so an admission client
    #    selects the stage's runtime mode and opens the gate through the
    #    contract services after the runtime is ready; a rejected or timed-out
    #    admission shuts the launch down instead of leaving a stage that
    #    silently cannot move.
    if provider and navigation_config.get("open_gate_on_startup", False):
        mode_name = str(robot_config.get("default_control_mode", "") or "")
        runtime_mode = str(
            ((robot_config.get("control_modes") or {}).get(mode_name) or {}).get("runtime_mode", "")
        ).strip()
        if not runtime_mode:
            raise ValueError(
                f"navigation.open_gate_on_startup requires control_modes.{mode_name!r}.runtime_mode "
                "to name the runtime mode this stage drives the base in"
            )
        admission = Node(
            package="robot_navigation",
            executable="navigation_admission",
            name="navigation_admission",
            output="screen",
            parameters=[{"runtime_mode": runtime_mode, "open_gate": True, "use_sim_time": use_sim}],
        )
        nodes.append(admission)
        nodes.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=admission,
                    on_exit=lambda event, _context: (
                        []
                        if event.returncode == 0
                        else [EmitEvent(Shutdown(reason=f"navigation admission failed (rc={event.returncode})"))]
                    ),
                )
            )
        )

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


def _validate_global_localization_provider(nav_stage: str, navigation_config: dict[str, Any]) -> None:
    """Require exactly one stage-appropriate map-to-odom authority."""
    slam_enabled = navigation_config.get("slam_toolbox", {}).get("enabled", False)
    nav2_config = navigation_config.get("nav2_bringup", {})
    nav2_enabled = nav2_config.get("enabled", False)
    amcl_enabled = nav2_enabled and nav2_config.get("use_amcl", False)
    valid = {
        "mapping": slam_enabled and not nav2_enabled,
        "navigation": not slam_enabled and amcl_enabled,
        "hybrid": not slam_enabled and amcl_enabled,
    }

    if not valid.get(nav_stage, False):
        raise ValueError(
            f"nav_stage '{nav_stage}' must enable exactly one matching global localization provider "
            "(mapping=slam_toolbox, navigation=Nav2 AMCL)"
        )


def _validate_navigation_velocity_contract(robot_config: dict[str, Any], navigation_config: dict[str, Any]) -> None:
    """Require the collision-monitor chain to land on the admitted base channel.

    With ``dyn_avoid_enabled`` the Nav2 collision monitor must be the last hop
    onto ``/cmd_vel`` (openspec lekiwi-application-migration task 3.2): the
    controller publishes ``/cmd_vel_nav`` and the monitor's filtered output is
    what the runtime admits. Any other velocity topic under dynamic avoidance
    would either bypass the monitor or never reach the base. The
    command server's stop topic shares the same channel so stops are admitted
    like any other producer.
    """
    nav2_config = navigation_config.get("nav2_bringup", {})
    if not nav2_config.get("dyn_avoid_enabled", False):
        return

    stop_topic = navigation_config.get("command_server", {}).get("stop_velocity_topic", "/cmd_vel_safe")
    if stop_topic != "/cmd_vel":
        raise ValueError(
            "dynamic navigation stop_velocity_topic must be /cmd_vel so command-server stops reach the base "
            "through the same admitted channel as the collision-monitor output"
        )
    if navigation_config.get("cmd_vel_bridge"):
        raise ValueError(
            "navigation.cmd_vel_bridge was removed; the runtime base node owns cmd_vel projection "
            "(openspec lekiwi-application-migration task 2.2)"
        )


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
    else:
        rviz_config_file = resolve_ros_path(rviz_config_file)

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


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration.
# ---------------------------------------------------------------------------

# Business launch arguments owned by this entry (migrated verbatim from the
# former builtin_registry table; ownership decides who declares an argument).
_NAVIGATION_ARGUMENTS = (
    ArgumentSpec(
        name="with_navigation",
        declared_default="",
        absent_fallback="",
        description="Enable navigation nodes. If empty, auto-detects from robot.navigation.enabled",
    ),
    ArgumentSpec(
        name="navigation_mode",
        declared_default="",
        absent_fallback="",
        description="Override robot.navigation.default_mode (for example: full, odom_only, imu_only)",
    ),
)


def build_navigation(context: LaunchBuildContext) -> BuildResult:
    """Navigation entry; runtime_ready timing.

    The mock backend and a false ``with_navigation`` resolution never
    reach this build function (registry-level filtering). ``force_enable=True``
    keeps the legacy generate semantics: the overall enablement decision
    was already made by the registry predicate.
    """
    from robot_navigation.launch_builders.navigation import generate_navigation_nodes

    nodes = generate_navigation_nodes(
        context.robot_config,
        use_sim=context.node_use_sim_time,
        navigation_mode=context.arguments.get("navigation_mode", ""),
        force_enable=True,
    )
    logger.info(f"Prepared {len(nodes)} navigation node(s)")
    return BuildResult(actions=list(nodes))


NAVIGATION_BUILDER = BuilderSpec(
    id="navigation",
    factory=f"{__name__}:build_navigation",
    timing="runtime_ready",
    mock_skip_key="navigation",
    arguments=_NAVIGATION_ARGUMENTS,
    # CLI switch (with its navigation.enabled YAML fallback)
    # owns overall participation; the build function no longer repeats
    # the resolution.
    enabled=resolve_with_navigation,
)
