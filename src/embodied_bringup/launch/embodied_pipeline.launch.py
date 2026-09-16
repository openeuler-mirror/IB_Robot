"""Launch the base robot stack plus embodied runtime nodes."""

import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from embodied_bringup.launch_builders.embodied import _resolve_development_source_root, generate_embodied_nodes
from robot_config.interface_binding import required_interface_ids
from robot_config.loader import load_robot_config_dict, validate_embodied_launch_dict
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool, resolve_ros_path

logger = get_colored_logger("embodied_bringup.launch")


def _required_controllers(config: dict, active_control_mode: str, use_sim: bool) -> list[str]:
    if use_sim and str(config.get("simulation", {}).get("platform", "gazebo")).lower() == "mock":
        return []
    control_modes = config.get("control_modes", {})
    mode_config = control_modes.get(active_control_mode, {}) if isinstance(control_modes, dict) else {}
    if use_sim and mode_config.get("sim_controllers") is not None:
        return list(mode_config.get("sim_controllers", []))
    if not use_sim and mode_config.get("hardware_controllers") is not None:
        return list(mode_config.get("hardware_controllers", []))
    controllers = mode_config.get("controllers")
    if controllers is not None:
        return list(controllers)
    return list(config.get("ros2_control", {}).get("controllers", []))


def _controller_startup_timeout(config: dict, use_sim: bool) -> float:
    configured = config.get("controller_startup_timeout", 120.0 if use_sim else 30.0)
    if isinstance(configured, dict):
        configured = configured.get("sim" if use_sim else "hardware", 120.0 if use_sim else 30.0)
    timeout = float(configured)
    if timeout <= 0.0:
        raise ValueError("controller_startup_timeout must be greater than zero")
    return timeout


def _controller_ready_waiter(config: dict, active_control_mode: str, use_sim: bool, auto_start: bool):
    if not auto_start:
        return None
    required = _required_controllers(config, active_control_mode, use_sim)
    if not required:
        return None
    timeout = _controller_startup_timeout(config, use_sim)
    return Node(
        package="robot_config",
        executable="wait_for_controllers",
        name="wait_for_embodied_controllers",
        arguments=[
            *required,
            "--controller-manager",
            "controller_manager",
            "--timeout",
            str(timeout),
            "--service-wait-timeout",
            str(min(timeout, 5.0)),
        ],
        output="screen",
    )


def _start_runtime_after_controller_readiness(runtime_actions):
    frozen_actions = tuple(runtime_actions)

    def _handler(event, _context):
        if event.returncode == 0:
            logger.info("Controllers are active; starting embodied runtime and IK workers")
            return list(frozen_actions)
        reason = f"Embodied controller readiness failed (returncode={event.returncode})"
        logger.error(reason)
        return [EmitEvent(event=Shutdown(reason=reason))]

    return _handler


def _load_config(robot_config_name: str, config_path_override: str, nav_stage: str = "") -> dict:
    try:
        robot_config_share = get_package_share_directory("robot_config")
    except Exception:
        robot_config_share = str(Path(__file__).parents[2] / "robot_config")

    config_path = (
        Path(config_path_override)
        if config_path_override
        else Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml"
    )
    config = load_robot_config_dict(config_path, nav_stage=nav_stage, defer_interface_binding=True)
    config["_config_path"] = str(config_path)
    return config


def launch_setup(context, *_args, **_kwargs):
    robot_config_name = context.launch_configurations.get("robot_config", "so101_single_arm")
    config_path_override = context.launch_configurations.get("config_path", "")
    control_mode_override = context.launch_configurations.get("control_mode", "")
    nav_stage = context.launch_configurations.get("nav_stage", "").strip()
    with_embodied_str = context.launch_configurations.get("with_embodied", "")
    with_perception_str = context.launch_configurations.get("with_perception", "")
    entry_mode_override = context.launch_configurations.get("entry_mode", "")
    authorize_motion_str = context.launch_configurations.get("authorize_motion", "false")

    config = _load_config(robot_config_name, config_path_override, nav_stage)
    if control_mode_override:
        config["default_control_mode"] = control_mode_override

    embodied_config = config.setdefault("embodied", {})
    embodied_config["enabled"] = (
        bool(embodied_config.get("enabled", False))
        if with_embodied_str == ""
        else parse_bool(with_embodied_str, default=True)
    )
    if entry_mode_override:
        embodied_config["entry_mode"] = entry_mode_override
    if with_perception_str != "":
        perception_config = embodied_config.setdefault("perception", {})
        perception_config["enabled"] = parse_bool(with_perception_str, default=False)

    # Fail fast on inconsistent launch overrides (e.g. a visual game enabled while
    # with_perception:=false) instead of starting a node graph that routes to a
    # dead topic. Reuses the same rules as robot_config.validate_config.
    launch_errors = validate_embodied_launch_dict(config)
    if launch_errors:
        for error in launch_errors:
            logger.error(f"Invalid embodied launch configuration: {error}")
        raise RuntimeError("embodied launch configuration is inconsistent: " + "; ".join(launch_errors))

    active_control_mode = config.get("default_control_mode", "moveit_planning")
    motion_authorized = parse_bool(authorize_motion_str, default=False)
    base_launch_path = Path(get_package_share_directory("robot_config")) / "launch" / "robot.launch.py"
    requested_moveit = context.launch_configurations.get("with_moveit", "")
    if (
        requested_moveit == ""
        and config.get("nav_stage") == "hybrid"
        and active_control_mode == "base_navigation"
        and "moveit" in str(config.get("skill_required_control_mode", "")).lower()
    ):
        requested_moveit = "true"
    base_launch_arguments = {
        "robot_config": robot_config_name,
        "config_path": config_path_override,
        "use_sim": context.launch_configurations.get("use_sim", "false"),
        "sim_platform": context.launch_configurations.get("sim_platform", ""),
        "use_mock": context.launch_configurations.get("use_mock", "false"),
        "auto_start_controllers": context.launch_configurations.get("auto_start_controllers", "true"),
        "control_mode": active_control_mode,
        "nav_stage": nav_stage,
        "with_moveit": requested_moveit,
        "moveit_display": context.launch_configurations.get("moveit_display", "false"),
        "with_embodied": "false",
        "with_perception": "false",
    }

    if config.get("runtime", {}).get("provider") or required_interface_ids(config):
        # Base launch owns normalization, provider startup and the live snapshot.
        # Keep embodied settings in that snapshot; the base has no embodied builders.
        del base_launch_arguments["with_embodied"]
        del base_launch_arguments["with_perception"]
        base_launch_arguments["config_path"] = config["_config_path"]
        catalog_root = embodied_config.get("skill_catalog_source_root", "")
        if catalog_root and not Path(catalog_root).is_absolute():
            resolved_root = _resolve_development_source_root(Path(config["_config_path"]), catalog_root)
            if resolved_root is not None:
                embodied_config["skill_catalog_source_root"] = str(resolved_root)
        spec = importlib.util.spec_from_file_location("embodied_base_robot_launch", base_launch_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load base robot launch: {base_launch_path}")
        base_launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_launch)

        return [
            GroupAction(
                # Match include argument lifetime for the asynchronous readiness callback.
                scoped=False,
                actions=[
                    *[SetLaunchConfiguration(name, value) for name, value in base_launch_arguments.items()],
                    OpaqueFunction(
                        function=base_launch.launch_setup,
                        kwargs={
                            "loaded_config": config,
                            "extra_consumers": lambda effective: _construct_embodied_actions(
                                effective,
                                effective.get("default_control_mode", "moveit_planning"),
                                motion_authorized=motion_authorized,
                                use_sim=parse_bool(base_launch_arguments["use_sim"], default=False),
                                # Runtime readiness replaces the legacy controller barrier.
                                auto_start=False,
                            ),
                        },
                    ),
                ],
            )
        ]

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(base_launch_path)),
            launch_arguments=base_launch_arguments.items(),
        ),
        *_construct_embodied_actions(
            config,
            active_control_mode,
            motion_authorized=motion_authorized,
            use_sim=parse_bool(base_launch_arguments["use_sim"], default=False),
            auto_start=parse_bool(base_launch_arguments["auto_start_controllers"], default=True),
        ),
    ]


def _parallel_ik_worker_action(config: dict, use_sim_time: str):
    if config.get("runtime", {}).get("provider"):
        return None
    grasp = config.get("grasp_execution", {})
    if not grasp.get("enabled", False) or not grasp.get("auto_start_dependencies", True):
        return None
    ik = grasp.get("ik", {})
    count = int(ik.get("worker_count", 0))
    if count <= 0 or not ik.get("auto_start_workers", True):
        return None
    if count > 8:
        raise ValueError("grasp_execution.ik.worker_count must be between 0 and 8")
    prefix = str(ik.get("worker_namespace_prefix", "/ik_worker")).strip("/")
    if not prefix:
        raise ValueError("grasp_execution.ik.worker_namespace_prefix must not be empty")
    # Providerless (legacy) bring-up: the IK/FK worker pool is a robot-suite
    # launch file; the generic bringup only includes what the YAML names.
    workers_launch = resolve_ros_path(str(ik.get("workers_launch", "") or "")).strip()
    if not workers_launch:
        raise ValueError(
            "grasp_execution.ik.workers_launch is required when auto-starting IK workers without "
            "runtime.provider (e.g. $(find so101_motion)/launch/ik_workers.launch.py)"
        )
    path = Path(workers_launch)
    if not path.is_file():
        raise FileNotFoundError(f"grasp_execution.ik.workers_launch not found: {path}")
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(path)),
        launch_arguments={
            "worker_count": str(count),
            "namespace_prefix": prefix,
            "use_sim_time": use_sim_time,
            "joint_state_topic": str(config.get("moveit", {}).get("joint_state_topic", "/joint_states")),
        }.items(),
    )


def _construct_embodied_actions(config, active_control_mode, *, motion_authorized, use_sim, auto_start):
    actions = []
    embodied_config = config["embodied"]
    if embodied_config["enabled"]:
        logger.info("Preparing embodied runtime nodes from embodied_bringup")
        visual_games = embodied_config.get("visual_games", {})
        visual_games_enabled = isinstance(visual_games, dict) and any(
            isinstance(policy, dict) and policy.get("enabled") is True for policy in visual_games.values()
        )
        visual_actions = generate_embodied_nodes(
            config,
            active_control_mode,
            motion_authorized=motion_authorized,
            include_motion=False,
            include_perception=visual_games_enabled,
            use_sim=use_sim,
        )
        actions.extend(visual_actions)
        runtime_actions = []
        worker = _parallel_ik_worker_action(config, str(use_sim).lower())
        if worker is not None:
            runtime_actions.append(worker)
        runtime_actions.extend(
            generate_embodied_nodes(
                config,
                active_control_mode,
                motion_authorized=motion_authorized,
                include_visual_games=False,
                include_perception=not visual_games_enabled,
                use_sim=use_sim,
            )
        )
        ready_waiter = _controller_ready_waiter(config, active_control_mode, use_sim, auto_start)
        if ready_waiter is None:
            actions.extend(runtime_actions)
        else:
            actions.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=ready_waiter,
                        on_exit=_start_runtime_after_controller_readiness(runtime_actions),
                    )
                )
            )
            actions.append(ready_waiter)
            logger.info(
                f"Started {len(visual_actions)} visual runtime action(s); "
                f"deferring {len(runtime_actions)} motion runtime action(s) until controllers are active"
            )
    else:
        logger.info("Embodied runtime disabled by with_embodied:=false")
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            # Fast DDS shared-memory segments can remain orphaned on the
            # OpenHarmony board after a pipeline is stopped.  In that state a
            # Python ActionServer may publish feedback/status while its service
            # endpoints are not discoverable.  Keep the launch deterministic by
            # using UDPv4 for the whole graph; all nodes inherit the same
            # transport and action services remain discoverable after restart.
            SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
            DeclareLaunchArgument("robot_config", default_value="so101_single_arm"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("use_sim", default_value="false"),
            DeclareLaunchArgument("sim_platform", default_value=""),
            DeclareLaunchArgument("use_mock", default_value="false"),
            DeclareLaunchArgument("auto_start_controllers", default_value="true"),
            DeclareLaunchArgument("control_mode", default_value=""),
            DeclareLaunchArgument("nav_stage", default_value=""),
            DeclareLaunchArgument("with_moveit", default_value=""),
            DeclareLaunchArgument("moveit_display", default_value="false"),
            DeclareLaunchArgument("with_embodied", default_value=""),
            DeclareLaunchArgument("with_perception", default_value=""),
            DeclareLaunchArgument("entry_mode", default_value=""),
            DeclareLaunchArgument("authorize_motion", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
