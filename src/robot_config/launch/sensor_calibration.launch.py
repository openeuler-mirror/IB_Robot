"""Sensor launch with manual base control for extrinsic calibration.

This entry point owns the configured sensors, FAST-LIO, and the minimum
control chain needed for manual base teleoperation. It does not construct
Nav2, SLAM Toolbox, inference, or autonomous motion components.

Two shapes, selected by the configuration (openspec
lekiwi-application-migration task 1.5):

- ``runtime.provider`` set (the migrated ``lekiwi_sensor_calib``): the LeKiwi
  runtime owns the sensors, FAST-LIO, the chassis and the navigation gate. The
  deployment peripherals/fast_lio fragment and the launch-time admission step
  (base_navigation mode + open gate, per ``navigation.open_gate_on_startup``)
  are handed to the provider's ``runtime.launch.py``, and this entry waits for
  the runtime to be ready before returning.
- provider-less (legacy configurations): the sensors, FAST-LIO, ros2_control
  and the control chain are built here as before. The generic cmd_vel bridge
  is gone; a provider-less calibration configuration has no base drive.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from robot_config.launch_builders.control import generate_ros2_control_nodes
from robot_config.launch_builders.fast_lio import generate_fast_lio_nodes
from robot_config.launch_builders.perception import (
    generate_camera_nodes,
    generate_lidar_nodes,
    generate_tf_nodes,
    generate_virtual_camera_relays,
)
from robot_config.launch_builders.runtime import generate_runtime_provider_actions, runtime_provider
from robot_config.launch_builders.static_tf import generate_static_tf_nodes
from robot_config.loader import load_robot_config_dict
from robot_config.utils import resolve_ros_path


def _provider_calibration_actions(robot_config) -> list:
    """Delegate sensors, odometry, chassis and gate admission to the runtime."""
    provider_actions, waiter = generate_runtime_provider_actions(
        robot_config,
        use_sim=False,
        readiness_timeout_s=300.0,
    )
    # The runtime brings up base_navigation plus the open gate itself via the
    # admission client launched by the navigation builder; this entry point
    # only needs the runtime stack and the readiness waiter. The admission
    # node is appended here because the calibration launch builds no
    # navigation nodes of its own.
    navigation_config = robot_config.get("navigation", {}) or {}
    if navigation_config.get("open_gate_on_startup", False):
        from launch.actions import EmitEvent, RegisterEventHandler
        from launch.event_handlers import OnProcessExit
        from launch.events import Shutdown
        from launch_ros.actions import Node

        mode_name = str(robot_config.get("default_control_mode", "") or "")
        runtime_mode = str(
            ((robot_config.get("control_modes") or {}).get(mode_name) or {}).get("runtime_mode", "")
        ).strip()
        if not runtime_mode:
            raise ValueError(
                f"navigation.open_gate_on_startup requires control_modes.{mode_name!r}.runtime_mode "
                "to name the runtime mode calibration teleoperation drives the base in"
            )
        admission = Node(
            package="robot_navigation",
            executable="navigation_admission",
            name="navigation_admission",
            output="screen",
            parameters=[{"runtime_mode": runtime_mode, "open_gate": True}],
        )
        provider_actions.append(admission)
        provider_actions.append(
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
    return [*provider_actions, waiter]


def launch_setup(context, *args, **kwargs):
    config_path = context.launch_configurations.get("config_path", "")
    if not config_path:
        config_path = f"$(find robot_config)/config/robots/{context.launch_configurations['robot_config']}.yaml"
    robot_config = load_robot_config_dict(resolve_ros_path(config_path))

    if runtime_provider(robot_config):
        return _provider_calibration_actions(robot_config)

    actions = []
    actions.extend(generate_camera_nodes(robot_config, use_sim=False))
    actions.extend(generate_lidar_nodes(robot_config, use_sim=False))
    actions.extend(generate_tf_nodes(robot_config, use_sim=False))
    actions.extend(generate_virtual_camera_relays(robot_config))
    navigation_config = robot_config.get("navigation", {})
    actions.extend(generate_static_tf_nodes(navigation_config, use_sim=False))
    actions.extend(generate_fast_lio_nodes(navigation_config, use_sim=False))

    control_nodes, _controller_names, controller_spawners, _robot_description = generate_ros2_control_nodes(
        robot_config,
        use_sim=False,
        auto_start_controllers="true",
        controller_startup_timeout=300.0,
    )
    actions.extend(control_nodes)
    actions.extend(controller_spawners)
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="lekiwi_sensor_calib"),
            DeclareLaunchArgument("config_path", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
