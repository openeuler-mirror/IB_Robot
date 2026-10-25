"""LeKiwi runtime launch entry (robot-runtime-packaging spec).

robot_state_publisher + ros2_control (lekiwi_hardware, shared bus, real or
SDK simulation platform) + controllers + so101_motion for the arm + the
LeKiwi base node (cmd_vel, odometry, navigation gating) + sensor peripherals
(D12: profile defaults, deployment overrides via peripherals:=<file>) with
the fast_lio lidar odometry chain + runtime facade.

    ros2 launch lekiwi_robot runtime.launch.py profile:=<path-or-name> \
        [simulated:=true] [display:=true] [peripherals:=<fragment.yaml>]
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from lekiwi_robot.base_profile import arm_motion_model, base_node_parameters, lidar_odometry_owns_tf
from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import (
    motion_launch_arguments,
    peripheral_stack_actions,
    render_robot_description,
    resolve_profile_argument,
    resolve_sim_selection,
    runtime_stack_actions,
)
from robot_runtime.peripherals import fast_lio_nodes, load_peripherals_file, merge_peripherals
from robot_runtime.profile import load_profile


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("lekiwi_robot", LaunchConfiguration("profile").perform(context))
    profile = load_profile(profile_path)
    use_sim, sim_platform = resolve_sim_selection(
        profile,
        LaunchConfiguration("use_sim").perform(context),
        LaunchConfiguration("sim_platform").perform(context),
        LaunchConfiguration("simulated").perform(context),
        source=str(profile_path),
    )
    simulated = use_sim
    display = LaunchConfiguration("display").perform(context).strip().lower() in ("1", "true")

    instance_id = LaunchConfiguration("instance_id").perform(context).strip()
    if instance_id:
        profile["runtime"]["instance_id"] = instance_id
    peripherals_file = LaunchConfiguration("peripherals").perform(context).strip()
    fragment = load_peripherals_file(peripherals_file) if peripherals_file else {}
    # The profile stays raw: peripheral_stack_actions and runtime_stack_actions
    # both apply the deployment file themselves (upstream contract). The merged
    # list is computed once here only for the public interface description.
    merged_peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    fast_lio_config = {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}
    # Chassis geometry comes from the profile only; a missing key or a limit
    # the wheels cannot realize fails here, naming it (lekiwi_robot.base_profile).
    base_params = base_node_parameters(profile, profile_path)
    if lidar_odometry_owns_tf(fast_lio_config, simulated):
        base_params["publish_tf"] = False
    base_node = Node(
        package="lekiwi_robot",
        executable="base_node",
        name="lekiwi_base",
        output="screen",
        parameters=[base_params],
    )
    # A base-only deployment has no arm, so it declares no `motion` section and
    # must not start the arm motion services. Composing them anyway would make
    # the runtime advertise FK/IK for joints the hardware never opens.
    motion_actions = []
    if profile.get("motion"):
        motion_actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(get_package_share_directory("so101_motion"), "launch", "motion.launch.py")
                ),
                launch_arguments={
                    **motion_launch_arguments(profile, profile_path, simulated, sim_platform or "", display),
                    "robot_description": arm_motion_model(profile, profile_path, simulated),
                }.items(),
            )
        )
    # The whole robot (arm and wheels) is the public model: ros2_control and
    # robot_state_publisher load it, and the facade derives the robot model and
    # the teleoperation interfaces from the same rendering. The arm-only model
    # above is what so101_motion plans against; it is not the public model.
    robot_description = render_robot_description(profile, profile_path, simulated)
    description = build_description(
        {**profile, "fast_lio": fast_lio_config},
        peripherals=merged_peripherals,
        use_sim=simulated,
        robot_description=robot_description,
    )
    peripherals = [
        *peripheral_stack_actions(
            profile, peripherals_file, simulated, description=description, sim_platform=sim_platform or ""
        ),
        *fast_lio_nodes(fast_lio_config, bridge_package="lekiwi_robot", use_sim=simulated),
    ]
    return [
        *runtime_stack_actions(
            profile,
            profile_path,
            simulated,
            peripherals_file,
            instance_id,
            robot_description=robot_description,
            interface_description=description,
            sim_platform=sim_platform or "",
        ),
        base_node,
        *motion_actions,
        *peripherals,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Runtime profile path or shipped profile name (required)"),
            DeclareLaunchArgument(
                "use_sim",
                default_value="",
                description="Run simulated (true) or physical (false); default: the profile's selection",
            ),
            DeclareLaunchArgument(
                "sim_platform",
                default_value="",
                description="Simulation platform: sdk | gazebo (sdk: the vendor SDK virtual bus, no physics); "
                "implies use_sim:=true when given alone",
            ),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Deprecated alias of use_sim (a warning names use_sim)",
            ),
            DeclareLaunchArgument("display", default_value="false", description="Launch RViz"),
            DeclareLaunchArgument("instance_id", default_value="", description="Public robot instance identity"),
            DeclareLaunchArgument(
                "peripherals",
                default_value="",
                description="Deployment peripherals fragment {peripherals: [...], fast_lio: {...}} "
                "(overrides profile defaults by (type, name))",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
