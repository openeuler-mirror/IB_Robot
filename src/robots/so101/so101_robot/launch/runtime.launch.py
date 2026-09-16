"""SO-101 runtime launch entry (robot-runtime-packaging spec).

Brings the SO-101 to the ACTIVE lifecycle from a runtime profile alone:
robot_state_publisher + ros2_control (so101_hardware adapter, real or SDK
simulated transport) + controllers + so101_motion (MoveIt + motion server)
+ the runtime facade + sensor peripherals (D12: profile defaults,
deployment overrides via peripherals:=<file>). No IB-Robot generic package is
read or launched.

    ros2 launch so101_robot runtime.launch.py profile:=<path-or-name> \
        [simulated:=true] [display:=true] [peripherals:=<fragment.yaml>]

``profile`` is a file path, or the name of a profile shipped in
``so101_robot/profiles`` (e.g. ``so101_single_arm``).
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import (
    motion_launch_arguments,
    peripheral_stack_actions,
    render_robot_description,
    resolve_profile_argument,
    resolve_simulated,
    runtime_stack_actions,
)
from robot_runtime.peripherals import load_peripherals_file, merge_peripherals
from robot_runtime.profile import load_profile


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("so101_robot", LaunchConfiguration("profile").perform(context))
    profile = load_profile(profile_path)
    simulated = resolve_simulated(profile, LaunchConfiguration("simulated").perform(context))
    display = LaunchConfiguration("display").perform(context).strip().lower() in ("1", "true")
    initial_mode = LaunchConfiguration("initial_mode").perform(context).strip()
    if initial_mode:
        if initial_mode not in profile["modes"] or initial_mode == "initial":
            raise ValueError(f"unsupported initial_mode {initial_mode!r}")
        profile["modes"]["initial"] = initial_mode
    instance_id = LaunchConfiguration("instance_id").perform(context).strip()
    if instance_id:
        profile["runtime"]["instance_id"] = instance_id
    peripherals_file = LaunchConfiguration("peripherals").perform(context).strip()
    fragment = load_peripherals_file(peripherals_file) if peripherals_file else {}
    profile["peripherals"] = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    robot_description = render_robot_description(profile, profile_path, simulated)
    description = build_description(profile, simulated=simulated, robot_description=robot_description)

    motion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("so101_motion"), "launch", "motion.launch.py")
        ),
        launch_arguments={
            **motion_launch_arguments(profile, profile_path, simulated, display),
            "robot_description": robot_description,
        }.items(),
    )
    from so101_robot.teleoperation import generate_teleoperation_nodes

    return [
        *runtime_stack_actions(
            profile,
            profile_path,
            simulated,
            peripherals_file,
            instance_id,
            robot_description=robot_description,
            interface_description=description,
        ),
        motion,
        *peripheral_stack_actions(profile, "", simulated, description=description),
        *generate_teleoperation_nodes(profile, description, {"robot_description": robot_description}),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Runtime profile path or shipped profile name (required)"),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Override profile.simulated: run so101_hardware on the SDK simulated transport",
            ),
            DeclareLaunchArgument("display", default_value="false", description="Launch RViz"),
            DeclareLaunchArgument("instance_id", default_value="", description="Public robot instance identity"),
            DeclareLaunchArgument("initial_mode", default_value="", description="Initial runtime mode override"),
            DeclareLaunchArgument(
                "peripherals",
                default_value="",
                description="Deployment peripherals fragment {peripherals: [...]} "
                "(overrides profile defaults by (type, name))",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
