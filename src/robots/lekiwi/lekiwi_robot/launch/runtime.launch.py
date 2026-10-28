"""LeKiwi runtime launch entry (robot-runtime-packaging spec).

robot_state_publisher + ros2_control (lekiwi_hardware, shared bus, real or
SDK simulation platform) + controllers + so101_motion for the arm + the
LeKiwi base node (cmd_vel, odometry, navigation gating) + sensor peripherals
(D12: profile defaults, deployment overrides via peripherals:=<file>) with
the fast_lio lidar odometry chain + runtime facade.

On a physics platform the whole stack runs on simulation time against the
simulator: the description is rendered once with the platform's renderer (the
public model and interface description derive from that exact render), the
base node, FAST-LIO and the scan converter carry ``use_sim_time``, and
``runtime_stack_actions`` hands the embodiment the world/scene name and spawn
pose. On Gazebo (D7) the world comes from ``gazebo_world``; on MuJoCo
(lekiwi-mujoco-runtime-platform D7) the scene comes from ``mujoco_scene``
(default the shared ``empty`` scene) and headless mode follows
``IBROBOT_MUJOCO_HEADLESS``.

    ros2 launch lekiwi_robot runtime.launch.py profile:=<path-or-name> \
        [use_sim:=true] [sim_platform:=gazebo|mujoco] \
        [gazebo_world:=<name>] [mujoco_scene:=<name>] [spawn_x/y/z/yaw:=…] \
        [display:=true] [peripherals:=<fragment.yaml>]
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
    render_gazebo_description,
    render_mujoco_description,
    render_robot_description,
    resolve_profile_argument,
    resolve_sim_selection,
    runtime_stack_actions,
)
from robot_runtime.peripherals import fast_lio_nodes, load_peripherals_file, merge_peripherals
from robot_runtime.profile import PHYSICS_SIM_PLATFORMS, load_profile


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
    # list is computed once here for the Gazebo rendering and the public
    # interface description (D5 inventory-first: one effective inventory for
    # sensor generation, bridges, derived processors and the description).
    merged_peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    fast_lio_config = {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}
    # Chassis geometry comes from the profile only; a missing key or a limit
    # the wheels cannot realize fails here, naming it (lekiwi_robot.base_profile).
    base_params = base_node_parameters(profile, profile_path)
    if lidar_odometry_owns_tf(fast_lio_config, simulated):
        base_params["publish_tf"] = False
    # Physics platforms run the base node on simulation time (D7); wheel
    # odometry always owns odom -> base_link in simulation (D4).
    base_params["use_sim_time"] = simulated and (sim_platform or "") in PHYSICS_SIM_PLATFORMS
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
    # D7: one platform-aware render — on a physics platform the public model
    # and the interface description derive from the exact platform rendering
    # (sensors included); hardware and SDK keep the ordinary renderer. On
    # MuJoCo (lekiwi-mujoco-runtime-platform D7) the render composes the robot
    # MJCF into the named scene (empty scene name composes ``empty``).
    mujoco_scene = LaunchConfiguration("mujoco_scene").perform(context).strip()
    if simulated and sim_platform in PHYSICS_SIM_PLATFORMS:
        spawn_pose = tuple(
            float(LaunchConfiguration(f"spawn_{axis}").perform(context)) for axis in ("x", "y", "z", "yaw")
        )
    else:
        spawn_pose = (0.0, 0.0, 0.0, 0.0)
    if simulated and sim_platform == "gazebo":
        robot_description = render_gazebo_description(
            {**profile, "peripherals": merged_peripherals}, profile_path, spawn_pose=spawn_pose
        )
    elif simulated and sim_platform == "mujoco":
        robot_description = render_mujoco_description(
            {**profile, "peripherals": merged_peripherals}, profile_path, spawn_pose=spawn_pose, scene=mujoco_scene
        )
    else:
        robot_description = render_robot_description(profile, profile_path, simulated)
    description = build_description(
        {**profile, "fast_lio": fast_lio_config},
        peripherals=merged_peripherals,
        use_sim=simulated,
        sim_platform=sim_platform,
        robot_description=robot_description,
    )
    peripherals = [
        *peripheral_stack_actions(
            profile, peripherals_file, simulated, description=description, sim_platform=sim_platform or ""
        ),
        *fast_lio_nodes(
            fast_lio_config, bridge_package="lekiwi_robot", use_sim=simulated, sim_platform=sim_platform or ""
        ),
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
            gazebo_world=LaunchConfiguration("gazebo_world").perform(context).strip(),
            mujoco_scene=mujoco_scene,
            spawn_pose=spawn_pose,
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
                description="Simulation platform: sdk | gazebo | mujoco (sdk: the vendor SDK virtual bus, no "
                "physics; gazebo/mujoco: the physics simulators); implies use_sim:=true when given alone",
            ),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Deprecated alias of use_sim (a warning names use_sim)",
            ),
            DeclareLaunchArgument(
                "gazebo_world",
                default_value="",
                description="Gazebo world name to spawn into (empty: the runtime starts a minimal empty world)",
            ),
            DeclareLaunchArgument(
                "mujoco_scene",
                default_value="",
                description="MuJoCo scene name to compose the robot into (empty: the shared 'empty' scene)",
            ),
            DeclareLaunchArgument("spawn_x", default_value="0.0", description="Robot spawn pose x (physics platforms)"),
            DeclareLaunchArgument("spawn_y", default_value="0.0", description="Robot spawn pose y (physics platforms)"),
            DeclareLaunchArgument(
                # Default lifts the base so the wheels (lowest point at
                # wheel_base_offset_z - wheel_radius = -0.0406 m below
                # base_link) start just above the ground plane (D7).
                "spawn_z",
                default_value="0.05",
                description="Robot spawn pose z (physics platforms); default lifts the wheels above the ground",
            ),
            DeclareLaunchArgument(
                "spawn_yaw", default_value="0.0", description="Robot spawn yaw in radians (physics platforms)"
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
