# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright 2026 IB_Robot Contributors
#
# Derived from nav2_bringup/launch/navigation_launch.py as shipped with
# ros-humble-nav2-bringup (Apache-2.0). Modifications for IB-Robot
# (openspec lekiwi-application-migration task 3.2): one additional launch
# argument, use_collision_monitor. The stock file remaps the velocity
# smoother's output onto cmd_vel, which cannot coexist with the Nav2
# collision monitor publishing on cmd_vel -- two publishers would drive the
# base, one of them unfiltered. When use_collision_monitor is true this file
# remaps the smoother's output to cmd_vel_smoothed instead, so the chain is
# controller -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel_smoothed ->
# collision_monitor -> /cmd_vel (the base's single admitted channel), and the
# collision monitor's cmd_vel_in_topic must be /cmd_vel_smoothed. The node
# set, lifecycle ordering and every other remap are unchanged from upstream.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode, ParameterFile
from nav2_common.launch import RewrittenYaml


def _flag(context, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).strip().lower() in ("1", "true", "yes")


def _navigation_nodes(context):
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    container_name = LaunchConfiguration("container_name")
    container_name_full = (namespace, "/", container_name)
    use_respawn = LaunchConfiguration("use_respawn")
    log_level = LaunchConfiguration("log_level")

    with_collision_monitor = _flag(context, "use_collision_monitor")
    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]

    # Map fully qualified names to relative ones so the node's namespace can
    # be prepended (upstream comment; kept for parity).
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    # With the collision monitor active, the smoother's output stays on
    # cmd_vel_smoothed so the monitor is the single publisher on the base's
    # /cmd_vel channel; without it the stock smoother -> cmd_vel remap keeps
    # the base driven directly by the smoothed output.
    smoother_output = "cmd_vel_smoothed" if with_collision_monitor else "cmd_vel"

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        "use_sim_time": use_sim_time,
        "autostart": autostart,
    }

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites=param_substitutions,
            convert_types=True,
        ),
        allow_substs=True,
    )

    node_specs = [
        ("nav2_controller", "controller_server", "controller_server", remappings + [("cmd_vel", "cmd_vel_nav")]),
        ("nav2_smoother", "smoother_server", "smoother_server", remappings),
        ("nav2_planner", "planner_server", "planner_server", remappings),
        ("nav2_behaviors", "behavior_server", "behavior_server", remappings),
        ("nav2_bt_navigator", "bt_navigator", "bt_navigator", remappings),
        ("nav2_waypoint_follower", "waypoint_follower", "waypoint_follower", remappings),
        (
            "nav2_velocity_smoother",
            "velocity_smoother",
            "velocity_smoother",
            remappings + [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", smoother_output)],
        ),
    ]
    composable_specs = [
        (
            "nav2_controller",
            "nav2_controller::ControllerServer",
            "controller_server",
            remappings + [("cmd_vel", "cmd_vel_nav")],
        ),
        ("nav2_smoother", "nav2_smoother::SmootherServer", "smoother_server", remappings),
        ("nav2_planner", "nav2_planner::PlannerServer", "planner_server", remappings),
        ("nav2_behaviors", "behavior_server::BehaviorServer", "behavior_server", remappings),
        ("nav2_bt_navigator", "nav2_bt_navigator::BtNavigator", "bt_navigator", remappings),
        ("nav2_waypoint_follower", "nav2_waypoint_follower::WaypointFollower", "waypoint_follower", remappings),
        (
            "nav2_velocity_smoother",
            "nav2_velocity_smoother::VelocitySmoother",
            "velocity_smoother",
            remappings + [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", smoother_output)],
        ),
    ]

    load_nodes = GroupAction(
        condition=IfCondition(PythonExpression(["not ", LaunchConfiguration("use_composition")])),
        actions=[
            *(
                Node(
                    package=package,
                    executable=executable,
                    name=name,
                    output="screen",
                    respawn=use_respawn,
                    respawn_delay=2.0,
                    parameters=[configured_params],
                    arguments=["--ros-args", "--log-level", log_level],
                    remappings=node_remappings,
                )
                for package, executable, name, node_remappings in node_specs
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                arguments=["--ros-args", "--log-level", log_level],
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"autostart": autostart},
                    {"node_names": lifecycle_nodes},
                ],
            ),
        ],
    )

    load_composable_nodes = LoadComposableNodes(
        condition=IfCondition(LaunchConfiguration("use_composition")),
        target_container=container_name_full,
        composable_node_descriptions=[
            *(
                ComposableNode(
                    package=package,
                    plugin=plugin,
                    name=name,
                    parameters=[configured_params],
                    remappings=node_remappings,
                )
                for package, plugin, name, node_remappings in composable_specs
            ),
            ComposableNode(
                package="nav2_lifecycle_manager",
                plugin="nav2_lifecycle_manager::LifecycleManager",
                name="lifecycle_manager_navigation",
                parameters=[{"use_sim_time": use_sim_time, "autostart": autostart, "node_names": lifecycle_nodes}],
            ),
        ],
    )

    return [
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
        load_nodes,
        load_composable_nodes,
    ]


def generate_launch_description():
    actions = [
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Top-level namespace",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(get_package_share_directory("nav2_bringup"), "params", "nav2_params.yaml"),
            description="Full path to the ROS2 parameters file to use for all launched nodes",
        ),
        DeclareLaunchArgument(
            "autostart",
            default_value="true",
            description="Automatically startup the nav2 stack",
        ),
        DeclareLaunchArgument(
            "use_composition",
            default_value="False",
            description="Use composed bringup if True",
        ),
        DeclareLaunchArgument(
            "container_name",
            default_value="nav2_container",
            description="the name of container that nodes will load in if use composition",
        ),
        DeclareLaunchArgument(
            "use_respawn",
            default_value="False",
            description="Whether to respawn if a node crashes. Applied when composition is disabled.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="log level",
        ),
        DeclareLaunchArgument(
            "use_collision_monitor",
            default_value="false",
            description=(
                "Remap the velocity smoother's output to cmd_vel_smoothed so the collision monitor "
                "(configured with cmd_vel_in_topic=/cmd_vel_smoothed, cmd_vel_out_topic=/cmd_vel) is the "
                "single publisher on the base /cmd_vel channel."
            ),
        ),
        OpaqueFunction(function=_navigation_nodes),
    ]
    return LaunchDescription(actions)
