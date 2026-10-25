import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get directories
    pkg_dir = FindPackageShare(package="robot_navigation").find("robot_navigation")

    # Launch configurations
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    use_composition = LaunchConfiguration("use_composition")
    use_respawn = LaunchConfiguration("use_respawn")
    namespace = LaunchConfiguration("namespace")
    log_level = LaunchConfiguration("log_level")
    use_amcl = LaunchConfiguration("use_amcl")
    auto_global_localization = LaunchConfiguration("auto_global_localization")
    controller_plugin = LaunchConfiguration("controller_plugin")
    local_costmap_plugin = LaunchConfiguration("local_costmap_plugin")
    observation_sources = LaunchConfiguration("observation_sources")
    use_collision_monitor = LaunchConfiguration("use_collision_monitor")
    dwb_critics_key = LaunchConfiguration("dwb_critics_key")
    mppi_critics_key = LaunchConfiguration("mppi_critics_key")

    stdout_linebuf_envvar = SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1")

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites={
            "use_sim_time": use_sim_time,
            "yaml_filename": map_file,
            "controller_server.ros__parameters.FollowPath.plugin": controller_plugin,
            "local_costmap.local_costmap.ros__parameters.voxel_layer.plugin": local_costmap_plugin,
            "local_costmap.local_costmap.ros__parameters.voxel_layer.observation_sources": observation_sources,
        },
        key_rewrites={"dwb_critics": dwb_critics_key, "mppi_critics": mppi_critics_key},
        convert_types=True,
    )

    localization_nodes = [
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                {"use_sim_time": use_sim_time},
                {"autostart": autostart},
                {"bond_timeout": 10.0},
                {"node_names": ["map_server"]},
            ],
        ),
        Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params, {"tf_broadcast": True}],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
            condition=IfCondition(use_amcl),
        ),
        Node(
            package="robot_navigation",
            executable="amcl_global_localization",
            name="amcl_global_localization",
            output="screen",
            parameters=[
                {
                    "startup_delay": 3.0,
                    "required_scans": 10,
                    "scan_topic": "/scan",
                    "service_name": "/reinitialize_global_localization",
                }
            ],
            condition=IfCondition(auto_global_localization),
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_amcl",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                {"use_sim_time": use_sim_time},
                {"autostart": autostart},
                {"bond_timeout": 10.0},
                {"node_names": ["amcl"]},
            ],
            condition=IfCondition(use_amcl),
        ),
    ]

    navigation_bringup = GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                # Our adaptation of nav2_bringup's navigation_launch.py (Apache-2.0,
                # upstream header kept): with the collision monitor active it
                # remaps the velocity smoother's output to cmd_vel_smoothed so
                # the monitor is the single publisher on the base /cmd_vel
                # channel (task 3.2: controller -> /cmd_vel_nav -> velocity_
                # smoother -> /cmd_vel_smoothed -> collision monitor -> /cmd_vel).
                PythonLaunchDescriptionSource(os.path.join(pkg_dir, "launch", "navigation_launch.py")),
                launch_arguments={
                    "namespace": namespace,
                    "use_sim_time": use_sim_time,
                    "autostart": "false",
                    # Pass the rewritten file so profile-specific controller and costmap
                    # parameters reach navigation_launch.py, not only localization nodes.
                    "params_file": configured_params,
                    "use_composition": use_composition,
                    "use_respawn": use_respawn,
                    "container_name": "nav2_container",
                    "log_level": log_level,
                    "use_collision_monitor": use_collision_monitor,
                }.items(),
            )
        ],
    )

    navigation_startup = Node(
        package="robot_navigation",
        executable="navigation_lifecycle_coordinator",
        name="navigation_lifecycle_coordinator",
        output="screen",
        condition=IfCondition(autostart),
        parameters=[
            {
                "startup_delay_sec": 3.0,
                "service_name": "/lifecycle_manager_navigation/manage_nodes",
                "namespace": namespace,
                "service_wait_timeout_sec": 30.0,
                "request_timeout_sec": 60.0,
                "retry_count": 15,
                "retry_interval_sec": 5.0,
            }
        ],
    )

    collision_monitor_nodes = [
        Node(
            package="nav2_collision_monitor",
            executable="collision_monitor",
            name="collision_monitor",
            output="screen",
            parameters=[configured_params],
            condition=IfCondition(use_collision_monitor),
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_collision_monitor",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time},
                {"autostart": autostart},
                {"bond_timeout": 10.0},
                {"node_names": ["collision_monitor"]},
            ],
            condition=IfCondition(use_collision_monitor),
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time", default_value="false", description="Use simulation (Gazebo) clock if true"
            ),
            DeclareLaunchArgument(
                "map",
                default_value=os.path.expanduser("~/.ros/ibrobot/maps/map.yaml"),
                description="Full path to map yaml file",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(pkg_dir, "config", "nav2_params.yaml"),
                description="Full path to the Nav2 parameters file",
            ),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("use_composition", default_value="False"),
            DeclareLaunchArgument("use_respawn", default_value="False"),
            DeclareLaunchArgument("use_amcl", default_value="false"),
            DeclareLaunchArgument("auto_global_localization", default_value="false"),
            DeclareLaunchArgument("controller_plugin", default_value="dwb_core::DWBLocalPlanner"),
            DeclareLaunchArgument("local_costmap_plugin", default_value="nav2_costmap_2d::ObstacleLayer"),
            DeclareLaunchArgument("observation_sources", default_value="scan"),
            DeclareLaunchArgument("use_collision_monitor", default_value="false"),
            DeclareLaunchArgument("dwb_critics_key", default_value="critics"),
            DeclareLaunchArgument("mppi_critics_key", default_value="mppi_critics"),
            DeclareLaunchArgument("log_level", default_value="info"),
            stdout_linebuf_envvar,
            *localization_nodes,
            navigation_bringup,
            navigation_startup,
            *collision_monitor_nodes,
        ]
    )
