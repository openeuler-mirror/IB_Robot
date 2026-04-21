import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get directories
    pkg_dir = FindPackageShare(package="robot_navigation").find("robot_navigation")
    robot_description_dir = FindPackageShare(package="robot_description").find("robot_description")
    nav2_bringup_dir = FindPackageShare(package="nav2_bringup").find("nav2_bringup")
    default_model_path = os.path.join(robot_description_dir, "urdf", "lerobot", "lekiwi", "lekiwi.urdf.xacro")

    # Launch configurations
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")

    # ==================== Robot Specific Nodes ====================
    # Joint State Publisher
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[
            {
                "robot_description": ParameterValue(Command(["xacro ", default_model_path]), value_type=str),
            }
        ],
    )
    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            {
                "robot_description": ParameterValue(Command(["xacro ", default_model_path]), value_type=str),
            }
        ],
    )

    # ==================== RViz2 ====================
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg_dir, "config", "config.rviz")],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ==================== Nav2 Goal Client ====================
    nav2_goal_client_node = Node(
        package="robot_navigation",
        executable="nav2_goal_client",
        name="nav2_goal_client",
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )

    # ==================== Official Nav2 Navigation Stack ====================
    # Use official nav2_bringup bringup_launch.py
    # This includes: localization (map_server + AMCL), navigation stack,
    #                 and lifecycle managers
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, "launch", "bringup_launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map": map_file,
            "params_file": params_file,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time", default_value="false", description="Use simulation (Gazebo) clock if true"
            ),
            DeclareLaunchArgument(
                "map", default_value="~/workspace/map/rtabmap.yaml", description="Full path to map yaml file"
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(pkg_dir, "config", "nav2_params.yaml"),
                description="Full path to the Nav2 parameters file",
            ),
            # Robot specific nodes
            robot_state_publisher_node,
            joint_state_publisher_node,
            # Official Nav2 bringup stack
            nav2_bringup,
            # Nav2 Goal Client (with voice control)
            nav2_goal_client_node,
            # RViz2
            rviz_node,
        ]
    )
