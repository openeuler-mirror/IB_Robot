import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration("use_sim_time")

    # New parameter to control whether to show GUI sliders or use real robot
    use_gui = LaunchConfiguration("use_gui")

    # Parameter for which joint_states topic to use
    joint_states_topic = LaunchConfiguration("joint_states_topic")

    description_dir = get_package_share_directory("so101_description")

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(description_dir, "urdf", "lerobot", "so101", "so101.urdf.xacro"),
        description="Absolute path to robot urdf file",
    )

    robot_description = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]), value_type=str)

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}],
        remappings=[("/joint_states", joint_states_topic)],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        remappings=[("/joint_states", "/so101_follower/joint_commands")],
        condition=IfCondition(use_gui),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(description_dir, "rviz", "display.rviz")],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false", description="Use sim time if true"),
            DeclareLaunchArgument(
                "use_gui", default_value="true", description="Whether to show joint_state_publisher_gui sliders"
            ),
            DeclareLaunchArgument(
                "joint_states_topic", default_value="/joint_states", description="Topic to get joint states from"
            ),
            model_arg,
            joint_state_publisher_gui_node,
            robot_state_publisher_node,
            rviz_node,
        ]
    )
