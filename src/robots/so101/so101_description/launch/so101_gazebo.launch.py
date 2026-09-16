import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_description = get_package_share_directory("so101_description")

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(robot_description, "urdf", "lerobot", "so101", "so101.urdf.xacro"),
        description="Absolute path to robot urdf file",
    )

    use_cameras_arg = DeclareLaunchArgument(
        name="use_cameras", default_value="true", description="Enable cameras (wrist + top)"
    )

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH", value=[str(Path(robot_description).parent.resolve())]
    )

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                LaunchConfiguration("model"),
                " use_sim:=true",
                " use_cameras:=",
                LaunchConfiguration("use_cameras"),
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True, "frame_prefix": ""}],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory("ros_gz_sim"), "launch"), "/gz_sim.launch.py"]
        ),
        launch_arguments=[("gz_args", [" -v 4 -r empty.sdf "])],
    )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "so101"],
    )

    # Clock bridge
    gz_ros2_clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
    )

    # Joint state bridge for ros2_control
    gz_ros2_joint_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/empty/model/so101/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        ],
    )

    # Camera bridges (if cameras enabled)
    gz_ros2_wrist_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        condition=LaunchConfigurationEquals("use_cameras", "true"),
        arguments=[
            "/world/empty/model/so101/link/wrist_camera_link/sensor/wrist_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
    )

    gz_ros2_top_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        condition=LaunchConfigurationEquals("use_cameras", "true"),
        arguments=[
            "/world/empty/model/so101/link/top_camera_link/sensor/top_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
    )

    return LaunchDescription(
        [
            model_arg,
            use_cameras_arg,
            gazebo_resource_path,
            robot_state_publisher_node,
            gazebo,
            gz_spawn_entity,
            gz_ros2_clock_bridge,
            gz_ros2_joint_bridge,
            gz_ros2_wrist_camera_bridge,
            gz_ros2_top_camera_bridge,
        ]
    )
