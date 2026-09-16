"""SO-101 motion stack: MoveIt move_group + motion server (+ optional RViz, IK workers).

Included by so101_robot/launch/runtime.launch.py. Every robot-instance value
comes from launch arguments the runtime launch derives from the runtime
profile; nothing here reads robot_config.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("display", default_value="false", description="Launch RViz"),
        DeclareLaunchArgument("robot_description", default_value="", description="Effective runtime URDF"),
        DeclareLaunchArgument("joint_names", description="Arm joint names (space-separated, required)"),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
        DeclareLaunchArgument("arm_group_name", description="MoveIt planning group (required)"),
        DeclareLaunchArgument("base_link", description="Base link frame (required)"),
        DeclareLaunchArgument("ee_link", description="End-effector link frame (required)"),
        DeclareLaunchArgument("shoulder_link", description="Shoulder link frame (required)"),
        DeclareLaunchArgument("motion_status_hold_s", default_value="0.0"),
        DeclareLaunchArgument("motion_feedback_timeout_s", default_value="0.3"),
        DeclareLaunchArgument("motion_feedback_tolerance_rad", default_value="0.12"),
        DeclareLaunchArgument("motion_require_tf_sync", default_value="true"),
        DeclareLaunchArgument("motion_hardware_feedback_topic", default_value=""),
        DeclareLaunchArgument("runtime_status_topic", default_value="/runtime_status"),
        DeclareLaunchArgument(
            "trajectory_modes",
            default_value="trajectory",
            description="Space-separated runtime modes in which executing moves are permitted",
        ),
        DeclareLaunchArgument(
            "ik_worker_count",
            default_value="0",
            description="Isolated MoveIt IK worker processes (0 = primary move_group only)",
        ),
        DeclareLaunchArgument("ik_worker_namespace_prefix", default_value="ik_worker"),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    effective_model = {"robot_description": ParameterValue(LaunchConfiguration("robot_description"), value_type=str)}

    robot_description_dir = get_package_share_directory("so101_description")
    so101_urdf_path = os.path.join(robot_description_dir, "urdf", "lerobot", "so101", "so101.urdf.xacro")
    moveit_config = (
        MoveItConfigsBuilder("so101", package_name="so101_motion")
        .robot_description(file_path=so101_urdf_path)
        .robot_description_semantic(file_path="config/so101/so101.srdf")
        .robot_description_kinematics(file_path="config/so101/kinematics.yaml")
        .joint_limits(file_path="config/so101/joint_limits.yaml")
        .pilz_cartesian_limits(file_path="config/so101/pilz_cartesian_limits.yaml")
        .trajectory_execution(file_path="config/so101/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            effective_model,
            {"use_sim_time": use_sim_time},
            {"publish_robot_description_semantic": True},
        ],
        remappings=[("joint_states", joint_state_topic)],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(get_package_share_directory("so101_motion"), "config", "so101", "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            effective_model,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        remappings=[("joint_states", joint_state_topic)],
        condition=IfCondition(LaunchConfiguration("display")),
    )

    # IK worker namespaces the motion server exposes as ComputeIk endpoints:
    # "" (primary move_group) plus <prefix>_<i> for i in range(count).
    ik_namespaces = PythonExpression(
        [
            "[''] + ['",
            LaunchConfiguration("ik_worker_namespace_prefix"),
            "_' + str(i) for i in range(int('",
            LaunchConfiguration("ik_worker_count"),
            "'))]",
        ]
    )

    motion_server_node = Node(
        package="so101_motion",
        executable="motion_server.py",
        name="motion_server",
        output="screen",
        parameters=[
            {"arm_group_name": LaunchConfiguration("arm_group_name")},
            {"base_link": LaunchConfiguration("base_link")},
            {"ee_link": LaunchConfiguration("ee_link")},
            {"shoulder_link": LaunchConfiguration("shoulder_link")},
            {"joint_names": PythonExpression(["'", LaunchConfiguration("joint_names"), "'.split()"])},
            {"motion_status_hold_s": LaunchConfiguration("motion_status_hold_s")},
            {"motion_feedback_timeout_s": LaunchConfiguration("motion_feedback_timeout_s")},
            {"motion_feedback_tolerance_rad": LaunchConfiguration("motion_feedback_tolerance_rad")},
            {"motion_require_tf_sync": LaunchConfiguration("motion_require_tf_sync")},
            {"motion_hardware_feedback_topic": LaunchConfiguration("motion_hardware_feedback_topic")},
            {"joint_state_topic": joint_state_topic},
            {"runtime_status_topic": LaunchConfiguration("runtime_status_topic")},
            {"trajectory_modes": PythonExpression(["'", LaunchConfiguration("trajectory_modes"), "'.split()"])},
            {"ik_worker_namespaces": ik_namespaces},
            {"use_sim_time": use_sim_time},
        ],
        remappings=[("joint_states", joint_state_topic)],
    )

    ik_workers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("so101_motion"), "launch", "ik_workers.launch.py")
        ),
        launch_arguments={
            "worker_count": LaunchConfiguration("ik_worker_count"),
            "namespace_prefix": LaunchConfiguration("ik_worker_namespace_prefix"),
            "use_sim_time": use_sim_time,
            "joint_state_topic": joint_state_topic,
        }.items(),
        condition=IfCondition(PythonExpression(["int('", LaunchConfiguration("ik_worker_count"), "') > 0"])),
    )

    return LaunchDescription([*args, move_group_node, rviz_node, motion_server_node, ik_workers])
