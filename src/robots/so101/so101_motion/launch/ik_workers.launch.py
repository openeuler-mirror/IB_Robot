import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

_DISABLED_CAPABILITIES = " ".join(
    (
        "move_group/MoveGroupCartesianPathService",
        "move_group/MoveGroupExecuteService",
        "move_group/MoveGroupExecuteTrajectoryAction",
        "move_group/MoveGroupMoveAction",
        "move_group/MoveGroupPlanService",
        "move_group/MoveGroupQueryPlannersService",
        "move_group/MoveGroupStateValidationService",
        "move_group/MoveGroupGetPlanningSceneService",
        "move_group/ApplyPlanningSceneService",
        "move_group/ClearOctomapService",
        "move_group/TfPublisher",
    )
)


def _launch_setup(context):
    worker_count = int(LaunchConfiguration("worker_count").perform(context))
    if not 1 <= worker_count <= 8:
        raise ValueError("worker_count must be between 1 and 8")
    namespace_prefix = LaunchConfiguration("namespace_prefix").perform(context).strip("/")
    if not namespace_prefix:
        raise ValueError("namespace_prefix must not be empty")
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("1", "true", "yes")
    log_level = LaunchConfiguration("log_level").perform(context)
    joint_state_topic = LaunchConfiguration("joint_state_topic").perform(context)

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

    common_parameters = {
        "use_sim_time": use_sim_time,
        "allow_trajectory_execution": False,
        "publish_robot_description_semantic": False,
        "disable_capabilities": _DISABLED_CAPABILITIES,
    }
    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            namespace=f"{namespace_prefix}_{index}",
            name="move_group",
            output="screen",
            parameters=[moveit_config.to_dict(), common_parameters],
            remappings=[
                ("joint_states", joint_state_topic),
                ("tf", "/tf"),
                ("tf_static", "/tf_static"),
            ],
            arguments=["--ros-args", "--log-level", log_level],
        )
        for index in range(worker_count)
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "worker_count",
                default_value="4",
                description="Number of isolated MoveIt kinematics worker processes (1-8)",
            ),
            DeclareLaunchArgument(
                "namespace_prefix",
                default_value="ik_worker",
                description="Worker namespace prefix; services become /<prefix>_<index>/compute_ik and compute_fk",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            DeclareLaunchArgument("log_level", default_value="warn"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
