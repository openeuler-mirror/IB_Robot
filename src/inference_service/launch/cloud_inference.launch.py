from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("pipeline_id", default_value="policy"),
        DeclareLaunchArgument("model_path", description="Absolute path to the unified policy bundle"),
        DeclareLaunchArgument("deployment", description="Named deployment from inference_manifest.json"),
        DeclareLaunchArgument("request_timeout", default_value="5.0"),
        DeclareLaunchArgument("runtime_options_json", default_value="{}"),
        DeclareLaunchArgument("robot_config_path", description="Absolute path to the robot configuration YAML"),
        DeclareLaunchArgument("recording", default_value="false"),
        DeclareLaunchArgument(
            "node_name",
            default_value=["inference_", LaunchConfiguration("pipeline_id"), "_cloud"],
        ),
        DeclareLaunchArgument(
            "request_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/request"],
        ),
        DeclareLaunchArgument(
            "result_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/result"],
        ),
        DeclareLaunchArgument(
            "heartbeat_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/heartbeat"],
        ),
        DeclareLaunchArgument(
            "video_descriptor_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/video/descriptors"],
        ),
        DeclareLaunchArgument(
            "video_status_topic",
            default_value=["/inference/", LaunchConfiguration("pipeline_id"), "/video/status"],
        ),
    ]
    cloud_node = Node(
        package="inference_service",
        executable="pure_inference_node",
        name=LaunchConfiguration("node_name"),
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("recording")),
        parameters=[
            {
                "pipeline_id": LaunchConfiguration("pipeline_id"),
                "model_path": LaunchConfiguration("model_path"),
                "deployment": LaunchConfiguration("deployment"),
                "request_timeout": LaunchConfiguration("request_timeout"),
                "runtime_options_json": ParameterValue(LaunchConfiguration("runtime_options_json"), value_type=str),
                "robot_config_path": LaunchConfiguration("robot_config_path"),
                "node_name": LaunchConfiguration("node_name"),
                "request_topic": LaunchConfiguration("request_topic"),
                "result_topic": LaunchConfiguration("result_topic"),
                "heartbeat_topic": LaunchConfiguration("heartbeat_topic"),
                "video_descriptor_topic": LaunchConfiguration("video_descriptor_topic"),
                "video_status_topic": LaunchConfiguration("video_status_topic"),
                "use_sim_time": False,
            }
        ],
    )
    recording_node = Node(
        package="inference_service",
        executable="recording_node",
        name="recording_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("recording")),
        parameters=[
            {
                "robot_config_path": LaunchConfiguration("robot_config_path"),
                "pipeline_id": LaunchConfiguration("pipeline_id"),
                # cloud_node is excluded when recording, so the recorder is the
                # only remaining peer that can supply the ROLE_CLOUD heartbeat
                # the edge needs before it will bind a video session.
                "heartbeat_topic": LaunchConfiguration("heartbeat_topic"),
                "video_descriptor_topic": LaunchConfiguration("video_descriptor_topic"),
                "video_status_topic": LaunchConfiguration("video_status_topic"),
                "runtime_options_json": ParameterValue(LaunchConfiguration("runtime_options_json"), value_type=str),
            }
        ],
    )
    return LaunchDescription([*arguments, cloud_node, recording_node])
