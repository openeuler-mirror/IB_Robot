"""Launch the IB-Robot tracing web API with safe local defaults."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import ExecutableInPackage


def _launch(context):
    host = LaunchConfiguration("host").perform(context)
    port = LaunchConfiguration("port").perform(context)
    trace_root = LaunchConfiguration("trace_root").perform(context)
    allowed_hosts = LaunchConfiguration("allowed_hosts").perform(context)
    allow_lan = LaunchConfiguration("allow_unauthenticated_lan").perform(context).lower() in {"1", "true", "yes"}
    certfile = LaunchConfiguration("certfile").perform(context)
    keyfile = LaunchConfiguration("keyfile").perform(context)
    if bool(certfile) != bool(keyfile):
        raise RuntimeError("certfile and keyfile must be provided together")
    arguments = ["--host", host, "--port", port, "--trace-root", trace_root]
    for option in ("max_source_bytes", "max_events", "max_result_bytes"):
        value = LaunchConfiguration(option).perform(context)
        if value:
            arguments.extend(("--" + option.replace("_", "-"), value))
    for allowed_host in (item.strip() for item in allowed_hosts.split(",")):
        if allowed_host:
            arguments.extend(("--allowed-host", allowed_host))
    if allow_lan:
        arguments.append("--allow-unauthenticated-lan")
    if certfile and keyfile:
        arguments.extend(("--certfile", certfile, "--keyfile", keyfile))
    return [
        ExecuteProcess(
            cmd=[
                ExecutableInPackage(package="ibrobot_tracing_web", executable="ibrobot-tracing-web"),
                *arguments,
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("port", default_value="8000"),
            DeclareLaunchArgument("trace_root", default_value="~/.ros/tracing"),
            DeclareLaunchArgument("allowed_hosts", default_value="localhost,127.0.0.1"),
            DeclareLaunchArgument("allow_unauthenticated_lan", default_value="false"),
            DeclareLaunchArgument("certfile", default_value=""),
            DeclareLaunchArgument("keyfile", default_value=""),
            DeclareLaunchArgument("max_source_bytes", default_value=""),
            DeclareLaunchArgument("max_events", default_value=""),
            DeclareLaunchArgument("max_result_bytes", default_value=""),
            OpaqueFunction(function=_launch),
        ]
    )
