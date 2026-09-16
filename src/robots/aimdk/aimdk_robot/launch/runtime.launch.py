"""AgiBot X2 runtime launch entry (robot-runtime-packaging spec).

Brings the X2 to ACTIVE from a runtime profile alone: one bridge process that
serves the public runtime contract over the vendor MC tier, plus — in simulated
transport — the in-repo vendor mock that stands in for the robot's own stack.

    ros2 launch aimdk_robot runtime.launch.py profile:=<path-or-name> \
        [simulated:=true] [instance_id:=<id>] [initial_mode:=<mode>]

No controller manager, no ros2_control hardware component and no sensor driver
is launched: the vendor owns real-time control and already publishes its
sensors. Node startup is staged because the platform documents that bulk node
launches disturb motion control (no more than ~2 nodes per second).
"""

from __future__ import annotations

import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import resolve_profile_argument, resolve_simulated
from robot_runtime.peripherals import synthetic_perception_nodes
from robot_runtime.profile import load_profile

#: Geometry used only when the runtime itself generates images (simulated
#: transport). The physical profiles declare no geometry on purpose: the vendor
#: documents that it varies across hardware revisions and must be read from
#: CameraInfo. Here the runtime is the producer, so it is free to choose.
SIMULATED_IMAGE_PROFILE = {"width": 640, "height": 480, "fps": 30, "encoding": "rgb8"}

#: Seconds between node starts. The vendor documents that more than ~2 nodes
#: per second joining DDS discovery can destabilize motion control.
NODE_STAGGER_S = 0.6


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("aimdk_robot", LaunchConfiguration("profile").perform(context))
    profile = load_profile(profile_path)
    simulated = resolve_simulated(profile, LaunchConfiguration("simulated").perform(context))
    instance_id = LaunchConfiguration("instance_id").perform(context).strip()
    initial_mode = LaunchConfiguration("initial_mode").perform(context).strip()

    def runtime_node(description_json: str = "") -> Node:
        return Node(
            package="aimdk_robot",
            executable="aimdk_runtime",
            name="aimdk_runtime",
            output="screen",
            parameters=[
                {
                    "profile": str(profile_path),
                    "simulated": simulated,
                    "initial_mode": initial_mode,
                    "instance_id": instance_id,
                    "interface_description_json": description_json,
                }
            ],
        )

    if not simulated:
        return [runtime_node()]

    # In simulated transport the platform's sensor streams have no publisher, so
    # the contract layer's synthetic publishers stand in for them on exactly the
    # declared endpoints.
    description = build_description(profile, simulated=True)
    for spec in description["interfaces"].values():
        if spec["message_type"] == "sensor_msgs/msg/Image" and spec.get("configured_profile") is None:
            spec["configured_profile"] = dict(SIMULATED_IMAGE_PROFILE)
    description.pop("digest", None)
    from robot_runtime.interface_description import description_digest

    description["digest"] = description_digest(description)
    perception = synthetic_perception_nodes(profile.get("peripherals"), description)
    runtime = runtime_node(json.dumps(description))

    vendor = profile.get("vendor") or {}
    mock = Node(
        package="aimdk_robot",
        executable="aimdk_vendor_mock",
        name="aimdk_vendor_mock",
        output="screen",
        parameters=[{"hand_type": str((vendor.get("hand") or {}).get("type", "claw")).split("_")[0]}],
    )
    # The platform stand-in comes up first; the bridge follows one stagger later,
    # then the synthetic sensor publishers, honouring the node-rate limit.
    return [
        mock,
        TimerAction(period=NODE_STAGGER_S, actions=[runtime]),
        TimerAction(period=2 * NODE_STAGGER_S, actions=perception),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Runtime profile path or shipped profile name (required)"),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Override profile.simulated: run against the in-repo vendor mock",
            ),
            DeclareLaunchArgument("instance_id", default_value="", description="Public robot instance identity"),
            DeclareLaunchArgument("initial_mode", default_value="", description="Initial runtime mode override"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
