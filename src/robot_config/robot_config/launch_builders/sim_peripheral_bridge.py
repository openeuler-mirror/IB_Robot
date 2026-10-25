"""Peripheral bridge node generators for Gazebo simulation.

Uses ros_gz_bridge bridge_node with a YAML config file to map Gazebo sensor
topics directly to the ROS topic naming contract — no remappings= needed.

Why bridge_node instead of parameter_bridge + remappings=:
  parameter_bridge with Node(remappings=...) causes the process to start but
  fail to register with the ROS 2 daemon (processes appear in ps aux but not
  in ros2 node list), so no topics are published. bridge_node with a YAML
  config avoids this by specifying the ROS topic name directly in the config.

Topic naming contract (output ROS topics):
  /camera/{name}/image_raw
  /camera/{name}/camera_info
  /camera/{name}/aligned_depth_to_color/image_raw  (when depth is enabled)
where {name} is YAML peripherals[].name (e.g. "top", "wrist").

This contract matches real hardware: perception.py remaps both
usb_cam and realsense2_camera outputs to the same topic names.

Gazebo sensor path construction (camera example):
  /world/{world_name}/model/{model_name}/link/{parent_frame}
    /sensor/{sensor_name}/{sensor_name}/image

where:
  world_name   = world file's <world name="..."> (fixed: "demo")
  model_name   = robot_config["name"]                    (e.g. "so101_single_arm")
  parent_frame = peripherals[].transform.parent_frame    (e.g. "base", "wrist")
  sensor_name  = f"{name}_camera"                        (e.g. "top_camera")

Why parent_frame, not frame_id:
  Ignition Gazebo 6 (gz-sim 6.x) merges fixed-joint child links into their
  parent link during URDF→SDF conversion. The camera link (frame_id, e.g.
  "camera_top_frame") is a fixed-joint child of the parent link ("base").
  After merging, the Gazebo entity path uses the parent link name:
    so101_single_arm::base::top_camera   (not ::camera_top_frame::top_camera)
  Using frame_id in the bridge path subscribes to a nonexistent Gz topic.
"""

import yaml
from launch_ros.actions import Node


def _build_camera_bridge_entries(periph: dict, model_name: str, world_name: str) -> list:
    """Return bridge config dicts for one camera peripheral.

    Thin re-import wrapper (openspec ``so101-gazebo-runtime-transport`` D6.1):
    the robot-agnostic builder lives in ``robot_runtime.gazebo``; this call
    site keeps the generic provider-less behaviour (color, color info and the
    legacy aligned-depth entry; the runtime providers pass ``full_streams`` to
    bridge every advertised stream).
    """
    from robot_runtime.gazebo import camera_bridge_entries

    return camera_bridge_entries(periph)


def _build_lidar_bridge_entries(
    periph: dict,
    model_name: str,
    world_name: str,
    simulation_config: dict | None = None,
    platform: str = "gazebo",
) -> list:
    """Return one LaserScan bridge config dict for a lidar peripheral."""
    params = periph.get("params", {})
    ros_topic = params.get("laser_scan_topic_name") or periph.get("topic") or "/scan"
    ros_topic = ros_topic if ros_topic.startswith("/") else f"/{ros_topic}"
    gz_topic = ros_topic
    return [
        {
            "ros_topic_name": ros_topic,
            "gz_topic_name": gz_topic,
            "ros_type_name": "sensor_msgs/msg/LaserScan",
            "gz_type_name": "gz.msgs.LaserScan",
            "direction": "GZ_TO_ROS",
            "lazy": False,
        }
    ]


def _build_imu_bridge_entries(
    periph: dict,
    model_name: str,
    world_name: str,
    simulation_config: dict | None = None,
    platform: str = "gazebo",
) -> list:
    """Return one IMU bridge config dict for an imu peripheral."""
    ros_topic = periph.get("topic") or "/imu/data"
    ros_topic = ros_topic if ros_topic.startswith("/") else f"/{ros_topic}"
    gz_topic = ros_topic
    return [
        {
            "ros_topic_name": ros_topic,
            "gz_topic_name": gz_topic,
            "ros_type_name": "sensor_msgs/msg/Imu",
            "gz_type_name": "gz.msgs.IMU",
            "direction": "GZ_TO_ROS",
            "lazy": False,
        }
    ]


# Registry: peripheral type → bridge entry builder function.
_PERIPHERAL_BRIDGE_BUILDERS = {
    "camera": _build_camera_bridge_entries,
    "lidar": _build_lidar_bridge_entries,
    "imu": _build_imu_bridge_entries,
    # "rgbd":  _build_rgbd_bridge_entries,   # reserved
}

# Fixed path for the generated bridge config (overwritten on each launch).
_BRIDGE_CONFIG_PATH = "/tmp/ros_gz_camera_bridge.yaml"


def generate_peripheral_sim_bridges(
    peripherals: list,
    model_name: str,
    world_name: str = "demo",
) -> list:
    """Generate a single bridge_node covering all peripheral cameras.

    Writes a YAML config to /tmp/ros_gz_camera_bridge.yaml and returns one
    bridge_node Node that reads it. bridge_node maps each Gazebo sensor topic
    directly to its ROS contract name — no remappings= required.

    Topic naming contract (output ROS topics):
      /camera/{name}/image_raw
      /camera/{name}/camera_info
      /camera/{name}/aligned_depth_to_color/image_raw when the camera requests
      depth via streams, align_depth, enable_depth, or enable_pointcloud.
    where {name} is YAML peripherals[].name.

    Args:
        peripherals: List of peripheral dicts from robot_config["peripherals"].
        model_name:  Gazebo model name (robot_config["name"]).
        world_name:  Gazebo world name, default "demo" (matches the sim_models scene worlds).

    Returns:
        List with one bridge_node Node action, or [] if no known peripherals.
    """
    bridge_entries = []
    for periph in peripherals:
        ptype = periph.get("type")
        builder = _PERIPHERAL_BRIDGE_BUILDERS.get(ptype)
        if builder:
            bridge_entries.extend(builder(periph, model_name, world_name))

    if not bridge_entries:
        return []

    with open(_BRIDGE_CONFIG_PATH, "w") as f:
        yaml.dump(bridge_entries, f, default_flow_style=False)

    return [
        Node(
            package="ros_gz_bridge",
            executable="bridge_node",
            name="peripheral_bridge",
            parameters=[{"config_file": _BRIDGE_CONFIG_PATH}],
            output="screen",
        )
    ]
