"""Perception system launch builders.

This module handles:
- Camera driver nodes (usb_cam, camera_ros, realsense2_camera)
- LiDAR driver nodes
- Static TF publishers for peripheral frames
- Virtual camera relays
"""

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.perception")


def generate_camera_nodes(robot_config, use_sim=False):
    """Generate physical camera driver nodes from configuration.

    Args:
        robot_config: Robot configuration dict
        use_sim: Simulation mode (if True, skip physical cameras)

    Returns:
        List of Node actions for cameras
    """
    is_sim = parse_bool(use_sim, default=False)
    if is_sim:
        logger.info("Skipping physical camera drivers in sim mode")
        return []

    nodes = []

    peripherals = robot_config.get("peripherals", [])
    logger.info(f"Generating nodes for {len(peripherals)} peripherals (use_sim={is_sim})")
    for periph in peripherals:
        periph_type = periph.get("type")

        # Skip virtual cameras in first pass
        if periph_type == "virtual_camera":
            continue

        if periph_type != "camera":
            continue

        name = periph["name"]
        driver = periph.get("driver", "opencv")
        logger.info(f"Creating camera node: {name} (driver={driver})")

        if driver == "opencv":
            # Use usb_cam package
            index = periph.get("index", 0)
            video_device = f"/dev/video{index}" if isinstance(index, int) else index

            params = {
                "use_sim_time": is_sim,
                "camera_name": name,
                "framerate": float(periph.get("fps", 30)),
                "image_width": periph.get("width", 640),
                "image_height": periph.get("height", 480),
                "pixel_format": periph.get("pixel_format", "mjpeg"),
                "brightness": periph.get("brightness", 0),
                "camera_frame_id": periph.get("frame_id", f"camera_{name}_frame"),
                "video_device": video_device,
            }

            if "camera_info_url" in periph:
                params["camera_info_url"] = periph["camera_info_url"]

            # Optional parameters
            for key in ["contrast", "saturation", "sharpness"]:
                if key in periph:
                    params[key] = periph[key]

            logger.info(f"  Camera params: {params}")

            nodes.append(
                Node(
                    package="usb_cam",
                    executable="usb_cam_node_exe",
                    name=f"{name}_camera",
                    parameters=[params],
                    remappings=[
                        ("image_raw", f"/camera/{name}/image_raw"),
                        ("camera_info", f"/camera/{name}/camera_info"),
                    ],
                    output="screen",
                )
            )

        elif driver == "camera_ros":
            params = {
                "camera": periph.get("index", periph.get("camera", 0)),
                "format": periph.get("format", "MJPEG"),
                "width": periph.get("width", 640),
                "height": periph.get("height", 480),
                "framerate": float(periph.get("fps", 30)),
            }
            if "camera_info_url" in periph:
                params["camera_info_url"] = periph["camera_info_url"]

            print(f"[robot_config]   camera_ros params: {params}")

            nodes.append(
                Node(
                    package="camera_ros",
                    executable="camera_node",
                    namespace=f"/camera/{name}",
                    name=f"{name}_camera",
                    parameters=[params],
                    output="screen",
                    respawn=bool(periph.get("respawn", True)),
                )
            )

        elif driver == "realsense":
            # Use realsense2_camera package
            w = periph.get("width", 640)
            h = periph.get("height", 480)
            fps = periph.get("fps", 30)
            streams = periph.get("streams") or (
                ["color"]
                + (["depth"] if periph.get("align_depth", False) else [])
                + (["pointcloud"] if periph.get("enable_pointcloud", False) else [])
            )
            params = {
                "use_sim_time": is_sim,
                "camera_name": name,
                "rgb_camera.color_profile": f"{w}x{h}x{fps}",
                "color_format": periph.get("pixel_format", "bgr8").upper(),
                "camera_frame_id": periph.get("frame_id", f"camera_{name}_frame"),
                "align_depth.enable": "depth" in streams,
                "pointcloud.enable": "pointcloud" in streams,
                "pointcloud.stream_filter": 2 if "pointcloud" in streams else 0,
                "pointcloud.ordered_pc": False,
                "enable_sync": periph.get("enable_sync", True),
            }

            if "depth_width" in periph:
                params["depth_width"] = periph["depth_width"]
                params["depth_height"] = periph["depth_height"]
            if "depth_fps" in periph:
                params["depth_fps"] = periph["depth_fps"]
            if "serial_number" in periph:
                params["serial_no"] = str(periph["serial_number"])

            logger.info(f"  RealSense params: {params}")

            nodes.append(
                Node(
                    package="realsense2_camera",
                    executable="realsense2_camera_node",
                    name=f"{name}_camera",
                    parameters=[params],
                    remappings=[
                        (f"/camera/{name}/color/image_raw", f"/camera/{name}/image_raw"),
                        (f"/camera/{name}/color/camera_info", f"/camera/{name}/camera_info"),
                    ],
                    output="screen",
                )
            )

    return nodes


def generate_lidar_nodes(robot_config, use_sim=False):
    """Generate physical LiDAR driver nodes from configuration."""
    is_sim = parse_bool(use_sim, default=False)
    if is_sim:
        print("[robot_config] Skipping physical lidar drivers in sim mode")
        return []

    nodes = []
    peripherals = robot_config.get("peripherals", [])
    print(f"[robot_config] Generating lidar nodes from {len(peripherals)} peripherals (use_sim={is_sim})")

    for periph in peripherals:
        if periph.get("type") != "lidar":
            continue

        name = periph["name"]
        driver = periph.get("driver", "")
        if driver != "ldlidar":
            continue

        params = dict(periph.get("params", {}))
        if periph.get("frame_id") and "frame_id" not in params:
            params["frame_id"] = periph["frame_id"]
        if "port" in periph and "port_name" not in params:
            params["port_name"] = periph["port"]
        params.setdefault("use_sim_time", is_sim)

        print(f"[robot_config] Creating lidar node: {name} (driver={driver})")
        print(f"[robot_config]   LiDAR params: {params}")

        nodes.append(
            Node(
                package="ldlidar_ros2",
                executable="ldlidar_ros2_node",
                name=f"{name}_lidar",
                parameters=[params],
                output="screen",
                respawn=bool(periph.get("respawn", True)),
            )
        )

    return nodes


def generate_virtual_camera_relays(robot_config):
    """Generate virtual camera relay nodes.

    Creates topic_tools relay nodes to duplicate existing camera topics
    for virtual cameras (e.g., wrist camera relayed from top camera).

    Args:
        robot_config: Robot configuration dict

    Returns:
        List of Node actions for virtual camera relays
    """
    nodes = []

    peripherals = robot_config.get("peripherals", [])
    for periph in peripherals:
        if periph.get("type") != "camera":
            continue

        driver = periph.get("driver", "")

        # Check if this is a virtual camera (driver == "virtual")
        if driver != "virtual":
            continue

        name = periph["name"]
        source_topic = periph.get("source_topic")

        # Construct target topic
        target_topic = f"/camera/{name}/image_raw"

        if not source_topic:
            logger.warning(f"Virtual camera {name} missing source_topic")
            continue

        logger.info(f"Creating virtual camera relay: {name}")
        logger.info(f"  {source_topic} -> {target_topic}")

        nodes.append(
            Node(
                package="topic_tools",
                executable="relay",
                name=f"{name}_relay",
                arguments=[source_topic, target_topic],
                output="screen",
            )
        )

    return nodes


def generate_tf_nodes(robot_config, use_sim=False):
    """Generate static TF publisher nodes for peripheral frames.

    Args:
        robot_config: Robot configuration dict
        use_sim: Simulation mode (if True, TF is published by robot_state_publisher from URDF)

    Returns:
        List of Node actions for TF publishers
    """
    is_sim = parse_bool(use_sim, default=False)
    if is_sim:
        # sim 模式下相机 TF 由 robot_state_publisher 从 URDF 发布，无需静态发布节点
        return []

    nodes = []

    peripherals = robot_config.get("peripherals", [])
    for periph in peripherals:
        name = periph.get("name")
        frame_id = periph.get("frame_id")
        optical_frame_id = periph.get("optical_frame_id")
        transform = periph.get("transform", {})

        if not all([frame_id, transform]):
            continue

        parent_frame = transform.get("parent_frame", "base_link")
        x = transform.get("x", 0.0)
        y = transform.get("y", 0.0)
        z = transform.get("z", 0.0)
        roll = transform.get("roll", 0.0)
        pitch = transform.get("pitch", 0.0)
        yaw = transform.get("yaw", 0.0)

        # Main frame TF
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"static_tf_{name}",
                arguments=[
                    str(x),
                    str(y),
                    str(z),
                    str(roll),
                    str(pitch),
                    str(yaw),
                    parent_frame,
                    frame_id,
                ],
                output="screen",
            )
        )

        # Optical frame TF (standard rotation for camera sensors)
        # Skip for RealSense: driver publishes its own optical frame TF internally
        if periph.get("driver") == "realsense":
            logger.info(
                f"  Skipping optical frame TF for RealSense (driver publishes {frame_id} -> {optical_frame_id})"
            )
        elif periph.get("type") == "camera" and optical_frame_id:
            nodes.append(
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name=f"static_tf_{name}_optical",
                    arguments=[
                        "0",
                        "0",
                        "0",
                        "-0.5",
                        "0.5",
                        "-0.5",
                        "0.5",  # ROS optical frame convention
                        frame_id,
                        optical_frame_id,
                    ],
                    output="screen",
                )
            )

    return nodes
