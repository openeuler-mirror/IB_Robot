"""Sensor peripheral composition for ``<robot>_robot`` runtime launches.

D12 (sensor peripherals belong to the runtime): the robot runtime exposes the
robot's complete hardware surface as ROS interfaces. The driver knowledge --
which ROS driver node a peripheral maps to, its parameters, and the static TF
mounting its frames -- lives here in the contract layer and is invoked by each
runtime's ``launch/runtime.launch.py``. ``robot_config`` passes the deployment
peripheral inventory through as data (``peripherals_file``) and never
interprets drivers.

Peripheral sources are merged by ``(type, name)``:

- the runtime profile's ``peripherals:`` list is the robot's default sensor
  set (the robot's hardware surface);
- the deployment robot YAML's ``peripherals:`` list overrides matching
  entries and may add deployment-specific ones (lidar IPs, camera serials).

In simulated transport the physical drivers are replaced by
``synthetic_perception`` publishers on the same topics, so the perception
contract holds in both modes (the same principle the SDK simulated transport
applies to motors).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from launch_ros.actions import Node

from robot_runtime.interface_description import (
    build_description,
    camera_streams,
    merge_peripherals,
    peripheral_interfaces,
)
from robot_runtime.path_subst import resolve_path

logger = logging.getLogger("robot_runtime.peripherals")

PERCEPTION_CAMERA = "perception.camera"
PERCEPTION_LIDAR = "perception.lidar"

_IMAGE_TYPE = "sensor_msgs/msg/Image"
_CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"
_POINTCLOUD_TYPE = "sensor_msgs/msg/PointCloud2"


def load_peripherals_file(path: str) -> dict[str, Any]:
    """Load a deployment peripherals fragment ``{peripherals: [...], fast_lio: {...}}``."""
    raw = (path or "").strip()
    if not raw:
        return {}
    with open(resolve_path(raw), encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"peripherals file must be a mapping: {raw}")
    return data


def perception_capabilities(peripherals: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Perception capability specs (RuntimeStatus surface) for a peripheral list."""
    caps: dict[str, dict[str, Any]] = {}
    for spec in peripheral_interfaces(peripherals).values():
        caps.setdefault(spec["capability"], {"topics": []})["topics"].append(spec["endpoint"])
    return caps


def synthetic_perception_nodes(peripherals: list[dict[str, Any]] | None, description: dict | None = None) -> list[Node]:
    """Simulated-transport perception: publish the declared topics with synthetic data."""
    if not peripherals and not (
        description and any(key.startswith("localization.") for key in description["interfaces"])
    ):
        return []
    if description is None:
        description = build_description(
            {
                "runtime": {"name": "synthetic_perception", "version": "0.1.0"},
                "capabilities": {},
                "peripherals": peripherals,
            },
            simulated=True,
        )
    return [
        Node(
            package="robot_runtime",
            executable="synthetic_perception",
            name="synthetic_perception",
            output="screen",
            parameters=[{"description_json": json.dumps(description)}],
        )
    ]


# ---------------------------------------------------------------------------
# Physical driver composition (ported from robot_config launch builders; the
# robot_config copies remain for provider-less legacy configurations only)
# ---------------------------------------------------------------------------

_ISP_BOOL_KEYS = {"auto_white_balance", "autoexposure", "autofocus"}
# Conservative UVC bounds; the driver further clips to the device's range.
_ISP_INT_RANGES = {
    "brightness": (-255, 255),
    "contrast": (0, 255),
    "saturation": (0, 255),
    "sharpness": (0, 255),
    "gain": (0, 255),
    "white_balance": (2000, 10000),
    "exposure": (0, 20000),
    "focus": (0, 1023),
}
_ISP_KEYS = tuple(_ISP_INT_RANGES) + tuple(sorted(_ISP_BOOL_KEYS))


def load_isp_override(camera_name: str) -> dict[str, Any]:
    """Fail-safe per-camera ISP override under ``$ROS_HOME/ibrobot/camera_isp_overrides``.

    Calibration state produced by ``camera_isp_calibrator`` (dataset_tools)
    persists across launches without touching the YAML SSOT. Any error or
    unknown key degrades to an empty/partial dict. Unset or empty ``ROS_HOME``
    falls back to ``~/.ros``, matching the calibrator's storage convention.
    """
    ros_home = os.environ.get("ROS_HOME") or str(Path.home() / ".ros")
    path = Path(ros_home) / "ibrobot" / "camera_isp_overrides" / f"{camera_name}.json"
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        logger.info("No ISP override for camera %s (%s)", camera_name, path)
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable ISP override %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("ignoring non-mapping ISP override %s", path)
        return {}
    override = {}
    dropped = []
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if key not in _ISP_KEYS:
            dropped.append(key)
            continue
        if key in _ISP_BOOL_KEYS:
            if isinstance(value, bool):
                override[key] = value
                continue
        elif not isinstance(value, bool) and isinstance(value, int | float):
            try:
                ivalue = int(value)
            except (ValueError, OverflowError):
                pass
            else:
                lo, hi = _ISP_INT_RANGES[key]
                if lo <= ivalue <= hi:
                    override[key] = ivalue
                    continue
        logger.warning("dropping invalid ISP value for %s: %s=%r", camera_name, key, value)
    if dropped:
        logger.warning("dropping unknown ISP keys for %s: %s", camera_name, sorted(dropped))
    return override


def _camera_nodes(periph: dict[str, Any], use_sim: bool) -> list[Node]:
    name = str(periph["name"])
    driver = str(periph.get("driver", "opencv"))
    nodes: list[Node] = []

    if driver == "opencv":
        index = periph.get("index", 0)
        video_device = f"/dev/video{index}" if isinstance(index, int) else index
        params: dict[str, Any] = {
            "use_sim_time": use_sim,
            "camera_name": name,
            "framerate": float(periph.get("fps", 30)),
            "image_width": periph.get("width", 640),
            "image_height": periph.get("height", 480),
            "pixel_format": periph.get("pixel_format", "mjpeg"),
            "brightness": periph.get("brightness", 0),
            "frame_id": periph.get("frame_id", f"camera_{name}_frame"),
            "video_device": video_device,
        }
        if "camera_info_url" in periph:
            params["camera_info_url"] = periph["camera_info_url"]
        for key in (*_ISP_KEYS, "io_method"):
            if key in periph:
                params[key] = periph[key]
        params.update(load_isp_override(name))
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
        return nodes

    if driver == "camera_ros":
        params = {
            "camera": periph.get("index", periph.get("camera", 0)),
            "format": periph.get("format", "MJPEG"),
            "width": periph.get("width", 640),
            "height": periph.get("height", 480),
            "framerate": float(periph.get("fps", 30)),
        }
        if "camera_info_url" in periph:
            params["camera_info_url"] = periph["camera_info_url"]
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
        return nodes

    if driver == "realsense":
        w = periph.get("width", 640)
        h = periph.get("height", 480)
        fps = periph.get("fps", 30)
        driver_camera_name = periph.get("driver_camera_name", f"{name}_camera")
        driver_topic_prefix = periph.get("driver_topic_prefix", f"/camera/{driver_camera_name}")
        frame_id = periph.get("frame_id", f"{driver_camera_name}_link")
        direct_topic_remap = bool(periph.get("direct_topic_remap", False))
        align_depth = periph.get("align_depth", False)
        streams = camera_streams(periph)
        align_depth = align_depth and "depth" in streams and "color" in streams
        driver_params = {
            "camera_namespace": "camera",
            "camera_name": driver_camera_name,
            "base_frame_id": frame_id,
            "tf_prefix": "",
            "publish_tf": True,
            "enable_color": "color" in streams,
            "enable_depth": "depth" in streams,
            "enable_infra": False,
            "enable_infra1": False,
            "enable_infra2": False,
            "enable_motion": False,
            "enable_rgbd": False,
            "rgb_camera.color_profile": f"{w}x{h}x{fps}",
            "depth_module.depth_profile": f"{w}x{h}x{fps}",
            "align_depth.enable": align_depth,
            "pointcloud.enable": "pointcloud" in streams,
            "pointcloud.stream_filter": 2 if "pointcloud" in streams else 0,
            "pointcloud.ordered_pc": False,
            "enable_sync": bool(periph.get("enable_sync", True)),
            "initial_reset": bool(periph.get("initial_reset", True)),
            "enable_gyro": False,
            "enable_accel": False,
            "unite_imu_method": 0,
        }
        if "color_format" in periph:
            driver_params["rgb_camera.color_format"] = periph["color_format"]
        if "depth_format" in periph:
            driver_params["depth_module.depth_format"] = periph["depth_format"]
        if "depth_width" in periph:
            depth_fps = periph.get("depth_fps", fps)
            driver_params["depth_module.depth_profile"] = (
                f"{periph['depth_width']}x{periph['depth_height']}x{depth_fps}"
            )
        if "serial_number" in periph:
            driver_params["serial_no"] = str(periph["serial_number"])

        raw_depth_source_topic = f"{driver_topic_prefix}/depth/image_rect_raw"
        aligned_depth_source_topic = (
            f"{driver_topic_prefix}/aligned_depth_to_color/image_raw" if align_depth else raw_depth_source_topic
        )
        aligned_depth_target_topic = (
            f"/camera/{name}/aligned_depth_to_color/image_raw"
            if align_depth
            else f"/camera/{name}/depth/image_rect_raw"
        )
        aligned_camera_info_source_topic = (
            f"{driver_topic_prefix}/aligned_depth_to_color/camera_info"
            if align_depth
            else f"{driver_topic_prefix}/depth/camera_info"
        )
        driver_remappings = []
        if direct_topic_remap:
            # Large RGB-D payloads must not cross a second DDS pair merely to
            # normalize topic names; CameraInfo still uses the small relay so
            # its frame_id is normalized to the configured optical frame.
            driver_remappings.extend(
                [
                    (f"{driver_topic_prefix}/color/image_raw", f"/camera/{name}/image_raw"),
                    (raw_depth_source_topic, f"/camera/{name}/depth/image_rect_raw"),
                ]
            )
            if align_depth:
                driver_remappings.append((aligned_depth_source_topic, aligned_depth_target_topic))
            if "pointcloud" in streams:
                driver_remappings.append(
                    (f"{driver_topic_prefix}/depth/color/points", f"/camera/{name}/depth/color/points")
                )

        nodes.append(
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace="camera",
                name=driver_camera_name,
                parameters=[driver_params],
                remappings=driver_remappings,
                arguments=["--ros-args", "--log-level", "info"],
                output="screen",
                emulate_tty=True,
            )
        )

        relay_topics = [
            (
                f"{driver_topic_prefix}/color/camera_info",
                f"/camera/{name}/camera_info",
                f"{name}_camera_info_relay",
                _CAMERA_INFO_TYPE,
                periph.get("optical_frame_id"),
            ),
            (
                f"{driver_topic_prefix}/depth/camera_info",
                f"/camera/{name}/depth/camera_info",
                f"{name}_depth_camera_info_relay",
                _CAMERA_INFO_TYPE,
                periph.get("optical_frame_id"),
            ),
        ]
        if align_depth:
            relay_topics.append(
                (
                    aligned_camera_info_source_topic,
                    f"/camera/{name}/aligned_depth_to_color/camera_info",
                    f"{name}_aligned_depth_camera_info_relay",
                    _CAMERA_INFO_TYPE,
                    periph.get("optical_frame_id"),
                )
            )
        if not direct_topic_remap:
            relay_topics.extend(
                [
                    (
                        f"{driver_topic_prefix}/color/image_raw",
                        f"/camera/{name}/image_raw",
                        f"{name}_color_image_relay",
                        _IMAGE_TYPE,
                        periph.get("optical_frame_id"),
                    ),
                    (
                        raw_depth_source_topic,
                        f"/camera/{name}/depth/image_rect_raw",
                        f"{name}_depth_image_relay",
                        _IMAGE_TYPE,
                        periph.get("optical_frame_id"),
                    ),
                ]
            )
            if align_depth:
                relay_topics.append(
                    (
                        aligned_depth_source_topic,
                        aligned_depth_target_topic,
                        f"{name}_aligned_depth_image_relay",
                        _IMAGE_TYPE,
                        periph.get("optical_frame_id"),
                    )
                )
        if "pointcloud" in streams and not direct_topic_remap:
            relay_topics.append(
                (
                    f"{driver_topic_prefix}/depth/color/points",
                    f"/camera/{name}/depth/color/points",
                    f"{name}_pointcloud_relay",
                    _POINTCLOUD_TYPE,
                    periph.get("optical_frame_id"),
                )
            )
        for source_topic, target_topic, relay_name, message_type, target_frame_id in relay_topics:
            if (
                ("color/camera_info" in source_topic or "color/image_raw" in source_topic)
                and "color" not in streams
                and "aligned_depth" not in source_topic
            ):
                continue
            if "/depth/" in source_topic and "depth" not in streams:
                continue
            if source_topic == target_topic:
                continue
            relay_args = [source_topic, target_topic, message_type]
            if target_frame_id:
                relay_args.append(target_frame_id)
            nodes.append(
                Node(
                    package="robot_runtime",
                    executable="topic_relay",
                    name=relay_name,
                    arguments=relay_args,
                    output="screen",
                )
            )
        return nodes

    logger.warning("unsupported camera driver %r for peripheral %r, skipping", driver, name)
    return nodes


def _livox_mid360_nodes(periph: dict[str, Any], use_sim: bool) -> list[Node]:
    config_path = resolve_path(periph["user_config_path"])
    with open(config_path, encoding="utf-8") as config_file:
        livox_config = json.load(config_file)

    host_ip = periph["host_ip"]
    host_net_info = livox_config["MID360"]["host_net_info"]
    for key in ("cmd_data_ip", "push_msg_ip", "point_data_ip", "imu_data_ip"):
        host_net_info[key] = host_ip
    livox_config["lidar_configs"][0]["ip"] = periph["lidar_ip"]

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="ibrobot_mid360_", suffix=".json", delete=False, encoding="utf-8"
    ) as runtime_config:
        json.dump(livox_config, runtime_config, indent=2)

    driver_params = {
        "xfer_format": periph.get("xfer_format", 1),
        "multi_topic": periph.get("multi_topic", 0),
        "data_src": periph.get("data_src", 0),
        "publish_freq": float(periph.get("publish_freq", 10.0)),
        "output_data_type": periph.get("output_data_type", 0),
        "frame_id": periph.get("frame_id", "livox_frame"),
        "lvx_file_path": periph.get("lvx_file_path", "/tmp/livox_test.lvx"),
        "user_config_path": runtime_config.name,
        "cmdline_input_bd_code": periph.get("cmdline_input_bd_code", "livox0000000001"),
        "use_sim_time": use_sim,
    }
    nodes = [
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name=str(periph.get("node_name", "livox_lidar_publisher")),
            parameters=[driver_params],
            remappings=[
                ("/livox/lidar", str(periph.get("pointcloud_topic", "/livox/lidar"))),
                ("/livox/imu", str(periph.get("imu_topic", "/livox/imu"))),
            ],
            output="screen",
            respawn=bool(periph.get("respawn", True)),
        )
    ]

    scan_converter = periph.get("scan_converter", {})
    if scan_converter.get("enabled", False):
        scan_params = {
            "target_frame": scan_converter.get("target_frame", periph.get("frame_id", "livox_frame")),
            "transform_tolerance": scan_converter.get("transform_tolerance", 0.01),
            "min_height": scan_converter.get("min_height", -0.2),
            "max_height": scan_converter.get("max_height", 0.5),
            "angle_min": scan_converter.get("angle_min", -3.14159265),
            "angle_max": scan_converter.get("angle_max", 3.14159265),
            "angle_increment": scan_converter.get("angle_increment", 0.00872665),
            "scan_time": scan_converter.get("scan_time", 0.1),
            "range_min": scan_converter.get("range_min", 0.1),
            "range_max": scan_converter.get("range_max", 20.0),
            "use_inf": scan_converter.get("use_inf", True),
            "inf_epsilon": scan_converter.get("inf_epsilon", 1.0),
            "use_sim_time": use_sim,
        }
        if "queue_size" in scan_converter:
            scan_params["queue_size"] = int(scan_converter["queue_size"])
        nodes.append(
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name=f"{periph['name']}_pointcloud_to_laserscan",
                parameters=[scan_params],
                remappings=[
                    ("cloud_in", str(scan_converter.get("pointcloud_topic", "/cloud_registered_body"))),
                    ("scan", str(scan_converter.get("scan_topic", "/scan"))),
                ],
                output="screen",
                respawn=bool(scan_converter.get("respawn", True)),
            )
        )
    return nodes


def _ldlidar_nodes(periph: dict[str, Any], use_sim: bool) -> list[Node]:
    params = dict(periph.get("params", {}))
    if periph.get("frame_id") and "frame_id" not in params:
        params["frame_id"] = periph["frame_id"]
    if "port" in periph and "port_name" not in params:
        params["port_name"] = periph["port"]
    params.setdefault("use_sim_time", use_sim)
    return [
        Node(
            package="ldlidar_ros2",
            executable="ldlidar_ros2_node",
            name=f"{periph['name']}_lidar",
            parameters=[params],
            output="screen",
            respawn=bool(periph.get("respawn", True)),
        )
    ]


def _static_tf_node(name: str, arguments: list[str]) -> Node:
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=arguments,
        output="screen",
    )


def _peripheral_tf_nodes(periph: dict[str, Any]) -> list[Node]:
    """Static TF mounting a peripheral's frames on the robot (skip in sim: URDF publishes)."""
    name = periph.get("name")
    frame_id = periph.get("frame_id")
    transform = periph.get("transform", {})
    if not all([name, frame_id, transform]):
        return []

    nodes: list[Node] = []
    parent_frame = transform.get("parent_frame", "base_link")
    x, y, z = (transform.get(key, 0.0) for key in ("x", "y", "z"))
    roll, pitch, yaw = (transform.get(key, 0.0) for key in ("roll", "pitch", "yaw"))
    quaternion = [transform.get(key) for key in ("qx", "qy", "qz", "qw")]
    if all(value is not None for value in quaternion):
        rotation_arguments = [
            "--qx",
            str(quaternion[0]),
            "--qy",
            str(quaternion[1]),
            "--qz",
            str(quaternion[2]),
            "--qw",
            str(quaternion[3]),
        ]
    else:
        rotation_arguments = ["--roll", str(roll), "--pitch", str(pitch), "--yaw", str(yaw)]

    nodes.append(
        _static_tf_node(
            f"static_tf_{name}",
            [
                "--x",
                str(x),
                "--y",
                str(y),
                "--z",
                str(z),
                *rotation_arguments,
                "--frame-id",
                parent_frame,
                "--child-frame-id",
                frame_id,
            ],
        )
    )

    if periph.get("driver") == "realsense":
        # realsense2_camera prefixes the configured base_frame_id with the
        # camera name; bridge the configured frame to the driver root so the
        # native color/depth TF tree stays connected to the robot.
        driver_camera_name = periph.get("driver_camera_name", f"{name}_camera")
        driver_frame_id = periph.get("driver_frame_id", f"{driver_camera_name}_{frame_id}")
        if frame_id != driver_frame_id:
            nodes.append(
                _static_tf_node(
                    f"static_tf_{name}_driver_bridge",
                    [
                        "--x",
                        "0",
                        "--y",
                        "0",
                        "--z",
                        "0",
                        "--roll",
                        "0",
                        "--pitch",
                        "0",
                        "--yaw",
                        "0",
                        "--frame-id",
                        frame_id,
                        "--child-frame-id",
                        driver_frame_id,
                    ],
                )
            )
    if periph.get("type") == "camera" and periph.get("optical_frame_id"):
        # ROS optical frame convention
        nodes.append(
            _static_tf_node(
                f"static_tf_{name}_optical",
                [
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--qx",
                    "-0.5",
                    "--qy",
                    "0.5",
                    "--qz",
                    "-0.5",
                    "--qw",
                    "0.5",
                    "--frame-id",
                    frame_id,
                    "--child-frame-id",
                    str(periph["optical_frame_id"]),
                ],
            )
        )
    return nodes


def peripheral_nodes(peripherals: list[dict[str, Any]] | None, use_sim: bool) -> list[Node]:
    """Physical driver nodes + static TF for a peripheral list (empty in sim)."""
    if use_sim:
        return []
    nodes: list[Node] = []
    peripheral_interfaces(peripherals)  # Validate the advertised surface before launching drivers.
    for periph in merge_peripherals(peripherals, []):
        ptype = str(periph.get("type"))
        if ptype == "camera":
            if periph.get("driver") == "virtual":
                continue  # topic-level relay, composed by the generic layer
            nodes.extend(_camera_nodes(periph, use_sim))
        elif ptype == "lidar":
            driver = str(periph.get("driver", ""))
            if driver == "livox_mid360":
                nodes.extend(_livox_mid360_nodes(periph, use_sim))
            elif driver == "ldlidar":
                nodes.extend(_ldlidar_nodes(periph, use_sim))
            else:
                logger.warning("unsupported lidar driver %r for peripheral %r, skipping", driver, periph.get("name"))
        elif ptype == "microphone":
            continue  # audio contract data consumed by the voice stack, no ROS node
        else:
            logger.warning("unsupported peripheral type %r (%r), skipping", ptype, periph.get("name"))
        nodes.extend(_peripheral_tf_nodes(periph))
    return nodes


def fast_lio_nodes(config: dict[str, Any], *, bridge_package: str, use_sim: bool) -> list[Node]:
    """Lidar odometry chain: fastlio_mapping + the odom bridge (robot-side host package)."""
    if not config.get("enabled", False) or use_sim:
        return []
    params_file = config.get("params_file", "")
    if not params_file:
        raise ValueError("fast_lio.params_file is required")

    raw_odom_topic = config.get("raw_odom_topic", "/fast_lio/odometry_raw")
    mapping = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fast_lio",
        output="screen",
        parameters=[resolve_path(params_file), {"use_sim_time": False}],
        remappings=[
            ("/Odometry", raw_odom_topic),
            ("/livox/lidar", config.get("lidar_topic", "/livox/lidar")),
            ("/livox/imu", config.get("imu_topic", "/livox/imu")),
            ("/tf", config.get("isolated_tf_topic", "/fast_lio/tf_raw")),
            ("/tf_static", config.get("isolated_tf_static_topic", "/fast_lio/tf_static_raw")),
        ],
        respawn=bool(config.get("respawn", True)),
        respawn_delay=float(config.get("respawn_delay", 2.0)),
    )
    bridge = Node(
        package=bridge_package,
        executable="fast_lio_odom_bridge",
        name="fast_lio_odom_bridge",
        output="screen",
        parameters=[
            {
                "source_topic": raw_odom_topic,
                "output_topic": config.get("output_topic", "/odometry/filtered"),
                "source_odom_frame": config.get("source_odom_frame", "camera_init"),
                "source_body_frame": config.get("source_body_frame", "body"),
                "output_odom_frame": config.get("odom_frame", "odom"),
                "output_base_frame": config.get("base_frame", "base_link"),
                "body_to_base_translation": config.get("body_to_base_translation", [0.0, 0.0, 0.0]),
                "body_to_base_rotation": config.get("body_to_base_rotation", [0.0, 0.0, 0.0, 1.0]),
                "publish_tf": bool(config.get("publish_tf", True)),
                "planar_output": bool(config.get("planar_output", False)),
                "max_future_skew_sec": float(config.get("max_future_skew_sec", 0.1)),
            }
        ],
    )
    return [mapping, bridge]
