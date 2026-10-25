"""Robot-agnostic Gazebo sensor helpers shared by runtime providers and generic launch.

Moved from ``robot_config`` (openspec change ``so101-gazebo-runtime-transport``
D6.1): the camera-sensor URDF fragment builder, the gz camera bridge entries
and the override-aware pose resolver are robot-agnostic algorithms, so they
live in the contract layer and ``robot_config`` re-imports them — provider-less
generic Gazebo robots keep their existing camera topics, frames and poses
through the same call sites.

The functions here never import ``robot_config``; robot-specific values (the
generic preset table) are injected by the caller through ``default_pose_loader``.

Simulated lidar and IMU sensors (openspec ``lekiwi-gazebo-runtime-platform``
D5) join the cameras here: ``lidar_sensor_sdf``/``imu_sensor_sdf`` generate
``<gazebo reference>`` sensor blocks from a peripheral declaration (no URDF
link for the sensor frame), ``lidar_imu_sensor_urdf`` composes them for a
peripheral list, and ``lidar_bridge_entries``/``imu_bridge_entries`` add the
matching ros_gz bridge entries.

Extensions for the provider path (D6, spec "Simulated cameras honour the
advertised camera contract") are opt-in so the re-imported generic behaviour
is unchanged:

- ``frame_contract=True`` adds the optical-frame link/joint and stamps the gz
  sensor with ``<gz_frame_id>`` so the published ``header.frame_id`` equals the
  advertised frame (gz Fortress otherwise publishes the scoped
  ``model/link/sensor`` name).
- ``full_streams=True`` bridges every stream the descriptor advertises
  (color, color info, native depth, aligned depth and their infos) instead of
  the legacy color + aligned-depth subset.
- ``spawn_pose`` carries the full spawn transform (x, y, z, yaw); world-parented
  sensors are placed by the inverse of that transform. With yaw 0 this
  degenerates to the historical xyz subtraction.

Pose resolution precedence (D6.3): user-saved override (``~/.ros/ibrobot/
sim_camera_overrides/<name>.yaml``, written by the sim_models camera adjuster)
> declared per-platform default (``use_default_transform`` preset via the
caller's loader, or the peripheral's ``simulation.<platform>.transform``) >
the plain ``transform`` field.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml

from robot_runtime.profile import LIDAR_SIM_PLATFORMS


def _override_base() -> Path:
    """The user override directory (resolved lazily so tests can redirect HOME)."""
    return Path.home() / ".ros" / "ibrobot" / "sim_camera_overrides"


# Standard body→optical rotation (x right, y down, z forward) as extrinsic
# XYZ rpy; identical to the static TF the physical driver path publishes.
_OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)


def _parse_bool(value, default: bool = False) -> bool:
    """Parse YAML-bool-ish values (mirrors ``robot_config.utils.parse_bool``)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return default


def camera_requests_depth(periph: dict) -> bool:
    """Whether a camera peripheral requests any depth-capable stream.

    Mirrors ``robot_config.peripherals.camera.camera_dict_requests_depth``
    exactly so the shared builders keep the generic path's behaviour.
    """
    streams = periph.get("streams") or []
    if isinstance(streams, str):
        streams = [streams]
    stream_names = {str(stream).strip().lower() for stream in streams}
    return (
        "depth" in stream_names
        or "pointcloud" in stream_names
        or _parse_bool(periph.get("align_depth", False), default=False)
        or _parse_bool(periph.get("enable_depth", False), default=False)
        or _parse_bool(periph.get("enable_pointcloud", False), default=False)
    )


def load_camera_override(camera_name: str, *, fallback: dict | None = None) -> dict | None:
    """Load a user-saved Gazebo calibration override for one camera.

    Returns None when the override file is absent, unreadable, or still a
    placeholder stub; callers distinguish user-saved calibration from defaults.
    The file format is unchanged from the sim_models camera adjuster.
    ``fallback`` supplies the parent_frame/fovy defaults for override files
    that omit them (the generic path passes its preset table entry).
    """
    try:
        override_path = _override_base() / f"{camera_name}.yaml"
        if not override_path.exists():
            return None
        data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        pose = data.get("pose", {}) or {}
        # Skip stub overrides (all-zero placeholder poses would silently break
        # the simulation); mirrors sim_camera_adjuster's handling.
        pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        if bool(data.get("_is_stub")) or all(float(pose.get(k, 0.0)) == 0.0 for k in pose_keys):
            return None
        default = fallback or {}
        return {
            "parent_frame": data.get("parent_frame", default.get("parent_frame", "base")),
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "roll": float(pose.get("roll", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "fovy": float(data.get("fovy_deg", default.get("fovy", 60))),
        }
    except Exception:
        return None


def resolve_camera_pose(periph: dict, platform: str = "gazebo", *, default_pose_loader=None) -> tuple[dict, Any]:
    """Resolve a camera's effective (transform, fovy) for a simulation platform.

    Precedence: user override > declared default > explicit transform. The
    declared default is either the caller's preset table (``use_default_transform``,
    the generic provider-less mechanism) or the peripheral's own
    ``simulation.<platform>.transform`` (+ ``fovy``), the runtime-profile
    mechanism. Overrides apply in Gazebo convention and only on the gazebo
    platform, exactly as before this change.
    """
    name = periph.get("name", "camera")
    transform = dict(periph.get("transform") or {})
    fovy = periph.get("fovy", 60)
    default: dict | None = None
    if periph.get("use_default_transform", False) and default_pose_loader is not None:
        default = default_pose_loader(platform, name)
    if default is None:
        section = (periph.get("simulation") or {}).get(platform) or {}
        declared = section.get("transform") if isinstance(section, dict) else None
        if isinstance(declared, dict) and declared:
            default = dict(declared)
            if section.get("fovy") is not None:
                default["fovy"] = section["fovy"]
    if default is not None:
        override = load_camera_override(name) if platform == "gazebo" else None
        if override is not None:
            return override, override.get("fovy", 60)
        transform = {key: default.get(key, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
        transform["parent_frame"] = default.get("parent_frame", "base")
        fovy = default.get("fovy", fovy)
    return transform, fovy


def _rotate_yaw_inverse(pose: dict, spawn_pose: tuple[float, float, float, float]) -> dict:
    """Place a world-parented sensor under the model-local URDF ``world`` link.

    The URDF ``world`` link sits at the robot's spawn pose in the simulator,
    not at the true world origin, so a camera calibrated in world coordinates
    must be transformed by the inverse of the full spawn transform
    (translation and yaw); the historical xyz subtraction is the yaw-0 case.
    """
    x, y, z, yaw = (float(value) for value in spawn_pose)
    if yaw == 0.0:
        return {**pose, "x": pose["x"] - x, "y": pose["y"] - y, "z": pose["z"] - z}
    cos, sin = math.cos(-yaw), math.sin(-yaw)
    dx, dy, dz = pose["x"] - x, pose["y"] - y, pose["z"] - z
    return {
        **pose,
        "x": cos * dx - sin * dy,
        "y": sin * dx + cos * dy,
        "z": dz,
        "yaw": pose["yaw"] - yaw,
    }


def advertised_camera_frame(periph: dict) -> str:
    """The frame id the runtime description advertises for this camera's images."""
    image_frame, _info_frame = advertised_camera_frames(periph)
    if image_frame is not None:
        return image_frame
    name = periph.get("name", "camera")
    if str(periph.get("driver", "opencv")) == "realsense":
        return str(periph.get("optical_frame_id") or f"camera_{name}_optical_frame")
    return str(periph.get("frame_id") or f"camera_{name}_frame")


def advertised_camera_frames(periph: dict) -> tuple[str | None, str | None]:
    """(image frame, info frame) the description advertises — the single source.

    Convention: opencv/default → ``frame_id``; realsense → ``optical_frame_id``
    (the image frame is absent for ``direct_topic_remap`` — the native driver
    topic carries its own frames); camera_ros → no frames (the driver
    publishes its own). Used by the interface description and, via
    :func:`advertised_camera_frame`, by the gz ``frame_id`` and the MuJoCo
    sensor ``frame_name``.
    """
    name = periph.get("name", "camera")
    driver = str(periph.get("driver", "opencv"))
    if driver == "camera_ros":
        return None, None
    if driver == "realsense":
        optical = str(periph.get("optical_frame_id") or f"camera_{name}_optical_frame")
        return (None if periph.get("direct_topic_remap") else optical), optical
    frame = str(periph.get("frame_id") or f"camera_{name}_frame")
    return frame, frame


#: Lidar drivers the Gazebo platform stands in for; other drivers get no
#: simulated sensor and are not advertised on the platform (D5).
LIDAR_SIM_DRIVERS = ("livox_mid360", "ldlidar")

#: Lidar drivers whose physical device carries an integrated IMU.
LIDAR_INTEGRATED_IMU_DRIVERS = ("livox_mid360",)

# Default sensor parameters (starting values, D1: tunable through the
# peripheral's ``simulation.<platform>`` section; not schema).
_LIDAR_DEFAULTS = {
    "update_rate": 10.0,
    "horizontal_samples": 360,
    "horizontal_min_deg": -180.0,
    "horizontal_max_deg": 180.0,
    "vertical_samples": 1,
    "vertical_min_deg": -7.0,
    "vertical_max_deg": 52.0,
    "range_min": 0.1,
    "range_max": 40.0,
}
_IMU_DEFAULT_RATE = 200.0


def resolve_sensor_pose(periph: dict, platform: str = "gazebo") -> dict:
    """A non-camera sensor's simulator attachment transform (D5/D13).

    Precedence mirrors :func:`resolve_camera_pose` without the user-override
    layer (only cameras are user-calibrated): the peripheral's
    ``simulation.<platform>.transform`` (the simulator attachment pose, D13)
    over the plain ``transform`` (the public static TF). The resolved
    transform's ``parent_frame`` is the URDF link the sensor attaches to.
    """
    section = (periph.get("simulation") or {}).get(platform) or {}
    declared = section.get("transform") if isinstance(section, dict) else None
    if isinstance(declared, dict) and declared:
        return dict(declared)
    return dict(periph.get("transform") or {})


def _sensor_param(periph: dict, platform: str, key: str, default):
    section = (periph.get("simulation") or {}).get(platform) or {}
    if isinstance(section, dict) and section.get(key) is not None:
        return section[key]
    return default


def _pose_values(transform: dict) -> tuple[float, ...]:
    return tuple(float(transform.get(key, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw"))


def lidar_sensor_sdf(periph: dict, *, platform: str = "gazebo") -> str:
    """The ``<gazebo reference>`` block simulating one lidar peripheral (D5).

    A ``gpu_lidar`` sensor attached to the resolved parent link at the
    resolved mount pose. No URDF link or joint is added for the sensor frame:
    RSP must not publish a second parent for it (D13). One vertical sample
    renders a planar scan; ``<gz_frame_id>`` makes the bridged messages carry
    the advertised frame id.
    """
    name = periph["name"]
    transform = resolve_sensor_pose(periph, platform)
    parent = str(transform.get("parent_frame", "base_link"))
    x, y, z, roll, pitch, yaw = _pose_values(transform)
    horizontal_samples = int(
        _sensor_param(periph, platform, "horizontal_samples", _LIDAR_DEFAULTS["horizontal_samples"])
    )
    horizontal_min = math.radians(
        float(_sensor_param(periph, platform, "horizontal_min_deg", _LIDAR_DEFAULTS["horizontal_min_deg"]))
    )
    horizontal_max = math.radians(
        float(_sensor_param(periph, platform, "horizontal_max_deg", _LIDAR_DEFAULTS["horizontal_max_deg"]))
    )
    vertical_samples = int(_sensor_param(periph, platform, "vertical_samples", _LIDAR_DEFAULTS["vertical_samples"]))
    if vertical_samples <= 1:
        vertical_samples, vertical_min, vertical_max = 1, 0.0, 0.0
    else:
        vertical_min = math.radians(
            float(_sensor_param(periph, platform, "vertical_min_deg", _LIDAR_DEFAULTS["vertical_min_deg"]))
        )
        vertical_max = math.radians(
            float(_sensor_param(periph, platform, "vertical_max_deg", _LIDAR_DEFAULTS["vertical_max_deg"]))
        )
    update_rate = float(_sensor_param(periph, platform, "update_rate", periph.get("publish_freq", 10.0)))
    range_min = float(_sensor_param(periph, platform, "range_min", _LIDAR_DEFAULTS["range_min"]))
    range_max = float(_sensor_param(periph, platform, "range_max", _LIDAR_DEFAULTS["range_max"]))
    frame = str(periph.get("frame_id", f"{name}_frame"))
    return f"""
    <!-- {name} lidar (generated from the peripheral declaration) -->
    <gazebo reference="{escape(parent)}">
        <sensor type="gpu_lidar" name="{escape(str(name))}_lidar">
            <always_on>true</always_on>
            <update_rate>{update_rate}</update_rate>
            <gz_frame_id>{escape(frame)}</gz_frame_id>
            <topic>/{escape(str(name))}_lidar</topic>
            <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
            <ray>
                <scan>
                    <horizontal>
                        <samples>{horizontal_samples}</samples>
                        <resolution>1</resolution>
                        <min_angle>{horizontal_min}</min_angle>
                        <max_angle>{horizontal_max}</max_angle>
                    </horizontal>
                    <vertical>
                        <samples>{vertical_samples}</samples>
                        <resolution>1</resolution>
                        <min_angle>{vertical_min}</min_angle>
                        <max_angle>{vertical_max}</max_angle>
                    </vertical>
                </scan>
                <range>
                    <min>{range_min}</min>
                    <max>{range_max}</max>
                    <resolution>0.01</resolution>
                </range>
            </ray>
            <visualize>false</visualize>
        </sensor>
    </gazebo>"""


def imu_sensor_sdf(periph: dict, *, platform: str = "gazebo") -> str:
    """The ``<gazebo reference>`` block simulating one IMU (D5).

    Co-located with the lidar when the lidar driver has an integrated IMU
    (MID-360: same attachment pose and advertised frame, gz topic
    ``/<lidar name>_imu``, rate from ``simulation.<platform>.imu``), or on an
    ``imu`` peripheral's own mount (gz topic ``/<name>_imu``).
    """
    name = periph["name"]
    integrated = periph.get("type") == "lidar"
    section = (periph.get("simulation") or {}).get(platform) or {}
    if not isinstance(section, dict):
        section = {}
    if integrated:
        imu_section = section.get("imu")
        rate = float(
            imu_section.get("update_rate", _IMU_DEFAULT_RATE) if isinstance(imu_section, dict) else _IMU_DEFAULT_RATE
        )
    else:
        rate = float(section.get("update_rate", _IMU_DEFAULT_RATE))
    transform = resolve_sensor_pose(periph, platform)
    parent = str(transform.get("parent_frame", "base_link"))
    x, y, z, roll, pitch, yaw = _pose_values(transform)
    frame = str(periph.get("frame_id", f"{name}_frame"))
    return f"""
    <!-- {name} IMU (generated from the peripheral declaration) -->
    <gazebo reference="{escape(parent)}">
        <sensor type="imu" name="{escape(str(name))}_imu">
            <always_on>true</always_on>
            <update_rate>{rate}</update_rate>
            <gz_frame_id>{escape(frame)}</gz_frame_id>
            <topic>/{escape(str(name))}_imu</topic>
            <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
        </sensor>
    </gazebo>"""


def lidar_imu_sensor_urdf(peripherals: list, *, platform: str = "gazebo") -> str:
    """Lidar/IMU ``<gazebo reference>`` sensor blocks for a peripheral list (D5).

    Only peripherals the simulator stands in for are generated, next to the
    camera function; disabled entries are absent (callers pass the effective
    inventory, D5 inventory-first). A platform outside
    :data:`LIDAR_SIM_PLATFORMS` generates nothing: the rendering path
    consults the same constant (D14.2), so a platform that renders no lidar
    also reports no lidar-simulation capability.
    """
    if platform not in LIDAR_SIM_PLATFORMS:
        return ""
    parts: list[str] = []
    for periph in peripherals:
        if periph.get("disabled", False) or periph.get("enabled") is False:
            continue
        ptype = periph.get("type")
        if ptype == "lidar":
            driver = str(periph.get("driver", ""))
            if driver not in LIDAR_SIM_DRIVERS:
                continue
            parts.append(lidar_sensor_sdf(periph, platform=platform))
            if driver in LIDAR_INTEGRATED_IMU_DRIVERS:
                parts.append(imu_sensor_sdf(periph, platform=platform))
        elif ptype == "imu":
            parts.append(imu_sensor_sdf(periph, platform=platform))
    return "\n".join(parts)


def _sensor_bridge_entry(ros_topic: str, gz_topic: str, ros_type: str, gz_type: str) -> dict:
    return {
        "ros_topic_name": ros_topic,
        "gz_topic_name": gz_topic,
        "ros_type_name": ros_type,
        "gz_type_name": gz_type,
        "direction": "GZ_TO_ROS",
        "lazy": False,
    }


def lidar_bridge_entries(periph: dict) -> list[dict]:
    """ros_gz bridge entries for one lidar on its advertised endpoint (D5).

    The 3D lidar's cloud is bridged from the gz ``<topic>/points`` stream the
    ``gpu_lidar`` publishes next to its LaserScan. Gazebo reports no-hit
    beams as ``±inf``, so the cloud lands on the internal ``/<name>_lidar/raw``
    topic and the finite-point filter republishes it on the advertised
    endpoint (D5); a planar lidar bridges the LaserScan itself. Drivers the
    simulator does not stand in for bridge nothing.
    """
    driver = str(periph.get("driver", ""))
    if driver not in LIDAR_SIM_DRIVERS:
        return []
    name = periph["name"]
    if driver == "livox_mid360":
        return [
            _sensor_bridge_entry(
                f"/{name}_lidar/raw",
                f"/{name}_lidar/points",
                "sensor_msgs/msg/PointCloud2",
                "gz.msgs.PointCloudPacked",
            )
        ]
    return [
        _sensor_bridge_entry(
            str(periph.get("scan_topic", "/scan")), f"/{name}_lidar", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan"
        )
    ]


def lidar_finite_filter_nodes(peripherals: list) -> list:
    """The finite-point filter nodes for a peripheral list (D5).

    One ``robot_runtime/finite_cloud_filter`` per 3D lidar the simulator
    stands in for: internal ``/<name>_lidar/raw`` in, the advertised point
    cloud endpoint out, on simulation time.
    """
    from launch_ros.actions import Node

    nodes = []
    for periph in peripherals:
        if periph.get("type") != "lidar" or str(periph.get("driver", "")) != "livox_mid360":
            continue
        name = periph["name"]
        nodes.append(
            Node(
                package="robot_runtime",
                executable="finite_cloud_filter",
                name=f"{name}_finite_cloud_filter",
                output="screen",
                # a stateless relay on the simulation-time stamps: the source
                # cloud's header is preserved verbatim, so the node itself
                # needs no clock
                parameters=[
                    {
                        "input_topic": f"/{name}_lidar/raw",
                        "output_topic": str(periph.get("pointcloud_topic", "/livox/lidar")),
                    }
                ],
            )
        )
    return nodes


def imu_bridge_entries(periph: dict) -> list[dict]:
    """ros_gz bridge entries for one IMU on its advertised endpoint (D5).

    The integrated IMU of a stand-in lidar driver (MID-360) and a standalone
    ``imu`` peripheral both land on the internal ``/<name>_imu/raw`` topic:
    the simulator's pre-settle samples (the zero window and the landing
    impulse) are held back and the settled stream is republished on the
    advertised endpoint. Other peripherals bridge nothing.
    """
    ptype = periph.get("type")
    if ptype == "lidar":
        if str(periph.get("driver", "")) not in LIDAR_INTEGRATED_IMU_DRIVERS:
            return []
    elif ptype != "imu":
        return []
    return [
        _sensor_bridge_entry(
            f"/{periph['name']}_imu/raw", f"/{periph['name']}_imu", "sensor_msgs/msg/Imu", "gz.msgs.IMU"
        )
    ]


def imu_settle_filter_nodes(peripherals: list) -> list:
    """The pre-settle IMU filter nodes for a peripheral list (D5).

    One ``robot_runtime/imu_settle_filter`` per simulated IMU: internal
    ``/<name>_imu/raw`` in, the advertised IMU endpoint out. The filter holds
    the stream back until gravity has been stable for a full run, then
    relays every sample verbatim.
    """
    from launch_ros.actions import Node

    nodes = []
    for periph in peripherals:
        ptype = periph.get("type")
        integrated = ptype == "lidar" and str(periph.get("driver", "")) in LIDAR_INTEGRATED_IMU_DRIVERS
        if not integrated and ptype != "imu":
            continue
        if not imu_bridge_entries(periph):
            continue
        name = periph["name"]
        endpoint = (
            str(periph.get("imu_topic", "/livox/imu"))
            if ptype == "lidar"
            else str(periph.get("imu_topic", "/imu/data"))
        )
        nodes.append(
            Node(
                package="robot_runtime",
                executable="imu_settle_filter",
                name=f"{name}_imu_settle_filter",
                output="screen",
                parameters=[
                    {
                        "input_topic": f"/{name}_imu/raw",
                        "output_topic": endpoint,
                    }
                ],
            )
        )
    return nodes


def camera_sensor_urdf(
    peripherals: list,
    *,
    platform: str = "gazebo",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    default_pose_loader=None,
    frame_contract: bool = False,
) -> str:
    """Camera link/joint/<gazebo><sensor> XML for a peripheral list.

    Naming (unchanged from the generic path): URDF link = the YAML
    ``frame_id``, gz sensor = ``{name}_camera``, gz color topic =
    ``/{name}_camera/image`` (``rgbd_camera`` uses the bare sensor name and gz
    appends the stream suffixes). ``frame_contract=True`` additionally adds the
    optical link/joint and stamps ``<gz_frame_id>`` with the advertised frame.
    """
    parts: list[str] = []
    for periph in peripherals:
        if periph.get("type") != "camera":
            continue
        if periph.get("disabled", False) or periph.get("enabled") is False:
            continue
        if (periph.get("simulation") or {}).get("embedded_sensor", False):
            continue
        if periph.get("skip_urdf_without_transform", False) and not periph.get("transform"):
            continue
        name = periph["name"]
        frame_id = periph.get("frame_id", f"camera_{name}_frame")
        transform, cam_fovy = resolve_camera_pose(periph, platform, default_pose_loader=default_pose_loader)
        parent = transform.get("parent_frame", "world")
        pose = {key: float(transform.get(key, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
        # The URDF "world" link is model-local (it sits at the spawn pose in
        # the simulator): re-anchor world-calibrated cameras by the inverse of
        # the full spawn transform.
        if parent == "world":
            pose = _rotate_yaw_inverse(pose, spawn_pose)

        width = periph.get("width", 640)
        height = periph.get("height", 480)
        fps = periph.get("fps", 30)
        sensor_name = f"{name}_camera"
        has_depth = camera_requests_depth(periph)
        sensor_type = "rgbd_camera" if has_depth else "camera"
        topic_name = sensor_name if has_depth else f"{sensor_name}/image"

        # fovy (vertical, degrees) → horizontal_fov (radians)
        fovy_rad = math.radians(cam_fovy)
        hfov_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * width / height)

        frame_id_element = (
            f"\n            <gz_frame_id>{escape(advertised_camera_frame(periph))}</gz_frame_id>"
            if frame_contract
            else ""
        )
        parts.append(
            f"""
    <!-- {name} camera (generated from YAML peripherals) -->
    <link name="{frame_id}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.01"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
    </link>
    <joint name="{name}_camera_joint" type="fixed">
        <parent link="{parent}"/>
        <child  link="{frame_id}"/>
        <origin xyz="{pose["x"]} {pose["y"]} {pose["z"]}" rpy="{pose["roll"]} {pose["pitch"]} {pose["yaw"]}"/>
    </joint>"""
        )
        if frame_contract:
            optical_frame = str(periph.get("optical_frame_id") or f"camera_{name}_optical_frame")
            parts.append(
                f"""
    <link name="{optical_frame}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.001"/>
            <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
        </inertial>
    </link>
    <joint name="{name}_optical_joint" type="fixed">
        <parent link="{frame_id}"/>
        <child  link="{optical_frame}"/>
        <origin xyz="0 0 0" rpy="{_OPTICAL_RPY[0]} {_OPTICAL_RPY[1]} {_OPTICAL_RPY[2]}"/>
    </joint>"""
            )
        parts.append(
            f"""
    <gazebo reference="{frame_id}">
        <sensor type="{sensor_type}" name="{sensor_name}">{frame_id_element}
            <update_rate>{fps}</update_rate>
            <camera>
                <horizontal_fov>{hfov_rad:.4f}</horizontal_fov>
                <image>
                    <width>{width}</width>
                    <height>{height}</height>
                    <format>R8G8B8</format>
                </image>
                <clip><near>0.01</near><far>100</far></clip>
            </camera>
            <always_on>true</always_on>
            <visualize>true</visualize>
            <topic>{topic_name}</topic>
        </sensor>
    </gazebo>"""
        )
    return "\n".join(parts)


def camera_bridge_entries(periph: dict, *, full_streams: bool = False) -> list[dict]:
    """ros_gz bridge config entries for one camera peripheral.

    The ROS contract topics are unchanged (``/camera/{name}/image_raw`` etc.);
    gz topics are absolute sensor topics (``/{name}_camera/...``). With
    ``full_streams`` the entries cover every stream the runtime descriptor
    advertises: color, color info, native depth, aligned depth and their
    camera infos (gz ``rgbd_camera`` serves native and aligned depth from the
    same ``depth_image`` topic).
    """
    name = periph["name"]
    sensor_name = f"{name}_camera"
    gz_base = f"/{sensor_name}"

    def entry(ros_topic: str, gz_topic: str, ros_type: str, gz_type: str) -> dict:
        return {
            "ros_topic_name": ros_topic,
            "gz_topic_name": gz_topic,
            "ros_type_name": ros_type,
            "gz_type_name": gz_type,
            "direction": "GZ_TO_ROS",
            "lazy": False,
        }

    entries = [
        entry(f"/camera/{name}/image_raw", f"{gz_base}/image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        entry(
            f"/camera/{name}/camera_info", f"{gz_base}/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"
        ),
    ]
    if not full_streams:
        # Legacy generic subset: aligned depth only, when any depth is requested.
        if camera_requests_depth(periph):
            entries.append(
                entry(
                    f"/camera/{name}/aligned_depth_to_color/image_raw",
                    f"{gz_base}/depth_image",
                    "sensor_msgs/msg/Image",
                    "gz.msgs.Image",
                )
            )
        return entries
    # Full contract: every stream the runtime descriptor advertises (the
    # descriptor's stream rules, not the looser depth-detection heuristic).
    # gz rgbd_camera serves native and aligned depth from the same
    # depth_image topic; it publishes no depth camera_info (the depth image
    # shares the color intrinsics, bridged as the color camera_info) and no
    # color-aligned pointcloud, which the descriptor drops on this transport.
    from robot_runtime.interface_description import camera_streams

    streams = camera_streams(periph)
    if "depth" not in streams:
        return entries
    entries.append(
        entry(
            f"/camera/{name}/depth/image_rect_raw", f"{gz_base}/depth_image", "sensor_msgs/msg/Image", "gz.msgs.Image"
        )
    )
    if periph.get("align_depth") and "color" in streams:
        entries.append(
            entry(
                f"/camera/{name}/aligned_depth_to_color/image_raw",
                f"{gz_base}/depth_image",
                "sensor_msgs/msg/Image",
                "gz.msgs.Image",
            )
        )
    return entries


def bridge_config_node(entries: list[dict], config_path: str | Path):
    """One ros_gz ``bridge_node`` reading a YAML config covering all entries."""
    from launch_ros.actions import Node

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(entries, stream, default_flow_style=False)
    return Node(
        package="ros_gz_bridge",
        executable="bridge_node",
        name="peripheral_bridge",
        parameters=[{"config_file": str(path)}],
        output="screen",
    )
