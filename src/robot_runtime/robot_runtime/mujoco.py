"""Robot-agnostic MuJoCo helpers shared by the runtime provider (D2/D4/D5/D6).

Mirrors ``robot_runtime.gazebo`` (openspec change ``so101-gazebo-runtime-
transport`` D6.1): the camera pose resolution, MJCF camera generation, URDF
camera TF, the simulator CameraPlugin parameters and the depth fan-out relays
are robot-agnostic algorithms living in the contract layer. Robot-specific
values (the MJCF template, meshes, SDF mesh names and the default camera
poses) come from the runtime profile.

Conventions:
- A MuJoCo camera looks along −Z with +Y up; the ROS optical frame looks
  along +Z with +Y down. The optical frame is the MJCF camera frame rotated
  by π about X, and that constant is applied in exactly one place
  (``_OPTICAL_FROM_MJCF_RPY``).
- User-saved simulation camera overrides (``~/.ros/ibrobot/
  sim_camera_overrides/<name>.yaml``) stay in the Gazebo convention; on this
  platform they are converted with ``gazebo_rpy_to_mujoco_rpy`` and keep
  their precedence over the profile defaults.
- World-parented cameras are placed in the MJCF worldbody at their calibrated
  world pose (the MJCF world is the true world; the robot base carries the
  spawn pose). Their URDF TF is anchored under the model-local ``world`` link
  using the inverse of the full spawn transform, exactly as on Gazebo.

Simulator baseline: ``mujoco_ros2_control`` 0.1.2 / ``mujoco_vendor`` 0.1.1
(libmujoco 3.12.0). Cameras are configured through the simulator's
``mujoco_ros2_control_plugins/CameraPlugin`` parameters, never through the
deprecated URDF ``<sensor>`` auto-registration; ``simulator_versions`` /
``check_simulator_versions`` keep the composed ``.mjb`` on exactly the
simulator's MuJoCo version.
"""

from __future__ import annotations

import math
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_runtime.gazebo import _rotate_yaw_inverse, load_camera_override, resolve_camera_pose

# The optical frame is the MJCF camera frame rotated by π about X (MuJoCo
# looks along −Z with +Y up; ROS optical looks along +Z with +Y down).
# Extrinsic XYZ rpy, same convention as URDF ``rpy`` and MJCF ``eulerseq="XYZ"``.
_OPTICAL_FROM_MJCF_RPY = (math.pi, 0.0, 0.0)

_IMAGE_TYPE = "sensor_msgs/msg/Image"
_CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"

#: Lidar drivers the MuJoCo platform stands in for (D5); other drivers get no
#: simulated sensor and are not advertised on the platform, exactly like a
#: driver Gazebo does not stand in for. Only the MID-360 in this stage: its
#: ray grid maps to ``mujoco.plugin.lidar`` and its integrated IMU to the
#: co-located MJCF sensor triple; 2D lidars have no MuJoCo counterpart yet.
MUJOCO_LIDAR_SIM_DRIVERS = ("livox_mid360",)


# ---------------------------------------------------------------------------
# Orientation conversion (moved from robot_config camera_presets, D5)
# ---------------------------------------------------------------------------

_R_FIX_GZ_TO_MJ = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


def gazebo_rpy_to_mujoco_rpy(gz_roll: float, gz_pitch: float, gz_yaw: float) -> tuple[float, float, float]:
    """Convert (roll, pitch, yaw) from the Gazebo camera convention to MuJoCo.

    Gazebo SDF camera link convention: optical = +X, image-up = +Z, image-right = -Y.
    MuJoCo camera body convention:     optical = -Z, image-up = +Y, image-right = +X.

    R_fix maps the MuJoCo camera basis into the Gazebo camera basis (columns =
    e_x^M, e_y^M, e_z^M expressed in Gazebo coords). With R_WG the calibrated
    Gazebo camera rotation in world, the equivalent MuJoCo camera rotation is
    R_WM = R_WG @ R_fix (det = +1, a proper rotation). Input and output are
    extrinsic XYZ Euler angles in radians, matching both URDF ``rpy`` and
    MuJoCo ``eulerseq="XYZ"``.
    """
    cr, sr = math.cos(gz_roll), math.sin(gz_roll)
    cp, sp = math.cos(gz_pitch), math.sin(gz_pitch)
    cy, sy = math.cos(gz_yaw), math.sin(gz_yaw)
    R_WG = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    R_WM = tuple(tuple(sum(R_WG[i][k] * _R_FIX_GZ_TO_MJ[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    sp_out = max(-1.0, min(1.0, -R_WM[2][0]))
    pitch = math.asin(sp_out)
    cp_out = math.cos(pitch)
    if abs(cp_out) > 1e-6:
        roll = math.atan2(R_WM[2][1], R_WM[2][2])
        yaw = math.atan2(R_WM[1][0], R_WM[0][0])
    else:
        # Gimbal lock (pitch = ±pi/2); roll and yaw are coupled. Convention:
        # set roll = 0, resolve yaw.
        roll = 0.0
        yaw = math.atan2(-R_WM[0][1], R_WM[1][1])
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Pose resolution (D5): override (converted) > simulation.mujoco > transform
# ---------------------------------------------------------------------------


def resolve_mujoco_camera_pose(periph: dict) -> tuple[dict, Any]:
    """Resolve a camera's effective (transform, fovy) in MuJoCo-native convention.

    Precedence: user-saved override (stored in the Gazebo convention, converted
    here) > the peripheral's ``simulation.mujoco.transform`` (+ ``fovy``) > the
    plain ``transform`` field. The shared resolver in ``robot_runtime.gazebo``
    handles the last two; the override conversion is MuJoCo-specific and lives
    here.
    """
    override = load_camera_override(str(periph.get("name", "camera")))
    if override is not None:
        roll, pitch, yaw = gazebo_rpy_to_mujoco_rpy(
            float(override.get("roll", 0.0)), float(override.get("pitch", 0.0)), float(override.get("yaw", 0.0))
        )
        transform = {
            "parent_frame": override.get("parent_frame", "base"),
            "x": float(override.get("x", 0.0)),
            "y": float(override.get("y", 0.0)),
            "z": float(override.get("z", 0.0)),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        return transform, override.get("fovy", 60)
    return resolve_camera_pose(periph, "mujoco")


def _enabled_cameras(peripherals: list) -> list[dict]:
    """Enabled camera peripherals (disabled cameras emit nothing, D5)."""
    return [
        periph
        for periph in peripherals or []
        if periph.get("type") == "camera" and not periph.get("disabled", False) and periph.get("enabled") is not False
    ]


# ---------------------------------------------------------------------------
# MJCF camera generation (D5)
# ---------------------------------------------------------------------------


def camera_mjcf_elements(peripherals: list) -> list[tuple[str | None, str]]:
    """MJCF ``<camera>`` elements for the enabled cameras, as (parent, xml).

    The parent is ``None`` for world-parented cameras (the element belongs in
    the ``worldbody`` at its calibrated world pose; the MJCF world is the true
    world, so no spawn compensation applies) or the MJCF body name for
    robot-mounted cameras (robot-relative, following the robot to its spawn
    pose). The pose is MuJoCo-native (extrinsic XYZ euler, matching the SO-101
    template's ``eulerseq="XYZ"``).
    """
    elements: list[tuple[str | None, str]] = []
    for periph in _enabled_cameras(peripherals):
        name = periph["name"]
        transform, fovy = resolve_mujoco_camera_pose(periph)
        parent = str(transform.get("parent_frame", "world"))
        pos = " ".join(f"{float(transform.get(key, 0.0)):.6f}" for key in ("x", "y", "z"))
        euler = " ".join(f"{float(transform.get(key, 0.0)):.6f}" for key in ("roll", "pitch", "yaw"))
        resolution = f"{int(periph.get('width', 640))} {int(periph.get('height', 480))}"
        xml = (
            f'<camera name="{name}_camera" pos="{pos}" euler="{euler}" '
            f'fovy="{float(fovy):.6f}" resolution="{resolution}"/>'
        )
        elements.append((None if parent in ("world", "worldbody") else parent, xml))
    return elements


# ---------------------------------------------------------------------------
# URDF camera TF (D5): the mount link and the optical frame, no gz tags
# ---------------------------------------------------------------------------


def camera_tf_urdf(peripherals: list, *, spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> str:
    """Camera link + optical-frame joints for TF only (no simulator sensor tags).

    The mount link (``frame_id``) sits at the MuJoCo-native camera pose; the
    optical frame (``optical_frame_id``, the advertised frame on this platform)
    is that camera rotated by π about X. World-parented cameras are re-anchored
    by the inverse of the full spawn transform because the URDF ``world`` link
    is model-local (it sits at the robot's spawn pose, not the true origin).
    """
    parts: list[str] = []
    for periph in _enabled_cameras(peripherals):
        name = periph["name"]
        frame_id = str(periph.get("frame_id", f"camera_{name}_frame"))
        optical_frame = str(periph.get("optical_frame_id", f"camera_{name}_optical_frame"))
        transform, _fovy = resolve_mujoco_camera_pose(periph)
        parent = str(transform.get("parent_frame", "world"))
        pose = {key: float(transform.get(key, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
        if parent in ("world", "worldbody"):
            parent = "world"
            pose = _rotate_yaw_inverse(pose, spawn_pose)
        parts.append(
            f"""
    <!-- {name} camera mount + optical frame (MuJoCo platform, TF only) -->
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
    </joint>
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
        <origin xyz="0 0 0" rpy="{_OPTICAL_FROM_MJCF_RPY[0]} {_OPTICAL_FROM_MJCF_RPY[1]} {_OPTICAL_FROM_MJCF_RPY[2]}"/>
    </joint>"""
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Simulator versions (D1/D2): the CameraPlugin release line and one MuJoCo
# ---------------------------------------------------------------------------

# mujoco_ros2_control 0.1.0 moved cameras into the CameraPlugin; the
# generated plugin parameters only work from that release on.
_CAMERA_PLUGIN_MIN_CONTROL_VERSION = "0.1.0"

_APT_UPGRADE_HINT = (
    "sudo apt install --only-upgrade ros-humble-mujoco-ros2-control "
    "ros-humble-mujoco-ros2-control-msgs ros-humble-mujoco-ros2-control-plugins ros-humble-mujoco-vendor"
)


@dataclass(frozen=True)
class SimulatorVersions:
    """The simulator release the composed ``.mjb`` must agree with (D2).

    ``control`` is the ``mujoco_ros2_control`` package version (the CameraPlugin
    release line starts at 0.1.0); ``mujoco`` is the ``X.Y.Z`` of the libmujoco
    that ``mujoco_vendor`` ships. Each field is ``None`` when its package or
    file cannot be found — an unknown version must never pass the check.
    """

    control: str | None = None
    mujoco: str | None = None


def _version_key(version: str) -> tuple[int, ...]:
    """The leading dotted-numeric parts of a version (``0.1.2-1`` → ``(0, 1, 2)``)."""
    key: list[int] = []
    for part in version.strip().split("."):
        leading = part[: len(part) - len(part.lstrip("0123456789"))]
        if not leading:
            break
        key.append(int(leading))
    return tuple(key)


def simulator_versions() -> SimulatorVersions:
    """Read the simulator release from the ament index.

    ``control`` comes from the ``<version>`` of the ``mujoco_ros2_control``
    ``package.xml``; ``mujoco`` from the ``libmujoco.so.X.Y.Z`` under
    ``<prefix>/opt/mujoco_vendor/lib``. Missing packages or files leave the
    field ``None`` instead of guessing.
    """
    control: str | None = None
    mujoco: str | None = None
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        package_xml = Path(get_package_share_directory("mujoco_ros2_control")) / "package.xml"
        version = (ET.parse(package_xml).getroot().findtext("version") or "").strip()
        control = version or None
    except (PackageNotFoundError, OSError, ET.ParseError):
        control = None
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_prefix

        lib_dir = Path(get_package_prefix("mujoco_vendor")) / "opt" / "mujoco_vendor" / "lib"
        candidates = sorted(
            (
                match.group(1)
                for path in lib_dir.glob("libmujoco.so.*")
                if (match := re.fullmatch(r"libmujoco\.so\.(\d+\.\d+\.\d+)", path.name))
            ),
            key=_version_key,
        )
        mujoco = candidates[-1] if candidates else None
    except (PackageNotFoundError, OSError):
        mujoco = None
    return SimulatorVersions(control=control, mujoco=mujoco)


def check_simulator_versions(versions: SimulatorVersions | None = None) -> None:
    """Fail fast unless the simulator can load the composed model (D2).

    Two invariants: the simulator is a CameraPlugin release
    (``mujoco_ros2_control`` >= 0.1.0), and the Python ``mujoco`` that
    compiles the ``.mjb`` has exactly the version of the simulator's
    libmujoco. Any violation raises ``RuntimeProfileError`` with an
    installation diagnostic before the model is composed; nothing is a
    silent no-op on the provider path.
    """
    from robot_runtime.launch_support import RuntimeProfileError

    if versions is None:
        versions = simulator_versions()
    if versions.control is None:
        raise RuntimeProfileError(
            "mujoco_ros2_control not found; install ros-humble-mujoco-ros2-control >= 0.1.2 "
            f"({_APT_UPGRADE_HINT} once installed)"
        )
    if _version_key(versions.control) < _version_key(_CAMERA_PLUGIN_MIN_CONTROL_VERSION):
        raise RuntimeProfileError(
            f"mujoco_ros2_control {versions.control} is older than the CameraPlugin release "
            f"({_CAMERA_PLUGIN_MIN_CONTROL_VERSION}); run the apt upgrade: {_APT_UPGRADE_HINT}"
        )
    if versions.mujoco is None:
        raise RuntimeProfileError(
            "mujoco_vendor libmujoco not found under <prefix>/opt/mujoco_vendor/lib; "
            "install ros-humble-mujoco-vendor 0.1.1 "
            f"({_APT_UPGRADE_HINT} once installed)"
        )
    try:
        import mujoco  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeProfileError(
            f"the MuJoCo embodiment needs the mujoco Python package ({exc}); "
            f"install mujoco=={versions.mujoco} (requirements/mujoco-sim.txt)"
        ) from exc
    if mujoco.__version__ != versions.mujoco:
        raise RuntimeProfileError(
            f"Python mujoco {mujoco.__version__} != mujoco_vendor libmujoco {versions.mujoco}; "
            "the composed .mjb would not load. "
            f"Install mujoco=={versions.mujoco} (requirements/mujoco-sim.txt); a venv set up with the "
            "LIBERO benchmark dependencies cannot run the MuJoCo simulation"
        )


# ---------------------------------------------------------------------------
# Cameras through the simulator CameraPlugin (D3, mujoco_ros2_control 0.1.2)
# ---------------------------------------------------------------------------

# One fixed CameraPlugin instance carries every enabled camera; the per-camera
# parameter key is the MJCF camera name (``camera_mjcf_elements``).
CAMERA_PLUGIN_INSTANCE = "ibrobot_cameras"
CAMERA_PLUGIN_TYPE = "mujoco_ros2_control_plugins/CameraPlugin"


def camera_depth_topics(periph: dict) -> tuple[str, bool]:
    """The (depth_topic, advertised) pair for one camera's plugin entry.

    The plugin always publishes one depth stream per camera, so the
    ``depth_topic`` param is always set: to the advertised native-depth
    endpoint when the camera advertises depth, otherwise to a runtime-private,
    never-advertised topic (D5).
    """
    from robot_runtime.interface_description import camera_streams

    name = periph["name"]
    if "depth" in camera_streams(periph):
        return f"/camera/{name}/depth/image_rect_raw", True
    return f"/_runtime/mujoco/{name}/depth", False


def camera_plugin_parameters(profile: dict, peripherals: list) -> dict:
    """The nested ``mujoco_plugins`` CameraPlugin parameter tree (D3).

    One plugin instance carries every enabled camera. The per-camera key is
    the MJCF camera name (``<name>_camera``, what ``camera_mjcf_elements``
    writes), and the four values are exactly the ones the 0.0.3 ``<sensor>``
    blocks carried: the frame is the shared advertised frame (5b.3:
    ``advertised_camera_frame`` — opencv → ``frame_id``, realsense →
    optical), the image/info topics are the advertised color endpoints, and
    the depth topic follows ``camera_depth_topics``. The single publish rate
    is ``mujoco_camera_publish_rate``. An empty camera list returns ``{}``
    (no plugin is configured, and with ``auto_register_cameras=false`` no
    deprecated auto-registration fills in).
    """
    cameras = _enabled_cameras(peripherals)
    if not cameras:
        return {}
    from robot_runtime.gazebo import advertised_camera_frame

    instance: dict[str, Any] = {
        "type": CAMERA_PLUGIN_TYPE,
        "camera_publish_rate": float(mujoco_camera_publish_rate({**profile, "peripherals": peripherals})),
    }
    for periph in cameras:
        name = periph["name"]
        depth_topic, _advertised = camera_depth_topics(periph)
        instance[f"{name}_camera"] = {
            "policy": "streaming",
            "frame_name": advertised_camera_frame(periph),
            "image_topic": f"/camera/{name}/image_raw",
            "info_topic": f"/camera/{name}/camera_info",
            "depth_topic": depth_topic,
        }
    return {"mujoco_plugins": {CAMERA_PLUGIN_INSTANCE: instance}}


# ---------------------------------------------------------------------------
# Scene object observation (issue #195): FreeJointStatePublisherPlugin
# ---------------------------------------------------------------------------

# One plugin instance publishes every dynamic scene body; the scene task's
# MuJoCo backend subscribes to the same topic through
# sim_models.scene_compiler.free_joint_states_topic.
FREE_JOINT_STATE_PLUGIN_INSTANCE = "ibrobot_object_states"
FREE_JOINT_STATE_PLUGIN_TYPE = "mujoco_ros2_control_plugins/FreeJointStatePublisherPlugin"


def free_joint_state_plugin_parameters(scene: str) -> dict:
    """The nested ``mujoco_plugins`` FreeJointStatePublisherPlugin tree (issue #195).

    Scenes with dynamic objects (``type: dynamic`` in their layout.yaml) get
    one plugin instance publishing every dynamic body's pose at 50 Hz on the
    scene-namespaced absolute topic ``/sim/<scene>/free_joint_states`` — the
    MuJoCo counterpart of Gazebo's ``/sim/<scene>/object_poses``. The topic
    name resolves through ``sim_models.scene_compiler.free_joint_states_topic``
    so the provider and the scene task's backend can never drift apart. An
    empty scene, a robot-only launch (``scene=""``/None) or a scene without
    dynamic objects returns ``{}`` — no plugin is configured.
    """
    if not scene:
        return {}
    from sim_models.scene_compiler import free_joint_states_topic, scene_dynamic_objects

    body_names = scene_dynamic_objects(scene)
    if not body_names:
        return {}
    return {
        "mujoco_plugins": {
            FREE_JOINT_STATE_PLUGIN_INSTANCE: {
                "type": FREE_JOINT_STATE_PLUGIN_TYPE,
                "body_names": body_names,
                "frame_id": "",
                "topic": free_joint_states_topic(scene),
                "publish_rate": 50.0,
            }
        }
    }


def merge_mujoco_plugin_parameters(*param_trees: dict) -> dict:
    """Merge ``mujoco_plugins`` parameter trees into one (later trees win per key).

    Every generator (cameras, scene object states) returns a tree keyed
    ``{"mujoco_plugins": {<instance>: {...}}}``; the simulator node takes a
    single params file, so the trees merge instance-wise before being written.
    """
    instances: dict[str, Any] = {}
    for tree in param_trees:
        for instance, params in (tree.get("mujoco_plugins") or {}).items():
            instances[instance] = {**instances.get(instance, {}), **params}
    return {"mujoco_plugins": instances} if instances else {}


# ---------------------------------------------------------------------------
# The single camera rate (D5): the plugin has one rate for all cameras
# ---------------------------------------------------------------------------


def mujoco_camera_publish_rate(profile: dict) -> float:
    """The effective MuJoCo camera rate: ``simulation.mujoco.camera_publish_rate``,
    defaulting to the minimum enabled camera fps (the descriptor advertises
    exactly this rate on ``mujoco``)."""
    configured = ((profile.get("simulation") or {}).get("mujoco") or {}).get("camera_publish_rate")
    if configured is not None:
        return float(configured)
    fps = [float(p.get("fps", 30)) for p in _enabled_cameras(profile.get("peripherals") or [])]
    return min(fps) if fps else 5.0


# ---------------------------------------------------------------------------
# Simulated lidar and IMU on MuJoCo (lekiwi-mujoco-runtime-platform D5)
# ---------------------------------------------------------------------------

# One Mujoco3dLidarPlugin instance carries every simulated lidar; the
# per-sensor parameter key is the MJCF sensor name (``lidar_imu_mjcf``).
LIDAR_PLUGIN_INSTANCE = "ibrobot_lidars"
LIDAR_PLUGIN_TYPE = "mujoco_ros2_control_plugins/Mujoco3dLidarPlugin"

# One BaseVelocityPlugin instance moves the floating base from the twist the
# base's own wheel odometry reports (D3); the command topic is
# simulator-internal, fed by the odometry-twist relay.
BASE_VELOCITY_PLUGIN_INSTANCE = "ibrobot_floating_base"
BASE_VELOCITY_PLUGIN_TYPE = "mujoco_ros2_control_plugins/BaseVelocityPlugin"
FLOATING_BASE_CMD_TOPIC = "/simulator/floating_base/cmd_vel"

# Floating-base odometry constants (D9): the free joint MujocoSystemInterface
# reports, on its simulator-namespaced topic. Never advertised, never consumed
# by a runtime node — the ground-truth stream subscribes to it for tests.
ODOM_FREE_JOINT_NAME = "floating_base_joint"
FLOATING_BASE_STATE_TOPIC = "/simulator/floating_base_state"

# BaseVelocityPlugin command timeout (D3.4): on the simulation clock, the
# same order as base_node's wheel-command cadence — a relay that dies leaves
# the base stopped within a quarter second.
BASE_VELOCITY_CMD_TIMEOUT_S = 0.25


def _simulated_lidars(peripherals: list) -> list[dict]:
    """Lidar peripherals the MuJoCo platform stands in for (D5)."""
    return [
        periph
        for periph in peripherals or []
        if periph.get("type") == "lidar"
        and str(periph.get("driver", "")) in MUJOCO_LIDAR_SIM_DRIVERS
        and not periph.get("disabled", False)
        and periph.get("enabled") is not False
    ]


def lidar_plugin_parameters(peripherals: list) -> dict:
    """The nested ``mujoco_plugins`` Mujoco3dLidarPlugin tree (D5.3).

    One plugin instance carries every simulated lidar. The per-sensor key is
    the MJCF sensor name (``<name>_lidar``, what ``lidar_imu_mjcf`` writes);
    the frame is the shared advertised frame id and the topic the internal
    raw stream the finite-cloud filter consumes. An empty list returns
    ``{}`` — no plugin is configured.
    """
    lidars = _simulated_lidars(peripherals)
    if not lidars:
        return {}
    instance: dict[str, Any] = {"type": LIDAR_PLUGIN_TYPE}
    for periph in lidars:
        name = periph["name"]
        instance[f"{name}_lidar"] = {
            "frame_name": str(periph.get("frame_id", f"{name}_frame")),
            "topic": f"/{name}_lidar/raw",
        }
    return {"mujoco_plugins": {LIDAR_PLUGIN_INSTANCE: instance}}


def base_velocity_plugin_parameters(floating_base: dict | None) -> dict:
    """The nested ``mujoco_plugins`` BaseVelocityPlugin tree (D3/D5.3).

    ``floating_base`` is the profile's ``simulation.mujoco.floating_base``
    section: ``body`` names the MJCF body with the free joint the plugin
    drives and ``twist_source_topic`` the base's own wheel odometry the relay
    forwards. The plugin takes a ``TwistStamped`` on the simulator-internal
    command topic; a command older than ``cmd_timeout`` (simulation clock)
    reads as zero. No ``max_linear_velocity`` / ``max_yaw_rate`` keys are
    set: upstream defaults are +inf, so the base executes exactly the twist
    ``/odom`` reports (D3.2 — the one authoritative clamp lives in
    ``base_node``).
    """
    section = floating_base or {}
    body = str(section.get("body", "")).strip()
    if not body:
        return {}
    return {
        "mujoco_plugins": {
            BASE_VELOCITY_PLUGIN_INSTANCE: {
                "type": BASE_VELOCITY_PLUGIN_TYPE,
                "body": body,
                "cmd_vel_topic": FLOATING_BASE_CMD_TOPIC,
                "use_stamped_twist": True,
                "cmd_timeout": BASE_VELOCITY_CMD_TIMEOUT_S,
            }
        }
    }


def _euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Extrinsic XYZ euler (URDF rpy == MJCF eulerseq="XYZ") to w-x-y-z."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def lidar_imu_mjcf(peripherals: list) -> tuple[list[tuple[str, str]], list[str], bool]:
    """The MJCF fragments simulating the lidars and their integrated IMUs (D5.2).

    Returns ``(site_elements, sensor_elements, needs_lidar_extension)``:
    ``site_elements`` are ``(parent_body, xml)`` pairs — a ``<site>`` at the
    resolved mount pose in the transform's parent body, positioned so the
    base's own geoms stay out of the ray fan where the mount allows it;
    ``sensor_elements`` are top-level ``<sensor>`` children — the
    ``mujoco.plugin.lidar`` plugin sensor on the site plus the co-located
    ``framequat``/``gyro``/``accelerometer`` triple of the integrated IMU
    (zero extrinsic, as on Gazebo). ``needs_lidar_extension`` says the model
    must declare the engine extension (exactly once; a profile without a
    simulated lidar gets no site, sensor or extension — the SO-101 render
    stays byte-identical).
    """
    from robot_runtime.gazebo import _LIDAR_DEFAULTS, LIDAR_INTEGRATED_IMU_DRIVERS, _sensor_param
    from robot_runtime.profile import resolve_sensor_pose

    sites: list[tuple[str, str]] = []
    sensors: list[str] = []
    needs_extension = False
    for periph in _simulated_lidars(peripherals):
        name = periph["name"]
        transform = resolve_sensor_pose(periph, "mujoco")
        parent = str(transform.get("parent_frame", "base_link"))
        x, y, z, roll, pitch, yaw = (float(transform.get(key, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw"))
        quat = _euler_xyz_to_quat(roll, pitch, yaw)
        pos = f"{x:.6f} {y:.6f} {z:.6f}"
        quat_text = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
        sites.append((parent, f'<site name="{name}_lidar_site" pos="{pos}" quat="{quat_text}" size="0.005"/>'))
        horizontal = int(_sensor_param(periph, "mujoco", "horizontal_samples", _LIDAR_DEFAULTS["horizontal_samples"]))
        vertical = int(_sensor_param(periph, "mujoco", "vertical_samples", _LIDAR_DEFAULTS["vertical_samples"]))
        if vertical <= 1:
            # a planar scan: one elevation sample (the plugin requires a
            # bounded range even for a single row)
            vertical, elevation = 1, "0 0"
        else:
            vertical_min = math.radians(
                float(_sensor_param(periph, "mujoco", "vertical_min_deg", _LIDAR_DEFAULTS["vertical_min_deg"]))
            )
            vertical_max = math.radians(
                float(_sensor_param(periph, "mujoco", "vertical_max_deg", _LIDAR_DEFAULTS["vertical_max_deg"]))
            )
            elevation = f"{vertical_min:.6f} {vertical_max:.6f}"
        range_min = float(_sensor_param(periph, "mujoco", "range_min", _LIDAR_DEFAULTS["range_min"]))
        range_max = float(_sensor_param(periph, "mujoco", "range_max", _LIDAR_DEFAULTS["range_max"]))
        update_rate = float(_sensor_param(periph, "mujoco", "update_rate", periph.get("publish_freq", 10.0)))
        sensors.append(
            f'<plugin name="{name}_lidar" plugin="mujoco.plugin.lidar" objtype="site" objname="{name}_lidar_site">\n'
            f'  <config key="resolution" value="{horizontal} {vertical}"/>\n'
            '  <config key="azimuth_range" value="-3.14159265 3.14159265"/>\n'
            f'  <config key="elevation_range" value="{elevation}"/>\n'
            f'  <config key="min_range" value="{range_min:g}"/>\n'
            f'  <config key="max_range" value="{range_max:g}"/>\n'
            f'  <config key="update_rate" value="{update_rate:g}"/>\n'
            '  <config key="async" value="0"/>\n'
            "</plugin>"
        )
        needs_extension = True
        if str(periph.get("driver", "")) in LIDAR_INTEGRATED_IMU_DRIVERS:
            sensors.extend(
                [
                    f'<framequat name="{name}_imu_quat" objtype="site" objname="{name}_lidar_site"/>',
                    f'<gyro name="{name}_imu_gyro" site="{name}_lidar_site"/>',
                    f'<accelerometer name="{name}_imu_accel" site="{name}_lidar_site"/>',
                ]
            )
    return sites, sensors, needs_extension


def imu_sensor_urdf(peripherals: list) -> str:
    """The ros2_control ``<sensor>`` blocks for the integrated IMUs (D5.3).

    One ``<sensor>`` per simulated lidar whose driver carries an integrated
    IMU: ``mujoco_type imu`` mapping to the MJCF sensor triple
    ``<name>_imu_quat/_gyro/_accel`` (the upstream default suffixes). The
    ``imu_sensor_broadcaster`` spawned against it publishes the internal raw
    stream the settle filter consumes. Rendered into the URDF's
    ``<ros2_control>`` block by ``render_mujoco_description``.
    """
    from robot_runtime.gazebo import LIDAR_INTEGRATED_IMU_DRIVERS

    parts: list[str] = []
    for periph in _simulated_lidars(peripherals):
        if str(periph.get("driver", "")) not in LIDAR_INTEGRATED_IMU_DRIVERS:
            continue
        name = periph["name"]
        state_interfaces = (
            ("orientation", ("x", "y", "z", "w")),
            ("angular_velocity", ("x", "y", "z")),
            ("linear_acceleration", ("x", "y", "z")),
        )
        interfaces = "\n".join(
            f'      <state_interface name="{group}.{axis}"/>' for group, axes in state_interfaces for axis in axes
        )
        parts.append(
            f"<!-- {name} integrated IMU (generated from the peripheral declaration) -->\n"
            f'    <sensor name="{name}_imu">\n'
            '      <param name="mujoco_type">imu</param>\n'
            f'      <param name="mujoco_sensor_name">{name}_imu</param>\n'
            f"{interfaces}\n"
            "    </sensor>"
        )
    return "\n".join(parts)


def mujoco_lidar_filter_nodes(peripherals: list) -> list:
    """The finite-point filter nodes for the MuJoCo lidars (D5.3).

    One ``robot_runtime/finite_cloud_filter`` per simulated lidar: the
    plugin's raw cloud (``/<name>_lidar/raw``, bare ``x y z`` float32 with
    NaN misses) in, the advertised point cloud endpoint out, with the zero
    ``intensity`` field appended for FAST-LIO's generic handler.
    """
    from launch_ros.actions import Node

    nodes = []
    for periph in _simulated_lidars(peripherals):
        name = periph["name"]
        nodes.append(
            Node(
                package="robot_runtime",
                executable="finite_cloud_filter",
                name=f"{name}_finite_cloud_filter",
                output="screen",
                parameters=[
                    {
                        "input_topic": f"/{name}_lidar/raw",
                        "output_topic": str(periph.get("pointcloud_topic", "/livox/lidar")),
                        "add_intensity": True,
                    }
                ],
            )
        )
    return nodes


def mujoco_imu_filter_nodes(peripherals: list) -> list:
    """The pre-settle IMU filter nodes for the MuJoCo integrated IMUs (D5.3).

    One ``robot_runtime/imu_settle_filter`` per simulated integrated IMU: the
    broadcaster's raw stream (``/<name>_imu/raw`` — the ``~/imu`` publisher
    remapped process-wide) in, the advertised IMU endpoint out, holding the
    pre-settle samples back until gravity has been stable.
    """
    from launch_ros.actions import Node

    from robot_runtime.gazebo import LIDAR_INTEGRATED_IMU_DRIVERS

    nodes = []
    for periph in _simulated_lidars(peripherals):
        if str(periph.get("driver", "")) not in LIDAR_INTEGRATED_IMU_DRIVERS:
            continue
        name = periph["name"]
        nodes.append(
            Node(
                package="robot_runtime",
                executable="imu_settle_filter",
                name=f"{name}_imu_settle_filter",
                output="screen",
                parameters=[
                    {
                        "input_topic": f"/{name}_imu/raw",
                        "output_topic": str(periph.get("imu_topic", "/livox/imu")),
                    }
                ],
            )
        )
    return nodes


def odometry_twist_relay_node(floating_base: dict | None) -> list:
    """The odometry-twist relay node for a floating base (D3.3).

    Subscribes the base's own wheel odometry (``twist_source_topic``) and
    forwards the body twist it reports as a ``TwistStamped`` on the
    simulator-internal command topic the ``BaseVelocityPlugin`` consumes. The
    base frame is the floating-base body — the odometry's ``child_frame_id``.
    A profile without a ``simulation.mujoco.floating_base`` section gets no
    relay (no free joint to drive).
    """
    from launch_ros.actions import Node

    section = floating_base or {}
    body = str(section.get("body", "")).strip()
    if not body:
        return []
    return [
        Node(
            package="robot_runtime",
            executable="odometry_twist_relay",
            name="odometry_twist_relay",
            output="screen",
            parameters=[
                {
                    "twist_source_topic": str(section.get("twist_source_topic", "/odom")),
                    "cmd_vel_topic": FLOATING_BASE_CMD_TOPIC,
                    "base_frame": body,
                    "use_sim_time": True,
                }
            ],
        )
    ]


def imu_broadcaster_entries(peripherals: list) -> dict:
    """The generated IMU broadcaster controller entries (D5.3).

    A mapping of controller name to its ``ros__parameters`` for every
    simulated lidar with an integrated IMU; ``merged_controllers_config``
    writes them next to the shipped controllers configuration (the shipped
    file is never edited).
    """
    from robot_runtime.gazebo import LIDAR_INTEGRATED_IMU_DRIVERS

    entries: dict[str, Any] = {}
    for periph in _simulated_lidars(peripherals):
        if str(periph.get("driver", "")) not in LIDAR_INTEGRATED_IMU_DRIVERS:
            continue
        name = periph["name"]
        entries[f"{name}_imu_broadcaster"] = {
            "sensor_name": f"{name}_imu",
            "frame_id": str(periph.get("frame_id", f"{name}_frame")),
        }
    return entries


def imu_broadcaster_remappings(peripherals: list) -> list[tuple[str, str]]:
    """The process-wide remaps putting the IMU broadcasters on the raw topics.

    The ``imu_sensor_broadcaster`` publishes on ``~/imu`` (its node name is
    the controller's); a process-level remap rule reaches the controller node
    created inside the controller manager, so the raw stream lands on
    ``/<name>_imu/raw`` — the internal topic the settle filter consumes
    (verified against mujoco_ros2_control 0.1.2).
    """
    from robot_runtime.gazebo import LIDAR_INTEGRATED_IMU_DRIVERS

    remaps: list[tuple[str, str]] = []
    for periph in _simulated_lidars(peripherals):
        if str(periph.get("driver", "")) not in LIDAR_INTEGRATED_IMU_DRIVERS:
            continue
        name = periph["name"]
        remaps.append((f"/{name}_imu_broadcaster/imu", f"/{name}_imu/raw"))
    return remaps


# ---------------------------------------------------------------------------
# Robot MJCF rendering (D2, D4)
# ---------------------------------------------------------------------------


def render_robot_mjcf(
    profile: dict[str, Any],
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    peripherals: list | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """Render the runtime-private robot MJCF with the cameras injected (D2/D4).

    The template and meshes come from the profile's ``simulation.mujoco.template``
    / ``meshes_dir`` (``$(find …)`` resolved by ``path_subst``); the base
    placeholder takes the spawn position and the yaw quaternion
    (``{{ROBOT_BASE_POS}}`` / ``{{ROBOT_BASE_QUAT}}``, w-x-y-z); the MJCF
    ``<camera>`` elements come from the resolved camera poses (D5) and the
    lidar site / plugin sensor / integrated-IMU triple from the effective
    inventory (D5.2 — a profile without a simulated lidar renders no site,
    sensor or engine-extension element). The file is written to a per-launch
    temporary directory, never to a fixed path.
    """
    from robot_runtime.path_subst import resolve_path

    mujoco_cfg = (profile.get("simulation") or {}).get("mujoco") or {}
    template = mujoco_cfg.get("template")
    meshes_dir = mujoco_cfg.get("meshes_dir")
    if not template or not meshes_dir:
        raise RuntimeError(
            "runtime profile missing simulation.mujoco.template / simulation.mujoco.meshes_dir "
            "(the MuJoCo embodiment needs the robot MJCF template and mesh directory)"
        )
    template_path = Path(resolve_path(str(template)))
    if not template_path.is_file():
        raise RuntimeError(f"simulation.mujoco.template not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")
    content = content.replace("{{MESHES_DIR}}", str(Path(resolve_path(str(meshes_dir)))))
    x, y, z, yaw = (float(value) for value in spawn_pose)
    content = content.replace("{{ROBOT_BASE_POS}}", f"{x:.6f} {y:.6f} {z:.6f}")
    # MuJoCo free/base orientation is w-x-y-z; a pure-yaw spawn quaternion.
    content = content.replace("{{ROBOT_BASE_QUAT}}", f"{math.cos(yaw / 2):.6f} 0.0 0.0 {math.sin(yaw / 2):.6f}")

    effective = peripherals if peripherals is not None else profile.get("peripherals") or []
    root = ET.fromstring(content)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"MuJoCo template has no <worldbody>: {template_path}")
    for parent, camera_xml in camera_mjcf_elements(effective):
        element = ET.fromstring(camera_xml)
        if parent is None:
            worldbody.append(element)
            continue
        body = worldbody.find(f'.//body[@name="{parent}"]')
        if body is None:
            raise RuntimeError(f"camera parent body {parent!r} not found in the MuJoCo template {template_path}")
        body.append(element)

    # D5.2: the simulated lidar site in its mount body, the plugin sensor and
    # the integrated-IMU triple under <sensor>, and the engine extension the
    # plugin sensor needs (exactly once).
    sites, sensors, needs_extension = lidar_imu_mjcf(effective)
    for parent, site_xml in sites:
        body = worldbody.find(f'.//body[@name="{parent}"]')
        if body is None:
            raise RuntimeError(f"lidar parent body {parent!r} not found in the MuJoCo template {template_path}")
        body.append(ET.fromstring(site_xml))
    if sensors:
        sensor_section = root.find("sensor")
        if sensor_section is None:
            sensor_section = ET.SubElement(root, "sensor")
        for sensor_xml in sensors:
            sensor_section.append(ET.fromstring(sensor_xml))
    if needs_extension:
        extension = root.find("extension")
        if extension is None:
            extension = ET.SubElement(root, "extension")
        if extension.find('plugin[@plugin="mujoco.plugin.lidar"]') is None:
            ET.SubElement(extension, "plugin", {"plugin": "mujoco.plugin.lidar"})

    out_path = Path(out_dir or tempfile.mkdtemp(prefix="ibrobot_mujoco_")) / "robot.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out_path, encoding="unicode", xml_declaration=True)
    return out_path


# ---------------------------------------------------------------------------
# Depth fan-out relays (D5 depth branch, spike T0 keeps depth)
# ---------------------------------------------------------------------------


def mujoco_camera_relays(peripherals: list) -> list:
    """The per-camera relays fanning the plugin's single depth/info pair out.

    The plugin has one depth publisher and one CameraInfo publisher per
    camera: the depth image already lands on the advertised native-depth
    endpoint, and the CameraInfo on the advertised color endpoint. The relays
    copy the depth image to the aligned-depth endpoint and the CameraInfo to
    every advertised depth and aligned-depth info endpoint. The frame id is
    already the advertised optical frame (``frame_name``), so no rewrite.
    """
    from launch_ros.actions import Node

    from robot_runtime.interface_description import camera_streams

    nodes = []
    for periph in _enabled_cameras(peripherals):
        name = periph["name"]
        if "depth" not in camera_streams(periph):
            continue
        aligned = bool(periph.get("align_depth")) and "color" in camera_streams(periph)
        relay_specs = []
        if aligned:
            relay_specs.append(
                (
                    f"/camera/{name}/depth/image_rect_raw",
                    [f"/camera/{name}/aligned_depth_to_color/image_raw"],
                    f"{name}_aligned_depth_relay",
                    _IMAGE_TYPE,
                )
            )
        info_targets = [f"/camera/{name}/depth/camera_info"]
        if aligned:
            info_targets.append(f"/camera/{name}/aligned_depth_to_color/camera_info")
        relay_specs.append(
            (
                f"/camera/{name}/camera_info",
                info_targets,
                f"{name}_depth_info_relay",
                _CAMERA_INFO_TYPE,
            )
        )
        for source, targets, relay_name, message_type in relay_specs:
            for index, target in enumerate(targets):
                nodes.append(
                    Node(
                        package="robot_runtime",
                        executable="topic_relay",
                        name=f"{relay_name}{'' if len(targets) == 1 else f'_{index}'}",
                        # the plugin's publishers are reliable depth-1: large
                        # images suffer best-effort fragment loss without
                        # repair, so the relay subscribes reliable
                        arguments=[source, target, message_type, "--reliable"],
                        parameters=[{"use_sim_time": True}],
                        output="screen",
                    )
                )
    return nodes
