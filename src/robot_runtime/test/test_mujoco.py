"""MuJoCo platform tests (so101-mujoco-runtime-platform tasks 3.1-3.4,
upgrade-mujoco-ros2-control-0-1 D2/D3/D6).

Covers the robot-agnostic MuJoCo helpers (pose resolution with override
conversion, MJCF camera generation, camera TF, the simulator version check,
the CameraPlugin parameter tree, the depth fan-out relays), the
launch-structure contract of the MuJoCo embodiment and the SO-101 profile's
runtime-owned MuJoCo camera defaults (golden values).
"""

from __future__ import annotations

import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from launch_ros.actions import Node

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import (
    MUJOCO_SIMULATION_NODE,
    mujoco_embodiment_actions,
    render_mujoco_description,
    runtime_stack_actions,
)
from robot_runtime.mujoco import (
    CAMERA_PLUGIN_INSTANCE,
    CAMERA_PLUGIN_TYPE,
    FREE_JOINT_STATE_PLUGIN_INSTANCE,
    FREE_JOINT_STATE_PLUGIN_TYPE,
    SimulatorVersions,
    camera_mjcf_elements,
    camera_plugin_parameters,
    camera_tf_urdf,
    check_simulator_versions,
    free_joint_state_plugin_parameters,
    gazebo_rpy_to_mujoco_rpy,
    merge_mujoco_plugin_parameters,
    mujoco_camera_publish_rate,
    mujoco_camera_relays,
    render_robot_mjcf,
    resolve_mujoco_camera_pose,
    simulator_versions,
)

mujoco = pytest.importorskip("mujoco")

# ---------------------------------------------------------------------------
# Shared fixtures: a minimal robot-agnostic profile the MuJoCo helpers accept
# ---------------------------------------------------------------------------

MJCF_TEMPLATE = """<mujoco model="probe">
  <compiler eulerseq="XYZ" angle="radian"/>
  <worldbody>
    <body name="base" pos="{{ROBOT_BASE_POS}}" quat="{{ROBOT_BASE_QUAT}}">
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="gripper" pos="0 0 0">
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

XACRO = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:arg name="use_sim" default="true"/>
  <xacro:arg name="simulated" default="false"/>
  <xacro:arg name="sim_plugin" default="gz_ros2_control/GazeboSimSystem"/>
  <xacro:arg name="sim_effort_state" default="false"/>
  <xacro:arg name="mujoco_model" default=""/>
  <xacro:arg name="mujoco_headless" default="false"/>
  <link name="world"/>
  <link name="base"/>
  <link name="gripper"/>
  <joint name="world_base" type="fixed">
    <parent link="world"/><child link="base"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="base_gripper" type="fixed">
    <parent link="base"/><child link="gripper"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <xacro:if value="$(arg use_sim)">
    <ros2_control name="RobotSystem" type="system">
      <hardware>
        <plugin>$(arg sim_plugin)</plugin>
        <xacro:if value="${'$(arg mujoco_model)' != ''}">
          <param name="mujoco_model">$(arg mujoco_model)</param>
          <param name="auto_register_cameras">false</param>
        </xacro:if>
        <xacro:if value="$(arg mujoco_headless)">
          <param name="headless">true</param>
        </xacro:if>
      </hardware>
      <joint name="1">
        <command_interface name="position"/>
        <state_interface name="position"/>
        <state_interface name="velocity"/>
        <xacro:if value="$(arg sim_effort_state)">
          <state_interface name="effort"/>
        </xacro:if>
      </joint>
    </ros2_control>
  </xacro:if>
</robot>
"""

# SO-101 golden values: the old camera_presets.py "mujoco" column, now
# runtime-owned in the SO-101 profile (task 3.4).
TOP = {
    "type": "camera",
    "name": "top",
    "driver": "opencv",
    "index": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "pixel_format": "mjpeg2rgb",
    "frame_id": "camera_top_frame",
    "optical_frame_id": "camera_top_optical_frame",
    "simulation": {
        "mujoco": {
            "transform": {
                "parent_frame": "world",
                "x": 0.0419,
                "y": 0.1123,
                "z": 0.6136,
                "roll": 0.052274,
                "pitch": 0.003013,
                "yaw": 3.136436,
            },
            "fovy": 47,
        }
    },
    "transform": {"parent_frame": "base", "x": 0.0, "y": 0.0, "z": 0.0},
}
WRIST = {
    "type": "camera",
    "name": "wrist",
    "driver": "opencv",
    "index": 2,
    "width": 640,
    "height": 480,
    "fps": 60,
    "pixel_format": "mjpeg2rgb",
    "frame_id": "camera_wrist_frame",
    "optical_frame_id": "camera_wrist_optical_frame",
    "simulation": {
        "mujoco": {
            "transform": {
                "parent_frame": "gripper",
                "x": 0.002265,
                "y": 0.073023,
                "z": -0.016289,
                "roll": 0.031197,
                "pitch": -0.490648,
                "yaw": -1.607245,
            },
            "fovy": 46,
        }
    },
    "transform": {"parent_frame": "gripper", "x": 0.0, "y": 0.0, "z": 0.0},
}
FRONT = {
    "type": "camera",
    "name": "front",
    "driver": "realsense",
    "serial_number": "",
    "width": 640,
    "height": 480,
    "fps": 30,
    "pixel_format": "bgr8",
    "enable_pointcloud": False,
    "align_depth": True,
    "frame_id": "camera_front_link",
    "optical_frame_id": "camera_front_optical_frame",
    "simulation": {
        "mujoco": {
            "transform": {
                "parent_frame": "base",
                "x": 0.0,
                "y": -0.28,
                "z": 0.55,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 3.1415,
            },
            "fovy": 70,
        }
    },
    "transform": {"parent_frame": "base", "x": 0.0, "y": 0.0, "z": 0.0},
}


def _profile(tmp_path: Path, peripherals=None, with_joint_current=True):
    mjcf = tmp_path / "robot.xml.template"
    mjcf.write_text(MJCF_TEMPLATE, encoding="utf-8")
    meshes = tmp_path / "meshes"
    meshes.mkdir(exist_ok=True)
    xacro = tmp_path / "robot.xacro"
    xacro.write_text(XACRO, encoding="utf-8")
    controllers = tmp_path / "controllers.yaml"
    controllers.write_text("controller_manager:\n  ros__parameters:\n    update_rate: 100\n", encoding="utf-8")
    profile = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "modes": {
            "initial": "idle",
            "idle": {"controllers": [], "transitions": ["trajectory"]},
            "trajectory": {
                "controllers": ["arm_trajectory_controller"],
                "allows_trajectory": True,
                "transitions": ["idle"],
            },
        },
        "capabilities": {"joint.state": {}, "joint.trajectory": {}, "runtime.stop": {}},
        "joints": ["1"],
        "simulated": False,
        "sim_platforms": ["sdk", "gazebo", "mujoco"],
        "sim_platform": "sdk",
        "controller_manager": "controller_manager",
        "joint_state_topic": "/joint_states",
        "command_channels": [],
        "trajectory_actions": ["/arm_trajectory_controller/follow_joint_trajectory"],
        "description": {"xacro": str(xacro)},
        "ros2_control": {"controllers_config": str(controllers)},
        "peripherals": peripherals if peripherals is not None else [TOP, WRIST, FRONT],
        "hardware": {},
        "motion": {
            "arm_group_name": "arm",
            "base_link": "base",
            "ee_link": "gripper",
            "shoulder_link": "shoulder",
        },
        "simulation": {
            "mujoco": {
                "template": str(mjcf),
                "meshes_dir": str(meshes),
                "sdf_meshes": [],
            }
        },
    }
    if with_joint_current:
        profile["interfaces"] = {
            "joint.current": {
                "capability": "joint.state",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/so101_follower/joint_currents",
                "message_type": "ibrobot_msgs/msg/JointCurrent",
                "joint_names": ["1"],
            }
        }
        profile["simulation"]["mujoco"]["joint_current"] = {"scale": [2.0]}
    return profile


# ---------------------------------------------------------------------------
# Rotation helpers for the TF ↔ MJCF agreement test
# ---------------------------------------------------------------------------


def _rpy_to_matrix(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _mat_vec(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def _quat_to_matrix(w, x, y, z):
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _urdf_joint_transforms(tf_xml: str, camera: str):
    """(xyz, R) of the mount joint and the optical joint from camera_tf_urdf."""
    root = ET.fromstring(f"<robot>{tf_xml}</robot>")
    mounts = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = tuple(float(v) for v in origin.get("xyz").split())
        r, p, y = (float(v) for v in origin.get("rpy").split())
        mounts[joint.get("name")] = (xyz, _rpy_to_matrix(r, p, y))
    return mounts[f"{camera}_camera_joint"], mounts[f"{camera}_optical_joint"]


# ---------------------------------------------------------------------------
# 3.1: the robot MJCF and the camera pose plumbing
# ---------------------------------------------------------------------------


def test_rendered_mjcf_places_base_at_spawn_position_and_yaw(tmp_path):
    profile = _profile(tmp_path)
    spawn = (0.5, -0.25, 0.1, 1.25)
    path = render_robot_mjcf(profile, spawn)
    model = mujoco.MjModel.from_xml_path(str(path))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert tuple(round(v, 6) for v in model.body_pos[bid]) == spawn[:3]
    assert tuple(round(v, 6) for v in model.body_quat[bid]) == (
        round(math.cos(spawn[3] / 2), 6),
        0.0,
        0.0,
        round(math.sin(spawn[3] / 2), 6),
    )


def test_tf_optical_frame_matches_the_mjcf_camera_under_yaw_spawn(tmp_path):
    """The optical frame's TF pose equals the pose the image is rendered from.

    A world-parented camera (spawn-compensated in TF, raw in the MJCF
    worldbody) and a gripper-parented camera (robot-relative on both sides)
    under a non-zero-yaw spawn: the optical frame's viewing axis (+Z) equals
    the MJCo camera's -Z, its image-down (+Y) the camera's -Y, and its +X the
    camera's +X, in the true world frame.
    """
    profile = _profile(tmp_path, peripherals=[TOP, WRIST])
    spawn = (0.3, -0.1, 0.05, 0.9)
    mjcf_path = render_robot_mjcf(profile, spawn)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tf_xml = camera_tf_urdf(profile["peripherals"], spawn_pose=spawn)
    spawn_rotation = _rpy_to_matrix(0.0, 0.0, spawn[3])

    for camera in ("top", "wrist"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{camera}_camera")
        # cam_xpos/cam_xmat are the camera's pose in the true world frame
        cam_pos = tuple(data.cam_xpos[cid])
        cam_rot = tuple(tuple(row) for row in data.cam_xmat[cid].reshape(3, 3))
        (mount_xyz, mount_rot), (opt_xyz, opt_rot) = _urdf_joint_transforms(tf_xml, camera)
        # URDF chain: the model root (world link) sits at the spawn pose in
        # the true world, so the optical pose is spawn @ mount @ optical.
        local = tuple(mount_xyz[i] + sum(mount_rot[i][k] * opt_xyz[k] for k in range(3)) for i in range(3))
        optical_pos = tuple(spawn[i] + sum(spawn_rotation[i][k] * local[k] for k in range(3)) for i in range(3))
        optical_rot = _mat_mul(_mat_mul(spawn_rotation, mount_rot), opt_rot)
        assert optical_pos == pytest.approx(cam_pos, abs=1e-6), camera
        camera_axes = {
            axis: _mat_vec(cam_rot, axis_vector)
            for axis, axis_vector in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)))
        }
        # optical +X == camera +X; optical +Y (image down) == camera -Y; optical +Z (view) == camera -Z
        assert _mat_vec(optical_rot, (1, 0, 0)) == pytest.approx(camera_axes["x"], abs=1e-6)
        assert _mat_vec(optical_rot, (0, 1, 0)) == pytest.approx(tuple(-v for v in camera_axes["y"]), abs=1e-6)
        assert _mat_vec(optical_rot, (0, 0, 1)) == pytest.approx(tuple(-v for v in camera_axes["z"]), abs=1e-6)


def test_saved_override_is_converted_to_mujoco_convention(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    override_dir = tmp_path / ".ros" / "ibrobot" / "sim_camera_overrides"
    override_dir.mkdir(parents=True)
    (override_dir / "top.yaml").write_text(
        "parent_frame: world\nfovy_deg: 55\npose: {x: 0.1, y: 0.2, z: 0.3, roll: 0.1, pitch: 0.2, yaw: 0.3}\n",
        encoding="utf-8",
    )
    transform, fovy = resolve_mujoco_camera_pose(TOP)
    roll, pitch, yaw = gazebo_rpy_to_mujoco_rpy(0.1, 0.2, 0.3)
    assert (transform["roll"], transform["pitch"], transform["yaw"]) == pytest.approx((roll, pitch, yaw))
    assert (transform["x"], transform["y"], transform["z"]) == (0.1, 0.2, 0.3)
    assert transform["parent_frame"] == "world"
    assert fovy == 55


def test_disabled_cameras_emit_nothing(tmp_path):
    disabled = {**TOP, "disabled": True}
    peripherals = [disabled, WRIST]
    assert [(parent, "wrist_camera" in xml) for parent, xml in camera_mjcf_elements(peripherals)] == [("gripper", True)]
    assert "camera_top" not in camera_tf_urdf(peripherals)
    parameters = camera_plugin_parameters(_profile(tmp_path, peripherals=peripherals), peripherals)
    cameras = parameters["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE]
    assert "top_camera" not in cameras and "wrist_camera" in cameras


def test_camera_plugin_parameters_match_the_advertised_contract(tmp_path):
    peripherals = [TOP, WRIST, FRONT]
    profile = _profile(tmp_path, peripherals=peripherals)
    parameters = camera_plugin_parameters(profile, peripherals)
    assert set(parameters) == {"mujoco_plugins"}
    instance = parameters["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE]
    assert instance["type"] == CAMERA_PLUGIN_TYPE
    assert instance["camera_publish_rate"] == mujoco_camera_publish_rate(profile)
    assert [key for key in instance if key.endswith("_camera")] == ["top_camera", "wrist_camera", "front_camera"]
    for periph in peripherals:
        name = periph["name"]
        entry = instance[f"{name}_camera"]
        assert set(entry) == {"policy", "frame_name", "image_topic", "info_topic", "depth_topic"}
        assert entry["policy"] == "streaming"
        # 5b.3: the shared advertised frame — opencv → frame_id, realsense → optical
        expected_frame = str(periph["optical_frame_id"] if periph.get("driver") == "realsense" else periph["frame_id"])
        assert entry["frame_name"] == expected_frame
        assert entry["image_topic"] == f"/camera/{name}/image_raw"
        assert entry["info_topic"] == f"/camera/{name}/camera_info"
        if name == "front":
            # the realsense advertises depth: the plugin's depth stream lands
            # on the advertised native-depth endpoint
            assert entry["depth_topic"] == "/camera/front/depth/image_rect_raw"
        else:
            # opencv cameras are color-only: the always-published depth goes
            # to the runtime-private, never-advertised topic
            assert entry["depth_topic"] == f"/_runtime/mujoco/{name}/depth"


def test_camera_plugin_parameters_without_cameras_is_empty(tmp_path):
    peripherals = [{**TOP, "disabled": True}, {**WRIST, "disabled": True}, {**FRONT, "disabled": True}]
    assert camera_plugin_parameters(_profile(tmp_path, peripherals=peripherals), peripherals) == {}


# ---------------------------------------------------------------------------
# issue #195: the FreeJointStatePublisherPlugin parameters for scene objects
# ---------------------------------------------------------------------------


def test_free_joint_state_plugin_parameters_publish_dynamic_scene_bodies():
    parameters = free_joint_state_plugin_parameters("pick_banana")
    instance = parameters["mujoco_plugins"][FREE_JOINT_STATE_PLUGIN_INSTANCE]
    assert instance["type"] == FREE_JOINT_STATE_PLUGIN_TYPE
    assert instance["body_names"] == ["plate", "banana"], "every layout dynamic body, in layout order"
    assert instance["topic"] == "/sim/pick_banana/free_joint_states", "scene-namespaced, like Gazebo's object_poses"
    assert instance["frame_id"] == "", "world frame"
    assert instance["publish_rate"] == 50.0


def test_free_joint_state_plugin_parameters_are_empty_without_dynamic_objects():
    assert free_joint_state_plugin_parameters("") == {}, "robot-only launch: no plugin"
    assert free_joint_state_plugin_parameters("empty") == {}, "empty scene has no dynamic bodies"
    assert free_joint_state_plugin_parameters("does_not_exist") == {}


def test_merge_mujoco_plugin_parameters_combines_trees_instance_wise(tmp_path):
    peripherals = [TOP]
    cameras = camera_plugin_parameters(_profile(tmp_path, peripherals=peripherals), peripherals)
    objects = free_joint_state_plugin_parameters("pick_banana")

    merged = merge_mujoco_plugin_parameters(cameras, objects)

    assert set(merged["mujoco_plugins"]) == {CAMERA_PLUGIN_INSTANCE, FREE_JOINT_STATE_PLUGIN_INSTANCE}
    assert merged["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE] == cameras["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE]
    assert (
        merged["mujoco_plugins"][FREE_JOINT_STATE_PLUGIN_INSTANCE]
        == (objects["mujoco_plugins"][FREE_JOINT_STATE_PLUGIN_INSTANCE])
    )
    assert merge_mujoco_plugin_parameters() == {}
    assert merge_mujoco_plugin_parameters({}, {}) == {}
    # a later tree wins per key inside one instance
    later = {"mujoco_plugins": {FREE_JOINT_STATE_PLUGIN_INSTANCE: {"publish_rate": 10.0}}}
    assert (
        merge_mujoco_plugin_parameters(objects, later)["mujoco_plugins"][FREE_JOINT_STATE_PLUGIN_INSTANCE][
            "publish_rate"
        ]
        == 10.0
    )


# ---------------------------------------------------------------------------
# D2: the simulator version check
# ---------------------------------------------------------------------------


def test_check_simulator_versions_passes_on_a_match(monkeypatch):
    monkeypatch.setattr(mujoco, "__version__", "3.12.0")
    check_simulator_versions(SimulatorVersions(control="0.1.2", mujoco="3.12.0"))


def test_check_simulator_versions_names_both_versions_on_a_mismatch(monkeypatch):
    monkeypatch.setattr(mujoco, "__version__", "3.4.0")
    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match=r"Python mujoco 3\.4\.0 != mujoco_vendor libmujoco 3\.12\.0"):
        check_simulator_versions(SimulatorVersions(control="0.1.2", mujoco="3.12.0"))


def test_check_simulator_versions_missing_control_package_diagnoses_the_install():
    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match="mujoco_ros2_control not found; install"):
        check_simulator_versions(SimulatorVersions(control=None, mujoco="3.12.0"))


def test_check_simulator_versions_missing_vendor_library_diagnoses_the_install(monkeypatch):
    monkeypatch.setattr(mujoco, "__version__", "3.12.0")
    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match="mujoco_vendor libmujoco not found"):
        check_simulator_versions(SimulatorVersions(control="0.1.2", mujoco=None))


def test_check_simulator_versions_too_old_control_diagnoses_the_upgrade(monkeypatch):
    """A pre-CameraPlugin release fails even with matching MuJoCo versions."""
    monkeypatch.setattr(mujoco, "__version__", "3.4.0")
    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match=r"mujoco_ros2_control 0\.0\.3 is older than the CameraPlugin"):
        check_simulator_versions(SimulatorVersions(control="0.0.3", mujoco="3.4.0"))


def test_check_simulator_versions_missing_python_mujoco_is_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", None)
    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match="needs the mujoco Python package"):
        check_simulator_versions(SimulatorVersions(control="0.1.2", mujoco="3.12.0"))


def test_simulator_versions_reads_a_fake_prefix_tree(tmp_path, monkeypatch):
    """simulator_versions() reads package.xml + libmujoco from the ament index."""
    import ament_index_python.packages as packages
    from ament_index_python.packages import PackageNotFoundError

    prefix = tmp_path / "opt-ros"
    (prefix / "share" / "mujoco_ros2_control").mkdir(parents=True)
    (prefix / "share" / "mujoco_ros2_control" / "package.xml").write_text(
        '<?xml version="1.0"?><package><name>mujoco_ros2_control</name><version>0.1.2</version></package>',
        encoding="utf-8",
    )
    lib_dir = prefix / "opt" / "mujoco_vendor" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libmujoco.so.3.12.0").touch()
    monkeypatch.setattr(packages, "get_package_share_directory", lambda name: str(prefix / "share" / name))
    monkeypatch.setattr(packages, "get_package_prefix", lambda name: str(prefix))
    assert simulator_versions() == SimulatorVersions(control="0.1.2", mujoco="3.12.0")

    # a missing package or library leaves the field None, never a guess
    def _raise_lookup(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(packages, "get_package_share_directory", _raise_lookup)
    assert simulator_versions() == SimulatorVersions(control=None, mujoco="3.12.0")
    empty_prefix = tmp_path / "empty-ros"
    empty_prefix.mkdir()
    monkeypatch.setattr(packages, "get_package_share_directory", lambda name: str(prefix / "share" / name))
    monkeypatch.setattr(packages, "get_package_prefix", lambda name: str(empty_prefix))
    assert simulator_versions() == SimulatorVersions(control="0.1.2", mujoco=None)


def test_camera_rate_defaults_to_min_fps_and_honours_the_profile():
    assert mujoco_camera_publish_rate(_profile(Path(tempfile.mkdtemp()))) == 30.0  # min(30, 60, 30)
    profile = _profile(Path(tempfile.mkdtemp()))
    profile["simulation"]["mujoco"]["camera_publish_rate"] = 12.5
    assert mujoco_camera_publish_rate(profile) == 12.5


# ---------------------------------------------------------------------------
# 3.3: the MuJoCo embodiment launch structure
# ---------------------------------------------------------------------------


def _nodes(actions):
    return [a for a in actions if isinstance(a, Node)]


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _param_value(value):
    """Resolve a normalized launch parameter the way evaluation would."""
    if isinstance(value, tuple | list):
        if value and all(not hasattr(item, "text") and not isinstance(item, tuple | list) for item in value):
            return list(value)  # plainly typed array (bool/int/float)
        if any(isinstance(item, tuple | list) for item in value):
            return [_param_value(item) for item in value]
        import yaml

        return yaml.safe_load(_text(value)) if value else []
    return value


def _params(node):
    params = {}
    for entry in node._Node__parameters or []:
        if isinstance(entry, dict):
            params.update({_text(key): _param_value(value) for key, value in entry.items()})
    return params


def test_mujoco_action_set_has_exactly_one_simulator_and_sim_time_everywhere(tmp_path):
    profile = _profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml")
    actions = mujoco_embodiment_actions(
        profile, tmp_path / "profile.yaml", robot_description=description, mujoco_scene=""
    )
    nodes = _nodes(actions)
    executables = {node._Node__node_executable for node in nodes}
    # exactly one simulator+controller-manager process, and it is the plugin's
    mujoco_nodes = [node for node in nodes if node._Node__node_executable == "ros2_control_node"]
    assert len(mujoco_nodes) == 1
    assert mujoco_nodes[0]._Node__package == "mujoco_ros2_control"
    # no generic controller-manager ros2_control_node (the plugin embeds it);
    # the controller_manager package only appears as the spawners
    assert not [
        node
        for node in nodes
        if node._Node__package == "controller_manager" and node._Node__node_executable == "ros2_control_node"
    ]
    assert "robot_state_publisher" in executables
    assert "runtime_facade" in executables
    assert "spawner" in executables
    assert "joint_current_publisher" in executables
    assert "topic_relay" in executables  # the depth fan-out
    # every node runs on simulation time
    for node in nodes:
        params = _params(node)
        if node._Node__node_executable == "spawner":
            continue  # spawners carry use_sim_time via parameters list (checked below)
        assert params.get("use_sim_time") is True, node._Node__node_name
    spawners = [node for node in nodes if node._Node__node_executable == "spawner"]
    assert all(_params(spawner).get("use_sim_time") is True for spawner in spawners)
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    assert _params(facade)["sim_platform"] == "mujoco"
    assert _params(facade)["use_sim"] is True
    joint_current = next(node for node in nodes if node._Node__node_executable == "joint_current_publisher")
    assert _params(joint_current)["scale"] == [2.0]  # the mujoco platform section


def test_mujoco_urdf_carries_plugin_params_and_camera_contract(tmp_path):
    profile = _profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml")
    assert "mujoco_ros2_control/MujocoSystemInterface" in description
    assert "mujoco_model" in description and ".mjb" in description
    assert '<state_interface name="effort"' in description
    root = ET.fromstring(description)
    # D3: no simulator-specific camera <sensor> blocks, and the deprecated
    # auto-registration is explicitly off; the rate is a plugin parameter now
    assert root.findall(".//ros2_control/sensor") == []
    params = {param.get("name"): param.text for param in root.findall(".//ros2_control/hardware/param")}
    assert params["auto_register_cameras"] == "false"
    assert "camera_publish_rate" not in params
    # the camera TF links exist for every enabled camera
    for periph in (TOP, WRIST, FRONT):
        assert f'link name="{periph["frame_id"]}"' in description
        assert f'link name="{periph["optical_frame_id"]}"' in description


def test_mujoco_embodiment_passes_the_camera_plugin_params_file(tmp_path, monkeypatch):
    """D3/D6: the params file is keyed under the upstream node name and
    carries the CameraPlugin tree (structure only — the version result is
    injected and the composition faked, so no installed simulator needed)."""
    import yaml

    import robot_runtime.mujoco as mujoco_helpers
    import sim_models.scene_compiler as scene_compiler

    # structure only: the injected simulator version follows this module's
    # imported mujoco (already importorskip'ed), and the composition is faked
    monkeypatch.setattr(
        mujoco_helpers,
        "simulator_versions",
        lambda: mujoco_helpers.SimulatorVersions(control="0.1.2", mujoco=mujoco.__version__),
    )
    monkeypatch.setattr(
        scene_compiler, "compose_mujoco_model", lambda *args, **kwargs: Path("/tmp/ibrobot_fake_composed.mjb")
    )
    profile = _profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml")
    actions = mujoco_embodiment_actions(
        profile, tmp_path / "profile.yaml", robot_description=description, mujoco_scene=""
    )
    simulator_nodes = [node for node in _nodes(actions) if node._Node__node_executable == "ros2_control_node"]
    assert [node._Node__package for node in simulator_nodes] == ["mujoco_ros2_control"]
    from launch_ros.parameter_descriptions import ParameterFile

    def _param_file_paths(node):
        # launch_ros normalizes a parameter-file path to a list of substitutions
        return [
            "".join(getattr(part, "text", None) or str(part) for part in entry.param_file)
            for entry in node._Node__parameters
            if isinstance(entry, ParameterFile)
        ]

    param_files = [Path(path) for path in _param_file_paths(simulator_nodes[0])]
    keyed = [path for path in param_files if MUJOCO_SIMULATION_NODE in yaml.safe_load(path.read_text())]
    assert len(keyed) == 1, f"exactly one {MUJOCO_SIMULATION_NODE} params file, got {param_files}"
    tree = yaml.safe_load(Path(keyed[0]).read_text())[MUJOCO_SIMULATION_NODE]["ros__parameters"]
    assert tree["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE]["type"] == CAMERA_PLUGIN_TYPE
    # the empty scene composes no dynamic objects: no free-joint publisher
    assert FREE_JOINT_STATE_PLUGIN_INSTANCE not in tree["mujoco_plugins"]


def test_mujoco_embodiment_adds_the_free_joint_state_plugin_for_dynamic_scenes(tmp_path, monkeypatch):
    """Issue #195: a scene with dynamic objects gets the
    FreeJointStatePublisherPlugin in the same params file as the cameras."""
    import yaml

    import robot_runtime.mujoco as mujoco_helpers
    import sim_models.scene_compiler as scene_compiler

    monkeypatch.setattr(
        mujoco_helpers,
        "simulator_versions",
        lambda: mujoco_helpers.SimulatorVersions(control="0.1.2", mujoco=mujoco.__version__),
    )
    monkeypatch.setattr(
        scene_compiler, "compose_mujoco_model", lambda *args, **kwargs: Path("/tmp/ibrobot_fake_composed.mjb")
    )
    profile = _profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml")
    actions = mujoco_embodiment_actions(
        profile, tmp_path / "profile.yaml", robot_description=description, mujoco_scene="pick_banana"
    )
    simulator_nodes = [node for node in _nodes(actions) if node._Node__node_executable == "ros2_control_node"]
    from launch_ros.parameter_descriptions import ParameterFile

    param_files = [
        Path("".join(getattr(part, "text", None) or str(part) for part in entry.param_file))
        for entry in simulator_nodes[0]._Node__parameters
        if isinstance(entry, ParameterFile)
    ]
    keyed = [path for path in param_files if MUJOCO_SIMULATION_NODE in yaml.safe_load(path.read_text())]
    assert len(keyed) == 1
    tree = yaml.safe_load(Path(keyed[0]).read_text())[MUJOCO_SIMULATION_NODE]["ros__parameters"]
    instance = tree["mujoco_plugins"][FREE_JOINT_STATE_PLUGIN_INSTANCE]
    assert instance["type"] == FREE_JOINT_STATE_PLUGIN_TYPE
    assert instance["body_names"] == ["plate", "banana"]
    assert instance["topic"] == "/sim/pick_banana/free_joint_states"
    # the cameras still ride in the same file
    assert tree["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE]["type"] == CAMERA_PLUGIN_TYPE


def test_descriptor_streams_match_the_generated_sensors_and_relays(tmp_path):
    peripherals = [TOP, WRIST, FRONT]
    description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": peripherals,
        },
        use_sim=True,
        sim_platform="mujoco",
    )
    interfaces = description["interfaces"]

    # color is rgb8 on mujoco even for the bgr8 realsense device; depth 32FC1
    assert interfaces["camera.front.color"]["configured_profile"]["encoding"] == "rgb8"
    assert interfaces["camera.top.color"]["configured_profile"]["encoding"] == "rgb8"
    assert interfaces["camera.front.depth"]["configured_profile"]["encoding"] == "32FC1"
    # the plugin's publishers are reliable depth-1: the contract offers
    # reliable delivery (best-effort readers lose large image fragments)
    assert interfaces["camera.top.color"]["qos"]["reliability"] == "reliable"
    assert interfaces["camera.front.depth"]["qos"]["reliability"] == "reliable"
    assert interfaces["camera.front.depth_info"]["qos"]["reliability"] == "reliable"
    # non-mujoco platforms keep the historical best-effort sensor contract
    physical = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": peripherals,
        }
    )
    assert physical["interfaces"]["camera.top.color"]["qos"]["reliability"] == "best_effort"
    # 5b.3: the advertised frames follow the shared convention (opencv →
    # frame_id, realsense → optical), identical to hardware/sdk/gazebo
    assert interfaces["camera.top.color"]["frame_id"] == "camera_top_frame"
    assert interfaces["camera.front.color"]["frame_id"] == "camera_front_optical_frame"
    # the effective plugin rate is advertised on every stream
    assert interfaces["camera.top.color"]["configured_profile"]["fps"] == 30.0
    assert interfaces["camera.front.depth"]["configured_profile"]["fps"] == 30.0
    # depth info endpoints are advertised (the relays fill them)
    assert "camera.front.depth_info" in interfaces
    assert "camera.front.aligned_depth_info" in interfaces

    # every advertised camera endpoint is produced by a plugin entry or a
    # relay; private depth topics (color-only cameras) are never advertised
    peripherals = [TOP, WRIST, FRONT]
    produced: set[str] = set()
    for entry in camera_plugin_parameters(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": peripherals,
        },
        peripherals,
    )["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE].values():
        if isinstance(entry, dict):
            produced.add(entry["image_topic"])
            produced.add(entry["info_topic"])
            if not entry["depth_topic"].startswith("/_runtime/"):
                produced.add(entry["depth_topic"])
    relay_nodes = mujoco_camera_relays(peripherals)
    for node in relay_nodes:
        args = node._Node__arguments
        produced.add(args[1])  # the relay target
    advertised = {
        interface["endpoint"]
        for key, interface in interfaces.items()
        if key.startswith("camera.") and interface["kind"] == "topic" and "CameraInfo" not in interface["message_type"]
    } | {
        interface["camera_info_topic"]
        for key, interface in interfaces.items()
        if key.startswith("camera.") and interface.get("camera_info_topic")
    }
    assert advertised == produced, f"advertised {sorted(advertised)} vs produced {sorted(produced)}"


def test_runtime_stack_actions_dispatches_mujoco(tmp_path):
    profile = _profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml")
    actions = runtime_stack_actions(
        profile, tmp_path / "profile.yaml", True, sim_platform="mujoco", robot_description=description
    )
    nodes = _nodes(actions)
    assert [n._Node__package for n in nodes if n._Node__node_executable == "ros2_control_node"] == [
        "mujoco_ros2_control"
    ]


def test_mujoco_relays_run_reliable():
    """The depth fan-out relays subscribe and publish reliable (QoS root cause
    of the 3.5 rate investigation: the plugin's reliable depth-1 publishers
    lose large image fragments for best-effort readers)."""
    relays = mujoco_camera_relays([TOP, WRIST, FRONT])
    assert relays, "the front camera's depth streams need relays"
    for node in relays:
        assert "--reliable" in node._Node__arguments, node._Node__node_name


def test_topic_relay_reliable_flag_selects_the_qos():
    """TopicRelay's --reliable flag upgrades both sides to reliable QoS."""
    import pytest as _pytest

    _pytest.importorskip("rclpy")
    from robot_runtime.topic_relay import TopicRelay

    _pytest.importorskip("mujoco")  # keep the skip semantics of this module
    import rclpy
    from rclpy.qos import QoSReliabilityPolicy

    rclpy.init()
    try:
        best_effort = TopicRelay("/source", "/target", "std_msgs/msg/Header")
        reliable = TopicRelay("/source2", "/target2", "std_msgs/msg/Header", reliable=True)
        assert best_effort._subscription.qos_profile.reliability == QoSReliabilityPolicy.BEST_EFFORT
        assert reliable._subscription.qos_profile.reliability == QoSReliabilityPolicy.RELIABLE
        assert reliable._publisher.qos_profile.reliability == QoSReliabilityPolicy.RELIABLE
        best_effort.destroy_node()
        reliable.destroy_node()
    finally:
        rclpy.shutdown()


def test_empty_scene_composes_ground_and_light(tmp_path):
    """5b.2: an empty mujoco_scene composes with the sim_models `empty` scene."""
    import mujoco

    from sim_models.scene_compiler import composed_model_path

    profile = _profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml", scene="")
    composed = composed_model_path("empty")
    assert str(composed) in description
    model = mujoco.MjModel.from_binary_path(str(composed))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0, "lit ground plane present"
    assert model.nlight >= 1


def test_explicit_scene_none_stays_robot_only(tmp_path):
    """5b.3 companion: scene=None (a direct API request) keeps robot-only composition."""
    from sim_models.scene_compiler import composed_model_path

    profile = _profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml", scene=None)
    assert str(composed_model_path(None)) in description


# ---------------------------------------------------------------------------
# lekiwi-mujoco-runtime-platform D3/D5/D9: lidar/IMU MJCF, plugin parameters,
# the floating-base constants
# ---------------------------------------------------------------------------


def _mid360(**simulation):
    return {
        "type": "lidar",
        "name": "mid360",
        "driver": "livox_mid360",
        "frame_id": "livox_frame",
        "pointcloud_topic": "/livox/lidar",
        "imu_topic": "/livox/imu",
        "publish_freq": 10.0,
        "transform": {
            "parent_frame": "base_link",
            "x": -0.08,
            "y": 0.0,
            "z": 0.2,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 1.5707963,
        },
        "simulation": {"mujoco": dict(simulation)},
    }


def test_lidar_imu_mjcf_injects_site_plugin_sensor_and_imu_triple():
    from robot_runtime.mujoco import lidar_imu_mjcf

    # the D1 LeKiwi grid (the profile declares what the Gazebo section declares)
    periph = _mid360(horizontal_samples=360, vertical_samples=32, vertical_min_deg=-7.0, vertical_max_deg=52.0)
    sites, sensors, needs_extension = lidar_imu_mjcf([periph])
    assert needs_extension
    assert len(sites) == 1
    parent, site_xml = sites[0]
    assert parent == "base_link"
    site = ET.fromstring(site_xml)
    assert site.tag == "site" and site.get("name") == "mid360_lidar_site"
    # the resolved mount pose, yaw in the w-x-y-z quaternion (extrinsic XYZ)
    assert site.get("pos") == "-0.080000 0.000000 0.200000"
    quat = [float(v) for v in site.get("quat").split()]
    assert abs(quat[0] - math.cos(math.pi / 4)) < 1e-5 and abs(quat[3] - math.sin(math.pi / 4)) < 1e-5
    # the plugin sensor on the site, the grid from the peripheral's params
    plugin = ET.fromstring(sensors[0])
    assert plugin.get("plugin") == "mujoco.plugin.lidar" and plugin.get("objname") == "mid360_lidar_site"
    config = {c.get("key"): c.get("value") for c in plugin}
    assert config["resolution"] == "360 32"  # the D1 defaults
    assert config["azimuth_range"] == "-3.14159265 3.14159265"
    assert config["elevation_range"] == f"{math.radians(-7.0):.6f} {math.radians(52.0):.6f}"
    assert config["async"] == "0"
    # the integrated IMU triple on the same site (co-located, zero extrinsic)
    names = [ET.fromstring(s).get("name") for s in sensors[1:]]
    assert names == ["mid360_imu_quat", "mid360_imu_gyro", "mid360_imu_accel"]
    assert all(ET.fromstring(s).get("site") == "mid360_lidar_site" for s in sensors[2:4])
    # a profile without a simulated lidar renders nothing
    assert lidar_imu_mjcf([]) == ([], [], False)
    assert lidar_imu_mjcf([{**_mid360(), "driver": "ldlidar"}]) == ([], [], False)


def test_lidar_imu_mjcf_prefers_the_simulation_mujoco_transform():
    from robot_runtime.mujoco import lidar_imu_mjcf

    periph = _mid360(vertical_samples=32)
    periph["simulation"]["mujoco"]["transform"] = {"parent_frame": "chassis", "x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.0}
    sites, _sensors, _needs = lidar_imu_mjcf([periph])
    parent, site_xml = sites[0]
    assert parent == "chassis"
    assert ET.fromstring(site_xml).get("pos") == "1.000000 2.000000 3.000000"


def test_lidar_imu_mjcf_planar_lidar_renders_one_elevation_sample():
    from robot_runtime.mujoco import lidar_imu_mjcf

    periph = {**_mid360(), "driver": "livox_mid360"}
    periph["simulation"]["mujoco"]["vertical_samples"] = 1
    _sites, sensors, _needs = lidar_imu_mjcf([periph])
    config = {c.get("key"): c.get("value") for c in ET.fromstring(sensors[0])}
    assert config["resolution"] == "360 1" and config["elevation_range"] == "0 0"


def test_render_robot_mjcf_injects_lidar_into_the_template(tmp_path):
    template = tmp_path / "robot.xml.template"
    template.write_text(MJCF_TEMPLATE.replace('name="base"', 'name="base_link"'), encoding="utf-8")
    profile = {
        "simulation": {
            "mujoco": {"template": str(template), "meshes_dir": str(tmp_path)},
        },
        "peripherals": [_mid360()],
    }
    out = render_robot_mjcf(profile)
    root = ET.parse(out).getroot()
    # the site inside the mount body
    site = root.find('.//body[@name="base_link"]/site[@name="mid360_lidar_site"]')
    assert site is not None
    # the sensor section with the plugin sensor and the IMU triple
    sensor_names = [s.get("name") for s in root.find("sensor")]
    assert sensor_names == ["mid360_lidar", "mid360_imu_quat", "mid360_imu_gyro", "mid360_imu_accel"]
    # the engine extension, exactly once
    extensions = root.findall("extension/plugin[@plugin='mujoco.plugin.lidar']")
    assert len(extensions) == 1
    # a profile without a lidar renders no site/sensor/extension (byte-identical SO-101 path)
    out_plain = render_robot_mjcf({**profile, "peripherals": []})
    plain = ET.parse(out_plain).getroot()
    assert plain.find(".//site") is None and plain.find("sensor") is None and plain.find("extension") is None


def test_lidar_plugin_parameters_target_the_advertised_frame_and_raw_topic():
    from robot_runtime.mujoco import LIDAR_PLUGIN_INSTANCE, LIDAR_PLUGIN_TYPE, lidar_plugin_parameters

    tree = lidar_plugin_parameters([_mid360()])
    instance = tree["mujoco_plugins"][LIDAR_PLUGIN_INSTANCE]
    assert instance["type"] == LIDAR_PLUGIN_TYPE
    assert instance["mid360_lidar"] == {"frame_name": "livox_frame", "topic": "/mid360_lidar/raw"}
    # only the drivers MuJoCo simulates; nothing without a lidar
    assert lidar_plugin_parameters([{**_mid360(), "driver": "ldlidar"}]) == {}
    assert lidar_plugin_parameters([]) == {}
    # disabled entries are absent
    assert lidar_plugin_parameters([{**_mid360(), "enabled": False}]) == {}


def test_base_velocity_plugin_parameters_carry_no_second_clamp():
    from robot_runtime.mujoco import (
        BASE_VELOCITY_PLUGIN_INSTANCE,
        BASE_VELOCITY_PLUGIN_TYPE,
        FLOATING_BASE_CMD_TOPIC,
        base_velocity_plugin_parameters,
    )

    tree = base_velocity_plugin_parameters({"body": "base_link", "twist_source_topic": "/odom"})
    instance = tree["mujoco_plugins"][BASE_VELOCITY_PLUGIN_INSTANCE]
    assert instance["type"] == BASE_VELOCITY_PLUGIN_TYPE
    assert instance["body"] == "base_link"
    assert instance["cmd_vel_topic"] == FLOATING_BASE_CMD_TOPIC
    assert instance["use_stamped_twist"] is True
    assert instance["cmd_timeout"] == 0.25
    # D3.2: no limit keys — upstream defaults are +inf and the single
    # authoritative clamp lives in base_node
    assert "max_linear_velocity" not in instance and "max_yaw_rate" not in instance
    # no floating-base section: no plugin (the robot has no free joint to drive)
    assert base_velocity_plugin_parameters(None) == {}
    assert base_velocity_plugin_parameters({}) == {}


def test_floating_base_constants_are_simulator_namespaced():
    from robot_runtime.mujoco import FLOATING_BASE_CMD_TOPIC, FLOATING_BASE_STATE_TOPIC, ODOM_FREE_JOINT_NAME

    assert ODOM_FREE_JOINT_NAME == "floating_base_joint"
    assert FLOATING_BASE_STATE_TOPIC == "/simulator/floating_base_state"
    assert FLOATING_BASE_CMD_TOPIC == "/simulator/floating_base/cmd_vel"


def test_imu_sensor_urdf_maps_the_mjcf_triple_with_the_default_suffixes():
    from robot_runtime.mujoco import imu_sensor_urdf

    xml = imu_sensor_urdf([_mid360()])
    sensor = ET.fromstring(f"<ros2_control>{xml}</ros2_control>").find("sensor")
    assert sensor.get("name") == "mid360_imu"
    params = {p.get("name"): p.text for p in sensor.findall("param")}
    assert params == {"mujoco_type": "imu", "mujoco_sensor_name": "mid360_imu"}
    interfaces = [s.get("name") for s in sensor.findall("state_interface")]
    assert len(interfaces) == 10
    assert interfaces[:4] == ["orientation.x", "orientation.y", "orientation.z", "orientation.w"]
    assert interfaces[-3:] == ["linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]
    # no integrated IMU -> no sensor block
    assert imu_sensor_urdf([]) == ""
    assert imu_sensor_urdf([{**_mid360(), "driver": "ldlidar"}]) == ""


# ---------------------------------------------------------------------------
# lekiwi-mujoco-runtime-platform D3/D5: the floating-base + lidar embodiment
# ---------------------------------------------------------------------------

LEKIWI_MJCF = """<mujoco model="lekiwi_probe">
  <compiler eulerseq="XYZ" angle="radian"/>
  <worldbody>
    <body name="base_link" pos="{{ROBOT_BASE_POS}}" quat="{{ROBOT_BASE_QUAT}}">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.13 0.13 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def _lekiwi_profile(tmp_path: Path):
    mjcf = tmp_path / "lekiwi.xml.template"
    mjcf.write_text(LEKIWI_MJCF, encoding="utf-8")
    meshes = tmp_path / "meshes"
    meshes.mkdir(exist_ok=True)
    xacro = tmp_path / "robot.xacro"
    xacro.write_text(XACRO, encoding="utf-8")
    controllers = tmp_path / "controllers.yaml"
    controllers.write_text(
        "controller_manager:\n"
        "  ros__parameters:\n"
        "    update_rate: 100\n"
        "    base_velocity_controller:\n"
        "      type: velocity_controllers/JointGroupVelocityController\n"
        "    joint_state_broadcaster:\n"
        "      type: joint_state_broadcaster/JointStateBroadcaster\n",
        encoding="utf-8",
    )
    return {
        "runtime": {"name": "lekiwi_robot", "version": "0.1.0"},
        "modes": {
            "initial": "idle",
            "idle": {
                "controllers": [],
                "controllers_by_platform": {"mujoco": ["base_velocity_controller"]},
                "transitions": ["base_navigation"],
            },
            "base_navigation": {
                "controllers": ["base_velocity_controller"],
                "allows_base": True,
                "transitions": ["idle"],
            },
        },
        "capabilities": {"joint.state": {}, "runtime.stop": {}},
        "joints": ["7", "8", "9"],
        "sim_platforms": ["sdk", "gazebo", "mujoco"],
        "sim_platform": "sdk",
        "controller_manager": "controller_manager",
        "joint_state_topic": "/joint_states",
        "command_channels": [],
        "trajectory_actions": [],
        "description": {"xacro": str(xacro)},
        "ros2_control": {"controllers_config": str(controllers), "broadcasters": ["joint_state_broadcaster"]},
        "peripherals": [_mid360(horizontal_samples=360, vertical_samples=32)],
        "hardware": {},
        "simulation": {
            "mujoco": {
                "template": str(mjcf),
                "meshes_dir": str(meshes),
                "sdf_meshes": [],
                "floating_base": {"body": "base_link", "twist_source_topic": "/odom"},
            }
        },
    }


def test_mujoco_embodiment_composes_the_floating_base_and_lidar(tmp_path, monkeypatch):
    """D3/D5: the plugin tree, the generated controllers file, the filters,
    the relay, the remaps and the platform spawner set of the LeKiwi shape."""
    import yaml

    import robot_runtime.mujoco as mujoco_helpers
    import sim_models.scene_compiler as scene_compiler

    monkeypatch.setattr(
        mujoco_helpers,
        "simulator_versions",
        lambda: mujoco_helpers.SimulatorVersions(control="0.1.2", mujoco=mujoco.__version__),
    )
    monkeypatch.setattr(
        scene_compiler, "compose_mujoco_model", lambda *args, **kwargs: Path("/tmp/ibrobot_fake_composed.mjb")
    )
    profile = _lekiwi_profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml")
    actions = mujoco_embodiment_actions(
        profile, tmp_path / "profile.yaml", robot_description=description, mujoco_scene=""
    )
    simulator = [node for node in _nodes(actions) if node._Node__node_executable == "ros2_control_node"][0]
    from launch_ros.parameter_descriptions import ParameterFile

    param_files = [
        Path("".join(getattr(part, "text", None) or str(part) for part in entry.param_file))
        for entry in simulator._Node__parameters
        if isinstance(entry, ParameterFile)
    ]
    keyed = [path for path in param_files if MUJOCO_SIMULATION_NODE in yaml.safe_load(path.read_text())]
    assert len(keyed) == 1
    tree = yaml.safe_load(Path(keyed[0]).read_text())[MUJOCO_SIMULATION_NODE]["ros__parameters"]["mujoco_plugins"]
    # the floating-base drive: stamped twist, internal topic, timeout, no clamp
    base = tree["ibrobot_floating_base"]
    assert base["type"] == "mujoco_ros2_control_plugins/BaseVelocityPlugin"
    assert base["body"] == "base_link" and base["cmd_vel_topic"] == "/simulator/floating_base/cmd_vel"
    assert base["use_stamped_twist"] is True and base["cmd_timeout"] == 0.25
    assert "max_linear_velocity" not in base and "max_yaw_rate" not in base
    # the lidar publisher
    lidars = tree["ibrobot_lidars"]
    assert lidars["type"] == "mujoco_ros2_control_plugins/Mujoco3dLidarPlugin"
    assert lidars["mid360_lidar"] == {"frame_name": "livox_frame", "topic": "/mid360_lidar/raw"}
    # the generated controllers file carries the IMU broadcaster
    controllers_files = [
        path for path in param_files if path not in keyed and "mid360_imu_broadcaster" in path.read_text()
    ]
    assert len(controllers_files) == 1, f"one generated controllers file, got {param_files}"
    generated = yaml.safe_load(Path(controllers_files[0]).read_text())
    assert generated["controller_manager"]["ros__parameters"]["mid360_imu_broadcaster"] == {
        "type": "imu_sensor_broadcaster/IMUSensorBroadcaster"
    }
    assert generated["mid360_imu_broadcaster"]["ros__parameters"] == {
        "sensor_name": "mid360_imu",
        "frame_id": "livox_frame",
    }
    # the shipped entries survive the merge
    assert "base_velocity_controller" in generated["controller_manager"]["ros__parameters"]
    # the process-wide IMU remap (launch stores substitutions, not strings)
    remaps = {
        tuple("".join(getattr(part, "text", None) or str(part) for part in side) for side in remap)
        for remap in simulator._Node__remappings
    }
    assert ("/mid360_imu_broadcaster/imu", "/mid360_imu/raw") in remaps
    # the filters, the relay and the static TF
    executables = {
        (node._Node__package, node._Node__node_executable, node._Node__node_name): node for node in _nodes(actions)
    }

    def _plain(parameters_entry):
        # launch stores parameter names and string values as substitution
        # tuples, appending a "\n...\n" marker to unperformed scalars
        def _text(value):
            if isinstance(value, list | tuple):
                text = "".join(getattr(part, "text", None) or str(part) for part in value)
                return text.removesuffix("\n...\n")
            return value

        return {_text(key): _text(value) for key, value in parameters_entry.items()}

    cloud_filter = executables[("robot_runtime", "finite_cloud_filter", "mid360_finite_cloud_filter")]
    assert _plain(cloud_filter._Node__parameters[0]) == {
        "input_topic": "/mid360_lidar/raw",
        "output_topic": "/livox/lidar",
        "add_intensity": True,
    }
    imu_filter = executables[("robot_runtime", "imu_settle_filter", "mid360_imu_settle_filter")]
    assert _plain(imu_filter._Node__parameters[0]) == {
        "input_topic": "/mid360_imu/raw",
        "output_topic": "/livox/imu",
    }
    relay = executables[("robot_runtime", "odometry_twist_relay", "odometry_twist_relay")]
    assert _plain(relay._Node__parameters[0]) == {
        "twist_source_topic": "/odom",
        "cmd_vel_topic": "/simulator/floating_base/cmd_vel",
        "base_frame": "base_link",
        "use_sim_time": True,
    }
    static_tf = [node for node in _nodes(actions) if node._Node__node_executable == "static_transform_publisher"]
    assert static_tf, "the sensor frame absent from the model gets the runtime static TF"
    # the platform spawner set: the base controller spawns ACTIVE on mujoco
    # (D3.1) and the generated IMU broadcaster joins the broadcasters
    spawners = [node for node in _nodes(actions) if node._Node__node_executable == "spawner"]
    active_spawner = [n for n in spawners if "--inactive" not in n._Node__arguments][0]
    assert "base_velocity_controller" in active_spawner._Node__arguments
    assert "mid360_imu_broadcaster" in active_spawner._Node__arguments


def test_render_mujoco_description_injects_the_imu_sensor_into_ros2_control(tmp_path, monkeypatch):
    import robot_runtime.mujoco as mujoco_helpers
    import sim_models.scene_compiler as scene_compiler

    monkeypatch.setattr(
        mujoco_helpers,
        "simulator_versions",
        lambda: mujoco_helpers.SimulatorVersions(control="0.1.2", mujoco=mujoco.__version__),
    )
    monkeypatch.setattr(
        scene_compiler, "compose_mujoco_model", lambda *args, **kwargs: Path("/tmp/ibrobot_fake_composed.mjb")
    )
    profile = _lekiwi_profile(tmp_path)
    description = render_mujoco_description(profile, tmp_path / "profile.yaml")
    root = ET.fromstring(description)
    sensors = root.findall("ros2_control/sensor")
    assert [s.get("name") for s in sensors] == ["mid360_imu"]
    params = {p.get("name"): p.text for p in sensors[0].findall("param")}
    assert params == {"mujoco_type": "imu", "mujoco_sensor_name": "mid360_imu"}
    assert len(sensors[0].findall("state_interface")) == 10
    # a profile without a simulated lidar renders no sensor (SO-101 unchanged)
    plain = render_mujoco_description(_profile(tmp_path), tmp_path / "profile.yaml")
    assert ET.fromstring(plain).find("ros2_control/sensor") is None
