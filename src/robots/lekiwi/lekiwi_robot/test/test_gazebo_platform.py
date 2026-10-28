"""LeKiwi Gazebo platform tests (openspec lekiwi-gazebo-runtime-platform).

Drives the shipped ``runtime.launch.py`` and the shipped profiles: the
platform declaration and its rejection surface (D1), the one-description
renders per platform (D2), the launch plumbing through the real entry —
world/spawn arguments, the platform-aware single render, simulation-time
wiring (D7) — and the composed MID-360 mount forwarding (D13).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import launch
import yaml
from launch_ros.actions import Node

_PACKAGE = Path(__file__).resolve().parents[1]
_LAUNCH = _PACKAGE / "launch" / "runtime.launch.py"
LIDAR_PROFILE = _PACKAGE / "profiles" / "lekiwi_lidar.yaml"
GRASP_PROFILE = _PACKAGE / "profiles" / "lekiwi_nav_grasp.yaml"
MOUNT = _PACKAGE.parent / "lekiwi_description" / "config" / "lekiwi_mid360_mount.yaml"


def _launch_module():
    spec = importlib.util.spec_from_file_location("lekiwi_robot_runtime_launch_gazebo", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(profile, extra):
    context = launch.LaunchContext()
    configurations = {
        "profile": str(profile),
        "simulated": "",
        "use_sim": "",
        "sim_platform": "",
        "gazebo_world": "",
        "mujoco_scene": "",
        "spawn_x": "0.0",
        "spawn_y": "0.0",
        "spawn_z": "0.05",
        "spawn_yaw": "0.0",
        "display": "false",
        "instance_id": "",
        "peripherals": "",
    }
    configurations.update(extra)
    context.launch_configurations.update(configurations)
    return _launch_module()._launch_setup(context)


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _param_value(value):
    if isinstance(value, tuple | list):
        if value and all(not hasattr(item, "text") and not isinstance(item, tuple | list) for item in value):
            return list(value)
        if any(isinstance(item, tuple | list) for item in value):
            return [_param_value(item) for item in value]
        return yaml.safe_load(_text(value)) if value else []
    return value


def _params(node):
    params = {}
    for entry in node._Node__parameters or []:
        if isinstance(entry, dict):
            params.update({_text(key): _param_value(value) for key, value in entry.items()})
    return params


def _deferred_actions(actions):
    from launch.actions import RegisterEventHandler

    deferred = []
    for action in actions:
        if isinstance(action, RegisterEventHandler):
            handler = getattr(action.event_handler, "_OnActionEventBase__on_event", None)
            if handler is not None:
                deferred.extend(handler(type("E", (), {"returncode": 0})(), None) or [])
    return deferred


def _mount_translation():
    mount = yaml.safe_load(MOUNT.read_text(encoding="utf-8"))
    return mount["translation_m"], [math.radians(angle) for angle in mount["rpy_deg"]]


# ---------------------------------------------------------------------------
# D1: profile selection
# ---------------------------------------------------------------------------


def test_shipped_lidar_profile_declares_sdk_and_gazebo():
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    assert profile["sim_platforms"] == ["sdk", "gazebo", "mujoco"]
    assert profile["sim_platform"] == "sdk"
    assert profile["use_sim"] is False


def test_profile_mount_defaults_equal_the_mount_file():
    """D13: both pose keys default to the mount relative to base_link."""
    from robot_runtime.path_subst import resolve_path
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    mount_path = Path(resolve_path(profile["mid360_mount_file"]))
    assert mount_path.is_file(), f"the profile's mount reference must resolve: {mount_path}"
    translation, rpy = _mount_translation()

    lidar = next(periph for periph in profile["peripherals"] if periph.get("type") == "lidar")
    expected = {
        "parent_frame": "base_link",
        "x": translation[0],
        "y": translation[1],
        "z": translation[2],
        "roll": rpy[0],
        "pitch": rpy[1],
        "yaw": rpy[2],
    }
    assert lidar["transform"] == expected
    assert lidar["simulation"]["gazebo"]["transform"] == expected
    # lekiwi-mujoco-runtime-platform D1: the same mount drives the MuJoCo
    # attachment pose (one helper, both platforms)
    assert lidar["simulation"]["mujoco"]["transform"] == expected


def test_nav_grasp_still_rejects_gazebo_before_any_action():
    import pytest

    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match="sim_platform 'gazebo' is not supported") as excinfo:
        _render(GRASP_PROFILE, {"use_sim": "true", "sim_platform": "gazebo"})
    message = str(excinfo.value)
    assert "sdk" in message and "lekiwi_nav_grasp" in message


# ---------------------------------------------------------------------------
# D2: one description across execution platforms
# ---------------------------------------------------------------------------


def test_physical_and_sdk_renders_carry_no_simulation_content():
    from robot_runtime.launch_support import render_robot_description
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    for simulated in (False, True):
        urdf = render_robot_description(profile, LIDAR_PROFILE, simulated)
        assert "gz_ros2_control-system" not in urdf
        assert "<mu1>" not in urdf and "<fdir1>" not in urdf
        assert "<cylinder" not in urdf
        assert '<mesh filename="package://lekiwi_description/meshes/omni_wheel.stl"/>' in urdf


def test_gazebo_render_carries_the_plugin_and_the_wheel_treatment():
    from robot_runtime.launch_support import render_gazebo_description
    from robot_runtime.peripherals import merge_peripherals
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    urdf = render_gazebo_description(
        {**profile, "peripherals": merge_peripherals(profile.get("peripherals"), [])}, LIDAR_PROFILE
    )
    # the simulator control plugin with the profile's controller parameters
    assert "gz_ros2_control-system" in urdf
    assert "<parameters>" in urdf and "lekiwi_base_controllers.yaml" in urdf
    # cylinder collisions and directional friction for exactly the three wheels
    assert urdf.count('<cylinder length="0.025" radius="0.051"/>') == 3
    for wheel in ("left_wheel_link", "back_wheel_link", "right_wheel_link"):
        assert f'<gazebo reference="{wheel}">' in urdf
    assert urdf.count("<fdir1>0 1 0</fdir1>") == 3
    # the wheel joint axes flip under the gz gate: the SDK's mount angles
    # (150/-90/30 deg) are each wheel's positive-spin drive direction,
    # antiparallel to the URDF joints' +z rotation as simulated — the flip
    # aligns the simulated joint convention with the physical motors' so the
    # production base_node kinematics drive and integrate identically (D3/D4)
    assert urdf.count('<axis xyz="0 0 -1"/>') == 3
    # joint names and the root frame are unchanged; meshes resolve absolutely
    assert '<joint name="7" type="continuous">' in urdf
    assert '<link name="base_link">' in urdf
    assert "package://" not in urdf


def test_mujoco_plugin_selection_keeps_the_gazebo_treatment_out():
    """The gz content is gated by the plugin, not by use_sim alone (D2)."""
    from robot_runtime.launch_support import render_xacro
    from robot_runtime.path_subst import resolve_path
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    urdf = render_xacro(
        resolve_path(profile["description"]["xacro"]),
        {
            "use_sim": "true",
            "simulated": "false",
            "sim_plugin": "mujoco_ros2_control/MujocoSystemInterface",
            "base_only": "true",
        },
    )
    assert "gz_ros2_control-system" not in urdf
    assert "<mu1>" not in urdf
    assert "<cylinder" not in urdf


# ---------------------------------------------------------------------------
# D7: launch plumbing through the real entry
# ---------------------------------------------------------------------------


def test_launch_declares_the_gazebo_arguments():
    from launch.actions import DeclareLaunchArgument

    description = _launch_module().generate_launch_description()
    arguments = [action for action in description.entities if isinstance(action, DeclareLaunchArgument)]
    names = {_text(action._DeclareLaunchArgument__name) for action in arguments}
    assert {"gazebo_world", "spawn_x", "spawn_y", "spawn_z", "spawn_yaw"} <= names
    defaults = {
        _text(action._DeclareLaunchArgument__name): _text(action._DeclareLaunchArgument__default_value)
        for action in arguments
        if action._DeclareLaunchArgument__default_value
    }
    # the default spawn z lifts the wheels just above the ground plane (D7)
    assert float(defaults["spawn_z"]) == 0.05


def test_gazebo_launch_threads_world_pose_and_sim_time():
    actions = _render(
        LIDAR_PROFILE,
        {
            "use_sim": "true",
            "sim_platform": "gazebo",
            "gazebo_world": "demo",
            "spawn_x": "0.5",
            "spawn_y": "-0.25",
            "spawn_z": "0.05",
            "spawn_yaw": "1.25",
        },
    )
    nodes = [a for a in actions if isinstance(a, Node)]
    executables = {node._Node__node_executable for node in nodes}
    # the gazebo embodiment: no ros2_control_node, entity creation deferred
    assert "ros2_control_node" not in executables
    create = next(node for node in _deferred_actions(actions) if node._Node__node_executable == "create")
    args = create._Node__arguments
    for flag, value in (("-x", "0.5"), ("-y", "-0.25"), ("-z", "0.05"), ("-Y", "1.25"), ("-world", "demo")):
        assert value == args[args.index(flag) + 1], f"create {flag} must receive {value}"
    # the facade and the base node run on simulation time
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    assert _params(facade)["use_sim_time"] is True
    assert _params(facade)["sim_platform"] == "gazebo"
    base = next(node for node in nodes if node._Node__node_executable == "base_node")
    assert _params(base)["use_sim_time"] is True
    # the sensor bridge covers the raw gz cloud (the finite filter serves
    # the advertised /livox/lidar endpoint) and the raw integrated IMU (the
    # settle filter serves the advertised /livox/imu endpoint); the lidar
    # frame's static TF is part of the embodiment
    bridge = next(node for node in nodes if node._Node__node_executable == "bridge_node")
    entries = yaml.safe_load(Path(_params(bridge)["config_file"]).read_text(encoding="utf-8"))
    assert {entry["ros_topic_name"] for entry in entries} == {"/mid360_lidar/raw", "/mid360_imu/raw"}
    filter_node = next(node for node in nodes if node._Node__node_executable == "finite_cloud_filter")
    assert _params(filter_node)["output_topic"] == "/livox/lidar"
    tf = next(node for node in nodes if node._Node__node_executable == "static_transform_publisher")
    tf_args = tf._Node__arguments
    assert tf_args[tf_args.index("--frame-id") + 1] == "base_link"
    assert tf_args[tf_args.index("--child-frame-id") + 1] == "livox_frame"


def test_gazebo_launch_composes_fast_lio_and_the_scan_converter_on_sim_time(tmp_path):
    fragment = tmp_path / "fragment.yaml"
    fragment.write_text(
        yaml.safe_dump({"fast_lio": {"enabled": True}}), encoding="utf-8"
    )  # sim_params_file comes from the profile
    actions = _render(LIDAR_PROFILE, {"use_sim": "true", "sim_platform": "gazebo", "peripherals": str(fragment)})
    nodes = [a for a in actions if isinstance(a, Node)]
    by_executable = {}
    for node in nodes:
        by_executable.setdefault(node._Node__node_executable, []).append(node)
    assert len(by_executable["fastlio_mapping"]) == 1
    assert _params(by_executable["fastlio_mapping"][0])["use_sim_time"] is True
    bridge = by_executable["fast_lio_odom_bridge"][0]
    assert _params(bridge)["use_sim_time"] is True
    # exactly one odom -> base_link publisher in simulation (D4)
    assert _params(bridge)["publish_tf"] is False
    base = by_executable["base_node"][0]
    assert _params(base)["publish_tf"] is True
    # the derived scan converter runs on sim time next to its source (D5)
    converter = by_executable["pointcloud_to_laserscan_node"][0]
    assert _params(converter)["use_sim_time"] is True


def test_sdk_launch_keeps_synthetic_perception_as_the_single_scan_producer(tmp_path):
    fragment = tmp_path / "fragment.yaml"
    fragment.write_text(yaml.safe_dump({"fast_lio": {"enabled": True}}), encoding="utf-8")
    actions = _render(LIDAR_PROFILE, {"use_sim": "true", "sim_platform": "sdk", "peripherals": str(fragment)})
    nodes = [a for a in actions if isinstance(a, Node)]
    executables = {node._Node__node_executable for node in nodes}
    # no lidar odometry, no converter: the synthetic publishers own the surface
    assert "fastlio_mapping" not in executables
    assert "pointcloud_to_laserscan_node" not in executables
    assert "livox_ros_driver2_node" not in executables
    synthetic = next(node for node in nodes if node._Node__node_executable == "synthetic_perception")
    import json

    description = json.loads(_params(synthetic)["description_json"])
    assert description["interfaces"]["lidar.mid360.scan"]["endpoint"] == "/scan"
    assert description["interfaces"]["lidar.mid360.points"]["message_type"] == "livox_ros_driver2/msg/CustomMsg"
    # the base node stays on wall time on the sdk platform
    base = next(node for node in nodes if node._Node__node_executable == "base_node")
    assert _params(base)["use_sim_time"] is False


def test_hardware_launch_composes_the_physical_drivers(tmp_path):
    actions = _render(LIDAR_PROFILE, {"use_sim": "false"})
    nodes = [a for a in actions if isinstance(a, Node)]
    executables = {node._Node__node_executable for node in nodes}
    assert "ros2_control_node" in executables
    assert "livox_ros_driver2_node" in executables
    assert "pointcloud_to_laserscan_node" in executables
    base = next(node for node in nodes if node._Node__node_executable == "base_node")
    assert _params(base)["use_sim_time"] is False
    assert _params(base)["publish_tf"] is True


# ---------------------------------------------------------------------------
# D13: one MID-360 mount through the composed provider path
# ---------------------------------------------------------------------------


def test_rotated_mount_drives_sensor_pose_public_tf_and_extrinsics(tmp_path):
    """A non-default rotated mount (as the application fragment forwards it):
    the simulator sensor sits at the mount, livox_frame has exactly one
    parent (body), and FAST-LIO's extrinsic is the mount's inverse."""
    from robot_runtime.launch_support import render_gazebo_description
    from robot_runtime.peripherals import load_peripherals_file, merge_peripherals
    from robot_runtime.profile import load_profile

    translation = [-0.12, 0.05, 0.28]
    rpy = [0.2, -0.15, 1.9]  # non-default rotation
    fragment = tmp_path / "fragment.yaml"
    fragment.write_text(
        yaml.safe_dump(
            {
                "peripherals": [
                    {
                        "type": "lidar",
                        "name": "mid360",
                        "frame_id": "livox_frame",
                        "transform": {"parent_frame": "body", "x": 0.0, "y": 0.0, "z": 0.0},
                        "simulation": {
                            "gazebo": {
                                "transform": {
                                    "parent_frame": "base_link",
                                    "x": translation[0],
                                    "y": translation[1],
                                    "z": translation[2],
                                    "roll": rpy[0],
                                    "pitch": rpy[1],
                                    "yaw": rpy[2],
                                }
                            }
                        },
                    }
                ],
                "fast_lio": {
                    "enabled": True,
                    # the inverse of the mount above (sensor_mount semantics)
                    "body_to_base_translation": [0.05, 0.12, -0.28],
                    "body_to_base_rotation": [-0.0647, 0.0486, 0.5856, 0.8053],
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_profile(LIDAR_PROFILE)
    merged = merge_peripherals(profile.get("peripherals"), load_peripherals_file(str(fragment)).get("peripherals"))

    # the rendered model: both sensors at the mount, stamped livox_frame,
    # and no link for the sensor frame (one parent, from the static TF)
    urdf = render_gazebo_description({**profile, "peripherals": merged}, LIDAR_PROFILE)
    pose = f"<pose>{translation[0]} {translation[1]} {translation[2]} {rpy[0]} {rpy[1]} {rpy[2]}</pose>"
    assert urdf.count(pose) == 2, "the lidar and its IMU both sit at the mount"
    assert urdf.count("<gz_frame_id>livox_frame</gz_frame_id>") == 2
    assert '<link name="livox_frame">' not in urdf

    # through the real launch entry: the static TF parents livox_frame under
    # body, and the FAST-LIO bridge receives the mount's inverse extrinsic
    actions = _render(
        LIDAR_PROFILE,
        {"use_sim": "true", "sim_platform": "gazebo", "peripherals": str(fragment), "gazebo_world": "demo"},
    )
    nodes = [a for a in actions if isinstance(a, Node)]
    tf = [node for node in nodes if node._Node__node_executable == "static_transform_publisher"]
    assert len(tf) == 1, "exactly one parent for livox_frame"
    tf_args = tf[0]._Node__arguments
    assert tf_args[tf_args.index("--frame-id") + 1] == "body"
    assert tf_args[tf_args.index("--child-frame-id") + 1] == "livox_frame"
    bridge = next(node for node in nodes if node._Node__node_executable == "fast_lio_odom_bridge")
    params = _params(bridge)
    assert params["body_to_base_translation"] == [0.05, 0.12, -0.28]
    assert params["body_to_base_rotation"] == [-0.0647, 0.0486, 0.5856, 0.8053]
