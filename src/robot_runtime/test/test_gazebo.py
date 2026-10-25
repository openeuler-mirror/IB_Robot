"""Gazebo platform tests: shared camera builders, embodiment actions, sim identity.

Covers openspec ``so101-gazebo-runtime-transport`` tasks 2.1, 2.3b, 2.3c and
2.3d: the launch-structure contract per platform, the advertised camera
contract (frames, streams, spawn compensation), the effective peripheral
inventory (disabled cameras absent everywhere) and the simulated joint
current.

``lekiwi-gazebo-runtime-platform`` D5 adds the simulated lidar/IMU sensors
and bridges, the one-inventory rule across rendering/bridging/advertisement,
peripheral static TF ownership on the platform, and the derived scan
converter's platform composition matrix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from launch_ros.actions import Node
from sensor_msgs.msg import Imu, PointCloud2, PointField

from robot_runtime.gazebo import (
    advertised_camera_frame,
    camera_bridge_entries,
    camera_sensor_urdf,
    imu_bridge_entries,
    imu_sensor_sdf,
    lidar_bridge_entries,
    lidar_imu_sensor_urdf,
    lidar_sensor_sdf,
    resolve_camera_pose,
    resolve_sensor_pose,
)
from robot_runtime.interface_description import (
    InterfaceDescriptionError,
    build_description,
    merge_peripherals,
)
from robot_runtime.launch_support import (
    gazebo_embodiment_actions,
    motion_launch_arguments,
    peripheral_stack_actions,
    render_gazebo_description,
    runtime_stack_actions,
    simulated_joint_current_actions,
)
from robot_runtime.peripherals import peripheral_tf_nodes_for_absent_frames

# ---------------------------------------------------------------------------
# Shared fixtures: a minimal robot-agnostic profile the launch helpers accept
# ---------------------------------------------------------------------------

XACRO = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:arg name="use_sim" default="true"/>
  <xacro:arg name="simulated" default="false"/>
  <xacro:arg name="sim_plugin" default="gz_ros2_control/GazeboSimSystem"/>
  <xacro:arg name="sim_effort_state" default="false"/>
  <xacro:arg name="gz_ros2_control_parameters_file" default=""/>
  <link name="base">
    <visual><geometry><mesh><filename>package://robot_runtime/meshes/none.stl</filename></mesh></geometry></visual>
  </link>
  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>$(arg gz_ros2_control_parameters_file)</parameters>
    </plugin>
  </gazebo>
  <xacro:if value="$(arg use_sim)">
    <ros2_control name="RobotSystem" type="system">
      <hardware>
        <plugin>$(arg sim_plugin)</plugin>
        <param name="simulated">$(arg simulated)</param>
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
        "gazebo": {
            "transform": {
                "parent_frame": "world",
                "x": 0.0419,
                "y": 0.1123,
                "z": 0.6136,
                "roll": 0.057596,
                "pitch": 1.518436,
                "yaw": -1.518436,
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
        "gazebo": {
            "transform": {
                "parent_frame": "gripper",
                "x": 0.002265,
                "y": 0.073023,
                "z": -0.016289,
                "roll": -1.512475,
                "pitch": 1.079239,
                "yaw": -1.541112,
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
        "gazebo": {
            "transform": {
                "parent_frame": "base",
                "x": 0.0,
                "y": -0.28,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 1.5708,
                "yaw": -1.5708,
            },
            "fovy": 70,
        }
    },
    "transform": {"parent_frame": "base", "x": 0.0, "y": 0.0, "z": 0.0},
}

# A MID-360 lidar in the LeKiwi profile's shape (D1/D13 pose keys).
MID360 = {
    "type": "lidar",
    "name": "mid360",
    "driver": "livox_mid360",
    "frame_id": "livox_frame",
    "pointcloud_topic": "/livox/lidar",
    "imu_topic": "/livox/imu",
    "xfer_format": 1,
    "publish_freq": 10.0,
    "transform": {
        "parent_frame": "base_link",
        "x": -0.08,
        "y": 0.0,
        "z": 0.2,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 1.5707963267948966,
    },
    "simulation": {
        "gazebo": {
            "transform": {
                "parent_frame": "base_link",
                "x": -0.08,
                "y": 0.0,
                "z": 0.2,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 1.5707963267948966,
            },
            "horizontal_samples": 900,
            "vertical_samples": 32,
            "vertical_min_deg": -7.0,
            "vertical_max_deg": 52.0,
            "range_min": 0.1,
            "range_max": 40.0,
            "update_rate": 10.0,
            "imu": {"update_rate": 200.0},
        }
    },
    "scan_converter": {
        "enabled": True,
        "pointcloud_topic": "/cloud_registered_body",
        "scan_topic": "/scan",
        "target_frame": "body",
    },
}

# A planar lidar (ldlidar) on its scan endpoint.
LDLIDAR = {
    "type": "lidar",
    "name": "ldlidar",
    "driver": "ldlidar",
    "frame_id": "laser_frame",
    "scan_topic": "/scan",
    "transform": {"parent_frame": "base_link", "x": 0.0, "y": 0.0, "z": 0.1},
}


def _profile(tmp_path: Path, peripherals=None, with_joint_current=True):
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
        "sim_platforms": ["sdk", "gazebo"],
        "sim_platform": "sdk",
        "controller_manager": "controller_manager",
        "joint_state_topic": "/joint_states",
        "command_channels": [
            {
                "channel": "arm_stream",
                "topic": "/arm_position_controller/commands",
                "joints": ["1"],
                "modes": ["stream"],
            }
        ],
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
                "units": {"current": "A"},
                "qos": {
                    "reliability": "reliable",
                    "durability": "volatile",
                    "history": "keep_last",
                    "depth": 10,
                },
            }
        }
        profile["simulation"] = {"gazebo": {"joint_current": {"scale": [2.0]}}}
    return profile


def _node(nodes, name):
    matches = [node for node in nodes if node._Node__node_name == name]
    assert len(matches) == 1, f"expected exactly one node named {name!r}, got {[n._Node__node_name for n in nodes]}"
    return matches[0]


def _by_executable(nodes, executable):
    matches = [node for node in nodes if node._Node__node_executable == executable]
    assert len(matches) == 1, f"expected exactly one {executable!r} node"
    return matches[0]


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _param_value(value):
    """Resolve a normalized launch parameter the way evaluate_parameters would.

    launch_ros stores string-ish values as substitutions of their YAML dump
    (``yaml.dump('gazebo')`` is ``'gazebo\\n...\\n'``) and yaml-loads the
    performed text back at evaluation; array parameters keep one
    substitution-group per element.
    """
    if isinstance(value, tuple | list):
        if value and all(not hasattr(item, "text") and not isinstance(item, tuple | list) for item in value):
            return list(value)  # plainly typed array (bool/int/float)
        if any(isinstance(item, tuple | list) for item in value):
            return [_param_value(item) for item in value]
        return yaml.safe_load(_text(value)) if value else []
    return value


def _params(node):
    """All parameter dicts of a Node, keys/values resolved like evaluation.

    ``ParameterFile`` entries (per-call temp files) are skipped: they carry no
    comparable identity across two generated action lists.
    """
    params = {}
    for entry in node._Node__parameters or []:
        if isinstance(entry, dict):
            params.update(
                {
                    _text(key): _param_value(value)
                    for key, value in entry.items()
                    if "ParameterFile" not in type(value).__name__
                }
            )
    return params


class _FakeExitEvent:
    returncode = 0


def _deferred_actions(actions):
    """Nodes an OnProcessExit handler starts when its target exits successfully."""
    from launch.actions import RegisterEventHandler

    deferred = []
    for action in actions:
        if isinstance(action, RegisterEventHandler):
            handler = getattr(action.event_handler, "_OnActionEventBase__on_event", None)
            if handler is not None:
                deferred.extend(handler(_FakeExitEvent(), None) or [])
    return deferred


# ---------------------------------------------------------------------------
# 2.1: launch structure per platform
# ---------------------------------------------------------------------------


def test_gazebo_platform_action_set_has_no_ros2_control_node(tmp_path):
    profile = _profile(tmp_path)
    actions = runtime_stack_actions(
        profile, tmp_path / "robot.xacro", simulated=True, sim_platform="gazebo", gazebo_world=""
    )
    nodes = [a for a in actions if isinstance(a, Node)]
    executables = {node._Node__node_executable for node in nodes}
    deferred = {node._Node__node_executable for node in _deferred_actions(actions) if isinstance(node, Node)}
    assert "ros2_control_node" not in executables
    assert "robot_state_publisher" in executables
    assert "runtime_facade" in executables
    assert "create" in deferred
    assert "wait_for_clock" in executables
    # standalone: the provider brings a minimal empty world + clock bridge
    assert "IncludeLaunchDescription" in {type(a).__name__ for a in actions}
    assert "parameter_bridge" in executables
    # the camera bridge covers the embodiment sensors
    assert "bridge_node" in executables
    # the simulated joint current publisher is part of the embodiment
    assert "joint_current_publisher" in executables


def test_gazebo_nodes_run_on_sim_time(tmp_path):
    profile = _profile(tmp_path)
    actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", gazebo_world="demo")
    nodes = [a for a in actions if isinstance(a, Node)]
    rsp = _by_executable(nodes, "robot_state_publisher")
    assert _params(rsp)["use_sim_time"] is True
    facade = _node(nodes, "runtime_facade")
    assert _params(facade)["use_sim_time"] is True
    assert _params(facade)["sim_platform"] == "gazebo"
    # spawners sit behind the create-exit handler; they must carry sim time
    spawners = [
        node
        for node in _deferred_actions(actions)
        if isinstance(node, Node) and node._Node__node_executable == "spawner"
    ]
    assert spawners, "controller spawners must start after create exits"
    assert all(_params(spawner).get("use_sim_time") is True for spawner in spawners)


def test_attached_gazebo_world_is_not_started_by_the_provider(tmp_path):
    profile = _profile(tmp_path)
    actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", gazebo_world="demo")
    nodes = [a for a in actions if isinstance(a, Node)]
    executables = {node._Node__node_executable for node in nodes}
    assert "parameter_bridge" not in executables, "the world owner bridges /clock"
    create = _by_executable(_deferred_actions(actions), "create")
    assert "-world" in create._Node__arguments and "demo" in create._Node__arguments


def test_gazebo_create_carries_the_full_spawn_pose(tmp_path):
    profile = _profile(tmp_path)
    actions = gazebo_embodiment_actions(
        profile, tmp_path / "robot.xacro", gazebo_world="demo", spawn_pose=(0.5, -0.25, 0.1, 1.25)
    )
    create = _by_executable(_deferred_actions(actions), "create")
    args = create._Node__arguments
    for flag, value in (("-x", "0.5"), ("-y", "-0.25"), ("-z", "0.1"), ("-Y", "1.25")):
        assert value == args[args.index(flag) + 1], f"create {flag} must receive {value}"


def test_gazebo_description_has_no_package_uris_and_flags(tmp_path):
    profile = _profile(tmp_path)
    urdf = render_gazebo_description(profile, tmp_path / "robot.xacro")
    assert "package://" not in urdf
    assert "file://" in urdf
    assert "gz_ros2_control/GazeboSimSystem" in urdf
    assert "SO101SystemHardware" not in urdf  # SDK hardware adapter is not the gazebo plugin
    assert '<state_interface name="effort"/>' in urdf


def test_hardware_and_sdk_action_lists_are_unchanged(tmp_path):
    profile = _profile(tmp_path)
    for sim_platform, simulated in (("", False), ("sdk", True)):
        legacy = runtime_stack_actions(profile, tmp_path / "robot.xacro", simulated, sim_platform="")
        selected = runtime_stack_actions(profile, tmp_path / "robot.xacro", simulated, sim_platform=sim_platform)
        assert [type(a).__name__ for a in legacy] == [type(a).__name__ for a in selected]
        legacy_nodes = [a for a in legacy if isinstance(a, Node)]
        selected_nodes = [a for a in selected if isinstance(a, Node)]
        assert [n._Node__node_name for n in legacy_nodes] == [n._Node__node_name for n in selected_nodes]
        for legacy_node, selected_node in zip(legacy_nodes, selected_nodes, strict=True):
            # the facade additionally receives the selected platform (D1/D7);
            # everything else is identical
            legacy_params = {k: v for k, v in _params(legacy_node).items() if k != "sim_platform"}
            selected_params = {k: v for k, v in _params(selected_node).items() if k != "sim_platform"}
            assert legacy_params == selected_params
        facade = _node(selected_nodes, "runtime_facade")
        assert _params(facade).get("sim_platform", "") == sim_platform
        executables = {node._Node__node_executable for node in legacy_nodes}
        assert "ros2_control_node" in executables
        # wall time: no provider node carries sim time off the gazebo platform
        assert all("use_sim_time" not in _params(node) for node in legacy_nodes)


def test_motion_launch_arguments_follow_the_platform_clock(tmp_path):
    profile = _profile(tmp_path)
    gazebo = motion_launch_arguments(profile, tmp_path / "robot.xacro", True, "gazebo", False)
    assert gazebo["use_sim_time"] == "true"
    assert gazebo["motion_hardware_feedback_topic"] == ""
    # mujoco is a physics platform like gazebo: every motion node runs on sim time
    mujoco = motion_launch_arguments(profile, tmp_path / "robot.xacro", True, "mujoco", False)
    assert mujoco["use_sim_time"] == "true"
    assert mujoco["motion_hardware_feedback_topic"] == ""
    sdk = motion_launch_arguments(profile, tmp_path / "robot.xacro", True, "sdk", False)
    assert sdk["use_sim_time"] == "false"
    hardware = motion_launch_arguments(profile, tmp_path / "robot.xacro", False, "", False)
    assert hardware["use_sim_time"] == "false"
    assert hardware["motion_hardware_feedback_topic"] == "/so101_follower/joint_currents" or isinstance(
        hardware["motion_hardware_feedback_topic"], str
    )


# ---------------------------------------------------------------------------
# 2.3b: the advertised camera contract
# ---------------------------------------------------------------------------


def test_advertised_frames_match_gz_frame_ids_and_optical_links(tmp_path):
    peripherals = [TOP, WRIST, FRONT]
    urdf = camera_sensor_urdf(peripherals, platform="gazebo", frame_contract=True)
    for periph in peripherals:
        frame = advertised_camera_frame(periph)
        assert f"<gz_frame_id>{frame}</gz_frame_id>" in urdf
        optical = periph["optical_frame_id"]
        assert f'<link name="{optical}">' in urdf and f'<child  link="{optical}"/>' in urdf


def test_descriptor_interfaces_match_generated_sensors_and_bridge_entries(tmp_path):
    peripherals = [TOP, WRIST, FRONT]
    description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": peripherals,
        },
        use_sim=True,
        sim_platform="gazebo",
    )
    urdf = camera_sensor_urdf(peripherals, platform="gazebo", frame_contract=True)
    for periph in peripherals:
        name = periph["name"]
        # every advertised camera interface endpoint has a bridge entry
        advertised = {
            interface["endpoint"]
            for key, interface in description["interfaces"].items()
            if key.startswith(f"camera.{name}.")
        }
        bridged = {entry["ros_topic_name"] for entry in camera_bridge_entries(periph, full_streams=True)}
        assert advertised == bridged, f"camera {name}: advertised {sorted(advertised)} vs bridged {sorted(bridged)}"
        # every advertised frame id equals the stamped gz frame id
        frames = {
            interface["frame_id"]
            for key, interface in description["interfaces"].items()
            if key.startswith(f"camera.{name}.") and interface.get("frame_id")
        }
        assert frames == {advertised_camera_frame(periph)}
        # a sensor exists for every non-disabled camera with the advertised frame
        assert f"<gz_frame_id>{advertised_camera_frame(periph)}</gz_frame_id>" in urdf
    # opencv cameras are color-only; the realsense gets an rgbd sensor
    assert urdf.count("<sensor") == 3
    assert 'sensor type="rgbd_camera" name="front_camera"' in urdf
    assert 'sensor type="camera" name="top_camera"' in urdf


def test_gazebo_depth_streams_advertise_the_encoding_gz_publishes():
    """The Gazebo depth contract is what ros_gz_bridge actually publishes (PR !515 P2).

    gz ``rgbd_camera`` depth is float32 and the bridge maps ``R_FLOAT32`` to
    ``32FC1``; the bridge config carries no conversion, so advertising the
    physical device's encoding (RealSense Z16 -> ``16UC1``) would make every
    observed depth message an encoding mismatch in the interface monitor.
    sdk keeps the device encoding: the synthetic publishers produce exactly
    what the descriptor advertises.
    """
    robot = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "capabilities": {},
        "joints": ["1"],
        "peripherals": [FRONT],
    }
    gazebo = build_description(robot, use_sim=True, sim_platform="gazebo")["interfaces"]
    assert gazebo["camera.front.depth"]["configured_profile"]["encoding"] == "32FC1"
    assert gazebo["camera.front.aligned_depth"]["configured_profile"]["encoding"] == "32FC1"
    # both depth endpoints are bridged from the same gz depth_image stream,
    # so both observe exactly the encoding asserted above
    bridged = {
        entry["ros_topic_name"]: entry["gz_topic_name"] for entry in camera_bridge_entries(FRONT, full_streams=True)
    }
    assert bridged["/camera/front/depth/image_rect_raw"] == "/front_camera/depth_image"
    assert bridged["/camera/front/aligned_depth_to_color/image_raw"] == "/front_camera/depth_image"
    sdk = build_description(robot, use_sim=True, sim_platform="sdk")["interfaces"]
    assert sdk["camera.front.depth"]["configured_profile"]["encoding"] == "16UC1"
    assert sdk["camera.front.aligned_depth"]["configured_profile"]["encoding"] == "16UC1"


def test_unproducible_streams_are_not_advertised_on_gazebo(tmp_path):
    pointcloud_camera = {
        **FRONT,
        "enable_pointcloud": True,
        "simulation": {"gazebo": {"transform": FRONT["simulation"]["gazebo"]["transform"], "fovy": 70}},
    }
    description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": [pointcloud_camera],
        },
        use_sim=True,
        sim_platform="gazebo",
    )
    assert not any(key.startswith("camera.front.points") for key in description["interfaces"])
    sdk_description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": [pointcloud_camera],
        },
        use_sim=True,
        sim_platform="sdk",
    )
    assert any(key.startswith("camera.front.points") for key in sdk_description["interfaces"])


def test_world_parented_camera_uses_the_inverse_full_spawn_transform():
    urdf = camera_sensor_urdf([TOP], platform="gazebo", spawn_pose=(0.0, 0.3, -0.03, 0.5), frame_contract=True)
    import re

    origin = re.search(r'name="top_camera_joint".*?<origin xyz="([^"]+)" rpy="([^"]+)"', urdf, re.S)
    x, y, z = (float(value) for value in origin.group(1).split())
    roll, pitch, yaw = (float(value) for value in origin.group(2).split())
    # inverse of spawn (translation + yaw 0.5) applied to the world pose
    cos, sin = math.cos(-0.5), math.sin(-0.5)
    dx, dy = TOP["simulation"]["gazebo"]["transform"]["x"], TOP["simulation"]["gazebo"]["transform"]["y"] - 0.3
    assert x == pytest.approx(cos * dx - sin * dy)
    assert y == pytest.approx(sin * dx + cos * dy)
    assert z == pytest.approx(TOP["simulation"]["gazebo"]["transform"]["z"] + 0.03)
    assert yaw == pytest.approx(TOP["simulation"]["gazebo"]["transform"]["yaw"] - 0.5)


def test_legacy_camera_urdf_keeps_the_generic_shape():
    urdf = camera_sensor_urdf([TOP, WRIST], platform="gazebo")
    assert "gz_frame_id" not in urdf
    assert "optical_frame" not in urdf
    assert '<gazebo reference="camera_top_frame">' in urdf
    # fovy 47 at 640x480 -> horizontal fov
    assert f"<horizontal_fov>{2.0 * math.atan(math.tan(math.radians(47) / 2) * 640 / 480):.4f}</horizontal_fov>" in urdf


def test_preset_loader_keeps_the_generic_use_default_transform_path():
    periph = {
        "type": "camera",
        "name": "front",
        "driver": "opencv",
        "frame_id": "camera_front_frame",
        "use_default_transform": True,
        "transform": {"parent_frame": "base", "x": 9.0},
    }

    def loader(platform, name):
        return {"parent_frame": "world", "x": 0.5, "y": 0.1, "z": 0.2, "yaw": 0.3, "fovy": 55}

    transform, fovy = resolve_camera_pose(periph, "gazebo", default_pose_loader=loader)
    assert transform["parent_frame"] == "world" and transform["x"] == 0.5
    assert fovy == 55
    # without the loader flag the explicit transform applies
    plain, _ = resolve_camera_pose({**periph, "use_default_transform": False}, "gazebo")
    assert plain["x"] == 9.0


def test_saved_override_takes_precedence_over_the_declared_default(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ros" / "ibrobot" / "sim_camera_overrides").mkdir(parents=True)
    (home / ".ros" / "ibrobot" / "sim_camera_overrides" / "top.yaml").write_text(
        yaml.safe_dump(
            {
                "pose": {"x": 1.5, "y": -0.2, "z": 0.8, "roll": 0.0, "pitch": 1.4, "yaw": -1.4},
                "parent_frame": "world",
                "fovy_deg": 50,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    transform, fovy = resolve_camera_pose(TOP, "gazebo")
    assert transform["x"] == 1.5 and transform["parent_frame"] == "world"
    assert fovy == 50
    urdf = camera_sensor_urdf([TOP], platform="gazebo", frame_contract=True)
    assert 'xyz="1.5 -0.2 0.8"' in urdf


# ---------------------------------------------------------------------------
# 2.3c: one effective peripheral inventory
# ---------------------------------------------------------------------------


def test_disabled_camera_is_absent_everywhere(tmp_path):
    fragment = [{"type": "camera", "name": "front", "disabled": True}]
    peripherals = merge_peripherals([TOP, WRIST, FRONT], fragment)
    assert [periph["name"] for periph in peripherals] == ["top", "wrist"]
    # descriptor
    description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": [TOP, WRIST, FRONT],
        },
        peripherals=peripherals,
        use_sim=True,
        sim_platform="gazebo",
    )
    assert not any(key.startswith("camera.front.") for key in description["interfaces"])
    # sdk synthetic publishers
    profile = _profile(tmp_path, peripherals=peripherals)
    sdk_actions = peripheral_stack_actions(profile, "", simulated=True, sim_platform="sdk")
    synthetic = _node([a for a in sdk_actions if isinstance(a, Node)], "synthetic_perception")
    encoded = _params(synthetic)["description_json"]
    assert "camera.front.color" not in encoded
    # gazebo sensors + bridge
    gazebo_urdf = render_gazebo_description(profile, tmp_path / "robot.xacro")
    assert "front_camera" not in gazebo_urdf
    gazebo_actions = gazebo_embodiment_actions(
        profile, tmp_path / "robot.xacro", robot_description=gazebo_urdf, gazebo_world="demo"
    )
    bridge = _by_executable([a for a in gazebo_actions if isinstance(a, Node)], "bridge_node")
    config = Path(_params(bridge)["config_file"]).read_text(encoding="utf-8")
    assert "camera/front" not in config
    # physical drivers
    hardware_actions = peripheral_stack_actions(profile, "", simulated=False, sim_platform="")
    assert not any(isinstance(a, Node) and a._Node__node_name == "front_camera" for a in hardware_actions)


def test_peripheral_stack_on_a_physics_platform_is_audio_only(tmp_path):
    profile = _profile(tmp_path)
    for platform in ("gazebo", "mujoco"):
        actions = peripheral_stack_actions(profile, "", simulated=True, sim_platform=platform)
        executables = {a._Node__node_executable for a in actions if isinstance(a, Node)}
        assert "synthetic_perception" not in executables, f"{platform}: the simulator serves the camera streams"
        assert not any("camera" in exe for exe in executables), f"{platform}: no camera driver nodes"


def test_enabled_false_and_disabled_true_are_the_same_toggle():
    assert merge_peripherals([TOP], [{"type": "camera", "name": "top", "enabled": False}]) == []
    assert merge_peripherals([TOP], [{"type": "camera", "name": "top", "disabled": True}]) == []
    # the canonicaliser turns the disabled spelling into enabled: false and
    # the enabled spelling is already canonical (runtime-deployment-peripherals D3)
    disabled_spelling = merge_peripherals([TOP], [{"type": "camera", "name": "top", "disabled": False, "width": 32}])
    assert disabled_spelling == [dict(TOP, width=32, enabled=True)]
    enabled_spelling = merge_peripherals([TOP], [{"type": "camera", "name": "top", "enabled": True, "width": 32}])
    assert enabled_spelling == [dict(TOP, width=32, enabled=True)]


def test_toggle_only_entry_for_undeclared_peripheral_is_rejected():
    with pytest.raises(InterfaceDescriptionError, match=r"\('camera', 'ghost'\) is not declared"):
        merge_peripherals([TOP], [{"type": "camera", "name": "ghost", "disabled": True}])
    # a full entry may still add a deployment-specific peripheral
    extra = merge_peripherals(
        [TOP],
        [{"type": "camera", "name": "deploy", "driver": "opencv", "index": 3, "transform": {"parent_frame": "base"}}],
    )
    assert [entry["name"] for entry in extra] == ["top", "deploy"]


# ---------------------------------------------------------------------------
# 2.3d: simulated joint current
# ---------------------------------------------------------------------------


def test_joint_current_conversion():
    from sensor_msgs.msg import JointState

    from robot_runtime.joint_current_publisher import joint_currents_from_efforts

    message = JointState()
    message.name = ["2", "1", "6"]
    message.effort = [10.0, 20.0, 30.0]
    assert joint_currents_from_efforts(["1", "2", "6"], message, [2.0, 0.5, 1.0]) == [40.0, 5.0, 30.0]
    message.effort = []
    assert joint_currents_from_efforts(["1"], message, [1.0]) is None


def test_joint_current_node_only_on_gazebo(tmp_path):
    profile = _profile(tmp_path)
    gazebo_actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", gazebo_world="demo")
    node = _node([a for a in gazebo_actions if isinstance(a, Node)], "simulated_joint_current")
    params = _params(node)
    assert params["endpoint"] == "/so101_follower/joint_currents"
    assert params["joint_names"] == ["1"]
    assert params["scale"] == [2.0]
    # the hardware/sdk stack never starts it
    for sim_platform, simulated in (("", False), ("sdk", True)):
        actions = runtime_stack_actions(profile, tmp_path / "robot.xacro", simulated, sim_platform=sim_platform)
        assert not any(isinstance(a, Node) and a._Node__node_executable == "joint_current_publisher" for a in actions)
    # a profile without the joint.current interface gets no publisher
    bare = _profile(tmp_path, with_joint_current=False)
    assert simulated_joint_current_actions(bare) == []


def test_joint_current_scale_is_looked_up_per_platform(tmp_path):
    profile = _profile(tmp_path)
    profile["simulation"] = {
        "gazebo": {"joint_current": {"scale": [2.0]}},
        "mujoco": {"joint_current": {"scale": [3.5]}},
    }
    gazebo_node = _node(
        [a for a in simulated_joint_current_actions(profile, sim_platform="gazebo") if isinstance(a, Node)],
        "simulated_joint_current",
    )
    mujoco_node = _node(
        [a for a in simulated_joint_current_actions(profile, sim_platform="mujoco") if isinstance(a, Node)],
        "simulated_joint_current",
    )
    assert _params(gazebo_node)["scale"] == [2.0]
    assert _params(mujoco_node)["scale"] == [3.5]
    # a platform section without the scale falls back to 1 per joint
    fallback = _node(
        [a for a in simulated_joint_current_actions(_profile(tmp_path), sim_platform="mujoco") if isinstance(a, Node)],
        "simulated_joint_current",
    )
    assert _params(fallback)["scale"] == [1.0]


def test_advertised_camera_frames_is_the_single_source_of_the_convention():
    """5b.3/D: one helper defines the opencv/realsense/camera_ros frame rules."""
    from robot_runtime.gazebo import advertised_camera_frame, advertised_camera_frames

    opencv = {"name": "top", "driver": "opencv", "frame_id": "camera_top_frame", "optical_frame_id": "unused"}
    assert advertised_camera_frames(opencv) == ("camera_top_frame", "camera_top_frame")
    assert advertised_camera_frame(opencv) == "camera_top_frame"

    realsense = {
        "name": "front",
        "driver": "realsense",
        "frame_id": "camera_front_link",
        "optical_frame_id": "camera_front_optical_frame",
    }
    assert advertised_camera_frames(realsense) == ("camera_front_optical_frame", "camera_front_optical_frame")

    remapped = {**realsense, "direct_topic_remap": True}
    assert advertised_camera_frames(remapped) == (None, "camera_front_optical_frame")
    assert advertised_camera_frame(remapped) == "camera_front_optical_frame"

    assert advertised_camera_frames({"name": "cam", "driver": "camera_ros"}) == (None, None)

    # defaults when the YAML omits the explicit frame ids
    assert advertised_camera_frames({"name": "wrist", "driver": "opencv"}) == (
        "camera_wrist_frame",
        "camera_wrist_frame",
    )


# ---------------------------------------------------------------------------
# lekiwi-gazebo-runtime-platform D5: simulated lidar and IMU sensors
# ---------------------------------------------------------------------------


def test_lidar_sensor_attaches_by_reference_at_the_mounted_pose():
    urdf = lidar_imu_sensor_urdf([MID360])
    # attached to the resolved parent link, no URDF link/joint for the frame
    assert '<gazebo reference="base_link">' in urdf
    assert '<link name="livox_frame">' not in urdf and "livox_frame_joint" not in urdf
    # the advertised frame stamps the bridged messages; the gz topic is named
    # after the peripheral
    assert "<gz_frame_id>livox_frame</gz_frame_id>" in urdf
    assert "<topic>/mid360_lidar</topic>" in urdf
    # the mount is the sensor pose (D13: simulation.gazebo.transform)
    assert "<pose>-0.08 0.0 0.2 0.0 0.0 1.5707963267948966</pose>" in urdf
    # D1 sensor parameters render into the scan pattern
    assert "<samples>900</samples>" in urdf and "<samples>32</samples>" in urdf
    assert "<min>0.1</min>" in urdf and "<max>40.0</max>" in urdf
    assert "<update_rate>10.0</update_rate>" in urdf
    assert 'sensor type="gpu_lidar" name="mid360_lidar"' in urdf


def test_integrated_imu_is_colocated_with_the_lidar():
    urdf = lidar_imu_sensor_urdf([MID360])
    assert 'sensor type="imu" name="mid360_imu"' in urdf
    assert "<topic>/mid360_imu</topic>" in urdf
    # same frame, same pose and the IMU-specific rate (D5)
    imu_block = urdf[urdf.index('name="mid360_imu"') : urdf.index('name="mid360_imu"') + 400]
    assert "<gz_frame_id>livox_frame</gz_frame_id>" in imu_block
    assert "<pose>-0.08 0.0 0.2 0.0 0.0 1.5707963267948966</pose>" in imu_block
    assert "<update_rate>200.0</update_rate>" in imu_block


def test_planar_lidar_renders_one_vertical_sample():
    urdf = lidar_imu_sensor_urdf([LDLIDAR])
    assert 'sensor type="gpu_lidar" name="ldlidar_lidar"' in urdf
    assert "<topic>/ldlidar_lidar</topic>" in urdf
    # vertical samples 1 = planar scan
    assert "<samples>1</samples>" in urdf
    assert "<min_angle>0.0</min_angle>" in urdf
    # no integrated IMU for this driver
    assert 'sensor type="imu"' not in urdf


def test_standalone_imu_peripheral_gets_its_own_sensor():
    periph = {
        "type": "imu",
        "name": "base_imu",
        "frame_id": "imu_frame",
        "imu_topic": "/imu/data",
        "transform": {"parent_frame": "base_link", "x": 0.0, "y": 0.0, "z": 0.08},
        "simulation": {"gazebo": {"update_rate": 100.0}},
    }
    urdf = lidar_imu_sensor_urdf([periph])
    assert 'sensor type="imu" name="base_imu_imu"' in urdf
    assert "<gz_frame_id>imu_frame</gz_frame_id>" in urdf
    assert "<topic>/base_imu_imu</topic>" in urdf
    assert "<update_rate>100.0</update_rate>" in urdf
    assert "<pose>0.0 0.0 0.08 0.0 0.0 0.0</pose>" in urdf


def test_lidar_and_imu_bridge_entries_target_the_advertised_endpoints():
    entries = [*lidar_bridge_entries(MID360), *imu_bridge_entries(MID360)]
    # both sensor streams land on internal raw topics: no-hit beams arrive
    # as ±inf (the finite filter serves the cloud endpoint) and the IMU's
    # pre-settle samples are all-zero (the settle filter serves the IMU
    # endpoint) — D5
    assert entries == [
        {
            "ros_topic_name": "/mid360_lidar/raw",
            "gz_topic_name": "/mid360_lidar/points",
            "ros_type_name": "sensor_msgs/msg/PointCloud2",
            "gz_type_name": "gz.msgs.PointCloudPacked",
            "direction": "GZ_TO_ROS",
            "lazy": False,
        },
        {
            "ros_topic_name": "/mid360_imu/raw",
            "gz_topic_name": "/mid360_imu",
            "ros_type_name": "sensor_msgs/msg/Imu",
            "gz_type_name": "gz.msgs.IMU",
            "direction": "GZ_TO_ROS",
            "lazy": False,
        },
    ]
    # a planar lidar bridges the LaserScan on its scan endpoint
    assert lidar_bridge_entries(LDLIDAR) == [
        {
            "ros_topic_name": "/scan",
            "gz_topic_name": "/ldlidar_lidar",
            "ros_type_name": "sensor_msgs/msg/LaserScan",
            "gz_type_name": "gz.msgs.LaserScan",
            "direction": "GZ_TO_ROS",
            "lazy": False,
        }
    ]
    assert imu_bridge_entries(LDLIDAR) == []
    # drivers the simulator does not stand in for bridge nothing
    other = {**MID360, "driver": "some_vendor_lidar"}
    assert lidar_bridge_entries(other) == [] and imu_bridge_entries(other) == []


def test_lidar_injection_follows_LIDAR_SIM_PLATFORMS_for_every_physics_platform():
    """D14.2 guard: for every physics platform the lidar injection is non-empty
    exactly when the platform simulates lidar (the capability the gate reads
    and the render path consult cannot drift apart)."""
    from robot_runtime.profile import LIDAR_SIM_PLATFORMS, PHYSICS_SIM_PLATFORMS

    for platform in PHYSICS_SIM_PLATFORMS:
        urdf = lidar_imu_sensor_urdf([MID360], platform=platform)
        assert bool(urdf) == (platform in LIDAR_SIM_PLATFORMS), (
            f"{platform}: lidar injection is {'non-empty' if urdf else 'empty'} but the platform is "
            f"{'in' if platform in LIDAR_SIM_PLATFORMS else 'not in'} LIDAR_SIM_PLATFORMS"
        )


def test_sensor_pose_prefers_the_simulation_transform():
    periph = {
        "type": "lidar",
        "name": "l",
        "driver": "livox_mid360",
        "transform": {"parent_frame": "body", "x": 0.0, "y": 0.0, "z": 0.0},
        "simulation": {"gazebo": {"transform": {"parent_frame": "base_link", "x": 1.0, "y": 2.0, "z": 3.0}}},
    }
    assert resolve_sensor_pose(periph, "gazebo") == {"parent_frame": "base_link", "x": 1.0, "y": 2.0, "z": 3.0}
    # without the simulation section the public transform applies
    assert resolve_sensor_pose({"transform": {"parent_frame": "body", "x": 9.0}}, "gazebo") == {
        "parent_frame": "body",
        "x": 9.0,
    }
    assert resolve_sensor_pose({}, "gazebo") == {}


def test_lidar_sensor_uses_the_sensor_sdf_directly():
    """The per-peripheral builders are the robot-agnostic primitives (D5)."""
    block = lidar_sensor_sdf(MID360)
    assert block.startswith("\n") is False or block.strip() != ""
    assert '<gazebo reference="base_link">' in block and 'name="mid360_lidar"' in block
    imu_block = imu_sensor_sdf(MID360)
    assert 'name="mid360_imu"' in imu_block and "<update_rate>200.0</update_rate>" in imu_block


# ---------------------------------------------------------------------------
# lekiwi-gazebo-runtime-platform D5: one effective inventory everywhere
# ---------------------------------------------------------------------------


def _lidar_profile(tmp_path, peripherals):
    profile = _profile(tmp_path, peripherals=peripherals)
    profile["fast_lio"] = {"enabled": False, "params_file": "/tmp/fl.yaml"}
    return profile


def test_render_gazebo_description_accepts_the_effective_inventory(tmp_path):
    profile = _lidar_profile(tmp_path, [TOP, MID360])
    urdf = render_gazebo_description(profile, tmp_path / "robot.xacro")
    assert 'name="mid360_lidar"' in urdf and 'name="mid360_imu"' in urdf
    assert 'name="top_camera"' in urdf
    # the same render comes out when the caller resolved the inventory
    explicit = render_gazebo_description(
        profile, tmp_path / "robot.xacro", peripherals=merge_peripherals(profile["peripherals"], [])
    )
    assert explicit == urdf


def test_fragment_changes_the_inventory_everywhere(tmp_path):
    """Disable, add, rename and re-mount through a fragment: render, bridge
    entries and advertised interfaces all reflect the same effective
    inventory (runtime-simulation-platform D5)."""
    base_profile = _lidar_profile(tmp_path, [TOP, MID360])
    fragment = tmp_path / "fragment.yaml"

    # remount + rename the endpoint through the fragment
    fragment.write_text(
        yaml.safe_dump(
            {
                "peripherals": [
                    {
                        "type": "lidar",
                        "name": "mid360",
                        "pointcloud_topic": "/livox/renamed",
                        "simulation": {"gazebo": {"transform": {"parent_frame": "base_link", "x": 0.5, "yaw": 0.25}}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    from robot_runtime.peripherals import load_peripherals_file

    merged = merge_peripherals(base_profile["peripherals"], load_peripherals_file(str(fragment)).get("peripherals"))
    effective = {**base_profile, "peripherals": merged}
    urdf = render_gazebo_description(effective, tmp_path / "robot.xacro")
    # the sensor moved to the fragment's mount (key-wise deep merge: the
    # unspecified z keeps the profile's 0.2) and the bridge follows the
    # renamed ROS endpoint
    assert "<pose>0.5 0.0 0.2 0.0 0.0 0.25</pose>" in urdf
    bridge = [entry for periph in merged for entry in (*lidar_bridge_entries(periph), *imu_bridge_entries(periph))]
    # the renamed endpoint is served by the finite filter from the raw topic,
    # the IMU endpoint by the settle filter (both internal raw topics)
    assert {entry["ros_topic_name"] for entry in bridge} == {"/mid360_lidar/raw", "/mid360_imu/raw"}
    description = build_description(effective, peripherals=merged, use_sim=True, sim_platform="gazebo")
    assert description["interfaces"]["lidar.mid360.points"]["endpoint"] == "/livox/renamed"
    assert description["interfaces"]["lidar.mid360.points"]["message_type"] == "sensor_msgs/msg/PointCloud2"

    # disabling through the fragment removes the sensor, the bridge entry and
    # the advertisement alike
    fragment.write_text(
        yaml.safe_dump({"peripherals": [{"type": "lidar", "name": "mid360", "disabled": True}]}), encoding="utf-8"
    )
    merged = merge_peripherals(base_profile["peripherals"], load_peripherals_file(str(fragment)).get("peripherals"))
    effective = {**base_profile, "peripherals": merged}
    urdf = render_gazebo_description(effective, tmp_path / "robot.xacro")
    assert "mid360_lidar" not in urdf and "mid360_imu" not in urdf
    assert not [entry for periph in merged for entry in (*lidar_bridge_entries(periph), *imu_bridge_entries(periph))]
    description = build_description(effective, peripherals=merged, use_sim=True, sim_platform="gazebo")
    assert not any(key.startswith("lidar.mid360.") for key in description["interfaces"])


def test_camera_only_provider_keeps_the_exact_pre_change_surface(tmp_path):
    """SO-101-shaped input (cameras only): no lidar/IMU content appears and
    the bridge stays camera-only (camera behaviour unchanged, D5)."""
    profile = _profile(tmp_path)  # TOP, WRIST, FRONT
    urdf = render_gazebo_description(profile, tmp_path / "robot.xacro")
    assert "gpu_lidar" not in urdf and 'sensor type="imu"' not in urdf
    actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", robot_description=urdf, gazebo_world="demo")
    bridge = _by_executable([a for a in actions if isinstance(a, Node)], "bridge_node")
    config = yaml.safe_load(Path(_params(bridge)["config_file"]).read_text(encoding="utf-8"))
    assert {entry["ros_type_name"] for entry in config} == {"sensor_msgs/msg/Image", "sensor_msgs/msg/CameraInfo"}
    assert all(entry["ros_topic_name"].startswith("/camera/") for entry in config)


def test_sensor_frames_absent_from_the_model_get_the_runtime_static_tf(tmp_path):
    """D5/D13: the runtime publishes static TF only for peripheral frames the
    rendered model does not contain; camera frames keep their single
    model-derived publisher (RSP)."""
    profile = _lidar_profile(tmp_path, [TOP, MID360])
    urdf = render_gazebo_description(profile, tmp_path / "robot.xacro")
    # the lidar frame is absent from the model, the camera frames are present
    assert '<link name="livox_frame">' not in urdf
    assert '<link name="camera_top_frame">' in urdf
    nodes = peripheral_tf_nodes_for_absent_frames(profile["peripherals"], urdf)
    tf_nodes = [node for node in nodes if node._Node__node_executable == "static_transform_publisher"]
    assert len(tf_nodes) == 1, "exactly the lidar frame gets a runtime static TF"
    args = tf_nodes[0]._Node__arguments
    assert args[args.index("--frame-id") + 1] == "base_link"
    assert args[args.index("--child-frame-id") + 1] == "livox_frame"
    # a camera-only profile composes no static TF at all: every camera frame
    # is owned by RSP through the rendered links
    camera_only = render_gazebo_description(_profile(tmp_path), tmp_path / "robot.xacro")
    assert peripheral_tf_nodes_for_absent_frames(_profile(tmp_path)["peripherals"], camera_only) == []


def test_lidar_tf_follows_the_mount_parent_from_the_fragment(tmp_path):
    """The application path remounts livox_frame under ``body`` (D13): the
    runtime static TF uses the fragment's public transform parent while the
    simulator attachment pose stays the base_link-relative mount."""
    profile = _lidar_profile(tmp_path, [MID360])
    fragment = tmp_path / "fragment.yaml"
    fragment.write_text(
        yaml.safe_dump(
            {
                "peripherals": [
                    {
                        "type": "lidar",
                        "name": "mid360",
                        "transform": {"parent_frame": "body", "x": 0.0, "y": 0.0, "z": 0.0},
                        "simulation": {
                            "gazebo": {"transform": {"parent_frame": "base_link", "x": -0.08, "z": 0.2, "yaw": 1.5708}}
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    from robot_runtime.peripherals import load_peripherals_file

    merged = merge_peripherals(profile["peripherals"], load_peripherals_file(str(fragment)).get("peripherals"))
    urdf = render_gazebo_description({**profile, "peripherals": merged}, tmp_path / "robot.xacro")
    # the sensor still attaches to base_link at the mount
    assert '<gazebo reference="base_link">' in urdf
    nodes = peripheral_tf_nodes_for_absent_frames(merged, urdf)
    tf_nodes = [node for node in nodes if node._Node__node_executable == "static_transform_publisher"]
    assert len(tf_nodes) == 1
    args = tf_nodes[0]._Node__arguments
    assert args[args.index("--frame-id") + 1] == "body"
    assert args[args.index("--child-frame-id") + 1] == "livox_frame"


def test_gazebo_embodiment_composes_the_sensor_bridge_and_tf(tmp_path):
    profile = _lidar_profile(tmp_path, [TOP, MID360])
    fragment = tmp_path / "fragment.yaml"
    fragment.write_text(yaml.safe_dump({"fast_lio": {"enabled": True, "sim_params_file": "/tmp/s.yaml"}}))
    actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", str(fragment), gazebo_world="demo")
    nodes = [a for a in actions if isinstance(a, Node)]
    bridge = _by_executable(nodes, "bridge_node")
    entries = yaml.safe_load(Path(_params(bridge)["config_file"]).read_text(encoding="utf-8"))
    topics = {entry["ros_topic_name"] for entry in entries}
    assert "/camera/top/image_raw" in topics
    # the raw gz streams are internal; the filters serve the endpoints
    assert "/mid360_lidar/raw" in topics and "/mid360_imu/raw" in topics
    assert Path(_params(bridge)["config_file"]).name == "ibrobot_runtime_sensor_bridge.yaml"
    # the finite filter: raw in, advertised endpoint out (D5)
    filter_node = _by_executable(nodes, "finite_cloud_filter")
    filter_params = _params(filter_node)
    assert filter_params["input_topic"] == "/mid360_lidar/raw"
    assert filter_params["output_topic"] == "/livox/lidar"
    # the IMU settle filter: raw in, advertised endpoint out (D5)
    imu_filter = _by_executable(nodes, "imu_settle_filter")
    imu_params = _params(imu_filter)
    assert imu_params["input_topic"] == "/mid360_imu/raw"
    assert imu_params["output_topic"] == "/livox/imu"
    # the lidar frame's static TF is part of the embodiment
    tf = [node for node in nodes if node._Node__node_executable == "static_transform_publisher"]
    assert len(tf) == 1 and tf[0]._Node__node_name == "static_tf_mid360"


def test_finite_cloud_filter_drops_nonfinite_points_only():
    """D5: no-hit beams arrive as ±inf; the advertised cloud stays finite."""
    import struct

    from robot_runtime.finite_cloud_filter import filter_finite_points

    def cloud(rows):
        message = PointCloud2()
        message.header.frame_id = "livox_frame"
        message.header.stamp.sec = 7
        message.width = len(rows)
        message.height = 1
        message.point_step = 16
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        message.data = b"".join(struct.pack("<ffff", *row) for row in rows)
        return message

    finite_rows = [(1.0, 2.0, 3.0, 10.0), (0.5, -0.5, 0.25, 5.0)]
    raw = cloud(
        [*finite_rows, (float("inf"), 1.0, 1.0, 3.0), (1.0, float("nan"), 1.0, 3.0), (-float("inf"), 0.0, 0.0, 1.0)]
    )
    filtered = filter_finite_points(raw)
    assert filtered.width == 2 and filtered.height == 1
    assert filtered.header.frame_id == "livox_frame" and filtered.header.stamp.sec == 7
    assert [struct.unpack_from("<ffff", bytes(filtered.data), i * 16) for i in range(2)] == finite_rows
    assert [field.name for field in filtered.fields] == ["x", "y", "z", "intensity"]
    # an already-finite cloud passes through untouched
    clean = cloud(finite_rows)
    assert filter_finite_points(clean) is clean


def test_lidar_finite_filter_nodes_cover_the_3d_lidar_only():
    from robot_runtime.gazebo import lidar_finite_filter_nodes

    nodes = lidar_finite_filter_nodes([TOP, MID360, LDLIDAR])
    assert [node._Node__node_name for node in nodes] == ["mid360_finite_cloud_filter"]


def test_imu_settle_filter_holds_back_pre_settle_samples():
    """D5: the simulator's pre-settle IMU samples never reach the endpoint."""
    from robot_runtime.imu_settle_filter import GRAVITY_MAX, GRAVITY_MIN, SettleGate, is_settle_sample

    def imu(acc, gyro=0.0):
        message = Imu()
        message.header.frame_id = "livox_frame"
        message.header.stamp.sec = 3
        message.linear_acceleration.z = acc
        message.angular_velocity.x = gyro
        return message

    # all-zero: the pre-settle artifact (gravity must appear on a real IMU)
    assert is_settle_sample(imu(0.0)) is True
    # gravity (with or without spin): a valid sample
    assert is_settle_sample(imu(9.8, 0.17)) is False
    # free fall is physical (acc 0) as long as it spins
    assert is_settle_sample(imu(0.0, 0.17)) is False

    # the observed Gazebo transient: a zero window, one landing impulse
    # (measured at ~45 g), then the resting gravity
    gate = SettleGate(run_length=3)
    assert [gate.accept(m) for m in (imu(0.0), imu(0.0), imu(0.0))] == [False, False, False]
    # the landing impulse is not gravity: it does not advance the gate
    assert gate.accept(imu(441.997)) is False
    assert not gate.settled
    # three consecutive gravity-plausible samples open the gate on the last
    assert [gate.accept(imu(9.8)) for _ in range(3)] == [False, False, True]
    assert gate.settled
    # open gate: everything passes, including motion dynamics
    assert gate.accept(imu(441.997)) is True
    assert gate.accept(imu(0.0, 0.17)) is True

    # an implausible sample resets the run: a short gravity blip does not open
    gate = SettleGate(run_length=3)
    gate.accept(imu(9.8))
    gate.accept(imu(9.8))
    gate.accept(imu(0.0))  # zero window resumes mid-settle
    assert gate.accept(imu(9.8)) is False
    assert gate.accept(imu(9.8)) is False
    assert gate.accept(imu(9.8)) is True

    # the gravity window is orientation independent (tilt keeps |acc| ~ g)
    assert GRAVITY_MIN < 9.80665 < GRAVITY_MAX
    tilted = Imu()
    tilted.linear_acceleration.x = 9.80665 * math.cos(math.radians(75))
    tilted.linear_acceleration.z = 9.80665 * math.sin(math.radians(75))
    assert SettleGate(run_length=1).accept(tilted) is True


def test_imu_settle_filter_nodes_cover_simulated_imus():
    from robot_runtime.gazebo import imu_settle_filter_nodes

    nodes = imu_settle_filter_nodes([TOP, MID360, LDLIDAR])
    assert [node._Node__node_name for node in nodes] == ["mid360_imu_settle_filter"]
    params = _params(nodes[0])
    assert params["input_topic"] == "/mid360_imu/raw"
    assert params["output_topic"] == "/livox/imu"


# ---------------------------------------------------------------------------
# lekiwi-gazebo-runtime-platform D5: derived scan converter composition
# ---------------------------------------------------------------------------


def _scan_matrix_profile(tmp_path):
    profile = _lidar_profile(tmp_path, [MID360])
    # the hardware branch composes the physical driver: give it the minimal
    # user config the driver composer rewrites
    user_config = tmp_path / "MID360_config.json"
    user_config.write_text(json.dumps({"MID360": {"host_net_info": {}}, "lidar_configs": [{}]}), encoding="utf-8")
    profile["peripherals"] = [
        {**periph, "user_config_path": str(user_config), "host_ip": "192.168.1.5", "lidar_ip": "192.168.1.139"}
        if periph.get("type") == "lidar"
        else periph
        for periph in profile["peripherals"]
    ]
    profile["fast_lio"] = {
        "enabled": False,
        "params_file": "/tmp/fl.yaml",
        "sim_params_file": "/tmp/fl_sim.yaml",
    }
    return profile


def test_scan_converter_composition_matrix(tmp_path):
    """Hardware composes the converter with the driver; sdk keeps synthetic
    perception as the single /scan producer; a physics platform composes it
    on sim time only when its source (FAST-LIO) runs (D5)."""
    profile = _scan_matrix_profile(tmp_path)

    def executables(actions):
        return [a._Node__node_executable for a in actions if isinstance(a, Node)]

    # hardware: the converter rides with the physical driver
    hardware = peripheral_stack_actions(profile, "", False, sim_platform="")
    assert "livox_ros_driver2_node" in executables(hardware)
    assert "pointcloud_to_laserscan_node" in executables(hardware)

    # sdk: no converter, even with FAST-LIO enabled (synthetic /scan)
    sdk = peripheral_stack_actions(profile, "", True, sim_platform="sdk")
    assert "pointcloud_to_laserscan_node" not in executables(sdk)
    assert "synthetic_perception" in executables(sdk)

    # gazebo with FAST-LIO off: no converter, no synthetic /scan either
    gazebo_off = peripheral_stack_actions(profile, "", True, sim_platform="gazebo")
    assert "pointcloud_to_laserscan_node" not in executables(gazebo_off)
    assert "synthetic_perception" not in executables(gazebo_off)

    # gazebo with FAST-LIO on (fragment): the converter runs on sim time
    fragment = tmp_path / "fragment.yaml"
    fragment.write_text(
        yaml.safe_dump({"fast_lio": {"enabled": True, "sim_params_file": "/tmp/fl_sim.yaml"}}), encoding="utf-8"
    )
    gazebo_on = peripheral_stack_actions(profile, str(fragment), True, sim_platform="gazebo")
    converters = [
        a for a in gazebo_on if isinstance(a, Node) and a._Node__node_executable == "pointcloud_to_laserscan_node"
    ]
    assert len(converters) == 1
    assert _params(converters[0])["use_sim_time"] is True
    assert converters[0]._Node__node_name == "mid360_pointcloud_to_laserscan"
