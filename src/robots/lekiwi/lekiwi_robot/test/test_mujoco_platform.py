"""LeKiwi MuJoCo platform tests (openspec lekiwi-mujoco-runtime-platform).

The structure part of D10, against the shipped ``runtime.launch.py`` and the
shipped ``lekiwi_lidar`` profile with ``sim_platform:=mujoco``:

- D1: the platform declaration (profile section, MID-360 platform parameters)
  and the rejection surface (``lekiwi_nav_grasp`` keeps rejecting it);
- D2/D2a/D9: the MuJoCo render — the composed MJCF (free joint, lidar
  extension/sensor, IMU sensors, wheel contact parameters) and the URDF's
  hardware parameters and wheel-name transmissions;
- D7: the launch plumbing — ``mujoco_scene``, the spawn pose, simulation time;
- D3.1/D3.2: the mode resolution keeps the base controller active in idle on
  MuJoCo only, and the base node receives the published cmd_vel envelope;
- D5.3: the embodiment's merged plugin parameters (base velocity without
  limit keys, lidar) and the generated controllers file (IMU broadcaster).

The real-run part of D10 (ACTIVE, sensors publishing, one odom publisher,
ground-truth topic) lives with the gates (tasks 5.1/5.2).
"""

from __future__ import annotations

import importlib.util
import math
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import launch
import numpy
import pytest
import rclpy
import yaml
from geometry_msgs.msg import Twist, Vector3
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from mujoco_test_support import require_mujoco
from nav_msgs.msg import Odometry
from rclpy.node import Node as RclpyNode
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy as QoSReliability
from sensor_msgs.msg import Imu, PointCloud2

from ibrobot_msgs.msg import RuntimeStatus

_PACKAGE = Path(__file__).resolve().parents[1]
_LAUNCH = _PACKAGE / "launch" / "runtime.launch.py"
LIDAR_PROFILE = _PACKAGE / "profiles" / "lekiwi_lidar.yaml"
GRASP_PROFILE = _PACKAGE / "profiles" / "lekiwi_nav_grasp.yaml"
MOUNT = _PACKAGE.parent / "lekiwi_description" / "config" / "lekiwi_mid360_mount.yaml"

# The Python mujoco module is imported only inside the gated code paths (the
# render composition): a module-level import (or importorskip) would skip or
# break collection of the whole file on a machine without the simulator
# (openEuler), hiding the profile/structure tests that need no simulator.


def _launch_module():
    spec = importlib.util.spec_from_file_location("lekiwi_robot_runtime_launch_mujoco", _LAUNCH)
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


def _param_file_paths(node):
    return [
        Path("".join(getattr(part, "text", None) or str(part) for part in entry.param_file))
        for entry in node._Node__parameters
        if isinstance(entry, ParameterFile)
    ]


def _mount_translation():
    mount = yaml.safe_load(MOUNT.read_text(encoding="utf-8"))
    return mount["translation_m"], [math.radians(angle) for angle in mount["rpy_deg"]]


def _mujoco_actions(**extra):
    """The launch actions for the MuJoCo platform in the nav_room scene."""
    return _render(
        LIDAR_PROFILE,
        {
            "use_sim": "true",
            "sim_platform": "mujoco",
            "mujoco_scene": "nav_room",
            "spawn_x": "-1.5",
            "spawn_y": "-1.5",
            "spawn_z": "0.05",
            "spawn_yaw": "0.0",
            **extra,
        },
    )


def _simulator_node(actions):
    nodes = [action for action in actions if isinstance(action, Node)]
    simulator = [node for node in nodes if node._Node__node_executable == "ros2_control_node"]
    assert len(simulator) == 1 and simulator[0]._Node__package == "mujoco_ros2_control"
    return simulator[0], nodes


def _plugin_tree(simulator):
    files = [
        path
        for path in _param_file_paths(simulator)
        if "mujoco_ros2_control_node" in yaml.safe_load(path.read_text(encoding="utf-8"))
    ]
    assert len(files) == 1, f"exactly one simulator params file, got {[_param_file_paths(simulator)]}"
    tree = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    return files[0], tree["mujoco_ros2_control_node"]["ros__parameters"]["mujoco_plugins"]


def _rendered_urdf(actions):
    simulator, _nodes = _simulator_node(actions)
    # the URDF rides the controller-manager params file the simulator loads
    files = [
        path
        for path in _param_file_paths(simulator)
        if "mujoco_ros2_control_node" not in yaml.safe_load(path.read_text(encoding="utf-8"))
    ]
    for path in files:
        tree = yaml.safe_load(path.read_text(encoding="utf-8"))
        text = str((tree.get("controller_manager") or {}).get("ros__parameters", {}).get("robot_description") or "")
        if "mujoco_ros2_control/MujocoSystemInterface" in text:
            return text
    raise AssertionError(f"no controller-manager params file carries the URDF: {files}")


# ---------------------------------------------------------------------------
# D1: profile selection
# ---------------------------------------------------------------------------


def test_shipped_lidar_profile_declares_the_mujoco_platform():
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    assert profile["sim_platforms"] == ["sdk", "gazebo", "mujoco"]
    assert profile["sim_platform"] == "sdk"
    mujoco_section = profile["simulation"]["mujoco"]
    assert mujoco_section["template"] == "$(find lekiwi_description)/mujoco/lekiwi_base.xml.template"
    assert mujoco_section["meshes_dir"] == "$(find lekiwi_description)/meshes/mujoco"
    assert mujoco_section["sdf_meshes"] == []
    assert mujoco_section["floating_base"] == {"body": "base_link", "twist_source_topic": "/odom"}


def test_profile_mujoco_mount_and_grid_match_the_gazebo_surface():
    """D1: the MuJoCo attachment pose defaults to the same mount, with the
    coarser CPU ray-cast grid; the IMU rate key is absent (the CM clock owns
    the rate on MuJoCo, D5.3)."""
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    translation, rpy = _mount_translation()
    expected = {
        "parent_frame": "base_link",
        "x": translation[0],
        "y": translation[1],
        "z": translation[2],
        "roll": rpy[0],
        "pitch": rpy[1],
        "yaw": rpy[2],
    }
    lidar = next(periph for periph in profile["peripherals"] if periph.get("type") == "lidar")
    mujoco_sim = lidar["simulation"]["mujoco"]
    assert mujoco_sim["transform"] == expected
    assert mujoco_sim["horizontal_samples"] == 360  # coarser than Gazebo's 900
    assert mujoco_sim["vertical_samples"] == 32
    assert mujoco_sim["vertical_min_deg"] == -7.0
    assert mujoco_sim["vertical_max_deg"] == 52.0
    assert mujoco_sim["range_min"] == 0.1
    assert mujoco_sim["range_max"] == 40.0
    assert mujoco_sim["update_rate"] == 10.0
    assert "imu" not in mujoco_sim


def test_nav_grasp_still_rejects_mujoco_before_any_action():
    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match="sim_platform 'mujoco' is not supported") as excinfo:
        _render(GRASP_PROFILE, {"use_sim": "true", "sim_platform": "mujoco"})
    message = str(excinfo.value)
    assert "sdk" in message and "lekiwi_nav_grasp" in message


# ---------------------------------------------------------------------------
# D2/D2a/D9: the MuJoCo render
# ---------------------------------------------------------------------------


def _composed_model(actions):
    """The composed MJCF path the URDF's mujoco_model points at."""
    urdf = _rendered_urdf(actions)
    root = ET.fromstring(urdf)
    params = {param.get("name"): param.text for param in root.findall(".//ros2_control/hardware/param")}
    return Path(params["mujoco_model"])


def test_mujoco_render_carries_the_platform_structure():
    require_mujoco()  # the render composes the model for real
    actions = _mujoco_actions()
    composed = _composed_model(actions)
    # D5.1a: the LeKiwi model declares the lidar engine extension, so the
    # compiler delivers the XML (never a .mjb)
    assert composed.suffix == ".xml" and composed.is_file()
    root = ET.parse(composed).getroot()

    # the floating base at the spawn pose
    base = root.find("worldbody/body[@name='base_link']")
    assert base is not None
    assert [float(v) for v in base.get("pos").split()] == pytest.approx([-1.5, -1.5, 0.05])
    free = base.find("joint[@name='floating_base_joint']")
    assert free is not None and free.get("type") == "free"

    # the engine extension and the injected lidar sensor + IMU triple (D5.2)
    assert root.find("extension/plugin[@plugin='mujoco.plugin.lidar']") is not None
    sensors = {sensor.get("name"): sensor for sensor in root.findall("sensor/*")}
    lidar = sensors["mid360_lidar"]
    assert lidar.get("plugin") == "mujoco.plugin.lidar" and lidar.get("objname") == "mid360_lidar_site"
    config = {child.get("key"): child.get("value") for child in lidar}
    assert config["resolution"] == "360 32"
    assert config["update_rate"] == "10"
    site = base.find("site[@name='mid360_lidar_site']")
    translation, rpy = _mount_translation()
    assert [float(v) for v in site.get("pos").split()] == pytest.approx(translation)
    assert site.get("quat") is not None  # the mount orientation, w-x-y-z
    for name in ("mid360_imu_quat", "mid360_imu_gyro", "mid360_imu_accel"):
        assert name in sensors

    # D2: the wheels and their contact parameters (the delivered XML keeps
    # the template's default classes; the frictionless sphere parameters
    # govern every wheel-plane pair). The collision geoms render in group 3
    # (review round 2: the viewer hides groups above 2 by default, so the
    # visual meshes — not the analytic hulls — are what the robot looks like;
    # contacts and rays ignore the group).
    wheel_default = root.find("default/default[@class='wheel_collision']/geom")
    assert wheel_default.get("condim") == "1" and wheel_default.get("priority") == "2"
    assert wheel_default.get("friction") == "0 0 0"
    assert wheel_default.get("group") == "3"
    wheel_bodies = {}
    for body in root.findall("worldbody/body/body"):
        for joint in body.findall("joint"):
            if joint.get("name", "").startswith("wheel_"):
                wheel_bodies[joint.get("name")] = body
    assert set(wheel_bodies) == {"wheel_7", "wheel_8", "wheel_9"}
    for joint_name, body in wheel_bodies.items():
        collision = body.find(f"geom[@name='{joint_name}']")
        assert collision is not None and collision.get("class") == "wheel_collision"
    actuator = root.find("actuator/*[@name='wheel_7']")
    assert actuator is not None
    assert [float(v) for v in actuator.get("ctrlrange").split()] == pytest.approx([-5.5224, 5.5224])


def test_mujoco_urdf_carries_the_transmissions_and_odometry_params():
    """D2a/D9: the wheel-name mappings and the floating-base odometry
    constants, matching robot_runtime.mujoco (the ground-truth subscriber and
    the relay resolve the same names)."""
    from robot_runtime.mujoco import FLOATING_BASE_STATE_TOPIC, ODOM_FREE_JOINT_NAME

    require_mujoco()
    actions = _mujoco_actions()
    urdf = _rendered_urdf(actions)
    root = ET.fromstring(urdf)
    params = {param.get("name"): param.text for param in root.findall(".//ros2_control/hardware/param")}
    assert params["odom_free_joint_name"] == ODOM_FREE_JOINT_NAME
    assert params["odom_topic"] == FLOATING_BASE_STATE_TOPIC
    assert params["auto_register_cameras"] == "false"
    transmissions = root.findall(".//ros2_control/transmission")
    assert len(transmissions) == 3
    for index in ("7", "8", "9"):
        match = [t for t in transmissions if t.find("joint").get("name") == index]
        assert len(match) == 1
        assert match[0].find("actuator").get("name") == f"wheel_{index}"
        assert match[0].find("plugin").text == "transmission_interface/SimpleTransmission"
    # the runtime-injected IMU sensor rides the ros2_control block (D5.3)
    sensor = root.find(".//ros2_control/sensor[@name='mid360_imu']")
    assert sensor is not None
    sensor_params = {param.get("name"): param.text for param in sensor.findall("param")}
    assert sensor_params["mujoco_type"] == "imu"
    assert sensor_params["mujoco_sensor_name"] == "mid360_imu"


def test_hardware_and_sdk_renders_carry_no_mujoco_content():
    """D2a pin: only the MuJoCo branch emits the transmissions; the physical,
    SDK and Gazebo renders keep their exact pre-change output."""
    from robot_runtime.launch_support import render_gazebo_description, render_robot_description
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    for simulated in (False, True):
        urdf = render_robot_description(profile, LIDAR_PROFILE, simulated)
        assert "transmission" not in urdf
        assert "odom_free_joint_name" not in urdf
    gazebo = render_gazebo_description(profile, LIDAR_PROFILE)
    assert "transmission" not in gazebo
    assert "odom_free_joint_name" not in gazebo


# ---------------------------------------------------------------------------
# D5.3: the embodiment's plugin parameters and generated controllers
# ---------------------------------------------------------------------------


def test_plugin_parameters_carry_base_velocity_and_lidar():
    from robot_runtime.mujoco import (
        BASE_VELOCITY_PLUGIN_INSTANCE,
        FLOATING_BASE_CMD_TOPIC,
        LIDAR_PLUGIN_INSTANCE,
    )

    require_mujoco()
    actions = _mujoco_actions()
    simulator, _nodes = _simulator_node(actions)
    _file, plugins = _plugin_tree(simulator)
    base_velocity = plugins[BASE_VELOCITY_PLUGIN_INSTANCE]
    assert base_velocity["body"] == "base_link"
    assert base_velocity["cmd_vel_topic"] == FLOATING_BASE_CMD_TOPIC
    assert base_velocity["use_stamped_twist"] is True
    assert base_velocity["cmd_timeout"] == 0.25
    # D3.2: no limit keys — upstream defaults are +inf, so the base executes
    # exactly the twist /odom reports (the one clamp lives in base_node)
    assert "max_linear_velocity" not in base_velocity
    assert "max_yaw_rate" not in base_velocity
    lidar = plugins[LIDAR_PLUGIN_INSTANCE]["mid360_lidar"]
    assert lidar["frame_name"] == "livox_frame"
    assert lidar["topic"] == "/mid360_lidar/raw"


def test_generated_controllers_carry_the_imu_broadcaster():
    require_mujoco()
    actions = _mujoco_actions()
    simulator, nodes = _simulator_node(actions)
    spawners = [node for node in nodes if node._Node__node_executable == "spawner"]
    spawned = []
    for spawner in spawners:
        for argument in spawner._Node__arguments:
            text = _text(argument) if hasattr(argument, "text") else str(argument)
            if text.startswith("--"):
                break  # the controller names lead, the flags follow
            spawned.append(text)
    assert "mid360_imu_broadcaster" in spawned
    # on MuJoCo the base controller starts active (D3.1: idle owns it)
    assert "base_velocity_controller" in spawned
    # the generated file merges the shipped controllers with the broadcaster
    files = [
        path for path in _param_file_paths(simulator) if "mid360_imu_broadcaster" in path.read_text(encoding="utf-8")
    ]
    assert len(files) == 1
    tree = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    entry = tree["mid360_imu_broadcaster"]["ros__parameters"]
    assert entry == {"sensor_name": "mid360_imu", "frame_id": "livox_frame"}
    # the shipped controller set survives the merge
    assert "base_velocity_controller" in tree["controller_manager"]["ros__parameters"]


# ---------------------------------------------------------------------------
# D3.1/D3.2/D7: mode resolution, the envelope, launch plumbing
# ---------------------------------------------------------------------------


def test_mode_resolution_keeps_the_base_controller_in_idle_only_on_mujoco():
    """D3.1: on MuJoCo the base controller is never released — idle keeps it
    active so no mode transition or stop deactivates it (idle and
    base_navigation then own the same controller set); every other platform
    keeps idle empty and starts the base controller inactive."""
    from robot_runtime.launch_support import profile_controllers
    from robot_runtime.profile import load_profile

    profile = load_profile(LIDAR_PROFILE)
    for platform in (None, "sdk", "gazebo"):
        broadcasters, active, inactive = profile_controllers(profile, platform)
        assert active == [], (platform, active)
        assert inactive == ["base_velocity_controller"], (platform, inactive)
        assert broadcasters == ["joint_state_broadcaster"]
    broadcasters, active, inactive = profile_controllers(profile, "mujoco")
    # idle owns the base controller on MuJoCo: it starts active and no mode
    # switch ever deactivates it (base_node is the gate on that platform)
    assert active == ["base_velocity_controller"]
    assert inactive == []
    assert broadcasters == ["joint_state_broadcaster"]


def test_mujoco_launch_threads_scene_spawn_and_sim_time():
    require_mujoco()
    actions = _mujoco_actions()
    simulator, nodes = _simulator_node(actions)
    # no generic controller-manager process (the plugin embeds it)
    assert not [
        node
        for node in nodes
        if node._Node__package == "controller_manager" and node._Node__node_executable == "ros2_control_node"
    ]
    # the spawn pose reaches the composed model (checked above) and the
    # facade / base node / relay run on simulation time with the platform
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    assert _params(facade)["use_sim_time"] is True
    assert _params(facade)["sim_platform"] == "mujoco"
    base = next(node for node in nodes if node._Node__node_executable == "base_node")
    base_params = _params(base)
    assert base_params["use_sim_time"] is True
    # D3.2: the published envelope reaches the base node
    assert base_params["max_vx"] == 0.26
    assert base_params["max_vy"] == 0.23
    assert base_params["max_wz"] == 1.5
    # D3.3: the odometry-twist relay forwards the base's own wheel odometry
    relay = next(node for node in nodes if node._Node__node_executable == "odometry_twist_relay")
    relay_params = _params(relay)
    assert relay_params["twist_source_topic"] == "/odom"
    assert relay_params["cmd_vel_topic"] == "/simulator/floating_base/cmd_vel"
    assert relay_params["base_frame"] == "base_link"
    assert relay_params["use_sim_time"] is True


def test_launch_declares_the_mujoco_arguments():
    from launch.actions import DeclareLaunchArgument

    description = _launch_module().generate_launch_description()
    arguments = [action for action in description.entities if isinstance(action, DeclareLaunchArgument)]
    names = {_text(action._DeclareLaunchArgument__name) for action in arguments}
    assert "mujoco_scene" in names
    defaults = {
        _text(action._DeclareLaunchArgument__name): _text(action._DeclareLaunchArgument__default_value)
        for action in arguments
        if action._DeclareLaunchArgument__default_value
    }
    assert defaults["mujoco_scene"] == ""


def test_filters_and_static_tf_bring_the_streams_to_the_advertised_endpoints():
    """D5.3: the finite-cloud filter and the IMU settle filter run on the raw
    simulator streams, and the static TF covers the livox frame."""
    require_mujoco()
    actions = _mujoco_actions()
    simulator, nodes = _simulator_node(actions)
    cloud = next(node for node in nodes if node._Node__node_executable == "finite_cloud_filter")
    cloud_params = _params(cloud)
    assert cloud_params["input_topic"] == "/mid360_lidar/raw"
    assert cloud_params["output_topic"] == "/livox/lidar"
    assert cloud_params["add_intensity"] is True
    imu = next(node for node in nodes if node._Node__node_executable == "imu_settle_filter")
    imu_params = _params(imu)
    assert imu_params["input_topic"] == "/mid360_imu/raw"
    assert imu_params["output_topic"] == "/livox/imu"
    static_tf = [node for node in nodes if node._Node__node_executable == "static_transform_publisher"]
    assert static_tf, "the livox_frame static TF must run (the model has no livox body)"


# ---------------------------------------------------------------------------
# D10: the real provider-alone run (task 5.1) — headless, fresh ROS domain
# ---------------------------------------------------------------------------


class _RunProbe(RclpyNode):
    """Everything the provider-alone run observes (D10)."""

    def __init__(self):
        super().__init__("mujoco_platform_run_probe")
        self.status = None
        self.clouds = []
        self.imus = []
        self.base_states = []
        self.create_subscription(RuntimeStatus, "/runtime_status", self._on_status, 10)
        self.create_subscription(
            PointCloud2, "/livox/lidar", self._on_cloud, QoSProfile(reliability=QoSReliability.RELIABLE, depth=5)
        )
        self.create_subscription(
            Imu, "/livox/imu", self._on_imu, QoSProfile(reliability=QoSReliability.RELIABLE, depth=10)
        )
        # the ground-truth topic, subscribed by this test only (D4/D9)
        self.create_subscription(Odometry, "/simulator/floating_base_state", self._on_base_state, 10)

    def _on_status(self, msg):
        self.status = msg

    def _on_cloud(self, msg):
        self.clouds.append(msg)

    def _on_imu(self, msg):
        self.imus.append(msg)

    def _on_base_state(self, msg):
        self.base_states.append(msg)


@pytest.mark.parametrize("scene", ["empty", "nav_room"])
def test_provider_alone_run_reaches_active_and_streams_the_sensor_surface(tmp_path, scene):
    """D10 (task 5.1): the real provider-alone MuJoCo run, in ``empty`` and
    ``nav_room`` — ACTIVE in idle, the advertised cloud (``x y z intensity``
    float32, finite, ``livox_frame``) and IMU on simulation time, exactly one
    ``odom -> base_link`` TF publisher, the ground-truth topic subscribed by
    this test only, and a closed navigation gate keeping the base still under
    ``/cmd_vel``."""
    require_mujoco()
    import random

    domain = str(random.randint(150, 229))
    environment = dict(os.environ)
    environment.update({"ROS_DOMAIN_ID": domain, "IBROBOT_MUJOCO_HEADLESS": "1", "ROS_LOCALHOST_ONLY": "1"})
    os.environ["ROS_DOMAIN_ID"] = domain
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    spawn = {"empty": ("0.0", "0.0"), "nav_room": ("-1.5", "-1.5")}[scene]
    launch_log = tmp_path / "launch.log"
    launch = subprocess.Popen(
        [
            "ros2",
            "launch",
            "lekiwi_robot",
            "runtime.launch.py",
            "profile:=lekiwi_lidar",
            "sim_platform:=mujoco",
            f"mujoco_scene:={scene}",
            f"spawn_x:={spawn[0]}",
            f"spawn_y:={spawn[1]}",
            "spawn_z:=0.05",
            "spawn_yaw:=0.0",
        ],
        stdout=launch_log.open("w"),
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    probe = None
    executor = None
    try:
        rclpy.init()
        probe = _RunProbe()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(probe)
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and (probe.status is None or probe.status.lifecycle != "ACTIVE"):
            executor.spin_once(timeout_sec=0.2)
            if launch.poll() is not None:
                pytest.fail(f"the launch exited early ({launch.returncode}); see {launch_log}")
        assert probe.status is not None and probe.status.lifecycle == "ACTIVE", (
            f"the runtime did not reach ACTIVE; see {launch_log}"
        )
        assert probe.status.active_mode == "idle"
        # settle, then sample the sensor surface
        time.sleep(4.0)
        probe.clouds.clear()
        probe.imus.clear()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and (len(probe.clouds) < 3 or len(probe.imus) < 10):
            executor.spin_once(timeout_sec=0.2)
        assert len(probe.clouds) >= 3, f"too few /livox/lidar messages; see {launch_log}"
        assert len(probe.imus) >= 10, f"too few /livox/imu messages; see {launch_log}"

        # the cloud: x y z intensity float32, finite, livox_frame, sim stamps
        cloud = probe.clouds[-1]
        fields = {field.name: (field.datatype, field.offset) for field in cloud.fields}
        assert set(fields) == {"x", "y", "z", "intensity"}
        assert all(datatype == 7 for datatype, _ in fields.values())  # float32
        assert cloud.point_step == 16
        assert cloud.is_dense is True
        assert cloud.width * cloud.height > 0
        arr = numpy.frombuffer(bytes(cloud.data), dtype=numpy.float32).reshape(-1, 4)
        assert numpy.isfinite(arr).all(), "the filtered cloud must be entirely finite"
        assert cloud.header.frame_id == "livox_frame"
        clock = probe.get_clock().now().nanoseconds * 1e-9
        stamp = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9
        assert 0.0 < stamp <= clock + 0.1, "the cloud stamp must be simulation time"

        # the IMU on simulation time
        imu = probe.imus[-1]
        imu_stamp = imu.header.stamp.sec + imu.header.stamp.nanosec * 1e-9
        assert 0.0 < imu_stamp <= clock + 0.1, "the IMU stamp must be simulation time"

        # exactly one odom -> base_link publisher: the /tf publisher set
        tf_publishers = {info.node_name for info in probe.get_publishers_info_by_topic("/tf")}
        assert tf_publishers == {"lekiwi_base", "robot_state_publisher"}, (
            f"unexpected /tf publishers: {sorted(tf_publishers)}"
        )

        # the ground-truth topic: this test is the only subscriber (D4/D9)
        subscribers = {
            info.node_name for info in probe.get_subscriptions_info_by_topic("/simulator/floating_base_state")
        }
        assert subscribers == {"mujoco_platform_run_probe"}, f"ground-truth topic subscribers: {subscribers}"

        # the closed gate: /cmd_vel must not move the base (idle + gate closed)
        command = probe.create_publisher(Twist, "/cmd_vel", 10)
        before = probe.base_states[-1] if probe.base_states else None
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and before is None:
            executor.spin_once(timeout_sec=0.1)
            before = probe.base_states[-1] if probe.base_states else None
        assert before is not None, "no floating-base state arrived"
        start = time.monotonic()
        while time.monotonic() - start < 4.0:
            executor.spin_once(timeout_sec=0.02)
            command.publish(Twist(linear=Vector3(x=0.2)))
            time.sleep(0.05)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if len(probe.base_states) and probe.base_states[-1].header.stamp != before.header.stamp:
                break
        after = probe.base_states[-1]
        displacement = math.hypot(
            after.pose.pose.position.x - before.pose.pose.position.x,
            after.pose.pose.position.y - before.pose.pose.position.y,
        )
        assert displacement < 0.01, f"the closed gate let the base move {displacement:.4f} m under /cmd_vel"
    finally:
        if executor is not None:
            executor.shutdown()
        if probe is not None:
            probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if launch.poll() is None:
            os.killpg(os.getpgid(launch.pid), signal.SIGINT)
            try:
                launch.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(launch.pid), signal.SIGKILL)
                launch.wait(timeout=10)
