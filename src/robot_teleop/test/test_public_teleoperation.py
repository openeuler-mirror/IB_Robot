"""Device-free tests of the migrated, actually wired teleoperation boundary."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sensor_msgs.msg import JointState, Joy

from robot_teleop.devices.leader_topic import LeaderTopicDevice
from robot_teleop.devices.xbox_controller import XboxTeleopDevice
from robot_teleop.public_target import map_gripper_ratio, resolve_public_target


def config():
    arm = ["1", "2", "3", "4", "5"]
    model = {
        "joint_groups": {"arm": arm, "gripper": ["6"]},
        "frames": {"base_link": "base", "ee_link": "gripper"},
        "joint_limits": {name: {"min": -2.0, "max": 2.0} for name in arm + ["6"]},
        "joint_conversions": {"modes": {"range_m100_100": {"6": {"min": -0.6, "max": 1.6}}}},
    }
    types = {
        "joints": "sensor_msgs/msg/JointState",
        "pose": "geometry_msgs/msg/PoseStamped",
        "linear": "geometry_msgs/msg/Vector3Stamped",
        "angular": "geometry_msgs/msg/Vector3Stamped",
        "start": "std_srvs/srv/Trigger",
        "stop": "std_srvs/srv/Trigger",
        "home": "ibrobot_msgs/action/ArmReturnHome",
        "lease": "std_msgs/msg/Empty",
    }
    interfaces = {
        f"motion.arm.{key}": {
            "message_type": kind,
            "target_group": "arm",
            "base_frame": "base",
            "tool_frame": "gripper",
            "joint_names": arm + ["6"] if key == "joints" else arm,
            "endpoint": f"/motion/arm/{key}",
            "command_stale_s": 0.2,
        }
        for key, kind in types.items()
    }
    interfaces["motion.arm.pose"]["pose_reference"] = "clutch_relative"
    for key in types:
        kind = "action" if key == "home" else "service" if key in ("start", "stop") else "topic"
        interfaces[f"motion.arm.{key}"].update({"kind": kind, "direction": "subscribe" if kind == "topic" else "serve"})
    for key, units in {
        "joints": {"position": "rad"},
        "pose": {"position": "m", "orientation": "quaternion"},
        "linear": {"linear": "m/s"},
        "angular": {"angular": "rad/s"},
    }.items():
        interfaces[f"motion.arm.{key}"]["units"] = units
    interfaces["joint.state"] = {"endpoint": "/joint_states"}
    interfaces["runtime.status"] = {"endpoint": "/runtime/status"}
    return {
        "robot_model": model,
        "runtime": {"provider": "so101_robot", "interface_description": {"interfaces": interfaces}},
        "teleoperation": {
            "enabled": True,
            "active_device": "so101_leader",
            "target": {
                "group": "arm",
                "gripper_group": "gripper",
                "interfaces": {key: f"motion.arm.{key}" for key in types},
            },
        },
    }


def test_public_target_order_and_ratio_are_not_radians():
    target = resolve_public_target(config())
    assert target["arm"] == ["1", "2", "3", "4", "5"]
    limits = {"min": -0.5, "max": 1.5}
    assert map_gripper_ratio(0, target["closed"], target["open"], limits) == -0.5
    assert map_gripper_ratio(1, target["closed"], target["open"], limits) == 1.5
    assert map_gripper_ratio(0.5, 1.6, -0.6, limits) == pytest.approx(0.5)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_ratio_rejected(value):
    with pytest.raises(ValueError):
        map_gripper_ratio(value, -0.6, 1.6, {"min": -2, "max": 2})


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_group", "other"),
        ("joint_names", ["2", "1"]),
        ("base_frame", "wrong"),
        ("message_type", "std_msgs/msg/String"),
    ],
)
def test_target_conflict_rejected(field, value):
    item = config()
    item["runtime"]["interface_description"]["interfaces"]["motion.arm.pose"][field] = value
    with pytest.raises(ValueError):
        resolve_public_target(item)


def test_safety_cannot_enlarge_runtime_limits():
    item = config()
    item["teleoperation"]["safety"] = {"joint_limits": {"1": {"min": -3, "max": 3}}}
    with pytest.raises(ValueError, match="enlarge"):
        resolve_public_target(item)


def test_safety_margin_narrows_relative_to_public_limits():
    item = config()
    item["teleoperation"]["safety"] = {"joint_limits_margin": 0.05}
    target = resolve_public_target(item)
    for _name, limit in target["limits"].items():
        assert limit["min"] == pytest.approx(-1.95)
        assert limit["max"] == pytest.approx(1.95)


def test_safety_margin_that_leaves_no_travel_rejected():
    item = config()
    item["teleoperation"]["safety"] = {"joint_limits_margin": 3.0}
    with pytest.raises(ValueError, match="no travel"):
        resolve_public_target(item)


def test_safety_margin_must_be_non_negative():
    item = config()
    item["teleoperation"]["safety"] = {"joint_limits_margin": -0.1}
    with pytest.raises(ValueError, match="non-negative"):
        resolve_public_target(item)


class Node:
    def __init__(self):
        self.now = 10.0
        self.warnings = []

    def get_logger(self):
        return SimpleNamespace(warn=self.warnings.append)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self.now * 1e9)))

    def create_subscription(self, *_args):
        return object()


def leader_message(position=None):
    msg = JointState()
    msg.header.stamp.sec = 10
    msg.header.frame_id = "leader_radians_gripper_ratio_v1"
    msg.name = ["1", "6"]
    msg.position = [0.25, 0.75] if position is None else position
    return msg


def test_ros_leader_fresh_complete_input_disconnect_and_fresh_rearm(monkeypatch):
    from robot_teleop.devices import leader_topic

    clock = [20.0]
    monkeypatch.setattr(leader_topic.time, "monotonic", lambda: clock[0])
    device = LeaderTopicDevice(
        {
            "joint_mapping": {"1": "shoulder", "6": "jaw"},
            "input_gripper_joint": "6",
            "input_stale_s": 0.2,
            "source_topic": "/input",
        },
        Node(),
    )
    device.connect()
    device._on_sample(leader_message())
    assert device.get_joint_targets() == {"shoulder": 0.25, "jaw": 0.75}
    clock[0] += 0.3
    device._node.now = 10.1
    assert device.get_joint_targets() == {}
    assert len(device._node.warnings) == 1
    warning = device._node.warnings[0]
    assert "stale on /input" in warning
    assert "last_valid_received_age=0.300s" in warning
    assert "last_valid_source_age=0.100s" in warning
    assert "timeout=0.200s" in warning
    assert "last_invalid_reason=none" in warning
    for _ in range(3):
        assert device.get_joint_targets() == {}
    assert len(device._node.warnings) == 1
    device.emergency_stop()
    device._on_sample(leader_message())
    assert device.get_joint_targets() == {}  # old timestamp cannot replay after rearm
    assert device.last_invalid_reason == "non-increasing source timestamp"
    msg = leader_message()
    msg.header.stamp.nanosec = 100_000_000
    device._node.now = 10.1
    device._on_sample(msg)
    assert device.get_joint_targets()
    device.disconnect()
    assert device.get_joint_targets() == {}


@pytest.mark.parametrize(
    "positions,reason",
    [
        ([0.1], "joint name/position length mismatch"),
        ([float("nan"), 0.5], "non-finite joint position"),
        ([0.1, 2.0], "gripper ratio outside [0,1]"),
    ],
)
def test_ros_leader_rejects_partial_and_invalid_units(positions, reason):
    device = LeaderTopicDevice(
        {"joint_mapping": {"1": "1", "6": "6"}, "input_gripper_joint": "6", "input_stale_s": 0.2}, Node()
    )
    for _ in range(3):
        device._on_sample(leader_message(positions))
        assert device.get_joint_targets() == {}
    assert device.last_invalid_reason == reason
    assert len(device._node.warnings) == 1
    assert f"last_invalid_reason={reason}" in device._node.warnings[0]


def test_ros_leader_stale_log_includes_invalid_frame_reason_and_resets_on_valid_input(monkeypatch):
    from robot_teleop.devices import leader_topic

    clock = [20.0]
    monkeypatch.setattr(leader_topic.time, "monotonic", lambda: clock[0])
    device = LeaderTopicDevice(
        {"joint_mapping": {"1": "1", "6": "6"}, "input_gripper_joint": "6", "input_stale_s": 0.2}, Node()
    )
    device._on_sample(leader_message())
    invalid = leader_message()
    invalid.header.frame_id = "wrong_units"
    for _ in range(3):
        device._on_sample(invalid)
        assert device.get_joint_targets() == {"1": 0.25, "6": 0.75}
    assert device._node.warnings == []
    assert device.last_invalid_reason == "unexpected frame_id: 'wrong_units'"

    clock[0] = 20.3
    device._node.now = 10.4
    assert device.get_joint_targets() == {}
    warning = device._node.warnings[0]
    assert "last_valid_received_age=0.300s" in warning
    assert "last_valid_source_age=0.400s" in warning
    assert "last_invalid_reason=unexpected frame_id: 'wrong_units'" in warning
    for _ in range(3):
        device._on_sample(invalid)
        assert device.get_joint_targets() == {}
    assert len(device._node.warnings) == 1

    valid = leader_message()
    valid.header.stamp.nanosec = 400_000_000
    device._on_sample(valid)
    assert device.get_joint_targets() == {"1": 0.25, "6": 0.75}
    assert device.last_invalid_reason == ""
    clock[0] = 20.6
    device._node.now = 10.7
    assert device.get_joint_targets() == {}
    assert len(device._node.warnings) == 2
    assert "last_invalid_reason=none" in device._node.warnings[-1]


def test_leader_checks_source_age_even_with_fresh_receipt():
    node = Node()
    device = LeaderTopicDevice(
        {"joint_mapping": {"1": "1", "6": "6"}, "input_gripper_joint": "6", "input_stale_s": 0.2}, node
    )
    device._on_sample(leader_message())
    node.now = 10.3
    assert device.get_joint_targets() == {}


def test_leader_rearm_excludes_queued_pre_request_samples():
    node = Node()
    device = LeaderTopicDevice(
        {"joint_mapping": {"1": "1", "6": "6"}, "input_gripper_joint": "6", "input_stale_s": 0.2}, node
    )
    device._on_sample(leader_message())
    node.now = 10.1
    device.prepare_rearm()
    queued = leader_message()
    queued.header.stamp.nanosec = 50_000_000
    device._on_sample(queued)
    assert device.get_joint_targets() == {}
    node.now = 10.15
    queued.header.stamp.nanosec = 150_000_000
    device._on_sample(queued)
    assert device.get_joint_targets()


class Backend:
    def __init__(self):
        self.is_enabled = False
        self._requested_enabled = False
        self._home_pending = False
        self.commands = []
        self.stops = 0
        self.stop_pending = False

    def enable(self):
        self.is_enabled = True
        return True

    def disable(self):
        self.is_enabled = False
        self.stops += 1

    def servo(self, **kwargs):
        self.commands.append(kwargs)

    def servo_pose(self, **kwargs):
        self.commands.append(kwargs)

    def keepalive(self):
        pass


def test_phone_public_backend_pose_loss_and_explicit_rearm():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from robot_teleop.phone.phone_device import PhoneDevice

    device = PhoneDevice({"managed_teleop": True, "cartesian_solver": "runtime"})
    device.servo_client = Backend()
    action = {"phone.enabled": True, "phone.pos": np.zeros(3), "phone.rot": Rotation.identity(), "phone.raw_inputs": {}}
    source = SimpleNamespace(action=action)
    device._phone_impl = SimpleNamespace(
        get_action=lambda: source.action,
        consume_stop_request=lambda: None,
        require_release=lambda *_args, **_kwargs: None,
    )
    device._is_connected = True
    device._first_state_received = True
    device.get_joint_targets()
    device.get_joint_targets()
    source.action = {**action, "phone.pos": np.array([0.02, 0.0, 0.0])}
    assert "6" in device.get_joint_targets()
    assert device.servo_client.commands[-1]["position"] != (0.0, 0.0, 0.0)
    source.action = None
    assert device.get_joint_targets() == {}
    assert device.servo_client.stops == 1
    source.action = action
    device.get_joint_targets()
    assert not device.servo_client.is_enabled
    source.action = {**action, "phone.enabled": False}
    device.get_joint_targets()
    source.action = action
    device.get_joint_targets()
    assert device.servo_client.is_enabled


@pytest.mark.parametrize("mode", ["pose", "velocity"])
def test_vr_managed_modes_release_and_rearm_are_explicit(mode):
    import numpy as np
    from scipy.spatial.transform import Rotation

    from robot_teleop.vr_teleop import VRTeleopNode, _ArmState, _ControllerData

    events = []
    node = SimpleNamespace(
        _estop_active=False,
        _so101_started=False,
        _so101_recalib_inflight=False,
        _so101_home_inflight=False,
        _managed_release_required=True,
        _so101_stop_pending=False,
        _so101_stalled=False,
        _controller_side="right",
        _arm_state={"right": _ArmState()},
        _so101_input_mode=mode,
        _secondary_prev=False,
        _managed_lease_pub=SimpleNamespace(publish=lambda _msg: events.append("lease")),
    )

    def stop(reason):
        node._so101_started = False
        events.append("stop")

    def start():
        node._so101_started = True
        events.append("start")

    node._stop_so101_servo = stop
    node._recalibrate_so101_baseline = start
    node._control_so101_pose = lambda _ctrl: events.append("pose")
    node._compute_velocities = lambda *_: (np.zeros(3), np.zeros(3))
    node._publish_so101 = lambda *_: events.append("velocity")
    ctrl = _ControllerData(np.zeros(3), Rotation.identity(), enabled=True)
    VRTeleopNode._control_managed(node, ctrl)
    assert events == []  # held at connection is not an enable edge
    ctrl.enabled = False
    VRTeleopNode._control_managed(node, ctrl)
    ctrl.enabled = True
    VRTeleopNode._control_managed(node, ctrl)
    assert events == ["start"]
    VRTeleopNode._control_managed(node, ctrl)
    assert mode in events
    VRTeleopNode._control_managed(node, None)
    assert events[-1] == "stop"
    VRTeleopNode._control_managed(node, ctrl)
    assert events[-1] == "stop"
    ctrl.enabled = False
    VRTeleopNode._control_managed(node, ctrl)
    ctrl.enabled = True
    VRTeleopNode._control_managed(node, ctrl)
    assert events[-1] == "start"


@pytest.mark.parametrize("mode", ["joint", "cartesian"])
def test_gamepad_command_stale_input_and_explicit_rearm(monkeypatch, mode):
    from robot_teleop.devices import xbox_controller

    monkeypatch.setattr(XboxTeleopDevice, "_load_mapping", lambda *_: {"joint_mode": {"1": 0}})
    monkeypatch.setattr(XboxTeleopDevice, "_print_usage", lambda *_args, **_kwargs: None)
    clock = [20.0]
    monkeypatch.setattr(xbox_controller.time, "monotonic", lambda: clock[0])
    node = Node()
    device = XboxTeleopDevice(
        {
            "managed_teleop": True,
            "default_mode": mode,
            "arm_joint_names": ["1"],
            "gripper_joint_names": ["6"],
            "gripper_closed": -0.6,
            "gripper_open": 1.6,
        },
        node,
    )
    device.servo_client = Backend()
    state = leader_message([0.0, 0.5])
    device._joint_state_callback(state)
    joy = Joy()
    joy.header.stamp.sec = 10
    joy.axes = [0.5] * 8
    joy.buttons = [0] * 6
    device._process_joy_event(joy)
    joy.buttons[0] = 1
    device._process_joy_event(joy)
    targets = device.get_joint_targets()
    assert targets["6"] >= -0.6
    if mode == "joint":
        assert targets["1"] > 0.0
    else:
        assert device.servo_client.commands
        assert "1" not in targets
    clock[0] += 0.3
    assert device.get_joint_targets() == {}
    assert device.servo_client.stops == 1
    device._joint_state_callback(state)
    device._process_joy_event(joy)  # held A cannot rearm
    assert device.get_joint_targets() == {}
    joy.buttons[0] = 0
    device._process_joy_event(joy)
    joy.buttons[0] = 1
    device._process_joy_event(joy)
    assert device.get_joint_targets()


def load_input_class():
    path = Path(__file__).resolve().parents[2] / "robots/so101/so101_hardware/so101_hardware/leader_input.py"
    spec = importlib.util.spec_from_file_location("sdk_leader_input_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LeaderInput


def test_sdk_source_only_uses_readonly_lifecycle(tmp_path):
    events = []

    class Leader:
        def __init__(self, config):
            assert config.calibration_version == 1

        def connect(self):
            events.append("connect")
            return True

        def read(self):
            events.append("read")
            return {"1": {"position": 0.3}, "6": {"position": 0.7}}

        def disconnect(self):
            events.append("disconnect")

    path = tmp_path / "calibration.json"
    path.write_text("{}")
    sdk = SimpleNamespace(LeaderConfig=SimpleNamespace, LeaderArm=Leader)
    source = load_input_class()(
        {"calibration_version": 1, "calibration_file": str(path), "joint_order": ["1", "6"], "gripper_joint": "6"}, sdk
    )
    source.connect()
    assert source.read() == {"1": 0.3, "6": 0.7}
    source.disconnect()
    assert events == ["connect", "read", "disconnect"]


@pytest.mark.parametrize("version", [None, 0, 2, True])
def test_sdk_provisioning_ack_is_required(version):
    with pytest.raises(ValueError, match="explicit provisioned"):
        load_input_class()({"calibration_version": version})


@pytest.mark.parametrize("name", ["so101_leader", "xbox_controller", "phone_teleop", "vr_teleop"])
def test_real_launch_builder_has_no_generic_solver_or_follower_files(monkeypatch, name):
    from robot_config.launch_builders import teleop

    item = config()
    item["teleoperation"]["safety"] = {
        "joint_limits": {"6": {"min": -0.3, "max": 1.2}},
        "joint_limits_margin": 0.1,
    }
    item["teleoperation"].update(
        {
            "active_device": name,
            "input_config": str(
                Path(__file__).resolve().parents[2] / "robot_config/config/teleop/so101_leader_inputs.yaml"
            ),
        }
    )
    nodes = []
    monkeypatch.setattr(teleop, "Node", lambda **kwargs: nodes.append(kwargs) or kwargs)
    teleop.generate_teleop_nodes(copy.deepcopy(item))
    assert any(node["package"] == "robot_teleop" for node in nodes)
    assert not any(node["package"] in ("so101_motion", "moveit_servo") for node in nodes)
    assert "follower_calib_file" not in str(nodes)
    assert "urdf_path" not in str(nodes)
    params = next(node["parameters"][0] for node in nodes if node["package"] == "robot_teleop")
    limits = json.loads(params["joint_limits"])
    assert limits == resolve_public_target(item)["limits"]
    assert limits["6"]["min"] == pytest.approx(-0.2)
    assert limits["6"]["max"] == pytest.approx(1.1)
    if name == "vr_teleop":
        assert params["so101_gripper_topic"] == "/motion/arm/joints"
        assert params["so101_gripper_closed"] == -0.6
        assert params["so101_gripper_open"] == 1.6


def test_real_runtime_projection_composes_executor_and_all_inputs(monkeypatch):
    import yaml

    from robot_config.launch_builders import teleop
    from robot_runtime.interface_description import build_description
    from robot_runtime.launch_support import render_robot_description
    from so101_robot import teleoperation

    root = Path(__file__).resolve().parents[2]
    path = root / "robots/so101/so101_robot/profiles/so101_single_arm.yaml"
    profile = yaml.safe_load(path.read_text())
    xml = render_robot_description(profile, path, True)
    descriptor = build_description(profile, simulated=True, robot_description=xml)
    item = yaml.safe_load((root / "robot_config/config/robots/so101_single_arm.yaml").read_text())["robot"]
    item["robot_model"] = descriptor["model"]
    item["runtime"]["interface_description"] = descriptor
    item["teleoperation"]["input_config"] = str(root / "robot_config/config/teleop/so101_leader_inputs.yaml")
    monkeypatch.setattr(teleoperation, "Node", lambda **kwargs: kwargs)
    executor = teleoperation.generate_teleoperation_nodes(profile, descriptor, {"robot_description": xml})[0]
    params = executor["parameters"][0]
    assert params["managed_teleop"] is True
    assert params["robot_description"] == xml
    assert params["joint_intent_topic"] == descriptor["interfaces"]["motion.arm.joints"]["endpoint"]
    monkeypatch.setattr(teleop, "Node", lambda **kwargs: kwargs)
    for device in ("so101_leader", "xbox_controller", "phone_teleop", "vr_teleop"):
        item["teleoperation"]["active_device"] = device
        nodes = teleop.generate_teleop_nodes(item)
        assert any(node["package"] == "robot_teleop" for node in nodes)
