"""Public descriptions are generated, versioned and distinct from observed state."""

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from builtin_interfaces.msg import Time
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image

from robot_runtime.interface_description import (
    build_description,
    description_digest,
    load_description,
    merge_peripherals,
    validate_description,
    write_description,
)
from robot_runtime.interface_monitor import InterfaceMonitor
from robot_runtime.synthetic_perception import SyntheticStreams
from robot_runtime.wait_for_runtime import validate_endpoint_graph, validate_runtime_snapshot


@pytest.fixture
def profile():
    return {
        "runtime": {"name": "test_robot", "version": "1.0.0", "instance_id": "test-1"},
        "capabilities": {"joint.state": {"rate_hz": 50}, "motion.ik": {"endpoints": ["/motion/compute_ik"]}},
        "joints": ["joint_1"],
        "hardware": {"port": "/dev/private-device", "calib_file": "/private/path"},
        "peripherals": [
            {
                "type": "camera",
                "name": "front",
                "driver": "opencv",
                "index": 0,
                "width": 16,
                "height": 12,
                "fps": 20,
                "pixel_format": "mjpeg2rgb",
                "frame_id": "front_optical",
            }
        ],
    }


def test_public_projection_and_roundtrip(profile, tmp_path):
    original = copy.deepcopy(profile)
    descriptor = build_description(profile, simulated=True)
    image = descriptor["interfaces"]["camera.front.color"]
    assert image["configured_profile"] == {"width": 16, "height": 12, "fps": 20, "encoding": "rgb8"}
    assert image["supported_profiles"] is None
    assert descriptor["states"] == {}
    assert descriptor["robot"]["id"] == "test-1"
    assert descriptor["execution"] == "simulated"
    assert descriptor["interfaces"]["motion.compute_ik"]["message_type"] == "ibrobot_msgs/srv/ComputeIk"
    encoded = json.dumps(descriptor)
    assert "/private" not in encoded and "opencv" not in encoded and "/dev/" not in encoded
    destination = tmp_path / "interfaces.yaml"
    write_description(descriptor, destination)
    assert load_description(destination) == descriptor
    assert profile == original


def test_partial_overrides_and_disable_are_effective(profile):
    original = copy.deepcopy(profile["peripherals"])
    merged = merge_peripherals(original, [{"type": "camera", "name": "front", "width": 32}])
    descriptor = build_description(profile, merged)
    assert descriptor["interfaces"]["camera.front.color"]["configured_profile"]["height"] == 12
    assert descriptor["interfaces"]["camera.front.color"]["configured_profile"]["width"] == 32
    assert original == profile["peripherals"]
    disabled = merge_peripherals(original, [{"type": "camera", "name": "front", "enabled": False}])
    assert not any(name.startswith("camera.") for name in build_description(profile, disabled)["interfaces"])


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda doc: doc.update(schema_version=2), "schema_version"),
        (lambda doc: doc["interfaces"]["camera.front.color"].update(driver="opencv"), "Additional properties"),
        (lambda doc: doc["interfaces"]["camera.front.color"]["configured_profile"].update(width=0), "width"),
        (
            lambda doc: doc["interfaces"]["camera.front.color"].update(message_type="sensor_msgs/msg/CameraInfo"),
            "should not be valid",
        ),
        (
            lambda doc: doc["interfaces"]["camera.front.color"].update(camera_info_topic="/absent"),
            "CameraInfo interface",
        ),
    ],
)
def test_schema_rejects_invalid_description(profile, mutate, reason):
    doc = build_description(profile)
    mutate(doc)
    doc["digest"] = description_digest(doc)
    with pytest.raises(ValueError, match=reason):
        validate_description(doc)


def test_digest_and_unknown_states(profile):
    doc = build_description(profile)
    digest = doc["digest"]
    state = {"state": "unknown", "observed_profile": None, "observed_frame_id": None, "last_seen": None, "detail": ""}
    doc["states"]["camera.front.color"] = state
    assert description_digest(doc) == digest
    state.update(state="ready", last_seen=time.time())
    with pytest.raises(ValueError, match="observed_profile"):
        validate_description(doc)
    state["observed_profile"] = {"width": 16, "height": 12, "fps": None, "encoding": "rgb8"}
    validate_description(doc)
    doc["interfaces"]["camera.front.color"]["configured_profile"]["width"] = 32
    with pytest.raises(ValueError, match="digest"):
        validate_description(doc)


def test_supported_modes_not_inferred(profile):
    periph = profile["peripherals"][0]
    periph["supported_profiles"] = {"color": [{"width": 16, "height": 12, "fps": 20, "encoding": "rgb8"}]}
    assert (
        build_description(profile)["interfaces"]["camera.front.color"]["supported_profiles"]
        == periph["supported_profiles"]["color"]
    )
    periph["width"] = 32
    with pytest.raises(ValueError, match="supported_profiles"):
        build_description(profile)


def test_depth_and_livox_wire_types(profile):
    profile["peripherals"] = [
        {
            "type": "camera",
            "name": "front",
            "driver": "realsense",
            "width": 32,
            "height": 24,
            "depth_width": 16,
            "depth_height": 12,
            "align_depth": True,
            "streams": ["color", "depth", "pointcloud"],
        },
        {"type": "lidar", "name": "mid360", "driver": "livox_mid360", "xfer_format": 1},
    ]
    interfaces = build_description(profile)["interfaces"]
    assert interfaces["camera.front.depth"]["configured_profile"]["encoding"] == "16UC1"
    assert interfaces["camera.front.depth"]["configured_profile"]["width"] == 16
    assert interfaces["camera.front.aligned_depth"]["configured_profile"]["width"] == 32
    assert interfaces["camera.front.points"]["message_type"] == "sensor_msgs/msg/PointCloud2"
    assert interfaces["lidar.mid360.points"]["message_type"] == "livox_ros_driver2/msg/CustomMsg"
    profile["peripherals"][0]["streams"] = ["depth"]
    profile["peripherals"][1]["xfer_format"] = 0
    interfaces = build_description(profile)["interfaces"]
    assert "camera.front.color" not in interfaces and "camera.front.aligned_depth" not in interfaces
    assert interfaces["lidar.mid360.points"]["message_type"] == "sensor_msgs/msg/PointCloud2"


class FakeNode:
    def __init__(self):
        self.messages, self.timers, self.offered, self.subscribed = {}, [], {}, {}

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscribed[topic] = (message_type, qos)
        return None

    def create_publisher(self, message_type, topic, qos):
        self.offered[topic] = (message_type, qos)
        return SimpleNamespace(publish=lambda msg: self.messages.update({topic: msg}))

    def create_timer(self, period, callback):
        self.timers.append((period, callback))
        return None

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=int(time.time()))))


def test_synthetic_streams_match_the_descriptor(profile):
    doc = build_description(profile, simulated=True)
    node = FakeNode()
    SyntheticStreams(node, doc)
    for period, callback in node.timers:
        assert period == 1 / 20
        callback()
    image = node.messages["/camera/front/image_raw"]
    assert (image.width, image.height, image.encoding, image.step) == (16, 12, "rgb8", 48)
    assert len(image.data) == 16 * 12 * 3
    assert image.header.frame_id == "front_optical"
    info = node.messages["/camera/front/camera_info"]
    assert (info.width, info.height) == (16, 12)
    assert info.k[0] != 0


def test_unknown_configuration_is_not_fabricated(profile):
    doc = build_description(profile)
    doc["interfaces"]["camera.front.color"]["configured_profile"] = None
    doc["digest"] = description_digest(doc)
    validate_description(doc)
    with pytest.raises(ValueError, match="configured image profile"):
        SyntheticStreams(FakeNode(), doc)
    monitor = InterfaceMonitor(FakeNode(), doc)
    image = Image(width=8, height=8, encoding="mono8", step=8, data=bytes(64))
    image.header.frame_id = "front_optical"
    for _ in range(3):
        monitor.observe("camera.front.color", doc["interfaces"]["camera.front.color"], image)
    assert monitor.states()["camera.front.color"]["state"] == "ready"


def test_observed_metadata_and_staleness(profile, monkeypatch):
    doc = build_description(profile)
    monitor = InterfaceMonitor(FakeNode(), doc)
    now = [0.0]
    monkeypatch.setattr("robot_runtime.interface_monitor.time.monotonic", lambda: now[0])
    image = Image(width=16, height=12, encoding="rgb8", step=48, data=bytes(576))
    image.header.frame_id = "front_optical"
    spec = doc["interfaces"]["camera.front.color"]
    monitor.observe("camera.front.color", spec, image)
    assert monitor.states()["camera.front.color"]["observed_profile"]["fps"] is None
    for _ in range(3):
        now[0] += 0.05
        monitor.observe("camera.front.color", spec, image)
    state = monitor.states()["camera.front.color"]
    assert state["state"] == "ready"
    assert state["observed_profile"]["fps"] == pytest.approx(20)
    image.encoding = "bgr8"
    monitor.observe("camera.front.color", spec, image)
    assert monitor.states()["camera.front.color"]["state"] == "mismatch"
    now[0] += 3
    assert monitor.states()["camera.front.color"]["state"] == "stale"


def test_waiter_validates_identity_and_graph(profile):
    doc = build_description(profile)
    status = SimpleNamespace(
        runtime_name="test_robot",
        runtime_version="1.0.0",
        lifecycle="ACTIVE",
        capabilities=["runtime.status"],
        interface_description_json=json.dumps(doc),
    )
    validate_runtime_snapshot(status, runtime_name="test_robot", instance_id="test-1")
    with pytest.raises(ValueError, match="identity"):
        validate_runtime_snapshot(status, runtime_name="wrong_robot")
    with pytest.raises(ValueError, match="unknown"):
        validate_runtime_snapshot(status, runtime_name="test_robot", interfaces=["camera.front.color"])
    with pytest.raises(ValueError, match="ROS graph"):
        validate_endpoint_graph(SimpleNamespace(get_service_names_and_types=lambda: []), doc, ["runtime.get_status"])


def test_graph_qos_uses_requested_offered_compatibility(profile):
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    doc = build_description(profile)
    spec = doc["interfaces"]["camera.front.color"]
    spec["qos"]["reliability"] = "reliable"
    info = SimpleNamespace(
        topic_type=spec["message_type"],
        qos_profile=QoSProfile(
            depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE
        ),
    )
    node = SimpleNamespace(get_publishers_info_by_topic=lambda _: [info])
    with pytest.raises(ValueError, match="ROS graph"):
        validate_endpoint_graph(node, doc, ["camera.front.color"])
    spec["qos"]["reliability"] = "best_effort"
    validate_endpoint_graph(node, doc, ["camera.front.color"])


def test_shipped_runtime_profiles_export_valid_public_descriptions():
    from robot_runtime.profile import load_profile

    root = Path(__file__).resolve().parents[2]
    paths = list((root / "robots").glob("*/*_robot/profiles/*.yaml"))
    assert paths
    for path in paths:
        profile = load_profile(path)
        validate_description(build_description(profile, simulated=True))


def test_monitor_subscription_mirrors_declared_qos(profile):
    doc = build_description(profile)
    # A latched reliable topic such as robot.description is absent from the
    # minimal fixture, so inject one: robot_state_publisher publishes it
    # reliable + transient_local, and a late joiner must subscribe the same
    # way or the historical sample never arrives (stuck "No sample received").
    doc["interfaces"]["robot.description"] = {
        "endpoint": "/robot_description",
        "message_type": "std_msgs/msg/String",
        "kind": "topic",
        "direction": "publish",
        "capability": "joint.state",
        "qos": {"reliability": "reliable", "durability": "transient_local", "history": "keep_last", "depth": 10},
    }
    node = FakeNode()
    InterfaceMonitor(node, doc)
    description_qos = node.subscribed["/robot_description"][1]
    assert description_qos.reliability == ReliabilityPolicy.RELIABLE
    assert description_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    camera_qos = node.subscribed["/camera/front/image_raw"][1]
    assert camera_qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert camera_qos.durability == DurabilityPolicy.VOLATILE
