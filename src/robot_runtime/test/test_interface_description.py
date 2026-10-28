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
    InterfaceDescriptionError,
    _teleoperation_interfaces,
    build_description,
    description_digest,
    description_sim_platform,
    load_description,
    merge_peripherals,
    peripheral_interfaces,
    validate_description,
    write_description,
)
from robot_runtime.interface_monitor import InterfaceMonitor
from robot_runtime.synthetic_perception import SyntheticStreams
from robot_runtime.wait_for_runtime import validate_endpoint_graph, validate_runtime_snapshot


@pytest.mark.parametrize("stale", [0.0, -0.1, 1.001, float("nan"), float("inf"), True])
def test_teleoperation_deadline_rejects_out_of_contract_profile_values(stale):
    profile = {
        "teleoperation": {"enabled": True, "target_group": "arm", "endpoint_prefix": "/arm", "command_stale_s": stale}
    }
    model = {"joint_groups": {"arm": ["1"], "all": ["1", "6"]}, "frames": {"base_link": "base", "ee_link": "tool"}}
    with pytest.raises(InterfaceDescriptionError, match="command_stale_s"):
        _teleoperation_interfaces(profile, model)


@pytest.mark.parametrize("stale", [0.2, 0.5, 1.0])
def test_teleoperation_deadline_projects_hardware_profile_budget(stale):
    profile = {
        "teleoperation": {"enabled": True, "target_group": "arm", "endpoint_prefix": "/arm", "command_stale_s": stale}
    }
    model = {"joint_groups": {"arm": ["1"], "all": ["1", "6"]}, "frames": {"base_link": "base", "ee_link": "tool"}}
    interfaces = _teleoperation_interfaces(profile, model)
    assert all(interface["command_stale_s"] == stale for interface in interfaces.values())


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
    descriptor = build_description(profile, use_sim=True)
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


@pytest.mark.parametrize(
    ("use_sim", "sim_platform", "execution"),
    [
        (False, None, "physical"),
        (True, "sdk", "simulated"),
        (True, "gazebo", "simulated"),
        (True, "mujoco", "simulated"),
    ],
)
def test_description_records_sim_platform_and_execution(profile, use_sim, sim_platform, execution):
    doc = build_description(profile, use_sim=use_sim, sim_platform=sim_platform)
    assert doc.get("sim_platform") == sim_platform  # absent when physical
    assert doc["execution"] == execution
    validate_description(doc)  # build already validated; the schema accepts the field


def test_simulated_profile_key_stays_a_use_sim_alias(profile):
    assert build_description(profile, use_sim=True)["sim_platform"] == "sdk"
    assert "sim_platform" not in build_description(profile, use_sim=False)
    profile_with_sim = {**profile, "simulated": True}
    assert build_description(profile_with_sim)["sim_platform"] == "sdk"


def test_physical_execution_with_platform_rejected(profile):
    with pytest.raises(InterfaceDescriptionError, match="conflicting execution selection"):
        build_description(profile, use_sim=False, sim_platform="sdk")


def test_unknown_sim_platform_rejected(profile):
    with pytest.raises(InterfaceDescriptionError, match="unknown sim_platform 'teleport'"):
        build_description(profile, use_sim=True, sim_platform="teleport")


def test_digest_covers_sim_platform(profile):
    doc = build_description(profile, use_sim=True, sim_platform="sdk")
    doc["sim_platform"] = "gazebo"
    with pytest.raises(ValueError, match="digest"):
        validate_description(doc)


def test_mujoco_description_validates_and_keeps_a_stable_digest(profile):
    # the schema enum and the reader accept mujoco (D1), and re-digesting the
    # same document reproduces the digest exactly as for the older platforms
    doc = build_description(profile, use_sim=True, sim_platform="mujoco")
    assert description_sim_platform(doc) == "mujoco"
    assert doc["digest"] == description_digest(doc)
    validate_description(doc)
    again = build_description(profile, use_sim=True, sim_platform="mujoco")
    assert again["digest"] == doc["digest"]


def test_mujoco_depth_stream_dims_come_from_the_rendered_camera(profile):
    # The plugin renders one MJCF camera at width x height for every stream
    # (D5): unequal device depth overrides must not leak into the mujoco
    # contract (finding 7).
    peripherals = [
        {
            "type": "camera",
            "name": "front",
            "driver": "realsense",
            "width": 16,
            "height": 12,
            "depth_width": 8,
            "depth_height": 6,
            "align_depth": True,
            "streams": ["color", "depth"],
        }
    ]
    descriptor = build_description({**profile, "peripherals": peripherals}, use_sim=True, sim_platform="mujoco")
    for stream in ("camera.front.depth", "camera.front.aligned_depth"):
        configured = descriptor["interfaces"][stream]["configured_profile"]
        assert (configured["width"], configured["height"]) == (16, 12), f"{stream} uses the rendered dims"
    # gazebo renders its rgbd sensor at the colour dims/rate too
    # (runtime-deployment-peripherals D8): the depth streams advertise what
    # the sensor actually renders, never the device's depth_* overrides.
    gazebo = build_description({**profile, "peripherals": peripherals}, use_sim=True, sim_platform="gazebo")
    for stream in ("camera.front.depth", "camera.front.aligned_depth"):
        configured = gazebo["interfaces"][stream]["configured_profile"]
        assert (configured["width"], configured["height"], configured["fps"]) == (16, 12, 30)
    # off every simulator the device-specific depth dims are still honoured
    sdk = build_description({**profile, "peripherals": peripherals}, use_sim=True, sim_platform="sdk")
    assert sdk["interfaces"]["camera.front.depth"]["configured_profile"]["width"] == 8


def test_pre_platform_description_stays_valid_and_defaults_from_execution(profile):
    # A snapshot exported before the sim_platform field existed keeps its
    # digest and validates unchanged; readers default it from execution
    # without mutating the document (runtime-simulation-platform D7).
    doc = build_description(profile, use_sim=True, sim_platform="sdk")
    del doc["sim_platform"]
    doc["digest"] = description_digest(doc)
    validate_description(doc)
    assert description_sim_platform(doc) == "sdk"
    assert "sim_platform" not in doc  # the reader must not insert the default
    physical = build_description(profile, use_sim=False)
    physical["digest"] = description_digest(physical)
    validate_description(physical)
    assert description_sim_platform(physical) is None
    assert "sim_platform" not in physical


def test_description_sim_platform_rejects_unknown_value(profile):
    doc = build_description(profile, use_sim=True, sim_platform="sdk")
    doc["sim_platform"] = "teleport"
    doc["digest"] = description_digest(doc)
    with pytest.raises(ValueError, match="sim_platform"):
        validate_description(doc)  # the schema enum rejects the unknown value
    with pytest.raises(InterfaceDescriptionError, match="unknown value 'teleport'"):
        description_sim_platform(doc)


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


# ---------------------------------------------------------------------------
# lekiwi-gazebo-runtime-platform D5/D6: the lidar contract on Gazebo
# ---------------------------------------------------------------------------


def _lidar_profile(enabled_scan=True, fast_lio=None):
    return {
        "runtime": {"name": "test_robot", "version": "1.0.0"},
        "capabilities": {},
        "joints": ["1"],
        "peripherals": [
            {
                "type": "lidar",
                "name": "mid360",
                "driver": "livox_mid360",
                "xfer_format": 1,
                "frame_id": "livox_frame",
                "pointcloud_topic": "/livox/lidar",
                "imu_topic": "/livox/imu",
                **({"scan_converter": {"enabled": enabled_scan, "scan_topic": "/scan"}} if enabled_scan else {}),
            }
        ],
        "fast_lio": fast_lio or {},
    }


def test_gazebo_advertises_the_point_cloud_regardless_of_xfer_format():
    profile = _lidar_profile(enabled_scan=False)
    gazebo = build_description(profile, use_sim=True, sim_platform="gazebo")["interfaces"]
    # the physical driver's CustomMsg transfer cannot be produced by the
    # simulator: the cloud is advertised as PointCloud2 (D5)
    assert gazebo["lidar.mid360.points"]["message_type"] == "sensor_msgs/msg/PointCloud2"
    assert gazebo["lidar.mid360.imu"]["message_type"] == "sensor_msgs/msg/Imu"
    # hardware and sdk keep the driver's wire format
    hardware = build_description(profile, use_sim=False)["interfaces"]
    assert hardware["lidar.mid360.points"]["message_type"] == "livox_ros_driver2/msg/CustomMsg"
    sdk = build_description(profile, use_sim=True, sim_platform="sdk")["interfaces"]
    assert sdk["lidar.mid360.points"]["message_type"] == "livox_ros_driver2/msg/CustomMsg"


def test_gazebo_scan_advertised_iff_its_source_runs():
    """The scan's source is FAST-LIO's registered body cloud: on Gazebo the
    scan is advertised only when lidar odometry runs (D5); hardware and sdk
    keep the scan_converter-driven advertisement."""
    off = build_description(_lidar_profile(fast_lio={"enabled": False}), use_sim=True, sim_platform="gazebo")
    assert not any(key.startswith("lidar.mid360.scan") for key in off["interfaces"])

    on = build_description(
        _lidar_profile(fast_lio={"enabled": True, "sim_params_file": "/tmp/s.yaml"}),
        use_sim=True,
        sim_platform="gazebo",
    )
    assert on["interfaces"]["lidar.mid360.scan"]["endpoint"] == "/scan"
    # its source stream is advertised with it
    assert on["interfaces"]["localization.cloud"]["endpoint"] == "/cloud_registered_body"

    for use_sim, sim_platform in ((False, None), (True, "sdk")):
        other = build_description(
            _lidar_profile(fast_lio={"enabled": False}), use_sim=use_sim, sim_platform=sim_platform
        )
        assert other["interfaces"]["lidar.mid360.scan"]["endpoint"] == "/scan", (use_sim, sim_platform)


def test_gazebo_scan_rule_covers_every_lidar_with_the_converter_enabled():
    profile = _lidar_profile(fast_lio={"enabled": False})
    interfaces = peripheral_interfaces(profile["peripherals"], sim_platform="gazebo", scan_source_runs=False)
    assert not any(key.startswith("lidar.mid360.scan") for key in interfaces)
    interfaces = peripheral_interfaces(profile["peripherals"], sim_platform="gazebo", scan_source_runs=True)
    assert interfaces["lidar.mid360.scan"]["endpoint"] == "/scan"


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
    doc = build_description(profile, use_sim=True)
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


def test_waiter_declared_interfaces_check_declaration_only(profile):
    """--declared-interfaces gates on the runtime's declaration, not on
    readiness: a missing interface is a named error, an interface that
    exists but has published no sample yet passes (the shared readiness
    barrier must not block on a slow driver or a late simulator)."""
    doc = build_description(profile)
    status = SimpleNamespace(
        runtime_name="test_robot",
        runtime_version="1.0.0",
        lifecycle="ACTIVE",
        capabilities=["runtime.status"],
        interface_description_json=json.dumps(doc),
    )
    # every camera interface exists, none has a ready state
    assert doc["states"] == {}
    validate_runtime_snapshot(
        status,
        runtime_name="test_robot",
        declared_interfaces=["camera.front.color", "camera.front.color_info"],
    )
    with pytest.raises(ValueError, match="runtime does not declare interface 'camera.front.aligned_depth'"):
        validate_runtime_snapshot(status, runtime_name="test_robot", declared_interfaces=["camera.front.aligned_depth"])
    # a non-topic or non-publish interface is not a valid declaration gate
    with pytest.raises(ValueError, match="must be a published topic"):
        validate_runtime_snapshot(status, runtime_name="test_robot", declared_interfaces=["runtime.get_status"])


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
        validate_description(build_description(profile, use_sim=True))


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


def test_synthetic_perception_publishes_an_unstamped_message(profile):
    """A declared sensor interface need not be a stamped message.

    Assuming a header raised AttributeError inside the publish timer, which
    tore down the synthetic publisher and silenced every other topic it served
    — so the visible failure was an unrelated camera interface.
    """
    from std_msgs.msg import Float64MultiArray

    from robot_runtime.synthetic_perception import SyntheticStreams

    published: list[object] = []

    class _Pub:
        def publish(self, message):
            published.append(message)

    class _Node:
        def create_publisher(self, *_args, **_kwargs):
            return _Pub()

        def create_timer(self, _period, callback):
            callback()
            return object()

        def get_clock(self):
            raise AssertionError("an unstamped message must not reach the clock")

    descriptor = build_description(profile, use_sim=True)
    # Only the headerless interface, so any use of the clock is this one's.
    descriptor["interfaces"] = {
        "touch.hands": {
            "capability": "perception.touch",
            "kind": "topic",
            "direction": "publish",
            "endpoint": "/aimdk/hand_touch",
            "message_type": "std_msgs/msg/Float64MultiArray",
            "rate_hz": 10.0,
            "qos": {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 10},
        }
    }
    descriptor["digest"] = description_digest(descriptor)
    SyntheticStreams(_Node(), descriptor)
    assert published and isinstance(published[0], Float64MultiArray)


def test_camera_frames_come_from_the_shared_advertised_frames_helper(profile, monkeypatch):
    """5b.3/D: the description must not re-derive the frame convention."""
    import robot_runtime.interface_description as description_module

    def sentinel(_periph):
        return ("SENTINEL_IMAGE_FRAME", "SENTINEL_INFO_FRAME")

    monkeypatch.setattr(description_module, "advertised_camera_frames", sentinel)
    doc = build_description(profile, use_sim=True)
    assert doc["interfaces"]["camera.front.color"]["frame_id"] == "SENTINEL_IMAGE_FRAME"
    assert doc["interfaces"]["camera.front.color_info"]["frame_id"] == "SENTINEL_INFO_FRAME"


# ---------------------------------------------------------------------------
# runtime-deployment-peripherals D3/D5: toggle canonicalisation, the
# completeness rule, driver changes and enabled-camera value checks
# ---------------------------------------------------------------------------

_CAMERA = {"type": "camera", "name": "top", "driver": "opencv", "width": 640, "height": 480, "fps": 30}


def test_profile_disabled_true_is_overridden_by_enabled_true():
    # D3: canonicalisation happens in every layer, so a deployment's
    # enabled: true really overrides a profile's legacy disabled: true.
    merged = merge_peripherals([dict(_CAMERA, disabled=True)], [{"type": "camera", "name": "top", "enabled": True}])
    assert [entry["name"] for entry in merged] == ["top"]
    assert merged[0]["enabled"] is True


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "camera", "name": "top", "disabled": True, "enabled": True},
        {"type": "camera", "name": "top", "disabled": False, "enabled": False},
    ],
)
def test_contradictory_toggle_spellings_are_rejected(entry):
    with pytest.raises(InterfaceDescriptionError, match=r"peripheral \('camera', 'top'\) sets contradictory"):
        merge_peripherals([_CAMERA], [entry])
    # the defaults layer is canonicalised with the same rule
    with pytest.raises(InterfaceDescriptionError, match="sets contradictory"):
        merge_peripherals([dict(_CAMERA, **entry)], [])


def test_agreeing_toggle_spellings_accept_the_enabled_value():
    # disabled: X == enabled: not X: both spellings of the same toggle agree
    # and the canonical enabled value wins.
    merged = merge_peripherals([_CAMERA], [{"type": "camera", "name": "top", "disabled": True, "enabled": False}])
    assert merged == []
    kept = merge_peripherals(
        [_CAMERA], [{"type": "camera", "name": "top", "disabled": False, "enabled": True, "width": 32}]
    )
    assert kept[0]["enabled"] is True and kept[0]["width"] == 32


def test_driver_change_keeps_the_base_disable():
    # D3: a changed driver replaces the device settings but inherits the
    # base enabled state unless the override sets one.
    merged = merge_peripherals(
        [dict(_CAMERA, disabled=True)],
        [{"type": "camera", "name": "top", "driver": "camera_ros", "width": 1280}],
    )
    assert merged == []
    # an explicit enabled on the driver change wins
    reenabled = merge_peripherals(
        [dict(_CAMERA, disabled=True)],
        [{"type": "camera", "name": "top", "driver": "camera_ros", "enabled": True, "width": 1280}],
    )
    assert reenabled[0]["driver"] == "camera_ros" and reenabled[0]["enabled"] is True
    assert reenabled[0].get("index") is None, "the old driver's settings must not leak"


def test_misspelled_partial_entry_is_rejected():
    with pytest.raises(
        InterfaceDescriptionError,
        match=r"peripheral \('camera', 'tpo'\) is not declared by the runtime: an override must "
        "name a declared peripheral, or declare driver to add a new device",
    ):
        merge_peripherals([_CAMERA], [{"type": "camera", "name": "tpo", "width": 1280}])


def test_toggle_only_undeclared_entry_is_still_rejected():
    with pytest.raises(InterfaceDescriptionError, match=r"\('camera', 'ghost'\) is not declared"):
        merge_peripherals([_CAMERA], [{"type": "camera", "name": "ghost", "disabled": True}])


def test_complete_new_device_with_driver_is_accepted():
    merged = merge_peripherals(
        [_CAMERA],
        [{"type": "camera", "name": "deploy", "driver": "realsense", "serial_number": "X", "width": 848}],
    )
    assert [entry["name"] for entry in merged] == ["top", "deploy"]
    assert merged[1]["driver"] == "realsense"


def test_empty_overrides_never_apply_the_completeness_rule():
    # merge_peripherals(defaults, []) callers (interfaces, nodes, stacks) must
    # keep accepting profile defaults of any shape.
    assert merge_peripherals([{"type": "camera", "name": "top", "width": 640}], []) == [
        {"type": "camera", "name": "top", "width": 640}
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 0),
        ("width", "1280"),
        ("width", 1280.5),
        ("width", True),
        ("height", 0),
        ("depth_width", 0),
        ("depth_height", -1),
        ("fps", 0),
        ("fps", "30"),
        ("fps", True),
        ("fps", float("nan")),
        ("fps", float("inf")),
        ("depth_fps", float("nan")),
    ],
)
def test_invalid_enabled_camera_values_are_rejected(field, value):
    with pytest.raises(InterfaceDescriptionError, match=rf"camera top: {field} must be"):
        merge_peripherals([_CAMERA], [{"type": "camera", "name": "top", field: value}])


def test_unpaired_depth_dimensions_are_rejected():
    with pytest.raises(InterfaceDescriptionError, match="camera top: depth_width and depth_height must be"):
        merge_peripherals([_CAMERA], [{"type": "camera", "name": "top", "depth_width": 640}])
    with pytest.raises(InterfaceDescriptionError, match="camera top: depth_width and depth_height must be"):
        merge_peripherals([_CAMERA], [{"type": "camera", "name": "top", "depth_height": 480}])


def test_depth_fps_without_depth_dimensions_is_rejected():
    with pytest.raises(InterfaceDescriptionError, match="camera top: depth_fps requires depth_width/depth_height"):
        merge_peripherals([_CAMERA], [{"type": "camera", "name": "top", "depth_fps": 30}])


def test_disabled_camera_is_not_value_checked():
    merged = merge_peripherals(
        [_CAMERA], [{"type": "camera", "name": "top", "disabled": True, "fps": 0, "width": "bad"}]
    )
    assert merged == []


def test_unknown_keys_pass_through_the_merge():
    merged = merge_peripherals([_CAMERA], [{"type": "camera", "name": "top", "vendor_knob": 7}])
    assert merged[0]["vendor_knob"] == 7


def test_broken_profile_default_fails_the_same_value_checks():
    # D5: profile defaults are validated by the same pass.
    with pytest.raises(InterfaceDescriptionError, match="camera top: fps must be"):
        merge_peripherals([dict(_CAMERA, fps=0)], [])


def test_gazebo_depth_streams_advertise_the_rendered_colour_dims():
    # D8: the gz rgbd sensor renders one camera at the colour dims/rate; the
    # depth streams advertise exactly that, never the device's depth_* overrides.
    camera = {
        "type": "camera",
        "name": "front",
        "driver": "realsense",
        "width": 1280,
        "height": 720,
        "fps": 15,
        "depth_width": 640,
        "depth_height": 360,
        "depth_fps": 30,
        "align_depth": True,
        "streams": ["color", "depth"],
    }
    profile = {"runtime": {"name": "t", "version": "1"}, "capabilities": {}, "peripherals": [camera]}
    description = build_description(profile, use_sim=True, sim_platform="gazebo")
    for stream in ("camera.front.depth", "camera.front.aligned_depth"):
        configured = description["interfaces"][stream]["configured_profile"]
        assert (configured["width"], configured["height"], configured["fps"]) == (1280, 720, 15)
    # off the simulator the device-specific depth dims are still honoured
    sdk = build_description(profile, use_sim=True, sim_platform="sdk")
    depth = sdk["interfaces"]["camera.front.depth"]["configured_profile"]
    assert (depth["width"], depth["height"], depth["fps"]) == (640, 360, 30)


# ---------------------------------------------------------------------------
# lekiwi-mujoco-runtime-platform D5/D6: the lidar contract on MuJoCo
# ---------------------------------------------------------------------------


def _mujoco_lidar_profile(controllers_path, *, driver="livox_mid360", update_rate=100):
    controllers_path.write_text(
        "controller_manager:\n  ros__parameters:\n    update_rate: " + str(update_rate) + "\n",
        encoding="utf-8",
    )
    profile = _lidar_profile(enabled_scan=False, fast_lio={"enabled": True})
    profile["ros2_control"] = {"controllers_config": str(controllers_path)}
    profile["peripherals"][0]["driver"] = driver
    return profile


def test_mujoco_advertises_the_point_cloud_and_the_imu_rate(tmp_path):
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml")
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.points"]["message_type"] == "sensor_msgs/msg/PointCloud2"
    assert interfaces["lidar.mid360.imu"]["message_type"] == "sensor_msgs/msg/Imu"
    # the IMU rate is the effective controller-manager rate, not a constant (D5.3)
    assert interfaces["lidar.mid360.imu"]["rate_hz"] == 100.0
    # a different controllers file advertises its own rate
    profile = _mujoco_lidar_profile(tmp_path / "controllers2.yaml", update_rate=250)
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.imu"]["rate_hz"] == 250.0


def test_mujoco_does_not_advertise_lidars_it_does_not_simulate(tmp_path):
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml", driver="ldlidar")
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert not any(key.startswith("lidar.") for key in interfaces), "ldlidar is not simulated on MuJoCo (D5)"
    # the same driver stays advertised on Gazebo (it simulates it) and off-sim
    gazebo = build_description(profile, use_sim=True, sim_platform="gazebo")["interfaces"]
    assert gazebo["lidar.mid360.scan"]["message_type"] == "sensor_msgs/msg/LaserScan"
    hardware = build_description(profile, use_sim=False)["interfaces"]
    assert hardware["lidar.mid360.scan"]["message_type"] == "sensor_msgs/msg/LaserScan"


def test_mujoco_without_a_simulated_lidar_needs_no_controllers_config():
    # SO-101-shaped: no lidar at all and no ros2_control.controllers_config —
    # the IMU rate must not be demanded for a profile without an integrated IMU
    profile = _lidar_profile(enabled_scan=False)
    profile["peripherals"] = []
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert not any(key.startswith("lidar.") for key in interfaces)


def test_mujoco_imu_rate_follows_the_effective_inventory_not_the_raw_profile(tmp_path):
    """Review round 1: the IMU rate is computed from the same effective
    inventory (profile + deployment fragment) the description itself uses —
    a fragment that adds or disables the lidar decides whether a rate is
    needed at all, and a disabled lidar never demands a controllers file."""
    # fragment ADDS the lidar: the raw profile has none, the effective list
    # has the MID-360 — the IMU entry carries the controller-manager rate
    bare = _lidar_profile(enabled_scan=False)
    bare["peripherals"] = []
    bare["ros2_control"] = {"controllers_config": str(tmp_path / "controllers.yaml")}
    (tmp_path / "controllers.yaml").write_text(
        "controller_manager:\n  ros__parameters:\n    update_rate: 125\n", encoding="utf-8"
    )
    effective = _lidar_profile(enabled_scan=False, fast_lio={"enabled": True})["peripherals"]
    interfaces = build_description(bare, peripherals=effective, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.imu"]["rate_hz"] == 125.0

    # fragment DISABLES the lidar: no IMU is advertised and a missing
    # controllers_config must not fail the description
    disabled = _mujoco_lidar_profile(tmp_path / "unused.yaml")
    disabled["peripherals"][0]["enabled"] = False
    interfaces = build_description(disabled, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert not any(key.startswith("lidar.") for key in interfaces)


def test_mujoco_points_rate_honours_the_platform_update_rate(tmp_path):
    """The advertised cloud rate resolves the same precedence as the composed
    MJCF: simulation.mujoco.update_rate over publish_freq (review round 1)."""
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml")
    lidar = profile["peripherals"][0]
    lidar["publish_freq"] = 10.0
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.points"]["rate_hz"] == 10.0  # publish_freq stands

    lidar["simulation"] = {"mujoco": {"update_rate": 20.0}}
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.points"]["rate_hz"] == 20.0  # the platform override wins

    # other platforms never read the mujoco section
    gazebo = build_description(profile, use_sim=True, sim_platform="gazebo")["interfaces"]
    assert gazebo["lidar.mid360.points"]["rate_hz"] == 10.0


def test_mujoco_imu_rate_fails_closed_on_a_missing_update_rate(tmp_path):
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml")
    (tmp_path / "controllers.yaml").write_text("controller_manager:\n  ros__parameters:\n", encoding="utf-8")
    with pytest.raises(Exception, match="update_rate missing or invalid"):
        build_description(profile, use_sim=True, sim_platform="mujoco")
