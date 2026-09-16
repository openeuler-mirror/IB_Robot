"""Public-description binding and the launch-owned consumer snapshot boundary."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from robot_config.interface_binding import (
    InterfaceBindingError,
    bind_robot_interfaces,
    required_interface_ids,
    resolve_robot_interfaces,
)


@pytest.fixture
def description_api():
    return importlib.import_module("robot_runtime.interface_description")


@pytest.fixture
def descriptor(description_api):
    qos = {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 5}
    profile = {"width": 640, "height": 480, "fps": 30.0, "encoding": "bgr8"}
    doc = {
        "schema_version": 1,
        "robot": {"id": "unit-1", "type": "test", "runtime_name": "test_robot", "runtime_version": "1.0.0"},
        "execution": "simulated",
        "interfaces": {
            "camera.front.color": {
                "capability": "perception.camera",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/front/image",
                "message_type": "sensor_msgs/msg/Image",
                "qos": qos,
                "frame_id": "front_optical",
                "camera_info_topic": "/unit/front/info",
                "configured_profile": profile,
                "supported_profiles": None,
            },
            "camera.front.color_info": {
                "capability": "perception.camera",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/front/info",
                "message_type": "sensor_msgs/msg/CameraInfo",
                "qos": qos,
                "frame_id": "front_optical",
            },
            "joint.state": {
                "capability": "joint.state",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/joints",
                "message_type": "sensor_msgs/msg/JointState",
                "qos": qos,
            },
            "base.cmd_vel": {
                "capability": "base.cmd_vel",
                "kind": "topic",
                "direction": "subscribe",
                "endpoint": "/unit/velocity",
                "message_type": "geometry_msgs/msg/Twist",
                "qos": {**qos, "reliability": "reliable"},
            },
            "motion.compute_fk": {
                "capability": "motion.fk",
                "kind": "service",
                "direction": "serve",
                "endpoint": "/motion/compute_fk",
                "message_type": "ibrobot_msgs/srv/ComputeFk",
            },
        },
        "states": {
            "camera.front.color": {
                "state": "ready",
                "observed_profile": copy.deepcopy(profile),
                "observed_frame_id": "front_optical",
                "last_seen": 1_800_000_000.0,
                "detail": "",
            },
            "joint.state": {
                "state": "ready",
                "observed_profile": None,
                "observed_frame_id": None,
                "last_seen": 1_800_000_000.0,
                "detail": "",
            },
        },
    }
    doc["digest"] = description_api.description_digest(doc)
    description_api.validate_description(doc)
    return doc


@pytest.fixture
def config():
    return {
        "name": "consumer",
        "type": "test",
        "runtime": {"provider": "test_robot", "profile": "test", "instance_id": "unit-1", "target": "simulation"},
        "default_control_mode": "teleop",
        "contract": {
            "observations": [
                {
                    "key": "observation.images.front",
                    "interface": "camera.front.color",
                    "image": {"resize": [224, 224], "encoding": "rgb8"},
                    "align": {"strategy": "asof", "stamp": "header", "tol_ms": 25},
                    "requires": {"width": 640, "height": 480, "encoding": "bgr8", "min_fps": 20},
                },
                {"key": "observation.state", "interface": "joint.state", "selector": {"names": ["one"]}},
            ],
            "actions": [
                {
                    "key": "action",
                    "publish": {"interface": "base.cmd_vel", "strategy": {"type": "immediate"}},
                    "from_tensor": {"field": "linear.x"},
                    "safety_behavior": "hold",
                }
            ],
        },
    }


@pytest.fixture
def model_descriptor(descriptor, description_api):
    from robot_runtime.model_metadata import build_model_metadata

    descriptor["model"] = build_model_metadata(
        {
            "joints": ["one"],
            "arm_joints": ["one"],
            "gripper_joints": [],
            "motion": {"base_link": "base", "ee_link": "tool", "shoulder_link": "shoulder"},
        },
        simulated=True,
        robot_description='<robot name="test"><link name="base"/><link name="tool"/><link name="shoulder"/>'
        '<joint name="one" type="revolute">'
        '<limit lower="-1" upper="1" effort="10" velocity="10"/></joint></robot>',
    )
    descriptor["digest"] = description_api.description_digest(descriptor)
    description_api.validate_description(descriptor)
    return descriptor


@pytest.mark.parametrize("frame", ["base_link", "ee_link", "shoulder_link"])
@pytest.mark.parametrize("override", ["different_frame", None])
def test_binding_rejects_conflicting_moveit_frames(config, model_descriptor, frame, override):
    config["moveit"] = {frame: override, "arm_group_name": "application_arm"}
    original, original_doc = copy.deepcopy(config), copy.deepcopy(model_descriptor)
    with pytest.raises(InterfaceBindingError) as error:
        bind_robot_interfaces(config, model_descriptor)
    assert error.value.code == "model_mismatch"
    assert error.value.path == f"moveit.{frame}"
    assert config == original and model_descriptor == original_doc


@pytest.mark.parametrize("declare_frames", [False, True])
def test_binding_preserves_nonoverlapping_moveit_settings(config, model_descriptor, declare_frames):
    frames = model_descriptor["model"]["frames"]
    config["moveit"] = {"arm_group_name": "application_arm", "planning_time": 2.0}
    if declare_frames:
        config["moveit"].update(frames)
    original, original_doc = copy.deepcopy(config), copy.deepcopy(model_descriptor)
    bound = bind_robot_interfaces(config, model_descriptor)
    assert bound["moveit"] == {**original["moveit"], **frames}
    assert config == original and model_descriptor == original_doc
    assert bind_robot_interfaces(bound, model_descriptor) == bound


def test_binding_is_pure_and_preserves_preprocessing(config, descriptor):
    original, original_doc = copy.deepcopy(config), copy.deepcopy(descriptor)
    bound = bind_robot_interfaces(config, descriptor, require_ready=True)
    obs = bound["contract"]["observations"][0]
    assert config == original and descriptor == original_doc
    assert obs["topic"] == "/unit/front/image"
    assert obs["type"] == "sensor_msgs/msg/Image"
    assert obs["qos"]["reliability"] == "best_effort"
    assert obs["image"] == {"resize": [224, 224], "encoding": "rgb8"}
    assert obs["align"] == original["contract"]["observations"][0]["align"]
    assert obs["_interface_source"]["profile"]["encoding"] == "bgr8"
    assert obs["_interface_source"]["profile_origin"] == "observed"
    assert obs["_interface_source"]["uncertainty"] == []
    assert "peripherals" not in bound
    assert bound["contract"]["observations"][1]["selector"] == {"names": ["one"]}
    action = bound["contract"]["actions"][0]
    assert action["publish"]["topic"] == "/unit/velocity"
    assert action["from_tensor"] == original["contract"]["actions"][0]["from_tensor"]
    assert action["publish"]["strategy"] == {"type": "immediate"}
    assert action["safety_behavior"] == "hold"
    assert bind_robot_interfaces(bound, descriptor, require_ready=True) == bound


def test_binding_has_no_ros_io(config, descriptor, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_ros(name, *args, **kwargs):
        if name.split(".")[0] in {"rclpy", "launch", "launch_ros", "ament_index_python"}:
            pytest.fail(f"pure binding imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ros)
    assert bind_robot_interfaces(config, descriptor)["contract"]["observations"][0]["topic"] == "/unit/front/image"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("interface", "camera.missing", "unknown_interface"),
        ("interface", "motion.compute_fk", "kind_mismatch"),
        ("interface", "base.cmd_vel", "direction_mismatch"),
        ("topic", "/incorrect", "topic_mismatch"),
        ("type", "sensor_msgs/msg/CameraInfo", "type_mismatch"),
        ("interface", "", "invalid_interface"),
    ],
)
def test_rejects_invalid_observation_binding(config, descriptor, field, value, code):
    config["contract"]["observations"][0][field] = value
    with pytest.raises(InterfaceBindingError, match="observation.images.front") as error:
        bind_robot_interfaces(config, descriptor)
    assert error.value.code == code


@pytest.mark.parametrize("logical_id,code", [("absent", "unknown_interface"), ("joint.state", "direction_mismatch")])
def test_rejects_invalid_action_binding(config, descriptor, logical_id, code):
    config["contract"]["actions"][0]["publish"]["interface"] = logical_id
    with pytest.raises(InterfaceBindingError, match=code):
        bind_robot_interfaces(config, descriptor)


@pytest.mark.parametrize(
    "field,value", [("provider", "other_robot"), ("instance_id", "unit-2"), ("runtime_version", "9")]
)
def test_identity_mismatch(config, descriptor, field, value):
    config["runtime"][field] = value
    with pytest.raises(InterfaceBindingError, match=f"identity_mismatch: runtime.{field}"):
        bind_robot_interfaces(config, descriptor)


def test_robot_type_and_execution_must_match(config, descriptor):
    config["type"] = "another_type"
    with pytest.raises(InterfaceBindingError, match="identity_mismatch: type"):
        bind_robot_interfaces(config, descriptor)
    config["type"] = "test"
    config["runtime"]["target"] = "hardware"
    with pytest.raises(InterfaceBindingError, match="identity_mismatch: runtime.target"):
        bind_robot_interfaces(config, descriptor)


def test_unsupported_schema_and_missing_type(config, descriptor):
    descriptor["schema_version"] = 900
    with pytest.raises(InterfaceBindingError, match="invalid_descriptor"):
        bind_robot_interfaces(config, descriptor)
    descriptor["schema_version"] = 1
    del descriptor["interfaces"]["camera.front.color"]["message_type"]
    with pytest.raises(InterfaceBindingError, match="invalid_descriptor"):
        bind_robot_interfaces(config, descriptor)


def test_digest_mismatch(config, descriptor):
    descriptor["digest"] = "0" * 64
    with pytest.raises(InterfaceBindingError, match="invalid_descriptor"):
        bind_robot_interfaces(config, descriptor)


@pytest.mark.parametrize(
    "requires",
    [
        {"width": 1280},
        {"height": 720},
        {"encoding": "rgb8"},
        {"min_fps": 60},
        {"message_type": "sensor_msgs/msg/CameraInfo"},
        {"capability": "perception.lidar"},
    ],
)
def test_unsatisfied_source_requirements(config, descriptor, requires):
    config["contract"]["observations"][0]["requires"] = requires
    with pytest.raises(InterfaceBindingError, match="requirement_unsatisfied.*observation.images.front"):
        bind_robot_interfaces(config, descriptor)


@pytest.mark.parametrize("requires", [{"width": True}, {"height": 0}, {"min_fps": float("nan")}, {"fps": 30}, []])
def test_invalid_requirements_fail_named(config, descriptor, requires):
    config["contract"]["observations"][0]["requires"] = requires
    with pytest.raises(InterfaceBindingError, match="invalid_requires"):
        bind_robot_interfaces(config, descriptor)


@pytest.mark.parametrize(
    "qos", [{"reliability": "reliable"}, {"durability": "transient_local"}, {"depth": 0}, {"extra": 1}]
)
def test_incompatible_observation_qos(config, descriptor, qos):
    config["contract"]["observations"][0]["qos"] = qos
    with pytest.raises(InterfaceBindingError, match="qos_mismatch"):
        bind_robot_interfaces(config, descriptor)


def test_qos_requested_offered_direction_and_local_depth(config, descriptor):
    observation = config["contract"]["observations"][0]
    observation["qos"] = {"depth": 2, "reliability": "best_effort"}
    assert bind_robot_interfaces(config, descriptor)["contract"]["observations"][0]["qos"]["depth"] == 2
    publish = config["contract"]["actions"][0]["publish"]
    publish["qos"] = {"reliability": "best_effort"}
    with pytest.raises(InterfaceBindingError, match="qos_mismatch.*base.cmd_vel"):
        bind_robot_interfaces(config, descriptor)
    publish["qos"] = {"reliability": "reliable", "durability": "transient_local"}
    assert bind_robot_interfaces(config, descriptor)["contract"]["actions"][0]["publish"]["qos"] == {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 5,
    }


@pytest.mark.parametrize("state", ["unknown", "mismatch", "stale", None])
def test_offline_configured_uncertainty_and_live_readiness(config, descriptor, state):
    if state is None:
        del descriptor["states"]["camera.front.color"]
    else:
        descriptor["states"]["camera.front.color"]["state"] = state
    source = bind_robot_interfaces(config, descriptor)["contract"]["observations"][0]["_interface_source"]
    assert source["profile_origin"] == "configured"
    assert source["observed_profile"] is None
    assert "profile_not_observed" in source["uncertainty"]
    with pytest.raises(InterfaceBindingError, match="interface_not_ready.*camera.front.color"):
        bind_robot_interfaces(config, descriptor, require_ready=True)


def test_unknown_observed_fps_never_borrows_configured_measurement(config, descriptor):
    descriptor["states"]["camera.front.color"]["observed_profile"]["fps"] = None
    with pytest.raises(InterfaceBindingError, match="requires.min_fps"):
        bind_robot_interfaces(config, descriptor, require_ready=True)
    del config["contract"]["observations"][0]["requires"]["min_fps"]
    source = bind_robot_interfaces(config, descriptor)["contract"]["observations"][0]["_interface_source"]
    assert source["profile"]["fps"] is None
    assert "fps_unknown" in source["uncertainty"]


def test_action_requirements(config, descriptor):
    config["contract"]["actions"][0]["publish"]["requires"] = {"message_type": "sensor_msgs/msg/JointState"}
    with pytest.raises(InterfaceBindingError, match="requirement_unsatisfied.*base.cmd_vel"):
        bind_robot_interfaces(config, descriptor)


def test_unknown_configured_encoding_cannot_satisfy_a_requirement(config, descriptor, description_api):
    descriptor["states"] = {}
    descriptor["interfaces"]["camera.front.color"]["configured_profile"]["encoding"] = None
    descriptor["digest"] = description_api.description_digest(descriptor)
    with pytest.raises(InterfaceBindingError, match="requires.encoding"):
        bind_robot_interfaces(config, descriptor)


def test_legacy_config_needs_no_description():
    config = {"name": "legacy", "contract": {"observations": [{"key": "state", "topic": "/joint_states"}]}}
    assert resolve_robot_interfaces(config) is config
    assert required_interface_ids(config) == []


def test_explicit_deferral_does_not_make_an_offline_contract(config, tmp_path):
    from robot_config.loader import build_contract_from_robot_config_dict, load_contract_config, load_robot_config_dict

    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    with pytest.raises(InterfaceBindingError, match="description_required"):
        load_robot_config_dict(path)
    structural = load_robot_config_dict(path, defer_interface_binding=True)
    assert structural["_interfaces_deferred"] is True
    assert "topic" not in structural["contract"]["observations"][0]
    with pytest.raises(InterfaceBindingError, match="description_required"):
        build_contract_from_robot_config_dict(structural)
    with pytest.raises(InterfaceBindingError, match="description_required"):
        load_contract_config(structural["contract"])


def test_yaml_marker_cannot_enable_deferred_validation(tmp_path):
    from robot_config.loader import load_robot_config_dict

    config = {
        "name": "invalid",
        "_interfaces_deferred": True,
        "contract": {"observations": [{"key": "image", "topic": "/image", "transport": {"mode": "invalid"}}]},
    }
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    with pytest.raises(ValueError, match="transport.mode"):
        load_robot_config_dict(path)


def test_public_raw_section_loader_cannot_bypass_binding(config, descriptor, tmp_path):
    from robot_config.loader import load_robot_section

    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    with pytest.raises(InterfaceBindingError, match="description_required"):
        load_robot_section(config_path)
    snapshot = tmp_path / "description.yaml"
    snapshot.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    config["runtime"]["interface_description"] = snapshot.name
    # This lower-level production API must not require model bundles just to bind.
    config["perception_services"] = {"services": [{"id": "uninstalled", "bundle_path": "/missing/model"}]}
    config_path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    path, bound = load_robot_section(config_path)
    assert path == config_path
    assert bound["contract"]["observations"][0]["topic"] == "/unit/front/image"
    config["runtime"]["instance_id"] = "wrong-unit"
    config_path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    with pytest.raises(InterfaceBindingError, match="identity_mismatch"):
        load_robot_section(config_path)


def test_legacy_raw_section_unchanged(tmp_path):
    from robot_config.loader import load_robot_section

    config = {"name": "legacy", "contract": {"observations": [{"key": "state", "topic": "/joint_states"}]}}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    assert load_robot_section(path) == (path, config)


def test_standalone_contract_rejects_unchecked_logical_ids(config, tmp_path):
    from robot_config.generators.contract import load_contract_with_robot_config

    path = tmp_path / "contract.yaml"
    config["contract"]["observations"][0].update(topic="/unchecked", type="sensor_msgs/msg/Image")
    path.write_text(yaml.safe_dump(config["contract"]), encoding="utf-8")
    with pytest.raises(InterfaceBindingError, match="description_required"):
        load_contract_with_robot_config(path)


def test_snapshot_yaml_parse_error_is_named(config, tmp_path):
    path = tmp_path / "description.yaml"
    path.write_text("interfaces: [", encoding="utf-8")
    config["runtime"]["interface_description"] = str(path)
    with pytest.raises(InterfaceBindingError, match="invalid_descriptor"):
        resolve_robot_interfaces(config)


def test_waiter_failure_stops_launch_without_constructing_consumers(config, monkeypatch):
    from launch import LaunchDescription, LaunchService
    from launch.actions import ExecuteProcess

    from robot_config.launch_builders import runtime as builder

    calls = []

    def provider(_config, *, description_output, **_kwargs):
        calls.append(description_output)
        return [], ExecuteProcess(cmd=[sys.executable, "-c", "raise SystemExit(1)"])

    monkeypatch.setattr(builder, "generate_runtime_provider_actions", provider)
    service = LaunchService()
    actions = builder.generate_bound_runtime_actions(
        config, lambda _effective: pytest.fail("consumer constructed before readiness"), use_sim=True
    )
    service.include_launch_description(LaunchDescription(actions))
    service.run()
    assert len(calls) == 1
    assert not Path(calls[0]).parent.exists()


def test_legacy_launch_still_constructs_consumers_directly(monkeypatch, tmp_path):
    from launch import LaunchContext
    from launch_ros.actions import Node

    module = _launch_module()
    path = tmp_path / "legacy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "legacy",
                    "default_control_mode": "teleop",
                    "contract": {"observations": [{"key": "state", "topic": "/legacy/joints"}]},
                }
            }
        ),
        encoding="utf-8",
    )
    context = LaunchContext()
    context.launch_configurations.update({"config_path": str(path), "record": "true", "record_mode": "episodic"})
    monkeypatch.setattr(
        module, "generate_bound_runtime_actions", lambda *_a, **_k: pytest.fail("legacy binding opt-in")
    )
    monkeypatch.setattr(module, "generate_ros2_control_nodes", lambda *_a, **_k: ([], [], [], {}))
    nodes = module.launch_setup(context)
    recorder = next(node for node in nodes if isinstance(node, Node) and node.node_package == "dataset_tools")
    params = {
        "".join(key.text for key in name): value
        for group in recorder._Node__parameters
        for name, value in group.items()
    }
    assert yaml.safe_load("".join(part.text for part in params["robot_config_path"])) == str(path)


@pytest.mark.parametrize("snapshot_format", ["yaml", "json", "embedded"])
def test_offline_snapshot_dict_typed_parity_and_rtp_source_size(config, descriptor, tmp_path, snapshot_format):
    from robot_config.generators.contract import generate_contract_from_robot_config, load_contract_with_robot_config
    from robot_config.loader import (
        build_contract_from_robot_config_dict,
        load_contract_config,
        load_robot_config,
        load_robot_config_dict,
    )

    obs = config["contract"]["observations"][0]
    obs["peripheral"] = "old_camera"
    config["peripherals"] = [{"type": "camera", "name": "old_camera", "width": 12, "height": 10, "fps": 2}]
    obs["transport"] = {"mode": "rtp", "stream_id": "front", "endpoint": {"host": "127.0.0.1", "port": 55000}}
    original_transport = copy.deepcopy(obs["transport"])
    config["control_modes"] = {"teleop": {"inference": {"pipelines": {"policy": {"execution_mode": "distributed"}}}}}
    if snapshot_format == "embedded":
        config["runtime"]["interface_description"] = descriptor
    else:
        path = tmp_path / f"description.{snapshot_format}"
        path.write_text(
            json.dumps(descriptor) if snapshot_format == "json" else yaml.safe_dump(descriptor), encoding="utf-8"
        )
        config["runtime"]["interface_description"] = path.name
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    raw = load_robot_config_dict(path)
    typed = load_robot_config(path)
    contract = build_contract_from_robot_config_dict(raw)
    assert typed.to_contract() == contract
    assert load_contract_config(raw["contract"], robot_config=raw) == typed.contract
    assert raw["peripherals"] == config["peripherals"]
    assert raw["contract"]["observations"][0]["transport"] == original_transport
    image = contract.observations[0]
    assert image.image == {"resize": [224, 224], "encoding": "rgb8"}
    assert (image.transport.media.width, image.transport.media.height, image.transport.media.frame_rate_hz) == (
        640,
        480,
        30,
    )
    generated = tmp_path / "contract.yaml"
    generated.write_text(generate_contract_from_robot_config(typed), encoding="utf-8")
    assert load_contract_with_robot_config(generated) == contract
    roundtrip = tmp_path / "materialized.yaml"
    roundtrip.write_text(yaml.safe_dump({"robot": raw}), encoding="utf-8")
    assert load_robot_config(roundtrip).to_contract() == contract


def test_wire_encoding_and_policy_resize_reach_the_decoder(config, descriptor):
    import numpy as np
    from sensor_msgs.msg import Image

    from robot_config.contract_utils import decode_value, iter_specs
    from robot_config.loader import build_contract_from_robot_config_dict

    bound = bind_robot_interfaces(config, descriptor, require_ready=True)
    spec = next(iter_specs(build_contract_from_robot_config_dict(bound)))
    message = Image(height=480, width=640, encoding="bgr8", step=640 * 3, data=bytes([0, 0, 255]) * (480 * 640))
    decoded = decode_value("sensor_msgs/msg/Image", message, spec)
    assert spec.image_resize == (224, 224) and spec.image_encoding == "rgb8"
    assert decoded.shape == (3, 224, 224)
    np.testing.assert_array_equal(decoded[:, 0, 0], [1.0, 0.0, 0.0])
    assert bound["contract"]["observations"][0]["_interface_source"]["profile"]["encoding"] == "bgr8"


def test_explicit_transport_media_does_not_change_source_or_preprocessing(config, descriptor):
    from robot_config.loader import build_contract_from_robot_config_dict

    transport = {
        "mode": "rtp",
        "stream_id": "front",
        "endpoint": {"host": "127.0.0.1", "port": 55000},
        "media": {"width": 160, "height": 120, "frame_rate_hz": 10},
    }
    config["contract"]["observations"][0]["transport"] = transport
    bound = bind_robot_interfaces(config, descriptor)
    obs = build_contract_from_robot_config_dict(bound).observations[0]
    assert (obs.transport.media.width, obs.transport.media.height, obs.transport.media.frame_rate_hz) == (160, 120, 10)
    assert obs.image["resize"] == [224, 224]
    assert obs._interface_source["profile"] == {"width": 640, "height": 480, "fps": 30.0, "encoding": "bgr8"}
    assert bound["contract"]["observations"][0]["transport"] == transport


def _launch_module():
    path = Path(__file__).resolve().parents[1] / "launch" / "robot.launch.py"
    spec = importlib.util.spec_from_file_location("robot_launch_interface_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exit_callback(actions, event):
    from launch.actions import RegisterEventHandler
    from launch.event_handlers import OnProcessExit

    handler = next(
        action.event_handler
        for action in actions
        if isinstance(action, RegisterEventHandler) and isinstance(action.event_handler, OnProcessExit)
    )
    assert handler.matches(event)
    return handler


@pytest.mark.parametrize(
    "outcome",
    [
        "ready",
        "waiter_failed",
        "bad_descriptor",
        "missing_descriptor",
        "stale",
        "wrong_instance",
        "wrong_runtime",
        "missing_profile",
        "fps_unknown",
        "fps_too_low",
        "size_mismatch",
        "encoding_mismatch",
    ],
)
def test_robot_launch_snapshot_gate(config, descriptor, description_api, monkeypatch, tmp_path, outcome):
    from launch import LaunchContext
    from launch.actions import EmitEvent, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
    from launch.events.process import ProcessExited
    from launch_ros.actions import Node

    from robot_config.launch_builders import runtime as builder

    module = _launch_module()
    path = tmp_path / "input.yaml"
    config["default_control_mode"] = "unused"
    path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "config_path": str(path),
            "record": "true",
            "record_mode": "episodic",
            "control_mode": "teleop",
            "with_inference": "true",
        }
    )
    provider_path = tmp_path / "runtime.launch.py"
    provider_path.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(builder, "resolve_runtime_launch", lambda _provider: str(provider_path))
    monkeypatch.setattr(builder.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(module, "get_sim_backend", lambda *_: pytest.fail("provider simulation must not start twice"))
    monkeypatch.setattr(
        module, "generate_ros2_control_nodes", lambda *_a, **_k: pytest.fail("duplicate hardware start")
    )
    monkeypatch.setattr(
        module,
        "confirm_interactive_startup_p_pose",
        lambda cfg, **_kwargs: cfg.setdefault("recording", {}).update(default_task="preflight completed"),
    )
    seen = []
    execution_paths = []
    real_recording = module.generate_recording_nodes

    def execution(effective, *_args, **kwargs):
        assert kwargs["use_sim_time"] is False
        assert Path(effective["_config_path"]).is_file()
        assert effective["contract"]["observations"][0]["topic"] == "/unit/front/image"
        execution_paths.append(effective["_config_path"])
        return []

    def recording(effective, *args, **kwargs):
        snapshot = Path(effective["_config_path"])
        assert snapshot != path and snapshot.is_file()
        assert yaml.safe_load(snapshot.read_text())["robot"]["contract"] == effective["contract"]
        assert yaml.safe_load(snapshot.read_text())["robot"]["default_control_mode"] == "teleop"
        assert yaml.safe_load(snapshot.read_text())["robot"]["recording"]["default_task"] == "preflight completed"
        seen.append(effective)
        return real_recording(effective, *args, **kwargs)

    monkeypatch.setattr(module, "generate_recording_nodes", recording)
    monkeypatch.setattr(module, "generate_execution_nodes", execution)
    actions = module.launch_setup(context)
    assert not seen and not execution_paths
    assert sum(isinstance(action, IncludeLaunchDescription) for action in actions) == 1
    waiter = next(action for action in actions if isinstance(action, Node))
    arguments = waiter._Node__arguments
    description_path = Path(arguments[arguments.index("--description-output") + 1])
    assert "--require-interfaces" in arguments and "--instance-id" in arguments
    assert "base.cmd_vel" in arguments and "camera.front.color" in arguments
    if outcome == "bad_descriptor":
        descriptor["schema_version"] = 9
    elif outcome == "stale":
        descriptor["states"]["camera.front.color"]["state"] = "stale"
    elif outcome == "wrong_instance":
        descriptor["robot"]["id"] = "wrong-unit"
    elif outcome == "wrong_runtime":
        descriptor["robot"]["runtime_name"] = "wrong_robot"
    elif outcome == "missing_profile":
        descriptor["states"]["camera.front.color"]["observed_profile"] = None
    elif outcome == "fps_unknown":
        descriptor["states"]["camera.front.color"]["observed_profile"]["fps"] = None
    elif outcome == "fps_too_low":
        descriptor["states"]["camera.front.color"]["observed_profile"]["fps"] = 10.0
    elif outcome == "size_mismatch":
        descriptor["states"]["camera.front.color"]["observed_profile"]["width"] = 320
    elif outcome == "encoding_mismatch":
        descriptor["states"]["camera.front.color"]["observed_profile"]["encoding"] = "mono8"
    descriptor["digest"] = description_api.description_digest(descriptor)
    if outcome != "missing_descriptor":
        description_path.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    event = ProcessExited(
        action=waiter,
        name="waiter",
        cmd=[],
        cwd=None,
        env=None,
        pid=1,
        returncode=1 if outcome == "waiter_failed" else 0,
    )
    returned = list(_exit_callback(actions, event).handle(event, context))
    if outcome == "ready":
        assert len(seen) == 1
        assert execution_paths == [seen[0]["_config_path"]]
        assert not any(isinstance(action, IncludeLaunchDescription) for action in returned)
        recorder = next(
            action for action in returned if isinstance(action, Node) and action.node_package == "dataset_tools"
        )
        params = {
            "".join(key.text for key in name): value
            for group in recorder._Node__parameters
            for name, value in group.items()
        }
        assert yaml.safe_load("".join(part.text for part in params["robot_config_path"])) == seen[0]["_config_path"]
        assert not any(isinstance(action, ExecuteProcess) and action is waiter for action in returned)
    else:
        assert not seen and not execution_paths
        assert len(returned) == 1 and isinstance(returned[0], EmitEvent)
        assert not (description_path.parent / "robot.yaml").exists()
    from launch.events import Shutdown

    for action in actions:
        if isinstance(action, RegisterEventHandler) and action.event_handler.matches(Shutdown()):
            list(action.event_handler.handle(Shutdown(), context) or [])
    if outcome == "ready":
        from robot_config.loader import load_robot_config_dict

        assert load_robot_config_dict(seen[0]["_config_path"])["contract"] == seen[0]["contract"]


def test_recording_uses_public_topics(config, descriptor):
    from robot_config.launch_builders.recording import get_recording_topics

    config["peripherals"] = [{"type": "camera", "name": "not_public"}]
    topics = get_recording_topics(bind_robot_interfaces(config, descriptor))
    assert "/unit/front/image" in topics and "/unit/front/info" in topics and "/unit/joints" in topics
    assert "/unit/velocity" in topics
    assert "/joint_states" not in topics and "/camera/not_public/image_raw" not in topics


@pytest.mark.parametrize("waiter_code", [0, 1])
def test_launch_service_materializes_before_consumer_process(config, descriptor, monkeypatch, tmp_path, waiter_code):
    """Exercise actual launch process events; only the provider/waiter process boundary is substituted."""
    from launch import LaunchDescription, LaunchService
    from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler, TimerAction
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown

    from robot_config.launch_builders import runtime as builder

    started = tmp_path / "provider_count"
    consumed = tmp_path / "consumed"
    calls = []

    def provider(_config, *, description_output, **_kwargs):
        calls.append(description_output)
        owner = ExecuteProcess(
            cmd=[
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).open('a').write('start' + chr(10))",
                str(started),
            ]
        )
        waiter_script = (
            "from pathlib import Path\nimport sys, time\n"
            "while not Path(sys.argv[4]).exists():\n    time.sleep(0.01)\n"
            "Path(sys.argv[1]).write_text(sys.argv[2])\nsys.exit(int(sys.argv[3]))\n"
        )
        waiter = ExecuteProcess(
            cmd=[
                sys.executable,
                "-c",
                waiter_script,
                description_output,
                yaml.safe_dump(descriptor),
                str(waiter_code),
                str(started),
            ]
        )
        return [owner], waiter

    def consumer(effective):
        assert Path(effective["_config_path"]).is_file()
        process = ExecuteProcess(
            cmd=[
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; from robot_config.loader import load_robot_config; c = load_robot_config(sys.argv[1]).to_contract(); assert c.observations[0].topic == '/unit/front/image'; Path(sys.argv[2]).write_text('bound')",
                effective["_config_path"],
                str(consumed),
            ]
        )
        return [
            RegisterEventHandler(OnProcessExit(target_action=process, on_exit=[EmitEvent(event=Shutdown())])),
            process,
        ]

    monkeypatch.setattr(builder, "generate_runtime_provider_actions", provider)
    monkeypatch.setattr(builder.tempfile, "tempdir", str(tmp_path))
    actions = builder.generate_bound_runtime_actions(config, consumer, use_sim=True)
    service = LaunchService()
    service.include_launch_description(
        LaunchDescription(
            [*actions, TimerAction(period=10.0, actions=[EmitEvent(event=Shutdown(reason="test timeout"))])]
        )
    )
    service.run()
    assert len(calls) == 1
    assert started.read_text() == "start\n"
    assert consumed.exists() is (waiter_code == 0)
    if waiter_code == 0:
        from robot_config.loader import load_robot_config

        assert (
            load_robot_config(Path(calls[0]).parent / "robot.yaml").to_contract().observations[0].topic
            == "/unit/front/image"
        )


def test_binding_rejects_selectors_that_name_no_interface_joint(config, descriptor, description_api):
    # `<field>.<joint>` selectors resolve by joint name; a suffix outside the
    # interface's declared joints reads the wrong joint (or NaN) silently.
    descriptor["interfaces"]["joint.state"]["joint_names"] = ["1", "2", "3", "4", "5", "6"]
    descriptor["digest"] = description_api.description_digest(descriptor)
    config["contract"]["observations"][1]["selector"] = {"names": ["position.1", "position.9"]}
    with pytest.raises(InterfaceBindingError, match="selector_mismatch"):
        bind_robot_interfaces(config, descriptor, require_ready=True)


def test_binding_accepts_selectors_that_name_interface_joints(config, descriptor, description_api):
    descriptor["interfaces"]["joint.state"]["joint_names"] = ["1", "2", "3", "4", "5", "6"]
    descriptor["digest"] = description_api.description_digest(descriptor)
    config["contract"]["observations"][1]["selector"] = {"names": ["position.1", "position.6"]}
    bound = bind_robot_interfaces(config, descriptor, require_ready=True)
    assert bound["contract"]["observations"][1]["selector"]["names"] == ["position.1", "position.6"]


def test_binding_rejects_command_stream_selector_count_mismatch(config, descriptor, description_api):
    # Array-typed command streams consume selector values in declared joint
    # order; a count mismatch silently truncates or overruns the stream.
    descriptor["interfaces"]["base.cmd_vel"]["joint_names"] = ["1", "2"]
    descriptor["interfaces"]["base.cmd_vel"]["message_type"] = "std_msgs/msg/Float64MultiArray"
    descriptor["digest"] = description_api.description_digest(descriptor)
    config["contract"]["actions"][0]["publish"] = {
        "interface": "base.cmd_vel",
        "strategy": {"type": "immediate"},
        "selector": {"names": ["action.0", "action.1", "action.2"]},
    }
    with pytest.raises(InterfaceBindingError, match="selector values for 2 joints"):
        bind_robot_interfaces(config, descriptor, require_ready=True)
