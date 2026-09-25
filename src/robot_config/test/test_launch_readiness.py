"""Unit tests for launch readiness helpers."""

import builtins
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from launch import LaunchContext
from launch.actions import ExecuteProcess, GroupAction, OpaqueFunction, RegisterEventHandler
from launch.events import Shutdown
from launch_ros.actions import Node

from inference_manifest import BundleFile, canonical_bundle_digest
from robot_config.inference_config import InferenceConfigError
from robot_config.launch_builders import tracing as tracing_builder
from robot_config.launch_builders.control import (
    generate_auxiliary_actuator_nodes,
    generate_controller_spawners,
    generate_ros2_control_nodes,
    validate_runtime_resources,
)
from robot_config.launch_builders.execution import (
    _attention_viz_request,
    generate_action_dispatcher_node,
    generate_execution_nodes,
    generate_inference_node,
)
from robot_config.launch_builders.hand_sources import (
    apply_hand_profile,
    confirm_interactive_startup_p_pose,
    generate_hand_source_nodes,
)
from robot_config.launch_builders.navigation import generate_navigation_nodes
from robot_config.launch_builders.perception import generate_camera_nodes, generate_tf_nodes
from robot_config.launch_builders.sim_backend import get_sim_backend
from robot_config.launch_builders.teleop import generate_teleop_nodes, validate_teleop_config
from robot_config.loader import load_robot_config_dict
from robot_config.wait_for_controllers import missing_inactive_controllers

_LAUNCH_PATH = Path(__file__).resolve().parents[1] / "launch" / "robot.launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("robot_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None
assert _LAUNCH_SPEC.loader is not None
robot_launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(robot_launch)

_HAND_JOINTS = [f"hand_{index}" for index in range(7)]


def _joint_limits(names):
    return {name: {"min": -1.0, "max": 1.0} for name in names}


_BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
_DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"


def _block_voice_asr_service_import(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("voice_asr_service"):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def _text(substitutions):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in substitutions)


def _node_parameters(node):
    def decode_parameter(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode_parameter(tuple(item)) for item in value]
        if all(isinstance(item, bool | int | float) for item in value):
            return list(value)
        text = _text(value)
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text.strip()

    raw = node._Node__parameters[0]
    parsed = {}
    for key, value in raw.items():
        name = _text(key)
        parsed[name] = decode_parameter(value)
    return parsed


def _node_remappings(node):
    remappings = []
    for src, dst in node._Node__remappings:
        remappings.append((_text(src), _text(dst)))
    return remappings


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _create_inference_bundle(root, deployments=None):
    root.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "type": "act",
            "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})
    (root / "model.safetensors").write_bytes(b"test-weights")
    paths = ("config.json", "model.safetensors", "policy_postprocessor.json", "policy_preprocessor.json")
    entries = [BundleFile(path=path) for path in paths]
    deployment_values = deployments or {"cpu": {"backend": "torch", "device": "cpu"}}
    deployment_values = {
        name: {
            "uuid": _DEPLOYMENT_UUID,
            "revision": 1,
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {
                "backend": value.get("backend", "torch"),
                "target": {"runtime": "torch"},
                "profile": {"device": value.get("device", "cpu")},
            },
        }
        for name, value in deployment_values.items()
    }
    _write_json(
        root / "inference_manifest.json",
        {
            "schema_version": 3,
            "bundle": {
                "uuid": _BUNDLE_UUID,
                "revision": 1,
                "name": root.name,
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(_BUNDLE_UUID, 1, root.name, entries),
                },
            },
            "model": {
                "interface": "policy",
                "model_type": "act",
                "operation": "predict",
                "inputs": [{"semantic": "observation.state", "dtype": "float32", "shape": [6]}],
                "outputs": [{"semantic": "action", "dtype": "float32", "shape": [6]}],
            },
            "deployments": deployment_values,
        },
    )
    return root


def _inference_robot_config(config_path, pipelines, *, selection=None):
    executor = {"type": "topic", "mode": "model_inference"}
    if selection is not None:
        executor["inference_pipeline"] = selection
    return {
        "_config_path": str(config_path),
        "name": "test_robot",
        "joints": {"all": ["1", "2", "3", "4", "5", "6"]},
        "control_modes": {
            "model_inference": {
                "inference": {"enabled": True, "pipelines": pipelines},
                "executor": executor,
            }
        },
    }


def test_missing_inactive_controllers_returns_only_non_active():
    controllers = [
        SimpleNamespace(name="joint_state_broadcaster", state="active"),
        SimpleNamespace(name="arm_position_controller", state="inactive"),
        SimpleNamespace(name="gripper_position_controller", state="active"),
    ]

    pending = missing_inactive_controllers(
        controllers,
        ["joint_state_broadcaster", "arm_position_controller", "missing_controller"],
    )

    assert pending == ["arm_position_controller", "missing_controller"]


def test_generate_controller_spawners_uses_one_timeout_aware_group_process():
    spawners = generate_controller_spawners(
        ["joint_state_broadcaster", "arm_position_controller"],
        use_sim=True,
        controller_manager_timeout=42.5,
    )

    assert len(spawners) == 1
    spawner = spawners[0]
    assert isinstance(spawner, Node)
    assert spawner.node_package == "robot_config"
    assert spawner.node_executable == "controller_spawner"
    cmd_text = [item[0].text for item in spawner.cmd if item and hasattr(item[0], "text")]
    assert "joint_state_broadcaster" in cmd_text
    assert "arm_position_controller" in cmd_text
    assert "--controller-manager" in cmd_text
    assert "controller_manager" in cmd_text
    assert cmd_text[cmd_text.index("--service-call-timeout") + 1] == "10.0"
    assert cmd_text[cmd_text.index("--switch-timeout") + 1] == "42.5"


def test_generate_controller_spawners_supports_inactive_controller_group():
    spawners = generate_controller_spawners(
        ["arm_trajectory_controller"],
        use_sim=False,
        controller_manager_timeout=30.0,
        inactive_controller_names=["base_velocity_controller"],
    )

    assert len(spawners) == 1
    arguments = _text(spawners[0]._Node__arguments)
    assert "arm_trajectory_controller" in arguments
    assert "--inactive-controller" in arguments
    assert "base_velocity_controller" in arguments


def test_generate_controller_spawners_rejects_overlapping_groups():
    with pytest.raises(ValueError, match="both active and inactive"):
        generate_controller_spawners(
            ["base_velocity_controller"],
            inactive_controller_names=["base_velocity_controller"],
        )


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_real_hardware_controller_spawners_use_configured_readiness_timeout():
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "lekiwi_handeye_realsense_grasp.yaml"
    robot_config = load_robot_config_dict(config_path)
    robot_config["default_control_mode"] = "moveit_planning"

    nodes, controller_names, deferred_spawners, _robot_description = generate_ros2_control_nodes(
        robot_config,
        use_sim=False,
        auto_start_controllers="true",
        controller_startup_timeout=120.0,
    )

    assert controller_names == [
        "joint_state_broadcaster",
        "arm_joint_state_broadcaster",
        "arm_trajectory_controller",
        "gripper_trajectory_controller",
    ]
    assert len(deferred_spawners) == 1
    assert "--inactive-controller" in _text(deferred_spawners[0]._Node__arguments)
    assert all(spawner.node_package == "robot_config" for spawner in deferred_spawners)
    assert all(spawner.node_executable == "controller_spawner" for spawner in deferred_spawners)
    assert all(node.node_executable != "controller_spawner" for node in nodes)
    assert "120.0" in _text(deferred_spawners[0]._Node__arguments)


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_hybrid_real_hardware_spawner_keeps_arm_state_stream_active():
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "lekiwi_nav_grasp.yaml"
    robot_config = load_robot_config_dict(config_path)

    _nodes, controller_names, deferred_spawners, _robot_description = generate_ros2_control_nodes(
        robot_config,
        use_sim=False,
        auto_start_controllers="true",
        controller_startup_timeout=120.0,
    )

    assert controller_names == [
        "joint_state_broadcaster",
        "arm_joint_state_broadcaster",
        "base_velocity_controller",
    ]
    assert len(deferred_spawners) == 1
    arguments = [_text(argument) for argument in deferred_spawners[0]._Node__arguments]
    inactive_controllers = [
        arguments[index + 1] for index, argument in enumerate(arguments) if argument == "--inactive-controller"
    ]
    assert inactive_controllers == ["arm_trajectory_controller", "gripper_trajectory_controller"]


def test_controller_startup_processes_are_serialized():
    processes = [ExecuteProcess(cmd=["true"]) for _ in range(4)]

    actions = robot_launch._serialize_process_startup(processes, "test controller startup")

    assert len(actions) == len(processes)
    assert all(isinstance(action, RegisterEventHandler) for action in actions[:-1])
    assert actions[-1] is processes[0]


def test_group_spawner_is_the_controller_readiness_barrier():
    group_spawner = ExecuteProcess(cmd=["true"])
    fallback_waiter = ExecuteProcess(cmd=["false"])

    barrier = robot_launch._controller_readiness_barrier([group_spawner], fallback_waiter)

    assert barrier is group_spawner
    assert robot_launch._controller_readiness_barrier([], fallback_waiter) is fallback_waiter

    with pytest.raises(ValueError, match="single controller group spawner"):
        robot_launch._controller_readiness_barrier([group_spawner, ExecuteProcess(cmd=["true"])], fallback_waiter)


def test_controller_readiness_handler_is_registered_before_barrier_can_start():
    source = _LAUNCH_PATH.read_text(encoding="utf-8")
    block = source[source.index("if controller_dependent_actions:") : source.index("# ========== N. Tracing")]

    assert "actions.insert(" in block
    assert "target_action=controller_ready_barrier" in block


def _relay_targets(nodes):
    """Collect (source_topic, target_topic) pairs from robot_config topic_relay nodes."""
    pairs = []
    for node in nodes:
        if getattr(node, "_Node__package", None) != "robot_config":
            continue
        args = getattr(node, "_Node__arguments", None) or []
        if len(args) >= 2:
            pairs.append((_text(args[0]), _text(args[1])))
    return pairs


def test_realsense_camera_topics_are_remapped_to_robot_config_contract_names():
    nodes = generate_camera_nodes(
        {
            "peripherals": [
                {
                    "type": "camera",
                    "name": "wrist",
                    "driver": "realsense",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "align_depth": True,
                    "enable_pointcloud": True,
                }
            ]
        },
        use_sim=False,
    )

    relay_pairs = _relay_targets(nodes)
    driver = next(
        node
        for node in nodes
        if vars(node).get("_Node__package") == "realsense2_camera"
        and vars(node).get("_Node__node_executable") == "realsense2_camera_node"
    )
    driver_params = _node_parameters(driver)
    assert driver_params["base_frame_id"] == "wrist_camera_link"
    assert driver_params["align_depth.enable"] is True
    assert driver_params["pointcloud.enable"] is True
    assert driver_params["enable_infra"] is False
    assert driver_params["enable_infra1"] is False
    assert driver_params["enable_infra2"] is False
    assert (
        "/camera/wrist_camera/color/image_raw",
        "/camera/wrist/image_raw",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/color/camera_info",
        "/camera/wrist/camera_info",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/depth/image_rect_raw",
        "/camera/wrist/depth/image_rect_raw",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/aligned_depth_to_color/image_raw",
        "/camera/wrist/aligned_depth_to_color/image_raw",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/aligned_depth_to_color/camera_info",
        "/camera/wrist/aligned_depth_to_color/camera_info",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/depth/color/points",
        "/camera/wrist/depth/color/points",
    ) in relay_pairs


def test_realsense_direct_topic_remap_avoids_large_payload_relays():
    nodes = generate_camera_nodes(
        {
            "peripherals": [
                {
                    "type": "camera",
                    "name": "wrist",
                    "driver": "realsense",
                    "width": 640,
                    "height": 360,
                    "fps": 30,
                    "align_depth": True,
                    "enable_pointcloud": False,
                    "direct_topic_remap": True,
                    "optical_frame_id": "camera_wrist_optical_frame",
                }
            ]
        },
        use_sim=False,
    )

    driver = next(node for node in nodes if getattr(node, "_Node__package", None) == "realsense2_camera")
    assert _node_remappings(driver) == [
        ("/camera/wrist_camera/color/image_raw", "/camera/wrist/image_raw"),
        ("/camera/wrist_camera/depth/image_rect_raw", "/camera/wrist/depth/image_rect_raw"),
        (
            "/camera/wrist_camera/aligned_depth_to_color/image_raw",
            "/camera/wrist/aligned_depth_to_color/image_raw",
        ),
    ]
    relay_pairs = _relay_targets(nodes)
    assert relay_pairs == [
        ("/camera/wrist_camera/color/camera_info", "/camera/wrist/camera_info"),
        ("/camera/wrist_camera/depth/camera_info", "/camera/wrist/depth/camera_info"),
        (
            "/camera/wrist_camera/aligned_depth_to_color/camera_info",
            "/camera/wrist/aligned_depth_to_color/camera_info",
        ),
    ]


def test_realsense_tf_bridge_targets_the_driver_prefixed_base_frame():
    nodes = generate_tf_nodes(
        {
            "peripherals": [
                {
                    "type": "camera",
                    "name": "wrist",
                    "driver": "realsense",
                    "driver_camera_name": "wrist_camera",
                    "frame_id": "camera_wrist_link",
                    "optical_frame_id": "camera_wrist_optical_frame",
                    "transform": {"parent_frame": "gripper"},
                }
            ]
        },
        use_sim=False,
    )

    bridge = next(node for node in nodes if vars(node).get("_Node__node_name") == "static_tf_wrist_driver_bridge")
    arguments = [_text(value) for value in bridge._Node__arguments]
    assert arguments[arguments.index("--frame-id") + 1] == "camera_wrist_link"
    assert arguments[arguments.index("--child-frame-id") + 1] == "wrist_camera_camera_wrist_link"


def test_start_actions_handler_snapshots_action_list():
    original_actions = ["first"]
    handler = robot_launch._start_actions_on_success(
        original_actions,
        success_message="ok",
        failure_reason="failed",
    )
    original_actions.append("second")

    returned_actions = handler(SimpleNamespace(returncode=0), None)

    assert returned_actions == ["first"]


def test_robot_launch_loads_without_voice_asr_service_when_asr_disabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)
    launch_spec = importlib.util.spec_from_file_location("robot_launch_without_voice_asr", _LAUNCH_PATH)
    assert launch_spec is not None
    assert launch_spec.loader is not None
    launch_module = importlib.util.module_from_spec(launch_spec)

    launch_spec.loader.exec_module(launch_module)

    assert launch_module.generate_launch_description() is not None


def test_voice_asr_builder_skips_missing_package_when_disabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)
    from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

    nodes = generate_voice_asr_nodes({"voice_asr": {"enabled": False}})

    assert nodes == []


def test_voice_asr_builder_reports_missing_package_when_enabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)
    from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

    with pytest.raises(
        ModuleNotFoundError,
        match="voice_asr.enabled=true requires the voice_asr_service package to be installed",
    ):
        generate_voice_asr_nodes({"voice_asr": {"enabled": True}})


def test_controller_startup_timeout_comes_from_yaml_mapping():
    robot_config = {
        "controller_startup_timeout": {
            "sim": 42.5,
            "hardware": 7.5,
        }
    }

    assert robot_launch._resolve_controller_startup_timeout(robot_config, use_sim=True) == 42.5
    assert robot_launch._resolve_controller_startup_timeout(robot_config, use_sim=False) == 7.5


def test_gazebo_start_backend_uses_readiness_probe_instead_of_timer():
    adapter = get_sim_backend("gazebo")
    actions, create_node = adapter.start_backend({"name": "test_robot", "gazebo_world_name": "demo"})

    assert isinstance(create_node, Node)
    assert any(isinstance(action, Node) for action in actions)
    assert any(isinstance(action, RegisterEventHandler) for action in actions)


def test_shared_loader_preserves_source_path_metadata():
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_single_arm_legacy.yaml"
    robot_config = load_robot_config_dict(config_path)

    assert robot_config["name"] == "so101_single_arm"
    assert robot_config["_config_path"] == str(config_path.resolve())
    assert robot_config["_config_sources"] == [str(config_path.resolve())]


def test_launch_loader_uses_shared_dict_loader():
    robot_config = robot_launch.load_robot_config("so101_single_arm_legacy")

    assert robot_config["name"] == "so101_single_arm"
    assert robot_config["_config_path"].endswith("config/robots/so101_single_arm_legacy.yaml")


def test_default_trace_session_auto_suffixes_on_collision(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tracing_builder, "_trace_session_exists", lambda name: name == tracing_builder.DEFAULT_TRACE_SESSION_NAME
    )
    monkeypatch.setattr(
        tracing_builder,
        "datetime",
        SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _fmt: "20260428_180000")),
    )

    session_name, trace_dir = tracing_builder._resolve_trace_session(
        tracing_builder.DEFAULT_TRACE_SESSION_NAME,
        tmp_path,
    )

    assert session_name == "ib_robot_trace_20260428_180000"
    assert trace_dir == tmp_path / session_name


def test_custom_trace_session_collision_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing_builder, "_trace_session_exists", lambda _name: True)

    try:
        tracing_builder._resolve_trace_session("custom_trace", tmp_path)
    except RuntimeError as exc:
        assert "custom_trace" in str(exc)
    else:
        raise AssertionError("Expected custom tracing session collision to raise RuntimeError")


def test_tracing_startup_does_not_import_or_write_topology(monkeypatch, tmp_path):
    monkeypatch.delenv("IB_TRACE_ENABLED", raising=False)
    monkeypatch.setattr(tracing_builder, "_trace_session_exists", lambda _name: False)
    commands = []
    monkeypatch.setattr(tracing_builder, "_run_trace_command", lambda command, _reason: commands.append(command))
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"ibrobot_tracing.topology", "robot_config.tracing_topology"}:
            raise ImportError("optional topology is unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    spec = importlib.util.spec_from_file_location("tracing_without_topology", tracing_builder.__file__)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

    actions = tracing_builder.generate_tracing_actions(True, "test_trace", tmp_path)

    assert len(actions) == 1
    assert isinstance(actions[0], RegisterEventHandler)
    assert [command[1] for command in commands] == ["create", "enable-event", "enable-event", "start"]
    assert not (tmp_path / "test_trace" / "ibrobot-topology.json").exists()
    assert "IB_TRACE_ENABLED" not in os.environ


@pytest.mark.parametrize("previous_flag", [None, "0", "1"])
@pytest.mark.parametrize("fail_in_scope", [False, True])
def test_tracing_environment_is_scoped_for_immediate_and_deferred_processes(monkeypatch, previous_flag, fail_in_scope):
    if previous_flag is None:
        monkeypatch.delenv("IB_TRACE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("IB_TRACE_ENABLED", previous_flag)
    context = LaunchContext()
    original_environment = dict(context.environment)
    original_handlers = tuple(context._event_handlers)
    runtime_environment = {"RUNTIME_OPTION": "keep", "IB_TRACE_ENABLED": "0"}
    inherited = ExecuteProcess(cmd=["true"])
    explicit = ExecuteProcess(cmd=["true"], env=runtime_environment)
    additional = ExecuteProcess(cmd=["true"], additional_env={"IB_TRACE_ENABLED": "0", "EXTRA": "keep"})
    deferred = ExecuteProcess(cmd=["true"])
    deferred_explicit = ExecuteProcess(cmd=["true"], env=runtime_environment)
    deferred_handler = robot_launch._start_actions_on_success(
        [tracing_builder.scope_tracing_environment([deferred, deferred_explicit])],
        success_message="ready",
        failure_reason="not ready",
    )
    constructed_later = []

    def build_in_scope(context):
        assert context.environment["IB_TRACE_ENABLED"] == "1"
        if fail_in_scope:
            raise RuntimeError("business launch failed")
        constructed_later.append(ExecuteProcess(cmd=["true"]))
        return constructed_later

    def prepare(action):
        if isinstance(action, ExecuteProcess):
            action.process_description.prepare(context, action)
        else:
            for child in action.execute(context) or []:
                prepare(child)

    group = tracing_builder.scope_tracing_environment(
        [inherited, explicit, additional, OpaqueFunction(function=build_in_scope)]
    )
    if fail_in_scope:
        with pytest.raises(RuntimeError, match="business launch failed"):
            prepare(group)
        shutdown = Shutdown(reason="business launch failed")
        for handler in tuple(context._event_handlers):
            if handler.matches(shutdown):
                for action in handler.handle(shutdown, context):
                    prepare(action)
        assert context.environment == original_environment
        assert LaunchContext().environment.get("IB_TRACE_ENABLED") == previous_flag
        return

    prepare(group)
    assert context.environment == original_environment
    assert tuple(context._event_handlers) == original_handlers
    for action in deferred_handler(SimpleNamespace(returncode=0), context):
        prepare(action)

    for process in [inherited, explicit, additional, deferred, deferred_explicit, *constructed_later]:
        assert process.process_description.final_env["IB_TRACE_ENABLED"] == "1"
    assert explicit.process_description.final_env["RUNTIME_OPTION"] == "keep"
    assert additional.process_description.final_env["EXTRA"] == "keep"
    assert runtime_environment == {"RUNTIME_OPTION": "keep", "IB_TRACE_ENABLED": "0"}
    assert context.environment == original_environment
    assert os.environ.get("IB_TRACE_ENABLED") == previous_flag
    subsequent = ExecuteProcess(cmd=["true"])
    prepare(subsequent)
    assert subsequent.process_description.final_env.get("IB_TRACE_ENABLED") == previous_flag
    assert LaunchContext().environment.get("IB_TRACE_ENABLED") == previous_flag


@pytest.mark.parametrize("previous_flag", [None, "0", "1"])
@pytest.mark.parametrize("failure_stage", ["create", "ust", "python", "start"])
def test_tracing_failure_cleans_only_owned_session_without_changing_environment(
    monkeypatch, tmp_path, previous_flag, failure_stage
):
    if previous_flag is None:
        monkeypatch.delenv("IB_TRACE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("IB_TRACE_ENABLED", previous_flag)
    monkeypatch.setattr(tracing_builder, "_trace_session_exists", lambda _name: False)
    startup_commands = [
        ["lttng", "create", "test_trace", "--output", str(tmp_path / "test_trace")],
        ["lttng", "enable-event", "--session", "test_trace", "--userspace", "ros2:*"],
        ["lttng", "enable-event", "--session", "test_trace", "--python", "ib_trace.*"],
        ["lttng", "start", "test_trace"],
    ]
    failure_index = ["create", "ust", "python", "start"].index(failure_stage)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        assert os.environ.get("IB_TRACE_ENABLED") == previous_flag
        failed = failure_index < len(startup_commands) and command == startup_commands[failure_index]
        return SimpleNamespace(returncode=int(failed), stderr=f"{failure_stage} failed" if failed else "", stdout="")

    monkeypatch.setattr(tracing_builder.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        tracing_builder.generate_tracing_actions(True, "test_trace", tmp_path)

    assert os.environ.get("IB_TRACE_ENABLED") == previous_flag
    expected_commands = startup_commands[: failure_index + 1]
    if failure_stage != "create":
        expected_commands += [["lttng", "stop", "test_trace"], ["lttng", "destroy", "test_trace"]]
    assert commands == expected_commands


@pytest.mark.parametrize("previous_flag", [None, "0", "1"])
def test_tracing_keeps_startup_error_when_cleanup_raises(monkeypatch, tmp_path, previous_flag):
    if previous_flag is None:
        monkeypatch.delenv("IB_TRACE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("IB_TRACE_ENABLED", previous_flag)
    monkeypatch.setattr(tracing_builder, "_trace_session_exists", lambda _name: False)

    def run(command, _reason):
        if command[1] == "start":
            raise RuntimeError("start failed")

    def cleanup(*_args):
        raise OSError("cleanup failed")

    monkeypatch.setattr(tracing_builder, "_run_trace_command", run)
    monkeypatch.setattr(
        tracing_builder,
        "_make_trace_shutdown_handler",
        lambda _name: cleanup,
    )

    with pytest.raises(RuntimeError, match="start failed"):
        tracing_builder.generate_tracing_actions(True, "test_trace", tmp_path)

    assert os.environ.get("IB_TRACE_ENABLED") == previous_flag


def test_generate_navigation_nodes_for_lekiwi_mode():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": True,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        }
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_lekiwi_sim_uses_sim_controller_override():
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "lekiwi.yaml"
    robot_config = load_robot_config_dict(config_path)

    _nodes, controller_names, _deferred_spawners, _robot_description = generate_ros2_control_nodes(
        robot_config,
        use_sim=True,
        auto_start_controllers="true",
    )

    assert controller_names == ["joint_state_broadcaster", "base_controller"]


def test_generate_navigation_nodes_honors_force_enable_override():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": False,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        },
        force_enable=True,
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_launch_setup_enables_navigation_when_requested():
    context = LaunchContext()
    context.launch_configurations["robot_config"] = "lekiwi"
    context.launch_configurations["use_sim"] = "false"
    context.launch_configurations["auto_start_controllers"] = "false"
    context.launch_configurations["control_mode"] = "teleop"
    context.launch_configurations["with_navigation"] = "true"
    context.launch_configurations["navigation_mode"] = "full"

    actions = robot_launch.launch_setup(context)
    nav_nodes = [
        action for action in actions if isinstance(action, Node) and action.node_package == "robot_localization"
    ]

    assert len(nav_nodes) == 1


def test_simulation_scheduler_is_rejected_before_generating_nodes(tmp_path, monkeypatch):
    src_config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_single_arm.yaml"
    # The migrated provider config binds logical interfaces against the live
    # runtime at launch time; offline structural loads defer that binding.
    robot_config = load_robot_config_dict(src_config_path, defer_interface_binding=True)
    robot_config.pop("_config_path", None)
    robot_config["control_modes"]["model_inference"]["inference"]["scheduler"] = {"enable": True}
    config_path = tmp_path / "scheduled_sim.yaml"
    config_path.write_text(yaml.safe_dump({"robot": robot_config}, sort_keys=False), encoding="utf-8")
    context = LaunchContext()
    context.launch_configurations.update(
        config_path=str(config_path),
        use_sim="true",
        # The migrated provider config supports the SDK simulated transport;
        # the scheduler/simulation rejection below must fire before any nodes.
        sim_platform="sdk",
        control_mode="model_inference",
        with_inference="true",
    )

    def unexpected_control_generation(*args, **kwargs):
        pytest.fail("control generation must not run for unsupported scheduled simulation")

    monkeypatch.setattr(robot_launch, "generate_ros2_control_nodes", unexpected_control_generation)
    with pytest.raises(ValueError, match="simulation does not support"):
        robot_launch.launch_setup(context)


@pytest.mark.parametrize("enable_tracing", [False, True])
# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_launch_setup_uses_mock_sim_backend_without_controllers(monkeypatch, tmp_path, enable_tracing):
    monkeypatch.delenv("IB_TRACE_ENABLED", raising=False)
    monkeypatch.setattr(tracing_builder, "_resolve_trace_session", lambda name, _root: (name, tmp_path / name))
    monkeypatch.setattr(tracing_builder, "_run_trace_command", lambda _command, _reason: None)
    src_config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_single_arm_legacy.yaml"
    robot_config = load_robot_config_dict(src_config_path)
    robot_config.pop("_config_path", None)
    bundle = _create_inference_bundle(tmp_path / "model")
    robot_config["control_modes"]["model_inference"]["inference"]["pipelines"]["policy"]["model_path"] = str(bundle)

    config_path = tmp_path / "so101_mock.yaml"
    config_path.write_text(yaml.safe_dump({"robot": robot_config}, sort_keys=False), encoding="utf-8")

    context = LaunchContext()
    context.launch_configurations["robot_config"] = "so101_single_arm_legacy"
    context.launch_configurations["config_path"] = str(config_path)
    context.launch_configurations["use_sim"] = "true"
    context.launch_configurations["sim_platform"] = "mock"
    context.launch_configurations["auto_start_controllers"] = "true"
    context.launch_configurations["control_mode"] = "model_inference"
    context.launch_configurations["with_navigation"] = "false"
    context.launch_configurations["enable_tracing"] = str(enable_tracing).lower()

    actions = robot_launch.launch_setup(context)
    assert "IB_TRACE_ENABLED" not in os.environ
    if enable_tracing:
        assert len(actions) == 1
        assert isinstance(actions[0], GroupAction)
        actions = actions[0].get_sub_entities()
    node_packages = [action.node_package for action in actions if isinstance(action, Node)]

    assert node_packages.count("hardware_mock") == 1
    assert "controller_manager" not in node_packages
    assert "ros_gz_sim" not in node_packages

    timed_nodes = [
        action
        for action in actions
        if isinstance(action, Node) and action.node_executable in {"pipeline_policy_node", "action_dispatcher_node"}
    ]
    assert len(timed_nodes) == 2
    assert all(_node_parameters(node)["use_sim_time"] is False for node in timed_nodes)

    inference_nodes = [node for node in timed_nodes if node.node_package == "inference_service"]
    assert _node_parameters(inference_nodes[0])["use_sim"] is True
    inference_environment = {_text(key): _text(value) for key, value in inference_nodes[0].env}
    assert inference_environment.get("IB_TRACE_ENABLED") == ("1" if enable_tracing else None)


def test_launch_loader_preserves_config_path_for_runtime_consumers():
    robot_config = robot_launch.load_robot_config("lekiwi")

    assert robot_config["name"] == "lekiwi"
    assert robot_config["_config_path"].endswith("config/robots/lekiwi.yaml")


def test_inference_execution_mode_cli_override_targets_one_pipeline():
    robot_config = {
        "control_modes": {
            "model_inference": {
                "inference": {
                    "pipelines": {
                        "primary": {"execution_mode": "monolithic"},
                        "backup": {"execution_mode": "monolithic"},
                    }
                }
            }
        }
    }
    context = LaunchContext()
    context.launch_configurations["inference_pipeline"] = "backup"
    context.launch_configurations["inference_execution_mode"] = "distributed"

    robot_launch._apply_inference_cli_overrides(context, robot_config, "model_inference")

    pipelines = robot_config["control_modes"]["model_inference"]["inference"]["pipelines"]
    assert pipelines["primary"]["execution_mode"] == "monolithic"
    assert pipelines["backup"]["execution_mode"] == "distributed"


def test_inference_execution_mode_cli_override_requires_pipeline():
    context = LaunchContext()
    context.launch_configurations["inference_execution_mode"] = "distributed"

    with pytest.raises(ValueError, match="requires inference_pipeline"):
        robot_launch._apply_inference_cli_overrides(context, {}, "model_inference")


def test_inference_execution_mode_cli_override_rejects_unknown_pipeline():
    robot_config = {
        "control_modes": {"model_inference": {"inference": {"pipelines": {"policy": {"execution_mode": "monolithic"}}}}}
    }
    context = LaunchContext()
    context.launch_configurations["inference_pipeline"] = "missing"
    context.launch_configurations["inference_execution_mode"] = "distributed"

    with pytest.raises(ValueError, match="unknown pipeline"):
        robot_launch._apply_inference_cli_overrides(context, robot_config, "model_inference")


def test_generate_one_pipeline_node_uses_only_unified_parameters(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {
            "policy": {
                "model_path": str(bundle),
                "deployment": "cpu",
                "execution_mode": "monolithic",
                "request_timeout": 2.5,
                "default_task": "pick banana",
                "runtime_options": {"perf_enabled": True, "perf_log_every": 3},
                "transport": {
                    "action_server": "/custom/dispatch",
                    "reset_service": "/custom/reset",
                    "health_topic": "/custom/health",
                    "action_topic": "/custom/actions",
                },
            }
        },
    )

    nodes = generate_inference_node(robot_config, "model_inference")

    assert len(nodes) == 1
    assert nodes[0].node_executable == "pipeline_policy_node"
    params = _node_parameters(nodes[0])
    assert params["pipeline_id"] == "policy"
    assert params["model_path"] == str(bundle.resolve())
    assert params["deployment"] == "cpu"
    assert params["request_timeout"] == 2.5
    assert params["default_task"] == "pick banana"
    assert json.loads(params["runtime_options_json"]) == {"perf_enabled": True, "perf_log_every": 3}
    assert params["action_server"] == "/custom/dispatch"
    assert params["reset_service"] == "/custom/reset"
    assert params["health_topic"] == "/custom/health"
    assert params["action_topic"] == "/custom/actions"
    for legacy_field in ("checkpoint", "model", "models", "device", "policy_path"):
        assert legacy_field not in params


def test_generate_two_pipeline_nodes_and_route_dispatcher_to_selected_pipeline(tmp_path):
    first = _create_inference_bundle(tmp_path / "first")
    second = _create_inference_bundle(tmp_path / "second")
    pipelines = {
        "primary": {"model_path": str(first), "deployment": "cpu", "execution_mode": "monolithic"},
        "backup": {
            "model_path": str(second),
            "deployment": "cpu",
            "execution_mode": "monolithic",
            "transport": {
                "action_server": "/backup/dispatch",
                "reset_service": "/backup/reset",
            },
        },
    }
    robot_config = _inference_robot_config(tmp_path / "robot.yaml", pipelines, selection="backup")

    nodes = generate_execution_nodes(robot_config, "model_inference")

    inference_nodes = [node for node in nodes if node.node_package == "inference_service"]
    assert [node.node_executable for node in inference_nodes] == ["pipeline_policy_node", "pipeline_policy_node"]
    dispatcher = next(node for node in nodes if node.node_package == "action_dispatch")
    params = _node_parameters(dispatcher)
    assert params["inference_action_server"] == "/backup/dispatch"
    assert params["inference_reset_service"] == "/backup/reset"
    assert params["inference_timeout_sec"] == 5.0


def test_multiple_pipelines_require_explicit_executor_selection(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    pipelines = {
        "first": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
        "second": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
    }
    robot_config = _inference_robot_config(tmp_path / "robot.yaml", pipelines)

    with pytest.raises(InferenceConfigError, match="executor.inference_pipeline"):
        generate_action_dispatcher_node(robot_config, "model_inference")


def test_launch_generation_rejects_invalid_deployment(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {"policy": {"model_path": str(bundle), "deployment": "missing", "execution_mode": "monolithic"}},
    )

    with pytest.raises(InferenceConfigError, match="Deployment 'missing'"):
        generate_inference_node(robot_config, "model_inference")


def test_launch_generation_emits_distributed_edge_with_transport_parameters(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {"edge": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "distributed"}},
    )

    nodes = generate_inference_node(robot_config, "model_inference")

    assert len(nodes) == 1
    assert nodes[0].node_executable == "pipeline_policy_node"
    params = _node_parameters(nodes[0])
    assert params["execution_mode"] == "distributed"
    assert params["request_topic"] == "/inference/edge/request"
    assert params["result_topic"] == "/inference/edge/result"
    assert params["heartbeat_topic"] == "/inference/edge/heartbeat"
    assert params["video_descriptor_topic"] == "/inference/edge/video/descriptors"
    assert params["video_status_topic"] == "/inference/edge/video/status"


def test_launch_generation_supports_mixed_execution_modes(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {
            "local": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
            "edge": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "distributed"},
        },
        selection="local",
    )

    nodes = generate_execution_nodes(robot_config, "model_inference")

    inference_nodes = [node for node in nodes if node.node_package == "inference_service"]
    assert len(inference_nodes) == 2
    modes = [_node_parameters(node)["execution_mode"] for node in inference_nodes]
    assert modes == ["monolithic", "distributed"]


def test_launch_generation_rejects_pipeline_endpoint_conflicts(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {
            "first": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
            "second": {
                "model_path": str(bundle),
                "deployment": "cpu",
                "execution_mode": "monolithic",
                "transport": {"action_server": "/inference/first/dispatch"},
            },
        },
        selection="first",
    )

    with pytest.raises(InferenceConfigError, match="endpoint conflict"):
        generate_inference_node(robot_config, "model_inference")


def test_attention_viz_request_uses_robot_config_only():
    enabled, mode, _viz_config = _attention_viz_request({"attention_viz": {"enabled": False, "mode": "file"}})

    assert enabled is False
    assert mode == "file"


def test_generate_teleop_nodes_injects_target_joint_names_into_device_config():
    nodes = generate_teleop_nodes(
        {
            "joints": {
                "arm": ["1", "2", "3", "4", "5"],
                "gripper": ["6"],
            },
            "teleoperation": {
                "enabled": True,
                "active_device": "left_leader",
                "devices": [
                    {
                        "name": "left_leader",
                        "type": "leader_arm",
                        "port": "/dev/ttyACM0",
                        "target": {
                            "arm_joint_names": ["joint1_left", "joint2_left"],
                            "gripper_joint_names": ["joint6_left"],
                        },
                    }
                ],
                "safety": {
                    "joint_limits": {
                        "joint1_left": {"min": -1.0, "max": 1.0},
                        "joint2_left": {"min": -1.0, "max": 1.0},
                        "joint6_left": {"min": 0.0, "max": 1.0},
                    }
                },
            },
        }
    )

    params = _node_parameters(nodes[0])
    device_config = json.loads(params["device_config"].strip("'"))

    assert device_config["arm_joint_names"] == ["joint1_left", "joint2_left"]
    assert device_config["gripper_joint_names"] == ["joint6_left"]

    publish_groups = json.loads(params["publish_groups"].strip("'"))
    assert publish_groups == [
        {
            "name": "arm",
            "joint_names": ["joint1_left", "joint2_left"],
            "topic": "/arm_position_controller/commands",
        },
        {
            "name": "gripper",
            "joint_names": ["joint6_left"],
            "topic": "/gripper_position_controller/commands",
        },
    ]


def test_dual_arm_legacy_targets_keep_topics_and_joint_order(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_dual_arm.yaml"
    robot_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]
    calibration = tmp_path / "leader.json"
    calibration.write_text("{}\n", encoding="utf-8")
    for device in robot_config["teleoperation"]["devices"]:
        device["calib_file"] = str(calibration)

    nodes = generate_teleop_nodes(robot_config)

    assert len(nodes) == 2
    expected = [
        (
            ["joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left"],
            ["joint6_left"],
            "/arm_position_controller_left/commands",
            "/gripper_position_controller_left/commands",
        ),
        (
            ["joint1_right", "joint2_right", "joint3_right", "joint4_right", "joint5_right"],
            ["joint6_right"],
            "/arm_position_controller_right/commands",
            "/gripper_position_controller_right/commands",
        ),
    ]
    for node, (arm_joints, gripper_joints, arm_topic, gripper_topic) in zip(nodes, expected, strict=True):
        params = _node_parameters(node)
        assert params["arm_joint_names"] == arm_joints
        assert params["gripper_joint_names"] == gripper_joints
        assert params["arm_command_topic"] == arm_topic
        assert params["gripper_command_topic"] == gripper_topic
        assert json.loads(params["publish_groups"].strip("'")) == [
            {"name": "arm", "joint_names": arm_joints, "topic": arm_topic},
            {"name": "gripper", "joint_names": gripper_joints, "topic": gripper_topic},
        ]


def test_generate_auxiliary_aero_hand_node_uses_configured_topics_and_rates():
    joint_names = [f"hand_{index}" for index in range(7)]
    nodes = generate_auxiliary_actuator_nodes(
        {
            "teleoperation": {"safety": {"joint_limits": _joint_limits(joint_names)}},
            "auxiliary_actuators": [
                {
                    "name": "aero_hand_right",
                    "type": "aero_hand",
                    "mock": True,
                    "joint_names": joint_names,
                    "command_topic": "/aero_hand_right/commands",
                    "joint_state_topic": "/aero_hand_right/joint_states",
                    "command_frequency": 50.0,
                    "state_frequency": 20.0,
                }
            ],
        }
    )

    assert len(nodes) == 1
    params = _node_parameters(nodes[0])
    assert params["mock"] is True
    assert params["joint_names"] == joint_names
    assert params["command_topic"] == "/aero_hand_right/commands"
    assert params["joint_state_topic"] == "/aero_hand_right/joint_states"
    assert params["command_frequency"] == 50.0
    assert params["state_frequency"] == 20.0
    assert params["command_lower_limits"] == [-1.0] * 7
    assert params["command_upper_limits"] == [1.0] * 7
    assert params["estop_topic"] == "/emergency_stop"
    assert params["estop_behavior"] == "hold"
    assert "safe_pose" not in params


def test_generate_auxiliary_aero_hand_safe_pose_contract():
    safe_pose = [0.1 * index for index in range(7)]
    nodes = generate_auxiliary_actuator_nodes(
        {
            "auxiliary_actuators": {
                "aero_hand_right": {
                    "type": "aero_hand",
                    "mock": True,
                    "joint_names": [f"hand_{index}" for index in range(7)],
                    "command_topic": "/aero_hand_right/commands",
                    "joint_state_topic": "/aero_hand_right/joint_states",
                    "estop_topic": "/safety/stop",
                    "estop_behavior": "safe_pose",
                    "safe_pose": safe_pose,
                }
            }
        }
    )

    params = _node_parameters(nodes[0])
    assert params["estop_topic"] == "/safety/stop"
    assert params["estop_behavior"] == "safe_pose"
    assert params["safe_pose"] == safe_pose


def test_generate_auxiliary_actuator_resolves_port_environment(monkeypatch):
    monkeypatch.setenv("AERO_HAND_RIGHT_PORT", "/dev/aero-right")
    joint_names = [f"hand_{index}" for index in range(7)]
    nodes = generate_auxiliary_actuator_nodes(
        {
            "teleoperation": {"safety": {"joint_limits": _joint_limits(joint_names)}},
            "auxiliary_actuators": {
                "aero_hand_right": {
                    "type": "aero_hand",
                    "mock": False,
                    "port": "$(env AERO_HAND_RIGHT_PORT)",
                    "joint_names": joint_names,
                    "command_topic": "/hand/commands",
                    "joint_state_topic": "/hand/joint_states",
                }
            },
        }
    )

    assert _node_parameters(nodes[0])["port"] == "/dev/aero-right"


def test_auxiliary_aero_hands_reject_duplicate_real_serial_ports():
    joint_names = [f"hand_{index}" for index in range(7)]
    config = {
        "teleoperation": {
            "safety": {
                "joint_limits": _joint_limits(
                    [f"{joint}_{side}" for side in ("left", "right") for joint in joint_names]
                )
            }
        },
        "auxiliary_actuators": [
            {
                "name": side,
                "type": "aero_hand",
                "mock": False,
                "port": "/dev/aero",
                "joint_names": [f"{joint}_{side}" for joint in joint_names],
                "command_topic": f"/{side}/commands",
                "joint_state_topic": f"/{side}/joint_states",
            }
            for side in ("left", "right")
        ],
    }

    with pytest.raises(ValueError, match="serial port is used more than once"):
        generate_auxiliary_actuator_nodes(config)


def test_real_auxiliary_aero_hand_requires_hardware_boundary_limits():
    with pytest.raises(ValueError, match="requires safety joint_limits"):
        generate_auxiliary_actuator_nodes(
            {
                "auxiliary_actuators": {
                    "aero_hand_right": {
                        "type": "aero_hand",
                        "mock": False,
                        "port": "/dev/aero-right",
                        "joint_names": [f"hand_{index}" for index in range(7)],
                        "command_topic": "/hand/commands",
                        "joint_state_topic": "/hand/joint_states",
                    }
                }
            }
        )


def test_runtime_resources_reject_ros2_control_and_auxiliary_port_conflict():
    config = {
        "ros2_control": {"port": "/dev/robot"},
        "auxiliary_actuators": {
            "hand": {
                "type": "aero_hand",
                "mock": False,
                "port": "/dev/robot",
            }
        },
    }

    with pytest.raises(ValueError, match=r"/dev/robot: ros2_control, auxiliary_actuators.hand"):
        validate_runtime_resources(config)


def test_runtime_resources_reject_multiple_real_mhandpro_sdk_owners():
    config = {
        "hand_sources": {
            "left": {"type": "mhandpro", "mock": False, "sides": ["left"]},
            "right": {"type": "mhandpro", "mock": False, "sides": ["right"]},
        }
    }

    with pytest.raises(ValueError, match=r"mhandpro_sdk: hand_sources.left, hand_sources.right"):
        validate_runtime_resources(config)


def test_runtime_resources_allow_one_shared_mhandpro_and_disjoint_serial_ports():
    validate_runtime_resources(
        {
            "ros2_control": {"port": "/dev/follower"},
            "teleoperation": {
                "enabled": True,
                "active_devices": ["leader", "glove"],
                "devices": [
                    {"name": "leader", "type": "leader_arm", "port": "/dev/leader"},
                    {"name": "glove", "type": "hand_retarget"},
                ],
            },
            "auxiliary_actuators": {"hand": {"type": "aero_hand", "mock": False, "port": "/dev/aero"}},
            "hand_sources": {"mhandpro": {"type": "mhandpro", "mock": False, "sides": ["left", "right"]}},
        }
    )


def test_simulation_skips_real_external_hand_components(monkeypatch):
    monkeypatch.delenv("AERO_HAND_RIGHT_PORT", raising=False)
    monkeypatch.delenv("MHANDPRO_SDK_LIB", raising=False)
    config = {
        "auxiliary_actuators": {
            "hand": {
                "type": "aero_hand",
                "mock": False,
                "port": "$(env AERO_HAND_RIGHT_PORT)",
                "joint_names": _HAND_JOINTS,
                "command_topic": "/hand/commands",
                "joint_state_topic": "/hand/joint_states",
            }
        },
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": False,
                "lib_path": "$(env MHANDPRO_SDK_LIB)",
            }
        },
    }

    validate_runtime_resources(config, use_sim=True, control_mode="teleop")
    assert generate_auxiliary_actuator_nodes(config, use_sim=True, control_mode="teleop") == []
    assert generate_hand_source_nodes(config, use_sim=True, control_mode="teleop") == []


def test_external_hand_components_respect_active_control_modes():
    config = {
        "auxiliary_actuators": {
            "hand": {
                "type": "aero_hand",
                "mock": True,
                "active_control_modes": ["teleop"],
                "joint_names": _HAND_JOINTS,
                "command_topic": "/hand/commands",
                "joint_state_topic": "/hand/joint_states",
            }
        },
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": True,
                "active_control_modes": ["teleop"],
            }
        },
    }

    assert generate_auxiliary_actuator_nodes(config, control_mode="model_inference") == []
    assert generate_hand_source_nodes(config, control_mode="model_inference") == []
    assert len(generate_auxiliary_actuator_nodes(config, use_sim=True, control_mode="teleop")) == 1
    assert len(generate_hand_source_nodes(config, use_sim=True, control_mode="teleop")) == 1


def test_generate_generic_three_channel_auxiliary_actuator():
    nodes = generate_auxiliary_actuator_nodes(
        {
            "auxiliary_actuators": {
                "amazing_hand_right": {
                    "type": "amazing_hand",
                    "mock": True,
                    "driver": {"package": "amazing_hand_hardware", "executable": "amazing_hand_node"},
                    "joint_names": ["thumb", "index", "middle"],
                    "command_topic": "/amazing_hand_right/commands",
                    "joint_state_topic": "/amazing_hand_right/joint_states",
                    "parameters": {"device_id": "right"},
                }
            }
        }
    )

    assert len(nodes) == 1
    assert nodes[0].node_package == "amazing_hand_hardware"
    assert nodes[0].node_executable == "amazing_hand_node"
    params = _node_parameters(nodes[0])
    assert params["joint_names"] == ["thumb", "index", "middle"]
    assert params["device_id"] == "right"


def test_required_auxiliary_process_exits_shutdown_launch():
    nodes = generate_auxiliary_actuator_nodes(
        {
            "auxiliary_actuators": {
                "hand": {
                    "type": "aero_hand",
                    "mock": True,
                    "driver": {"package": "aero_hand_hardware", "executable": "aero_hand_node"},
                    "joint_names": _HAND_JOINTS,
                    "command_topic": "/hand/commands",
                    "joint_state_topic": "/hand/joint_states",
                }
            }
        }
    )

    assert nodes[0]._ExecuteLocal__on_exit is not None


def test_required_teleop_process_exits_shutdown_launch():
    nodes = generate_teleop_nodes(
        {
            "teleoperation": {
                "enabled": True,
                "active_device": "leader",
                "devices": [
                    {
                        "name": "leader",
                        "type": "leader_arm",
                        "port": "/dev/leader",
                        "target": {"arm_joint_names": ["1"], "gripper_joint_names": ["6"]},
                    }
                ],
            },
            "joints": {"arm": ["1"], "gripper": ["6"]},
            "safety": {"joint_limits": _joint_limits(["1", "6"])},
        },
        {},
    )

    assert nodes[0]._ExecuteLocal__on_exit is not None


def test_generate_mock_mhandpro_source_for_both_sides():
    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": True,
                    "sides": ["left", "right"],
                    "topic_prefix": "/hands/mhandpro",
                }
            }
        }
    )

    assert len(nodes) == 1
    assert nodes[0].node_package == "robot_teleop"
    assert nodes[0].node_executable == "mhandpro_source_node"
    params = _node_parameters(nodes[0])
    assert params["sides"] == ["left", "right"]
    assert params["topic_prefix"] == "/hands/mhandpro"
    assert params["mock"] is True
    assert params["publish_raw_frame"] is False
    assert params["calibrate_p_pose_on_startup"] is False
    assert params["failure_policy"] == "require_all"
    assert params["auto_reconnect"] is True
    assert params["reconnect_initial_delay"] == 1.0
    assert params["reconnect_max_delay"] == 10.0
    assert params["reconnect_max_attempts"] == 0
    assert nodes[0]._ExecuteLocal__on_exit is None


def test_real_mhandpro_source_exits_shutdown_launch(tmp_path):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": False,
                    "lib_path": str(library),
                }
            }
        }
    )

    assert nodes[0]._ExecuteLocal__on_exit is not None


def test_mhandpro_raw_frame_publication_requires_explicit_opt_in():
    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": True,
                    "publish_raw_frame": True,
                }
            }
        }
    )

    assert _node_parameters(nodes[0])["publish_raw_frame"] is True


def test_interactive_startup_p_pose_prompts_and_enables_in_process_calibration(monkeypatch, tmp_path):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    config = {
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": False,
                "lib_path": str(library),
                "sides": ["right"],
                "require_p_pose": True,
                "startup_p_pose": "interactive",
            }
        }
    }
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt))

    confirm_interactive_startup_p_pose(config, use_sim=False, control_mode="teleop")
    nodes = generate_hand_source_nodes(config, use_sim=False, control_mode="teleop")

    assert len(prompts) == 1
    assert "P-pose" in prompts[0]
    assert _node_parameters(nodes[0])["calibrate_p_pose_on_startup"] is True


def test_interactive_startup_p_pose_requires_foreground_terminal(monkeypatch, tmp_path):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    config = {
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": False,
                "lib_path": str(library),
                "startup_p_pose": "interactive",
            }
        }
    }
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

    with pytest.raises(RuntimeError, match="requires a terminal"):
        confirm_interactive_startup_p_pose(config, use_sim=False, control_mode="teleop")


def test_unknown_startup_p_pose_mode_is_rejected():
    with pytest.raises(ValueError, match="startup_p_pose"):
        generate_hand_source_nodes(
            {
                "hand_sources": {
                    "mhandpro": {
                        "type": "mhandpro",
                        "mock": True,
                        "startup_p_pose": "automatic",
                    }
                }
            }
        )


@pytest.mark.parametrize(
    ("profile", "sides", "actuator_count", "device_count"),
    [
        ("right", ["right"], 1, 1),
        ("left", ["left"], 1, 1),
        ("dual", ["left", "right"], 2, 2),
    ],
)
def test_hand_profile_selects_sides_actuators_and_devices(profile, sides, actuator_count, device_count):
    config = {
        "joints": {"hand": []},
        "hand_profiles": {
            "default_profile": "right",
            "profiles": {
                "right": {
                    "sides": ["right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_right"],
                    "active_devices": ["aero_glove_right"],
                },
                "left": {
                    "sides": ["left"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_left"],
                    "active_devices": ["aero_glove_left"],
                },
                "dual": {
                    "sides": ["left", "right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_left", "aero_hand_right"],
                    "active_devices": ["aero_glove_left", "aero_glove_right"],
                },
            },
        },
        "hand_sources": {"mhandpro": {"type": "mhandpro", "mock": True}},
        "auxiliary_actuators": {
            "aero_hand_left": {
                "profile_managed": True,
                "enabled": False,
                "joint_names": [f"left_{index}" for index in range(7)],
            },
            "aero_hand_right": {
                "profile_managed": True,
                "enabled": False,
                "joint_names": [f"right_{index}" for index in range(7)],
            },
        },
        "teleoperation": {
            "devices": [{"name": "aero_glove_left"}, {"name": "aero_glove_right"}],
        },
    }

    assert apply_hand_profile(config, profile) == profile
    assert config["hand_sources"]["mhandpro"]["sides"] == sides
    assert sum(item["enabled"] for item in config["auxiliary_actuators"].values()) == actuator_count
    assert len(config["teleoperation"]["active_devices"]) == device_count
    assert len(config["joints"]["hand"]) == actuator_count * 7


def test_hand_profile_defaults_to_robot_yaml_selection():
    config = {
        "joints": {"hand": []},
        "hand_profiles": {
            "default_profile": "right",
            "profiles": {
                "right": {
                    "sides": ["right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_right"],
                    "active_devices": ["aero_glove_right"],
                }
            },
        },
        "hand_sources": {"mhandpro": {}},
        "auxiliary_actuators": {
            "aero_hand_right": {
                "profile_managed": True,
                "enabled": False,
                "joint_names": [f"right_{index}" for index in range(7)],
            }
        },
        "teleoperation": {"devices": [{"name": "aero_glove_right"}]},
    }

    assert apply_hand_profile(config) == "right"
    assert config["teleoperation"]["active_devices"] == ["aero_glove_right"]


def test_unknown_hand_profile_is_rejected():
    config = {"hand_profiles": {"default_profile": "right", "profiles": {"right": {}}}}

    with pytest.raises(ValueError, match="unknown hand_profile"):
        apply_hand_profile(config, "triple")


def test_mhandpro_source_rejects_unknown_dual_hand_failure_policy():
    with pytest.raises(ValueError, match="failure_policy"):
        generate_hand_source_nodes(
            {
                "hand_sources": {
                    "mhandpro": {
                        "type": "mhandpro",
                        "mock": True,
                        "sides": ["left", "right"],
                        "failure_policy": "ignore_missing",
                    }
                }
            }
        )


def test_real_mhandpro_source_resolves_external_library(tmp_path, monkeypatch):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    monkeypatch.setenv("MHANDPRO_SDK_LIB", str(library))

    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": False,
                    "lib_path": "$(env MHANDPRO_SDK_LIB)",
                }
            }
        }
    )

    assert _node_parameters(nodes[0])["lib_path"] == str(library)


def test_real_mhandpro_source_requires_external_library_when_environment_is_unset(monkeypatch):
    monkeypatch.delenv("MHANDPRO_SDK_LIB", raising=False)

    with pytest.raises(RuntimeError, match="external vendor library"):
        generate_hand_source_nodes(
            {
                "hand_sources": {
                    "mhandpro": {
                        "type": "mhandpro",
                        "mock": False,
                        "lib_path": "$(env MHANDPRO_SDK_LIB)",
                    }
                }
            }
        )


def test_hand_retarget_resolves_output_contract_from_auxiliary_actuator():
    joints = ["thumb", "index", "middle"]
    nodes = generate_teleop_nodes(
        {
            "joints": {"arm": ["1"], "gripper": ["6"]},
            "auxiliary_actuators": {
                "amazing_hand_right": {
                    "type": "amazing_hand",
                    "mock": True,
                    "driver": {"package": "amazing_hand_hardware", "executable": "amazing_hand_node"},
                    "joint_names": joints,
                    "command_topic": "/amazing_hand_right/commands",
                    "joint_state_topic": "/amazing_hand_right/joint_states",
                }
            },
            "teleoperation": {
                "enabled": True,
                "active_device": "amazing_retarget",
                "safety": {"joint_limits": _joint_limits(joints)},
                "devices": [
                    {
                        "name": "amazing_retarget",
                        "type": "hand_retarget",
                        "side": "right",
                        "source_topic": "/hand_sources/mhandpro/right/state",
                        "retargeter": {
                            "type": "synergy_matrix",
                            "input_features": ["index_mcp_flex"],
                            "matrix": [[1.0], [1.0], [1.0]],
                        },
                        "target": {"actuator": "amazing_hand_right"},
                    }
                ],
            },
        }
    )

    params = _node_parameters(nodes[0])
    assert json.loads(params["publish_groups"].strip("'")) == [
        {"name": "hand", "joint_names": joints, "topic": "/amazing_hand_right/commands"}
    ]
    device_config = json.loads(params["device_config"].strip("'"))
    assert device_config["joint_names"] == joints


def test_phone_placo_uses_explicit_arm_group_topic():
    nodes = generate_teleop_nodes(
        {
            "joints": {"arm": ["1", "2"], "gripper": ["6"]},
            "moveit": {
                "base_link": "base",
                "ee_link": "gripper",
                "so101_placo_servo_config_path": "$(find so101_motion)/config/so101_placo_servo.yaml",
            },
            "ros2_control": {"reset_positions": {"1": 0.0, "2": 0.0}},
            "teleoperation": {
                "enabled": True,
                "active_device": "phone",
                "cartesian": {"solver": "placo_servo"},
                "safety": {"joint_limits": _joint_limits(["1", "2", "6"])},
                "devices": [
                    {
                        "name": "phone",
                        "type": "phone",
                        "phone_config": {
                            "backend": "webphone",
                            "end_effector_bounds": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]},
                        },
                        "target": {
                            "publish_groups": [
                                {"name": "arm", "joint_names": ["1", "2"], "topic": "/custom/arm"},
                                {"name": "gripper", "joint_names": ["6"], "topic": "/custom/gripper"},
                            ]
                        },
                    }
                ],
            },
        }
    )

    teleop_params = _node_parameters(nodes[0])
    device_config = json.loads(teleop_params["device_config"].strip("'"))
    placo = device_config["cartesian_backend_config"]
    assert placo["command_lease_topic"] == "/so101_placo_servo_node/command_lease"
    placo_node = next(node for node in nodes if _text(node.node_executable) == "so101_placo_servo_node.py")
    assert _node_parameters(placo_node)["command_out_topic"] == "/custom/arm"


def test_synergy_hand_retarget_does_not_require_aero_calibration():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "amazing",
            "safety": {"joint_limits": _joint_limits(["thumb", "index", "middle"])},
            "devices": [
                {
                    "name": "amazing",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0], [1.0], [1.0]],
                    },
                    "target": {
                        "publish_groups": [
                            {
                                "name": "hand",
                                "joint_names": ["thumb", "index", "middle"],
                                "topic": "/amazing/commands",
                            }
                        ]
                    },
                }
            ],
        }
    )

    assert errors == []


def test_active_devices_allow_disjoint_arm_and_hand_topics():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_devices": ["leader", "glove"],
            "safety": {"joint_limits": _joint_limits(["1", *_HAND_JOINTS])},
            "devices": [
                {"name": "leader", "type": "leader_arm", "port": "/dev/leader"},
                {
                    "name": "glove",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0]] * 7,
                    },
                    "target": {
                        "publish_groups": [
                            {"name": "hand", "joint_names": _HAND_JOINTS, "topic": "/aero_hand/commands"}
                        ]
                    },
                },
            ],
        }
    )

    assert errors == []


def test_vr_and_glove_allow_disjoint_command_topics():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_devices": ["vr", "glove"],
            "safety": {"joint_limits": _joint_limits(_HAND_JOINTS)},
            "devices": [
                {"name": "vr", "type": "vr_teleop", "vr_config": {"output_profile": "so101"}},
                {
                    "name": "glove",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0]] * 7,
                    },
                    "target": {
                        "publish_groups": [
                            {"name": "hand", "joint_names": _HAND_JOINTS, "topic": "/aero_hand/commands"}
                        ]
                    },
                },
            ],
        }
    )

    assert errors == []


def test_two_gloves_reject_shared_aero_hand_topic():
    devices = []
    for name in ("first", "second"):
        devices.append(
            {
                "name": name,
                "type": "hand_retarget",
                "side": "right",
                "source_topic": f"/hands/{name}/state",
                "retargeter": {
                    "type": "synergy_matrix",
                    "input_features": ["index_mcp_flex"],
                    "matrix": [[1.0]] * 7,
                },
                "target": {
                    "publish_groups": [{"name": "hand", "joint_names": _HAND_JOINTS, "topic": "/aero_hand/commands"}]
                },
            }
        )

    errors = validate_teleop_config({"enabled": True, "active_devices": ["first", "second"], "devices": devices})

    assert any("share command topic" in error for error in errors)


def test_hand_retarget_requires_explicit_publish_group():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "glove",
            "devices": [
                {
                    "name": "glove",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0]] * 7,
                    },
                }
            ],
        }
    )

    assert any("requires explicit target.publish_groups" in error for error in errors)


def test_vr_rejects_target_override_that_does_not_match_its_builder():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "vr",
            "devices": [
                {
                    "name": "vr",
                    "type": "vr_teleop",
                    "vr_config": {"output_profile": "so101"},
                    "target": {
                        "publish_groups": [{"name": "other", "joint_names": ["1"], "topic": "/not_the_vr_output"}]
                    },
                }
            ],
        }
    )

    assert any("vr_teleop uses vr_config outputs" in error for error in errors)


@pytest.mark.parametrize("second_type", ["leader_arm", "phone", "xbox_controller"])
def test_active_arm_devices_reject_shared_command_topics(second_type):
    second = {"name": "second", "type": second_type}
    if second_type == "leader_arm":
        second["port"] = "/dev/leader2"
    if second_type == "phone":
        second["phone_config"] = {"phone_os": "android"}

    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_devices": ["leader", "second"],
            "safety": {"joint_limits": _joint_limits(["1"])},
            "devices": [
                {"name": "leader", "type": "leader_arm", "port": "/dev/leader"},
                second,
            ],
        }
    )

    assert any("share arm command topic" in error for error in errors)


def test_generate_joy_teleop_nodes_for_mobile_base():
    nodes = generate_teleop_nodes(
        {
            "teleoperation": {
                "enabled": True,
                "active_device": "lekiwi_gamepad",
                "devices": [
                    {
                        "name": "lekiwi_gamepad",
                        "type": "joy_teleop",
                        "config_path": "$(find robot_config)/config/lekiwi/lekiwi_teleop.yaml",
                        "input_device": "/dev/input/js0",
                    }
                ],
            }
        }
    )

    assert len(nodes) == 2
    assert all(isinstance(node, Node) for node in nodes)
