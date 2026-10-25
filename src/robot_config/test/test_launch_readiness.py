"""Unit tests for launch readiness helpers."""

import builtins
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from launch import LaunchContext
from launch.actions import ExecuteProcess, GroupAction, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
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
    generate_action_dispatcher_node,
    generate_execution_nodes,
    generate_inference_node,
)
from robot_config.launch_builders.perception import generate_camera_nodes, generate_tf_nodes
from robot_config.launch_builders.sim_backend import get_sim_backend
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


@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_hybrid_arm_state_stream_is_a_runtime_profile_broadcaster():
    """lekiwi_nav_grasp delegates the ros2_control chain to the runtime.

    The hybrid stage's old spawner test asserted that
    arm_joint_state_broadcaster stays active while base_velocity_controller
    owns the base. Under the provider (openspec
    lekiwi-application-migration task 1.2) that guarantee moved to the runtime
    profile: broadcasters are always-on, mode controllers are scoped, and the
    application configuration restates none of it.
    """
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "lekiwi_nav_grasp.yaml"
    profile_path = Path(__file__).resolve().parents[3] / "src/robots/lekiwi/lekiwi_robot/profiles/lekiwi_nav_grasp.yaml"
    robot_config = load_robot_config_dict(config_path)
    profile = __import__("yaml").safe_load(profile_path.read_text(encoding="utf-8"))

    assert "ros2_control" not in robot_config
    assert profile["ros2_control"]["broadcasters"] == ["joint_state_broadcaster", "arm_joint_state_broadcaster"]
    assert profile["modes"]["base_navigation"]["controllers"] == ["base_velocity_controller"]
    assert profile["modes"]["trajectory"]["controllers"] == [
        "arm_trajectory_controller",
        "gripper_trajectory_controller",
    ]
    for mode in robot_config["control_modes"].values():
        assert "controllers" not in mode


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
    from launch.actions import EmitEvent, LogInfo
    from launch.events.process import ProcessExited

    from robot_config.launch_registry.orchestrator import _assemble_actions
    from robot_config.launch_registry.registry import LaunchBuildContext, PlanOutputs, PlanSegment

    barrier = ExecuteProcess(cmd=["never-executed"])
    consumer = LogInfo(msg="deferred")
    outputs = PlanOutputs(
        actions=[barrier, consumer],
        runtime_ready_actions=[consumer],
        segments=[
            PlanSegment("startup", "immediate", (barrier,)),
            PlanSegment("consumer", "runtime_ready", (consumer,)),
        ],
    )
    context = LaunchBuildContext(robot_config={}, controller_readiness_barrier=barrier)
    actions = _assemble_actions(SimpleNamespace(arguments={}), context, outputs, enable_tracing=False)
    assert isinstance(actions[0], RegisterEventHandler)
    assert actions[1:] == [barrier]
    for code in (0, 1):
        event = ProcessExited(action=barrier, name="barrier", cmd=[], cwd=None, env=None, pid=1, returncode=code)
        handler = actions[0].event_handler
        assert handler.matches(event)
        released = list(handler.handle(event, LaunchContext()))
        if code == 0:
            assert released == [consumer]
        else:
            assert len(released) == 1 and isinstance(released[0], EmitEvent)
            assert isinstance(released[0].event, Shutdown)


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


def test_robot_launch_description_generation_works_without_extensions(monkeypatch):
    """An environment without the optional business packages must still
    generate a launch description: the registry degrades to builtin-only and
    the migrated business entries (voice_asr among them, now an
    ibrobot.launch_builders extension) simply do not participate. The
    pre-migration companion blocked the voice_asr_service import; the
    migration turned that scenario into "extension package not installed",
    simulated here hermetically through the registry's injection seam."""
    original_build = robot_launch.build_launch_registry

    def builtin_only_registry():
        return original_build(entry_points=[])

    monkeypatch.setattr(robot_launch, "build_launch_registry", builtin_only_registry)

    assert robot_launch.generate_launch_description() is not None


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
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_arm_aero_hand.yaml"
    robot_config = load_robot_config_dict(config_path)

    assert robot_config["name"] == "so101_arm_aero_hand"
    assert robot_config["_config_path"] == str(config_path.resolve())
    assert robot_config["_config_sources"] == [str(config_path.resolve())]


def test_launch_loader_uses_shared_dict_loader():
    robot_config = robot_launch.load_robot_config("so101_arm_aero_hand")

    assert robot_config["name"] == "so101_arm_aero_hand"
    assert robot_config["_config_path"].endswith("config/robots/so101_arm_aero_hand.yaml")


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


def test_generate_tracing_actions_enables_python_trace_logging(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing_builder, "_trace_session_exists", lambda _name: False)
    monkeypatch.setattr(tracing_builder, "_run_trace_command", lambda _command, _reason: None)
    monkeypatch.setattr(tracing_builder, "_start_trace_session", lambda _name, _path: None)

    actions = tracing_builder.generate_tracing_actions(True, "test_trace", tmp_path)

    assert any(action.__class__.__name__ == "SetEnvironmentVariable" for action in actions)


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

    assert len(actions) == 2
    assert isinstance(actions[0], SetEnvironmentVariable)
    environment = LaunchContext()
    actions[0].execute(environment)
    assert environment.environment["IB_TRACE_LOG_LEVEL"] == "INFO"
    assert isinstance(actions[1], RegisterEventHandler)
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


def test_lekiwi_sim_uses_sim_controller_override():
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "lekiwi.yaml"
    robot_config = load_robot_config_dict(config_path)

    _nodes, controller_names, _deferred_spawners, _robot_description = generate_ros2_control_nodes(
        robot_config,
        use_sim=True,
        auto_start_controllers="true",
    )

    assert controller_names == ["joint_state_broadcaster", "base_controller"]


def test_launch_setup_enables_navigation_when_requested(monkeypatch):
    """The with_navigation switch still reaches the navigation entry's
    participation after the entry moved to robot_navigation: a synthetic
    navigation entry injected through the registry's seam (hermetic — no
    ambient extensions) participates for with_navigation:=true and its
    action reaches the assembled launch. The real node generation is
    covered by robot_navigation's suite."""
    from importlib.metadata import EntryPoint
    from types import ModuleType

    from robot_config.launch_registry.registry import BuilderSpec, BuildResult

    module = ModuleType("launch_readiness_synthetic_navigation")
    module.build_navigation = lambda _context: BuildResult(
        actions=[Node(package="robot_localization", executable="ekf_node", name="ekf_node")]
    )
    module.NAVIGATION_BUILDER = BuilderSpec(
        id="navigation",
        factory="launch_readiness_synthetic_navigation:build_navigation",
        arguments=(),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original_build = robot_launch.build_launch_registry

    def build_with_synthetic_navigation():
        return original_build(
            entry_points=[
                EntryPoint(
                    name="navigation",
                    value="launch_readiness_synthetic_navigation:NAVIGATION_BUILDER",
                    group="ibrobot.launch_builders",
                )
            ]
        )

    monkeypatch.setattr(robot_launch, "build_launch_registry", build_with_synthetic_navigation)
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


def test_one_registration_extends_production_entry_end_to_end(monkeypatch):
    """Task 6.2: adding ONE registration (a minimal module + a single
    BuilderSpec with its own CLI argument, prepare hook, enable switch and
    a runtime-ready action) must extend the REAL production entry end to
    end — the new CLI argument is enumerated by generate_launch_description,
    the hook runs inside the real preparation pipeline under its own config
    key, the enable switch controls participation, and the runtime-ready
    action is gated behind the REAL controller-readiness barrier (invisible
    in the immediate stream until the barrier exits; success releases it,
    failure shuts down without release) — with NO extra modification to the
    main launch file or the facade (design decision 4 / spec 6.2)."""
    import sys
    from types import ModuleType

    from launch.actions import EmitEvent, LogInfo
    from launch.event_handlers import OnProcessExit
    from launch.events.process import ProcessExited

    from robot_config.launch_registry import builtin_registry as builtin_registry_module
    from robot_config.launch_registry.launch_options import ArgumentSpec
    from robot_config.launch_registry.registry import BuilderSpec, BuildResult, HookSpec

    factory_calls = []
    built_actions = []

    def extension_hook(config, arguments):
        config["extension_probe"] = {"executed": True, "mode": arguments["extension_mode"]}

    def extension_build(context):
        # The hook's write under its own config key must be visible to the
        # factory through the real preparation -> plan data flow.
        factory_calls.append(
            (context.arguments["extension_mode"], dict(context.robot_config.get("extension_probe", {})))
        )
        action = LogInfo(msg="extension runtime-ready action")
        built_actions.append(action)
        return BuildResult(actions=[action])

    # One registration: the extension spec (CLI argument + prepare hook
    # with its own config key + enable switch + runtime-ready action) joins
    # the builtin table through the same build_launch_registry seam the
    # production entry uses (migrated from build_builtin_registry when the
    # entry switched to entry-point composition).
    original_build = builtin_registry_module.build_launch_registry

    def build_with_extension():
        registry = original_build(entry_points=[])
        registry.add(
            BuilderSpec(
                id="extension_probe",
                factory="test_launch_readiness_extension_probe:extension_build",
                timing="runtime_ready",
                enabled=lambda context: context.arguments.get("extension_mode") == "true",
                arguments=(
                    ArgumentSpec(
                        name="extension_mode",
                        declared_default="false",
                        absent_fallback="false",
                        description="Extension probe switch",
                    ),
                ),
                hooks=(
                    HookSpec(
                        phase="config_overrides",
                        factory="test_launch_readiness_extension_probe:extension_hook",
                        config_keys=("extension_probe",),
                    ),
                ),
            )
        )
        return registry

    monkeypatch.setattr(robot_launch, "build_launch_registry", build_with_extension)

    probe_module = ModuleType("test_launch_readiness_extension_probe")
    probe_module.extension_build = extension_build
    probe_module.extension_hook = extension_hook
    monkeypatch.setitem(sys.modules, probe_module.__name__, probe_module)

    # 1. Declaration stage: the contributed CLI argument is enumerated by
    # the real generate_launch_description without touching the entry file.
    description = robot_launch.generate_launch_description()
    declared_names = {getattr(entity, "name", None) for entity in description.entities}
    assert "extension_mode" in declared_names  # the new argument IS declared
    assert "robot_config" in declared_names  # core table still declared

    def _launch(extension_mode):
        context = LaunchContext()
        context.launch_configurations["robot_config"] = "lekiwi"
        context.launch_configurations["use_sim"] = "false"
        # auto_start_controllers=true arms the REAL controller readiness
        # barrier (the controller group spawner) of this hardware launch.
        context.launch_configurations["auto_start_controllers"] = "true"
        context.launch_configurations["control_mode"] = "teleop"
        context.launch_configurations["extension_mode"] = extension_mode
        return robot_launch.launch_setup(context)

    # 2. Enabled branch: the hook ran inside the real preparation pipeline
    # (its own config key is visible to the factory), the factory ran once
    # with the CLI value, and the runtime-ready action is deferred behind
    # the real barrier — invisible in the immediate stream.
    actions = _launch("true")
    assert factory_calls == [("true", {"executed": True, "mode": "true"})]
    extension_action = built_actions[0]
    assert extension_action not in actions  # gated: invisible before the barrier exits

    barrier = next(
        action
        for action in actions
        if isinstance(action, Node) and action.node_executable in {"controller_spawner", "wait_for_controllers"}
    )
    assert isinstance(barrier, ExecuteProcess)  # the real readiness barrier process
    exit_handlers = [
        action.event_handler
        for action in actions
        if isinstance(action, RegisterEventHandler) and isinstance(action.event_handler, OnProcessExit)
    ]
    assert exit_handlers  # the success-gated consumer handler is registered before the barrier

    event_context = LaunchContext()
    for returncode in (0, 1):
        event = ProcessExited(action=barrier, name="barrier", cmd=[], cwd=None, env=None, pid=1, returncode=returncode)
        handler = next(handler for handler in exit_handlers if handler.matches(event))
        released = list(handler.handle(event, event_context))
        if returncode == 0:
            # Success releases the deferred set: the extension action rides
            # the same readiness gating as the builtin runtime-ready entries.
            assert extension_action in released
            assert any(isinstance(action, Node) for action in released)
        else:
            # Failure shuts down without releasing the deferred set.
            assert len(released) == 1 and isinstance(released[0], EmitEvent)
            assert isinstance(released[0].event, Shutdown)
            assert extension_action not in released

    # 3. Disabled branch: the same registration with the switch off never
    # resolves the factory (zero calls, zero actions created).
    disabled_actions = _launch("false")
    assert len(factory_calls) == 1
    assert len(built_actions) == 1
    assert extension_action not in disabled_actions


@pytest.mark.parametrize(
    ("target", "execution_mode", "transport_mode"),
    [
        (None, "monolithic", "dds"),  # simulation selected by CLI use_sim alone
        ("simulation", "monolithic", "dds"),
        ("benchmark", "monolithic", "dds"),
        ("benchmark", "distributed", "dds"),
        ("benchmark", "distributed", "rtp"),
    ],
)
def test_virtual_target_launch_keeps_robot_scheduler_dormant(
    tmp_path, monkeypatch, target, execution_mode, transport_mode
):
    from copy import deepcopy

    from robot_config.launch_builders.benchmark import BENCHMARK_BUILDER
    from robot_config.launch_builders.execution import EXECUTION_BUILDER
    from robot_config.launch_registry.registry import LaunchBuilderRegistry

    bundle = _create_inference_bundle(tmp_path / "bundle")
    config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {"policy": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": execution_mode}},
        selection="policy",
    )
    config["runtime"] = {"target": "hardware"}
    if target is None:
        config.pop("runtime")
    config["default_control_mode"] = "model_inference"
    config["runtime_scheduler"] = {"enable": True}  # dormant tuning needs no scheduled endpoints
    config["contract"] = {
        "observations": [
            {
                "key": "observation.images.image",
                "topic": "/camera/image",
                "type": "sensor_msgs/msg/Image",
                "image": {"encoding": "rgb8", "resize": [32, 32]},
            }
        ],
        "actions": [],
    }
    if target == "benchmark":
        config["benchmark"] = {
            "type": "libero",
            "adapter": "libero",
            "evaluation": {
                "enabled": True,
                "suite": "libero_spatial",
                "observation_transport": {"mode": transport_mode},
                "output": {"root": str(tmp_path / "output")},
            },
        }
        mode = config["control_modes"]["model_inference"]
        mode["executor"]["type"] = "benchmark"
        mode["dispatch"] = {"scheduler": "wait_for_feedback"}
    original = deepcopy(config)
    registry = LaunchBuilderRegistry()
    registry.add(BENCHMARK_BUILDER)
    registry.add(EXECUTION_BUILDER)
    # Isolate registration and local DDS admission; the production preparation,
    # orchestrator, benchmark and execution builders all run without stubs.
    monkeypatch.setattr(robot_launch, "build_launch_registry", lambda: registry)
    monkeypatch.setattr(robot_launch, "check_launch_dds", lambda context: None)
    monkeypatch.delenv("WORKSPACE", raising=False)  # materialize run YAML inside tmp_path
    context = LaunchContext()
    context.launch_configurations.update(config_path=str(tmp_path / "robot.yaml"), with_inference="true")
    if target is not None:
        context.launch_configurations.update(runtime_target=target, sim_platform="mock")
    else:
        context.launch_configurations.update(use_sim="true")

    actions = robot_launch.launch_setup(context, loaded_config=config)
    nodes = {node.node_executable: node for node in actions if isinstance(node, Node)}
    expected = {"pipeline_policy_node", "action_dispatcher_node"}
    if target == "benchmark":
        expected.update({"benchmark_environment_node", "benchmark_evaluator_node"})
    if execution_mode == "distributed":
        expected.add("pure_inference_node")
    assert set(nodes) == expected
    policy = _node_parameters(nodes["pipeline_policy_node"])
    assert "runtime_policy_json" not in policy
    assert config == original
    if target == "benchmark":
        evaluator = _node_parameters(nodes["benchmark_evaluator_node"])
        assert evaluator["inference_health_topic"] == policy["health_topic"] == "/inference/policy/health"
        assert evaluator["prepare_policy_episode_service"] == "/action_dispatcher/prepare_episode"
        assert evaluator["run_policy_action_server"] == "/action_dispatcher/run_policy"
        run_config = yaml.safe_load(Path(evaluator["robot_config_path"]).read_text())["robot"]
        assert run_config["runtime"] == config["runtime"]
        assert "use_sim" not in run_config
        assert run_config["runtime_scheduler"]["enable"] is True
        assert policy["robot_config_path"] == evaluator["robot_config_path"]
        environment = _node_parameters(nodes["benchmark_environment_node"])
        assert environment["frame_ingress_enabled"] is (transport_mode == "rtp")
        if transport_mode == "rtp":
            ingress = json.loads(environment["frame_ingress_contract_json"])
            assert ingress["pipeline_id"] == "policy"
            assert ingress["heartbeat_topic"] == policy["heartbeat_topic"]
            assert len(ingress["streams"]) == 1


@pytest.mark.parametrize("enable_tracing", [False, True])
# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_launch_setup_uses_mock_sim_backend_without_controllers(monkeypatch, tmp_path, enable_tracing):
    monkeypatch.delenv("IB_TRACE_ENABLED", raising=False)
    monkeypatch.setattr(tracing_builder, "_resolve_trace_session", lambda name, _root: (name, tmp_path / name))
    monkeypatch.setattr(tracing_builder, "_run_trace_command", lambda _command, _reason: None)
    src_config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_arm_aero_hand.yaml"
    robot_config = load_robot_config_dict(src_config_path)
    robot_config.pop("_config_path", None)
    bundle = _create_inference_bundle(tmp_path / "model")
    robot_config["control_modes"]["model_inference"]["inference"]["pipelines"]["policy"]["model_path"] = str(bundle)

    config_path = tmp_path / "so101_mock.yaml"
    config_path.write_text(yaml.safe_dump({"robot": robot_config}, sort_keys=False), encoding="utf-8")

    context = LaunchContext()
    context.launch_configurations["robot_config"] = "so101_arm_aero_hand"
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


def test_simulation_skips_real_auxiliary_actuators(monkeypatch):
    """The auxiliary_actuators half of the former mixed sim-skip test (the
    hand_sources half moved to robot_teleop with its entry)."""
    monkeypatch.delenv("AERO_HAND_RIGHT_PORT", raising=False)
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
    }

    validate_runtime_resources(config, use_sim=True, control_mode="teleop")
    assert generate_auxiliary_actuator_nodes(config, use_sim=True, control_mode="teleop") == []


def test_auxiliary_actuators_respect_active_control_modes():
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
    }

    assert generate_auxiliary_actuator_nodes(config, control_mode="model_inference") == []
    assert len(generate_auxiliary_actuator_nodes(config, use_sim=True, control_mode="teleop")) == 1


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
