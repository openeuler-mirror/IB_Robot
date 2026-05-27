"""Unit tests for launch readiness helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from launch import LaunchContext
from launch.actions import RegisterEventHandler
from launch_ros.actions import Node

from robot_config.launch_builders import tracing as tracing_builder
from robot_config.launch_builders.control import (
    generate_controller_spawners,
    generate_ros2_control_nodes,
)
from robot_config.launch_builders.execution import (
    _attention_viz_request,
    generate_inference_node,
)
from robot_config.launch_builders.navigation import generate_navigation_nodes
from robot_config.launch_builders.sim_backend import get_sim_backend
from robot_config.launch_builders.teleop import generate_teleop_nodes
from robot_config.loader import load_robot_config_dict
from robot_config.wait_for_controllers import missing_inactive_controllers

_LAUNCH_PATH = Path(__file__).resolve().parents[1] / "launch" / "robot.launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("robot_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None
assert _LAUNCH_SPEC.loader is not None
robot_launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(robot_launch)


def _text(substitutions):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in substitutions)


def _node_parameters(node):
    raw = node._Node__parameters[0]
    parsed = {}
    for key, value in raw.items():
        name = _text(key)
        if isinstance(value, tuple):
            parsed[name] = _text(value).strip()
        else:
            parsed[name] = value
    return parsed


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


def test_generate_controller_spawners_groups_activation():
    spawners = generate_controller_spawners(
        ["joint_state_broadcaster", "arm_position_controller"],
        use_sim=True,
    )

    assert len(spawners) == 1
    assert isinstance(spawners[0], Node)
    cmd_text = [item[0].text for item in spawners[0].cmd if item and hasattr(item[0], "text")]
    assert "--controller-manager" in cmd_text
    assert "controller_manager" in cmd_text
    assert "--activate-as-group" in cmd_text


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
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_single_arm.yaml"
    robot_config = load_robot_config_dict(config_path)

    assert robot_config["name"] == "so101_single_arm"
    assert robot_config["_config_path"] == str(config_path.resolve())


def test_launch_loader_uses_shared_dict_loader():
    robot_config = robot_launch.load_robot_config("so101_single_arm")

    assert robot_config["name"] == "so101_single_arm"
    assert robot_config["_config_path"].endswith("config/robots/so101_single_arm.yaml")


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


def test_launch_loader_preserves_config_path_for_runtime_consumers():
    robot_config = robot_launch.load_robot_config("lekiwi")

    assert robot_config["name"] == "lekiwi"
    assert robot_config["_config_path"].endswith("config/robots/lekiwi.yaml")


def test_generate_inference_node_binds_shared_rknn_resources(monkeypatch, tmp_path):
    workspace = tmp_path
    model_dir = workspace / "models" / "502000" / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.rknn"
    model_file.write_bytes(b"rknn")

    monkeypatch.setenv("WORKSPACE", str(workspace))

    try:
        node = generate_inference_node(
            {
                "_config_path": "/tmp/so101_single_arm.yaml",
                "models": {
                    "so101_act_rknn": {
                        "path": "./models/502000/pretrained_model",
                        "policy_type": "act",
                        "device": "rknn",
                    }
                },
                "control_modes": {
                    "model_inference": {
                        "inference": {
                            "enabled": True,
                            "model": "so101_act_rknn",
                        }
                    }
                },
            },
            "model_inference",
        )

        params = _node_parameters(node)

        assert str(model_dir.resolve()) in params["checkpoint"]
        assert "rknn" in str(params["device"])
    finally:
        model_file.unlink(missing_ok=True)


def test_generate_inference_node_uses_policy_path_only_for_rknn(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE", str(tmp_path))

    model_dir = tmp_path / "models" / "502000" / "pretrained_model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.rknn").write_bytes(b"rknn")

    node = generate_inference_node(
        {
            "_config_path": "/tmp/so101_single_arm.yaml",
            "models": {
                "so101_act_rknn": {
                    "path": "./models/502000/pretrained_model",
                    "policy_type": "act",
                    "device": "rknn",
                }
            },
            "control_modes": {
                "model_inference": {
                    "inference": {
                        "enabled": True,
                        "model": "so101_act_rknn",
                    }
                }
            },
        },
        "model_inference",
    )

    params = _node_parameters(node)

    assert str(model_dir) in params["checkpoint"]
    assert node.env is not None
    assert all(_text(key) != "RKNN_MODEL_PATH" for key, _value in node.env)


def test_attention_viz_request_uses_robot_config_only():
    enabled, mode, _viz_config = _attention_viz_request(
        {"attention_viz": {"enabled": False, "mode": "file"}}
    )

    assert enabled is False
    assert mode == "file"


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
