"""Unit tests for launch readiness helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from launch.actions import RegisterEventHandler
from launch_ros.actions import Node

from robot_config.launch_builders.control import generate_controller_spawners
from robot_config.launch_builders.sim_backend import get_sim_backend
from robot_config.wait_for_controllers import missing_inactive_controllers

_LAUNCH_PATH = Path(__file__).resolve().parents[1] / "launch" / "robot.launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("robot_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None
assert _LAUNCH_SPEC.loader is not None
robot_launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(robot_launch)


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
    cmd_text = [
        item[0].text
        for item in spawners[0].cmd
        if item
        and hasattr(item[0], "text")
    ]
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
    actions, create_node = adapter.start_backend(
        {"name": "test_robot", "gazebo_world_name": "demo"}
    )

    assert isinstance(create_node, Node)
    assert any(isinstance(action, Node) for action in actions)
    assert any(isinstance(action, RegisterEventHandler) for action in actions)
