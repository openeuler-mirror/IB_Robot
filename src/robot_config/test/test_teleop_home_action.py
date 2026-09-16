"""Launch contract tests for transactional teleop Home."""

from pathlib import Path

import pytest

from robot_config.launch_builders import teleop


def _robot_config() -> dict:
    placo_config = (
        Path(__file__).resolve().parents[2] / "robots" / "so101" / "so101_motion" / "config" / "so101_placo_servo.yaml"
    )
    return {
        "joints": {"arm": ["1", "2"], "gripper": ["6"]},
        "ros2_control": {"reset_positions": {"1": 0.1, "2": -0.2}},
        "moveit": {
            "base_link": "base",
            "ee_link": "gripper",
            "so101_placo_servo_config_path": str(placo_config),
        },
        "teleoperation": {
            "enabled": True,
            "active_device": "vr",
            "cartesian": {"solver": "placo_servo"},
            "safety": {"estop_topic": "/safety/estop"},
            "devices": [],
        },
    }


class _FakeNode:
    def __init__(self, **kwargs):
        self.parameters = kwargs.get("parameters", [])


def test_placo_home_uses_ordered_reset_positions_and_estop(monkeypatch):
    monkeypatch.setattr(teleop, "Node", _FakeNode)
    config = _robot_config()
    device = {
        "name": "left_vr",
        "type": "vr_teleop",
        "target": {"arm_joint_names": ["2", "1"]},
    }

    node = teleop._create_so101_placo_servo_node(
        config,
        device,
        {},
        home_action="/left_arm/return_home",
    )

    params = node.parameters[0]
    assert params["arm_joint_names"] == ["2", "1"]
    assert params["home_joint_positions"] == [-0.2, 0.1]
    assert params["home_action"] == "/left_arm/return_home"
    assert params["estop_topic"] == "/safety/estop"


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf"), "0.0"])
def test_placo_home_rejects_invalid_reset_position(monkeypatch, bad_value):
    monkeypatch.setattr(teleop, "Node", _FakeNode)
    config = _robot_config()
    config["ros2_control"]["reset_positions"]["2"] = bad_value

    with pytest.raises(teleop.ConfigError, match="finite number"):
        teleop._create_so101_placo_servo_node(config, {"name": "vr", "type": "vr_teleop"}, {})


def test_placo_home_rejects_missing_reset_position(monkeypatch):
    monkeypatch.setattr(teleop, "Node", _FakeNode)
    config = _robot_config()
    del config["ros2_control"]["reset_positions"]["2"]

    with pytest.raises(teleop.ConfigError, match="missing arm joint"):
        teleop._create_so101_placo_servo_node(config, {"name": "vr", "type": "vr_teleop"}, {})


def test_vr_and_placo_share_home_action_and_estop(monkeypatch):
    config = _robot_config()
    device = {
        "name": "vr",
        "type": "vr_teleop",
        "control_frequency": 50.0,
        "vr_config": {
            "output_profile": "so101",
            "so101_input_mode": "pose",
            "so101_home_action": "/left_arm/return_home",
        },
    }
    placo_kwargs = {}

    monkeypatch.setattr(teleop, "Node", _FakeNode)
    monkeypatch.setattr(teleop, "prepare_lerobot_env", lambda: {})

    def _fake_placo(*_args, **kwargs):
        placo_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(teleop, "_create_so101_placo_servo_node", _fake_placo)

    nodes = teleop._generate_vr_teleop_nodes(config, device, {})

    vr_params = nodes[0].parameters[0]
    assert vr_params["so101_home_action"] == "/left_arm/return_home"
    assert vr_params["estop_topic"] == "/safety/estop"
    assert placo_kwargs["home_action"] == "/left_arm/return_home"
