"""Unit tests for the teleoperation-only launch entry."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext
from launch_ros.actions import Node

_LAUNCH_PATH = Path(__file__).resolve().parents[1] / "launch" / "teleop_operator.launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("teleop_operator_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None
assert _LAUNCH_SPEC.loader is not None
teleop_operator_launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(teleop_operator_launch)

_PROFILE_DIR = Path(__file__).resolve().parents[1] / "config" / "robots"


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

    return {_text(key): decode_parameter(value) for key, value in node._Node__parameters[0].items()}


def _calibrated_home(tmp_path, *names):
    """Create the calibration files the operator host must hold locally."""
    calibrate = tmp_path / ".calibrate"
    calibrate.mkdir(parents=True, exist_ok=True)
    for name in names:
        (calibrate / f"{name}.json").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _context(config_path, robot_config="lekiwi_rtp_distributed"):
    context = LaunchContext()
    context.launch_configurations["robot_config"] = robot_config
    context.launch_configurations["config_path"] = str(config_path)
    return context


def _profile_copy(tmp_path, name="lekiwi_rtp_distributed"):
    """Copy a shipped profile so tests can tweak it without touching the repo."""
    profile = yaml.safe_load((_PROFILE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    target = tmp_path / f"{name}.yaml"
    target.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return profile, target


def _write(profile, target):
    target.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_generate_launch_description_declares_config_arguments():
    description = teleop_operator_launch.generate_launch_description()
    declared = {
        entity.name for entity in description.entities if hasattr(entity, "name") and isinstance(entity.name, str)
    }

    assert "robot_config" in declared
    assert "config_path" in declared


def test_launch_setup_starts_teleop_without_ros2_control(tmp_path, monkeypatch):
    """The operator host drives the arm; the follower stays on the robot host."""
    monkeypatch.setenv("HOME", str(_calibrated_home(tmp_path, "so101_leader_calibrate", "so101_follower_calibrate")))
    _, config_path = _profile_copy(tmp_path)

    actions = teleop_operator_launch.launch_setup(_context(config_path))

    packages = [action.node_package for action in actions if isinstance(action, Node)]
    assert "robot_teleop" in packages
    assert "controller_manager" not in packages
    assert "lekiwi_hardware" not in packages


def test_launch_setup_ignores_the_robot_host_enabled_flag(tmp_path, monkeypatch):
    """`teleoperation.enabled` answers a different question than this entry."""
    monkeypatch.setenv("HOME", str(_calibrated_home(tmp_path, "so101_leader_calibrate", "so101_follower_calibrate")))
    profile, config_path = _profile_copy(tmp_path)
    assert profile["robot"]["teleoperation"]["enabled"] is False, "profile is expected to ship disabled"

    actions = teleop_operator_launch.launch_setup(_context(config_path))

    assert [action for action in actions if isinstance(action, Node)]


def test_launch_setup_resolves_the_follower_calibration(tmp_path, monkeypatch):
    """End-to-end through the entry: the derived path reaches the device."""
    home = _calibrated_home(tmp_path, "so101_leader_calibrate", "so101_follower_calibrate")
    monkeypatch.setenv("HOME", str(home))
    _, config_path = _profile_copy(tmp_path)

    actions = teleop_operator_launch.launch_setup(_context(config_path))

    leader = next(
        action for action in actions if isinstance(action, Node) and "device_config" in _node_parameters(action)
    )
    device_config = json.loads(_node_parameters(leader)["device_config"])
    assert device_config["follower_calib_file"] == str(home / ".calibrate" / "so101_follower_calibrate.json")


def test_launch_setup_rejects_a_profile_without_teleop_devices(tmp_path):
    """A profile with nothing to drive must fail loudly, not launch nothing."""
    profile, config_path = _profile_copy(tmp_path)
    profile["robot"]["teleoperation"]["devices"] = []
    profile["robot"]["teleoperation"].pop("active_devices", None)
    _write(profile, config_path)

    with pytest.raises(RuntimeError, match="teleoperation"):
        teleop_operator_launch.launch_setup(_context(config_path))


def test_launch_setup_accepts_any_profile_not_just_lekiwi(tmp_path, monkeypatch):
    """The entry is generic; the profile is an argument, not a hard-coded name."""
    monkeypatch.setenv("HOME", str(_calibrated_home(tmp_path, "so101_leader_calibrate", "so101_follower_calibrate")))
    _, config_path = _profile_copy(tmp_path, name="so101_rtp_distributed")

    actions = teleop_operator_launch.launch_setup(_context(config_path, robot_config="so101_rtp_distributed"))

    assert [action for action in actions if isinstance(action, Node)]
