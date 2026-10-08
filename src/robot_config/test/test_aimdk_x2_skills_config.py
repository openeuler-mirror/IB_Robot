"""The X2 skills deployment adds only the delegated skill stack to the X2 base config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.interface_binding import InterfaceBindingError, bind_robot_interfaces, required_interface_ids
from robot_config.loader import (
    load_robot_config_dict,
    validate_robot_config_dict,
    validate_runtime_named_motion_config,
)

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"
CONFIG = ROBOTS / "aimdk_x2_skills.yaml"
# src/robot_config/test/ -> src/
PROFILE = Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra.yaml"


@pytest.fixture(scope="module")
def config():
    # The named-motion action is a live logical binding, resolved at launch.
    return load_robot_config_dict(CONFIG, defer_interface_binding=True)


def test_configuration_is_valid(config):
    validate_robot_config_dict(config, deferred_interfaces=True)


def test_it_extends_the_x2_runtime_without_an_arm_surface(config):
    assert config["runtime"]["provider"] == "aimdk_robot"
    assert config["embodied"]["enabled"] is True
    assert config["embodied"]["arm_surface"] is False
    assert config["embodied"]["runtime_named_motion"] == {
        "enabled": True,
        "interface": "motion.named",
        "kind": "action",
        "type": "ibrobot_msgs/action/ExecuteNamedMotion",
        "requires": {"capability": "motion.named"},
    }
    assert config["skill_required_control_mode"] == "named_motion"


def test_named_motion_mode_selects_the_runtime_idle_mode_without_a_dispatcher(config):
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    mode = config["control_modes"]["named_motion"]

    assert mode["runtime_mode"] == "idle"
    assert "idle" in profile["modes"]
    assert profile["vendor"]["named_motion"]["allowed_modes"] == ["idle"]
    assert mode["executor"]["enabled"] is False


@pytest.mark.parametrize("name", ["aimdk_x2.yaml", "aimdk_x2_interaction_demo.yaml"])
def test_base_and_demo_deployments_keep_the_skill_stack_disabled(name):
    config = load_robot_config_dict(ROBOTS / name, defer_interface_binding=True)

    assert config.get("embodied", {}).get("enabled", False) is False


DECLARATION = {
    "enabled": True,
    "interface": "motion.named",
    "kind": "action",
    "type": "ibrobot_msgs/action/ExecuteNamedMotion",
}


def _named_motion_robot(**overrides):
    robot = {
        "runtime": {"provider": "aimdk_robot"},
        "capabilities": {"requires": ["motion.named"]},
        "embodied": {"runtime_named_motion": dict(DECLARATION)},
    }
    robot.update(overrides)
    return robot


def test_runtime_named_motion_switch_accepts_a_runtime_advertising_named_motions():
    assert validate_runtime_named_motion_config(_named_motion_robot()) == []
    posture_only = _named_motion_robot(capabilities={"requires": ["motion.posture"]})
    assert validate_runtime_named_motion_config(posture_only) == []
    disabled = _named_motion_robot(embodied={"runtime_named_motion": {"enabled": False}})
    assert validate_runtime_named_motion_config(disabled) == []


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"runtime": {}}, "embodied.runtime_named_motion requires a runtime.provider"),
        (
            {"capabilities": {"requires": ["joint.state"]}},
            "embodied.runtime_named_motion requires capabilities motion.named or motion.posture",
        ),
        (
            {"embodied": {"runtime_named_motion": {**DECLARATION, "enabled": "yes"}}},
            "embodied.runtime_named_motion.enabled",
        ),
        ({"embodied": {"runtime_named_motion": True}}, "embodied.runtime_named_motion.enabled"),
        ({"embodied": {"runtime_named_motion": {"enabled": True}}}, "embodied.runtime_named_motion.interface"),
        (
            {"embodied": {"runtime_named_motion": {**DECLARATION, "kind": "service"}}},
            "embodied.runtime_named_motion.kind",
        ),
        (
            {"embodied": {"runtime_named_motion": {**DECLARATION, "type": "ibrobot_msgs/srv/SpeakText"}}},
            "embodied.runtime_named_motion.type",
        ),
        (
            {"embodied": {"runtime_named_motion": {**DECLARATION, "target": "left"}}},
            "embodied.runtime_named_motion has unknown",
        ),
    ],
)
def test_runtime_named_motion_switch_rejects_inconsistent_configuration(overrides, error):
    errors = validate_runtime_named_motion_config(_named_motion_robot(**overrides))

    assert any(item.startswith(error) for item in errors)


def _description(interfaces):
    from robot_runtime.interface_description import description_digest

    description = {
        "schema_version": 1,
        "robot": {"id": "aimdk_robot", "type": "x2_ultra", "runtime_name": "aimdk_robot", "runtime_version": "0"},
        "execution": "simulated",
        "interfaces": interfaces,
        "states": {},
    }
    description["digest"] = description_digest(description)
    return description


NAMED_MOTION_INTERFACE = {
    "capability": "motion.named",
    "kind": "action",
    "direction": "serve",
    "endpoint": "/motion/execute_named",
    "message_type": "ibrobot_msgs/action/ExecuteNamedMotion",
}


def test_enabled_named_motion_is_a_live_logical_binding(config):
    assert "motion.named" in required_interface_ids(config)
    with pytest.raises(InterfaceBindingError, match="description_required"):
        load_robot_config_dict(CONFIG)


def test_binding_resolves_the_action_and_carries_the_live_description(config):
    description = _description({"motion.named": dict(NAMED_MOTION_INTERFACE)})

    bound = bind_robot_interfaces(config, description, require_ready=True)

    declaration = bound["embodied"]["runtime_named_motion"]
    assert declaration["endpoint"] == "/motion/execute_named"
    assert bound["runtime"]["interface_description"]["interfaces"]["motion.named"]["endpoint"] == (
        "/motion/execute_named"
    )
    validate_robot_config_dict(bound)


@pytest.mark.parametrize(
    "interfaces",
    [{}, {"motion.named": {**NAMED_MOTION_INTERFACE, "message_type": "ibrobot_msgs/srv/SpeakText", "kind": "service"}}],
)
def test_binding_fails_closed_when_the_runtime_lacks_the_action(config, interfaces):
    with pytest.raises(InterfaceBindingError):
        bind_robot_interfaces(config, _description(interfaces), require_ready=True)
