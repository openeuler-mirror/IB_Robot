"""The X2 deployment configuration binds only to what the X2 runtime provides.

Pure configuration checks: no ROS, no vendor SDK, no robot. Capability and
control-mode declarations are read from the YAML because they are consumed as
configuration data (``RobotConfig`` keeps the runtime section as a mapping).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config, validate_config
from robot_runtime.capabilities import all_capabilities

CONFIG = Path(__file__).resolve().parents[1] / "config" / "robots" / "aimdk_x2.yaml"
# src/robot_config/test/ -> src/
PROFILE = Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra.yaml"


@pytest.fixture(scope="module")
def config():
    return load_robot_config(str(CONFIG))


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["robot"]


@pytest.fixture(scope="module")
def profile():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_configuration_is_valid(config):
    assert validate_config(config) == []


def test_it_selects_the_x2_runtime_provider(config, raw):
    assert config.runtime["provider"] == "aimdk_robot"
    assert raw["runtime"]["profile"] == "x2_ultra"


def test_required_capabilities_are_registered_names(raw):
    required = set(raw["capabilities"]["requires"])
    assert required
    assert required <= all_capabilities()


def test_required_capabilities_are_all_provided_by_the_profile(raw, profile):
    """Fail-closed binding: the deployment may not require what X2 cannot serve."""
    missing = sorted(set(raw["capabilities"]["requires"]) - set(profile["capabilities"]))
    assert missing == [], f"deployment requires capabilities the X2 runtime does not declare: {missing}"


def test_it_requires_the_vendor_capabilities_the_wrapper_exposes(raw):
    """The point of the wrapper: platform capabilities reach the application."""
    required = set(raw["capabilities"]["requires"])
    assert {"motion.named", "motion.posture", "interaction.tts", "power.state"} <= required


def test_it_does_not_require_a_public_model(raw):
    """No model is published for the X2 in this stage (design D5)."""
    assert raw["runtime"]["require_model"] is False


def test_it_does_not_require_trajectory_or_kinematics_capabilities(raw):
    required = set(raw["capabilities"]["requires"])
    assert not required & {
        "joint.trajectory",
        "motion.fk",
        "motion.ik",
        "motion.move_to_joint",
        "motion.move_to_pose",
    }


def test_control_modes_name_runtime_modes_the_profile_declares(raw, profile):
    declared = {name for name in profile["modes"] if name != "initial"}
    modes = raw["control_modes"]
    assert modes
    for name, mode in modes.items():
        runtime_mode = mode.get("runtime_mode", "")
        assert runtime_mode in declared, f"control mode {name} names undeclared runtime mode {runtime_mode!r}"


def test_audio_capture_is_not_launched_twice(raw):
    """X2 publishes the audio contract itself; the shared ALSA nodes stay off."""
    assert raw["audio_io"]["enabled"] is False
