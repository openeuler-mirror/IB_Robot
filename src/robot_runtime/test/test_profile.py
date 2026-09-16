"""Unit tests for runtime profile validation (robot-runtime-packaging spec)."""

import pytest
import yaml

from robot_runtime.profile import ProfileError, load_profile, stop_bounds, validate_profile


def _minimal() -> dict:
    return {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "modes": {"initial": "idle", "idle": {"transitions": []}},
        "capabilities": {"joint.state": {"joint_count": 2, "rate_hz": 50.0}},
        "joints": ["a", "b"],
    }


def test_minimal_profile_validates_and_fills_defaults():
    data = _minimal()
    validate_profile(data, "unit")
    assert data["simulated"] is False
    assert data["controller_manager"] == "controller_manager"
    assert data["command_channels"] == [] and data["trajectory_actions"] == []
    assert data["stop_default_policy"] == "HOLD"


@pytest.mark.parametrize("missing", ["runtime", "modes", "capabilities", "joints"])
def test_missing_required_key_names_key_and_source(missing):
    data = _minimal()
    del data[missing]
    with pytest.raises(ProfileError, match=rf"missing required key '{missing}'.*my_profile"):
        validate_profile(data, "my_profile.yaml")


def test_missing_runtime_name_names_nested_key():
    data = _minimal()
    data["runtime"]["name"] = ""
    with pytest.raises(ProfileError, match="'runtime.name'"):
        validate_profile(data)


def test_unknown_capability_rejected():
    data = _minimal()
    data["capabilities"]["joint.teleport"] = {}
    with pytest.raises(ProfileError, match=r"unknown capabilities \['joint.teleport'\]"):
        validate_profile(data)


def test_capability_list_form_normalized_to_mapping():
    data = _minimal()
    data["capabilities"] = ["joint.state", "motion.fk"]
    validate_profile(data)
    assert data["capabilities"] == {"joint.state": {}, "motion.fk": {}}


def test_stop_bounds_required_positive_when_declared():
    data = _minimal()
    data["capabilities"]["runtime.stop"] = {"cancel_bound_s": 0.5, "idle_bound_s": 0}
    with pytest.raises(ProfileError, match="runtime.stop.idle_bound_s"):
        validate_profile(data)
    data["capabilities"]["runtime.stop"] = {"cancel_bound_s": 0.5, "idle_bound_s": 1.0, "torque_off_bound_s": 2.0}
    validate_profile(data)
    assert stop_bounds(data) == {"cancel_bound_s": 0.5, "idle_bound_s": 1.0, "torque_off_bound_s": 2.0}


def test_idle_mode_is_mandatory():
    data = _minimal()
    data["modes"] = {"initial": "stream", "stream": {}}
    with pytest.raises(ProfileError, match="must declare 'idle'"):
        validate_profile(data)


def test_command_channel_entries_need_channel_and_topic():
    data = _minimal()
    data["command_channels"] = [{"topic": "/x"}]
    with pytest.raises(ProfileError, match="need 'channel' and 'topic'"):
        validate_profile(data)


def test_load_profile_from_file_and_missing_file(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(_minimal()), encoding="utf-8")
    assert load_profile(path)["runtime"]["name"] == "test_robot"
    with pytest.raises(ProfileError, match="not found"):
        load_profile(tmp_path / "nope.yaml")


def test_mock_default_profile_is_valid():
    from robot_runtime.mock_runtime_node import default_profile

    for base in (True, False):
        data = default_profile(base=base)
        validate_profile(data, "mock")
        assert ("base.cmd_vel" in data["capabilities"]) is base
