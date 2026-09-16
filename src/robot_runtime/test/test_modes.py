"""Unit tests for the runtime mode model (transition validation, rejection counting)."""

import pytest

from robot_runtime.modes import ModeModel, ModeModelConfig, ModeSpec


def _config() -> ModeModelConfig:
    return ModeModelConfig.from_profile(
        {
            "modes": {
                "initial": "idle",
                "idle": {"controllers": [], "transitions": ["stream", "trajectory"]},
                "stream": {"controllers": ["arm_position_controller"], "allows_stream": True, "transitions": ["idle"]},
                "trajectory": {
                    "controllers": ["arm_trajectory_controller"],
                    "allows_trajectory": True,
                    "transitions": ["idle", "stream"],
                },
            }
        }
    )


def test_from_profile_builds_specs_and_transitions():
    config = _config()
    assert config.initial_mode == "idle"
    assert config.modes["stream"] == ModeSpec("stream", ("arm_position_controller",), False, True, False)
    assert config.transitions["trajectory"] == {"idle", "stream"}


def test_from_profile_rejects_undeclared_transition_target():
    with pytest.raises(ValueError, match="undeclared modes"):
        ModeModelConfig.from_profile({"modes": {"initial": "idle", "idle": {"transitions": ["ghost"]}}})


def test_from_profile_rejects_unknown_initial():
    with pytest.raises(ValueError, match="not a declared mode"):
        ModeModelConfig.from_profile({"modes": {"initial": "nope", "idle": {}}})


def test_valid_transition_and_commit():
    model = ModeModel(_config())
    decision = model.can_switch("stream")
    assert decision.allowed and decision.mode == "stream"
    model.commit("stream")
    assert model.mode == "stream"
    assert model.spec().allows_stream and not model.spec().allows_trajectory


def test_invalid_transition_lists_alternatives():
    model = ModeModel(_config())
    model.commit("stream")
    decision = model.can_switch("trajectory")
    assert not decision.allowed
    assert "valid: ['idle']" in decision.reason
    assert model.valid_transitions() == {"idle"}
    assert model.mode == "stream"


def test_undeclared_mode_rejected_by_name():
    model = ModeModel(_config())
    decision = model.can_switch("teleop")
    assert not decision.allowed and "'teleop' is not declared" in decision.reason


def test_same_mode_is_a_no_op_success():
    model = ModeModel(_config())
    assert model.can_switch("idle").allowed


def test_rejected_commands_counted_per_channel():
    model = ModeModel(_config())
    assert model.note_rejected("arm_stream") == 1
    assert model.note_rejected("arm_stream") == 2
    assert model.note_rejected("gripper_stream") == 1
    assert model.rejections() == {"arm_stream": 2, "gripper_stream": 1}
