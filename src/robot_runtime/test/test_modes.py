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


# ---------------------------------------------------------------------------
# D3.1: per-platform controller overrides
# ---------------------------------------------------------------------------


def _platform_profile() -> dict:
    # The LeKiwi-shaped declaration: on MuJoCo the base controller is never
    # released (a MuJoCo velocity actuator keeps its last target), so idle
    # carries it there and the base node is the gate.
    return {
        "modes": {
            "initial": "idle",
            "idle": {
                "controllers": [],
                "controllers_by_platform": {"mujoco": ["base_velocity_controller"]},
                "transitions": ["base_navigation"],
            },
            "base_navigation": {
                "controllers": ["base_velocity_controller"],
                "allows_base": True,
                "transitions": ["idle"],
            },
        }
    }


def test_platform_override_replaces_controllers_for_that_platform_only():
    config = ModeModelConfig.from_profile(_platform_profile(), "mujoco")
    assert config.modes["idle"].controllers == ("base_velocity_controller",)
    assert config.modes["base_navigation"].controllers == ("base_velocity_controller",)
    # every other selection keeps the plain controllers (gazebo, sdk, hardware)
    for selection in (None, "gazebo", "sdk"):
        other = ModeModelConfig.from_profile(_platform_profile(), selection)
        assert other.modes["idle"].controllers == ()
        assert other.modes["base_navigation"].controllers == ("base_velocity_controller",)
    # the declared table stays visible for introspection either way
    assert config.modes["idle"].controllers_by_platform == {"mujoco": ("base_velocity_controller",)}


def test_an_empty_platform_override_stays_empty():
    profile = _platform_profile()
    profile["modes"]["base_navigation"]["controllers_by_platform"] = {"mujoco": []}
    config = ModeModelConfig.from_profile(profile, "mujoco")
    assert config.modes["base_navigation"].controllers == ()
    assert ModeModelConfig.from_profile(profile, "gazebo").modes["base_navigation"].controllers == (
        "base_velocity_controller",
    )


def test_unknown_platform_key_is_rejected_even_when_not_selected():
    profile = _platform_profile()
    profile["modes"]["idle"]["controllers_by_platform"] = {"teleport": ["x"]}
    with pytest.raises(ValueError, match="controllers_by_platform key 'teleport' is not a physics platform"):
        ModeModelConfig.from_profile(profile)
    with pytest.raises(ValueError, match="controllers_by_platform key 'teleport' is not a physics platform"):
        ModeModelConfig.from_profile(profile, "mujoco")


def test_sdk_cannot_be_a_platform_override_key():
    profile = _platform_profile()
    profile["modes"]["idle"]["controllers_by_platform"] = {"sdk": ["x"]}
    with pytest.raises(ValueError, match="not a physics platform"):
        ModeModelConfig.from_profile(profile, "sdk")


def test_non_list_platform_override_is_rejected():
    profile = _platform_profile()
    profile["modes"]["idle"]["controllers_by_platform"] = {"mujoco": "base_velocity_controller"}
    with pytest.raises(ValueError, match="must be a list"):
        ModeModelConfig.from_profile(profile, "mujoco")


def test_mode_controller_table_validates_every_mode_up_front():
    profile = _platform_profile()
    profile["modes"]["base_navigation"]["controllers_by_platform"] = {"gazebo": 3}
    from robot_runtime.modes import mode_controller_table

    with pytest.raises(ValueError, match="must be a list"):
        mode_controller_table(profile, "mujoco")
    # the mujoco table itself resolves through the same single resolver
    table = mode_controller_table(_platform_profile(), "mujoco")
    assert table == {"idle": ("base_velocity_controller",), "base_navigation": ("base_velocity_controller",)}
    assert mode_controller_table(_platform_profile(), None)["idle"] == ()
