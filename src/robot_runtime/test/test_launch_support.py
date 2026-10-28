"""Launch-support unit tests: simulation selection (runtime-simulation-platform D1)."""

import logging

import pytest

from robot_runtime.launch_support import RuntimeProfileError, resolve_sim_selection


def _profile(**extra):
    profile = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "use_sim": False,
        "sim_platforms": ["sdk"],
        "sim_platform": "sdk",
    }
    profile.update(extra)
    return profile


def test_no_selection_uses_profile_default():
    assert resolve_sim_selection(_profile(), "", "", "") == (False, None)
    assert resolve_sim_selection(_profile(use_sim=True), "", "", "") == (True, "sdk")


def test_platform_alone_implies_simulation():
    assert resolve_sim_selection(_profile(sim_platforms=["sdk", "gazebo"]), "", "gazebo", "") == (True, "gazebo")
    # ... even when the profile itself defaults to physical execution
    assert resolve_sim_selection(_profile(), "", "sdk", "") == (True, "sdk")


def test_use_sim_alone_uses_profile_default_platform():
    assert resolve_sim_selection(_profile(), "true", "", "") == (True, "sdk")
    assert resolve_sim_selection(_profile(sim_platforms=["sdk", "gazebo"], sim_platform="gazebo"), "true", "", "") == (
        True,
        "gazebo",
    )


def test_simulated_alias_maps_to_use_sim_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="robot_runtime.launch_support"):
        assert resolve_sim_selection(_profile(), "", "", "true") == (True, "sdk")
        assert resolve_sim_selection(_profile(), "", "", "false") == (False, None)
        assert resolve_sim_selection(_profile(use_sim=True), "", "", "1") == (True, "sdk")
        assert resolve_sim_selection(_profile(), "", "", "no") == (False, None)
    assert any(
        "simulated:=true is deprecated" in record.message and "use_sim" in record.message for record in caplog.records
    )
    assert any(
        "simulated:=false is deprecated" in record.message and "use_sim" in record.message for record in caplog.records
    )


def test_agreeing_use_sim_and_simulated_selection_is_accepted():
    assert resolve_sim_selection(_profile(), "true", "", "true") == (True, "sdk")
    assert resolve_sim_selection(_profile(), "false", "", "false") == (False, None)


def test_conflicting_use_sim_and_simulated_selection_fails_naming_both():
    with pytest.raises(RuntimeProfileError, match=r"use_sim:=True.*simulated:=false"):
        resolve_sim_selection(_profile(), "true", "", "false")
    with pytest.raises(RuntimeProfileError, match="pass at most one"):
        resolve_sim_selection(_profile(), "false", "", "true")


def test_physical_execution_with_platform_fails_naming_both():
    with pytest.raises(RuntimeProfileError, match=r"use_sim:=false.*sim_platform:='gazebo'"):
        resolve_sim_selection(_profile(sim_platforms=["sdk", "gazebo"]), "false", "gazebo", "")


def test_unknown_platform_value_fails():
    with pytest.raises(RuntimeProfileError, match="unknown sim_platform 'teleport'"):
        resolve_sim_selection(_profile(), "true", "teleport", "")


def test_unparseable_boolean_values_fail():
    with pytest.raises(RuntimeProfileError, match="cannot parse use_sim:='banana'"):
        resolve_sim_selection(_profile(), "banana", "", "")
    with pytest.raises(RuntimeProfileError, match="cannot parse simulated:='banana'"):
        resolve_sim_selection(_profile(), "", "", "banana")


def test_undeclared_platform_fails_naming_platform_set_and_profile():
    profile = _profile()
    with pytest.raises(
        RuntimeProfileError,
        match=r"sim_platform 'gazebo' is not supported.*supported: \['sdk'\].*profile: /tmp/p.yaml",
    ):
        resolve_sim_selection(profile, "true", "gazebo", "", source="/tmp/p.yaml")


def test_declared_gazebo_platform_resolves():
    profile = _profile(sim_platforms=["sdk", "gazebo"])
    assert resolve_sim_selection(profile, "true", "gazebo", "") == (True, "gazebo")
    assert resolve_sim_selection(profile, "", "gazebo", "") == (True, "gazebo")


def test_declared_mujoco_platform_resolves_and_undeclared_fails():
    profile = _profile(sim_platforms=["sdk", "gazebo", "mujoco"])
    assert resolve_sim_selection(profile, "true", "mujoco", "") == (True, "mujoco")
    assert resolve_sim_selection(profile, "", "mujoco", "") == (True, "mujoco")
    with pytest.raises(
        RuntimeProfileError,
        match=r"sim_platform 'mujoco' is not supported.*supported: \['sdk'\].*profile: /tmp/p.yaml",
    ):
        resolve_sim_selection(_profile(), "true", "mujoco", "", source="/tmp/p.yaml")


def test_profile_without_selection_keys_keeps_legacy_behaviour():
    profile = {"runtime": {"name": "test_robot", "version": "0.1.0"}, "simulated": True}
    assert resolve_sim_selection(profile, "", "", "") == (True, "sdk")
    assert resolve_sim_selection(profile, "", "", "false") == (False, None)


# ---------------------------------------------------------------------------
# D3.1: the spawner list follows the mode table's per-platform overrides
# ---------------------------------------------------------------------------


def _mode_profile() -> dict:
    return {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "ros2_control": {"broadcasters": ["joint_state_broadcaster"]},
        "modes": {
            "initial": "idle",
            "idle": {
                "controllers": [],
                "controllers_by_platform": {"mujoco": ["base_velocity_controller"]},
                "transitions": ["base_navigation"],
            },
            "base_navigation": {"controllers": ["base_velocity_controller"], "transitions": ["idle"]},
        },
    }


def test_profile_controllers_applies_the_platform_override():
    from robot_runtime.launch_support import profile_controllers

    broadcasters, active, inactive = profile_controllers(_mode_profile(), "mujoco")
    assert broadcasters == ["joint_state_broadcaster"]
    # idle (initial) carries the base controller on mujoco: spawned ACTIVE
    assert active == ["base_velocity_controller"]
    # base_navigation resolves to the same controller: no duplicate, nothing inactive
    assert inactive == []


def test_profile_controllers_without_the_platform_keep_the_plain_sets():
    from robot_runtime.launch_support import profile_controllers

    for selection in (None, "gazebo", "sdk"):
        _broadcasters, active, inactive = profile_controllers(_mode_profile(), selection)
        assert active == [] and inactive == ["base_velocity_controller"], selection


def test_profile_controllers_rejects_a_bad_platform_table():
    from robot_runtime.launch_support import profile_controllers

    profile = _mode_profile()
    profile["modes"]["idle"]["controllers_by_platform"] = {"teleport": ["x"]}
    with pytest.raises(ValueError, match="not a physics platform"):
        profile_controllers(profile, "mujoco")
