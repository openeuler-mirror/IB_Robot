"""Unit tests of the shared simulator-environment rule (D10/D15).

Only the pure decision logic is tested here: the environment probes
themselves need a real simulator installation (the live gates exercise them
with ``IBROBOT_REQUIRE_SIM_PLATFORMS`` set).
"""

from __future__ import annotations

import subprocess

import pytest

from robot_runtime.testing import sim_environment
from robot_runtime.testing.sim_environment import (
    REQUIRED_SIM_PLATFORMS_ENV,
    environment_missing,
    gazebo_available,
    required_sim_platforms,
    skip_or_fail,
)


def test_the_requirement_list_parses_names_case_and_spaces(monkeypatch):
    monkeypatch.setenv(REQUIRED_SIM_PLATFORMS_ENV, " MUJOCO , Gazebo ,")
    assert required_sim_platforms() == {"mujoco", "gazebo"}


def test_an_unset_or_empty_requirement_list_is_no_requirement(monkeypatch):
    monkeypatch.delenv(REQUIRED_SIM_PLATFORMS_ENV, raising=False)
    assert required_sim_platforms() == set()
    monkeypatch.setenv(REQUIRED_SIM_PLATFORMS_ENV, " , ")
    assert required_sim_platforms() == set()


def test_an_unknown_platform_has_no_environment_check():
    reason = environment_missing("synthetic")
    assert reason is not None and "no environment check for platform 'synthetic'" in reason


def test_a_required_platform_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setenv(REQUIRED_SIM_PLATFORMS_ENV, "mujoco")
    with pytest.raises(pytest.fail.Exception, match="navigation simulation: platform 'mujoco' was required"):
        skip_or_fail("mujoco", "the simulator is unavailable", context="navigation simulation")


def test_an_unrequired_platform_stays_a_skip(monkeypatch):
    monkeypatch.delenv(REQUIRED_SIM_PLATFORMS_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="the simulator is unavailable"):
        skip_or_fail("gazebo", "the simulator is unavailable", context="navigation simulation")


def test_gazebo_accepts_either_stack(monkeypatch):
    """The probe covers Fortress (``ign``) and newer gz installs alike."""

    class _Result:
        returncode = 0

    def _run_ign(cmd, **_kwargs):
        if cmd[0] == "ign":
            return _Result()
        raise FileNotFoundError(cmd[0])

    def _run_gz(cmd, **_kwargs):
        if cmd[0] == "gz":
            return _Result()
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(sim_environment.subprocess, "run", _run_ign)
    assert gazebo_available() is True
    monkeypatch.setattr(sim_environment.subprocess, "run", _run_gz)
    assert gazebo_available() is True


def test_gazebo_unavailable_when_no_clique_starts(monkeypatch):
    def _missing(cmd, **_kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(sim_environment.subprocess, "run", _missing)
    assert gazebo_available() is False
    monkeypatch.setattr(
        sim_environment.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, returncode=1),
    )
    assert gazebo_available() is False
