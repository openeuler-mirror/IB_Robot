from pathlib import Path

import pytest

from robot_config import config_path as config_path_module
from robot_config import resolve_robot_config_path

SOURCE_ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"


def test_resolve_robot_config_path_prefers_explicit_path_over_all_other_sources(tmp_path, monkeypatch):
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text("robot: {name: explicit}\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_CONFIG", "so101_single_arm")
    monkeypatch.setenv("ROBOT_NAME", "missing_robot")

    assert resolve_robot_config_path(config_name="missing_robot", config_path=explicit_path) == explicit_path.resolve()


def test_resolve_robot_config_path_uses_explicit_name_then_environment(monkeypatch):
    monkeypatch.setattr(config_path_module, "get_package_share_directory", lambda _package: "/missing/share")
    monkeypatch.setenv("ROBOT_CONFIG", "missing_robot")
    monkeypatch.setenv("ROBOT_NAME", "also_missing")

    assert (
        resolve_robot_config_path(config_name="so101_single_arm") == (SOURCE_ROBOTS / "so101_single_arm.yaml").resolve()
    )


def test_resolve_robot_config_path_checks_share_before_checkout(tmp_path, monkeypatch):
    share_config = tmp_path / "share" / "config" / "robots" / "so101_single_arm.yaml"
    share_config.parent.mkdir(parents=True)
    share_config.write_text("robot: {name: from_share}\n", encoding="utf-8")
    monkeypatch.setattr(config_path_module, "get_package_share_directory", lambda _package: str(tmp_path / "share"))

    assert resolve_robot_config_path(config_name="so101_single_arm") == share_config.resolve()


def test_resolve_robot_config_path_uses_robot_config_then_robot_name(monkeypatch):
    monkeypatch.setattr(config_path_module, "get_package_share_directory", lambda _package: "/missing/share")
    monkeypatch.setenv("ROBOT_CONFIG", "so101_single_arm")
    monkeypatch.setenv("ROBOT_NAME", "missing_robot")
    assert resolve_robot_config_path() == (SOURCE_ROBOTS / "so101_single_arm.yaml").resolve()

    monkeypatch.delenv("ROBOT_CONFIG")
    monkeypatch.setenv("ROBOT_NAME", "so101_single_arm")
    assert resolve_robot_config_path() == (SOURCE_ROBOTS / "so101_single_arm.yaml").resolve()


def test_resolve_robot_config_path_requires_explicit_selection(monkeypatch):
    # robot-agnostic-core: library callers obtain no implicit default robot;
    # only a launch entry point may document one.
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)

    with pytest.raises(ValueError, match="ROBOT_CONFIG"):
        resolve_robot_config_path()


def test_resolve_robot_config_path_reports_missing_explicit_path(tmp_path):
    missing_path = tmp_path / "not-here.yaml"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        resolve_robot_config_path(config_path=missing_path)
