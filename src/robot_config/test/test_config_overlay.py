"""Tests for robot configuration inheritance and hand-path boundaries."""

from pathlib import Path

import pytest

from robot_config.launch_builders.teleop import validate_teleop_config
from robot_config.loader import _load_robot_section, load_robot_config_dict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "robots"


def test_so101_aero_profile_inherits_base_without_copying_unrelated_sections() -> None:
    # The Aero Hand overlay stays on the provider-less legacy path; its base is
    # the retained legacy single-arm configuration, not the migrated
    # runtime-provider configuration.
    base_path, base = _load_robot_section(CONFIG_DIR / "so101_single_arm_legacy.yaml")
    resolved_path, resolved = _load_robot_section(CONFIG_DIR / "so101_arm_aero_hand.yaml")

    assert base_path.name == "so101_single_arm_legacy.yaml"
    assert resolved_path.name == "so101_arm_aero_hand.yaml"
    assert resolved["name"] == "so101_arm_aero_hand"
    assert resolved["joints"]["arm"] == base["joints"]["arm"]
    assert resolved["joints"]["gripper"] == base["joints"]["gripper"]
    for key in ("control_modes", "moveit", "peripherals", "ros2_control", "embodied", "voice_asr", "voice_tts"):
        assert resolved[key] == base[key], f"unexpected drift in inherited robot.{key}"

    assert [device["name"] for device in resolved["teleoperation"]["devices"]][-1] == "aero_glove_right"
    assert resolved["teleoperation"]["active_devices"] == ["so101_leader", "aero_glove_right"]
    assert resolved["hand_sources"]["mhandpro"]["exclusive_resources"] == ["mhandpro_sdk"]
    assert resolved["auxiliary_actuators"]["aero_hand_right"]["command_topic"] == "/aero_hand_right/commands"


def test_overlay_append_requires_a_base_list(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("robot:\n  name: base\n  value: 1\n", encoding="utf-8")
    (tmp_path / "overlay.yaml").write_text(
        "robot:\n  name: overlay\n  base_config: base.yaml\n  value:\n    __append__: [2]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a list in the base configuration"):
        _load_robot_section(tmp_path / "overlay.yaml")


def test_overlay_append_rejects_ambiguous_mapping(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("robot:\n  name: base\n  values: [1]\n", encoding="utf-8")
    (tmp_path / "overlay.yaml").write_text(
        "robot:\n  name: overlay\n  base_config: base.yaml\n  values:\n    __append__: [2]\n    extra: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain only"):
        _load_robot_section(tmp_path / "overlay.yaml")


@pytest.mark.parametrize(
    ("base_value", "overlay_value"),
    [
        ("{enabled: true}", "[enabled]"),
        ("[base]", "{enabled: true}"),
        ("enabled", "{enabled: true}"),
    ],
)
def test_overlay_rejects_container_type_mismatch(tmp_path: Path, base_value: str, overlay_value: str) -> None:
    (tmp_path / "base.yaml").write_text(
        f"robot:\n  name: base\n  nested:\n    value: {base_value}\n",
        encoding="utf-8",
    )
    (tmp_path / "overlay.yaml").write_text(
        f"robot:\n  name: overlay\n  base_config: base.yaml\n  nested:\n    value: {overlay_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"overlay type mismatch at robot\.nested\.value"):
        _load_robot_section(tmp_path / "overlay.yaml")


def test_overlay_rejects_cross_directory_base_config(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    overlay_dir = tmp_path / "overlay"
    base_dir.mkdir()
    overlay_dir.mkdir()
    (base_dir / "base.yaml").write_text("robot:\n  name: base\n", encoding="utf-8")
    (overlay_dir / "overlay.yaml").write_text(
        "robot:\n  name: overlay\n  base_config: ../base/base.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must reference a sibling robot YAML"):
        _load_robot_section(overlay_dir / "overlay.yaml")


def test_legacy_mhandpro_glove_is_rejected_as_active_runtime_path() -> None:
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "legacy_glove",
            "devices": [{"name": "legacy_glove", "type": "mhandpro_glove"}],
        }
    )

    assert any("has been removed" in error for error in errors)


def test_overlay_cycle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("robot:\n  name: a\n  base_config: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("robot:\n  name: b\n  base_config: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="base_config cycle detected"):
        _load_robot_section(tmp_path / "a.yaml")


def test_normalized_loader_reports_base_to_overlay_source_chain(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    middle = tmp_path / "middle.yaml"
    overlay = tmp_path / "overlay.yaml"
    base.write_text("robot:\n  name: base\n  values: [1]\n", encoding="utf-8")
    middle.write_text(
        "robot:\n  name: middle\n  base_config: base.yaml\n  values:\n    __append__: [2]\n",
        encoding="utf-8",
    )
    overlay.write_text("robot:\n  name: overlay\n  base_config: middle.yaml\n", encoding="utf-8")

    resolved = load_robot_config_dict(overlay)

    assert resolved["values"] == [1, 2]
    assert resolved["_config_path"] == str(overlay.resolve())
    assert resolved["_config_sources"] == [str(base.resolve()), str(middle.resolve()), str(overlay.resolve())]
