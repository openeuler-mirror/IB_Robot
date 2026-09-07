from pathlib import Path

import pytest
import yaml

from task_dispatch.task_executor_node import _load_robot_yaml


def _write_robot(path: Path, robot: dict) -> None:
    path.write_text(yaml.safe_dump({"robot": robot}, sort_keys=False), encoding="utf-8")


def test_base_config_overlay_inherits_gripper_joint(tmp_path: Path) -> None:
    base = tmp_path / "base_robot.yaml"
    overlay = tmp_path / "overlay_robot.yaml"
    _write_robot(
        base,
        {
            "name": "base_robot",
            "joints": {"arm": ["1", "2"], "gripper": ["6"]},
            "control_modes": {"moveit_planning": {"controllers": ["arm_trajectory_controller"]}},
        },
    )
    _write_robot(
        overlay,
        {
            "base_config": "base_robot",
            "joints": {"arm": ["1", "2", "3"]},
        },
    )

    loaded = _load_robot_yaml(str(overlay))

    assert loaded["name"] == "base_robot"
    assert loaded["joints"]["gripper"] == ["6"]
    assert loaded["joints"]["arm"] == ["1", "2", "3"]
    assert loaded["control_modes"]["moveit_planning"]["controllers"] == ["arm_trajectory_controller"]


def test_base_config_must_reference_sibling(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay_robot.yaml"
    _write_robot(overlay, {"base_config": "../outside_robot", "name": "overlay_robot"})

    with pytest.raises(ValueError, match="sibling robot YAML"):
        _load_robot_yaml(str(overlay))


def test_base_config_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first_robot.yaml"
    second = tmp_path / "second_robot.yaml"
    _write_robot(first, {"base_config": "second_robot", "name": "first_robot"})
    _write_robot(second, {"base_config": "first_robot", "name": "second_robot"})

    with pytest.raises(ValueError, match="cycle detected"):
        _load_robot_yaml(str(first))
