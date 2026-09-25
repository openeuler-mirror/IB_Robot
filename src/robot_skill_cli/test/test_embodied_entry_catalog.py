"""Catalog-compilation contract between robot_config YAML and the compiled snapshot.

These tests read robot_config's shipped robot YAML and assert what
``robot_skill_cli.catalog.compile_local_snapshot`` produces from it. They live
here, not in robot_config, because robot_skill_cli is the consumer: robot_config
cannot declare a test dependency on it without creating a dependency cycle
(robot_skill_cli already depends on robot_config), and colcon refuses to order a
workspace that contains one.

Config paths are resolved relative to the workspace root rather than to this
package, because the inputs are owned by robot_config and skill_catalog.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict
from robot_skill_cli.catalog import compile_local_snapshot

GRIPPER_TRAJECTORY_DURATION_SEC = 1.0

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_ROBOT_CONFIG_ROBOTS = _WORKSPACE_ROOT / "src" / "robot_config" / "config" / "robots"
_SKILL_CATALOG_CONFIG = _WORKSPACE_ROOT / "src" / "skill_catalog" / "config"


def _robot_config_path(config_name: str) -> Path:
    return _ROBOT_CONFIG_ROBOTS / f"{config_name}.yaml"


def _snapshot(config_path: Path):
    return compile_local_snapshot(load_robot_config_dict(config_path), config_path)


@pytest.mark.parametrize(
    "config_name",
    ["so101_single_arm_legacy"],
)
def test_compiled_profile_includes_dance_basic(config_name):
    config_path = _robot_config_path(config_name)

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    skill_templates = _snapshot(config_path).templates

    assert "dance_basic" in skill_templates
    primitive_sequence = skill_templates["dance_basic"]["primitive_sequence"]
    assert primitive_sequence
    trajectory_step = next(
        step for step in primitive_sequence if step["primitive_name"] == "move_through_joint_positions"
    )
    assert trajectory_step["joint_waypoints"]


def test_compiled_skills_match_profile_enabled_set():
    config_path = _robot_config_path("so101_single_arm_legacy")
    profile_path = _SKILL_CATALOG_CONFIG / "profiles" / "so101_single_arm.yaml"
    expected = {entry["name"] for entry in yaml.safe_load(profile_path.read_text(encoding="utf-8"))["enabled_skills"]}

    assert set(_snapshot(config_path).enabled_skill_names) == expected


def test_embodied_config_keeps_only_supported_direct_skills():
    config_path = _robot_config_path("so101_single_arm_legacy")
    skill_templates = _snapshot(config_path).templates

    assert "dance_basic" in skill_templates
    assert "pick_named_target" not in skill_templates
    assert "place_named_pose" not in skill_templates
    assert "observe_target_area" not in skill_templates


@pytest.mark.parametrize(
    ("skill_name", "pose_name"),
    [
        ("recover_safe_pose", "home"),
        ("inspect_scene", "observe_table"),
        ("recover_zero_pose", "zero"),
    ],
)
def test_embodied_named_pose_skills_map_to_configured_poses(skill_name, pose_name):
    config_path = _robot_config_path("so101_single_arm_legacy")
    config = load_robot_config_dict(config_path)
    skill_templates = _snapshot(config_path).templates

    assert pose_name in config["embodied"]["named_poses"]
    step = skill_templates[skill_name]["primitive_sequence"][0]
    assert dict(step) == {"primitive_name": "move_to_named_pose", "pose_name": pose_name}


@pytest.mark.parametrize(
    "skill_name",
    ["wave_hello", "nod_yes", "shake_no", "act_cute", "happy_spin_upright"],
)
def test_social_gesture_duration_estimate_covers_configured_motion(skill_name):
    config_path = _robot_config_path("so101_single_arm_legacy")
    skill = _snapshot(config_path).templates[skill_name]
    manifest_path = _SKILL_CATALOG_CONFIG / "skills" / skill_name / "manifest.yaml"
    description = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["description"]

    configured_duration = 0.0
    if skill.get("initial_gripper_state") in {"open", "closed"}:
        configured_duration += GRIPPER_TRAJECTORY_DURATION_SEC
    for step in skill["primitive_sequence"]:
        primitive_name = step["primitive_name"]
        if primitive_name == "move_to_joint_positions":
            configured_duration += float(step.get("duration_sec", 0.4))
        elif primitive_name == "move_through_joint_positions":
            configured_duration += len(step["joint_waypoints"]) * float(step["waypoint_duration_sec"])
        elif primitive_name in {"open_gripper", "close_gripper"}:
            configured_duration += GRIPPER_TRAJECTORY_DURATION_SEC

    assert float(description["duration_sec_estimate"]) >= configured_duration
