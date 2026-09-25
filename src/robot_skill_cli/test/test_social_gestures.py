from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from embodied_common.skill_templates import get_skill_templates
from embodied_common.trajectory_templates import _pose_within_workspace, expand_trajectory_template
from robot_config.loader import load_robot_config_dict
from robot_skill_cli.catalog import compile_local_snapshot

SOCIAL_GESTURES = (
    "wave_hello",
    "nod_yes",
    "shake_no",
    "celebrate",
    "greet_observe_raise",
    "act_cute",
    "happy_spin_upright",
)
REQUIRED_WORKSPACE_POINTS = {"joint3", "upper_arm_mid", "forearm_mid", "wrist_mid", "ee"}
TRAJECTORY_GESTURES = (
    "wave_hello",
    "nod_yes",
    "shake_no",
    "act_cute",
    "happy_spin_upright",
)
EXPECTED_TEMPLATE_TYPES = {"single_joint_wave_v1", "wave_dance_v1"}
GRIPPER_ONLY_PRIMITIVES = {"open_gripper", "close_gripper"}


@pytest.fixture(scope="module")
def robot_config() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    # Offline-complete variant of the provider-bound so101_single_arm config:
    # gesture templates need real joint groups, and the provider config gets
    # them only from a live runtime description.
    config_path = repo_root / "src" / "robot_config" / "config" / "robots" / "so101_single_arm_legacy.yaml"
    return load_robot_config_dict(config_path)


@pytest.fixture(scope="module")
def embodied_config(robot_config: dict) -> dict:
    return robot_config["embodied"]


@pytest.fixture(scope="module")
def skill_templates(robot_config: dict) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "src" / "robot_config" / "config" / "robots" / "so101_single_arm_legacy.yaml"
    snapshot = compile_local_snapshot(robot_config, config_path)
    profile_path = repo_root / "src" / "skill_catalog" / "config" / "profiles" / "so101_single_arm.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    implementation_names = {item["name"]: item["implementation"] for item in profile["enabled_skills"]}
    templates = {}
    for name, _frozen in snapshot.templates.items():
        manifest_path = repo_root / "src" / "skill_catalog" / "config" / "skills" / name / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        implementation_name = implementation_names[name]
        implementation_path = manifest_path.parent / manifest["implementations"][implementation_name]
        implementation = yaml.safe_load(implementation_path.read_text(encoding="utf-8"))
        implementation["description"] = manifest["description"]
        templates[name] = implementation
    return templates


@pytest.fixture(scope="module")
def joint_limits(robot_config: dict) -> dict:
    teleop_limits = robot_config.get("teleoperation", {}).get("safety", {}).get("joint_limits")
    assert teleop_limits, "joint limits must be configured for social gesture validation"
    return teleop_limits


def _trajectory_step(skill_template: dict) -> dict:
    for step in skill_template["primitive_sequence"]:
        if step.get("trajectory_template"):
            return step
    raise AssertionError("social gesture must include a trajectory_template step")


def _previous_arm_primitive(primitive_sequence: list[dict], index: int) -> dict | None:
    for candidate in reversed(primitive_sequence[:index]):
        if candidate.get("primitive_name") not in GRIPPER_ONLY_PRIMITIVES:
            return candidate
    return None


@pytest.mark.parametrize("gesture", SOCIAL_GESTURES)
def test_social_gesture_registration_is_complete(gesture: str, skill_templates: dict) -> None:
    assert gesture in skill_templates


@pytest.mark.parametrize("gesture", TRAJECTORY_GESTURES)
def test_social_gesture_trajectory_expands_to_waypoints(gesture: str, skill_templates: dict) -> None:
    trajectory_template = _trajectory_step(skill_templates[gesture])["trajectory_template"]
    waypoints = expand_trajectory_template(trajectory_template)

    assert waypoints
    assert all(waypoint["primitive_name"] == "move_to_joint_positions" for waypoint in waypoints)
    assert all(waypoint["joint_positions"] for waypoint in waypoints)


@pytest.mark.parametrize("gesture", SOCIAL_GESTURES)
def test_social_gesture_does_not_force_return_home(gesture: str, skill_templates: dict) -> None:
    final_step = skill_templates[gesture]["primitive_sequence"][-1]

    assert not (final_step.get("primitive_name") == "move_to_named_pose" and final_step.get("pose_name") == "home")


def test_skill_templates_expand_trajectory_templates_for_runtime(skill_templates: dict) -> None:
    expanded_templates = get_skill_templates(skill_templates)
    trajectory_step = next(
        step
        for step in expanded_templates["wave_hello"]["primitive_sequence"]
        if step.get("primitive_name") == "move_through_joint_positions"
    )

    assert "trajectory_template" not in trajectory_step
    assert trajectory_step["joint_waypoints"]
    assert trajectory_step["waypoint_duration_sec"] == 0.05


def test_dance_duration_estimate_covers_generated_motion(skill_templates: dict) -> None:
    dance = skill_templates["dance_basic"]
    trajectory_step = _trajectory_step(dance)
    trajectory_template = trajectory_step["trajectory_template"]
    generated_motion_sec = (
        len(expand_trajectory_template(trajectory_template)) * trajectory_template["waypoint_duration_sec"]
    )
    entry_motion_sec = sum(
        float(step.get("duration_sec", 0.0))
        for step in dance["primitive_sequence"]
        if step.get("primitive_name") == "move_to_joint_positions"
    )
    estimate = dance["description"]["duration_sec_estimate"]

    assert estimate == 22.0
    assert estimate >= entry_motion_sec + generated_motion_sec


def test_trajectory_templates_do_not_use_ignored_return_to_base(skill_templates: dict) -> None:
    for skill_name, template in skill_templates.items():
        for step in template.get("primitive_sequence", []):
            trajectory_template = step.get("trajectory_template")
            if trajectory_template is not None:
                assert "return_to_base" not in trajectory_template, skill_name


def test_all_absolute_joint_trajectories_have_safe_entries(skill_templates: dict) -> None:
    expanded_templates = get_skill_templates(skill_templates)
    checked = 0

    for skill_name, template in expanded_templates.items():
        primitive_sequence = template.get("primitive_sequence", [])
        for index, step in enumerate(primitive_sequence):
            if step.get("primitive_name") != "move_through_joint_positions":
                continue

            checked += 1
            previous = _previous_arm_primitive(primitive_sequence, index)
            assert previous is not None, f"{skill_name} has no arm entry"
            assert previous.get("primitive_name") == "move_to_joint_positions", skill_name
            assert float(previous.get("duration_sec", 0.0)) > 0.0, skill_name
            first_waypoint = step["joint_waypoints"][0]["joint_positions"]
            assert previous.get("joint_positions") == first_waypoint, skill_name

    assert checked == 6


def test_all_social_gesture_trajectory_template_types_are_covered(skill_templates: dict) -> None:
    template_types = {
        _trajectory_step(skill_templates[gesture])["trajectory_template"]["type"] for gesture in TRAJECTORY_GESTURES
    }

    assert template_types == EXPECTED_TEMPLATE_TYPES


def test_celebrate_moves_around_observe_table(skill_templates: dict) -> None:
    primitive_sequence = skill_templates["celebrate"]["primitive_sequence"]

    assert primitive_sequence[0] == {"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}
    assert primitive_sequence[-1] == {"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}
    assert [(step["motion_direction"], step["motion_distance"]) for step in primitive_sequence[1:-1]] == [
        ("up", 0.04),
        ("down", 0.08),
        ("up", 0.04),
        ("left", 0.04),
        ("right", 0.08),
        ("left", 0.04),
    ]


def test_happy_spin_upright_uses_base_joint_spin(skill_templates: dict) -> None:
    trajectory_template = _trajectory_step(skill_templates["happy_spin_upright"])["trajectory_template"]

    assert trajectory_template["base_pose"] == {
        "1": 0.02,
        "2": 0.0,
        "3": -1.22,
        "4": -0.18,
        "5": 0.02,
    }
    assert trajectory_template["joints"]["1"]["terms"] == [
        {
            "amplitude": 0.55,
            "harmonic": 1,
            "phase": 0.0,
        }
    ]


def test_greet_observe_raise_moves_from_observe_table(skill_templates: dict) -> None:
    primitive_sequence = skill_templates["greet_observe_raise"]["primitive_sequence"]

    assert primitive_sequence[0] == {"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}
    assert primitive_sequence[-1] == {"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}
    assert [(step["motion_direction"], step["motion_distance"]) for step in primitive_sequence[1:-1]] == [
        ("up", 0.04),
        ("down", 0.04),
        ("up", 0.04),
        ("down", 0.04),
    ]


@pytest.mark.parametrize("gesture", TRAJECTORY_GESTURES)
def test_social_gesture_waypoints_stay_within_joint_limits(
    gesture: str,
    skill_templates: dict,
    joint_limits: dict,
) -> None:
    trajectory_template = _trajectory_step(skill_templates[gesture])["trajectory_template"]
    waypoints = expand_trajectory_template(trajectory_template)

    for waypoint_index, waypoint in enumerate(waypoints):
        for joint_name, position in waypoint["joint_positions"].items():
            limits = joint_limits[joint_name]
            assert limits["min"] <= position <= limits["max"], (
                f"{gesture} waypoint {waypoint_index} joint {joint_name}={position} "
                f"outside [{limits['min']}, {limits['max']}]"
            )


@pytest.mark.parametrize("gesture", TRAJECTORY_GESTURES)
def test_social_gesture_workspace_limits_are_structured(gesture: str, skill_templates: dict) -> None:
    trajectory_template = _trajectory_step(skill_templates[gesture])["trajectory_template"]
    workspace_limits = trajectory_template["workspace_limits"]

    assert workspace_limits["model"] == "so101_arm_v1"
    assert set(workspace_limits["points"]) >= REQUIRED_WORKSPACE_POINTS

    for bounds_by_axis in workspace_limits["points"].values():
        assert bounds_by_axis
        for axis_name, bounds in bounds_by_axis.items():
            assert axis_name in {"x", "y", "z"}
            assert isinstance(bounds, list)
            assert len(bounds) == 2
            assert bounds[0] <= bounds[1]


@pytest.mark.parametrize("gesture", ("act_cute", "happy_spin_upright"))
def test_wave_dance_zero_hold_waypoints_stay_within_workspace(gesture: str, skill_templates: dict) -> None:
    trajectory_template = _trajectory_step(skill_templates[gesture])["trajectory_template"]
    waypoints = expand_trajectory_template(trajectory_template)

    zero_hold_count = trajectory_template["zero_hold_count"]
    zero_hold_waypoints = waypoints[-zero_hold_count:]

    assert len(zero_hold_waypoints) == zero_hold_count
    for waypoint in zero_hold_waypoints:
        assert _pose_within_workspace(trajectory_template["workspace_limits"], waypoint["joint_positions"])
