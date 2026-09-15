"""Default neutral skill template definitions for the embodied pipeline."""

from __future__ import annotations

import copy
from typing import Any

from embodied_common.primitive_contracts import SUPPORTED_PRIMITIVES as SUPPORTED_PRIMITIVES
from embodied_common.trajectory_templates import expand_trajectory_template

# Must stay consistent with skill_library.resolver move_through_joint_positions default.
DEFAULT_WAYPOINT_DURATION_SEC = 0.08

DEFAULT_SKILL_TEMPLATES: dict[str, dict[str, Any]] = {
    "inspect_scene": {
        "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}],
    },
    "recover_safe_pose": {
        "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "home"}],
    },
    "recover_zero_pose": {
        "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "zero"}],
    },
    "open_gripper_skill": {
        "primitive_sequence": [{"primitive_name": "open_gripper"}],
    },
    "close_gripper_skill": {
        "primitive_sequence": [{"primitive_name": "close_gripper"}],
    },
    "move_relative_ee": {
        "primitive_sequence": [
            {
                "primitive_name": "move_relative_ee",
                "motion_direction_from_request": True,
                "motion_distance_from_request": True,
            }
        ],
    },
    "rotate_gripper_cw": {
        "primitive_sequence": [{"primitive_name": "rotate_gripper_cw", "motion_distance_from_request": True}],
    },
    "rotate_gripper_ccw": {
        "primitive_sequence": [{"primitive_name": "rotate_gripper_ccw", "motion_distance_from_request": True}],
    },
}

SUPPORTED_SKILL_EXECUTORS = {
    "grasp_pipeline",
    "placement_pipeline",
    "semantic_map_query",
    "imitate_human_motion",
    "sound_following",
}
DEFAULT_ALLOWED_SKILLS = list(DEFAULT_SKILL_TEMPLATES.keys())


def is_skill_disabled(template: Any) -> bool:
    return isinstance(template, dict) and template.get("disabled") is True


def _expand_skill_templates(raw_templates: dict[str, Any]) -> dict[str, Any]:
    resolved_templates = copy.deepcopy(raw_templates)
    for template in resolved_templates.values():
        primitive_sequence = template.get("primitive_sequence")
        if not isinstance(primitive_sequence, list):
            continue
        for step in primitive_sequence:
            if not isinstance(step, dict):
                continue
            trajectory_template = step.get("trajectory_template")
            if not trajectory_template:
                continue
            if not isinstance(trajectory_template, dict):
                raise ValueError("trajectory_template must be a mapping")
            step["joint_waypoints"] = expand_trajectory_template(trajectory_template)
            step["waypoint_duration_sec"] = float(
                trajectory_template.get(
                    "waypoint_duration_sec",
                    step.get("waypoint_duration_sec", DEFAULT_WAYPOINT_DURATION_SEC),
                )
            )
            del step["trajectory_template"]
    return resolved_templates


def get_skill_templates(raw_templates: dict[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_SKILL_TEMPLATES if raw_templates is None else raw_templates
    enabled_templates = {
        skill_name: template for skill_name, template in source.items() if not is_skill_disabled(template)
    }
    return _expand_skill_templates(enabled_templates)
