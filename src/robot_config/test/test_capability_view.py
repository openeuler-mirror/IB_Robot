import copy

import pytest

from embodied_common.capability_view import build_capability_view
from embodied_common.skill_templates import get_skill_templates
from robot_config.timeout_policy import DEFAULT_EMBODIED_TIMEOUT_POLICY, resolve_embodied_timeout_policy


def _description(summary):
    return {
        "summary": summary,
        "category": "motion",
        "when_to_use": ["test the public capability view"],
        "aliases_en": ["test"],
        "motion_scope": ["arm"],
        "intensity": "subtle",
    }


def _capability(summary, parameters=None):
    return {
        "schema_version": 1,
        "summary": summary,
        "domain": "manipulation",
        "moves_robot": True,
        "required_control_mode": "moveit_planning",
        "parameters": {} if parameters is None else parameters,
        "recovery_policy": "never_retry",
    }


def _normalized_config():
    return {
        "name": "test_robot",
        "embodied": {
            "named_poses": {
                "home": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}},
                "tray": {"position": {"x": 0.4, "y": 0.5, "z": 0.6}},
            },
            "named_targets": {"cup": {"grasp_pose": "home"}, "block": {"grasp_pose": "tray"}},
            "timeouts": {"task_budget_sec": 45.0, "default_skill_timeout_sec": 30.0, "rpc_timeout_sec": 5.0},
            "skill_templates": {
                "place": {
                    "description": _description("Place an item at a configured location."),
                    "capability": _capability("Place an item at a configured location."),
                    "initial_gripper_state": "open",
                    "primitive_sequence": [
                        {
                            "primitive_name": "move_to_named_pose",
                            "place_name_from_request": True,
                        }
                    ],
                },
                "target_pose": {
                    "description": _description("Move to a named target pose."),
                    "capability": _capability("Move to a named target pose."),
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", "target_pose_key": "grasp_pose"}],
                },
                "nudge": {
                    "description": _description("Move the end effector by a requested offset."),
                    "capability": _capability(
                        "Move the end effector by a requested offset.",
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["motion_direction", "motion_distance"],
                            "properties": {
                                "motion_direction": {
                                    "enum": ["forward", "backward", "left", "right", "up", "down"],
                                    "type": "string",
                                },
                                "motion_distance": {
                                    "exclusiveMinimum": 0,
                                    "type": "number",
                                    "unit": "meters",
                                },
                            },
                        },
                    ),
                    "primitive_sequence": [
                        {
                            "primitive_name": "move_relative_ee",
                            "motion_direction_from_request": True,
                            "motion_distance_from_request": True,
                        }
                    ],
                },
                "rotate": {
                    "description": _description("Rotate the gripper by a requested angle."),
                    "capability": _capability(
                        "Rotate the gripper by a requested angle.",
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["motion_distance"],
                            "properties": {
                                "motion_distance": {
                                    "exclusiveMinimum": 0,
                                    "type": "number",
                                    "unit": "degrees",
                                },
                            },
                        },
                    ),
                    "primitive_sequence": [
                        {"primitive_name": "rotate_gripper_cw", "motion_distance_from_request": True}
                    ],
                },
            },
        },
    }


def _skill(view, skill_name):
    return next(skill for skill in view["skills"] if skill["name"] == skill_name)


def _build_capability_view(config, timeout_policy=None):
    if timeout_policy is None:
        timeout_policy = resolve_embodied_timeout_policy(config["embodied"])
    return build_capability_view(config, timeout_policy=timeout_policy)


def test_capability_view_projects_explicit_capability_metadata_without_private_template_details():
    config = _normalized_config()
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "required": ["motion_direction", "motion_distance"],
        "properties": {
            "motion_direction": {
                "type": "string",
                "enum": ["forward", "backward", "left", "right", "up", "down"],
            },
            "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
        },
    }
    for skill_name, template in config["embodied"]["skill_templates"].items():
        template["capability"] = _capability(f"Public summary for {skill_name}.", parameters)

    view = _build_capability_view(config)
    nudge = _skill(view, "nudge")

    assert nudge == {
        "name": "nudge",
        "summary": "Public summary for nudge.",
        "domain": "manipulation",
        "moves_robot": True,
        "required_control_mode": "moveit_planning",
        "parameters": parameters,
        "recovery_policy": "never_retry",
    }
    parameters["properties"]["motion_distance"]["unit"] = "private"
    assert nudge["parameters"]["properties"]["motion_distance"]["unit"] == "meters"
    serialized = repr(view)
    for forbidden in ("primitive_sequence", "target_pose_key", "position", "grasp_pose", "ros_interface"):
        assert forbidden not in serialized


def test_capability_view_exposes_only_explicit_capability_metadata_and_request_schema():
    config = _normalized_config()

    view = _build_capability_view(config)

    assert view["robot_name"] == "test_robot"
    assert view["pose_names"] == ["home", "tray"]
    resolved = resolve_embodied_timeout_policy(config["embodied"])
    assert view["timeout_policy"] == {
        name: value for name, value in resolved.items() if not name.startswith("visual_game_")
    }
    assert [skill["name"] for skill in view["skills"]] == ["nudge", "place", "rotate", "target_pose"]

    assert _skill(view, "place")["parameters"] == {}
    assert _skill(view, "target_pose")["parameters"] == {}

    nudge_schema = _skill(view, "nudge")["parameters"]
    assert nudge_schema["required"] == ["motion_direction", "motion_distance"]
    assert nudge_schema["properties"]["motion_direction"] == {
        "enum": ["forward", "backward", "left", "right", "up", "down"],
        "type": "string",
    }
    assert nudge_schema["properties"]["motion_distance"] == {
        "exclusiveMinimum": 0,
        "type": "number",
        "unit": "meters",
    }
    assert _skill(view, "rotate")["parameters"]["properties"]["motion_distance"]["unit"] == "degrees"

    assert _skill(view, "place")["summary"] == "Place an item at a configured location."
    serialized = repr(view)
    for forbidden in (
        "primitive_sequence",
        "initial_gripper_state",
        "target_pose_key",
        "position",
        "grasp_pose",
        "description",
    ):
        assert forbidden not in serialized


def test_capability_view_does_not_project_legacy_description_fields():
    config = _normalized_config()
    description = config["embodied"]["skill_templates"]["place"]["description"]
    description.update(
        {
            "aliases": {"private": {"ros_interface": "/private/aliases"}},
            "aliases_zh": ["放置"],
            "anchor_pose": "tray",
            "duration_sec_estimate": 3.0,
            "requires_motion_params": True,
            "do_not_use": [
                {
                    "condition": "not a placement",
                    "instead_use": "nudge",
                    "ros_interface": "/private/redirect",
                }
            ],
            "rule_entry": True,
            "primitive_sequence": [{"primitive_name": "move_to_joint_positions"}],
            "joint_positions": {"1": 1.0},
            "pose": {"position": {"x": 1.0}},
            "ros_interface": "/private/action",
            "unknown_field": "private",
        }
    )

    public_skill = _skill(_build_capability_view(config), "place")

    assert public_skill["summary"] == "Place an item at a configured location."
    serialized = repr(public_skill)
    for forbidden in (
        "primitive_sequence",
        "private",
        "joint_positions",
        "pose",
        "ros_interface",
        "rule_entry",
        "unknown_field",
        "category",
        "when_to_use",
    ):
        assert forbidden not in serialized


def test_capability_view_normalizes_raw_templates_before_exposing_skills():
    config = {
        "name": "test_robot",
        "embodied": {
            "named_poses": {},
            "named_targets": {},
            "timeouts": {},
            "skill_templates": {
                "expanded": {
                    "description": _description("Expand a trajectory before exposing this skill."),
                    "capability": _capability("Expand a trajectory before exposing this skill."),
                    "primitive_sequence": [
                        {
                            "primitive_name": "move_through_joint_positions",
                            "trajectory_template": {
                                "type": "single_joint_wave_v1",
                                "active_waypoint_count": 1,
                                "base_pose": {"1": 0.0},
                                "joint": "1",
                                "amplitude": 0.0,
                            },
                        }
                    ],
                },
                "disabled_private": {
                    "disabled": True,
                    "description": _description("This disabled skill must remain private."),
                    "primitive_sequence": [{"primitive_name": "unknown_private_primitive"}],
                },
            },
        },
    }

    view = _build_capability_view(config)

    assert [skill["name"] for skill in view["skills"]] == ["expanded"]


def test_capability_view_uses_parameters_declared_by_each_skill():
    config = {
        "name": "test_robot",
        "embodied": {
            "named_poses": {"home": {}, "tray": {}},
            "named_targets": {
                "both": {"grasp_pose": "home", "lift_pose": "tray"},
                "grasp_only": {"grasp_pose": "home"},
                "lift_only": {"lift_pose": "tray"},
                "unrelated": {"observe_pose": "home"},
                "empty_pose": {"grasp_pose": ""},
                "non_string_pose": {"grasp_pose": 1},
                "unknown_pose": {"grasp_pose": "missing_pose"},
                "surrounded_pose": {"grasp_pose": " home "},
                "multi_empty_lift": {"grasp_pose": "home", "lift_pose": ""},
                "multi_non_string_lift": {"grasp_pose": "home", "lift_pose": 1},
                "multi_unknown_lift": {"grasp_pose": "home", "lift_pose": "missing_pose"},
            },
            "timeouts": {},
            "skill_templates": {
                "single_target": {
                    "description": _description("Move to a target grasp pose."),
                    "capability": _capability(
                        "Move to a target grasp pose.",
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["target_name"],
                            "properties": {"target_name": {"enum": ["configured_target"], "type": "string"}},
                        },
                    ),
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", "target_pose_key": "grasp_pose"}],
                },
                "multi_target": {
                    "description": _description("Move through target grasp and lift poses."),
                    "capability": _capability(
                        "Move through target grasp and lift poses.",
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["target_name"],
                            "properties": {"target_name": {"enum": ["configured_target"], "type": "string"}},
                        },
                    ),
                    "primitive_sequence": [
                        {"primitive_name": "move_to_named_pose", "target_pose_key": "grasp_pose"},
                        {"primitive_name": "move_to_named_pose", "target_pose_key": "lift_pose"},
                    ],
                },
            },
        },
    }

    view = _build_capability_view(config)

    assert _skill(view, "single_target")["parameters"]["properties"]["target_name"] == {
        "enum": ["configured_target"],
        "type": "string",
    }
    assert _skill(view, "multi_target")["parameters"]["properties"]["target_name"] == {
        "enum": ["configured_target"],
        "type": "string",
    }
    assert "target_pose_key" not in repr(view)


def test_capability_view_rejects_invalid_raw_trajectory_template():
    config = {
        "name": "test_robot",
        "embodied": {
            "named_poses": {},
            "named_targets": {},
            "timeouts": {},
            "skill_templates": {
                "invalid_trajectory": {
                    "description": _description("This trajectory is invalid."),
                    "capability": _capability("This trajectory is invalid."),
                    "primitive_sequence": [
                        {
                            "primitive_name": "move_through_joint_positions",
                            "trajectory_template": {"type": "unsupported"},
                        }
                    ],
                }
            },
        },
    }

    with pytest.raises(ValueError, match="Unsupported trajectory_template type"):
        _build_capability_view(config)


def test_capability_view_uses_resolved_timeout_policy_for_digest():
    omitted_timeouts = _normalized_config()
    omitted_timeouts["embodied"].pop("timeouts")
    explicit_defaults = copy.deepcopy(omitted_timeouts)
    explicit_defaults["embodied"]["timeouts"] = dict(DEFAULT_EMBODIED_TIMEOUT_POLICY)

    omitted_policy = resolve_embodied_timeout_policy(omitted_timeouts["embodied"])
    explicit_policy = resolve_embodied_timeout_policy(explicit_defaults["embodied"])
    omitted_view = _build_capability_view(omitted_timeouts, omitted_policy)
    explicit_view = _build_capability_view(explicit_defaults, explicit_policy)

    assert omitted_policy == explicit_policy
    assert omitted_view["timeout_policy"] == {
        name: value for name, value in omitted_policy.items() if not name.startswith("visual_game_")
    }
    assert omitted_view["capability_digest"] == explicit_view["capability_digest"]


def test_capability_view_digest_is_deterministic_for_equivalent_normalized_data():
    config = _normalized_config()
    reordered = copy.deepcopy(config)
    reordered["embodied"]["skill_templates"] = dict(reversed(list(reordered["embodied"]["skill_templates"].items())))

    assert _build_capability_view(config)["capability_digest"] == _build_capability_view(reordered)["capability_digest"]


def test_capability_view_does_not_derive_public_fields_from_primitive_details():
    config = _normalized_config()
    config["embodied"]["skill_templates"]["nudge"]["primitive_sequence"].append(
        {"primitive_name": "rotate_gripper_cw", "motion_distance_from_request": True}
    )

    view = _build_capability_view(config)

    assert _skill(view, "nudge")["parameters"]["properties"]["motion_distance"]["unit"] == "meters"
    assert "rotate_gripper_cw" not in repr(view)


def test_capability_view_rejects_templates_without_explicit_capability_metadata():
    config = {
        "name": "fallback_robot",
        "embodied": {
            "named_poses": {"home": {}},
            "named_targets": {},
            "timeouts": {},
            "skill_templates": get_skill_templates(None),
        },
    }
    with pytest.raises(ValueError, match="skill_templates.close_gripper_skill.capability must be a mapping"):
        _build_capability_view(config)


def test_capability_view_keeps_an_explicit_empty_skill_catalog_empty():
    config = {
        "name": "fallback_robot",
        "embodied": {
            "named_poses": {"home": {}},
            "named_targets": {},
            "timeouts": {},
        },
    }

    config["embodied"]["skill_templates"] = {}
    assert _build_capability_view(config)["skills"] == []


def test_capability_view_keeps_an_omitted_skill_catalog_empty():
    config = {
        "name": "fallback_robot",
        "embodied": {
            "named_poses": {"home": {}},
            "named_targets": {},
            "timeouts": {},
        },
    }

    assert _build_capability_view(config)["skills"] == []
