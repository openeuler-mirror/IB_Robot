"""Tests for the agent-facing skill `description` contract and its validator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.loader import _validate_skill_description

SO101 = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm_legacy.yaml"
SKILLS = SO101.parents[3] / "skill_catalog" / "config" / "skills"
PROFILE = SO101.parents[3] / "skill_catalog" / "config" / "profiles" / "so101_single_arm.yaml"

_REQUIRED_FIELDS = ("summary", "category", "when_to_use", "motion_scope", "intensity")


def _description(skill_name: str) -> dict:
    description = yaml.safe_load((SKILLS / skill_name / "manifest.yaml").read_text(encoding="utf-8")).get("description")
    assert description, f"{skill_name} is missing a description block"
    return description


def _enabled_skills() -> set[str]:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    return {entry["name"] for entry in profile["enabled_skills"]}


def test_so101_skills_all_carry_description_contract():
    skill_names = _enabled_skills()

    assert skill_names
    for skill_name in skill_names:
        description = _description(skill_name)
        for field in _REQUIRED_FIELDS:
            assert field in description, f"{skill_name}.description missing {field}"
        assert isinstance(description["when_to_use"], list) and description["when_to_use"]
        assert description["intensity"] in {"subtle", "moderate", "large"}
        assert description["summary"]


@pytest.mark.parametrize(
    "skill_name",
    ["wave_hello", "nod_yes", "shake_no", "celebrate", "greet_observe_raise", "act_cute", "happy_spin_upright"],
)
def test_social_gestures_declare_disambiguation(skill_name: str):
    description = _description(skill_name)
    do_not_use = description.get("do_not_use")

    assert do_not_use, f"{skill_name} must declare do_not_use to disambiguate from near-synonyms"
    for entry in do_not_use:
        assert entry.get("condition"), f"{skill_name} do_not_use entry needs a condition"
        assert entry.get("instead_use"), f"{skill_name} do_not_use entry needs instead_use"


def test_so101_description_aliases_match_supported_skills():
    """Every instead_use target must be enabled by the same profile."""
    known_skills = _enabled_skills()

    for skill_name in known_skills:
        for entry in _description(skill_name).get("do_not_use", []):
            target = entry["instead_use"]
            assert target in known_skills, f"{skill_name}.description.do_not_use points to unknown skill '{target}'"
            assert target != skill_name, f"{skill_name}.description.do_not_use points to itself"


def _collect_errors(description: dict, valid_skills: set[str], named_poses: dict | None = None) -> list[str]:
    errors: list[str] = []
    _validate_skill_description("demo", description, valid_skills, named_poses or {}, errors)
    return errors


def test_validator_accepts_well_formed_description():
    description = {
        "summary": "A well-formed demo skill.",
        "category": "demo",
        "when_to_use": ["demo case"],
        "do_not_use": [{"condition": "other case", "instead_use": "other"}],
        "motion_scope": ["wrist"],
        "anchor_pose": "none",
        "intensity": "subtle",
        "duration_sec_estimate": 1.5,
        "requires_motion_params": False,
    }
    assert _collect_errors(description, {"demo", "other"}) == []


def test_validator_rejects_unknown_instead_use_target():
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "do_not_use": [{"condition": "y", "instead_use": "ghost_skill"}],
    }
    errors = _collect_errors(description, {"demo"})
    assert any("ghost_skill" in e for e in errors)


def test_validator_rejects_self_referencing_instead_use():
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "do_not_use": [{"condition": "y", "instead_use": "demo"}],
    }
    errors = _collect_errors(description, {"demo"})
    assert any("same skill" in e for e in errors)


def test_validator_rejects_bad_vocab_and_values():
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "motion_scope": ["knee"],
        "intensity": "extreme",
        "duration_sec_estimate": -2.0,
        "anchor_pose": "nonexistent_pose",
        "rule_entry": "yes",
    }
    errors = _collect_errors(description, {"demo"})
    joined = " ".join(errors)
    assert "motion_scope" in joined
    assert "intensity" in joined
    assert "duration_sec_estimate" in joined
    assert "anchor_pose" in joined
    assert "rule_entry" in joined


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", {"text": "demo"}),
        ("category", True),
        ("intensity", False),
        ("anchor_pose", 123),
    ],
)
def test_validator_rejects_non_string_description_fields(field, value):
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "intensity": "subtle",
    }
    description[field] = value

    errors = _collect_errors(description, {"demo"})

    assert any(field in error and "string" in error for error in errors)


@pytest.mark.parametrize("duration", [True, "1.5", float("nan"), float("inf"), float("-inf"), 10**1000])
def test_validator_rejects_non_finite_or_non_numeric_duration(duration):
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "duration_sec_estimate": duration,
    }

    errors = _collect_errors(description, {"demo"})

    assert any("duration_sec_estimate must be a finite number" in error for error in errors)


def test_validator_rejects_missing_description():
    errors = _collect_errors(None, {"demo"})
    assert any("description is required" in error for error in errors)
