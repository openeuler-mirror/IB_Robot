import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("check_agent_skill.py")
CANONICAL_SKILL = SCRIPT_PATH.parents[1] / ".agents" / "skills" / "ibrobot-control" / "SKILL.md"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_agent_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def valid_skill(tmp_path):
    skill_path = tmp_path / "ibrobot-control" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_bytes(CANONICAL_SKILL.read_bytes())
    return skill_path


def test_canonical_skill_contract_passes():
    assert _load_checker().validate_skill(CANONICAL_SKILL) == []


def test_checker_requires_immediate_binding_and_cancel_truthfulness(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("Immediately bind that exact tuple once", "Bind the tuple later")
    content = content.replace("not a second user confirmation gate", "a user confirmation gate")
    content = content.replace("Cancellation requested is not robot stopped", "Cancellation is complete")
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing immediate internal plan binding requirement" in errors
    assert "missing no-second-confirmation-gate requirement" in errors
    assert "missing cancellation truthfulness requirement" in errors


def test_checker_requires_plan_command_order_and_prohibitions(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    status = "1. Query the Gateway: `robot-skill status`."
    listing = "2. Discover capabilities: `robot-skill list-skills`."
    content = content.replace(status, "TEMP").replace(listing, status).replace("TEMP", listing)
    content = content.replace("must not enable motion authorization", "may enable motion authorization")
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "robot-skill workflow commands are missing or out of order" in errors
    assert "missing prohibition: enable motion authorization" in errors


def test_checker_requires_exact_display_and_flush_before_binding(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("exact ordered steps", "summary")
    content = content.replace("then flush the", "then continue", 1)
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "workflow must display the exact plan, digest, registry identity, and fresh task ID" in errors
    assert "plan presentation flush must appear between validate-plan and confirm-plan" in errors


def test_checker_requires_plan_cancel_and_routing(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("cancel-plan", "cancel")
    content = content.replace(
        "Natural-language single-Skill and Workflow requests both use the plan workflow above through the composite entry.",
        "Natural language is routed by the model.",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing Agent plan cancel command" in errors
    assert "missing natural-language single-Skill/Workflow routing rule" in errors


def test_checker_requires_typed_workflow_step_schema_version_contract(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace(
        "`workflow-json` is an array of flat `WorkflowStep` objects.",
        "`workflow-json` is an array of flat objects.",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing typed WorkflowStep schema_version requirement" in errors


def test_checker_requires_no_domain_based_workflow_step_rewrite(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace(
        "Never infer or rewrite `WorkflowStep.schema_version` from the skill domain.",
        "Infer `WorkflowStep.schema_version` from the skill domain.",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing prohibition on domain-based WorkflowStep schema rewrites" in errors
