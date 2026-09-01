#!/usr/bin/env python3
"""Validate the repository contract for the ibrobot-control Agent Skill.

This script validates the ``ibrobot-control`` plan and cancellation contracts.
When additional Agent skills are added, extend this script to accept a
``--skill-name`` parameter and load per-skill rule sets instead of
duplicating the hardcoded checks below.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

PLAN_COMMANDS = (
    "status",
    "list-skills",
    "plan-workflow",
    "describe",
    "validate-plan",
    "confirm-plan",
    "execute-plan",
)
REQUIRED_RULES = {
    "then flush the presentation to the user": "missing synchronous plan presentation flush requirement",
    "immediately bind that exact tuple once": "missing immediate internal plan binding requirement",
    "not a second user confirmation gate": "missing no-second-confirmation-gate requirement",
    "must not launch or restart the pipeline": "missing prohibition: launch or restart pipeline",
    "must not enable motion authorization": "missing prohibition: enable motion authorization",
    "must not modify ros parameters": "missing prohibition: modify ROS parameters",
    "must not call primitive, moveit, controller, or raw ros2 motion commands": (
        "missing prohibition: primitive, MoveIt, controller, or raw ros2 motion commands"
    ),
    "cancellation requested is not robot stopped": "missing cancellation truthfulness requirement",
    "must not automatically retry after failure, timeout, or unknown result": (
        "missing prohibition: automatic retry after failure, timeout, or unknown result"
    ),
    "natural-language single-skill and workflow requests both use the plan workflow above through the composite entry": (
        "missing natural-language single-Skill/Workflow routing rule"
    ),
    "stop on any failure": "missing stop-on-failure rule",
    "must not source environment scripts, select a ros domain, or discover another repository/config": (
        "missing prohibition: environment or repository discovery"
    ),
    "on any nonzero exit, report the exact cli error and stop": "missing truthful nonzero-exit reporting rule",
    "call `plan-workflow` exactly once with the user's original wording": "missing single-attempt workflow planning rule",
    "do not retry alternate phrasings": "missing prohibition: alternate workflow paraphrase retries",
    "`workflow-json` is an array of flat `workflowstep` objects": (
        "missing typed WorkflowStep schema_version requirement"
    ),
    "every object must include an explicit `schema_version`": ("missing typed WorkflowStep schema_version requirement"),
    "never infer or rewrite `workflowstep.schema_version` from the skill domain": (
        "missing prohibition on domain-based WorkflowStep schema rewrites"
    ),
}


def _parse_frontmatter(content: str) -> tuple[dict[str, str], list[str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ["missing YAML frontmatter"]
    try:
        closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return {}, ["unterminated YAML frontmatter"]

    frontmatter_text = "\n".join(lines[: closing_index + 1])
    errors = []
    if len(frontmatter_text) > 1024:
        errors.append("YAML frontmatter exceeds 1024 characters")
    values = {}
    for line in lines[1:closing_index]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            errors.append(f"invalid YAML frontmatter line: {line}")
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values, errors


def _validate_command_order(content: str) -> bool:
    positions = []
    for command in PLAN_COMMANDS:
        match = re.search(rf"`robot-skill\b[^`]*\b{re.escape(command)}\b[^`]*`", content)
        if match is None:
            return False
        positions.append(match.start())
    return positions == sorted(positions)


def _required_workflow(content: str) -> str | None:
    match = re.search(
        r"^## Natural-Language Plan Workflow\s*$\n(.*?)(?=^## |\Z)",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        match = re.search(
            r"^## Diagnostic Plan Workflow\s*$\n(.*?)(?=^## |\Z)",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
    return None if match is None else match.group(1)


def _required_composite_workflow(content: str) -> str | None:
    match = re.search(
        r"^## Composite Workflow Entry\s*$\n(.*?)(?=^## |\Z)",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )
    return None if match is None else match.group(1)


def validate_skill(skill_path: Path) -> list[str]:
    if not skill_path.is_file():
        return [f"skill file does not exist: {skill_path}"]
    content = skill_path.read_text(encoding="utf-8")
    frontmatter, errors = _parse_frontmatter(content)

    name = frontmatter.get("name", "")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        errors.append("frontmatter name must contain only lowercase letters, numbers, and hyphens")
    if name != "ibrobot-control" or skill_path.parent.name != name:
        errors.append("frontmatter name and skill directory must both be ibrobot-control")
    description = frontmatter.get("description", "")
    if not description.startswith("Use when "):
        errors.append("frontmatter description must start with 'Use when '")

    normalized = " ".join(content.lower().split())
    workflow = _required_workflow(content)
    composite = _required_composite_workflow(content)
    if composite is None or re.search(r"robot-skill\s+run-workflow\b", composite) is None:
        errors.append("composite workflow entry is missing run-workflow")
    if workflow is None or any(
        re.search(rf"`robot-skill\b[^`]*\b{re.escape(command)}\b[^`]*`", workflow) is None for command in PLAN_COMMANDS
    ):
        errors.append("required workflow section is missing ordered robot-skill commands")
    elif not _validate_command_order(workflow):
        errors.append("robot-skill workflow commands are missing or out of order")
    validate_match = None if workflow is None else re.search(r"`robot-skill\b[^`]*\bvalidate-plan\b[^`]*`", workflow)
    confirm_match = None if workflow is None else re.search(r"`robot-skill\b[^`]*\bconfirm-plan\b[^`]*`", workflow)
    if (
        validate_match is not None
        and confirm_match is not None
        and re.search(
            r"flush\s+the\s+presentation\s+to\s+the\s+user",
            workflow[validate_match.end() : confirm_match.start()],
            flags=re.IGNORECASE,
        )
        is None
    ):
        errors.append("plan presentation flush must appear between validate-plan and confirm-plan")
    if workflow is not None and not all(
        phrase in " ".join(workflow.lower().split())
        for phrase in ("exact ordered steps", "plan digest", "registry identity", "fresh task id")
    ):
        errors.append("workflow must display the exact plan, digest, registry identity, and fresh task ID")
    if re.search(r"`robot-skill\b[^`]*\bcancel\b[^`]*--task-id\b[^`]*`", content) is None:
        errors.append("missing task-ID cancel command")
    if re.search(r"`robot-skill\b[^`]*\bcancel-plan\b[^`]*--task-id\b[^`]*`", content) is None:
        errors.append("missing Agent plan cancel command")
    if "sigint/sigterm" not in normalized:
        errors.append("missing current execute signal cancellation guidance")
    for requirement, message in REQUIRED_RULES.items():
        if requirement not in normalized:
            errors.append(message)
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-path", required=True, type=Path)
    args = parser.parse_args(argv)
    errors = validate_skill(args.skill_path)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: ibrobot-control skill contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
