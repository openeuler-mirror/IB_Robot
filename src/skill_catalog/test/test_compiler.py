from pathlib import Path

import pytest

from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_DIGEST, PRIMITIVE_DESCRIPTORS
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import SkillCompileContext, SkillCompileError, SkillRobotContext
from skill_catalog.source import DevelopmentStagingSkillSource


def _context() -> SkillCompileContext:
    robot = SkillRobotContext(
        robot_name="test_robot",
        context_schema_version=1,
        robot_config_digest="a" * 64,
        named_poses={"home": {"position": {"x": 0.0}}},
        named_targets={},
        arm_joint_names=("1",),
        joint_limits={"1": {"lower": -1.0, "upper": 1.0}},
        workspace_limits={},
        required_control_mode="moveit_planning",
        timeout_policy={"task_budget_sec": 60.0},
        relative_motion_reference_frame="base",
        relative_motion_step_m=0.01,
        relative_motion_direction_mapping={},
        gripper_open_position=1.0,
        gripper_closed_position=0.0,
        execution_endpoints={
            "skill_action": "/embodied/execute_skill",
            "primitive_action": "/embodied/execute_primitive",
            "validate_skill_service": "/embodied/validate_skill",
            "validate_primitive_service": "/embodied/validate_primitive",
            "gateway_status_service": "/embodied/get_skill_gateway_status",
            "begin_workflow_service": "/embodied/begin_workflow_execution",
            "finalize_workflow_service": "/embodied/finalize_workflow_execution",
            "task_executor_action": "/embodied/execute_task",
            "arm_trajectory_action": "/arm/execute_trajectory",
            "move_configuration_service": "/arm/move_configuration",
        },
    )
    return SkillCompileContext(
        robot=robot,
        primitive_contracts=PRIMITIVE_DESCRIPTORS,
        primitive_contract_digest=PRIMITIVE_CONTRACT_DIGEST,
        delegated_executors={},
    )


def _write_catalog(root: Path, *, invalid: bool = False) -> None:
    package = root / "config" / "skills" / "open_gripper_skill"
    package.mkdir(parents=True)
    (root / "config" / "profiles").mkdir(parents=True)
    (root / "config" / "profiles" / "test_robot.yaml").write_text(
        """schema_version: 1
name: test_robot
robot_name: test_robot
enabled_skills:
  - name: open_gripper_skill
    implementation: test_robot
    planner_visible: true
""",
        encoding="utf-8",
    )
    (package / "manifest.yaml").write_text(
        """schema_version: 1
name: open_gripper_skill
version: 1.0.0
semantic_level: atomic_operator
description:
  summary: Open the gripper.
  category: gripper
  when_to_use: [release an object]
  motion_scope: [gripper]
  intensity: subtle
capability:
  schema_version: 1
  summary: Open the gripper.
  domain: manipulation
  moves_robot: true
  required_control_mode: moveit_planning
  parameters:
    type: object
    properties: {}
    required: []
    additionalProperties: false
  recovery_policy: never_retry
implementations:
  test_robot: implementations/test_robot.yaml
""",
        encoding="utf-8",
    )
    (package / "implementations").mkdir()
    (package / "implementations" / "test_robot.yaml").write_text(
        f"""schema_version: 1
kind: primitive_sequence
robot: test_robot
initial_gripper_state: none
timeout_sec: {61.0 if invalid else 5.0}
primitive_sequence:
  - primitive_name: open_gripper
""",
        encoding="utf-8",
    )


def test_compiler_builds_immutable_snapshot(tmp_path):
    _write_catalog(tmp_path)
    snapshot = compile_skill_catalog(
        DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context()
    )
    assert snapshot.enabled_skill_names == ("open_gripper_skill",)
    assert snapshot.planner_visible_skill_names == ("open_gripper_skill",)
    assert snapshot.capability_view["open_gripper_skill"]["semantic_level"] == "atomic_operator"
    assert "schema_version" not in snapshot.capability_view["open_gripper_skill"]
    assert "required_capabilities" not in snapshot.capability_view["open_gripper_skill"]
    with pytest.raises(TypeError):
        snapshot.templates["new"] = {}


def test_compiler_rejects_robot_timeout_limit(tmp_path):
    _write_catalog(tmp_path, invalid=True)
    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context())
    assert raised.value.code == "SKILL_LIMIT_VIOLATION"


def test_compiler_rejects_non_hidden_skill_directory_without_manifest(tmp_path):
    _write_catalog(tmp_path)
    (tmp_path / "config" / "skills" / "broken_skill").mkdir()

    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context())

    assert any(
        diagnostic.source_relative_path == "config/skills/broken_skill"
        and diagnostic.error_code == "SKILL_SCHEMA_INVALID"
        for diagnostic in raised.value.diagnostics
    )


def test_compiler_validates_manifest_outside_enabled_profile(tmp_path):
    _write_catalog(tmp_path)
    disabled = tmp_path / "config" / "skills" / "disabled_broken"
    disabled.mkdir()
    (disabled / "manifest.yaml").write_text("schema_version: 1\nname: wrong_name\n", encoding="utf-8")

    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context())

    assert any(
        diagnostic.source_relative_path == "config/skills/disabled_broken/manifest.yaml"
        for diagnostic in raised.value.diagnostics
    )


def test_compiler_reports_malformed_yaml_as_schema_diagnostic(tmp_path):
    _write_catalog(tmp_path)
    malformed = tmp_path / "config" / "skills" / "malformed"
    malformed.mkdir()
    (malformed / "manifest.yaml").write_text("name: [unterminated\n", encoding="utf-8")

    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context())

    assert any(diagnostic.error_code == "SKILL_SCHEMA_INVALID" for diagnostic in raised.value.diagnostics)


def test_compiler_accepts_valid_required_capabilities(tmp_path):
    _write_catalog(tmp_path)
    manifest_path = tmp_path / "config" / "skills" / "open_gripper_skill" / "manifest.yaml"
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest = manifest.replace(
        "  recovery_policy: never_retry",
        "  recovery_policy: never_retry\n  required_capabilities: [joint.trajectory, gripper.1d]",
    )
    manifest_path.write_text(manifest, encoding="utf-8")

    snapshot = compile_skill_catalog(
        DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context()
    )
    assert tuple(snapshot.capability_view["open_gripper_skill"]["required_capabilities"]) == (
        "joint.trajectory",
        "gripper.1d",
    )


def test_compiler_rejects_empty_required_capabilities_list(tmp_path):
    _write_catalog(tmp_path)
    manifest_path = tmp_path / "config" / "skills" / "open_gripper_skill" / "manifest.yaml"
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest = manifest.replace(
        "  recovery_policy: never_retry",
        "  recovery_policy: never_retry\n  required_capabilities: []",
    )
    manifest_path.write_text(manifest, encoding="utf-8")

    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context())

    assert any(
        diagnostic.error_code == "SKILL_SCHEMA_INVALID" and "required_capabilities" in diagnostic.message
        for diagnostic in raised.value.diagnostics
    )


def test_compiler_rejects_non_string_required_capabilities_entry(tmp_path):
    _write_catalog(tmp_path)
    manifest_path = tmp_path / "config" / "skills" / "open_gripper_skill" / "manifest.yaml"
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest = manifest.replace(
        "  recovery_policy: never_retry",
        "  recovery_policy: never_retry\n  required_capabilities: [123]",
    )
    manifest_path.write_text(manifest, encoding="utf-8")

    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context())

    assert any(
        diagnostic.error_code == "SKILL_SCHEMA_INVALID" and "required_capabilities" in diagnostic.message
        for diagnostic in raised.value.diagnostics
    )


@pytest.mark.parametrize("value", ["null", "joint.trajectory", "['']", "['   ']", "{}"])
def test_compiler_rejects_malformed_required_capabilities(tmp_path, value):
    _write_catalog(tmp_path)
    manifest_path = tmp_path / "config" / "skills" / "open_gripper_skill" / "manifest.yaml"
    manifest = manifest_path.read_text(encoding="utf-8").replace(
        "  recovery_policy: never_retry",
        f"  recovery_policy: never_retry\n  required_capabilities: {value}",
    )
    manifest_path.write_text(manifest, encoding="utf-8")
    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=_context())
    assert any(diagnostic.field_path == "capability.required_capabilities" for diagnostic in raised.value.diagnostics)
