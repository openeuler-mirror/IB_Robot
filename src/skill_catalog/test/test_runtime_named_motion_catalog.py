"""Catalog rules for delegated skills that run a runtime-owned named motion."""

from dataclasses import replace
from pathlib import Path

import pytest

from embodied_common.dispatch_binding import delegated_executor_identity
from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_DIGEST, PRIMITIVE_DESCRIPTORS
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import (
    DelegatedExecutorDescriptor,
    SkillCompileContext,
    SkillCompileError,
    SkillRobotContext,
)
from skill_catalog.source import DevelopmentStagingSkillSource
from skill_catalog.validator import validate_robot_context

ENDPOINTS = {
    "skill_action": "/embodied/execute_skill",
    "primitive_action": "/embodied/execute_primitive",
    "validate_skill_service": "/embodied/validate_skill",
    "validate_primitive_service": "/embodied/validate_primitive",
    "gateway_status_service": "/embodied/get_skill_gateway_status",
    "begin_workflow_service": "/embodied/begin_workflow_execution",
    "finalize_workflow_service": "/embodied/finalize_workflow_execution",
    "task_executor_action": "/task_executor/execute_task_plan",
    "arm_trajectory_action": "/arm_trajectory_controller/follow_joint_trajectory",
    "move_configuration_service": "/runtime/move_to_joint",
}

IMPLEMENTATION = """schema_version: 1
kind: delegated_executor
robot: runtime_v1
executor: runtime_named_motion
binding:
  motion: handshake
required_args: []
timeout_sec: 45.0
"""


def _robot(required_control_mode: str = "named_motion") -> SkillRobotContext:
    return SkillRobotContext(
        robot_name="motion_owning_robot",
        context_schema_version=1,
        robot_config_digest="a" * 64,
        named_poses={},
        named_targets={},
        arm_joint_names=(),
        joint_limits={},
        workspace_limits={},
        required_control_mode=required_control_mode,
        timeout_policy={"task_budget_sec": 120.0},
        relative_motion_reference_frame="base",
        relative_motion_step_m=0.03,
        relative_motion_direction_mapping={},
        gripper_open_position=1.0,
        gripper_closed_position=0.0,
        execution_endpoints=dict(ENDPOINTS),
    )


def _context(executor: str = "runtime_named_motion") -> SkillCompileContext:
    descriptor = DelegatedExecutorDescriptor(
        **delegated_executor_identity(
            name=executor,
            endpoint_name="/motion/execute_named",
            endpoint_kind="ros_action",
            configuration={"runtime_name": "vendor_runtime"},
        )
    )
    return SkillCompileContext(
        robot=_robot(),
        primitive_contracts=PRIMITIVE_DESCRIPTORS,
        primitive_contract_digest=PRIMITIVE_CONTRACT_DIGEST,
        delegated_executors={executor: descriptor},
    )


def _write_catalog(root: Path, *, implementation: str = IMPLEMENTATION, parameters: str = "{}") -> None:
    package = root / "config" / "skills" / "handshake"
    (package / "implementations").mkdir(parents=True)
    (root / "config" / "profiles").mkdir(parents=True)
    (root / "config" / "profiles" / "motion_owning_robot.yaml").write_text(
        """schema_version: 1
name: motion_owning_robot
robot_name: motion_owning_robot
enabled_skills:
  - name: handshake
    implementation: runtime_v1
    planner_visible: true
""",
        encoding="utf-8",
    )
    (package / "manifest.yaml").write_text(
        f"""schema_version: 1
name: handshake
version: 1.0.0
semantic_level: skill
description:
  summary: Offer a handshake.
  category: social_greeting
  when_to_use: [greet someone formally]
  motion_scope: [arm]
  intensity: moderate
capability:
  schema_version: 1
  summary: Offer a handshake.
  domain: social
  moves_robot: true
  required_control_mode: named_motion
  parameters:
    type: object
    properties: {parameters}
    required: []
    additionalProperties: false
  recovery_policy: never_retry
  required_capabilities: [motion.named]
implementations:
  runtime_v1: implementations/runtime_v1.yaml
""",
        encoding="utf-8",
    )
    (package / "implementations" / "runtime_v1.yaml").write_text(implementation, encoding="utf-8")


def _compile(root: Path, context: SkillCompileContext | None = None):
    return compile_skill_catalog(
        DevelopmentStagingSkillSource(root), profile_name="motion_owning_robot", context=context or _context()
    )


def _diagnostics(raised) -> list[tuple[str, str]]:
    return [(item.field_path, item.message) for item in raised.value.diagnostics]


def test_bound_named_motion_skill_compiles_without_arguments(tmp_path):
    _write_catalog(tmp_path)

    snapshot = _compile(tmp_path)

    assert snapshot.enabled_skill_names == ("handshake",)
    assert snapshot.planner_visible_skill_names == ("handshake",)
    assert snapshot.templates["handshake"]["binding"] == {"motion": "handshake"}
    assert snapshot.templates["handshake"]["executor"] == "runtime_named_motion"
    assert snapshot.capability_view["handshake"]["required_control_mode"] == "named_motion"
    assert list(snapshot.capability_view["handshake"]["required_capabilities"]) == ["motion.named"]


def test_named_motion_mode_is_valid_in_a_v1_robot_context():
    assert validate_robot_context(_robot()) == []
    # The original v1 vocabulary is unchanged.
    for mode in ("teleop", "model_inference", "moveit_planning"):
        assert validate_robot_context(replace(_robot(), required_control_mode=mode)) == []


@pytest.mark.parametrize(
    "binding",
    [
        "",
        "binding: {}\n",
        "binding:\n  motion: ''\n",
        "binding:\n  motion: Wave-Hand\n",
        "binding:\n  motion: 7\n",
        "binding:\n  motion: wave\n  target: right\n",
        "binding: wave\n",
    ],
)
def test_binding_must_name_exactly_one_motion(tmp_path, binding):
    implementation = IMPLEMENTATION.replace("binding:\n  motion: handshake\n", binding)
    _write_catalog(tmp_path, implementation=implementation)

    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path)

    assert ("binding", "binding must be exactly {motion: <runtime motion name>}") in _diagnostics(raised)


def test_named_motion_skill_takes_no_caller_arguments(tmp_path):
    implementation = IMPLEMENTATION.replace("required_args: []", "required_args: [target_name]")
    _write_catalog(tmp_path, implementation=implementation, parameters="{target_name: {type: string, freeform: true}}")

    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path)

    messages = [message for _, message in _diagnostics(raised)]
    assert "runtime_named_motion takes no caller arguments; required_args must be []" in messages
    assert "runtime_named_motion skills must declare no capability parameters" in messages


@pytest.mark.parametrize("required", ["", "  required_capabilities: [joint.state]\n"])
def test_named_motion_skill_must_require_a_named_motion_capability(tmp_path, required):
    _write_catalog(tmp_path)
    manifest_path = tmp_path / "config" / "skills" / "handshake" / "manifest.yaml"
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(manifest.replace("  required_capabilities: [motion.named]\n", required), encoding="utf-8")

    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path)

    assert (
        "capability.required_capabilities",
        "runtime_named_motion skills must require motion.named or motion.posture",
    ) in _diagnostics(raised)


def test_other_delegated_executors_reject_a_binding(tmp_path):
    implementation = IMPLEMENTATION.replace("runtime_named_motion", "sound_following")
    _write_catalog(tmp_path, implementation=implementation)

    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path, _context("sound_following"))

    fields = [field for field, _ in _diagnostics(raised)]
    assert "binding" in fields


def test_other_delegated_executors_still_require_arguments(tmp_path):
    implementation = IMPLEMENTATION.replace("runtime_named_motion", "sound_following").replace(
        "binding:\n  motion: handshake\n", ""
    )
    _write_catalog(tmp_path, implementation=implementation)

    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path, _context("sound_following"))

    assert ("required_args", "required_args must be a non-empty unique string list") in _diagnostics(raised)
