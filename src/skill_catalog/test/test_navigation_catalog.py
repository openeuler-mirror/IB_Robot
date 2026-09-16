from dataclasses import replace
from pathlib import Path

import pytest

from embodied_common.primitive_contracts import primitive_contract_for_version
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import DelegatedExecutorDescriptor, SkillCompileContext, SkillCompileError, SkillRobotContext
from skill_catalog.source import DevelopmentStagingSkillSource
from skill_catalog.validator import validate_manifest

CATALOG_ROOT = Path(__file__).resolve().parents[1]
NAVIGATION_SKILLS = ("nav_straight", "nav_turn", "nav_abs_coordinate", "resolve_object_pose")


def _parameter_schema(properties):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


@pytest.mark.parametrize(
    ("name", "properties"),
    [
        (
            "nav_straight",
            {
                "direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]},
                "distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
        ),
        (
            "nav_turn",
            {
                "direction": {"type": "string", "enum": ["left", "right"]},
                "degree": {"type": "number", "exclusiveMinimum": 0, "unit": "degrees"},
            },
        ),
        (
            "nav_abs_coordinate",
            {
                "x": {"type": "number", "unit": "meters"},
                "y": {"type": "number", "unit": "meters"},
                "yaw": {"type": "number", "unit": "degrees"},
            },
        ),
    ],
)
def test_navigation_manifest_parameter_schemas_are_supported(name, properties):
    manifest = {
        "schema_version": 2,
        "name": name,
        "version": "1.0.0",
        "semantic_level": "atomic_operator",
        "description": {
            "summary": "Execute one navigation command.",
            "category": "navigation",
            "when_to_use": ["move the mobile base"],
            "motion_scope": ["base"],
            "intensity": "moderate",
        },
        "capability": {
            "schema_version": 2,
            "summary": "Execute one navigation command.",
            "domain": "navigation",
            "moves_robot": True,
            "required_control_mode": "base_navigation",
            "parameters": _parameter_schema(properties),
            "recovery_policy": "ask_user",
        },
        "implementations": {"lekiwi_navigation_v2": "implementations/lekiwi_navigation_v2.yaml"},
    }

    assert validate_manifest(manifest, package_name=name, source_relative_path=f"{name}/manifest.yaml") == []


def _navigation_context() -> SkillCompileContext:
    primitive_contract = primitive_contract_for_version(2)
    robot = SkillRobotContext(
        robot_name="lekiwi_lidar",
        context_schema_version=2,
        robot_config_digest="a" * 64,
        named_poses={},
        named_targets={},
        arm_joint_names=(),
        joint_limits={},
        workspace_limits={},
        required_control_mode="base_navigation",
        timeout_policy={"task_budget_sec": 600.0},
        relative_motion_reference_frame="base",
        relative_motion_step_m=0.03,
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
            "navigation_action": "/navigation/execute",
        },
    )
    return SkillCompileContext(
        robot=robot,
        primitive_contracts=primitive_contract.descriptors,
        primitive_contract_digest=primitive_contract.digest,
        delegated_executors={
            "semantic_map_query": DelegatedExecutorDescriptor(
                name="semantic_map_query",
                contract_version="1",
                endpoint_kind="ros_service",
                endpoint_name="/semantic_mapping/resolve_target",
                configuration_digest="a" * 64,
                model_deployment_name="",
                model_fingerprint="",
                model_bundle_digest="",
            ),
        },
    )


def test_lekiwi_lidar_profile_compiles_three_atomic_navigation_skills():
    snapshot = compile_skill_catalog(
        DevelopmentStagingSkillSource(CATALOG_ROOT),
        profile_name="lekiwi_lidar",
        context=_navigation_context(),
    )

    navigation_skills = tuple(sorted(name for name in NAVIGATION_SKILLS if name != "resolve_object_pose"))
    assert snapshot.enabled_skill_names == navigation_skills
    assert snapshot.planner_visible_skill_names == navigation_skills
    assert snapshot.primitive_contract_digest == primitive_contract_for_version(2).digest
    for skill_name in navigation_skills:
        assert snapshot.semantic_levels[skill_name] == "atomic_operator"
        assert snapshot.capability_view[skill_name]["schema_version"] == 2
        template = snapshot.templates[skill_name]
        if skill_name == "resolve_object_pose":
            assert template.get("executor") == "semantic_map_query"
        else:
            primitive_sequence = template["primitive_sequence"]
            assert len(primitive_sequence) == 1
            assert dict(primitive_sequence[0]) == {
                "primitive_name": skill_name,
                **{
                    f"{parameter}_from_request": True
                    for parameter in snapshot.parameter_schemas[skill_name]["required"]
                },
            }
            assert snapshot.requirements[skill_name] == frozenset({"navigation", "validate_skill"})


def test_navigation_compile_uses_context_selected_primitive_contract():
    context = _navigation_context()
    v1_contract = primitive_contract_for_version(1)

    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(
            DevelopmentStagingSkillSource(CATALOG_ROOT),
            profile_name="lekiwi_lidar",
            context=replace(
                context,
                primitive_contracts=v1_contract.descriptors,
                primitive_contract_digest=v1_contract.digest,
            ),
        )

    assert any(diagnostic.error_code == "SKILL_SNAPSHOT_DIGEST_MISMATCH" for diagnostic in raised.value.diagnostics)


def test_navigation_profile_rejects_v1_context_and_selected_contract():
    context = _navigation_context()
    v1_contract = primitive_contract_for_version(1)
    legacy_robot = replace(
        context.robot,
        context_schema_version=1,
        execution_endpoints={
            name: endpoint
            for name, endpoint in context.robot.execution_endpoints.items()
            if name != "navigation_action"
        },
    )

    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(
            DevelopmentStagingSkillSource(CATALOG_ROOT),
            profile_name="lekiwi_lidar",
            context=replace(
                context,
                robot=legacy_robot,
                primitive_contracts=v1_contract.descriptors,
                primitive_contract_digest=v1_contract.digest,
            ),
        )

    assert any(
        diagnostic.error_code in {"SKILL_SCHEMA_INVALID", "SKILL_REFERENCE_MISSING"}
        for diagnostic in raised.value.diagnostics
    )
