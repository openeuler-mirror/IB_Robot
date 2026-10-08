from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_DIGEST, PRIMITIVE_DESCRIPTORS
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.consumer import CatalogConsumerError, CatalogIdentity, verify_snapshot_response
from skill_catalog.digest import (
    derive_capability_digest,
    derive_capability_view_from_registry,
    derive_registry_digest,
    to_canonical_json,
)
from skill_catalog.models import SkillCompileContext, SkillRobotContext
from skill_catalog.source import DevelopmentStagingSkillSource


def _snapshot(tmp_path, *, required_capabilities: str = ""):
    package = tmp_path / "config" / "skills" / "open_gripper_skill"
    package.mkdir(parents=True)
    (tmp_path / "config" / "profiles").mkdir(parents=True)
    (tmp_path / "config" / "profiles" / "test_robot.yaml").write_text(
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
        f"""schema_version: 1
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
  parameters: {{type: object, properties: {{}}, required: [], additionalProperties: false}}
  recovery_policy: never_retry
{required_capabilities}implementations:
  test_robot: implementations/test_robot.yaml
""",
        encoding="utf-8",
    )
    (package / "implementations").mkdir()
    (package / "implementations" / "test_robot.yaml").write_text(
        """schema_version: 1
kind: primitive_sequence
robot: test_robot
initial_gripper_state: none
timeout_sec: 5.0
primitive_sequence:
- primitive_name: open_gripper
""",
        encoding="utf-8",
    )
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
    context = SkillCompileContext(robot, PRIMITIVE_DESCRIPTORS, PRIMITIVE_CONTRACT_DIGEST, {})
    return compile_skill_catalog(DevelopmentStagingSkillSource(tmp_path), profile_name="test_robot", context=context)


def _response(snapshot, *, payload=None, registry_digest=None, capability_digest=None):
    return SimpleNamespace(
        success=True,
        error_code="",
        message="",
        registry_epoch="epoch",
        generation=1,
        registry_digest=registry_digest or snapshot.registry_digest,
        capability_digest=capability_digest or snapshot.capability_digest,
        provenance_digest=snapshot.provenance_digest,
        snapshot_json=to_canonical_json(payload) if payload is not None else snapshot.snapshot_json,
    )


def _raw_digest(preimage):
    return hashlib.sha256(to_canonical_json(preimage).encode("utf-8")).hexdigest()


def _set_capability_entry_version(payload, version):
    registry_capability = payload["registry_preimage"]["skills"][0]["template"]["capability"]
    public_capability = payload["capability_preimage"]["capability_view"]["open_gripper_skill"]
    registry_capability["schema_version"] = version
    public_capability["schema_version"] = version


def test_consumer_accepts_compiler_snapshot(tmp_path):
    snapshot = _snapshot(tmp_path)

    view = verify_snapshot_response(_response(snapshot), CatalogIdentity("epoch", 1, snapshot.registry_digest))

    assert view.enabled_names == {"open_gripper_skill"}
    assert view.robot_context["robot_config_digest"] == "a" * 64


def test_consumer_rejects_individually_valid_but_cross_inconsistent_preimages(tmp_path):
    snapshot = _snapshot(tmp_path)
    payload = json.loads(snapshot.snapshot_json)
    payload["capability_preimage"]["profile_name"] = "other-profile"
    capability_digest = derive_capability_digest(payload["capability_preimage"])

    with pytest.raises(CatalogConsumerError, match="disagree"):
        verify_snapshot_response(
            _response(snapshot, payload=payload, capability_digest=capability_digest),
            CatalogIdentity("epoch", 1, snapshot.registry_digest),
        )


def test_consumer_rejects_capability_view_not_derived_from_registry(tmp_path):
    snapshot = _snapshot(tmp_path)
    payload = copy.deepcopy(json.loads(snapshot.snapshot_json))
    payload["capability_preimage"]["capability_view"]["open_gripper_skill"]["summary"] = "forged"
    capability_digest = derive_capability_digest(payload["capability_preimage"])

    with pytest.raises(CatalogConsumerError, match="capability view disagrees"):
        verify_snapshot_response(
            _response(snapshot, payload=payload, capability_digest=capability_digest),
            CatalogIdentity("epoch", 1, snapshot.registry_digest),
        )


def test_derived_capability_view_v1_omits_version_and_v2_includes_two(tmp_path):
    snapshot = _snapshot(tmp_path)
    registry = json.loads(snapshot.registry_preimage_json)

    v1_view = derive_capability_view_from_registry(registry)
    registry["skills"][0]["template"]["capability"]["schema_version"] = 2
    v2_view = derive_capability_view_from_registry(registry)

    assert "schema_version" not in v1_view["open_gripper_skill"]
    assert v2_view["open_gripper_skill"]["schema_version"] == 2


@pytest.mark.parametrize("version", [0, 1, 3])
def test_derived_capability_view_rejects_noncanonical_explicit_versions(tmp_path, version):
    snapshot = _snapshot(tmp_path)
    registry = json.loads(snapshot.registry_preimage_json)
    registry["skills"][0]["template"]["capability"]["schema_version"] = version

    with pytest.raises(ValueError, match="capability fields"):
        derive_capability_view_from_registry(registry)


def test_derived_capability_view_rejects_extra_fields_for_v1_and_v2(tmp_path):
    snapshot = _snapshot(tmp_path)

    for version in (None, 2):
        registry = json.loads(snapshot.registry_preimage_json)
        capability = registry["skills"][0]["template"]["capability"]
        if version is not None:
            capability["schema_version"] = version
        capability["unknown"] = True

        with pytest.raises(ValueError, match="capability fields"):
            derive_capability_view_from_registry(registry)


def test_consumer_accepts_exact_v2_derived_capability_fields(tmp_path):
    snapshot = _snapshot(tmp_path)
    payload = json.loads(snapshot.snapshot_json)
    _set_capability_entry_version(payload, 2)
    registry_digest = derive_registry_digest(payload["registry_preimage"])
    capability_digest = derive_capability_digest(payload["capability_preimage"])

    view = verify_snapshot_response(
        _response(
            snapshot,
            payload=payload,
            registry_digest=registry_digest,
            capability_digest=capability_digest,
        ),
        CatalogIdentity("epoch", 1, registry_digest),
    )

    assert view.capability_view["open_gripper_skill"]["schema_version"] == 2


@pytest.mark.parametrize(("version", "extra_field"), [(0, None), (1, None), (3, None), (2, "unknown")])
def test_consumer_rejects_noncanonical_capability_field_sets(tmp_path, version, extra_field):
    snapshot = _snapshot(tmp_path)
    payload = json.loads(snapshot.snapshot_json)
    _set_capability_entry_version(payload, version)
    if extra_field is not None:
        payload["registry_preimage"]["skills"][0]["template"]["capability"][extra_field] = True
        payload["capability_preimage"]["capability_view"]["open_gripper_skill"][extra_field] = True
    registry_digest = _raw_digest(payload["registry_preimage"])
    capability_digest = _raw_digest(payload["capability_preimage"])

    with pytest.raises(CatalogConsumerError, match="capability fields"):
        verify_snapshot_response(
            _response(
                snapshot,
                payload=payload,
                registry_digest=registry_digest,
                capability_digest=capability_digest,
            ),
            CatalogIdentity("epoch", 1, registry_digest),
        )


def test_consumer_accepts_compiled_required_capabilities(tmp_path):
    snapshot = _snapshot(tmp_path, required_capabilities="  required_capabilities: [motion.named]\n")

    view = verify_snapshot_response(_response(snapshot), CatalogIdentity("epoch", 1, snapshot.registry_digest))

    assert list(view.capability_view["open_gripper_skill"]["required_capabilities"]) == ["motion.named"]
    registry = json.loads(snapshot.registry_preimage_json)
    assert derive_capability_view_from_registry(registry)["open_gripper_skill"]["required_capabilities"] == [
        "motion.named"
    ]


@pytest.mark.parametrize("required", [[], [""], "motion.named", [7]])
def test_derived_capability_view_rejects_malformed_required_capabilities(tmp_path, required):
    snapshot = _snapshot(tmp_path, required_capabilities="  required_capabilities: [motion.named]\n")
    registry = json.loads(snapshot.registry_preimage_json)
    registry["skills"][0]["template"]["capability"]["required_capabilities"] = required

    with pytest.raises(ValueError, match="capability fields"):
        derive_capability_view_from_registry(registry)
