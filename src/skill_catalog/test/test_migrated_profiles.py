from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import DelegatedExecutorDescriptor, SkillCompileContext, SkillRobotContext
from skill_catalog.source import DevelopmentStagingSkillSource

from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_DIGEST, PRIMITIVE_DESCRIPTORS
from robot_config.loader import load_robot_config_dict, robot_config_digest
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from robot_skill_cli.catalog import compile_local_snapshot

ROOT = Path(__file__).resolve().parents[2]
CATALOG_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG_DIR = ROOT / "robot_config" / "config" / "robots"
PROFILES = (
    "so101_single_arm",
    "lekiwi_handeye_realsense_grasp",
    "lekiwi_handeye_realsense_grasp_pc",
    "so101_rtp_distributed",
)
SO101_V1_BASE_COMMIT = "18bbaa419d16b91a66f6a5857d096ff21f3f0549"
SO101_V1_REGISTRY_DIGEST = "c8c29a1ed666289b41f236d3ae84c84883cfd6b38f1282f67224c8e81d2394b6"
SO101_V1_CAPABILITY_DIGEST = "9899815166f5684ba69b8401c94e623a90c41460be308e7250d3d3850395f6a4"


def _context(config: dict, capability_digest: str) -> SkillCompileContext:
    embodied = config["embodied"]
    execution = embodied.get("execution", {})
    endpoint = config.get("grasp_execution", {}).get("action_name", "/manipulation/execute_pick")
    delegated = {}
    if config["name"] in {"lekiwi_handeye_realsense_grasp", "lekiwi_handeye_realsense_grasp_pc"}:
        descriptor = DelegatedExecutorDescriptor(
            name="grasp_pipeline",
            contract_version="1",
            endpoint_kind="ros_action",
            endpoint_name=endpoint,
            configuration_digest=hashlib.sha256(
                json.dumps(
                    {"endpoint_kind": "ros_action", "endpoint_name": endpoint},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            # The catalog compiler only needs a complete identity. Runtime launch
            # performs the strict manifest validation against the selected bundle.
            model_deployment_name="ascend_310p" if config["name"] == "lekiwi_handeye_realsense_grasp" else "torch_cuda",
            model_fingerprint="0" * 64,
            model_bundle_digest="1" * 64,
        )
        delegated[descriptor.name] = descriptor
        place = DelegatedExecutorDescriptor(
            name="placement_pipeline",
            contract_version="1",
            endpoint_kind="ros_action",
            endpoint_name=config.get("placement_execution", {}).get("action_name", "/manipulation/execute_place"),
            configuration_digest=hashlib.sha256(
                json.dumps(
                    {"endpoint_kind": "ros_action", "endpoint_name": "/manipulation/execute_place"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            model_deployment_name="",
            model_fingerprint="",
            model_bundle_digest="",
        )
        delegated[place.name] = place
    robot = SkillRobotContext(
        robot_name=config["name"],
        context_schema_version=1,
        robot_config_digest=capability_digest,
        named_poses=embodied.get("named_poses", {}),
        named_targets=embodied.get("named_targets", {}),
        arm_joint_names=tuple(config.get("joints", {}).get("arm", [])),
        joint_limits=config.get("teleoperation", {}).get("safety", {}).get("joint_limits", {}),
        workspace_limits=embodied.get("safety", {}).get("workspace", {}),
        required_control_mode=config["skill_required_control_mode"],
        timeout_policy=resolve_embodied_timeout_policy(embodied),
        relative_motion_reference_frame=execution.get("relative_motion_reference_frame", "base"),
        relative_motion_step_m=execution.get("relative_motion_step_m", 0.03),
        relative_motion_direction_mapping=execution.get("relative_motion_direction_mapping", {}),
        gripper_open_position=execution.get("gripper_open_position", 1.0),
        gripper_closed_position=execution.get("gripper_closed_position", 0.0),
        execution_endpoints={
            "skill_action": embodied.get("skill_action_name", "/embodied/execute_skill"),
            "primitive_action": embodied.get("primitive_action_name", "/embodied/execute_primitive"),
            "validate_skill_service": embodied.get("validate_skill_service", "/embodied/validate_skill"),
            "validate_primitive_service": embodied.get("validate_primitive_service", "/embodied/validate_primitive"),
            "gateway_status_service": embodied.get(
                "skill_gateway_status_service", "/embodied/get_skill_gateway_status"
            ),
            "begin_workflow_service": embodied.get("begin_workflow_service", "/embodied/begin_workflow_execution"),
            "finalize_workflow_service": embodied.get(
                "finalize_workflow_service", "/embodied/finalize_workflow_execution"
            ),
            "task_executor_action": execution.get("task_executor_action_name", "/task_executor/execute_task_plan"),
            "arm_trajectory_action": execution.get(
                "arm_trajectory_action_name", "/arm_controller/follow_joint_trajectory"
            ),
            "move_configuration_service": execution.get(
                "move_configuration_service", "/moveit_gateway/move_to_configuration"
            ),
        },
    )
    return SkillCompileContext(
        robot=robot,
        primitive_contracts=PRIMITIVE_DESCRIPTORS,
        primitive_contract_digest=PRIMITIVE_CONTRACT_DIGEST,
        delegated_executors=delegated,
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_migrated_profile_preserves_legacy_templates_capabilities_and_visibility(profile: str, monkeypatch) -> None:
    monkeypatch.setenv("WORKSPACE", str(ROOT.parent))
    config = load_robot_config_dict(ROBOT_CONFIG_DIR / f"{profile}.yaml")
    compiled = compile_skill_catalog(
        DevelopmentStagingSkillSource(CATALOG_ROOT),
        profile_name=profile,
        context=_context(config, robot_config_digest(config)),
    )
    expected_enabled = {
        "so101_single_arm": {
            "inspect_scene",
            "recover_safe_pose",
            "recover_zero_pose",
            "move_relative_ee",
            "open_gripper_skill",
            "close_gripper_skill",
            "rotate_gripper_cw",
            "rotate_gripper_ccw",
            "dance_basic",
            "wave_hello",
            "nod_yes",
            "shake_no",
            "celebrate",
            "act_cute",
            "greet_observe_raise",
            "happy_spin_upright",
        },
        "so101_rtp_distributed": {
            "inspect_scene",
            "recover_safe_pose",
            "recover_zero_pose",
            "move_relative_ee",
            "open_gripper_skill",
            "close_gripper_skill",
            "rotate_gripper_cw",
            "rotate_gripper_ccw",
            "dance_basic",
            "wave_hello",
            "nod_yes",
            "shake_no",
            "celebrate",
            "act_cute",
            "greet_observe_raise",
            "happy_spin_upright",
        },
        "lekiwi_handeye_realsense_grasp": {
            "inspect_scene",
            "recover_safe_pose",
            "recover_zero_pose",
            "move_relative_ee",
            "open_gripper_skill",
            "close_gripper_skill",
            "pick_object",
            "place_in_container",
        },
        "lekiwi_handeye_realsense_grasp_pc": {
            "inspect_scene",
            "recover_safe_pose",
            "recover_zero_pose",
            "move_relative_ee",
            "open_gripper_skill",
            "close_gripper_skill",
            "pick_object",
            "place_in_container",
        },
    }[profile]
    assert set(compiled.enabled_skill_names) == expected_enabled
    assert set(compiled.planner_visible_skill_names) == expected_enabled


def test_equivalent_so101_profiles_select_shared_stable_implementation_variant() -> None:
    for profile in ("so101_single_arm", "so101_rtp_distributed"):
        profile_config = yaml.safe_load(
            (CATALOG_ROOT / "config" / "profiles" / f"{profile}.yaml").read_text(encoding="utf-8")
        )
        assert all(entry["implementation"] == "so101_arm_v1" for entry in profile_config["enabled_skills"])


def test_so101_v1_registry_and_capability_digests_match_base_identity(monkeypatch) -> None:
    monkeypatch.setenv("WORKSPACE", str(ROOT.parent))
    config_path = ROBOT_CONFIG_DIR / "so101_single_arm.yaml"
    snapshot = compile_local_snapshot(load_robot_config_dict(config_path), config_path)

    assert snapshot.registry_digest == SO101_V1_REGISTRY_DIGEST, SO101_V1_BASE_COMMIT
    assert snapshot.capability_digest == SO101_V1_CAPABILITY_DIGEST, SO101_V1_BASE_COMMIT
