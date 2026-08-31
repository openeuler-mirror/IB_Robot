"""ROS-independent loading and querying of robot skill capabilities."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.digest import (
    derive_capability_digest,
    derive_provenance_digest,
    derive_registry_digest,
    to_canonical_json,
)
from skill_catalog.models import DelegatedExecutorDescriptor, SkillCompileContext, SkillRobotContext
from skill_catalog.source import AmentShareSkillSource, DevelopmentStagingSkillSource, DirectoryReleaseSkillSource

from embodied_common.capability_view import project_capability_timeout_policy
from embodied_common.dispatch_binding import delegated_executor_identity, load_delegated_model_identity
from embodied_common.primitive_contracts import primitive_contract_for_version
from embodied_common.visual_game_contracts import build_visual_game_capability_view
from robot_config.config_path import resolve_robot_config_path
from robot_config.loader import (
    get_effective_visual_game_policies,
    load_robot_config_dict,
    robot_config_digest,
    robot_context_schema_version,
    robot_execution_endpoints,
    robot_supported_control_modes,
)
from robot_config.timeout_policy import resolve_embodied_timeout_policy

_LIST_CAPABILITY_FIELDS = (
    "summary",
    "domain",
    "moves_robot",
    "required_control_mode",
)


class UnknownSkillError(ValueError):
    """Raised when a catalog query names a skill that is not enabled."""

    code = "UNKNOWN_SKILL"

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"unknown skill: {skill_name}")


class UnknownGameError(ValueError):
    """Raised when a visual game is absent or disabled in robot_config."""

    code = "UNKNOWN_GAME"

    def __init__(self, game_name: str) -> None:
        self.game_name = game_name
        super().__init__(f"unknown or disabled visual game: {game_name}")


@dataclass(frozen=True)
class CatalogContext:
    view: dict[str, Any]
    game_view: dict[str, Any]


@dataclass(frozen=True)
class GatewayTransport:
    status_service: str
    validate_skill_service: str
    skill_action_name: str
    primitive_action_name: str = "/embodied/execute_primitive"
    validate_primitive_service: str = "/embodied/validate_primitive"
    snapshot_service: str = "/embodied/get_skill_snapshot"
    reload_service: str = "/embodied/reload_skill_catalog"
    plan_service: str = "/embodied/plan_agent_command"
    validate_plan_service: str = "/embodied/validate_agent_plan"
    confirm_plan_service: str = "/embodied/confirm_agent_plan"
    execute_plan_action: str = "/embodied/execute_agent_plan"
    start_visual_game_service: str = "/embodied/start_visual_game"
    get_visual_game_result_service: str = "/embodied/get_visual_game_result"


def _game_view(robot_config: dict[str, Any], timeout_policy: dict[str, float]) -> dict[str, Any]:
    embodied = robot_config.get("embodied", {})
    games = get_effective_visual_game_policies(robot_config)
    return build_visual_game_capability_view(
        str(robot_config.get("name", "")),
        games,
        timeout_sec=timeout_policy["visual_game_timeout_sec"],
        result_retention_sec=timeout_policy["visual_game_result_retention_sec"],
        result_capacity=embodied.get("visual_game_result_capacity", 128),
        start_service=embodied.get("start_visual_game_service", "/embodied/start_visual_game"),
        result_service=embodied.get("get_visual_game_result_service", "/embodied/get_visual_game_result"),
        event_topic=embodied.get("visual_game_event_topic", "/embodied/visual_game_events"),
    )


def load_catalog_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> CatalogContext:
    """Load one normalized config for public catalog use (ROS-independent)."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return CatalogContext(
        view=_snapshot_capability_view(compile_local_snapshot(robot_config, resolved_path)),
        game_view=_game_view(robot_config, timeout_policy),
    )


def load_runtime_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatalogContext, GatewayTransport]:
    """Load both catalog view and ROS transport for runtime Gateway commands."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return (
        CatalogContext(
            view=_snapshot_capability_view(compile_local_snapshot(robot_config, resolved_path)),
            game_view=_game_view(robot_config, timeout_policy),
        ),
        _gateway_transport(embodied),
    )


def load_visual_game_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> CatalogContext:
    """Load visual-game metadata without compiling the motion Skill catalog."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    return _visual_game_context(robot_config)


def load_visual_game_runtime_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatalogContext, GatewayTransport]:
    """Load visual-game metadata and ROS transport without motion dependencies."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    return _visual_game_context(robot_config), _gateway_transport(embodied)


def _visual_game_context(robot_config: dict[str, Any]) -> CatalogContext:
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return CatalogContext(
        view={"robot_name": robot_config["name"], "timeout_policy": timeout_policy},
        game_view=_game_view(robot_config, timeout_policy),
    )


def _gateway_transport(embodied: dict[str, Any]) -> GatewayTransport:
    return GatewayTransport(
        status_service=embodied.get("skill_gateway_status_service", "/embodied/get_skill_gateway_status"),
        snapshot_service=embodied.get("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot"),
        reload_service=embodied.get("skill_catalog_reload_service", "/embodied/reload_skill_catalog"),
        validate_skill_service=embodied.get("validate_skill_service", "/embodied/validate_skill"),
        skill_action_name=embodied.get("skill_action_name", "/embodied/execute_skill"),
        primitive_action_name=embodied.get("primitive_action_name", "/embodied/execute_primitive"),
        validate_primitive_service=embodied.get("validate_primitive_service", "/embodied/validate_primitive"),
        plan_service=embodied.get("plan_service", "/embodied/plan_agent_command"),
        validate_plan_service=embodied.get("validate_plan_service", "/embodied/validate_agent_plan"),
        confirm_plan_service=embodied.get("confirm_plan_service", "/embodied/confirm_agent_plan"),
        execute_plan_action=embodied.get("execute_plan_action", "/embodied/execute_agent_plan"),
        start_visual_game_service=embodied.get("start_visual_game_service", "/embodied/start_visual_game"),
        get_visual_game_result_service=embodied.get(
            "get_visual_game_result_service", "/embodied/get_visual_game_result"
        ),
    )


def compile_local_snapshot(robot_config: dict[str, Any], config_path: Path):
    """Compile the profile selected by one normalized robot config."""
    embodied = robot_config["embodied"]
    execution = embodied.get("execution", {})
    source_mode = embodied.get("skill_catalog_source_mode", "installed")
    source_root = Path(embodied.get("skill_catalog_source_root", ""))
    if source_mode == "development":
        if not source_root.is_absolute():
            repository_root = next((parent for parent in config_path.parents if parent.name == "src"), None)
            if repository_root is None:
                source_mode = "installed"
            else:
                source_root = repository_root.parent / source_root
        if source_mode == "installed":
            source = AmentShareSkillSource()
        else:
            source = DevelopmentStagingSkillSource(source_root)
    elif source_mode == "production":
        source = DirectoryReleaseSkillSource(source_root)
    else:
        source = AmentShareSkillSource()
    # Keep the ROS-independent CLI catalog context aligned with the runtime
    # SkillExecutorNode.  Delegated executors are selected by the enabled
    # execution configuration, not by one particular profile name; otherwise
    # the PC grasp profile cannot validate its own ``grasp_pipeline`` entry.
    delegated = {}
    grasp_execution = robot_config.get("grasp_execution", {})
    if grasp_execution.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="grasp_pipeline",
                endpoint_name=grasp_execution.get("action_name", "/manipulation/execute_pick"),
                configuration=grasp_execution,
                **load_delegated_model_identity(grasp_execution),
            )
        )
        delegated[descriptor.name] = descriptor

    placement_execution = robot_config.get("placement_execution", {})
    if placement_execution.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="placement_pipeline",
                endpoint_name=placement_execution.get("action_name", "/manipulation/execute_place"),
                configuration=placement_execution,
            )
        )
        delegated[descriptor.name] = descriptor

    hri_runtime = embodied.get("imitate_human_motion", {})
    if isinstance(hri_runtime, dict) and hri_runtime.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="imitate_human_motion",
                endpoint_name=hri_runtime.get("action_name", "/hri/imitate_human_motion"),
                configuration={"implementation": "mock_v1"},
            )
        )
        delegated[descriptor.name] = descriptor

    semantic_mapping = robot_config.get("semantic_mapping", {})
    if isinstance(semantic_mapping, dict) and semantic_mapping.get("enabled", False):
        interfaces = semantic_mapping.get("interfaces", {})
        target_service = interfaces.get("query_service", "/semantic_mapping/get_objects")
        target_watch = semantic_mapping.get("target_watch", {})
        stand_off_distance_m = target_watch.get("stand_off_distance_m", 0.3)
        if isinstance(target_service, str) and target_service.strip():
            configuration = {
                "query_service": target_service,
                "stand_off_distance_m": float(stand_off_distance_m),
            }
            descriptor = DelegatedExecutorDescriptor(
                **delegated_executor_identity(
                    name="semantic_map_query",
                    endpoint_name=target_service,
                    endpoint_kind="ros_service",
                    configuration=configuration,
                )
            )
            delegated[descriptor.name] = descriptor
    robot_context = SkillRobotContext(
        robot_name=robot_config["name"],
        context_schema_version=robot_context_schema_version(robot_config),
        robot_config_digest=robot_config_digest(robot_config),
        named_poses=embodied.get("named_poses", {}),
        named_targets=embodied.get("named_targets", {}),
        arm_joint_names=tuple(robot_config.get("joints", {}).get("arm", [])),
        joint_limits=robot_config.get("teleoperation", {}).get("safety", {}).get("joint_limits", {}),
        workspace_limits=embodied.get("safety", {}).get("workspace", {}),
        required_control_mode=robot_config["skill_required_control_mode"],
        timeout_policy=project_capability_timeout_policy(resolve_embodied_timeout_policy(embodied)),
        relative_motion_reference_frame=execution.get("relative_motion_reference_frame", "base"),
        relative_motion_step_m=execution.get("relative_motion_step_m", 0.03),
        relative_motion_direction_mapping=execution.get("relative_motion_direction_mapping", {}),
        gripper_open_position=execution.get("gripper_open_position", 1.0),
        gripper_closed_position=execution.get("gripper_closed_position", 0.0),
        execution_endpoints=robot_execution_endpoints(robot_config),
        supported_control_modes=robot_supported_control_modes(robot_config),
    )
    primitive_contract = primitive_contract_for_version(robot_context.context_schema_version)
    return compile_skill_catalog(
        source,
        profile_name=embodied["skill_catalog_profile"],
        context=SkillCompileContext(
            robot=robot_context,
            primitive_contracts=primitive_contract.descriptors,
            primitive_contract_digest=primitive_contract.digest,
            delegated_executors=delegated,
        ),
    )


def _snapshot_capability_view(snapshot):
    def thaw(value):
        if isinstance(value, Mapping):
            return {str(key): thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    skills = [thaw(snapshot.capability_view[name]) for name in sorted(snapshot.capability_view)]
    for skill in skills:
        skill["timeout_sec"] = float(snapshot.templates[skill["name"]]["timeout_sec"])
    return {
        "robot_name": snapshot.robot_name,
        "skills": skills,
        "pose_names": sorted(snapshot.robot_context.named_poses),
        "timeout_policy": project_capability_timeout_policy(snapshot.robot_context.timeout_policy),
        "capability_digest": snapshot.capability_digest,
        "profile_name": snapshot.profile_name,
    }


def load_capability_catalog(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the normalized config and return its public capability view."""
    return load_catalog_context(config_name=config_name, config_path=config_path).view


def capability_view_from_snapshot(snapshot: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Verify one exact Gateway snapshot and return its public catalog view."""
    expected_identity = (
        status["registry_epoch"],
        int(status["registry_generation"]),
        status["registry_digest"],
    )
    actual_identity = (
        snapshot["registry_epoch"],
        int(snapshot["generation"]),
        snapshot["registry_digest"],
    )
    if not snapshot["success"] or actual_identity != expected_identity:
        raise ValueError("SKILL_REGISTRY_VERSION_MISMATCH: snapshot identity does not match Gateway status")
    try:
        payload = json.loads(snapshot["snapshot_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot payload is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot schema is invalid")
    if to_canonical_json(payload) != snapshot["snapshot_json"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot payload is not canonical")
    registry_preimage = payload.get("registry_preimage")
    capability_preimage = payload.get("capability_preimage")
    provenance_preimage = payload.get("provenance_preimage")
    if not all(isinstance(value, dict) for value in (registry_preimage, capability_preimage, provenance_preimage)):
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot preimages are invalid")
    if derive_registry_digest(registry_preimage) != snapshot["registry_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: registry digest does not match")
    if derive_capability_digest(capability_preimage) != snapshot["capability_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: capability digest does not match")
    if derive_provenance_digest(provenance_preimage) != snapshot["provenance_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: provenance digest does not match")
    capability_mapping = capability_preimage.get("capability_view")
    if not isinstance(capability_mapping, dict):
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: capability view is invalid")
    planner_visible_names = {str(name) for name in capability_preimage.get("planner_visible_skill_names", [])}
    semantic_levels = {
        str(entry["name"]): str(entry["semantic_level"])
        for entry in registry_preimage.get("skills", [])
        if isinstance(entry, dict) and "name" in entry and "semantic_level" in entry
    }
    timeout_caps = {
        str(entry["name"]): float(entry["template"]["timeout_sec"])
        for entry in registry_preimage.get("skills", [])
        if isinstance(entry, dict) and isinstance(entry.get("template"), dict) and "timeout_sec" in entry["template"]
    }
    skills = [copy.deepcopy(capability_mapping[name]) for name in sorted(capability_mapping)]
    for skill in skills:
        skill["planner_visible"] = skill["name"] in planner_visible_names
        skill["semantic_level"] = semantic_levels.get(skill["name"], "")
        if skill["name"] in timeout_caps:
            skill["timeout_sec"] = timeout_caps[skill["name"]]
    return {
        "robot_name": capability_preimage["robot_name"],
        "skills": skills,
        "pose_names": list(capability_preimage["named_pose_names"]),
        "timeout_policy": project_capability_timeout_policy(capability_preimage["timeout_policy"]),
        "capability_digest": snapshot["capability_digest"],
        "profile_name": capability_preimage["profile_name"],
    }


def _skill_entry(view: dict[str, Any], skill_name: str) -> dict[str, Any]:
    normalized_name = skill_name.strip()
    for skill in view["skills"]:
        if skill["name"] == normalized_name:
            return skill
    raise UnknownSkillError(normalized_name)


def list_skills(view: dict[str, Any]) -> dict[str, Any]:
    skills = []
    for skill in view["skills"]:
        entry = {
            "name": skill["name"],
            "contract_schema_version": int(skill.get("schema_version", 1)),
            **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
        }
        skills.append(entry)
    return {
        "robot_name": view["robot_name"],
        "config_digest": view["capability_digest"],
        "skills": skills,
    }


def describe_skill(view: dict[str, Any], skill_name: str) -> dict[str, Any]:
    skill = _skill_entry(view, skill_name)
    result = {
        "robot_name": view["robot_name"],
        "name": skill["name"],
        "schema_version": int(skill.get("schema_version", 1)),
        **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
        "parameters": copy.deepcopy(skill["parameters"]),
        "recovery_policy": copy.deepcopy(skill["recovery_policy"]),
        "timeout_policy": copy.deepcopy(view["timeout_policy"]),
        "config_digest": view["capability_digest"],
    }
    if "timeout_sec" in skill:
        result["timeout_sec"] = float(skill["timeout_sec"])
    return result


def list_poses(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_name": view["robot_name"],
        "config_digest": view["capability_digest"],
        "poses": list(view["pose_names"]),
    }


def list_games(game_view: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_name": game_view["robot_name"],
        "config_digest": game_view["config_digest"],
        "games": [
            {
                "name": game["name"],
                "summary": game["summary"],
                "result_field": game["result_schema"]["field"],
            }
            for game in game_view["games"]
        ],
    }


def require_enabled_game(game_view: dict[str, Any], game_name: str) -> dict[str, Any]:
    normalized_name = game_name.strip()
    for game in game_view["games"]:
        if game["name"] == normalized_name:
            return copy.deepcopy(game)
    raise UnknownGameError(normalized_name)


def describe_game(game_view: dict[str, Any], game_name: str) -> dict[str, Any]:
    game = require_enabled_game(game_view, game_name)
    return {
        "robot_name": game_view["robot_name"],
        "name": game["name"],
        "summary": game["summary"],
        "required_inputs": game["required_inputs"],
        "result_schema": game["result_schema"],
        "timeout_sec": game_view["timeout_sec"],
        "result_retention_sec": game_view["result_retention_sec"],
        "result_capacity": game_view["result_capacity"],
        "config_digest": game_view["config_digest"],
    }
