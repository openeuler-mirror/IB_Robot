"""Closed-schema validation for skill packages, implementations and profiles."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from skill_catalog.models import (
    DELEGATED_ENDPOINT_KINDS,
    DISPATCH_KIND_CAPABILITY_BY_CONTEXT_VERSION,
    EXECUTION_ENDPOINT_ROLES_BY_CONTEXT_VERSION,
    DelegatedExecutorDescriptor,
    SkillDiagnostic,
    SkillRobotContext,
)

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
SEMANTIC_LEVELS = frozenset({"atomic_operator", "skill"})
CONTROL_MODES = frozenset({"teleop", "model_inference", "moveit_planning", "base_navigation"})
CONTROL_MODES_BY_SCHEMA_VERSION = {
    1: frozenset({"teleop", "model_inference", "moveit_planning"}),
    2: CONTROL_MODES,
    3: CONTROL_MODES,
}
RECOVERY_POLICIES = frozenset({"never_retry", "ask_user", "recover_safe_pose"})
MOTION_SCOPES = frozenset({"base", "shoulder", "elbow", "wrist", "gripper", "arm"})
INITIAL_GRIPPER_STATES = frozenset({"open", "closed", "hold", "none"})
CAPABILITY_PARAMETER_NAMES_V1 = frozenset(
    {
        "target_name",
        "container_name",
        "place_name",
        "motion_direction",
        "motion_distance",
        "arm_side",
        "imitation_duration_sec",
    }
)
CAPABILITY_PARAMETER_NAMES = frozenset(
    {
        "direction",
        "distance",
        "degree",
        "x",
        "y",
        "yaw",
        "stand_off_distance_m",
    }
    | CAPABILITY_PARAMETER_NAMES_V1
)
CAPABILITY_PARAMETER_NAMES_BY_SCHEMA_VERSION = {
    1: CAPABILITY_PARAMETER_NAMES_V1,
    2: CAPABILITY_PARAMETER_NAMES,
}
SOURCE_SCHEMA_VERSIONS = frozenset({1, 2})

MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "semantic_level",
        "description",
        "description_variants",
        "capability",
        "implementations",
    }
)
DESCRIPTION_FIELDS = frozenset(
    {
        "summary",
        "category",
        "when_to_use",
        "motion_scope",
        "intensity",
        "aliases_zh",
        "aliases_en",
        "anchor_pose",
        "duration_sec_estimate",
        "requires_motion_params",
        "rule_entry",
        "do_not_use",
    }
)
CAPABILITY_FIELDS = frozenset(
    {
        "schema_version",
        "summary",
        "domain",
        "moves_robot",
        "required_control_mode",
        "parameters",
        "recovery_policy",
        "required_capabilities",
    }
)
PROFILE_FIELDS = frozenset({"schema_version", "name", "robot_name", "enabled_skills"})
PROFILE_ENTRY_FIELDS = frozenset({"name", "implementation", "planner_visible"})


def validate_profile(
    profile: Mapping[str, Any], *, profile_name: str, robot_name: str, source_relative_path: str = ""
) -> list[SkillDiagnostic]:
    diagnostics: list[SkillDiagnostic] = []
    _exact_fields(profile, PROFILE_FIELDS, diagnostics, source_relative_path=source_relative_path)
    if profile.get("schema_version") != 1:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "schema_version must be 1",
            source_relative_path=source_relative_path,
            field_path="schema_version",
        )
    if profile.get("name") != profile_name:
        _error(
            diagnostics,
            "SKILL_PROFILE_NOT_FOUND",
            "profile name must match the file name",
            source_relative_path=source_relative_path,
            field_path="name",
        )
    if profile.get("robot_name") != robot_name:
        _error(
            diagnostics,
            "SKILL_LIMIT_VIOLATION",
            "profile robot_name does not match robot context",
            source_relative_path=source_relative_path,
            field_path="robot_name",
        )
    entries = profile.get("enabled_skills")
    if not isinstance(entries, list) or not entries:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "enabled_skills must be a non-empty list",
            source_relative_path=source_relative_path,
            field_path="enabled_skills",
        )
        return diagnostics
    names: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"enabled_skills[{index}]"
        if not isinstance(entry, Mapping):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "entry must be an object",
                source_relative_path=source_relative_path,
                field_path=path,
            )
            continue
        _exact_fields(
            entry, PROFILE_ENTRY_FIELDS, diagnostics, source_relative_path=source_relative_path, field_path=path
        )
        name = entry.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "entry name has invalid format",
                source_relative_path=source_relative_path,
                field_path=f"{path}.name",
            )
        elif name in names:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "duplicate enabled skill name",
                source_relative_path=source_relative_path,
                field_path=f"{path}.name",
            )
        else:
            names.add(name)
        if not isinstance(entry.get("implementation"), str) or not entry.get("implementation", "").strip():
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "implementation must be a non-empty string",
                source_relative_path=source_relative_path,
                field_path=f"{path}.implementation",
            )
        if not isinstance(entry.get("planner_visible"), bool):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "planner_visible must be boolean",
                source_relative_path=source_relative_path,
                field_path=f"{path}.planner_visible",
            )
    return diagnostics


def validate_manifest(
    manifest: Mapping[str, Any], *, package_name: str, source_relative_path: str
) -> list[SkillDiagnostic]:
    diagnostics: list[SkillDiagnostic] = []
    _exact_fields(manifest, MANIFEST_FIELDS, diagnostics, source_relative_path=source_relative_path)
    manifest_version = manifest.get("schema_version")
    if (
        not isinstance(manifest_version, int)
        or isinstance(manifest_version, bool)
        or manifest_version not in SOURCE_SCHEMA_VERSIONS
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "schema_version must be 1 or 2",
            source_relative_path=source_relative_path,
            field_path="schema_version",
        )
    if (
        manifest.get("name") != package_name
        or not isinstance(manifest.get("name"), str)
        or not NAME_PATTERN.fullmatch(str(manifest.get("name", "")))
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "manifest name must match the package directory",
            source_relative_path=source_relative_path,
            field_path="name",
        )
    if not isinstance(manifest.get("version"), str) or not SEMVER_PATTERN.fullmatch(str(manifest.get("version", ""))):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "version must use the v1 SemVer grammar",
            source_relative_path=source_relative_path,
            field_path="version",
        )
    if manifest.get("semantic_level") not in SEMANTIC_LEVELS:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "semantic_level is invalid",
            source_relative_path=source_relative_path,
            field_path="semantic_level",
        )
    description = manifest.get("description")
    if not isinstance(description, Mapping):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "description must be an object",
            source_relative_path=source_relative_path,
            field_path="description",
        )
    else:
        _validate_description(description, diagnostics, source_relative_path)
    description_variants = manifest.get("description_variants", {})
    if not isinstance(description_variants, Mapping):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "description_variants must be an object",
            source_relative_path=source_relative_path,
            field_path="description_variants",
        )
    else:
        for variant_name, variant_description in description_variants.items():
            if (
                not isinstance(variant_name, str)
                or not variant_name.strip()
                or not isinstance(variant_description, Mapping)
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "description variant must have a non-empty name and object value",
                    source_relative_path=source_relative_path,
                    field_path="description_variants",
                )
                continue
            _validate_description(
                variant_description,
                diagnostics,
                source_relative_path,
                field_prefix=f"description_variants.{variant_name}",
            )
    capability = manifest.get("capability")
    if not isinstance(capability, Mapping):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "capability must be an object",
            source_relative_path=source_relative_path,
            field_path="capability",
        )
    else:
        _validate_capability(capability, diagnostics, source_relative_path, schema_version=manifest_version)
    implementations = manifest.get("implementations")
    if not isinstance(implementations, Mapping) or not implementations:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "implementations must be a non-empty mapping",
            source_relative_path=source_relative_path,
            field_path="implementations",
        )
    elif any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str)
        for key, value in implementations.items()
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "implementation paths must be non-empty strings",
            source_relative_path=source_relative_path,
            field_path="implementations",
        )
    return diagnostics


def validate_implementation(
    implementation: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    implementation_name: str,
    context: SkillRobotContext,
    delegated_executors: Mapping[str, DelegatedExecutorDescriptor],
    primitive_contracts: Mapping[str, Any],
    source_relative_path: str,
) -> tuple[list[SkillDiagnostic], dict[str, Any], frozenset[str]]:
    diagnostics: list[SkillDiagnostic] = []
    normalized = dict(implementation)
    source_version = manifest.get("schema_version")
    kind = implementation.get("kind")
    if kind == "primitive_sequence":
        allowed = {"schema_version", "kind", "robot", "initial_gripper_state", "timeout_sec", "primitive_sequence"}
        _exact_fields(implementation, allowed, diagnostics, source_relative_path=source_relative_path)
        if implementation.get("schema_version") != source_version:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "implementation schema_version must match manifest schema_version",
                source_relative_path=source_relative_path,
                field_path="schema_version",
            )
        if implementation.get("robot") != implementation_name:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "robot must match implementation variant",
                source_relative_path=source_relative_path,
                field_path="robot",
            )
        if implementation.get("initial_gripper_state") not in INITIAL_GRIPPER_STATES:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "initial_gripper_state is invalid",
                source_relative_path=source_relative_path,
                field_path="initial_gripper_state",
            )
        timeout = implementation.get("timeout_sec")
        if not _positive_finite(timeout):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "timeout_sec must be finite and positive",
                source_relative_path=source_relative_path,
                field_path="timeout_sec",
            )
        steps = implementation.get("primitive_sequence")
        if not isinstance(steps, list) or not steps:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "primitive_sequence must be non-empty",
                source_relative_path=source_relative_path,
                field_path="primitive_sequence",
            )
            return diagnostics, normalized, frozenset()
        requirements: set[str] = set()
        normalized_steps: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            step_path = f"primitive_sequence[{index}]"
            if not isinstance(step, Mapping):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "primitive step must be an object",
                    source_relative_path=source_relative_path,
                    field_path=step_path,
                )
                continue
            primitive_name = step.get("primitive_name")
            descriptor = primitive_contracts.get(primitive_name) if isinstance(primitive_name, str) else None
            if descriptor is None:
                _error(
                    diagnostics,
                    "SKILL_REFERENCE_MISSING",
                    "unsupported primitive",
                    source_relative_path=source_relative_path,
                    field_path=f"{step_path}.primitive_name",
                )
                continue
            if descriptor.schema_version != source_version:
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "primitive descriptor schema_version must match source schema_version",
                    source_relative_path=source_relative_path,
                    field_path=f"{step_path}.primitive_name",
                )
            endpoint_role = descriptor.dispatch_kind
            if endpoint_role not in context.execution_endpoints:
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    f"primitive dispatch requires execution endpoint role '{endpoint_role}'",
                    source_relative_path=source_relative_path,
                    field_path=f"{step_path}.primitive_name",
                )
            dispatch_capabilities = DISPATCH_KIND_CAPABILITY_BY_CONTEXT_VERSION.get(context.context_schema_version, {})
            implied_capability = dispatch_capabilities.get(endpoint_role)
            if implied_capability is None or implied_capability not in descriptor.required_runtime_capabilities:
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "primitive dispatch capability contract is invalid",
                    source_relative_path=source_relative_path,
                    field_path=f"{step_path}.primitive_name",
                )
            if primitive_name == "move_to_pose":
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "move_to_pose is internal and cannot be catalog source",
                    source_relative_path=source_relative_path,
                    field_path=f"{step_path}.primitive_name",
                )
            _validate_primitive_step(step, descriptor, diagnostics, source_relative_path, step_path)
            requirements.update(descriptor.required_runtime_capabilities)
            normalized_step = dict(step)
            if "trajectory_template" in normalized_step:
                trajectory_template = dict(normalized_step.pop("trajectory_template"))
                try:
                    from embodied_common.trajectory_templates import expand_trajectory_template

                    expanded = expand_trajectory_template(trajectory_template)
                except (TypeError, ValueError, OSError) as exc:
                    _error(
                        diagnostics,
                        "SKILL_SCHEMA_INVALID",
                        str(exc),
                        source_relative_path=source_relative_path,
                        field_path=f"{step_path}.trajectory_template",
                    )
                else:
                    normalized_step["joint_waypoints"] = expanded
                    normalized_step["waypoint_duration_sec"] = float(
                        trajectory_template.get("waypoint_duration_sec", 0.08)
                    )
            normalized_steps.append(normalized_step)
        normalized["primitive_sequence"] = normalized_steps
        if manifest.get("semantic_level") == "atomic_operator" and len(normalized_steps) != 1:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "atomic_operator must contain exactly one primitive",
                source_relative_path=source_relative_path,
                field_path="primitive_sequence",
            )
        if normalized_steps:
            _validate_robot_compatibility(normalized_steps, context, diagnostics, source_relative_path)
        return diagnostics, normalized, frozenset(requirements)
    if kind == "delegated_executor":
        allowed = {"schema_version", "kind", "robot", "executor", "required_args", "timeout_sec"}
        _exact_fields(implementation, allowed, diagnostics, source_relative_path=source_relative_path)
        if implementation.get("schema_version") != source_version:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "implementation schema_version must match manifest schema_version",
                source_relative_path=source_relative_path,
                field_path="schema_version",
            )
        if implementation.get("robot") != implementation_name:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "robot must match implementation variant",
                source_relative_path=source_relative_path,
                field_path="robot",
            )
        executor_name = implementation.get("executor")
        executor = delegated_executors.get(executor_name)
        if executor is None:
            _error(
                diagnostics,
                "SKILL_REFERENCE_MISSING",
                "delegated executor is not registered",
                source_relative_path=source_relative_path,
                field_path="executor",
            )
        else:
            _validate_executor(executor, diagnostics, source_relative_path, "executor")
        required_args = implementation.get("required_args")
        capability_parameters = manifest.get("capability", {}).get("parameters", {}).get("properties", {})
        if (
            not isinstance(required_args, list)
            or not required_args
            or any(not isinstance(arg, str) for arg in required_args)
            or len(set(required_args)) != len(required_args)
        ):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "required_args must be a non-empty unique string list",
                source_relative_path=source_relative_path,
                field_path="required_args",
            )
        else:
            for arg in required_args:
                if arg not in capability_parameters:
                    _error(
                        diagnostics,
                        "SKILL_SCHEMA_INVALID",
                        "required_args must reference capability parameters",
                        source_relative_path=source_relative_path,
                        field_path=f"required_args[{arg}]",
                    )
        if not _positive_finite(implementation.get("timeout_sec")):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "timeout_sec must be finite and positive",
                source_relative_path=source_relative_path,
                field_path="timeout_sec",
            )
        return diagnostics, normalized, frozenset({f"delegated_executor:{executor_name}"})
    _error(
        diagnostics,
        "SKILL_SCHEMA_INVALID",
        "implementation kind must be primitive_sequence or delegated_executor",
        source_relative_path=source_relative_path,
        field_path="kind",
    )
    return diagnostics, normalized, frozenset()


def _validate_description(
    description: Mapping[str, Any],
    diagnostics: list[SkillDiagnostic],
    path: str,
    *,
    field_prefix: str = "description",
) -> None:
    _exact_fields(description, DESCRIPTION_FIELDS, diagnostics, source_relative_path=path, field_path=field_prefix)
    for field_name in ("summary", "category"):
        if not isinstance(description.get(field_name), str) or not description.get(field_name, "").strip():
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "must be a non-empty string",
                source_relative_path=path,
                field_path=f"{field_prefix}.{field_name}",
            )
    when_to_use = description.get("when_to_use")
    if (
        not isinstance(when_to_use, list)
        or not when_to_use
        or any(not isinstance(item, str) or not item.strip() for item in when_to_use)
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "when_to_use must be a non-empty string list",
            source_relative_path=path,
            field_path=f"{field_prefix}.when_to_use",
        )
    motion_scope = description.get("motion_scope")
    if (
        not isinstance(motion_scope, list)
        or not motion_scope
        or len(set(motion_scope)) != len(motion_scope)
        or any(item not in MOTION_SCOPES for item in motion_scope)
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "motion_scope contains invalid or duplicate values",
            source_relative_path=path,
            field_path=f"{field_prefix}.motion_scope",
        )
    if description.get("intensity") not in {"subtle", "moderate", "large"}:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "intensity is invalid",
            source_relative_path=path,
            field_path=f"{field_prefix}.intensity",
        )
    for alias_key in ("aliases_zh", "aliases_en"):
        aliases = description.get(alias_key, [])
        if (
            not isinstance(aliases, list)
            or len(set(aliases)) != len(aliases)
            or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
        ):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "aliases must be a unique string list",
                source_relative_path=path,
                field_path=f"{field_prefix}.{alias_key}",
            )
    if "duration_sec_estimate" in description and not _positive_finite(description["duration_sec_estimate"]):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "duration_sec_estimate must be finite and positive",
            source_relative_path=path,
            field_path=f"{field_prefix}.duration_sec_estimate",
        )
    for field_name in ("requires_motion_params", "rule_entry"):
        if field_name in description and not isinstance(description[field_name], bool):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "must be boolean",
                source_relative_path=path,
                field_path=f"{field_prefix}.{field_name}",
            )
    do_not_use = description.get("do_not_use", [])
    if not isinstance(do_not_use, list):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "do_not_use must be a list",
            source_relative_path=path,
            field_path=f"{field_prefix}.do_not_use",
        )
    else:
        for index, entry in enumerate(do_not_use):
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"condition", "instead_use"}
                or any(not isinstance(value, str) or not value.strip() for value in entry.values())
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "do_not_use entry must contain condition and instead_use",
                    source_relative_path=path,
                    field_path=f"{field_prefix}.do_not_use[{index}]",
                )


def _validate_capability(
    capability: Mapping[str, Any],
    diagnostics: list[SkillDiagnostic],
    path: str,
    *,
    schema_version: int | None,
) -> None:
    _exact_fields(capability, CAPABILITY_FIELDS, diagnostics, source_relative_path=path, field_path="capability")
    if capability.get("schema_version") != schema_version:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "capability schema_version must match manifest schema_version",
            source_relative_path=path,
            field_path="capability.schema_version",
        )
    for field_name in ("summary", "domain"):
        if not isinstance(capability.get(field_name), str) or not capability.get(field_name, "").strip():
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "must be a non-empty string",
                source_relative_path=path,
                field_path=f"capability.{field_name}",
            )
    if not isinstance(capability.get("moves_robot"), bool):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "moves_robot must be boolean",
            source_relative_path=path,
            field_path="capability.moves_robot",
        )
    allowed_control_modes = CONTROL_MODES_BY_SCHEMA_VERSION.get(schema_version, frozenset())
    if capability.get("required_control_mode") not in allowed_control_modes:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "required_control_mode is invalid",
            source_relative_path=path,
            field_path="capability.required_control_mode",
        )
    allowed_parameter_names = CAPABILITY_PARAMETER_NAMES_BY_SCHEMA_VERSION.get(schema_version, frozenset())
    _validate_parameter_schema(
        capability.get("parameters"),
        diagnostics,
        path,
        "capability.parameters",
        allowed_parameter_names=allowed_parameter_names,
    )
    _validate_required_capabilities(capability, diagnostics, path)
    if capability.get("recovery_policy") not in RECOVERY_POLICIES:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "recovery_policy is invalid",
            source_relative_path=path,
            field_path="capability.recovery_policy",
        )


def _validate_required_capabilities(
    capability: Mapping[str, Any],
    diagnostics: list[SkillDiagnostic],
    path: str,
) -> None:
    """Validate format; RuntimeStatus supplies the runtime capability vocabulary."""
    if "required_capabilities" not in capability:
        return
    required = capability["required_capabilities"]
    if not isinstance(required, list) or not required:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "required_capabilities must be a non-empty list when present",
            source_relative_path=path,
            field_path="capability.required_capabilities",
        )
        return
    for item in required:
        if not isinstance(item, str) or not item.strip():
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "required_capabilities entries must be non-empty strings",
                source_relative_path=path,
                field_path="capability.required_capabilities",
            )
            return


def _validate_parameter_schema(
    schema: Any,
    diagnostics: list[SkillDiagnostic],
    path: str,
    field_path: str,
    *,
    allowed_parameter_names: frozenset[str],
) -> None:
    if not isinstance(schema, Mapping):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "parameters must be an object",
            source_relative_path=path,
            field_path=field_path,
        )
        return
    allowed_root = {"type", "properties", "required", "additionalProperties"}
    if set(schema) - allowed_root or schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "parameters must use the closed object schema",
            source_relative_path=path,
            field_path=field_path,
        )
        return
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        not isinstance(properties, Mapping)
        or not isinstance(required, list)
        or len(set(required)) != len(required)
        or any(name not in properties for name in required)
        or set(properties) - allowed_parameter_names
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "parameter properties or required list is invalid",
            source_relative_path=path,
            field_path=field_path,
        )
        return
    for name, property_schema in properties.items():
        if not isinstance(property_schema, Mapping):
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "parameter property must be an object",
                source_relative_path=path,
                field_path=f"{field_path}.properties.{name}",
            )
            continue
        positive_numeric_units = {
            "motion_distance": {"meters", "degrees"},
            "imitation_duration_sec": {"seconds"},
            "distance": {"meters"},
            "degree": {"degrees"},
            "stand_off_distance_m": {"meters"},
        }
        signed_numeric_units = {"x": "meters", "y": "meters", "yaw": "degrees"}
        if name in positive_numeric_units:
            if (
                set(property_schema) != {"type", "exclusiveMinimum", "unit"}
                or property_schema.get("type") != "number"
                or property_schema.get("exclusiveMinimum") != 0
                or property_schema.get("unit") not in positive_numeric_units[name]
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    f"{name} schema is invalid",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )
        elif name in signed_numeric_units:
            if (
                set(property_schema) != {"type", "unit"}
                or property_schema.get("type") != "number"
                or property_schema.get("unit") != signed_numeric_units[name]
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    f"{name} schema is invalid",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )
        else:
            allowed = {"type", "enum", "freeform"}
            if set(property_schema) - allowed or property_schema.get("type") != "string":
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "string parameter schema is invalid",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )
            if property_schema.get("freeform") is True and name not in {"target_name", "container_name"}:
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "only target_name and container_name may be freeform",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )
            if property_schema.get("freeform") is not True and (
                not isinstance(property_schema.get("enum"), list) or not property_schema["enum"]
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "non-freeform strings need a non-empty enum",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )
            if name == "motion_direction" and any(
                item not in {"forward", "backward", "left", "right", "up", "down"}
                for item in property_schema.get("enum", [])
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "motion_direction enum is invalid",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )
            if name == "direction" and any(
                item not in {"forward", "backward", "left", "right"} for item in property_schema.get("enum", [])
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "direction enum is invalid",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )
            if name == "arm_side" and any(
                item not in {"left", "right", "auto"} for item in property_schema.get("enum", [])
            ):
                _error(
                    diagnostics,
                    "SKILL_SCHEMA_INVALID",
                    "arm_side enum is invalid",
                    source_relative_path=path,
                    field_path=f"{field_path}.properties.{name}",
                )


def _validate_primitive_step(
    step: Mapping[str, Any], descriptor: Any, diagnostics: list[SkillDiagnostic], path: str, field_path: str
) -> None:
    schema = _json_value(descriptor.parameter_contract)
    validator = Draft202012Validator(schema)
    for error in validator.iter_errors(dict(step)):
        _error(diagnostics, "SKILL_SCHEMA_INVALID", error.message, source_relative_path=path, field_path=field_path)
    primitive_name = step.get("primitive_name")
    if (
        primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}
        and "motion_distance" in step
        and "motion_distance_from_request" in step
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "motion distance must be specified once",
            source_relative_path=path,
            field_path=field_path,
        )


def _validate_robot_compatibility(
    steps: Sequence[Mapping[str, Any]], context: SkillRobotContext, diagnostics: list[SkillDiagnostic], path: str
) -> None:
    for index, step in enumerate(steps):
        primitive_name = step.get("primitive_name")
        if primitive_name == "move_to_named_pose":
            pose_name = step.get("pose_name")
            if isinstance(pose_name, str) and pose_name not in context.named_poses:
                _error(
                    diagnostics,
                    "SKILL_REFERENCE_MISSING",
                    "named pose does not exist",
                    source_relative_path=path,
                    field_path=f"primitive_sequence[{index}].pose_name",
                )
        if primitive_name in {"move_to_joint_positions", "move_to_configuration"}:
            values = step.get("joint_positions") or step.get("joint_position_offsets")
            if isinstance(values, Mapping):
                _validate_joint_mapping(values, context, diagnostics, path, f"primitive_sequence[{index}]")
        if primitive_name == "move_through_joint_positions":
            for waypoint_index, waypoint in enumerate(step.get("joint_waypoints", [])):
                if isinstance(waypoint, Mapping):
                    _validate_joint_mapping(
                        waypoint.get("joint_positions", {}),
                        context,
                        diagnostics,
                        path,
                        f"primitive_sequence[{index}].joint_waypoints[{waypoint_index}]",
                    )


def _validate_joint_mapping(
    values: Mapping[str, Any],
    context: SkillRobotContext,
    diagnostics: list[SkillDiagnostic],
    path: str,
    field_path: str,
) -> None:
    if set(values) != set(context.arm_joint_names):
        _error(
            diagnostics,
            "SKILL_LIMIT_VIOLATION",
            "joint mapping must cover exactly the robot arm joints",
            source_relative_path=path,
            field_path=field_path,
        )
    for joint_name, value in values.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
            _error(
                diagnostics,
                "SKILL_LIMIT_VIOLATION",
                "joint value must be finite",
                source_relative_path=path,
                field_path=f"{field_path}.{joint_name}",
            )
            continue
        limits = context.joint_limits.get(joint_name, {})
        lower = limits.get("lower")
        upper = limits.get("upper")
        if (
            isinstance(lower, int | float)
            and float(value) < float(lower)
            or isinstance(upper, int | float)
            and float(value) > float(upper)
        ):
            _error(
                diagnostics,
                "SKILL_LIMIT_VIOLATION",
                "joint value exceeds robot limit",
                source_relative_path=path,
                field_path=f"{field_path}.{joint_name}",
            )


def _validate_executor(
    executor: DelegatedExecutorDescriptor, diagnostics: list[SkillDiagnostic], path: str, field_path: str
) -> None:
    if executor.endpoint_kind not in DELEGATED_ENDPOINT_KINDS:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "executor endpoint_kind is invalid",
            source_relative_path=path,
            field_path=field_path,
        )
    if (
        not executor.endpoint_name.strip()
        or not executor.contract_version.strip()
        or not re.fullmatch(r"[0-9a-f]{64}", executor.configuration_digest)
    ):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "executor identity is incomplete",
            source_relative_path=path,
            field_path=field_path,
        )
    model_fields = (executor.model_deployment_name, executor.model_fingerprint, executor.model_bundle_digest)
    if any(model_fields) and not all(model_fields):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "model identity fields must be all empty or all present",
            source_relative_path=path,
            field_path=field_path,
        )
    if executor.name == "grasp_pipeline" and not all(model_fields):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "grasp_pipeline requires inference manifest model identity",
            source_relative_path=path,
            field_path=field_path,
        )


def validate_robot_context(context: SkillRobotContext) -> list[SkillDiagnostic]:
    diagnostics: list[SkillDiagnostic] = []
    endpoint_roles = EXECUTION_ENDPOINT_ROLES_BY_CONTEXT_VERSION.get(context.context_schema_version)
    if endpoint_roles is None:
        _error(
            diagnostics, "SKILL_SCHEMA_INVALID", "unsupported robot context schema", field_path="context_schema_version"
        )
    allowed_control_modes = CONTROL_MODES_BY_SCHEMA_VERSION.get(context.context_schema_version, frozenset())
    if context.required_control_mode not in allowed_control_modes:
        _error(diagnostics, "SKILL_SCHEMA_INVALID", "invalid required control mode", field_path="required_control_mode")
    supported_control_modes = tuple(getattr(context, "supported_control_modes", ()))
    if context.context_schema_version < 3 and supported_control_modes:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "supported_control_modes requires robot context schema v3",
            field_path="supported_control_modes",
        )
    if context.context_schema_version == 3 and len(set(supported_control_modes)) < 2:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "hybrid robot context requires at least two supported_control_modes",
            field_path="supported_control_modes",
        )
    if supported_control_modes:
        unsupported = sorted(set(supported_control_modes) - allowed_control_modes)
        if unsupported:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                f"unsupported control mode(s): {unsupported}",
                field_path="supported_control_modes",
            )
        if context.required_control_mode not in supported_control_modes:
            _error(
                diagnostics,
                "SKILL_SCHEMA_INVALID",
                "required_control_mode must be one of supported_control_modes",
                field_path="required_control_mode",
            )
    if endpoint_roles is not None and set(context.execution_endpoints) != endpoint_roles:
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            f"execution endpoint roles must match the v{context.context_schema_version} closed set",
            field_path="execution_endpoints",
        )
    return diagnostics


def _exact_fields(
    value: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    diagnostics: list[SkillDiagnostic],
    *,
    source_relative_path: str,
    field_path: str = "",
) -> None:
    for key in sorted(set(value) - set(allowed)):
        _error(
            diagnostics,
            "SKILL_SCHEMA_INVALID",
            "unknown field",
            source_relative_path=source_relative_path,
            field_path=f"{field_path}.{key}".strip("."),
        )


def _error(
    diagnostics: list[SkillDiagnostic], code: str, message: str, *, source_relative_path: str = "", field_path: str = ""
) -> None:
    diagnostics.append(
        SkillDiagnostic.error(code, message, source_relative_path=source_relative_path, field_path=field_path)
    )


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value
