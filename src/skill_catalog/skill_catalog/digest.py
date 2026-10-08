"""Canonical JSON and digest helpers (section 6.4, 5.1, 13).

Every process (compiler, runtime, CLI, safety) MUST use these helpers so that
the same frozen data always produces byte-identical preimages and digests.
The rules below mirror ``docs/lightweight_skill_package_registry_design_zh.md``
section 6.4 exactly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from embodied_common.canon import (
    SCHEMA_VERSION,  # noqa: F401  re-exported for backward compatibility
    deep_freeze,  # noqa: F401  re-exported for backward compatibility
    sha256_bytes,  # noqa: F401  re-exported for backward compatibility
    sha256_text,
    to_canonical_json,
)

# --------------------------------------------------------------------------- #
# Canonical JSON and deep-freeze helpers                                       #
# --------------------------------------------------------------------------- #
# ``to_canonical_json``, ``deep_freeze``, ``sha256_text`` and ``sha256_bytes``
# are re-exported from ``embodied_common.canon`` (see imports above). That
# module is the single source of truth for canonical serialization so that
# ``robot_config``, ``embodied_common`` and ``skill_catalog`` never diverge.


# --------------------------------------------------------------------------- #
# Registry / capability / provenance preimages (section 6.4)                  #
# --------------------------------------------------------------------------- #

REGISTRY_PREIMAGE_SCHEMA_VERSION = 1
CAPABILITY_PREIMAGE_SCHEMA_VERSION = 1
PROVENANCE_PREIMAGE_SCHEMA_VERSION = 1

REGISTRY_PREIMAGE_FIELDS = frozenset(
    {
        "schema_version",
        "robot_name",
        "profile_name",
        "primitive_contract_digest",
        "robot_context",
        "delegated_executors",
        "skills",
        "aliases",
        "parameter_schemas",
        "requirements",
        "enabled_skill_names",
        "planner_visible_skill_names",
    }
)
CAPABILITY_PREIMAGE_FIELDS = frozenset(
    {
        "schema_version",
        "robot_name",
        "profile_name",
        "capability_view",
        "enabled_skill_names",
        "planner_visible_skill_names",
        "named_pose_names",
        "timeout_policy",
    }
)
PROVENANCE_PREIMAGE_FIELDS = frozenset({"schema_version", "source_release_digest", "skill_package_digests"})


def _robot_context_to_preimage(robot_context: Any) -> dict[str, Any]:
    preimage = {
        "context_schema_version": _attr(robot_context, "context_schema_version"),
        "robot_config_digest": _attr(robot_context, "robot_config_digest"),
        "named_poses": _as_mapping(_attr(robot_context, "named_poses")),
        "named_targets": _as_mapping(_attr(robot_context, "named_targets")),
        "arm_joint_names": list(_as_tuple(_attr(robot_context, "arm_joint_names"))),
        "joint_limits": _as_mapping(_attr(robot_context, "joint_limits")),
        "workspace_limits": _as_mapping(_attr(robot_context, "workspace_limits")),
        "required_control_mode": _attr(robot_context, "required_control_mode"),
        "timeout_policy": _as_mapping(_attr(robot_context, "timeout_policy")),
        "relative_motion_reference_frame": _attr(robot_context, "relative_motion_reference_frame"),
        "relative_motion_step_m": _attr(robot_context, "relative_motion_step_m"),
        "relative_motion_direction_mapping": _as_mapping(_attr(robot_context, "relative_motion_direction_mapping")),
        "gripper_open_position": _attr(robot_context, "gripper_open_position"),
        "gripper_closed_position": _attr(robot_context, "gripper_closed_position"),
        "execution_endpoints": _as_mapping(_attr(robot_context, "execution_endpoints")),
    }
    if preimage["context_schema_version"] >= 3:
        preimage["supported_control_modes"] = list(
            _as_tuple(_attr(robot_context, "supported_control_modes", default=()))
        )
    return preimage


def _delegated_executor_to_preimage(executor: Any) -> dict[str, Any]:
    return {
        "name": _attr(executor, "name"),
        "contract_version": _attr(executor, "contract_version"),
        "endpoint_kind": _attr(executor, "endpoint_kind"),
        "endpoint_name": _attr(executor, "endpoint_name"),
        "configuration_digest": _attr(executor, "configuration_digest"),
        "model_deployment_name": _attr(executor, "model_deployment_name"),
        "model_fingerprint": _attr(executor, "model_fingerprint"),
        "model_bundle_digest": _attr(executor, "model_bundle_digest"),
    }


def _skill_to_registry_entry(
    name: str,
    template: Any,
    *,
    semantic_levels: Mapping[str, str],
    capability_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a ``registry_preimage_v1.skills[]`` entry (section 6.4).

    ``template`` is the per-skill compiled body stored in the snapshot. It may
    carry ``version`` and ``implementation_identity`` metadata keys alongside
    the execution body (``primitive_sequence`` etc.). The preimage keeps
    ``version``/``implementation_identity`` as separate top-level skill fields
    and emits the clean execution body as ``template``.
    """

    body = _as_mapping(template)
    version = body.get("version", "0.0.0")
    implementation_identity = body.get("implementation_identity", "")
    clean_body = {key: value for key, value in body.items() if key not in ("version", "implementation_identity")}
    # The public capability is duplicated inside the registry-bound template so
    # consumers can deterministically reconstruct capability_preimage_v1. This
    # prevents a valid capability digest from being paired with an unrelated
    # execution registry.
    clean_body["capability"] = _as_mapping(capability_view[name])
    return {
        "name": name,
        "version": version,
        "semantic_level": semantic_levels.get(name, ""),
        "implementation_identity": implementation_identity,
        "template": clean_body,
    }


def derive_registry_preimage(
    *,
    robot_name: str,
    profile_name: str,
    primitive_contract_digest: str,
    robot_context: Any,
    delegated_executors: Mapping[str, Any],
    templates: Mapping[str, Mapping[str, Any]],
    semantic_levels: Mapping[str, str],
    aliases: Mapping[str, Any],
    parameter_schemas: Mapping[str, Any],
    requirements: Mapping[str, Any],
    capability_view: Mapping[str, Any],
    enabled_skill_names: Iterable[str],
    planner_visible_skill_names: Iterable[str],
) -> dict[str, Any]:
    """Build ``registry_preimage_v1`` (section 6.4).

    Top-level fields are the precise closed set; callers cannot add fields.
    """

    skills = [
        _skill_to_registry_entry(
            name,
            templates[name],
            semantic_levels=semantic_levels,
            capability_view=capability_view,
        )
        for name in sorted(templates)
    ]
    executors = [_delegated_executor_to_preimage(delegated_executors[name]) for name in sorted(delegated_executors)]
    return {
        "schema_version": REGISTRY_PREIMAGE_SCHEMA_VERSION,
        "robot_name": robot_name,
        "profile_name": profile_name,
        "primitive_contract_digest": primitive_contract_digest,
        "robot_context": _robot_context_to_preimage(robot_context),
        "delegated_executors": executors,
        "skills": skills,
        "aliases": {name: list(_as_tuple(aliases[name])) for name in sorted(aliases)},
        "parameter_schemas": {name: _as_mapping(parameter_schemas[name]) for name in sorted(parameter_schemas)},
        "requirements": {name: sorted(_as_tuple(requirements[name])) for name in sorted(requirements)},
        "enabled_skill_names": sorted(enabled_skill_names),
        "planner_visible_skill_names": sorted(planner_visible_skill_names),
    }


def derive_capability_view_from_registry(registry_preimage: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the public view using only registry-bound normalized entries."""

    skills = registry_preimage.get("skills")
    schemas = registry_preimage.get("parameter_schemas")
    if not isinstance(skills, list) or not isinstance(schemas, Mapping):
        raise ValueError("registry skills and parameter_schemas are required")
    enabled = set(registry_preimage.get("enabled_skill_names", ()))
    visible = set(registry_preimage.get("planner_visible_skill_names", ()))
    result: dict[str, Any] = {}
    for entry in skills:
        if not isinstance(entry, Mapping):
            raise ValueError("registry skill entry must be an object")
        name = entry.get("name")
        template = entry.get("template")
        if not isinstance(name, str) or not name or name in result or not isinstance(template, Mapping):
            raise ValueError("registry skill entry is invalid")
        capability = template.get("capability")
        if not isinstance(capability, Mapping):
            raise ValueError("registry skill capability is missing")
        capability_version = capability.get("schema_version", 1)
        if "schema_version" in capability and capability_version != 2:
            raise ValueError("registry skill capability fields are invalid")
        expected_fields = {
            "name",
            "summary",
            "domain",
            "semantic_level",
            "planner_visible",
            "moves_robot",
            "required_control_mode",
            "parameters",
            "recovery_policy",
        }
        if capability_version == 2:
            expected_fields.add("schema_version")
        # Optional, and emitted by the compiler only when the manifest declares it.
        if "required_capabilities" in capability:
            required = capability["required_capabilities"]
            if (
                not isinstance(required, list | tuple)
                or not required
                or any(not isinstance(item, str) or not item.strip() for item in required)
            ):
                raise ValueError("registry skill capability fields are invalid")
            expected_fields.add("required_capabilities")
        if set(capability) != expected_fields:
            raise ValueError("registry skill capability fields are invalid")
        rebuilt = dict(capability)
        rebuilt["name"] = name
        rebuilt["semantic_level"] = entry.get("semantic_level")
        rebuilt["planner_visible"] = name in visible
        rebuilt["parameters"] = _as_mapping(schemas.get(name, {}))
        if name not in enabled:
            raise ValueError("registry contains a non-enabled skill entry")
        result[name] = rebuilt
    if set(result) != enabled or not visible <= enabled:
        raise ValueError("registry enabled or planner-visible set is inconsistent")
    return {name: result[name] for name in sorted(result)}


def derive_registry_digest(registry_preimage: Mapping[str, Any]) -> str:
    return sha256_text(to_canonical_json(registry_preimage))


def derive_capability_preimage(
    *,
    robot_name: str,
    profile_name: str,
    capability_view: Mapping[str, Any],
    enabled_skill_names: Iterable[str],
    planner_visible_skill_names: Iterable[str],
    named_pose_names: Iterable[str],
    timeout_policy: Mapping[str, float],
) -> dict[str, Any]:
    """Build ``capability_preimage_v1`` (section 6.4)."""

    return {
        "schema_version": CAPABILITY_PREIMAGE_SCHEMA_VERSION,
        "robot_name": robot_name,
        "profile_name": profile_name,
        "capability_view": {name: _as_mapping(capability_view[name]) for name in sorted(capability_view)},
        "enabled_skill_names": sorted(enabled_skill_names),
        "planner_visible_skill_names": sorted(planner_visible_skill_names),
        "named_pose_names": sorted(named_pose_names),
        "timeout_policy": {name: timeout_policy[name] for name in sorted(timeout_policy)},
    }


def derive_capability_digest(capability_preimage: Mapping[str, Any]) -> str:
    return sha256_text(to_canonical_json(capability_preimage))


def derive_provenance_preimage(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Build ``provenance_preimage_v1`` (section 6.4).

    ``provenance`` must already match the v1 structure
    (``source_release_digest`` + ``skill_package_digests``).
    """

    skill_package_digests = _as_mapping(provenance.get("skill_package_digests", {}))
    return {
        "schema_version": PROVENANCE_PREIMAGE_SCHEMA_VERSION,
        "source_release_digest": provenance.get("source_release_digest", ""),
        "skill_package_digests": {name: skill_package_digests[name] for name in sorted(skill_package_digests)},
    }


def derive_provenance_digest(provenance_preimage: Mapping[str, Any]) -> str:
    return sha256_text(to_canonical_json(provenance_preimage))


# --------------------------------------------------------------------------- #
# Source release / package digests (section 13)                              #
# --------------------------------------------------------------------------- #


def compute_file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_manifest(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the release manifest preimage (section 13).

    ``files`` entries must be ``{"path": <posix relative>, "size": int, "sha256": str}``.
    The manifest is sorted by path for determinism.
    """

    ordered = sorted(files, key=lambda entry: entry["path"])
    return {"schema_version": SCHEMA_VERSION, "files": ordered}


def compute_release_digest_from_manifest(files: list[dict[str, Any]]) -> str:
    return sha256_text(to_canonical_json(build_release_manifest(files)))


def compute_skill_package_digest(files: list[dict[str, Any]]) -> str:
    """Per-skill package digest (section 13) using the same file-manifest rule."""

    return compute_release_digest_from_manifest(files)


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #


def _attr(obj: Any, name: str, *, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"expected mapping, got {type(value).__name__}")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple | list | set | frozenset):
        return tuple(value)
    raise TypeError(f"expected sequence, got {type(value).__name__}")
