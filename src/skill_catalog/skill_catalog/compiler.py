"""Deterministic four-pass compiler for immutable skill catalog snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from embodied_common.primitive_contracts import (
    primitive_contract_for_version,
)
from skill_catalog.digest import compute_skill_package_digest
from skill_catalog.models import (
    SkillCatalogError,
    SkillCompileContext,
    SkillCompileError,
    SkillDiagnostic,
    SkillSnapshot,
    sort_diagnostics,
)
from skill_catalog.source import (
    SkillPackageLocation,
    SkillSource,
    build_package_file_manifest,
    load_yaml_mapping,
)
from skill_catalog.validator import validate_implementation, validate_manifest, validate_profile, validate_robot_context


class SkillCatalogCompiler:
    def compile(
        self,
        source: SkillSource,
        *,
        profile_name: str,
        context: SkillCompileContext,
    ) -> SkillSnapshot:
        diagnostics = self._validate_compile_context(context)
        release = source.resolve_active_release()
        source_release_digest = source.compute_release_digest(release)
        if release.source_release_digest and source_release_digest != release.source_release_digest:
            diagnostics.append(
                SkillDiagnostic.error(
                    "SKILL_SOURCE_CHANGED_DURING_COMPILE",
                    "source release digest changed before compilation",
                )
            )

        profile_path = f"config/profiles/{profile_name}.yaml"
        profile = source.load_profile(release, profile_name)
        diagnostics.extend(
            validate_profile(
                profile,
                profile_name=profile_name,
                robot_name=context.robot.robot_name,
                source_relative_path=profile_path,
            )
        )

        packages = source.discover_packages(release)
        package_index: dict[str, SkillPackageLocation] = {}
        package_digests: dict[str, str] = {}
        discovered_manifests: dict[str, Mapping[str, Any]] = {}
        for package in packages:
            if package.name in package_index:
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_SCHEMA_INVALID",
                        "duplicate skill package name",
                        source_relative_path=package.source_relative_path,
                    )
                )
            package_index[package.name] = package
            package_digests[package.name] = compute_skill_package_digest(
                build_package_file_manifest(package.package_dir)
            )

            if not package.manifest_path.is_file():
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_SCHEMA_INVALID",
                        "non-hidden skill package is missing manifest.yaml",
                        source_relative_path=package.source_relative_path,
                    )
                )
                continue
            try:
                manifest = load_yaml_mapping(package.manifest_path)
            except SkillCatalogError as exc:
                diagnostic = exc.diagnostic()
                if not diagnostic.source_relative_path:
                    diagnostic = SkillDiagnostic.error(
                        diagnostic.error_code,
                        diagnostic.message,
                        source_relative_path=_relative_path(package.manifest_path, release.root),
                        field_path=diagnostic.field_path,
                    )
                diagnostics.append(diagnostic)
                continue
            discovered_manifests[package.name] = manifest
            manifest_diagnostics = validate_manifest(
                manifest,
                package_name=package.name,
                source_relative_path=_relative_path(package.manifest_path, release.root),
            )
            diagnostics.extend(manifest_diagnostics)

        profile_entries = profile.get("enabled_skills", []) if isinstance(profile, Mapping) else []
        if not isinstance(profile_entries, list):
            profile_entries = []

        templates: dict[str, Mapping[str, Any]] = {}
        semantic_levels: dict[str, str] = {}
        aliases: dict[str, tuple[str, ...]] = {}
        parameter_schemas: dict[str, Mapping[str, Any]] = {}
        requirements: dict[str, frozenset[str]] = {}
        capability_view: dict[str, dict[str, Any]] = {}
        enabled_names: list[str] = []
        planner_visible_names: list[str] = []
        manifests: dict[str, Mapping[str, Any]] = {}
        alias_owners: dict[tuple[str, str], str] = {}

        for entry_index, entry in enumerate(profile_entries):
            if not isinstance(entry, Mapping):
                continue
            skill_name = entry.get("name")
            implementation_name = entry.get("implementation")
            planner_visible = entry.get("planner_visible")
            if not isinstance(skill_name, str) or not isinstance(implementation_name, str):
                continue
            package = package_index.get(skill_name)
            if package is None:
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_PACKAGE_NOT_FOUND",
                        "enabled skill package does not exist",
                        source_relative_path=profile_path,
                        field_path=f"enabled_skills[{entry_index}].name",
                    )
                )
                continue
            if not package.manifest_path.is_file():
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_SCHEMA_INVALID",
                        "skill package is missing manifest.yaml",
                        source_relative_path=package.source_relative_path,
                    )
                )
                continue
            manifest = discovered_manifests.get(skill_name)
            if manifest is None:
                continue
            manifests[skill_name] = manifest
            manifest_relative_path = _relative_path(package.manifest_path, release.root)
            manifest_diagnostics = [item for item in diagnostics if item.source_relative_path == manifest_relative_path]
            if any(item.severity == 1 for item in manifest_diagnostics):
                continue
            implementation_paths = manifest.get("implementations", {})
            implementation_relative_path = implementation_paths.get(implementation_name)
            if not isinstance(implementation_relative_path, str):
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_REFERENCE_MISSING",
                        "selected implementation does not exist",
                        source_relative_path=profile_path,
                        field_path=f"enabled_skills[{entry_index}].implementation",
                    )
                )
                continue
            try:
                implementation_path = _resolve_package_path(package, implementation_relative_path)
            except ValueError as exc:
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_SCHEMA_INVALID",
                        str(exc),
                        source_relative_path=manifest_relative_path,
                        field_path=f"implementations.{implementation_name}",
                    )
                )
                continue
            if not implementation_path.is_file():
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_REFERENCE_MISSING",
                        "selected implementation file does not exist",
                        source_relative_path=_relative_path(implementation_path, release.root),
                    )
                )
                continue
            implementation = load_yaml_mapping(implementation_path)
            implementation_source_path = _relative_path(implementation_path, release.root)
            implementation_diagnostics, normalized, skill_requirements = validate_implementation(
                implementation,
                manifest=manifest,
                implementation_name=implementation_name,
                context=context.robot,
                delegated_executors=context.delegated_executors,
                primitive_contracts=context.primitive_contracts,
                source_relative_path=implementation_source_path,
            )
            diagnostics.extend(implementation_diagnostics)
            selected_description = manifest.get("description", {})
            description_variants = manifest.get("description_variants", {})
            if isinstance(description_variants, Mapping):
                selected_description = description_variants.get(implementation_name, selected_description)
            self._validate_manifest_references(
                manifest,
                skill_name=skill_name,
                enabled_names={item.get("name") for item in profile_entries if isinstance(item, Mapping)},
                robot_context=context.robot,
                diagnostics=diagnostics,
                source_relative_path=manifest_relative_path,
                description=selected_description,
            )
            self._validate_aliases(
                manifest,
                skill_name=skill_name,
                owners=alias_owners,
                diagnostics=diagnostics,
                source_relative_path=manifest_relative_path,
                description=selected_description,
            )
            if any(
                item.severity == 1 and item.source_relative_path in {manifest_relative_path, implementation_source_path}
                for item in diagnostics
            ):
                continue

            capability = manifest["capability"]
            description = selected_description
            # A unified mobile-manipulator profile may expose skills from more
            # than one mutually-exclusive motion domain.  The capability's
            # required mode is selected at execution time; only validate that
            # it is a mode supported by the robot context.
            supported_control_modes = getattr(context.robot, "supported_control_modes", ())
            mode_mismatch = (
                capability["required_control_mode"] not in supported_control_modes
                if supported_control_modes
                else capability["required_control_mode"] != context.robot.required_control_mode
            )
            if mode_mismatch:
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_LIMIT_VIOLATION",
                        "capability control mode is not supported by robot context",
                        source_relative_path=manifest_relative_path,
                        field_path="capability.required_control_mode",
                    )
                )
                continue
            timeout_cap = float(normalized["timeout_sec"])
            task_budget = context.robot.timeout_policy.get("task_budget_sec")
            if isinstance(task_budget, int | float) and timeout_cap > float(task_budget):
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_LIMIT_VIOLATION",
                        "implementation timeout exceeds robot task budget",
                        source_relative_path=implementation_source_path,
                        field_path="timeout_sec",
                    )
                )
                continue

            implementation_identity = f"{skill_name}@{manifest['version']}#{implementation_name}"
            templates[skill_name] = {
                **normalized,
                "version": manifest["version"],
                "implementation_identity": implementation_identity,
            }
            semantic_levels[skill_name] = manifest["semantic_level"]
            aliases[skill_name] = tuple(
                str(value).strip() for key in ("aliases_zh", "aliases_en") for value in description.get(key, [])
            )
            parameter_schemas[skill_name] = capability["parameters"]
            requirements[skill_name] = skill_requirements
            capability_view[skill_name] = {
                "name": skill_name,
                "summary": capability["summary"],
                "domain": capability["domain"],
                "semantic_level": manifest["semantic_level"],
                "planner_visible": planner_visible,
                "moves_robot": capability["moves_robot"],
                "required_control_mode": capability["required_control_mode"],
                "parameters": capability["parameters"],
                "recovery_policy": capability["recovery_policy"],
            }
            if "required_capabilities" in capability:
                capability_view[skill_name]["required_capabilities"] = list(capability["required_capabilities"])
            if manifest["schema_version"] == 2:
                capability_view[skill_name]["schema_version"] = 2
            enabled_names.append(skill_name)
            if planner_visible is True:
                planner_visible_names.append(skill_name)

        errors = [diagnostic for diagnostic in diagnostics if diagnostic.severity == 1]
        if errors:
            raise SkillCompileError(sort_diagnostics(diagnostics))
        final_source_digest = source.compute_release_digest(release)
        if final_source_digest != source_release_digest:
            raise SkillCompileError(
                (
                    SkillDiagnostic.error(
                        "SKILL_SOURCE_CHANGED_DURING_COMPILE",
                        "source tree changed during compile",
                    ),
                )
            )

        return SkillSnapshot(
            robot_name=context.robot.robot_name,
            profile_name=profile_name,
            primitive_contract_digest=context.primitive_contract_digest,
            robot_context=context.robot,
            delegated_executors=context.delegated_executors,
            templates=templates,
            semantic_levels=semantic_levels,
            aliases=aliases,
            parameter_schemas=parameter_schemas,
            requirements=requirements,
            provenance={
                "schema_version": 1,
                "source_release_digest": source_release_digest,
                "skill_package_digests": package_digests,
            },
            enabled_skill_names=tuple(sorted(enabled_names)),
            planner_visible_skill_names=tuple(sorted(planner_visible_names)),
            capability_view=capability_view,
        )

    @staticmethod
    def _validate_compile_context(context: SkillCompileContext) -> list[SkillDiagnostic]:
        diagnostics = validate_robot_context(context.robot)
        try:
            selected_contract = primitive_contract_for_version(context.robot.context_schema_version)
        except ValueError as exc:
            diagnostics.append(SkillDiagnostic.error("SKILL_SCHEMA_INVALID", str(exc)))
            selected_contract = None
        if selected_contract is not None and context.primitive_contract_digest != selected_contract.digest:
            diagnostics.append(
                SkillDiagnostic.error(
                    "SKILL_SNAPSHOT_DIGEST_MISMATCH", "primitive contract digest does not match local SSOT"
                )
            )
        if selected_contract is not None and (
            set(context.primitive_contracts) != set(selected_contract.descriptors)
            or any(
                context.primitive_contracts.get(name) is not descriptor
                for name, descriptor in selected_contract.descriptors.items()
            )
        ):
            diagnostics.append(
                SkillDiagnostic.error(
                    "SKILL_SNAPSHOT_DIGEST_MISMATCH", "primitive contracts must come from the canonical registry"
                )
            )
        for name, executor in context.delegated_executors.items():
            if executor.name != name:
                diagnostics.append(
                    SkillDiagnostic.error("SKILL_SCHEMA_INVALID", "delegated executor name must match its mapping key")
                )
        return diagnostics

    @staticmethod
    def _validate_manifest_references(
        manifest: Mapping[str, Any],
        *,
        skill_name: str,
        enabled_names: set[Any],
        robot_context: Any,
        diagnostics: list[SkillDiagnostic],
        source_relative_path: str,
        description: Mapping[str, Any] | None = None,
    ) -> None:
        description = description if description is not None else manifest.get("description", {})
        anchor_pose = description.get("anchor_pose")
        if anchor_pose not in (None, "none") and anchor_pose not in robot_context.named_poses:
            diagnostics.append(
                SkillDiagnostic.error(
                    "SKILL_REFERENCE_MISSING",
                    "anchor_pose does not exist in robot context",
                    source_relative_path=source_relative_path,
                    field_path="description.anchor_pose",
                )
            )
        for index, rule in enumerate(description.get("do_not_use", [])):
            instead_use = rule.get("instead_use") if isinstance(rule, Mapping) else None
            if instead_use == skill_name or instead_use not in enabled_names:
                diagnostics.append(
                    SkillDiagnostic.error(
                        "SKILL_REFERENCE_MISSING",
                        "instead_use must reference another enabled catalog entry",
                        source_relative_path=source_relative_path,
                        field_path=f"description.do_not_use[{index}].instead_use",
                    )
                )

    @staticmethod
    def _validate_aliases(
        manifest: Mapping[str, Any],
        *,
        skill_name: str,
        owners: dict[tuple[str, str], str],
        diagnostics: list[SkillDiagnostic],
        source_relative_path: str,
        description: Mapping[str, Any] | None = None,
    ) -> None:
        description = description if description is not None else manifest.get("description", {})
        for language in ("zh", "en"):
            for alias in description.get(f"aliases_{language}", []):
                normalized = alias.strip()
                key = (language, normalized.casefold())
                owner = owners.get(key)
                if owner is not None and owner != skill_name:
                    diagnostics.append(
                        SkillDiagnostic.error(
                            "SKILL_SCHEMA_INVALID",
                            f"alias conflicts with {owner}",
                            source_relative_path=source_relative_path,
                            field_path=f"description.aliases_{language}",
                        )
                    )
                owners[key] = skill_name


def compile_skill_catalog(
    source: SkillSource,
    *,
    profile_name: str,
    context: SkillCompileContext,
) -> SkillSnapshot:
    return SkillCatalogCompiler().compile(source, profile_name=profile_name, context=context)


def _resolve_package_path(package: SkillPackageLocation, relative_path: str) -> Path:
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or str(pure_path) != relative_path:
        raise ValueError("implementation path must be a normalized relative POSIX path")
    candidate = package.package_dir.joinpath(*pure_path.parts)
    package_root = package.package_dir.resolve()
    if candidate.is_symlink() or package_root not in candidate.resolve().parents:
        raise ValueError("implementation path must stay inside its package and cannot be a symlink")
    return candidate


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
