"""Strict schema v3 manifest loading before backend initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from inference_manifest.errors import ManifestIntegrityError, ManifestValidationError
from inference_manifest.integrity import (
    deployment_fingerprint,
    integrity_status_for_deployment,
    runtime_profile_fingerprint,
    verify_bundle_digest,
    verify_deployment_artifacts,
)
from inference_manifest.json_utils import load_json_strict
from inference_manifest.metadata import PolicyFeature, PolicyMetadata, load_policy_metadata
from inference_manifest.models import (
    HOST_SEMANTIC_PREFIX,
    INTERNAL_SEMANTIC_PREFIX,
    Deployment,
    InferenceManifest,
    ModelDescriptor,
    SemanticTensor,
    TensorBinding,
    ValidatedDeployment,
    _runtime_profiles,
)
from inference_manifest.paths import normalize_unique_paths, resolve_bundle_file
from inference_manifest.schema import validate_manifest_schema

MANIFEST_FILENAME = "inference_manifest.json"


# This alias is intentionally source-level only.  The returned object is the
# v3 ValidatedDeployment snapshot and no v2 parser or field mapping remains.
ValidatedManifest = ValidatedDeployment


def load_inference_manifest(
    bundle_root: str | Path,
    deployment_name: str,
    *,
    verify_on_demand: bool = False,
) -> ValidatedDeployment:
    """Validate one bundle and selected deployment without importing a backend SDK."""

    validated = _load_inference_manifest(
        bundle_root,
        deployment_name,
        verify_all_bundle_files=True,
        verify_deployment_artifacts=True,
        require_native_weights=None,
    )
    if verify_on_demand:
        report = verify_deployment_artifacts(
            validated.bundle_root,
            deployment_name,
            mode="verify_on_demand",
            expected_deployment=validated.deployment,
        )
        if report.state == "mismatch":
            raise ManifestIntegrityError(
                f"artifact_digest_mismatch for deployment {deployment_name!r}: {report.mismatch_details}"
            )
        # Keep the snapshot immutable while exposing the explicit verification
        # result to callers that requested it.
        return ValidatedDeployment(
            **{
                **validated.__dict__,
                "integrity_status": report,
            }
        )
    return validated


def load_inference_manifest_metadata(bundle_root: str | Path, deployment_name: str) -> ValidatedDeployment:
    """Validate edge-owned semantics and identity without requiring cloud-only artifacts."""

    return _load_inference_manifest(
        bundle_root,
        deployment_name,
        verify_all_bundle_files=False,
        verify_deployment_artifacts=False,
        require_native_weights=False,
    )


def _load_inference_manifest(
    bundle_root: str | Path,
    deployment_name: str,
    *,
    verify_all_bundle_files: bool,
    verify_deployment_artifacts: bool,
    require_native_weights: bool | None,
) -> ValidatedDeployment:
    root = Path(bundle_root)
    manifest_path = root / MANIFEST_FILENAME
    guidance = (
        "Model startup requires a bundle directory containing inference_manifest.json. "
        "Package the model with the project's exporter/packager and select a named deployment; "
        "unpackaged model files are not supported."
    )
    try:
        has_manifest = bool(str(bundle_root).strip()) and root.is_dir() and manifest_path.is_file()
    except OSError as exc:
        raise ManifestValidationError(f"bundle_manifest_required: {guidance} Path: {manifest_path}. {exc}") from exc
    if not has_manifest:
        raise ManifestValidationError(f"bundle_manifest_required: {guidance} Path: {manifest_path}")
    raw = load_json_strict(manifest_path)
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"Inference manifest must be a JSON object: {manifest_path}")

    # Check the version before schema validation, policy metadata, artifact
    # resolution, or any operation that could import a runtime dependency.
    schema_version = raw.get("schema_version")
    if schema_version != 3:
        raise ManifestValidationError(
            f"unsupported schema_version {schema_version!r} in {manifest_path}; supported versions: [3]"
        )
    validate_manifest_schema(raw, str(manifest_path))
    manifest = _parse_typed_manifest(raw, manifest_path)

    try:
        deployment = manifest.deployments[deployment_name]
    except KeyError as exc:
        raise ManifestValidationError(
            f"Deployment {deployment_name!r} is not present in {manifest_path}; "
            f"available deployments: {sorted(manifest.deployments)}"
        ) from exc

    bundle_entries = tuple(manifest.bundle.files)
    selected_artifacts = tuple(deployment.artifacts.values())
    bundle_paths = set(normalize_unique_paths((entry.path for entry in bundle_entries), "bundle.files"))
    artifact_paths = {artifact.path for artifact in selected_artifacts}
    overlap = sorted(bundle_paths & artifact_paths)
    if overlap:
        raise ManifestValidationError(f"bundle files and deployment artifacts must use distinct paths: {overlap}")

    is_policy = manifest.model.interface == "policy"
    if require_native_weights is None:
        require_native_weights = is_policy and any(
            profile.backend == "torch"
            for candidate in manifest.deployments.values()
            for profile in _runtime_profiles(candidate.runtime_profile, candidate.role_runtime_profiles)
        )

    if verify_all_bundle_files:
        for entry in bundle_entries:
            resolve_bundle_file(root, entry.path)
    verify_bundle_digest(
        manifest.bundle.uuid,
        manifest.bundle.revision,
        manifest.bundle.name,
        bundle_entries,
        manifest.bundle.digest.value,
    )

    policy = load_policy_metadata(root, require_native_weights=require_native_weights) if is_policy else None
    if policy is not None:
        if manifest.model.model_type != policy.policy_type:
            raise ManifestValidationError(
                f"Policy model_type {manifest.model.model_type!r} does not match "
                f"LeRobot config type {policy.policy_type!r}"
            )
        _validate_required_bundle_files(
            manifest,
            policy,
            allow_declared_native_weights=not require_native_weights,
        )

    required_paths = set(policy.required_files) if policy is not None else set()
    entries_to_verify = (
        bundle_entries
        if verify_all_bundle_files
        else tuple(entry for entry in bundle_entries if entry.path in required_paths)
    )
    if not verify_all_bundle_files:
        for entry in entries_to_verify:
            resolve_bundle_file(root, entry.path)

    resolved_artifacts: dict[str, Path] = {}
    if verify_deployment_artifacts:
        for role, artifact in deployment.artifacts.items():
            resolved_artifacts[role] = resolve_bundle_file(root, artifact.path)

    if deployment.execution:
        if policy is not None:
            _validate_policy_deployment(deployment, policy)
        else:
            _validate_semantic_contract(deployment, manifest.model)

    role_identities = dict(deployment.role_identities or {})
    role_runtime_profiles = dict(deployment.role_runtime_profiles or {})
    integrity_status = integrity_status_for_deployment(root, deployment_name, deployment)
    fingerprint = deployment_fingerprint(
        manifest.schema_version,
        manifest.bundle.digest.value,
        deployment_name,
        deployment,
        identity=manifest.model.identity,
        semantic_contract=manifest.model,
        role_identities=role_identities,
        role_runtime_profiles=role_runtime_profiles or None,
    )
    profile_fingerprint = runtime_profile_fingerprint(
        manifest.schema_version,
        manifest.bundle.digest.value,
        deployment_name,
        deployment,
        role_runtime_profiles=role_runtime_profiles or None,
    )
    bundle_root = root.resolve(strict=True)
    resolved_manifest_path = manifest_path.resolve(strict=True)
    return ValidatedDeployment(
        bundle_root=bundle_root,
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        deployment_name=deployment_name,
        deployment=deployment,
        top_level_identity=manifest.model.identity,
        role_identities=role_identities,
        role_runtime_profiles=role_runtime_profiles,
        selected_deployment=deployment,
        semantic_contract=manifest.model,
        resolved_artifacts=resolved_artifacts,
        role_artifact_bindings=dict(deployment.bindings),
        declared_metadata={
            "bundle": manifest.bundle.model_dump(mode="json"),
            "deployment": deployment.metadata,
            "artifacts": {
                role: artifact.model_dump(mode="json", exclude_none=True)
                for role, artifact in deployment.artifacts.items()
            },
        },
        integrity_status=integrity_status,
        deployment_fingerprint=fingerprint,
        runtime_profile_fingerprint=profile_fingerprint,
        policy=policy,
    )


def _parse_typed_manifest(raw: dict[str, Any], manifest_path: Path) -> InferenceManifest:
    content = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    try:
        return InferenceManifest.model_validate_json(content)
    except ValidationError as exc:
        raise ManifestValidationError(f"Typed manifest validation failed for {manifest_path}: {exc}") from exc


def _validate_required_bundle_files(
    manifest: InferenceManifest,
    policy: PolicyMetadata,
    *,
    allow_declared_native_weights: bool = False,
) -> None:
    declared = {entry.path for entry in manifest.bundle.files}
    required = set(policy.required_files)
    missing = sorted(required - declared)
    if missing:
        raise ManifestValidationError(
            f"Manifest bundle.files omits required LeRobot semantic files: {missing}. "
            "Rerun the owning exporter or packaging workflow."
        )

    optional = {"model.safetensors"} if allow_declared_native_weights else set()
    unexpected_semantic = sorted(path for path in declared - required - optional if _is_reserved_semantic_path(path))
    if unexpected_semantic:
        raise ManifestValidationError(
            f"Manifest bundle.files contains unreferenced LeRobot semantic files: {unexpected_semantic}"
        )


def _is_reserved_semantic_path(path: str) -> bool:
    name = Path(path).name
    return (
        name == "model.safetensors"
        or name.startswith("policy_preprocessor_step_")
        or name.startswith("policy_postprocessor_step_")
    )


def _validate_feature_compatibility(deployment: Deployment, policy: PolicyMetadata) -> None:
    for role in deployment.execution:
        group = deployment.bindings[role]
        for binding in (*group.inputs, *group.outputs):
            feature = _policy_feature_for_semantic(binding.semantic, policy)
            if feature is None:
                continue
            if not _shape_matches_feature(binding, feature, policy.policy_type):
                raise ManifestValidationError(
                    f"Binding {binding.semantic!r} for role {role!r} has shape {binding.shape} "
                    f"incompatible with LeRobot feature shape {feature.shape}"
                )


def _validate_policy_deployment(deployment: Deployment, policy: PolicyMetadata) -> None:
    output_semantics = {
        binding.semantic for role in deployment.execution for binding in deployment.bindings[role].outputs
    }
    if "action" not in output_semantics:
        raise ManifestValidationError("Compiled policy deployment must declare an action output binding")
    _validate_feature_compatibility(deployment, policy)


def _validate_semantic_contract(deployment: Deployment, model: ModelDescriptor) -> None:
    if not model.inputs and not model.outputs:
        return

    declared_inputs = {descriptor.semantic: descriptor for descriptor in model.inputs}
    declared_outputs = {descriptor.semantic: descriptor for descriptor in model.outputs}
    bound_inputs = _external_bindings(deployment, "inputs")
    bound_outputs = _external_bindings(deployment, "outputs")
    orchestrated = _is_host_orchestrated(deployment)
    _validate_contract_direction("input", declared_inputs, bound_inputs, host_orchestrated=orchestrated)
    _validate_contract_direction("output", declared_outputs, bound_outputs, host_orchestrated=orchestrated)


def _is_host_orchestrated(deployment: Deployment) -> bool:
    return any(
        binding.semantic.startswith(HOST_SEMANTIC_PREFIX)
        for role in deployment.execution
        for binding in (*deployment.bindings[role].inputs, *deployment.bindings[role].outputs)
    )


def _external_bindings(deployment: Deployment, direction: str) -> dict[str, list[tuple[str, TensorBinding]]]:
    bindings: dict[str, list[tuple[str, TensorBinding]]] = {}
    for role in deployment.execution:
        for binding in getattr(deployment.bindings[role], direction):
            if binding.semantic.startswith((INTERNAL_SEMANTIC_PREFIX, HOST_SEMANTIC_PREFIX)):
                continue
            bindings.setdefault(binding.semantic, []).append((role, binding))
    return bindings


def _validate_contract_direction(
    direction: str,
    declared: dict[str, SemanticTensor],
    bound: dict[str, list[tuple[str, TensorBinding]]],
    *,
    host_orchestrated: bool = False,
) -> None:
    missing = sorted(set(declared) - set(bound))
    if missing and not host_orchestrated:
        raise ManifestValidationError(f"Compiled deployment omits declared semantic {direction} bindings: {missing}")
    unexpected = sorted(set(bound) - set(declared))
    if unexpected:
        raise ManifestValidationError(f"Compiled deployment has undeclared semantic {direction} bindings: {unexpected}")

    for semantic, descriptor in declared.items():
        for role, binding in bound.get(semantic, ()):
            mismatches: list[str] = []
            if binding.dtype != descriptor.dtype:
                mismatches.append(f"dtype expected {descriptor.dtype}, actual {binding.dtype}")
            if not _shapes_compatible(binding.shape, descriptor.shape):
                mismatches.append(f"shape expected {descriptor.shape}, actual {binding.shape}")
            batch_added_image_layout = (
                descriptor.layout is None
                and binding.layout in {"NCHW", "NHWC"}
                and len(binding.shape) == len(descriptor.shape) + 1
            )
            if binding.layout != descriptor.layout and not batch_added_image_layout:
                mismatches.append(f"layout expected {descriptor.layout}, actual {binding.layout}")
            if mismatches:
                raise ManifestValidationError(
                    f"Binding {semantic!r} for role {role!r} conflicts with declared model {direction} contract: "
                    + "; ".join(mismatches)
                )


def _shapes_compatible(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    if len(actual) < len(expected):
        return False
    actual_suffix = actual[-len(expected) :]
    return all(
        actual_dimension == -1 or expected_dimension == -1 or actual_dimension == expected_dimension
        for actual_dimension, expected_dimension in zip(actual_suffix, expected, strict=True)
    )


def _policy_feature_for_semantic(semantic: str, policy: PolicyMetadata) -> PolicyFeature | None:
    if policy.policy_type in {"pi05", "smolvla"} and semantic in {
        "observation.language.tokens",
        "observation.language.attention_mask",
    }:
        return None
    if semantic.startswith("observation."):
        try:
            return policy.input_features[semantic]
        except KeyError as exc:
            raise ManifestValidationError(
                f"Binding references unknown LeRobot input feature {semantic!r}; "
                f"available inputs: {sorted(policy.input_features)}"
            ) from exc
    if semantic == "action":
        return policy.output_features[semantic]
    return None


def _shape_matches_feature(binding: TensorBinding, feature: PolicyFeature, policy_type: str) -> bool:
    expected = feature.shape
    if policy_type in {"pi05", "smolvla"} and feature.type.upper() == "VISUAL" and len(expected) == 3:
        if len(binding.shape) < 3:
            return False
        actual = binding.shape[-3:]
        channel_axis = -1 if binding.layout == "NHWC" else 0
        actual_channels = actual[channel_axis]
        return actual_channels == -1 or actual_channels == expected[0]
    if feature.type.upper() == "VISUAL" and len(expected) == 3 and binding.layout == "NHWC":
        expected = (expected[1], expected[2], expected[0])
    if len(binding.shape) < len(expected):
        return False
    actual_suffix = binding.shape[-len(expected) :]
    if policy_type in {"pi05", "smolvla"} and feature.type.upper() in {"STATE", "ACTION"}:
        return all(actual == -1 or actual >= wanted for actual, wanted in zip(actual_suffix, expected, strict=True))
    return all(actual == -1 or actual == wanted for actual, wanted in zip(actual_suffix, expected, strict=True))
