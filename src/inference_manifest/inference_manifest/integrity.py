"""Canonical deployment identity and artifact integrity helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from inference_manifest.errors import ManifestIntegrityError, ManifestValidationError
from inference_manifest.json_utils import load_json_strict
from inference_manifest.models import (
    BundleFile,
    Deployment,
    InferenceManifest,
    ModelDescriptor,
    ModelIdentity,
    RoleRuntimeProfile,
    SemanticIdentity,
    Sha256,
    StrictFrozenModel,
    StrictString,
    _json_safe,
    _without_provider_identity,
)
from inference_manifest.paths import normalize_unique_paths, resolve_bundle_file

_REGENERATE_GUIDANCE = (
    "Rerun the owning exporter or packaging workflow to regenerate the manifest; do not edit digests manually."
)
_INTEGRITY_FILENAME = "inference_integrity.json"
_DEFAULT_VERIFIER = "inference_manifest/0.1.0"
ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"

ArtifactIntegrityMode = Literal["declared_only", "verify_on_install", "verify_on_demand"]
ArtifactIntegrityState = Literal["declared", "verified", "mismatch", "unverified"]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _model_value(value: Any, *, exclude_none: bool = True) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=exclude_none)
    return value


def _semantic_contract_value(value: ModelDescriptor | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, ModelDescriptor):
        value = ModelDescriptor.model_validate_json(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return value.model_dump(
        mode="json",
        exclude_none=True,
        include={
            "interface",
            "model_type",
            "operation",
            "inputs",
            "outputs",
            "architecture_class",
            "domain",
            "lineage",
            "semantic_identity",
        },
    )


class ArtifactDigestMismatch(StrictFrozenModel):
    """One stable artifact digest mismatch reported by the verifier."""

    code: Literal["artifact_digest_mismatch"] = ARTIFACT_DIGEST_MISMATCH
    role: str
    path: str
    expected: Sha256
    actual: Sha256


class ArtifactIntegrityStatus(StrictFrozenModel):
    """Runtime-visible integrity state, without implying a fresh hash on load."""

    mode: ArtifactIntegrityMode
    state: ArtifactIntegrityState
    artifact_digests: dict[str, Sha256] = Field(default_factory=dict)
    verified_at: datetime | None = None
    verifier: StrictString | None = None
    mismatches: tuple[ArtifactDigestMismatch, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse_rfc3339_timestamp(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized = dict(value)
            timestamp = normalized.get("verified_at")
            if isinstance(timestamp, str):
                normalized["verified_at"] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return normalized
        return value

    @model_validator(mode="after")
    def normalize_integrity_values(self) -> ArtifactIntegrityStatus:
        if self.verified_at is not None:
            if self.verified_at.tzinfo is None or self.verified_at.utcoffset() is None:
                raise ValueError("verified_at must be an RFC3339 timezone-aware UTC timestamp")
            object.__setattr__(self, "verified_at", self.verified_at.astimezone(timezone.utc))
        object.__setattr__(self, "details", _json_safe(self.details, "integrity details"))
        if self.state == "mismatch" and not self.mismatches:
            raise ValueError("mismatch integrity state requires mismatch details")
        return self

    @property
    def mismatch_details(self) -> tuple[ArtifactDigestMismatch, ...]:
        return self.mismatches

    @property
    def error_code(self) -> str | None:
        return ARTIFACT_DIGEST_MISMATCH if self.mismatches else None

    @property
    def code(self) -> str | None:
        return self.error_code


class IntegrityReport(ArtifactIntegrityStatus):
    """Persisted result of install-time or on-demand artifact verification."""

    deployment_name: str

    def require_verified(self) -> IntegrityReport:
        if self.state != "verified":
            code = self.error_code or "artifact_integrity_unverified"
            raise ManifestIntegrityError(f"{code} for deployment {self.deployment_name!r}")
        return self


def canonical_bundle_digest(
    bundle_uuid: str,
    bundle_revision: int,
    bundle_name: str,
    files: Iterable[BundleFile],
) -> str:
    """Hash the lightweight bundle declaration without reading bundle files."""

    entries = tuple(files)
    normalized_paths = normalize_unique_paths((entry.path for entry in entries), "bundle.files")
    payload = {
        "format": "ibrobot.bundle-structure-v2",
        "uuid": bundle_uuid,
        "revision": bundle_revision,
        "name": bundle_name,
        "files": sorted(normalized_paths),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def verify_bundle_digest(
    bundle_uuid: str,
    bundle_revision: int,
    bundle_name: str,
    files: Iterable[BundleFile],
    expected: str,
) -> str:
    actual = canonical_bundle_digest(bundle_uuid, bundle_revision, bundle_name, files)
    if actual != expected:
        raise ManifestIntegrityError(
            f"Bundle digest mismatch: expected {expected}, actual {actual}. {_REGENERATE_GUIDANCE}"
        )
    return actual


def _identity_value(identity: ModelIdentity | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if identity is None:
        return None
    if not isinstance(identity, ModelIdentity):
        identity = ModelIdentity.model_validate(identity)
    return identity.model_dump(mode="json", exclude_none=True)


def _profile_values(
    deployment: Deployment,
    role_runtime_profiles: Mapping[str, RoleRuntimeProfile] | None,
) -> dict[str, RoleRuntimeProfile]:
    raw_profiles: Mapping[str, RoleRuntimeProfile] | None
    if role_runtime_profiles is not None:
        raw_profiles = role_runtime_profiles
    elif deployment.role_runtime_profiles is not None:
        raw_profiles = deployment.role_runtime_profiles
    else:
        raw_profiles = None
    if raw_profiles is not None:
        return {
            role: profile
            if isinstance(profile, RoleRuntimeProfile)
            else RoleRuntimeProfile.model_validate_json(json.dumps(profile, ensure_ascii=False, separators=(",", ":")))
            for role, profile in raw_profiles.items()
        }
    if deployment.runtime_profile is not None:
        profile = deployment.runtime_profile
        if not isinstance(profile, RoleRuntimeProfile):
            profile = RoleRuntimeProfile.model_validate_json(
                json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
            )
        return {"__deployment__": profile}
    return {}


def _artifact_identity(deployment: Deployment) -> dict[str, Any]:
    # Paths are deliberately omitted.  A path is a local instance detail; the
    # declared digest and artifact format are the portable deployment identity.
    return {
        role: {
            "format": artifact.format,
            **({"share_group": artifact.share_group} if artifact.share_group is not None else {}),
            **({"sha256": artifact.sha256} if artifact.sha256 is not None else {}),
        }
        for role, artifact in sorted(deployment.artifacts.items())
    }


def _binding_identity(deployment: Deployment) -> dict[str, Any]:
    return {role: _model_value(bindings) for role, bindings in sorted(deployment.bindings.items())}


def _contract_identity(deployment: Deployment) -> dict[str, Any]:
    contract = deployment.execution_contract.model_dump(mode="json", exclude_none=True)
    links = contract.get("state_links")
    if links:
        contract["state_links"] = sorted(
            links,
            key=lambda link: (
                link["role"],
                link["state_name"],
                link["owner"],
                link["source"],
                link["target"],
                link["scope"],
                link["state_bank"],
            ),
        )
    return contract


def _audio_contract_identity(deployment: Deployment) -> dict[str, Any] | None:
    contract = deployment.audio_contract
    return contract.model_dump(mode="json", exclude_none=True) if contract is not None else None


def _deployment_payload(
    schema_version: int,
    bundle_digest: str,
    deployment_name: str,
    deployment: Deployment,
    *,
    identity: ModelIdentity | Mapping[str, Any] | None = None,
    semantic_contract: ModelDescriptor | Mapping[str, Any] | None = None,
    role_identities: Mapping[str, ModelIdentity] | None = None,
    role_runtime_profiles: Mapping[str, RoleRuntimeProfile] | None = None,
) -> dict[str, Any]:
    identities = role_identities if role_identities is not None else deployment.role_identities or {}
    profiles = _profile_values(deployment, role_runtime_profiles)
    profile_identity: dict[str, Any] = {}
    for role, profile in sorted(profiles.items()):
        projection = profile.deployment_projection()
        profile_identity[role] = {
            "backend": profile.backend,
            "target_runtime": profile.target_runtime,
            "runtime_abi": profile.runtime_abi,
            "target_soc": profile.target.soc,
            "projection": _without_provider_identity(_model_value(projection, exclude_none=False)),
        }

    role_identity_value = {role: _identity_value(value) for role, value in sorted(identities.items())}
    return {
        "format": "ibrobot.deployment-structure-v3",
        "schema_version": schema_version,
        "bundle_digest": bundle_digest,
        "deployment_name": deployment_name,
        "deployment_uuid": deployment.uuid,
        "deployment_revision": deployment.revision,
        "identity": _identity_value(identity),
        "semantic_contract": _semantic_contract_value(semantic_contract) if semantic_contract is not None else None,
        "role_identities": role_identity_value,
        "execution_contract": _contract_identity(deployment),
        "audio_contract": _audio_contract_identity(deployment),
        "execution": list(deployment.execution),
        "profiles": profile_identity,
        "artifacts": _artifact_identity(deployment),
        "bindings": _binding_identity(deployment),
        "device_links": [_model_value(link) for link in deployment.device_links],
    }


def deployment_fingerprint(
    schema_version: int,
    bundle_digest: str,
    deployment_name: str,
    deployment: Deployment | Any,
    identity: ModelIdentity | Mapping[str, Any] | None = None,
    role_identities: Mapping[str, ModelIdentity] | None = None,
    role_runtime_profiles: Mapping[str, RoleRuntimeProfile] | None = None,
    *,
    model: ModelIdentity | Mapping[str, Any] | None = None,
    semantic_contract: ModelDescriptor | Mapping[str, Any] | None = None,
    provider_identity: Any = None,
) -> str:
    """Return the portable deployment identity fingerprint.

    ``ValidatedDeployment`` is accepted as a convenience, but its runtime
    profile fingerprint is never folded into this value.
    """

    del provider_identity
    validated = deployment if hasattr(deployment, "selected_deployment") else None
    if validated is not None:
        deployment = validated.selected_deployment
        identity = validated.top_level_identity
        role_identities = validated.role_identities
        role_runtime_profiles = validated.role_runtime_profiles
    if identity is None:
        identity = model
    if isinstance(model, ModelDescriptor):
        identity = model.identity
        semantic_contract = model
    elif isinstance(model, Mapping) and ("inputs" in model or "outputs" in model or "semantic_identity" in model):
        descriptor = ModelDescriptor.model_validate_json(json.dumps(model, ensure_ascii=False, separators=(",", ":")))
        identity = descriptor.identity
        semantic_contract = descriptor
    payload = _deployment_payload(
        schema_version,
        bundle_digest,
        deployment_name,
        deployment,
        identity=identity,
        semantic_contract=semantic_contract,
        role_identities=role_identities,
        role_runtime_profiles=role_runtime_profiles,
    )
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _runtime_profile_payload(
    schema_version: int,
    bundle_digest: str,
    deployment_name: str,
    deployment: Deployment,
    *,
    role_runtime_profiles: Mapping[str, RoleRuntimeProfile] | None = None,
    instance_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profiles = _profile_values(deployment, role_runtime_profiles)
    values: dict[str, Any] = {}
    for role, profile in sorted(profiles.items()):
        projection = profile.runtime_instance_projection(instance_metadata if role == "__deployment__" else None)
        values[role] = _without_provider_identity(
            {
                "backend": profile.backend,
                "target_runtime": profile.target_runtime,
                "runtime_abi": profile.runtime_abi,
                "target_soc": profile.target.soc,
                "projection": _model_value(projection, exclude_none=False),
            }
        )
    return {
        "format": "ibrobot.runtime-profile-instance-v1",
        "schema_version": schema_version,
        "bundle_digest": bundle_digest,
        "deployment_name": deployment_name,
        "profiles": values,
    }


def runtime_profile_fingerprint(
    schema_version: int,
    bundle_digest: str,
    deployment_name: str,
    deployment: Deployment | Any,
    *,
    role_runtime_profiles: Mapping[str, RoleRuntimeProfile] | None = None,
    instance_metadata: Mapping[str, Any] | None = None,
    provider_identity: Any = None,
) -> str:
    """Hash complete typed runtime profiles while ignoring provider identity."""

    del provider_identity
    if hasattr(deployment, "selected_deployment"):
        validated = deployment
        deployment = validated.selected_deployment
        role_runtime_profiles = validated.role_runtime_profiles
    payload = _runtime_profile_payload(
        schema_version,
        bundle_digest,
        deployment_name,
        deployment,
        role_runtime_profiles=role_runtime_profiles,
        instance_metadata=instance_metadata,
    )
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


runtime_instance_fingerprint = runtime_profile_fingerprint
runtime_profile_instance_fingerprint = runtime_profile_fingerprint


def canonical_runtime_profile_json(
    profile: RoleRuntimeProfile | Any,
    *,
    instance_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Serialize a complete runtime profile with deterministic key ordering."""

    if isinstance(profile, RoleRuntimeProfile):
        value = profile.runtime_instance_projection(instance_metadata)
    elif hasattr(profile, "backend") and hasattr(profile, "model_dump"):
        value = {
            "backend": profile.backend,
            "profile": _model_value(profile, exclude_none=False),
            "instance_metadata": _without_provider_identity(dict(instance_metadata or {})),
        }
    else:
        value = profile
    return _canonical_json(_without_provider_identity(_model_value(value, exclude_none=False)))


def canonical_deployment_profile_json(profile: RoleRuntimeProfile | Any) -> str:
    """Serialize the deployment-only profile projection canonically."""

    if isinstance(profile, RoleRuntimeProfile) or hasattr(profile, "deployment_projection"):
        value = profile.deployment_projection()
    else:
        value = profile
    return _canonical_json(_without_provider_identity(_model_value(value, exclude_none=False)))


def canonical_semantic_identity_json(identity: SemanticIdentity) -> str:
    """Serialize a validated semantic identity without deployment provenance."""

    if not isinstance(identity, SemanticIdentity):
        raise TypeError("identity must be a validated SemanticIdentity")
    value = identity.model_dump(mode="json", exclude_none=True)
    return _canonical_json(value)


def semantic_identity_fingerprint(identity: SemanticIdentity) -> str:
    """Hash only validated semantic model-space metadata, never model artifacts."""

    return hashlib.sha256(canonical_semantic_identity_json(identity).encode("utf-8")).hexdigest()


def _report_path(bundle_root: Path) -> Path:
    return bundle_root / _INTEGRITY_FILENAME


def _declared_integrity_status(deployment: Deployment) -> ArtifactIntegrityStatus:
    declared = {
        role: artifact.sha256 for role, artifact in sorted(deployment.artifacts.items()) if artifact.sha256 is not None
    }
    if deployment.artifacts and len(declared) == len(deployment.artifacts):
        state: Literal["declared", "unverified"] = "declared"
    else:
        state = "unverified"
    return ArtifactIntegrityStatus(
        mode="declared_only",
        state=state,
        artifact_digests=declared,
    )


def read_integrity_report(bundle_root: str | Path, deployment_name: str | None = None) -> IntegrityReport | None:
    """Read an adjacent verifier report, if one exists."""

    root = Path(bundle_root).resolve(strict=True)
    path = _report_path(root)
    if not path.exists():
        return None
    try:
        value = load_json_strict(path)
        report = IntegrityReport.model_validate_json(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (OSError, ValueError) as exc:
        raise ManifestIntegrityError(f"Invalid integrity report {path}: {exc}") from exc
    if deployment_name is not None and report.deployment_name != deployment_name:
        return None
    return report


def integrity_status_for_deployment(
    bundle_root: str | Path,
    deployment_name: str,
    deployment: Deployment,
) -> ArtifactIntegrityStatus:
    report = read_integrity_report(bundle_root, deployment_name)
    return report if report is not None else _declared_integrity_status(deployment)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_integrity_report(bundle_root: Path, report: IntegrityReport) -> None:
    path = _report_path(bundle_root)
    path.write_text(
        json.dumps(report.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def verify_deployment_artifacts(
    bundle_root: str | Path,
    deployment_name: str,
    *,
    mode: Literal["verify_on_install", "verify_on_demand"] = "verify_on_install",
    verifier: str = _DEFAULT_VERIFIER,
) -> IntegrityReport:
    """Hash selected deployment artifacts and persist an adjacent integrity report.

    The function returns a mismatch report instead of silently translating or
    rewriting a bad digest.  Publication callers can gate on ``state`` or
    ``error_code == 'artifact_digest_mismatch'``.
    """

    root = Path(bundle_root).resolve(strict=True)
    manifest_path = root / "inference_manifest.json"
    raw = load_json_strict(manifest_path)
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"Inference manifest must be a JSON object: {manifest_path}")
    if raw.get("schema_version") != 3:
        raise ManifestValidationError(
            f"Unsupported schema_version {raw.get('schema_version')!r} in {manifest_path}; supported versions: [3]"
        )
    from inference_manifest.schema import validate_manifest_schema

    validate_manifest_schema(raw, str(manifest_path))
    try:
        manifest = InferenceManifest.model_validate_json(json.dumps(raw, ensure_ascii=False, separators=(",", ":")))
    except ValueError as exc:
        raise ManifestValidationError(f"Typed manifest validation failed for {manifest_path}: {exc}") from exc
    try:
        deployment = manifest.deployments[deployment_name]
    except KeyError as exc:
        raise ManifestValidationError(f"Deployment {deployment_name!r} is not present in {manifest_path}") from exc
    verify_bundle_digest(
        manifest.bundle.uuid,
        manifest.bundle.revision,
        manifest.bundle.name,
        manifest.bundle.files,
        manifest.bundle.digest.value,
    )

    actual_digests: dict[str, str] = {}
    mismatches: list[ArtifactDigestMismatch] = []
    missing_digests: list[str] = []
    for role, artifact in sorted(deployment.artifacts.items()):
        path = resolve_bundle_file(root, artifact.path)
        actual = _sha256_file(path)
        actual_digests[role] = actual
        if artifact.sha256 is None:
            missing_digests.append(role)
        elif actual != artifact.sha256:
            mismatches.append(
                ArtifactDigestMismatch(
                    role=role,
                    path=artifact.path,
                    expected=artifact.sha256,
                    actual=actual,
                )
            )

    now = datetime.now(timezone.utc)
    if mismatches:
        state: Literal["verified", "mismatch", "unverified"] = "mismatch"
    elif missing_digests or not deployment.artifacts:
        state = "unverified"
    else:
        state = "verified"
    report = IntegrityReport(
        deployment_name=deployment_name,
        mode=mode,
        state=state,
        artifact_digests=actual_digests,
        verified_at=now if state == "verified" else None,
        verifier=verifier,
        mismatches=tuple(mismatches),
        details={
            **({"missing_declared_digests": missing_digests} if missing_digests else {}),
            **({"code": ARTIFACT_DIGEST_MISMATCH} if mismatches else {}),
        },
    )
    _write_integrity_report(root, report)
    return report


def verify_deployment_artifacts_on_demand(
    bundle_root: str | Path,
    deployment_name: str,
    *,
    verifier: str = _DEFAULT_VERIFIER,
) -> IntegrityReport:
    """Explicit diagnostic/security entry point for full content verification."""

    return verify_deployment_artifacts(
        bundle_root,
        deployment_name,
        mode="verify_on_demand",
        verifier=verifier,
    )
