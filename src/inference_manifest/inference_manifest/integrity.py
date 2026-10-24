"""Canonical deployment identity and artifact integrity helpers."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from os import stat_result
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

from pydantic import Field, TypeAdapter, model_validator

from inference_manifest.errors import ManifestIntegrityError, ManifestPathError, ManifestValidationError
from inference_manifest.json_utils import load_json_strict
from inference_manifest.models import (
    BundleFile,
    Deployment,
    DeploymentArtifact,
    InferenceManifest,
    ModelDescriptor,
    ModelIdentity,
    RoleRuntimeProfile,
    SemanticIdentity,
    Sha256,
    StrictFrozenModel,
    StrictString,
    _json_safe,
    _runtime_profiles,
    _without_provider_identity,
)
from inference_manifest.paths import normalize_unique_paths, resolve_bundle_file

_REGENERATE_GUIDANCE = (
    "Rerun the owning exporter or packaging workflow to regenerate the manifest; do not edit digests manually."
)
_INTEGRITY_FILENAME = "inference_integrity.json"
_DEFAULT_VERIFIER = "inference_manifest/0.1.0"
ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"
ARTIFACT_INTEGRITY_UNVERIFIED = "artifact_integrity_unverified"

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
    deployment_identity: Sha256 | None = None

    def require_verified(self) -> IntegrityReport:
        if self.state != "verified":
            code = self.error_code or ARTIFACT_INTEGRITY_UNVERIFIED
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
        # Preserve legacy identities when the optional parameters are absent.
        # This is cache identity, not a user-supplied file checksum.
        **({"extra_parameters": _model_value(deployment.extra_parameters)} if deployment.extra_parameters else {}),
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
        **({"extra_parameters": _model_value(deployment.extra_parameters)} if deployment.extra_parameters else {}),
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


def integrity_report_path(bundle_root: str | Path) -> Path:
    """Return the root report entry without following a possible leaf symlink."""
    return Path(bundle_root) / _INTEGRITY_FILENAME


def _declared_integrity_status(deployment: Deployment, *, include_missing: bool = False) -> ArtifactIntegrityStatus:
    declared = {
        role: artifact.sha256 for role, artifact in sorted(deployment.artifacts.items()) if artifact.sha256 is not None
    }
    missing = sorted(deployment.artifacts.keys() - declared.keys())
    if deployment.artifacts and not missing:
        state: Literal["declared", "unverified"] = "declared"
    else:
        state = "unverified"
    return ArtifactIntegrityStatus(
        mode="declared_only",
        state=state,
        artifact_digests=declared,
        details={"missing_declared_digests": missing} if include_missing and missing else {},
    )


def read_integrity_report(bundle_root: str | Path, deployment_name: str | None = None) -> IntegrityReport | None:
    """Read an adjacent verifier report, if one exists."""

    root = Path(bundle_root).resolve(strict=True)
    path = integrity_report_path(root)
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
    if report is not None and report.deployment_identity == _report_deployment_identity(deployment):
        return report
    return _declared_integrity_status(deployment)


def _report_deployment_identity(deployment: Deployment) -> str:
    """Bind reports to the full declaration, including lineage and artifact SHA values."""
    payload = deployment.model_dump(mode="json", exclude_none=True)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def inherit_deployment_version(candidate: Deployment, previous: Deployment | None) -> Deployment:
    """Preserve lineage and increment revision exactly when the declaration changes."""
    if previous is None:
        return candidate
    changed = _canonical_json(candidate.model_dump(mode="json", exclude={"uuid", "revision"})) != _canonical_json(
        previous.model_dump(mode="json", exclude={"uuid", "revision"})
    )
    return candidate.model_copy(update={"uuid": previous.uuid, "revision": previous.revision + int(changed)})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file using the shared chunked reader."""
    return _sha256_file(path)


_SHA256_ADAPTER = TypeAdapter(Sha256)


def bind_artifact_digest(artifact: DeploymentArtifact, digest: str, role: str) -> DeploymentArtifact:
    """Bind a computed digest, rejecting changed bytes for retained evidence."""
    digest = _SHA256_ADAPTER.validate_python(digest)
    if artifact.sha256 is not None and artifact.sha256 != digest:
        raise ManifestIntegrityError(
            f"{ARTIFACT_DIGEST_MISMATCH}: retained artifact role={role!r} path={artifact.path!r}"
        )
    return artifact.model_copy(update={"sha256": digest})


class _FileState(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, info: stat_result) -> _FileState:
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_state(path: Path) -> _FileState:
    return _FileState.from_stat(path.stat())


def _cached_file_digest(path: Path, cache: dict[_FileState, str], *, context: str) -> str:
    """Reuse a stable file's digest within the caller's publication phase."""
    state = _file_state(path)
    if state not in cache:
        digest = _sha256_file(path)
        if _file_state(path) != state:
            raise ManifestIntegrityError(f"{ARTIFACT_INTEGRITY_UNVERIFIED}: artifact changed while hashing{context}")
        cache[state] = digest
    return cache[state]


def _read_artifact_digests(
    bundle_root: str | Path, deployment: Deployment, published: PublishedArtifactIndex | None = None
) -> dict[str, str]:
    """Hash each unique file identity once while resolving every declaration."""
    digests: dict[_FileState, str] = {}
    actual = {}
    for role, artifact in sorted(deployment.artifacts.items()):
        try:
            path = resolve_bundle_file(Path(bundle_root), artifact.path)
            if published is not None:
                actual[role] = published.read_and_verify_digest(path)
                continue
            actual[role] = _cached_file_digest(path, digests, context=f" role={role!r}")
        except (ManifestPathError, OSError) as exc:
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: role={role!r} path={artifact.path!r}: {exc}"
            ) from exc
    return actual


class PublishedArtifactIndex:
    """Index old declarations and cache content for one coordinated publication.

    Sources may name another role or deployment, or a symbolic/hardlink alias.
    The declaration index snapshots file identities before writes; it is not an
    index of the evolving manifest. Cached content is reusable only while the
    identity, size, mtime and ctime match, and is checked on every digest lookup.

    After preflight proves disjoint targets and protects retained/source paths,
    callers may keep using this index for untouched files while atomically
    replacing other directory entries. Never use it to validate newly copied
    destinations: finalize those independently, without this index. In-place
    artifact writes or retargeting indexed aliases invalidate this contract.
    Discard the index when publication ends; a new publication rebuilds it.
    Callers must exclude concurrent writers for the entire publication.
    ``manifest`` is the same declaration snapshot used to build this index.
    """

    def __init__(self, bundle_root: Path, manifest: InferenceManifest | None):
        self.manifest = manifest
        self._digests: dict[tuple[int, int], set[str]] = {}
        self._actual: dict[_FileState, str] = {}
        if manifest is not None:
            for deployment_name, deployment in manifest.deployments.items():
                for role, artifact in deployment.artifacts.items():
                    if artifact.sha256 is None:
                        continue
                    try:
                        info = (bundle_root / artifact.path).stat()
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        raise ManifestIntegrityError(
                            f"{ARTIFACT_INTEGRITY_UNVERIFIED}: published artifact "
                            f"deployment={deployment_name!r} role={role!r} path={artifact.path!r}: {exc}"
                        ) from exc
                    if stat.S_ISREG(info.st_mode):
                        self._digests.setdefault((info.st_dev, info.st_ino), set()).add(artifact.sha256)

    def declared_digest(self, path: Path) -> str | None:
        """Return old evidence without reading content; reject conflicting claims."""
        info = path.stat()
        digests = self._digests.get((info.st_dev, info.st_ino), set())
        if len(digests) > 1:
            raise ManifestIntegrityError(f"{ARTIFACT_DIGEST_MISMATCH}: conflicting retained artifact digests: {path}")
        return next(iter(digests), None)

    def _digest_unchecked(self, path: Path) -> str:
        """Return and cache actual content without accepting or rejecting old claims."""
        return _cached_file_digest(path, self._actual, context=f": {path}")

    def read_and_verify_digest(self, path: Path) -> str:
        """Read content (or reuse its state cache) and check any old declaration.

        Without an old declaration, still compute and return actual content.
        A cache entry is valid only for the same file identity and stat state.
        """
        expected = self.declared_digest(path)
        actual = self._digest_unchecked(path)
        if expected is not None and actual != expected:
            raise ManifestIntegrityError(f"{ARTIFACT_DIGEST_MISMATCH}: retained artifact source={str(path)!r}")
        return actual

    def _accept_computed_digest(self, path: Path, actual: str) -> str:
        """Check a supplied digest against old evidence/cache without reading bytes.

        The caller must ensure the digest describes this path's current bytes
        and coordinate writers between computation and this call. With neither
        an old declaration nor cached content, the supplied digest is trusted
        and cached; this method does not independently verify file content.
        """
        actual = _SHA256_ADAPTER.validate_python(actual)
        expected = self.declared_digest(path)
        state = _file_state(path)
        cached = self._actual.get(state)
        if cached is not None and cached != actual:
            raise ManifestIntegrityError(f"{ARTIFACT_INTEGRITY_UNVERIFIED}: artifact changed while checking: {path}")
        if expected is not None and expected != actual:
            raise ManifestIntegrityError(f"{ARTIFACT_DIGEST_MISMATCH}: retained artifact source={str(path)!r}")
        self._actual[state] = actual
        return actual

    def verify_declared_digest(self, path: Path) -> str | None:
        """Verify existing evidence, returning None without reading if undeclared.

        When evidence exists, read content (or use its state cache) and compare.
        None means no old declaration, not verified content; copied outputs
        still require finalization at their final paths before publication.
        """
        expected = self.declared_digest(path)
        if expected is None:
            return None
        return self.read_and_verify_digest(path)

    def _verify_matching_content(self, source: Path, destination: Path) -> bool:
        """Compare undeclared history, reusing stable content digests.

        Reuse either side's cached digest only for its current file state.
        Only use digest comparison when both sides are cached. Otherwise compare
        block by block and stop at the first difference, even if one side is cached. Equal bytes
        share a streaming digest; check both file states and old evidence before
        accepting the match. Final publication still verifies copied targets.
        """
        before = _file_state(source), _file_state(destination)
        if all(state in self._actual for state in before):
            actual = self._actual[before[0]]
            matches = actual == self._actual[before[1]]
        else:
            digest = hashlib.sha256()
            with source.open("rb") as left, destination.open("rb") as right:
                while True:
                    chunk = left.read(64 * 1024)
                    if chunk != right.read(64 * 1024):
                        return False
                    if not chunk:
                        break
                    digest.update(chunk)
            actual = digest.hexdigest()
            matches = True
        if before != (_file_state(source), _file_state(destination)):
            raise ManifestIntegrityError(f"{ARTIFACT_INTEGRITY_UNVERIFIED}: artifact changed while comparing")
        if not matches:
            return False
        self._accept_computed_digest(source, actual)
        self._accept_computed_digest(destination, actual)
        return True

    def reusable_digest(self, source: Path, destination: Path) -> str | None:
        """Return verified equal content, or None for an unusable destination.

        Callers validate regular-file paths and choose permission/replacement
        policies. Corrupt sources and conflicting destination declarations
        raise; a damaged destination with one old claim is not reusable.
        This is preparation evidence, never final verification after a copy.
        """
        self.verify_declared_digest(source)
        declared = self.declared_digest(destination)
        try:
            if declared is None and not source.samefile(destination):
                if not self._verify_matching_content(source, destination):
                    return None
                return self.read_and_verify_digest(source)
            actual = self.read_and_verify_digest(source)
            if declared is not None and actual != declared:
                return None
            return actual if self.read_and_verify_digest(destination) == actual else None
        except ManifestIntegrityError:
            return None


class ArtifactDeployment(Protocol):
    @property
    def deployment(self) -> Deployment: ...

    @property
    def bundle_root(self) -> Path: ...


def _inspect_deployment_artifacts(
    bundle_root: Path,
    deployment: Deployment,
    *,
    verify_content: bool = True,
) -> ArtifactIntegrityStatus:
    """Read-only core shared by strict runtime checks and persisted reports."""
    declared = _declared_integrity_status(deployment, include_missing=True)
    if not verify_content:
        return declared
    actual_digests = _read_artifact_digests(bundle_root, deployment)
    mismatches: list[ArtifactDigestMismatch] = []
    for role, artifact in sorted(deployment.artifacts.items()):
        if artifact.sha256 is not None and artifact.sha256 != actual_digests[role]:
            mismatches.append(
                ArtifactDigestMismatch(
                    role=role, path=artifact.path, expected=artifact.sha256, actual=actual_digests[role]
                )
            )
    state: ArtifactIntegrityState = "verified"
    if mismatches:
        state = "mismatch"
    elif declared.state == "unverified":
        state = "unverified"
    return ArtifactIntegrityStatus(
        mode="verify_on_demand",
        state=state,
        artifact_digests=actual_digests,
        mismatches=tuple(mismatches),
        details={
            **declared.details,
            **({"code": ARTIFACT_DIGEST_MISMATCH} if mismatches else {}),
        },
    )


def validate_deployment_artifact_integrity(
    validated_manifest: ArtifactDeployment, *, verify_content: bool = True
) -> None:
    """Require declared/verified compiled artifacts without writing a report.

    Empty artifacts are allowed only for Torch profiles, whose weights are
    declared as bundle assets. Other backends must declare compiled artifacts.
    Remote checks require digests but do not access local artifact files.
    Failures use ManifestIntegrityError and the publication verifier's codes.
    """
    root = validated_manifest.bundle_root
    deployment = validated_manifest.deployment
    profiles = _runtime_profiles(deployment.runtime_profile, deployment.role_runtime_profiles)
    if not deployment.artifacts and profiles and all(profile.backend == "torch" for profile in profiles):
        return
    declared = _declared_integrity_status(deployment, include_missing=True)
    if declared.state == "unverified":
        raise ManifestIntegrityError(f"{ARTIFACT_INTEGRITY_UNVERIFIED}: {declared.details or 'no compiled artifacts'}")
    result = _inspect_deployment_artifacts(
        root,
        deployment,
        verify_content=verify_content,
    )
    if result.state not in {"verified", "declared"}:
        code = result.error_code or ARTIFACT_INTEGRITY_UNVERIFIED
        raise ManifestIntegrityError(f"{code}: {result.mismatch_details or result.details or 'no compiled artifacts'}")


def finalize_deployment_artifacts(
    bundle_root: str | Path, deployment: Deployment, *, published: PublishedArtifactIndex | None = None
) -> Deployment:
    """Bind a packaging candidate to its final artifact bytes before publication.

    Exporters call this after copying/compiler output and before comparing
    deployment revisions. New artifacts omit sha256; retained artifacts keep
    their previous digest, which is verified rather than refreshed. Loading and
    writing an existing manifest also preserve evidence of content tampering.

    Supply the publication index for pre-write retained candidates, including
    candidates missing their old SHA, to recover evidence and share reads across
    deployments. Omit it for independently copied final destinations.
    """
    digests = _read_artifact_digests(bundle_root, deployment, published)
    artifacts = {
        role: bind_artifact_digest(artifact, digests[role], role) for role, artifact in deployment.artifacts.items()
    }
    return deployment.model_copy(update={"artifacts": artifacts})


def _write_integrity_report(bundle_root: Path, report: IntegrityReport) -> None:
    path = integrity_report_path(bundle_root)
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
    expected_deployment: Deployment | None = None,
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
    if expected_deployment is not None and _report_deployment_identity(deployment) != _report_deployment_identity(
        expected_deployment
    ):
        raise ManifestIntegrityError(f"{ARTIFACT_INTEGRITY_UNVERIFIED}: deployment changed before verification")
    verify_bundle_digest(
        manifest.bundle.uuid,
        manifest.bundle.revision,
        manifest.bundle.name,
        manifest.bundle.files,
        manifest.bundle.digest.value,
    )

    result = _inspect_deployment_artifacts(root, deployment)
    report = IntegrityReport(
        **result.model_dump(exclude={"mode", "verified_at", "verifier"}),
        deployment_name=deployment_name,
        deployment_identity=_report_deployment_identity(deployment),
        mode=mode,
        verified_at=datetime.now(timezone.utc) if result.state == "verified" else None,
        verifier=verifier,
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
