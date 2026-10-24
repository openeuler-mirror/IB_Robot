"""Finalize selected publication roles without rereading already checked artifacts."""

import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from inference_manifest import (
    ARTIFACT_INTEGRITY_UNVERIFIED,
    CompiledDeployment,
    Deployment,
    DeploymentArtifact,
    InferenceManifest,
    ManifestIntegrityError,
    ManifestPathError,
    PublishedArtifactIndex,
    bind_artifact_digest,
    finalize_deployment_artifacts,
    inherit_deployment_version,
    integrity_report_path,
    normalize_bundle_path,
    resolve_bundle_file,
)
from inference_manifest.paths import depends_on_file_replacement


@dataclass(frozen=True)
class ArtifactPublication:
    """One deployment's descriptors and copy sources, classified together."""

    artifacts: dict[str, DeploymentArtifact]
    sources: dict[str, Path]

    @property
    def retained(self) -> dict[str, DeploymentArtifact]:
        return {role: artifact for role, artifact in self.artifacts.items() if role not in self.sources}

    @property
    def import_candidates(self) -> dict[str, tuple[str, Path]]:
        return {role: (self.artifacts[role].path, source) for role, source in self.sources.items()}


def classify_om_artifacts(
    root: Path, inputs: Mapping[str, tuple[str, Path]], published: PublishedArtifactIndex
) -> ArtifactPublication:
    """Inherit same-file evidence and verify different-file sources before copies."""
    artifacts = {}
    sources = {}
    for role, (path, source) in inputs.items():
        destination = root / path
        retained = destination.is_file() and source.samefile(destination)
        digest = published.declared_digest(source) if retained else published.verify_declared_digest(source)
        artifacts[role] = DeploymentArtifact(path=path, format="om", sha256=digest)
        if not retained:
            sources[role] = source
    return ArtifactPublication(artifacts, sources)


def _finalize_copied_artifact_roles(
    root: Path,
    deployment: CompiledDeployment,
    roles: Iterable[str],
) -> CompiledDeployment:
    """Bind copied roles at their final paths, independently of preflight caches.

    The subset is internal to digest finalization; only the complete deployment
    is published and subjected to schema and binding validation.
    """
    selected = {role: deployment.artifacts[role] for role in roles}
    finalized = finalize_deployment_artifacts(root, deployment.model_copy(update={"artifacts": selected}))
    return deployment.model_copy(update={"artifacts": {**deployment.artifacts, **finalized.artifacts}})


def _destination_entry(root: Path, relative_path: str) -> Path:
    """Validate a write path, including missing leaves, before any mutation.

    Return the directory entry used by replace(), but also check the resolved
    leaf: asset writers follow that leaf, and neither operation may escape.
    The publication's existing single-writer requirement applies here too.
    """
    try:
        destination = root / normalize_bundle_path(relative_path)
        resolved_root = root.resolve(strict=False)
        entry = destination.parent.resolve(strict=False) / destination.name
        entry.relative_to(resolved_root)
        resolved = destination.resolve(strict=False)
        resolved.relative_to(resolved_root)
        if destination.is_dir():
            raise ValueError("file publication target is a directory")
        if destination.exists() and not destination.is_file():
            raise ValueError("file publication target is not a regular file")
        for target in (entry, resolved):
            for parent in target.parents:
                if parent == resolved_root:
                    break
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"file publication target ancestor is not a directory: {parent}")
        return entry
    except (ValueError, OSError, RuntimeError) as exc:
        raise ManifestIntegrityError(
            f"{ARTIFACT_INTEGRITY_UNVERIFIED}: publication target {relative_path!r} "
            f"is invalid or escapes the bundle root: {exc}"
        ) from exc


def _reusable_destination_digest(
    root: Path, relative_path: str, source: Path, published: PublishedArtifactIndex
) -> str | None:
    """Return verified equal content, or None when replacement is required.

    Independent inputs to missing targets need no comparison hash. Sources
    associated with published evidence must still pass their old declaration.
    Each invocation rechecks file state through the publication index.
    """
    destination = root / relative_path
    if not destination.is_file():
        published.verify_declared_digest(source)
        return None
    resolved_destination = resolve_bundle_file(root, relative_path)
    return published.reusable_digest(source, resolved_destination)


def _verified_retained_artifact(
    root: Path, role: str, artifact: DeploymentArtifact, published: PublishedArtifactIndex
) -> DeploymentArtifact:
    try:
        resolved = resolve_bundle_file(root, artifact.path)
        return bind_artifact_digest(artifact, published.read_and_verify_digest(resolved), role)
    except (ManifestPathError, OSError) as exc:
        raise ManifestIntegrityError(
            f"{ARTIFACT_INTEGRITY_UNVERIFIED}: retained artifact role={role!r} path={artifact.path!r}: {exc}"
        ) from exc


def _preflight_asset_writes(
    root: Path,
    asset_writes: Iterable[str],
    protected: set[Path],
    asset_sources: Mapping[str, Path],
) -> None:
    """Check in-place writers before content reuse or atomic artifact replacement."""
    # Asset writers currently update files in place. Protect both retained
    # artifacts and copy sources from symlink traversal and shared inodes.
    asset_writes = tuple(asset_writes)
    assets = [_destination_entry(root, path) for path in asset_writes]
    asset_writes = tuple(normalize_bundle_path(path) for path in asset_writes)
    asset_targets = set()
    asset_inodes = set()
    for path in assets:
        target = path.resolve(strict=False)
        aliases = target in asset_targets
        if path.is_file():
            info = path.stat()
            inode = (info.st_dev, info.st_ino)
            aliases = aliases or inode in asset_inodes
            asset_inodes.add(inode)
        if aliases:
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: publication asset {str(path)!r} conflicts with another asset"
            )
        asset_targets.add(target)
    for path in protected:
        aliases = path.resolve(strict=False) in asset_targets
        if path.is_file():
            info = path.stat()
            aliases = aliases or (info.st_dev, info.st_ino) in asset_inodes
        if aliases:
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: artifact or source {str(path)!r} aliases a publication asset"
            )

    for destination, source in asset_sources.items():
        destination = normalize_bundle_path(destination)
        for relative, target in zip(asset_writes, assets, strict=True):
            if relative == destination:
                continue
            if source.resolve(strict=False) == target.resolve(strict=False) or (
                source.is_file() and target.is_file() and source.samefile(target)
            ):
                raise ManifestIntegrityError(
                    f"{ARTIFACT_INTEGRITY_UNVERIFIED}: source {str(source)!r} aliases another publication asset"
                )


def _preflight_replacement_dependencies(
    root: Path,
    import_candidates: Mapping[tuple[str, str], tuple[str, Path]],
    replacement_targets: Mapping[tuple[str, str], Path],
    retained_paths: Mapping[tuple[str, str], str],
    read_assets: set[Path],
    retained_asset_paths: set[Path],
    asset_sources: Mapping[str, Path],
) -> set[Path]:
    """Protect sources and retained paths after deciding which imports need copies."""
    targets = set(replacement_targets.values())
    # The manifest is the expected destination for its own atomic replacement;
    # check internal metadata only against artifact imports at this stage.
    for path in (read_assets - retained_asset_paths) | set(asset_sources.values()):
        if path.resolve(strict=False) in targets or (path.exists() and depends_on_file_replacement(path, targets)):
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: asset or metadata {str(path)!r} aliases a replaced target"
            )
    for (deployment_name, role), (_, source) in import_candidates.items():
        if depends_on_file_replacement(
            source, targets, excluded_target=replacement_targets.get((deployment_name, role))
        ):
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: source deployment={deployment_name!r} "
                f"role={role!r} path={str(source)!r} aliases another replaced artifact"
            )

    # Publication replaces the manifest and discards the previous verifier report.
    targets.add(_destination_entry(root, "inference_manifest.json"))
    targets.add(_destination_entry(root, integrity_report_path(root).name))
    for path in retained_asset_paths:
        if path.resolve(strict=False) in targets or (path.exists() and depends_on_file_replacement(path, targets)):
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: retained asset {str(path)!r} aliases a replaced target"
            )

    # Retained aliases must not bind a digest to bytes that will be replaced.
    for (deployment_name, role), relative_path in retained_paths.items():
        resolve_bundle_file(root, relative_path)
        if depends_on_file_replacement(root / relative_path, targets):
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: retained artifact deployment={deployment_name!r} "
                f"role={role!r} path={relative_path!r} aliases a replaced target"
            )

    return targets


def _preserve_deployments(
    root: Path,
    existing: InferenceManifest | None,
    published: PublishedArtifactIndex,
    replaced_deployments: set[str],
    targets: set[Path],
) -> dict[str, Deployment]:
    """Verify untouched deployments and inherit their versions after all checks."""
    if existing is not None:
        preserved_artifacts: dict[str, dict[str, DeploymentArtifact]] = {}
        for deployment_name, deployment in existing.deployments.items():
            if deployment_name in replaced_deployments:
                continue
            for role, artifact in deployment.artifacts.items():
                verified = _verified_retained_artifact(root, role, artifact, published)
                if depends_on_file_replacement(root / artifact.path, targets):
                    raise ManifestIntegrityError(
                        f"{ARTIFACT_INTEGRITY_UNVERIFIED}: retained artifact deployment={deployment_name!r} "
                        f"role={role!r} path={artifact.path!r} aliases a replaced target"
                    )
                preserved_artifacts.setdefault(deployment_name, {})[role] = verified

    preserved: dict[str, Deployment] = {}
    if existing is not None:
        for name, deployment in existing.deployments.items():
            if name in replaced_deployments:
                continue
            artifacts = preserved_artifacts.get(name, {})
            candidate = deployment.model_copy(update={"artifacts": artifacts})
            preserved[name] = inherit_deployment_version(candidate, deployment)
    return preserved


def preflight_publication(
    root: Path,
    published: PublishedArtifactIndex,
    publications: Mapping[str, ArtifactPublication],
    replaced_deployments: set[str],
    *,
    asset_writes: Iterable[str] = (),
    retained_assets: Iterable[str] = (),
    asset_sources: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Deployment], dict[str, ArtifactPublication]]:
    """Validate all retained evidence and reject writes that would retarget aliases.

    Return untouched deployments and publications ready for copying. Equal
    imports become verified retained roles; sources contain only replacements.
    Retained inputs need no prior finalization: their declared and indexed SHA
    evidence is checked and bound here, once. Asset sources may reuse their own
    destination, but cannot alias another asset writer or an artifact replaced
    later in this publication.

    Retained assets must be existing bundle-local files. Include old bundle
    entries that are not being rewritten, even if discovery now omits them.
    New metadata and asset write targets may still be absent before publication.

    Order: retained asset paths, asset writes/aliases, distinct import targets,
    current retained evidence, content reuse, dependencies, untouched deployments.
    Keep content reuse before dependency checks: equal imports do not replace
    entries. All validation finishes before callers mutate the bundle. Callers
    must keep sources and retained files unchanged until copying finishes,
    under the publication index's existing single-writer contract.
    """
    existing = published.manifest
    import_candidates = {
        (name, role): value for name, item in publications.items() for role, value in item.import_candidates.items()
    }
    retained_artifacts = {
        (name, role): value for name, item in publications.items() for role, value in item.retained.items()
    }
    asset_writes = tuple(asset_writes)
    asset_sources = asset_sources or {}
    protected = {root / artifact.path for artifact in retained_artifacts.values()}
    protected.update(source for _, source in import_candidates.values())
    protected.update(root / path for path, _ in import_candidates.values())
    if existing is not None:
        protected.update(
            root / artifact.path for dep in existing.deployments.values() for artifact in dep.artifacts.values()
        )
    retained_assets = set(retained_assets)
    if existing is not None:
        retained_assets.update(entry.path for entry in existing.bundle.files if entry.path not in asset_writes)
    for path in sorted(retained_assets):
        try:
            resolve_bundle_file(root, path)
        except (ManifestPathError, OSError) as exc:
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: retained asset path={path!r}: {exc}"
            ) from exc
    retained_asset_paths = {root / path for path in retained_assets}
    read_assets = set(retained_asset_paths)
    read_assets.update((root / "inference_manifest.json", integrity_report_path(root)))
    protected.update(read_assets)
    _preflight_asset_writes(root, asset_writes, protected, asset_sources)

    # Directory symlinks can collapse distinct deployment paths onto one entry.
    # Check every import, even equal-content imports that may skip copying.
    owners: dict[Path, tuple[str, str]] = {}
    for identity, (relative_path, _source) in import_candidates.items():
        entry = _destination_entry(root, relative_path)
        if entry in owners:
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: publication target {str(entry)!r} "
                f"conflicts between {owners[entry]!r} and {identity!r}"
            )
        owners[entry] = identity

    # Missing leaves can still conflict: one file cannot be another's directory.
    # Assets follow the final symlink; artifact and manifest writes replace entries.
    file_targets = set(owners)
    for path in (*asset_writes, "inference_manifest.json", integrity_report_path(root).name):
        file_targets.add(_destination_entry(root, path))
    file_targets.update((root / path).resolve(strict=False) for path in asset_writes)
    for target in file_targets:
        if any(parent in file_targets for parent in target.parents):
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: publication target {str(target)!r} "
                "conflicts with an ancestor file target"
            )

    verified_artifacts: dict[str, dict[str, DeploymentArtifact]] = {}
    for (deployment_name, role), artifact in retained_artifacts.items():
        verified_artifacts.setdefault(deployment_name, {})[role] = _verified_retained_artifact(
            root, role, artifact, published
        )

    replacement_targets: dict[tuple[str, str], Path] = {}
    retained_paths = {identity: artifact.path for identity, artifact in retained_artifacts.items()}
    for identity, (relative_path, source) in import_candidates.items():
        digest = _reusable_destination_digest(root, relative_path, source, published)
        if digest is None:
            replacement_targets[identity] = _destination_entry(root, relative_path)
        else:
            retained_paths[identity] = relative_path
            name, role = identity
            verified_artifacts.setdefault(name, {})[role] = bind_artifact_digest(
                publications[name].artifacts[role], digest, role
            )

    targets = _preflight_replacement_dependencies(
        root,
        import_candidates,
        replacement_targets,
        retained_paths,
        read_assets,
        retained_asset_paths,
        asset_sources,
    )

    preserved = _preserve_deployments(root, existing, published, replaced_deployments, targets)
    return preserved, {
        name: replace(
            item,
            artifacts={**item.artifacts, **verified_artifacts.get(name, {})},
            sources={role: source for role, source in item.sources.items() if (name, role) in replacement_targets},
        )
        for name, item in publications.items()
    }


def copy_and_finalize_artifact_roles(
    root: Path,
    deployment: CompiledDeployment,
    publication: ArtifactPublication,
) -> CompiledDeployment:
    """Copy only replacements from a completed preflight_publication result.

    Replace directory entries rather than mutating shared inodes. Equal files
    already carry the preflight digest; copied files are hashed at their final
    path independently of the preflight index. The caller must preserve the
    preflight's single-writer and unchanged-source/retained-file boundary.
    """
    artifacts = dict(publication.artifacts)
    copied = set()
    for role, source in publication.sources.items():
        artifact = artifacts[role]
        destination = root / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{uuid4()}.tmp")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        copied.add(role)
    return _finalize_copied_artifact_roles(root, deployment.model_copy(update={"artifacts": artifacts}), copied)
