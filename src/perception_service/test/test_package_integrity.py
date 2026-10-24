"""Shared publication contracts for both perception bundle entry points."""

import hashlib
import json
import os
import signal
import socket
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import _bundle_snapshot

from inference_manifest import (
    ManifestIntegrityError,
    PublishedArtifactIndex,
    load_inference_manifest,
    validate_deployment_artifact_integrity,
    verify_deployment_artifacts,
)
from perception_service.bundle_publication import ArtifactPublication, preflight_publication


@pytest.fixture(params=["ordinary", "ascend"])
def publication(request, tmp_path):
    if request.param == "ordinary":
        import perception_service.package_perception_bundles as packager

        spec = packager._specs()["ram_plus"]
        root = tmp_path / spec.name
        assets = spec.required_paths
        sources = {root / item.artifact: tmp_path / "_work" / root.name / item.source for item in spec.compiled}
        name = spec.compiled[0].deployment
        argument = root
    elif request.param == "graspgen":
        from conftest import GRASPGEN_DEPLOYMENT, package_graspgen_export

        export = request.getfixturevalue("graspgen_export")
        package_graspgen_export(export)
        deployment = load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT).deployment
        return SimpleNamespace(
            root=export.bundle,
            name=GRASPGEN_DEPLOYMENT,
            sources={
                export.bundle / artifact.path: export.om_dir / f"{role}.om"
                for role, artifact in deployment.artifacts.items()
            },
            publish=lambda: package_graspgen_export(export),
        )
    elif request.param == "ascend":
        import perception_service.package_ascend_perception_bundles as packager

        spec = packager._specs()["sam2"]
        root = tmp_path / spec.name
        assets = [destination for _, destination in spec.assets]
        sources = {
            root / item.destination: tmp_path / item.source
            for deployment in spec.deployments
            for item in deployment.artifacts
        }
        name = spec.deployments[0].name
        argument = tmp_path
    else:
        raise ValueError(f"unknown publication entry point: {request.param!r}")
    for path in [*(root / relative for relative in assets), *sources]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    return SimpleNamespace(
        root=root,
        name=name,
        sources=sources,
        packager=packager,
        publish=lambda: packager.package_bundle(argument, spec),
    )


@pytest.fixture
def digest_reads(publication, monkeypatch):
    import inference_manifest.integrity as integrity

    counts = Counter()

    def count(original):
        def hash_file(path, **kwargs):
            counts[path] += 1
            return original(path, **kwargs)

        return hash_file

    monkeypatch.setattr(integrity, "_sha256_file", count(integrity._sha256_file))
    return counts


@pytest.fixture
def inode_digest_reads(monkeypatch):
    import inference_manifest.integrity as integrity

    counts = Counter()
    original = integrity._sha256_file

    def count(path):
        info = path.stat()
        counts[info.st_dev, info.st_ino] += 1
        return original(path)

    monkeypatch.setattr(integrity, "_sha256_file", count)
    return counts


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
@pytest.mark.parametrize("multibyte", [False, True])
def test_publication_supports_filesystem_maximum_artifact_name(publication, multibyte):
    spec = publication.packager._specs()["ram_plus"]
    name_max = os.pathconf(publication.root, "PC_NAME_MAX")
    stem_bytes = name_max - len(".om")
    stem = "模" * (stem_bytes // 3) + "m" * (stem_bytes % 3) if multibyte else "m" * stem_bytes
    artifact = f"artifacts/ascend/{stem}.om"
    selected = replace(spec.compiled[0], artifact=artifact)
    spec = replace(spec, compiled=(selected,))
    source = next(iter(publication.sources.values()))
    source.parent.mkdir(parents=True, exist_ok=True)
    destination = publication.root / artifact

    for content in (b"first compilation", b"updated compilation"):
        source.write_bytes(content)
        publication.packager.package_bundle(publication.root, spec)
        assert destination.read_bytes() == content
        assert list(destination.parent.iterdir()) == [destination]
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, selected.deployment))


def test_failed_artifact_copy_preserves_destination_and_removes_temporary(publication, monkeypatch):
    import perception_service.bundle_publication as publication_helpers

    publication.publish()
    source = next(iter(publication.sources.values()))
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"new compilation")
    before = _bundle_snapshot(publication.root)

    def interrupted_copy(source, destination):
        destination.write_bytes(b"partial copy")
        raise OSError("injected copy failure")

    monkeypatch.setattr(publication_helpers.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before
    validate_deployment_artifact_integrity(load_inference_manifest(publication.root, publication.name))


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
@pytest.mark.parametrize(
    ("published_before", "layout"),
    [(False, "required"), (True, "required"), (False, "extra"), (True, "extra"), (True, "ancestor")],
)
def test_retained_asset_escape_is_rejected_before_reading_or_writing(
    publication, tmp_path, monkeypatch, published_before, layout
):
    root = publication.root
    relative = (
        publication.packager._specs()["ram_plus"].required_paths[0]
        if layout == "required"
        else "assets/retained/metadata.json"
    )
    asset = root / relative
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"original asset")
    if published_before:
        publication.publish()
    external = tmp_path / "outside"
    external.mkdir()
    (external / asset.name).write_bytes(b"external asset must not be read")
    asset.unlink()
    if layout == "ancestor":
        asset.parent.rmdir()
        asset.parent.symlink_to(external, target_is_directory=True)
    else:
        asset.symlink_to(external / asset.name)
    # A pending artifact update must not run after an invalid asset is found.
    source = next(iter(publication.sources.values()))
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"new compilation")
    before = _bundle_snapshot(root)
    original = publication.packager.sha256_file

    def checked_digest(path):
        if path.resolve().is_relative_to(external):
            pytest.fail("publication must reject external assets before reading their bytes")
        return original(path)

    def unexpected_write(*args, **kwargs):
        pytest.fail("publication must reject invalid assets before writing metadata")

    monkeypatch.setattr(publication.packager, "sha256_file", checked_digest)
    monkeypatch.setattr(Path, "write_text", unexpected_write)
    with pytest.raises(ManifestIntegrityError, match="retained asset.*escapes.*bundle root"):
        publication.publish()

    assert _bundle_snapshot(root) == before
    assert (external / asset.name).read_bytes() == b"external asset must not be read"


@pytest.mark.parametrize("publication", ["ascend"], indirect=True)
def test_first_ascend_publication_checks_preexisting_extra_assets(publication, tmp_path, monkeypatch):
    external = tmp_path / "outside.json"
    external.write_bytes(b"external metadata")
    (publication.root / "assets").mkdir(exist_ok=True)
    (publication.root / "assets/retained.json").symlink_to(external)
    before = _bundle_snapshot(publication.root)

    def unexpected_write(*args, **kwargs):
        pytest.fail("publication must reject invalid assets before writing metadata")

    monkeypatch.setattr(Path, "write_text", unexpected_write)
    with pytest.raises(ManifestIntegrityError, match="retained asset.*escapes.*bundle root"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before
    assert external.read_bytes() == b"external metadata"


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
@pytest.mark.parametrize("damage", ["missing", "directory", "broken_symlink"])
def test_previously_declared_asset_is_validated_even_when_discovery_omits_it(publication, damage):
    asset = publication.root / "assets/retained.json"
    asset.write_bytes(b"retained metadata")
    publication.publish()
    asset.unlink()
    if damage == "directory":
        asset.mkdir()
    elif damage == "broken_symlink":
        asset.symlink_to("missing.json")
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="retained asset.*assets/retained.json"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
def test_internal_retained_asset_links_and_missing_regenerated_metadata_remain_supported(publication):
    root = publication.root
    target = root / "retained.json"
    target.write_bytes(b"internal retained metadata")
    asset = root / "assets/retained.json"
    asset.symlink_to("../retained.json")
    publication.publish()
    for relative in ("assets/adapter.json", "assets/artifact-digests.json"):
        (root / relative).unlink()

    publication.publish()

    validated = load_inference_manifest(root, publication.name)
    validate_deployment_artifact_integrity(validated)
    assert "assets/retained.json" in {entry.path for entry in validated.manifest.bundle.files}
    assert asset.is_symlink()
    assert target.read_bytes() == b"internal retained metadata"


@pytest.mark.parametrize("publication", ["ordinary", "ascend"], indirect=True)
@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_retained_asset_alias_to_manifest_is_rejected_before_writes(publication, intermediate, declared):
    asset = publication.root / "assets/manifest-copy.json"
    if declared:
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"retained metadata")
    publication.publish()
    root = publication.root
    manifest = root / "inference_manifest.json"
    link = manifest
    if intermediate:
        link = root / "manifest-alias.json"
        link.symlink_to(manifest.name)
    asset.unlink(missing_ok=True)
    asset.symlink_to(os.path.relpath(link, asset.parent))
    before = _bundle_snapshot(root)

    with pytest.raises(ManifestIntegrityError, match="retained asset.*manifest-copy.json.*replaced target"):
        publication.publish()

    assert _bundle_snapshot(root) == before


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
@pytest.mark.parametrize("directory_link", [False, True])
def test_directory_import_target_is_rejected_before_mutating_retained_descendant(publication, directory_link):
    publication.publish()
    destination, source = next(iter(publication.sources.items()))
    destination.unlink()
    if directory_link:
        store = publication.root / "store"
        store.mkdir()
        destination.symlink_to(store, target_is_directory=True)
    else:
        destination.mkdir()
    helper = destination / "helper.om"
    helper.write_bytes(b"retained model")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"new compilation")
    manifest_path = publication.root / "inference_manifest.json"
    raw = json.loads(manifest_path.read_text())
    previous = raw["deployments"][publication.name]
    previous["artifacts"]["model"]["path"] = helper.relative_to(publication.root).as_posix()
    previous["artifacts"]["model"]["sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    raw["deployments"]["retained"] = raw["deployments"].pop(publication.name)
    manifest_path.write_text(json.dumps(raw))
    validate_deployment_artifact_integrity(load_inference_manifest(publication.root, "retained"))
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="publication target.*directory"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before
    validate_deployment_artifact_integrity(load_inference_manifest(publication.root, "retained"))


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
def test_packaging_wraps_published_artifact_symlink_loop_before_writes(publication):
    publication.publish()
    destination = next(iter(publication.sources))
    destination.unlink()
    destination.symlink_to(destination.name)
    before = _bundle_snapshot(publication.root)
    with pytest.raises(
        ManifestIntegrityError, match="artifact_integrity_unverified.*deployment=.*role=.*path="
    ) as error:
        publication.publish()
    assert isinstance(error.value.__cause__, OSError)
    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("target", ["../outside.json", "/outside.json"])
def test_preflight_keeps_integrity_error_for_invalid_asset_targets(tmp_path, target):
    with pytest.raises(ManifestIntegrityError, match="publication target.*invalid or escapes"):
        preflight_publication(tmp_path, PublishedArtifactIndex(tmp_path, None), {}, set(), asset_writes=(target,))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
@pytest.mark.parametrize("published_before", [False, True])
def test_adapter_cannot_overwrite_plain_torch_weights(tmp_path, alias, published_before):
    from perception_service.package_perception_bundles import _specs, package_bundle

    spec = _specs()["sam2"]
    root = tmp_path / spec.name
    weights = root / spec.required_paths[0]
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"torch checkpoint")
    if published_before:
        package_bundle(root, spec)
    adapter = root / "assets/adapter.json"
    adapter.unlink(missing_ok=True)
    if alias == "symlink":
        adapter.symlink_to(weights.name)
    else:
        adapter.hardlink_to(weights)
    before = _bundle_snapshot(root)

    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        package_bundle(root, spec)

    assert _bundle_snapshot(root) == before


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
@pytest.mark.parametrize("metadata", ["inference_manifest.json", "inference_integrity.json"])
def test_asset_cannot_overwrite_publication_metadata(publication, alias, metadata):
    publication.publish()
    target = publication.root / metadata
    if not target.exists():
        verify_deployment_artifacts(publication.root, publication.name)
    adapter = publication.root / "assets/adapter.json"
    adapter.unlink()
    if alias == "symlink":
        adapter.symlink_to(target)
    else:
        adapter.hardlink_to(target)
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ascend"], indirect=True)
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_asset_source_cannot_depend_on_another_asset_write(publication, tmp_path, alias):
    publication.publish()
    source = tmp_path / "new_asset.bin"
    source.write_bytes(b"replacement")
    destination = publication.root / "assets/extra.bin"
    destination.write_bytes(b"old asset")
    read_source = destination
    if alias != "direct":
        read_source = tmp_path / "asset_alias.bin"
        if alias == "symlink":
            read_source.symlink_to(destination)
        else:
            read_source.hardlink_to(destination)
    spec = replace(
        publication.packager._specs()["sam2"],
        assets=((str(source), "assets/extra.bin"), (str(read_source), "assets/copied.bin")),
    )
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        publication.packager.package_bundle(tmp_path, spec)

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ascend"], indirect=True)
def test_asset_source_cannot_depend_on_a_replaced_artifact(publication, tmp_path):
    publication.publish()
    artifact, source = next(iter(publication.sources.items()))
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"replacement OM")
    spec = replace(publication.packager._specs()["sam2"], assets=((str(artifact), "assets/from_artifact.bin"),))
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="asset or metadata.*aliases a replaced target"):
        publication.packager.package_bundle(tmp_path, spec)

    assert _bundle_snapshot(publication.root) == before


def test_sidecar_formatting_follows_existing_bundle_revision_policy(publication):
    publication.publish()
    first = load_inference_manifest(publication.root, publication.name)
    sidecar = publication.root / "assets/artifact-digests.json"
    sidecar.write_text(json.dumps(json.loads(sidecar.read_text()), separators=(",", ":")))

    publication.publish()

    second = load_inference_manifest(publication.root, publication.name)
    text_comparison = publication.packager.__name__.endswith("package_perception_bundles")
    assert second.manifest.deployments == first.manifest.deployments
    assert second.manifest.bundle.revision == first.manifest.bundle.revision + int(text_comparison)
    assert (second.manifest.bundle.digest != first.manifest.bundle.digest) == text_comparison
    assert (second.fingerprint != first.fingerprint) == text_comparison


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_asset_write_targets_cannot_alias_each_other(publication, alias):
    publication.publish()
    adapter = publication.root / "assets/adapter.json"
    sidecar = publication.root / "assets/artifact-digests.json"
    sidecar.unlink()
    if alias == "symlink":
        sidecar.symlink_to(adapter.name)
    else:
        sidecar.hardlink_to(adapter)
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="publication asset.*conflict"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


def test_missing_artifact_under_external_parent_is_rejected_before_writes(publication, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = next(iter(publication.sources))
    # Move only this deployment's parent; leave all other candidates available.
    parent = destination.parent
    for target, source in publication.sources.items():
        if target.parent == parent:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"independent compilation")
            target.unlink()
    parent.rmdir()
    parent.symlink_to(outside, target_is_directory=True)
    before = _bundle_snapshot(publication.root), _bundle_snapshot(outside)

    with pytest.raises(ManifestIntegrityError, match="escapes.*bundle root"):
        publication.publish()

    assert (_bundle_snapshot(publication.root), _bundle_snapshot(outside)) == before


@pytest.mark.parametrize("target", ["adapter", "asset_parent"])
def test_external_asset_target_is_rejected_before_writes(publication, tmp_path, target):
    outside = tmp_path / "outside"
    outside.mkdir()
    assets = publication.root / "assets"
    assets.mkdir(exist_ok=True)
    if target == "adapter":
        external = outside / "adapter.json"
        external.write_text("{}")
        (assets / "adapter.json").symlink_to(external)
    else:
        assets.rename(outside / "assets")
        assets.symlink_to(outside / "assets", target_is_directory=True)
    before = _bundle_snapshot(publication.root), _bundle_snapshot(outside)

    with pytest.raises(ManifestIntegrityError, match="escapes.*bundle root"):
        publication.publish()

    assert (_bundle_snapshot(publication.root), _bundle_snapshot(outside)) == before


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
@pytest.mark.parametrize("asset_name", ["adapter.json", "artifact-digests.json"])
def test_asset_writes_cannot_modify_retained_artifacts(publication, alias, asset_name):
    publication.publish()
    artifact = next(iter(publication.sources))
    asset = publication.root / "assets" / asset_name
    asset.unlink()
    # Keep valid JSON for the Ascend sidecar reader while retaining valid old SHA evidence.
    artifact.write_bytes(b"{}")
    if alias == "symlink":
        asset.symlink_to(artifact)
    else:
        asset.hardlink_to(artifact)
    manifest_path = publication.root / "inference_manifest.json"
    raw = json.loads(manifest_path.read_text())
    for deployment in raw["deployments"].values():
        for entry in deployment.get("artifacts", {}).values():
            if entry["path"] == artifact.relative_to(publication.root).as_posix():
                entry["sha256"] = hashlib.sha256(b"{}").hexdigest()
    manifest_path.write_text(json.dumps(raw))
    validate_deployment_artifact_integrity(load_inference_manifest(publication.root, publication.name))
    before = _bundle_snapshot(publication.root)
    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        publication.publish()
    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ascend"], indirect=True)
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_copied_assets_cannot_modify_retained_artifacts(publication, alias):
    publication.publish()
    artifact = next(iter(publication.sources))
    asset = publication.root / "assets/extra.bin"
    if alias == "symlink":
        asset.symlink_to(artifact)
    else:
        asset.hardlink_to(artifact)
    source = publication.root.parent / "replacement.bin"
    source.write_bytes(b"replacement asset")
    spec = replace(publication.packager._specs()["sam2"], assets=((source.name, "assets/extra.bin"),))
    before = _bundle_snapshot(publication.root)
    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        publication.packager.package_bundle(publication.root.parent, spec)
    assert _bundle_snapshot(publication.root) == before


def test_sidecar_repair_preserves_deployment_revision_but_revises_bundle_identity(publication, digest_reads):
    root, artifacts = publication.root, set(publication.sources)
    publication.publish()
    assert {path: digest_reads[path] for path in artifacts} == dict.fromkeys(artifacts, 1)
    first_spec = load_inference_manifest(root, publication.name)
    first = first_spec.manifest
    sidecar = root / "assets/artifact-digests.json"
    digests = json.loads(sidecar.read_text())
    for deployment in first.deployments.values():
        for artifact in deployment.artifacts.values():
            assert digests[artifact.path] == artifact.sha256
            digests[artifact.path] = "0" * 64
    sidecar.write_text(json.dumps(digests))
    digest_reads.clear()
    publication.publish()
    second_spec = load_inference_manifest(root, publication.name)
    second = second_spec.manifest
    assert second.deployments == first.deployments
    # The existing bundle revision policy includes sidecar repairs. Only the
    # deployment declaration's revision authority moved into the manifest.
    assert second.bundle.revision == first.bundle.revision + 1
    assert second.bundle.digest != first.bundle.digest
    assert second_spec.fingerprint != first_spec.fingerprint
    assert {path: digest_reads[path] for path in artifacts} == dict.fromkeys(artifacts, 1)
    publication.publish()
    assert load_inference_manifest(root, publication.name).fingerprint == second_spec.fingerprint

    # A legacy declaration acquires a digest once, then remains stable.
    path = root / "inference_manifest.json"
    legacy = json.loads(path.read_text())
    for deployment in legacy["deployments"].values():
        for artifact in deployment.get("artifacts", {}).values():
            artifact.pop("sha256", None)
    path.write_text(json.dumps(legacy))
    publication.publish()
    migrated = load_inference_manifest(root, publication.name).manifest
    for name, previous in second.deployments.items():
        assert migrated.deployments[name].revision == previous.revision + int(bool(previous.artifacts))
    publication.publish()
    assert load_inference_manifest(root, publication.name).manifest.deployments == migrated.deployments


def test_repackaging_with_sources_counts_comparison_and_finalization_reads(publication, digest_reads):
    for source in publication.sources.values():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"compiled")
    publication.publish()
    digest_reads.clear()
    publication.publish()
    for destination, source in publication.sources.items():
        assert digest_reads[source] == 1
        # Unchanged imports reuse the comparison digest for the manifest and sidecar.
        assert digest_reads[destination] == 1


def test_republication_rejects_corrupt_retained_artifact_after_sources_are_removed(publication):
    for source in publication.sources.values():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"compiled")
    publication.publish()
    manifest_path = publication.root / "inference_manifest.json"
    before = manifest_path.read_bytes()
    for source in publication.sources.values():
        source.unlink()
    next(iter(publication.sources)).write_bytes(b"corrupted")
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
        publication.publish()
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_compiler_alias_preserves_digest_and_rejects_corruption(publication, alias):
    publication.publish()
    destination, source = next(iter(publication.sources.items()))
    source.parent.mkdir(parents=True, exist_ok=True)
    if alias == "symlink":
        source.symlink_to(destination)
    else:
        source.hardlink_to(destination)
    previous = load_inference_manifest(publication.root, publication.name).deployment
    publication.publish()
    assert load_inference_manifest(publication.root, publication.name).deployment == previous
    destination.write_bytes(b"corrupt aliased OM")
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize(
    ("publication", "retained_index"), [("ordinary", 1), ("ascend", 1), ("ascend", -1)], indirect=["publication"]
)
def test_partial_publication_checks_retained_roles_and_accepts_new_sources(publication, retained_index):
    publication.publish()
    paths = list(publication.sources)
    updated, retained = paths[0], paths[retained_index]
    source = publication.sources[updated]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"new compilation")
    original = retained.read_bytes()
    retained.write_bytes(b"corrupt retained role")
    before = _bundle_snapshot(publication.root)
    with pytest.raises(ManifestIntegrityError, match="retained artifact"):
        publication.publish()
    assert _bundle_snapshot(publication.root) == before
    retained.write_bytes(original)
    # Restoring only the damaged retained role must restore the old publication.
    previous = load_inference_manifest(publication.root, publication.name).manifest
    for name in previous.deployments:
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, name))
    publication.publish()
    validated = load_inference_manifest(publication.root, publication.name)
    for name in validated.manifest.deployments:
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, name))
    assert updated.read_bytes() == b"new compilation"
    assert retained.read_bytes() == original


def test_updating_one_hardlinked_target_preserves_other_published_artifacts(publication):
    publication.publish()
    updated, retained = list(publication.sources)[:2]
    retained.unlink()
    retained.hardlink_to(updated)
    original = retained.read_bytes()
    source = publication.sources[updated]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"new compiled output")

    publication.publish()

    assert updated.read_bytes() == source.read_bytes()
    assert retained.read_bytes() == original
    assert not updated.samefile(retained)
    manifest = load_inference_manifest(publication.root, publication.name).manifest
    for name in manifest.deployments:
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, name))


@pytest.mark.parametrize("intermediate_link", [False, True])
def test_updating_a_symlink_target_is_rejected_before_any_bundle_write(publication, intermediate_link):
    publication.publish()
    updated_path, retained_path = list(publication.sources)[:2]
    updated_source = publication.sources[updated_path]
    if intermediate_link:
        target = updated_path.with_suffix(".target.om")
        target.write_bytes(updated_path.read_bytes())
        updated_path.unlink()
        updated_path.symlink_to(target.name)
    retained_path.unlink()
    retained_path.symlink_to(os.path.relpath(updated_path, retained_path.parent))
    updated_source.parent.mkdir(parents=True, exist_ok=True)
    updated_source.write_bytes(b"new compilation")
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="retained artifact.*aliases a replaced target") as error:
        publication.publish()

    manifest = load_inference_manifest(publication.root, publication.name).manifest
    name, role = next(
        (name, role)
        for name, deployment in manifest.deployments.items()
        for role, artifact in deployment.artifacts.items()
        if publication.root / artifact.path == retained_path
    )
    assert f"deployment={name!r}" in str(error.value)
    assert f"role={role!r}" in str(error.value)
    assert f"path={str(retained_path.relative_to(publication.root))!r}" in str(error.value)
    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
def test_source_symlink_to_another_replaced_target_is_rejected_before_writes(publication):
    publication.publish()
    updated_path, imported_path = list(publication.sources)[:2]
    updated_source = publication.sources[updated_path]
    imported_source = publication.sources[imported_path]
    imported_source.parent.mkdir(parents=True, exist_ok=True)
    imported_source.unlink(missing_ok=True)
    imported_source.symlink_to(os.path.relpath(updated_path, imported_source.parent))
    updated_source.parent.mkdir(parents=True, exist_ok=True)
    updated_source.write_bytes(b"new compilation")
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="source.*aliases another replaced artifact") as error:
        publication.publish()

    manifest = load_inference_manifest(publication.root, publication.name).manifest
    name, role = next(
        (name, role)
        for name, deployment in manifest.deployments.items()
        for role, artifact in deployment.artifacts.items()
        if publication.root / artifact.path == imported_path
    )
    assert f"deployment={name!r}" in str(error.value)
    assert f"role={role!r}" in str(error.value)
    assert f"path={str(imported_source)!r}" in str(error.value)
    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
def test_preflight_verifies_retained_artifacts_in_unselected_deployments(publication):
    publication.publish()
    manifest = load_inference_manifest(publication.root, publication.name).manifest
    retained_name = next(
        name for name, deployment in manifest.deployments.items() if name != publication.name and deployment.artifacts
    )
    retained = manifest.deployments[retained_name]
    retained_artifact = next(iter(retained.artifacts.values()))
    (publication.root / retained_artifact.path).write_bytes(b"corrupt unselected artifact")
    spec = publication.packager._specs()["ram_plus"]
    selected = next(item for item in spec.compiled if item.deployment == publication.name)
    source = publication.sources[publication.root / selected.artifact]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"new compilation")
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
        publication.packager.package_bundle(publication.root, replace(spec, compiled=(selected,)))

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
def test_cross_artifact_published_source_is_hashed_once(publication, inode_digest_reads):
    publication.publish()
    source_paths = list(publication.sources.items())
    destination_a, source_a = source_paths[0]
    destination_b, source_b = source_paths[1]
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(destination_a.read_bytes())
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_b.unlink(missing_ok=True)
    source_b.symlink_to(os.path.relpath(destination_a, source_b.parent))
    inode_digest_reads.clear()

    publication.publish()

    info = destination_a.stat()
    assert inode_digest_reads[info.st_dev, info.st_ino] == 1
    manifest = load_inference_manifest(publication.root, publication.name).manifest
    for name in manifest.deployments:
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, name))


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_cross_artifact_source_is_checked_before_any_replacement(publication, alias):
    publication.publish()
    old, target = list(publication.sources)[:2]
    old.write_bytes(b"damaged published artifact")
    # The old artifact has its own independent replacement, so merely checking
    # retained target roles would miss its use as another role/deployment's input.
    replacement = publication.sources[old]
    replacement.parent.mkdir(parents=True, exist_ok=True)
    replacement.write_bytes(b"new independent compilation")
    source = publication.sources[target]
    source.parent.mkdir(parents=True, exist_ok=True)
    if alias == "symlink":
        source.symlink_to(old)
    else:
        source.hardlink_to(old)
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


def test_equal_import_symlink_cannot_follow_a_later_replacement(publication):
    publication.publish()
    retained, updated = list(publication.sources)[:2]
    retained.unlink()
    retained.symlink_to(os.path.relpath(updated, retained.parent))
    for destination, content in ((retained, retained.read_bytes()), (updated, b"new compilation")):
        source = publication.sources[destination]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="retained artifact.*aliases a replaced target"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before
    manifest = load_inference_manifest(publication.root, publication.name).manifest
    for name in manifest.deployments:
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, name))


def test_independent_replacements_do_not_hide_conflicting_target_evidence(publication):
    publication.publish()
    first, second = list(publication.sources)[:2]
    second.unlink()
    second.hardlink_to(first)
    manifest_path = publication.root / "inference_manifest.json"
    raw = json.loads(manifest_path.read_text())
    for deployment in raw["deployments"].values():
        for artifact in deployment.get("artifacts", {}).values():
            if artifact["path"] == second.relative_to(publication.root).as_posix():
                artifact["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(raw))
    for destination in (first, second):
        source = publication.sources[destination]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"independent replacement")
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="conflicting retained artifact digests"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


def test_first_publication_has_bounded_source_and_target_digest_reads(publication, track_digest_reads):
    for destination, source in publication.sources.items():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(destination.read_bytes())
        destination.unlink()

    # Measure digest/comparison bytes, excluding copy2's source reads.
    # Allow a source-validation pass as well as final target verification.
    tracked = set(publication.sources) | set(publication.sources.values())
    bytes_read = track_digest_reads(tracked)
    publication.publish()

    for destination, source in publication.sources.items():
        assert bytes_read[source] <= source.stat().st_size
        assert 0 < bytes_read[destination] <= 2 * destination.stat().st_size
    manifest = load_inference_manifest(publication.root, publication.name).manifest
    for name in manifest.deployments:
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, name))


@pytest.mark.parametrize("publication", ["ordinary", "ascend", "graspgen"], indirect=True)
def test_updated_artifact_skips_reading_old_target_bytes(publication, track_digest_reads):
    publication.publish()
    destination, source = next(iter(publication.sources.items()))
    old_content = b"old published artifact" * 65536
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(old_content)
    publication.publish()
    assert destination.read_bytes() == old_content

    new_content = b"new compilation"
    source.write_bytes(new_content)
    bytes_read = track_digest_reads({destination})
    publication.publish()
    destination_bytes_read = bytes_read[destination]

    validated = load_inference_manifest(publication.root, publication.name)
    relative_path = destination.relative_to(publication.root).as_posix()
    artifact = next(artifact for artifact in validated.deployment.artifacts.values() if artifact.path == relative_path)
    assert artifact.sha256 == hashlib.sha256(new_content).hexdigest()
    assert destination_bytes_read == len(new_content)


@pytest.mark.parametrize("publication", ["ordinary", "ascend", "graspgen"], indirect=True)
def test_legacy_tail_update_compares_once_before_independent_finalization(publication, track_digest_reads):
    destination, source = next(iter(publication.sources.items()))
    old_content = b"x" * (3 * 1024 * 1024)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(old_content)
    publication.publish()
    manifest_path = publication.root / "inference_manifest.json"
    raw = json.loads(manifest_path.read_text())
    for deployment in raw["deployments"].values():
        for artifact in deployment.get("artifacts", {}).values():
            artifact.pop("sha256", None)
    manifest_path.write_text(json.dumps(raw))

    new_content = old_content[:-1] + b"!"
    source.write_bytes(new_content)
    reads = track_digest_reads({source, destination})
    publication.publish()
    # One comparison, plus independent verification of the copied destination.
    assert reads[source] == len(new_content)
    assert reads[destination] == 2 * len(new_content)
    validated = load_inference_manifest(publication.root, publication.name)
    relative = destination.relative_to(publication.root).as_posix()
    artifact = next(item for item in validated.deployment.artifacts.values() if item.path == relative)
    assert artifact.sha256 == hashlib.sha256(new_content).hexdigest()
    assert destination.read_bytes() == new_content


@pytest.mark.parametrize("indexed", [False, True])
def test_preflight_checks_retained_digest_without_caller_finalization(publication, indexed):
    publication.publish()
    manifest = load_inference_manifest(publication.root, publication.name).manifest
    artifact = next(iter(manifest.deployments[publication.name].artifacts.values()))
    (publication.root / artifact.path).write_bytes(b"changed retained artifact")
    before = _bundle_snapshot(publication.root)
    existing = manifest if indexed else None

    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch"):
        preflight_publication(
            publication.root,
            PublishedArtifactIndex(publication.root, existing),
            {publication.name: ArtifactPublication({"model": artifact}, {})},
            set(manifest.deployments),
        )

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ascend"], indirect=True)
@pytest.mark.parametrize(("published_before", "targets_exist"), [(False, False), (False, True), (True, True)])
def test_parent_symlinks_cannot_merge_distinct_import_targets(publication, published_before, targets_exist):
    if published_before:
        publication.publish()
    destinations = list(publication.sources)
    first_parent, second_parent = dict.fromkeys(path.parent for path in destinations)
    for destination, source in publication.sources.items():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(destination.parent.name.encode())
        if destination.parent == second_parent:
            shared_path = first_parent / destination.name
            if not shared_path.exists():
                shared_path.write_bytes(destination.read_bytes())
            destination.unlink()
    second_parent.rmdir()
    second_parent.symlink_to(first_parent, target_is_directory=True)
    if not targets_exist:
        for path in first_parent.iterdir():
            path.unlink()
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="publication target.*conflict"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ascend"], indirect=True)
def test_parent_symlinks_cannot_nest_import_file_targets(publication):
    destinations = list(publication.sources)
    first_parent, second_parent = dict.fromkeys(path.parent for path in destinations)
    for destination, source in publication.sources.items():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(destination.read_bytes())
        destination.unlink()
    second_parent.rmdir()
    second_parent.symlink_to(destinations[0], target_is_directory=True)
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="publication target.*conflict"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("kind", ["file", "symlink", "fifo"])
def test_preflight_rejects_non_directory_parent_components(tmp_path, kind):
    parent = tmp_path / "assets"
    if kind == "fifo":
        os.mkfifo(parent)
    elif kind == "symlink":
        target = tmp_path / "regular-file"
        target.write_bytes(b"asset")
        parent.symlink_to(target)
    else:
        parent.write_bytes(b"asset")
    before = _bundle_snapshot(tmp_path)

    with pytest.raises(ManifestIntegrityError, match="publication target.*directory"):
        preflight_publication(
            tmp_path, PublishedArtifactIndex(tmp_path, None), {}, set(), asset_writes=("assets/adapter.json",)
        )

    assert _bundle_snapshot(tmp_path) == before


@pytest.mark.parametrize("through_symlink", [False, True])
def test_preflight_rejects_nested_asset_targets(tmp_path, through_symlink):
    if through_symlink:
        (tmp_path / "adapter.json").symlink_to("future-file")
        targets = ("adapter.json", "future-file/child.json")
    else:
        targets = ("future-file", "future-file/child.json")
    before = _bundle_snapshot(tmp_path)

    with pytest.raises(ManifestIntegrityError, match="publication target.*conflict"):
        preflight_publication(tmp_path, PublishedArtifactIndex(tmp_path, None), {}, set(), asset_writes=targets)

    assert _bundle_snapshot(tmp_path) == before


@pytest.mark.parametrize("publication", ["ordinary", "ascend"], indirect=True)
@pytest.mark.parametrize("kind", ["fifo", "socket"])
@pytest.mark.parametrize("asset", ["adapter.json", "artifact-digests.json"])
def test_special_asset_targets_are_rejected_before_writes(publication, kind, asset, monkeypatch):
    target = publication.root / "assets" / asset
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "fifo":
        os.mkfifo(target)
    else:
        with socket.socket(socket.AF_UNIX) as endpoint, monkeypatch.context() as context:
            context.chdir(target.parent)
            endpoint.bind(target.name)
    before = _bundle_snapshot(publication.root)

    def timed_out(*args):
        raise TimeoutError("publication blocked on a special asset target")

    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(3)
    try:
        with pytest.raises(ManifestIntegrityError, match="publication target.*regular file"):
            publication.publish()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("publication", ["ordinary", "ascend", "graspgen"], indirect=True)
def test_restoring_original_artifact_bytes_discards_stale_mismatch(publication):
    publication.publish()
    destination, source = next(iter(publication.sources.items()))
    original = destination.read_bytes()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(original)
    old = load_inference_manifest(publication.root, publication.name).deployment
    destination.write_bytes(b"corrupted after publication")
    assert verify_deployment_artifacts(publication.root, publication.name).state == "mismatch"

    publication.publish()

    validated = load_inference_manifest(publication.root, publication.name)
    assert validated.deployment == old
    assert destination.read_bytes() == original
    assert validated.integrity_status.state == "declared"
    validate_deployment_artifact_integrity(validated)


def test_republication_rejects_retained_asset_depending_on_discarded_report(publication):
    publication.publish()
    verify_deployment_artifacts(publication.root, publication.name)
    (publication.root / "assets" / "report-copy.json").symlink_to("../inference_integrity.json")
    before = _bundle_snapshot(publication.root)

    with pytest.raises(ManifestIntegrityError, match="retained asset.*aliases a replaced target"):
        publication.publish()

    assert _bundle_snapshot(publication.root) == before


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
@pytest.mark.parametrize("use_as_source", [False, True])
def test_retained_aliases_share_one_content_read(publication, inode_digest_reads, alias, use_as_source):
    publication.publish()
    first, second = list(publication.sources)[:2]
    linked = publication.sources[second] if use_as_source else second
    linked.parent.mkdir(parents=True, exist_ok=True)
    linked.unlink(missing_ok=True)
    if alias == "symlink":
        linked.symlink_to(first)
    else:
        linked.hardlink_to(first)
    inode_digest_reads.clear()
    publication.publish()
    info = first.stat()
    assert inode_digest_reads[info.st_dev, info.st_ino] == 1
    manifest = load_inference_manifest(publication.root, publication.name).manifest
    for name in manifest.deployments:
        validate_deployment_artifact_integrity(load_inference_manifest(publication.root, name))


@pytest.mark.parametrize("publication", ["ordinary"], indirect=True)
@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("unselected", [False, True])
def test_legacy_artifact_depending_on_manifest_is_rejected_before_writes(publication, intermediate, unselected):
    publication.publish()
    manifest_path = publication.root / "inference_manifest.json"
    raw = json.loads(manifest_path.read_text())
    artifact = raw["deployments"][publication.name]["artifacts"]["model"]
    artifact.pop("sha256")
    target = publication.root / artifact["path"]
    target.unlink()
    link = manifest_path
    if intermediate:
        link = publication.root / "manifest-alias"
        link.symlink_to(manifest_path.name)
    target.symlink_to(link)
    manifest_path.write_text(json.dumps(raw))
    before = _bundle_snapshot(publication.root)
    spec = publication.packager._specs()["ram_plus"]
    if unselected:
        spec = replace(spec, compiled=tuple(item for item in spec.compiled if item.deployment != publication.name))

    with pytest.raises(ManifestIntegrityError, match="retained artifact.*aliases a replaced target") as error:
        publication.packager.package_bundle(publication.root, spec)

    assert f"deployment={publication.name!r}" in str(error.value)
    assert "role='model'" in str(error.value)
    assert f"path={artifact['path']!r}" in str(error.value)
    assert _bundle_snapshot(publication.root) == before
