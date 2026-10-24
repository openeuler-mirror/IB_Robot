# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GraspGen packages into a grasp-domain bundle, not a LeRobot policy bundle.

The previous packager wrote ``config.json``, ``policy_preprocessor.json`` and
``policy_postprocessor.json`` so that a non-policy model would survive the policy loader's
LeRobot asset checks, and it wrote them before any of the eight OM ABIs had been read - so
a missing artifact left a bundle that looked packaged and was not. These tests hold the
replacement to the generic runtime shape already used by staged models: the current
compatibility descriptor, the model's own constants in ``assets/adapter.json``, and
nothing written at all until every role has been resolved.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from conftest import (
    GRASPGEN_BATCH,
    GRASPGEN_DEPLOYMENT,
    GRASPGEN_POINTS,
    GRASPGEN_ROLE_ABI,
    _bundle_snapshot,
    package_graspgen_export,
    runtime_abi_json,
    write_json,
)

from inference_manifest import ManifestIntegrityError, load_inference_manifest, validate_deployment_artifact_integrity
from model_utils.graspgen_contract import (
    GRASPGEN_CONTRACT_VERSION,
    GRASPGEN_EXECUTION,
    graspgen_geometry,
)
from model_utils.inference_manifest_export import RuntimeABI, RuntimeTensor
from perception_service.graspgen_adapter import GraspGenAdapter
from perception_service.package_graspgen_ascend_bundle import _TORCH_ASSETS

_LEROBOT_ASSETS = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")


def test_first_graspgen_publication_checks_preexisting_extra_assets(graspgen_export, tmp_path, monkeypatch):
    export = graspgen_export
    external = tmp_path / "outside.json"
    external.write_bytes(b"external metadata")
    asset = export.bundle / "assets/retained.json"
    asset.parent.mkdir(parents=True)
    asset.symlink_to(external)
    before = _bundle_snapshot(export.bundle)
    original = Path.write_text

    def checked_write(path, *args, **kwargs):
        if path.is_relative_to(export.bundle):
            pytest.fail("publication must reject invalid assets before writing metadata")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", checked_write)
    with pytest.raises(ManifestIntegrityError, match="retained asset.*escapes.*bundle root"):
        package_graspgen_export(export)

    assert _bundle_snapshot(export.bundle) == before
    assert asset.is_symlink()
    assert external.read_bytes() == b"external metadata"


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_graspgen_asset_source_cannot_alias_another_write(graspgen_export, tmp_path, alias):
    export = graspgen_export
    package_graspgen_export(export)
    generator = export.bundle / "assets/generator_checkpoint.pth"
    source = generator
    if alias != "direct":
        source = tmp_path / "generator_alias.pth"
        if alias == "symlink":
            source.symlink_to(generator)
        else:
            source.hardlink_to(generator)
    export.onnx_manifest["source"]["discriminator_checkpoint"] = str(source)
    original = Path(export.onnx_manifest["source"]["generator_checkpoint"])
    original.write_bytes(b"new generator checkpoint")
    before = _bundle_snapshot(export.bundle)

    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        package_graspgen_export(export)

    assert _bundle_snapshot(export.bundle) == before


def test_graspgen_accepts_assets_read_from_their_own_destinations(graspgen_export):
    export = graspgen_export
    package_graspgen_export(export)
    first = load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT)
    for name, (_, path) in _TORCH_ASSETS.items():
        export.onnx_manifest["source"][name] = str(export.bundle / path)

    package_graspgen_export(export)

    assert load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT).fingerprint == first.fingerprint


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_graspgen_rejects_aliasing_asset_targets_before_writes(graspgen_export, alias):
    export = graspgen_export
    package_graspgen_export(export)
    generator = export.bundle / "assets/generator_checkpoint.pth"
    discriminator = export.bundle / "assets/discriminator_checkpoint.pth"
    discriminator.unlink()
    if alias == "symlink":
        discriminator.symlink_to(generator.name)
    else:
        discriminator.hardlink_to(generator)
    before = _bundle_snapshot(export.bundle)

    with pytest.raises(ManifestIntegrityError, match="publication asset.*conflict"):
        package_graspgen_export(export)

    assert _bundle_snapshot(export.bundle) == before
    assert discriminator.samefile(generator)


@pytest.mark.parametrize("target", ["om_parent", "adapter"])
def test_graspgen_rejects_external_targets_before_writes(graspgen_export, tmp_path, target):
    export = graspgen_export
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "om_parent":
        link = export.bundle / "artifacts/ascend" / GRASPGEN_DEPLOYMENT
        link.parent.mkdir(parents=True)
        link.symlink_to(outside, target_is_directory=True)
    else:
        external = outside / "adapter.json"
        external.write_bytes(b"{}")
        link = export.bundle / "assets/adapter.json"
        link.parent.mkdir(parents=True)
        link.symlink_to(external)
    paths_before = _bundle_snapshot(export.bundle)
    outside_before = _bundle_snapshot(outside)

    with pytest.raises(ManifestIntegrityError, match="escapes.*bundle root"):
        package_graspgen_export(export)

    assert _bundle_snapshot(export.bundle) == paths_before
    assert _bundle_snapshot(outside) == outside_before
    assert link.is_symlink()


def test_the_bundle_declares_a_perception_model_and_no_lerobot_assets(graspgen_export):
    manifest_path = package_graspgen_export(graspgen_export)

    bundle = graspgen_export.bundle
    assert manifest_path == bundle / "inference_manifest.json"
    validated = load_inference_manifest(bundle, GRASPGEN_DEPLOYMENT)
    validate_deployment_artifact_integrity(validated)
    assert validated.manifest.model.interface == "tensor_model"
    assert validated.manifest.model.model_type == "graspgen"
    assert validated.manifest.model.operation == "generate_grasps"
    assert validated.policy is None
    for asset in _LEROBOT_ASSETS:
        assert not (bundle / asset).exists()


def test_the_service_contract_is_the_only_thing_the_descriptor_declares(graspgen_bundle):
    """The poses are integrated on the host, so no OM slot may claim to produce them."""
    validated = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT)

    model = validated.manifest.model
    assert [tensor.semantic for tensor in model.inputs] == ["observation.object_points"]
    assert [tensor.semantic for tensor in model.outputs] == ["grasp.poses", "grasp.confidence"]
    head_outputs = {binding.semantic for binding in validated.deployment.bindings["discriminator_head"].outputs}
    assert head_outputs == {"host.graspgen.grasp_logits", "host.graspgen.grasp_scores"}


def test_the_roles_bind_in_the_shared_execution_order_over_two_device_links(graspgen_bundle):
    deployment = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT).deployment

    assert deployment.execution == GRASPGEN_EXECUTION
    assert deployment.target.soc == "Ascend310P3"
    assert deployment.target.runtime == "acl"
    assert {(link.semantic, link.producer, link.consumer) for link in deployment.device_links} == {
        ("internal.graspgen.generator_embedding", "generator_encoder_head", "denoiser"),
        ("internal.graspgen.discriminator_embedding", "discriminator_encoder_head", "discriminator_head"),
    }
    assert deployment.bindings["denoiser"].inputs[0].semantic == "internal.graspgen.generator_embedding"
    assert (
        deployment.bindings["generator_sa1"].outputs[0].runtime_name
        == "PartitionedCall_/ReduceMax_ReduceMax_1:0:features"
    )
    assert deployment.bindings["generator_sa1"].inputs[0].layout == "NCHW"
    assert deployment.artifacts["denoiser"].path == f"artifacts/ascend/{GRASPGEN_DEPLOYMENT}/denoiser.om"


def test_the_model_constants_land_in_the_adapter_asset_the_adapter_reads_back(graspgen_bundle):
    config = GraspGenAdapter.from_bundle(graspgen_bundle).config

    assert config.kappa == 2.02217
    assert config.diffusion_steps == 10
    assert config.grasp_batch_size == GRASPGEN_BATCH
    assert config.point_count == GRASPGEN_POINTS
    assert (config.npoints, config.radii, config.nsamples) == ((256, 64), (0.02, 0.04), (64, 128))

    manifest = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT).manifest
    assert {entry.path for entry in manifest.bundle.files} == {
        "assets/adapter.json",
        "assets/discriminator_checkpoint.pth",
        "assets/generator_checkpoint.pth",
        "assets/graspgen_config.yml",
    }
    assert set(manifest.deployments) == {"torch_cuda", GRASPGEN_DEPLOYMENT}


def test_the_packaged_geometry_drops_the_encoder_heads_null_stage(graspgen_bundle):
    """The ONNX manifest lists one entry per exported stage; the bundle lists sampled ones."""
    assets = json.loads((graspgen_bundle / "assets" / "adapter.json").read_text(encoding="utf-8"))

    assert assets["geometry"] == graspgen_geometry()
    assert None not in assets["geometry"]["npoints"]


def test_repackaging_an_unchanged_export_keeps_the_bundle_and_deployment_revisions(graspgen_export):
    package_graspgen_export(graspgen_export)
    first = load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT)
    package_graspgen_export(graspgen_export)
    second = load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT)

    assert second.manifest.bundle.uuid == first.manifest.bundle.uuid
    assert second.manifest.bundle.revision == first.manifest.bundle.revision == 1
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision == 1


@pytest.mark.parametrize("publication", ["first", "unchanged", "changed"])
def test_repackaging_keeps_artifact_verification_work_bounded(graspgen_export, monkeypatch, publication):
    import inference_manifest.integrity as integrity

    export = graspgen_export
    first = None
    if publication != "first":
        package_graspgen_export(export)
        first = load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT).deployment
    if publication == "changed":
        for source in export.om_dir.glob("*.om"):
            source.write_bytes(b"new compiled OM")
    reads = Counter()

    def count(original):
        def hash_file(path):
            reads[path] += 1
            return original(path)

        return hash_file

    monkeypatch.setattr(integrity, "_sha256_file", count(integrity._sha256_file))
    package_graspgen_export(export)
    current = load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT)
    for role, artifact in current.deployment.artifacts.items():
        source, destination = export.om_dir / f"{role}.om", export.bundle / artifact.path
        # One source validation and up to two target passes are bounded work,
        # independent of whether first publication chooses to validate sources.
        assert reads[source] <= 1
        assert 1 <= reads[destination] <= 2
        assert destination.read_bytes() == source.read_bytes()
    if first is not None:
        assert current.deployment.revision == first.revision + int(publication == "changed")
    validate_deployment_artifact_integrity(current)


def test_a_role_whose_abi_disagrees_with_its_semantics_is_rejected(graspgen_export):
    bad_abi = RuntimeABI(
        inputs=(
            RuntimeTensor("object_embedding", 0, "float32", (1, 512)),
            RuntimeTensor("sample", 1, "float32", (GRASPGEN_BATCH, 6)),
        ),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 128, 256)),),
    )
    write_json(graspgen_export.om_abi_dir / "denoiser.om.abi.json", runtime_abi_json(bad_abi))

    with pytest.raises(ValueError, match="runtime ABI has 2 inputs; expected 3"):
        package_graspgen_export(graspgen_export)

    assert list(graspgen_export.bundle.iterdir()) == []


def test_a_missing_role_leaves_the_bundle_exactly_as_it_was(graspgen_export):
    """Resolution runs to completion before the first byte is written."""
    (graspgen_export.om_dir / "discriminator_head.om").unlink()

    with pytest.raises(FileNotFoundError, match="discriminator_head"):
        package_graspgen_export(graspgen_export)

    assert list(graspgen_export.bundle.iterdir()) == []


def test_missing_runtime_abis_are_introspected_from_the_compiled_oms(graspgen_export, monkeypatch):
    for abi_path in graspgen_export.om_abi_dir.iterdir():
        abi_path.unlink()
    calls = []

    def fake_write_acl_om_abi(om_path, output_path, *, device_id, acl_config_path):
        role = Path(om_path).stem
        calls.append((role, device_id, acl_config_path))
        write_json(Path(output_path), runtime_abi_json(GRASPGEN_ROLE_ABI[role]))
        return Path(output_path)

    monkeypatch.setattr(
        "perception_service.package_graspgen_ascend_bundle.write_acl_om_abi",
        fake_write_acl_om_abi,
    )

    package_graspgen_export(graspgen_export, abi_device_id=2, acl_config_path="acl.json")

    assert calls == [(role, 2, "acl.json") for role in GRASPGEN_EXECUTION]


def test_introspection_can_be_refused_so_a_host_without_a_device_fails_loudly(graspgen_export):
    (graspgen_export.om_abi_dir / "generator_sa1.om.abi.json").unlink()

    with pytest.raises(FileNotFoundError, match="generator_sa1"):
        package_graspgen_export(graspgen_export, inspect_missing_abi=False)

    assert list(graspgen_export.bundle.iterdir()) == []


def test_an_export_from_another_contract_is_refused_before_anything_is_read(graspgen_export):
    graspgen_export.onnx_manifest["contract_version"] = GRASPGEN_CONTRACT_VERSION + 1

    with pytest.raises(ValueError, match="contract_version"):
        package_graspgen_export(graspgen_export)

    assert list(graspgen_export.bundle.iterdir()) == []


def test_replacing_compiled_bytes_changes_deployment_identity_and_publishes_digests(graspgen_export):
    package_graspgen_export(graspgen_export)
    first = load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT)
    role = GRASPGEN_EXECUTION[0]
    (graspgen_export.om_dir / f"{role}.om").write_bytes(b"replacement compiled model")
    package_graspgen_export(graspgen_export)
    second = load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT)
    validate_deployment_artifact_integrity(second)
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision + 1
    assert second.fingerprint != first.fingerprint
    package_graspgen_export(graspgen_export)
    assert load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT).deployment == second.deployment


@pytest.mark.parametrize("source_alias", ["cli_default", "normalized", "symlink", "hardlink"])
def test_in_place_repackaging_preserves_digests_and_rejects_corruption(graspgen_export, monkeypatch, source_alias):
    from perception_service.package_graspgen_ascend_bundle import main

    export = graspgen_export
    package_graspgen_export(export)
    manifest_path = export.bundle / "inference_manifest.json"
    initial_manifest = manifest_path.read_bytes()
    first = load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT)
    artifact_path = export.bundle / first.deployment.artifacts[GRASPGEN_EXECUTION[0]].path
    om_dir = artifact_path.parent
    if source_alias == "normalized":
        om_dir = om_dir / ".." / om_dir.name
    elif source_alias == "symlink":
        alias = export.bundle.parent / "om_alias"
        alias.symlink_to(om_dir, target_is_directory=True)
        om_dir = alias
    elif source_alias == "hardlink":
        alias = export.bundle.parent / "om_alias"
        alias.mkdir()
        for role in GRASPGEN_EXECUTION:
            (alias / f"{role}.om").hardlink_to(om_dir / f"{role}.om")
        om_dir = alias
    if source_alias == "cli_default":
        onnx_path = export.bundle.parent / "graspgen.onnx.json"
        write_json(onnx_path, export.onnx_manifest)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "package-graspgen",
                "--bundle-root",
                str(export.bundle),
                "--onnx-manifest",
                str(onnx_path),
                "--om-abi-dir",
                str(export.om_abi_dir),
                "--soc-version",
                "Ascend310P3",
            ],
        )

    def republish():
        if source_alias == "cli_default":
            main()
        else:
            package_graspgen_export(export, om_dir=om_dir)

    republish()
    assert manifest_path.read_bytes() == initial_manifest
    artifact_path.write_bytes(b"corrupted published OM")
    before = _bundle_snapshot(export.bundle)
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
        republish()
    assert _bundle_snapshot(export.bundle) == before


def test_mixed_graspgen_sources_are_not_copied_before_retained_roles_pass(graspgen_export):
    export = graspgen_export
    package_graspgen_export(export)
    first = load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT)
    for role in GRASPGEN_EXECUTION[1:]:
        source = export.om_dir / f"{role}.om"
        source.unlink()
        source.hardlink_to(export.bundle / first.deployment.artifacts[role].path)
    (export.om_dir / f"{GRASPGEN_EXECUTION[0]}.om").write_bytes(b"new compilation")
    (export.om_dir / f"{GRASPGEN_EXECUTION[-1]}.om").write_bytes(b"corrupted retained OM")
    before = _bundle_snapshot(export.bundle)

    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
        package_graspgen_export(export)

    assert _bundle_snapshot(export.bundle) == before


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_cross_deployment_om_sources_preserve_old_evidence(graspgen_export, alias, corrupt):
    export = graspgen_export
    package_graspgen_export(export)
    original = export.bundle / "artifacts/ascend" / GRASPGEN_DEPLOYMENT
    sources = original
    if alias != "direct":
        sources = export.bundle.parent / "aliases"
        sources.mkdir()
        for path in original.glob("*.om"):
            if alias == "symlink":
                (sources / path.name).symlink_to(path)
            else:
                (sources / path.name).hardlink_to(path)
    if corrupt:
        (original / "denoiser.om").write_bytes(b"corrupted published OM")
    before = _bundle_snapshot(export.bundle)

    if corrupt:
        with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
            package_graspgen_export(export, deployment_name="copy", om_dir=sources)
        assert _bundle_snapshot(export.bundle) == before
    else:
        package_graspgen_export(export, deployment_name="copy", om_dir=sources)
        for name in (GRASPGEN_DEPLOYMENT, "copy"):
            validate_deployment_artifact_integrity(load_inference_manifest(export.bundle, name))


def test_updating_graspgen_does_not_modify_hardlinked_retained_deployment(graspgen_export):
    export = graspgen_export
    package_graspgen_export(export)
    package_graspgen_export(export, deployment_name="copy")
    original = export.bundle / "artifacts/ascend" / GRASPGEN_DEPLOYMENT / "denoiser.om"
    retained = export.bundle / "artifacts/ascend/copy/denoiser.om"
    retained.unlink()
    retained.hardlink_to(original)
    before = retained.read_bytes()
    (export.om_dir / "denoiser.om").write_bytes(b"new compilation")

    package_graspgen_export(export)

    assert original.read_bytes() == b"new compilation"
    assert retained.read_bytes() == before
    assert not retained.samefile(original)
    for name in (GRASPGEN_DEPLOYMENT, "copy"):
        validate_deployment_artifact_integrity(load_inference_manifest(export.bundle, name))


@pytest.mark.parametrize("asset_path", ["assets/adapter.json", *(path for _, path in _TORCH_ASSETS.values())])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_graspgen_asset_writes_reject_artifact_aliases_before_writes(graspgen_export, asset_path, alias):
    export = graspgen_export
    package_graspgen_export(export)
    om_dir = export.bundle / "artifacts/ascend" / GRASPGEN_DEPLOYMENT
    artifact = om_dir / "denoiser.om"
    asset = export.bundle / asset_path
    # The artifact points at the asset (the inverse of the shared packager tests).
    artifact.unlink()
    if alias == "symlink":
        artifact.symlink_to(asset)
    else:
        artifact.hardlink_to(asset)
    manifest_path = export.bundle / "inference_manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw["deployments"][GRASPGEN_DEPLOYMENT]["artifacts"]["denoiser"]["sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(raw))
    validate_deployment_artifact_integrity(load_inference_manifest(export.bundle, GRASPGEN_DEPLOYMENT))
    before = _bundle_snapshot(export.bundle)
    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        package_graspgen_export(export, om_dir=om_dir)
    assert _bundle_snapshot(export.bundle) == before
    assert artifact.is_symlink() == (alias == "symlink")


def test_graspgen_asset_mapping_drives_copy_adapter_and_preflight(graspgen_export, monkeypatch):
    import perception_service.package_graspgen_ascend_bundle as packager

    destination = "assets/renamed_generator.pth"
    monkeypatch.setitem(packager._TORCH_ASSETS, "generator_checkpoint", ("generator_checkpoint", destination))
    package_graspgen_export(graspgen_export)
    bundle = graspgen_export.bundle
    adapter = json.loads((bundle / "assets/adapter.json").read_text())
    assert adapter["generator_checkpoint"] == destination
    original_source = Path(graspgen_export.onnx_manifest["source"]["generator_checkpoint"])
    assert (bundle / destination).read_bytes() == original_source.read_bytes()
    om_dir = bundle / "artifacts/ascend" / GRASPGEN_DEPLOYMENT
    artifact = om_dir / "denoiser.om"
    (bundle / destination).unlink()
    (bundle / destination).hardlink_to(artifact)
    before = _bundle_snapshot(bundle)
    with pytest.raises(ManifestIntegrityError, match="aliases.*asset"):
        package_graspgen_export(graspgen_export, om_dir=om_dir)
    assert _bundle_snapshot(bundle) == before


@pytest.mark.parametrize("retained_state", ["corrupt", "symlink"])
def test_graspgen_checks_unselected_deployment_before_writing(graspgen_export, retained_state):
    export = graspgen_export
    package_graspgen_export(export)
    package_graspgen_export(export, deployment_name="copy")
    original = export.bundle / "artifacts/ascend" / GRASPGEN_DEPLOYMENT / "denoiser.om"
    retained = export.bundle / "artifacts/ascend/copy/denoiser.om"
    if retained_state == "corrupt":
        retained.write_bytes(b"damaged retained OM")
        expected = "artifact_digest_mismatch"
    else:
        retained.unlink()
        retained.symlink_to(original)
        expected = "retained artifact.*aliases a replaced target"
    (export.om_dir / "denoiser.om").write_bytes(b"new compilation")
    before = _bundle_snapshot(export.bundle)

    with pytest.raises(ManifestIntegrityError, match=expected):
        package_graspgen_export(export)

    assert _bundle_snapshot(export.bundle) == before
    if retained_state == "symlink":
        assert retained.is_symlink()
        for name in (GRASPGEN_DEPLOYMENT, "copy"):
            validate_deployment_artifact_integrity(load_inference_manifest(export.bundle, name))
