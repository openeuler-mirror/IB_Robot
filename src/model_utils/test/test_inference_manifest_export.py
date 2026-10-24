from __future__ import annotations

import hashlib
import json
import stat
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from inference_manifest import (
    DeploymentTarget,
    ExecutionContract,
    ManifestIntegrityError,
    RoleRuntimeProfile,
    TensorBinding,
    TorchDeployment,
    TorchRuntimeProfile,
    load_inference_manifest,
    validate_deployment_artifact_integrity,
)
from model_utils.acl_abi_inspection import write_acl_om_abi
from model_utils.inference_manifest_export import (
    RuntimeABI,
    RuntimeTensor,
    artifact_bindings,
    compiled_deployment,
    copy_policy_metadata_bundle,
    package_deployment_artifact,
    read_runtime_abi,
    read_tcim_abi,
    refresh_bundle_revision,
    upsert_deployment,
)
from model_utils.package_compiled_deployment import package_compiled_deployment
from model_utils.package_pi05_ascend_deployments import package_pi05_ascend_deployments
from model_utils.package_torch_deployment import package_torch_deployments


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_bundle(root: Path, policy_type: str = "act") -> None:
    _write_json(
        root / "config.json",
        {
            "type": policy_type,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})


def _act_abi() -> RuntimeABI:
    return RuntimeABI(
        inputs=(
            RuntimeTensor("state", 0, "float32", (1, 6)),
            RuntimeTensor("image", 1, "float32", (1, 3, 16, 24)),
        ),
        outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
    )


def _act_bindings():
    return artifact_bindings(
        _act_abi(),
        input_semantics={"state": "observation.state", "image": "observation.images.top"},
        output_semantics={"action": "action"},
        image_layouts={"observation.images.top": "NCHW"},
    )


def _torch_deployment(device: str = "cpu") -> TorchDeployment:
    return TorchDeployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device=device),
        ),
    )


@pytest.mark.parametrize("alias", ["direct", "symlink", "chain", "hardlink"])
@pytest.mark.parametrize("retained", [False, True])
def test_upsert_rejects_manifest_dependencies_before_writing(tmp_path, alias, retained):
    from inference_manifest import DeploymentArtifact

    _create_bundle(tmp_path)
    model = tmp_path / "policy.rknn"
    model.write_bytes(b"model")
    candidate = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (model, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    first = upsert_deployment(tmp_path, "rknn", candidate)
    target = first.manifest_path
    if alias != "direct":
        target = tmp_path / "manifest-alias.json"
        if alias == "hardlink":
            target.hardlink_to(first.manifest_path)
        else:
            link = first.manifest_path
            if alias == "chain":
                link = tmp_path / "intermediate.json"
                link.symlink_to(first.manifest_path.name)
            target.symlink_to(link.name)
    artifact = DeploymentArtifact(path=target.name, format="json")
    if retained:
        raw = json.loads(first.manifest_path.read_text())
        raw["deployments"]["rknn"]["artifacts"]["metadata"] = artifact.model_dump(mode="json", exclude_none=True)
        # Atomic replacement preserves an independent hardlink's bytes.
        replacement = tmp_path / "legacy.json"
        replacement.write_text(json.dumps(raw))
        replacement.replace(first.manifest_path)
        name = "another"
    else:
        candidate = candidate.model_copy(update={"artifacts": {**candidate.artifacts, "metadata": artifact}})
        name = "rknn"
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    if alias == "hardlink":
        result = upsert_deployment(tmp_path, name, candidate)
        validate_deployment_artifact_integrity(result)
        assert target.read_bytes() == before[target.name]
    else:
        with pytest.raises(ManifestIntegrityError, match="artifact_integrity_unverified.*metadata.*manifest"):
            upsert_deployment(tmp_path, name, candidate)
        assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("entrypoint", ["index", "package", "upsert"])
@pytest.mark.parametrize("damage", ["symlink_loop", "non_directory", "permission"])
def test_packaging_wraps_published_index_filesystem_errors(tmp_path, monkeypatch, entrypoint, damage):
    from model_utils.inference_manifest_export import load_published_artifact_index

    _create_bundle(tmp_path)
    path = tmp_path / "artifacts" / "policy.rknn"
    path.parent.mkdir()
    path.write_bytes(b"published model")
    candidate = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (path, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "rknn", candidate)
    source = tmp_path / "new.rknn"
    source.write_bytes(b"new compilation")
    if damage == "symlink_loop":
        path.unlink()
        path.symlink_to(path.name)
    elif damage == "non_directory":
        path.unlink()
        path.parent.rmdir()
        path.parent.write_bytes(b"not a directory")
    else:
        original_stat = Path.stat

        def stat_path(value, *args, **kwargs):
            if value == path:
                raise PermissionError(13, "Permission denied", str(value))
            return original_stat(value, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat_path)
    before = (tmp_path / "inference_manifest.json").read_bytes()
    with pytest.raises(
        ManifestIntegrityError,
        match="artifact_integrity_unverified.*deployment='rknn'.*role='policy'.*artifacts/policy.rknn",
    ) as error:
        if entrypoint == "index":
            load_published_artifact_index(tmp_path)
        elif entrypoint == "package":
            package_deployment_artifact(tmp_path, source, backend="rknn", deployment_name="rknn", role="policy")
        else:
            upsert_deployment(tmp_path, "rknn", candidate)
    assert isinstance(error.value.__cause__, OSError)
    assert str(error.value.__cause__) in str(error.value)
    assert (tmp_path / "inference_manifest.json").read_bytes() == before


@pytest.mark.parametrize("state", ["verified", "mismatch"])
def test_upsert_invalidates_reports_from_previous_deployment(tmp_path, state):
    from inference_manifest import validate_deployment_artifact_integrity, verify_deployment_artifacts

    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"model")
    candidate = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    first = upsert_deployment(tmp_path, "rknn", candidate)
    if state == "mismatch":
        artifact.write_bytes(b"tampered")
    assert verify_deployment_artifacts(tmp_path, "rknn").state == state
    source = tmp_path / "new.rknn"
    source.write_bytes(b"new model generation")
    generation = package_deployment_artifact(tmp_path, source, backend="rknn", deployment_name="rknn", role="policy")
    replacement = candidate.model_copy(
        update={
            "artifacts": {
                "policy": candidate.artifacts["policy"].model_copy(
                    update={"path": generation.relative_to(tmp_path).as_posix()}
                )
            }
        }
    )
    current = upsert_deployment(tmp_path, "rknn", replacement)
    assert current.deployment.revision == first.deployment.revision + 1
    assert current.integrity_status.state == "declared"
    validate_deployment_artifact_integrity(current)


def test_upsert_deployment_preserves_existing_deployments_and_structural_identity(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "artifacts" / "policy.rknn"
    artifact.parent.mkdir()
    artifact.write_bytes(b"rknn")
    torch_deployment = _torch_deployment()
    with pytest.raises(ValueError, match="model.safetensors"):
        upsert_deployment(tmp_path, "cpu", torch_deployment)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    upsert_deployment(tmp_path, "cpu", torch_deployment)
    first = load_inference_manifest(tmp_path, "cpu")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    validated = upsert_deployment(tmp_path, "rknn", deployment)

    assert set(validated.manifest.deployments) == {"cpu", "rknn"}
    assert "model.safetensors" in {entry.path for entry in validated.manifest.bundle.files}
    assert load_inference_manifest(tmp_path, "cpu").deployment.backend == "torch"
    assert validated.manifest.bundle.uuid == first.manifest.bundle.uuid
    assert validated.manifest.bundle.revision == first.manifest.bundle.revision
    assert validated.deployment.uuid != first.deployment.uuid


def test_upsert_deployment_preserves_uuid_and_automatically_revises_structural_changes(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn-v1")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    first = upsert_deployment(tmp_path, "rknn", deployment)
    no_op = upsert_deployment(tmp_path, "rknn", deployment)
    changed_profile = deployment.runtime_profile.model_copy(
        update={"target": deployment.runtime_profile.target.model_copy(update={"runtime": "rknn-lite2-2.3"})}
    )
    changed = upsert_deployment(tmp_path, "rknn", deployment.model_copy(update={"runtime_profile": changed_profile}))

    assert no_op.deployment.uuid == first.deployment.uuid
    assert no_op.deployment.revision == first.deployment.revision
    assert changed.deployment.uuid == first.deployment.uuid
    assert changed.deployment.revision == first.deployment.revision + 1
    assert changed.fingerprint != first.fingerprint

    artifact.write_bytes(b"rknn-v2")
    from inference_manifest import ManifestIntegrityError
    from model_utils.inference_manifest_export import update_deployment

    before = (tmp_path / "inference_manifest.json").read_bytes()
    with pytest.raises(ManifestIntegrityError, match="retained artifact"):
        update_deployment(
            tmp_path,
            "rknn",
            changed.deployment.model_copy(update={"extra_parameters": {"threshold": 0.5}}),
            expected_uuid=changed.deployment.uuid,
            expected_revision=changed.deployment.revision,
        )
    assert (tmp_path / "inference_manifest.json").read_bytes() == before
    replacement = changed.deployment.model_copy(
        update={"artifacts": {"policy": changed.deployment.artifacts["policy"].model_copy(update={"sha256": None})}}
    )
    with pytest.raises(ManifestIntegrityError, match="retained artifact"):
        upsert_deployment(tmp_path, "rknn", replacement)
    assert (tmp_path / "inference_manifest.json").read_bytes() == before
    source = tmp_path / "new.rknn"
    source.write_bytes(b"rknn-v2")
    generation = package_deployment_artifact(tmp_path, source, backend="rknn", deployment_name="rknn", role="policy")
    replacement = replacement.model_copy(
        update={
            "artifacts": {
                "policy": replacement.artifacts["policy"].model_copy(
                    update={"path": generation.relative_to(tmp_path).as_posix()}
                )
            }
        }
    )
    repackaged = upsert_deployment(tmp_path, "rknn", replacement)
    assert repackaged.deployment.uuid == changed.deployment.uuid
    assert repackaged.deployment.revision == changed.deployment.revision + 1
    assert repackaged.deployment.artifacts["policy"].sha256 == hashlib.sha256(b"rknn-v2").hexdigest()
    assert repackaged.fingerprint != changed.fingerprint


def test_refresh_bundle_revision_preserves_uuid_and_changes_deployment_fingerprint(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    first = upsert_deployment(tmp_path, "cpu", _torch_deployment())
    config_path = tmp_path / "config.json"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    refreshed = refresh_bundle_revision(tmp_path)
    second = load_inference_manifest(tmp_path, "cpu")

    assert refreshed.bundle.uuid == first.manifest.bundle.uuid
    assert refreshed.bundle.revision == first.manifest.bundle.revision + 1
    assert refreshed.bundle.digest.value != first.manifest.bundle.digest.value
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision
    assert second.fingerprint != first.fingerprint


def test_package_torch_deployments_generates_cpu_and_cuda_by_default(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    validated = package_torch_deployments(tmp_path)

    assert [item.deployment_name for item in validated] == ["torch-cpu", "torch-cuda"]
    assert [item.deployment.device for item in validated] == ["cpu", "cuda"]
    manifest = load_inference_manifest(tmp_path, "torch-cpu").manifest
    assert set(manifest.deployments) == {"torch-cpu", "torch-cuda"}


def test_package_torch_deployments_supports_device_selection_and_prefix(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    validated = package_torch_deployments(tmp_path, devices=("cpu",), deployment_prefix="native")

    assert validated[0].deployment_name == "native-cpu"
    assert validated[0].deployment.device == "cpu"


@pytest.mark.parametrize("policy_type", ["act", "pi05"])
def test_native_devices_share_one_model_descriptor_without_selector(tmp_path, policy_type):
    _create_bundle(tmp_path, policy_type=policy_type)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    package_torch_deployments(tmp_path, devices=("cpu", "cuda"))
    result = package_torch_deployments(tmp_path, devices=("npu",))[0]

    assert set(result.manifest.deployments) == {"torch-cpu", "torch-cuda", "torch-npu"}
    assert result.manifest.model.model_type == policy_type
    assert result.manifest.model.architecture_class is None


def test_generic_torch_packager_preserves_model_owned_extra_parameters(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    parameters = {"unknown_to_manifest": {"future_options": [True, 3, None]}}
    result = package_torch_deployments(
        tmp_path, devices=("cpu",), deployment_name="custom-mode", extra_parameters=parameters
    )[0]
    assert result.deployment.extra_parameters == parameters
    assert load_inference_manifest(tmp_path, "custom-mode").deployment.extra_parameters == parameters
    assert result.deployment.artifacts == {}
    with pytest.raises(ValueError, match="one device"):
        package_torch_deployments(tmp_path, deployment_name="ambiguous")


def test_pi05_named_deployments_share_bundle_and_preserve_legacy_mode(tmp_path):
    _create_bundle(tmp_path, policy_type="pi05")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    config_before = (tmp_path / "config.json").read_bytes()
    inode_before = weights.stat().st_ino
    legacy = package_torch_deployments(tmp_path, devices=("npu",))[0]
    results = package_pi05_ascend_deployments(tmp_path)
    fidelity = load_inference_manifest(tmp_path, "torch-npu-fidelity")
    fast = load_inference_manifest(tmp_path, "torch-npu-fast")
    assert [r.deployment_name for r in results] == ["torch-npu-fidelity", "torch-npu-fast"]
    assert fidelity.deployment.extra_parameters["denoising"] == {"solver": "euler", "schedule": "uniform", "steps": 10}
    assert fast.deployment.extra_parameters["denoising"] == {"solver": "ab2", "schedule": "uniform", "steps": 3}
    assert fidelity.deployment.extra_parameters["language_capacity"] == 200
    assert fast.deployment.extra_parameters["language_capacity"] == 48
    assert fidelity.deployment_fingerprint != fast.deployment_fingerprint
    assert fidelity.manifest.bundle.uuid == fast.manifest.bundle.uuid == legacy.manifest.bundle.uuid
    assert fidelity.manifest.bundle.revision == legacy.manifest.bundle.revision
    assert load_inference_manifest(tmp_path, "torch-npu").deployment_fingerprint == legacy.deployment_fingerprint
    assert weights.stat().st_ino == inode_before and weights.read_bytes() == b"weights"
    assert (tmp_path / "config.json").read_bytes() == config_before
    assert fidelity.deployment.artifacts == fast.deployment.artifacts == {}
    again = package_pi05_ascend_deployments(tmp_path)
    assert [r.deployment.revision for r in again] == [r.deployment.revision for r in results]


def test_pi05_packaging_rejects_non_pi05_bundle_before_writing(tmp_path):
    _create_bundle(tmp_path, policy_type="act")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="pi05 bundle"):
        package_pi05_ascend_deployments(tmp_path)
    assert not (tmp_path / "inference_manifest.json").exists()


def test_extra_parameter_update_revises_only_selected_deployment(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    first = package_torch_deployments(tmp_path, devices=("cpu",), extra_parameters={"arbitrary": 1})[0]
    second = package_torch_deployments(tmp_path, devices=("cpu",), extra_parameters={"arbitrary": 2})[0]
    assert first.deployment.uuid == second.deployment.uuid
    assert second.deployment.revision == first.deployment.revision + 1
    assert first.manifest.bundle.digest == second.manifest.bundle.digest
    assert first.deployment_fingerprint != second.deployment_fingerprint


def test_unequal_size_generations_are_rejected_without_hashing(tmp_path, monkeypatch):
    import inference_manifest.integrity as integrity

    source = tmp_path / "new.om"
    source.write_bytes(b"new compiled model")
    for index in range(3):
        historical = tmp_path / f"artifacts/ascend/old/generations/{index}/model.om"
        historical.parent.mkdir(parents=True)
        historical.write_bytes(b"x" * (index + 1))

    def unexpected_hash(path):
        pytest.fail(f"Unnecessary comparison hash: {path}")

    monkeypatch.setattr(integrity, "_sha256_file", unexpected_hash)
    result = package_deployment_artifact(tmp_path, source, backend="ascend", deployment_name="new", role="model")
    assert result.read_bytes() == source.read_bytes()
    assert "/new/generations/" in str(result)


@pytest.mark.parametrize("difference_block", [0, 1])
@pytest.mark.parametrize("published_current", [False, True])
def test_equal_size_history_with_early_differences_has_bounded_comparison_reads(
    tmp_path, track_digest_reads, published_current, difference_block
):
    source = tmp_path / "new.om"
    content = b"new" + b"x" * (3 * 1024 * 1024)
    source.write_bytes(content)
    historical_paths = []
    for index in range(4):
        historical = tmp_path / f"artifacts/ascend/old/generations/{index}/model.om"
        historical.parent.mkdir(parents=True)
        offset = difference_block * 64 * 1024
        historical.write_bytes(content[:offset] + bytes([index]) + content[offset + 1 :])
        historical_paths.append(historical)
    if published_current:
        _create_bundle(tmp_path)
        candidate = compiled_deployment(
            tmp_path,
            backend="ascend",
            target_soc="Ascend310P3",
            target_runtime="acl",
            artifacts={"model": (historical_paths[0], "om")},
            execution=("model",),
            bindings={"model": _act_bindings()},
        )
        current = upsert_deployment(tmp_path, "new", candidate)
        assert current.deployment.artifacts["model"].sha256 is not None
    bytes_read = track_digest_reads(historical_paths)
    result = package_deployment_artifact(tmp_path, source, backend="ascend", deployment_name="new", role="model")
    assert result.read_bytes() == content
    if published_current:
        # Comparing the current declaration caches the source SHA before the
        # unreferenced history is scanned. Its old bytes need not be read.
        assert bytes_read[historical_paths[0]] == 0
    unmatched = historical_paths[1:] if published_current else historical_paths
    assert all(0 < bytes_read[path] <= (difference_block + 1) * 64 * 1024 for path in unmatched), dict(bytes_read)
    # The comparison of a subsequent match also supplies its verified digest.
    historical_paths.append(result)
    bytes_read.clear()
    assert (
        package_deployment_artifact(tmp_path, source, backend="ascend", deployment_name="new", role="model") == result
    )
    assert bytes_read[result] == len(content)


def test_declared_digest_excludes_different_generation_without_content_reads(tmp_path, monkeypatch):
    import inference_manifest.integrity as integrity

    _create_bundle(tmp_path)
    source = tmp_path / "new.rknn"
    source.write_bytes(b"new model")
    old = tmp_path / "artifacts/rknn/A/generations/old/policy.rknn"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old model")
    candidate = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (old, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "A", candidate)
    original_open = Path.open

    def open_file(path, mode="r", *args, **kwargs):
        if path == old and mode == "rb":
            pytest.fail("A mismatching declared SHA should exclude this generation without reading it")
        return original_open(path, mode, *args, **kwargs)

    reads = Counter()
    original_hash = integrity._sha256_file

    def hash_file(path):
        reads[path] += 1
        return original_hash(path)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(integrity, "_sha256_file", hash_file)
    result = package_deployment_artifact(tmp_path, source, backend="rknn", deployment_name="A", role="policy")
    assert result != old
    assert result.read_bytes() == source.read_bytes()
    assert reads[source] == 1


@pytest.mark.parametrize("changed_file", [None, "source", "history", "source_same_bytes", "history_same_bytes"])
@pytest.mark.parametrize("cached_side", [None, "source", "history"])
def test_undeclared_equal_generation_bounds_digest_reads_and_checks_changes(
    tmp_path, track_digest_reads, changed_file, cached_side
):
    from inference_manifest import PublishedArtifactIndex

    source = tmp_path / "new.om"
    historical = tmp_path / "artifacts/ascend/old/generations/old/model.om"
    historical.parent.mkdir(parents=True)
    content = b"model" * (512 * 1024)
    source.write_bytes(content)
    historical.write_bytes(content)
    assert not source.samefile(historical)
    published = PublishedArtifactIndex(tmp_path, None)
    if cached_side is not None:
        published.read_and_verify_digest(source if cached_side == "source" else historical)
    mutated = False

    def after_read(path, data):
        nonlocal mutated
        uncached_path = source if cached_side == "history" else historical
        if changed_file and path == uncached_path and not data and not mutated:
            changed = source if changed_file.startswith("source") else historical
            # A same-byte rewrite must not be accepted as a stable comparison.
            changed.write_bytes(content if changed_file.endswith("same_bytes") else b"!" * len(content))
            mutated = True

    bytes_read = track_digest_reads({source, historical}, after_read=after_read)
    result = package_deployment_artifact(
        tmp_path, source, backend="ascend", deployment_name="new", role="model", published=published
    )
    if changed_file:
        assert mutated
        assert result != historical
        assert result.read_bytes() == source.read_bytes()
    else:
        assert result == historical
        # Without block hashes, an uncached candidate requires comparing both
        # streams. A repeated match can reuse both complete, state-bound digests.
        assert bytes_read == {source: len(content), historical: len(content)}
        bytes_read.clear()
        assert published.reusable_digest(source, historical) == hashlib.sha256(content).hexdigest()
        assert not bytes_read


@pytest.mark.parametrize("cached_side", ["source", "history"])
@pytest.mark.parametrize("offset", [0, 64 * 1024, 256 * 1024 - 1])
def test_cached_digest_comparison_stops_at_first_different_block(tmp_path, track_digest_reads, cached_side, offset):
    source, history = tmp_path / "source.om", tmp_path / "history.om"
    content = b"x" * (256 * 1024)
    source.write_bytes(content)
    history.write_bytes(content[:offset] + b"!" + content[offset + 1 :])
    from inference_manifest import PublishedArtifactIndex

    published = PublishedArtifactIndex(tmp_path, None)
    cached, uncached = (source, history) if cached_side == "source" else (history, source)
    published.read_and_verify_digest(cached)
    reads = track_digest_reads({source, history})
    assert published.reusable_digest(source, history) is None
    expected = (offset // (64 * 1024) + 1) * 64 * 1024
    assert reads[cached] == expected
    assert reads[uncached] == expected


@pytest.mark.parametrize("role_count", [2, 6])
@pytest.mark.parametrize("hardlinked_targets", [False, True])
def test_undeclared_shared_source_reuses_cached_reads_before_independent_upsert(
    tmp_path, track_digest_reads, role_count, hardlinked_targets
):
    from model_utils.inference_manifest_export import load_published_artifact_index

    _create_bundle(tmp_path)
    content = b"compiled" * (32 * 1024)
    source = tmp_path / "compiled.om"
    source.write_bytes(content)
    roles = ["policy", *(f"extra{index}" for index in range(role_count - 1))]
    paths = [tmp_path / f"artifacts/ascend/A/generations/old/{role}.om" for role in roles]
    paths[0].parent.mkdir(parents=True)
    for path in paths:
        if hardlinked_targets and path != paths[0]:
            path.hardlink_to(paths[0])
        else:
            path.write_bytes(content)
    candidate = compiled_deployment(
        tmp_path,
        backend="ascend",
        target_soc="Ascend310P3",
        target_runtime="acl",
        artifacts={role: (path, "om") for role, path in zip(roles, paths, strict=True)},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    first = upsert_deployment(tmp_path, "A", candidate)
    raw = json.loads(first.manifest_path.read_text())
    for artifact in raw["deployments"]["A"]["artifacts"].values():
        artifact.pop("sha256")
    _write_json(first.manifest_path, raw)
    published = load_published_artifact_index(tmp_path)
    bytes_read = track_digest_reads({source, *paths})
    for role, path in zip(roles, paths, strict=True):
        assert (
            package_deployment_artifact(
                tmp_path, source, backend="ascend", deployment_name="A", role=role, published=published
            )
            == path
        )
    unique_targets = 1 if hardlinked_targets else role_count
    assert bytes_read[source] == unique_targets * len(content)
    assert sum(bytes_read[path] for path in paths) == unique_targets * len(content)

    current = upsert_deployment(tmp_path, "A", candidate)
    assert sum(bytes_read[path] for path in paths) == 2 * unique_targets * len(content)
    assert all(
        artifact.sha256 == hashlib.sha256(content).hexdigest() for artifact in current.deployment.artifacts.values()
    )


@pytest.mark.parametrize("evidence", ["original", "cleared", "new_digest"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "replaced_entry"])
def test_upsert_recovers_published_evidence_before_accepting_candidate(tmp_path, evidence, alias):
    _create_bundle(tmp_path)
    path = tmp_path / "policy.rknn"
    path.write_bytes(b"original model")
    candidate = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (path, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    first = upsert_deployment(tmp_path, "rknn", candidate)
    if alias == "replaced_entry":
        temporary = tmp_path / "replacement.rknn"
        temporary.write_bytes(b"changed model")
        temporary.replace(path)
    else:
        path.write_bytes(b"changed model")
    if alias in {"symlink", "hardlink"}:
        linked = tmp_path / "alias.rknn"
        if alias == "symlink":
            linked.symlink_to(path)
        else:
            linked.hardlink_to(path)
        path = linked
    sha = hashlib.sha256(b"changed model").hexdigest() if evidence == "new_digest" else None
    if evidence != "original" or alias in {"symlink", "hardlink"}:
        candidate = first.deployment.model_copy(
            update={
                "artifacts": {
                    "policy": first.deployment.artifacts["policy"].model_copy(update={"path": path.name, "sha256": sha})
                }
            }
        )
    before = first.manifest_path.read_bytes()
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch"):
        upsert_deployment(tmp_path, "rknn", candidate)
    assert first.manifest_path.read_bytes() == before


def test_package_deployment_artifact_reuses_identical_immutable_generation(tmp_path):
    source = tmp_path / "policy.rknn"
    source.write_bytes(b"rknn")

    packaged = package_deployment_artifact(
        tmp_path,
        source,
        backend="rknn",
        deployment_name="rk3588",
        role="policy",
        force_copy=True,
    )

    assert packaged != source
    assert packaged.read_bytes() == b"rknn"
    assert packaged.relative_to(tmp_path).parts[:4] == ("artifacts", "rknn", "rk3588", "generations")
    assert packaged.name == "policy.rknn"
    repackaged = package_deployment_artifact(
        tmp_path,
        packaged,
        backend="rknn",
        deployment_name="rk3588",
        role="policy",
    )
    assert repackaged == packaged

    source.chmod(0o755)
    executable = package_deployment_artifact(
        tmp_path,
        source,
        backend="rknn",
        deployment_name="rk3588-executable",
        role="policy",
    )
    assert executable != packaged
    assert executable.stat().st_mode & 0o111


def test_package_deployment_artifact_prefers_target_deployments_current_generation(tmp_path):
    _create_bundle(tmp_path)
    older = tmp_path / "artifacts/rknn/older/generations/000/policy.rknn"
    current = tmp_path / "artifacts/rknn/target/generations/999/policy.rknn"
    source = tmp_path / "source.rknn"
    for path in (older, current, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"same")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (current, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "target", deployment)

    packaged = package_deployment_artifact(
        tmp_path,
        source,
        backend="rknn",
        deployment_name="target",
        role="policy",
    )

    assert packaged == current


@pytest.mark.parametrize("input_kind", ["self", "symlink", "hardlink", "independent"])
def test_compiled_packaging_retains_published_evidence_and_accepts_independent_inputs(tmp_path, input_kind):
    _create_bundle(tmp_path)
    source = tmp_path / "compiled.om"
    source.write_bytes(b"compiled")
    abi = _act_abi()
    _write_json(
        tmp_path / "abi.json", {"inputs": [asdict(t) for t in abi.inputs], "outputs": [asdict(t) for t in abi.outputs]}
    )
    spec = {
        "execution": ["policy"],
        "roles": {
            "policy": {
                "artifact": str(source),
                "format": "om",
                "abi": "abi.json",
                "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                "output_semantics": {"action": "action"},
                "image_layouts": {"observation.images.top": "NCHW"},
            }
        },
    }

    def publish():
        _write_json(tmp_path / "spec.json", spec)
        return package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="ascend",
            backend="ascend",
            target_soc="Ascend310P3",
            target_runtime="acl",
            spec_path=tmp_path / "spec.json",
        )

    first = publish().deployment
    destination = tmp_path / first.artifacts["policy"].path
    if input_kind == "self":
        source = destination
    elif input_kind in {"symlink", "hardlink"}:
        source = tmp_path / "alias.om"
        if input_kind == "symlink":
            source.symlink_to(destination)
        else:
            source.hardlink_to(destination)
    spec["roles"]["policy"]["artifact"] = str(source)
    assert publish().deployment == first
    destination.write_bytes(b"corrupt")
    before = (tmp_path / "inference_manifest.json").read_bytes()
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch"):
        validate_deployment_artifact_integrity(load_inference_manifest(tmp_path, "ascend"))

    if input_kind == "independent":
        # Even equal bytes from an independent compiler input need a new
        # generation if the previous immutable generation was damaged.
        source.write_bytes(b"corrupt")
        updated = publish()
        assert updated.deployment.artifacts["policy"].path != first.artifacts["policy"].path
        assert updated.deployment.revision == first.revision + 1
        validate_deployment_artifact_integrity(updated)
    else:
        with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
            publish()
        assert (tmp_path / "inference_manifest.json").read_bytes() == before


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("rename", ["role", "deployment"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_compiler_source_evidence_survives_role_and_deployment_changes(tmp_path, alias, rename, corrupt):
    _create_bundle(tmp_path)
    source = tmp_path / "compiled.om"
    source.write_bytes(b"compiled")
    abi = _act_abi()
    _write_json(
        tmp_path / "abi.json", {"inputs": [asdict(t) for t in abi.inputs], "outputs": [asdict(t) for t in abi.outputs]}
    )

    def publish(path, role, name):
        _write_json(
            tmp_path / "spec.json",
            {
                "execution": [role],
                "roles": {
                    role: {
                        "artifact": str(path),
                        "format": "om",
                        "abi": "abi.json",
                        "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                        "output_semantics": {"action": "action"},
                        "image_layouts": {"observation.images.top": "NCHW"},
                    }
                },
            },
        )
        return package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name=name,
            backend="ascend",
            target_soc="Ascend310P3",
            target_runtime="acl",
            spec_path=tmp_path / "spec.json",
        )

    first = publish(source, "encoder", "A")
    old = tmp_path / first.deployment.artifacts["encoder"].path
    source = old
    if alias != "direct":
        source = tmp_path / "alias.om"
        if alias == "symlink":
            source.symlink_to(old)
        else:
            source.hardlink_to(old)
    if corrupt:
        old.write_bytes(b"damaged old generation")
    manifest_before = first.manifest_path.read_bytes()
    artifacts_before = {path: path.read_bytes() for path in (tmp_path / "artifacts").rglob("*") if path.is_file()}
    role, name = ("decoder", "A") if rename == "role" else ("encoder", "B")
    if corrupt:
        with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
            publish(source, role, name)
        assert first.manifest_path.read_bytes() == manifest_before
        assert {
            path: path.read_bytes() for path in (tmp_path / "artifacts").rglob("*") if path.is_file()
        } == artifacts_before
    else:
        result = publish(source, role, name)
        assert result.deployment.artifacts[role].sha256 == first.deployment.artifacts["encoder"].sha256
        validate_deployment_artifact_integrity(result)


def test_shared_generation_reuse_hashes_once_then_checks_final_publication(tmp_path, monkeypatch):
    import inference_manifest.integrity as integrity
    import model_utils.inference_manifest_export as exporter

    _create_bundle(tmp_path)
    source = tmp_path / "compiled.om"
    source.write_bytes(b"compiled")
    path = package_deployment_artifact(tmp_path, source, backend="ascend", deployment_name="A", role="policy")

    def candidate():
        return compiled_deployment(
            tmp_path,
            backend="ascend",
            target_soc="Ascend310P3",
            target_runtime="acl",
            artifacts={"policy": (path, "om")},
            execution=("policy",),
            bindings={"policy": _act_bindings()},
        )

    for index in range(4):
        upsert_deployment(tmp_path, str(index), candidate())
    reads = Counter()
    original = integrity._sha256_file

    def count(path):
        reads[path] += 1
        return original(path)

    monkeypatch.setattr(integrity, "_sha256_file", count)
    monkeypatch.setattr(
        exporter.filecmp,
        "cmp",
        lambda *args, **kwargs: pytest.fail("artifact reuse must compare the digests already used for integrity"),
    )
    assert package_deployment_artifact(tmp_path, source, backend="ascend", deployment_name="0", role="policy") == path
    assert reads[path] == 1
    result = upsert_deployment(tmp_path, "0", candidate())
    assert reads[path] == 2
    validate_deployment_artifact_integrity(result)


def test_compiled_roles_share_manifest_parse_and_file_identity_index(tmp_path, monkeypatch):
    import model_utils.inference_manifest_export as exporter

    _create_bundle(tmp_path)
    old = tmp_path / "published.om"
    old.write_bytes(b"compiled")
    first = compiled_deployment(
        tmp_path,
        backend="ascend",
        target_soc="Ascend310P3",
        target_runtime="acl",
        artifacts={"policy": (old, "om")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    previous = upsert_deployment(tmp_path, "A", first).deployment
    paths = {}
    for role in ("encoder", "decoder", "head"):
        path = tmp_path / f"{role}.om"
        path.hardlink_to(old)
        paths[role] = (path, "om")
    parses = []
    original_load = exporter._load_existing_manifest

    def load(path):
        parses.append(path)
        return original_load(path)

    monkeypatch.setattr(exporter, "_load_existing_manifest", load)
    result = compiled_deployment(
        tmp_path,
        backend="ascend",
        target_soc="Ascend310P3",
        target_runtime="acl",
        artifacts=paths,
        execution=tuple(paths),
        bindings={role: _act_bindings() for role in paths},
    )
    assert len(parses) == 1
    assert {artifact.sha256 for artifact in result.artifacts.values()} == {previous.artifacts["policy"].sha256}


def test_artifact_binding_requires_complete_semantic_and_layout_mapping():
    with pytest.raises(ValueError, match="No semantic mapping"):
        artifact_bindings(
            _act_abi(),
            input_semantics={"state": "observation.state"},
            output_semantics={"action": "action"},
        )
    with pytest.raises(ValueError, match="explicit runtime layout"):
        artifact_bindings(
            _act_abi(),
            input_semantics={"state": "observation.state", "image": "observation.images.top"},
            output_semantics={"action": "action"},
        )


def test_read_runtime_abi_rejects_noncontiguous_indices(tmp_path):
    metadata = tmp_path / "abi.json"
    _write_json(
        metadata,
        {
            "inputs": [{"name": "state", "index": 1, "dtype": "float32", "shape": [1, 6]}],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )

    with pytest.raises(ValueError, match="contiguous from zero"):
        read_runtime_abi(metadata)


def test_read_runtime_abi_accepts_sparse_output_indices(tmp_path):
    metadata = tmp_path / "abi.json"
    _write_json(
        metadata,
        {
            "inputs": [{"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]}],
            "outputs": [{"name": "action", "index": 3, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )

    abi = read_runtime_abi(metadata)

    assert abi.outputs[0].index == 3


def test_write_acl_om_abi_lazily_uses_runtime_descriptor(tmp_path, monkeypatch):
    model = tmp_path / "model.om"
    model.write_bytes(b"om")
    calls = []

    class FakeRuntime:
        @staticmethod
        def set_device(device_id):
            calls.append(("set_device", device_id))
            return 0

        @staticmethod
        def reset_device(device_id):
            calls.append(("reset_device", device_id))
            return 0

        @staticmethod
        def create_context(device_id):
            calls.append(("create_context", device_id))
            return "context", 0

        @staticmethod
        def set_context(context):
            calls.append(("set_context", context))
            return 0

        @staticmethod
        def destroy_context(context):
            calls.append(("destroy_context", context))
            return 0

    class FakeModel:
        names = {"input": ["image"], "output": ["action"]}
        shapes = {"input": [[1, 3, 16, 24]], "output": [[1, 2, 8]]}
        dtypes = {"input": [1], "output": [0]}

        @staticmethod
        def load_from_file(path):
            calls.append(("load", path))
            return 7, 0

        @staticmethod
        def create_desc():
            return object()

        @staticmethod
        def get_desc(desc, model_id):
            return 0

        @classmethod
        def get_num_inputs(cls, desc):
            return 1

        @classmethod
        def get_num_outputs(cls, desc):
            return 1

        @classmethod
        def get_input_name_by_index(cls, desc, index):
            return cls.names["input"][index]

        @classmethod
        def get_output_name_by_index(cls, desc, index):
            return cls.names["output"][index]

        @classmethod
        def get_input_dims(cls, desc, index):
            return {"dims": cls.shapes["input"][index]}, 0

        @classmethod
        def get_output_dims(cls, desc, index):
            return {"dims": cls.shapes["output"][index]}, 0

        @classmethod
        def get_input_data_type(cls, desc, index):
            return cls.dtypes["input"][index]

        @classmethod
        def get_output_data_type(cls, desc, index):
            return cls.dtypes["output"][index]

        @staticmethod
        def destroy_desc(desc):
            calls.append(("destroy_desc",))

        @staticmethod
        def unload(model_id):
            calls.append(("unload", model_id))
            return 0

    class FakeACL:
        rt = FakeRuntime()
        mdl = FakeModel()

        @staticmethod
        def init(config_path=None):
            calls.append(("init", config_path))
            return 0

        @staticmethod
        def finalize():
            calls.append(("finalize",))
            return 0

    monkeypatch.setattr("model_utils.acl_abi_inspection.importlib.import_module", lambda name: FakeACL())

    output = write_acl_om_abi(model, tmp_path / "model.om.abi.json")
    abi = read_runtime_abi(output)

    assert abi.inputs == (RuntimeTensor("image", 0, "float16", (1, 3, 16, 24)),)
    assert abi.outputs == (RuntimeTensor("action", 0, "float32", (1, 2, 8)),)
    assert calls[0] == ("init", None)
    assert ("create_context", 0) in calls
    assert ("destroy_context", "context") in calls
    assert calls[-1] == ("finalize",)


def test_acl_config_path_is_limited_to_offline_abi_inspection(tmp_path, monkeypatch):
    calls = []

    class FakeACL:
        class rt:
            @staticmethod
            def set_device(_device_id):
                return 0

            @staticmethod
            def reset_device(_device_id):
                return 0

            @staticmethod
            def create_context(_device_id):
                return "context", 0

            @staticmethod
            def set_context(_context):
                return 0

            @staticmethod
            def destroy_context(_context):
                return 0

        class mdl:
            @staticmethod
            def load_from_file(_path):
                return 1, 0

            @staticmethod
            def create_desc():
                return object()

            @staticmethod
            def get_desc(_descriptor, _model_id):
                return 0

            @staticmethod
            def get_num_inputs(_descriptor):
                return 1

            @staticmethod
            def get_num_outputs(_descriptor):
                return 1

            @staticmethod
            def get_input_name_by_index(_descriptor, _index):
                return "input"

            @staticmethod
            def get_output_name_by_index(_descriptor, _index):
                return "output"

            @staticmethod
            def get_input_dims(_descriptor, _index):
                return {"dims": [1]}, 0

            @staticmethod
            def get_output_dims(_descriptor, _index):
                return {"dims": [1]}, 0

            @staticmethod
            def get_input_data_type(_descriptor, _index):
                return 0

            @staticmethod
            def get_output_data_type(_descriptor, _index):
                return 0

            @staticmethod
            def destroy_desc(_descriptor):
                return 0

            @staticmethod
            def unload(_model_id):
                return 0

        @staticmethod
        def init(config_path=None):
            calls.append(config_path)
            return 0

        @staticmethod
        def finalize():
            return 0

    model = tmp_path / "model.om"
    model.write_bytes(b"om")
    monkeypatch.setattr("model_utils.acl_abi_inspection.importlib.import_module", lambda _name: FakeACL)

    write_acl_om_abi(model, tmp_path / "abi.json", acl_config_path="inspection.json")

    assert calls == ["inspection.json"]


def test_write_acl_om_abi_explains_missing_acl_runtime(tmp_path, monkeypatch):
    om_path = tmp_path / "policy.om"
    om_path.write_bytes(b"om")

    def missing_acl(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("model_utils.acl_abi_inspection.importlib.import_module", missing_acl)

    with pytest.raises(RuntimeError, match="source the CANN environment or provide a pre-generated"):
        write_acl_om_abi(om_path, tmp_path / "policy.om.abi.json")


def test_runtime_image_layout_is_authoritative():
    abi = RuntimeABI(
        inputs=(RuntimeTensor("image", 0, "float32", (1, 16, 24, 3), "NHWC"),),
        outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
    )

    bindings = artifact_bindings(
        abi,
        input_semantics={"image": "observation.images.top"},
        output_semantics={"action": "action"},
    )

    assert bindings.inputs[0].layout == "NHWC"
    with pytest.raises(ValueError, match="runtime ABI reports NHWC"):
        artifact_bindings(
            abi,
            input_semantics={"image": "observation.images.top"},
            output_semantics={"action": "action"},
            image_layouts={"observation.images.top": "NCHW"},
        )


def test_upsert_deployment_restores_previous_manifest_on_strict_validation_failure(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn")
    valid = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "rknn", valid)
    original = (tmp_path / "inference_manifest.json").read_bytes()
    invalid_bindings = artifact_bindings(
        RuntimeABI(
            inputs=(RuntimeTensor("unknown", 0, "float32", (1, 6)),),
            outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
        ),
        input_semantics={"unknown": "observation.unknown"},
        output_semantics={"action": "action"},
    )
    invalid = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": invalid_bindings},
    )

    with pytest.raises(ValueError, match="unknown LeRobot input feature"):
        upsert_deployment(tmp_path, "rknn", invalid)

    assert (tmp_path / "inference_manifest.json").read_bytes() == original


def test_manifest_writer_preserves_existing_mode_and_uses_readable_default(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    first = upsert_deployment(tmp_path, "cpu", _torch_deployment())
    path = first.manifest_path
    assert stat.S_IMODE(path.stat().st_mode) == 0o644

    path.chmod(0o640)
    upsert_deployment(tmp_path, "cuda", _torch_deployment("cuda"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_upsert_deployment_does_not_rehash_preserved_deployment_artifacts(tmp_path):
    _create_bundle(tmp_path)
    first_artifact = tmp_path / "first.rknn"
    first_artifact.write_bytes(b"first")
    first = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (first_artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "first", first)
    first_artifact.write_bytes(b"changed")

    second_artifact = tmp_path / "second.om"
    second_artifact.write_bytes(b"second")
    second = compiled_deployment(
        tmp_path,
        backend="ascend",
        target_soc="Ascend310P3",
        target_runtime="acl",
        artifacts={"policy": (second_artifact, "om")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    validated = upsert_deployment(tmp_path, "second", second)
    assert set(validated.manifest.deployments) == {"first", "second"}


def test_upsert_deployment_rejects_external_semantic_dependencies(tmp_path):
    _create_bundle(tmp_path, "smolvla")
    preprocessor = json.loads((tmp_path / "policy_preprocessor.json").read_text(encoding="utf-8"))
    preprocessor["steps"] = [
        {
            "registry_name": "tokenizer_processor",
            "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
        }
    ]
    _write_json(tmp_path / "policy_preprocessor.json", preprocessor)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    with pytest.raises(ValueError, match="all semantic dependencies to be local bundle assets"):
        upsert_deployment(tmp_path, "rknn", deployment)

    assert not (tmp_path / "inference_manifest.json").exists()


def test_package_hisilicon_requires_complete_runtime_abi_and_executable_worker(tmp_path):
    _create_bundle(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = artifacts / "policy.om"
    worker = artifacts / "worker"
    model.write_bytes(b"om")
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    _write_json(
        tmp_path / "policy_abi.json",
        {
            "inputs": [
                {"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {"name": "image", "index": 1, "dtype": "float32", "shape": [1, 3, 16, 24]},
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )
    spec = tmp_path / "hisilicon.json"
    _write_json(
        spec,
        {
            "execution": ["policy"],
            "roles": {
                "policy": {
                    "artifact": "artifacts/policy.om",
                    "format": "om",
                    "abi": "policy_abi.json",
                    "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                    "output_semantics": {"action": "action"},
                    "image_layouts": {"observation.images.top": "NCHW"},
                }
            },
            "artifacts": {"worker": {"path": "artifacts/worker", "format": "executable"}},
        },
    )

    validated = package_compiled_deployment(
        bundle_root=tmp_path,
        deployment_name="sd3403",
        backend="hisilicon",
        target_soc="sd3403",
        target_runtime="hisilicon-worker",
        spec_path=spec,
    )

    assert validated.deployment.backend == "hisilicon"
    assert set(validated.deployment.artifacts) == {"policy", "worker"}
    assert load_inference_manifest(tmp_path, "sd3403").deployment.target.soc == "sd3403"
    for artifact in validated.deployment.artifacts.values():
        assert artifact.sha256 == hashlib.sha256((tmp_path / artifact.path).read_bytes()).hexdigest()

    worker.chmod(0o644)
    with pytest.raises(ValueError, match="not executable"):
        package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="sd3403",
            backend="hisilicon",
            target_soc="sd3403",
            target_runtime="hisilicon-worker",
            spec_path=spec,
        )


def test_package_compiled_deployment_rejects_backend_target_mismatch(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.om"
    artifact.write_bytes(b"om")
    _write_json(
        tmp_path / "policy_abi.json",
        {
            "inputs": [
                {"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {
                    "name": "image",
                    "index": 1,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 24],
                    "layout": "NCHW",
                },
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )
    spec = tmp_path / "package.json"
    _write_json(
        spec,
        {
            "execution": ["policy"],
            "roles": {
                "policy": {
                    "artifact": "policy.om",
                    "format": "om",
                    "abi": "policy_abi.json",
                    "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                    "output_semantics": {"action": "action"},
                }
            },
        },
    )

    with pytest.raises(ValueError, match="RKNN deployment requires"):
        package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="bad",
            backend="rknn",
            target_soc="rk3588",
            target_runtime="acl",
            spec_path=spec,
        )


def test_tensor_binding_type_is_strict():
    with pytest.raises(ValueError):
        TensorBinding(semantic="action", runtime_name="action", index=0, dtype="float32", shape=(1, 0))


def test_read_tcim_abi_uses_model_descriptors(tmp_path):
    metadata = tmp_path / "model.json"
    _write_json(
        metadata,
        {
            "Golden": {"inputs": [], "outputs": []},
            "Model": {
                "inputs": [{"name": "x_t", "shape": [1, 2, 8], "dtype": {"code": "float", "bits": 16}}],
                "outputs": [{"name": "v_t", "shape": [1, 2, 8], "dtype": {"code": "float", "bits": 16}}],
            },
        },
    )

    abi = read_tcim_abi(metadata)

    assert abi.inputs[0].dtype == "float16"
    assert abi.outputs[0].shape == (1, 2, 8)


def test_copy_policy_metadata_bundle_copies_only_required_semantic_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _create_bundle(source)
    (source / "model.safetensors").write_bytes(b"native")
    destination = tmp_path / "compiled"

    copied = copy_policy_metadata_bundle(source, destination)

    assert set(copied) == {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    assert (destination / "config.json").is_file()
    assert not (destination / "model.safetensors").exists()


def test_copy_policy_metadata_bundle_only_revises_changed_content(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "compiled"
    source.mkdir()
    destination.mkdir()
    _create_bundle(source)
    _create_bundle(destination)
    artifact = destination / "policy.rknn"
    artifact.write_bytes(b"rknn")
    deployment = compiled_deployment(
        destination,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    first = upsert_deployment(destination, "rknn", deployment)

    copy_policy_metadata_bundle(source, destination)
    unchanged = load_inference_manifest(destination, "rknn")
    assert unchanged.manifest.bundle.revision == first.manifest.bundle.revision

    _write_json(source / "policy_preprocessor.json", {"name": "updated", "steps": []})
    copy_policy_metadata_bundle(source, destination)
    changed = load_inference_manifest(destination, "rknn")
    assert changed.manifest.bundle.revision == first.manifest.bundle.revision + 1
    assert json.loads((destination / "policy_preprocessor.json").read_text())["name"] == "updated"


def test_copy_policy_metadata_bundle_rolls_back_files_and_manifest_on_validation_failure(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "compiled"
    source.mkdir()
    destination.mkdir()
    _create_bundle(source)
    _create_bundle(destination)
    artifact = destination / "policy.rknn"
    artifact.write_bytes(b"rknn")
    deployment = compiled_deployment(
        destination,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(destination, "rknn", deployment)
    manifest_before = (destination / "inference_manifest.json").read_bytes()
    config_before = (destination / "config.json").read_bytes()
    config = json.loads((source / "config.json").read_text())
    config["input_features"].pop("observation.state")
    _write_json(source / "config.json", config)

    with pytest.raises(ValueError, match="unknown LeRobot input feature"):
        copy_policy_metadata_bundle(source, destination)

    assert (destination / "config.json").read_bytes() == config_before
    assert (destination / "inference_manifest.json").read_bytes() == manifest_before


@pytest.mark.parametrize("role_count", [2, 6])
def test_complete_packaging_shares_index_and_alias_reads_until_independent_upsert(tmp_path, monkeypatch, role_count):
    import inference_manifest.integrity as integrity
    import model_utils.inference_manifest_export as exporter

    _create_bundle(tmp_path)
    source = tmp_path / "compiled.om"
    source.write_bytes(b"compiled")
    abi = _act_abi()
    _write_json(
        tmp_path / "abi.json", {"inputs": [asdict(t) for t in abi.inputs], "outputs": [asdict(t) for t in abi.outputs]}
    )
    spec = {
        "execution": ["policy"],
        "roles": {
            "policy": {
                "artifact": str(source),
                "format": "om",
                "abi": "abi.json",
                "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                "output_semantics": {"action": "action"},
                "image_layouts": {"observation.images.top": "NCHW"},
            }
        },
        "artifacts": {f"extra{index}": {"path": str(source), "format": "om"} for index in range(role_count - 1)},
    }

    def publish():
        _write_json(tmp_path / "spec.json", spec)
        return package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="A",
            backend="ascend",
            target_soc="Ascend310P3",
            target_runtime="acl",
            spec_path=tmp_path / "spec.json",
        )

    first = publish()
    old = tmp_path / first.deployment.artifacts["policy"].path
    spec["roles"]["policy"]["artifact"] = str(old)
    for index, entry in enumerate(spec["artifacts"].values()):
        alias = tmp_path / f"alias{index}.om"
        alias.hardlink_to(old)
        entry["path"] = str(alias)
    scans = []
    reads = Counter()
    original_init = exporter.PublishedArtifactIndex.__init__
    original_hash = integrity._sha256_file

    def scan(self, root, manifest):
        scans.append(manifest)
        original_init(self, root, manifest)

    def count(path):
        info = path.stat()
        reads[info.st_dev, info.st_ino] += 1
        return original_hash(path)

    monkeypatch.setattr(exporter.PublishedArtifactIndex, "__init__", scan)
    monkeypatch.setattr(integrity, "_sha256_file", count)
    result = publish()
    # One preparation snapshot and one independent final publication check,
    # regardless of how many roles or aliases the input spec contains.
    assert len(scans) <= 2
    info = old.stat()
    assert reads[info.st_dev, info.st_ino] == 2
    assert result.deployment == first.deployment
    validate_deployment_artifact_integrity(result)
