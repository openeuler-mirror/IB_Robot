from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from inference_manifest import (
    AscendRuntimeProfile,
    AudioContract,
    BackendRuntimeProfile,
    BundleFile,
    Deployment,
    ExecutionContract,
    InferenceManifest,
    ManifestIntegrityError,
    ManifestValidationError,
    ModelIdentity,
    ModelRuntimeSpec,
    PublishedArtifactIndex,
    RoleRuntimeProfile,
    RoleRuntimeSpec,
    RuntimeInstanceProjection,
    TorchRuntimeProfile,
    canonical_bundle_digest,
    deployment_fingerprint,
    finalize_deployment_artifacts,
    inherit_deployment_version,
    load_inference_manifest,
    load_inference_manifest_metadata,
    runtime_profile_fingerprint,
    validate_deployment_artifact_integrity,
    validate_manifest_schema,
    verify_deployment_artifacts,
    write_inference_manifest,
)

BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"


@pytest.mark.parametrize("other_target", [False, True])
def test_replacement_dependency_exclusion_still_checks_the_remaining_chain(tmp_path, other_target):
    from inference_manifest.paths import depends_on_file_replacement

    artifact = tmp_path / "artifact.om"
    artifact.write_bytes(b"model")
    middle = tmp_path / "middle.om"
    middle.symlink_to(artifact.name)
    own = tmp_path / "own.om"
    own.symlink_to(middle.name)
    source = tmp_path / "source.om"
    source.symlink_to(own.name)
    targets = {own, artifact} if other_target else {own}

    assert depends_on_file_replacement(source, targets)
    assert depends_on_file_replacement(source, targets, excluded_target=own) is other_target
    # An independent directory entry keeps its inode when a target is replaced.
    hardlink = tmp_path / "hardlink.om"
    hardlink.hardlink_to(artifact)
    assert not depends_on_file_replacement(hardlink, targets, excluded_target=own)


@pytest.mark.parametrize("digest", ["", "abc", "A" * 64, None, 123])
def test_bind_artifact_digest_rejects_invalid_sha256(digest):
    from inference_manifest import DeploymentArtifact, bind_artifact_digest

    artifact = DeploymentArtifact(path="model.om", format="om")
    with pytest.raises(ValueError):
        bind_artifact_digest(artifact, digest, "model")
    assert artifact.sha256 is None


def test_published_index_distinguishes_undeclared_content_from_caller_evidence(tmp_path, monkeypatch):
    import inference_manifest.integrity as integrity

    path = tmp_path / "model.om"
    path.write_bytes(b"model")
    reads = []
    original = integrity._sha256_file

    def read(candidate):
        reads.append(candidate)
        return original(candidate)

    monkeypatch.setattr(integrity, "_sha256_file", read)
    published = PublishedArtifactIndex(tmp_path, None)
    assert published.verify_declared_digest(path) is None
    assert reads == []
    expected = hashlib.sha256(b"model").hexdigest()
    assert published.read_and_verify_digest(path) == expected
    assert reads == [path]
    with pytest.raises(ManifestIntegrityError, match="changed while checking"):
        published._accept_computed_digest(path, "0" * 64)

    fresh = PublishedArtifactIndex(tmp_path, None)
    assert fresh._accept_computed_digest(path, expected) == expected
    assert fresh.read_and_verify_digest(path) == expected
    assert reads == [path]


def test_manifest_rejects_runtime_capability_claims(tmp_path):
    deployment = _request_deployment()
    deployment.update(
        {
            "execution": ["encoder", "decoder"],
            "artifacts": {role: {"path": f"{role}.om", "format": "om"} for role in ("encoder", "decoder")},
            "bindings": {
                role: {
                    "inputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 4]}],
                    "outputs": [{"semantic": "prob", "index": 0, "dtype": "float32", "shape": [1, 1]}],
                }
                for role in ("encoder", "decoder")
            },
        }
    )
    value = _bundle(tmp_path, deployment)
    validate_manifest_schema(value, tmp_path / "inference_manifest.json")
    parsed = Deployment.model_validate(deployment)
    assert parsed.execution == ("encoder", "decoder")
    deployment["functional_execution"] = {"schema_version": 1, "supports_async_stage_submission": True}
    with pytest.raises(ManifestValidationError):
        validate_manifest_schema(value, tmp_path / "inference_manifest.json")
    with pytest.raises(ValueError, match="functional_execution"):
        Deployment.model_validate(deployment)


def _bundle(root: Path, deployment: dict, *, model: dict | None = None) -> dict:
    (root / "payload.json").write_text("{}", encoding="utf-8")
    entry = BundleFile(path="payload.json")
    return {
        "schema_version": 3,
        "bundle": {
            "uuid": BUNDLE_UUID,
            "revision": 1,
            "name": "v3-test",
            "files": [entry.model_dump(mode="json")],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(BUNDLE_UUID, 1, "v3-test", (entry,)),
            },
        },
        "model": model
        or {
            "interface": "tensor_model",
            "model_type": "silero_vad",
            "operation": "vad",
            "inputs": [{"semantic": "audio", "dtype": "float32", "shape": [1, 4]}],
            "outputs": [{"semantic": "prob", "dtype": "float32", "shape": [1, 1]}],
        },
        "deployments": {"test": deployment},
    }


def _request_deployment(*, device_id: int = 0, artifact: bool = False) -> dict:
    value = {
        "uuid": DEPLOYMENT_UUID,
        "revision": 1,
        "execution_contract": {
            "state_scope": "request",
            "execution_structure": "direct",
            "cancellation_granularity": "request_boundary",
        },
        "runtime_profile": {
            "backend": "ascend",
            "target": {"soc": "Ascend310P", "runtime": "acl", "runtime_abi": "cann-8.0"},
            "profile": {"device_id": device_id},
        },
    }
    if artifact:
        value["artifacts"] = {
            "model": {
                "path": "model.om",
                "format": "om",
                "sha256": hashlib.sha256(b"model").hexdigest(),
            }
        }
    return value


@pytest.mark.parametrize("loader", [load_inference_manifest, load_inference_manifest_metadata])
@pytest.mark.parametrize(
    "entry",
    ["missing_root", "raw_file", "unpackaged", "manifest_directory", "dangling_manifest", "manifest_fifo"],
)
def test_bundle_required_before_reading_model_files(tmp_path, monkeypatch, loader, entry):
    root = tmp_path / "model"
    if entry == "raw_file":
        root.write_bytes(b"unpackaged weights")
    elif entry != "missing_root":
        root.mkdir()
        (root / "model.safetensors").write_bytes(b"unpackaged weights")
        manifest = root / "inference_manifest.json"
        if entry == "manifest_directory":
            manifest.mkdir()
        elif entry == "dangling_manifest":
            manifest.symlink_to("absent.json")
        elif entry == "manifest_fifo":
            os.mkfifo(manifest)

    def unexpected_read(*args, **kwargs):
        pytest.fail("An invalid bundle entry must fail before opening model files")

    monkeypatch.setattr("inference_manifest.loader.load_json_strict", unexpected_read)
    with pytest.raises(ManifestValidationError, match="bundle_manifest_required") as error:
        loader(root, "test")
    assert str(root / "inference_manifest.json") in str(error.value)


@pytest.mark.parametrize("loader", [load_inference_manifest, load_inference_manifest_metadata])
@pytest.mark.parametrize("empty_path", ["", " ", "\t\n"])
def test_empty_bundle_path_cannot_use_a_manifest_from_working_directory(tmp_path, monkeypatch, loader, empty_path):
    write_inference_manifest(tmp_path / "inference_manifest.json", _bundle(tmp_path, _request_deployment()))
    monkeypatch.chdir(tmp_path)
    assert loader(".", "test").deployment_name == "test"
    with pytest.raises(ManifestValidationError, match="bundle_manifest_required"):
        loader(empty_path, "test")


def test_publication_index_rechecks_changed_source_after_cached_verification(tmp_path):
    path = tmp_path / "model.om"
    path.write_bytes(b"model")
    manifest = InferenceManifest.model_validate(_bundle(tmp_path, _request_deployment(artifact=True)))
    published = PublishedArtifactIndex(tmp_path, manifest)
    assert published.verify_declared_digest(path) == hashlib.sha256(b"model").hexdigest()
    path.write_bytes(b"damaged publication")
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*retained artifact"):
        published.verify_declared_digest(path)


@pytest.mark.parametrize("indexed", [False, True])
def test_finalization_rejects_changes_during_hashing_with_and_without_index(tmp_path, monkeypatch, indexed):
    import inference_manifest.integrity as integrity

    path = tmp_path / "model.om"
    path.write_bytes(b"model")
    deployment = Deployment.model_validate(_request_deployment(artifact=True))
    original = integrity._sha256_file

    def hash_then_change(candidate):
        digest = original(candidate)
        candidate.write_bytes(b"changed during content hashing")
        return digest

    monkeypatch.setattr(integrity, "_sha256_file", hash_then_change)
    published = PublishedArtifactIndex(tmp_path, None) if indexed else None
    context = "model.om" if indexed else "role='model'"
    with pytest.raises(
        ManifestIntegrityError, match=f"artifact_integrity_unverified.*changed while hashing.*{context}"
    ):
        finalize_deployment_artifacts(tmp_path, deployment, published=published)


def test_publication_index_rejects_conflicting_digests_for_hardlinked_declarations(tmp_path):
    path = tmp_path / "model.om"
    path.write_bytes(b"model")
    alias = tmp_path / "alias.om"
    alias.hardlink_to(path)
    value = _bundle(tmp_path, _request_deployment(artifact=True))
    other = copy.deepcopy(value["deployments"]["test"])
    other["artifacts"]["model"].update(path="alias.om", sha256="0" * 64)
    value["deployments"]["other"] = other
    published = PublishedArtifactIndex(tmp_path, InferenceManifest.model_validate(value))
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch.*conflicting"):
        published.verify_declared_digest(path)


def test_extra_parameters_are_generic_json_and_change_deployment_identity(tmp_path):
    raw = _request_deployment()
    value = _bundle(tmp_path, raw)
    original = Deployment.model_validate(raw)
    raw["extra_parameters"] = {"future_model": {"threshold": 0.4, "labels": ["a", None, True]}}
    validate_manifest_schema(value, "extra parameters")
    configured = Deployment.model_validate(raw)
    assert configured.extra_parameters == raw["extra_parameters"]
    assert deployment_fingerprint(3, "bundle", "test", original) != deployment_fingerprint(
        3, "bundle", "test", configured
    )
    assert runtime_profile_fingerprint(3, "bundle", "test", original) != runtime_profile_fingerprint(
        3, "bundle", "test", configured
    )
    raw["extra_parameters"] = {}
    empty = Deployment.model_validate(raw)
    assert deployment_fingerprint(3, "bundle", "test", original) == deployment_fingerprint(3, "bundle", "test", empty)


@pytest.mark.parametrize("parameters", [None, [], {"invalid": float("nan")}, {"invalid": object()}])
def test_extra_parameters_reject_non_json_values(parameters):
    with pytest.raises(ValueError):
        Deployment.model_validate({**_request_deployment(), "extra_parameters": parameters})


def test_extra_parameters_identity_is_independent_of_key_order():
    first = Deployment.model_validate({**_request_deployment(), "extra_parameters": {"a": 1, "b": {"x": True}}})
    second = Deployment.model_validate({**_request_deployment(), "extra_parameters": {"b": {"x": True}, "a": 1}})
    assert deployment_fingerprint(3, "bundle", "test", first) == deployment_fingerprint(3, "bundle", "test", second)


def test_packaging_finalization_hashes_final_bytes_without_mutating_candidate(tmp_path):
    artifact = tmp_path / "model.om"
    artifact.write_bytes(b"final compiled bytes")
    candidate = Deployment.model_validate(_request_deployment(artifact=True))
    candidate = candidate.model_copy(
        update={"artifacts": {"model": candidate.artifacts["model"].model_copy(update={"sha256": None})}}
    )
    finalized = finalize_deployment_artifacts(tmp_path, candidate)
    assert finalized.artifacts["model"].sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert candidate.artifacts["model"].sha256 is None
    assert finalized.uuid == candidate.uuid
    assert finalized.revision == candidate.revision
    artifact.unlink()
    with pytest.raises(ManifestIntegrityError, match="artifact_integrity_unverified.*model.om"):
        finalize_deployment_artifacts(tmp_path, candidate)


def test_packaging_finalization_hashes_hardlinked_declarations_once(tmp_path, monkeypatch):
    import inference_manifest.integrity as integrity

    artifact = tmp_path / "model.om"
    artifact.write_bytes(b"shared model")
    alias = tmp_path / "alias.om"
    alias.hardlink_to(artifact)
    raw = _request_deployment(artifact=True)
    raw["artifacts"]["model"]["sha256"] = hashlib.sha256(b"shared model").hexdigest()
    raw["artifacts"]["alias"] = {
        "path": "alias.om",
        "format": "om",
        "sha256": hashlib.sha256(b"shared model").hexdigest(),
    }
    deployment = Deployment.model_validate(raw)
    hashed = []
    original = integrity._sha256_file

    def counted(path):
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(integrity, "_sha256_file", counted)
    finalized = finalize_deployment_artifacts(tmp_path, deployment)

    assert len(hashed) == 1
    assert finalized.artifacts["model"].sha256 == finalized.artifacts["alias"].sha256


def test_schema_v2_and_legacy_identity_are_rejected(tmp_path: Path):
    value = _bundle(tmp_path, _request_deployment())
    value["schema_version"] = 2
    (tmp_path / "inference_manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match=r"supported versions: \[3\]"):
        load_inference_manifest(tmp_path, "test")

    value = _bundle(tmp_path, _request_deployment())
    value["model"]["family"] = "lerobot"
    with pytest.raises(ManifestValidationError, match="family"):
        validate_manifest_schema(value, "legacy")


def test_policy_model_type_must_match_lerobot_config(tmp_path: Path):
    config = {
        "type": "act",
        "input_features": {"observation.state": {"type": "STATE", "shape": [2]}},
        "output_features": {"action": {"type": "ACTION", "shape": [2]}},
    }
    for name, content in {
        "config.json": config,
        "policy_preprocessor.json": {"steps": []},
        "policy_postprocessor.json": {"steps": []},
    }.items():
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    files = tuple(
        BundleFile(path=name)
        for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json", "model.safetensors")
    )
    value = {
        "schema_version": 3,
        "bundle": {
            "uuid": BUNDLE_UUID,
            "revision": 1,
            "name": "policy",
            "files": [entry.model_dump(mode="json") for entry in files],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(BUNDLE_UUID, 1, "policy", files),
            },
        },
        "model": {"interface": "policy", "model_type": "act", "operation": "predict"},
        "deployments": {
            "cpu": {
                "uuid": DEPLOYMENT_UUID,
                "revision": 1,
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {
                    "backend": "torch",
                    "target": {"runtime": "torch"},
                    "profile": {"device": "cpu"},
                },
            }
        },
    }
    write_inference_manifest(tmp_path / "inference_manifest.json", value)
    assert load_inference_manifest(tmp_path, "cpu").policy.policy_type == "act"

    value["model"]["model_type"] = "pi05"
    write_inference_manifest(tmp_path / "inference_manifest.json", value)
    with pytest.raises(ManifestValidationError, match="does not match LeRobot config type"):
        load_inference_manifest(tmp_path, "cpu")


@pytest.mark.parametrize(
    ("model_type", "operation"),
    [
        ("act", "predict"),
        ("pi05", "predict"),
        ("smolvla", "predict"),
        ("sam2", "prompt"),
        ("sam2", "automatic"),
        ("grounding_dino", "detect"),
        ("zipvoice", "synthesize"),
        ("fullsubnet", "enhance"),
        ("silero_vad", "vad"),
        ("sherpa_onnx", "recognize"),
    ],
)
def test_canonical_model_identity_mapping(model_type: str, operation: str):
    interface = "policy" if model_type in {"act", "pi05", "smolvla"} else "tensor_model"
    identity = ModelIdentity(interface=interface, model_type=model_type, operation=operation)
    assert identity.model_type == model_type
    assert identity.operation == operation


def test_audio_contract_is_typed_and_deployment_scoped(tmp_path: Path):
    contract = AudioContract(
        sample_rate_hz=16000,
        channels=1,
        sample_dtype="float32",
        chunk_size=320,
        execution_mode="streaming",
    )
    assert contract.sample_rate_hz == 16000
    assert contract.chunk_size == 320

    with pytest.raises(ValueError, match="at least one applicable constraint"):
        AudioContract()

    value = _bundle(tmp_path, _request_deployment())
    value["deployments"]["test"]["audio_contract"] = {
        "sample_rate_hz": 16000,
        "sample_dtype": "float32",
        "execution_mode": "offline",
    }
    validated = InferenceManifest.model_validate(value)
    assert validated.deployments["test"].audio_contract.sample_rate_hz == 16000


def test_policy_vla_and_noncanonical_operations_are_rejected():
    with pytest.raises(ValueError, match="vla"):
        ModelIdentity(interface="policy", model_type="vla", operation="predict")
    with pytest.raises(ValueError, match="predict"):
        ModelIdentity(interface="policy", model_type="act", operation="sample")
    with pytest.raises(ValueError, match="grounding_dino"):
        ModelIdentity(interface="tensor_model", model_type="grounding_dino", operation="raw")
    with pytest.raises(ValueError, match="sherpa_onnx"):
        ModelIdentity(interface="tensor_model", model_type="sherpa_onnx", operation="vad")

    descriptor = InferenceManifest.model_validate_json(
        json.dumps(
            {
                "schema_version": 3,
                "bundle": {
                    "name": "classification",
                    "files": [{"path": "config.json"}],
                    "digest": {"algorithm": "sha256", "scope": "structure", "value": "0" * 64},
                },
                "model": {
                    "interface": "policy",
                    "model_type": "pi05",
                    "operation": "predict",
                    "architecture_class": "vla",
                },
                "deployments": {"d": _request_deployment()},
            }
        )
    ).model
    assert descriptor.architecture_class == "vla"


def test_execution_contract_and_state_link_rules():
    with pytest.raises(ValueError, match="direct execution"):
        ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            orchestration_visibility="executor",
            cancellation_granularity="request_boundary",
        )
    with pytest.raises(ValueError, match="iterative execution"):
        ExecutionContract(
            state_scope="request",
            execution_structure="iterative",
            cancellation_granularity="checkpoint",
        )
    with pytest.raises(ValueError, match="request execution"):
        ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
            state_bank_mode="per_stream",
            max_open_streams=1,
        )

    contract = ExecutionContract.model_validate_json(
        json.dumps(
            {
                "state_scope": "stream",
                "execution_structure": "direct",
                "cancellation_granularity": "checkpoint",
                "stateful": True,
                "state_bank_mode": "runtime_exclusive",
                "max_open_streams": 1,
                "state_links": [
                    {
                        "role": "__runtime__",
                        "state_name": "aggregate.state",
                        "owner": "streaming_runtime",
                        "source": "aggregate.input",
                        "target": "aggregate.output",
                        "scope": "stream",
                        "state_bank": "aggregate.bank",
                    }
                ],
            }
        )
    )
    assert contract.state_links[0].role == "__runtime__"

    with pytest.raises(ValueError, match="invalid_state_link_identifier"):
        ExecutionContract.model_validate_json(
            json.dumps(
                {
                    "state_scope": "stream",
                    "execution_structure": "direct",
                    "cancellation_granularity": "checkpoint",
                    "state_bank_mode": "per_stream",
                    "max_open_streams": 2,
                    "state_links": [
                        {
                            "role": "model",
                            "state_name": "device-3-bank",
                            "owner": "session",
                            "source": "state.in",
                            "target": "state.out",
                            "scope": "stream",
                            "state_bank": "bank",
                        }
                    ],
                }
            )
        )


def test_composite_roles_require_matching_typed_profiles_and_build_role_specs(tmp_path: Path):
    model = {
        "interface": "tensor_model",
        "model_type": "speech_direction",
        "operation": "enhance_and_vad",
        "inputs": [{"semantic": "audio", "dtype": "float32", "shape": [1, 4]}],
        "outputs": [{"semantic": "audio", "dtype": "float32", "shape": [1, 4]}],
    }
    deployment = {
        "execution_contract": {
            "state_scope": "request",
            "execution_structure": "direct",
            "cancellation_granularity": "request_boundary",
        },
        "role_identities": {
            "enhancer": {"interface": "tensor_model", "model_type": "fullsubnet", "operation": "enhance"},
            "vad": {"interface": "tensor_model", "model_type": "silero_vad", "operation": "vad"},
        },
        "role_runtime_profiles": {
            "enhancer": {
                "backend": "ascend",
                "target": {"runtime": "acl", "runtime_abi": "cann-8.0"},
                "profile": {"device_id": 0},
            },
            "vad": {
                "backend": "rknn",
                "target": {"runtime": "rknn-lite"},
                "profile": {"target_name": "rk3588", "core_mask": 3, "device_id": 0},
            },
        },
        "artifacts": {
            "enhancer": {"path": "enhancer.om", "format": "om"},
            "vad": {"path": "vad.om", "format": "om"},
        },
        "execution": ["enhancer", "vad"],
        "bindings": {
            "enhancer": {
                "inputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 4]}],
                "outputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 4]}],
            },
            "vad": {
                "inputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 4]}],
                "outputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 4]}],
            },
        },
    }
    manifest = InferenceManifest.model_validate_json(json.dumps(_bundle(tmp_path, deployment, model=model)))
    role_profiles = manifest.deployments["test"].role_runtime_profiles
    assert role_profiles is not None
    spec = ModelRuntimeSpec(
        deployment=manifest.deployments["test"],
        role_runtime_specs={
            role: RoleRuntimeSpec(
                deployment_binding=role,
                backend=profile.backend,
                target_runtime=profile.target_runtime,
                runtime_abi=profile.runtime_abi,
                backend_profile=profile.backend_profile,
            )
            for role, profile in role_profiles.items()
        },
    )
    assert spec.role_runtime_specs["enhancer"].backend == "ascend"
    assert spec.role_runtime_specs["vad"].backend == "rknn"

    missing = copy.deepcopy(deployment)
    del missing["role_runtime_profiles"]["vad"]
    with pytest.raises(ValueError, match="exactly the same roles"):
        InferenceManifest.model_validate_json(json.dumps(_bundle(tmp_path, missing, model=model)))


def test_profile_projection_and_fingerprint_separation():
    profile0 = RoleRuntimeProfile(
        backend="ascend",
        target={"runtime": "acl", "runtime_abi": "cann-8.0"},
        profile={"device_id": 0},
    )
    profile1 = RoleRuntimeProfile(
        backend="ascend",
        target={"runtime": "acl", "runtime_abi": "cann-8.0"},
        profile={"device_id": 1},
    )
    deployment0 = Deployment.model_validate_json(
        json.dumps(
            {
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": profile0.model_dump(mode="json"),
            }
        )
    )
    deployment1 = deployment0.model_copy(update={"runtime_profile": profile1})
    identity = ModelIdentity(interface="policy", model_type="act", operation="predict")
    assert deployment_fingerprint(3, "a" * 64, "test", deployment0, identity=identity) == deployment_fingerprint(
        3, "a" * 64, "test", deployment1, identity=identity
    )
    assert runtime_profile_fingerprint(3, "a" * 64, "test", deployment0) != runtime_profile_fingerprint(
        3, "a" * 64, "test", deployment1
    )
    assert runtime_profile_fingerprint(
        3, "a" * 64, "test", deployment0, provider_identity="provider-a"
    ) == runtime_profile_fingerprint(3, "a" * 64, "test", deployment0, provider_identity="provider-b")

    with pytest.raises(ValueError, match="acl_config_path"):
        AscendRuntimeProfile.model_validate({"device_id": 0, "acl_config_path": "/tmp/acl.json"})
    torch_a = TorchRuntimeProfile(device="cpu", dtype="float16", thread_count=2)
    torch_b = TorchRuntimeProfile.model_validate_json(
        json.dumps({"thread_count": 2, "dtype": "float16", "device": "cpu"})
    )
    assert torch_a.deployment_projection().canonical_bytes() == torch_b.deployment_projection().canonical_bytes()
    assert isinstance(torch_a.runtime_instance_projection(), RuntimeInstanceProjection)

    class UnclassifiedProfile(BackendRuntimeProfile):
        backend_name = "test"
        option: str | None = None

    with pytest.raises(ValueError, match="unknown_profile_projection_field"):
        UnclassifiedProfile().deployment_projection()

    spec = ModelRuntimeSpec(runtime_profile=AscendRuntimeProfile(device_id=3))
    assert spec.runtime_profile.device_id == 3
    assert spec.target_runtime == "acl"


@pytest.mark.parametrize("old,new", [(False, 0), (0, False), (True, 1), (1, True)])
def test_revision_distinguishes_json_boolean_and_number(old, new):
    previous = Deployment.model_validate({**_request_deployment(), "extra_parameters": {"nested": [old]}})
    candidate = previous.model_copy(update={"extra_parameters": {"nested": [new]}})
    assert inherit_deployment_version(candidate, previous).revision == previous.revision + 1


def test_finalization_rejects_modified_retained_artifact(tmp_path):
    (tmp_path / "model.om").write_bytes(b"tampered")
    candidate = Deployment.model_validate(_request_deployment(artifact=True))
    with pytest.raises(ManifestIntegrityError, match="retained artifact"):
        finalize_deployment_artifacts(tmp_path, candidate)


def test_deployment_revision_inheritance_uses_one_identity_comparison():
    previous = Deployment.model_validate(_request_deployment())
    unchanged = previous.model_copy(update={"revision": previous.revision + 3})
    inherited = inherit_deployment_version(unchanged, previous)
    assert inherited.uuid == previous.uuid
    assert inherited.revision == previous.revision

    changed = previous.model_copy(update={"extra_parameters": {"threshold": 0.5}})
    revised = inherit_deployment_version(changed, previous)
    assert revised.uuid == previous.uuid
    assert revised.revision == previous.revision + 1


def test_artifact_integrity_is_explicit_and_persisted(tmp_path: Path):
    (tmp_path / "model.om").write_bytes(b"model")
    value = _bundle(tmp_path, _request_deployment(artifact=True))
    write_inference_manifest(tmp_path / "inference_manifest.json", value)
    loaded = load_inference_manifest(tmp_path, "test")
    assert loaded.integrity_status.state == "declared"

    report = verify_deployment_artifacts(tmp_path, "test")
    assert report.state == "verified"
    assert (tmp_path / "inference_integrity.json").is_file()
    assert load_inference_manifest(tmp_path, "test").integrity_status.state == "verified"

    (tmp_path / "model.om").write_bytes(b"changed")
    mismatch = verify_deployment_artifacts(tmp_path, "test")
    assert mismatch.state == "mismatch"
    assert mismatch.error_code == "artifact_digest_mismatch"
    assert load_inference_manifest(tmp_path, "test").integrity_status.state == "mismatch"
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch"):
        load_inference_manifest(tmp_path, "test", verify_on_demand=True)


def test_on_demand_verification_rejects_manifest_replaced_after_load(tmp_path, monkeypatch):
    import inference_manifest.loader as loader

    (tmp_path / "model.om").write_bytes(b"model")
    deployment_a = _request_deployment(artifact=True)
    manifest_a = _bundle(tmp_path, deployment_a)
    write_inference_manifest(tmp_path / "inference_manifest.json", manifest_a)
    manifest_b = copy.deepcopy(manifest_a)
    manifest_b["deployments"]["test"]["revision"] = 2
    original_load = loader.load_json_strict
    reads = 0

    def read_a_then_publish_b(path):
        nonlocal reads
        result = original_load(path)
        reads += 1
        if reads == 1:
            write_inference_manifest(tmp_path / "inference_manifest.json", manifest_b)
        return result

    monkeypatch.setattr(loader, "load_json_strict", read_a_then_publish_b)
    with pytest.raises(ManifestIntegrityError, match="deployment changed before verification"):
        load_inference_manifest(tmp_path, "test", verify_on_demand=True)
    assert reads == 1


def test_runtime_artifact_verification_is_read_only_and_detects_changed_bytes(tmp_path):
    (tmp_path / "model.om").write_bytes(b"model")
    write_inference_manifest(
        tmp_path / "inference_manifest.json", _bundle(tmp_path, _request_deployment(artifact=True))
    )
    validated = load_inference_manifest(tmp_path, "test")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    validate_deployment_artifact_integrity(validated)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before
    (tmp_path / "model.om").write_bytes(b"changed")
    with pytest.raises(ManifestIntegrityError, match="artifact_digest_mismatch"):
        validate_deployment_artifact_integrity(validated)
    (tmp_path / "model.om").unlink()
    validate_deployment_artifact_integrity(validated, verify_content=False)


@pytest.mark.parametrize("entrypoint", ["finalize", "readonly", "report"])
def test_compiled_symlink_loop_uses_integrity_error_contract(tmp_path, entrypoint):
    artifact = tmp_path / "model.om"
    artifact.write_bytes(b"model")
    write_inference_manifest(
        tmp_path / "inference_manifest.json", _bundle(tmp_path, _request_deployment(artifact=True))
    )
    validated = load_inference_manifest(tmp_path, "test")
    artifact.unlink()
    artifact.symlink_to("model.om")
    with pytest.raises(ManifestIntegrityError, match="artifact_integrity_unverified"):
        if entrypoint == "finalize":
            finalize_deployment_artifacts(tmp_path, validated.deployment)
        elif entrypoint == "readonly":
            validate_deployment_artifact_integrity(validated)
        else:
            verify_deployment_artifacts(tmp_path, "test")
    assert not (tmp_path / "inference_integrity.json").exists()


def test_path_resolution_runtime_error_does_not_depend_on_interpreter_wording(tmp_path, monkeypatch):
    from inference_manifest.errors import ManifestPathError
    from inference_manifest.paths import resolve_bundle_file

    error = RuntimeError("alternate interpreter path-resolution failure")

    def fail_resolve(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    with pytest.raises(ManifestPathError) as caught:
        resolve_bundle_file(tmp_path, "model.om")
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("digests", [[], [True], [False], [True, False]])
def test_metadata_load_and_remote_integrity_check_agree_on_declarations(tmp_path, digests):
    raw = _request_deployment()
    raw["artifacts"] = {
        f"role{index}": {
            "path": f"absent{index}.om",
            "format": "om",
            **({"sha256": "0" * 64} if declared else {}),
        }
        for index, declared in enumerate(digests)
    }
    write_inference_manifest(tmp_path / "inference_manifest.json", _bundle(tmp_path, raw))
    validated = load_inference_manifest_metadata(tmp_path, "test")
    declared = bool(digests) and all(digests)
    assert validated.integrity_status.state == ("declared" if declared else "unverified")
    assert validated.integrity_status.artifact_digests == {
        f"role{index}": "0" * 64 for index, present in enumerate(digests) if present
    }
    assert validated.integrity_status.details == {}
    if declared:
        validate_deployment_artifact_integrity(validated, verify_content=False)
    else:
        with pytest.raises(ManifestIntegrityError, match="artifact_integrity_unverified") as error:
            validate_deployment_artifact_integrity(validated, verify_content=False)
        for index, present in enumerate(digests):
            if not present:
                assert f"role{index}" in str(error.value)


@pytest.mark.parametrize(
    "backend,runtime",
    [
        ("torch", "torch"),
        ("ascend", "acl"),
        ("rknn", "rknn-lite2"),
        ("onnx", "onnxruntime"),
        ("hmm", "hmm"),
        ("hisilicon", "hisilicon-worker"),
        ("tensorrt", "tensorrt"),
    ],
)
@pytest.mark.parametrize("verify_content", [False, True])
def test_empty_compiled_artifacts_are_unverified_but_native_torch_is_allowed(
    tmp_path, backend, runtime, verify_content
):
    raw = _request_deployment()
    raw["runtime_profile"] = {
        "backend": backend,
        "target": {"runtime": runtime},
        "profile": {"device": "cpu"} if backend == "torch" else {"device_id": 0} if backend == "ascend" else {},
    }
    write_inference_manifest(tmp_path / "inference_manifest.json", _bundle(tmp_path, raw))
    validated = load_inference_manifest(tmp_path, "test")
    if backend == "torch":
        validate_deployment_artifact_integrity(validated, verify_content=verify_content)
    else:
        with pytest.raises(ManifestIntegrityError, match="artifact_integrity_unverified"):
            validate_deployment_artifact_integrity(validated, verify_content=verify_content)
    assert verify_deployment_artifacts(tmp_path, "test").state == "unverified"


@pytest.mark.parametrize("mutation", ["missing_digest", "mismatch"])
def test_read_only_and_report_verification_share_failure_classification(tmp_path, mutation):
    raw = _request_deployment(artifact=True)
    if mutation == "missing_digest":
        raw["artifacts"]["model"].pop("sha256")
    (tmp_path / "model.om").write_bytes(b"changed" if mutation == "mismatch" else b"model")
    write_inference_manifest(tmp_path / "inference_manifest.json", _bundle(tmp_path, raw))
    validated = load_inference_manifest(tmp_path, "test")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    code = "artifact_digest_mismatch" if mutation == "mismatch" else "artifact_integrity_unverified"
    with pytest.raises(ManifestIntegrityError, match=code):
        validate_deployment_artifact_integrity(validated)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    report = verify_deployment_artifacts(tmp_path, "test")
    with pytest.raises(ManifestIntegrityError, match=code):
        report.require_verified()
    if mutation == "mismatch":
        with pytest.raises(ManifestIntegrityError, match=code):
            load_inference_manifest(tmp_path, "test", verify_on_demand=True)
    else:
        assert report.details["missing_declared_digests"] == ["model"]
        (tmp_path / "model.om").unlink()
        with pytest.raises(ManifestIntegrityError, match=code):
            validate_deployment_artifact_integrity(validated, verify_content=False)


@pytest.mark.parametrize("change", ["legacy", "revision", "uuid", "sha256"])
def test_reports_only_cover_the_exact_deployment_declaration(tmp_path, change):
    (tmp_path / "model.om").write_bytes(b"model")
    raw = _request_deployment(artifact=True)
    write_inference_manifest(tmp_path / "inference_manifest.json", _bundle(tmp_path, raw))
    verify_deployment_artifacts(tmp_path, "test")
    if change == "legacy":
        report = tmp_path / "inference_integrity.json"
        value = json.loads(report.read_text())
        value.pop("deployment_identity")
        report.write_text(json.dumps(value))
    else:
        if change == "revision":
            raw["revision"] += 1
        elif change == "uuid":
            raw["uuid"] = BUNDLE_UUID
        else:
            raw["artifacts"]["model"]["sha256"] = "0" * 64
        write_inference_manifest(tmp_path / "inference_manifest.json", _bundle(tmp_path, raw))
    assert load_inference_manifest(tmp_path, "test").integrity_status.state == "declared"


@pytest.mark.parametrize("partial", [False, True])
def test_strict_missing_digest_rejects_without_reading_but_report_collects_content(tmp_path, monkeypatch, partial):
    import inference_manifest.integrity as integrity

    raw = _request_deployment(artifact=True)
    raw["artifacts"]["model"].pop("sha256")
    if partial:
        raw["artifacts"]["extra"] = {"path": "extra.om", "format": "om", "sha256": "0" * 64}
        (tmp_path / "extra.om").write_bytes(b"extra")
    (tmp_path / "model.om").write_bytes(b"model")
    write_inference_manifest(tmp_path / "inference_manifest.json", _bundle(tmp_path, raw))
    validated = load_inference_manifest(tmp_path, "test")
    reads = []
    original = integrity._sha256_file

    def count(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(integrity, "_sha256_file", count)
    with pytest.raises(ManifestIntegrityError, match="artifact_integrity_unverified"):
        validate_deployment_artifact_integrity(validated)
    assert reads == []
    report = verify_deployment_artifacts(tmp_path, "test")
    assert set(report.artifact_digests) == set(raw["artifacts"])
    assert len(reads) == len(raw["artifacts"])
