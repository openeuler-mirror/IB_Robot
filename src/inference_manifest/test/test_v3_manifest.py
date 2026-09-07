from __future__ import annotations

import copy
import hashlib
import json
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
    RoleRuntimeProfile,
    RoleRuntimeSpec,
    RuntimeInstanceProjection,
    TorchRuntimeProfile,
    canonical_bundle_digest,
    deployment_fingerprint,
    load_inference_manifest,
    runtime_profile_fingerprint,
    validate_manifest_schema,
    verify_deployment_artifacts,
    write_inference_manifest,
)

BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"


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
