from __future__ import annotations

import builtins
import copy
import json
import os
import subprocess
import sys

import pytest

from inference_manifest import (
    BundleFile,
    CompiledDeployment,
    ManifestIntegrityError,
    ManifestPathError,
    ManifestValidationError,
    SemanticIdentity,
    canonical_bundle_digest,
    canonical_manifest_bytes,
    canonical_semantic_identity_json,
    deployment_fingerprint,
    load_inference_manifest,
    normalize_bundle_path,
    normalize_unique_paths,
    semantic_identity_fingerprint,
    write_inference_manifest,
)
from inference_manifest.models import DeviceLink
from test.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    create_non_policy_bundle,
    create_policy_bundle,
    make_manifest,
    make_non_policy_manifest,
    write_manifest,
)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/file",
        "../outside",
        "folder/../outside",
        "./config.json",
        "folder//file",
        "folder\\file",
        "C:/models/file",
    ],
)
def test_bundle_paths_reject_unsafe_forms(path):
    with pytest.raises(ManifestPathError):
        normalize_bundle_path(path)


def test_duplicate_normalized_paths_are_rejected():
    with pytest.raises(ManifestPathError, match="duplicate normalized path 'config.json'"):
        normalize_unique_paths(["config.json", "config.json"], "bundle.files")


def test_symlink_escape_and_broken_link_are_rejected(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (bundle / "escape.bin").symlink_to(outside)
    (bundle / "broken.bin").symlink_to(tmp_path / "missing.bin")

    from inference_manifest import resolve_bundle_file

    with pytest.raises(ManifestPathError, match="escapes the bundle root"):
        resolve_bundle_file(bundle, "escape.bin")
    with pytest.raises(ManifestPathError, match="broken symlink"):
        resolve_bundle_file(bundle, "broken.bin")


def test_schema_version_unknown_fields_aliases_and_duplicate_json_keys(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)

    manifest["schema_version"] = 2
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match=r"unsupported schema_version 2.*inference_manifest.json"):
        load_inference_manifest(tmp_path, "cpu")

    manifest = make_manifest(tmp_path, paths)
    manifest["unexpected"] = True
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match=r"schema validation failed.*unexpected"):
        load_inference_manifest(tmp_path, "cpu")

    manifest = make_manifest(tmp_path, paths)
    manifest["deployments"]["cpu"]["runtime_profile"]["target"]["runtime"] = "ascend_om"
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match="ascend_om"):
        load_inference_manifest(tmp_path, "cpu")

    (tmp_path / "inference_manifest.json").write_text(
        '{"schema_version":3,"schema_version":3}',
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="duplicate JSON key 'schema_version'"):
        load_inference_manifest(tmp_path, "cpu")


def test_schema_v1_requires_regeneration(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    manifest["schema_version"] = 1
    manifest["bundle"].pop("uuid")
    manifest["bundle"].pop("revision")
    manifest["bundle"]["digest"].pop("scope")
    for entry in manifest["bundle"]["files"]:
        entry["sha256"] = "1" * 64
    deployment = manifest["deployments"]["rk3588"]
    deployment.pop("uuid")
    deployment.pop("revision")
    for artifact in deployment["artifacts"].values():
        artifact["sha256"] = "2" * 64
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=r"unsupported schema_version 1.*supported versions"):
        load_inference_manifest(tmp_path, "rk3588")


def test_historical_v2_deployment_is_rejected_before_typed_construction() -> None:
    with pytest.raises(Exception, match="execution_contract"):
        CompiledDeployment.model_validate_json(
            json.dumps(
                {
                    "uuid": "123e4567-e89b-42d3-a456-426614174001",
                    "revision": 1,
                    "backend": "ascend",
                    "target": {"soc": "Ascend310P1", "runtime": "acl"},
                }
            )
        )


def _stateful_deployment() -> dict:
    return {
        "uuid": "123e4567-e89b-42d3-a456-426614174001",
        "revision": 1,
        "execution_contract": {
            "state_scope": "stream",
            "execution_structure": "direct",
            "cancellation_granularity": "checkpoint",
            "stateful": True,
            "state_links": [
                {
                    "role": "model",
                    "state_name": "recurrent.state",
                    "owner": "session",
                    "source": "state.in",
                    "target": "state.out",
                    "scope": "runtime",
                    "state_bank": "model.bank",
                }
            ],
            "state_bank_mode": "runtime_exclusive",
            "max_open_streams": 1,
        },
        "runtime_profile": {
            "backend": "ascend",
            "target": {"soc": "Ascend310P1", "runtime": "acl"},
            "profile": {"device_id": 0},
        },
        "artifacts": {"model": {"path": "artifacts/model.om", "format": "om"}},
        "execution": ["model"],
        "bindings": {
            "model": {
                "inputs": [
                    {"semantic": "features", "index": 0, "dtype": "float32", "shape": [1, 2]},
                    {"semantic": "host.state_in", "index": 1, "dtype": "float32", "shape": [1, 4]},
                ],
                "outputs": [
                    {"semantic": "scores", "index": 0, "dtype": "float32", "shape": [1, 2]},
                    {"semantic": "host.state_out", "index": 1, "dtype": "float32", "shape": [1, 4]},
                ],
            }
        },
    }


def test_state_links_pair_host_state_by_semantic() -> None:
    deployment = CompiledDeployment.model_validate_json(json.dumps(_stateful_deployment()))

    assert deployment.execution_contract.state_links[0].role == "model"
    assert deployment.execution_contract.state_links[0].state_bank == "model.bank"


def test_state_links_may_cover_only_stateful_execution_roles() -> None:
    value = _stateful_deployment()
    value["artifacts"]["stateless"] = {"path": "artifacts/stateless.om", "format": "om"}
    value["execution"].append("stateless")
    value["bindings"]["stateless"] = {
        "inputs": [{"semantic": "host.derived", "index": 0, "dtype": "float32", "shape": [1, 2]}],
        "outputs": [{"semantic": "host.result", "index": 0, "dtype": "float32", "shape": [1, 2]}],
    }

    deployment = CompiledDeployment.model_validate_json(json.dumps(value))

    assert {link.role for link in deployment.execution_contract.state_links} == {"model"}
    assert deployment.execution == ("model", "stateless")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value["execution_contract"].update(state_bank_mode="per_stream"), "scope='stream'"),
        (lambda value: value["execution_contract"].update(max_open_streams=2), "max_open_streams=1"),
        (
            lambda value: value["execution_contract"]["state_links"][0].update(state_name="device-3-bank"),
            "invalid_state_link_identifier",
        ),
        (lambda value: value["execution_contract"].update(state_scope="request"), "request execution"),
    ],
)
def test_state_links_reject_invalid_contracts(mutation, expected) -> None:
    value = _stateful_deployment()
    mutation(value)

    with pytest.raises(Exception, match=expected):
        CompiledDeployment.model_validate_json(json.dumps(value))


def test_missing_manifest_fails_before_bundle_metadata_loading(tmp_path):
    with pytest.raises(ManifestValidationError, match=r"bundle_manifest_required:.*inference_manifest.json"):
        load_inference_manifest(tmp_path, "cpu")


def test_v3_policy_manifests_retain_strict_lerobot_validation(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "cpu")
    assert validated.manifest.model.interface == "policy"
    assert validated.manifest.model.model_type == "act"
    assert validated.manifest.model.operation == "predict"
    assert validated.policy is not None

    mismatched = copy.deepcopy(manifest)
    mismatched["model"]["model_type"] = "smolvla"
    write_manifest(tmp_path, mismatched)
    with pytest.raises(ManifestValidationError, match=r"model_type 'smolvla'.*LeRobot config type 'act'"):
        load_inference_manifest(tmp_path, "cpu")

    write_manifest(tmp_path, manifest)
    (tmp_path / "policy_preprocessor.json").unlink()
    with pytest.raises(ManifestPathError, match="policy_preprocessor.json"):
        load_inference_manifest(tmp_path, "cpu")


def test_unknown_model_kind_is_rejected(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["kind"] = "detector"
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=r"model.*kind"):
        load_inference_manifest(tmp_path, "ascend")


def _semantic_identity():
    return {
        "logical_model_revision": "google/siglip2-base-patch16-224@v1",
        "preprocessing_contract": "siglip2-dual-encoder-v1",
        "output_semantics": "normalized-joint-image-text-embedding-v1",
        "embedding": {
            "embedding_space_id": "siglip2-base-patch16-224:v1",
            "dimension": 768,
            "normalization": "l2",
            "image_preprocessing": "siglip2-image-224-v1",
            "text_preprocessing": "siglip2-tokenizer-64-v1",
        },
    }


def test_perception_manifest_accepts_complete_semantic_embedding_identity(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["semantic_identity"] = _semantic_identity()
    write_manifest(tmp_path, manifest)

    identity = load_inference_manifest(tmp_path, "ascend").manifest.model.semantic_identity

    assert identity is not None
    assert identity.embedding is not None
    assert identity.embedding.embedding_space_id == "siglip2-base-patch16-224:v1"
    assert identity.embedding.dimension == 768


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda embedding: embedding.pop("text_preprocessing"), "text_preprocessing"),
        (lambda embedding: embedding.update(dimension=0), "dimension"),
        (lambda embedding: embedding.update(extra="invalid"), "extra"),
    ],
)
def test_semantic_embedding_identity_is_strict_and_complete(tmp_path, mutation, expected):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    identity = _semantic_identity()
    mutation(identity["embedding"])
    manifest["model"]["semantic_identity"] = identity
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=expected):
        load_inference_manifest(tmp_path, "ascend")


def test_semantic_identity_json_and_fingerprint_are_canonical_and_provenance_independent():
    identity = SemanticIdentity.model_validate(_semantic_identity())
    reordered = SemanticIdentity.model_validate(
        {
            "output_semantics": identity.output_semantics,
            "embedding": identity.embedding.model_dump(),
            "preprocessing_contract": identity.preprocessing_contract,
            "logical_model_revision": identity.logical_model_revision,
        }
    )

    expected_json = (
        '{"embedding":{"dimension":768,"embedding_space_id":"siglip2-base-patch16-224:v1",'
        '"image_preprocessing":"siglip2-image-224-v1","normalization":"l2",'
        '"text_preprocessing":"siglip2-tokenizer-64-v1"},'
        '"logical_model_revision":"google/siglip2-base-patch16-224@v1",'
        '"output_semantics":"normalized-joint-image-text-embedding-v1",'
        '"preprocessing_contract":"siglip2-dual-encoder-v1"}'
    )
    assert canonical_semantic_identity_json(identity) == expected_json
    assert canonical_semantic_identity_json(reordered) == expected_json
    assert semantic_identity_fingerprint(identity) == semantic_identity_fingerprint(reordered)
    assert len(semantic_identity_fingerprint(identity)) == 64


@pytest.mark.parametrize("output_semantic", ["tag_logits", "image_embedding", "masks", "boxes"])
def test_compiled_non_policy_bundle_accepts_declared_semantic_outputs(tmp_path, output_semantic):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths, output_semantic=output_semantic)
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "ascend")

    assert validated.policy is None
    assert validated.manifest.model.interface == "tensor_model"
    assert validated.manifest.model.model_type == "ram_plus"
    assert validated.manifest.model.outputs[0].semantic == output_semantic
    assert validated.deployment.bindings["model"].outputs[0].semantic == output_semantic
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "policy_preprocessor.json").exists()
    assert not (tmp_path / "policy_postprocessor.json").exists()
    assert not (tmp_path / "model.safetensors").exists()


def test_torch_non_policy_bundle_does_not_require_lerobot_files(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths, deployment_name="cpu")
    manifest["deployments"]["cpu"] = {
        "uuid": manifest["deployments"]["cpu"]["uuid"],
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
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "cpu")

    assert validated.deployment.backend == "torch"
    assert validated.policy is None


def test_compiled_policy_bundle_still_requires_action_output(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    manifest["deployments"]["rk3588"]["bindings"]["policy"]["outputs"][0]["semantic"] = "tag_logits"
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="action"):
        load_inference_manifest(tmp_path, "rk3588")


@pytest.mark.parametrize(
    ("direction", "field", "value", "message"),
    [
        ("outputs", "semantic", "scores", "declared semantic output bindings"),
        ("outputs", "dtype", "float16", "tag_logits.*dtype expected float32, actual float16"),
        ("outputs", "shape", [1, 1000], "tag_logits.*shape expected.*4585.*actual.*1000"),
        ("inputs", "layout", "NHWC", "observation.image.*layout expected NCHW, actual NHWC"),
    ],
)
def test_non_policy_binding_must_match_declared_semantic_contract(tmp_path, direction, field, value, message):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["deployments"]["ascend"]["bindings"]["model"][direction][0][field] = value
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=message):
        load_inference_manifest(tmp_path, "ascend")


def test_non_policy_manifest_requires_complete_semantic_contract(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["outputs"] = []
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=r"model.outputs.*should be non-empty"):
        load_inference_manifest(tmp_path, "ascend")


def test_non_policy_bundle_integrity_artifacts_and_paths_are_validated(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)

    invalid_digest = copy.deepcopy(manifest)
    invalid_digest["bundle"]["digest"]["value"] = "0" * 64
    write_manifest(tmp_path, invalid_digest)
    with pytest.raises(ManifestIntegrityError, match="Bundle digest mismatch"):
        load_inference_manifest(tmp_path, "ascend")

    missing_artifact = copy.deepcopy(manifest)
    (tmp_path / "artifacts" / "ram_plus.om").unlink()
    write_manifest(tmp_path, missing_artifact)
    with pytest.raises(ManifestPathError, match="does not exist"):
        load_inference_manifest(tmp_path, "ascend")

    escaped_artifact = copy.deepcopy(manifest)
    escaped_artifact["deployments"]["ascend"]["artifacts"]["model"]["path"] = "../ram_plus.om"
    write_manifest(tmp_path, escaped_artifact)
    with pytest.raises(ManifestValidationError, match="parent traversal"):
        load_inference_manifest(tmp_path, "ascend")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uuid", "not-a-uuid"),
        ("uuid", "00000000-0000-0000-0000-000000000000"),
        ("revision", 0),
        ("revision", True),
        ("revision", "1"),
    ],
)
def test_bundle_identity_requires_canonical_uuid_and_positive_integer_revision(tmp_path, field, value):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    manifest["bundle"][field] = value
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError):
        load_inference_manifest(tmp_path, "cpu")


@pytest.mark.parametrize("alias", ["ascend_om", "ascend_om_3403", "3403", "om"])
def test_removed_backend_aliases_are_rejected(tmp_path, alias):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    manifest["deployments"]["cpu"]["runtime_profile"]["target"]["runtime"] = alias
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=alias):
        load_inference_manifest(tmp_path, "cpu")


def test_bundle_content_changes_do_not_require_runtime_rehashing(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    write_manifest(tmp_path, manifest)
    before = load_inference_manifest(tmp_path, "cpu")
    config = tmp_path / "config.json"
    config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = load_inference_manifest(tmp_path, "cpu")
    assert after.fingerprint == before.fingerprint


def test_artifact_content_changes_are_left_to_backend_validation(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    write_manifest(tmp_path, manifest)
    artifact = manifest["deployments"]["rk3588"]["artifacts"]["policy"]
    before = load_inference_manifest(tmp_path, "rk3588")
    (tmp_path / artifact["path"]).write_bytes(b"changed-compiled-policy")
    after = load_inference_manifest(tmp_path, "rk3588")
    assert after.fingerprint == before.fingerprint


def test_canonical_bundle_digest_is_order_independent():
    entries = [
        BundleFile(path="z.json"),
        BundleFile(path="a.json"),
    ]

    assert canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "test", entries) == canonical_bundle_digest(
        TEST_BUNDLE_UUID, 1, "test", reversed(entries)
    )
    assert canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "test", entries) != canonical_bundle_digest(
        TEST_BUNDLE_UUID, 2, "test", entries
    )


def test_deployment_fingerprint_is_stable_and_tracks_identity_changes():
    deployment = CompiledDeployment.model_validate_json(
        json.dumps(
            {
                "uuid": "123e4567-e89b-42d3-a456-426614174001",
                "revision": 1,
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {
                    "backend": "rknn",
                    "target": {"soc": "rk3588", "runtime": "rknn-lite"},
                    "profile": {"target_name": "rk3588", "core_mask": 7, "device_id": 0},
                },
                "artifacts": {"policy": {"path": "policy.rknn", "format": "rknn"}},
                "execution": ["policy"],
                "bindings": {
                    "policy": {
                        "inputs": [
                            {
                                "semantic": "observation.state",
                                "runtime_name": "state",
                                "dtype": "float32",
                                "shape": [1, 6],
                            }
                        ],
                        "outputs": [
                            {
                                "semantic": "action",
                                "runtime_name": "actions",
                                "dtype": "float32",
                                "shape": [1, -1, 6],
                            }
                        ],
                    }
                },
            }
        )
    )

    baseline = deployment_fingerprint(3, "4" * 64, "rk3588", deployment)
    assert baseline == deployment_fingerprint(3, "4" * 64, "rk3588", deployment)
    assert baseline != deployment_fingerprint(3, "5" * 64, "rk3588", deployment)
    assert baseline != deployment_fingerprint(3, "4" * 64, "other", deployment)

    changed_profile = deployment.runtime_profile.model_copy(
        update={"target": deployment.runtime_profile.target.model_copy(update={"runtime": "rknn-lite-2"})}
    )
    changed = deployment.model_copy(update={"runtime_profile": changed_profile})
    assert baseline != deployment_fingerprint(3, "4" * 64, "rk3588", changed)
    revised = deployment.model_copy(update={"revision": 2})
    assert baseline != deployment_fingerprint(3, "4" * 64, "rk3588", revised)


def test_device_link_source_defaults_preserve_canonical_identity():
    implicit = DeviceLink(
        semantic="internal.cache",
        producer="prefill",
        consumer="decode",
        transport="device_pointer",
        owner="producer",
    )
    explicit = implicit.model_copy(update={"producer_binding": "output"})
    input_sourced = implicit.model_copy(update={"producer_binding": "input"})

    assert implicit.producer_binding == "output"
    assert implicit.model_dump(mode="json", exclude_defaults=True) == explicit.model_dump(
        mode="json", exclude_defaults=True
    )
    assert "producer_binding" not in implicit.model_dump(mode="json", exclude_defaults=True)
    assert input_sourced.model_dump(mode="json", exclude_defaults=True)["producer_binding"] == "input"

    with pytest.raises(ValueError, match="owner='producer'"):
        DeviceLink(
            semantic="internal.cache",
            producer="prefill",
            consumer="decode",
            producer_binding="input",
            transport="device_pointer",
            owner="consumer",
        )


def test_missing_deployment_reports_available_names(tmp_path):
    paths = create_policy_bundle(tmp_path)
    write_manifest(tmp_path, make_manifest(tmp_path, paths))

    with pytest.raises(ManifestValidationError, match=r"'missing'.*available deployments: \['cpu'\]"):
        load_inference_manifest(tmp_path, "missing")


def test_compiled_execution_bindings_and_feature_compatibility(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "rk3588")
    assert validated.deployment.backend == "rknn"
    assert validated.policy.policy_type == "act"

    invalid_roles = copy.deepcopy(manifest)
    invalid_roles["deployments"]["rk3588"]["execution"] = ["missing"]
    write_manifest(tmp_path, invalid_roles)
    with pytest.raises(ManifestValidationError, match="binding roles must exactly match execution roles"):
        load_inference_manifest(tmp_path, "rk3588")

    invalid_layout = copy.deepcopy(manifest)
    del invalid_layout["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][1]["layout"]
    write_manifest(tmp_path, invalid_layout)
    with pytest.raises(ManifestValidationError, match=r"layout.*required"):
        load_inference_manifest(tmp_path, "rk3588")

    non_image_layout = copy.deepcopy(manifest)
    non_image_layout["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][0]["layout"] = "NCHW"
    write_manifest(tmp_path, non_image_layout)
    with pytest.raises(ManifestValidationError, match=r"not be valid"):
        load_inference_manifest(tmp_path, "rk3588")

    invalid_shape = copy.deepcopy(manifest)
    invalid_shape["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][0]["shape"] = [1, 5]
    write_manifest(tmp_path, invalid_shape)
    with pytest.raises(ManifestValidationError, match=r"observation.state.*incompatible"):
        load_inference_manifest(tmp_path, "rk3588")


def test_compiled_deployment_allows_verified_auxiliary_artifacts_and_sparse_output_indices(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="hisilicon", compiled=True, backend="hisilicon")
    deployment = manifest["deployments"]["hisilicon"]
    deployment["runtime_profile"]["target"] = {"soc": "sd3403", "runtime": "hisilicon-worker"}
    deployment["artifacts"]["policy"]["format"] = "om"
    deployment["bindings"]["policy"]["outputs"][0]["index"] = 1

    worker_path = tmp_path / "artifacts" / "worker"
    worker_path.write_bytes(b"worker")
    deployment["artifacts"]["worker"] = {
        "path": "artifacts/worker",
        "format": "executable",
    }
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "hisilicon")

    assert validated.deployment.execution == ("policy",)
    assert set(validated.deployment.artifacts) == {"policy", "worker"}
    assert validated.deployment.bindings["policy"].outputs[0].index == 1

    worker_path.write_bytes(b"changed-worker")
    assert load_inference_manifest(tmp_path, "hisilicon").deployment.backend == "hisilicon"


@pytest.mark.parametrize("policy_type", ["pi05", "smolvla"])
def test_vla_bindings_allow_declared_state_and_action_padding(tmp_path, policy_type):
    paths = tuple(
        path for path in create_policy_bundle(tmp_path, policy_type=policy_type) if path != "model.safetensors"
    )
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    bindings = manifest["deployments"]["rk3588"]["bindings"]["policy"]
    bindings["inputs"][0]["shape"] = [1, 32]
    bindings["outputs"][0]["shape"] = [1, 4, 32]
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "rk3588")

    assert validated.policy.policy_type == policy_type


def _host_orchestrated_manifest(tmp_path) -> tuple[dict, dict]:
    """A two-role perception deployment joined by the host, plus its second role's bindings.

    The first role consumes what the caller declared and hands an ``internal.`` embedding
    to the second over a device link. The second also consumes a ``host.`` tensor the host
    computes between them, and the semantic the service publishes is integrated on the
    host from that role's raw output rather than read out of a slot. This is the shape
    GraspGen has, expressed with nothing GraspGen-specific in it.
    """
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["outputs"] = [{"semantic": "grasp.poses", "dtype": "float32", "shape": [-1, 4, 4]}]
    deployment = manifest["deployments"]["ascend"]
    encoder = deployment["bindings"]["model"]
    encoder["outputs"] = [
        {
            "semantic": "internal.embedding",
            "runtime_name": "embedding",
            "index": 0,
            "dtype": "float32",
            "shape": [1, 512],
        }
    ]
    head = {
        "inputs": [
            {
                "semantic": "internal.embedding",
                "runtime_name": "object_embedding",
                "index": 0,
                "dtype": "float32",
                "shape": [1, 512],
            },
            {
                "semantic": "host.sample",
                "runtime_name": "sample",
                "index": 1,
                "dtype": "float32",
                "shape": [16, 6],
            },
        ],
        "outputs": [
            {
                "semantic": "host.logits",
                "runtime_name": "logits",
                "index": 0,
                "dtype": "float32",
                "shape": [16, 1],
            }
        ],
    }
    head_artifact = "artifacts/head.om"
    (tmp_path / head_artifact).write_bytes(b"compiled-head")
    deployment["artifacts"]["head"] = {"path": head_artifact, "format": "om"}
    deployment["execution"] = ["model", "head"]
    deployment["bindings"]["head"] = head
    deployment["device_links"] = [
        {
            "semantic": "internal.embedding",
            "producer": "model",
            "consumer": "head",
            "transport": "device_pointer",
            "owner": "producer",
        }
    ]
    return manifest, head


def test_host_orchestrated_deployment_need_not_bind_every_declared_semantic(tmp_path):
    """``grasp.poses`` is integrated between roles, so no OM slot carries it."""
    manifest, _ = _host_orchestrated_manifest(tmp_path)
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "ascend")

    assert validated.deployment.execution == ("model", "head")
    assert {descriptor.semantic for descriptor in validated.manifest.model.outputs} == {"grasp.poses"}


def test_host_inputs_do_not_need_an_in_graph_producer(tmp_path):
    """``host.sample`` has no producing role by construction, unlike an ``internal.`` tensor."""
    manifest, head = _host_orchestrated_manifest(tmp_path)
    write_manifest(tmp_path, manifest)

    inputs = load_inference_manifest(tmp_path, "ascend").deployment.bindings["head"].inputs

    assert [binding.semantic for binding in inputs] == ["internal.embedding", "host.sample"]
    assert head["inputs"][1]["semantic"].startswith("host.")


def test_internal_inputs_still_need_an_in_graph_producer(tmp_path):
    """Relaxing the producer rule for ``host.`` must not relax it for ``internal.``."""
    manifest, head = _host_orchestrated_manifest(tmp_path)
    head["inputs"][1]["semantic"] = "internal.sample"
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="internal.sample"):
        load_inference_manifest(tmp_path, "ascend")


def test_host_orchestration_does_not_excuse_an_undeclared_external_binding(tmp_path):
    """The ``missing`` check relaxes; the ``unexpected`` check does not."""
    manifest, head = _host_orchestrated_manifest(tmp_path)
    head["inputs"][1]["semantic"] = "gripper.width"
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="undeclared semantic input bindings"):
        load_inference_manifest(tmp_path, "ascend")


def test_a_straight_through_deployment_still_binds_every_declared_semantic(tmp_path):
    """Without a ``host.`` binding the 1:1 mapping stays strict."""
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["outputs"].append({"semantic": "tag_scores", "dtype": "float32", "shape": [1, 4585]})
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="omits declared semantic output bindings"):
        load_inference_manifest(tmp_path, "ascend")


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_vla_visual_bindings_allow_compiled_resize_dimensions(tmp_path, layout):
    paths = tuple(path for path in create_policy_bundle(tmp_path, policy_type="smolvla") if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    image_binding = manifest["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][1]
    image_binding["layout"] = layout
    image_binding["shape"] = [1, 3, 512, 512] if layout == "NCHW" else [1, 512, 512, 3]
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "rk3588")

    assert validated.policy.policy_type == "smolvla"


def test_rank_four_layout_is_scoped_to_image_semantics(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    image_binding = manifest["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][1]

    image_binding["semantic"] = "depth.frame"
    manifest["model"]["inputs"][1]["semantic"] = "depth.frame"
    del image_binding["layout"]
    write_manifest(tmp_path, manifest)
    validated = load_inference_manifest(tmp_path, "rk3588")
    assert validated.deployment.bindings["policy"].inputs[1].layout is None

    image_binding["layout"] = "NCHW"
    write_manifest(tmp_path, manifest)
    validated = load_inference_manifest(tmp_path, "rk3588")
    assert validated.deployment.bindings["policy"].inputs[1].layout == "NCHW"

    image_binding["semantic"] = "observation.images.top"
    manifest["model"]["inputs"][1]["semantic"] = "observation.images.top"
    del image_binding["layout"]
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match=r"layout.*required"):
        load_inference_manifest(tmp_path, "rk3588")

    image_binding["layout"] = "NCHW"
    write_manifest(tmp_path, manifest)
    validated = load_inference_manifest(tmp_path, "rk3588")
    assert validated.deployment.bindings["policy"].inputs[1].layout == "NCHW"


def test_selected_bundle_and_artifact_paths_cannot_duplicate(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    artifact = manifest["deployments"]["rk3588"]["artifacts"]["policy"]
    manifest["bundle"]["files"].append({"path": artifact["path"]})
    entries = [BundleFile.model_validate(entry) for entry in manifest["bundle"]["files"]]
    manifest["bundle"]["digest"]["value"] = canonical_bundle_digest(
        manifest["bundle"]["uuid"],
        manifest["bundle"]["revision"],
        manifest["bundle"]["name"],
        entries,
    )
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="distinct paths"):
        load_inference_manifest(tmp_path, "rk3588")


def test_canonical_writer_is_deterministic_and_replaces_atomically(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    destination = tmp_path / "nested" / "inference_manifest.json"

    first = write_inference_manifest(destination, manifest)
    first_content = first.read_bytes()
    assert first_content == canonical_manifest_bytes(manifest)

    reordered = {
        "deployments": manifest["deployments"],
        "model": manifest["model"],
        "bundle": manifest["bundle"],
        "schema_version": 3,
    }
    write_inference_manifest(destination, reordered)
    assert destination.read_bytes() == first_content
    assert not list(destination.parent.glob("*.tmp"))

    unsafe = copy.deepcopy(manifest)
    unsafe["bundle"]["files"][0]["path"] = "../config.json"
    with pytest.raises(ManifestValidationError, match="parent traversal"):
        write_inference_manifest(destination, unsafe)


def test_manifest_package_and_loading_are_dependency_neutral(monkeypatch, tmp_path):
    forbidden_roots = {
        "acl",
        "geometry_msgs",
        "inference_service",
        "model_utils",
        "rclpy",
        "rknn",
        "rknnlite",
        "robot_config",
        "sensor_msgs",
        "std_msgs",
        "tcim",
        "torch",
        "torch_npu",
    }
    code = f"""
import sys
import inference_manifest

forbidden = {forbidden_roots!r}
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
if loaded:
    raise SystemExit(f"forbidden imports loaded by inference_manifest: {{loaded}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=os.environ.copy(),
        text=True,
    )
    assert result.returncode == 0, result.stderr

    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    write_manifest(tmp_path, make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True))
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in forbidden_roots:
            attempted.append(name)
            raise AssertionError(f"forbidden dependency import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    load_inference_manifest(tmp_path, "rk3588")

    assert attempted == []
