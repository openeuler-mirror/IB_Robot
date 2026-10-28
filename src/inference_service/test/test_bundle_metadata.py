from __future__ import annotations

import copy
import json

import pytest

from inference_manifest import (
    BundleFile,
    ManifestValidationError,
    canonical_bundle_digest,
    load_inference_manifest,
    load_inference_manifest_metadata,
    load_policy_metadata,
)
from test.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    TEST_DEPLOYMENT_UUID,
    create_policy_bundle,
    make_manifest,
    write_manifest,
)


@pytest.mark.parametrize(
    ("policy_type", "local_tokenizer", "expected_external", "expected_local_assets"),
    [
        ("act", False, (), ()),
        ("pi05", False, (("tokenizer_name", "bert-base-uncased"),), ()),
        (
            "smolvla",
            False,
            (
                ("tokenizer_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
                ("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
            ),
            (),
        ),
        (
            "smolvla",
            True,
            (),
            ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"),
        ),
    ],
)
def test_policy_family_discovery_summaries_and_digests(
    tmp_path,
    policy_type,
    local_tokenizer,
    expected_external,
    expected_local_assets,
):
    paths = create_policy_bundle(tmp_path, policy_type, local_tokenizer=local_tokenizer)
    metadata = load_policy_metadata(tmp_path, require_native_weights=True)

    assert metadata.policy_type == policy_type
    assert metadata.input_features["observation.state"].shape == (6,)
    assert metadata.input_features["observation.images.top"].type == "VISUAL"
    assert metadata.output_features["action"].shape == (6,)
    assert tuple((item.source, item.identifier) for item in metadata.external_dependencies) == expected_external
    assert all(path in metadata.required_files for path in expected_local_assets)

    manifest = make_manifest(tmp_path, paths)
    write_manifest(tmp_path, manifest)
    validated = load_inference_manifest(tmp_path, "cpu")
    declared_entries = tuple(BundleFile.model_validate(entry) for entry in manifest["bundle"]["files"])

    assert validated.manifest.bundle.digest.value == canonical_bundle_digest(
        manifest["bundle"]["uuid"],
        manifest["bundle"]["revision"],
        manifest["bundle"]["name"],
        declared_entries,
    )
    assert validated.policy == metadata


def test_metadata_loading_is_read_only_and_preserves_original_device(tmp_path):
    create_policy_bundle(tmp_path, "act")
    config_path = tmp_path / "config.json"
    before = config_path.read_bytes()

    metadata = load_policy_metadata(tmp_path, require_native_weights=True)

    assert metadata.policy_type == "act"
    assert config_path.read_bytes() == before
    assert b'"device": "cuda"' in before


def test_policy_metadata_reads_optional_action_generation_dimensions(tmp_path):
    create_policy_bundle(tmp_path, "pi05")
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"chunk_size": 50, "max_action_dim": 32})
    config_path.write_text(json.dumps(config), encoding="utf-8")

    metadata = load_policy_metadata(tmp_path)

    assert metadata.nominal_chunk_size == 50
    assert metadata.max_action_dimension == 32


def test_diffusion_metadata_uses_observation_and_action_steps(tmp_path):
    create_policy_bundle(tmp_path, "diffusion")
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"n_obs_steps": 2, "horizon": 64, "n_action_steps": 44, "chunk_size": 99})
    config_path.write_text(json.dumps(config), encoding="utf-8")

    metadata = load_policy_metadata(tmp_path)

    assert metadata.n_obs_steps == 2
    assert metadata.nominal_chunk_size == 44


@pytest.mark.parametrize(("policy_type", "expected"), [("act", 1), ("diffusion", 2)])
def test_policy_metadata_defaults_observation_steps_by_policy(tmp_path, policy_type, expected):
    create_policy_bundle(tmp_path, policy_type)

    metadata = load_policy_metadata(tmp_path)

    assert metadata.n_obs_steps == expected


@pytest.mark.parametrize(
    ("policy_type", "key", "value"),
    [
        ("pi05", "chunk_size", 0),
        ("pi05", "max_action_dim", -1),
        ("diffusion", "n_obs_steps", 0),
        ("diffusion", "n_action_steps", 0),
        ("diffusion", "n_action_steps", 1.5),
    ],
)
def test_policy_metadata_rejects_invalid_action_generation_dimensions(tmp_path, policy_type, key, value):
    create_policy_bundle(tmp_path, policy_type)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[key] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match=key):
        load_policy_metadata(tmp_path)


def test_missing_required_semantic_file_is_rejected_but_arbitrary_extra_is_allowed(tmp_path):
    paths = create_policy_bundle(tmp_path, "act")
    manifest = make_manifest(tmp_path, paths)
    omitted = "policy_preprocessor_step_0_normalizer_processor.safetensors"
    manifest["bundle"]["files"] = [entry for entry in manifest["bundle"]["files"] if entry["path"] != omitted]
    entries = tuple(BundleFile.model_validate(entry) for entry in manifest["bundle"]["files"])
    manifest["bundle"]["digest"]["value"] = canonical_bundle_digest(
        manifest["bundle"]["uuid"],
        manifest["bundle"]["revision"],
        manifest["bundle"]["name"],
        entries,
    )
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=omitted):
        load_inference_manifest(tmp_path, "cpu")

    extra_file = tmp_path / "notes.bin"
    extra_file.write_bytes(b"non-semantic deployment notes")
    extra_manifest = make_manifest(tmp_path, paths + ("notes.bin",))
    write_manifest(tmp_path, extra_manifest)

    assert load_inference_manifest(tmp_path, "cpu").policy.policy_type == "act"


def test_unreferenced_reserved_semantic_file_is_rejected(tmp_path):
    paths = create_policy_bundle(tmp_path, "act")
    orphan = "policy_preprocessor_step_9_orphan.safetensors"
    (tmp_path / orphan).write_bytes(b"orphan")
    manifest = make_manifest(tmp_path, paths + (orphan,))
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=r"unreferenced LeRobot semantic files.*orphan"):
        load_inference_manifest(tmp_path, "cpu")


def test_torch_requires_native_weights_while_compiled_metadata_does_not(tmp_path):
    paths = create_policy_bundle(tmp_path, "act", include_weights=False)
    torch_manifest = make_manifest(tmp_path, paths)
    write_manifest(tmp_path, torch_manifest)
    with pytest.raises(ManifestValidationError, match="model.safetensors"):
        load_inference_manifest(tmp_path, "cpu")

    compiled_manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    write_manifest(tmp_path, compiled_manifest)
    assert load_inference_manifest(tmp_path, "rk3588").policy.native_weights_required is False


def test_multi_deployment_bundle_keeps_native_weights_when_compiled_is_selected(tmp_path):
    paths = create_policy_bundle(tmp_path, "act")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    manifest["deployments"]["cpu"] = {
        "uuid": TEST_DEPLOYMENT_UUID,
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

    validated = load_inference_manifest(tmp_path, "rk3588")

    assert validated.policy.native_weights_required is True
    assert "model.safetensors" in validated.policy.required_files


def test_metadata_only_loader_does_not_require_cloud_artifacts_or_native_weights(tmp_path):
    paths = create_policy_bundle(tmp_path, "act")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    manifest["deployments"]["cpu"] = {
        "uuid": TEST_DEPLOYMENT_UUID,
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
    (tmp_path / "artifacts/policy.rknn").unlink()
    (tmp_path / "model.safetensors").unlink()

    validated = load_inference_manifest_metadata(tmp_path, "rk3588")

    assert validated.policy.native_weights_required is False
    assert validated.deployment.backend == "rknn"


def test_local_tokenizer_reference_must_not_escape_or_go_missing(tmp_path):
    create_policy_bundle(tmp_path, "smolvla", local_tokenizer=True)
    preprocessor_path = tmp_path / "policy_preprocessor.json"
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    preprocessor["steps"][1]["config"]["tokenizer_name"] = "./missing-tokenizer"
    preprocessor_path.write_text(json.dumps(preprocessor), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="Invalid local tokenizer_name reference"):
        load_policy_metadata(tmp_path, require_native_weights=True)

    outside = tmp_path.parent / "outside-tokenizer"
    outside.mkdir(exist_ok=True)
    absolute = copy.deepcopy(preprocessor)
    absolute["steps"][1]["config"]["tokenizer_name"] = str(outside)
    preprocessor_path.write_text(json.dumps(absolute), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="escapes the policy bundle"):
        load_policy_metadata(tmp_path, require_native_weights=True)


def test_policy_fixture_digest_tracks_revision_not_file_bytes(tmp_path):
    paths = create_policy_bundle(tmp_path, "pi05")
    before_entries = tuple(BundleFile(path=path) for path in paths)
    before = canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "pi05", before_entries)

    state_path = tmp_path / "policy_preprocessor_step_0_normalizer_processor.safetensors"
    state_path.write_bytes(b"changed-state")
    after_entries = tuple(BundleFile(path=path) for path in reversed(paths))
    unchanged = canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "pi05", after_entries)
    revised = canonical_bundle_digest(TEST_BUNDLE_UUID, 2, "pi05", after_entries)

    assert before == unchanged
    assert before != revised
