import hashlib
import json

import pytest

from inference_manifest import (
    load_inference_manifest,
    validate_deployment_artifact_integrity,
    verify_deployment_artifacts,
)
from perception_service.package_ascend_perception_bundles import _specs, package_bundle
from perception_service.semantic_model_adapters import SigLIP2ImageAdapter


def _write_sources(root, spec):
    for deployment in spec.deployments:
        for artifact in deployment.artifacts:
            path = root / artifact.source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact.role.encode())
    for source, _destination in spec.assets:
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")


def test_compiled_ascend_bundle_specs_validate(tmp_path):
    for family, spec in _specs().items():
        _write_sources(tmp_path, spec)
        manifest_path = package_bundle(tmp_path, spec)
        assert manifest_path.is_file()
        for deployment in spec.deployments:
            validated = load_inference_manifest(manifest_path.parent, deployment.name)
            validate_deployment_artifact_integrity(validated)
            assert validated.deployment.target.soc == deployment.soc
            assert set(validated.deployment.artifacts) == {artifact.role for artifact in deployment.artifacts}
        if family == "grounding_dino":
            assert len(validated.deployment.execution) == 12
            assert validated.deployment.device_links
            assert (manifest_path.parent / "assets/bert-base-uncased/vocab.txt").read_bytes() == b"asset"
            vision_outputs = validated.deployment.bindings["vision"].outputs
            assert [binding.runtime_name for binding in vision_outputs] == [
                "/input_proj.0/input_proj.0.1/Add:0:src0",
                "/input_proj.1/input_proj.1.1/Add:0:src1",
                "/input_proj.2/input_proj.2.1/Add:0:src2",
                "/input_proj.3/input_proj.3.1/Add:0:src3",
            ]
            assert validated.deployment.bindings["flatten"].outputs[0].runtime_name == "/Concat_4:0:visual"


def test_nested_integrity_report_name_is_a_versioned_asset(tmp_path):
    spec = _specs()["sam2"]
    _write_sources(tmp_path, spec)
    root = tmp_path / spec.name
    asset = root / "assets/inference_integrity.json"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"original asset")
    package_bundle(tmp_path, spec)
    first = load_inference_manifest(root, spec.deployments[0].name)
    sidecar = root / "assets/artifact-digests.json"
    assert (
        json.loads(sidecar.read_text())["assets/inference_integrity.json"]
        == hashlib.sha256(b"original asset").hexdigest()
    )

    verify_deployment_artifacts(root, spec.deployments[0].name)
    package_bundle(tmp_path, spec)
    unchanged = load_inference_manifest(root, spec.deployments[0].name)
    assert unchanged.manifest.bundle.revision == first.manifest.bundle.revision
    assert not (root / "inference_integrity.json").exists()
    assert "inference_integrity.json" not in json.loads(sidecar.read_text())

    asset.write_bytes(b"changed asset")
    package_bundle(tmp_path, spec)
    changed = load_inference_manifest(root, spec.deployments[0].name)
    assert changed.manifest.bundle.revision == first.manifest.bundle.revision + 1
    assert changed.fingerprint != first.fingerprint
    assert (
        json.loads(sidecar.read_text())["assets/inference_integrity.json"]
        == hashlib.sha256(b"changed asset").hexdigest()
    )


def test_packager_rejects_existing_legacy_bundles(tmp_path):
    expected_identities = {
        "sam2": ("sam2_prompt", "sam2", "prompt"),
        "grounding_dino": ("grounding_dino_raw", "grounding_dino", "raw"),
    }
    for family, (legacy_family, _expected_family, _expected_operation) in expected_identities.items():
        spec = _specs()[family]
        _write_sources(tmp_path, spec)
        manifest_path = package_bundle(tmp_path, spec)
        legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy["model"]["family"] = legacy_family
        legacy["model"].pop("operation", None)
        manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

        with pytest.raises(Exception, match="schema_version|model_type|interface"):
            package_bundle(tmp_path, spec)


def test_sam2_ascend_bundle_name_identifies_box_prompt_contract():
    spec = _specs()["sam2"]

    assert spec.name == "sam2.1_hiera_tiny_prompt_ascend"
    assert spec.operation == "prompt"
    assert spec.name != "sam2.1_hiera_tiny"


def test_siglip2_ascend_bundle_declares_loadable_embedding_identity(tmp_path, monkeypatch):
    spec = _specs()["siglip2"]
    _write_sources(tmp_path, spec)
    manifest_path = package_bundle(tmp_path, spec)
    validated = load_inference_manifest(manifest_path.parent, "ascend_310b")
    monkeypatch.setattr(
        "perception_service.semantic_model_adapters._load_siglip2_tokenizer",
        lambda _path: object(),
    )

    adapter = SigLIP2ImageAdapter.from_bundle(
        manifest_path.parent,
        validated.manifest.model.semantic_identity,
        model=validated.manifest.model,
        deployment=validated.deployment,
    )

    assert adapter.dimension == 1152
    assert validated.manifest.model.semantic_identity.logical_model_revision == (
        "google/siglip2-so400m-patch14-384@main"
    )
    assert validated.deployment.bindings["vision"].inputs[0].runtime_name == "image"
    assert validated.deployment.bindings["text"].inputs[0].runtime_name == "input_ids"
    assert validated.deployment.bindings["vision"].outputs[0].runtime_name is None
    assert validated.deployment.bindings["text"].outputs[0].runtime_name is None


def test_compiled_ascend_bundle_revisions_follow_artifact_content(tmp_path):
    spec = _specs()["sam2"]
    _write_sources(tmp_path, spec)
    manifest_path = package_bundle(tmp_path, spec)
    first = load_inference_manifest(manifest_path.parent, "ascend_310p").manifest

    package_bundle(tmp_path, spec)
    unchanged = load_inference_manifest(manifest_path.parent, "ascend_310p").manifest
    assert unchanged.bundle.revision == first.bundle.revision
    assert unchanged.deployments["ascend_310p"].revision == first.deployments["ascend_310p"].revision

    changed_artifact = tmp_path / spec.deployments[0].artifacts[0].source
    changed_artifact.write_bytes(b"changed encoder")
    package_bundle(tmp_path, spec)
    changed = load_inference_manifest(manifest_path.parent, "ascend_310p").manifest
    assert changed.bundle.revision == first.bundle.revision + 1
    assert changed.deployments["ascend_310p"].revision == first.deployments["ascend_310p"].revision + 1
    assert changed.deployments["ascend_310b"].revision == first.deployments["ascend_310b"].revision


@pytest.mark.parametrize("state", ["verified", "mismatch"])
def test_repackaging_does_not_reuse_a_previous_deployments_report(tmp_path, state):
    from inference_manifest import verify_deployment_artifacts

    spec = _specs()["sam2"]
    _write_sources(tmp_path, spec)
    root = package_bundle(tmp_path, spec).parent
    deployment = spec.deployments[0]
    artifact = deployment.artifacts[0]
    old = load_inference_manifest(root, deployment.name)
    if state == "mismatch":
        (root / artifact.destination).write_bytes(b"tampered")
    assert verify_deployment_artifacts(root, deployment.name).state == state
    (tmp_path / artifact.source).write_bytes(b"new model generation")
    package_bundle(tmp_path, spec)
    new = load_inference_manifest(root, deployment.name)
    assert new.deployment.revision == old.deployment.revision + 1
    assert new.integrity_status.state == "declared"
    validate_deployment_artifact_integrity(new)
