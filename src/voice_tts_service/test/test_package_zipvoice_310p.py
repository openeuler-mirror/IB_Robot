import hashlib
import json
from pathlib import Path

import numpy as np

from inference_manifest import load_inference_manifest, validate_deployment_artifact_integrity
from voice_tts_service import package_zipvoice_310p as packager


def _write(root: Path, relative: str, data: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_packager_creates_valid_scalar_abi_bundle_from_verified_delivery(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "bundle"
    payloads = {
        packager.TEXT_OM: b"text-om",
        packager.FLOW_OM: b"flow-om",
        packager.TOKENS: b"_\t0\n.\t1\n",
        packager.VOCOS_CHECKPOINT: b"vocos",
    }
    for relative, data in payloads.items():
        _write(source, relative, data)
    text_fixture = source / packager.TEXT_GOLDEN
    text_fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        text_fixture,
        prompt_tokens=np.zeros((1, 29), dtype=np.int64),
        prompt_features_len=np.asarray(302, dtype=np.int64),
    )
    flow_fixture = source / packager.FLOW_GOLDEN
    np.savez(flow_fixture, speech_condition=np.zeros((1, 1537, 100), dtype=np.float32))
    for relative in (packager.TEXT_GOLDEN, packager.FLOW_GOLDEN):
        payloads[relative] = (source / relative).read_bytes()
    monkeypatch.setattr(
        packager,
        "EXPECTED_SHA256",
        {relative: hashlib.sha256(data).hexdigest() for relative, data in payloads.items()},
    )
    manifest_path = packager.package_bundle(source, destination)
    validated = load_inference_manifest(destination, "ascend_310p")

    validate_deployment_artifact_integrity(validated)
    assert manifest_path == destination / "inference_manifest.json"
    assert validated.deployment.target.soc == "Ascend310P1"
    assert validated.deployment.bindings["text_encoder"].inputs[1].shape == ()
    assert validated.deployment.bindings["text_encoder"].outputs[0].runtime_name == "/Where_5:0:text_condition"
    assert validated.deployment.bindings["flow_decoder_1537"].outputs[0].shape == (1, 1537, 100)
    assert validated.deployment.bindings["flow_decoder_1537"].outputs[0].runtime_name.endswith(":0:v")
    assert not (destination / "assets/adapter.json").exists()
    assert not (destination / "assets/vendor").exists()
    runtime = json.loads((destination / "assets/zipvoice_310p.json").read_text(encoding="utf-8"))
    assert "vendor_python_path" not in runtime
    assert "vocos_vendor_path" not in runtime


def test_packager_rejects_unverified_same_name_asset(tmp_path, monkeypatch):
    relative = packager.TEXT_OM
    _write(tmp_path, relative, b"wrong-model")
    monkeypatch.setattr(packager, "EXPECTED_SHA256", {relative: "0" * 64})

    try:
        packager._require_source(tmp_path, relative)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("checksum mismatch was not rejected")


def test_packager_rejects_non_empty_destination(tmp_path):
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    try:
        packager.package_bundle(tmp_path, destination)
    except ValueError as exc:
        assert "must be empty" in str(exc)
    else:
        raise AssertionError("non-empty destination was not rejected")
