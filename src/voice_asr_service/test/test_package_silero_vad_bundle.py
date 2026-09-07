"""Tests for the standalone Silero VAD bundle packager."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voice_asr_service.package_silero_vad_bundle import (
    DEFAULT_OM_REL,
    DEFAULT_ONNX_REL,
    package_silero_vad_bundle,
)


def _write_model(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    onnx = tmp_path / "models" / "silero-vad" / "assets" / "silero_vad.onnx"
    om = tmp_path / "models" / "silero-vad" / "artifacts" / "ascend" / "ascend_310p" / "silero_vad_v6_310p_mixed16.om"
    _write_model(onnx, b"fake-v6-onnx-bytes")
    _write_model(om, b"fake-310p-om-bytes")
    return tmp_path


def test_bundle_contains_both_deployments(tmp_path: Path, workspace: Path) -> None:
    bundle_root = tmp_path / "models" / "silero-vad"
    manifest_path = package_silero_vad_bundle(bundle_root, workspace=workspace)
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["model"]["model_type"] == "silero_vad"
    assert manifest["model"]["operation"] == "vad"
    assert set(manifest["deployments"]) == {"ascend_310p", "torch_cpu"}

    onnx = manifest["deployments"]["torch_cpu"]
    assert onnx["runtime_profile"]["backend"] == "onnx"
    assert onnx["runtime_profile"]["target"]["runtime"] == "onnx"
    assert onnx["artifacts"]["silero_vad"] == {
        "format": "onnx",
        "path": DEFAULT_ONNX_REL,
        "sha256": hashlib.sha256(b"fake-v6-onnx-bytes").hexdigest(),
    }
    assert onnx["execution"] == ["silero_vad"]
    assert onnx["bindings"]["silero_vad"]["inputs"]
    assert onnx["execution_contract"]["stateful"] is True
    assert onnx["audio_contract"]["sample_rate_hz"] == 16000

    ascend = manifest["deployments"]["ascend_310p"]
    artifact = ascend["artifacts"]["silero_vad"]
    assert artifact["path"] == DEFAULT_OM_REL
    assert artifact["sha256"] == hashlib.sha256(b"fake-310p-om-bytes").hexdigest()
    assert ascend["execution"] == ["silero_vad"]
    assert ascend["execution_contract"]["state_scope"] == "stream"
    assert ascend["execution_contract"]["stateful"] is True
    assert "host.silero.sample_rate" not in {item["semantic"] for item in ascend["bindings"]["silero_vad"]["inputs"]}

    assert (bundle_root / DEFAULT_ONNX_REL).read_bytes() == b"fake-v6-onnx-bytes"
    assert (bundle_root / DEFAULT_OM_REL).read_bytes() == b"fake-310p-om-bytes"
    files = {entry["path"] for entry in manifest["bundle"]["files"]}
    assert DEFAULT_ONNX_REL not in files
    assert "assets/adapter.json" in files
    assert DEFAULT_OM_REL not in files, "deployment artifacts must stay out of bundle.files"


def test_bundle_without_ascend_source(tmp_path: Path) -> None:
    onnx = tmp_path / "models" / "silero-vad" / "assets" / "silero_vad.onnx"
    _write_model(onnx, b"host-only")
    bundle_root = tmp_path / "models" / "silero-vad"
    package_silero_vad_bundle(bundle_root, workspace=tmp_path)

    manifest = json.loads((bundle_root / "inference_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["deployments"]) == {"torch_cpu"}
    assert not (bundle_root / DEFAULT_OM_REL).exists()


def test_missing_onnx_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_speech_direction_models"):
        package_silero_vad_bundle(tmp_path / "models" / "silero-vad", workspace=tmp_path)


def test_repackaging_is_stable(tmp_path: Path, workspace: Path) -> None:
    bundle_root = tmp_path / "models" / "silero-vad"
    package_silero_vad_bundle(bundle_root, workspace=workspace)
    first = json.loads((bundle_root / "inference_manifest.json").read_text(encoding="utf-8"))
    package_silero_vad_bundle(bundle_root, workspace=workspace)
    second = json.loads((bundle_root / "inference_manifest.json").read_text(encoding="utf-8"))
    assert first["bundle"]["uuid"] == second["bundle"]["uuid"]
    assert second["bundle"]["revision"] == first["bundle"]["revision"], "unchanged files must not bump revision"
