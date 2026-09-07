"""Contract tests for schema-v3 sherpa ASR bundles."""

from pathlib import Path

import pytest

from inference_manifest import load_inference_manifest
from inference_manifest.errors import ManifestValidationError
from voice_asr_service.package_sherpa_asr_bundle import package_sherpa_asr_bundle


def _artifact_dir(tmp_path: Path, names: tuple[str, ...]) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir()
    result = {}
    for name in names:
        path = source / f"{name}.onnx" if name != "tokens" else source / "tokens.txt"
        path.write_bytes(name.encode())
        result[name] = path
    return result


def test_package_streaming_transducer_roles(tmp_path: Path) -> None:
    artifacts = _artifact_dir(tmp_path, ("tokens", "encoder", "decoder", "joiner"))
    bundle = tmp_path / "bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=True)

    validated = load_inference_manifest(bundle, "torch_cuda")
    assert validated.identity.model_type == "sherpa_onnx"
    assert set(validated.resolved_artifacts) == {"tokens", "encoder", "decoder", "joiner"}


def test_package_streaming_paraformer_roles(tmp_path: Path) -> None:
    artifacts = _artifact_dir(tmp_path, ("tokens", "encoder", "decoder"))
    bundle = tmp_path / "bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=True, streaming_kind="paraformer")

    validated = load_inference_manifest(bundle, "torch_cpu")
    assert set(validated.resolved_artifacts) == {"tokens", "encoder", "decoder"}


def test_package_offline_roles_are_explicit(tmp_path: Path) -> None:
    artifacts = _artifact_dir(tmp_path, ("tokens", "model"))
    bundle = tmp_path / "bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=False, include_cuda=False)

    validated = load_inference_manifest(bundle, "torch_cpu")
    assert set(validated.resolved_artifacts) == {"tokens", "model"}
    with pytest.raises(ManifestValidationError):
        load_inference_manifest(bundle, "torch_cuda")
