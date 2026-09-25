"""Tests for audio contract validation across audio launch boundaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1]
_WORKSPACE_SRC = _SRC.parent
for package_root in (_SRC, _WORKSPACE_SRC / "inference_manifest", _WORKSPACE_SRC / "voice_asr_service"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from voice_asr_service.package_sherpa_asr_bundle import package_sherpa_asr_bundle  # noqa: E402
from voice_asr_service.package_silero_vad_bundle import package_silero_vad_bundle  # noqa: E402

_SPEECH_DIRECTION_LAUNCH = _SRC / "launch" / "speech_direction.launch.py"


def _asr_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    artifacts = {}
    for name in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
        path = source / name
        path.write_bytes(name.encode())
        artifacts[name.removesuffix(".txt") if name == "tokens.txt" else name.removesuffix(".onnx")] = path
    bundle = tmp_path / "asr-bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=True, include_cuda=False)
    return bundle


def _silero_bundle(tmp_path: Path, *, include_ascend: bool = False) -> Path:
    onnx = tmp_path / "silero.onnx"
    onnx.write_bytes(b"fake-silero-onnx")
    om = tmp_path / "silero.om"
    om.write_bytes(b"fake-silero-om")
    om_310b = tmp_path / "silero_310b.om"
    om_310b.write_bytes(b"fake-silero-310b-om")
    bundle = tmp_path / "silero-vad"
    package_silero_vad_bundle(
        bundle,
        onnx_source=onnx,
        om_source=om if include_ascend else None,
        om_310b_source=om_310b if include_ascend else None,
        include_ascend=include_ascend,
    )
    return bundle


def _load_speech_direction_launch():
    spec = importlib.util.spec_from_file_location("speech_direction_launch_under_test", _SPEECH_DIRECTION_LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fullsubnet_bundle(tmp_path: Path) -> Path:
    from voice_asr_service.package_fullsubnet_bundle import package_fullsubnet_bundle

    bundle = tmp_path / "fullsubnet"
    (bundle / "assets").mkdir(parents=True)
    (bundle / "assets" / "adapter.json").write_text("{}", encoding="utf-8")
    (bundle / "assets" / "cum_fullsubnet_best_model_218epochs.tar").write_bytes(b"checkpoint")
    (bundle / "assets" / "cum_fullsubnet_best_model_218epochs.manifest.json").write_bytes(b"{}")
    fb = bundle / "artifacts" / "ascend" / "fullsubnet"
    fb.mkdir(parents=True)
    (fb / "fullsubnet_cum_stateful_fb_b4_t2_fp16.om").write_bytes(b"fb-om")
    (fb / "fullsubnet_cum_stateful_sb_b4_t2_fp16.om").write_bytes(b"sb-om")
    fb310b = bundle / "artifacts" / "ascend_310b" / "fullsubnet"
    fb310b.mkdir(parents=True)
    (fb310b / "fullsubnet_cum_stateful_fb_b4_t2_310b_origin.om").write_bytes(b"fb-om-310b")
    (fb310b / "fullsubnet_cum_stateful_sb_b4_t2_310b_origin.om").write_bytes(b"sb-om-310b")
    package_fullsubnet_bundle(bundle)
    return bundle


def test_speech_direction_platform_matches_pass_validation(tmp_path):
    launch = _load_speech_direction_launch()
    models_root = tmp_path / "models"
    models_root.mkdir()
    _silero_bundle(models_root)
    _fullsubnet_bundle(models_root)

    params = {
        "silero_vad_backend": "onnx",
        "silero_vad_deployment": "torch_cpu",
        "silero_vad_inference_bundle": str(models_root / "silero-vad"),
        "fullsubnet_backend": "stateful_torch_cuda",
        "fullsubnet_deployment": "torch_cuda",
        "speech_direction_inference_bundle": str(models_root / "fullsubnet"),
    }
    launch._validate_deployment_platform(params, models_root)


def test_speech_direction_backend_deployment_mismatch_is_rejected(tmp_path):
    launch = _load_speech_direction_launch()
    models_root = tmp_path / "models"
    models_root.mkdir()
    _silero_bundle(models_root, include_ascend=True)
    _fullsubnet_bundle(models_root)

    params = {
        "silero_vad_backend": "onnx",
        "silero_vad_deployment": "ascend_310p",
        "silero_vad_inference_bundle": str(models_root / "silero-vad"),
        "fullsubnet_backend": "ascend",
        "fullsubnet_deployment": "ascend_310p",
        "speech_direction_inference_bundle": str(models_root / "fullsubnet"),
    }
    with pytest.raises(ValueError, match="manifest backend 是 ascend"):
        launch._validate_deployment_platform(params, models_root)


def test_speech_direction_missing_deployment_bundle_is_rejected(tmp_path):
    launch = _load_speech_direction_launch()
    models_root = tmp_path / "models"
    models_root.mkdir()

    params = {
        "silero_vad_backend": "onnx",
        "silero_vad_deployment": "torch_cpu",
        "fullsubnet_backend": "ascend",
        "fullsubnet_deployment": "ascend_310p",
        "speech_direction_inference_bundle": str(models_root / "fullsubnet"),
    }
    with pytest.raises(ValueError, match="无法从"):
        launch._validate_deployment_platform(params, models_root)
