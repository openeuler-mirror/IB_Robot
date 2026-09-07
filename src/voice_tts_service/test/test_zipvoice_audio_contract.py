"""Synthesis audio contract tests for ZipVoice bundle metadata and validation."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
_WORKSPACE_SRC = _SRC.parent
for package_root in (_SRC, _WORKSPACE_SRC / "inference_manifest", _WORKSPACE_SRC / "voice_tts_service"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from inference_manifest import load_inference_manifest  # noqa: E402
from voice_tts_service import package_zipvoice_310p as packager  # noqa: E402
from voice_tts_service.model_service_plugin import validate_synthesis_audio_contract  # noqa: E402

_ZIPVOICE_BUNDLE = _WORKSPACE_SRC.parent / "models" / "zipvoice"


def _fake_validated(contract):
    return SimpleNamespace(deployment=SimpleNamespace(audio_contract=contract))


def test_synthesis_contract_accepts_declared_or_absent_contracts():
    validate_synthesis_audio_contract(_fake_validated(None))

    from inference_manifest import AudioContract

    validate_synthesis_audio_contract(
        _fake_validated(
            AudioContract(
                sample_rate_hz=24000,
                channels=1,
                channel_semantics="mono",
                sample_dtype="float32",
                execution_mode="offline",
            )
        )
    )


def test_synthesis_contract_rejects_sample_rate_mismatch():
    from inference_manifest import AudioContract

    with pytest.raises(ValueError, match="sample_rate_hz=16000"):
        validate_synthesis_audio_contract(
            _fake_validated(AudioContract(sample_rate_hz=16000, channels=1, sample_dtype="float32"))
        )


def test_synthesis_contract_rejects_non_mono_output():
    from inference_manifest import AudioContract

    with pytest.raises(ValueError, match="channels=2"):
        validate_synthesis_audio_contract(
            _fake_validated(AudioContract(sample_rate_hz=24000, channels=2, sample_dtype="float32"))
        )


def test_synthesis_contract_rejects_non_float_output():
    from inference_manifest import AudioContract

    with pytest.raises(ValueError, match="sample_dtype=int16"):
        validate_synthesis_audio_contract(
            _fake_validated(AudioContract(sample_rate_hz=24000, channels=1, sample_dtype="int16"))
        )


def test_packager_declares_synthesis_contract_without_microphone_fields(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "bundle"
    payloads = {
        packager.TEXT_OM: b"text-om",
        packager.FLOW_OM: b"flow-om",
        packager.TOKENS: b"_\t0\n.\t1\n",
        packager.VOCOS_CHECKPOINT: b"vocos",
    }
    for relative, data in payloads.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
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

    packager.package_bundle(source, destination)
    validated = load_inference_manifest(destination, "ascend_310p")

    contract = validated.deployment.audio_contract
    assert contract is not None
    assert contract.sample_rate_hz == 24000
    assert contract.channels == 1
    assert contract.sample_dtype == "float32"
    assert contract.execution_mode == "offline"
    assert contract.frame_size is None, "synthesis must not declare microphone frame constraints"
    assert contract.chunk_size is None, "synthesis must not declare microphone chunk constraints"


@pytest.mark.skipif(
    not (_ZIPVOICE_BUNDLE / "inference_manifest.json").is_file(),
    reason="local zipvoice bundle is not present",
)
def test_local_zipvoice_bundle_declares_synthesis_only_contracts():
    for name in ("ascend_310p", "ubuntu_onnx"):
        validated = load_inference_manifest(_ZIPVOICE_BUNDLE, name)
        contract = validated.deployment.audio_contract
        assert contract is not None, f"{name} must declare its synthesis audio contract"
        assert contract.sample_rate_hz == 24000
        assert contract.channels == 1
        assert contract.sample_dtype == "float32"
        assert contract.frame_size is None
        assert contract.chunk_size is None
