from types import SimpleNamespace

import pytest

from torch_models.pi05_ascend_310p import provider


@pytest.fixture
def local_bundle(tmp_path):
    (tmp_path / "model.safetensors").touch()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    return {
        "config": SimpleNamespace(num_inference_steps=10, dtype="float32"),
        "bundle_root": tmp_path,
        "tokenizer_path": str(tokenizer),
        "device_name": "Ascend310P1",
    }


def test_provider_accepts_validated_environment(monkeypatch, local_bundle):
    monkeypatch.setattr(provider, "package_version", lambda _name: "5.3.0")
    selected = provider.create_provider()
    selected.validate(**local_bundle)
    configured = selected.configure_config(local_bundle["config"], model_dtype="fp16")

    assert configured.dtype == "float16"
    assert selected.load_options == {"skip_weight_init": True}


@pytest.mark.parametrize("version", ["5.4.0", "5.5.4"])
def test_provider_rejects_accuracy_incompatible_transformers(monkeypatch, local_bundle, version):
    monkeypatch.setattr(provider, "package_version", lambda _name: version)
    with pytest.raises(ValueError, match="requires Transformers 5.3.0"):
        provider.create_provider().validate(**local_bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("device_name", "Ascend910B", "requires Ascend310P"), ("tokenizer_path", None, "bundled tokenizer")],
)
def test_provider_rejects_unsupported_environment(local_bundle, field, value, message):
    local_bundle[field] = value
    with pytest.raises(ValueError, match=message):
        provider.create_provider().validate(**local_bundle)


def test_provider_rejects_absent_weights(local_bundle):
    (local_bundle["bundle_root"] / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="bundled model.safetensors"):
        provider.create_provider().validate(**local_bundle)


def test_provider_passes_preparation_context_to_owned_model():
    calls = []
    policy = SimpleNamespace(prepare_for_inference=lambda **kwargs: calls.append(kwargs))
    options = {
        "deployment_fingerprint": "fixture",
        "torch_module": object(),
        "torch_npu_module": object(),
        "device_name": "Ascend310P1",
    }
    provider.create_provider().prepare(policy=policy, **options)
    assert calls == [options]


def test_provider_owns_optional_stage_metadata():
    records = []
    policy = SimpleNamespace(model=SimpleNamespace(get_action_fused_stage_timing_records=lambda: records))
    selected = provider.create_provider()
    assert selected.execution_metadata(policy) == {}
    records.append({"prefix_ms": 143.0, "denoise_total_ms": 80.0})
    assert selected.execution_metadata(policy) == {"pi05_stage_timing": records[-1]}
