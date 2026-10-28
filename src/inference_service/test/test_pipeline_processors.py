from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from inference_manifest import load_inference_manifest
from inference_service.backends import RuntimeContext
from inference_service.pipeline.processors import create_lerobot_processor_views
from test.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


def test_local_tokenizer_reference_is_resolved_against_bundle(monkeypatch, tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    paths = create_policy_bundle(bundle, "smolvla", local_tokenizer=True, include_weights=False)
    write_manifest(bundle, make_manifest(bundle, paths, deployment_name="rk3588", compiled=True))
    context = RuntimeContext(load_inference_manifest(bundle, "rk3588"))
    captured: dict[str, object] = {}

    class FakeConfig:
        device = "cuda"

        @classmethod
        def from_pretrained(cls, path, local_files_only):
            captured["config_path"] = path
            captured["local_files_only"] = local_files_only
            return cls()

    policies = ModuleType("lerobot.configs.policies")
    policies.PreTrainedConfig = FakeConfig
    factory = ModuleType("lerobot.policies.factory")

    def make_processors(**kwargs):
        captured.update(kwargs)
        return (lambda inputs: inputs), (lambda action: action)

    factory.make_pre_post_processors = make_processors
    monkeypatch.setitem(sys.modules, policies.__name__, policies)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)

    preprocessor, postprocessor = create_lerobot_processor_views()
    preprocessor.load(context)

    assert captured["config_path"] == str(bundle.resolve())
    assert captured["local_files_only"] is True
    assert captured["preprocessor_overrides"] == {
        "device_processor": {"device": "cpu"},
        "tokenizer_processor": {"tokenizer_name": str((bundle / "tokenizer").resolve())},
    }
    assert captured["postprocessor_overrides"] == {"device_processor": {"device": "cpu"}}
    assert preprocessor({"observation.state": 1}) == {"observation.state": 1}
    action = postprocessor(np.ones((1, 2), dtype=np.float32))
    assert isinstance(action, torch.Tensor)
    assert torch.equal(action, torch.ones((1, 2)))


def test_processor_pair_reset_resets_preprocessor_and_postprocessor(monkeypatch, tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    paths = create_policy_bundle(bundle, include_weights=False)
    write_manifest(bundle, make_manifest(bundle, paths, deployment_name="rk3588", compiled=True))
    context = RuntimeContext(load_inference_manifest(bundle, "rk3588"))
    preprocessor = SimpleNamespace(reset_calls=0)
    postprocessor = SimpleNamespace(reset_calls=0)
    preprocessor.reset = lambda: setattr(preprocessor, "reset_calls", preprocessor.reset_calls + 1)
    postprocessor.reset = lambda: setattr(postprocessor, "reset_calls", postprocessor.reset_calls + 1)

    policies = ModuleType("lerobot.configs.policies")
    policies.PreTrainedConfig = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(device="cuda")
    )
    factory = ModuleType("lerobot.policies.factory")
    factory.make_pre_post_processors = lambda **_kwargs: (preprocessor, postprocessor)
    monkeypatch.setitem(sys.modules, policies.__name__, policies)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)

    preprocessor_view, _ = create_lerobot_processor_views()
    preprocessor_view.load(context)
    preprocessor_view.reset()

    assert preprocessor.reset_calls == 1
    assert postprocessor.reset_calls == 1


def test_processor_pair_reset_attempts_both_processors_before_raising(monkeypatch, tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    paths = create_policy_bundle(bundle, include_weights=False)
    write_manifest(bundle, make_manifest(bundle, paths, deployment_name="rk3588", compiled=True))
    context = RuntimeContext(load_inference_manifest(bundle, "rk3588"))
    postprocessor = SimpleNamespace(reset_calls=0)

    def fail_reset():
        raise RuntimeError("preprocessor reset failed")

    def reset_postprocessor():
        postprocessor.reset_calls += 1

    preprocessor = SimpleNamespace(reset=fail_reset)
    postprocessor.reset = reset_postprocessor
    policies = ModuleType("lerobot.configs.policies")
    policies.PreTrainedConfig = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(device="cuda")
    )
    factory = ModuleType("lerobot.policies.factory")
    factory.make_pre_post_processors = lambda **_kwargs: (preprocessor, postprocessor)
    monkeypatch.setitem(sys.modules, policies.__name__, policies)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)

    preprocessor_view, _ = create_lerobot_processor_views()
    preprocessor_view.load(context)

    with pytest.raises(RuntimeError, match="preprocessor reset failed"):
        preprocessor_view.reset()

    assert postprocessor.reset_calls == 1
