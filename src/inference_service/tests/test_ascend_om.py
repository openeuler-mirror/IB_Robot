#!/usr/bin/env python3
"""Tests for Ascend OM inference_service integration."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference_service.core import (  # noqa: E402
    AscendOM3403PolicyWrapper,
    AscendOMPolicyWrapper,
    PureInferenceEngine,
    create_ascend_om_policy_wrapper,
    resolve_device,
)


class FakeRuntimeSession:
    def __init__(self, output):
        self.output = output
        self.loaded = None
        self.inputs = None

    def load(self, policy_path, config, device):
        self.loaded = (policy_path, config, device)

    def execute(self, inputs):
        self.inputs = inputs
        return [self.output]

    def release(self):
        pass


def _write_act_config(tmp_path, extra=None):
    config = {
        "type": "act",
        "chunk_size": 2,
        "input_features": {"observation.state": {"shape": [3]}},
        "output_features": {"action": {"shape": [6]}},
    }
    if extra:
        config.update(extra)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_pi05_config(tmp_path, extra=None):
    config = {
        "type": "pi05",
        "chunk_size": 50,
        "max_action_dim": 32,
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.language.tokens": {"shape": [48]},
            "observation.language.attention_mask": {"shape": [48]},
        },
        "output_features": {"action": {"shape": [6]}},
    }
    if extra:
        config.update(extra)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_manifest(tmp_path, manifest):
    (tmp_path / "config.om.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_resolve_device_accepts_ascend_om_aliases():
    assert resolve_device("ascend_om").type == "cpu"
    assert resolve_device("ascend_om_3403").type == "cpu"
    assert resolve_device("ascend-om").type == "cpu"


def test_create_ascend_wrapper_by_device_name():
    assert isinstance(create_ascend_om_policy_wrapper("ascend_om"), AscendOMPolicyWrapper)
    assert isinstance(create_ascend_om_policy_wrapper("ascend-om-3403"), AscendOM3403PolicyWrapper)


def test_pure_engine_selects_ascend_wrapper(monkeypatch, tmp_path):
    model_path = tmp_path / "model.om"
    model_path.write_bytes(b"om")
    _write_act_config(tmp_path)
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om",
            "artifacts": {"policy": "model.om"},
            "execution": ["policy"],
        },
    )
    runtime = FakeRuntimeSession(np.arange(12, dtype=np.float32))
    monkeypatch.setattr(
        "inference_service.core.compiled_policy.create_runtime_session",
        lambda backend, config=None: runtime,
    )

    engine = PureInferenceEngine(policy_path=str(tmp_path), device="ascend_om")
    result = engine({"observation.state": torch.ones(1, 3)})

    assert result.policy_type == "act"
    assert result.backend_type == "ascend_om"
    assert result.action.shape == (2, 6)
    assert runtime.inputs[0].shape == (1, 3)


def test_pure_engine_selects_compiled_pi05_wrapper(monkeypatch, tmp_path):
    vlm = tmp_path / "vlm.om"
    action_expert = tmp_path / "action_expert.om"
    vlm.write_bytes(b"vlm")
    action_expert.write_bytes(b"ae")
    _write_pi05_config(tmp_path)
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "pi05",
            "backend": "ascend_om",
            "artifacts": {"vlm": "vlm.om", "action_expert": "action_expert.om"},
            "execution": ["vlm", "action_expert"],
        },
    )
    runtime = FakeRuntimeSession(torch.zeros(1, 50, 32))
    monkeypatch.setattr(
        "inference_service.core.compiled_policy.create_runtime_session",
        lambda backend, config=None: runtime,
    )

    engine = PureInferenceEngine(policy_path=str(tmp_path), device="ascend_om")
    result = engine(
        {
            "observation.images.front": torch.ones(1, 3, 224, 224),
            "observation.language.tokens": torch.ones(1, 48, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(1, 48, dtype=torch.bool),
        }
    )

    assert result.policy_type == "pi05"
    assert result.backend_type == "ascend_om"
    assert result.action.shape == (50, 6)
    assert runtime.inputs.images[0].shape == (1, 3, 224, 224)


def test_pure_engine_selects_ascend_3403_wrapper(monkeypatch, tmp_path):
    model_path = tmp_path / "model.om"
    model_path.write_bytes(b"om")
    worker_path = tmp_path / "main"
    worker_path.write_text("#!/bin/sh\n", encoding="utf-8")
    worker_path.chmod(0o755)
    _write_act_config(tmp_path)
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifacts": {"policy": "model.om", "worker": "main"},
            "execution": ["policy", "worker"],
        },
    )
    runtime = FakeRuntimeSession(np.arange(16, dtype=np.float32))
    monkeypatch.setattr(
        "inference_service.core.compiled_policy.create_runtime_session",
        lambda backend, config=None: runtime,
    )

    engine = PureInferenceEngine(policy_path=str(tmp_path), device="ascend_om_3403")
    result = engine({"observation.state": torch.ones(1, 3)})

    assert result.policy_type == "act"
    assert result.backend_type == "ascend_om_3403"
    assert result.action.shape == (2, 6)
    assert result.chunk_size == 2
