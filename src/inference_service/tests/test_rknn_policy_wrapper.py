from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch

from inference_service.core.rknn.policy_wrapper import RKNNPolicyWrapper


def test_rknn_wrapper_requires_config_json(tmp_path):
    (tmp_path / "model.rknn").write_bytes(b"rknn")
    wrapper = RKNNPolicyWrapper()

    with pytest.raises(FileNotFoundError, match="config.json"):
        wrapper.load(str(tmp_path), torch.device("cpu"))


def test_rknn_wrapper_preserves_input_feature_order(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "chunk_size": 100,
                "input_features": {
                    "observation.images.wrist": {"shape": [3, 4, 5]},
                    "observation.state": {"shape": [6]},
                    "observation.images.top": {"shape": [3, 4, 5]},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.rknn").write_bytes(b"rknn")

    captured: dict[str, object] = {}

    class FakeRKNNLite:
        def load_rknn(self, path):
            captured["model_path"] = path
            return 0

        def init_runtime(self, target=None):
            captured["target"] = target
            return 0

        def inference(self, inputs):
            captured["inputs"] = inputs
            return [np.zeros((1, 2, 6), dtype=np.float32)]

        def release(self):
            captured["released"] = True

    fake_api = types.ModuleType("rknnlite.api")
    fake_api.RKNNLite = FakeRKNNLite
    fake_pkg = types.ModuleType("rknnlite")
    fake_pkg.api = fake_api
    monkeypatch.setitem(sys.modules, "rknnlite", fake_pkg)
    monkeypatch.setitem(sys.modules, "rknnlite.api", fake_api)

    wrapper = RKNNPolicyWrapper()
    wrapper.load(str(tmp_path), torch.device("cpu"))

    batch = {
        "observation.images.wrist": torch.full((3, 4, 5), 1.0),
        "observation.state": torch.full((6,), 2.0),
        "observation.images.top": torch.full((3, 4, 5), 3.0),
    }

    wrapper.infer(batch)

    inputs = captured["inputs"]
    assert isinstance(inputs, list)
    assert len(inputs) == 3
    assert inputs[0].shape == (1, 3, 4, 5)
    assert inputs[1].shape == (1, 6)
    assert inputs[2].shape == (1, 3, 4, 5)
    assert float(inputs[0][0, 0, 0, 0]) == 1.0
    assert float(inputs[1][0, 0]) == 2.0
    assert float(inputs[2][0, 0, 0, 0]) == 3.0
