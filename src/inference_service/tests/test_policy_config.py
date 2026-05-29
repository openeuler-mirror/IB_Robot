from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_policy_config_module():
    path = Path(__file__).parents[1] / "inference_service" / "core" / "_policy_config.py"
    spec = importlib.util.spec_from_file_location("ibrobot_policy_config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_lerobot_config(monkeypatch):
    calls = {}
    lerobot_mod = types.ModuleType("lerobot")
    configs_mod = types.ModuleType("lerobot.configs")
    policies_mod = types.ModuleType("lerobot.configs.policies")

    class FakePreTrainedConfig:
        @classmethod
        def from_pretrained(cls, path):
            calls["path"] = path
            calls["path_exists_during_load"] = Path(path).exists()
            calls["weights_exists_during_load"] = (Path(path) / "weights.safetensors").exists()
            calls["config"] = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
            return calls["config"]

    policies_mod.PreTrainedConfig = FakePreTrainedConfig
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_mod)
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs_mod)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", policies_mod)
    return calls


def _install_fake_lerobot_factory(monkeypatch, calls):
    lerobot_mod = types.ModuleType("lerobot")
    policies_pkg = types.ModuleType("lerobot.policies")
    factory_mod = types.ModuleType("lerobot.policies.factory")

    class FakePolicy:
        def __init__(self):
            self.config = types.SimpleNamespace(chunk_size=1)

        @classmethod
        def from_pretrained(cls, path):
            calls["path"] = path
            calls["path_exists_during_load"] = Path(path).exists()
            calls["weights_exists_during_load"] = (Path(path) / "model.safetensors").exists()
            calls["config"] = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
            if calls["config"].get("device") == "cuda":
                raise RuntimeError("CUDA requested but not available")
            return cls()

        def to(self, device):
            calls["to_device"] = str(device)
            return self

        def eval(self):
            calls["eval"] = True
            return self

    factory_mod.get_policy_class = lambda policy_type: FakePolicy
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_mod)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_mod)


def test_pretrained_config_uses_runtime_device_without_touching_source(monkeypatch, tmp_path):
    policy_config = _load_policy_config_module()
    calls = _install_fake_lerobot_config(monkeypatch)
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "device": "cuda",
                "is_rknn_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    loaded = policy_config.load_pretrained_policy_config(str(tmp_path), runtime_device="cpu")

    assert loaded["device"] == "cpu"
    assert "is_rknn_enabled" not in loaded
    runtime_path = Path(calls["path"])
    assert runtime_path != tmp_path
    assert calls["path_exists_during_load"] is True
    assert calls["weights_exists_during_load"] is True
    assert not runtime_path.exists()
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["device"] == "cuda"


def test_pretrained_config_inserts_runtime_device_when_missing(monkeypatch, tmp_path):
    policy_config = _load_policy_config_module()
    calls = _install_fake_lerobot_config(monkeypatch)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
            }
        ),
        encoding="utf-8",
    )

    loaded = policy_config.load_pretrained_policy_config(str(tmp_path), runtime_device="cpu")

    assert loaded["device"] == "cpu"
    runtime_path = Path(calls["path"])
    assert runtime_path != tmp_path
    assert calls["path_exists_during_load"] is True
    assert not runtime_path.exists()
    assert "device" not in json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))


def test_materialize_runtime_policy_path_cleans_tempdir_on_copy_error(monkeypatch, tmp_path):
    policy_config = _load_policy_config_module()
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "weights.safetensors").write_bytes(b"weights")
    (policy_dir / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "ibrobot_policy_failure"

    def fake_mkdtemp(prefix):
        assert prefix == "ibrobot_policy_"
        runtime_dir.mkdir()
        return str(runtime_dir)

    def fail_link_or_copy(_src, _dst):
        raise OSError("copy failed")

    monkeypatch.setattr(policy_config.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(policy_config, "_link_or_copy", fail_link_or_copy)

    with pytest.raises(OSError, match="copy failed"):
        policy_config.materialize_runtime_policy_path(str(policy_dir), runtime_device="cpu")

    assert not runtime_dir.exists()


def test_read_local_policy_config_device_reports_source_metadata(tmp_path):
    policy_config = _load_policy_config_module()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )

    assert policy_config.read_local_policy_config_device(str(tmp_path)) == "cuda"
    assert policy_config.read_local_policy_config_device(str(tmp_path / "missing")) is None


def test_lerobot_policy_wrapper_from_pretrained_uses_runtime_device(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from inference_service.core.pure_inference_engine import LeRobotPolicyWrapper

    calls = {}
    _install_fake_lerobot_factory(monkeypatch, calls)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "device": "cuda",
                "chunk_size": 1,
            }
        ),
        encoding="utf-8",
    )

    wrapper = LeRobotPolicyWrapper()
    wrapper.load(str(tmp_path), torch.device("cpu"))

    assert calls["config"]["device"] == "cpu"
    assert calls["path_exists_during_load"] is True
    assert calls["weights_exists_during_load"] is True
    assert not Path(calls["path"]).exists()
    assert calls["to_device"] == "cpu"
    assert calls["eval"] is True
    assert wrapper.policy_type == "act"
