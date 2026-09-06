"""ONNX Runtime model-session conformance tests against a fake ORT SDK."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry, RuntimeContext
from inference_service.backends.errors import BackendLoadError
from inference_service.backends.registry import _validate_onnx
from inference_service.model_sessions import OnnxRuntimeModelSession, build_onnx_model_session
from inference_service.unified_runtime import ExecutionContext, ModelRequest
from tests.manifest_fixtures import TEST_BUNDLE_UUID, TEST_DEPLOYMENT_UUID

_STATIC_BACKEND_REGISTRY = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
_REQUEST_CONTRACT = {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary",
}


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeGraph:
    """One ONNX graph stub: named I/O plus a callable execution body."""

    def __init__(self, inputs: tuple[str, ...], outputs: tuple[str, ...], body) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.body = body
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(dict(feed))
        return list(self.body(feed))


def _fake_ort(graphs: dict[str, _FakeGraph]) -> Any:
    class _InferenceSession:
        def __init__(self, path: str, sess_options=None, providers=None) -> None:
            try:
                self.graph = graphs[path]
            except KeyError as exc:
                raise RuntimeError(f"model not found: {path}") from exc
            self.sess_options = sess_options
            self.providers = providers

        def get_inputs(self):
            return [_FakeNode(name) for name in self.graph.inputs]

        def get_outputs(self):
            return [_FakeNode(name) for name in self.graph.outputs]

        def run(self, _output_names, feed):
            return self.graph.run(feed)

    class _OnnxRuntime:
        __version__ = "fake-ort-1.0"

        class GraphOptimizationLevel:
            ORT_DISABLE_ALL = "disable_all"
            ORT_ENABLE_BASIC = "enable_basic"
            ORT_ENABLE_EXTENDED = "enable_extended"
            ORT_ENABLE_ALL = "enable_all"

        class SessionOptions:
            def __init__(self) -> None:
                self.graph_optimization_level = None

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    _OnnxRuntime.InferenceSession = _InferenceSession
    return _OnnxRuntime


def _write_onnx_bundle(root: Path, manifest: dict[str, Any]) -> None:
    for deployment in manifest["deployments"].values():
        for artifact in deployment["artifacts"].values():
            path = root / artifact["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-onnx-graph")
    asset = root / "assets" / "adapter.json"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(json.dumps({"backend": "onnx"}), encoding="utf-8")
    entries = [BundleFile(path="assets/adapter.json")]
    manifest["bundle"]["files"] = [entry.model_dump(mode="json") for entry in entries]
    manifest["bundle"]["digest"]["value"] = canonical_bundle_digest(
        TEST_BUNDLE_UUID, 1, manifest["bundle"]["name"], entries
    )
    (root / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _base_manifest(name: str, model: dict[str, Any], deployment: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "bundle": {
            "uuid": TEST_BUNDLE_UUID,
            "revision": 1,
            "name": name,
            "files": [],
            "digest": {"algorithm": "sha256", "scope": "structure", "value": "0" * 64},
        },
        "model": model,
        "deployments": {"onnx_cpu": deployment},
    }


def _stateless_deployment() -> dict[str, Any]:
    return {
        "uuid": TEST_DEPLOYMENT_UUID,
        "revision": 1,
        "execution_contract": dict(_REQUEST_CONTRACT),
        "runtime_profile": {
            "backend": "onnx",
            "target": {"runtime": "onnx"},
            "profile": {"device": "cpu"},
        },
        "artifacts": {"model": {"path": "artifacts/model.onnx", "format": "onnx"}},
        "execution": ["model"],
        "bindings": {
            "model": {
                "inputs": [
                    {
                        "semantic": "host.silero.audio",
                        "runtime_name": "input",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [4],
                    }
                ],
                "outputs": [
                    {
                        "semantic": "host.silero.prob",
                        "runtime_name": "prob",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1],
                    }
                ],
            }
        },
    }


def _stateless_manifest() -> dict[str, Any]:
    return _base_manifest(
        "test-onnx-silero",
        {
            "interface": "tensor_model",
            "model_type": "silero_vad",
            "operation": "vad",
            "inputs": [{"semantic": "host.silero.audio", "dtype": "float32", "shape": [4]}],
            "outputs": [{"semantic": "host.silero.prob", "dtype": "float32", "shape": [1]}],
        },
        _stateless_deployment(),
    )


def _stateful_deployment() -> dict[str, Any]:
    return {
        "uuid": TEST_DEPLOYMENT_UUID,
        "revision": 1,
        "execution_contract": {
            "state_scope": "stream",
            "execution_structure": "direct",
            "cancellation_granularity": "request_boundary",
            "stateful": True,
            "state_links": [
                {
                    "role": "model",
                    "state_name": "hidden",
                    "owner": "session",
                    "source": "host.silero.rnn_hidden_in",
                    "target": "host.silero.rnn_hidden_out",
                    "scope": "runtime",
                    "state_bank": "model.bank",
                }
            ],
            "state_bank_mode": "runtime_exclusive",
            "max_open_streams": 1,
        },
        "runtime_profile": {
            "backend": "onnx",
            "target": {"runtime": "onnx"},
            "profile": {"device": "cpu"},
        },
        "artifacts": {"model": {"path": "artifacts/model.onnx", "format": "onnx"}},
        "execution": ["model"],
        "bindings": {
            "model": {
                "inputs": [
                    {
                        "semantic": "host.silero.audio",
                        "runtime_name": "input",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [4],
                    },
                    {
                        "semantic": "host.silero.rnn_hidden_in",
                        "runtime_name": "state",
                        "index": 1,
                        "dtype": "float32",
                        "shape": [2],
                    },
                ],
                "outputs": [
                    {
                        "semantic": "host.silero.prob",
                        "runtime_name": "prob",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1],
                    },
                    {
                        "semantic": "host.silero.rnn_hidden_out",
                        "runtime_name": "state_out",
                        "index": 1,
                        "dtype": "float32",
                        "shape": [2],
                    },
                ],
            }
        },
    }


def _stateful_manifest() -> dict[str, Any]:
    return _base_manifest(
        "test-onnx-silero-stateful",
        {
            "interface": "tensor_model",
            "model_type": "silero_vad",
            "operation": "vad",
            "inputs": [{"semantic": "host.silero.audio", "dtype": "float32", "shape": [4]}],
            "outputs": [{"semantic": "host.silero.prob", "dtype": "float32", "shape": [1]}],
        },
        _stateful_deployment(),
    )


def _multi_role_manifest() -> dict[str, Any]:
    deployment = {
        "uuid": TEST_DEPLOYMENT_UUID,
        "revision": 1,
        "execution_contract": dict(_REQUEST_CONTRACT),
        "runtime_profile": {
            "backend": "onnx",
            "target": {"runtime": "onnx"},
            "profile": {"device": "cpu"},
        },
        "artifacts": {
            "fullsubnet": {"path": "artifacts/fullsubnet.onnx", "format": "onnx"},
            "silero_vad": {"path": "artifacts/silero_vad.onnx", "format": "onnx"},
        },
        "execution": ["fullsubnet", "silero_vad"],
        "bindings": {
            "fullsubnet": {
                "inputs": [
                    {
                        "semantic": "host.fullsubnet.spectrum",
                        "runtime_name": "spectrum",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [4],
                    }
                ],
                "outputs": [
                    {
                        "semantic": "internal.enhanced",
                        "runtime_name": "enhanced",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [4],
                    }
                ],
            },
            "silero_vad": {
                "inputs": [
                    {
                        "semantic": "internal.enhanced",
                        "runtime_name": "audio",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [4],
                    }
                ],
                "outputs": [
                    {
                        "semantic": "host.silero.prob",
                        "runtime_name": "prob",
                        "index": 0,
                        "dtype": "float32",
                        "shape": [1],
                    }
                ],
            },
        },
    }
    return _base_manifest(
        "test-onnx-speech-direction",
        {
            "interface": "tensor_model",
            "model_type": "speech_direction",
            "operation": "enhance_and_vad",
            "inputs": [{"semantic": "host.fullsubnet.spectrum", "dtype": "float32", "shape": [4]}],
            "outputs": [{"semantic": "host.silero.prob", "dtype": "float32", "shape": [1]}],
        },
        deployment,
    )


def _context(tmp_path: Path, manifest: dict[str, Any]) -> RuntimeContext:
    _write_onnx_bundle(tmp_path, manifest)
    return RuntimeContext(load_inference_manifest(str(tmp_path), "onnx_cpu"))


def test_onnx_registry_descriptor_validates_context(tmp_path) -> None:
    context = _context(tmp_path, _stateless_manifest())
    assert _STATIC_BACKEND_REGISTRY.validate(context).name == "onnx"
    assert _validate_onnx(context.deployment) is None


def test_onnx_target_validator_rejects_non_onnx_runtime() -> None:
    wrong_runtime = SimpleNamespace(
        role_runtime_profiles={},
        target=SimpleNamespace(runtime="torch"),
        execution=(),
        artifacts={},
    )
    assert _validate_onnx(wrong_runtime) == "target.runtime must be exactly 'onnx'"
    wrong_format = SimpleNamespace(
        role_runtime_profiles={},
        target=SimpleNamespace(runtime="onnx"),
        execution=("model",),
        artifacts={"model": SimpleNamespace(format="om")},
    )
    assert "format 'onnx'" in _validate_onnx(wrong_format)
    uncompiled = SimpleNamespace(role_runtime_profiles={}, target=None, execution=(), artifacts={})
    assert _validate_onnx(uncompiled) == "onnx requires a compiled deployment"


def test_onnx_session_executes_manifest_bound_graph(tmp_path) -> None:
    graph = _FakeGraph(
        ("input",),
        ("prob",),
        lambda feed: [np.asarray([float(np.mean(feed["input"]))], dtype=np.float32)],
    )
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    context = _context(tmp_path, _stateless_manifest())
    session = OnnxRuntimeModelSession(ort_loader=lambda: ort)
    session.load(context)

    outputs = session.execute(
        ModelRequest({"host.silero.audio": np.full(4, 2.0, dtype=np.float32)}),
        ExecutionContext("onnx-1"),
    )

    np.testing.assert_allclose(outputs["host.silero.prob"], np.asarray([2.0], dtype=np.float32))
    assert session.runtime_version == "fake-ort-1.0"
    assert session.capabilities.stateful is False
    assert graph.feeds[0]["input"].shape == (4,)
    session.close()


def test_onnx_stateful_session_keeps_and_resets_host_state(tmp_path) -> None:
    graph = _FakeGraph(
        ("input", "state"),
        ("prob", "state_out"),
        lambda feed: [
            np.asarray([float(np.sum(feed["input"]))], dtype=np.float32),
            feed["state"] + np.float32(np.sum(feed["input"])),
        ],
    )
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    context = _context(tmp_path, _stateful_manifest())
    session = build_onnx_model_session(context, ort_loader=lambda: ort)
    session.load(context)

    request = ModelRequest({"host.silero.audio": np.ones(4, dtype=np.float32)})
    first = session.execute(request, ExecutionContext("onnx-state-1"))
    second = session.execute(request, ExecutionContext("onnx-state-2"))

    np.testing.assert_allclose(first["host.silero.prob"], np.asarray([4.0], dtype=np.float32))
    np.testing.assert_allclose(second["host.silero.prob"], np.asarray([4.0], dtype=np.float32))
    assert "host.silero.rnn_hidden_out" not in first
    assert "host.silero.rnn_hidden_in" not in first
    np.testing.assert_allclose(graph.feeds[0]["state"], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(graph.feeds[1]["state"], np.full(2, 4.0, dtype=np.float32))

    session.reset(ExecutionContext("onnx-reset"))
    third = session.execute(request, ExecutionContext("onnx-state-3"))
    np.testing.assert_allclose(third["host.silero.prob"], np.asarray([4.0], dtype=np.float32))
    np.testing.assert_allclose(graph.feeds[2]["state"], np.zeros(2, dtype=np.float32))
    session.close()


def test_onnx_session_chains_multi_role_host_orchestration(tmp_path) -> None:
    fullsubnet = _FakeGraph(
        ("spectrum",),
        ("enhanced",),
        lambda feed: [feed["spectrum"] * 2.0],
    )
    silero = _FakeGraph(
        ("audio",),
        ("prob",),
        lambda feed: [np.asarray([float(np.mean(feed["audio"]))], dtype=np.float32)],
    )
    ort = _fake_ort(
        {
            str(tmp_path / "artifacts/fullsubnet.onnx"): fullsubnet,
            str(tmp_path / "artifacts/silero_vad.onnx"): silero,
        }
    )
    context = _context(tmp_path, _multi_role_manifest())
    session = OnnxRuntimeModelSession(ort_loader=lambda: ort)
    session.load(context)

    outputs = session.execute(
        ModelRequest({"host.fullsubnet.spectrum": np.ones(4, dtype=np.float32)}),
        ExecutionContext("onnx-chain-1"),
    )

    np.testing.assert_allclose(outputs["host.silero.prob"], np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(fullsubnet.feeds[0]["spectrum"], np.ones(4, dtype=np.float32))
    np.testing.assert_allclose(silero.feeds[0]["audio"], np.full(4, 2.0, dtype=np.float32))
    session.close()


def test_onnx_session_rejects_non_onnx_target_runtime(tmp_path) -> None:
    graph = _FakeGraph(("input",), ("prob",), lambda feed: [np.zeros(1, dtype=np.float32)])
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    manifest = _stateless_manifest()
    manifest["deployments"]["onnx_cpu"]["runtime_profile"]["target"]["runtime"] = "tensorrt"
    context = _context(tmp_path, manifest)
    session = OnnxRuntimeModelSession(ort_loader=lambda: ort)
    with pytest.raises(BackendLoadError, match="target runtime 'tensorrt' must be 'onnx'"):
        session.load(context)


def test_onnx_session_rejects_unknown_runtime_options(tmp_path) -> None:
    graph = _FakeGraph(("input",), ("prob",), lambda feed: [np.zeros(1, dtype=np.float32)])
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    context = _context(tmp_path, _stateless_manifest())
    context = RuntimeContext(
        context.validated_manifest,
        runtime_options={"device_id": 0},
        runtime_profile=context.runtime_profile,
    )
    session = OnnxRuntimeModelSession(ort_loader=lambda: ort)
    with pytest.raises(BackendLoadError, match="unknown ONNX Runtime model-session options"):
        session.load(context)


def test_onnx_session_rejects_unavailable_provider(tmp_path) -> None:
    manifest = _stateless_manifest()
    manifest["deployments"]["onnx_cpu"]["runtime_profile"]["profile"] = {"device": "cuda"}
    _write_onnx_bundle(tmp_path, manifest)
    validated = load_inference_manifest(str(tmp_path), "onnx_cpu")

    class _NoCudaOrt:
        __version__ = "fake-ort-1.0"

        class GraphOptimizationLevel:
            ORT_DISABLE_ALL = "disable_all"
            ORT_ENABLE_BASIC = "enable_basic"
            ORT_ENABLE_EXTENDED = "enable_extended"
            ORT_ENABLE_ALL = "enable_all"

        class SessionOptions:
            def __init__(self) -> None:
                self.graph_optimization_level = None

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    session = OnnxRuntimeModelSession(ort_loader=lambda: _NoCudaOrt)
    with pytest.raises(BackendLoadError, match="execution provider 'CUDAExecutionProvider' is unavailable"):
        session.load(RuntimeContext(validated))


def test_onnx_session_rejects_unbound_graph_inputs(tmp_path) -> None:
    graph = _FakeGraph(("input", "sr"), ("prob",), lambda feed: [np.zeros(1, dtype=np.float32)])
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    context = _context(tmp_path, _stateless_manifest())
    session = OnnxRuntimeModelSession(ort_loader=lambda: ort)
    with pytest.raises(BackendLoadError, match="unbound inputs"):
        session.load(context)


def test_build_onnx_model_session_selects_state_mode(tmp_path) -> None:
    _write_onnx_bundle(tmp_path, _stateful_manifest())
    validated = load_inference_manifest(str(tmp_path), "onnx_cpu")
    session = build_onnx_model_session(RuntimeContext(validated))
    assert session.capabilities.stateful is True
    assert session.capabilities.resettable is True
