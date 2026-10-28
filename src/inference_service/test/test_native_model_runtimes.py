from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import TorchRuntimeProfile
from inference_service.backends import BackendState, RuntimeContext
from inference_service.model_sessions import TorchModelSession
from inference_service.pipeline import SequentialModelExecutor
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    LoadRollback,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    OutcomeEvidence,
    RuntimeAssembly,
    RuntimeLatency,
)


class _FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeTorch:
    __version__ = "fake-torch-1.0"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            return None

    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def as_tensor(value) -> _FakeTensor:
        return _FakeTensor(value)

    @staticmethod
    def is_tensor(value) -> bool:
        return isinstance(value, _FakeTensor)

    @staticmethod
    def inference_mode():
        from contextlib import nullcontext

        return nullcontext()


class _FakeModule:
    def __init__(self) -> None:
        self.calls = 0

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, inputs):
        self.calls += 1
        return {"scores": _FakeTensor(inputs["features"].value + 1.0)}


def _context(tmp_path) -> RuntimeContext:
    manifest = SimpleNamespace(
        model=SimpleNamespace(
            interface="tensor_model",
            model_type="fake_torch",
            operation="infer",
            inputs=(),
            outputs=(SimpleNamespace(semantic="scores", dtype="float32", shape=(1, 2)),),
        )
    )
    validated = SimpleNamespace(
        manifest=manifest,
        deployment=SimpleNamespace(backend="torch"),
        deployment_name="torch_cpu",
        runtime_profile=TorchRuntimeProfile(device="cpu"),
        bundle_root=tmp_path,
        deployment_fingerprint="sha256:test",
        top_level_identity=SimpleNamespace(interface="tensor_model", model_type="fake_torch", operation="infer"),
    )
    return RuntimeContext(validated, runtime_profile=validated.runtime_profile)


def test_torch_model_resource_uses_native_request_and_context(tmp_path) -> None:
    module = _FakeModule()
    session = TorchModelSession(lambda _context: module, torch_loader=_FakeTorch)
    context = _context(tmp_path)
    session.load(context)

    outputs = session.execute(
        ModelRequest({"features": np.ones((1, 2), dtype=np.float32)}),
        ExecutionContext("torch-1"),
    )

    np.testing.assert_array_equal(outputs["scores"], np.full((1, 2), 2.0, dtype=np.float32))
    assert module.calls == 1
    assert session.health().state is BackendState.READY
    session.close()


def test_native_session_is_owned_by_runtime_handle(tmp_path) -> None:
    session = TorchModelSession(lambda _context: _FakeModule(), torch_loader=_FakeTorch)
    context = _context(tmp_path)
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=session,
            session=session,
            execution_contract=ExecutionContract(),
            load_context=context,
        )
    )
    handle.load(context)
    result = handle.execute(
        ModelRequest({"features": np.zeros((1, 2), dtype=np.float32)}),
        ExecutionContext("handle-1"),
    )

    assert isinstance(result, ModelResult)
    assert result.successful
    handle.close()
    assert session.health().state is BackendState.CLOSED


def test_native_executor_does_not_own_model_resource() -> None:
    resource = SimpleNamespace(load=lambda _context: None, close=lambda: None)

    class Stage:
        def execute(self, frame, *, deadline):
            del deadline
            frame.values["output"] = frame.values["input"]

    class Adapter:
        def adapt(self, frame):
            return ModelResult(
                outputs={"output": frame.values["output"]},
                latency=RuntimeLatency(0.1, 0.1),
                evidence=OutcomeEvidence.completed("adaptation"),
            )

    executor = SequentialModelExecutor((Stage(),), Adapter(), components=(resource,))
    executor.load(SimpleNamespace())
    executor.close()
    assert resource.load is not None


def test_load_rollback_is_native_and_reverse_order() -> None:
    events: list[str] = []
    rollback = LoadRollback()
    rollback.defer(lambda: events.append("first"))
    rollback.defer(lambda: events.append("second"))
    assert rollback.rollback() == ()
    assert events == ["second", "first"]


def test_native_runtime_rejects_expired_request_before_resource_execution() -> None:
    class Runtime:
        def execute(self, _request, _context):
            raise AssertionError("backend must not execute")

    handle = ModelRuntimeHandle(Runtime(), execution_contract="request-direct")
    handle.load()
    with pytest.raises(ExecutionFailure) as error:
        handle.execute(ModelRequest({"x": 1}), ExecutionContext.create("expired", timeout=0))
    assert error.value.code == "deadline_exceeded"
    handle.close()
