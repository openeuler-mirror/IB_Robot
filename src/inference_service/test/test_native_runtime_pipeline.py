from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.pipeline import SequentialModelExecutor
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    OutcomeEvidence,
    RuntimeAssembly,
    RuntimeLatency,
)


class _Runtime:
    def __init__(self, *, delay: float = 0.0, fail: Exception | None = None) -> None:
        self.delay = delay
        self.fail = fail
        self.loaded = 0
        self.closed = 0
        self.calls = 0

    def load(self, _context: object) -> None:
        self.loaded += 1

    def execute(self, request: ModelRequest, context: ExecutionContext):
        context.check("backend")
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        return {"output": np.asarray(request.inputs["input"]) + 1}

    def close(self) -> None:
        self.closed += 1


def _handle(runtime: _Runtime, *, contract: str = "request-direct") -> ModelRuntimeHandle:
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=runtime,
            session=runtime,
            execution_contract=ExecutionContract(
                execution_structure="iterative" if contract.endswith("iterative") else "direct",
                orchestration_visibility="executor" if contract.endswith("iterative") else None,
            ),
        )
    )
    handle.load()
    return handle


def test_native_runtime_owns_load_execute_and_close_once() -> None:
    runtime = _Runtime()
    handle = _handle(runtime)
    result = handle.execute(ModelRequest({"input": np.float32(2)}), ExecutionContext("request-1"))

    assert isinstance(result, ModelResult)
    assert result.successful
    assert result.outputs["output"] == 3
    handle.close()
    handle.close()
    assert runtime.loaded == 1
    assert runtime.closed == 1


def test_native_runtime_deadline_rejects_before_backend() -> None:
    runtime = _Runtime()
    handle = _handle(runtime)
    with pytest.raises(ExecutionFailure) as error:
        handle.execute(
            ModelRequest({"input": np.float32(1)}),
            ExecutionContext.create("expired", timeout=0),
        )

    assert error.value.code == "deadline_exceeded"
    assert runtime.calls == 0
    handle.close()


def test_native_runtime_started_failure_reports_recovery_evidence() -> None:
    runtime = _Runtime(fail=RuntimeError("backend failed"))
    handle = _handle(runtime)
    with pytest.raises(ExecutionFailure) as error:
        handle.execute(ModelRequest({"input": np.float32(1)}), ExecutionContext("failed"))

    assert error.value.evidence.state_mutated is False
    assert error.value.evidence.outcome_known is True
    handle.close()


def test_native_runtime_control_does_not_reopen_closed_runtime() -> None:
    runtime = _Runtime()
    handle = _handle(runtime)
    handle.close()
    with pytest.raises(ExecutionFailure, match="closed"):
        handle.execute(ModelRequest({"input": np.float32(1)}), ExecutionContext("closed"))


def test_sequential_executor_is_an_orchestrator_without_resource_ownership() -> None:
    runtime = _Runtime()

    class Stage:
        def execute(self, frame, *, deadline):
            del deadline
            frame.control.raise_if_canceled("stage")
            frame.values["output"] = np.asarray(frame.values["input"]) + 1

    class Adapter:
        def adapt(self, frame):
            return ModelResult(
                outputs={"output": frame.values["output"]},
                latency=RuntimeLatency(0.1, 0.1),
                evidence=OutcomeEvidence.completed("adaptation"),
            )

    executor = SequentialModelExecutor((Stage(),), Adapter(), components=(runtime,))
    executor.load(SimpleNamespace())
    result = executor.execute(ModelRequest({"input": np.float32(3)}), ExecutionContext("stage"))

    assert result.outputs["output"] == 4
    assert runtime.loaded == 0
    executor.close()
    assert runtime.closed == 0


def test_policy_and_tensor_model_share_the_same_native_handle_contract() -> None:
    policy_runtime = _Runtime()
    tensor_runtime = _Runtime()

    policy = _handle(policy_runtime)
    tensor = _handle(tensor_runtime)

    policy_result = policy.execute(ModelRequest({"input": np.float32(1)}), ExecutionContext("policy"))
    tensor_result = tensor.execute(ModelRequest({"input": np.float32(2)}), ExecutionContext("tensor"))

    assert isinstance(policy_result, ModelResult)
    assert isinstance(tensor_result, ModelResult)
    assert policy_result.outputs["output"] == 2
    assert tensor_result.outputs["output"] == 3
    assert policy.assembly.runtime_executor is not None
    assert tensor.assembly.runtime_executor is not None
    policy.close()
    tensor.close()
