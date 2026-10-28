from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from inference_manifest import TorchRuntimeProfile
from inference_service.distributed import StructuredError
from inference_service.pipeline.factory import register_policy_session_builders
from inference_service.pipeline.runtime import _finalize_policy_assembly
from inference_service.unified_runtime import (
    ExecutionContext,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    ModelRuntimeKey,
    OutcomeState,
    RuntimeAssemblerRegistry,
    RuntimeAssembly,
    RuntimeProviders,
    SessionBuilderRegistry,
)


def _providers() -> RuntimeProviders:
    return RuntimeProviders.create(SimpleNamespace(), SimpleNamespace())


def _context(model_type: str, operation: str = "predict") -> SimpleNamespace:
    identity = SimpleNamespace(
        interface="policy" if model_type in {"pi05", "smolvla", "diffusion"} else "tensor_model",
        model_type=model_type,
        operation=operation,
    )
    return SimpleNamespace(
        identity=identity,
        model_type=model_type,
        backend="torch",
        backend_profile=TorchRuntimeProfile(device="cpu"),
        target_runtime="torch",
        runtime_abi=None,
        deployment=SimpleNamespace(
            execution_contract=SimpleNamespace(
                state_scope="request",
                execution_structure="iterative",
                orchestration_visibility="executor",
            )
        ),
        deployment_name="cpu",
    )


class _IterativeExecutor:
    stages = (SimpleNamespace(plan=(), state_adapter=object()),)
    components = ()

    def __init__(self) -> None:
        self.loaded = False
        self.closed = False
        self.calls: list[str] = []
        self.components = (self,)
        self.component_contexts = {}

    def load(self, _context: object) -> None:
        self.loaded = True

    def execute(self, request, context):
        context.check("test")
        self.calls.append(context.request_id)
        return {"value": request.inputs["value"]}

    def health(self):
        return SimpleNamespace(ready=True)

    def close(self) -> None:
        self.closed = True


def test_policy_iterative_path_uses_factory_handle_and_model_result() -> None:
    executor = _IterativeExecutor()
    assembly = RuntimeAssembly(
        runtime_executor=executor,
        session=executor,
        execution_contract="request-iterative",
    )
    _finalize_policy_assembly(assembly, _context("pi05"), executor, resettable=False)
    handle = ModelRuntimeHandle(assembly)

    assert isinstance(handle, ModelRuntimeHandle)
    assert handle.assembly.execution_contract.name == "request-iterative"
    handle.load()
    result = handle.execute(ModelRequest({"value": 7}), ExecutionContext("pi05-request"))

    assert isinstance(result, ModelResult)
    assert result.outputs["value"] == 7
    assert result.evidence.state is OutcomeState.COMPLETED
    handle.close()
    assert executor.closed


def test_torch_vla_policy_assemblers_use_direct_request_contract() -> None:
    session_registry = SessionBuilderRegistry()
    assembler_registry = RuntimeAssemblerRegistry()
    register_policy_session_builders(session_registry, assembler_registry)

    for model_type in ("pi05", "smolvla"):
        key = ModelRuntimeKey("policy", model_type, "predict", "torch", "request-direct", None)
        assert assembler_registry.get(key) is not None


def test_native_session_runtime_owns_one_session_lifecycle() -> None:
    class Session:
        def __init__(self) -> None:
            self.load_count = 0
            self.close_count = 0

        def load(self, _context: object) -> None:
            self.load_count += 1

        def execute(self, request, _context):
            return {"audio": np.asarray(request.inputs["audio"])}

        def close(self) -> None:
            self.close_count += 1

    session = Session()
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=session,
            session=session,
            execution_contract="request-iterative",
            runtime_id="zipvoice-test",
        )
    )
    handle.load()
    result = handle.execute(ModelRequest({"audio": np.ones(2, dtype=np.float32)}), ExecutionContext("tts-1"))

    assert isinstance(result, ModelResult)
    np.testing.assert_array_equal(result.outputs["audio"], np.ones(2, dtype=np.float32))
    handle.close()
    assert session.load_count == 1
    assert session.close_count == 1


def test_structured_error_remains_the_policy_distributed_wire_value() -> None:
    error = StructuredError(code="cancel_failed", message="cancel failed", stage="cancel")

    assert error.code == "cancel_failed"
    assert error.stage == "cancel"
