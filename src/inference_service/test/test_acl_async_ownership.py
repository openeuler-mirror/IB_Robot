import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from inference_manifest import ArtifactBindings, TensorBinding
from inference_service.backends.ascend.model import AclModel
from inference_service.backends.errors import BackendInferenceError, BackendLifecycleError


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("invalid", [None, "input_shape", "output_shape", "output_dtype", "missing_output"])
def test_role_execution_uses_shared_validation_and_capture(isolated, invalid):
    from inference_manifest import CompiledDeployment
    from inference_service.backends.types import BackendState
    from inference_service.model_sessions.ascend import AscendOmModelSession
    from inference_service.unified_runtime import ExecutionContext, ModelRequest

    deployment = CompiledDeployment.model_validate(
        {
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {"backend": "ascend", "target": {"runtime": "acl"}, "profile": {"device_id": 0}},
            "execution": ["encoder"],
            "artifacts": {"encoder": {"path": "encoder.om", "format": "om"}},
            "bindings": {"encoder": _bindings()},
        }
    )
    captured = []
    inputs = {"input": np.ones((1, 1) if invalid == "input_shape" else (1,), dtype=np.float32)}
    output = np.ones(
        (1, 1) if invalid == "output_shape" else (1,), dtype=np.float64 if invalid == "output_dtype" else np.float32
    )
    session = AscendOmModelSession(
        runtime_manager=object(), diagnostic_capture=lambda name, value: captured.append(name)
    )
    session._context = SimpleNamespace(deployment=deployment)
    session._state = BackendState.READY
    session._models = {
        "encoder": SimpleNamespace(
            execute=lambda *args, **kwargs: {} if invalid == "missing_output" else {0: output}, close=lambda: None
        )
    }
    if isolated:
        # Exercise the actual isolated submission branch, including its Future.
        def submit(*args, **kwargs):
            future = Future()
            future.set_result({} if invalid == "missing_output" else {0: output})
            return SimpleNamespace(completion=future)

        session._models["encoder"].submit_async_isolated = submit
        session._request_stream = lambda *args: "stream"
    try:
        if invalid:
            with pytest.raises(BackendInferenceError, match="shape|dtype|semantics"):
                session.execute_role(
                    "encoder", inputs, ModelRequest(inputs), ExecutionContext("role"), isolated=isolated
                )
        else:
            result = session.execute_role(
                "encoder", inputs, ModelRequest(inputs), ExecutionContext("role"), isolated=isolated
            )
            np.testing.assert_array_equal(result["output"], output)
            assert captured == ["encoder_in_input", "encoder_out_output"]
    finally:
        session.close()


def test_iterative_role_reuses_key_before_future_callbacks_finish(monkeypatch):
    from inference_service.model_sessions.ascend import AscendOmModelSession
    from inference_service.scheduler.operations import OperationIdentity, OperationKind, OperationRegistry
    from inference_service.unified_runtime import ExecutionContext

    registry = OperationRegistry(max_records=4, max_waiters_per_operation=1)
    identity = OperationIdentity(str(uuid4()), 1, str(uuid4()), 1, str(uuid4()))
    context = ExecutionContext("iterative-request")
    release_callbacks = threading.Event()
    callback_started = threading.Event()
    caller_thread = threading.current_thread()
    finish = registry.finish
    threads = []
    operation_ids = []

    def delayed_finish(*args, **kwargs):
        if threading.current_thread() is not caller_thread:
            callback_started.set()
            assert release_callbacks.wait(5)
        return finish(*args, **kwargs)

    monkeypatch.setattr(registry, "finish", delayed_finish)

    def operation_factory(role):
        operation, created = registry.create_or_get(
            kind=OperationKind.DISPATCH,
            idempotency_key=(1, context.request_id, role),
            identity=identity,
            deadline_mono_ns=10**30,
            waiter_id=context.request_id,
        )
        assert created
        operation_ids.append(operation.operation_id)
        return operation, registry

    class Completion(Future):
        def add_done_callback(self, fn):
            super().add_done_callback(fn)
            worker = threading.Thread(target=self.set_result, args=({0: "actions"},))
            threads.append(worker)
            worker.start()

    session = AscendOmModelSession(runtime_manager=object())
    session._models = {
        "action_expert": SimpleNamespace(
            submit_async_isolated=lambda *args, **kwargs: SimpleNamespace(completion=Completion())
        )
    }
    try:
        for _ in range(2):
            callback_started.clear()
            result = session._submit_isolated_role(
                "action_expert", {}, context, read_outputs=True, stream="stream", operation_factory=operation_factory
            )
            assert result == {0: "actions"}
            assert callback_started.wait(5)
            assert registry.find((1, context.request_id, "action_expert")) is None
        assert len(set(operation_ids)) == 2
    finally:
        release_callbacks.set()
        for worker in threads:
            worker.join(5)
            assert not worker.is_alive()
    assert all(registry.get(operation_id) is None for operation_id in operation_ids)


def _bindings():
    return ArtifactBindings(
        inputs=(TensorBinding(semantic="input", index=0, dtype="float32", shape=(1,)),),
        outputs=(TensorBinding(semantic="output", index=0, dtype="float32", shape=(1,)),),
    )


@pytest.fixture
def model_probe(monkeypatch):
    state = SimpleNamespace(submit=0, sync=0, destroyed=[], submitted=0)

    def submit(*_args):
        state.submitted += 1
        return state.submit

    acl = SimpleNamespace(
        mdl=SimpleNamespace(execute_async=submit),
        rt=SimpleNamespace(synchronize_stream=lambda _stream: state.sync),
    )
    lease = SimpleNamespace(acl=acl, bind_current_thread=lambda: None)
    model = AclModel(lease, "encoder", Path("fake.om"), _bindings(), async_execution=True)
    created = []

    def create(*_args):
        dataset = object()
        created.append(dataset)
        return dataset, []

    monkeypatch.setattr(model, "_create_dataset", create)
    monkeypatch.setattr(
        model, "_destroy_dataset", lambda dataset, buffers: state.destroyed.append(dataset) if dataset else None
    )
    monkeypatch.setattr(model, "_prepare_async_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "_read_dataset_outputs", lambda *args: {0: "result"})
    yield model, state, created
    state.sync = 0
    model.close()


@pytest.mark.parametrize("failure", ["submit", "sync"])
def test_uncertain_execution_retains_datasets_until_trusted_drain(model_probe, failure):
    model, state, created = model_probe
    setattr(state, failure, 1)
    with pytest.raises(BackendInferenceError) as error:
        model.submit_async_isolated({}, stream="stream").completion.result(timeout=2)
    assert error.value.operation_started and not error.value.outcome_known
    assert state.destroyed == []
    assert len(created) == 2
    with pytest.raises(BackendInferenceError, match="quarantined"):
        model.submit_async_isolated({}, stream="stream")
    assert state.submitted == 1
    state.sync = 1
    with pytest.raises(RuntimeError, match="synchronize_stream"):
        model.close()
    assert state.destroyed == []
    assert not model._closed
    state.sync = 0
    model.close()
    assert set(state.destroyed) == set(created)
    assert model._closed


def test_output_read_failure_is_known_and_releases_datasets(model_probe, monkeypatch):
    model, state, created = model_probe

    def fail(*args):
        raise ValueError("invalid output")

    monkeypatch.setattr(model, "_read_dataset_outputs", fail)
    with pytest.raises(BackendInferenceError) as error:
        model.submit_async_isolated({}, stream="stream").completion.result(timeout=2)
    assert error.value.operation_started and error.value.outcome_known
    assert set(state.destroyed) == set(created)


def test_legacy_model_has_no_async_worker_or_lock():
    model = AclModel(SimpleNamespace(acl=object()), "legacy", Path("fake.om"), _bindings())
    assert model._execution_lock is None
    assert model._completion_executor is None
    with pytest.raises(BackendInferenceError, match="disabled"):
        model.submit_async_isolated({}, stream="stream")


@pytest.mark.parametrize("failure", ["submit", "sync"])
def test_shared_execution_quarantines_before_reuse_and_drains_before_free(model_probe, monkeypatch, failure):
    model, state, _ = model_probe
    model.model_id = 1
    model.input_dataset, model.output_dataset = object(), object()
    datasets = {model.input_dataset, model.output_dataset}
    unloaded = []
    monkeypatch.setattr(model._acl.mdl, "unload", lambda _model: unloaded.append(True), raising=False)
    setattr(state, failure, 1)
    with pytest.raises(BackendInferenceError) as error:
        model.execute({}, stream="stream")
    assert error.value.operation_started and not error.value.outcome_known
    with pytest.raises(BackendInferenceError, match="quarantined"):
        model.execute({}, stream="stream")
    with pytest.raises(BackendInferenceError, match="quarantined"):
        model.submit_async_isolated({}, stream="stream")
    assert state.submitted == 1
    state.sync = 1
    with pytest.raises(RuntimeError, match="synchronize_stream"):
        model.close()
    assert state.destroyed == [] and unloaded == []
    state.sync = 0
    model.close()
    model.close()
    assert set(state.destroyed) == datasets and len(state.destroyed) == 2
    assert unloaded == [True]


def test_manager_retries_real_session_drain_without_releasing_dependencies(model_probe):
    from inference_service.backends.types import BackendState, RuntimeContext
    from inference_service.model_sessions.ascend import AscendOmModelSession
    from inference_service.pipeline.errors import PipelineManagerError
    from inference_service.pipeline.manager import InferencePipelineManager
    from inference_service.pipeline.runtime import InferencePipeline
    from inference_service.unified_runtime import (
        ExecutionContext,
        ExecutionFailure,
        LifecycleState,
        ModelRequest,
        ModelRuntimeHandle,
        RuntimeAssembly,
    )

    model, state, _ = model_probe
    session = AscendOmModelSession(runtime_manager=object(), priority_scheduling=True)
    session._load = lambda *_args: None
    closed = []
    session._models = {
        "producer": SimpleNamespace(close=lambda: closed.append("producer")),
        "consumer": model,
    }
    session._priority_streams = SimpleNamespace(close=lambda: closed.append("streams"))
    session._lease = SimpleNamespace(close=lambda: closed.append("lease"))
    provider = SimpleNamespace(close=lambda: closed.append("provider"))
    executor = SimpleNamespace(close=lambda: closed.append("executor"))
    context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor,
            owned_components=(provider, session, executor),
            load_context=context,
        )
    )
    pipeline = object.__new__(InferencePipeline)
    pipeline._pipeline_id = "policy"
    pipeline._context = context
    pipeline._unified_handle = handle
    manager = InferencePipelineManager((pipeline,), retry_pending_close=True)
    manager.start()
    state.submit = state.sync = 1
    with pytest.raises(BackendInferenceError):
        model.submit_async_isolated({}, stream="stream")
    with pytest.raises(PipelineManagerError, match="shutdown"):
        manager.close()
    assert handle.health.state is LifecycleState.CLOSING
    assert session.health().state is BackendState.FAILED
    assert closed == ["executor"]
    with pytest.raises(ExecutionFailure):
        handle.execute(ModelRequest({}), ExecutionContext("after-close"))
    with pytest.raises(PipelineManagerError, match="closed"):
        manager.start()
    state.sync = 0
    manager.close()
    manager.close()
    assert closed == ["executor", "producer", "streams", "lease", "provider"]
    assert handle.health.state is LifecycleState.CLOSED
    assert session.health().state is BackendState.CLOSED


def test_scheduled_session_close_keeps_streams_and_lease_until_models_drain(model_probe):
    from inference_service.backends.types import BackendState
    from inference_service.model_sessions.ascend import AscendOmModelSession

    model, state, _ = model_probe
    session = AscendOmModelSession(runtime_manager=object(), priority_scheduling=True)
    closed = []
    session._models = {"encoder": model}
    session._priority_streams = SimpleNamespace(close=lambda: closed.append("streams"))
    session._lease = SimpleNamespace(close=lambda: closed.append("lease"))
    session._context = object()
    state.submit = state.sync = 1
    with pytest.raises(BackendInferenceError):
        model.submit_async_isolated({}, stream="stream")
    with pytest.raises(BackendLifecycleError, match="drain failed"):
        session.close()
    assert session.health().state is BackendState.FAILED
    assert session._context is not None
    assert closed == []
    state.sync = 0
    session.close()
    assert closed == ["streams", "lease"]
    assert session.health().state is BackendState.CLOSED


@pytest.mark.parametrize("scheduled", [False, True])
def test_ascend_session_factory_enables_async_only_when_scheduled(tmp_path, monkeypatch, scheduled):
    from inference_manifest import CompiledDeployment
    from inference_service.backends import RuntimeContext
    from inference_service.model_sessions import ascend

    path = tmp_path / "model.om"
    path.write_bytes(b"fake")
    deployment = CompiledDeployment.model_validate(
        {
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {"backend": "ascend", "target": {"runtime": "acl"}, "profile": {"device_id": 0}},
            "execution": ["model"],
            "bindings": {"model": _bindings()},
            "artifacts": {"model": {"path": "model.om", "format": "om"}},
        }
    )
    options = []
    lease = SimpleNamespace(close=lambda: None)

    def factory(_lease, role, model_path, bindings, **kwargs):
        options.append(kwargs)
        return SimpleNamespace(load_descriptor=lambda: None, prepare_datasets=lambda **kwargs: None, close=lambda: None)

    monkeypatch.setattr(
        ascend.AclPriorityStreamPool, "create", lambda _lease: SimpleNamespace(level_count=8, close=lambda: None)
    )
    session = ascend.AscendOmModelSession(
        runtime_manager=SimpleNamespace(acquire=lambda device: lease),
        model_factory=factory,
        priority_scheduling=scheduled,
    )
    context = RuntimeContext(
        SimpleNamespace(deployment=deployment, bundle_root=tmp_path), priority_scheduling=scheduled
    )
    try:
        session.load(context)
        assert options == [{"async_execution": True} if scheduled else {}]
    finally:
        session.close()
