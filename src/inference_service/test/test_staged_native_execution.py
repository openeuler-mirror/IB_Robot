import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from ibrobot_tracing import TraceEmitter
from inference_manifest import ArtifactBindings, TensorBinding
from inference_service.backends.types import BackendCapabilities, BackendPriorityMapping
from inference_service.codecs import build_execution_plan
from inference_service.pipeline import runtime as runtime_module
from inference_service.pipeline import staged_executor as staged_module
from inference_service.pipeline.staged_executor import StagedModelExecutor, StagedScheduling
from inference_service.pipeline.stages import HostComputeStage, ModelStage
from inference_service.unified_runtime import ExecutionContext, ModelRequest
from inference_service.unified_runtime.errors import CancellationRequested


@pytest.fixture(autouse=True, params=[False, True], ids=["trace-off", "trace-on"])
def staged_tracing_mode(request, monkeypatch):
    records = []
    logger = logging.Logger("staged-test", logging.INFO)
    handler = logging.Handler()
    handler.emit = lambda record: records.append(json.loads(record.getMessage().split(" ", 1)[1]))
    logger.addHandler(handler)
    emitter = TraceEmitter(logger, enabled=request.param)
    monkeypatch.setattr(staged_module, "trace", emitter)
    monkeypatch.setattr(runtime_module, "trace", emitter)
    return emitter, records


def test_worker_identity_and_reused_snapshot_are_observed_without_extra_execution(staged_tracing_mode):
    emitter, records = staged_tracing_mode
    calls = []

    class Session:
        def execute_role(self, role, inputs, request, context, **kwargs):
            calls.append((role, context.request_id))
            assert not any("trace" in key for key in request.metadata)
            if role == "encoder":
                return {"internal.features": np.array([3], dtype=np.float32)}
            return {"action": inputs["internal.features"] + inputs["state"]}

    executor = _executor(Session())
    try:
        with emitter.trace_context("caller"):
            assert executor.submit_frame(_request("producer", 1, 0)).result(timeout=2) == 1
            for request_id, state in [("first", 10), ("second", 20)]:
                result = executor.execute(
                    _request(request_id, 9, state), ExecutionContext.create(request_id, timeout=2)
                )
                np.testing.assert_array_equal(result, [state + 3])
            emitter.event("caller_after")
        assert calls == [("encoder", "producer"), ("decoder", "first"), ("decoder", "second")]
        if emitter.enabled:
            starts = {r["fields"]["span_id"]: r for r in records if r["event"] == "span_begin"}
            ends = {r["fields"]["span_id"]: r for r in records if r["event"] == "span_end"}
            assert starts.keys() == ends.keys() and len(starts) == 3
            assert {(r["fields"]["span_name"], r["fields"]["trace_id"]) for r in starts.values()} == {
                ("visual_stage", "producer"),
                ("action_stage", "first"),
                ("action_stage", "second"),
            }
            selected = [r["fields"] for r in records if r["event"] == "visual_snapshot_selected"]
            assert [(r["trace_id"], r["version"]) for r in selected] == [("first", 1), ("second", 1)]
            assert records[-1]["fields"]["trace_id"] == "caller"
        else:
            assert not records
    finally:
        executor.close()


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_worker_failure_keeps_original_error_and_closes_trace(staged_tracing_mode, role):
    emitter, records = staged_tracing_mode
    fault = RuntimeError("vendor failure")

    class Session:
        def execute_role(self, current, inputs, request, context, **kwargs):
            if current == role:
                raise fault
            return {"internal.features": np.array([3], dtype=np.float32)}

    executor = _executor(Session())
    try:
        if role == "encoder":
            with pytest.raises(RuntimeError) as error:
                executor.submit_frame(_request("producer", 1, 0)).result(timeout=2)
        else:
            executor.submit_frame(_request("producer", 1, 0)).result(timeout=2)
            with pytest.raises(RuntimeError) as error:
                executor.execute(_request("consumer", 1, 0), ExecutionContext.create("consumer", timeout=2))
        assert error.value is fault
        if emitter.enabled:
            starts = [r for r in records if r["event"] == "span_begin"]
            ends = [r for r in records if r["event"] == "span_end"]
            assert len(starts) == len(ends)
            assert ends[-1]["fields"]["status"] == "error"
        else:
            assert not records
    finally:
        executor.close()


def _handle(executor):
    from inference_service.unified_runtime import ModelRuntimeHandle, RuntimeAssembly

    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor, owned_components=(executor,), resettable=True, retry_failed_reset=True
        )
    )
    executor.frame_submitter = handle.submit_frame
    handle.load(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
    return handle


def _binding(semantic, index=0):
    return TensorBinding(semantic=semantic, index=index, dtype="float32", shape=(1,))


def _executor(session, *, load=True, frame_base_priority=0, max_snapshot_age_ms=5000):
    from inference_service.model_sessions.base import ModelSession

    if not isinstance(session, ModelSession):
        session.capabilities = BackendCapabilities(
            supports_isolated_stage_execution=True,
            priority_mapping=BackendPriorityMapping(tuple(range(8))),
            supports_cancellation=getattr(getattr(session, "capabilities", None), "supports_cancellation", False),
        )
    bindings = {
        "encoder": ArtifactBindings(inputs=(_binding("image"),), outputs=(_binding("internal.features"),)),
        "decoder": ArtifactBindings(
            inputs=(_binding("internal.features"), _binding("state", 1)), outputs=(_binding("action"),)
        ),
    }
    executor = StagedModelExecutor(
        (
            HostComputeStage(lambda values: {"_selected_prompt": values["prompt"]}),
            ModelStage("encoder", session),
            ModelStage("decoder", session),
        ),
        SimpleNamespace(adapt=lambda frame: frame.values["action"]),
        components=(session,),
        execution_plan=build_execution_plan(("encoder", "decoder"), bindings),
        scheduling=StagedScheduling(
            priorities={"encoder": 1, "decoder": 0},
            triggers={"encoder": "frame_arrival", "decoder": "dispatch"},
            frame_base_priority=frame_base_priority,
            max_snapshot_age_ms=max_snapshot_age_ms,
        ),
    )
    if load:
        executor.load(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
    return executor


@pytest.mark.parametrize("isolated,streams", [(False, False), (False, True), (True, False)])
def test_staged_load_requires_actual_backend_capabilities(isolated, streams):
    session = SimpleNamespace()
    executor = _executor(session, load=False)
    session.capabilities = BackendCapabilities(
        supports_isolated_stage_execution=isolated,
        priority_mapping=BackendPriorityMapping(tuple(range(8))) if streams else None,
    )
    try:
        with pytest.raises(ValueError, match="loaded isolated async"):
            executor.load(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
        assert not executor.supports_priority_zero_deadline_admission
    finally:
        executor.close()


def _request(request_id, image, state, prompt="task", *, priority=0):
    return ModelRequest(
        {"image": np.array([image], dtype=np.float32), "state": np.array([state], dtype=np.float32), "prompt": prompt},
        {"request_id": request_id, "priority": priority},
    )


@pytest.mark.parametrize("unknown", [False, True])
def test_staged_reset_with_real_stateless_session_recovers_after_draining_unknown(unknown):
    from inference_service.backends.types import RuntimeContext
    from inference_service.model_sessions.ascend import AscendOmModelSession
    from inference_service.scheduler.operations import Certainty
    from inference_service.unified_runtime import ModelRuntimeHandle, RuntimeAssembly

    session = AscendOmModelSession(runtime_manager=object(), priority_scheduling=True)
    session._load = lambda *_args: session._update_loaded_capabilities(
        supports_isolated_stage_execution=True, priority_mapping=BackendPriorityMapping(tuple(range(8)))
    )
    executor = _executor(session, load=False)
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor,
            session=session,
            owned_components=(session, executor),
            resettable=True,
            retry_failed_reset=True,
        )
    )
    context = RuntimeContext(
        SimpleNamespace(
            deployment=SimpleNamespace(backend="ascend"),
        )
    )
    try:
        handle.load(context)
        generation = executor._generation
        if unknown:
            operation, registry = executor._new_operation(ExecutionContext("uncertain"), "encoder")
            operation.claim_send()
            registry.finish(operation.operation_id, certainty=Certainty.UNKNOWN)
            registry.detach_waiter(operation.operation_id, "uncertain")
            # Device quarantine is drained (no-op on the empty session), the
            # no-waiter UNKNOWN record is fenced, and reset opens a new
            # generation instead of bricking the pipeline.
            handle.reset()
            assert registry.get(operation.operation_id) is None
            assert len(registry) == 0
            assert executor._generation == generation + 1
            assert handle.health.ready
        else:
            handle.reset()
            assert handle.health.ready and session.health().ready
            assert executor._generation == generation + 1
    finally:
        handle.close()


def test_staged_reset_still_refuses_unknown_records_with_live_waiters():
    from inference_service.scheduler.operations import Certainty

    executor = _executor(SimpleNamespace())
    try:
        operation, registry = executor._new_operation(ExecutionContext("uncertain"), "encoder")
        operation.claim_send()
        registry.finish(operation.operation_id, certainty=Certainty.UNKNOWN)
        with pytest.raises(RuntimeError, match="known outcomes"):
            executor.reset()
        assert registry.get(operation.operation_id) is operation
        assert executor._generation == 1
        # Retrying after the waiter leaves drains and fences the record.
        registry.detach_waiter(operation.operation_id, "uncertain")
        executor.reset()
        assert registry.get(operation.operation_id) is None
        assert executor._generation == 2
    finally:
        executor.close()


def test_staged_reset_drain_failure_is_retryable():
    from inference_service.unified_runtime import ExecutionFailure

    class DrainingSession(SimpleNamespace):
        drain_failures = 0

        def execute_role(self, *args, **kwargs):
            return {"internal.features": args[2]["image"]}

        def drain_uncertain_operations(self):
            if DrainingSession.drain_failures > 0:
                DrainingSession.drain_failures -= 1
                raise RuntimeError("device drain failed")

    session = DrainingSession()
    executor = _executor(session)
    handle = _handle(executor)
    try:
        DrainingSession.drain_failures = 1
        with pytest.raises(ExecutionFailure, match="device drain failed"):
            handle.reset()
        assert executor._generation == 1
        # The failed drain left the executor intact; a retry succeeds.
        handle.reset()
        assert executor._generation == 2
        assert handle.health.ready
    finally:
        handle.close()


def test_native_stages_handoff_snapshot_and_use_current_action_state():
    calls = []
    features = np.array([3], dtype=np.float32)

    class Session:
        def execute_role(self, role, inputs, request, context, *, isolated, operation_factory):
            assert isolated
            assert isinstance(context, ExecutionContext)
            calls.append((role, context.request_id, request.metadata["priority"]))
            operation, registry = operation_factory(role)
            assert operation is not None and registry is not None
            from inference_service.scheduler.operations import Certainty

            registry.finish(operation.operation_id, certainty=Certainty.COMPLETED)
            registry.detach_waiter(operation.operation_id, context.request_id)
            if role == "encoder":
                return {"internal.features": features}
            return {"action": inputs["internal.features"] + inputs["state"]}

    executor = _executor(Session())
    try:
        assert executor.submit_frame(_request("frame", 1, 10)).result(timeout=2) == 1
        features.fill(100)
        result = executor.execute(_request("dispatch", 9, 20), ExecutionContext.create("dispatch", timeout=2))
        np.testing.assert_array_equal(result, [23])
        assert calls == [("encoder", "frame", 1), ("decoder", "dispatch", 0)]
        executor.reset()
        result = executor.execute(_request("next", 9, 2, "new"), ExecutionContext.create("next", timeout=2))
        np.testing.assert_array_equal(result, [102])
        assert calls[-2][0] == "encoder"
        assert calls[-2][1].startswith("next:refresh:")
        assert calls[-2][2] == 1
        assert calls[-1] == ("decoder", "next", 0)
    finally:
        executor.close()


@pytest.mark.parametrize("cache_state", ["missing", "stale", "prompt_changed"])
def test_visual_refresh_uses_configured_priority_without_changing_action_priority(cache_state):
    from dataclasses import replace

    calls = []

    class Session:
        def execute_role(self, role, inputs, request, context, **_kwargs):
            calls.append((role, request.metadata["priority"]))
            return {"internal.features": inputs["image"]} if role == "encoder" else {"action": inputs["state"]}

    executor = _executor(Session(), frame_base_priority=2)
    handle = _handle(executor)
    try:
        if cache_state != "missing":
            handle.submit_frame(_request("frame", 1, 2), ExecutionContext.create("frame", timeout=2)).result(2)
            assert calls == [("encoder", 3)]
            if cache_state == "stale":
                executor._snapshot = replace(executor._snapshot, observation_mono_ns=0)
        calls.clear()
        request = _request("dispatch", 1, 2, "new" if cache_state == "prompt_changed" else "task", priority=5)
        handle.execute(request, ExecutionContext.create("dispatch", timeout=2))
        assert calls == [("encoder", 3), ("decoder", 5)]
    finally:
        handle.close()


@pytest.mark.parametrize("max_age_ms,refresh", [(5000, False), (50, True)])
def test_snapshot_expiry_includes_producer_latency(monkeypatch, max_age_ms, refresh):
    clock_ns = [10_000_000_000]
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock_ns[0])
    calls = []

    class Session:
        def execute_role(self, role, inputs, request, context, **_kwargs):
            calls.append(role)
            if role == "encoder" and len(calls) == 1:
                clock_ns[0] += 100_000_000
            return {"internal.features": inputs["image"]} if role == "encoder" else {"action": inputs["state"]}

    executor = _executor(Session(), max_snapshot_age_ms=max_age_ms)
    try:
        executor.submit_frame(_request("frame", 1, 2)).result(2)
        executor.execute(_request("dispatch", 1, 2), ExecutionContext.create("dispatch", timeout=2))
        assert calls == (["encoder", "encoder", "decoder"] if refresh else ["encoder", "decoder"])
    finally:
        executor.close()


def test_refresh_cannot_renew_expired_observation_or_spin(monkeypatch):
    monkeypatch.setattr(time, "monotonic_ns", lambda: 10_000_000_000)
    calls = []

    class Session:
        def execute_role(self, role, inputs, request, context, **_kwargs):
            calls.append(role)
            return {"internal.features": inputs["image"]}

    executor = _executor(Session(), max_snapshot_age_ms=50)
    request = _request("old", 1, 2)
    request = ModelRequest(request.inputs, {**request.metadata, "observation_monotonic_ns": 9_000_000_000})
    try:
        with pytest.raises(TimeoutError, match="visual observation expired"):
            executor.execute(request, ExecutionContext("old"))
        assert calls == ["encoder"]
    finally:
        executor.close()


@pytest.mark.parametrize("requested_seconds,override_seconds", [(None, None), (10, None), (0.1, None), (10, 0.2)])
def test_frame_submission_combines_pipeline_and_caller_deadlines(requested_seconds, override_seconds):
    from inference_service.backends import InferenceRequest
    from inference_service.pipeline.runtime import InferencePipeline

    pipeline = object.__new__(InferencePipeline)
    pipeline._pipeline_id = "policy"
    pipeline._request_timeout = 1
    pipeline._session_executor = SimpleNamespace(submit_frame=lambda: None)
    contexts = []
    pipeline._unified_handle = SimpleNamespace(submit_frame=lambda _request, context: contexts.append(context))
    before = datetime.now(timezone.utc)
    requested = before + timedelta(seconds=requested_seconds) if requested_seconds is not None else None
    override = before + timedelta(seconds=override_seconds) if override_seconds is not None else None
    pipeline.submit_frame(InferenceRequest("frame", inputs={}, deadline=requested), deadline=override)
    expiry = contexts[0].deadline.expires_at
    effective = override or requested
    if effective is not None and effective < before + timedelta(seconds=1):
        assert expiry == effective
    else:
        assert before + timedelta(seconds=1) <= expiry <= datetime.now(timezone.utc) + timedelta(seconds=1)


def test_timed_out_camera_frame_does_not_publish_snapshot():
    from inference_service.backends import InferenceRequest
    from inference_service.pipeline.runtime import InferencePipeline
    from inference_service.unified_runtime import ExecutionFailure

    entered, release = threading.Event(), threading.Event()

    class Session:
        def execute_role(self, role, inputs, request, context, **_kwargs):
            assert role == "encoder"
            entered.set()
            assert release.wait(2)
            return {"internal.features": inputs["image"]}

    executor = _executor(Session())
    handle = _handle(executor)
    pipeline = object.__new__(InferencePipeline)
    pipeline._pipeline_id = "policy"
    pipeline._request_timeout = 0.05
    pipeline._session_executor = executor
    pipeline._unified_handle = handle
    try:
        future = pipeline.submit_frame(InferenceRequest("frame", inputs=_request("frame", 1, 2).inputs))
        assert entered.wait(2)
        time.sleep(0.06)
        release.set()
        with pytest.raises(ExecutionFailure, match="deadline"):
            future.result(2)
        assert executor._snapshot is None
    finally:
        release.set()
        handle.close()


def test_dispatch_cancellation_interrupts_snapshot_wait_and_drain_finishes():
    entered = threading.Event()
    release = threading.Event()

    class Session:
        def execute_role(self, role, inputs, request, context, *, isolated, operation_factory):
            assert role == "encoder"
            entered.set()
            assert release.wait(2)
            return {"internal.features": inputs["image"]}

    executor = _executor(Session())
    context = ExecutionContext.create("cancel", timeout=2)
    with ThreadPoolExecutor(max_workers=1) as caller:
        try:
            future = caller.submit(executor.execute, _request("cancel", 1, 2), context)
            assert entered.wait(1)
            context.cancellation_token.cancel("test")
            with pytest.raises(CancellationRequested):
                future.result(timeout=1)
        finally:
            release.set()
            executor.close()


def test_completion_callback_cannot_overtake_reserved_pending_frame():
    entered, release = threading.Event(), threading.Event()
    newest_entered, newest_release = threading.Event(), threading.Event()
    calls, reentrant = [], []

    def execute_role(role, inputs, request, context, **kwargs):
        calls.append(context.request_id)
        if context.request_id == "first":
            entered.set()
            assert release.wait(2)
        if context.request_id == "newest":
            newest_entered.set()
            assert newest_release.wait(2)
        return {"internal.features": inputs["image"]}

    executor = _executor(SimpleNamespace(execute_role=execute_role))
    try:
        first = executor.submit_frame(_request("first", 1, 0))
        assert entered.wait(1)
        pending = executor.submit_frame(_request("pending", 2, 0))
        first.add_done_callback(lambda _: reentrant.append(executor.submit_frame(_request("newest", 3, 0))))
        release.set()
        assert pending.result(timeout=2) == 2
        assert newest_entered.wait(1)
        with ThreadPoolExecutor(max_workers=1) as control:
            resetting = threading.Event()

            def reset():
                resetting.set()
                executor.reset()

            reset_future = control.submit(reset)
            try:
                assert resetting.wait(1)
                assert not reset_future.done()
                assert executor._visual_active == 1
            finally:
                newest_release.set()
            reset_future.result(timeout=2)
        assert reentrant[0].result(timeout=2) == 3
        assert calls == ["first", "pending", "newest"]
        assert executor._snapshot is None
        assert executor._visual_active == 0
    finally:
        release.set()
        newest_release.set()
        executor.close()


@pytest.mark.parametrize("colliding_frame", [False, True])
def test_frame_request_id_collision_is_rejected_without_losing_active_owner(colliding_frame):
    from inference_service.unified_runtime import ExecutionFailure

    entered, release = threading.Event(), threading.Event()

    def execute_role(role, inputs, request, context, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"internal.features": inputs["image"]}

    handle = _handle(_executor(SimpleNamespace(execute_role=execute_role)))
    try:
        first = handle.submit_frame(_request("same", 1, 0), ExecutionContext("same"))
        assert entered.wait(1)
        with pytest.raises(ExecutionFailure) as error:
            if colliding_frame:
                handle.submit_frame(_request("same", 2, 0), ExecutionContext("same"))
            else:
                handle.execute(_request("same", 2, 0), ExecutionContext("same"))
        assert error.value.code == "request_conflict"
        assert handle.active_executions == 1
        release.set()
        first.result(timeout=2)
        assert handle.active_executions == 0
    finally:
        release.set()
        handle.close()


def test_policy_facade_assembles_and_submits_to_staged_executor_with_device_links(staged_tracing_mode):
    from inference_manifest import CompiledDeployment
    from inference_service.backends import BackendCapabilities, InferenceRequest
    from inference_service.codecs.policies import PI05PolicyCodec
    from inference_service.pipeline.executor import SequentialModelExecutor
    from inference_service.pipeline.runtime import InferencePipeline, _PolicySessionHandle
    from inference_service.unified_runtime import RuntimeAssembly

    calls = []
    session = SimpleNamespace(
        execute_role=lambda role, inputs, request, context, **kwargs: (
            calls.append(role) or {"internal.features": inputs["image"]}
        ),
    )
    original = _executor(session)
    try:
        bindings = {role: original._execution_plan.role(role).bindings for role in ("encoder", "decoder")}
        deployment = CompiledDeployment.model_validate(
            {
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {"backend": "ascend", "target": {"runtime": "acl"}, "profile": {"device_id": 0}},
                "execution": ["encoder", "decoder"],
                "bindings": bindings,
                "artifacts": {role: {"path": f"{role}.om", "format": "om"} for role in bindings},
                "device_links": [
                    {
                        "semantic": "internal.features",
                        "producer": "encoder",
                        "consumer": "decoder",
                        "transport": "device_pointer",
                        "owner": "producer",
                    }
                ],
            }
        )
        context = SimpleNamespace(
            deployment=deployment,
            policy=SimpleNamespace(policy_type="act"),
            model_type="act",
            backend_profile=deployment.runtime_profile.backend_profile,
            deployment_name="ascend",
            priority_scheduling=True,
        )
        sequential = SequentialModelExecutor(
            original.stages[1:],
            original._result_adapter,
            components=(session,),
            execution_plan=build_execution_plan(deployment.execution, bindings, deployment.device_links),
        )
        handle = _PolicySessionHandle(sequential, context, "action", None, None, BackendCapabilities(), None, None)

        class TestPolicyCodec(PI05PolicyCodec):
            @staticmethod
            def validate_staged_deployment(value):
                assert value.execution == ("encoder", "decoder")

        pipeline = InferencePipeline(
            "policy",
            context,
            runtime_assembly=RuntimeAssembly(runtime_executor=sequential, session=session),
            session_handle=handle,
            codec=TestPolicyCodec(SimpleNamespace(output_features={"action": SimpleNamespace(shape=(1,))})),
            stage_policy="independent",
            frame_base_priority=3,
            stage_scheduling={"encoder": {"max_snapshot_age_ms": 250}},
        )
        try:
            assert isinstance(pipeline._runtime_assembly.runtime_executor, StagedModelExecutor)
            assert pipeline._session_executor._scheduling.frame_base_priority == 3
            assert pipeline._session_executor._max_snapshot_age_ns == 250_000_000
            assert pipeline._runtime_assembly.retry_failed_reset
            assert pipeline._runtime_assembly.owned_components[-1].resource is pipeline._session_executor
            pipeline._unified_handle.load(context)
            future = pipeline.submit_frame(
                InferenceRequest("frame", inputs={"image": np.ones(1), "state": np.zeros(1)})
            )
            assert future.result(timeout=2) == 1
            assert calls == ["encoder"]
            assert "internal.features" in pipeline._session_executor._snapshot.execution.host_tensors
            emitter, records = staged_tracing_mode
            if emitter.enabled:
                starts = [record["fields"] for record in records if record["event"] == "span_begin"]
                ends = [record["fields"] for record in records if record["event"] == "span_end"]
                assert {record["span_id"] for record in starts} == {record["span_id"] for record in ends}
                assert {record["span_name"] for record in starts} == {"visual_stage", "preprocess"}
                assert all(record["trace_id"] == "frame" for record in starts)
                assert not any(record["event"].startswith("flow_") for record in records)
        finally:
            pipeline._session_executor.close()
    finally:
        original.close()


def test_frame_handle_cancellation_keeps_ownership_until_worker_finishes():
    from inference_service.unified_runtime import ExecutionFailure

    entered, release = threading.Event(), threading.Event()

    def execute_role(role, inputs, request, context, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"internal.features": inputs["image"]}

    executor = _executor(SimpleNamespace(execute_role=execute_role))
    handle = _handle(executor)
    try:
        future = handle.submit_frame(_request("frame", 1, 2), ExecutionContext("frame"))
        assert entered.wait(1)
        assert handle.diagnostics().active_executions == 1
        handle.cancel("frame")
        assert not future.cancel()
        assert handle.active_executions == 1
        release.set()
        with pytest.raises(ExecutionFailure):
            future.result(timeout=2)
        assert handle.active_executions == 0
        assert executor._snapshot is None
    finally:
        release.set()
        handle.close()


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_handle_control_drains_frames_before_reset_or_resource_close(operation):
    from inference_service.unified_runtime import ExecutionFailure

    entered, release, controlling = threading.Event(), threading.Event(), threading.Event()

    def execute_role(role, inputs, request, context, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"internal.features": inputs["image"]}

    executor = _executor(SimpleNamespace(execute_role=execute_role))
    handle = _handle(executor)
    original_drain = handle._wait_quiescent_locked

    def drain(context, **kwargs):
        controlling.set()
        return original_drain(context, **kwargs)

    handle._wait_quiescent_locked = drain
    with ThreadPoolExecutor(max_workers=1) as caller:
        try:
            frame = handle.submit_frame(_request("frame", 1, 2), ExecutionContext("frame"))
            assert entered.wait(1)
            control = caller.submit(getattr(handle, operation), deadline=None)
            assert controlling.wait(1)
            assert not control.done()
            assert executor.generation == 1
            assert not executor._closed
            with pytest.raises(ExecutionFailure):
                handle.submit_frame(_request("late", 1, 2), ExecutionContext("late"))
            release.set()
            frame.result(timeout=2)
            control.result(timeout=2)
            assert handle.active_executions == 0
            assert executor._closed if operation == "close" else executor.generation == 2
        finally:
            release.set()
            handle.close()


def test_handle_tracks_latest_pending_and_dispatch_refresh_without_id_collision():
    from inference_service.unified_runtime import ExecutionFailure

    entered, release = threading.Event(), threading.Event()
    images = []

    def execute_role(role, inputs, request, context, **kwargs):
        if role == "encoder":
            images.append(float(inputs["image"][0]))
            entered.set()
            assert release.wait(2)
            return {"internal.features": inputs["image"]}
        return {"action": inputs["internal.features"] + inputs["state"]}

    executor = _executor(SimpleNamespace(execute_role=execute_role))
    handle = _handle(executor)
    try:
        first = handle.submit_frame(_request("first", 1, 2), ExecutionContext("first"))
        assert entered.wait(1)
        old = handle.submit_frame(_request("old", 2, 2), ExecutionContext("old"))
        newest = handle.submit_frame(_request("new", 3, 2), ExecutionContext("new"))
        with pytest.raises(ExecutionFailure) as superseded:
            old.result(timeout=1)
        assert superseded.value.evidence.outcome_known
        assert handle.active_executions == 2
        release.set()
        first.result(timeout=2)
        newest.result(timeout=2)
        assert images == [1, 3]
        assert handle.active_executions == 0
        # Coalesced frames are not failures: health stays clean.
        assert handle.health.failure_count == 0
        assert handle.health.ready
        # A new prompt requires a producer request while Dispatch is admitted.
        handle.execute(_request("dispatch", 4, 2, "different"), ExecutionContext("dispatch"))
        assert images == [1, 3, 4]
        assert handle.active_executions == 0
    finally:
        release.set()
        handle.close()


def test_unknown_frame_failure_closes_handle_admission_and_updates_health():
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.unified_runtime import ExecutionFailure, LifecycleState

    def execute_role(*args, **kwargs):
        raise BackendInferenceError("unknown submission", operation_started=True, outcome_known=False)

    handle = _handle(_executor(SimpleNamespace(execute_role=execute_role)))
    try:
        future = handle.submit_frame(_request("frame", 1, 2), ExecutionContext("frame"))
        with pytest.raises(ExecutionFailure) as error:
            future.result(timeout=2)
        assert not error.value.evidence.outcome_known
        assert handle.state is LifecycleState.RESET_REQUIRED
        assert handle.health.failure_count == 1
        assert handle.active_executions == 0
        with pytest.raises(ExecutionFailure):
            handle.submit_frame(_request("next", 1, 2), ExecutionContext("next"))
    finally:
        handle.close()


def test_frame_submit_failure_releases_handle_record_and_does_not_block_retry():
    from inference_service.unified_runtime import ExecutionFailure

    executor = _executor(SimpleNamespace())
    handle = _handle(executor)

    def reject(*args, **kwargs):
        raise RuntimeError("submission rejected")

    executor.submit_frame = reject
    try:
        for _ in range(4):
            with pytest.raises(ExecutionFailure):
                handle.submit_frame(_request("retry", 1, 2), ExecutionContext("retry")).result(timeout=1)
            assert handle.active_executions == 0
            assert handle.ready
    finally:
        handle.close()


def test_executors_share_component_health_aggregation_and_cancellation():
    from datetime import datetime, timezone

    from inference_service.backends.types import BackendHealth, BackendState
    from inference_service.pipeline.executor import SequentialModelExecutor

    canceled = []
    recent = datetime.now(timezone.utc)
    first = SimpleNamespace(
        health=lambda: BackendHealth(state=BackendState.READY, ready=True, failure_count=2),
        capabilities=SimpleNamespace(supports_cancellation=True),
        cancel=lambda request_id, **kwargs: canceled.append(request_id),
    )
    second = SimpleNamespace(
        health=lambda: BackendHealth(
            state=BackendState.READY, ready=True, failure_count=3, last_successful_inference_time=recent
        )
    )
    staged = _executor(first)
    staged._components = (first, second)
    sequential = SequentialModelExecutor(staged.stages, staged._result_adapter, components=(first, second, first))
    try:
        assert staged.health() == sequential.health()
        assert staged.health().failure_count == 5
        assert staged.health().last_successful_inference_time == recent
        for executor in (staged, sequential):
            executor.cancel("same-request")
        assert canceled == ["same-request", "same-request"]
    finally:
        staged.close()
