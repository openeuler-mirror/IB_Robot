import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from inference_service.backends.errors import BackendInferenceError
from inference_service.pipeline.staged_executor import (
    StagedModelExecutor,
    StagedScheduling,
    VisualFrameSupersededError,
)
from inference_service.pipeline.stages import ModelStage
from inference_service.scheduler.operations import (
    Certainty,
    OperationIdentity,
    OperationKind,
    OperationRegistry,
    OperationState,
)
from inference_service.unified_runtime import ExecutionContext


def _identity():
    return OperationIdentity(str(uuid4()), 1, str(uuid4()), 1, str(uuid4()))


def test_waiter_detach_does_not_delete_or_complete_operation():
    registry = OperationRegistry(max_records=2, max_waiters_per_operation=2)
    operation, created = registry.create_or_get(
        kind=OperationKind.OPEN,
        idempotency_key=("open", "session"),
        identity=_identity(),
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    assert created and operation.claim_send()
    operation.detach_waiter("caller")
    assert operation.waiter_count == 0
    assert registry.get(operation.operation_id) is operation
    assert operation.state is OperationState.SEND_PENDING


def test_accepted_operation_cannot_be_reclassified_not_started():
    registry = OperationRegistry(max_records=1, max_waiters_per_operation=1)
    operation, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "request"),
        identity=_identity(),
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    operation.claim_send()
    operation.transition(OperationState.ACCEPTED)
    with pytest.raises(RuntimeError, match="cannot become NOT_STARTED"):
        operation.finish(certainty=Certainty.NOT_STARTED)
    operation.finish(certainty=Certainty.UNKNOWN)
    assert operation.state is OperationState.QUARANTINED
    assert registry.remove_terminal(operation.operation_id) is False


def test_duplicate_waiters_share_one_operation_and_bounds_are_hard():
    registry = OperationRegistry(max_records=1, max_waiters_per_operation=2)
    identity = _identity()
    first, created = registry.create_or_get(
        kind=OperationKind.CLOSE,
        idempotency_key=("close", "session"),
        identity=identity,
        deadline_mono_ns=10,
        waiter_id="one",
    )
    replay, replay_created = registry.create_or_get(
        kind=OperationKind.CLOSE,
        idempotency_key=("close", "session"),
        identity=identity,
        deadline_mono_ns=20,
        waiter_id="two",
    )
    assert created and not replay_created and replay is first
    with pytest.raises(RuntimeError, match="waiter_capacity"):
        registry.create_or_get(
            kind=OperationKind.CLOSE,
            idempotency_key=("close", "session"),
            identity=identity,
            deadline_mono_ns=30,
            waiter_id="three",
        )


def test_completed_operations_can_be_detached_and_reclaimed_repeatedly():
    registry = OperationRegistry(max_records=1, max_waiters_per_operation=1)
    for sequence in range(1000):
        operation, created = registry.create_or_get(
            kind=OperationKind.DISPATCH,
            idempotency_key=("dispatch", sequence),
            identity=_identity(),
            deadline_mono_ns=10,
            waiter_id="caller",
        )
        assert created
        operation.claim_send()
        operation.finish(certainty=Certainty.COMPLETED, result=sequence)
        registry.detach_waiter(operation.operation_id, "caller")
        assert registry.get(operation.operation_id) is None
    assert len(registry) == 0


def test_late_known_terminal_reclaims_a_detached_operation() -> None:
    registry = OperationRegistry(max_records=1, max_waiters_per_operation=1)
    operation, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "late"),
        identity=_identity(),
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    operation.claim_send()
    registry.detach_waiter(operation.operation_id, "caller")

    assert registry.finish(operation.operation_id, certainty=Certainty.COMPLETED, result="done")
    assert registry.get(operation.operation_id) is None
    assert len(registry) == 0


def test_late_unknown_terminal_remains_quarantined() -> None:
    registry = OperationRegistry(max_records=1, max_waiters_per_operation=1)
    operation, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "unknown"),
        identity=_identity(),
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    operation.claim_send()
    registry.detach_waiter(operation.operation_id, "caller")

    assert not registry.finish(operation.operation_id, certainty=Certainty.UNKNOWN, error="uncertain")
    assert registry.get(operation.operation_id) is operation
    assert operation.state is OperationState.QUARANTINED


def test_detaching_last_waiter_reclaims_an_already_known_terminal() -> None:
    registry = OperationRegistry(max_records=1, max_waiters_per_operation=1)
    operation, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "terminal-before-detach"),
        identity=_identity(),
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    operation.claim_send()
    assert not registry.finish(operation.operation_id, certainty=Certainty.NOT_STARTED, error="rejected")

    registry.detach_waiter(operation.operation_id, "caller")

    assert registry.get(operation.operation_id) is None


def test_pipeline_boot_fence_removes_only_old_boot_operations():
    registry = OperationRegistry(max_records=2, max_waiters_per_operation=1)
    old_identity = _identity()
    new_identity = _identity()
    for key, identity in (("old", old_identity), ("new", new_identity)):
        operation, _ = registry.create_or_get(
            kind=OperationKind.DISPATCH,
            idempotency_key=("dispatch", key),
            identity=identity,
            deadline_mono_ns=10,
            waiter_id="caller",
        )
        operation.claim_send()
        operation.detach_waiter("caller")

    assert registry.fence_boot(old_identity.expected_boot_id) == 1
    assert len(registry) == 1


@pytest.mark.parametrize("different", ["session_id", "binding_id", "binding_incarnation", "expected_boot_id"])
def test_binding_drain_reclaims_only_its_own_unknown_operation(different):
    registry = OperationRegistry(max_records=2, max_waiters_per_operation=1)
    identity = _identity()
    other = replace(identity, **{different: 2 if different == "binding_incarnation" else str(uuid4())})
    operations = []
    for key, owner in (("drained", identity), ("other", other)):
        operation, _ = registry.create_or_get(
            kind=OperationKind.DISPATCH,
            idempotency_key=(key,),
            identity=owner,
            deadline_mono_ns=10,
            waiter_id="caller",
        )
        operation.claim_send()
        registry.finish(operation.operation_id, certainty=Certainty.UNKNOWN)
        registry.detach_waiter(operation.operation_id, "caller")
        operations.append(operation)
    binding = asdict(identity)
    binding.pop("logical_generation")
    assert registry.fence_binding(**binding) == 1
    assert registry.get(operations[0].operation_id) is None
    assert registry.get(operations[1].operation_id) is operations[1]
    registry.create_or_get(
        kind=OperationKind.OPEN,
        idempotency_key=("new",),
        identity=identity,
        deadline_mono_ns=20,
        waiter_id="next",
    )


@pytest.mark.parametrize("finish_before_drain", [False, True])
def test_binding_drain_retains_waiter_but_reclaims_late_unknown(finish_before_drain):
    registry = OperationRegistry(max_records=1, max_waiters_per_operation=1)
    identity = _identity()
    operation, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("request",),
        identity=identity,
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    operation.claim_send()
    if finish_before_drain:
        registry.finish(operation.operation_id, certainty=Certainty.UNKNOWN)
    binding = asdict(identity)
    binding.pop("logical_generation")
    assert registry.fence_binding(**binding) == 0
    assert registry.get(operation.operation_id) is operation
    if not finish_before_drain:
        registry.finish(operation.operation_id, certainty=Certainty.UNKNOWN)
    assert operation.certainty is Certainty.UNKNOWN
    registry.detach_waiter(operation.operation_id, "caller")
    assert len(registry) == 0
    assert not registry.finish(operation.operation_id, certainty=Certainty.UNKNOWN)


def test_uncertain_fence_removes_only_quarantined_no_waiter_records():
    registry = OperationRegistry(max_records=4, max_waiters_per_operation=2)
    identity = _identity()
    other_identity = _identity()
    # Detached quarantined UNKNOWN: fenced by drain-then-reset.
    fenced, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "fenced"),
        identity=identity,
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    fenced.claim_send()
    fenced.finish(certainty=Certainty.UNKNOWN)
    registry.detach_waiter(fenced.operation_id, "caller")
    # UNKNOWN with a live waiter: may still resolve, stays behind.
    waited, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "waited"),
        identity=identity,
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    waited.claim_send()
    waited.finish(certainty=Certainty.UNKNOWN)
    # Still in flight (no terminal state): stays behind.
    inflight, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "inflight"),
        identity=identity,
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    inflight.claim_send()
    # Different boot: untouched.
    other, _ = registry.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "other-boot"),
        identity=other_identity,
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    other.claim_send()
    other.finish(certainty=Certainty.UNKNOWN)
    registry.detach_waiter(other.operation_id, "caller")

    assert registry.fence_uncertain(identity.expected_boot_id) == 1
    assert registry.get(fenced.operation_id) is None
    assert registry.get(waited.operation_id) is waited
    assert registry.get(inflight.operation_id) is inflight
    assert registry.get(other.operation_id) is other
    assert len(registry) == 3


def test_staged_reset_waits_for_visual_completion():
    executor = object.__new__(StagedModelExecutor)
    executor._snapshot_condition = threading.Condition(threading.RLock())
    executor._resetting = False
    executor._visual_pending = None
    executor._visual_active = 1
    executor._action_active = 1
    executor._components = ()
    executor._generation = 1
    executor._snapshot = None
    executor._identity = _identity()
    executor._operations = OperationRegistry(max_records=2, max_waiters_per_operation=2)
    executor.begin_generation = lambda: 2

    def finish():
        with executor._snapshot_condition:
            executor._visual_active = 0
            executor._action_active = 0
            executor._snapshot_condition.notify_all()

    timer = threading.Timer(0.01, finish)
    timer.start()
    StagedModelExecutor.reset(executor, datetime.now(timezone.utc) + timedelta(seconds=1))
    timer.join()


def test_staged_submit_frame_keeps_only_latest_pending_request():
    executor = object.__new__(StagedModelExecutor)
    executor._snapshot_condition = threading.Condition(threading.RLock())
    executor._closed = False
    executor._resetting = False
    executor._generation = 1
    executor._visual_active = 0
    executor._visual_pending = None
    executor._components = ()
    executor._visual_worker = ThreadPoolExecutor(max_workers=1)
    executor._action_worker = ThreadPoolExecutor(max_workers=1)
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def run(request, _deadline, _generation):
        started.set()
        release.wait(timeout=1.0)
        calls.append(request.metadata["request_id"])
        return len(calls)

    executor._run_visual = run
    first = executor.submit_frame(SimpleNamespace(request_id="first", inputs={}))
    assert started.wait(timeout=1.0)
    second = executor.submit_frame(SimpleNamespace(request_id="second", inputs={}))
    third = executor.submit_frame(SimpleNamespace(request_id="third", inputs={}))

    with pytest.raises(VisualFrameSupersededError):
        second.result(timeout=1.0)
    release.set()
    assert first.result(timeout=1.0) == 1
    assert third.result(timeout=1.0) == 2
    assert calls == ["first", "third"]
    executor.close()


def test_staged_visual_unknown_calls_error_handler_with_started_operation():
    executor = object.__new__(StagedModelExecutor)
    executor._execution_plan = None
    executor._scheduling = StagedScheduling({}, {"visual": "frame_arrival"})
    executor._stages = (
        SimpleNamespace(
            execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                BackendInferenceError(
                    "async completion is uncertain",
                    code="async_execution_uncertain",
                    operation_started=True,
                    outcome_known=False,
                )
            )
        ),
    )
    executor._action_start = 1
    executor._generation = 1
    executor._snapshot_condition = threading.Condition(threading.RLock())
    executor._snapshot = None
    executor._snapshot_version = 0
    executor._identity = _identity()
    failures: list[tuple[Exception, bool]] = []
    executor._error_handler = lambda exc, started: failures.append((exc, started))

    request = SimpleNamespace(request_id="visual-unknown", inputs={})
    with pytest.raises(BackendInferenceError, match="async completion"):
        executor._run_visual(request, ExecutionContext(request.request_id), 1)
    assert len(failures) == 1
    assert failures[0][1] is True


def test_staged_snapshot_rejects_stale_result(monkeypatch):
    executor = object.__new__(StagedModelExecutor)
    executor._snapshot_condition = threading.Condition(threading.RLock())
    executor._generation = 2
    executor._snapshot = type(
        "Snapshot",
        (),
        {"generation": 2, "observation_mono_ns": 1, "values": {"_selected_prompt": "task"}},
    )()
    executor._closed = False
    executor._resetting = False
    refreshes = []

    def refresh(*args, **kwargs):
        refreshes.append(True)
        raise RuntimeError("refresh requested")

    executor.submit_frame = refresh
    executor.frame_submitter = None
    executor._max_snapshot_age_ns = 5_000_000_000
    monkeypatch.setattr("inference_service.pipeline.staged_executor.time.monotonic_ns", lambda: 6_000_000_002)

    with pytest.raises(RuntimeError, match="refresh requested"):
        StagedModelExecutor._wait_matching_snapshot(executor, SimpleNamespace(request_id="stale"), "task", None)
    assert refreshes == [True]


def test_staged_partition_uses_configured_triggers_not_pi05_role_names():
    producer = ModelStage("producer", object())
    terminal = ModelStage("terminal", object())
    executor = object.__new__(StagedModelExecutor)
    executor._stages = (producer, terminal)
    executor._scheduling = StagedScheduling(
        priorities={},
        triggers={"producer": "frame_arrival", "terminal": "dispatch"},
    )

    assert StagedModelExecutor._partition_stages(executor) == 1
