"""Tracing observes the accepted owner without adding a second lifecycle."""

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_legacy_active_plan import deliver, make_node

from action_dispatch import action_dispatcher_node as dispatcher
from action_dispatch.chunk_planning import ChunkPlan
from action_dispatch.executors.completion import CompletionStatus, ExecutionCompletion
from ibrobot_tracing.instrumentation import TraceEmitter


class _ObservedLock:
    def __init__(self):
        self.lock = threading.RLock()
        self.contended = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.contended.set()
            self.lock.acquire()
        return self

    def __exit__(self, *_args):
        self.lock.release()


@pytest.fixture
def trace_records(monkeypatch):
    records = []
    logger = Mock()
    logger.info.side_effect = lambda _format, _prefix, payload: records.append(json.loads(payload))
    monkeypatch.setattr(dispatcher, "trace", TraceEmitter(logger, enabled=True))
    return records


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize(
    "mode,indices", [("first", [0]), ("all", [0, 1, 2]), ("sample:2", [0, 2]), ("sample:bad", [0])]
)
def test_first_and_sampling_use_consumption_not_chunk_position(monkeypatch, trace_records, smoothing, mode, indices):
    node = make_node(smoothing=smoothing)
    node._trace_action_steps = mode
    node._chunk_planner = SimpleNamespace(plan=lambda actions, **kw: ChunkPlan(actions, 1, 4, 0))
    deliver(node, monkeypatch, {"action": np.arange(12).reshape(6, 2)}, request="accepted")
    node._current_request_id = "newer-request"

    for _ in range(4):
        node._control_loop()

    metadata = [call.args[1] for call in node._executor.execute.call_args_list]
    assert [item["execute_index"] for item in metadata] == ([-1] * 4 if smoothing else [1, 2, 3, 4])
    assert all(set(item) == {"request_id", "execute_index", "queue_size"} for item in metadata)
    assert all(item["request_id"] == "accepted" for item in metadata)
    assert node._active_plan.snapshot().consumed == 3

    executions = [r for r in trace_records if r["event"] in ("first_action_execute", "action_execute")]
    assert [r["fields"]["consumed_index"] for r in executions] == indices
    assert [r["event"] for r in executions].count("first_action_execute") == 1
    assert executions[0]["fields"]["chunk_position"] == (None if smoothing else 1)
    assert all(r["fields"]["request_id"] == "accepted" for r in executions)
    assert executions[0]["fields"]["queue_before"] == 3
    assert executions[0]["fields"]["queue_after"] == 2
    flows = [(r["event"], r["fields"]["edge_id"]) for r in trace_records if r["event"].startswith("flow_")]
    assert flows == [
        ("flow_receive", "result_to_decode"),
        ("flow_send", "decode_to_queue"),
        ("flow_receive", "decode_to_queue"),
        ("flow_send", "queue_to_execute"),
        ("flow_receive", "queue_to_execute"),
    ]


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("hold", [False, True])
def test_zero_consumed_empty_interval_never_traces_first(monkeypatch, trace_records, smoothing, hold):
    node = make_node(smoothing=smoothing)
    node._trace_action_steps = "all"
    node._last_action = np.zeros(2) if hold else None
    node._chunk_planner = SimpleNamespace(plan=lambda actions, **kw: ChunkPlan(actions, 4, 4, 0))
    deliver(node, monkeypatch, {"action": np.ones((6, 2))})
    before = node._active_plan.snapshot()
    node._control_loop()
    assert node._active_plan.snapshot() == before
    assert before.consumed == before.remaining == 0
    assert node._executor.execute.call_count == int(hold)
    if hold:
        assert set(node._executor.execute.call_args.args[1]) == {"request_id", "execute_index", "queue_size"}
    assert not any(r["event"] in ("first_action_execute", "action_execute") for r in trace_records)
    assert not any(r["fields"].get("edge_id") == "queue_to_execute" for r in trace_records)


@pytest.mark.parametrize("invalidate", [False, True])
def test_decode_retains_admission_and_rechecks_source(monkeypatch, trace_records, invalidate):
    node = make_node()
    node._inflight_request_id = "decoding"
    node._inference_in_progress = True

    def decode(_value):
        assert node._inflight_request_id == "decoding"
        assert node._inference_in_progress
        node._control_loop()
        node._request_inference.assert_not_called()
        if invalidate:
            node._reset_cb(None, SimpleNamespace())
        return {"action": np.ones((6, 2))}

    monkeypatch.setattr(dispatcher.TensorMsgConverter, "from_variant", decode)
    future = Future()
    future.set_result(
        SimpleNamespace(
            result=SimpleNamespace(
                success=True, action_chunk=None, inference_latency_ms=1, backend_latency_ms=1, execution_horizon=0
            )
        )
    )
    node._result_cb(future, "decoding", 0, 0)
    assert not node._inference_in_progress
    assert node._dispatch_count == int(not invalidate)
    assert node._active_plan.snapshot().remaining == (0 if invalidate else 6)
    assert sum(r["event"] == "queue_refill" for r in trace_records) == int(not invalidate)


@pytest.mark.parametrize("smoothing", [False, True])
def test_benchmark_traces_only_matching_commit(monkeypatch, trace_records, smoothing):
    node = make_node(benchmark=True, smoothing=smoothing)
    node._chunk_planner = SimpleNamespace(plan=lambda actions, **kw: ChunkPlan(actions, 1, 4, 0))
    deliver(node, monkeypatch, {"action": np.ones((6, 2))}, request="benchmark")
    assert not any(r["fields"].get("edge_id") == "queue_to_execute" for r in trace_records)
    trace_records.clear()
    before = node._active_plan.snapshot()
    node._submit_next_action_wait_for_feedback()
    assert node._active_plan.snapshot() == before
    assert trace_records == []
    completion = ExecutionCompletion(node._reservation_context.correlation_id, CompletionStatus.COMPLETED, 200, 7, 0)
    transition = node._scheduler.on_completion(completion)
    node._apply_completion_transition_wait_for_feedback(transition, completion)
    node._apply_completion_transition_wait_for_feedback(transition, completion)
    assert [r["event"] for r in trace_records] == ["action_commit"]
    fields = trace_records[0]["fields"]
    assert fields["request_id"] == "benchmark"
    assert fields["execute_index"] == (-1 if smoothing else 1)
    assert "chunk_position" not in node._executor.submit.call_args.args[1].metadata
    assert (fields["queue_before"], fields["queue_after"]) == (3, 2)
    assert node._active_plan.snapshot().consumed == node._episode.committed_count == 1


@pytest.mark.parametrize("smoothing", [False, True])
def test_refill_cannot_steal_inflight_publication_identity(monkeypatch, trace_records, smoothing):
    node = make_node(smoothing=smoothing)
    node._dispatch_lock = _ObservedLock()
    node._chunk_planner = SimpleNamespace(plan=lambda actions, **kw: ChunkPlan(actions, 1, 4, 0))
    deliver(node, monkeypatch, {"action": np.ones((6, 2))}, request="old")
    entered, release = threading.Event(), threading.Event()

    def publish(*_args):
        entered.set()
        assert release.wait(5)

    node._executor.execute.side_effect = publish
    node._inflight_request_id = "new"
    node._inference_in_progress = True
    future = Future()
    future.set_result(
        SimpleNamespace(
            result=SimpleNamespace(
                success=True,
                action_chunk={"action": np.ones((6, 2))},
                inference_latency_ms=1,
                backend_latency_ms=1,
                execution_horizon=0,
            )
        )
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        tick = pool.submit(node._control_loop)
        try:
            assert entered.wait(5)
            refill = pool.submit(node._result_cb, future, "new", 0, 0)
            assert node._dispatch_lock.contended.wait(5)
            assert node._active_plan.snapshot().source.request_id == "old"
        finally:
            release.set()
        tick.result(timeout=5)
        refill.result(timeout=5)

    assert node._executor.execute.call_args.args[1]["request_id"] == "old"
    assert node._active_plan.snapshot().source.request_id == "new"
    assert node._active_plan.snapshot().consumed == 0
    node._control_loop()
    metadata = node._executor.execute.call_args.args[1]
    assert metadata["request_id"] == "new" and metadata["execute_index"] == (-1 if smoothing else 1)
    first = [r for r in trace_records if r["event"] == "first_action_execute"]
    assert [r["fields"]["request_id"] for r in first] == ["old", "new"]


@pytest.mark.parametrize("smoothing", [False, True])
def test_trace_on_off_preserves_executor_arguments_and_plan(monkeypatch, smoothing):
    outcomes = []
    for enabled in (False, True):
        emitter = TraceEmitter(Mock(), enabled=enabled)
        monkeypatch.setattr(dispatcher, "trace", emitter)
        node = make_node(smoothing=smoothing)
        node._trace_action_steps = "all"
        node._chunk_planner = SimpleNamespace(plan=lambda actions, **kw: ChunkPlan(actions, 1, 4, 0))
        lock_observations = []
        emitter.logger.info.side_effect = lambda *_args, observed=lock_observations, node=node: observed.append(
            node._dispatch_lock._is_owned()
        )
        deliver(node, monkeypatch, {"action": np.arange(12).reshape(6, 2)}, request="same")
        for _ in range(4):
            node._control_loop()
        calls = [(call.args[0].tolist(), call.args[1], call.kwargs) for call in node._executor.execute.call_args_list]
        outcomes.append((calls, node._active_plan.snapshot(), node._dispatch_count, node._hold_count))
        assert not any(lock_observations)
    assert outcomes[0] == outcomes[1]


def test_trace_off_does_not_create_trace_uuid_or_extra_snapshot(monkeypatch):
    node = make_node()
    deliver(node, monkeypatch, {"action": np.ones((6, 2))})
    monkeypatch.setattr(dispatcher, "trace", TraceEmitter(Mock(), enabled=False))
    monkeypatch.setattr(dispatcher.uuid, "uuid4", lambda: pytest.fail("trace allocated UUID"))
    snapshot = Mock(wraps=node._active_plan.snapshot)
    monkeypatch.setattr(node._active_plan, "snapshot", snapshot)
    node._control_loop()
    # Two control snapshots, one inside reserve, and the pre-existing queue_after read.
    assert snapshot.call_count == 4
    assert set(node._executor.execute.call_args.args[1]) == {"request_id", "execute_index", "queue_size"}


@pytest.mark.parametrize("tracing", [False, True])
def test_immediate_refill_keeps_pre_request_queue_size_and_publish_order(monkeypatch, tracing):
    node = make_node()
    emitter = TraceEmitter(Mock(), enabled=tracing)
    monkeypatch.setattr(dispatcher, "trace", emitter)
    lock_observations = []
    emitter.logger.info.side_effect = lambda *_args: lock_observations.append(node._dispatch_lock._is_owned())
    deliver(node, monkeypatch, {"action": np.ones((1, 2))}, request="old")
    order = []
    node._queue_size_pub.publish.side_effect = lambda message: order.append(("queue", message.data))

    def request():
        order.append(("request",))
        deliver(node, monkeypatch, {"action": np.tile([7.0, 8.0], (3, 1))}, request="new")

    node._request_inference.side_effect = request
    node._executor.execute.side_effect = lambda action, metadata: order.append(("execute", action.tolist(), metadata))
    node._control_loop()
    assert order == [
        ("queue", 1),
        ("request",),
        ("execute", [7.0, 8.0], {"request_id": "new", "execute_index": 0, "queue_size": 1}),
    ]
    assert node._active_plan.snapshot().remaining == 2
    assert not any(lock_observations)
