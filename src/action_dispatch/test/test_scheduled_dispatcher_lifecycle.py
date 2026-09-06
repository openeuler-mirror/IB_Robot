"""Scheduled dispatcher lifecycle race regressions."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from action_dispatch import scheduled_action_dispatcher_node as dispatcher_module
from action_dispatch.active_plan import ActivePlan, PlanSource
from action_dispatch.chunk_planning import ChunkPlan, create_chunk_planner
from action_dispatch.scheduled_action_dispatcher_node import (
    DispatcherState,
    ScheduledActionDispatcherNode,
)
from action_dispatch.temporal_smoother import TemporalSmootherManager
from robot_config.dispatch_strategies import DispatchStrategyError, resolve_dispatch_strategies


def _plan_node(*, smoothing=False, capacity=4):
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._lifecycle_lock = threading.RLock()
    node._pending_failure = None
    node._state = DispatcherState.ACTIVE
    node._session_id = "session"
    node._session_generation = 3
    node._received_results = set()
    node._inflight_request_id = "request"
    node._inflight_goal_handle = None
    node._plan_length_at_inference_start = 0
    node._smoothing_enabled = smoothing
    node._strategy_selection = resolve_dispatch_strategies(
        blending="temporal_ensemble" if smoothing else "none", entrypoint="scheduled"
    )
    node.set_parameters_atomically = Mock(return_value=SimpleNamespace(successful=True))
    node._watermark = 2
    node._smoother = TemporalSmootherManager(enabled=smoothing, device="cpu")
    node._queue_plan = ActivePlan(capacity=capacity, watermark=2, overflow="fail_closed")
    node._smoothed_plan = ActivePlan(capacity=capacity, watermark=2, smoother=node._smoother)
    node._chunk_planner = create_chunk_planner("full_chunk")
    node._safe_stop_plan = SimpleNamespace(total_positions=2)
    node._last_action = None
    node._executor = Mock()
    node._smoothing_pub = Mock()
    node._queue_size_pub = Mock()
    node.get_logger = Mock()
    return node


def _source(node):
    return PlanSource(
        node._inflight_request_id, session_id=node._session_id, session_generation=node._session_generation
    )


def _result(node, actions):
    from tensormsg.converter import TensorMsgConverter

    return SimpleNamespace(
        request_id=node._inflight_request_id,
        session_id=node._session_id,
        session_generation=node._session_generation,
        success=True,
        chunk_size=len(actions),
        action_chunk=TensorMsgConverter.to_variant({"action": actions}),
        execution_horizon=0,
    )


class _Future:
    def __init__(self, value) -> None:
        self._value = value

    def result(self):
        return self._value


class _PendingFuture:
    def add_done_callback(self, callback) -> None:
        self.callback = callback


def test_joint_snapshot_uses_local_monotonic_receive_time(monkeypatch):
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._safe_stop_plan = SimpleNamespace(joint_order=["1", "2"])
    node._joint_snapshot = None
    monkeypatch.setattr(dispatcher_module.time, "monotonic_ns", lambda: 123456789)
    message = SimpleNamespace(
        name=["2", "1"],
        position=[2.0, 1.0],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=9_999_999_999, nanosec=0)),
    )

    ScheduledActionDispatcherNode._joint_cb(node, message)

    assert node._joint_snapshot.valid
    assert node._joint_snapshot.positions == [1.0, 2.0]
    assert node._joint_snapshot.received_monotonic_ns == 123456789


def test_late_open_success_while_closing_triggers_compensating_close():
    session_id = "00112233-4455-4677-8899-aabbccddeeff"
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.CLOSING
    node._session_id = session_id
    node._session_generation = 0
    node._pending_open_session_id = session_id
    node._pending_open_completion = threading.Event()
    node._close_after_open = True
    node._inference_pipeline = "policy"
    node._inference_fallback_chain = []
    node._inference_priority = 0
    close_calls: list[int] = []
    node._begin_close_session = lambda **kwargs: close_calls.append(kwargs["expected_identity"][1])
    result = SimpleNamespace(
        success=True,
        session_id=session_id,
        session_generation=7,
    )

    ScheduledActionDispatcherNode._open_result_callback(
        node,
        _Future(SimpleNamespace(result=result)),
        session_id,
        None,
    )

    assert node._session_generation == 7
    assert node._state == DispatcherState.CLOSING
    assert node._pending_open_completion.is_set()
    assert close_calls == [7]


def test_open_sends_only_logical_session_identity_and_deadline():
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.STOPPED
    node._received_results = set()
    node._queue_plan = ActivePlan(capacity=4, watermark=2)
    node._smoothed_plan = ActivePlan(capacity=4, watermark=2)
    node._inference_pipeline = "policy"
    node._inference_fallback_chain = ["backup"]
    node._inference_priority = 4
    node._default_open_timeout_ns = 1_000_000_000
    goals = []
    node._open_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda goal: goals.append(goal) or _PendingFuture(),
    )

    ScheduledActionDispatcherNode._open_new_session(node)

    assert len(goals) == 1
    assert goals[0].session_id
    assert goals[0].deadline.sec > 0


def test_shutdown_wait_can_drive_action_callbacks_with_spin_once():
    completed = threading.Event()
    spin_calls = 0

    def spin_once(*, timeout_sec):
        nonlocal spin_calls
        assert timeout_sec > 0
        spin_calls += 1
        completed.set()

    assert ScheduledActionDispatcherNode._wait_for_event(completed, 1_000_000_000, spin_once=spin_once)
    assert spin_calls == 1


def test_control_loop_uses_smoothed_plan_length_for_waterline_and_publishes_size():
    node = _plan_node(smoothing=True)
    node._active_plan.accept(ChunkPlan(np.zeros((3, 2)), replenishment_watermark=3), _source(node))
    node._inflight_request_id = ""
    published: list[int] = []
    dispatches: list[bool] = []
    node._queue_size_pub = SimpleNamespace(publish=lambda message: published.append(message.data))
    node._request_dispatch = lambda: dispatches.append(True)
    node._execute_next_action = lambda: None

    ScheduledActionDispatcherNode._control_loop(node)

    assert published == [3]
    assert dispatches == [True]


def test_enqueue_chunk_skips_actions_executed_during_inference():
    from tensormsg.converter import TensorMsgConverter

    node = _plan_node()
    node._plan_length_at_inference_start = 2
    action_chunk = TensorMsgConverter.to_variant(
        {"action": np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])}
    )

    ScheduledActionDispatcherNode._enqueue_chunk(
        node,
        action_chunk,
        reported_chunk_size=4,
        source=_source(node),
    )

    assert node._active_plan.snapshot().next_position == 2
    np.testing.assert_array_equal(node._active_plan.take_action().action, [4, 5])
    np.testing.assert_array_equal(node._active_plan.take_action().action, [6, 7])


def test_enqueue_chunk_applies_policy_execution_horizon():
    node = _plan_node()
    node._chunk_planner = create_chunk_planner("auto_horizon")
    result = _result(node, np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    result.execution_horizon = 2

    node._on_dispatch_result(result)

    snapshot = node._active_plan.snapshot()
    # The auto_horizon strategy truncates the executable plan to the horizon
    # prefix and drops the plan watermark to 0 (consume-then-replan).
    assert snapshot.remaining == 2
    assert snapshot.watermark == 0
    assert np.array_equal(np.asarray(list(node._queue_plan._queue)), np.asarray([[0.0, 1.0], [2.0, 3.0]]))


def test_enqueue_chunk_full_chunk_strategy_ignores_execution_horizon():
    node = _plan_node()
    node._chunk_planner = create_chunk_planner("full_chunk")
    result = _result(node, np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    result.execution_horizon = 2

    node._on_dispatch_result(result)

    snapshot = node._active_plan.snapshot()
    # full_chunk keeps the complete chunk and the configured watermark behavior.
    assert snapshot.remaining == 4
    assert snapshot.watermark == node._watermark
    assert np.array_equal(
        np.asarray(list(node._queue_plan._queue)),
        np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
    )


def test_enqueue_chunk_accepts_expired_horizon_as_empty_plan():
    """A fully expired prefix (S > H) is accepted, not treated as invalid.

    Watermark prefetch after a horizon=0 fallback can consume more actions
    than the next result's prefix. The scheduled path must accept the
    normalized empty plan (clearing the old one) instead of failing the
    result and safe-stopping the session.
    """
    node = _plan_node()
    node._chunk_planner = create_chunk_planner("auto_horizon")
    node._active_plan.accept(ChunkPlan(np.zeros((1, 2))), _source(node))
    node._plan_length_at_inference_start = 4  # 3 consumed + 1 remaining -> S=3
    result = _result(node, np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    result.execution_horizon = 2  # H=2 < S=3: fully expired prefix

    node._on_dispatch_result(result)

    snapshot = node._active_plan.snapshot()
    assert snapshot.remaining == 0
    assert snapshot.watermark == 0
    assert node._inflight_request_id == ""
    assert node._pending_failure is None
    # The shared watermark rule replenishes immediately on the next tick.
    assert ScheduledActionDispatcherNode._current_plan_length_locked(node) <= snapshot.watermark


def test_enqueue_chunk_expired_horizon_with_smoother_accepts_empty_plan():
    node = _plan_node(smoothing=True)
    node._chunk_planner = create_chunk_planner("auto_horizon")
    node._active_plan.accept(ChunkPlan(np.zeros((1, 2))), _source(node))
    node._plan_length_at_inference_start = 4
    result = _result(node, np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    result.execution_horizon = 2

    node._on_dispatch_result(result)

    snapshot = node._active_plan.snapshot()
    assert snapshot.remaining == 0
    assert snapshot.watermark == 0
    assert node._pending_failure is None


def test_enqueue_chunk_rejects_overflow_without_truncating_actions():
    from tensormsg.converter import TensorMsgConverter

    node = _plan_node(capacity=2)
    node._active_plan.accept(ChunkPlan(np.array([[9.0, 9.0]])), PlanSource("old"))
    before = node._active_plan.snapshot()
    action_chunk = TensorMsgConverter.to_variant({"action": np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])})

    with pytest.raises(ValueError, match="queue capacity"):
        ScheduledActionDispatcherNode._enqueue_chunk(
            node,
            action_chunk,
            reported_chunk_size=3,
            source=_source(node),
        )

    assert node._active_plan.snapshot() == before
    np.testing.assert_array_equal(node._active_plan.take_action().action, [9, 9])


def test_enqueue_chunk_rejects_reported_chunk_size_mismatch():
    from tensormsg.converter import TensorMsgConverter

    node = _plan_node()
    action_chunk = TensorMsgConverter.to_variant({"action": np.asarray([[0.0, 1.0], [2.0, 3.0]])})

    with pytest.raises(ValueError, match="reported chunk_size=1"):
        ScheduledActionDispatcherNode._enqueue_chunk(
            node,
            action_chunk,
            reported_chunk_size=1,
            source=_source(node),
        )

    assert node._active_plan.snapshot().remaining == 0


def test_smoothed_empty_plan_holds_last_action():
    node = _plan_node(smoothing=True)
    node._last_action = np.asarray([1.0, 2.0])
    executed: list[np.ndarray] = []
    node._executor = SimpleNamespace(execute=lambda action: executed.append(action.copy()))

    ScheduledActionDispatcherNode._execute_next_action(node)

    assert len(executed) == 1
    assert np.array_equal(executed[0], np.asarray([1.0, 2.0]))


@pytest.mark.parametrize("code", ["unsupported_priority", "hardware_priority_unavailable"])
def test_backend_priority_rejections_are_never_retried(code):
    node = _plan_node()
    node._retry_max_attempts = 3
    node._retry_initial_ms = 10
    node._retry_max_ms = 100
    failures: list[str] = []
    node._fail_and_close_locked = failures.append
    node.create_timer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("unsupported backend priorities must not schedule a retry")
    )

    ScheduledActionDispatcherNode._retry_dispatch_not_started(node, "request", 0, code)

    assert failures == [f"scheduled dispatch rejected: {code}"]


def test_not_started_retry_limit_counts_retries_after_initial_request():
    node = _plan_node()
    node._retry_max_attempts = 3
    node._retry_initial_ms = 10
    node._retry_max_ms = 100
    failures: list[str] = []
    node._fail_and_close_locked = failures.append
    node.create_timer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("retry index 3 must exhaust a three-retry budget")
    )

    ScheduledActionDispatcherNode._retry_dispatch_not_started(node, "request", 3)

    assert failures == ["scheduled dispatch retries exhausted"]


def test_dispatch_retry_reuses_one_observation_snapshot():
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.ACTIVE
    node._session_id = "00112233-4455-4677-8899-aabbccddeeff"
    node._session_generation = 3
    node._inflight_request_id = ""
    node._inflight_deadline_utc_ns = 0
    node._inflight_observation_time_ns = 0
    node._inflight_goal_handle = None
    node._default_request_timeout_ns = 2_000_000_000
    node._inference_pipeline = "policy"
    node._inference_fallback_chain = ["fallback"]
    node._inference_priority = 0
    node._inference_prompt = ""
    node._current_plan_length_locked = lambda: 0
    observation_time_ns = 123_456_789_012
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=observation_time_ns))
    goals = []
    node._dispatch_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda goal: goals.append(goal) or _PendingFuture(),
    )

    ScheduledActionDispatcherNode._request_dispatch(node)
    first_request_id = node._inflight_request_id
    observation_time_ns += 99_000_000
    ScheduledActionDispatcherNode._request_dispatch(node, attempt=1, replace_request_id=first_request_id)

    assert len(goals) == 2
    first_stamp = goals[0].obs_timestamp.sec * 1_000_000_000 + goals[0].obs_timestamp.nanosec
    second_stamp = goals[1].obs_timestamp.sec * 1_000_000_000 + goals[1].obs_timestamp.nanosec
    assert first_stamp == second_stamp == 123_456_789_012
    assert goals[0].fallback_chain == goals[1].fallback_chain == ["fallback"]
    assert goals[1].deadline == goals[0].deadline
    assert goals[1].request_id != goals[0].request_id


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("toggle", [False, True])
def test_retry_retains_consumption_since_original_observation(smoothing, toggle):
    node = _plan_node(smoothing=smoothing, capacity=20)
    node._active_plan.accept(ChunkPlan(np.ones((10, 2))), PlanSource("old"))
    node._inflight_request_id = ""
    node._default_request_timeout_ns = 10_000_000_000
    node._inference_pipeline = "policy"
    node._inference_fallback_chain = []
    node._inference_priority = 0
    node._inference_prompt = ""
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123))
    node._dispatch_client = Mock()
    node._dispatch_client.send_goal_async.side_effect = lambda goal: _PendingFuture()
    node._request_dispatch()
    original = node._inflight_request_id
    for _ in range(3):
        node._execute_next_action()
    if toggle:
        # Switch to a retained store with a different remaining length.
        other = node._queue_plan if smoothing else node._smoothed_plan
        other.accept(ChunkPlan(np.ones((6, 2))), PlanSource("retained"))
        node._toggle_smoothing_cb(None, None)
    node._request_dispatch(attempt=1, replace_request_id=original)
    for _ in range(2):
        node._execute_next_action()
    planner = Mock(wraps=node._chunk_planner)
    node._chunk_planner = planner
    node._on_dispatch_result(_result(node, np.ones((10, 2))))
    assert planner.plan.call_args.kwargs["actions_executed"] == 5
    assert node._active_plan.snapshot().remaining == 5
    if not node._smoothing_enabled:
        assert node._active_plan.snapshot().next_position == 5


def test_current_expired_retry_fails_under_lifecycle_lock():
    node = _plan_node()
    node._inflight_deadline_utc_ns = 1
    node._inflight_observation_time_ns = 123
    node._failure_handling = False
    node._safe_stop = Mock(return_value=True)

    def close():
        assert node._lifecycle_lock._is_owned()
        assert node._state is DispatcherState.CLOSING
        assert node._inflight_request_id == "request"
        node._clear_inflight_locked()
        return True

    node._close_session_sync = Mock(side_effect=close)
    node._request_dispatch(attempt=1, replace_request_id="request")
    assert node._state is DispatcherState.FAILED
    node._safe_stop.assert_called_once()
    node._close_session_sync.assert_called_once()


def test_expired_retry_cannot_close_restarted_session():
    node = _plan_node()
    node._inflight_deadline_utc_ns = 1
    node._inflight_observation_time_ns = 123
    node._default_open_timeout_ns = 1_000_000_000
    entered, release = threading.Event(), threading.Event()
    fail = node._fail_current_request
    errors = []

    def paused_failure(*args):
        entered.set()
        assert release.wait(5)
        fail(*args)

    def retry():
        try:
            node._request_dispatch(attempt=1, replace_request_id="request")
        except Exception as exc:
            errors.append(exc)

    def stop():
        with node._state_lock:
            node._clear_inflight_locked()
            node._state = DispatcherState.CLOSING
        return True

    def open_session(*, completion):
        with node._state_lock:
            node._session_id = "new-session"
            node._session_generation = 4
            node._inflight_request_id = "new-request"
            node._state = DispatcherState.ACTIVE
        completion.set()

    node._fail_current_request = paused_failure
    node._safe_stop = stop
    node._close_session_sync = Mock(return_value=True)
    node._open_new_session = open_session
    node._fail_and_close_locked = Mock()
    worker = threading.Thread(target=retry)
    worker.start()
    try:
        assert entered.wait(5)
        assert node._inflight_request_id == "request"
        response = node._restart_cb(None, SimpleNamespace())
        assert response.success
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    assert node._state is DispatcherState.ACTIVE
    assert (node._session_id, node._session_generation, node._inflight_request_id) == ("new-session", 4, "new-request")
    node._close_session_sync.assert_called_once()
    node._fail_and_close_locked.assert_not_called()


@pytest.mark.parametrize("failure", ["rejected", "unknown", "deadline", "decode"])
def test_stale_failure_leaves_two_worker_executor_free_for_close(monkeypatch, failure):
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.task import Future

    context = Context()
    context.init()
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    node = _plan_node()
    node._safe_stop = Mock(return_value=True)  # No robot publishers or hardware.
    node._pending_open_session_id = ""
    node._close_in_progress = False
    node._default_request_timeout_ns = 3_000_000_000
    closing, stale_entered = threading.Event(), threading.Event()
    resume = threading.Event()
    current = node._request_is_current

    def checked(*args):
        value = current(*args)
        if value:
            stale_entered.set()
            assert resume.wait(5)
        return value

    result = _result(node, np.ones((1, 2)))
    if failure == "decode":
        from tensormsg.converter import TensorMsgConverter

        def decode(message):
            stale_entered.set()
            assert resume.wait(5)
            raise ValueError("decode failure")

        monkeypatch.setattr(TensorMsgConverter, "from_variant", decode)
    else:
        node._request_is_current = checked
    goal_future = Future(executor=executor)
    result_future = Future(executor=executor)
    node._close_client = Mock()

    def send_close(goal):
        closing.set()
        return goal_future

    node._close_client.send_goal_async.side_effect = send_close

    def stale():
        if failure == "rejected":
            node._dispatch_rejected("request")
        elif failure == "unknown":
            node._on_dispatch_unknown("request", "session", 3)
        elif failure == "deadline":
            node._inflight_deadline_utc_ns = 1
            node._inflight_observation_time_ns = 123
            node._request_dispatch(attempt=1, replace_request_id="request")
        else:
            node._on_dispatch_result(result)

    spinner = threading.Thread(target=executor.spin)
    spinner.start()
    try:
        stale_task = executor.create_task(stale)
        assert stale_entered.wait(5)
        stop_task = executor.create_task(node._stop_cb, None, SimpleNamespace())
        assert closing.wait(5)
        resume.set()
        # Both action-client callbacks run on the real, bounded ROS executor.
        goal_future.set_result(SimpleNamespace(accepted=True, get_result_async=lambda: result_future))
        result_future.set_result(
            SimpleNamespace(result=SimpleNamespace(success=True, session_id="session", closed_session_generation=3))
        )
        finished = threading.Event()
        stop_task.add_done_callback(lambda future: finished.set())
        assert finished.wait(5)
        assert stop_task.result().success
        assert stale_task.done()
        assert stale_task.exception() is None
        assert node._state is DispatcherState.STOPPED
    finally:
        resume.set()
        executor.shutdown(timeout_sec=5)
        spinner.join(5)
        context.shutdown()
    assert not spinner.is_alive()


@pytest.mark.parametrize("smoothing", [False, True])
def test_selected_interval_and_zero_watermark_drive_consumption(smoothing):
    node = _plan_node(smoothing=smoothing)
    node._plan_length_at_inference_start = 2

    def plan(actions, *, actions_executed, execution_horizon=None):
        assert node._state_lock._is_owned()
        assert actions_executed == 2
        return ChunkPlan(actions, executed_during_inference=2, execution_horizon=5, replenishment_watermark=0)

    node._chunk_planner = SimpleNamespace(plan=plan)
    source = _source(node)
    node._on_dispatch_result(_result(node, np.arange(12).reshape(6, 2)))
    snapshot = node._active_plan.snapshot()
    assert (snapshot.remaining, snapshot.watermark, snapshot.source) == (3, 0, source)
    assert snapshot.next_position == (None if smoothing else 2)
    node._request_dispatch = Mock()
    for _ in range(3):
        node._control_loop()
    node._request_dispatch.assert_not_called()
    assert [call.args[0].tolist() for call in node._executor.execute.call_args_list] == [[4, 5], [6, 7], [8, 9]]
    exhausted = node._active_plan.snapshot()
    node._control_loop()
    node._request_dispatch.assert_called_once()
    assert node._active_plan.snapshot() == exhausted


def test_toggle_preserves_queue_metadata_and_discards_smoothed_metadata():
    node = _plan_node()
    node._queue_plan.accept(
        ChunkPlan(
            np.arange(10).reshape(5, 2), executed_during_inference=2, execution_horizon=5, replenishment_watermark=0
        ),
        PlanSource("queue"),
    )
    node._execute_next_action()
    queue_snapshot = node._queue_plan.snapshot()
    node._toggle_smoothing_cb(None, None)
    node._active_plan.accept(ChunkPlan(np.ones((2, 2)), replenishment_watermark=1), PlanSource("smooth"))
    node._toggle_smoothing_cb(None, None)
    assert node._active_plan.snapshot() == queue_snapshot
    assert node._smoothed_plan.snapshot().source is None
    node._toggle_smoothing_cb(None, None)
    assert node._active_plan.snapshot().remaining == 0
    assert node._active_plan.snapshot().watermark == 2
    with node._state_lock:
        node._clear_plans_locked()
    node._toggle_smoothing_cb(None, None)
    assert node._active_plan.snapshot().remaining == 0
    assert node._active_plan.snapshot().source is None
    assert node._active_plan.snapshot().next_position is None
    assert node._active_plan.snapshot().watermark == 2


@pytest.mark.parametrize("smoothing", [False, True])
def test_alignment_uses_progress_after_decode_and_accepts_fully_expired_interval(monkeypatch, smoothing):
    from tensormsg.converter import TensorMsgConverter

    node = _plan_node(smoothing=smoothing)
    node._active_plan.accept(ChunkPlan(np.ones((2, 2))), PlanSource("old"))
    node._plan_length_at_inference_start = 2
    result = _result(node, np.ones((1, 2)))
    source = _source(node)
    original = TensorMsgConverter.from_variant

    def decode(message):
        value = original(message)
        node._execute_next_action()
        return value

    monkeypatch.setattr(TensorMsgConverter, "from_variant", decode)
    node._on_dispatch_result(result)
    snapshot = node._active_plan.snapshot()
    assert snapshot.remaining == 0
    assert snapshot.source == source
    assert snapshot.next_position == (None if smoothing else 1)
    node._execute_next_action()
    assert node._active_plan.snapshot() == snapshot


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("invalid", ["overflow", "nan", "inf", "dimension", "empty", "missing", "rank"])
def test_invalid_result_safe_stops_then_closes_without_candidate_metadata(monkeypatch, smoothing, invalid):
    node = _plan_node(smoothing=smoothing, capacity=2)
    if smoothing and invalid == "overflow":
        node._on_dispatch_result(_result(node, np.ones((3, 2))))
        assert node._active_plan.snapshot().remaining == 3
        assert node._state == DispatcherState.ACTIVE
        return
    node._active_plan.accept(ChunkPlan(np.ones((1, 2)), replenishment_watermark=0), PlanSource("old"))
    node._failure_handling = False
    node._joint_snapshot = SimpleNamespace(valid=False)
    node._safe_stop_plan.channels = []
    order = []
    monkeypatch.setattr(dispatcher_module, "construct_safety_command", lambda **kwargs: order.append("safe") or [])
    node._close_session_sync = lambda: order.append("close") or True
    actions = {
        "overflow": np.ones((3, 2)),
        "nan": np.array([[np.nan, 0]]),
        "inf": np.array([[np.inf, 0]]),
        "dimension": np.ones((1, 3)),
        "empty": np.empty((0, 2)),
        "missing": np.ones((1, 2)),
        "rank": np.ones((2,)),
    }[invalid]
    result = _result(node, actions)
    if invalid == "missing":
        from tensormsg.converter import TensorMsgConverter

        result.action_chunk = TensorMsgConverter.to_variant({})
    node._on_dispatch_result(result)
    assert order == ["safe", "close"]
    assert node._state == DispatcherState.FAILED
    assert not node._inflight_request_id
    assert not node._received_results
    for owner in (node._queue_plan, node._smoothed_plan):
        snapshot = owner.snapshot()
        assert (snapshot.remaining, snapshot.source, snapshot.next_position, snapshot.watermark) == (0, None, None, 2)


@pytest.mark.parametrize("smoothing", [False, True])
def test_duplicate_and_stale_result_leave_plan_and_new_request_untouched(smoothing):
    node = _plan_node(smoothing=smoothing)
    result = _result(node, np.ones((2, 2)))
    node._on_dispatch_result(result)
    node._execute_next_action()
    before = node._active_plan.snapshot()
    node._inflight_request_id = "next"
    node._on_dispatch_result(result)
    result.request_id = "stale"
    node._on_dispatch_result(result)
    assert node._active_plan.snapshot() == before
    assert node._inflight_request_id == "next"


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("transition", ["close", "open"])
def test_decoded_result_cannot_restore_plan_after_lifecycle_change(monkeypatch, smoothing, transition):
    from tensormsg.converter import TensorMsgConverter

    node = _plan_node(smoothing=smoothing)
    result = _result(node, np.ones((2, 2)))
    reservations = []
    for owner in (node._queue_plan, node._smoothed_plan):
        owner.accept(ChunkPlan(np.ones((2, 2)), replenishment_watermark=0), PlanSource("old"))
        reservations.append(owner.reserve()[0])
    decoded = threading.Event()
    resume = threading.Event()
    original = TensorMsgConverter.from_variant

    def decode(message):
        value = original(message)
        decoded.set()
        assert resume.wait(5)
        return value

    monkeypatch.setattr(TensorMsgConverter, "from_variant", decode)
    errors = []

    def receive():
        try:
            node._on_dispatch_result(result)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=receive)
    worker.start()
    try:
        assert decoded.wait(5)
        if transition == "close":
            node._close_in_progress = False
            node._default_request_timeout_ns = 1_000_000_000
            node._close_client = SimpleNamespace(wait_for_server=lambda **kwargs: False)
            node._begin_close_session()
        else:
            node._state = DispatcherState.STOPPED
            node._default_open_timeout_ns = 1_000_000_000
            node._open_client = SimpleNamespace(wait_for_server=lambda **kwargs: False)
            node._open_new_session(session_id="new-session")
        cleared = [owner.snapshot() for owner in (node._queue_plan, node._smoothed_plan)]
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    assert [owner.snapshot() for owner in (node._queue_plan, node._smoothed_plan)] == cleared
    assert all(snapshot.source is None and snapshot.remaining == 0 for snapshot in cleared)
    assert all(snapshot.watermark == 2 and snapshot.next_position is None for snapshot in cleared)
    assert not node._received_results
    for owner, reservation in zip((node._queue_plan, node._smoothed_plan), reservations, strict=True):
        assert not owner.is_current(reservation)


@pytest.mark.parametrize("state", [DispatcherState.ACTIVE, DispatcherState.CLOSING])
def test_retry_timer_cannot_replace_a_new_request_or_reopen_closed_session(state):
    node = _plan_node()
    node._retry_max_attempts = 3
    node._retry_initial_ms = 10
    node._retry_max_ms = 100
    timers = []

    def timer(_delay, callback, **kwargs):
        timers.append(callback)
        return Mock()

    node.create_timer = timer
    node._retry_dispatch_not_started("request", 0)
    node._state = state
    node._inflight_request_id = "new-request"
    timers[0]()
    assert node._inflight_request_id == "new-request"
    assert node._state == state


@pytest.mark.parametrize(
    "values, enabled, error",
    [
        ({}, False, None),
        ({"blending_strategy": "temporal_ensemble"}, True, None),
        ({"blending_strategy": "none"}, False, None),
        ({"temporal_smoothing_enabled": True}, None, "removed"),
        ({"temporal_smoothing_enabled": False}, None, "removed"),
        ({"blending_strategy": "none", "temporal_smoothing_enabled": False}, None, "removed"),
        ({"blending_strategy": "temporal_ensemble", "temporal_smoothing_enabled": True}, None, "removed"),
        ({"blending_strategy": "temporal_ensemble", "temporal_smoothing_enabled": False}, None, "removed"),
        ({"executor_type": "benchmark"}, None, "scheduled entrypoint"),
        ({"scheduler_mode": "wait_for_feedback"}, None, "scheduled entrypoint"),
        ({"executor_type": "benchmark", "scheduler_mode": "wait_for_feedback"}, None, "scheduled entrypoint"),
        ({"chunking_strategy": "adaptive"}, None, "unknown chunking"),
        ({"blending_strategy": "rtc"}, None, "unknown blending"),
        ({"temporal_smoothing_enabled": "false"}, None, "removed"),
        ({"temporal_smoothing_enabled": 0}, None, "removed"),
        ({"chunking_strategy": False}, None, "expecting type"),
        ({"scheduler_mode": 0}, None, "expecting type"),
    ],
)
def test_scheduled_real_parameter_resolution(values, enabled, error):
    import rclpy
    from rclpy.context import Context
    from rclpy.exceptions import InvalidParameterTypeException
    from rclpy.node import Node
    from rclpy.parameter import Parameter

    context = Context()
    rclpy.init(context=context)
    node = object.__new__(ScheduledActionDispatcherNode)
    Node.__init__(
        node,
        "scheduled_parameter_test",
        context=context,
        parameter_overrides=[Parameter(name, value=value) for name, value in values.items()],
        use_global_arguments=False,
    )
    try:
        if error:
            with pytest.raises((ValueError, InvalidParameterTypeException), match=error):
                node._load_parameters()
        else:
            node._load_parameters()
            assert node._smoothing_enabled is enabled
            assert (node._strategy_selection.blending == "temporal_ensemble") is enabled
            assert node._strategy_selection.executor_type == "topic"
            assert node._strategy_selection.scheduler_mode == "continuous"
    finally:
        Node.destroy_node(node)
        context.shutdown()


@pytest.mark.parametrize("smoothing", [False, True])
def test_rejected_toggle_leaves_selection_stores_and_publication_unchanged(monkeypatch, smoothing):
    node = _plan_node(smoothing=smoothing)
    for owner in (node._queue_plan, node._smoothed_plan):
        owner.accept(ChunkPlan(np.ones((2, 2)), replenishment_watermark=0), PlanSource("old"))
    before = [owner.snapshot() for owner in (node._queue_plan, node._smoothed_plan)]
    selection = node._strategy_selection

    def reject(**kwargs):
        assert node._state_lock._is_owned()
        assert kwargs["entrypoint"] == "scheduled"
        assert kwargs["blending"] == ("none" if smoothing else "temporal_ensemble")
        raise DispatchStrategyError("unsupported proposed combination")

    monkeypatch.setattr(dispatcher_module, "resolve_dispatch_strategies", reject)
    response = SimpleNamespace()
    assert node._toggle_smoothing_cb(None, response) is response
    assert node._strategy_selection is selection
    assert node._smoothing_enabled is smoothing
    assert node._smoother.is_enabled is smoothing
    assert [owner.snapshot() for owner in (node._queue_plan, node._smoothed_plan)] == before
    node._smoothing_pub.publish.assert_not_called()
    node.get_logger().error.assert_called_once()


def test_publication_finishes_before_lifecycle_can_freeze():
    node = _plan_node()
    node._active_plan.accept(ChunkPlan(np.ones((2, 2))), PlanSource("old"))
    entered, release, stopping = threading.Event(), threading.Event(), threading.Event()
    order = []

    def execute(action):
        assert node._state_lock._is_owned()
        entered.set()
        assert release.wait(5)
        order.append("action")

    def stop():
        stopping.set()
        with node._state_lock:
            node._state = DispatcherState.CLOSING
            node._clear_plans_locked()
            order.append("freeze")

    node._executor.execute.side_effect = execute
    tick = threading.Thread(target=node._execute_next_action)
    stopper = threading.Thread(target=stop)
    tick.start()
    try:
        assert entered.wait(5)
        stopper.start()
        assert stopping.wait(5)
    finally:
        release.set()
        tick.join(5)
        if stopper.ident is not None:
            stopper.join(5)
    assert not tick.is_alive() and not stopper.is_alive()
    assert order == ["action", "freeze"]
    node._execute_next_action()
    assert node._executor.execute.call_count == 1


@pytest.mark.parametrize("replace", [False, True])
def test_busy_failure_is_retained_and_rechecked_on_timer(replace):
    node = _plan_node()
    source = _source(node)
    entered = threading.Event()
    node._fail_and_close_locked = Mock()
    lifecycle_lock = node._lifecycle_lock

    def fail():
        node._fail_current_request(source.request_id, source.session_id, source.session_generation, "late failure")
        entered.set()

    with lifecycle_lock:
        worker = threading.Thread(target=fail)
        worker.start()
        assert entered.wait(5)
        assert node._pending_failure == ("request", "session", 3, "late failure")
        if replace:
            with node._state_lock:
                node._session_id = "replacement"
                node._inflight_request_id = "new"
    worker.join(5)
    assert not worker.is_alive()
    node._drain_pending_failure()
    assert node._pending_failure is None
    if replace:
        node._fail_and_close_locked.assert_not_called()
        assert node._state == DispatcherState.ACTIVE
        assert node._inflight_request_id == "new"
    else:
        node._fail_and_close_locked.assert_called_once_with("late failure")
        assert node._state == DispatcherState.CLOSING


@pytest.mark.parametrize("smoothing", [False, True])
def test_toggle_during_request_preserves_consumption_alignment(smoothing):
    node = _plan_node(smoothing=smoothing, capacity=6)
    node._active_plan.accept(ChunkPlan(np.ones((5, 2))), PlanSource("old"))
    node._plan_length_at_inference_start = 5
    node._execute_next_action()
    node._execute_next_action()
    node._toggle_smoothing_cb(None, None)
    node._on_dispatch_result(_result(node, np.arange(12).reshape(6, 2)))
    assert node._active_plan.snapshot().remaining == 4
    node._execute_next_action()
    np.testing.assert_array_equal(node._executor.execute.call_args.args[0], [4, 5])


@pytest.mark.parametrize("failure", ["request", "open", "stop"])
@pytest.mark.parametrize("service", ["start", "stop", "restart"])
def test_real_timer_eventually_closes_busy_failure(failure, service):
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.task import Future
    from std_srvs.srv import Trigger

    context = Context()
    context.init()
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    timer_node = Node("failure_drain_test", context=context)
    node = _plan_node()
    node._failure_handling = False
    node._pending_open_session_id = ""
    node._close_in_progress = False
    node._default_request_timeout_ns = 3_000_000_000
    node._safe_stop = Mock(return_value=True)
    node._close_client = Mock()
    goal_future, result_future = Future(executor=executor), Future(executor=executor)
    closing, retried, finished = threading.Event(), threading.Event(), threading.Event()

    def send_close(goal):
        closing.set()
        return goal_future

    def drain():
        node._drain_pending_failure()
        retried.set()
        if node._state is DispatcherState.FAILED:
            finished.set()

    node._close_client.send_goal_async.side_effect = send_close
    timer_node.create_timer(0.01, drain, callback_group=MutuallyExclusiveCallbackGroup())
    executor.add_node(timer_node)
    spinner = threading.Thread(target=executor.spin)
    try:
        with node._lifecycle_lock:
            spinner.start()
            if failure == "stop":
                task = None
            elif failure == "request":
                task = executor.create_task(node._fail_current_request, "request", "session", 3, "failure")
            else:
                task = executor.create_task(node._fail_and_close, "failure", session_id="session")
            if task is not None:
                returned = threading.Event()
                task.add_done_callback(lambda future: returned.set())
                assert returned.wait(5)
                assert task.exception() is None
                retried.clear()
                assert retried.wait(5)
                assert node._pending_failure is not None
            assert not closing.is_set()
        if failure == "stop":
            task = executor.create_task(node._stop_cb, Trigger.Request(), Trigger.Response())
            task.add_done_callback(lambda future: finished.set())
        assert closing.wait(5)
        contender = executor.create_task(getattr(node, f"_{service}_cb"), Trigger.Request(), Trigger.Response())
        rejected = threading.Event()
        contender.add_done_callback(lambda future: rejected.set())
        assert rejected.wait(1)
        assert not contender.result().success
        assert contender.result().message == "lifecycle operation in progress"
        assert not finished.is_set()
        goal_future.set_result(SimpleNamespace(accepted=True, get_result_async=lambda: result_future))
        result_future.set_result(
            SimpleNamespace(result=SimpleNamespace(success=True, session_id="session", closed_session_generation=3))
        )
        assert finished.wait(5)
        if failure == "stop":
            assert task.result().success
            assert node._state is DispatcherState.STOPPED
        else:
            assert node._state is DispatcherState.FAILED
        node._safe_stop.assert_called_once()
        node._close_client.send_goal_async.assert_called_once()
        assert node._pending_failure is None
        assert node._session_id == ""
    finally:
        executor.shutdown(timeout_sec=5)
        spinner.join(5)
        timer_node.destroy_node()
        context.shutdown()
    assert not spinner.is_alive()


def test_production_failure_timer_closes_open_exception_with_frozen_ros_clock(monkeypatch):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.task import Future
    from rosgraph_msgs.msg import Clock

    queued, closing, finished = threading.Event(), threading.Event(), threading.Event()
    order = []
    original_failure = ScheduledActionDispatcherNode._fail_and_close
    original_drain = ScheduledActionDispatcherNode._drain_pending_failure

    def fail(node, *args, **kwargs):
        original_failure(node, *args, **kwargs)
        queued.set()

    def drain(node):
        original_drain(node)
        if node._state is DispatcherState.FAILED:
            finished.set()

    monkeypatch.setattr(
        ScheduledActionDispatcherNode, "_load_contract_and_plan", lambda node: setattr(node, "_action_specs", [])
    )
    monkeypatch.setattr(ScheduledActionDispatcherNode, "_fail_and_close", fail)
    monkeypatch.setattr(ScheduledActionDispatcherNode, "_drain_pending_failure", drain)
    monkeypatch.setattr(dispatcher_module, "TopicExecutor", Mock())
    monkeypatch.setattr(dispatcher_module.rclpy.action, "ActionClient", Mock())
    # Keep services and the frozen /clock separate from other domain-42 nodes.
    rclpy.init(args=["--ros-args", "-r", "__ns:=/frozen_failure_test", "-r", "/clock:=/frozen_failure_test/clock"])
    node = ScheduledActionDispatcherNode(
        parameter_overrides=[
            Parameter("use_sim_time", value=True),
            Parameter("smoothing_device", value="cpu"),
            Parameter("joint_state_topic", value="/frozen_failure_test/joint_states"),
        ]
    )
    node._control_timer.cancel()
    node._readiness_timer.cancel()
    node._failure_timer.cancel()  # Observe the mailbox before allowing the real timer to drain it.
    node._safe_stop = Mock(side_effect=lambda: order.append("safe_stop") or True)
    node._default_request_timeout_ns = 3_000_000_000
    clock_node = Node("frozen_failure_clock_test")
    publisher = clock_node.create_publisher(Clock, "/clock", qos_profile_sensor_data)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.add_node(clock_node)
    open_goal, open_result = Future(executor=executor), Future(executor=executor)
    close_goal, close_result = Future(executor=executor), Future(executor=executor)
    # Separate clients, since the constructor's ActionClient factory is mocked.
    node._open_client = Mock()
    node._open_client.send_goal_async.return_value = open_goal
    node._close_client = Mock()

    def send_close(goal):
        order.append("close")
        closing.set()
        return close_goal

    node._close_client.send_goal_async.side_effect = send_close
    spinner = threading.Thread(target=executor.spin)
    try:
        deadline = time.monotonic() + 5
        while publisher.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "dispatcher /clock subscription not discovered"
            executor.spin_once(timeout_sec=0.01)
        stamp = Clock()
        stamp.clock.sec = 123
        publisher.publish(stamp)
        while node.get_clock().now().nanoseconds != 123_000_000_000:
            assert time.monotonic() < deadline, "dispatcher did not receive /clock"
            executor.spin_once(timeout_sec=0.01)
        assert node.get_clock().ros_time_is_active
        with node._lifecycle_lock:
            node._state = DispatcherState.STOPPED
            completion = threading.Event()
            node._open_new_session(session_id="session", completion=completion)
            spinner.start()
            open_goal.set_result(SimpleNamespace(accepted=True, get_result_async=lambda: open_result))
            open_result.set_exception(RuntimeError("Open result transport failure"))
            assert queued.wait(5), "Open exception callback did not return under contention"
            assert completion.is_set()
            with node._state_lock:
                assert node._pending_failure == (None, "session", 0, "Open result outcome unknown")
            assert not closing.is_set()
            node._safe_stop.assert_not_called()
            node._failure_timer.reset()
        assert closing.wait(5), "failure timer stalled with frozen ROS time after lifecycle lock release"
        close_goal.set_result(SimpleNamespace(accepted=True, get_result_async=lambda: close_result))
        close_result.set_result(
            SimpleNamespace(result=SimpleNamespace(success=True, session_id="session", closed_session_generation=0))
        )
        assert finished.wait(5)
        assert node._state is DispatcherState.FAILED
        assert node._pending_failure is None
        assert node._session_id == ""
        assert node.get_clock().now().nanoseconds == 123_000_000_000
        assert order == ["safe_stop", "close"]
        node._safe_stop.assert_called_once()
        node._close_client.send_goal_async.assert_called_once()
    finally:
        executor.shutdown(timeout_sec=5)
        if spinner.ident is not None:
            spinner.join(5)
        Node.destroy_node(node)
        clock_node.destroy_node()
        rclpy.shutdown()
    assert not spinner.is_alive()


@pytest.mark.parametrize("replacement", [("new-session", 4), ("session", 4)])
def test_late_open_compensating_close_cannot_touch_replacement(replacement):
    node = _plan_node()
    node._state = DispatcherState.CLOSING
    node._session_generation = 0
    node._pending_open_session_id = "session"
    node._pending_open_completion = threading.Event()
    node._close_after_open = True
    node._close_in_progress = False
    node._close_client = Mock()
    completed, resume = threading.Event(), threading.Event()
    complete = node._complete_pending_open
    errors = []

    def pause_after_completion(*args):
        complete(*args)
        completed.set()
        assert resume.wait(5)

    def receive():
        try:
            result = SimpleNamespace(success=True, session_id="session", session_generation=3)
            node._open_result_callback(_Future(SimpleNamespace(result=result)), "session", None)
        except Exception as exc:
            errors.append(exc)

    node._complete_pending_open = pause_after_completion
    worker = threading.Thread(target=receive)
    worker.start()
    try:
        assert completed.wait(5)
        assert node._pending_open_completion.is_set()
        with node._state_lock:
            node._session_id, node._session_generation = replacement
            node._state = DispatcherState.ACTIVE
            node._inflight_request_id = "new-request"
            node._active_plan.accept(ChunkPlan(np.ones((2, 2))), _source(node))
            snapshot = node._active_plan.snapshot()
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    assert node._state is DispatcherState.ACTIVE
    assert (node._session_id, node._session_generation) == replacement
    assert node._inflight_request_id == "new-request"
    assert node._active_plan.snapshot() == snapshot
    node._close_client.send_goal_async.assert_not_called()


@pytest.mark.parametrize("request_id", ["request", None])
def test_stale_mailbox_retry_cannot_overwrite_new_valid_failure(request_id):
    node = _plan_node()
    stale = (request_id, "session", 3, "old failure")
    node._session_id = "replacement"
    node._session_generation = 4
    node._inflight_request_id = "new-request"
    valid = ("new-request", "replacement", 4, "new failure")
    node._pending_failure = valid
    node._fail_and_close_locked = Mock()
    returned = threading.Event()

    def retry():
        node._try_failure(*stale)
        returned.set()

    with node._lifecycle_lock:
        worker = threading.Thread(target=retry)
        worker.start()
        assert returned.wait(5)
        assert node._pending_failure == valid
    worker.join(5)
    assert not worker.is_alive()
    node._drain_pending_failure()
    node._fail_and_close_locked.assert_called_once_with("new failure")
    assert node._pending_failure is None
