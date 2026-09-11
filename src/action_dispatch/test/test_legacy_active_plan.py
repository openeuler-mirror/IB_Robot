"""Legacy node wiring, using real plan, smoother, scheduler and episode state."""

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from action_dispatch.action_dispatcher_node import ActionDispatcherNode, GoalExecutionContext
from action_dispatch.active_plan import ActivePlan, PlanSource
from action_dispatch.chunk_planning import ChunkPlan, FullChunkPlanner
from action_dispatch.episode import EpisodeGoalSpec, EpisodePhase, EpisodeStateMachine
from action_dispatch.executors.completion import CompletionStatus, ExecutionCompletion, ExecutionReceipt
from action_dispatch.schedulers.base import CompletionDecision, SchedulerTransition
from action_dispatch.schedulers.step_barrier import StepBarrierScheduler
from action_dispatch.temporal_smoother import TemporalSmootherManager
from robot_config.dispatch_strategies import DispatchStrategyError, resolve_dispatch_strategies


def make_node(*, smoothing=False, benchmark=False):
    node = object.__new__(ActionDispatcherNode)
    node._dispatch_lock = threading.RLock()
    node._smoother = TemporalSmootherManager() if smoothing else None
    node._smoothing_enabled = smoothing
    node._active_plan = ActivePlan(capacity=30, watermark=20, smoother=node._smoother)
    node._chunk_planner = FullChunkPlanner()
    node._action_specs = [SimpleNamespace(names=["action.0", "action.1"])]
    node._is_benchmark = benchmark
    node._is_running = True
    node._navigation_mode = False
    node._request_generation = 0
    node._inflight_request_id = ""
    node._inference_in_progress = False
    node._inference_started_at = 0.0
    node._inference_timeout_s = 10
    node._policy_reset_in_progress = False
    node._plan_length_at_inference_start = 0
    node._last_action = None
    node._reservation_context = None
    node._plan_reservation = None
    node._reserved_action = None
    node._benchmark_inference_timings = {}
    node._dispatch_count = 0
    node._total_inference_latency_ms = 0
    node._consecutive_failures = 0
    node._hold_count = 0
    node._last_queue_refill_monotonic_ns = time.monotonic_ns()
    node._last_stall_log_ns = time.monotonic_ns()
    node._last_stats_time = time.monotonic()
    node._stats_interval_s = 100
    node._scheduler_mode = "wait_for_feedback" if benchmark else "continuous"
    node._strategy_selection = resolve_dispatch_strategies(
        executor_type="benchmark" if benchmark else "topic",
        scheduler_mode=node._scheduler_mode,
        blending="temporal_ensemble" if smoothing else "none",
    )
    node._scheduler = StepBarrierScheduler(watermark=20, execution_timeout_sec=30)
    node._executor = Mock()
    node.set_parameters_atomically = Mock(return_value=SimpleNamespace(successful=True))
    node._executor.submit.side_effect = lambda action, ctx: ExecutionReceipt(ctx.correlation_id, True)
    node._queue_size_pub = Mock()
    node._smoothing_enabled_pub = Mock()
    node._request_inference = Mock()
    node._request_policy_reset = Mock()
    node.get_logger = Mock()
    node._episode = EpisodeStateMachine()
    node._active_goal_handle = None
    node._goal_contexts = {}
    if benchmark:
        node._episode.begin_preparing()
        token = node._episode.complete_prepare()
        spec = EpisodeGoalSpec(token, 7, 0, 100, 100, 60, 10, "test")
        assert node._episode.try_accept_goal(spec, time.monotonic_ns()) is None
        node._episode.bind_goal(time.monotonic_ns())
        node._active_goal_handle = Mock(is_cancel_requested=False)
        node._goal_contexts[b"goal"] = GoalExecutionContext(b"goal", node._episode.goal_generation, threading.Event())
    return node


def deliver(node, monkeypatch, payload, *, request="request"):
    node._inflight_request_id = request
    node._inference_in_progress = True
    result = SimpleNamespace(success=True, action_chunk=payload, inference_latency_ms=1, backend_latency_ms=1)
    future = Future()
    future.set_result(SimpleNamespace(result=result))
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.TensorMsgConverter.from_variant", lambda value: value)
    node._result_cb(future, request, node._request_generation, node._episode.goal_generation)
    return future


@pytest.mark.parametrize("smoothing", [False, True])
def test_selected_interval_and_zero_watermark_drive_actual_ticks(monkeypatch, smoothing):
    node = make_node(smoothing=smoothing)
    node._chunk_planner = SimpleNamespace(plan=lambda actions, **kw: ChunkPlan(actions, 2, 5, 0))
    deliver(node, monkeypatch, {"action": np.arange(12).reshape(6, 2)})
    for _ in range(3):
        node._control_loop()
        node._request_inference.assert_not_called()
    np.testing.assert_array_equal(
        [call.args[0] for call in node._executor.execute.call_args_list], [[4, 5], [6, 7], [8, 9]]
    )
    before = node._active_plan.snapshot()
    node._control_loop()
    node._request_inference.assert_called_once()
    assert node._active_plan.snapshot() == before


def test_capacity_pause_resume_retains_position_and_watermark(monkeypatch):
    node = make_node()
    node._plan_length_at_inference_start = 5
    node._chunk_planner = SimpleNamespace(
        plan=lambda actions, **kw: ChunkPlan(actions, kw["actions_executed"], None, 0)
    )
    deliver(node, monkeypatch, {"action": np.arange(100).reshape(50, 2)})
    before = node._active_plan.snapshot()
    assert (before.next_position, before.remaining, before.watermark) == (20, 30, 0)
    node._stop_nav_cb(None, SimpleNamespace())
    assert node._request_generation == 1
    node._control_loop()
    assert node._active_plan.snapshot() == before
    node._start_nav_cb(None, SimpleNamespace())
    for _ in range(5):
        node._control_loop()
    assert node._active_plan.snapshot().next_position == 25
    assert node._active_plan.snapshot().watermark == 0
    node._request_inference.assert_not_called()
    np.testing.assert_array_equal(node._executor.execute.call_args_list[0].args[0], [40, 41])


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action": np.zeros((2, 2, 2))},
        {"action": np.zeros((0, 2))},
        {"action": np.zeros((2, 3))},
        {"action": np.array([[np.nan, 1]])},
        {"action": np.array([[np.inf, 1]])},
    ],
)
def test_invalid_replacement_retains_continuous_but_aborts_benchmark(monkeypatch, smoothing, payload):
    for benchmark in (False, True):
        node = make_node(smoothing=smoothing, benchmark=benchmark)
        node._active_plan.accept(ChunkPlan(np.ones((3, 2)), replenishment_watermark=0), PlanSource("old"))
        before = node._active_plan.snapshot()
        deliver(node, monkeypatch, payload)
        assert not node._inference_in_progress
        assert node._dispatch_count == 0
        if benchmark:
            assert node._episode.phase is EpisodePhase.FAULTED
            assert node._episode.is_first_inference
            ctx = node._goal_contexts[b"goal"]
            assert ctx.termination_reason == "inference_failed"
            assert ctx.terminal_status == "aborted"
            assert ctx.result.published_actions == 0
            assert node._active_plan.snapshot().source is None
        else:
            assert node._active_plan.snapshot() == before
            node._control_loop()
            node._request_inference.assert_not_called()


@pytest.mark.parametrize("event", ["reset", "pause"])
@pytest.mark.parametrize("smoothing", [False, True])
def test_lifecycle_wins_over_decoding_result(monkeypatch, event, smoothing):
    node = make_node(smoothing=smoothing)
    node._active_plan.accept(ChunkPlan(np.ones((3, 2)), replenishment_watermark=0), PlanSource("old"))
    node._inflight_request_id = "late"
    node._inference_in_progress = True
    entered, release = threading.Event(), threading.Event()

    def decode(value):
        entered.set()
        assert release.wait(5)
        return {"action": np.zeros((5, 2))}

    monkeypatch.setattr("action_dispatch.action_dispatcher_node.TensorMsgConverter.from_variant", decode)
    future = Future()
    future.set_result(SimpleNamespace(result=SimpleNamespace(success=True, action_chunk=None)))
    errors = []

    def callback():
        try:
            node._result_cb(future, "late", 0, 0)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=callback)
    thread.start()
    try:
        assert entered.wait(5)
        if event == "reset":
            node._reset_cb(None, SimpleNamespace())
        else:
            node._stop_nav_cb(None, SimpleNamespace())
        after = node._active_plan.snapshot()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert node._active_plan.snapshot() == after
    assert after.source is None if event == "reset" else after.source.request_id == "old"


@pytest.mark.parametrize("smoothing", [False, True])
def test_benchmark_commits_once_and_preserves_feedback(monkeypatch, smoothing):
    node = make_node(benchmark=True, smoothing=smoothing)
    future = deliver(node, monkeypatch, {"action": np.arange(8).reshape(4, 2)})
    assert node._episode.phase is EpisodePhase.RUNNING
    before = node._active_plan.snapshot()
    node._result_cb(future, "request", 0, node._episode.goal_generation)
    assert node._active_plan.snapshot() == before
    node._submit_next_action_wait_for_feedback()
    assert node._active_plan.snapshot() == before
    completion = ExecutionCompletion(node._reservation_context.correlation_id, CompletionStatus.COMPLETED, 200, 7, 0)
    transition = node._scheduler.on_completion(completion)
    node._apply_completion_transition_wait_for_feedback(transition, completion)
    node._apply_completion_transition_wait_for_feedback(transition, completion)
    assert node._episode.committed_count == 1
    assert node._active_plan.snapshot().remaining == 3
    assert node._active_goal_handle.publish_feedback.call_count == 1
    assert node._episode.get_inference_timestamp() == 200
    np.testing.assert_array_equal(node._last_action, [0, 1])


def test_stale_reservation_does_not_consume_replacement():
    node = make_node()
    node._active_plan.accept(ChunkPlan(np.ones((3, 2))), PlanSource("old"))
    node._submit_next_action_wait_for_feedback()
    completion = ExecutionCompletion(node._reservation_context.correlation_id, CompletionStatus.COMPLETED, 200)
    node._active_plan.accept(ChunkPlan(np.zeros((2, 2))), PlanSource("new"))
    before = node._active_plan.snapshot()
    node._apply_completion_transition_wait_for_feedback(SchedulerTransition(CompletionDecision.COMMIT), completion)
    assert node._active_plan.snapshot() == before
    assert node._scheduler.fault_status == "plan_generation_mismatch"


@pytest.mark.parametrize("method", ["_cancel_episode", "_abort_episode", "_close_episode_terminal", "_reset_cb"])
def test_episode_cleanup_clears_exhausted_plan_metadata(method):
    node = make_node(benchmark=True)
    node._active_plan.accept(ChunkPlan(np.ones((1, 2)), replenishment_watermark=0), PlanSource("old"))
    node._submit_next_action_wait_for_feedback()
    reservation = node._plan_reservation
    node._active_plan.take_action()
    args = {
        "_cancel_episode": (),
        "_abort_episode": ("inference_failed",),
        "_close_episode_terminal": ("max_actions",),
        "_reset_cb": (None, SimpleNamespace()),
    }
    getattr(node, method)(*args[method])
    after = node._active_plan.snapshot()
    assert (after.remaining, after.source, after.next_position, after.watermark) == (0, None, None, 20)
    assert node._plan_reservation is None
    assert not node._active_plan.is_current(reservation)


def test_runtime_toggle_validation_precedes_all_mutation(monkeypatch):
    node = make_node(smoothing=True)
    node._active_plan.accept(ChunkPlan(np.ones((3, 2)), replenishment_watermark=0), PlanSource("old"))
    before = node._active_plan.snapshot()
    selection = node._strategy_selection
    validator = Mock(side_effect=DispatchStrategyError("unsupported combination"))
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.resolve_dispatch_strategies", validator)
    node._toggle_smoothing_cb(None, SimpleNamespace())
    assert node._strategy_selection is selection
    assert node._smoothing_enabled and node._smoother.is_enabled
    assert node._active_plan.snapshot() == before
    assert validator.call_args.kwargs["blending"] == "none"


def test_runtime_toggle_retains_metadata_and_clear_prevents_revival():
    node = make_node(smoothing=True)
    node._active_plan.accept(ChunkPlan(np.ones((3, 2)), replenishment_watermark=0), PlanSource("old"))
    before = node._active_plan.snapshot()
    node._toggle_smoothing_cb(None, SimpleNamespace())
    assert node._strategy_selection.blending == "none"
    assert not node._smoother.is_enabled
    assert node._active_plan.snapshot() == before
    node._reset_cb(None, SimpleNamespace())
    node._toggle_smoothing_cb(None, SimpleNamespace())
    assert node._strategy_selection.blending == "temporal_ensemble"
    assert node._active_plan.snapshot().source is None
    assert node._active_plan.snapshot().remaining == 0


def test_prepare_barrier_clears_plan_before_policy_reset():
    node = make_node()
    node._active_plan.accept(ChunkPlan(np.ones((3, 2)), replenishment_watermark=0), PlanSource("old"))
    node._policy_reset_timeout_s = 1

    def reset(prep_ctx):
        assert node._active_plan.snapshot().source is None
        assert node._active_plan.snapshot().remaining == 0
        node._complete_prepare_context(prep_ctx, True, "reset")

    node._request_policy_reset_for_prepare = reset
    response = node._prepare_episode_cb(None, SimpleNamespace())
    assert response.success
    assert response.preparation_id > 0
    assert node._request_generation == 1


def test_episode_rejected_commit_does_not_consume_or_publish(monkeypatch):
    node = make_node(benchmark=True)
    deliver(node, monkeypatch, {"action": np.ones((3, 2))})
    node._submit_next_action_wait_for_feedback()
    completion = ExecutionCompletion(node._reservation_context.correlation_id, CompletionStatus.COMPLETED, 200, 7, 0)
    before = node._active_plan.snapshot()
    monkeypatch.setattr(
        EpisodeStateMachine, "try_commit_step", lambda *args: SimpleNamespace(applied=False, identity_mismatch=False)
    )
    node._apply_completion_transition_wait_for_feedback(SchedulerTransition(CompletionDecision.COMMIT), completion)
    assert node._active_plan.snapshot() == before
    node._active_goal_handle.publish_feedback.assert_not_called()


def test_smoother_commit_does_not_index_after_episode_commit(monkeypatch):
    node = make_node(benchmark=True, smoothing=True)
    deliver(node, monkeypatch, {"action": np.ones((3, 2))})
    node._submit_next_action_wait_for_feedback()
    original = node._episode.try_commit_step

    def commit_episode(self, *args):
        outcome = original(*args)
        monkeypatch.setattr(node._active_plan, "is_current", Mock(side_effect=AssertionError("late validation")))
        return outcome

    monkeypatch.setattr(EpisodeStateMachine, "try_commit_step", commit_episode)
    monkeypatch.setattr(node._smoother, "get_next_action", Mock(side_effect=AssertionError("late indexing")))
    completion = ExecutionCompletion(node._reservation_context.correlation_id, CompletionStatus.COMPLETED, 200, 7, 0)
    node._apply_completion_transition_wait_for_feedback(node._scheduler.on_completion(completion), completion)
    assert node._episode.committed_count == 1
    assert node._active_plan.snapshot().remaining == 2


def test_stale_policy_reset_completion_cannot_release_new_reset():
    node = make_node()
    node.get_parameter = lambda name: SimpleNamespace(value="/reset")
    old, new = Future(), Future()
    node._policy_reset_client = Mock()
    node._policy_reset_client.call_async.side_effect = [old, new]
    node._request_policy_reset = ActionDispatcherNode._request_policy_reset.__get__(node)
    node._request_policy_reset()
    node._request_policy_reset()
    old.set_result(SimpleNamespace(success=True))
    assert node._policy_reset_in_progress
    assert node._policy_reset_future is new
    new.set_result(SimpleNamespace(success=True))
    assert not node._policy_reset_in_progress


def test_rejected_submission_never_consumes():
    node = make_node()
    node._active_plan.accept(ChunkPlan(np.ones((3, 2))), PlanSource("old"))
    node._executor.submit.side_effect = lambda action, ctx: ExecutionReceipt(ctx.correlation_id, False)
    before = node._active_plan.snapshot()
    node._submit_next_action_wait_for_feedback()
    assert node._active_plan.snapshot() == before
    assert node._scheduler.fault_status == "rejected"


@pytest.mark.parametrize("smoothing", [False, True])
def test_rejected_submission_aborts_episode_and_allows_prepare(monkeypatch, smoothing):
    node = make_node(benchmark=True, smoothing=smoothing)
    deliver(node, monkeypatch, {"action": np.ones((30, 2))})
    node._executor.drain_completions.return_value = ()
    node._executor.submit.side_effect = lambda action, ctx: ExecutionReceipt(ctx.correlation_id, False)
    commit = Mock(wraps=node._active_plan.commit)
    monkeypatch.setattr(node._active_plan, "commit", commit)
    goal = node._active_goal_handle
    ctx = node._goal_contexts[b"goal"]

    node._control_loop()

    assert ctx.done_event.wait(0.1)
    assert ctx.terminal_status == "aborted"
    assert ctx.termination_reason == "execution_rejected"
    assert ctx.result.published_actions == 0
    assert node._episode.phase is EpisodePhase.FAULTED
    assert not node._episode.is_gate_open
    assert node._episode.committed_count == 0
    commit.assert_not_called()
    goal.publish_feedback.assert_not_called()
    assert node._plan_reservation is None
    assert node._reservation_context is None
    assert node._active_plan.snapshot().source is None
    node._control_loop()
    node._executor.submit.assert_called_once()
    node._policy_reset_timeout_s = 1
    node._request_policy_reset_for_prepare = lambda prep: node._complete_prepare_context(prep, True, "reset")
    response = node._prepare_episode_cb(None, SimpleNamespace())
    assert response.success
    assert response.preparation_id > 0
    assert ctx.termination_reason == "execution_rejected"


def test_benchmark_wait_for_feedback_keeps_inference_and_consumption_gated(monkeypatch):
    node = make_node(benchmark=True)
    node._executor.drain_completions.return_value = ()
    deliver(node, monkeypatch, {"action": np.ones((30, 2))})
    node._control_loop()
    before = node._active_plan.snapshot()
    node._control_loop()
    node._request_inference.assert_not_called()
    node._executor.submit.assert_called_once()
    assert node._active_plan.snapshot() == before


@pytest.mark.parametrize("smoothing", [False, True])
def test_short_interval_pause_and_rejection_retain_request_policy(monkeypatch, smoothing):
    node = make_node(smoothing=smoothing)
    node._chunk_planner = SimpleNamespace(plan=lambda actions, **kw: ChunkPlan(actions, 2, 5, 0))
    deliver(node, monkeypatch, {"action": np.arange(12).reshape(6, 2)})
    before = node._active_plan.snapshot()
    node._stop_nav_cb(None, SimpleNamespace())
    node._control_loop()
    assert node._active_plan.snapshot() == before
    node._start_nav_cb(None, SimpleNamespace())
    deliver(node, monkeypatch, {"action": np.array([[np.nan, 1]])}, request="invalid")
    assert node._active_plan.snapshot() == before
    for _ in range(3):
        node._control_loop()
        node._request_inference.assert_not_called()
    np.testing.assert_array_equal(
        [call.args[0] for call in node._executor.execute.call_args_list], [[4, 5], [6, 7], [8, 9]]
    )
    node._control_loop()
    node._request_inference.assert_called_once()
