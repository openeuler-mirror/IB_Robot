"""Product-specific storage behavior, exercised through node callbacks."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from action_dispatch.action_dispatcher_node import ActionDispatcherNode
from action_dispatch.active_plan import ActivePlan, PlanSource
from action_dispatch.chunk_planning import ChunkPlan
from action_dispatch.scheduled_action_dispatcher_node import ScheduledActionDispatcherNode
from action_dispatch.temporal_smoother import TemporalSmootherManager
from robot_config.dispatch_strategies import resolve_dispatch_strategies


def test_legacy_toggle_keeps_manager_and_remaining_order():
    node = object.__new__(ActionDispatcherNode)
    node._dispatch_lock = threading.RLock()
    node._smoothing_enabled = True
    node._strategy_selection = resolve_dispatch_strategies(blending="temporal_ensemble")
    node.set_parameters_atomically = Mock(return_value=SimpleNamespace(successful=True))
    node._smoother = TemporalSmootherManager()
    node._active_plan = ActivePlan(capacity=10, watermark=0, smoother=node._smoother)
    node._active_plan.accept(ChunkPlan(np.arange(6).reshape(3, 2)), PlanSource("first"))
    manager = node._smoother
    node.get_logger = Mock()
    node._toggle_smoothing_cb(None, SimpleNamespace())
    assert node._smoother is manager
    assert not manager.is_enabled
    assert node._get_plan_length() == 3
    np.testing.assert_array_equal(node._active_plan.take_action().action, [0, 1])


def test_legacy_without_manager_cannot_toggle():
    node = object.__new__(ActionDispatcherNode)
    node._dispatch_lock = threading.RLock()
    node._smoother = None
    node._smoothing_enabled = False
    node.get_logger = Mock()
    node._toggle_smoothing_cb(None, SimpleNamespace())
    assert node._smoother is None
    assert not node._smoothing_enabled


def test_scheduled_toggle_retains_queue_but_discards_smoother():
    from robot_config.dispatch_strategies import resolve_dispatch_strategies

    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._smoothing_enabled = False
    node._strategy_selection = resolve_dispatch_strategies(entrypoint="scheduled")
    node.set_parameters_atomically = Mock(return_value=SimpleNamespace(successful=True))
    node._inflight_request_id = ""
    node._smoother = TemporalSmootherManager(enabled=False)
    node._queue_plan = ActivePlan(capacity=10, watermark=2)
    node._smoothed_plan = ActivePlan(capacity=10, watermark=2, smoother=node._smoother)
    node._queue_plan.accept(ChunkPlan(np.array([[9, 10]])), PlanSource("queue"))
    node._smoothing_pub = Mock()
    node.get_logger = Mock()
    node._toggle_smoothing_cb(None, SimpleNamespace())
    node._active_plan.accept(ChunkPlan(np.arange(6).reshape(3, 2)), PlanSource("smooth"))
    assert node._current_plan_length_locked() == 3
    node._toggle_smoothing_cb(None, SimpleNamespace())
    assert node._smoother.plan_length == 0
    assert node._current_plan_length_locked() == 1
    np.testing.assert_array_equal(node._active_plan.take_action().action, [9, 10])


def test_legacy_pause_resume_keeps_plan_and_last_action():
    node = object.__new__(ActionDispatcherNode)
    node._dispatch_lock = threading.RLock()
    node._is_benchmark = False
    node._is_running = True
    node._navigation_mode = False
    node._request_generation = 0
    node._active_plan = ActivePlan(capacity=10, watermark=0)
    node._active_plan.accept(ChunkPlan(np.array([[1, 2]])), PlanSource("first"))
    node._last_action = np.array([3, 4])
    node.get_logger = Mock()
    node._stop_nav_cb(None, SimpleNamespace())
    assert not node._is_running
    node._start_nav_cb(None, SimpleNamespace())
    assert node._is_running
    assert not node._inference_in_progress
    np.testing.assert_array_equal(node._active_plan.take_action().action, [1, 2])
    np.testing.assert_array_equal(node._last_action, [3, 4])


def test_benchmark_submission_only_reserves():
    from action_dispatch.executors.completion import ExecutionReceipt

    node = object.__new__(ActionDispatcherNode)
    node._dispatch_lock = threading.RLock()
    node._smoother = None
    node._active_plan = ActivePlan(capacity=10, watermark=0)
    node._active_plan.accept(ChunkPlan(np.array([[1, 2], [3, 4]])), PlanSource("request"))
    node._active_goal_handle = None
    node._executor = Mock()
    node._executor.submit.side_effect = lambda action, ctx: ExecutionReceipt(ctx.correlation_id, True)
    node._scheduler = Mock()
    node._submit_next_action_wait_for_feedback()
    assert node._active_plan.snapshot().remaining == 2
    assert node._active_plan.snapshot().next_position == 0
    assert node._active_plan.is_current(node._plan_reservation)
    node._executor.submit.assert_called_once()
