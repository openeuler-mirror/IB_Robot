"""Reciprocal runtime admission and one-way producer revocation."""

import json
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from std_srvs.srv import Trigger

from action_dispatch.action_dispatcher_node import ActionDispatcherNode
from action_dispatch.active_plan import ActivePlan, PlanSource
from action_dispatch.chunk_planning import ChunkPlan
from action_dispatch.policy_admission import PolicyAdmission
from action_dispatch.scheduled_action_dispatcher_node import DispatcherState, ScheduledActionDispatcherNode

IDENTITY = {"id": "unit-1", "type": "unit", "runtime_name": "unit_runtime", "runtime_version": "1.0.0"}


def status(mode="idle", lifecycle="ACTIVE", latched=False, *, name="unit_runtime", version="1.0.0", faults=()):
    return SimpleNamespace(
        active_mode=mode,
        lifecycle=lifecycle,
        stop_latched=latched,
        runtime_name=name,
        runtime_version=version,
        faults=list(faults),
        stamp=SimpleNamespace(sec=10, nanosec=0),
        interface_description_json=json.dumps(
            {"robot": {**IDENTITY, "runtime_name": name, "runtime_version": version}}
        ),
    )


class Runtime:
    def __init__(self):
        self.status = status()
        self.requests = []
        self.deferred = []
        self.defer = False
        self.reject = False
        self.get_client = Mock(service_is_ready=lambda: True, call_async=self.get)
        self.set_client = Mock(service_is_ready=lambda: True, call_async=self.set)

    def get(self, request):
        future = Future()
        future.set_result(SimpleNamespace(status=self.status))
        return future

    def set(self, request):
        self.requests.append(request.mode)
        future = Future()
        if self.defer:
            self.deferred.append(future)
        else:
            if not self.reject:
                self.status = status(request.mode)
            future.set_result(SimpleNamespace(success=not self.reject, message="mode response"))
        return future


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(
        "action_dispatch.policy_admission.load_robot_section",
        lambda _: (None, {"control_modes": {"model_inference": {"runtime_mode": "policy_stream"}}}),
    )
    runtime = Runtime()
    node = Mock()
    node.get_clock.return_value.now.return_value.nanoseconds = 10_000_000_000
    node.create_client.side_effect = [runtime.get_client, runtime.set_client]
    interfaces = {
        "runtime.status": {
            "kind": "topic",
            "direction": "publish",
            "message_type": "ibrobot_msgs/msg/RuntimeStatus",
            "endpoint": "/unit/status",
            "qos": {"depth": 7, "reliability": "reliable"},
        },
        "runtime.get_status": {
            "kind": "service",
            "direction": "serve",
            "message_type": "ibrobot_msgs/srv/GetRuntimeStatus",
            "endpoint": "/unit/get_status",
        },
        "runtime.set_mode": {
            "kind": "service",
            "direction": "serve",
            "message_type": "ibrobot_msgs/srv/SetRuntimeMode",
            "endpoint": "/unit/set_mode",
        },
    }
    guard = PolicyAdmission(
        node,
        "robot.yaml",
        {
            "interface_description": {
                "robot": IDENTITY,
                "interfaces": interfaces,
            }
        },
        threading.RLock(),
        Mock(),
    )
    yield guard, runtime, node
    guard.cancel()


def test_binds_public_runtime_endpoints_and_qos(admission):
    guard, _, node = admission
    assert [call.args[1] for call in node.create_client.call_args_list] == ["/unit/get_status", "/unit/set_mode"]
    assert node.create_subscription.call_args.args[1] == "/unit/status"
    assert node.create_subscription.call_args.args[3].depth == 7
    assert guard.mode == "policy_stream"


@pytest.mark.parametrize("mode", ["stream", "policy_stream", "trajectory"])
def test_new_policy_rejects_every_nonidle_mode_even_matching(admission, mode):
    guard, runtime, _ = admission
    runtime.status = status(mode)
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert runtime.requests == []
    assert not guard.owned


@pytest.mark.parametrize("lifecycle,latched", [("STOPPED", False), ("ACTIVE", True), ("FAULTED", False)])
def test_lifecycle_or_latch_rejects_without_clearing_stop(admission, lifecycle, latched):
    guard, runtime, _ = admission
    runtime.status = status(lifecycle=lifecycle, latched=latched)
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert runtime.requests == []


def test_acquire_repeated_own_start_release_and_reacquire(admission):
    guard, runtime, _ = admission
    result = Mock()
    guard.acquire(result)
    assert result.call_args.args[0] and guard.owned
    guard.acquire(result, own_active_session=True)
    assert result.call_args.args[0]
    assert runtime.requests == ["policy_stream"]
    assert guard.release()
    assert not guard.owned
    guard.acquire(result)
    assert result.call_args.args[0]
    assert runtime.requests == ["policy_stream", "idle", "policy_stream"]


@pytest.mark.parametrize(
    "revoked", [status(), status("policy_stream", "STOPPED"), status("policy_stream", latched=True), status("stream")]
)
def test_runtime_stop_revokes_and_recovery_never_restarts(admission, revoked):
    guard, runtime, _ = admission
    guard.acquire(Mock())
    guard.observe(revoked)
    assert not guard.owned
    guard.on_revoke.assert_called_once()
    guard.observe(status("policy_stream"))
    assert not guard.owned
    assert runtime.requests == ["policy_stream"]


def test_stop_while_acquiring_ignores_late_mode_success(admission):
    guard, runtime, _ = admission
    runtime.defer = True
    result = Mock()
    guard.acquire(result)
    guard.observe(status(lifecycle="STOPPED", latched=True))
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="late"))
    assert not guard.owned
    result.assert_called_once()
    assert not result.call_args.args[0]


def test_cancel_and_concurrent_start_do_not_inherit_pending_mode(admission):
    guard, runtime, _ = admission
    runtime.defer = True
    first, second = Mock(), Mock()
    guard.acquire(first)
    guard.acquire(second)
    assert not second.call_args.args[0]
    guard.cancel()
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="late"))
    assert not guard.owned
    assert not first.call_args.args[0]


def test_idle_after_target_observation_revokes_pending_admission(admission):
    guard, runtime, _ = admission
    runtime.defer = True
    result = Mock()
    guard.acquire(result)
    guard.observe(status("policy_stream"))
    guard.observe(status())
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="late"))
    assert not result.call_args.args[0]
    assert not guard.owned


def test_release_accepts_its_own_idle_notification(admission):
    guard, runtime, _ = admission
    guard.acquire(Mock())
    original = runtime.set_client.call_async

    def set_and_publish(request):
        future = original(request)
        guard.observe(runtime.status)
        return future

    runtime.set_client.call_async = set_and_publish
    assert guard.release()
    assert runtime.requests == ["policy_stream", "idle"]


def test_failed_mode_request_does_not_enable_or_retry(admission):
    guard, runtime, _ = admission
    runtime.reject = True
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert not guard.owned
    assert runtime.requests == ["policy_stream"]


def test_unavailable_status_service_fails_closed(admission):
    guard, runtime, _ = admission
    runtime.get_client.service_is_ready = lambda: False
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert not guard.owned
    assert runtime.requests == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "another_runtime"},
        {"version": "2.0.0"},
        {"faults": ["controller_failure"]},
    ],
)
@pytest.mark.parametrize("already_running", [False, True])
def test_status_identity_and_faults_reject_and_revoke(admission, kwargs, already_running):
    guard, runtime, _ = admission
    if already_running:
        guard.acquire(Mock())
        guard.observe(status("policy_stream", **kwargs))
        guard.on_revoke.assert_called_once()
        guard.observe(status("policy_stream"))
        assert not guard.owned
    else:
        runtime.status = status(**kwargs)
        result = Mock()
        guard.acquire(result)
        assert not result.call_args.args[0]
        assert runtime.requests == []


@pytest.mark.parametrize("stamp_sec,stamp_nanosec", [(7, 0), (11, 0), (10, -1), (10, 1_000_000_000)])
def test_stale_future_and_invalid_stamp_rejected(admission, stamp_sec, stamp_nanosec):
    guard, runtime, _ = admission
    runtime.status.stamp = SimpleNamespace(sec=stamp_sec, nanosec=stamp_nanosec)
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert not guard.owned
    assert runtime.requests == []


def test_silence_revokes_even_when_ros_clock_is_frozen(admission):
    guard, runtime, node = admission
    guard.acquire(Mock())
    assert guard.owned
    guard._last_status_received = time.monotonic() - 2.1
    guard._check_freshness()
    guard.on_revoke.assert_called_once()
    assert not guard.owned
    guard.observe(status("policy_stream"))
    guard._check_freshness()
    assert not guard.owned
    assert runtime.requests == ["policy_stream"]
    from rclpy.clock import ClockType

    assert node.create_timer.call_args.kwargs["clock"].clock_type is ClockType.STEADY_TIME


@pytest.mark.parametrize("stage", ["initial_status", "set_mode", "verify_status"])
def test_unresponsive_rpc_chain_is_bounded_and_late_success_is_ignored(admission, stage):
    guard, runtime, _ = admission
    guard._rpc_timeout_s = 0.02
    pending = Future()
    if stage == "initial_status":
        runtime.get_client.call_async = Mock(return_value=pending)
    elif stage == "set_mode":
        runtime.set_client.call_async = Mock(return_value=pending)
    else:
        runtime.get_client.call_async = Mock(side_effect=[runtime.get(None), pending])
    completed = threading.Event()
    result = Mock(side_effect=lambda *_: completed.set())
    started = time.monotonic()
    guard.acquire(result)
    assert completed.wait(1.0)
    assert time.monotonic() - started < 1.0
    assert not result.call_args.args[0]
    assert "timed out" in result.call_args.args[1]
    assert not guard.pending and not guard.owned
    pending.set_result(SimpleNamespace(status=status("policy_stream"), success=True, message="late success"))
    result.assert_called_once()
    assert not guard.owned
    client = runtime.set_client if stage == "set_mode" else runtime.get_client
    client.remove_pending_request.assert_called_once_with(pending)


def test_completed_attempt_timer_cannot_revoke_new_owner(admission):
    guard, _, _ = admission
    old_generation = guard.generation
    guard.acquire(Mock())
    guard._timeout(old_generation)
    assert guard.owned
    guard.on_revoke.assert_not_called()


@pytest.mark.parametrize(
    "description",
    [
        "not json",
        "null",
        "[]",
        "{}",
        json.dumps({"robot": {**IDENTITY, "id": "another-instance"}}),
        json.dumps({"robot": {**IDENTITY, "runtime_version": "other"}}),
    ],
)
def test_advertised_identity_must_match_bound_instance(admission, description):
    guard, runtime, _ = admission
    runtime.status.interface_description_json = description
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert runtime.requests == []


def test_unresponsive_release_is_bounded_and_cannot_restart(admission):
    guard, runtime, _ = admission
    guard.acquire(Mock())
    pending = Future()
    runtime.set_client.call_async = Mock(return_value=pending)
    assert not guard.release(timeout=0.01)
    assert not guard.owned and not guard.pending
    pending.set_result(SimpleNamespace(success=True, message="late idle"))
    guard.observe(status("policy_stream"))
    assert not guard.owned


def test_response_after_deadline_cannot_win_delayed_timer(admission):
    guard, runtime, _ = admission
    runtime.defer = True
    result = Mock()
    guard.acquire(result)
    guard._timeout_timer.cancel()
    guard._rpc_deadline = time.monotonic() - 0.01
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="late"))
    assert not result.call_args.args[0]
    assert "timed out" in result.call_args.args[1]
    assert not guard.owned


@pytest.mark.parametrize("scheduled", [False, True])
def test_dispatcher_revocation_clears_queue_inflight_and_requires_explicit_start(admission, scheduled):
    guard, runtime, _ = admission
    cls = ScheduledActionDispatcherNode if scheduled else ActionDispatcherNode
    node = object.__new__(cls)
    node.get_logger = Mock()
    node._executor = Mock()
    node._last_action = np.array([1.0])
    node._runtime_admission = guard
    plan = ActivePlan(capacity=10, watermark=2)
    plan.accept(ChunkPlan(np.ones((3, 1))), PlanSource("pending"))
    if scheduled:
        node._state_lock = guard.lock
        node._state = DispatcherState.ACTIVE
        node._session_id, node._session_generation = "session", 1
        node._queue_plan = plan
        node._smoothed_plan = ActivePlan(capacity=10, watermark=2)
        node._smoothing_enabled = False
        node._inflight_request_id = "pending"
        node._inflight_goal_handle = Mock()
        goal = node._inflight_goal_handle
    else:
        node._dispatch_lock = guard.lock
        node._is_running = True
        node._is_benchmark = False
        node._scheduler_mode = "continuous"
        node._active_plan = plan
        node._inference_in_progress = True
        node._request_generation = 5
        node._inflight_request_id = "pending"
    guard.on_revoke = node._runtime_revoked
    guard.acquire(Mock())
    guard.observe(status())
    assert plan.snapshot().remaining == 0
    assert node._last_action is None
    assert node._inflight_request_id == ""
    if scheduled:
        assert node._state is DispatcherState.FAILED
        goal.cancel_goal_async.assert_called_once()
    else:
        assert not node._is_running
        assert not node._inference_in_progress
        assert node._request_generation == 6
    node._control_loop()
    guard.observe(status("policy_stream"))
    node._control_loop()
    node._executor.execute.assert_not_called()


def test_legacy_start_hook_acquires_before_enabling(admission):
    guard, runtime, _ = admission
    node = object.__new__(ActionDispatcherNode)
    node._dispatch_lock = guard.lock
    node._runtime_admission = guard
    node._is_benchmark = False
    node._is_running = False
    node._request_generation = 0
    node.get_logger = Mock()
    runtime.status = status("stream")
    response = node._start_nav_cb(None, Trigger.Response())
    assert not response.success and not node._is_running
    runtime.status = status()
    response = node._start_nav_cb(None, Trigger.Response())
    assert response.success and node._is_running
    assert runtime.requests == ["policy_stream"]


def test_scheduled_open_hook_never_opens_before_admission(admission):
    guard, runtime, _ = admission
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = guard.lock
    node._state = DispatcherState.STOPPED
    node._runtime_admission = guard
    node._open_admitted_session = Mock()
    node.get_logger = Mock()
    runtime.status = status("policy_stream")
    completed = threading.Event()
    node._open_new_session(completion=completed)
    assert completed.is_set()
    node._open_admitted_session.assert_not_called()
    runtime.status = status()
    node._open_new_session()
    node._open_admitted_session.assert_called_once()
