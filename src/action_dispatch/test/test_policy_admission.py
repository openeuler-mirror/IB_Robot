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


def test_future_stamp_within_skew_tolerance_admits(admission):
    # Sim time: the facade's /clock-driven stamp can lead this node's clock by
    # a few ms; a bounded negative age is skew, not staleness.
    guard, runtime, _ = admission
    runtime.status = status()
    runtime.status.stamp = SimpleNamespace(sec=10, nanosec=20_000_000)  # age -20 ms
    result = Mock()
    guard.acquire(result)
    assert result.call_args.args[0]
    assert guard.owned
    assert runtime.requests == ["policy_stream"]


def test_future_stamp_beyond_skew_tolerance_rejected(admission):
    guard, runtime, _ = admission
    runtime.status = status()
    runtime.status.stamp = SimpleNamespace(sec=10, nanosec=200_000_000)  # age -200 ms
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert not guard.owned
    assert runtime.requests == []


def test_owned_mode_survives_future_stamp_heartbeat(admission):
    # A slightly-ahead heartbeat must not revoke a running owned session.
    guard, runtime, _ = admission
    guard.acquire(Mock())
    assert guard.owned
    heartbeat = status("policy_stream")
    heartbeat.stamp = SimpleNamespace(sec=10, nanosec=20_000_000)  # age -20 ms
    guard.observe(heartbeat)
    assert guard.owned
    guard.on_revoke.assert_not_called()


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
    if stage == "set_mode":
        # The unanswered switch is kept so the idle rollback can wait for it.
        runtime.set_client.remove_pending_request.assert_not_called()
    else:
        runtime.get_client.remove_pending_request.assert_called_once_with(pending)


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


def test_failed_verification_after_mode_switch_returns_runtime_to_idle(admission):
    # set_mode committed, then the verify status is invalid: the admission
    # fails AND the runtime is returned to idle instead of being stranded in
    # the admission mode; a retry can admit cleanly afterwards.
    guard, runtime, _ = admission
    original_get = runtime.get_client.call_async
    state = {"calls": 0}

    def get_with_faulted_verify(request):
        state["calls"] += 1
        if state["calls"] == 2:
            future = Future()
            future.set_result(SimpleNamespace(status=status("policy_stream", faults=["controller_failure"])))
            return future
        return original_get(request)

    runtime.get_client.call_async = get_with_faulted_verify
    result = Mock()
    guard.acquire(result)
    assert not result.call_args.args[0]
    assert not guard.owned and not guard.pending
    assert runtime.requests == ["policy_stream", "idle"]
    assert runtime.status.active_mode == "idle"
    second = Mock()
    guard.acquire(second)
    assert second.call_args.args[0] and guard.owned


def test_pending_fail_on_observed_target_mode_rolls_back_to_idle(admission):
    # The on-change status published after the facade committed the switch can
    # invalidate the pending admission before the set_mode response is
    # processed; the runtime must still be returned to idle, and a retry must
    # be able to admit cleanly afterwards.
    guard, runtime, _ = admission
    runtime.defer = True
    result = Mock()
    guard.acquire(result)  # initial status ok; set_mode in flight
    runtime.defer = False
    runtime.status = status("policy_stream")  # facade committed the switch
    invalid_target_mode = status("policy_stream")
    invalid_target_mode.stamp = SimpleNamespace(sec=10, nanosec=900_000_000)  # age -0.9 s: beyond tolerance
    guard.observe(invalid_target_mode)
    assert not result.call_args.args[0]
    assert not guard.owned and not guard.pending
    assert runtime.requests == ["policy_stream"]  # the rollback waits for our own switch's answer
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="late"))
    assert not guard.owned  # late set_mode response is ignored
    assert runtime.requests == ["policy_stream", "idle"]
    second = Mock()
    guard.acquire(second)
    assert second.call_args.args[0] and guard.owned
    assert runtime.requests == ["policy_stream", "idle", "policy_stream"]


def test_admission_failure_does_not_touch_other_producers_mode(admission):
    guard, runtime, _ = admission
    runtime.defer = True
    result = Mock()
    guard.acquire(result)  # set_mode in flight
    runtime.defer = False
    runtime.status = status("stream")  # another producer holds the stream mode
    foreign = status("stream")
    foreign.stamp = SimpleNamespace(sec=10, nanosec=900_000_000)  # invalid age
    guard.observe(foreign)
    runtime.deferred[0].set_result(SimpleNamespace(success=False, message="mode taken"))
    assert not result.call_args.args[0]
    assert runtime.requests == ["policy_stream"]  # no idle request against a foreign mode
    # The rollback finished (it checked the foreign mode and did not touch it):
    # a new acquire is no longer gated as "runtime transition in progress".
    followup = Mock()
    guard.acquire(followup)
    assert not followup.call_args.args[0]
    assert followup.call_args.args[1] == "runtime changed during admission"


def test_acquire_while_idle_rollback_in_progress_fails_fast(admission):
    guard, runtime, _ = admission
    runtime.defer = True
    guard.acquire(Mock())  # initial status ok; set_mode in flight
    runtime.defer = False
    runtime.status = status("policy_stream")  # facade committed the switch
    invalid_target_mode = status("policy_stream")
    invalid_target_mode.stamp = SimpleNamespace(sec=10, nanosec=900_000_000)  # invalid age
    guard.observe(invalid_target_mode)  # admission fails; rollback waits for the unanswered switch
    blocked = Mock()
    guard.acquire(blocked)
    assert not blocked.call_args.args[0]
    assert blocked.call_args.args[1] == "runtime transition in progress"
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="committed"))  # rollback proceeds
    assert runtime.requests == ["policy_stream", "idle"]
    recovered = Mock()
    guard.acquire(recovered)
    assert recovered.call_args.args[0] and guard.owned
    assert runtime.requests == ["policy_stream", "idle", "policy_stream"]


def test_idle_rollback_times_out_and_clears_state(admission):
    guard, runtime, _ = admission
    stuck = Future()
    runtime.defer = True
    guard.acquire(Mock())  # set_mode in flight
    runtime.defer = False
    runtime.status = status("policy_stream")  # facade committed the switch
    original_get = runtime.get_client.call_async
    runtime.get_client.call_async = Mock(return_value=stuck)  # rollback get_status hangs
    invalid_target_mode = status("policy_stream")
    invalid_target_mode.stamp = SimpleNamespace(sec=10, nanosec=900_000_000)  # invalid age
    guard.observe(invalid_target_mode)  # admission fails; rollback starts and hangs
    blocked = Mock()
    guard.acquire(blocked)
    assert blocked.call_args.args[1] == "runtime transition in progress"
    guard._rollback_timer.cancel()  # do not wait out the real 2 s bound
    guard._rollback_expired(guard._rollback_token)  # deadline passes without a rollback response
    runtime.get_client.call_async = original_get
    # The gate is cleared: a new acquire is refused on the stranded mode, not
    # on "runtime transition in progress".
    followup = Mock()
    guard.acquire(followup)
    assert not followup.call_args.args[0]
    assert followup.call_args.args[1] != "runtime transition in progress"
    assert guard._logger.warning.called
    # The expired attempt's late answers must not request idle on behalf of
    # whatever owns the runtime now.
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="late"))
    stuck.set_result(SimpleNamespace(status=status("policy_stream")))
    assert runtime.requests == ["policy_stream"]


def fail_after_switch_committed(guard, runtime, rollback_status):
    """Fail a pending admission after its set_mode committed; the rollback reads ``rollback_status``."""
    runtime.defer = True
    guard.acquire(Mock())  # set_mode in flight
    runtime.defer = False
    runtime.status = status("policy_stream")  # facade committed the switch
    runtime.get_client.call_async = Mock(return_value=rollback_status)
    invalid_target_mode = status("policy_stream")
    invalid_target_mode.stamp = SimpleNamespace(sec=10, nanosec=900_000_000)  # invalid age
    guard.observe(invalid_target_mode)
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="committed"))


def test_idle_rollback_never_requested_over_a_stop_latch(admission):
    # A stop engaging while the rollback reads the status: idle would clear the
    # latch (and re-energize a TORQUE_OFF stop), so it is never requested.
    guard, runtime, _ = admission
    stuck = Future()
    fail_after_switch_committed(guard, runtime, stuck)
    stuck.set_result(SimpleNamespace(status=status("policy_stream", latched=True)))
    assert runtime.requests == ["policy_stream"]
    assert not guard._rolling_back


def test_release_waits_for_inflight_rollback_before_reporting_success(admission):
    # stop_evaluate cancels first (starting the rollback), then releases: the
    # release must not report success until the runtime is back in idle.
    guard, runtime, _ = admission
    runtime.defer = True
    guard.acquire(Mock())
    runtime.status = status("policy_stream")
    guard.cancel()  # rollback starts and waits for the unanswered switch
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="committed"))
    assert runtime.requests == ["policy_stream", "idle"]  # its idle request is deferred
    released = []
    worker = threading.Thread(target=lambda: released.append(guard.release(timeout=2.0)))
    worker.start()
    time.sleep(0.2)
    assert worker.is_alive()  # still waiting for the idle confirmation
    runtime.status = status("idle")
    runtime.deferred[1].set_result(SimpleNamespace(success=True, message="idle"))
    worker.join(timeout=2.0)
    assert released == [True]


def test_release_reports_failure_when_rollback_is_rejected(admission):
    guard, runtime, _ = admission
    runtime.defer = True
    guard.acquire(Mock())
    runtime.status = status("policy_stream")
    guard.cancel()
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="committed"))
    runtime.deferred[1].set_result(SimpleNamespace(success=False, message="controller switch failed"))
    runtime.reject = True  # the runtime keeps refusing idle
    runtime.defer = False
    assert not guard.release(timeout=0.5)
    assert runtime.requests == ["policy_stream", "idle", "idle"]  # release re-verified and retried once


def test_rollback_waits_for_the_switch_instead_of_trusting_a_pre_switch_idle(admission):
    # Canceled while our set_mode is still being processed: the runtime still
    # reports idle, then commits the switch. Reading the mode before that
    # answer would see idle and strand the committed policy mode.
    guard, runtime, _ = admission
    runtime.defer = True
    guard.acquire(Mock())
    runtime.defer = False
    guard.cancel()  # runtime.status is still the pre-switch idle
    assert runtime.requests == ["policy_stream"]
    runtime.status = status("policy_stream")  # the switch commits after the cancel
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="committed"))
    assert runtime.requests == ["policy_stream", "idle"]
    assert runtime.status.active_mode == "idle"


def own_policy_mode(guard):
    result = Mock()
    guard.acquire(result)
    assert result.call_args.args[0] and guard.owned


def test_owned_release_never_requests_idle_over_a_latch(admission):
    guard, runtime, _ = admission
    own_policy_mode(guard)
    runtime.status = status("policy_stream", latched=True)  # a stop is engaging
    assert not guard.release(timeout=0.5)
    assert runtime.requests == ["policy_stream"]


def test_owned_release_still_returns_a_faulted_policy_mode_to_idle(admission):
    guard, runtime, _ = admission
    own_policy_mode(guard)
    runtime.status = status("policy_stream", lifecycle="DEGRADED", faults=["controller_failure"])
    assert guard.release(timeout=0.5)
    assert runtime.requests == ["policy_stream", "idle"]


def test_release_after_an_unconfirmed_release_reverifies_the_runtime(admission):
    guard, runtime, _ = admission
    own_policy_mode(guard)
    runtime.defer = True  # the idle request goes unanswered
    assert not guard.release(timeout=0.1)
    runtime.defer = False
    assert guard.release(timeout=0.5)  # not trusted as idle: checked and returned
    assert runtime.requests == ["policy_stream", "idle", "idle"]
    assert runtime.status.active_mode == "idle"


def test_release_during_switch_in_flight_rolls_back_once(admission):
    # Releasing while a pending admission's set_mode is already in flight must
    # return the mode exactly once (rollback), not zero and not twice.
    guard, runtime, _ = admission
    runtime.defer = True
    guard.acquire(Mock())
    runtime.defer = False
    runtime.status = status("policy_stream")
    released = []
    worker = threading.Thread(target=lambda: released.append(guard.release(timeout=2.0)))
    worker.start()
    time.sleep(0.1)
    runtime.deferred[0].set_result(SimpleNamespace(success=True, message="committed"))
    worker.join(timeout=2.0)
    assert released == [True]
    assert runtime.requests == ["policy_stream", "idle"]
    after = Mock()
    guard.acquire(after)  # runtime is idle again; admission succeeds
    assert after.call_args.args[0] and guard.owned
    assert runtime.requests == ["policy_stream", "idle", "policy_stream"]


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
