"""Admission and shutdown regressions without robot hardware."""

import os
import signal
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dataset_tools import record_cli
from dataset_tools.record_cli import RecordCLI


def ready(value):
    future = Future()
    future.set_result(value)
    return future


@pytest.fixture
def cli():
    node = object.__new__(RecordCLI)
    node.get_logger = lambda: Mock()
    node.get_parameter = lambda name: SimpleNamespace(
        value={
            "control_mode": "teleop",
            "admission_timeout_sec": 0.03,
            "admission_attempts": 3,
            "shutdown_timeout_sec": 0.2,
        }[name]
    )
    node._goal_lock = threading.RLock()
    node._owned_goal = None
    node._send_goal_future = None
    node._goal_pending = False
    node._cancel_future = None
    node._closing = False
    node._cancel_requested = False
    node._episode_finished_evt = threading.Event()
    node._goal_started_evt = threading.Event()
    node._goal_rejected_evt = threading.Event()
    node._runtime_status_event = threading.Event()
    node._runtime_status = None
    node._runtime_status_received = 0.0
    node._admission_future = None
    node._admission_client = None
    node._admission_dirty = False
    node._teleop_session_owned = False
    node._teleop_stop_client = None
    node._mode_client = None
    node._last_result_success = False
    node._should_reset_before_episode = lambda: False
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=time.time_ns()))
    return node


def service(response=True):
    return SimpleNamespace(
        wait_for_service=Mock(return_value=True),
        call_async=Mock(return_value=ready(SimpleNamespace(success=response, message="test"))),
    )


def test_admission_orders_idle_rearm_then_fresh_status(cli):
    calls = []
    cli._mode_client = service()
    cli._rearm_client = service()
    cli._mode_client.call_async.side_effect = lambda request: (
        calls.append(request.mode) or ready(SimpleNamespace(success=True))
    )
    cli._rearm_client.call_async.side_effect = lambda request: (
        calls.append("rearm") or ready(SimpleNamespace(success=True))
    )
    status = SimpleNamespace(
        lifecycle="ACTIVE", active_mode="stream", stop_latched=False, stamp=SimpleNamespace(sec=0, nanosec=0)
    )
    # A cached healthy sample must not satisfy this admission.
    cli._status_callback(status)

    def publish(timeout):
        mode = "idle" if "rearm" not in calls else "stream"
        calls.append(f"fresh {mode}")
        status.active_mode = mode
        status.stamp.sec, status.stamp.nanosec = divmod(time.time_ns(), 1_000_000_000)
        cli._status_callback(status)

    cli._runtime_status_event.wait = publish
    assert cli._admit_teleop_episode()
    assert calls == ["idle", "fresh idle", "rearm", "fresh stream"]
    # The CLI owns the session it just armed.
    assert cli._teleop_session_owned


@pytest.mark.parametrize("failed_step", ["mode", "rearm"])
def test_admission_timeout_blocks_retries_and_new_prompt_until_settled(cli, failed_step):
    cli._mode_client = service()
    cli._rearm_client = service()
    cli._wait_runtime_mode = lambda *args: True
    client = cli._mode_client if failed_step == "mode" else cli._rearm_client
    pending = Future()
    client.call_async.return_value = pending
    cli._action_client = Mock()
    assert cli.send_goal("pick") is False
    assert client.call_async.call_count == 1
    assert cli._goal_rejected_evt.is_set()
    assert cli.send_goal("explicit new prompt") is False
    assert client.call_async.call_count == 1
    cli._action_client.send_goal_async.assert_not_called()
    # Settling never starts a background retry.
    pending.set_result(SimpleNamespace(success=False))
    assert client.call_async.call_count == 1


def test_admission_retries_final_failures_only_three_times(cli):
    cli._mode_client = service()
    cli._rearm_client = service(False)
    cli._wait_runtime_mode = lambda *args: True
    assert not cli._admit_teleop_episode()
    assert cli._mode_client.call_async.call_count == 3
    assert cli._rearm_client.call_async.call_count == 3
    # Every rearm was explicitly refused: no session, no ownership.
    assert not cli._teleop_session_owned


@pytest.mark.parametrize(
    "status",
    [
        dict(lifecycle="DEGRADED", active_mode="stream", stop_latched=False, stamp=101),
        dict(lifecycle="ACTIVE", active_mode="idle", stop_latched=False, stamp=101),
        dict(lifecycle="ACTIVE", active_mode="stream", stop_latched=True, stamp=101),
        dict(lifecycle="ACTIVE", active_mode="stream", stop_latched=False, stamp=99),
        dict(lifecycle="ACTIVE", active_mode="stream", stop_latched=False, stamp=10**20),
    ],
)
def test_admission_rejects_unhealthy_or_stale_status(cli, status):
    cli._mode_client = service()
    cli._rearm_client = service()
    stamp = status.pop("stamp")
    msg = SimpleNamespace(**status, stamp=SimpleNamespace(sec=0, nanosec=stamp))

    def publish(timeout):
        if stamp == 101:
            msg.stamp.sec, msg.stamp.nanosec = divmod(time.time_ns(), 1_000_000_000)
        cli._status_callback(msg)

    cli._runtime_status_event.wait = publish
    assert not cli._wait_runtime_mode("stream", 0.03)


def test_cancel_before_acceptance_cancels_only_owned_goal_and_waits_result(cli):
    acceptance = Future()
    result = Future()
    canceled = threading.Event()
    handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result,
        cancel_goal_async=Mock(side_effect=lambda: canceled.set() or ready(SimpleNamespace(goals_canceling=[1]))),
    )
    cli._action_client = SimpleNamespace(send_goal_async=lambda *a, **k: acceptance)
    cli.send_goal("pick")
    cleanup = threading.Thread(target=cli.finish_before_shutdown)
    cleanup.start()
    acceptance.set_result(handle)
    assert canceled.wait(0.1)
    assert cleanup.is_alive()
    result.set_result(SimpleNamespace(result=SimpleNamespace(success=True, message="Saved")))
    cleanup.join(0.2)
    assert not cleanup.is_alive()
    handle.cancel_goal_async.assert_called_once()
    assert cli._last_result_success
    assert cli._owned_goal is None


def test_new_prompt_cannot_rearm_while_recording_or_awaiting_acceptance(cli):
    cli._mode_client = service()
    cli._admit_teleop_episode = Mock()
    cli._goal_pending = True
    assert cli.send_goal("pick") is False
    cli._goal_pending = False
    cli._owned_goal = object()
    assert cli.send_goal("pick") is False
    cli._admit_teleop_episode.assert_not_called()


def test_shutdown_bound_does_not_cancel_unrelated_goal(cli):
    cli.finish_before_shutdown()
    assert cli._cancel_future is None
    cli._goal_pending = True
    cli._episode_finished_evt.wait = Mock(return_value=False)
    cli.finish_before_shutdown()
    assert 0 < cli._episode_finished_evt.wait.call_args.args[0] <= 0.2


def test_shutdown_drains_rearm_before_idle(cli):
    pending = Future()
    cli._admission_future = pending
    cli._admission_dirty = True
    cli._mode_client = service()
    cleanup = threading.Thread(target=cli.finish_before_shutdown)
    cleanup.start()
    cli._mode_client.call_async.assert_not_called()
    pending.set_result(SimpleNamespace(success=True))
    cleanup.join(0.3)
    assert not cleanup.is_alive()
    assert cli._mode_client.call_async.call_args.args[0].mode == "idle"


def test_shutdown_does_not_race_idle_against_unresolved_rearm(cli):
    cli._admission_future = Future()
    cli._admission_dirty = True
    cli._mode_client = service()
    cli._wait_for_future = Mock(return_value=False)
    logger = Mock()
    cli.get_logger = lambda: logger
    cli.finish_before_shutdown()
    cli._mode_client.call_async.assert_not_called()
    assert "operator stop" in logger.error.call_args.args[0]


def test_shutdown_stops_owned_teleop_and_restores_idle(cli):
    order = []

    def record(name, response=True):
        def call(_request):
            order.append(name)
            return ready(SimpleNamespace(success=response, message="test"))

        return SimpleNamespace(wait_for_service=Mock(return_value=True), call_async=call)

    cli._teleop_session_owned = True
    cli._teleop_stop_client = record("stop")
    cli._mode_client = record("idle")
    cli.finish_before_shutdown()
    assert order == ["stop", "idle"]
    assert not cli._teleop_session_owned
    assert not cli._admission_dirty


@pytest.mark.parametrize("finalized", [False, True])
def test_shutdown_stops_owned_teleop_before_waiting_for_slow_recording_result(cli, finalized):
    logger = Mock()
    cli.get_logger = lambda: logger
    cli._owned_goal = SimpleNamespace(cancel_goal_async=Mock(return_value=ready(SimpleNamespace(goals_canceling=[1]))))
    cli._teleop_session_owned = True
    cli._teleop_stop_client = service()
    cli._mode_client = service()

    def await_result(timeout):
        assert timeout > 0.0
        cli._owned_goal.cancel_goal_async.assert_called_once()
        cli._teleop_stop_client.call_async.assert_called_once()
        assert not cli._teleop_session_owned
        assert cli._mode_client.call_async.call_args.args[0].mode == "idle"
        return finalized

    cli._episode_finished_evt.wait = await_result
    cli.finish_before_shutdown()
    assert logger.error.call_count == (0 if finalized else 1)
    if not finalized:
        assert "owned goal result" in logger.error.call_args.args[0]


def test_shutdown_does_not_stop_teleop_it_did_not_arm(cli):
    cli._teleop_session_owned = False
    cli._admission_dirty = False
    cli._teleop_stop_client = service()
    cli._mode_client = service()
    cli.finish_before_shutdown()
    cli._teleop_stop_client.call_async.assert_not_called()
    cli._mode_client.call_async.assert_not_called()


def test_shutdown_without_confirmed_hold_keeps_session_and_skips_idle(cli):
    cli._teleop_session_owned = True
    cli._teleop_stop_client = service(False)
    cli._mode_client = service()
    cli.finish_before_shutdown()
    cli._mode_client.call_async.assert_not_called()
    assert cli._teleop_session_owned


def test_shutdown_without_stop_service_reports_operator_stop(cli):
    cli._teleop_session_owned = True
    cli._teleop_stop_client = None
    cli._mode_client = service()
    logger = Mock()
    cli.get_logger = lambda: logger
    cli.finish_before_shutdown()
    cli._mode_client.call_async.assert_not_called()
    assert "no configured stop service" in logger.error.call_args.args[0]
    assert cli._teleop_session_owned


def test_shutdown_retries_pending_hold_until_confirmed(cli):
    cli._teleop_session_owned = True
    responses = [SimpleNamespace(success=False), SimpleNamespace(success=True)]
    cli._teleop_stop_client = SimpleNamespace(
        wait_for_service=Mock(return_value=True),
        call_async=Mock(side_effect=lambda _request: ready(responses.pop(0))),
    )
    cli._mode_client = service()
    cli.finish_before_shutdown()
    assert cli._teleop_stop_client.call_async.call_count == 2
    assert cli._mode_client.call_async.call_args.args[0].mode == "idle"


@pytest.mark.parametrize("late_success", [True, False])
def test_shutdown_late_rearm_resolution_is_fenced(cli, late_success):
    pending = Future()
    cli._admission_future = pending
    cli._admission_client = cli._rearm_client = service()
    cli._admission_dirty = True
    cli._teleop_session_owned = False
    cli._teleop_stop_client = service()
    cli._mode_client = service()
    cleanup = threading.Thread(target=cli.finish_before_shutdown)
    cleanup.start()
    pending.set_result(SimpleNamespace(success=late_success))
    cleanup.join(0.3)
    assert not cleanup.is_alive()
    if late_success:
        cli._teleop_stop_client.call_async.assert_called_once()
    else:
        cli._teleop_stop_client.call_async.assert_not_called()
    assert cli._mode_client.call_async.call_args.args[0].mode == "idle"


def test_sigint_cleanup_precedes_context_shutdown(monkeypatch, cli):
    order = []
    monkeypatch.setattr(record_cli.rclpy, "init", lambda **kwargs: order.append(kwargs["signal_handler_options"]))
    monkeypatch.setattr(record_cli.rclpy, "ok", lambda: True)
    monkeypatch.setattr(record_cli.rclpy, "shutdown", lambda: order.append("context shutdown"))
    monkeypatch.setattr(record_cli, "RecordCLI", lambda: cli)
    executor = Mock()
    executor.shutdown.side_effect = lambda **kwargs: order.append("executor shutdown")
    monkeypatch.setattr(record_cli, "MultiThreadedExecutor", lambda **kwargs: executor)
    cli.finish_before_shutdown = lambda: order.append("owned goal cleanup")
    cli.destroy_node = lambda: order.append("node destroy")
    monkeypatch.setattr(record_cli, "cli_loop", lambda node: os.kill(os.getpid(), signal.SIGINT))
    record_cli.main()
    assert order == [
        record_cli.SignalHandlerOptions.NO,
        "owned goal cleanup",
        "executor shutdown",
        "node destroy",
        "context shutdown",
    ]


def test_discovery_requires_explicit_recorder_opt_in_and_preserves_cli_override(cli):
    cli.create_client = Mock()
    cli.create_subscription = Mock()
    cli.set_parameters = Mock()
    cli.configure_from_recorder_info({"path": "/legacy"})
    cli.create_client.assert_not_called()
    config = dict(
        runtime_set_mode_service="/runtime/mode",
        teleop_rearm_service="/teleop/rearm",
        runtime_status_topic="/status",
        admission_timeout_sec=2.0,
        admission_attempts=3,
    )
    # An older recorder without a stop endpoint is still usable.
    cli.configure_from_recorder_info({"teleop_admission": config})
    assert cli.create_client.call_count == 2
    cli._mode_client = None
    cli.create_client.reset_mock()
    config["teleop_stop_service"] = "/teleop/stop"
    cli.configure_from_recorder_info({"teleop_admission": config})
    assert cli.create_client.call_count == 3
    cli.configure_from_recorder_info({"teleop_admission": config})
    assert cli.create_client.call_count == 3
