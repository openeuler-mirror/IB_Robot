import threading
import time
from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from rclpy.action import CancelResponse

from ibrobot_msgs.action import ExecuteNavigation
from robot_navigation import navigation_command_server
from robot_navigation.navigation_command_server import NavigationCommandServer, NavigationState


class _PendingFuture:
    def add_done_callback(self, _callback):
        return None

    def result(self):
        raise AssertionError("a pending future must not be read")


class _CompletedFuture:
    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return "done"


class _SettableFuture:
    def __init__(self):
        self._callbacks = []
        self._done = False
        self._result = None

    def add_done_callback(self, callback):
        if self._done:
            callback(self)
        else:
            self._callbacks.append(callback)

    def result(self):
        return self._result

    def set_result(self, result):
        self._result = result
        self._done = True
        for callback in self._callbacks:
            callback(self)


class _LateNavGoalHandle:
    accepted = True

    def __init__(self):
        self.cancel_count = 0
        self.canceled = threading.Event()
        self._result_future = _SettableFuture()
        self._result_future.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))

    def cancel_goal_async(self):
        self.cancel_count += 1
        self.canceled.set()
        return _CompletedResponseFuture(SimpleNamespace(goals_canceling=[object()]))

    def get_result_async(self):
        return self._result_future


class _CompletedResponseFuture:
    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class _ExecuteGoalHandle:
    def __init__(self):
        self.request = SimpleNamespace()
        self.aborted = False

    @property
    def is_cancel_requested(self):
        return False

    def abort(self):
        self.aborted = True


def _construct_server(monkeypatch, *, action_server_names=None, **parameter_overrides):
    parameters = {}
    action_server_names = action_server_names if action_server_names is not None else []

    def declare_parameter(_self, name, default):
        parameters[name] = parameter_overrides.get(name, default)

    def get_parameter(_self, name):
        return SimpleNamespace(value=parameters[name])

    class _ActionServer:
        def __init__(self, _node, _action_type, action_name, **_kwargs):
            action_server_names.append(action_name)

        def destroy(self):
            pass

    monkeypatch.setattr(navigation_command_server.Node, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(NavigationCommandServer, "declare_parameter", declare_parameter)
    monkeypatch.setattr(NavigationCommandServer, "get_parameter", get_parameter)
    monkeypatch.setattr(NavigationCommandServer, "create_service", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(NavigationCommandServer, "create_subscription", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(navigation_command_server, "ReentrantCallbackGroup", lambda: object())
    monkeypatch.setattr(navigation_command_server, "Buffer", lambda: object())
    monkeypatch.setattr(navigation_command_server, "TransformListener", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(navigation_command_server, "ActionClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(navigation_command_server, "ActionServer", _ActionServer)

    return NavigationCommandServer(), action_server_names


def test_blank_action_name_fails_before_action_server_creation(monkeypatch):
    action_server_names = []

    with pytest.raises(ValueError, match="action_name.*non-empty"):
        _construct_server(
            monkeypatch,
            action_server_names=action_server_names,
            action_name="   ",
        )

    assert action_server_names == []


def test_non_default_action_name_is_used_to_create_action_server(monkeypatch):
    server, action_server_names = _construct_server(
        monkeypatch,
        action_name="/robot/navigation/execute",
    )

    assert server.action_name == "/robot/navigation/execute"
    assert action_server_names == ["/robot/navigation/execute"]


def test_wait_future_has_a_finite_timeout():
    started = time.monotonic()

    assert NavigationCommandServer._wait_future(_PendingFuture(), 0.01) is None
    assert time.monotonic() - started < 0.5


def test_cancel_response_is_accepted_only_when_a_goal_is_listed():
    assert NavigationCommandServer._cancel_response_accepted(SimpleNamespace(goals_canceling=[object()]))
    assert not NavigationCommandServer._cancel_response_accepted(SimpleNamespace(goals_canceling=[]))


def test_cancel_callback_rejects_non_owner_goal():
    server = object.__new__(NavigationCommandServer)
    server._lock = threading.Lock()
    owner = object()
    server._active_execute_goal_handle = owner
    cancel_calls = []
    server._request_cancel = lambda: cancel_calls.append(True) or True

    assert server._cancel_callback(object()) == CancelResponse.REJECT
    assert server._cancel_callback(owner) == CancelResponse.ACCEPT
    assert cancel_calls == [True]


def test_result_wait_uses_cancel_terminal_timeout_after_cancel_request():
    cancel_requested = threading.Event()
    cancel_requested.set()
    started = time.monotonic()

    result, cancel_terminal_timeout = NavigationCommandServer._wait_result_future(
        _PendingFuture(),
        result_timeout=1.0,
        cancel_timeout=0.01,
        cancel_requested=cancel_requested,
    )

    assert result is None
    assert cancel_terminal_timeout is True
    assert time.monotonic() - started < 0.5


def test_result_wait_returns_a_completed_future_without_timeout():
    result, cancel_terminal_timeout = NavigationCommandServer._wait_result_future(
        _CompletedFuture(),
        result_timeout=1.0,
        cancel_timeout=0.01,
        cancel_requested=threading.Event(),
    )

    assert result == "done"
    assert cancel_terminal_timeout is False


def test_nav2_acceptance_timeout_keeps_fault_ownership_and_cancels_late_goal():
    send_future = _SettableFuture()
    server = object.__new__(NavigationCommandServer)
    server._lock = threading.Lock()
    server.state = NavigationState.IDLE
    server._generation = 0
    server._nav_goal_handle = None
    server._cancel_sent = False
    server._cancel_failed = False
    server._timeout_cancel_requested = False
    server._cancel_requested = threading.Event()
    server._cancel_complete = threading.Event()
    server._stop_confirmed = threading.Event()
    server._stop_confirmed.set()
    server._stop_gate = SimpleNamespace(reset=lambda: None)
    server.nav2_server_timeout = 0.01
    server.cancel_response_timeout = 0.01
    server.cancel_timeout = 0.01
    server.stop_confirmation_timeout = 0.01
    server._resolve_target = lambda _request: PoseStamped()
    server._wait_for_nav2_server = lambda: True
    # _start_goal clears the costmap gate; this test drives the Nav2
    # acceptance-timeout path, so start already-ready instead of polling.
    server._costmap_ready = threading.Event()
    server._wait_for_costmap_ready = lambda: True
    server._nav_client = SimpleNamespace(send_goal_async=lambda *_args, **_kwargs: send_future)
    execute_goal = _ExecuteGoalHandle()

    results = []
    execute_thread = threading.Thread(target=lambda: results.append(server._execute_callback(execute_goal)))
    execute_thread.start()
    _wait_for_state(server, NavigationState.FAULT)
    assert execute_thread.is_alive()

    late_goal = _LateNavGoalHandle()
    stop_confirmer = threading.Thread(
        target=lambda: (
            _wait_for_state(server, NavigationState.STOPPING),
            server._stop_confirmed.set(),
        )
    )
    stop_confirmer.start()
    send_future.set_result(late_goal)
    stop_confirmer.join(timeout=0.5)
    execute_thread.join(timeout=0.5)

    assert execute_thread.is_alive() is False
    assert results[0].error_code == ExecuteNavigation.Result.NAVIGATION_CANCELED
    assert execute_goal.aborted is True
    assert late_goal.canceled.wait(timeout=0.5)
    assert late_goal.cancel_count == 1
    assert server.state == NavigationState.IDLE


def _wait_for_state(server, expected):
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and server.state != expected:
        time.sleep(0.001)
    assert server.state == expected


def test_cancel_during_nav2_readiness_prevents_goal_dispatch():
    sent = []
    server = object.__new__(NavigationCommandServer)
    server._lock = threading.Lock()
    server.state = NavigationState.IDLE
    server._generation = 0
    server._nav_goal_handle = None
    server._cancel_sent = False
    server._cancel_failed = False
    server._timeout_cancel_requested = False
    server._cancel_requested = threading.Event()
    server._cancel_complete = threading.Event()
    server._stop_confirmed = threading.Event()
    server._stop_gate = SimpleNamespace(reset=lambda: None)
    server.stop_confirmation_timeout = 0.01
    server._resolve_target = lambda _request: PoseStamped()
    # The server gained a costmap readiness gate; this test is about the cancel
    # path, so start already-ready rather than exercising the wait.
    server._costmap_ready = threading.Event()
    server._costmap_ready.set()

    def ready_after_cancel():
        server._cancel_requested.set()
        server._stop_confirmed.set()
        return True

    server._wait_for_nav2_server = ready_after_cancel
    server._nav_client = SimpleNamespace(send_goal_async=lambda *_args, **_kwargs: sent.append(True))
    execute_goal = _ExecuteGoalHandle()

    result = server._execute_callback(execute_goal)

    assert result.error_code == ExecuteNavigation.Result.NAVIGATION_CANCELED
    assert sent == []
