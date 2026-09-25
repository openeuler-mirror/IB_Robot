import json
import signal
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_config.loader import load_robot_config_dict
from robot_skill_cli.catalog import compile_local_snapshot, load_capability_catalog
from skill_catalog.models import SkillSnapshot

# Offline-complete variant of the provider-bound so101_single_arm config; the
# catalog compiler needs real joint groups without a live runtime description.
CONFIG_PATH = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm_legacy.yaml"


class _CancelBridge:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.status_calls = []
        self.cancel_calls = []

    def start(self):
        return True

    def get_status(self, **kwargs):
        self.status_calls.append(kwargs)
        return self.statuses.pop(0)

    def cancel_task(self, task_id, **kwargs):
        self.cancel_calls.append((task_id, kwargs))
        return {"accepted": True}

    def close(self):
        pass


class _Future:
    def __init__(self, value=None, *, done=True, error=None):
        self.value = value
        self.completed = done
        self.error = error

    def done(self):
        return self.completed

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class _GoalHandle:
    def __init__(self, *, accepted=True, result=None, status=4, result_done=True, result_error=None):
        self.accepted = accepted
        self.result_future = _Future(
            SimpleNamespace(status=status, result=result), done=result_done, error=result_error
        )

    def get_result_async(self):
        return self.result_future


class _ExecuteBridge:
    def __init__(
        self,
        statuses,
        *,
        accepted=True,
        result_done=True,
        cancel_converges=True,
        goal_response_timeout=False,
        late_feedback_on_close=False,
        result_error=None,
    ):
        self.snapshot = _runtime_snapshot()
        for status in statuses:
            status.update(
                registry_epoch="epoch-1",
                registry_generation=1,
                registry_digest=self.snapshot.registry_digest,
                capability_digest=self.snapshot.capability_digest,
            )
        self.statuses = deque(statuses)
        self.calls = []
        self.sent = []
        self.cancel_converges = cancel_converges
        self.goal_response_timeout = goal_response_timeout
        self.late_feedback_on_close = late_feedback_on_close
        self.feedback_callback = None
        self.goal_wait_hook = None
        result = SimpleNamespace(
            success=True,
            error_code="",
            message="completed",
            executed_primitives=["private_one", "private_two"],
        )
        self.goal_handle = _GoalHandle(
            accepted=accepted,
            result=result,
            result_done=result_done,
            result_error=result_error,
        )

    def start(self):
        self.calls.append("start")
        return True

    def get_status(self, **kwargs):
        self.calls.append(("status", kwargs))
        return self.statuses.popleft()

    def validate_skill(self, payload, **kwargs):
        self.calls.append("validate")
        return {"allowed": True, "reason": "allowed"}

    def get_skill_snapshot(self, **_kwargs):
        return {
            "success": True,
            "registry_epoch": "epoch-1",
            "generation": 1,
            "registry_digest": self.snapshot.registry_digest,
            "capability_digest": self.snapshot.capability_digest,
            "provenance_digest": self.snapshot.provenance_digest,
            "snapshot_json": self.snapshot.snapshot_json,
        }

    def wait_for_skill_server(self, **kwargs):
        self.calls.append("wait_server")
        return True

    def send_skill_goal(self, payload, *, task_id, feedback_callback):
        self.calls.append("send")
        self.sent.append((payload, task_id))
        self.feedback_callback = feedback_callback
        feedback_callback({"state": "executing", "detail": "step 1 of 2"})
        return _Future(self.goal_handle)

    def wait_future(self, future, **kwargs):
        if self.goal_response_timeout and future.value is self.goal_handle:
            if self.goal_wait_hook is not None:
                self.goal_wait_hook()
            return False
        return future.done()

    def cancel_task(self, task_id, **kwargs):
        self.calls.append(("cancel_task", task_id, kwargs))
        return {"accepted": True, "return_code": 0}

    def cancel_goal(self, goal_handle, result_future, **kwargs):
        self.calls.append("cancel_goal")
        if self.cancel_converges:
            result_future.value.result.success = False
            result_future.value.result.error_code = "SKILL_CANCELLED"
            result_future.value.result.message = "cancelled"
            result_future.completed = True
        return self.cancel_converges

    def close(self):
        self.calls.append("close")
        if self.late_feedback_on_close and self.feedback_callback is not None:
            self.feedback_callback({"state": "executing", "detail": "late private_joint"})


class _ExecutePlanBridge:
    def __init__(self, *, terminal_status=4, success=True, error_code="", result_done=True):
        self.calls = []
        self.feedback_callback = None
        self.wait_hook = None
        self.server_hook = None
        self.server_ready = True
        self.status_hook = None
        self.result = {
            "success": success,
            "plan_id": "plan-1",
            "plan_digest": "digest-1",
            "workflow_digest": "",
            "completed_step_count": 1 if success else 0,
            "error_code": error_code,
            "message": "completed" if success else "stopped",
            "actual_registry_epoch": "epoch-1",
            "actual_registry_generation": 1,
            "actual_registry_digest": "registry-1",
        }
        result = SimpleNamespace(**self.result)
        self.result_future = _Future(
            SimpleNamespace(status=terminal_status, result=result),
            done=result_done,
        )
        self.goal_handle = _GoalHandle(
            accepted=True,
            result=result,
            status=terminal_status,
            result_done=result_done,
        )
        self.goal_handle.result_future = self.result_future

    def start(self):
        return True

    def get_status(self, **_kwargs):
        if self.status_hook is not None:
            self.status_hook()
        return {"task_budget_sec": 5.0}

    def wait_for_execute_plan_server(self, **_kwargs):
        if self.server_hook is not None:
            self.server_hook()
        return self.server_ready

    def send_agent_plan_goal(self, **kwargs):
        self.calls.append("send_agent_plan_goal")
        self.feedback_callback = kwargs["feedback_callback"]
        return _Future(self.goal_handle)

    def wait_future(self, future, *, interrupt_event=None, **_kwargs):
        if self.wait_hook is not None:
            self.wait_hook()
        if interrupt_event is not None and interrupt_event.is_set() and not future.done():
            return False
        return future.done()

    def cancel_agent_plan(self, task_id, **_kwargs):
        self.calls.append(("cancel_agent_plan", task_id))
        return {"accepted": True, "return_code": 0}

    def get_agent_plan_result(self, task_id, **_kwargs):
        self.calls.append(("get_agent_plan_result", task_id))
        return {"status": self.result_future.value.status, "result": dict(self.result)}

    def close(self):
        self.calls.append("close")


def _status(state, *, error_code=""):
    view = load_capability_catalog(config_path=CONFIG_PATH)
    return {
        "schema_version": 1,
        "robot_name": "so101_single_arm",
        "motion_authorized": True,
        "active_control_mode": "moveit_planning",
        "busy": state == "active",
        "active_task_id": "task-1" if state == "active" else "",
        "default_skill_timeout_sec": 30.0,
        "task_budget_sec": 180.0,
        "rpc_timeout_sec": 0.2,
        "config_digest": view["capability_digest"],
        "request_state": state,
        "request_error_code": error_code,
        "capabilities": [],
    }


def _runtime_snapshot() -> SkillSnapshot:
    config = load_robot_config_dict(CONFIG_PATH)
    return compile_local_snapshot(config, CONFIG_PATH)


def _gateway_status(*, request_state="", request_error_code="", ready=True):
    status = _status(request_state, error_code=request_error_code)
    status["capabilities"] = [
        {
            "name": "move_relative_ee",
            "ready": ready,
            "reason": "" if ready else "CAPABILITY_NOT_READY",
            "required_control_mode": "moveit_planning",
        }
    ]
    return status


def test_cancel_uses_task_id_only_lookup(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _CancelBridge([_status("active"), _status("active"), _status("terminal")])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(["--config-path", str(CONFIG_PATH), "cancel", "--task-id", "task-1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["data"]["already_terminal"] is False
    assert bridge.cancel_calls[0][0] == "task-1"
    assert all(call["task_id"] == "task-1" and call["payload_hash"] == "" for call in bridge.status_calls)


def test_cancel_terminal_record_is_idempotent(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _CancelBridge([_status("terminal")])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(["--config-path", str(CONFIG_PATH), "cancel", "--task-id", "task-1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["data"]["already_terminal"] is True
    assert bridge.cancel_calls == []


def test_cancel_unknown_record_returns_goal_not_found(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _CancelBridge([_status("")])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(["--config-path", str(CONFIG_PATH), "cancel", "--task-id", "task-1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"]["code"] == "GOAL_NOT_FOUND"
    assert bridge.cancel_calls == []


def test_execute_orders_gateway_checks_and_emits_one_public_result(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge([_gateway_status(), _gateway_status()])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-execute",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 0
    assert [event["event"] for event in events] == ["feedback", "result"]
    assert len([event for event in events if event["event"] == "result"]) == 1
    assert events[0]["data"] == {"state": "executing", "detail": "step 1 of 2"}
    assert events[1]["data"]["executed_step_count"] == 2
    assert "primitive" not in json.dumps(events)
    status_calls = [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "status"]
    assert status_calls[0][1]["task_id"] == ""
    assert status_calls[0][1]["payload_hash"] == ""
    assert status_calls[1][1]["task_id"] == "task-execute"
    assert status_calls[1][1]["payload_hash"] == events[1]["payload_hash"]
    assert bridge.calls.index("validate") < bridge.calls.index(status_calls[1])
    assert bridge.calls.index(status_calls[1]) < bridge.calls.index("wait_server") < bridge.calls.index("send")


def test_rejected_goal_uses_ledger_to_report_task_conflict(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="terminal", request_error_code="TASK_ID_CONFLICT"),
        ],
        accepted=False,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-conflict",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 3
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "TASK_ID_CONFLICT"


def _install_fake_signal_handlers(monkeypatch, cli, bridge, signal_number):
    handlers = {}

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    triggered = {"value": False}

    def trigger_signal(_seconds):
        if not triggered["value"]:
            triggered["value"] = True
            handlers[signal_number](signal_number, None)
            assert "cancel_goal" not in bridge.calls

    monkeypatch.setattr(cli.signal, "signal", install)
    monkeypatch.setattr(cli.time, "sleep", trigger_signal)
    bridge.goal_wait_hook = lambda: trigger_signal(0.0)


@pytest.mark.parametrize(("signal_number", "expected_exit"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_execute_signal_handler_only_sets_flag_and_waits_for_terminal(
    monkeypatch, capsys, signal_number, expected_exit
):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge([_gateway_status(), _gateway_status()], result_done=False)
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    _install_fake_signal_handlers(monkeypatch, cli, bridge, signal_number)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-signal",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == expected_exit
    assert bridge.calls.count("cancel_goal") == 1
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "SKILL_CANCELLED"


def test_execute_reports_unknown_stop_state_when_signal_cancel_does_not_converge(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [_gateway_status(), _gateway_status()],
        result_done=False,
        cancel_converges=False,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    _install_fake_signal_handlers(monkeypatch, cli, bridge, signal.SIGINT)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-unknown-stop",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 124
    assert events[-1]["data"]["error_code"] == "SKILL_CANCEL_TIMEOUT"
    assert events[-1]["data"]["message"] == "robot stop state is unknown"


def test_execute_goal_response_timeout_cancels_by_task_id_and_waits_for_terminal(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="active"),
            _gateway_status(request_state="terminal"),
        ],
        goal_response_timeout=True,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-goal-timeout",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    cancel_calls = [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "cancel_task"]
    assert exit_code == 124
    assert cancel_calls[0][1] == "task-goal-timeout"
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "RESULT_TIMEOUT"
    assert events[-1]["data"]["message"] == "task reached terminal after goal response timeout"


def test_execute_suppresses_feedback_after_terminal_result(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [_gateway_status(), _gateway_status()],
        late_feedback_on_close=True,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-late-feedback",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 0
    assert [event["event"] for event in events] == ["feedback", "result"]
    assert "private_joint" not in json.dumps(events)


def test_execute_signal_during_goal_response_wait_preserves_signal_exit(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="active"),
            _gateway_status(request_state="terminal"),
        ],
        goal_response_timeout=True,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    _install_fake_signal_handlers(monkeypatch, cli, bridge, signal.SIGINT)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-signal-goal-wait",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 130
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "SKILL_CANCELLED"


def test_execute_result_transport_error_cancels_and_emits_stable_result(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="active"),
            _gateway_status(request_state="terminal"),
        ],
        result_error=RuntimeError("transport failed"),
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-result-error",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    cancel_calls = [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "cancel_task"]
    assert exit_code == 4
    assert cancel_calls[0][1] == "task-result-error"
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "ROS_UNAVAILABLE"
    assert events[-1]["data"]["message"] == "terminal result unavailable"


def _execute_plan_args():
    return SimpleNamespace(
        plan_token="plan-token",
        confirmation_token="confirmation-token",
        task_id="agent-task-1",
        timeout_sec=5.0,
        plan_id="plan-1",
        plan_digest="digest-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="registry-1",
        expected_step_count=1,
    )


def _execute_plan_context():
    return SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 1.0, "task_budget_sec": 5.0}})


def test_execute_plan_restores_signal_handlers(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge()
    installed = {signal.SIGINT: object(), signal.SIGTERM: object()}
    original = dict(installed)

    def install(signum, handler):
        previous = installed[signum]
        installed[signum] = handler
        return previous

    monkeypatch.setattr(cli.signal, "signal", install)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 0
    assert installed == original
    assert [event["event"] for event in events] == ["result"]
    assert events[0]["data"]["success"] is True


def test_execute_plan_signal_during_acceptance_converges_to_terminal(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=5, success=False, error_code="SKILL_CANCELLED")
    bridge.goal_handle.result_future.completed = False
    pending_goal = _Future(bridge.goal_handle, done=False)
    bridge.send_agent_plan_goal = lambda **_kwargs: pending_goal
    handlers = {}

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(cli.signal, "signal", install)
    bridge.wait_hook = lambda: handlers[signal.SIGINT](signal.SIGINT, None)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 13
    assert events[-1]["data"]["success"] is False
    assert events[-1]["data"]["error_code"] == "SKILL_CANCELLED"
    assert ("cancel_agent_plan", "agent-task-1") in bridge.calls


def test_execute_plan_signal_race_preserves_success_exit(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge()
    handlers = {}

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(cli.signal, "signal", install)
    bridge.wait_hook = lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 0
    assert event["data"]["success"] is True


def test_execute_plan_signal_before_submission_converges_idempotent_retry(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=5, success=False, error_code="SKILL_CANCELLED")
    handlers = {}

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(cli.signal, "signal", install)

    def signal_twice():
        handlers[signal.SIGINT](signal.SIGINT, None)
        handlers[signal.SIGINT](signal.SIGINT, None)

    bridge.server_hook = signal_twice

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 13
    assert event["data"]["error_code"] == "SKILL_CANCELLED"
    assert "send_agent_plan_goal" not in bridge.calls
    assert [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "cancel_agent_plan"] == [
        ("cancel_agent_plan", "agent-task-1")
    ]


def test_execute_plan_signal_during_preflight_converges(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=5, success=False, error_code="SKILL_CANCELLED")
    handlers = {}

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(cli.signal, "signal", install)
    bridge.status_hook = lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 13
    assert event["data"]["error_code"] == "SKILL_CANCELLED"
    assert ("cancel_agent_plan", "agent-task-1") in bridge.calls


def test_execute_plan_server_unavailable_converges_existing_goal(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=5, success=False, error_code="SKILL_CANCELLED")
    bridge.server_ready = False
    monkeypatch.setattr(cli.signal, "signal", lambda _signum, _handler: signal.SIG_DFL)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 13
    assert event["data"]["error_code"] == "SKILL_CANCELLED"
    assert ("cancel_agent_plan", "agent-task-1") in bridge.calls


def test_execute_plan_rejected_goal_reports_rejection_without_cancel(monkeypatch, capsys):
    # The gateway rejects only malformed admissions and never creates the
    # goal (agent_plan_node._handle_goal; same-task retries take the
    # idempotent replay path instead of being rejected). Rejection therefore
    # implies there is no goal to converge, so the CLI mirrors
    # interactive_control's GOAL_REJECTED terminal: report the rejection
    # and never fire a speculative cancel.
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=5, success=False, error_code="SKILL_CANCELLED")
    bridge.goal_handle.accepted = False
    monkeypatch.setattr(cli.signal, "signal", lambda _signum, _handler: signal.SIG_DFL)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 13
    assert event["data"]["error_code"] == "GOAL_REJECTED"
    assert ("cancel_agent_plan", "agent-task-1") not in bridge.calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plan_id", "other-plan"),
        ("plan_digest", "other-digest"),
        ("actual_registry_epoch", "other-epoch"),
        ("actual_registry_generation", 2),
        ("actual_registry_digest", "other-registry"),
        ("completed_step_count", 0),
    ],
)
def test_execute_plan_rejects_mismatched_success_terminal(field, value, monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge()
    bridge.result[field] = value
    setattr(bridge.result_future.value.result, field, value)
    monkeypatch.setattr(cli.signal, "signal", lambda _signum, _handler: signal.SIG_DFL)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 15
    assert event["data"]["error_code"] == "SKILL_CANCEL_TIMEOUT"


@pytest.mark.parametrize(
    ("error_code", "expected_error"),
    [
        ("SKILL_CANCELLED", "SKILL_CANCEL_TIMEOUT"),
        ("SKILL_CANCEL_TIMEOUT", "SKILL_CANCEL_TIMEOUT"),
        ("GATEWAY_FINALIZATION_FAILED", "GATEWAY_FINALIZATION_FAILED"),
        ("SKILL_EXECUTION_BUSY", "SKILL_EXECUTION_BUSY"),
        ("NEW_CLEANUP_STATE_UNKNOWN", "SKILL_CANCEL_TIMEOUT"),
    ],
)
def test_execute_plan_rejects_unknown_aborted_terminal(error_code, expected_error, monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=6, success=False, error_code=error_code)
    monkeypatch.setattr(cli.signal, "signal", lambda _signum, _handler: signal.SIG_DFL)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 15
    assert event["data"]["error_code"] == expected_error


def test_execute_plan_fails_closed_for_unknown_pre_admission_plan(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=6, success=False, error_code="SKILL_AGENT_PLAN_NOT_FOUND")
    for field, value in (
        ("plan_id", ""),
        ("plan_digest", ""),
        ("actual_registry_epoch", ""),
        ("actual_registry_generation", 0),
        ("actual_registry_digest", ""),
    ):
        bridge.result[field] = value
        setattr(bridge.result_future.value.result, field, value)
    monkeypatch.setattr(cli.signal, "signal", lambda _signum, _handler: signal.SIG_DFL)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 15
    assert event["data"]["error_code"] == "SKILL_CANCEL_TIMEOUT"


def test_cancel_plan_queries_terminal_after_not_active():
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge()
    bridge.cancel_agent_plan = lambda task_id, **_kwargs: {"accepted": False, "return_code": 0}

    result = cli._run_cancel_plan(_execute_plan_args(), _execute_plan_context(), bridge)

    assert result["accepted"] is False
    assert result["terminal"] is True
    assert result["goal_status"] == 4


def test_execute_plan_nonterminal_result_cancels_once(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=2)
    monkeypatch.setattr(cli.signal, "signal", lambda _signum, _handler: signal.SIG_DFL)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 15
    assert event["data"]["error_code"] == "SKILL_CANCEL_TIMEOUT"
    assert [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "cancel_agent_plan"] == [
        ("cancel_agent_plan", "agent-task-1")
    ]


def test_execute_plan_waits_for_delayed_cancel_terminal(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=5, success=False, error_code="SKILL_CANCELLED")
    bridge.result_future.value.status = 2
    calls = 0

    def delayed_result(task_id, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": 2, "result": dict(bridge.result)}
        return {"status": 5, "result": dict(bridge.result)}

    bridge.get_agent_plan_result = delayed_result
    monkeypatch.setattr(cli.signal, "signal", lambda _signum, _handler: signal.SIG_DFL)

    exit_code = cli._run_execute_plan(_execute_plan_args(), _execute_plan_context(), bridge).exit_code

    event = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 13
    assert event["data"]["error_code"] == "SKILL_CANCELLED"
    assert calls == 2


def test_cancel_plan_mismatched_terminal_is_unknown():
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge()
    bridge.cancel_agent_plan = lambda task_id, **_kwargs: {"accepted": False, "return_code": 0}
    bridge.result["plan_digest"] = "other-digest"

    with pytest.raises(cli._CommandError) as exc_info:
        cli._run_cancel_plan(_execute_plan_args(), _execute_plan_context(), bridge)

    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"
    assert exc_info.value.exit_code == 15


def test_cancel_plan_waits_for_delayed_terminal():
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=5, success=False, error_code="SKILL_CANCELLED")
    calls = 0

    def delayed_result(task_id, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": 2, "result": dict(bridge.result)}
        return {"status": 5, "result": dict(bridge.result)}

    bridge.get_agent_plan_result = delayed_result

    result = cli._run_cancel_plan(_execute_plan_args(), _execute_plan_context(), bridge)

    assert result["terminal"] is True
    assert result["goal_status"] == 5
    assert result["result"]["error_code"] == "SKILL_CANCELLED"
    assert calls == 2


def test_closed_loop_runner_maps_terminal_states_to_process_exit_codes():
    from robot_skill_cli.closed_loop_runner import _terminal_exit_code

    assert _terminal_exit_code({"state": "succeeded"}) == 0
    assert _terminal_exit_code({"state": "failed"}) == 13
    assert _terminal_exit_code({"state": "unknown"}) == 15


@pytest.mark.parametrize("error_code", ["GATEWAY_FINALIZATION_FAILED", "SKILL_EXECUTION_BUSY"])
def test_cancel_plan_preserves_uncertain_terminal_code(error_code):
    from robot_skill_cli import cli

    bridge = _ExecutePlanBridge(terminal_status=6, success=False, error_code=error_code)

    with pytest.raises(cli._CommandError) as exc_info:
        cli._run_cancel_plan(_execute_plan_args(), _execute_plan_context(), bridge)

    assert exc_info.value.code == error_code
    assert exc_info.value.exit_code == 15
