"""Scheduler-specific recording reset regressions."""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

from dataset_tools.record_cli import RecordCLI

_PRE_SCHEDULER_PARAMETERS = {
    "control_mode",
    "dispatcher_reset_service",
    "policy_reset_service",
    "restart_session_service",
    "reset_before_episode",
    "reset_timeout_sec",
}


def _declared_parameter_names(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "declare_parameter"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }


def test_record_cli_ros_parameter_surface_matches_pre_scheduler_baseline():
    source = Path(__file__).resolve().parents[1] / "dataset_tools" / "record_cli.py"

    assert _declared_parameter_names(source) == _PRE_SCHEDULER_PARAMETERS


def test_send_goal_does_not_start_recording_after_session_restart_failure():
    node = object.__new__(RecordCLI)
    node._goal_started_evt = threading.Event()
    node._goal_rejected_evt = threading.Event()
    node._episode_finished_evt = threading.Event()
    node._last_result_success = True
    node._last_result_message = "old"
    node._goal_lock = threading.RLock()
    node._closing = False
    node._owned_goal = None
    node._goal_pending = False
    node._mode_client = None
    node._should_reset_before_episode = lambda: True
    node.prepare_new_episode = lambda: False
    node._action_client = SimpleNamespace(
        send_goal_async=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recording must not start after restart_session failure")
        )
    )

    RecordCLI.send_goal(node, "pick")

    assert node._goal_rejected_evt.is_set()
    assert node._episode_finished_evt.is_set()
    assert not node._goal_started_evt.is_set()
    assert node._last_result_success is False
    assert node._last_result_message == "inference session restart failed"


def test_legacy_prepare_new_episode_keeps_dispatcher_then_policy_reset_fallback():
    node = object.__new__(RecordCLI)
    node._restart_session_client = None
    calls: list[str] = []
    node._reset_dispatcher_state = lambda: calls.append("dispatcher") or False
    node._reset_policy_state = lambda: calls.append("policy")

    assert RecordCLI.prepare_new_episode(node)
    assert calls == ["dispatcher", "policy"]
