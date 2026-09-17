from __future__ import annotations

import pytest

from ibrobot_agent.chat_tui import ChatState, _build_key_bindings, _event_matches_session, parse_local_command


class _DummyNode:
    def __init__(self, stop_result: bool = False):
        self.stop_result = stop_result

    def stop(self):
        return self.stop_result


def test_enter_and_escape_enter_bindings_are_defined():
    bindings = _build_key_bindings(_DummyNode())
    keys = {tuple(binding.keys) for binding in bindings.bindings}

    assert ("c-m",) in keys
    assert ("escape", "c-m") in keys


def test_parse_local_command_keeps_natural_language_intact():
    assert parse_local_command("先回安全位，然后点头") == ("message", "先回安全位，然后点头")


def test_parse_local_commands_are_local_and_case_insensitive():
    assert parse_local_command("/status") == ("status", None)
    assert parse_local_command("/STOP now") == ("stop", "now")


def test_chat_state_tracks_active_requests_and_stop_target():
    state = ChatState("session-1")
    state.register("request-1")
    state.register("request-2")

    assert state.last_request_id == "request-2"
    assert state.stop_target() == "request-2"

    state.finish("request-2")
    assert state.stop_target() == "request-1"


def test_chat_state_returns_empty_stop_target_when_idle():
    assert ChatState("session-1").stop_target() == ""


def test_cancelled_before_execution_is_terminal_for_chat_state():
    state = ChatState("session-1")
    state.register("request-1")
    event = {
        "request_key": {
            "request_id": "request-1",
            "channel_id": "agent_cli",
            "principal_id": "local_operator",
            "robot_scope": "so101_single_arm",
        },
        "state": "CANCELLED_BEFORE_EXECUTION",
    }

    assert _event_matches_session(event, state)
    state.finish(event["request_key"]["request_id"])
    assert state.stop_target() == ""


def test_chat_state_does_not_register_unknown_response_as_active():
    state = ChatState("session-1")
    state.register("request-1")
    state.finish("request-1")

    assert "request-2" not in state.known_request_ids
    assert state.stop_target() == ""


def test_event_filter_uses_request_key_fields_without_session_id():
    state = ChatState("session-1")
    state.register("request-1")
    event = {
        "request_key": {
            "request_id": "request-1",
            "channel_id": "agent_cli",
            "principal_id": "local_operator",
            "robot_scope": "so101_single_arm",
        }
    }

    assert _event_matches_session(event, state)
    event["request_key"]["request_id"] = "other-request"
    assert not _event_matches_session(event, state)


def test_event_filter_respects_identity_overrides_for_non_default_profiles():
    state = ChatState(
        "session-1",
        channel_id="agent_incubation",
        principal_id="operator-2",
        robot_scope="so101_hardware",
    )
    state.register("request-1")
    event = {
        "request_key": {
            "request_id": "request-1",
            "channel_id": "agent_incubation",
            "principal_id": "operator-2",
            "robot_scope": "so101_hardware",
        }
    }

    assert _event_matches_session(event, state)
    event["request_key"]["channel_id"] = "agent_cli"
    assert not _event_matches_session(event, state)


def test_identity_mismatch_is_rejected_without_finishing_request():
    state = ChatState("session-1")
    state.register("request-1")
    event = {
        "request_key": {
            "request_id": "request-1",
            "channel_id": "other",
            "principal_id": "operator-2",
            "robot_scope": "so101",
        },
        "state": "FAILED",
    }

    assert not _event_matches_session(event, state)
    assert state.stop_target() == "request-1"


def test_main_passes_identity_overrides_to_run_chat(monkeypatch):
    import sys

    from ibrobot_agent import chat_tui

    captured = {}
    monkeypatch.setattr(chat_tui, "run_chat", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["chat_tui", "--channel-id", "agent_incubation", "--principal-id", "operator-2", "--robot-scope", "so101_hw"],
    )

    chat_tui.main()

    assert captured["channel_id"] == "agent_incubation"
    assert captured["principal_id"] == "operator-2"
    assert captured["robot_scope"] == "so101_hw"


def test_main_keeps_defaults_when_identity_flags_are_absent(monkeypatch):
    import sys

    from ibrobot_agent import chat_tui

    captured = {}
    monkeypatch.setattr(chat_tui, "run_chat", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["chat_tui"])

    chat_tui.main()

    assert captured["channel_id"] is None
    assert captured["principal_id"] is None
    assert captured["robot_scope"] is None


def test_chat_tui_uses_enter_for_single_line_submission():
    # The production prompt is intentionally single-line: Enter submits and
    # keeps the same terminal behavior across readline and prompt_toolkit.
    import inspect

    from ibrobot_agent import chat_tui

    source = inspect.getsource(chat_tui.run_chat)
    assert "multiline=False" in source


def test_chat_output_uses_prompt_toolkit_stdout_for_prompt_visibility():
    import inspect

    from ibrobot_agent import chat_tui

    assert "_write_terminal_line" in inspect.getsource(chat_tui._print_output)


def test_chat_output_uses_prompt_toolkit_synchronized_printer():
    import inspect

    from ibrobot_agent import chat_tui

    source = inspect.getsource(chat_tui._write_terminal_line)
    assert "patch_stdout" in source
    assert "print(line, flush=True)" in source


def test_chat_prompt_loop_uses_prompt_toolkit_stdout_patch():
    import inspect

    from ibrobot_agent import chat_tui

    source = inspect.getsource(chat_tui.run_chat)
    assert "patch_stdout()" in source
    assert "async with in_terminal()" in source
    assert "session.app.output.flush()" in source


def test_chat_output_formats_all_message_kinds_as_strings(monkeypatch):
    from ibrobot_agent import chat_tui

    lines = []
    monkeypatch.setattr(chat_tui, "_write_terminal_line", lines.append)

    chat_tui._print_output("system", {"message": "ready"})
    chat_tui._print_output("response", {"request_id": "r1", "state": "RECEIVED", "message": ""})
    chat_tui._print_output(
        "event",
        {"event_type": "conversation", "state": "ANSWERED", "user_message": "你好"},
    )

    assert lines == ["[系统] ready", "[受理] r1 RECEIVED ", "[Agent] conversation ANSWERED: 你好"]
    assert all(isinstance(line, str) for line in lines)


def test_chat_startup_diagnostics_mentions_event_subscription():
    import inspect

    from ibrobot_agent import chat_tui

    assert "agent_event订阅已建立" in inspect.getsource(chat_tui._print_startup_diagnostics)


@pytest.mark.parametrize("failure", [None, "writer", "digest", "identity"])
def test_presentation_receipt_only_follows_exact_plan_flush(failure):
    from types import SimpleNamespace

    from ibrobot_agent.chat_tui import _render_output_item
    from ibrobot_agent.presentation import presentation_digest

    state = ChatState("session")
    state.register("request")
    plan = {
        "steps": [{"schema_version": 1, "skill_name": "nod_yes"}],
        "task_ref": {"task_id": "task", "plan_id": "plan", "registry_digest": "rdig"},
    }
    payload = {
        "request_key": {
            "request_id": "request",
            "channel_id": state.channel_id,
            "principal_id": state.principal_id,
            "robot_scope": state.robot_scope,
        },
        "event_type": "presentation",
        "detail": {
            "presentation": plan,
            "presentation_digest": presentation_digest(plan),
            "receipt_token": "token",
        },
    }
    calls = []
    node = SimpleNamespace(state=state, acknowledge_presentation=lambda _: calls.append("receipt"))

    def write_and_flush(line):
        assert "nod_yes" in line and "task" in line and "rdig" in line
        if failure == "writer":
            raise OSError("output failed")
        calls.append("flushed")

    if failure == "digest":
        payload["detail"]["presentation"]["steps"][0]["skill_name"] = "wave_hello"
    if failure == "identity":
        payload["request_key"]["principal_id"] = "other"
    if failure in {"digest", "writer"}:
        with pytest.raises((ValueError, OSError)):
            _render_output_item(node, "event", payload, writer=write_and_flush)
    else:
        _render_output_item(node, "event", payload, writer=write_and_flush)
    assert calls == ([] if failure else ["flushed", "receipt"])


def test_async_chat_renderer_flushes_before_receipt(monkeypatch):
    import asyncio

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    from ibrobot_agent import chat_tui
    from ibrobot_agent.presentation import presentation_digest

    calls = []

    class Output(DummyOutput):
        def write(self, data):
            if "[Exact plan]" in data:
                calls.append("plan_written")

        def flush(self):
            if calls and calls[-1] == "plan_written":
                calls.append("flushed")

    class LocalChat(chat_tui.AgentChatNode):
        def wait_ready(self, timeout_sec=60.0):
            self.state.connected = True

        def send(self, text):
            self.state.register("render-test")
            plan = {"steps": [{"schema_version": 1, "skill_name": "nod_yes"}]}
            self._output.put(
                (
                    "event",
                    {
                        "event_type": "presentation",
                        "request_key": {
                            "request_id": "render-test",
                            "channel_id": self.state.channel_id,
                            "principal_id": self.state.principal_id,
                            "robot_scope": self.state.robot_scope,
                        },
                        "detail": {
                            "presentation": plan,
                            "presentation_digest": presentation_digest(plan),
                            "receipt_token": "token",
                        },
                    },
                )
            )
            return "render-test"

        def acknowledge_presentation(self, payload):
            assert calls == ["plan_written", "flushed"]
            calls.append("receipt")
            self.state.finish("render-test")

    session = PromptSession(input=DummyInput(), output=Output())
    submitted = False

    async def prompt(*args, **kwargs):
        nonlocal submitted
        if not submitted:
            submitted = True
            return "nod"
        for _ in range(100):
            if "receipt" in calls:
                return "/quit"
            await asyncio.sleep(0.01)
        raise RuntimeError("renderer did not complete")

    monkeypatch.setattr(session, "prompt_async", prompt)
    monkeypatch.setattr(chat_tui, "PromptSession", lambda **kwargs: session)
    monkeypatch.setattr(chat_tui, "AgentChatNode", LocalChat)
    chat_tui.run_chat(session_id="render-test")
    assert calls == ["plan_written", "flushed", "receipt"]
