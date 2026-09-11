from __future__ import annotations

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

    assert "print(line, flush=True)" in inspect.getsource(chat_tui._write_terminal_line)


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
