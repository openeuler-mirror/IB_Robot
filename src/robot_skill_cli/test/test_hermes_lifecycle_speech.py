from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

from robot_skill_cli import hermes_lifecycle_speech as speech


def _payload(event: str, command: str, *, result: str | None = None) -> dict:
    extra = {"turn_id": "turn-1"}
    if result is not None:
        extra.update({"status": "ok", "result": result})
    return {
        "hook_event_name": event,
        "session_id": "session-1",
        "tool_name": "terminal",
        "tool_input": {"command": command},
        "extra": extra,
    }


def test_event_gating() -> None:
    assert speech._event(_payload("pre_tool_call", "robot-skill status")) == "status_check_started"
    assert speech._event(_payload("pre_tool_call", "robot-skill plan-workflow --text x")) == "planning_started"
    success = '{"ok":true,"command":"confirm-plan","data":{"confirmed":true},"error":null}'
    assert speech._event(_payload("post_tool_call", "robot-skill confirm-plan --x", result=success)) == "plan_confirmed"
    assert (
        speech._event(_payload("post_tool_call", "robot-skill confirm-plan --x", result='{"confirmed":false}')) is None
    )
    wrapped = f"Process exited with code 0\nFinal output:\n{success}"
    assert speech._event(_payload("post_tool_call", "robot-skill confirm-plan --x", result=wrapped)) == "plan_confirmed"
    assert speech._event(_payload("pre_tool_call", "robot-skill run-workflow --text x")) == "planning_started"
    workflow_result = '{"event":"workflow_terminal","data":{"result":{"success":true}}}'
    assert (
        speech._event(_payload("post_tool_call", "robot-skill run-workflow --text x", result=workflow_result)) is None
    )
    internal_confirmation = '{"ok":true,"command":"confirm-plan","data":{"confirmed":true}}'
    assert (
        speech._event(_payload("post_tool_call", "robot-skill confirm-plan --internal", result=internal_confirmation))
        == "plan_confirmed"
    )


def test_composite_workflow_can_emit_authorization_at_confirm_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_LIFECYCLE_SPEECH_STATE", str(tmp_path))
    spawned = []
    monkeypatch.setattr(speech, "_spawn_play", lambda *args: spawned.append(args))

    speech.notify_plan_authorized(session_id="task-1")

    assert len(spawned) == 1
    assert spawned[0][2] == "plan_confirmed"
    assert spawned[0][3] == speech.FALLBACK_COPY["plan_confirmed"]


def test_confirmation_requires_successful_confirm_plan_envelope() -> None:
    assert not speech._result_success(
        {"ok": False, "command": "confirm-plan", "data": {"confirmed": True}, "error": {"code": "FAILED"}}
    )
    assert not speech._result_success({"ok": True, "command": "execute-plan", "data": {"success": True}, "error": None})
    assert not speech._result_success('log line {"success":true}\n{"ok":false,"command":"confirm-plan","data":null}')


def test_handle_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_LIFECYCLE_SPEECH_STATE", str(tmp_path))
    spawned = []
    monkeypatch.setattr(speech, "_spawn_play", lambda *args: spawned.append(args))
    payload = _payload("pre_tool_call", "robot-skill status")
    speech.handle(payload)
    speech.handle(payload)
    assert len(spawned) == 1


def test_pre_llm_remembers_task_without_starting_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_LIFECYCLE_SPEECH_STATE", str(tmp_path))
    generated = []
    monkeypatch.setattr(speech, "_spawn_model", lambda *args: generated.append(args))
    payload = {
        "hook_event_name": "pre_llm_call",
        "session_id": "session-1",
        "extra": {"turn_id": "turn-1", "user_message": "打开夹爪"},
    }
    speech.handle(payload)
    assert generated == []

    speech.handle(_payload("pre_tool_call", "robot-skill status"))
    assert generated[0][-1] == "打开夹爪"


def test_model_generation_is_spawned_once_per_turn(tmp_path: Path, monkeypatch) -> None:
    spawned = []
    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))

    speech._spawn_model(tmp_path, "turn-1", "打开夹爪")
    speech._spawn_model(tmp_path, "turn-1", "打开夹爪")

    assert len(spawned) == 1


def test_generate_uses_hermes_model_route(tmp_path: Path, monkeypatch) -> None:
    selected_models = []

    class FakeLLMClientService:
        def __init__(self, *, model: str) -> None:
            selected_models.append(model)

        def reply(self, _prompt: str) -> dict:
            return {
                "status": "ok",
                "content": json.dumps(speech.FALLBACK_COPY, ensure_ascii=False),
            }

    package = types.ModuleType("embodied_agent")
    package.__path__ = []
    client_module = types.ModuleType("embodied_agent.llm_client_service")
    client_module.LLMClientService = FakeLLMClientService
    monkeypatch.setitem(sys.modules, "embodied_agent", package)
    monkeypatch.setitem(sys.modules, "embodied_agent.llm_client_service", client_module)

    output = tmp_path / "copy.json"
    speech._generate(output, "打开夹爪")

    assert selected_models == ["gpt-5.6-sol"]
    assert json.loads(output.read_text(encoding="utf-8")) == speech.FALLBACK_COPY


def test_hook_entrypoint_accepts_payload(tmp_path: Path) -> None:
    env = {**os.environ, "IBROBOT_LIFECYCLE_SPEECH_STATE": str(tmp_path), "PYTHONPATH": "src/robot_skill_cli"}
    payload = _payload("pre_tool_call", "robot-skill status")
    result = subprocess.run(
        [sys.executable, "-m", "robot_skill_cli.hermes_lifecycle_speech"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0
