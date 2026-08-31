from __future__ import annotations

from robot_skill_cli import hermes_interim_speech as speech


def test_interim_message_is_dispatched_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_INTERIM_SPEECH_STATE", str(tmp_path))
    monkeypatch.setenv("IBROBOT_INTERIM_SPEAKER", "/profile/hooks/ibrobot-speak")
    spawned = []

    class _Stdin:
        def write(self, value):
            spawned.append(value)

        def close(self):
            pass

    class _Process:
        stdin = _Stdin()

    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: _Process())
    payload = {
        "hook_event_name": "on_interim_message",
        "session_id": "session-1",
        "extra": {"turn_id": "turn-1", "text": "收到，准备执行。"},
    }

    speech.handle(payload)
    speech.handle(payload)

    assert len(spawned) == 1
    assert "收到，准备执行。" in spawned[0]


def test_non_interim_event_is_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_INTERIM_SPEECH_STATE", str(tmp_path))
    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))

    speech.handle({"hook_event_name": "post_llm_call", "assistant_response": "完成。"})


def test_failed_speaker_start_can_be_retried(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_INTERIM_SPEECH_STATE", str(tmp_path))
    monkeypatch.setenv("IBROBOT_INTERIM_SPEAKER", "/profile/hooks/ibrobot-speak")
    attempts = 0

    class _Stdin:
        def write(self, _value):
            pass

        def close(self):
            pass

    class _Process:
        stdin = _Stdin()

    def fail_once(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("speaker unavailable")
        return _Process()

    monkeypatch.setattr(speech.subprocess, "Popen", fail_once)
    payload = {
        "hook_event_name": "on_interim_message",
        "session_id": "session-1",
        "extra": {"turn_id": "turn-retry", "text": "准备执行。"},
    }

    speech.handle(payload)
    speech.handle(payload)

    assert attempts == 2
