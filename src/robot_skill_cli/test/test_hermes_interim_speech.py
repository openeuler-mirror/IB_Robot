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


def test_different_partial_text_is_coalesced_per_turn(tmp_path, monkeypatch) -> None:
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
    for text in ("正在观察。", "正在观察环境。", "准备执行。"):
        speech.handle(
            {
                "hook_event_name": "on_interim_message",
                "session_id": "session-1",
                "extra": {"turn_id": "turn-1", "text": text},
            }
        )

    assert len(spawned) == 1


def test_interim_text_is_bounded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_INTERIM_SPEECH_STATE", str(tmp_path))
    monkeypatch.setenv("IBROBOT_INTERIM_SPEAKER", "/profile/hooks/ibrobot-speak")
    monkeypatch.setenv("IBROBOT_INTERIM_MAX_CHARS", "4")
    sent = []

    class _Stdin:
        def write(self, value):
            sent.append(value)

        def close(self):
            pass

    class _Process:
        stdin = _Stdin()

    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: _Process())
    speech.handle(
        {
            "hook_event_name": "on_interim_message",
            "session_id": "session-1",
            "extra": {"turn_id": "turn-1", "text": "准备执行动作。"},
        }
    )

    assert len(sent) == 1
    assert '"assistant_response": "准备执行"' in sent[0]


def test_stale_claim_files_are_cleaned_up(tmp_path, monkeypatch) -> None:
    import os
    import time

    monkeypatch.setenv("IBROBOT_INTERIM_SPEECH_STATE", str(tmp_path))
    monkeypatch.setenv("IBROBOT_INTERIM_SPEAKER", "/profile/hooks/ibrobot-speak")
    stale = tmp_path / "stale-claim"
    stale.write_text("", encoding="utf-8")
    eight_days_ago = time.time() - 8 * 24 * 60 * 60
    os.utime(stale, (eight_days_ago, eight_days_ago))
    recent = tmp_path / "recent-claim"
    recent.write_text("", encoding="utf-8")
    spawned = []

    class _Stdin:
        def write(self, value):
            spawned.append(value)

        def close(self):
            pass

    class _Process:
        stdin = _Stdin()

    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: _Process())
    speech.handle(
        {
            "hook_event_name": "on_interim_message",
            "session_id": "session-1",
            "extra": {"turn_id": "turn-1", "text": "准备执行。"},
        }
    )

    assert not stale.exists()
    assert recent.exists()
    assert len(spawned) == 1
