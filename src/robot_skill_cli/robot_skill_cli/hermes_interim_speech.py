"""Asynchronously speak Hermes interim assistant messages."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from robot_skill_cli.hermes_tts_hook import extract_response, sanitize_for_tts


def _text(payload: dict[str, Any]) -> str:
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    value = extra.get("text") or payload.get("text")
    return value.strip() if isinstance(value, str) else extract_response(payload)


def _claim_path(payload: dict[str, Any], text: str) -> Path:
    session_id = str(payload.get("session_id") or "")
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    turn_id = str(extra.get("turn_id") or payload.get("turn_id") or "")
    # Keep one bounded interim announcement per turn.
    digest = hashlib.sha256(f"{session_id}|{turn_id}".encode()).hexdigest()
    root = Path(os.environ.get("IBROBOT_INTERIM_SPEECH_STATE", "/tmp/ibrobot-interim-speech"))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / digest


def _claim(path: Path) -> bool:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    os.close(descriptor)
    return True


def _release(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


def _cleanup_claims(root: Path, *, max_age_sec: int = 7 * 24 * 60 * 60) -> None:
    now = time.time()
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        try:
            if now - entry.stat().st_mtime > max_age_sec:
                entry.unlink()
        except FileNotFoundError:
            continue


def handle(payload: dict[str, Any]) -> None:
    if payload.get("hook_event_name") != "on_interim_message":
        return
    text = _text(payload)
    try:
        max_chars = max(1, int(os.environ.get("IBROBOT_INTERIM_MAX_CHARS", "500")))
    except ValueError:
        max_chars = 500
    text = text[:max_chars].rstrip()
    if not text or not sanitize_for_tts(text).strip():
        return
    speaker = os.environ.get("IBROBOT_INTERIM_SPEAKER", "")
    if not speaker:
        return
    claim_path = _claim_path(payload, text)
    _cleanup_claims(claim_path.parent)
    if not _claim(claim_path):
        return
    child_payload = json.dumps(
        {"session_id": payload.get("session_id", ""), "assistant_response": text},
        ensure_ascii=False,
    )
    try:
        process = subprocess.Popen(
            [speaker],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            close_fds=True,
        )
        if process.stdin is None:
            raise OSError("speaker stdin is unavailable")
        process.stdin.write(child_payload)
        process.stdin.close()
    except (OSError, subprocess.SubprocessError):
        _release(claim_path)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if isinstance(payload, dict):
            handle(payload)
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
