"""Asynchronous Hermes robot-task lifecycle speech.

The shell hook is intentionally tiny at the protocol boundary.  Each Hermes
hook invocation is a short-lived process, so durable per-turn state lives in a
small runtime directory and work is handed to detached child processes.  No
child is on the robot control critical path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

FALLBACK_COPY = {
    "status_check_started": "任务已收到，小鸭子正在观察环境嘎嘎。",
    "planning_started": "看清楚了，让小鸭子先想想怎么执行。",
    "plan_authorized": "小鸭子知道了，开始工作嘎嘎。",
}
EVENTS = frozenset(FALLBACK_COPY)
_LOG_PATH = Path(os.environ.get("IBROBOT_LIFECYCLE_SPEECH_LOG", "/tmp/hermes-lifecycle-speech.log"))


def _log(message: str) -> None:
    try:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with _LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} {message}\n")
    except OSError:
        pass


def _state_root() -> Path:
    root = Path(os.environ.get("IBROBOT_LIFECYCLE_SPEECH_STATE", "/tmp/ibrobot-lifecycle-speech"))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _turn_key(payload: dict[str, Any]) -> str:
    raw = str(payload.get("session_id") or "")
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    raw += "|" + str(extra.get("turn_id") or "")
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _command(payload: dict[str, Any]) -> str:
    value = payload.get("tool_input")
    if isinstance(value, dict):
        value = value.get("command") or value.get("cmd") or value.get("input")
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def _event(payload: dict[str, Any]) -> str | None:
    command = _command(payload)
    if "robot-skill" not in command:
        return None
    hook_event = str(payload.get("hook_event_name") or "")
    if hook_event == "pre_tool_call":
        if " status" in command or command.rstrip().endswith(" status"):
            return "status_check_started"
        if " plan-workflow" in command:
            return "planning_started"
        if " run-workflow" in command:
            return "planning_started"
    if hook_event == "post_tool_call" and " confirm-plan" in command:
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        if str(extra.get("status") or "") == "ok" and _result_success(extra.get("result")):
            return "plan_authorized"
    if hook_event == "post_tool_call" and " run-workflow" in command:
        return None
    return None


def notify_plan_authorized(*, session_id: str, turn_id: str = "") -> None:
    """Emit the authorization event when a composite workflow confirms internally."""
    handle(
        {
            "hook_event_name": "post_tool_call",
            "session_id": session_id,
            "tool_name": "terminal",
            "tool_input": {"command": "robot-skill confirm-plan --internal"},
            "extra": {
                "turn_id": turn_id or session_id,
                "status": "ok",
                "result": '{"ok":true,"command":"confirm-plan","data":{"confirmed":true}}',
            },
        }
    )


def _result_success(result: Any) -> bool:
    def confirmed(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if "ok" in value:
            data = value.get("data")
            return (
                value.get("ok") is True
                and value.get("command") == "confirm-plan"
                and isinstance(data, dict)
                and (data.get("success") is True or data.get("confirmed") is True)
            )
        return value.get("success") is True or value.get("confirmed") is True

    if isinstance(result, dict):
        if result.get("event") == "workflow_terminal":
            result = result.get("data")
        if isinstance(result, dict) and isinstance(result.get("result"), dict):
            result = result["result"]
        return confirmed(result)
    if not isinstance(result, str):
        return False
    candidate = result.rpartition("Final output:")[2] if "Final output:" in result else result
    try:
        return _result_success(json.loads(candidate.strip()))
    except json.JSONDecodeError:
        return False


def _claim(root: Path, turn: str, event: str) -> bool:
    marker = root / f"{turn}.{event}.sent"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def _copy_path(root: Path, turn: str) -> Path:
    return root / f"{turn}.json"


def _task_path(root: Path, turn: str) -> Path:
    return root / f"{turn}.task"


def _remember_task(root: Path, turn: str, payload: dict[str, Any]) -> None:
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    task = extra.get("user_message")
    if not isinstance(task, str) or not task.strip():
        return
    _task_path(root, turn).write_text(task.strip(), encoding="utf-8")


def _remembered_task(root: Path, turn: str) -> str:
    try:
        return _task_path(root, turn).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _extract_task(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens[:-1]):
        if token == "--text":
            return tokens[index + 1]
    return ""


def _spawn_model(root: Path, turn: str, task: str) -> None:
    ready = root / f"{turn}.generation-started"
    try:
        fd = os.open(ready, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    os.close(fd)
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "robot_skill_cli.hermes_lifecycle_speech",
            "--generate",
            str(_copy_path(root, turn)),
            task,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _generate(output: Path, task: str) -> None:
    from embodied_agent.llm_client_service import LLMClientService

    prompt = (
        "请为机器人任务生成三个简短的中文生命周期播报，严格只输出 JSON 对象，键为 "
        "status_check_started、planning_started、plan_authorized。"
        "每句10到28个汉字；允许零到两句自然提及用户任务，不要每句都提。"
        "第一句表示收到并观察环境，第二句表示正在规划，第三句只能表示即将执行，"
        "不能声称已经完成，不要输出解释或 Markdown。用户任务：" + task
    )
    # Use the same configured Xunxing route as Hermes.  The shared service
    # defaults to qwen-vl-plus, which requires a separate Aliyun key and is
    # not available in the Hermes runtime environment.
    result = LLMClientService(model="gpt-5.6-sol").reply(prompt)
    if isinstance(result, dict) and result.get("status") != "ok":
        raise RuntimeError(str(result.get("error") or "lifecycle copy model request failed"))
    content = result.get("content", "") if isinstance(result, dict) else ""
    if not isinstance(content, str):
        raise RuntimeError("lifecycle copy model returned no text")
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("lifecycle copy model returned invalid JSON")
    value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("lifecycle copy model returned non-object")
    copy = {key: str(value[key]).strip() for key in EVENTS}
    if any(not text or len(text) > 80 for text in copy.values()):
        raise RuntimeError("lifecycle copy model returned invalid phrase")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(copy, stream, ensure_ascii=False)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    _log(f"COPY_READY turn={output.stem} task_chars={len(task)}")


def _copy(root: Path, turn: str, event: str) -> str:
    try:
        value = json.loads(_copy_path(root, turn).read_text(encoding="utf-8"))
        text = value.get(event)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return FALLBACK_COPY[event]


def _spawn_play(root: Path, turn: str, event: str, text: str) -> None:
    subprocess.Popen(
        [sys.executable, "-m", "robot_skill_cli.hermes_lifecycle_speech", "--play", turn, event, text],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _play(turn: str, event: str, text: str) -> None:
    _log(f"PLAY_START turn={turn} event={event} chars={len(text)}")
    payload = json.dumps({"session_id": turn, "assistant_response": text}, ensure_ascii=False)
    hook = os.environ.get("IBROBOT_HERMES_TTS_HOOK", "")
    timeout = float(os.environ.get("IBROBOT_LIFECYCLE_TTS_TIMEOUT_SEC", "360"))
    if hook:
        subprocess.run([hook], input=payload, text=True, timeout=timeout, check=False)
    else:
        subprocess.run(
            [sys.executable, "-m", "robot_skill_cli.hermes_tts_hook"],
            input=payload,
            text=True,
            timeout=timeout,
            check=False,
        )
    _log(f"PLAY_DONE turn={turn} event={event}")


def handle(payload: dict[str, Any]) -> None:
    root = _state_root()
    turn = _turn_key(payload)
    if payload.get("hook_event_name") == "pre_llm_call":
        _remember_task(root, turn, payload)
        return
    event = _event(payload)
    if event is None:
        return
    command = _command(payload)
    if event in {"status_check_started", "planning_started"}:
        _spawn_model(root, turn, _remembered_task(root, turn) or _extract_task(command))
    if not _claim(root, turn, event):
        return
    _log(f"EVENT turn={turn} event={event}")
    _spawn_play(root, turn, event, _copy(root, turn, event))


def main() -> int:
    if "--generate" in sys.argv:
        try:
            _generate(Path(sys.argv[2]), sys.argv[3])
            return 0
        except Exception as exc:
            _log(f"COPY_ERROR type={type(exc).__name__}")
            return 1
    if "--play" in sys.argv:
        try:
            _play(sys.argv[2], sys.argv[3], sys.argv[4])
            return 0
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            _log(f"PLAY_ERROR type={type(exc).__name__}")
            return 1
    try:
        raw = sys.stdin.read()
        if raw.strip():
            payload = json.loads(raw)
            if isinstance(payload, dict):
                handle(payload)
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError, ValueError):
        # Lifecycle speech is best-effort and must never block a robot tool.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
