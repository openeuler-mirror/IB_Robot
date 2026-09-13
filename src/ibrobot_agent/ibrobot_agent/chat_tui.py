"""Interactive terminal UI for the incubating Agent node."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import rclpy
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ibrobot_agent.contracts import TERMINAL_REQUEST_STATES as TERMINAL_STATES

STOP_WORDS = {"停止", "别动", "停", "stop", "halt"}


@dataclass
class ChatState:
    """UI state only; execution truth remains in the Agent ledger."""

    session_id: str
    channel_id: str = "agent_cli"
    principal_id: str = "local_operator"
    robot_scope: str = "so101_single_arm"
    last_request_id: str = ""
    active_request_ids: set[str] = field(default_factory=set)
    known_request_ids: set[str] = field(default_factory=set)
    connected: bool = False
    identity_warned: bool = False

    def register(self, request_id: str) -> None:
        self.last_request_id = request_id
        self.known_request_ids.add(request_id)
        self.active_request_ids.add(request_id)

    def finish(self, request_id: str) -> None:
        self.active_request_ids.discard(request_id)

    def stop_target(self) -> str:
        if self.last_request_id in self.active_request_ids:
            return self.last_request_id
        return next(iter(self.active_request_ids), "")


def parse_local_command(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "message", stripped
    command, _, argument = stripped[1:].partition(" ")
    return command.casefold(), argument.strip() or None


class AgentChatNode(Node):
    """ROS transport adapter used by the terminal UI."""

    def __init__(
        self,
        *,
        context: Context,
        state: ChatState,
        output: queue.Queue[tuple[str, dict[str, Any]]],
        topics: dict[str, str],
        ros_spin: threading.Event | None = None,
    ) -> None:
        super().__init__("agent_chat", context=context, use_global_arguments=False)
        self.state = state
        self._output = output
        self._request_publisher = self.create_publisher(String, topics["request"], 10)
        self._control_publisher = self.create_publisher(String, topics["control"], 10)
        self._response_subscription = self.create_subscription(String, topics["response"], self._on_response, 10)
        self._event_subscription = self.create_subscription(String, topics["event"], self._on_event, 10)
        self._ready_client = self.create_client(Trigger, "/ibrobot_agent_node/ready")
        self._ros_spin = ros_spin

    @property
    def request_topic(self) -> str:
        return self._request_publisher.topic_name

    def wait_ready(self, timeout_sec: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and self.context.ok():
            if not self._ready_client.wait_for_service(timeout_sec=0.5):
                continue
            future = self._ready_client.call_async(Trigger.Request())
            while not future.done() and time.monotonic() < deadline and self.context.ok():
                if self._ros_spin is None:
                    time.sleep(0.1)
                else:
                    self._ros_spin.wait(0.1)
            if future.done():
                response = future.result()
                if response is not None and response.success:
                    self.state.connected = True
                    self._output.put(("system", {"message": response.message}))
                    return
            time.sleep(0.5)
        raise RuntimeError("Agent did not become ready")

    def send(self, text: str) -> str:
        request_id = f"cli-{uuid.uuid4().hex[:12]}"
        self.state.register(request_id)
        message = String()
        message.data = json.dumps(
            {"schema_version": 1, "request_id": request_id, "session_id": self.state.session_id, "text": text},
            ensure_ascii=False,
        )
        self._request_publisher.publish(message)
        return request_id

    def request_ids_from_ledger(self) -> set[str]:
        return set(self.state.known_request_ids)

    def stop(self) -> bool:
        request_id = self.state.stop_target()
        if not request_id:
            self._output.put(("system", {"message": "当前会话没有可停止的请求"}))
            return False
        message = String()
        message.data = json.dumps({"operation": "stop", "request_id": request_id})
        self._control_publisher.publish(message)
        self._output.put(("system", {"message": f"已发送停止请求：{request_id}"}))
        return True

    def _on_response(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self._output.put(("system", {"message": "收到无法解析的 Agent 响应"}))
            return
        request_id = str(payload.get("request_id", ""))
        if request_id not in self.state.known_request_ids:
            return
        if payload.get("state") in TERMINAL_STATES:
            self.state.finish(request_id)
        self._output.put(("response", payload))

    def _on_event(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self._output.put(("system", {"message": "收到无法解析的 Agent 事件"}))
            return
        key = payload.get("request_key", {})
        if not isinstance(key, dict):
            return
        if (
            key.get("request_id") not in self.state.known_request_ids
            or key.get("channel_id") != self.state.channel_id
            or key.get("principal_id") != self.state.principal_id
            or key.get("robot_scope") != self.state.robot_scope
        ):
            if not self.state.identity_warned and key.get("request_id") in self.state.known_request_ids:
                self.state.identity_warned = True
                self._output.put(
                    (
                        "system",
                        {
                            "message": (
                                "Agent 事件身份不匹配，已忽略；请检查 channel_id/principal_id/robot_scope "
                                f"（当前期望 {self.state.channel_id}/{self.state.principal_id}/{self.state.robot_scope}）"
                            )
                        },
                    )
                )
            return
        request_id = str(key.get("request_id", ""))
        if payload.get("state") in TERMINAL_STATES:
            self.state.finish(request_id)
        self._output.put(("event", payload))


def _print_output(kind: str, payload: dict[str, Any]) -> None:
    """Write asynchronous results through the prompt toolkit output proxy."""
    if kind == "system":
        line = f"[系统] {payload.get('message', '')}"
    elif kind == "response":
        line = f"[受理] {payload.get('request_id', '')} {payload.get('state', '')} {payload.get('message', '')}"
    else:
        line = f"[Agent] {payload.get('event_type', '')} {payload.get('state', '')}: {payload.get('user_message', '')}"
    _write_terminal_line(line)


def _write_terminal_line(line: str) -> None:
    with patch_stdout(raw=True):
        print(line, flush=True)


def _event_matches_session(payload: dict[str, Any], state: ChatState) -> bool:
    key = payload.get("request_key", {})
    return (
        isinstance(key, dict)
        and key.get("request_id") in state.known_request_ids
        and key.get("channel_id") == state.channel_id
        and key.get("principal_id") == state.principal_id
        and key.get("robot_scope") == state.robot_scope
    )


def _print_startup_diagnostics(node: AgentChatNode, state: ChatState) -> None:
    print(
        f"Agent 会话已就绪：{state.session_id} | agent_event订阅已建立 | 输入通道：{node.request_topic}",
        flush=True,
    )


def _build_key_bindings(node: AgentChatNode) -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event) -> None:
        """Submit on Enter; use Escape+Enter when a multiline message is needed."""
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def _ctrl_c(event) -> None:
        if not node.stop():
            event.current_buffer.reset()

    return bindings


def _drain_output(output: queue.Queue[tuple[str, dict[str, Any]]]) -> None:
    while True:
        try:
            kind, payload = output.get_nowait()
        except queue.Empty:
            return
        _print_output(kind, payload)


def run_chat(
    *,
    session_id: str | None = None,
    topics: dict[str, str] | None = None,
    channel_id: str | None = None,
    principal_id: str | None = None,
    robot_scope: str | None = None,
) -> None:
    context = Context()
    rclpy.init(context=context)
    output: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    state = ChatState(
        session_id=session_id or f"cli-session-{uuid.uuid4().hex[:12]}",
        channel_id=channel_id or "agent_cli",
        principal_id=principal_id or "local_operator",
        robot_scope=robot_scope or "so101_single_arm",
    )
    spin_stop = threading.Event()
    node = AgentChatNode(
        context=context,
        state=state,
        output=output,
        topics=topics
        or {
            "request": "/agent/request",
            "response": "/agent/response",
            "event": "/agent/event",
            "control": "/agent/control",
        },
        ros_spin=spin_stop,
    )
    print("正在连接 ibrobot_agent_node...", flush=True)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)

    def spin() -> None:
        while not spin_stop.is_set() and context.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                return

    output_stop = threading.Event()
    output_thread: threading.Thread | None = None

    def render_output() -> None:
        while not output_stop.is_set() or not output.empty():
            try:
                kind, payload = output.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "event" and not _event_matches_session(payload, state):
                continue
            _print_output(kind, payload)

    spin_thread = threading.Thread(target=spin, name="agent-chat-ros", daemon=True)
    try:
        spin_thread.start()
        node.wait_ready()
        session: PromptSession[str] = PromptSession(history=InMemoryHistory())
        _print_startup_diagnostics(node, state)
        print("输入自然语言；/stop 停止；/status 查询状态；/skills 查询技能；/help 帮助；/quit 退出。", flush=True)
        output_thread = threading.Thread(target=render_output, name="agent-chat-output", daemon=True)
        output_thread.start()
        with patch_stdout():
            while context.ok():
                try:
                    text = session.prompt(
                        HTML("<prompt>小智&gt; </prompt>"),
                        key_bindings=_build_key_bindings(node),
                        style=Style.from_dict({"prompt": "ansicyan bold"}),
                        multiline=False,
                        mouse_support=False,
                    )
                except (EOFError, KeyboardInterrupt):
                    if node.stop():
                        continue
                    break
                command, _ = parse_local_command(text)
                stripped = text.strip()
                if command in {"quit", "exit"} or stripped in {"退出", "exit", "quit"}:
                    break
                if command == "help":
                    print("快捷命令：/stop /status /skills /clear /help /quit；Ctrl-C 停止当前请求。")
                elif command == "clear":
                    session.history = InMemoryHistory()
                    print("已清空本次输入历史。")
                elif command == "stop" or stripped in STOP_WORDS:
                    node.stop()
                elif command == "status":
                    node.send("当前状态")
                elif command == "skills":
                    node.send("当前有哪些技能")
                elif stripped:
                    request_id = node.send(stripped)
                    print(f"[{request_id}] 已提交，等待 Agent 处理。", flush=True)
    finally:
        output_stop.set()
        spin_stop.set()
        if context.ok():
            context.shutdown()
        spin_thread.join(timeout=2.0)
        if output_thread is not None:
            output_thread.join(timeout=2.0)
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Agent terminal UI")
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--channel-id",
        default=None,
        help="Agent channel identity; must match the profile's embodied.agent.channel_id",
    )
    parser.add_argument(
        "--principal-id",
        default=None,
        help="Agent principal identity; must match the profile's embodied.agent.principal_id",
    )
    parser.add_argument(
        "--robot-scope",
        default=None,
        help="Agent robot scope; must match the robot config name used by the Agent node",
    )
    args = parser.parse_args()
    try:
        run_chat(
            session_id=args.session_id,
            channel_id=args.channel_id,
            principal_id=args.principal_id,
            robot_scope=args.robot_scope,
        )
    except Exception as exc:
        print(f"Agent chat 启动失败：{type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from exc


__all__ = ["AgentChatNode", "ChatState", "main", "parse_local_command", "run_chat"]


if __name__ == "__main__":
    main()
