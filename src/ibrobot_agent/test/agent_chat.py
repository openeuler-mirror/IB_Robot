"""Small terminal channel for the incubating Agent node."""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
import uuid

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class AgentChat(Node):
    def __init__(
        self,
        *,
        session_id: str,
        request_topic: str,
        response_topic: str,
        event_topic: str,
        control_topic: str,
        context: Context,
    ):
        super().__init__("agent_chat", context=context, use_global_arguments=False)
        self.session_id = session_id
        self.last_request_id = ""
        self._last_request_at = 0.0
        self.responses: list[dict] = []
        self.events: list[dict] = []
        self.request_publisher = self.create_publisher(String, request_topic, 10)
        self.control_publisher = self.create_publisher(String, control_topic, 10)
        self.response_subscription = self.create_subscription(String, response_topic, self._on_response, 10)
        self.event_subscription = self.create_subscription(String, event_topic, self._on_event, 10)
        self.ready_client = self.create_client(Trigger, "/ibrobot_agent_node/ready")

    def _on_response(self, message: String) -> None:
        payload = json.loads(message.data)
        self.responses.append(payload)
        print(f"[agent] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)

    def _on_event(self, message: String) -> None:
        payload = json.loads(message.data)
        self.events.append(payload)
        print(
            f"[event] {payload.get('event_type')} {payload.get('state')}: {payload.get('user_message', '')}",
            flush=True,
        )

    def _spin_until_request_terminal(self, executor: SingleThreadedExecutor, request_id: str) -> None:
        deadline = time.monotonic() + 180.0
        terminal_states = {"ANSWERED", "SUCCEEDED", "FAILED", "CANCELLED", "UNKNOWN"}
        while time.monotonic() < deadline and self.context.ok():
            executor.spin_once(timeout_sec=0.1)
            if any(
                event.get("request_key", {}).get("request_id") == request_id and event.get("state") in terminal_states
                for event in self.events
            ):
                return

    def _session_event(self, event: dict) -> bool:
        return (
            event.get("request_key", {}).get("channel_id") == "agent_cli"
            and event.get("request_key", {}).get("principal_id") == "local_operator"
        )

    def wait_ready(self, executor: SingleThreadedExecutor, timeout_sec: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if not self.context.ok():
                raise RuntimeError("ROS context is not available")
            try:
                if not self.ready_client.wait_for_service(timeout_sec=0.5):
                    continue
            except RuntimeError as exc:
                raise RuntimeError("/ibrobot_agent_node/ready is unavailable") from exc
            future = self.ready_client.call_async(Trigger.Request())
            while not future.done() and time.monotonic() < deadline and self.context.ok():
                executor.spin_once(timeout_sec=0.1)
            response = future.result()
            if response is not None and response.success:
                print(f"Agent ready: {response.message}", flush=True)
                return
            time.sleep(0.5)
        raise RuntimeError("Agent did not become ready")

    def send(self, text: str) -> None:
        request_id = f"cli-{uuid.uuid4().hex[:12]}"
        self.last_request_id = request_id
        self._last_request_at = time.monotonic()
        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "session_id": self.session_id,
                "text": text,
            },
            ensure_ascii=False,
        )
        self.request_publisher.publish(message)

    def get_latest_request_id(self) -> str:
        return self.last_request_id

    def stop(self) -> None:
        request_id = self.last_request_id
        if not request_id or time.monotonic() - self._last_request_at > 300.0:
            print("当前会话没有可停止的请求", flush=True)
            return
        message = String()
        message.data = json.dumps({"operation": "stop", "request_id": request_id})
        self.control_publisher.publish(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--request-topic", default="/agent/request")
    parser.add_argument("--response-topic", default="/agent/response")
    parser.add_argument("--event-topic", default="/agent/event")
    parser.add_argument("--control-topic", default="/agent/control")
    args = parser.parse_args()
    context = Context()
    rclpy.init(context=context)
    node = AgentChat(
        context=context,
        session_id=args.session_id or f"cli-session-{uuid.uuid4().hex[:12]}",
        request_topic=args.request_topic,
        response_topic=args.response_topic,
        event_topic=args.event_topic,
        control_topic=args.control_topic,
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        node.wait_ready(executor)
        print("输入自然语言；输入‘停止’发送停止旁路；输入‘退出’结束会话。", flush=True)
        while context.ok():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            executor.spin_once(timeout_sec=0.1)
            if not readable:
                continue
            raw_line = sys.stdin.readline()
            if raw_line == "":
                break
            text = raw_line.strip()
            if not text:
                continue
            if text in {"退出", "exit", "quit"}:
                break
            if text in {"停止", "别动", "停", "stop", "halt"}:
                node.stop()
            else:
                node.send(text)
                node._spin_until_request_terminal(executor, node.last_request_id)
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)


if __name__ == "__main__":
    main()
