"""Execute exactly one approved SO-101 nod through Kimi and the Agent node."""

from __future__ import annotations

import json
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


def _spin_until(node: Node, predicate, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return
    raise RuntimeError("timed out waiting for Kimi SO-101 execution")


def main() -> None:
    rclpy.init()
    node = Node("agent_kimi_hardware_probe")
    responses: list[dict] = []
    events: list[dict] = []
    subscriptions = [
        node.create_subscription(String, "/agent/response", lambda msg: responses.append(json.loads(msg.data)), 10),
        node.create_subscription(String, "/agent/event", lambda msg: events.append(json.loads(msg.data)), 10),
    ]
    publisher = node.create_publisher(String, "/agent/request", 10)
    ready_client = node.create_client(Trigger, "/ibrobot_agent_node/ready")
    try:
        if not ready_client.wait_for_service(timeout_sec=90.0):
            raise RuntimeError("Agent ready service is unavailable")
        ready_response = None
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            future = ready_client.call_async(Trigger.Request())
            _spin_until(node, future.done, 5.0)
            ready_response = future.result()
            if ready_response is not None and ready_response.success:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Agent did not become ready: {getattr(ready_response, 'message', '')}")

        request_id = f"kimi-hardware-nod-{uuid.uuid4().hex[:12]}"
        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "session_id": "kimi-hardware-session",
                "text": "只执行一次点头动作，不要执行其他动作。",
            },
            ensure_ascii=False,
        )
        start = time.monotonic()
        publisher.publish(message)
        _spin_until(node, lambda: any(item.get("request_id") == request_id for item in responses), 10.0)
        _spin_until(
            node,
            lambda: any(
                item.get("request_key", {}).get("request_id") == request_id and item.get("event_type") == "terminal"
                for item in events
            ),
            150.0,
        )
        relevant = [item for item in events if item.get("request_key", {}).get("request_id") == request_id]
        terminal = next(item for item in relevant if item.get("event_type") == "terminal")
        assert terminal["state"] == "SUCCEEDED"
        assert {item["event_type"] for item in relevant} >= {"proposal_ready", "presentation", "terminal"}
        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "elapsed_ms": round((time.monotonic() - start) * 1000.0, 3),
                    "events": relevant,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    finally:
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
