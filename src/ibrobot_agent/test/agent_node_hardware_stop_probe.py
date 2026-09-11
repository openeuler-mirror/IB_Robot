"""Cancel one submitted SO-101 wave through the Agent node."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

LEDGER_PATH = "/tmp/opencode/ibrobot-agent-so101-hardware-stop/requests.sqlite3"


def _spin_until(node: Node, predicate, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return
    raise RuntimeError("timed out waiting for SO-101 hardware cancellation")


def _submitted(request_id: str) -> bool:
    connection = sqlite3.connect(LEDGER_PATH)
    row = connection.execute(
        "SELECT may_have_submitted FROM requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    connection.close()
    return row is not None and row[0] == 1


def main() -> None:
    rclpy.init()
    node = Node("agent_hardware_stop_probe")
    responses: list[dict] = []
    events: list[dict] = []
    subscriptions = [
        node.create_subscription(String, "/agent/response", lambda msg: responses.append(json.loads(msg.data)), 10),
        node.create_subscription(String, "/agent/event", lambda msg: events.append(json.loads(msg.data)), 10),
    ]
    request_publisher = node.create_publisher(String, "/agent/request", 10)
    control_publisher = node.create_publisher(String, "/agent/control", 10)
    ready_client = node.create_client(Trigger, "/ibrobot_agent_node/ready")
    try:
        if not ready_client.wait_for_service(timeout_sec=90.0):
            raise RuntimeError("Agent ready service is unavailable")
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            future = ready_client.call_async(Trigger.Request())
            _spin_until(node, future.done, 5.0)
            response = future.result()
            if response is not None and response.success:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Agent did not become ready")

        request_id = f"hardware-wave-stop-{uuid.uuid4().hex[:12]}"
        request = String()
        request.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "session_id": "hardware-stop-session",
                "text": "挥手",
            },
            ensure_ascii=False,
        )
        request_publisher.publish(request)
        _spin_until(node, lambda: _submitted(request_id), 30.0)

        stop = String()
        stop.data = json.dumps({"operation": "stop", "request_id": request_id})
        control_publisher.publish(stop)
        _spin_until(
            node,
            lambda: any(
                item.get("request_key", {}).get("request_id") == request_id and item.get("event_type") == "terminal"
                for item in events
            ),
            60.0,
        )
        relevant = [item for item in events if item.get("request_key", {}).get("request_id") == request_id]
        terminal = next(item for item in relevant if item.get("event_type") == "terminal")
        assert terminal["state"] == "CANCELLED"

        connection = sqlite3.connect(LEDGER_PATH)
        row = connection.execute(
            "SELECT state, may_have_submitted, task_ref_json FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        connection.close()
        assert row is not None and row[0] == "CANCELLED" and row[1] == 1 and row[2]
        print(json.dumps({"request_id": request_id, "events": relevant}, ensure_ascii=False, sort_keys=True))
    finally:
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
