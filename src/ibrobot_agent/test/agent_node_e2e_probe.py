"""No-motion ROS probe for the SO-101 Agent incubation pipeline."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

REQUIRED_NODES = {
    "/agent_plan_node",
    "/ibrobot_agent_node",
    "/safety_guard_node",
    "/skill_executor_node",
}


def _spin_until(node: Node, predicate, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return
    raise RuntimeError("timed out waiting for Agent no-motion probe condition")


def main() -> None:
    rclpy.init()
    node = Node("agent_no_motion_probe")
    responses: list[dict] = []
    events: list[dict] = []
    subscriptions = [
        node.create_subscription(String, "/agent/response", lambda msg: responses.append(json.loads(msg.data)), 10),
        node.create_subscription(String, "/agent/event", lambda msg: events.append(json.loads(msg.data)), 10),
    ]
    request_publisher = node.create_publisher(String, "/agent/request", 10)
    ready_client = node.create_client(Trigger, "/ibrobot_agent_node/ready")
    try:
        _spin_until(
            node,
            lambda: (
                {f"{namespace.rstrip('/')}/{name}" for name, namespace in node.get_node_names_and_namespaces()}
                >= REQUIRED_NODES
            ),
            60.0,
        )
        if not ready_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError("Agent ready service is unavailable")
        ready_response = None
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            ready_future = ready_client.call_async(Trigger.Request())
            _spin_until(node, ready_future.done, 5.0)
            ready_response = ready_future.result()
            if ready_response is not None and ready_response.success:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Agent did not become ready: {getattr(ready_response, 'message', '')}")

        suffix = uuid.uuid4().hex[:12]
        request_ids = {
            "status": f"mock-status-{suffix}",
            "wave": f"mock-wave-{suffix}",
        }
        for request_id, text, terminal_event in (
            (request_ids["status"], "当前状态", "read_only"),
            (request_ids["wave"], "挥手", "dry_run_complete"),
        ):
            message = String()
            message.data = json.dumps(
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "session_id": f"mock-session-{suffix}",
                    "channel_id": "spoofed-channel",
                    "principal_id": "spoofed-principal",
                    "robot_scope": "spoofed-robot",
                    "text": text,
                },
                ensure_ascii=False,
            )
            request_publisher.publish(message)
            _spin_until(
                node,
                lambda request_id=request_id: any(item.get("request_id") == request_id for item in responses),
                10.0,
            )
            _spin_until(
                node,
                lambda request_id=request_id, terminal_event=terminal_event: any(
                    item.get("request_key", {}).get("request_id") == request_id
                    and item.get("event_type") == terminal_event
                    for item in events
                ),
                15.0,
            )

        expected_event_types = {"read_only", "proposal_ready", "dry_run_complete"}
        _spin_until(
            node,
            lambda: expected_event_types <= {item.get("event_type") for item in events},
            15.0,
        )
        relevant = [
            event for event in events if event.get("request_key", {}).get("request_id") in set(request_ids.values())
        ]
        assert relevant
        assert all(event["request_key"]["robot_scope"] == "so101_single_arm" for event in relevant)
        assert all(event["request_key"]["channel_id"] == "agent_incubation" for event in relevant)
        assert all(event["request_key"]["principal_id"] == "local_operator" for event in relevant)
        connection = sqlite3.connect("/tmp/ibrobot-agent-so101-test/requests.sqlite3")
        rows = connection.execute(
            "SELECT request_id, state, may_have_submitted, terminal_json FROM requests "
            "WHERE request_id IN (?, ?) ORDER BY request_id",
            (request_ids["status"], request_ids["wave"]),
        ).fetchall()
        connection.close()
        assert len(rows) == 2
        assert all(row[1] == "ANSWERED" and row[2] == 0 for row in rows)
        assert any("DRY_RUN_ONLY" in row[3] for row in rows)
        print(json.dumps({"responses": responses, "events": relevant}, ensure_ascii=False, sort_keys=True))
    finally:
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
