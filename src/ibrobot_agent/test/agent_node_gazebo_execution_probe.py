"""Execute one deterministic SO-101 gesture through the Agent in Gazebo."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

import rclpy
from presentation_probe import display_probe_plan
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


def _spin_until(node: Node, predicate, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return
    raise RuntimeError("timed out waiting for Agent Gazebo execution")


def main() -> None:
    rclpy.init()
    node = Node("agent_gazebo_execution_probe")
    responses: list[dict] = []
    events: list[dict] = []
    subscriptions = [
        node.create_subscription(String, "/agent/response", lambda msg: responses.append(json.loads(msg.data)), 10),
        node.create_subscription(String, "/agent/event", lambda msg: events.append(json.loads(msg.data)), 10),
    ]
    publisher = node.create_publisher(String, "/agent/request", 10)
    control_publisher = node.create_publisher(String, "/agent/control", 10)
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

        request_id = f"gazebo-nod-{uuid.uuid4().hex[:12]}"
        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "session_id": "gazebo-execution-session",
                "text": "点头",
            },
            ensure_ascii=False,
        )
        publisher.publish(message)
        _spin_until(node, lambda: any(item.get("request_id") == request_id for item in responses), 10.0)
        display_probe_plan(node, events, request_id, "nod_yes", _spin_until, 30.0)
        _spin_until(
            node,
            lambda: any(
                item.get("request_key", {}).get("request_id") == request_id and item.get("event_type") == "terminal"
                for item in events
            ),
            90.0,
        )
        relevant = [item for item in events if item.get("request_key", {}).get("request_id") == request_id]
        terminal = next(item for item in relevant if item.get("event_type") == "terminal")
        assert terminal["state"] == "SUCCEEDED"
        assert {item["event_type"] for item in relevant} >= {"proposal_ready", "presentation", "terminal"}

        connection = sqlite3.connect("/tmp/ibrobot-agent-so101-gazebo/requests.sqlite3")
        row = connection.execute(
            "SELECT state, may_have_submitted, task_ref_json FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        connection.close()
        assert row is not None
        assert row[0] == "SUCCEEDED"
        assert row[1] == 1
        assert row[2]

        # Re-delivering the exact request returns the retained result and must
        # not create another submission event or action goal.
        publisher.publish(message)
        _spin_until(
            node,
            lambda: sum(item.get("request_id") == request_id for item in responses) >= 2,
            10.0,
        )
        connection = sqlite3.connect("/tmp/ibrobot-agent-so101-gazebo/requests.sqlite3")
        submission_count = connection.execute(
            "SELECT COUNT(*) FROM request_events WHERE request_id = ? AND event_type = 'mark_submitted'",
            (request_id,),
        ).fetchone()[0]
        connection.close()
        assert submission_count == 1

        stop_request_id = f"gazebo-wave-stop-{uuid.uuid4().hex[:12]}"
        wave_message = String()
        wave_message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": stop_request_id,
                "session_id": "gazebo-execution-session",
                "text": "挥手",
            },
            ensure_ascii=False,
        )
        publisher.publish(wave_message)
        display_probe_plan(node, events, stop_request_id, "wave_hello", _spin_until, 30.0)
        _spin_until(
            node,
            lambda: any(
                item.get("request_key", {}).get("request_id") == stop_request_id
                and item.get("event_type") == "presentation"
                for item in events
            ),
            30.0,
        )
        submission_deadline = time.monotonic() + 30.0
        while time.monotonic() < submission_deadline:
            connection = sqlite3.connect("/tmp/ibrobot-agent-so101-gazebo/requests.sqlite3")
            submitted = connection.execute(
                "SELECT may_have_submitted FROM requests WHERE request_id = ?",
                (stop_request_id,),
            ).fetchone()
            connection.close()
            if submitted is not None and submitted[0] == 1:
                break
            rclpy.spin_once(node, timeout_sec=0.05)
        else:
            raise RuntimeError("wave request did not reach the durable submission boundary")
        stop_message = String()
        stop_message.data = json.dumps({"operation": "stop", "request_id": stop_request_id})
        control_publisher.publish(stop_message)
        _spin_until(
            node,
            lambda: any(
                item.get("request_key", {}).get("request_id") == stop_request_id
                and item.get("event_type") == "terminal"
                for item in events
            ),
            60.0,
        )
        stop_events = [item for item in events if item.get("request_key", {}).get("request_id") == stop_request_id]
        stop_terminal = next(item for item in stop_events if item.get("event_type") == "terminal")
        assert stop_terminal["state"] == "CANCELLED"
        print(
            json.dumps(
                {
                    "success_request_id": request_id,
                    "success_events": relevant,
                    "stop_request_id": stop_request_id,
                    "stop_events": stop_events,
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
