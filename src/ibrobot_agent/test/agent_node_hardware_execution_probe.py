"""Execute exactly one approved SO-101 nod through the Agent node."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime

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
    raise RuntimeError("timed out waiting for SO-101 hardware Agent execution")


def _milliseconds(start_ns: int, end_ns: int) -> float:
    return round((end_ns - start_ns) / 1_000_000.0, 3)


def _event_milliseconds(events_by_type: dict[str, datetime], start: str, end: str) -> float:
    return round((events_by_type[end] - events_by_type[start]).total_seconds() * 1000.0, 3)


def main() -> None:
    rclpy.init()
    node = Node("agent_hardware_execution_probe")
    responses: list[dict] = []
    events: list[dict] = []
    response_arrivals_ns: dict[str, int] = {}
    event_arrivals_ns: dict[tuple[str, str], int] = {}

    def on_response(message: String) -> None:
        payload = json.loads(message.data)
        responses.append(payload)
        request_id = payload.get("request_id")
        if request_id:
            response_arrivals_ns.setdefault(str(request_id), time.monotonic_ns())

    def on_event(message: String) -> None:
        payload = json.loads(message.data)
        events.append(payload)
        request_id = payload.get("request_key", {}).get("request_id")
        event_type = payload.get("event_type")
        if request_id and event_type:
            event_arrivals_ns.setdefault((str(request_id), str(event_type)), time.monotonic_ns())

    subscriptions = [
        node.create_subscription(String, "/agent/response", on_response, 10),
        node.create_subscription(String, "/agent/event", on_event, 10),
    ]
    publisher = node.create_publisher(String, "/agent/request", 10)
    ready_client = node.create_client(Trigger, "/ibrobot_agent_node/ready")
    probe_start_wall_ns = time.time_ns()
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
        ready_wall_ns = time.time_ns()

        request_id = f"hardware-nod-{uuid.uuid4().hex[:12]}"
        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "session_id": "hardware-e2e-session",
                "text": "点头",
            },
            ensure_ascii=False,
        )
        publish_ns = time.monotonic_ns()
        publisher.publish(message)
        _spin_until(node, lambda: any(item.get("request_id") == request_id for item in responses), 10.0)
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
        connection = sqlite3.connect("/tmp/opencode/ibrobot-agent-so101-hardware/requests.sqlite3")
        row = connection.execute(
            "SELECT state, may_have_submitted, task_ref_json FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        event_rows = connection.execute(
            "SELECT event_type, created_at FROM request_events WHERE request_id = ? ORDER BY sequence",
            (request_id,),
        ).fetchall()
        connection.close()
        assert row is not None and row[0] == "SUCCEEDED" and row[1] == 1 and row[2]
        ledger_times: dict[str, datetime] = {}
        for event_type, created_at in event_rows:
            ledger_times.setdefault(str(event_type), datetime.fromisoformat(str(created_at)))
        required_ledger_events = {
            "admit",
            "begin_planning",
            "mark_proposal_ready",
            "bind_task",
            "record_confirmation",
            "mark_submitted",
            "finish",
        }
        assert required_ledger_events <= set(ledger_times)
        launch_start_wall_ns = int(os.environ.get("AGENT_PIPELINE_START_WALL_NS", probe_start_wall_ns))
        timings_ms = {
            "pipeline_start_to_agent_ready": _milliseconds(launch_start_wall_ns, ready_wall_ns),
            "probe_start_to_agent_ready": _milliseconds(probe_start_wall_ns, ready_wall_ns),
            "publish_to_accept_response": _milliseconds(publish_ns, response_arrivals_ns[request_id]),
            "publish_to_proposal_event": _milliseconds(publish_ns, event_arrivals_ns[(request_id, "proposal_ready")]),
            "publish_to_presentation_event": _milliseconds(publish_ns, event_arrivals_ns[(request_id, "presentation")]),
            "publish_to_terminal_event": _milliseconds(publish_ns, event_arrivals_ns[(request_id, "terminal")]),
            "ledger_admission_to_planning": _event_milliseconds(ledger_times, "admit", "begin_planning"),
            "ledger_planning": _event_milliseconds(ledger_times, "begin_planning", "mark_proposal_ready"),
            "ledger_gateway_prepare_to_presentation": _event_milliseconds(
                ledger_times, "mark_proposal_ready", "bind_task"
            ),
            "ledger_presentation_to_confirmation": _event_milliseconds(
                ledger_times, "bind_task", "record_confirmation"
            ),
            "ledger_confirmation_to_submission": _event_milliseconds(
                ledger_times, "record_confirmation", "mark_submitted"
            ),
            "ledger_robot_execution": _event_milliseconds(ledger_times, "mark_submitted", "finish"),
            "ledger_admission_to_terminal": _event_milliseconds(ledger_times, "admit", "finish"),
        }
        print(
            json.dumps(
                {"request_id": request_id, "events": relevant, "timings_ms": timings_ms},
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
