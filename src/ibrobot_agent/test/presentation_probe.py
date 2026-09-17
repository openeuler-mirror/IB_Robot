"""Display exactly the probe's approved step before acknowledging its presentation."""

from __future__ import annotations

import json
from types import SimpleNamespace

from std_msgs.msg import String

from ibrobot_agent.chat_tui import AgentChatNode, ChatState, _event_matches_session, _render_output_item


def display_probe_plan(node, events, request_id: str, skill_name: str, spin_until, timeout_sec: float) -> None:
    state = ChatState("probe")
    state.register(request_id)
    spin_until(
        node,
        lambda: any(
            e.get("event_type") == "presentation" and e.get("request_key", {}).get("request_id") == request_id
            for e in events
        ),
        timeout_sec,
    )
    event = next(
        e
        for e in events
        if e.get("event_type") == "presentation" and e.get("request_key", {}).get("request_id") == request_id
    )
    if not _event_matches_session(event, state):
        raise RuntimeError("presentation does not belong to the probe identity")
    expected = [
        {
            "schema_version": 1,
            "skill_name": skill_name,
            "target_name": "",
            "container_name": "",
            "place_name": "",
            "motion_direction": "",
            "motion_distance": 0.0,
            "timeout_sec": 0.0,
        }
    ]
    if event["detail"]["presentation"]["steps"] != expected:
        raise RuntimeError("presentation differs from the approved single-step probe")
    publisher = node.create_publisher(String, "/agent/control", 10)
    try:
        spin_until(node, lambda: publisher.get_subscription_count() > 0, 5.0)
        adapter = SimpleNamespace(state=state, _control_publisher=publisher)
        adapter.acknowledge_presentation = lambda payload: AgentChatNode.acknowledge_presentation(adapter, payload)
        _render_output_item(adapter, "event", event, writer=lambda line: print(line, flush=True))
        spin_until(
            node,
            lambda: any(
                e.get("event_type") == "presentation_rendered"
                and e.get("request_key", {}).get("request_id") == request_id
                for e in events
            ),
            5.0,
        )
    except Exception:
        publisher.publish(String(data=json.dumps({"operation": "stop", "request_id": request_id})))
        raise
    finally:
        node.destroy_publisher(publisher)
