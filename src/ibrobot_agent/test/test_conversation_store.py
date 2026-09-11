from __future__ import annotations

from datetime import datetime, timezone

from ibrobot_agent.contracts import AgentRequest
from ibrobot_agent.conversation_store import SQLiteConversationStore


def _request(request_id: str, *, session_id: str = "session-1", reply_to_request_id: str | None = None) -> AgentRequest:
    return AgentRequest(
        schema_version=1,
        request_id=request_id,
        session_id=session_id,
        channel_id="agent_incubation",
        principal_id="local_operator",
        robot_scope="so101_single_arm",
        text=f"message-{request_id}",
        received_at=datetime.now(timezone.utc),
        reply_to_request_id=reply_to_request_id,
    )


def test_conversation_persists_and_isolates_sessions(tmp_path):
    path = tmp_path / "conversation.sqlite3"
    first = SQLiteConversationStore(path, max_turns=2)
    request = _request("r1")
    first.append(request)
    first.append_assistant(request, "answer")
    first.append(_request("r2", session_id="session-2"))
    first.close()

    reopened = SQLiteConversationStore(path, max_turns=2)
    try:
        assert reopened.context(request) == [
            {"role": "user", "content": "message-r1"},
            {"role": "assistant", "content": "answer"},
        ]
        assert reopened.context(_request("r3", session_id="session-2")) == [{"role": "user", "content": "message-r2"}]
    finally:
        reopened.close()


def test_conversation_enforces_bounded_history(tmp_path):
    store = SQLiteConversationStore(tmp_path / "conversation.sqlite3", max_turns=1)
    try:
        first = _request("r1")
        store.append(first)
        store.append_assistant(first, "answer")
        store.append(_request("r2"))
        assert store.context(first) == [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "message-r2"},
        ]
    finally:
        store.close()


def test_clarification_is_session_bound_and_consumed_once(tmp_path):
    store = SQLiteConversationStore(tmp_path / "conversation.sqlite3")
    try:
        original = _request("clarification-1")
        store.save_clarification(original, ["skill_name"])
        assert store.consume_clarification(_request("answer-1", reply_to_request_id="clarification-1")) == {
            "request_id": "clarification-1",
            "original_text": "message-clarification-1",
            "missing_fields": ["skill_name"],
        }
        assert store.consume_clarification(_request("answer-2", reply_to_request_id="clarification-1")) is None

        second = _request("clarification-2")
        store.save_clarification(second, ["skill_name"])
        assert (
            store.consume_clarification(
                _request("answer-3", session_id="different-session", reply_to_request_id="clarification-2")
            )
            is None
        )
    finally:
        store.close()
