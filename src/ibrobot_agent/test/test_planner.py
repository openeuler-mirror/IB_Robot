from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from ibrobot_agent.contracts import AgentRequest, PlannerIdentity
from ibrobot_agent.planner import (
    PlannerAdapter,
    PlannerParseError,
    RulePlanner,
    build_planner_messages,
    parse_planner_outcome,
)


def _request(text: str) -> AgentRequest:
    return AgentRequest(
        schema_version=1,
        request_id="request-1",
        session_id="session-1",
        channel_id="test",
        principal_id="operator",
        robot_scope="so101_single_arm",
        text=text,
        received_at=datetime.now(timezone.utc),
    )


@pytest.mark.parametrize("text", ["不要挥手", "你会挥手吗？", "如果让你点头会怎样？"])
def test_rule_planner_does_not_promote_negative_or_hypothetical_text(text):
    outcome = RulePlanner().plan(_request(text), {}, {}, threading.Event())

    assert outcome.kind == "conversation"
    assert not outcome.steps


def test_rule_planner_preserves_ordered_so101_workflow():
    outcome = RulePlanner().plan(_request("先回安全位，然后点头"), {}, {}, threading.Event())

    assert outcome.kind == "workflow"
    assert [step.skill_name for step in outcome.steps] == ["recover_safe_pose", "nod_yes"]


@pytest.mark.parametrize("text", ["please nod", "不同意", "再见"])
def test_rule_planner_does_not_match_aliases_inside_other_words(text):
    outcome = RulePlanner().plan(_request(text), {}, {}, threading.Event())

    if text == "please nod":
        assert [step.skill_name for step in outcome.steps] == ["nod_yes"]
    elif text == "不同意":
        assert [step.skill_name for step in outcome.steps] == ["shake_no"]
    else:
        assert outcome.kind == "rejected"


def test_rule_planner_asks_for_a_specific_skill():
    outcome = RulePlanner().plan(_request("帮我做一个动作"), {}, {}, threading.Event())

    assert outcome.kind == "needs_clarification"
    assert outcome.missing_fields == ("skill_name",)


def test_parser_rejects_duplicate_json_keys():
    with pytest.raises(PlannerParseError, match="duplicate key"):
        parse_planner_outcome('{"kind":"conversation","kind":"workflow","user_message":"x"}')


@pytest.mark.parametrize(
    "raw",
    [
        '{"kind":"conversation","user_message":"x","endpoint":"/unsafe"}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","endpoint":"/unsafe"}]}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","timeout_sec":"1.0"}]}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","timeout_sec":true}]}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","timeout_sec":NaN}]}',
    ],
)
def test_parser_rejects_unknown_fields_and_coercible_or_non_finite_numbers(raw):
    with pytest.raises(PlannerParseError):
        parse_planner_outcome(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"kind":"conversation","user_message":"x"}\n```',
        'prefix {"kind":"conversation","user_message":"x"}',
        '{"kind":"conversation","user_message":"x"} suffix',
    ],
)
def test_parser_rejects_wrappers_and_mixed_text(raw):
    with pytest.raises(PlannerParseError):
        parse_planner_outcome(raw)


class FakeModelClient:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def complete(self, messages, *, force_json):
        self.messages = messages
        assert force_json is True
        return self.response


def _planner_adapter(response):
    client = FakeModelClient(response)
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kwargs: [{"role": "user", "content": kwargs["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
    )
    return planner, client


def test_planner_adapter_parses_strict_json_response():
    planner, client = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"你好"}',
            "tool_calls": [],
        }
    )

    outcome = planner.plan(_request("你好"), {}, {}, threading.Event())

    assert outcome.kind == "conversation"
    assert client.messages == [{"role": "user", "content": "你好"}]


def test_planner_adapter_rejects_tool_calls():
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"你好"}',
            "tool_calls": [{"function": {"name": "unsafe"}}],
        }
    )

    with pytest.raises(PlannerParseError, match="must not return tool calls"):
        planner.plan(_request("你好"), {}, {}, threading.Event())


def test_planner_adapter_builds_prompt_with_conversation_and_catalog():
    client = FakeModelClient(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"普通回答"}',
            "tool_calls": [],
        }
    )
    planner = PlannerAdapter(
        client,
        prompt_builder=build_planner_messages,
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
    )
    planner.plan(
        _request("这是什么"),
        {"messages": [{"role": "user", "content": "上一轮"}]},
        {"skills": [{"name": "nod_yes"}]},
        threading.Event(),
    )

    assert "上一轮" in client.messages[0]["content"] or "上一轮" in client.messages[1]["content"]


def test_planner_adapter_supports_ordinary_knowledge_answer():
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"我是小智。"}',
            "tool_calls": [],
        }
    )

    outcome = planner.plan(_request("你的名字是什么？"), {}, {}, threading.Event())

    assert outcome.kind == "conversation"
    assert outcome.user_message == "我是小智。"


def test_planner_adapter_rejects_execution_fields_on_conversation():
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"普通回答","summary":"误用"}',
            "tool_calls": [],
        }
    )

    with pytest.raises(PlannerParseError, match="conversation outcomes cannot carry a summary"):
        planner.plan(_request("这是什么"), {}, {}, threading.Event())
