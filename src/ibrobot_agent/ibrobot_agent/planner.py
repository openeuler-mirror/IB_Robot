"""Strict planner-response parsing for ibrobot_agent.

The planner layer accepts only a single JSON object and turns it into the
typed ``PlannerOutcome`` contract. Anything else fails closed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import PlannerIdentity, PlannerOutcome, planner_outcome_from_mapping


class PlannerParseError(ValueError):
    """Raised when a model response cannot be parsed into a planner outcome."""

    code = "SKILL_SCHEMA_INVALID"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlannerParseError(f"planner response contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise PlannerParseError(f"planner response contains non-finite number: {value}")


def parse_planner_outcome(raw_text: str, *, steps: Sequence[Any] | None = None) -> PlannerOutcome:
    """Parse one JSON response blob into a validated PlannerOutcome."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise PlannerParseError("planner response must be a non-empty JSON object")
    if raw_text != raw_text.strip() or not raw_text.startswith("{") or not raw_text.endswith("}"):
        raise PlannerParseError("planner response must contain exactly one JSON object")
    try:
        payload = json.loads(
            raw_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise PlannerParseError("planner response is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise PlannerParseError("planner response must be a JSON object")
    try:
        return planner_outcome_from_mapping(payload, steps=steps)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerParseError(str(exc)) from exc


def planner_outcome_from_payload(payload: Mapping[str, Any], *, steps: Sequence[Any] | None = None) -> PlannerOutcome:
    """Convert a pre-parsed payload mapping into a validated PlannerOutcome."""
    try:
        return planner_outcome_from_mapping(payload, steps=steps)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerParseError(str(exc)) from exc


class PlannerAdapter:
    """Adapt a model client returning ``complete()`` dictionaries to Planner."""

    def __init__(self, model_client: Any, *, prompt_builder: Any, planner_identity: Any) -> None:
        self._model_client = model_client
        self._prompt_builder = prompt_builder
        self.identity = planner_identity

    def plan(
        self, request: Any, context: Mapping[str, Any], catalog: Mapping[str, Any], cancel_token: Any
    ) -> PlannerOutcome:
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        messages = self._prompt_builder(request=request, context=context, catalog=catalog)
        response = self._model_client.complete(messages, force_json=True)
        if not isinstance(response, Mapping) or response.get("status") != "ok":
            detail = (
                response.get("error", "unknown model error")
                if isinstance(response, Mapping)
                else "invalid model response"
            )
            raise PlannerParseError(f"planner model call failed: {detail}")
        if response.get("tool_calls"):
            raise PlannerParseError("planner model must not return tool calls")
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        content = response.get("content")
        if not isinstance(content, str):
            raise PlannerParseError("planner model content must be a string")
        return parse_planner_outcome(content)


def build_planner_messages(
    *, request: Any, context: Mapping[str, Any], catalog: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build a channel-neutral prompt whose output is still locally validated."""
    system = (
        "You are the IB-Robot planner. Return exactly one JSON object and no markdown. "
        "Allowed kinds: conversation, read_only, needs_clarification, rejected, workflow. "
        "Never invent skills, parameters, endpoints, authorization, or execution tokens. "
        "For workflow, use only planner_visible skills from CATALOG and emit flat WorkflowStep objects. "
        "Questions, hypotheticals, quotations, and negated commands must not become workflows. "
        "For ordinary knowledge or identity questions, use kind=conversation and put the answer in user_message. "
        "The only permitted top-level fields are kind, user_message, query_kind, skill_name, missing_fields, "
        "reason_code, summary, and steps. Never return a field named text. Keep user_message <= 150 characters "
        "and summary <= 300 characters. For conversation/read_only/rejected, omit fields that do not belong "
        "to that kind (especially summary and steps). For every workflow step, schema_version must be the JSON "
        "integer 1 or 2, never a string, float, or omitted. Example conversation: "
        '{"kind":"conversation","user_message":"简短回答"}. Example workflow: '
        '{"kind":"workflow","user_message":"准备执行：nod_yes。","summary":"nod_yes",'
        '"steps":[{"schema_version":1,"skill_name":"nod_yes"}]}.'
    )
    payload = {
        "request": {"text": request.text, "reply_to_request_id": request.reply_to_request_id},
        "conversation": context,
        "catalog": catalog,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


class RulePlanner:
    """Deterministic SO-101 planner used for incubation and offline E2E tests.

    This is deliberately small and conservative.  It provides a usable natural
    language path without making a live model a prerequisite for validating the
    node and can later be replaced by :class:`PlannerAdapter` behind the same
    ``Planner`` protocol.
    """

    identity = PlannerIdentity(
        route="so101_rule_incubator",
        returned_model="deterministic-rule-planner",
        protocol="local-rule-v1",
        prompt_version="none",
        schema_version="1",
        config_digest="builtin-so101-v1",
    )

    _RULES = (
        (("挥手", "挥挥手", "打招呼", "wave", "hello"), "wave_hello"),
        (("点头", "同意", "nod", "yes"), "nod_yes"),
        (("摇头", "不同意", "shake", "no"), "shake_no"),
        (("打开夹爪", "张开夹爪", "开爪", "open gripper"), "open_gripper_skill"),
        (("关闭夹爪", "合上夹爪", "关爪", "close gripper"), "close_gripper_skill"),
        (("回安全位", "回到安全位", "回 home", "回home", "safe pose"), "recover_safe_pose"),
        (("回零位", "回到零位", "zero pose"), "recover_zero_pose"),
    )

    def plan(self, request, context, catalog, cancel_token):
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        text = request.text.strip()
        folded = text.casefold()
        if any(token in folded for token in ("不要", "别", "不想", "do not", "don't")):
            return PlannerOutcome(
                kind="conversation",
                user_message="我不会执行被明确否定的动作。请直接说明你希望机器人做什么。",
            )
        if any(token in folded for token in ("吗", "能不能", "会不会", "如果", "假如", "假设", "?", "？")):
            return PlannerOutcome(
                kind="conversation",
                user_message="这是能力询问或假设，不会触发机器人动作。",
            )
        if any(token in folded for token in ("有哪些能力", "有哪些技能", "能力列表", "list skills")):
            return PlannerOutcome(kind="read_only", user_message="我来查看当前机器人的能力。", query_kind="list_skills")
        if any(token in folded for token in ("当前状态", "状态怎么样", "status")):
            return PlannerOutcome(kind="read_only", user_message="我来查看当前机器人的状态。", query_kind="status")
        if any(token in folded for token in ("你好", "谢谢", "hello", "hi")) and not self._find_skills(folded):
            return PlannerOutcome(kind="conversation", user_message="你好，我可以帮助你控制 SO-101。")
        if any(token in folded for token in ("做一个动作", "做个动作", "动一下", "do something")):
            return PlannerOutcome(
                kind="needs_clarification",
                user_message="请明确选择当前目录中的一个动作，例如点头或挥手。",
                missing_fields=("skill_name",),
            )

        steps = []
        for aliases, skill_name in self._RULES:
            if any(alias.casefold() in folded for alias in aliases):
                steps.append(CanonicalWorkflowStep(1, skill_name))
        if any(separator in text for separator in ("然后", "再", "之后", "接着")):
            # Rules above are checked against the complete text, preserving the
            # user's order by matching each clause rather than the alias table.
            clauses = [part.strip() for part in re.split(r"然后|再|之后|接着", text) if part.strip()]
            ordered = []
            for clause in clauses:
                clause_steps = self._find_skills(clause.casefold())
                if clause_steps:
                    ordered.extend(clause_steps)
            if ordered:
                steps = ordered
        if not steps:
            return PlannerOutcome(
                kind="rejected",
                user_message="我无法把这句话安全映射到当前 SO-101 的已验证技能。",
                reason_code="UNSUPPORTED_TASK",
            )
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        summary = "、".join(step.skill_name for step in steps)
        return PlannerOutcome(
            kind="workflow", user_message=f"准备执行：{summary}。", summary=summary, steps=tuple(steps)
        )

    def _find_skills(self, text: str) -> list[CanonicalWorkflowStep]:
        return [
            CanonicalWorkflowStep(1, skill_name)
            for aliases, skill_name in self._RULES
            if any(alias.casefold() in text for alias in aliases)
        ]


__all__ = [
    "PlannerAdapter",
    "PlannerParseError",
    "RulePlanner",
    "build_planner_messages",
    "parse_planner_outcome",
    "planner_outcome_from_payload",
]
