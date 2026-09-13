"""ROS-free Agent orchestration for the IB-Robot incubation node."""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ibrobot_agent.contracts import (
    AgentEvent,
    AgentRequest,
    CatalogPort,
    ConversationStore,
    EventSink,
    ExecutionPort,
    ExecutionResult,
    Planner,
    PlannerOutcome,
    PlanProposal,
    RegistryIdentity,
    RequestKey,
    RequestStore,
    StopReceipt,
    planner_outcome_to_mapping,
    request_hash_text,
)
from ibrobot_agent.planner import RulePlanner

_LOGGER = logging.getLogger(__name__)


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    lowered = message.casefold()
    if "bearer " in lowered or "api key" in lowered or "apikey" in lowered:
        return type(exc).__name__
    return message[:300]


@dataclass(frozen=True)
class AcceptedResponse:
    accepted: bool
    request_id: str
    state: str
    reason_code: str = ""
    message: str = ""


class InMemoryConversationStore:
    """Bounded session memory used by the incubator until persistent memory is added."""

    def __init__(self, *, max_turns: int = 12, clarification_ttl_sec: float = 300.0) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if clarification_ttl_sec <= 0.0:
            raise ValueError("clarification_ttl_sec must be positive")
        self._max_turns = max_turns
        self._clarification_ttl_sec = clarification_ttl_sec
        self._lock = threading.RLock()
        self._messages: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
        self._clarifications: dict[tuple[str, str, str, str, str], dict[str, object]] = {}

    def append(self, request: AgentRequest, *, role: str = "user") -> None:
        key = (request.session_id, request.principal_id, request.channel_id, request.robot_scope)
        with self._lock:
            messages = self._messages.setdefault(key, [])
            messages.append({"role": role, "content": request.text})
            del messages[: -self._max_turns]

    def append_assistant(self, request: AgentRequest, content: str) -> None:
        key = (request.session_id, request.principal_id, request.channel_id, request.robot_scope)
        with self._lock:
            messages = self._messages.setdefault(key, [])
            messages.append({"role": "assistant", "content": content})
            del messages[: -self._max_turns]

    def context(self, request: AgentRequest) -> list[dict[str, str]]:
        key = (request.session_id, request.principal_id, request.channel_id, request.robot_scope)
        with self._lock:
            return [dict(item) for item in self._messages.get(key, [])]

    def save_clarification(self, request: AgentRequest, missing_fields) -> None:
        key = (*self._session_key(request), request.request_id)
        with self._lock:
            self._clarifications[key] = {
                "request_id": request.request_id,
                "original_text": request.text,
                "missing_fields": list(missing_fields),
                "expires_at": time.monotonic() + self._clarification_ttl_sec,
                "consumed": False,
            }

    def consume_clarification(self, request: AgentRequest) -> dict[str, object] | None:
        if request.reply_to_request_id is None:
            return None
        key = (*self._session_key(request), request.reply_to_request_id)
        with self._lock:
            value = self._clarifications.get(key)
            if value is None or value["consumed"] is True or float(value["expires_at"]) <= time.monotonic():
                return None
            value["consumed"] = True
            return dict(value)

    def close(self) -> None:
        return

    @staticmethod
    def _session_key(request: AgentRequest) -> tuple[str, str, str, str]:
        return request.session_id, request.principal_id, request.channel_id, request.robot_scope


class AgentService:
    """Coordinate planning and optional execution without importing ROS."""

    def __init__(
        self,
        *,
        planner: Planner,
        catalog: CatalogPort,
        store: RequestStore,
        execution: ExecutionPort | None = None,
        event_sink: EventSink | None = None,
        conversation: ConversationStore | None = None,
        execution_enabled: bool = False,
        allowed_skills: set[str] | None = None,
        max_planning_workers: int = 1,
        robot_scope: str = "so101_single_arm",
    ) -> None:
        if max_planning_workers != 1:
            raise ValueError("incubation node requires exactly one planning worker")
        self._planner = planner
        self._catalog = catalog
        self._store = store
        self._execution = execution
        self._events = event_sink
        self._conversation = conversation or InMemoryConversationStore()
        self._execution_enabled = execution_enabled
        self._allowed_skills = frozenset(allowed_skills or ())
        self._robot_scope = robot_scope
        self._quarantined = store.is_robot_quarantined(robot_scope)
        if execution_enabled and not self._allowed_skills:
            raise ValueError("execution-enabled AgentService requires a non-empty incubation allowlist")
        if execution_enabled and execution is None:
            raise ValueError("execution-enabled AgentService requires an execution port")
        if getattr(planner, "identity", None) is None:
            raise ValueError("planner must provide a PlannerIdentity")
        self._worker = threading.Thread(target=self._planning_loop, name="agent-planner", daemon=True)
        self._queue: list[AgentRequest] = []
        self._queued_or_active: set[RequestKey] = set()
        self._queue_condition = threading.Condition()
        self._stop = False
        self._active: dict[RequestKey, threading.Event] = {}
        self._active_lock = threading.RLock()
        self._worker.start()

    @property
    def execution_enabled(self) -> bool:
        return self._execution_enabled

    @property
    def healthy(self) -> bool:
        return not self._quarantined

    def close(self) -> None:
        with self._queue_condition:
            self._stop = True
            self._queue_condition.notify_all()
        with self._active_lock:
            for event in self._active.values():
                event.set()
        if self._execution is not None:
            with contextlib.suppress(Exception):
                self._execution.request_stop(None)
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            raise RuntimeError("Agent planning worker did not stop")
        self._conversation.close()
        self._store.close()

    def get_request(self, key: RequestKey):
        return self._store.get_request(key)

    def send_message(self, request: AgentRequest) -> AcceptedResponse:
        if request.robot_scope != self._robot_scope:
            return AcceptedResponse(
                False, request.request_id, "FAILED", "UNSUPPORTED_ROBOT", "robot scope is not bound"
            )
        key = request.to_key()
        try:
            record = self._store.admit(key, input_hash=request_hash_text(request), session_id=request.session_id)
        except Exception as exc:
            return AcceptedResponse(
                False, request.request_id, "FAILED", getattr(exc, "code", "REQUEST_REJECTED"), str(exc)
            )
        if record.state in {"SUCCEEDED", "FAILED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION", "ANSWERED", "UNKNOWN"}:
            terminal = record.terminal
            if terminal is not None and terminal.error_code == "ROBOT_QUARANTINED":
                return AcceptedResponse(False, request.request_id, "UNKNOWN", "ROBOT_QUARANTINED", terminal.message)
            if terminal is not None and terminal.error_code == "BUSY":
                return AcceptedResponse(False, request.request_id, "BUSY", "BUSY", terminal.message)
            return AcceptedResponse(True, request.request_id, record.state)
        if record.state != "RECEIVED":
            return AcceptedResponse(True, request.request_id, record.state)
        rejection: tuple[str, str, str] | None = None
        with self._queue_condition:
            if self._quarantined:
                rejection = ("ROBOT_QUARANTINED", "UNKNOWN", "robot scope is quarantined")
            elif key in self._queued_or_active:
                return AcceptedResponse(True, request.request_id, record.state)
            elif self._queued_or_active:
                rejection = ("BUSY", "BUSY", "robot scope has an active request")
            else:
                self._queued_or_active.add(key)
                self._queue.append(request)
                self._queue_condition.notify()
        if rejection is not None:
            reason_code, state, message = rejection
            try:
                self._store.finish(
                    key,
                    expected_generation=0,
                    result=ExecutionResult(
                        status="failed",
                        task_ref=None,
                        error_code=reason_code,
                        message=message,
                        detail={"admission": False},
                    ),
                )
            except Exception:
                self._quarantined = True
                _LOGGER.exception("Could not persist Agent admission rejection for %s", request.request_id)
                return AcceptedResponse(False, request.request_id, "UNKNOWN", "STORAGE_UNAVAILABLE", message)
            self._emit(key, reason_code.lower(), state, message)
            return AcceptedResponse(False, request.request_id, state, reason_code, message)
        self._conversation.append(request)
        return AcceptedResponse(True, request.request_id, "RECEIVED")

    def stop_request(self, key: RequestKey) -> StopReceipt:
        with self._active_lock:
            event = self._active.get(key)
            if event is not None:
                event.set()
        try:
            record = self._store.get_request(key)
            stopped_record = self._store.mark_stop(key, expected_generation=record.planning_generation)
            if self._execution is not None:
                with contextlib.suppress(Exception):
                    self._execution.request_stop(key)
            self._emit(key, "stop_requested", stopped_record.state, "stop requested")
            return StopReceipt(True, stopped_record, "", "stop requested")
        except Exception as exc:
            try:
                record = self._store.get_request(key)
            except Exception:
                raise
            return StopReceipt(False, record, getattr(exc, "code", "REQUEST_REJECTED"), str(exc))

    def _planning_loop(self) -> None:
        while True:
            with self._queue_condition:
                while not self._queue and not self._stop:
                    self._queue_condition.wait()
                if self._stop:
                    return
                request = self._queue.pop(0)
            try:
                self._plan_one(request)
            except Exception:
                _LOGGER.exception("Agent planner worker failed for request %s", request.request_id)
                self._recover_worker_failure(request)

    def _recover_worker_failure(self, request: AgentRequest) -> None:
        key = request.to_key()
        try:
            record = self._store.get_request(key)
            if record.state not in {
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "CANCELLED_BEFORE_EXECUTION",
                "ANSWERED",
                "UNKNOWN",
            }:
                self._finish_failure(key, record, "Agent planner worker failed")
        except Exception:
            self._quarantined = True
            _LOGGER.exception("Could not persist planner worker failure for request %s", request.request_id)

    def _plan_one(self, request: AgentRequest) -> None:
        key = request.to_key()
        try:
            record = self._store.begin_planning(key, expected_generation=0)
            cancel_token = threading.Event()
            with self._active_lock:
                self._active[key] = cancel_token
            gateway_error = None
            try:
                status = self._catalog.get_status()
                catalog = self._catalog.get_catalog(status)
            except Exception as exc:
                status = {}
                catalog = {}
                gateway_error = exc
            if isinstance(self._planner, RulePlanner) and not status:
                status = {
                    "robot_name": self._robot_scope,
                    "active_control_mode": "unknown",
                    "control_plane_ready": False,
                    "motion_authorized": False,
                    "registry_epoch": "offline",
                    "registry_generation": 0,
                    "registry_digest": "offline",
                }
                catalog = {"skills": [], "pose_names": []}
            clarification = self._conversation.consume_clarification(request)
            if request.reply_to_request_id is not None and clarification is None:
                self._finish_answered(
                    key,
                    record.planning_generation,
                    "澄清上下文已过期或不属于当前会话，请重新说明完整请求。",
                    error_code="CLARIFICATION_EXPIRED",
                    event_type="needs_clarification",
                )
                return
            read_only_outcome = self._deterministic_read_only_outcome(request.text)
            if read_only_outcome is not None:
                outcome = read_only_outcome
            else:
                outcome = self._planner.plan(
                    request,
                    {"messages": self._conversation.context(request), "clarification": clarification},
                    catalog,
                    cancel_token,
                )
            if cancel_token.is_set():
                self._finish_cancel_before_execution(key, record.planning_generation)
                return
            if outcome.kind != "workflow":
                if outcome.kind == "read_only" and not status:
                    raise RuntimeError("Gateway is unavailable for read-only robot queries") from gateway_error
                user_message = self._resolve_non_workflow_message(outcome, status, catalog)
                if outcome.kind == "needs_clarification":
                    self._conversation.save_clarification(request, outcome.missing_fields)
                self._store.record_event(
                    key,
                    event_type="planner_outcome",
                    state="PLANNING",
                    detail_json=json.dumps(planner_outcome_to_mapping(outcome), ensure_ascii=False, sort_keys=True),
                )
                self._conversation.append_assistant(request, user_message)
                self._finish_answered(key, record.planning_generation, user_message, event_type=outcome.kind)
                return
            if gateway_error is not None:
                raise RuntimeError("Gateway is unavailable for workflow validation") from gateway_error
            if not status or not status.get("registry_epoch") or not status.get("registry_digest"):
                raise RuntimeError("Gateway registry is not ready")
            if status.get("control_plane_ready", True) is False:
                raise RuntimeError("Gateway control plane is not ready")
            self._validate_workflow(outcome, catalog)
            identity = RegistryIdentity(
                epoch=str(status["registry_epoch"]),
                generation=int(status["registry_generation"]),
                digest=str(status["registry_digest"]),
            )
            proposal = PlanProposal(
                request=request,
                planning_generation=record.planning_generation,
                catalog_identity=identity,
                outcome=outcome,
                planner_identity=self._planner.identity,
            )
            record = self._store.mark_proposal_ready(key, expected_generation=record.planning_generation)
            self._store.record_event(
                key,
                event_type="proposal_snapshot",
                state="PROPOSAL_READY",
                detail_json=json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True),
            )
            self._emit(key, "proposal_ready", "PROPOSAL_READY", outcome.summary)
            if not self._execution_enabled or self._execution is None:
                self._finish_plan_without_execution(key, record.planning_generation, outcome.user_message)
                return
            record = self._store.mark_preparing(key, expected_generation=record.planning_generation)
            execution_result = self._execution.execute(
                proposal,
                expected_registry_identity=identity,
                presentation_callback=lambda presentation: self._present(
                    key, record.planning_generation, proposal, presentation
                ),
                submission_callback=lambda task_ref, detail: self._mark_may_submit(
                    key, record.planning_generation, task_ref, detail
                ),
                stop_event=cancel_token,
            )
            if execution_result.status == "unknown":
                current = self._store.get_request(key)
                self._finish_unknown(key, current, execution_result.message or execution_result.error_code)
            else:
                self._store.finish(key, expected_generation=record.planning_generation, result=execution_result)
                self._conversation.append_assistant(request, execution_result.message)
                self._emit(key, "terminal", self._request_state_for_result(execution_result), execution_result.message)
        except Exception as exc:
            try:
                record = self._store.get_request(key)
                if record.may_have_submitted:
                    self._finish_unknown(key, record, str(exc))
                elif record.stop_requested:
                    self._finish_cancel_before_execution(key, record.planning_generation)
                else:
                    self._finish_failure(key, record, _safe_error_message(exc))
            except Exception:
                self._quarantined = True
                _LOGGER.exception("Could not persist Agent request failure for %s", request.request_id)
        finally:
            with self._active_lock:
                self._active.pop(key, None)
            with self._queue_condition:
                self._queued_or_active.discard(key)

    def _finish_cancel_before_execution(self, key: RequestKey, generation: int) -> None:
        self._store.finish(
            key,
            expected_generation=generation,
            result=ExecutionResult(
                status="cancelled",
                task_ref=None,
                error_code="CANCELLED_BEFORE_EXECUTION",
                message="stopped before execution",
                detail={"submitted": False},
            ),
        )
        self._emit(key, "cancelled", "CANCELLED_BEFORE_EXECUTION", "stopped before execution")

    def _finish_answered(
        self,
        key: RequestKey,
        generation: int,
        message: str,
        *,
        error_code: str = "",
        event_type: str = "answered",
    ) -> None:
        self._store.mark_answered(
            key,
            expected_generation=generation,
            message=message,
            error_code=error_code,
        )
        self._emit(key, event_type, "ANSWERED", message)

    @staticmethod
    def _deterministic_read_only_outcome(text: str) -> PlannerOutcome | None:
        folded = text.casefold()
        if any(token in folded for token in ("当前状态", "状态怎么样", "状态如何", "status")):
            return PlannerOutcome(kind="read_only", user_message="我来查看当前机器人状态。", query_kind="status")
        if any(token in folded for token in ("有什么能力", "哪些能力", "有哪些技能", "技能列表", "能力列表")):
            return PlannerOutcome(kind="read_only", user_message="我来查看当前机器人技能。", query_kind="list_skills")
        if any(token in folded for token in ("有哪些姿态", "命名姿态", "姿态列表")):
            return PlannerOutcome(kind="read_only", user_message="我来查看当前机器人姿态。", query_kind="list_poses")
        return None

    @staticmethod
    def _resolve_non_workflow_message(
        outcome: PlannerOutcome,
        status: Mapping[str, Any],
        catalog: Mapping[str, Any],
    ) -> str:
        if outcome.kind != "read_only":
            return outcome.user_message
        if outcome.query_kind == "status":
            return (
                f"机器人 {status.get('robot_name', 'unknown')}，控制平面"
                f"{'已就绪' if status.get('control_plane_ready') else '未就绪'}，"
                f"当前模式 {status.get('active_control_mode', 'unknown')}，"
                f"{'运动已授权' if status.get('motion_authorized') else '运动未授权'}。"
            )
        if outcome.query_kind == "list_skills":
            names = sorted(
                str(item["name"])
                for item in catalog.get("skills", [])
                if isinstance(item, Mapping) and item.get("planner_visible") is True and item.get("name")
            )
            return AgentService._bounded_list_message("当前可规划技能", names)
        if outcome.query_kind == "list_poses":
            poses = [str(name) for name in catalog.get("pose_names", [])]
            return AgentService._bounded_list_message("当前命名姿态", sorted(poses))
        if outcome.query_kind == "describe_skill":
            capability = next(
                (
                    item
                    for item in catalog.get("skills", [])
                    if isinstance(item, Mapping) and item.get("name") == outcome.skill_name
                ),
                None,
            )
            if capability is None:
                return f"当前目录中没有技能 {outcome.skill_name}。"
            return f"{outcome.skill_name}：{capability.get('summary', '无公开说明')}"
        return outcome.user_message

    @staticmethod
    def _bounded_list_message(prefix: str, values: list[str], *, max_length: int = 145) -> str:
        if not values:
            return f"{prefix}：无。"
        visible = []
        used = len(prefix) + 1
        for value in values:
            addition = len(value) + (1 if visible else 0)
            if used + addition + 8 > max_length:
                break
            visible.append(value)
            used += addition
        remaining = len(values) - len(visible)
        suffix = f"，另有 {remaining} 项。" if remaining else "。"
        return f"{prefix}：{'、'.join(visible)}{suffix}"

    def _validate_workflow(self, outcome: PlannerOutcome, catalog: Mapping[str, Any]) -> None:
        skills = catalog.get("skills", [])
        if not isinstance(skills, list):
            raise RuntimeError("Gateway catalog skills are invalid")
        by_name = {item.get("name"): item for item in skills if isinstance(item, Mapping)}
        for step in outcome.steps:
            capability = by_name.get(step.skill_name)
            if not isinstance(capability, Mapping) or capability.get("planner_visible") is not True:
                raise RuntimeError(f"skill is not planner-visible: {step.skill_name}")
            if capability.get("semantic_level") not in {"atomic_operator", "skill"}:
                raise RuntimeError(f"skill is not executable: {step.skill_name}")
            if self._allowed_skills and step.skill_name not in self._allowed_skills:
                raise RuntimeError(f"skill is outside the incubation allowlist: {step.skill_name}")

    def _present(self, key: RequestKey, generation: int, proposal: PlanProposal, presentation: Any) -> None:
        self._store.bind_task(
            key,
            expected_generation=generation,
            task_ref=presentation.task_ref,
            presentation=presentation,
            proposal_json=json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True),
        )
        self._emit(key, "presentation", "MAY_EXECUTE", presentation.summary)

    def _mark_may_submit(
        self,
        key: RequestKey,
        generation: int,
        task_ref: Any,
        detail: Mapping[str, object],
    ) -> None:
        self._store.record_confirmation(key, expected_generation=generation, detail=detail)
        self._store.mark_submitted(key, expected_generation=generation, task_ref=task_ref)

    def _finish_failure(self, key: RequestKey, record: Any, message: str) -> None:
        self._store.finish(
            key,
            expected_generation=record.planning_generation,
            result=ExecutionResult(
                status="failed", task_ref=record.task_ref, error_code="AGENT_FAILED", message=message, detail={}
            ),
        )
        self._emit(key, "failed", "FAILED", message)

    def _finish_unknown(self, key: RequestKey, record: Any, message: str) -> None:
        reason = (message or "unknown execution outcome")[:300]
        self._store.quarantine(key, expected_generation=record.planning_generation, reason=reason)
        self._quarantined = True
        self._emit(key, "quarantined", "UNKNOWN", reason)

    @staticmethod
    def _request_state_for_result(result: Any) -> str:
        if result.status == "succeeded":
            return "SUCCEEDED"
        if result.status == "failed":
            return "FAILED"
        if result.status == "cancelled":
            return "CANCELLED" if result.task_ref is not None else "CANCELLED_BEFORE_EXECUTION"
        return "UNKNOWN"

    def _finish_plan_without_execution(self, key: RequestKey, generation: int, message: str) -> None:
        """Close dry-run planning without claiming that a robot task succeeded."""
        self._store.mark_answered(
            key,
            expected_generation=generation,
            message=message,
            error_code="DRY_RUN_ONLY",
        )
        self._emit(key, "dry_run_complete", "ANSWERED", message)

    def _emit(self, key: RequestKey, event_type: str, state: str, message: str) -> None:
        if self._events is None:
            return
        sequence = self._store.record_event(
            key,
            event_type=event_type,
            state=state,
            detail_json=json.dumps({"message": message}, ensure_ascii=False, sort_keys=True),
        )
        try:
            self._events.publish(
                AgentEvent(
                    schema_version=1,
                    request_key=key,
                    sequence=sequence,
                    event_type=event_type,
                    state=state,
                    user_message=message,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception:
            # The ledger is authoritative; a broken transport must not reopen
            # the request through the outer failure handler.
            _LOGGER.exception("Could not publish Agent event for %s", key.request_id)
