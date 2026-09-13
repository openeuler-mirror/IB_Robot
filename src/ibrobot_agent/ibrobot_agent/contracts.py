"""Typed core contracts for the ibrobot_agent package.

The package keeps this module free of ROS imports so the planning and storage
layers stay importable in unit tests and offline tooling.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

from embodied_common.canon import sha256_text, to_canonical_json
from embodied_common.workflow_contracts import CanonicalWorkflowStep, normalize_workflow_steps

QueryKind = Literal["status", "list_skills", "describe_skill", "list_poses"]
PlanKind = Literal[1, 2]
PlannerOutcomeKind = Literal["conversation", "read_only", "needs_clarification", "rejected", "workflow"]
AgentResponseStatus = Literal[
    "conversation",
    "read_only",
    "needs_clarification",
    "rejected",
    "planned",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "unknown",
    "busy",
]
ExecutionStatus = Literal["succeeded", "failed", "cancelled", "unknown"]
RequestState = Literal[
    "RECEIVED",
    "PLANNING",
    "PROPOSAL_READY",
    "PREPARING",
    "MAY_EXECUTE",
    "RUNNING",
    "STOPPING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "CANCELLED_BEFORE_EXECUTION",
    "ANSWERED",
    "UNKNOWN",
]
RobotAdmissionState = Literal["READY", "BUSY", "QUARANTINED"]

REQUEST_STATE_VALUES = {
    "RECEIVED",
    "PLANNING",
    "PROPOSAL_READY",
    "PREPARING",
    "MAY_EXECUTE",
    "RUNNING",
    "STOPPING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    "CANCELLED_BEFORE_EXECUTION",
    "ANSWERED",
    "UNKNOWN",
}
TERMINAL_REQUEST_STATES = frozenset(
    {"SUCCEEDED", "FAILED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION", "ANSWERED", "UNKNOWN"}
)
RESPONSE_STATUS_VALUES = {
    "conversation",
    "read_only",
    "needs_clarification",
    "rejected",
    "planned",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "unknown",
    "busy",
}

REASON_CODE_VALUES = {"UNSUPPORTED_TASK", "OUT_OF_SCOPE", "SAFETY_RESTRICTED"}
_PLANNER_FIELDS = {
    "kind",
    "user_message",
    "query_kind",
    "skill_name",
    "missing_fields",
    "reason_code",
    "steps",
    "summary",
}
_WORKFLOW_COMMON_FIELDS = {
    "schema_version",
    "skill_name",
    "target_name",
    "container_name",
    "place_name",
    "motion_direction",
    "motion_distance",
    "arm_side",
    "imitation_duration_sec",
    "timeout_sec",
}
_WORKFLOW_NAVIGATION_FIELDS = {"direction", "distance", "degree", "has_x", "x", "has_y", "y", "has_yaw", "yaw"}
_WORKFLOW_TEXT_FIELDS = {
    "skill_name",
    "target_name",
    "container_name",
    "place_name",
    "motion_direction",
    "arm_side",
    "direction",
}
_WORKFLOW_NUMBER_FIELDS = {
    "motion_distance",
    "imitation_duration_sec",
    "timeout_sec",
    "distance",
    "degree",
    "x",
    "y",
    "yaw",
}


def _non_empty_text(value: object, field_name: str, *, max_length: int | None = None) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if max_length is not None and len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return normalized


def _text(value: object, field_name: str, *, max_length: int | None = None) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if max_length is not None and len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return normalized


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


def _mapping_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class RequestKey:
    robot_scope: str
    channel_id: str
    principal_id: str
    request_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "robot_scope", _non_empty_text(self.robot_scope, "robot_scope"))
        object.__setattr__(self, "channel_id", _non_empty_text(self.channel_id, "channel_id"))
        object.__setattr__(self, "principal_id", _non_empty_text(self.principal_id, "principal_id"))
        object.__setattr__(self, "request_id", _non_empty_text(self.request_id, "request_id"))

    def to_dict(self) -> dict[str, str]:
        return {
            "robot_scope": self.robot_scope,
            "channel_id": self.channel_id,
            "principal_id": self.principal_id,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class RegistryIdentity:
    epoch: str
    generation: int
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", _non_empty_text(self.epoch, "epoch"))
        object.__setattr__(self, "generation", _positive_int(self.generation, "generation"))
        object.__setattr__(self, "digest", _non_empty_text(self.digest, "digest"))

    def to_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "generation": self.generation, "digest": self.digest}


@dataclass(frozen=True)
class PlannerIdentity:
    route: str
    returned_model: str
    protocol: str
    prompt_version: str
    schema_version: str
    config_digest: str

    def __post_init__(self) -> None:
        for field_name in ("route", "returned_model", "protocol", "prompt_version", "schema_version", "config_digest"):
            object.__setattr__(self, field_name, _non_empty_text(getattr(self, field_name), field_name))

    def to_dict(self) -> dict[str, str]:
        return {
            "route": self.route,
            "returned_model": self.returned_model,
            "protocol": self.protocol,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "config_digest": self.config_digest,
        }


@dataclass(frozen=True)
class AgentRequest:
    schema_version: int
    request_id: str
    session_id: str
    channel_id: str
    principal_id: str
    robot_scope: str
    text: str
    received_at: datetime
    reply_to_request_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("AgentRequest.schema_version must be 1")
        object.__setattr__(self, "request_id", _non_empty_text(self.request_id, "request_id"))
        object.__setattr__(self, "session_id", _non_empty_text(self.session_id, "session_id"))
        object.__setattr__(self, "channel_id", _non_empty_text(self.channel_id, "channel_id"))
        object.__setattr__(self, "principal_id", _non_empty_text(self.principal_id, "principal_id"))
        object.__setattr__(self, "robot_scope", _non_empty_text(self.robot_scope, "robot_scope"))
        object.__setattr__(self, "text", _non_empty_text(self.text, "text"))
        if self.reply_to_request_id is not None:
            object.__setattr__(
                self, "reply_to_request_id", _non_empty_text(self.reply_to_request_id, "reply_to_request_id")
            )
        if not isinstance(self.received_at, datetime):
            raise TypeError("received_at must be a datetime")

    def to_key(self) -> RequestKey:
        return RequestKey(
            robot_scope=self.robot_scope,
            channel_id=self.channel_id,
            principal_id=self.principal_id,
            request_id=self.request_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "channel_id": self.channel_id,
            "principal_id": self.principal_id,
            "robot_scope": self.robot_scope,
            "text": self.text,
            "received_at": self.received_at.isoformat(),
            "reply_to_request_id": self.reply_to_request_id,
        }


@dataclass(frozen=True)
class TaskRef:
    task_id: str
    plan_id: str
    plan_digest: str
    registry_epoch: str
    registry_generation: int
    registry_digest: str
    expected_step_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _non_empty_text(self.task_id, "task_id"))
        object.__setattr__(self, "plan_id", _non_empty_text(self.plan_id, "plan_id"))
        object.__setattr__(self, "plan_digest", _non_empty_text(self.plan_digest, "plan_digest"))
        object.__setattr__(self, "registry_epoch", _non_empty_text(self.registry_epoch, "registry_epoch"))
        object.__setattr__(self, "registry_generation", _positive_int(self.registry_generation, "registry_generation"))
        object.__setattr__(self, "registry_digest", _non_empty_text(self.registry_digest, "registry_digest"))
        object.__setattr__(self, "expected_step_count", _positive_int(self.expected_step_count, "expected_step_count"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "registry_epoch": self.registry_epoch,
            "registry_generation": self.registry_generation,
            "registry_digest": self.registry_digest,
            "expected_step_count": self.expected_step_count,
        }


@dataclass(frozen=True)
class PlannerOutcome:
    kind: PlannerOutcomeKind
    user_message: str
    query_kind: QueryKind | None = None
    skill_name: str | None = None
    missing_fields: tuple[str, ...] = ()
    reason_code: str | None = None
    steps: tuple[CanonicalWorkflowStep, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"conversation", "read_only", "needs_clarification", "rejected", "workflow"}:
            raise ValueError("PlannerOutcome.kind is invalid")
        object.__setattr__(self, "user_message", _non_empty_text(self.user_message, "user_message", max_length=150))
        normalized_missing_fields = tuple(
            _non_empty_text(field_name, "missing_fields item", max_length=64) for field_name in self.missing_fields
        )
        object.__setattr__(self, "missing_fields", normalized_missing_fields)
        if self.kind == "conversation":
            if (
                self.query_kind is not None
                or self.skill_name is not None
                or self.missing_fields
                or self.reason_code is not None
            ):
                raise ValueError("conversation outcomes cannot carry execution fields")
            if self.steps:
                raise ValueError("conversation outcomes cannot carry steps")
            if self.summary:
                raise ValueError("conversation outcomes cannot carry a summary")
            return
        if self.kind == "read_only":
            if self.query_kind not in {"status", "list_skills", "describe_skill", "list_poses"}:
                raise ValueError("read_only outcomes require a query_kind")
            if self.missing_fields or self.reason_code is not None or self.steps or self.summary:
                raise ValueError("read_only outcomes cannot carry workflow fields")
            if self.query_kind == "describe_skill":
                object.__setattr__(self, "skill_name", _non_empty_text(self.skill_name, "skill_name"))
            elif self.skill_name is not None:
                raise ValueError("skill_name is only allowed for describe_skill")
            return
        if self.kind == "needs_clarification":
            if not self.missing_fields:
                raise ValueError("needs_clarification outcomes require missing_fields")
            for field_name in self.missing_fields:
                _non_empty_text(field_name, "missing_fields item", max_length=64)
            if self.query_kind is not None or self.skill_name is not None or self.reason_code is not None:
                raise ValueError("needs_clarification outcomes cannot carry execution fields")
            if self.steps or self.summary:
                raise ValueError("needs_clarification outcomes cannot carry workflow fields")
            return
        if self.kind == "rejected":
            reason_code = _non_empty_text(self.reason_code, "reason_code", max_length=64)
            if reason_code not in REASON_CODE_VALUES:
                raise ValueError("reason_code is invalid")
            object.__setattr__(self, "reason_code", reason_code)
            if self.query_kind is not None or self.skill_name is not None or self.missing_fields:
                raise ValueError("rejected outcomes cannot carry execution fields")
            if self.steps or self.summary:
                raise ValueError("rejected outcomes cannot carry workflow fields")
            return
        if (
            self.query_kind is not None
            or self.skill_name is not None
            or self.missing_fields
            or self.reason_code is not None
        ):
            raise ValueError("workflow outcomes cannot carry read-only or rejection fields")
        object.__setattr__(self, "summary", _non_empty_text(self.summary, "summary", max_length=300))
        if not self.steps:
            raise ValueError("workflow outcomes require at least one step")
        for step in self.steps:
            if not isinstance(step, CanonicalWorkflowStep):
                raise TypeError("workflow outcome steps must be CanonicalWorkflowStep instances")
        object.__setattr__(self, "steps", tuple(self.steps))
        if len(self.steps) > 16:
            raise ValueError("workflow outcomes cannot contain more than 16 steps")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "user_message": self.user_message}
        if self.query_kind is not None:
            result["query_kind"] = self.query_kind
        if self.skill_name is not None:
            result["skill_name"] = self.skill_name
        if self.missing_fields:
            result["missing_fields"] = list(self.missing_fields)
        if self.reason_code is not None:
            result["reason_code"] = self.reason_code
        if self.summary:
            result["summary"] = self.summary
        if self.steps:
            result["steps"] = [step.to_dict() for step in self.steps]
        return result


@dataclass(frozen=True)
class PlanProposal:
    request: AgentRequest
    planning_generation: int
    catalog_identity: RegistryIdentity
    outcome: PlannerOutcome
    planner_identity: PlannerIdentity

    def __post_init__(self) -> None:
        if isinstance(self.planning_generation, bool) or self.planning_generation < 0:
            raise ValueError("planning_generation must be non-negative")
        if not isinstance(self.request, AgentRequest):
            raise TypeError("request must be an AgentRequest")
        if not isinstance(self.catalog_identity, RegistryIdentity):
            raise TypeError("catalog_identity must be a RegistryIdentity")
        if not isinstance(self.outcome, PlannerOutcome):
            raise TypeError("outcome must be a PlannerOutcome")
        if not isinstance(self.planner_identity, PlannerIdentity):
            raise TypeError("planner_identity must be a PlannerIdentity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "planning_generation": self.planning_generation,
            "catalog_identity": self.catalog_identity.to_dict(),
            "outcome": self.outcome.to_dict(),
            "planner_identity": self.planner_identity.to_dict(),
        }


@dataclass(frozen=True)
class Presentation:
    task_ref: TaskRef
    plan_kind: PlanKind
    steps: tuple[CanonicalWorkflowStep, ...]
    execution_mode: str
    proposed_task_budget_sec: float
    summary: str

    def __post_init__(self) -> None:
        if self.plan_kind not in {1, 2}:
            raise ValueError("plan_kind must be 1 or 2")
        object.__setattr__(self, "execution_mode", _non_empty_text(self.execution_mode, "execution_mode"))
        object.__setattr__(
            self, "proposed_task_budget_sec", _positive_float(self.proposed_task_budget_sec, "proposed_task_budget_sec")
        )
        object.__setattr__(self, "summary", _non_empty_text(self.summary, "summary", max_length=300))
        object.__setattr__(self, "steps", tuple(self.steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_ref": self.task_ref.to_dict(),
            "plan_kind": int(self.plan_kind),
            "steps": [step.to_dict() for step in self.steps],
            "execution_mode": self.execution_mode,
            "proposed_task_budget_sec": self.proposed_task_budget_sec,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecutionStatus
    task_ref: TaskRef | None
    error_code: str
    message: str
    detail: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed", "cancelled", "unknown"}:
            raise ValueError("ExecutionResult.status is invalid")
        object.__setattr__(self, "error_code", _text(self.error_code, "error_code", max_length=64))
        object.__setattr__(self, "message", _text(self.message, "message", max_length=300))
        if not isinstance(self.detail, Mapping):
            raise TypeError("detail must be a mapping")
        if self.task_ref is not None and not isinstance(self.task_ref, TaskRef):
            raise TypeError("task_ref must be a TaskRef or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_ref": None if self.task_ref is None else self.task_ref.to_dict(),
            "error_code": self.error_code,
            "message": self.message,
            "detail": _mapping_dict(self.detail),
        }


@dataclass(frozen=True)
class RequestRecord:
    key: RequestKey
    session_id: str
    state: RequestState
    planning_generation: int
    stop_requested: bool
    may_have_submitted: bool
    input_hash: str
    task_ref: TaskRef | None = None
    proposal_json: str = ""
    terminal: ExecutionResult | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, RequestKey):
            raise TypeError("key must be a RequestKey")
        object.__setattr__(self, "session_id", _non_empty_text(self.session_id, "session_id"))
        if self.state not in REQUEST_STATE_VALUES:
            raise ValueError("state is invalid")
        object.__setattr__(
            self, "planning_generation", _non_negative_int(self.planning_generation, "planning_generation")
        )
        object.__setattr__(self, "input_hash", _non_empty_text(self.input_hash, "input_hash"))
        object.__setattr__(self, "proposal_json", _text(self.proposal_json, "proposal_json"))
        if not isinstance(self.stop_requested, bool):
            raise TypeError("stop_requested must be a boolean")
        if not isinstance(self.may_have_submitted, bool):
            raise TypeError("may_have_submitted must be a boolean")
        if self.task_ref is not None and not isinstance(self.task_ref, TaskRef):
            raise TypeError("task_ref must be a TaskRef or None")
        if self.terminal is not None and not isinstance(self.terminal, ExecutionResult):
            raise TypeError("terminal must be an ExecutionResult or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "session_id": self.session_id,
            "state": self.state,
            "planning_generation": self.planning_generation,
            "stop_requested": self.stop_requested,
            "may_have_submitted": self.may_have_submitted,
            "input_hash": self.input_hash,
            "task_ref": None if self.task_ref is None else self.task_ref.to_dict(),
            "proposal_json": self.proposal_json,
            "terminal": None if self.terminal is None else self.terminal.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


RequestView = RequestRecord


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    record: RequestRecord
    reason_code: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "record": self.record.to_dict(),
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class StopReceipt:
    requested: bool
    record: RequestRecord
    reason_code: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "record": self.record.to_dict(),
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class AgentEvent:
    schema_version: int
    request_key: RequestKey
    sequence: int
    event_type: str
    state: RequestState
    user_message: str
    detail: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("AgentEvent.schema_version must be 1")
        if not isinstance(self.request_key, RequestKey):
            raise TypeError("request_key must be a RequestKey")
        object.__setattr__(self, "sequence", _positive_int(self.sequence, "sequence"))
        object.__setattr__(self, "event_type", _non_empty_text(self.event_type, "event_type"))
        object.__setattr__(self, "user_message", _text(self.user_message, "user_message", max_length=300))
        if not isinstance(self.state, str) or self.state not in REQUEST_STATE_VALUES:
            raise ValueError("state is invalid")
        if not isinstance(self.detail, Mapping):
            raise TypeError("detail must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_key": self.request_key.to_dict(),
            "sequence": self.sequence,
            "event_type": self.event_type,
            "state": self.state,
            "user_message": self.user_message,
            "detail": _mapping_dict(self.detail),
            "created_at": self.created_at,
        }


class Planner(Protocol):
    def plan(
        self,
        request: AgentRequest,
        context: Mapping[str, Any],
        catalog: Mapping[str, Any],
        cancel_token: threading.Event,
    ) -> PlannerOutcome: ...


class CatalogPort(Protocol):
    """Read-only runtime capability boundary injected by IB-Robot."""

    def get_status(self) -> Mapping[str, Any]: ...

    def get_catalog(self, status: Mapping[str, Any]) -> Mapping[str, Any]: ...


class EventSink(Protocol):
    def publish(self, event: AgentEvent) -> None: ...


class ConversationStore(Protocol):
    def append(self, request: AgentRequest, *, role: str = "user") -> None: ...

    def append_assistant(self, request: AgentRequest, content: str) -> None: ...

    def context(self, request: AgentRequest) -> list[dict[str, str]]: ...

    def save_clarification(self, request: AgentRequest, missing_fields: Sequence[str]) -> None: ...

    def consume_clarification(self, request: AgentRequest) -> dict[str, object] | None: ...

    def close(self) -> None: ...


class ExecutionPort(Protocol):
    def execute(
        self,
        proposal: PlanProposal,
        *,
        expected_registry_identity: RegistryIdentity,
        presentation_callback: Callable[[Presentation], None],
        submission_callback: Callable[[TaskRef, Mapping[str, object]], None],
        stop_event: threading.Event,
    ) -> ExecutionResult: ...

    def request_stop(self, request_key: RequestKey | None) -> None: ...


class RequestStore(Protocol):
    def admit(self, key: RequestKey, *, input_hash: str, session_id: str) -> RequestRecord: ...

    def get_request(self, key: RequestKey) -> RequestView: ...

    def is_robot_quarantined(self, robot_scope: str) -> bool: ...

    def begin_planning(self, key: RequestKey, *, expected_generation: int) -> RequestRecord: ...

    def mark_proposal_ready(self, key: RequestKey, *, expected_generation: int) -> RequestRecord: ...

    def mark_preparing(self, key: RequestKey, *, expected_generation: int) -> RequestRecord: ...

    def bind_task(
        self,
        key: RequestKey,
        *,
        expected_generation: int,
        task_ref: TaskRef,
        presentation: Presentation,
        proposal_json: str,
    ) -> RequestRecord: ...

    def mark_stop(self, key: RequestKey, *, expected_generation: int) -> RequestRecord: ...

    def mark_submitted(self, key: RequestKey, *, expected_generation: int, task_ref: TaskRef) -> RequestRecord: ...

    def record_confirmation(
        self,
        key: RequestKey,
        *,
        expected_generation: int,
        detail: Mapping[str, object],
    ) -> RequestRecord: ...

    def mark_answered(
        self,
        key: RequestKey,
        *,
        expected_generation: int,
        message: str,
        error_code: str = "",
    ) -> RequestRecord: ...

    def finish(self, key: RequestKey, *, expected_generation: int, result: ExecutionResult) -> RequestRecord: ...

    def quarantine(self, key: RequestKey, *, expected_generation: int, reason: str) -> RequestRecord: ...

    def record_event(self, key: RequestKey, *, event_type: str, state: str, detail_json: str) -> int: ...

    def close(self) -> None: ...


def planner_outcome_from_mapping(payload: Mapping[str, Any], *, steps: Sequence[Any] | None = None) -> PlannerOutcome:
    if not isinstance(payload, Mapping):
        raise TypeError("planner outcome must be a mapping")
    unknown_fields = sorted(set(payload) - _PLANNER_FIELDS)
    if unknown_fields:
        raise ValueError(f"planner outcome contains unknown fields: {', '.join(unknown_fields)}")
    for field_name in ("kind", "user_message"):
        if not isinstance(payload.get(field_name), str):
            raise TypeError(f"{field_name} must be a string")
    for field_name in ("query_kind", "skill_name", "reason_code", "summary"):
        if field_name in payload and payload[field_name] is not None and not isinstance(payload[field_name], str):
            raise TypeError(f"{field_name} must be a string")
    missing_fields = payload.get("missing_fields", ())
    if not isinstance(missing_fields, Sequence) or isinstance(missing_fields, str | bytes):
        raise TypeError("missing_fields must be a sequence")
    if any(not isinstance(item, str) for item in missing_fields):
        raise TypeError("missing_fields items must be strings")
    workflow_steps = steps if steps is not None else payload.get("steps", ())
    _validate_raw_workflow_steps(workflow_steps)
    normalized_steps = normalize_workflow_steps(workflow_steps) if workflow_steps else ()
    return PlannerOutcome(
        kind=payload["kind"],
        user_message=payload["user_message"],
        query_kind=payload.get("query_kind"),
        skill_name=payload.get("skill_name"),
        missing_fields=tuple(missing_fields),
        reason_code=payload.get("reason_code"),
        steps=normalized_steps,
        summary=payload.get("summary", ""),
    )


def _validate_raw_workflow_steps(steps: Sequence[Any]) -> None:
    if not isinstance(steps, Sequence) or isinstance(steps, str | bytes):
        raise TypeError("steps must be a sequence")
    if len(steps) > 16:
        raise ValueError("steps cannot contain more than 16 entries")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"steps[{index}] must be a mapping")
        schema_version = step.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version not in {1, 2}:
            raise TypeError(f"steps[{index}].schema_version must be integer 1 or 2")
        allowed_fields = _WORKFLOW_COMMON_FIELDS | (_WORKFLOW_NAVIGATION_FIELDS if schema_version == 2 else set())
        unknown_fields = sorted(set(step) - allowed_fields)
        if unknown_fields:
            raise ValueError(f"steps[{index}] contains unknown fields: {', '.join(unknown_fields)}")
        for field_name in _WORKFLOW_TEXT_FIELDS & set(step):
            if not isinstance(step[field_name], str):
                raise TypeError(f"steps[{index}].{field_name} must be a string")
        for field_name in _WORKFLOW_NUMBER_FIELDS & set(step):
            value = step[field_name]
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise TypeError(f"steps[{index}].{field_name} must be a finite number")
        for field_name in {"has_x", "has_y", "has_yaw"} & set(step):
            if not isinstance(step[field_name], bool):
                raise TypeError(f"steps[{index}].{field_name} must be a boolean")


def planner_outcome_to_mapping(outcome: PlannerOutcome) -> dict[str, Any]:
    return outcome.to_dict()


def request_hash_preimage(request: AgentRequest) -> dict[str, Any]:
    return {
        "schema_version": request.schema_version,
        "request_id": request.request_id,
        "session_id": request.session_id,
        "channel_id": request.channel_id,
        "principal_id": request.principal_id,
        "robot_scope": request.robot_scope,
        "text": request.text,
        "reply_to_request_id": request.reply_to_request_id,
    }


def request_hash_text(request: AgentRequest) -> str:
    return sha256_text(to_canonical_json(request_hash_preimage(request)))


__all__ = [
    "AdmissionResult",
    "AgentEvent",
    "AgentRequest",
    "AgentResponseStatus",
    "ExecutionPort",
    "CatalogPort",
    "ConversationStore",
    "EventSink",
    "ExecutionResult",
    "ExecutionStatus",
    "PlanKind",
    "PlanProposal",
    "Planner",
    "PlannerIdentity",
    "PlannerOutcome",
    "PlannerOutcomeKind",
    "QueryKind",
    "RegistryIdentity",
    "RequestKey",
    "RequestRecord",
    "RequestState",
    "RequestStore",
    "RequestView",
    "RobotAdmissionState",
    "StopReceipt",
    "Presentation",
    "TaskRef",
    "planner_outcome_from_mapping",
    "planner_outcome_to_mapping",
    "request_hash_preimage",
    "request_hash_text",
    "REASON_CODE_VALUES",
]
