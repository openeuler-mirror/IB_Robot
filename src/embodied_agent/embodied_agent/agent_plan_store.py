"""Short-lived immutable Agent plan state owned by ``embodied_agent``."""

from __future__ import annotations

import hmac
import math
import secrets
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_common.agent_execution_contract import (
    INTERACTIVE_CONFIRMATION,
    validate_agent_execution_mode,
)
from embodied_common.canon import sha256_text, to_canonical_json
from embodied_common.workflow_contracts import CanonicalWorkflowStep, normalize_workflow_steps

MAX_PLAN_STEPS = 16
DEFAULT_PLAN_TTL_SEC = 300.0
DEFAULT_MAX_RECORDS = 1024


def _float32(value: float) -> float:
    import struct

    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


class AgentPlanError(Exception):
    """Stable plan-store failure with no secret-bearing message."""

    def __init__(self, code: str, message: str = "agent plan request rejected") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentPlan:
    schema_version: int
    plan_id: str
    plan_token: str = field(repr=False)
    plan_kind: int
    raw_command: str
    workflow_steps: tuple[CanonicalWorkflowStep, ...]
    plan_digest: str
    registry_epoch: str
    registry_generation: int
    registry_digest: str
    expires_at: tuple[int, int]
    execution_mode: str = INTERACTIVE_CONFIRMATION

    SINGLE_SKILL = 1
    WORKFLOW = 2


@dataclass(frozen=True)
class PlanConfirmation:
    confirmed: bool
    confirmation_token: str = field(repr=False)
    plan_digest: str
    task_id: str
    task_budget_sec: float
    task_budget_started_at: tuple[int, int]
    task_budget_deadline: tuple[int, int]


@dataclass(frozen=True)
class AgentPlanExecution:
    plan: AgentPlan
    task_id: str
    state: str
    confirmation_token: str = field(repr=False)
    task_budget_sec: float = 0.0
    task_budget_started_at: tuple[int, int] = (0, 0)
    task_budget_deadline: tuple[int, int] = (0, 0)
    clock_at_confirmation: float = 0.0
    terminal_code: str = ""
    terminal_message: str = ""
    workflow_digest: str = ""
    completed_step_count: int = 0
    newly_accepted: bool = False
    execution_token: str = field(default="", repr=False)


@dataclass
class _PlanRecord:
    plan: AgentPlan
    request_id: str
    monotonic_expires_at: float
    state: str = "PLANNED"
    task_id: str = ""
    confirmation_token: str = field(default="", repr=False)
    task_budget_sec: float = 0.0
    task_budget_started_at: tuple[int, int] = (0, 0)
    task_budget_deadline: tuple[int, int] = (0, 0)
    clock_at_confirmation: float = 0.0
    terminal_code: str = ""
    terminal_message: str = ""
    workflow_digest: str = ""
    completed_step_count: int = 0
    execution_token: str = field(default="", repr=False)


class AgentPlanStore:
    """In-memory bounded store; callers provide their own state lock."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        token_factory: Callable[[], str] | None = None,
        plan_id_factory: Callable[[], str] | None = None,
        max_records: int = DEFAULT_MAX_RECORDS,
        ttl_sec: float = DEFAULT_PLAN_TTL_SEC,
    ) -> None:
        if max_records <= 0 or ttl_sec <= 0:
            raise ValueError("max_records and ttl_sec must be positive")
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._last_wall_time: float | None = None
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self._plan_id_factory = plan_id_factory or (lambda: str(uuid.uuid4()))
        self._max_records = max_records
        self._ttl_sec = float(ttl_sec)
        self._records: dict[str, _PlanRecord] = {}
        self._request_index: dict[str, str] = {}
        self._expired_tokens: OrderedDict[str, None] = OrderedDict()

    def create_plan(
        self,
        *,
        request_id: str,
        raw_command: str,
        workflow_steps: Sequence[Any],
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
        execution_mode: str = INTERACTIVE_CONFIRMATION,
    ) -> AgentPlan:
        self._purge()
        try:
            execution_mode = validate_agent_execution_mode(execution_mode)
        except ValueError as exc:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", str(exc)) from exc
        if not request_id.strip() or not raw_command.strip() or not registry_epoch or not registry_digest:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", "plan request fields are incomplete")
        try:
            steps = normalize_workflow_steps(workflow_steps, max_steps=MAX_PLAN_STEPS)
        except (TypeError, ValueError) as exc:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", str(exc)) from exc
        if registry_generation <= 0:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", "registry generation must be positive")

        existing_token = self._request_index.get(request_id)
        if existing_token:
            existing = self._records.get(existing_token)
            if existing is not None:
                if _plan_payload_matches(
                    existing.plan,
                    raw_command=raw_command,
                    workflow_steps=steps,
                    registry_epoch=registry_epoch,
                    registry_generation=registry_generation,
                    registry_digest=registry_digest,
                    execution_mode=execution_mode,
                ):
                    return existing.plan
                raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "request_id payload conflicts with the stored plan")

        self._ensure_capacity()
        plan = AgentPlan(
            schema_version=1,
            plan_id=self._plan_id_factory(),
            plan_token=_new_opaque_token(self._token_factory),
            plan_kind=AgentPlan.SINGLE_SKILL if len(steps) == 1 else AgentPlan.WORKFLOW,
            raw_command=raw_command,
            workflow_steps=steps,
            plan_digest=compute_plan_digest(
                raw_command=raw_command,
                workflow_steps=steps,
                registry_epoch=registry_epoch,
                registry_generation=registry_generation,
                registry_digest=registry_digest,
            ),
            registry_epoch=registry_epoch,
            registry_generation=registry_generation,
            registry_digest=registry_digest,
            expires_at=_wall_time(self._wall_now() + self._ttl_sec),
            execution_mode=execution_mode,
        )
        self._records[plan.plan_token] = _PlanRecord(
            plan=plan,
            request_id=request_id,
            monotonic_expires_at=self._clock() + self._ttl_sec,
        )
        self._request_index[request_id] = plan.plan_token
        return plan

    def get(self, plan_token: str) -> AgentPlan:
        record = self._get_record(plan_token)
        return record.plan

    def get_for_execution(self, plan_token: str) -> AgentPlan:
        """Return a plan for admission proof, including immutable terminal records."""
        record = self._get_record(plan_token, allow_terminal=True)
        return record.plan

    def validate(
        self,
        *,
        plan_token: str,
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
    ) -> AgentPlan:
        record = self._get_record(plan_token)
        self._require_identity(record.plan, registry_epoch, registry_generation, registry_digest)
        return record.plan

    def mark_validated(
        self,
        *,
        plan_token: str,
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
    ) -> AgentPlan:
        record = self._get_record(plan_token)
        self._require_identity(record.plan, registry_epoch, registry_generation, registry_digest)
        if record.state == "PLANNED":
            record.state = "VALIDATED"
        elif record.state != "VALIDATED":
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan can no longer be validated")
        return record.plan

    def confirm(
        self,
        *,
        plan_token: str,
        plan_digest: str,
        task_id: str,
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
        task_budget_sec: float,
        execution_mode: str = INTERACTIVE_CONFIRMATION,
    ) -> PlanConfirmation:
        record = self._get_record(plan_token)
        try:
            execution_mode = validate_agent_execution_mode(execution_mode)
        except ValueError as exc:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", str(exc)) from exc
        if record.plan.execution_mode != execution_mode:
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "execution mode does not match the planned mode")
        self._require_identity(record.plan, registry_epoch, registry_generation, registry_digest)
        if not hmac.compare_digest(record.plan.plan_digest, plan_digest):
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan digest does not match")
        if not task_id.strip():
            raise AgentPlanError("SKILL_SCHEMA_INVALID", "task_id must be non-empty")
        normalized_budget = _float32(task_budget_sec)
        if isinstance(task_budget_sec, bool) or not math.isfinite(normalized_budget) or normalized_budget <= 0.0:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", "task_budget_sec must be finite and positive")
        if record.state != "VALIDATED":
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan must be validated before confirmation")
        confirmation_token = _new_opaque_token(self._token_factory)
        started_at = _wall_time(self._wall_now())
        deadline = _wall_time(started_at[0] + started_at[1] / 1_000_000_000 + normalized_budget)
        record.state = "CONFIRMED"
        record.task_id = task_id
        record.confirmation_token = confirmation_token
        record.task_budget_sec = normalized_budget
        record.task_budget_started_at = started_at
        record.task_budget_deadline = deadline
        record.clock_at_confirmation = started_at[0] + started_at[1] / 1_000_000_000
        return PlanConfirmation(
            True,
            confirmation_token,
            record.plan.plan_digest,
            task_id,
            record.task_budget_sec,
            started_at,
            deadline,
        )

    def accept_execution(
        self,
        *,
        plan_token: str,
        confirmation_token: str,
        task_id: str,
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
        task_budget_sec: float,
    ) -> AgentPlanExecution:
        record = self._get_record(plan_token, allow_terminal=True)
        if record.task_id and record.task_id != task_id:
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan is bound to a different task")
        if (
            isinstance(task_budget_sec, bool)
            or not math.isfinite(task_budget_sec)
            or task_budget_sec <= 0.0
            or _float32(task_budget_sec) != record.task_budget_sec
        ):
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "execution budget does not match confirmation")
        if record.state in {"ACCEPTED", "TERMINAL"}:
            if not hmac.compare_digest(record.confirmation_token, confirmation_token):
                raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "confirmation token does not match")
            return AgentPlanExecution(
                plan=record.plan,
                task_id=record.task_id,
                state=record.state,
                confirmation_token=record.confirmation_token,
                task_budget_sec=record.task_budget_sec,
                task_budget_started_at=record.task_budget_started_at,
                task_budget_deadline=record.task_budget_deadline,
                clock_at_confirmation=record.clock_at_confirmation,
                terminal_code=record.terminal_code,
                terminal_message=record.terminal_message,
                workflow_digest=record.workflow_digest,
                completed_step_count=record.completed_step_count,
            )
        self._require_identity(record.plan, registry_epoch, registry_generation, registry_digest)
        if record.state != "CONFIRMED" or not hmac.compare_digest(record.confirmation_token, confirmation_token):
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan confirmation is invalid")
        record.state = "ACCEPTED"
        record.execution_token = _new_opaque_token(self._token_factory)
        return AgentPlanExecution(
            record.plan,
            task_id,
            record.state,
            record.confirmation_token,
            record.task_budget_sec,
            record.task_budget_started_at,
            record.task_budget_deadline,
            record.clock_at_confirmation,
            newly_accepted=True,
            execution_token=record.execution_token,
        )

    def mark_terminal(
        self,
        *,
        plan_token: str,
        task_id: str,
        execution_token: str,
        terminal_code: str = "",
        terminal_message: str = "",
        workflow_digest: str = "",
        completed_step_count: int = 0,
    ) -> AgentPlanExecution:
        record = self._get_record(plan_token, allow_terminal=True)
        if record.task_id != task_id:
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan is bound to a different task")
        if not record.execution_token or not hmac.compare_digest(record.execution_token, execution_token):
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "execution owner does not match")
        if record.state not in {"ACCEPTED", "TERMINAL"}:
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan has not been accepted")
        if record.state == "TERMINAL":
            if (
                record.terminal_code,
                record.terminal_message,
                record.workflow_digest,
                record.completed_step_count,
            ) != (terminal_code, terminal_message, workflow_digest, completed_step_count):
                raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "terminal result is immutable")
            return AgentPlanExecution(
                plan=record.plan,
                task_id=task_id,
                state=record.state,
                confirmation_token=record.confirmation_token,
                task_budget_sec=record.task_budget_sec,
                task_budget_started_at=record.task_budget_started_at,
                task_budget_deadline=record.task_budget_deadline,
                clock_at_confirmation=record.clock_at_confirmation,
                terminal_code=record.terminal_code,
                terminal_message=record.terminal_message,
                workflow_digest=record.workflow_digest,
                completed_step_count=record.completed_step_count,
            )
        record.state = "TERMINAL"
        record.terminal_code = terminal_code
        record.terminal_message = terminal_message
        record.workflow_digest = workflow_digest
        record.completed_step_count = completed_step_count
        record.monotonic_expires_at = self._clock() + self._ttl_sec
        return AgentPlanExecution(
            plan=record.plan,
            task_id=task_id,
            state=record.state,
            confirmation_token=record.confirmation_token,
            task_budget_sec=record.task_budget_sec,
            task_budget_started_at=record.task_budget_started_at,
            task_budget_deadline=record.task_budget_deadline,
            clock_at_confirmation=record.clock_at_confirmation,
            terminal_code=terminal_code,
            terminal_message=terminal_message,
            workflow_digest=workflow_digest,
            completed_step_count=completed_step_count,
        )

    def cancel(self, *, plan_token: str, task_id: str, execution_token: str) -> AgentPlanExecution:
        return self.mark_terminal(
            plan_token=plan_token,
            task_id=task_id,
            execution_token=execution_token,
            terminal_code="SKILL_CANCELLED",
        )

    def _wall_now(self) -> float:
        now = float(self._wall_clock())
        if not math.isfinite(now) or now < 0.0:
            raise AgentPlanError("CAPABILITY_NOT_READY", "ROS clock is invalid")
        if self._last_wall_time is not None and now < self._last_wall_time:
            raise AgentPlanError("CAPABILITY_NOT_READY", "ROS clock moved backwards")
        self._last_wall_time = now
        return now

    def _get_record(self, plan_token: str, *, allow_terminal: bool = False) -> _PlanRecord:
        self._purge()
        record = self._records.get(plan_token)
        if record is None:
            if plan_token in self._expired_tokens:
                raise AgentPlanError("SKILL_AGENT_PLAN_EXPIRED")
            raise AgentPlanError("SKILL_AGENT_PLAN_NOT_FOUND")
        if record.state == "TERMINAL" and not allow_terminal:
            raise AgentPlanError("SKILL_AGENT_PLAN_EXPIRED", "plan is no longer active")
        return record

    def _require_identity(self, plan: AgentPlan, epoch: str, generation: int, digest: str) -> None:
        if (plan.registry_epoch, plan.registry_generation, plan.registry_digest) != (epoch, generation, digest):
            raise AgentPlanError("SKILL_REGISTRY_VERSION_MISMATCH", "plan snapshot identity is stale")

    def _ensure_capacity(self) -> None:
        self._purge()
        while len(self._records) >= self._max_records:
            candidates = [
                (record.monotonic_expires_at, token)
                for token, record in self._records.items()
                if record.state != "ACCEPTED"
            ]
            if not candidates:
                raise AgentPlanError("SKILL_EXECUTION_BUSY", "plan store is full of active executions")
            _, token = min(candidates)
            self._remove(token)

    def _purge(self) -> None:
        now = self._clock()
        for token, record in list(self._records.items()):
            if record.monotonic_expires_at <= now and record.state != "ACCEPTED":
                self._expired_tokens[token] = None
                self._expired_tokens.move_to_end(token)
                while len(self._expired_tokens) > self._max_records:
                    self._expired_tokens.popitem(last=False)
                self._remove(token)

    def _remove(self, token: str) -> None:
        record = self._records.pop(token, None)
        if record is not None and self._request_index.get(record.request_id) == token:
            self._request_index.pop(record.request_id, None)


def compute_plan_digest(
    *,
    raw_command: str,
    workflow_steps: Sequence[Any],
    registry_epoch: str,
    registry_generation: int,
    registry_digest: str,
) -> str:
    steps = normalize_workflow_steps(workflow_steps, max_steps=MAX_PLAN_STEPS)
    preimage = {
        "schema_version": 1,
        "raw_command": raw_command,
        "workflow_steps": [step.to_dict() for step in steps],
        "registry_epoch": registry_epoch,
        "registry_generation": registry_generation,
        "registry_digest": registry_digest,
    }
    return sha256_text(to_canonical_json(preimage))


def _plan_payload_matches(
    plan: AgentPlan,
    *,
    raw_command: str,
    workflow_steps: Sequence[CanonicalWorkflowStep],
    registry_epoch: str,
    registry_generation: int,
    registry_digest: str,
    execution_mode: str,
) -> bool:
    return (
        plan.raw_command == raw_command
        and plan.workflow_steps == tuple(workflow_steps)
        and plan.registry_epoch == registry_epoch
        and plan.registry_generation == registry_generation
        and plan.registry_digest == registry_digest
        and plan.execution_mode == execution_mode
    )


def _new_opaque_token(factory: Callable[[], str]) -> str:
    token = factory()
    if not isinstance(token, str) or len(token.encode("utf-8")) < 16:
        raise AgentPlanError("SKILL_SCHEMA_INVALID", "token factory did not provide an opaque token")
    return token


def _wall_time(timestamp: float) -> tuple[int, int]:
    seconds = int(timestamp)
    nanoseconds = int(round((timestamp - seconds) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    return seconds, nanoseconds
