"""ROS-free state core for the global inference scheduler.

A product session is a logical lifecycle fence. It may lazily bind more than
one pipeline because every Dispatch request can select a different target and
fallback chain. Pipeline generations remain private to their bindings; callers
continue to use the logical generation returned by the public Open action.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from inference_service.scheduler.idempotency import validate_pipeline_id, validate_uuid4


class GlobalSessionState(str, Enum):
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    QUARANTINED = "quarantined"


class BindingState(str, Enum):
    OPENING = "opening"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class PipelineCandidate:
    """Static pipeline registry entry resolved from robot_config."""

    pipeline_id: str
    compatibility_group: str
    hardware_resource_id: str
    hardware_profile_fingerprint: str
    deployment_fingerprint: str
    runtime_policy_fingerprint: str
    endpoint_open: str
    endpoint_dispatch: str
    endpoint_close: str
    endpoint_serving_status: str
    profile_path: str
    profile_compatibility_fingerprint: str = ""
    required: bool = True
    public_capacity: dict[str, int] | None = None


@dataclass
class PipelineBinding:
    pipeline_id: str
    hardware_resource_id: str
    binding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    binding_incarnation: int = 1
    state: BindingState = BindingState.OPENING
    pipeline_generation: int = 0
    quarantine: bool = False
    observed_boot_id: str | None = None
    expected_boot_id: str | None = None
    close_operation_id: str | None = None


@dataclass
class SessionRecord:
    """Logical session plus all pipeline-local bindings used by it."""

    session_generation: int
    last_activity_mono_ns: int
    lease_expires_at_ns: int
    state: GlobalSessionState
    bindings: dict[str, PipelineBinding] = field(default_factory=dict)
    quarantine: bool = False
    unresolved_cleanup: bool = False
    close_requested: bool = False
    close_retry_pending: bool = False
    in_flight_requests: int = 0
    product_request_count: int = 0


@dataclass(frozen=True)
class LogicalOpenDecision:
    session_generation: int
    lease_expires_at_ns: int
    replay: bool = False


@dataclass(frozen=True)
class DispatchPlan:
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class BindingDecision:
    pipeline_id: str | None
    reason: str = ""
    pipeline_generation: int = 0
    needs_open: bool = False
    binding_id: str = ""
    binding_incarnation: int = 0
    expected_boot_id: str = ""


class SchedulerError(Exception):
    def __init__(self, message: str, *, code: str = "scheduler_error") -> None:
        super().__init__(message)
        self.code = code


class GlobalSchedulerCore:
    """Thread-safe lifecycle, binding, fencing, and quarantine owner."""

    def __init__(
        self,
        *,
        candidates: list[PipelineCandidate],
        max_session_records: int,
        max_product_requests_per_session: int,
        terminal_session_retention_ns: int,
        session_idle_timeout_ns: int,
        max_fallback_pipelines: int,
        now_ns: Callable[[], int],
    ) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._candidates = {candidate.pipeline_id: candidate for candidate in candidates}
        self._now_ns = now_ns
        self._session_idle_timeout_ns = session_idle_timeout_ns
        self._max_fallback = max_fallback_pipelines
        self._max_session_records = max_session_records
        self._max_product_requests_per_session = max_product_requests_per_session
        self._terminal_session_retention_ns = terminal_session_retention_ns
        self._sessions: dict[str, SessionRecord] = {}
        self._quarantined_pipelines: set[str] = set()
        self._session_terminal_ns: dict[str, int] = {}
        self._public_dispatches: set[tuple[str, int, str]] = set()

    # ------------------------------------------------------------------
    # Open lifecycle.
    # ------------------------------------------------------------------

    def open_session(self, *, session_id: str) -> LogicalOpenDecision:
        """Create or replay a route-independent logical session."""

        validate_uuid4(session_id, field="session_id")
        with self._condition:
            self._purge_terminal_sessions_locked()
            record = self._sessions.get(session_id)
            if record is not None:
                if record.state == GlobalSessionState.ACTIVE:
                    return LogicalOpenDecision(
                        record.session_generation,
                        record.lease_expires_at_ns,
                        replay=True,
                    )
                raise SchedulerError(f"session_{record.state.value}")
            if len(self._sessions) >= self._max_session_records:
                raise SchedulerError("max_session_records")
            now = self._now_ns()
            record = SessionRecord(
                session_generation=1,
                last_activity_mono_ns=now,
                lease_expires_at_ns=now + self._session_idle_timeout_ns,
                state=GlobalSessionState.ACTIVE,
            )
            self._sessions[session_id] = record
            return LogicalOpenDecision(record.session_generation, record.lease_expires_at_ns)

    def record_binding_open_success(
        self,
        *,
        session_id: str,
        pipeline_id: str,
        pipeline_generation: int,
        hardware_resource_id: str,
        binding_id: str | None = None,
        binding_incarnation: int | None = None,
        boot_id: str | None = None,
    ) -> bool:
        """CAS an OPENING binding to ACTIVE.

        Returns true when Close raced with Open and the accepted binding must be
        cleaned up instead of being used for the pending Dispatch.
        """
        if pipeline_generation <= 0:
            raise SchedulerError("invalid_pipeline_generation")
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return True
            binding = record.bindings.get(pipeline_id)
            if binding is None or binding.state != BindingState.OPENING:
                return record.close_requested
            if binding_id is not None and binding.binding_id != binding_id:
                return True
            if binding_incarnation is not None and binding.binding_incarnation != binding_incarnation:
                return True
            if boot_id is not None and binding.expected_boot_id != boot_id:
                binding.state = BindingState.QUARANTINED
                binding.quarantine = True
                record.state = GlobalSessionState.QUARANTINED
                record.quarantine = True
                record.unresolved_cleanup = True
                return True
            binding.pipeline_generation = pipeline_generation
            binding.observed_boot_id = boot_id or binding.expected_boot_id
            binding.hardware_resource_id = hardware_resource_id
            binding.state = BindingState.ACTIVE
            if record.close_requested:
                record.state = GlobalSessionState.CLOSING
            record.last_activity_mono_ns = self._now_ns()
            record.lease_expires_at_ns = record.last_activity_mono_ns + self._session_idle_timeout_ns
            self._condition.notify_all()
            return record.close_requested

    def release_binding_open(
        self,
        *,
        session_id: str,
        not_started: bool,
        pipeline_id: str | None = None,
        pipeline_generation: int = 0,
    ) -> None:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return
            if pipeline_id is None:
                pipeline_id = next(
                    (pid for pid, binding in record.bindings.items() if binding.state == BindingState.OPENING),
                    None,
                )
            binding = record.bindings.get(pipeline_id or "")
            if not_started:
                if binding is not None and binding.state == BindingState.OPENING:
                    record.bindings.pop(binding.pipeline_id, None)
                self._condition.notify_all()
                return
            if binding is not None:
                binding.state = BindingState.QUARANTINED
                binding.pipeline_generation = max(binding.pipeline_generation, pipeline_generation)
                binding.quarantine = True
                self._quarantined_pipelines.add(binding.pipeline_id)
            record.state = GlobalSessionState.QUARANTINED
            record.quarantine = True
            record.unresolved_cleanup = True
            record.last_activity_mono_ns = self._now_ns()
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Per-request routing and lazy pipeline bindings.
    # ------------------------------------------------------------------

    def resolve_dispatch_plan(
        self,
        *,
        session_id: str,
        session_generation: int,
        target_pipeline_id: str,
        fallback_chain: list[str],
        priority: int,
        request_id: str | None = None,
    ) -> DispatchPlan:
        with self._condition:
            record = self._active_session_locked(session_id, session_generation)
            if record.product_request_count >= self._max_product_requests_per_session:
                raise SchedulerError("max_product_requests_per_session")
            ordered = self._validate_candidate_chain_locked(
                target_pipeline_id,
                fallback_chain if priority == 0 else [],
            )
            record.last_activity_mono_ns = self._now_ns()
            record.lease_expires_at_ns = record.last_activity_mono_ns + self._session_idle_timeout_ns
            record.in_flight_requests += 1
            record.product_request_count += 1
            if request_id is not None:
                validate_uuid4(request_id, field="request_id")
                key = (session_id, session_generation, request_id)
                if key in self._public_dispatches:
                    record.in_flight_requests -= 1
                    record.product_request_count -= 1
                    raise SchedulerError("duplicate_public_dispatch")
                self._public_dispatches.add(key)
            return DispatchPlan(ordered)

    def prepare_dispatch_candidate(
        self,
        *,
        session_id: str,
        session_generation: int,
        pipeline_id: str,
        expected_boot_id: str | None = None,
    ) -> BindingDecision:
        with self._condition:
            record = self._active_session_locked(session_id, session_generation)
            if pipeline_id in self._quarantined_pipelines:
                return BindingDecision(None, "pipeline_quarantined")
            binding = record.bindings.get(pipeline_id)
            if binding is not None:
                if binding.state == BindingState.ACTIVE:
                    if expected_boot_id is not None and binding.expected_boot_id != expected_boot_id:
                        return BindingDecision(None, "boot_mismatch")
                    return BindingDecision(
                        pipeline_id,
                        pipeline_generation=binding.pipeline_generation,
                        binding_id=binding.binding_id,
                        binding_incarnation=binding.binding_incarnation,
                        expected_boot_id=binding.expected_boot_id or "",
                    )
                if binding.state == BindingState.OPENING:
                    return BindingDecision(
                        pipeline_id,
                        "binding_open_in_progress",
                        binding_id=binding.binding_id,
                        binding_incarnation=binding.binding_incarnation,
                        expected_boot_id=binding.expected_boot_id or "",
                    )
                return BindingDecision(None, "pipeline_quarantined")
            if self._pipeline_owned_by_other_session_locked(pipeline_id, record):
                return BindingDecision(None, "pipeline_busy")
            candidate = self._candidates[pipeline_id]
            binding = PipelineBinding(
                pipeline_id,
                candidate.hardware_resource_id,
                expected_boot_id=expected_boot_id,
                observed_boot_id=expected_boot_id,
            )
            record.bindings[pipeline_id] = binding
            return BindingDecision(
                pipeline_id,
                needs_open=True,
                binding_id=binding.binding_id,
                binding_incarnation=binding.binding_incarnation,
                expected_boot_id=expected_boot_id or "",
            )

    def release_dispatch_candidate(self, *, session_id: str, pipeline_id: str, not_started: bool) -> None:
        if not_started:
            self.release_binding_open(session_id=session_id, pipeline_id=pipeline_id, not_started=True)
        else:
            self.mark_session_failed(session_id, pipeline_id=pipeline_id)

    def wait_for_binding_open(
        self,
        *,
        session_id: str,
        pipeline_id: str,
        deadline_monotonic_ns: int,
    ) -> bool:
        """Wait until another request's lazy Open stops being in flight.

        A caller that observes an OPENING binding must not send Dispatch with
        generation zero. Returning after the binding is removed also lets the
        caller retry admission and become the new opener when the first Open
        was proven not to have started.
        """
        with self._condition:
            while True:
                record = self._sessions.get(session_id)
                binding = record.bindings.get(pipeline_id) if record is not None else None
                if binding is None or binding.state != BindingState.OPENING:
                    return True
                remaining_ns = deadline_monotonic_ns - self._now_ns()
                if remaining_ns <= 0:
                    return False
                self._condition.wait(remaining_ns / 1_000_000_000)

    def record_request_terminal(
        self,
        session_id: str,
        *,
        request_id: str | None = None,
        session_generation: int | None = None,
    ) -> None:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is not None and record.in_flight_requests > 0:
                record.in_flight_requests -= 1
                if request_id is not None:
                    generation = session_generation if session_generation is not None else record.session_generation
                    self._public_dispatches.discard((session_id, generation, request_id))
                self._retain_drained_failure_locked(session_id, record)
                self._condition.notify_all()

    # ------------------------------------------------------------------
    # Close and cleanup.
    # ------------------------------------------------------------------

    def begin_close(self, *, session_id: str, session_generation: int) -> None:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                raise SchedulerError("session_not_found")
            if session_generation not in (0, record.session_generation):
                raise SchedulerError("generation_mismatch")
            if record.state == GlobalSessionState.CLOSED:
                return
            if record.state == GlobalSessionState.CLOSING and record.close_requested and not record.close_retry_pending:
                raise SchedulerError("close_in_progress")
            record.close_requested = True
            record.close_retry_pending = False
            record.state = GlobalSessionState.CLOSING
            for binding in record.bindings.values():
                if binding.close_operation_id is None:
                    binding.close_operation_id = str(uuid.uuid4())
                elif binding.state is not BindingState.CLOSED:
                    # Retries re-send Close with a fresh operation id: the
                    # previous attempt's UNKNOWN outcome must not be replayed
                    # from the idempotency cache (P0-2 no-replay contract).
                    binding.close_operation_id = str(uuid.uuid4())
            self._condition.notify_all()

    def wait_for_bindings_to_settle(self, session_id: str, deadline_monotonic_ns: int) -> bool:
        with self._condition:
            while True:
                record = self._sessions.get(session_id)
                if record is None:
                    return True
                # Close is a drain barrier for both lazy Opens and product
                # Dispatches. Do not return while a request still owns the
                # logical session.
                if record.in_flight_requests == 0 and not any(
                    binding.state == BindingState.OPENING for binding in record.bindings.values()
                ):
                    return True
                remaining_ns = deadline_monotonic_ns - self._now_ns()
                if remaining_ns <= 0:
                    return False
                self._condition.wait(remaining_ns / 1_000_000_000)

    def close_bindings(self, session_id: str) -> list[PipelineBinding]:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return []
            result: list[PipelineBinding] = []
            for binding in record.bindings.values():
                if binding.state in (BindingState.ACTIVE, BindingState.QUARANTINED, BindingState.CLOSING):
                    binding.state = BindingState.CLOSING
                    if binding.close_operation_id is None:
                        binding.close_operation_id = str(uuid.uuid4())
                    result.append(
                        PipelineBinding(
                            pipeline_id=binding.pipeline_id,
                            hardware_resource_id=binding.hardware_resource_id,
                            binding_id=binding.binding_id,
                            binding_incarnation=binding.binding_incarnation,
                            state=binding.state,
                            pipeline_generation=binding.pipeline_generation,
                            quarantine=binding.quarantine,
                            observed_boot_id=binding.observed_boot_id,
                            expected_boot_id=binding.expected_boot_id,
                            close_operation_id=binding.close_operation_id,
                        )
                    )
            return sorted(result, key=lambda item: item.pipeline_id)

    def record_binding_close_success(self, session_id: str, pipeline_id: str) -> None:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return
            record.bindings.pop(pipeline_id, None)
            self._retain_drained_failure_locked(session_id, record)
            if not any(other.unresolved_cleanup and pipeline_id in other.bindings for other in self._sessions.values()):
                self._quarantined_pipelines.discard(pipeline_id)
            self._condition.notify_all()

    def record_close_not_started(self, session_id: str) -> None:
        """Allow another Close attempt without reopening product admission."""
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return
            record.state = GlobalSessionState.CLOSING
            record.close_retry_pending = True
            record.unresolved_cleanup = bool(record.bindings)
            record.last_activity_mono_ns = self._now_ns()
            self._condition.notify_all()

    def record_close_complete(self, session_id: str, *, success: bool, unresolved: bool = False) -> int:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return 0
            now = self._now_ns()
            if success and not record.bindings and record.in_flight_requests == 0:
                record.session_generation = max(1, record.session_generation + 1)
                record.state = GlobalSessionState.CLOSED
                record.quarantine = False
                record.unresolved_cleanup = False
                self._session_terminal_ns[session_id] = now
            else:
                record.state = GlobalSessionState.QUARANTINED
                record.quarantine = True
                record.unresolved_cleanup = unresolved or bool(record.bindings)
                if not record.unresolved_cleanup:
                    self._retain_drained_failure_locked(session_id, record)
            record.last_activity_mono_ns = now
            self._condition.notify_all()
            return record.session_generation

    # ------------------------------------------------------------------
    # Quarantine, lease, retention, and introspection.
    # ------------------------------------------------------------------

    def mark_session_failed(self, session_id: str, *, pipeline_id: str | None = None) -> None:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return
            record.state = GlobalSessionState.QUARANTINED
            record.quarantine = True
            record.unresolved_cleanup = True
            record.last_activity_mono_ns = self._now_ns()
            targets = [pipeline_id] if pipeline_id else list(record.bindings)
            for target in targets:
                binding = record.bindings.get(target)
                if binding is not None:
                    binding.state = BindingState.QUARANTINED
                    binding.quarantine = True
                    self._quarantined_pipelines.add(target)
            self._condition.notify_all()

    def reconcile_pipeline_boot(self, pipeline_id: str, *, boot_id: str | None = None) -> None:
        """Fence bindings from a previous pipeline process boot."""
        with self._condition:
            self._quarantined_pipelines.discard(pipeline_id)
            for session_id, record in self._sessions.items():
                binding = record.bindings.get(pipeline_id)
                if binding is None:
                    continue
                if boot_id is None or binding.expected_boot_id is None:
                    binding.state = BindingState.QUARANTINED
                    binding.quarantine = True
                    record.state = GlobalSessionState.QUARANTINED
                    record.quarantine = True
                    record.unresolved_cleanup = True
                    self._quarantined_pipelines.add(pipeline_id)
                    continue
                if binding.expected_boot_id == boot_id:
                    continue
                # A new boot fences the old process. Sending Close with the
                # old boot identity would be rejected by the new process.
                record.bindings.pop(pipeline_id, None)
                if record.state != GlobalSessionState.ACTIVE or record.close_requested or record.in_flight_requests > 0:
                    record.state = GlobalSessionState.QUARANTINED
                    record.quarantine = True
                    record.unresolved_cleanup = bool(record.bindings)
                    self._retain_drained_failure_locked(session_id, record)
            self._condition.notify_all()

    def _retain_drained_failure_locked(self, session_id: str, record: SessionRecord) -> None:
        """Retain the failure result only after every resource owner is fenced."""
        if record.state == GlobalSessionState.QUARANTINED and not record.bindings and record.in_flight_requests == 0:
            record.unresolved_cleanup = False
            self._session_terminal_ns.setdefault(session_id, self._now_ns())

    def expired_sessions(self) -> list[str]:
        with self._condition:
            now = self._now_ns()
            return [
                session_id
                for session_id, record in self._sessions.items()
                if record.state == GlobalSessionState.ACTIVE
                and record.in_flight_requests == 0
                and now >= record.lease_expires_at_ns
            ]

    def aged_quarantined_sessions(self, age_ns: int) -> list[str]:
        """Sessions stuck QUARANTINED/CLOSING beyond the recovery window.

        The idle sweep retries Close once. Failed retries retain ownership
        until a successful Close or a trusted boot fence.
        """

        with self._condition:
            now = self._now_ns()
            return [
                session_id
                for session_id, record in self._sessions.items()
                if record.state in (GlobalSessionState.QUARANTINED, GlobalSessionState.CLOSING)
                and record.in_flight_requests == 0
                and now - record.last_activity_mono_ns >= age_ns
            ]

    def session_record(self, session_id: str) -> SessionRecord | None:
        with self._condition:
            return self._sessions.get(session_id)

    def session_state(self, session_id: str) -> GlobalSessionState | None:
        with self._condition:
            record = self._sessions.get(session_id)
            return record.state if record else None

    def _active_session_locked(self, session_id: str, session_generation: int) -> SessionRecord:
        record = self._sessions.get(session_id)
        if record is None:
            raise SchedulerError("session_not_found")
        if record.state != GlobalSessionState.ACTIVE:
            raise SchedulerError("session_failed" if record.quarantine else "session_not_active")
        if record.session_generation != session_generation:
            raise SchedulerError("generation_mismatch")
        return record

    def _validate_candidate_chain_locked(self, target_pipeline_id: str, fallback_chain: list[str]) -> tuple[str, ...]:
        validate_pipeline_id(target_pipeline_id, field="target_pipeline_id")
        if len(fallback_chain) > self._max_fallback:
            raise SchedulerError("max_fallback_exceeded")
        ordered = (target_pipeline_id, *fallback_chain)
        if len(ordered) != len(set(ordered)):
            raise SchedulerError("invalid_fallback_chain")
        for pipeline_id in ordered:
            validate_pipeline_id(pipeline_id, field="fallback_pipeline_id")
            if pipeline_id not in self._candidates:
                raise SchedulerError(
                    "unknown_target_pipeline" if pipeline_id == target_pipeline_id else "unknown_fallback_pipeline"
                )
        target_group = self._candidates[target_pipeline_id].compatibility_group
        if any(self._candidates[pipeline_id].compatibility_group != target_group for pipeline_id in ordered[1:]):
            raise SchedulerError("fallback_compatibility_group_mismatch")
        return ordered

    def _pipeline_owned_by_other_session_locked(self, pipeline_id: str, owner: SessionRecord) -> bool:
        for record in self._sessions.values():
            if record is owner:
                continue
            binding = record.bindings.get(pipeline_id)
            if binding is not None and binding.state in {
                BindingState.OPENING,
                BindingState.ACTIVE,
                BindingState.CLOSING,
                BindingState.QUARANTINED,
            }:
                return True
        return False

    def _purge_terminal_sessions_locked(self) -> None:
        now = self._now_ns()
        expired = [
            session_id
            for session_id, terminal_ns in self._session_terminal_ns.items()
            if now - terminal_ns > self._terminal_session_retention_ns
            and not self._sessions[session_id].unresolved_cleanup
            and not self._sessions[session_id].bindings
            and self._sessions[session_id].in_flight_requests == 0
        ]
        for session_id in expired:
            self._session_terminal_ns.pop(session_id, None)
            self._sessions.pop(session_id, None)


__all__ = [
    "BindingDecision",
    "BindingState",
    "DispatchPlan",
    "GlobalSchedulerCore",
    "GlobalSessionState",
    "LogicalOpenDecision",
    "PipelineBinding",
    "PipelineCandidate",
    "SchedulerError",
    "SessionRecord",
]
