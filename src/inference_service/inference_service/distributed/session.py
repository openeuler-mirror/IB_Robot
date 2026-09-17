"""Handshake sessions, heartbeat invalidation, and request routing guards."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from inference_service.pipeline import PipelineState, PipelineStateMachine

from .types import (
    DistributedRequest,
    DistributedResult,
    Operation,
    PeerRole,
    PipelineIdentity,
    PipelineStatus,
    StreamReference,
    StructuredError,
    UnsupportedDistributedRuntimeError,
    identity_error,
)


class DistributedProtocolError(RuntimeError):
    def __init__(self, error: StructuredError) -> None:
        super().__init__(error.message)
        self.error = error
        self.code = error.code
        self.recoverable = error.recoverable
        self.details = error.details


@dataclass(frozen=True)
class SessionUpdate:
    invalidated_request_ids: tuple[str, ...] = ()
    error: StructuredError | None = None
    canceled_request_id: str = ""


class EdgeSession:
    """Own the edge view of one revocable cloud handshake session."""

    def __init__(
        self,
        identity: PipelineIdentity,
        *,
        heartbeat_timeout: float = 2.0,
        runtime_interface: str = "policy",
        runtime_model_type: str = "",
    ) -> None:
        if runtime_interface != "policy":
            raise UnsupportedDistributedRuntimeError(runtime_interface, runtime_model_type)
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive")
        self.identity = identity
        self._heartbeat_timeout = heartbeat_timeout
        self._state = PipelineStateMachine()
        self._lock = threading.RLock()
        self._sequence = 0
        self._session_id = ""
        self._generation = 0
        self._last_cloud_sequence = 0
        self._retired_sessions: set[tuple[str, int]] = set()
        self._last_heartbeat = 0.0
        self._pending: dict[str, tuple[Operation, str]] = {}
        self._reset_supported = False
        self._cancellation_supported = False
        self._last_error: StructuredError | None = None

    @property
    def state(self) -> PipelineState:
        with self._lock:
            return self._state.state

    @property
    def ready(self) -> bool:
        return self.state is PipelineState.READY

    @property
    def reset_supported(self) -> bool:
        with self._lock:
            return self._reset_supported

    @property
    def cancellation_supported(self) -> bool:
        with self._lock:
            return self._cancellation_supported

    @property
    def session(self) -> tuple[str, int]:
        with self._lock:
            return self._session_id, self._generation

    def start(self) -> None:
        with self._lock:
            self._state.transition(PipelineState.LOADING)
            self._state.transition(PipelineState.HANDSHAKING)

    def local_status(self) -> PipelineStatus:
        with self._lock:
            self._sequence += 1
            return PipelineStatus(
                role=PeerRole.EDGE,
                identity=self.identity,
                sequence=self._sequence,
                session_id=self._session_id,
                session_generation=self._generation,
                ready=self._state.state is PipelineState.READY,
                runtime_state=self._state.state.value,
                reset_supported=self._reset_supported,
                cancellation_supported=self._cancellation_supported,
                error=self._last_error,
            )

    def observe_cloud(self, status: PipelineStatus, *, now: float | None = None) -> SessionUpdate:
        with self._lock:
            if self._state.state is PipelineState.FAILED:
                return SessionUpdate(error=self._last_error)
            if status.role is not PeerRole.CLOUD:
                return self._invalidate_locked(
                    StructuredError(
                        code="unexpected_peer_role",
                        message="edge received a non-cloud distributed status",
                        stage="handshake",
                    )
                )
            mismatch = identity_error(self.identity, status.identity)
            if mismatch is not None:
                return self._invalidate_locked(mismatch)
            if status.error is not None:
                return self._invalidate_locked(status.error)
            if not status.ready or not status.session_id or status.session_generation < 1:
                return self._invalidate_locked(
                    StructuredError(
                        code="remote_not_ready",
                        message=f"cloud backend is not ready ({status.runtime_state})",
                        stage="readiness",
                        recoverable=True,
                    )
                )

            remote_session = (status.session_id, status.session_generation)
            current_session = (self._session_id, self._generation) if self._session_id else None
            if remote_session in self._retired_sessions:
                return SessionUpdate(
                    error=StructuredError(
                        code="stale_status",
                        message="discarded status for an invalidated cloud session",
                        stage="routing",
                        recoverable=True,
                    )
                )
            if current_session == remote_session and status.sequence <= self._last_cloud_sequence:
                return SessionUpdate(
                    error=StructuredError(
                        code="stale_status",
                        message="discarded out-of-order cloud status",
                        stage="routing",
                        recoverable=True,
                    )
                )

            invalidated: tuple[str, ...] = ()
            if current_session is not None and remote_session != current_session:
                self._retired_sessions.add(current_session)
                invalidated = tuple(sorted(self._pending))
                self._pending.clear()
            self._session_id = status.session_id
            self._generation = status.session_generation
            self._last_cloud_sequence = status.sequence
            self._last_heartbeat = time.monotonic() if now is None else now
            self._reset_supported = status.reset_supported
            self._cancellation_supported = status.cancellation_supported
            self._last_error = None
            if self._state.state in {PipelineState.HANDSHAKING, PipelineState.DEGRADED}:
                self._state.transition(PipelineState.READY)
            return SessionUpdate(invalidated_request_ids=invalidated)

    def expire_heartbeat(self, *, now: float | None = None) -> SessionUpdate:
        with self._lock:
            if self._state.state is not PipelineState.READY:
                return SessionUpdate()
            current = time.monotonic() if now is None else now
            if current - self._last_heartbeat <= self._heartbeat_timeout:
                return SessionUpdate()
            return self._invalidate_locked(
                StructuredError(
                    code="heartbeat_expired",
                    message="cloud heartbeat expired",
                    stage="transport",
                    recoverable=True,
                )
            )

    def prepare_request(
        self,
        operation: Operation,
        request_id: str,
        *,
        inputs: dict[str, object] | None = None,
        prompt: str | None = None,
        deadline: datetime | None = None,
        target_request_id: str = "",
        observation_timestamp_ns: int = 0,
        stream_references: tuple[StreamReference, ...] = (),
    ) -> DistributedRequest:
        with self._lock:
            return self._prepare_request_locked(
                operation,
                request_id,
                inputs=inputs,
                prompt=prompt,
                deadline=deadline,
                target_request_id=target_request_id,
                observation_timestamp_ns=observation_timestamp_ns,
                stream_references=stream_references,
            )

    def dispatch_request(
        self,
        operation: Operation,
        request_id: str,
        sender: Callable[[DistributedRequest], None],
        *,
        inputs: dict[str, object] | None = None,
        prompt: str | None = None,
        deadline: datetime | None = None,
        target_request_id: str = "",
        observation_timestamp_ns: int = 0,
        stream_references: tuple[StreamReference, ...] = (),
        aligned_timestamps_ns: tuple[int, ...] = (),
        aligned_tensors: tuple[Mapping[str, object], ...] = (),
    ) -> DistributedRequest:
        """Register and transmit one request without an invalidation window."""

        with self._lock:
            request = self._prepare_request_locked(
                operation,
                request_id,
                inputs=inputs,
                prompt=prompt,
                deadline=deadline,
                target_request_id=target_request_id,
                observation_timestamp_ns=observation_timestamp_ns,
                stream_references=stream_references,
                aligned_timestamps_ns=aligned_timestamps_ns,
                aligned_tensors=aligned_tensors,
            )
            try:
                sender(request)
            except Exception:
                self._pending.pop(request_id, None)
                raise
            return request

    def _prepare_request_locked(
        self,
        operation: Operation,
        request_id: str,
        *,
        inputs: dict[str, object] | None,
        prompt: str | None,
        deadline: datetime | None,
        target_request_id: str,
        observation_timestamp_ns: int,
        stream_references: tuple[StreamReference, ...],
        aligned_timestamps_ns: tuple[int, ...] = (),
        aligned_tensors: tuple[Mapping[str, object], ...] = (),
    ) -> DistributedRequest:
        if self._state.state is not PipelineState.READY or not self._session_id:
            raise DistributedProtocolError(
                StructuredError(
                    code="not_ready",
                    message=f"distributed pipeline is not ready ({self._state.state.value})",
                    stage="readiness",
                    recoverable=True,
                )
            )
        if request_id in self._pending:
            raise DistributedProtocolError(
                StructuredError(
                    code="duplicate_request_id",
                    message=f"distributed request {request_id!r} is already pending",
                    stage="routing",
                )
            )
        if operation is Operation.CANCEL:
            target = self._pending.get(target_request_id)
            if target is None or target[0] is not Operation.INFER:
                raise DistributedProtocolError(
                    StructuredError(
                        code="invalid_cancel_target",
                        message=f"cancellation target {target_request_id!r} is not a pending inference",
                        stage="routing",
                    )
                )
        if deadline is not None:
            if deadline.tzinfo is None:
                raise DistributedProtocolError(
                    StructuredError(
                        code="invalid_deadline",
                        message="distributed request deadline must be timezone-aware",
                        stage="validation",
                    )
                )
            if datetime.now(timezone.utc) >= deadline.astimezone(timezone.utc):
                raise DistributedProtocolError(
                    StructuredError(
                        code="deadline_exceeded",
                        message="distributed request deadline has already expired",
                        stage="admission",
                    )
                )
        request = DistributedRequest(
            operation=operation,
            pipeline_id=self.identity.pipeline_id,
            request_id=request_id,
            session_id=self._session_id,
            session_generation=self._generation,
            deployment_fingerprint=self.identity.deployment_fingerprint,
            inputs=inputs or {},
            prompt=prompt,
            deadline=deadline,
            target_request_id=target_request_id,
            observation_timestamp_ns=observation_timestamp_ns,
            stream_references=stream_references,
            aligned_timestamps_ns=aligned_timestamps_ns,
            aligned_tensors=aligned_tensors,
        )
        self._pending[request_id] = (operation, target_request_id)
        return request

    def accept_result(self, result: DistributedResult) -> SessionUpdate:
        with self._lock:
            expected = self._pending.get(result.request_id)
            expected_operation = expected[0] if expected is not None else None
            expected_target = expected[1] if expected is not None else ""
            if (
                result.pipeline_id != self.identity.pipeline_id
                or result.deployment_fingerprint != self.identity.deployment_fingerprint
                or result.session_id != self._session_id
                or result.session_generation != self._generation
                or expected_operation is None
                or (
                    result.operation is not expected_operation
                    and not (result.operation is Operation.UNKNOWN and not result.success)
                )
                or (expected_operation is Operation.CANCEL and result.target_request_id != expected_target)
            ):
                return SessionUpdate(
                    error=StructuredError(
                        code="stale_response",
                        message=f"discarded stale or mismatched response {result.request_id!r}",
                        stage="routing",
                        recoverable=True,
                    )
                )
            del self._pending[result.request_id]
            canceled = ""
            if result.operation is Operation.CANCEL and result.success:
                target = self._pending.get(result.target_request_id)
                if target is None or target[0] is not Operation.INFER:
                    return SessionUpdate(
                        error=StructuredError(
                            code="cancellation_unconfirmed",
                            message=f"cancellation target {result.target_request_id!r} is no longer pending",
                            stage="routing",
                            recoverable=True,
                        )
                    )
                del self._pending[result.target_request_id]
                canceled = result.target_request_id
            if not result.backend_ready:
                update = self._invalidate_locked(
                    StructuredError(
                        code="remote_backend_unavailable",
                        message=f"cloud backend left READY ({result.backend_state})",
                        stage="readiness",
                        recoverable=True,
                    )
                )
                return SessionUpdate(
                    invalidated_request_ids=update.invalidated_request_ids,
                    error=update.error,
                    canceled_request_id=canceled,
                )
            return SessionUpdate(canceled_request_id=canceled)

    def abandon_request(self, request_id: str) -> None:
        with self._lock:
            self._pending.pop(request_id, None)

    def close(self) -> tuple[str, ...]:
        with self._lock:
            if self._state.state is PipelineState.CLOSED:
                return ()
            if self._state.state is not PipelineState.CLOSING:
                self._state.transition(PipelineState.CLOSING)
            pending = tuple(sorted(self._pending))
            self._pending.clear()
            self._state.transition(PipelineState.CLOSED)
            return pending

    def fail(self, error: StructuredError) -> SessionUpdate:
        with self._lock:
            update = self._invalidate_locked(error)
            if self._state.state is not PipelineState.FAILED and self._state.can_transition(PipelineState.FAILED):
                self._state.transition(PipelineState.FAILED)
            return update

    def _invalidate_locked(self, error: StructuredError) -> SessionUpdate:
        invalidated = tuple(sorted(self._pending))
        self._pending.clear()
        if self._session_id:
            self._retired_sessions.add((self._session_id, self._generation))
        self._session_id = ""
        self._generation = 0
        self._last_cloud_sequence = 0
        self._reset_supported = False
        self._cancellation_supported = False
        self._last_error = error
        if self._state.state is PipelineState.READY:
            self._state.transition(PipelineState.DEGRADED)
        return SessionUpdate(invalidated_request_ids=invalidated, error=error)


class CloudSession:
    """Own cloud session generations and validate every routed operation."""

    def __init__(
        self,
        identity: PipelineIdentity,
        *,
        request_stream_validator: Callable[[tuple[StreamReference, ...]], None] | None = None,
        runtime_interface: str = "policy",
        runtime_model_type: str = "",
    ) -> None:
        if runtime_interface != "policy":
            raise UnsupportedDistributedRuntimeError(runtime_interface, runtime_model_type)
        self.identity = identity
        self._request_stream_validator = request_stream_validator
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._rollover_lock = threading.Lock()
        self._sequence = 0
        self._generation = 0
        self._session_id = ""
        self._last_edge_status: PipelineStatus | None = None
        self._edge_acknowledged = False
        self._last_edge_sequence = 0
        self._seen_request_ids: set[str] = set()
        self._active_operations = 0
        self._accepting_requests = False
        self._rollover_required = False
        self._rollover_epoch = 0
        self._last_error: StructuredError | None = None

    def observe_edge(
        self,
        status: PipelineStatus,
        *,
        backend_ready: bool,
        rollover_barrier: Callable[[], StructuredError | None] | None = None,
    ) -> StructuredError | None:
        with self._rollover_lock:
            return self._observe_edge(status, backend_ready=backend_ready, rollover_barrier=rollover_barrier)

    def _observe_edge(
        self,
        status: PipelineStatus,
        *,
        backend_ready: bool,
        rollover_barrier: Callable[[], StructuredError | None] | None,
    ) -> StructuredError | None:
        requires_rollover_barrier = False
        rollover_epoch = 0
        with self._condition:
            if status.role is not PeerRole.EDGE:
                return self._reject_locked("unexpected_peer_role", "cloud received a non-edge distributed status")
            mismatch = identity_error(self.identity, status.identity)
            if mismatch is not None:
                self._invalidate_locked(mismatch)
                return mismatch
            if status.runtime_state == PipelineState.FAILED.value:
                error = status.error or StructuredError(
                    code="remote_edge_failed",
                    message="edge pipeline entered FAILED state",
                    stage="readiness",
                )
                self._invalidate_locked(error)
                return error
            restarted = status.sequence <= self._last_edge_sequence
            self._last_edge_sequence = status.sequence
            self._last_edge_status = status
            if not backend_ready:
                error = StructuredError(
                    code="remote_backend_unavailable",
                    message="cloud backend is not ready",
                    stage="readiness",
                    recoverable=True,
                )
                self._invalidate_locked(error)
                return error

            session_mismatch = bool(status.session_id) and (
                status.session_id != self._session_id or status.session_generation != self._generation
            )
            dropped_acknowledgement = (
                bool(self._session_id)
                and not status.session_id
                and (self._edge_acknowledged or status.error is not None)
            )
            if not self._session_id or restarted or session_mismatch or dropped_acknowledgement:
                if self._session_id:
                    self._rollover_required = True
                requires_rollover_barrier = self._rollover_required
                self._accepting_requests = False
                self._condition.wait_for(lambda: self._active_operations == 0)
                if requires_rollover_barrier:
                    self._rollover_epoch += 1
                    rollover_epoch = self._rollover_epoch

        if requires_rollover_barrier and rollover_barrier is not None:
            error = rollover_barrier()
            if error is not None:
                with self._condition:
                    self._invalidate_locked(error)
                return error

        with self._condition:
            if requires_rollover_barrier and rollover_epoch != self._rollover_epoch:
                return self._last_error
            if not self._session_id or restarted or session_mismatch or dropped_acknowledgement:
                self._generation += 1
                self._session_id = uuid.uuid4().hex
                self._edge_acknowledged = False
                self._seen_request_ids.clear()
                self._accepting_requests = True
                self._rollover_required = False
            elif status.session_id == self._session_id and status.session_generation == self._generation:
                self._edge_acknowledged = True
            self._last_error = None
            return None

    def status(
        self,
        *,
        backend_ready: bool,
        backend_state: str,
        reset_supported: bool,
        cancellation_supported: bool,
    ) -> PipelineStatus:
        with self._lock:
            if not backend_ready and self._session_id:
                self._invalidate_locked(
                    StructuredError(
                        code="remote_backend_unavailable",
                        message=f"cloud backend is not ready ({backend_state})",
                        stage="readiness",
                        recoverable=True,
                    )
                )
            self._sequence += 1
            return PipelineStatus(
                role=PeerRole.CLOUD,
                identity=self.identity,
                sequence=self._sequence,
                session_id=self._session_id,
                session_generation=self._generation if self._session_id else 0,
                ready=backend_ready and bool(self._session_id) and self._accepting_requests,
                runtime_state=backend_state,
                reset_supported=reset_supported,
                cancellation_supported=cancellation_supported,
                error=self._last_error,
            )

    def validate_request(self, request: DistributedRequest) -> None:
        with self._lock:
            self._validate_request_locked(request)

    @contextmanager
    def operation(self, request: DistributedRequest):
        with self._condition:
            self._validate_request_locked(request)
            self._active_operations += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_operations -= 1
                self._condition.notify_all()

    def _validate_request_locked(self, request: DistributedRequest) -> None:
        checks = (
            (
                request.pipeline_id == self.identity.pipeline_id,
                "pipeline_not_found",
                "pipeline ID is not configured",
            ),
            (
                request.deployment_fingerprint == self.identity.deployment_fingerprint,
                "deployment_fingerprint_mismatch",
                "deployment fingerprint mismatch",
            ),
            (request.session_id == self._session_id, "session_mismatch", "session ID mismatch"),
            (
                request.session_generation == self._generation,
                "session_generation_mismatch",
                "session generation mismatch",
            ),
            (bool(self._session_id), "not_ready", "no live cloud handshake session"),
            (self._accepting_requests, "session_draining", "cloud session is draining active operations"),
        )
        for valid, code, message in checks:
            if not valid:
                raise DistributedProtocolError(
                    StructuredError(code=code, message=message, stage="routing", recoverable=True)
                )
        if self._request_stream_validator is not None and request.operation is Operation.INFER:
            try:
                self._request_stream_validator(request.stream_references)
            except Exception as exc:
                error = StructuredError(
                    code=str(getattr(exc, "code", "stream_negotiation_failed")),
                    message=str(exc) or type(exc).__name__,
                    stage="handshake",
                    recoverable=bool(getattr(exc, "recoverable", False)),
                    details=getattr(exc, "details", {}),
                )
                raise DistributedProtocolError(error) from exc
        if request.deadline is not None and datetime.now(timezone.utc) >= request.deadline.astimezone(timezone.utc):
            raise DistributedProtocolError(
                StructuredError(
                    code="deadline_exceeded",
                    message="distributed request deadline expired before cloud execution",
                    stage="admission",
                )
            )
        if request.request_id in self._seen_request_ids:
            raise DistributedProtocolError(
                StructuredError(
                    code="duplicate_request_id",
                    message=f"distributed request {request.request_id!r} was already admitted",
                    stage="routing",
                )
            )
        self._seen_request_ids.add(request.request_id)

    def invalidate(self, error: StructuredError) -> None:
        with self._lock:
            self._invalidate_locked(error)

    def _reject_locked(self, code: str, message: str) -> StructuredError:
        error = StructuredError(code=code, message=message, stage="handshake")
        self._invalidate_locked(error)
        return error

    def _invalidate_locked(self, error: StructuredError) -> None:
        if self._session_id:
            self._rollover_required = True
        self._rollover_epoch += 1
        self._accepting_requests = False
        self._session_id = ""
        self._edge_acknowledged = False
        self._seen_request_ids.clear()
        self._last_error = error
