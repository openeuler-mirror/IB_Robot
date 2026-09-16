"""Persistent ROS-free downstream operation ownership for scheduled inference."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class OperationKind(str, Enum):
    OPEN = "open"
    DISPATCH = "dispatch"
    CLOSE = "close"


class OperationState(str, Enum):
    CREATED = "created"
    SEND_PENDING = "send_pending"
    ACCEPTED = "accepted"
    RESULT_PENDING = "result_pending"
    TERMINAL = "terminal"
    QUARANTINED = "quarantined"


class Certainty(str, Enum):
    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperationIdentity:
    session_id: str
    logical_generation: int
    binding_id: str
    binding_incarnation: int
    expected_boot_id: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "expected_boot_id"):
            try:
                parsed = UUID(getattr(self, name))
            except ValueError as exc:
                raise ValueError(f"{name} must be a UUID4") from exc
            if parsed.version != 4:
                raise ValueError(f"{name} must be a UUID4")
        if self.logical_generation < 1 or self.binding_incarnation < 1:
            raise ValueError("generation and incarnation must be positive")


@dataclass
class DownstreamOperationContext:
    operation_id: str
    kind: OperationKind
    idempotency_key: tuple[object, ...]
    identity: OperationIdentity
    deadline_mono_ns: int
    state: OperationState = OperationState.CREATED
    certainty: Certainty | None = None
    result: Any = None
    error: str = ""
    binding_drained: bool = field(default=False, repr=False)
    _waiters: set[str] = field(default_factory=set, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def create(
        cls,
        *,
        kind: OperationKind,
        idempotency_key: tuple[object, ...],
        identity: OperationIdentity,
        deadline_mono_ns: int,
    ) -> DownstreamOperationContext:
        if deadline_mono_ns <= 0:
            raise ValueError("deadline_mono_ns must be positive")
        return cls(str(uuid4()), kind, idempotency_key, identity, deadline_mono_ns)

    @property
    def terminal(self) -> bool:
        return self.state in (OperationState.TERMINAL, OperationState.QUARANTINED)

    @property
    def waiter_count(self) -> int:
        with self._lock:
            return len(self._waiters)

    def attach_waiter(self, waiter_id: str, *, max_waiters: int) -> None:
        if not waiter_id:
            raise ValueError("waiter_id must be non-empty")
        with self._lock:
            if waiter_id in self._waiters:
                return
            if len(self._waiters) >= max_waiters:
                raise RuntimeError("operation_waiter_capacity")
            self._waiters.add(waiter_id)

    def detach_waiter(self, waiter_id: str) -> None:
        with self._lock:
            self._waiters.discard(waiter_id)

    def claim_send(self) -> bool:
        with self._lock:
            if self.state is not OperationState.CREATED:
                return False
            self.state = OperationState.SEND_PENDING
            return True

    def transition(self, state: OperationState) -> None:
        with self._lock:
            if self.terminal:
                return
            self.state = state

    def finish(self, *, certainty: Certainty, result: Any = None, error: str = "") -> None:
        with self._lock:
            if self.terminal:
                return
            if certainty is Certainty.NOT_STARTED and self.state in (
                OperationState.ACCEPTED,
                OperationState.RESULT_PENDING,
            ):
                raise RuntimeError("accepted operation cannot become NOT_STARTED")
            self.certainty, self.result, self.error = certainty, result, error
            self.state = OperationState.QUARANTINED if certainty is Certainty.UNKNOWN else OperationState.TERMINAL
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)


class OperationRegistry:
    """Bounded owner of live, detached, retained and quarantined operations."""

    def __init__(self, *, max_records: int, max_waiters_per_operation: int) -> None:
        if max_records < 1 or max_waiters_per_operation < 1:
            raise ValueError("operation registry bounds must be positive")
        self._max_records = max_records
        self._max_waiters = max_waiters_per_operation
        self._lock = threading.RLock()
        self._by_id: dict[str, DownstreamOperationContext] = {}
        self._by_key: dict[tuple[object, ...], str] = {}

    def create_or_get(
        self,
        *,
        kind: OperationKind,
        idempotency_key: tuple[object, ...],
        identity: OperationIdentity,
        deadline_mono_ns: int,
        waiter_id: str,
    ) -> tuple[DownstreamOperationContext, bool]:
        with self._lock:
            existing_id = self._by_key.get(idempotency_key)
            if existing_id is not None:
                operation = self._by_id[existing_id]
                if operation.identity != identity or operation.kind is not kind:
                    raise RuntimeError("operation_identity_conflict")
                operation.attach_waiter(waiter_id, max_waiters=self._max_waiters)
                return operation, False
            if len(self._by_id) >= self._max_records:
                raise RuntimeError("operation_record_capacity")
            operation = DownstreamOperationContext.create(
                kind=kind,
                idempotency_key=idempotency_key,
                identity=identity,
                deadline_mono_ns=deadline_mono_ns,
            )
            operation.attach_waiter(waiter_id, max_waiters=self._max_waiters)
            self._by_id[operation.operation_id] = operation
            self._by_key[idempotency_key] = operation.operation_id
            return operation, True

    def get(self, operation_id: str) -> DownstreamOperationContext | None:
        with self._lock:
            return self._by_id.get(operation_id)

    def find(self, idempotency_key: tuple[object, ...]) -> DownstreamOperationContext | None:
        with self._lock:
            operation_id = self._by_key.get(idempotency_key)
            return self._by_id.get(operation_id) if operation_id is not None else None

    def fence_operation(self, operation_id: str) -> bool:
        """Remove one quarantined UNKNOWN record with no remaining waiters.

        Used when a retry declines to replay a cached unknown outcome: the
        idempotency key becomes reusable so the next attempt actually
        re-sends. In-flight records or records with live waiters stay.
        """

        with self._lock:
            operation = self._by_id.get(operation_id)
            if (
                operation is None
                or operation.state is not OperationState.QUARANTINED
                or operation.certainty is not Certainty.UNKNOWN
                or operation.waiter_count > 0
            ):
                return False
            self._by_id.pop(operation_id, None)
            self._by_key.pop(operation.idempotency_key, None)
            return True

    def detach_waiter(self, operation_id: str, waiter_id: str) -> None:
        operation = self.get(operation_id)
        if operation is not None:
            operation.detach_waiter(waiter_id)
            self.remove_terminal(operation_id)

    def bind_future(self, operation_id: str, future: Future) -> None:
        """Bind a submitted runtime future to its persistent operation record.

        The callback is deliberately owned by the registry rather than by a
        request stack frame: a ROS waiter may detach while ACL execution is
        still in flight. Runtime failures with an unknown outcome remain in
        quarantine and are never made retryable.
        """

        if not isinstance(future, Future):
            raise TypeError("operation future must be a concurrent.futures.Future")
        operation = self.get(operation_id)
        if operation is None:
            raise KeyError(f"unknown operation {operation_id!r}")
        operation.transition(OperationState.ACCEPTED)
        operation.transition(OperationState.RESULT_PENDING)

        def complete(done: Future) -> None:
            try:
                result = done.result()
            except Exception as exc:  # noqa: BLE001
                certainty = Certainty.UNKNOWN if not bool(getattr(exc, "outcome_known", True)) else Certainty.COMPLETED
                self.finish(operation_id, certainty=certainty, error=str(exc))
            else:
                self.finish(operation_id, certainty=Certainty.COMPLETED, result=result)

        future.add_done_callback(complete)

    def finish(
        self,
        operation_id: str,
        *,
        certainty: Certainty,
        result: Any = None,
        error: str = "",
    ) -> bool:
        """Finish an operation and reclaim a known terminal with no waiters.

        Late callbacks use this path after an upstream waiter has timed out.
        UNKNOWN remains quarantined until a validated binding drain or boot fence.
        """

        operation = self.get(operation_id)
        if operation is None:
            return False
        operation.finish(certainty=certainty, result=result, error=error)
        return self.remove_terminal(operation_id)

    def remove_terminal(self, operation_id: str) -> bool:
        with self._lock:
            operation = self._by_id.get(operation_id)
            if (
                operation is None
                or (
                    not operation.binding_drained
                    and (not operation.terminal or operation.certainty is Certainty.UNKNOWN)
                )
                or operation.waiter_count > 0
            ):
                return False
            self._by_id.pop(operation_id)
            self._by_key.pop(operation.idempotency_key, None)
            return True

    def fence_binding(
        self, *, session_id: str, binding_id: str, binding_incarnation: int, expected_boot_id: str
    ) -> int:
        """Reclaim ownership covered by a validated successful binding Close.

        Keep live waiters attached to their original result. Mark their records
        so detachment also reclaims UNKNOWN or missing late results after drain.
        """
        if not session_id or not binding_id or binding_incarnation < 1 or not expected_boot_id:
            return 0
        with self._lock:
            removed = 0
            for operation in tuple(self._by_id.values()):
                identity = operation.identity
                if (
                    identity.session_id,
                    identity.binding_id,
                    identity.binding_incarnation,
                    identity.expected_boot_id,
                ) != (session_id, binding_id, binding_incarnation, expected_boot_id):
                    continue
                operation.binding_drained = True
                removed += self.remove_terminal(operation.operation_id)
            return removed

    def fence_boot(self, previous_boot_id: str) -> int:
        """Remove records whose old Pipeline boot can no longer complete."""

        with self._lock:
            operation_ids = [
                operation_id
                for operation_id, operation in self._by_id.items()
                if operation.identity.expected_boot_id == previous_boot_id
            ]
            for operation_id in operation_ids:
                operation = self._by_id.pop(operation_id)
                self._by_key.pop(operation.idempotency_key, None)
            return len(operation_ids)

    def fence_uncertain(self, previous_boot_id: str) -> int:
        """Clear quarantined UNKNOWN records once device execution is fenced.

        Only no-waiter QUARANTINED records are removed: records with live
        waiters or still-bound futures may still resolve and stay behind.
        """

        with self._lock:
            operation_ids = [
                operation_id
                for operation_id, operation in self._by_id.items()
                if operation.identity.expected_boot_id == previous_boot_id
                and operation.state is OperationState.QUARANTINED
                and operation.waiter_count == 0
            ]
            for operation_id in operation_ids:
                operation = self._by_id.pop(operation_id)
                self._by_key.pop(operation.idempotency_key, None)
            return len(operation_ids)

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)
