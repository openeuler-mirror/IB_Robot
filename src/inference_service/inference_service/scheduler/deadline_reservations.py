"""Atomic priority-zero deadline reservations by hardware resource."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DeadlineReservation:
    token: int
    pipeline_id: str
    hardware_resource_id: str
    deadline_ns: int
    session_id: str | None = None
    binding_id: str | None = None
    binding_incarnation: int | None = None
    expected_boot_id: str | None = None
    estimated_start_ns: int | None = None
    estimated_finish_ns: int | None = None

    @property
    def estimate_ns(self) -> int | None:
        if self.estimated_start_ns is None or self.estimated_finish_ns is None:
            return None
        return self.estimated_finish_ns - self.estimated_start_ns


class DeadlineReservationTable:
    """Serialize priority-zero dispatches per hardware resource.

    FIFO preserves admission order. EDF orders work that has not started by
    absolute monotonic deadline, using the reservation token as a stable FIFO
    tie-breaker. Once a reservation owns the resource it is non-preemptible;
    newly arriving work cannot bypass it.

    An execution estimate is optional. FIFO deadline admission supplies one
    from an offline profile and therefore rejects reservations whose estimated
    finish misses the deadline. EDF is online best-effort scheduling: it orders
    and expires work but does not claim profile-based finish feasibility.
    """

    def __init__(self, *, policy: Literal["fifo", "edf"] = "fifo") -> None:
        if policy not in {"fifo", "edf"}:
            raise ValueError("policy must be 'fifo' or 'edf'")
        self._policy = policy
        self._condition = threading.Condition(threading.Lock())
        self._next_token = 1
        self._reservations: dict[int, DeadlineReservation] = {}
        self._resource_tokens: dict[str, list[int]] = {}
        self._active_tokens: dict[str, int] = {}
        self._unknown_tokens: set[int] = set()

    def try_reserve(
        self,
        *,
        pipeline_id: str,
        hardware_resource_id: str,
        now_ns: int,
        deadline_ns: int,
        estimate_ns: int | None = None,
        session_id: str | None = None,
        binding_id: str | None = None,
        binding_incarnation: int | None = None,
        expected_boot_id: str | None = None,
    ) -> DeadlineReservation | None:
        if not hardware_resource_id:
            raise ValueError("hardware_resource_id must be non-empty")
        if estimate_ns is not None and estimate_ns <= 0:
            raise ValueError("estimate_ns must be positive")
        with self._condition:
            if deadline_ns <= now_ns:
                return None
            tokens = self._resource_tokens.get(hardware_resource_id, [])
            if any(token in self._unknown_tokens for token in tokens):
                return None
            start_ns: int | None = None
            finish_ns: int | None = None
            if estimate_ns is not None:
                if self._policy != "fifo":
                    raise ValueError("EDF does not accept profile-based finish estimates")
                estimated_finishes = [
                    self._reservations[token].estimated_finish_ns
                    for token in tokens
                    if self._reservations[token].estimated_finish_ns is not None
                ]
                tail_ns = max(estimated_finishes, default=now_ns)
                start_ns = max(now_ns, tail_ns)
                finish_ns = start_ns + estimate_ns
                if finish_ns > deadline_ns:
                    return None
            token = self._next_token
            self._next_token += 1
            reservation = DeadlineReservation(
                token=token,
                pipeline_id=pipeline_id,
                hardware_resource_id=hardware_resource_id,
                deadline_ns=deadline_ns,
                session_id=session_id,
                binding_id=binding_id,
                binding_incarnation=binding_incarnation,
                expected_boot_id=expected_boot_id,
                estimated_start_ns=start_ns,
                estimated_finish_ns=finish_ns,
            )
            self._reservations[token] = reservation
            resource_tokens = self._resource_tokens.setdefault(hardware_resource_id, [])
            resource_tokens.append(token)
            self._sort_waiting_locked(hardware_resource_id)
            self._condition.notify_all()
            return reservation

    def wait_for_turn(
        self,
        reservation: DeadlineReservation,
        *,
        deadline_ns: int,
        cancel_requested: Callable[[], bool] | None = None,
        now_ns: Callable[[], int] = time.monotonic_ns,
    ) -> str:
        """Wait until this reservation owns the resource dispatch turn.

        Returns ``ready``, ``deadline_exceeded``, ``request_canceled``, or
        ``reservation_released``. Deadline feasibility is checked again against
        the actual monotonic time when the turn becomes available.
        """

        with self._condition:
            while True:
                current = self._reservations.get(reservation.token)
                if current is None:
                    return "reservation_released"
                if cancel_requested is not None and cancel_requested():
                    self._remove_locked(current)
                    self._condition.notify_all()
                    return "request_canceled"
                current_time_ns = now_ns()
                effective_deadline_ns = min(deadline_ns, current.deadline_ns)
                if current_time_ns >= effective_deadline_ns:
                    self._remove_locked(current)
                    self._condition.notify_all()
                    return "deadline_exceeded"
                estimate_ns = current.estimate_ns
                if estimate_ns is not None and current_time_ns + estimate_ns > effective_deadline_ns:
                    self._remove_locked(current)
                    self._condition.notify_all()
                    return "deadline_exceeded"
                self._expire_waiting_locked(current.hardware_resource_id, current_time_ns)
                tokens = self._resource_tokens[current.hardware_resource_id]
                active_token = self._active_tokens.get(current.hardware_resource_id)
                if active_token == current.token:
                    return "ready"
                if active_token is None and tokens and tokens[0] == current.token:
                    # Claiming the dispatch turn is the non-preemption boundary.
                    # A reservation that has not reached this point remains
                    # eligible for EDF reordering by an earlier deadline.
                    self._active_tokens[current.hardware_resource_id] = current.token
                    return "ready"
                remaining_ns = effective_deadline_ns - current_time_ns
                if remaining_ns <= 0:
                    self._remove_locked(current)
                    self._condition.notify_all()
                    return "deadline_exceeded"
                self._condition.wait(min(0.05, remaining_ns / 1_000_000_000))

    def release(self, reservation: DeadlineReservation) -> None:
        """Release work known not to have started or known to have completed."""

        with self._condition:
            current = self._reservations.pop(reservation.token, None)
            self._unknown_tokens.discard(reservation.token)
            if current is None:
                return
            self._remove_resource_token_locked(current)
            self._condition.notify_all()

    def mark_unknown(self, reservation: DeadlineReservation) -> None:
        """Retain uncertain work so the resource fails closed until reconciliation."""

        with self._condition:
            if reservation.token in self._reservations:
                self._unknown_tokens.add(reservation.token)
                self._condition.notify_all()

    def reconcile_binding(
        self,
        *,
        pipeline_id: str,
        session_id: str,
        binding_id: str,
        binding_incarnation: int,
        expected_boot_id: str,
    ) -> None:
        """Release reservations covered by a validated Close drain.

        A late Open cleanup can finish before its caller marks UNKNOWN.
        Removing the owned reservation also makes that later mark a no-op.
        """

        if not session_id or not binding_id or not expected_boot_id or binding_incarnation < 1:
            return
        with self._condition:
            for reservation in tuple(self._reservations.values()):
                if (
                    reservation.pipeline_id == pipeline_id
                    and reservation.session_id == session_id
                    and reservation.binding_id == binding_id
                    and reservation.binding_incarnation == binding_incarnation
                    and reservation.expected_boot_id == expected_boot_id
                ):
                    self._remove_locked(reservation)
            self._condition.notify_all()

    def reconcile_pipeline(self, pipeline_id: str) -> None:
        """Clear uncertain work fenced by a pipeline reboot."""

        with self._condition:
            tokens = [token for token in self._unknown_tokens if self._reservations[token].pipeline_id == pipeline_id]
            for token in tokens:
                reservation = self._reservations.pop(token)
                self._unknown_tokens.remove(token)
                self._remove_resource_token_locked(reservation)
            self._condition.notify_all()

    def _remove_locked(self, reservation: DeadlineReservation) -> None:
        self._reservations.pop(reservation.token, None)
        self._unknown_tokens.discard(reservation.token)
        self._remove_resource_token_locked(reservation)

    def _remove_resource_token_locked(self, reservation: DeadlineReservation) -> None:
        if self._active_tokens.get(reservation.hardware_resource_id) == reservation.token:
            self._active_tokens.pop(reservation.hardware_resource_id, None)
        resource_tokens = self._resource_tokens.get(reservation.hardware_resource_id)
        if resource_tokens is None:
            return
        if reservation.token in resource_tokens:
            resource_tokens.remove(reservation.token)
        if not resource_tokens:
            self._resource_tokens.pop(reservation.hardware_resource_id, None)

    def _sort_waiting_locked(self, hardware_resource_id: str) -> None:
        if self._policy != "edf":
            return
        tokens = self._resource_tokens[hardware_resource_id]
        active_token = self._active_tokens.get(hardware_resource_id)
        waiting = [token for token in tokens if token != active_token]
        waiting.sort(key=lambda token: (self._reservations[token].deadline_ns, token))
        self._resource_tokens[hardware_resource_id] = [active_token, *waiting] if active_token is not None else waiting

    def _expire_waiting_locked(self, hardware_resource_id: str, now_ns: int) -> None:
        active_token = self._active_tokens.get(hardware_resource_id)
        expired = [
            self._reservations[token]
            for token in self._resource_tokens.get(hardware_resource_id, ())
            if token != active_token and self._reservations[token].deadline_ns <= now_ns
        ]
        for reservation in expired:
            self._remove_locked(reservation)
        if expired:
            self._condition.notify_all()


__all__ = ["DeadlineReservation", "DeadlineReservationTable"]
