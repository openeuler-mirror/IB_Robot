"""Minimal scheduler decision model and ABC.

This module defines the pure-Python types the dispatcher uses to query the
scheduler and the ``DispatchScheduler`` ABC. It imports only the standard
library and ``executors.completion`` so the scheduler contract can be loaded
and unit-tested without rclpy, NumPy, Torch, benchmark runtime or sim backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ..executors.completion import ExecutionCompletion, ExecutionContext, ExecutionReceipt


class ActionDecision(str, Enum):
    """What the dispatcher should do for the next action this tick."""

    WAIT = "wait"
    TAKE_NEXT = "take_next"
    HOLD_LAST = "hold_last"


class CompletionDecision(str, Enum):
    """What the dispatcher should do with a just-drained completion."""

    IGNORE = "ignore"
    COMMIT = "commit"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    """Read-only snapshot of dispatcher state the scheduler is allowed to see.

    The dispatcher builds this each control tick; the scheduler never owns the
    underlying queue, smoother or inference state.
    """

    plan_length: int
    watermark: int
    inference_in_progress: bool
    policy_reset_in_progress: bool
    has_last_action: bool


@dataclass(frozen=True, slots=True)
class SchedulerTransition:
    """Result of a scheduler state transition (completion or timeout)."""

    decision: CompletionDecision
    correlation_id: str | None = None
    fault_status: str | None = None
    message: str = ""


class DispatchScheduler(ABC):
    """Permission/state-transition contract for the action dispatcher.

    The scheduler answers "when may I request inference?" and "when may I
    submit the next action?" plus "what should I do with this completion?".
    It must not read action values, modify the queue/smoother, select models,
    interpret suites/tasks or generate reports.
    """

    @property
    @abstractmethod
    def scheduler_mode(self) -> str:
        """Return the exact SSOT scheduler mode string."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all in-flight context, faults and cached timestamps."""

    @abstractmethod
    def set_observation_timestamp(self, timestamp_ns: int) -> None:
        """Record the latest environment observation timestamp (nanoseconds)."""

    @abstractmethod
    def should_request_inference(self, snapshot: SchedulerSnapshot) -> bool:
        """Return whether the dispatcher may request inference this tick."""

    @abstractmethod
    def observation_timestamp_for_inference(self) -> int | None:
        """Return the timestamp to use for the next inference request, or None."""

    @abstractmethod
    def choose_action(self, snapshot: SchedulerSnapshot) -> ActionDecision:
        """Return what the dispatcher should do for the next action this tick."""

    @abstractmethod
    def on_submission(
        self,
        context: ExecutionContext,
        receipt: ExecutionReceipt,
        submitted_monotonic_ns: int,
    ) -> None:
        """Record that an action was submitted (accepted or rejected)."""

    @abstractmethod
    def on_completion(self, completion: ExecutionCompletion) -> SchedulerTransition:
        """Process a drained completion and return the transition decision."""

    @abstractmethod
    def on_tick(self, now_monotonic_ns: int) -> SchedulerTransition:
        """Check for execution timeout and return any transition."""

    @abstractmethod
    def mark_fault(self, reason: str) -> None:
        """Force the scheduler into fail-closed state.

        Used by the dispatcher when it detects a condition the scheduler could
        not detect from the completion alone (e.g. plan generation mismatch).
        Only ``reset`` can clear the fault afterwards.
        """

    @property
    @abstractmethod
    def inflight_correlation_id(self) -> str | None:
        """Return the in-flight action correlation id, or None."""

    @property
    @abstractmethod
    def fault_status(self) -> str | None:
        """Return the current fault status, or None if healthy."""

    @property
    @abstractmethod
    def can_accept_next(self) -> bool:
        """Return whether the scheduler can accept a new submission right now."""
