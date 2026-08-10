"""Step barrier scheduler (``wait_for_feedback`` mode).

Implements the completion-aware executor contract correctness-first backpressure rules:
- environment execution in-flight => no new inference;
- inference in-progress => no new action submission;
- completion before pop/hold/advance => only one submission, no logical pop,
  no counter advance, no hold;
- matching ``COMPLETED`` with a positive observation timestamp and matching
  optional episode/step identity => ``COMMIT`` exactly one action and update
  the latest timestamp used for the next inference;
- ``rejected``/``FAILED``/``timeout`` => ``FAIL_CLOSED`` (no pop, no retry);
- ``timeout`` => ``UNCERTAIN``; a late ``COMPLETED`` after ``UNCERTAIN`` is
  ``IGNORE`` and cannot recover;
- correlation mismatch / duplicate / stale completion => ``IGNORE``;
- only ``reset`` clears fault, in-flight and latest timestamp.

Timeouts use monotonic nanoseconds; they are never mixed with ROS or sim
observation time.
"""

from __future__ import annotations

from ..executors.completion import (
    CompletionStatus,
    ExecutionCompletion,
    ExecutionContext,
    ExecutionReceipt,
)
from .base import (
    ActionDecision,
    CompletionDecision,
    DispatchScheduler,
    SchedulerSnapshot,
    SchedulerTransition,
)


class StepBarrierScheduler(DispatchScheduler):
    """Single-in-flight, no-hold, fail-closed scheduler for feedback-gated execution."""

    def __init__(self, watermark: int, execution_timeout_sec: float) -> None:
        if not isinstance(execution_timeout_sec, int | float) or isinstance(execution_timeout_sec, bool):
            raise ValueError(
                f"execution_timeout_sec must be a positive number, got {type(execution_timeout_sec).__name__}"
            )
        if execution_timeout_sec <= 0:
            raise ValueError(f"execution_timeout_sec must be positive, got {execution_timeout_sec}")
        self._watermark = watermark
        self._execution_timeout_ns = int(float(execution_timeout_sec) * 1_000_000_000)
        self._inflight_context: ExecutionContext | None = None
        self._inflight_deadline_monotonic_ns: int = 0
        self._fault_status: str | None = None
        self._latest_observation_timestamp_ns: int | None = None

    @property
    def scheduler_mode(self) -> str:
        return "wait_for_feedback"

    def reset(self) -> None:
        self._inflight_context = None
        self._inflight_deadline_monotonic_ns = 0
        self._fault_status = None
        self._latest_observation_timestamp_ns = None

    def set_observation_timestamp(self, timestamp_ns: int) -> None:
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise ValueError(f"timestamp_ns must be a positive int (nanoseconds), got {type(timestamp_ns).__name__}")
        if timestamp_ns <= 0:
            raise ValueError(f"timestamp_ns must be positive, got {timestamp_ns}")
        self._latest_observation_timestamp_ns = timestamp_ns

    def should_request_inference(self, snapshot: SchedulerSnapshot) -> bool:
        if self._latest_observation_timestamp_ns is None:
            return False
        if self._fault_status is not None:
            return False
        if self._inflight_context is not None:
            return False
        if snapshot.inference_in_progress:
            return False
        if snapshot.policy_reset_in_progress:
            return False
        return snapshot.plan_length <= snapshot.watermark

    def observation_timestamp_for_inference(self) -> int | None:
        return self._latest_observation_timestamp_ns

    def choose_action(self, snapshot: SchedulerSnapshot) -> ActionDecision:
        if self._fault_status is not None:
            return ActionDecision.WAIT
        if self._inflight_context is not None:
            return ActionDecision.WAIT
        if snapshot.inference_in_progress:
            return ActionDecision.WAIT
        if snapshot.policy_reset_in_progress:
            return ActionDecision.WAIT
        if snapshot.plan_length > 0:
            return ActionDecision.TAKE_NEXT
        return ActionDecision.WAIT

    def on_submission(
        self,
        context: ExecutionContext,
        receipt: ExecutionReceipt,
        submitted_monotonic_ns: int,
    ) -> None:
        if self._fault_status is not None:
            # Faulted: leave fault in place, do not accept.
            return
        # Validate context/receipt correlation match.
        if context.correlation_id != receipt.correlation_id:
            self._fault_status = "submission_correlation_mismatch"
            self._inflight_context = None
            self._inflight_deadline_monotonic_ns = 0
            return
        if self._inflight_context is not None:
            # Second in-flight submission: fail-closed, do not overwrite the
            # first in-flight context.
            self._fault_status = "double_submission"
            self._inflight_context = None
            self._inflight_deadline_monotonic_ns = 0
            return
        if not receipt.accepted:
            # Rejected: definitive failure, fail-closed.
            self._fault_status = "rejected"
            return
        # Accepted: establish the unique in-flight context and deadline.
        self._inflight_context = context
        self._inflight_deadline_monotonic_ns = submitted_monotonic_ns + self._execution_timeout_ns

    def on_completion(self, completion: ExecutionCompletion) -> SchedulerTransition:
        # No in-flight: ignore all completions (covers stale/duplicate/late).
        if self._inflight_context is None:
            return SchedulerTransition(
                decision=CompletionDecision.IGNORE,
                message="no in-flight completion to match",
            )
        # Already faulted: ignore (cannot recover without reset).
        if self._fault_status is not None:
            return SchedulerTransition(
                decision=CompletionDecision.IGNORE,
                message=f"faulted: {self._fault_status}",
            )
        # Correlation mismatch: ignore.
        if completion.correlation_id != self._inflight_context.correlation_id:
            return SchedulerTransition(
                decision=CompletionDecision.IGNORE,
                correlation_id=completion.correlation_id,
                message="correlation mismatch",
            )

        # Matching correlation: dispatch on status.
        if completion.status is CompletionStatus.COMPLETED:
            return self._handle_completed(completion)
        if completion.status is CompletionStatus.FAILED:
            return self._fail_closed(completion, "failed", "matching failed")
        # UNCERTAIN completion: same as timeout, fail-closed.
        return self._fail_closed(completion, "uncertain", "matching uncertain")

    def _handle_completed(self, completion: ExecutionCompletion) -> SchedulerTransition:
        # Must carry a positive observation timestamp.
        if completion.observation_timestamp_ns is None or completion.observation_timestamp_ns <= 0:
            return self._fail_closed(
                completion, "missing_timestamp", "matching completed without observation timestamp"
            )
        # Episode identity check (if the context carried an episode id).
        ctx = self._inflight_context
        if ctx.episode_id is not None and (completion.episode_id is None or completion.episode_id != ctx.episode_id):
            return self._fail_closed(completion, "episode_id_mismatch", "episode_id mismatch")
        # Step identity check (if the context carried an expected step id).
        if ctx.expected_step_id is not None and (
            completion.step_id is None or completion.step_id != ctx.expected_step_id
        ):
            return self._fail_closed(completion, "step_id_mismatch", "step_id mismatch")
        # All checks passed: commit and update latest timestamp.
        self._latest_observation_timestamp_ns = completion.observation_timestamp_ns
        self._inflight_context = None
        self._inflight_deadline_monotonic_ns = 0
        return SchedulerTransition(
            decision=CompletionDecision.COMMIT,
            correlation_id=completion.correlation_id,
            message="matching completed",
        )

    def _fail_closed(self, completion: ExecutionCompletion, fault: str, reason: str) -> SchedulerTransition:
        self._fault_status = fault
        self._inflight_context = None
        self._inflight_deadline_monotonic_ns = 0
        return SchedulerTransition(
            decision=CompletionDecision.FAIL_CLOSED,
            correlation_id=completion.correlation_id,
            fault_status=fault,
            message=reason,
        )

    def on_tick(self, now_monotonic_ns: int) -> SchedulerTransition:
        if self._inflight_context is None:
            return SchedulerTransition(
                decision=CompletionDecision.IGNORE,
                message="no in-flight to time out",
            )
        if self._fault_status is not None:
            return SchedulerTransition(
                decision=CompletionDecision.IGNORE,
                message=f"faulted: {self._fault_status}",
            )
        if self._inflight_deadline_monotonic_ns == 0:
            return SchedulerTransition(
                decision=CompletionDecision.IGNORE,
                message="no deadline set",
            )
        if now_monotonic_ns < self._inflight_deadline_monotonic_ns:
            return SchedulerTransition(
                decision=CompletionDecision.IGNORE,
                message="in-flight, not timed out",
            )
        # Timeout: UNCERTAIN, fail-closed, no retry, no pop.
        self._fault_status = "timeout_uncertain"
        self._inflight_context = None
        self._inflight_deadline_monotonic_ns = 0
        return SchedulerTransition(
            decision=CompletionDecision.FAIL_CLOSED,
            fault_status=self._fault_status,
            message="execution timeout, uncertain",
        )

    def mark_fault(self, reason: str) -> None:
        """Force fail-closed. Used by the dispatcher for conditions the
        scheduler cannot detect from the completion alone (e.g. plan
        generation mismatch). Only ``reset`` can clear this.
        """
        self._fault_status = reason
        self._inflight_context = None
        self._inflight_deadline_monotonic_ns = 0

    @property
    def inflight_correlation_id(self) -> str | None:
        if self._inflight_context is None:
            return None
        return self._inflight_context.correlation_id

    @property
    def fault_status(self) -> str | None:
        return self._fault_status

    @property
    def can_accept_next(self) -> bool:
        return self._fault_status is None and self._inflight_context is None
