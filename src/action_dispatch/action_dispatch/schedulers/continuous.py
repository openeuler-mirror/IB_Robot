"""Continuous scheduler.

Mirrors the existing IB-Robot dispatcher control loop exactly:
- request inference when ``plan_length <= watermark`` and no inference or
  policy reset is pending;
- ``TAKE_NEXT`` when the plan is non-empty;
- ``HOLD_LAST`` when the plan is empty but a last action exists;
- ``WAIT`` otherwise;
- inference timestamp is ``None`` so the dispatcher keeps using
  ``get_clock().now()`` (current ROS time);
- no backpressure, no in-flight gate, no fault state, no timeout.

The continuous scheduler is stateless; ``reset``/``set_observation_timestamp``/
``on_submission``/``on_completion``/``on_tick`` are all no-ops so the
continuous path is provably zero-regression.
"""

from __future__ import annotations

from .base import (
    ActionDecision,
    CompletionDecision,
    DispatchScheduler,
    SchedulerSnapshot,
    SchedulerTransition,
)


class ContinuousScheduler(DispatchScheduler):
    """Stateless scheduler that reproduces the existing IB-Robot control loop."""

    def __init__(self, watermark: int) -> None:
        self._watermark = watermark

    @property
    def scheduler_mode(self) -> str:
        return "continuous"

    def reset(self) -> None:
        # Continuous is stateless; nothing to clear.
        return

    def set_observation_timestamp(self, timestamp_ns: int) -> None:
        # Continuous always uses ROS current time; ignore environment timestamps.
        return

    def should_request_inference(self, snapshot: SchedulerSnapshot) -> bool:
        return (
            snapshot.plan_length <= snapshot.watermark
            and not snapshot.inference_in_progress
            and not snapshot.policy_reset_in_progress
        )

    def observation_timestamp_for_inference(self) -> int | None:
        # None signals the dispatcher to use ``get_clock().now()`` unchanged.
        return None

    def choose_action(self, snapshot: SchedulerSnapshot) -> ActionDecision:
        if snapshot.plan_length > 0:
            return ActionDecision.TAKE_NEXT
        if snapshot.has_last_action:
            return ActionDecision.HOLD_LAST
        return ActionDecision.WAIT

    def on_submission(self, context, receipt, submitted_monotonic_ns: int) -> None:
        # Continuous does not gate on submissions.
        return

    def on_completion(self, completion) -> SchedulerTransition:
        # Continuous does not gate on completions; topic immediate completion
        # is processed exactly once via the receipt and ignored here.
        return SchedulerTransition(decision=CompletionDecision.IGNORE, message="continuous ignores completions")

    def on_tick(self, now_monotonic_ns: int) -> SchedulerTransition:
        # Continuous has no execution timeout.
        return SchedulerTransition(decision=CompletionDecision.IGNORE, message="continuous has no timeout")

    def mark_fault(self, reason: str) -> None:
        # Continuous is stateless and never faults; the dispatcher only calls
        # mark_fault on the wait_for_feedback path. No-op keeps continuous
        # fault_status as documented (always None).
        return

    @property
    def inflight_correlation_id(self) -> str | None:
        return None

    @property
    def fault_status(self) -> str | None:
        return None

    @property
    def can_accept_next(self) -> bool:
        return True
