"""Continuous scheduler.

Mirrors the existing IB-Robot dispatcher control loop exactly:
- request inference when ``plan_length <= watermark`` and no inference or
  policy reset is pending;
- ``TAKE_NEXT`` permits owner selection, including hold/empty;
- inference timestamp is ``None`` so the dispatcher keeps using
  ``get_clock().now()`` (current ROS time);
- no backpressure, no in-flight gate, no fault state, no timeout.

The continuous scheduler is stateless; ``set_observation_timestamp``/
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


def should_replenish_plan(
    plan_length: int,
    watermark: int,
    *,
    inference_in_progress: bool,
    policy_reset_in_progress: bool = False,
) -> bool:
    """Shared watermark replenishment rule for continuous dispatch loops.

    Single implementation of the ``plan_length <= watermark`` trigger used
    by ``ContinuousScheduler`` and the inline control loops of both
    dispatcher paths, so the legacy and scheduled paths cannot drift apart.
    """
    return plan_length <= watermark and not inference_in_progress and not policy_reset_in_progress


class ContinuousScheduler(DispatchScheduler):
    """Stateless scheduler that reproduces the existing IB-Robot control loop."""

    def __init__(self, watermark: int) -> None:
        self._watermark = watermark

    @property
    def scheduler_mode(self) -> str:
        return "continuous"

    def set_observation_timestamp(self, timestamp_ns: int) -> None:
        # Continuous always uses ROS current time; ignore environment timestamps.
        return

    def should_request_inference(self, snapshot: SchedulerSnapshot) -> bool:
        return should_replenish_plan(
            snapshot.plan_length,
            snapshot.watermark,
            inference_in_progress=snapshot.inference_in_progress,
            policy_reset_in_progress=snapshot.policy_reset_in_progress,
        )

    def observation_timestamp_for_inference(self) -> int | None:
        # None signals the dispatcher to use ``get_clock().now()`` unchanged.
        return None

    def choose_action(self, snapshot: SchedulerSnapshot) -> ActionDecision:
        return ActionDecision.TAKE_NEXT

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
