"""Pure provider-agnostic RunPolicy terminal result classifier."""

from __future__ import annotations

from dataclasses import dataclass

_COMPLETED = frozenset({"task_success", "terminated", "truncated", "max_actions", "max_duration"})
_INFRASTRUCTURE = frozenset(
    {
        "inference_failed",
        "inference_timeout",
        "execution_rejected",
        "execution_failed",
        "execution_uncertain",
        "identity_mismatch",
        "observation_startup_timeout",
        "external_reset",
    }
)


@dataclass(frozen=True, slots=True)
class RunPolicyResultData:
    protocol_success: bool
    message: str
    has_success: bool
    success: bool
    termination_reason: str
    episode_id: int
    has_final_step: bool
    final_step_id: int
    published_actions: int
    has_reward: bool
    reward: float
    terminated: bool
    truncated: bool
    standard_metrics_json: str
    native_metrics_json: str
    info_json: str
    round_trip_latency_ms: float


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    completed: bool
    partial: bool
    canceled: bool
    infrastructure_error: bool
    error_category: str
    termination_reason: str
    has_success: bool
    success: bool
    result: RunPolicyResultData


def classify_run_policy(action_status: str, result: RunPolicyResultData) -> EpisodeOutcome:
    reason = result.termination_reason
    if action_status == "canceled" or reason == "canceled":
        return EpisodeOutcome(False, True, True, False, "", "canceled", result.has_success, result.success, result)
    if action_status == "succeeded" and result.protocol_success and reason in _COMPLETED:
        return EpisodeOutcome(True, False, False, False, "", reason, result.has_success, result.success, result)
    if action_status == "aborted" or not result.protocol_success or reason in _INFRASTRUCTURE:
        return EpisodeOutcome(
            False,
            True,
            False,
            True,
            "infrastructure",
            reason or "transport_fault",
            result.has_success,
            result.success,
            result,
        )
    return EpisodeOutcome(
        False,
        True,
        False,
        True,
        "infrastructure",
        reason or "unknown_terminal",
        result.has_success,
        result.success,
        result,
    )
