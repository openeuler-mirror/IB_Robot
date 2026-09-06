"""Canonical action-dispatch strategy vocabulary and combination rules.

Single source of truth for dispatch strategy names, compatibility defaults
and legal executor/scheduler/chunking/blending combinations. The robot_config
launch builders validate against this module at launch-plan construction
time, and the action_dispatch nodes import the same validators defensively
at init so the two layers cannot silently diverge.

Strategy layers (see the action_dispatch package documentation):

- ``executor``  — where actions go (``topic`` / ``benchmark``).
- ``scheduler`` — when inference may be requested and actions submitted
  (``continuous`` / ``wait_for_feedback``).
- ``chunking``  — how an inference chunk enters the executable plan
  (``full_chunk`` / ``auto_horizon``; ``auto_horizon`` requires the
  ``topic`` executor and is rejected for benchmark episodes).
- ``blending``  — how candidates become one action per tick
  (``none`` / ``temporal_ensemble``).

This module is intentionally stdlib-only so both packages can import it
without ROS or numpy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_SCHEDULERS = ("continuous", "wait_for_feedback")
SUPPORTED_CHUNKING_STRATEGIES = ("full_chunk", "auto_horizon")
SUPPORTED_BLENDING_STRATEGIES = ("none", "temporal_ensemble")

DEFAULT_SCHEDULER = "continuous"
DEFAULT_CHUNKING = "full_chunk"


class DispatchStrategyError(ValueError):
    """Raised when a dispatch strategy selection is invalid or contradictory."""


@dataclass(frozen=True)
class DispatchStrategySelection:
    executor_type: str
    scheduler_mode: str
    chunking: str
    blending: str


def _resolve_name(value: object, default: str, supported: tuple[str, ...], label: str) -> str:
    if value is None or (isinstance(value, str) and value == ""):
        return default
    if not isinstance(value, str) or value not in supported:
        raise DispatchStrategyError(f"unknown {label} {value!r}; expected one of {supported}")
    return value


def reject_legacy_smoothing_config(executor_config: dict) -> None:
    """Reject the removed executor flag even when its value agrees with blending."""
    if "temporal_smoothing_enabled" in executor_config:
        raise DispatchStrategyError(
            "executor.temporal_smoothing_enabled has been removed; use dispatch.blending: none or temporal_ensemble"
        )


def resolve_dispatch_strategies(
    *,
    executor_type: object = None,
    scheduler_mode: object = None,
    chunking: object = None,
    blending: object = None,
    entrypoint: str = "legacy",
) -> DispatchStrategySelection:
    """Resolve raw startup values, or validate a proposed runtime selection.

    Blending is the sole fusion configuration; validate it before changing any store.
    The historical action executor alias belongs only to the launch boundary.
    """
    if entrypoint not in ("legacy", "scheduled"):
        raise DispatchStrategyError(f"unknown dispatch entrypoint {entrypoint!r}")
    executor = _resolve_name(executor_type, "topic", ("topic", "benchmark"), "executor type")
    scheduler = _resolve_name(scheduler_mode, DEFAULT_SCHEDULER, SUPPORTED_SCHEDULERS, "scheduler mode")
    if entrypoint == "scheduled" and (executor != "topic" or scheduler != "continuous"):
        raise DispatchStrategyError(
            f"scheduled entrypoint requires executor type 'topic' and scheduler_mode 'continuous'; "
            f"got executor_type={executor!r}, scheduler_mode={scheduler!r}"
        )
    validate_executor_scheduler_pairing(executor, scheduler)
    chunking = _resolve_name(chunking, DEFAULT_CHUNKING, SUPPORTED_CHUNKING_STRATEGIES, "chunking strategy")
    if chunking == "auto_horizon" and executor != "topic":
        # Benchmark episodes measure full-chunk policies against recorded
        # action sequences; consuming a result-level prefix there would
        # silently change the evaluated trajectories. Reject the
        # combination in the shared SSOT instead of ignoring the field
        # per path.
        raise DispatchStrategyError(
            f"chunking strategy 'auto_horizon' requires executor type 'topic' "
            f"(benchmark executes full chunks); got executor_type={executor!r}"
        )
    blending = _resolve_name(blending, "none", SUPPORTED_BLENDING_STRATEGIES, "blending strategy")
    return DispatchStrategySelection(executor, scheduler, chunking, blending)


def validate_executor_scheduler_pairing(executor_type: str, scheduler_mode: str) -> None:
    """Generic executor/scheduler pairing guard (canonical implementation).

    Legal combinations:

    - ``topic`` + ``continuous``
    - ``benchmark`` + ``wait_for_feedback``

    Illegal:

    - ``benchmark`` + ``continuous`` (benchmark requires step feedback)
    - ``wait_for_feedback`` + any executor other than ``benchmark``

    Unknown executor/scheduler strings are NOT aliased, case-folded or
    fallback-corrected here; they pass through so the executor/scheduler
    registries can fail-fast with their own clear error messages. The
    legacy ``action`` string is preserved verbatim (it is not silently
    rewritten to ``topic``).

    Raises:
        DispatchStrategyError: if a known special-pair rule is violated.
    """
    # benchmark requires wait_for_feedback; benchmark + continuous illegal.
    if executor_type == "benchmark" and scheduler_mode != "wait_for_feedback":
        raise DispatchStrategyError(
            f"executor type 'benchmark' requires scheduler_mode 'wait_for_feedback'; "
            f"got scheduler_mode={scheduler_mode!r}"
        )
    # wait_for_feedback requires benchmark; any non-benchmark executor illegal.
    if scheduler_mode == "wait_for_feedback" and executor_type != "benchmark":
        raise DispatchStrategyError(
            f"scheduler_mode 'wait_for_feedback' requires executor type 'benchmark'; "
            f"got executor_type={executor_type!r}"
        )


def validate_chunking_strategy(chunking: str) -> None:
    """Exact-match validation of a chunking strategy name (no aliases)."""
    if chunking not in SUPPORTED_CHUNKING_STRATEGIES:
        raise DispatchStrategyError(
            f"unknown chunking strategy {chunking!r}; expected one of {SUPPORTED_CHUNKING_STRATEGIES}"
        )


def validate_blending_strategy(blending: str) -> None:
    """Exact-match validation of a blending strategy name (no aliases)."""
    if blending not in SUPPORTED_BLENDING_STRATEGIES:
        raise DispatchStrategyError(
            f"unknown blending strategy {blending!r}; expected one of {SUPPORTED_BLENDING_STRATEGIES}"
        )


def validate_dispatch_strategies(
    *,
    executor_type: str,
    scheduler_mode: str,
    chunking: str | None = None,
    blending: str | None = None,
) -> None:
    """Validate the complete dispatch strategy combination.

    Pairing rules plus exact-name validation for chunking and blending.
    Defaults are resolved by :func:`resolve_dispatch_strategies`.
    """
    resolve_dispatch_strategies(
        executor_type=executor_type, scheduler_mode=scheduler_mode, chunking=chunking, blending=blending
    )


__all__ = [
    "DEFAULT_CHUNKING",
    "DEFAULT_SCHEDULER",
    "DispatchStrategyError",
    "DispatchStrategySelection",
    "SUPPORTED_BLENDING_STRATEGIES",
    "SUPPORTED_CHUNKING_STRATEGIES",
    "SUPPORTED_SCHEDULERS",
    "resolve_dispatch_strategies",
    "validate_blending_strategy",
    "validate_chunking_strategy",
    "validate_dispatch_strategies",
    "validate_executor_scheduler_pairing",
]
