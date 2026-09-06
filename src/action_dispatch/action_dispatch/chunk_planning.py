"""Shared chunk-planning boundary for action dispatch (pure Python + numpy).

Chunk planning owns how an accepted inference chunk enters the executable
plan: acceptance, execution-prefix skipping, truncation and consumption
metadata. It never publishes robot commands, starts inference, closes
sessions or decides blending weights — those belong to the executor,
scheduler, lifecycle and blending layers respectively.

ActivePlan applies the candidate and owns path-specific capacity handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from robot_config.dispatch_strategies import (
    SUPPORTED_CHUNKING_STRATEGIES as SUPPORTED_CHUNKING,
)

from .action_chunk import validate_execution_horizon


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Executable plan update produced by a chunk-planning strategy.

    Attributes:
        actions: The accepted action chunk, shape ``[steps, action_dim]``.
        executed_during_inference: Actions of this chunk already consumed
            while the inference request was in flight (alignment skip);
            adaptive-horizon plans clamp it to the horizon when the whole
            prefix expired, selecting the empty interval ``[H, H)``.
        execution_horizon: Executable prefix selected by an adaptive
            execution horizon (AutoHorizon). ``None`` keeps the full
            remaining chunk, which is the watermark-compatible behavior.
        replenishment_watermark: Plan-level replenishment threshold the
            per-tick scheduler should use while this plan is active.
            ``None`` keeps the configured watermark; an adaptive-horizon
            plan carries ``0`` so inference is re-requested once the prefix
            is consumed instead of watermark prefetching.
    """

    actions: np.ndarray
    executed_during_inference: int = 0
    execution_horizon: int | None = None
    replenishment_watermark: int | None = None

    @property
    def start(self) -> int:
        return self.executed_during_inference

    @property
    def stop(self) -> int:
        return self.actions.shape[0] if self.execution_horizon is None else self.execution_horizon

    @property
    def executable_actions(self) -> np.ndarray:
        """Unexecuted actions selected for the current plan."""
        end = self.execution_horizon if self.execution_horizon is not None else self.actions.shape[0]
        return self.actions[self.executed_during_inference : end]


def validate_chunk_plan(plan: ChunkPlan, *, action_dimension: int | None = None) -> ChunkPlan:
    """Validate the chunk-plan contract without owning queue policy."""
    if plan.actions.ndim != 2:
        raise ValueError(f"actions must have rank 2, got shape {plan.actions.shape}")
    if not all(plan.actions.shape):
        raise ValueError("actions must be nonempty")
    if type(plan.start) is not int or not 0 <= plan.start <= plan.actions.shape[0]:
        raise ValueError("executed_during_inference must be within the action chunk")
    if plan.execution_horizon is not None and (
        type(plan.stop) is not int or not 0 <= plan.stop <= plan.actions.shape[0]
    ):
        raise ValueError("execution_horizon must be within the action chunk")
    if plan.start > plan.stop:
        raise ValueError("selected interval start must not exceed stop")
    if plan.replenishment_watermark is not None and (
        type(plan.replenishment_watermark) is not int or plan.replenishment_watermark < 0
    ):
        raise ValueError("replenishment_watermark must be a nonnegative integer")
    if action_dimension is not None and plan.actions.shape[1] != action_dimension:
        raise ValueError(f"expected action dimension {action_dimension}, got {plan.actions.shape[1]}")
    if not np.isfinite(plan.actions).all():
        raise ValueError("actions contain non-finite values")
    return plan


class ChunkPlanner(ABC):
    """Chunk-planning strategy contract.

    Implementations decide how an incoming inference chunk becomes an
    executable plan without mutating accepted storage.
    """

    @abstractmethod
    def plan(
        self,
        actions: np.ndarray,
        *,
        actions_executed: int,
        execution_horizon: int | None = None,
    ) -> ChunkPlan:
        """Plan the executable update for an accepted inference chunk.

        ``execution_horizon`` is the optional result-level executable prefix
        (0/``None`` = full chunk). Whether a strategy honors it is part of
        the strategy contract: ``full_chunk`` ignores it, ``auto_horizon``
        truncates the plan and switches to consume-then-replan replenishment.
        """

    @property
    @abstractmethod
    def chunking_strategy(self) -> str:
        """Return the exact SSOT chunking strategy name."""


class FullChunkPlanner(ChunkPlanner):
    """Watermark-compatible full-chunk acceptance.

    Accepts the whole chunk and only aligns it with the actions consumed
    during inference; the plan is then replenished by the watermark rule
    owned by the per-tick scheduler. Result-level execution horizons are
    ignored: with ``full_chunk`` the dispatcher executes complete chunks
    regardless of any prefix the inference side may report. This preserves
    the pre-strategy behavior of both dispatcher paths byte-for-byte.
    """

    def plan(
        self,
        actions: np.ndarray,
        *,
        actions_executed: int,
        execution_horizon: int | None = None,
    ) -> ChunkPlan:
        if actions.ndim != 2:
            raise ValueError(f"actions must have rank 2, got shape {actions.shape}")
        skipped = max(0, min(int(actions_executed), actions.shape[0]))
        return validate_chunk_plan(ChunkPlan(actions=actions, executed_during_inference=skipped))

    @property
    def chunking_strategy(self) -> str:
        return "full_chunk"


class AutoHorizonPlanner(ChunkPlanner):
    """Adaptive execution-horizon planning (AutoHorizon).

    Consumes the result-level ``execution_horizon`` reported by the
    inference side: the executable plan is truncated to the horizon prefix
    and the plan's replenishment watermark drops to 0, so the dispatcher
    re-requests inference as soon as the prefix is consumed instead of
    watermark prefetching. A missing or zero horizon safely falls back to
    the full chunk with the configured watermark.

    A prefix that fully expired while the request was in flight
    (``actions_executed >= horizon``) is not an invalid chunk: the result
    is legal, only its executable prefix is stale. The planner normalizes
    it to the legal empty interval ``[horizon, horizon)`` with watermark
    0, the owner atomically accepts the empty plan (clearing the previous
    one), and the scheduler re-requests inference on the next tick.
    """

    def plan(
        self,
        actions: np.ndarray,
        *,
        actions_executed: int,
        execution_horizon: int | None = None,
    ) -> ChunkPlan:
        if actions.ndim != 2:
            raise ValueError(f"actions must have rank 2, got shape {actions.shape}")
        skipped = max(0, min(int(actions_executed), actions.shape[0]))
        horizon = validate_execution_horizon(execution_horizon, actions.shape[0])
        if horizon is None:
            return validate_chunk_plan(ChunkPlan(actions=actions, executed_during_inference=skipped))
        return validate_chunk_plan(
            ChunkPlan(
                actions=actions,
                executed_during_inference=min(skipped, horizon),
                execution_horizon=horizon,
                replenishment_watermark=0,
            )
        )

    def reset(self) -> None:
        return

    @property
    def chunking_strategy(self) -> str:
        return "auto_horizon"


def create_chunk_planner(chunking: str) -> ChunkPlanner:
    """Exact-name factory for chunk-planning strategies (no aliases)."""
    if chunking == "full_chunk":
        return FullChunkPlanner()
    if chunking == "auto_horizon":
        return AutoHorizonPlanner()
    raise ValueError(f"unknown chunking strategy {chunking!r}; expected one of {SUPPORTED_CHUNKING}")


__all__ = [
    "AutoHorizonPlanner",
    "ChunkPlan",
    "ChunkPlanner",
    "FullChunkPlanner",
    "SUPPORTED_CHUNKING",
    "create_chunk_planner",
    "validate_chunk_plan",
]
