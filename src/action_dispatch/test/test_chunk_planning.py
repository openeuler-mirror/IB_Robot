"""Chunk-planning boundary tests (pure, no ROS)."""

from __future__ import annotations

import numpy as np
import pytest

from action_dispatch.chunk_planning import (
    AutoHorizonPlanner,
    ChunkPlan,
    FullChunkPlanner,
    create_chunk_planner,
    validate_chunk_plan,
)


def _chunk(steps: int):
    return np.arange(steps * 2, dtype=np.float32).reshape(steps, 2)


def test_full_chunk_planner_accepts_whole_chunk():
    plan = FullChunkPlanner().plan(_chunk(4), actions_executed=0)

    assert plan.executed_during_inference == 0
    assert plan.execution_horizon is None
    assert np.array_equal(plan.executable_actions, _chunk(4))


def test_full_chunk_planner_skips_executed_prefix():
    plan = FullChunkPlanner().plan(_chunk(4), actions_executed=2)

    assert plan.executed_during_inference == 2
    assert np.array_equal(plan.executable_actions, _chunk(4)[2:])


def test_full_chunk_planner_clamps_skip():
    assert FullChunkPlanner().plan(_chunk(3), actions_executed=-5).executed_during_inference == 0
    assert FullChunkPlanner().plan(_chunk(3), actions_executed=99).executed_during_inference == 3
    assert FullChunkPlanner().plan(_chunk(3), actions_executed=99).executable_actions.shape[0] == 0


def test_full_chunk_planner_rejects_wrong_rank():
    with pytest.raises(ValueError, match="rank 2"):
        FullChunkPlanner().plan(np.zeros(6, dtype=np.float32), actions_executed=0)


def test_execution_horizon_truncates_executable_actions():
    plan = ChunkPlan(actions=_chunk(5), executed_during_inference=1, execution_horizon=3)

    assert np.array_equal(plan.executable_actions, _chunk(5)[1:3])


def test_validate_chunk_plan_rejects_non_finite_and_bad_bounds():
    nan_chunk = _chunk(2)
    nan_chunk[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_chunk_plan(ChunkPlan(actions=nan_chunk))
    with pytest.raises(ValueError, match="executed_during_inference"):
        validate_chunk_plan(ChunkPlan(actions=_chunk(2), executed_during_inference=3))
    with pytest.raises(ValueError, match="execution_horizon"):
        validate_chunk_plan(ChunkPlan(actions=_chunk(2), execution_horizon=3))
    with pytest.raises(ValueError, match="action dimension"):
        validate_chunk_plan(ChunkPlan(actions=_chunk(2)), action_dimension=7)


def test_create_chunk_planner_exact_names_only():
    assert isinstance(create_chunk_planner("full_chunk"), FullChunkPlanner)
    assert isinstance(create_chunk_planner("auto_horizon"), AutoHorizonPlanner)
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        create_chunk_planner("rtc")
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        create_chunk_planner("FULL_CHUNK")


def test_full_chunk_planner_reports_strategy_name():
    assert FullChunkPlanner().chunking_strategy == "full_chunk"


def test_auto_horizon_planner_truncates_to_execution_horizon():
    plan = AutoHorizonPlanner().plan(_chunk(4), actions_executed=1, execution_horizon=3)

    assert plan.execution_horizon == 3
    assert np.array_equal(plan.executable_actions, _chunk(4)[1:3])
    # Adaptive-horizon plans re-request inference once the prefix is consumed.
    assert plan.replenishment_watermark == 0


def test_auto_horizon_planner_falls_back_to_full_chunk_without_horizon():
    for missing in (None, 0):
        plan = AutoHorizonPlanner().plan(_chunk(3), actions_executed=0, execution_horizon=missing)

        assert plan.execution_horizon is None
        assert plan.replenishment_watermark is None
        assert np.array_equal(plan.executable_actions, _chunk(3))


def test_auto_horizon_planner_normalizes_expired_prefix_to_empty_plan():
    """S > H after a watermark prefetch is a legal, fully expired prefix.

    The planner must produce the legal empty interval [H, H) with
    watermark 0 instead of the invalid [S, H), so the owner accepts the
    (empty) plan and the scheduler re-requests inference.
    """
    plan = AutoHorizonPlanner().plan(_chunk(5), actions_executed=5, execution_horizon=3)

    assert plan.execution_horizon == 3
    assert plan.executed_during_inference == 3
    assert plan.start == plan.stop == 3
    assert plan.executable_actions.shape[0] == 0
    assert plan.replenishment_watermark == 0
    # The original horizon stays the exclusive stop; it is not rewritten
    # to a post-skip length.
    assert plan.actions.shape[0] == 5


@pytest.mark.parametrize("skip", [4, 5, 99])
def test_auto_horizon_planner_expired_prefix_is_never_an_invalid_interval(skip):
    plan = AutoHorizonPlanner().plan(_chunk(4), actions_executed=skip, execution_horizon=4)

    assert plan.start <= plan.stop
    assert plan.executable_actions.shape[0] == 0


def test_auto_horizon_planner_zero_horizon_after_watermark_prefetch_keeps_full_chunk():
    """A 0 (fallback) horizon after watermark prefetch plans the remainder."""
    plan = AutoHorizonPlanner().plan(_chunk(5), actions_executed=2, execution_horizon=0)

    assert plan.execution_horizon is None
    assert plan.replenishment_watermark is None
    assert np.array_equal(plan.executable_actions, _chunk(5)[2:])


def test_auto_horizon_empty_plan_clears_queue_owner_and_replenishes():
    from action_dispatch.active_plan import ActivePlan, PlanSource

    chunk = _chunk(5)
    owner = ActivePlan(capacity=10, watermark=4)
    owner.accept(
        AutoHorizonPlanner().plan(chunk, actions_executed=0, execution_horizon=5),
        PlanSource("prefetch"),
    )
    assert owner.snapshot().remaining == 5

    expired = AutoHorizonPlanner().plan(chunk, actions_executed=5, execution_horizon=3)
    snapshot = owner.accept(expired, PlanSource("expired"))

    assert snapshot.remaining == 0
    assert snapshot.watermark == 0
    # Consume-then-replan: the shared watermark rule re-requests immediately.
    from action_dispatch.schedulers.continuous import should_replenish_plan

    assert should_replenish_plan(snapshot.remaining, snapshot.watermark, inference_in_progress=False)
    assert owner.take_action().source == "empty"


def test_auto_horizon_empty_plan_clears_smoother_owner():
    from action_dispatch.active_plan import ActivePlan, PlanSource
    from action_dispatch.temporal_smoother import TemporalSmootherManager

    chunk = _chunk(4)
    manager = TemporalSmootherManager(enabled=True, chunk_size=4)
    owner = ActivePlan(capacity=10, watermark=4, smoother=manager)
    owner.accept(
        AutoHorizonPlanner().plan(chunk, actions_executed=0, execution_horizon=4),
        PlanSource("prefetch"),
    )
    assert owner.snapshot().remaining == 4

    expired = AutoHorizonPlanner().plan(chunk, actions_executed=4, execution_horizon=2)
    snapshot = owner.accept(expired, PlanSource("expired"))

    assert snapshot.remaining == 0
    assert snapshot.watermark == 0
    assert owner.take_action().source == "empty"


def test_auto_horizon_planner_rejects_out_of_range_horizon():
    with pytest.raises(ValueError, match="execution_horizon"):
        AutoHorizonPlanner().plan(_chunk(3), actions_executed=0, execution_horizon=4)


def test_auto_horizon_planner_reports_strategy_name():
    assert AutoHorizonPlanner().chunking_strategy == "auto_horizon"


def test_full_chunk_planner_ignores_execution_horizon():
    plan = FullChunkPlanner().plan(_chunk(4), actions_executed=0, execution_horizon=2)

    assert plan.execution_horizon is None
    assert plan.replenishment_watermark is None
    assert np.array_equal(plan.executable_actions, _chunk(4))


def test_fail_closed_owner_counts_horizon_prefix_against_capacity():
    from action_dispatch.active_plan import ActivePlan, PlanSource

    plan = AutoHorizonPlanner().plan(_chunk(6), actions_executed=0, execution_horizon=3)

    # Only the horizon prefix counts against the scheduled queue capacity.
    owner = ActivePlan(capacity=4, watermark=2, overflow="fail_closed")
    snapshot = owner.accept(plan, PlanSource("request"))

    assert snapshot.remaining == 3
    assert np.array_equal(np.asarray(owner._queue), _chunk(6)[:3])
