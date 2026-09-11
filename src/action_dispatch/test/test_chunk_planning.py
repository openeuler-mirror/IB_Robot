"""Chunk-planning boundary tests (pure, no ROS)."""

from __future__ import annotations

import numpy as np
import pytest

from action_dispatch.chunk_planning import (
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
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        create_chunk_planner("auto_horizon")
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        create_chunk_planner("FULL_CHUNK")


def test_full_chunk_planner_reports_strategy_name():
    assert FullChunkPlanner().chunking_strategy == "full_chunk"
