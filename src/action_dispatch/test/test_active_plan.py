"""Accepted-plan transactions, ownership and logical consumption."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from action_dispatch.active_plan import ActivePlan, PlanSource
from action_dispatch.chunk_planning import ChunkPlan, FullChunkPlanner, validate_chunk_plan
from action_dispatch.temporal_smoother import TemporalSmootherManager


def chunk(n=6):
    return np.arange(n * 2, dtype=np.float32).reshape(n, 2)


@pytest.mark.parametrize("start,stop", [(4, 3), (-1, 3), (0, 7), (1.5, 3), (0, True)])
def test_invalid_intervals(start, stop):
    with pytest.raises(ValueError):
        validate_chunk_plan(ChunkPlan(chunk(), start, stop))


@pytest.mark.parametrize(
    "actions", [np.empty((0, 2)), np.empty((2, 0)), np.zeros(2), np.array([[np.nan, 0]]), np.array([[np.inf, 0]])]
)
def test_invalid_raw_chunk(actions):
    with pytest.raises(ValueError):
        FullChunkPlanner().plan(actions, actions_executed=0)


@pytest.mark.parametrize("smoothing", [None, False, True])
def test_selected_interval_ownership_and_clear(smoothing):
    manager = None if smoothing is None else TemporalSmootherManager(enabled=smoothing)
    owner = ActivePlan(capacity=3, watermark=2, smoother=manager)
    data = chunk()
    source = PlanSource("req", 2, "session", 3)
    snapshot = owner.accept(ChunkPlan(data, 2, 5, 0), source)
    assert snapshot.remaining == 3
    assert snapshot.watermark == 0
    assert snapshot.source == source
    assert snapshot.next_position == (None if smoothing else 2)
    assert snapshot.consumed == 0
    with pytest.raises(FrozenInstanceError):
        snapshot.remaining = 9
    with pytest.raises(FrozenInstanceError):
        snapshot.consumed = 9
    with pytest.raises(FrozenInstanceError):
        snapshot.source.request_id = "other"
    data[:] = -1
    reservation, action = owner.reserve()
    action[:] = -2
    assert owner.snapshot() == snapshot
    for consumed, expected in enumerate(chunk()[2:5], start=1):
        np.testing.assert_array_equal(owner.take_action().action, expected)
        assert owner.snapshot().consumed == consumed
    assert not owner.commit(reservation)
    exhausted = owner.snapshot()
    assert owner.take_action(last_action=action).source == "hold"
    assert owner.take_action().source == "empty"
    assert owner.snapshot() == exhausted
    owner.clear()
    cleared = owner.snapshot()
    assert cleared.source is None and cleared.next_position is None
    assert cleared.watermark == 2 and cleared.remaining == 0
    assert cleared.consumed == 0
    assert cleared.revision > exhausted.revision


def test_capacity_clipping_includes_discarded_prefix():
    owner = ActivePlan(capacity=30, watermark=4)
    owner.accept(FullChunkPlanner().plan(chunk(50), actions_executed=5), PlanSource("r"))
    assert owner.snapshot().next_position == 20
    assert owner.snapshot().consumed == 0
    np.testing.assert_array_equal(owner.take_action().action, chunk(50)[20])
    assert owner.snapshot().next_position == 21
    assert owner.snapshot().consumed == 1


@pytest.mark.parametrize(
    "invalid",
    [ChunkPlan(chunk(), 4, 3), ChunkPlan(np.empty((0, 2))), ChunkPlan(np.array([[np.nan, 1]])), ChunkPlan(chunk(8))],
)
def test_failed_acceptance_preserves_snapshot_and_reservation(invalid):
    owner = ActivePlan(capacity=6, watermark=2, overflow="fail_closed")
    before = owner.accept(ChunkPlan(chunk(), 1, 4, 0), PlanSource("old"))
    reservation, action = owner.reserve()
    with pytest.raises(ValueError):
        owner.accept(invalid, PlanSource("new"))
    assert owner.snapshot() == before
    assert owner.is_current(reservation)
    np.testing.assert_array_equal(owner.reserve()[1], action)


def test_reservations_commit_once_and_never_consume_replacement():
    owner = ActivePlan(capacity=6, watermark=2)
    owner.accept(ChunkPlan(chunk()), PlanSource("r", 1))
    reservation, _ = owner.reserve()
    assert not owner.commit(replace(reservation, source=PlanSource("wrong")))
    assert owner.commit(reservation)
    assert not owner.commit(reservation)
    assert owner.snapshot().next_position == 1
    assert owner.snapshot().consumed == 1
    stale, _ = owner.reserve()
    owner.accept(ChunkPlan(chunk()), PlanSource("r", 2))
    assert owner.snapshot().consumed == 0
    assert not owner.commit(stale)
    current, _ = owner.reserve()
    owner.clear()
    assert not owner.commit(current)


@pytest.mark.parametrize("smoothing", [None, False, True])
@pytest.mark.parametrize("start,stop", [(0, 0), (6, 6)])
def test_empty_selected_interval_clears_executable_plan(smoothing, start, stop):
    manager = None if smoothing is None else TemporalSmootherManager(enabled=smoothing)
    owner = ActivePlan(capacity=6, watermark=2, smoother=manager)
    owner.accept(ChunkPlan(chunk()), PlanSource("old"))
    stale, _ = owner.reserve()
    owner.accept(ChunkPlan(chunk(), start, stop, 0), PlanSource("expired"))
    assert owner.snapshot().remaining == 0
    assert owner.snapshot().consumed == 0
    assert owner.snapshot().source.request_id == "expired"
    assert not owner.commit(stale)


@pytest.mark.parametrize("enabled", [False, True])
def test_smoother_has_no_queue_capacity_limit(enabled):
    owner = ActivePlan(
        capacity=1, watermark=0, overflow="fail_closed", smoother=TemporalSmootherManager(enabled=enabled)
    )
    assert owner.accept(ChunkPlan(chunk()), PlanSource("r")).remaining == 6


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("entry", ["prepare", "update", "owner"])
def test_smoother_allocation_failure_is_atomic(monkeypatch, existing, entry):
    manager = TemporalSmootherManager()
    owner = ActivePlan(capacity=6, watermark=2, smoother=manager)
    if existing:
        owner.accept(ChunkPlan(chunk(), 0, 3, 0), PlanSource("old"))
    before = owner.snapshot()
    smoother = manager._smoother
    actions, counts = smoother._smoothed_actions, smoother._action_counts
    action_values = None if actions is None else actions.clone()
    count_values = None if counts is None else counts.clone()

    def fail(*args, **kwargs):
        raise MemoryError("injected allocation failure")

    monkeypatch.setattr(torch, "ones", fail)
    with pytest.raises(MemoryError, match="injected"):
        if entry == "owner":
            owner.accept(ChunkPlan(chunk(), 0, 5, 1), PlanSource("new"))
        else:
            getattr(manager, entry)(chunk())
    assert owner.snapshot() == before
    assert smoother._smoothed_actions is actions
    assert smoother._action_counts is counts
    if existing:
        assert torch.equal(actions, action_values)
        assert torch.equal(counts, count_values)


def test_prepare_does_not_publish_until_commit_and_retains_weights():
    manager = TemporalSmootherManager(temporal_ensemble_coeff=0.0)
    manager.update(chunk())
    before = manager.get_plan().clone()
    prepared = manager.prepare(chunk() + 10)
    assert torch.equal(manager.get_plan(), before)
    manager.commit(prepared)
    np.testing.assert_allclose(manager.get_plan(), chunk() + 5)
    with pytest.raises(ValueError):
        manager.prepare(np.array([[np.inf, 0]]))
    np.testing.assert_allclose(manager.get_plan(), chunk() + 5)


def test_public_smoother_reads_cannot_mutate_owned_storage():
    manager = TemporalSmootherManager()
    owner = ActivePlan(capacity=6, watermark=0, smoother=manager)
    owner.accept(ChunkPlan(chunk()), PlanSource("r"))
    manager.get_plan().fill_(-1)
    manager.peek_next_action().fill_(-2)
    np.testing.assert_array_equal(owner.take_action().action, chunk()[0])


def test_smoothing_overflow_rejects_without_replacing_plan():
    manager = TemporalSmootherManager(temporal_ensemble_coeff=0.0)
    owner = ActivePlan(capacity=6, watermark=2, smoother=manager)
    large = np.full((2, 2), 3e38, dtype=np.float32)
    before = owner.accept(ChunkPlan(large), PlanSource("old"))
    with pytest.raises(ValueError, match="non-finite"):
        owner.accept(ChunkPlan(large, replenishment_watermark=0), PlanSource("new"))
    assert owner.snapshot() == before
    np.testing.assert_array_equal(owner.take_action().action, large[0])


@pytest.mark.parametrize("smoothing", [False, True])
def test_reservation_replacement_order_is_event_controlled(smoothing):
    import threading

    owner = ActivePlan(capacity=6, watermark=0, smoother=TemporalSmootherManager() if smoothing else None)
    owner.accept(ChunkPlan(chunk()), PlanSource("old"))
    reserved, replaced = threading.Event(), threading.Event()
    results = []

    def consume():
        reservation, _ = owner.reserve()
        reserved.set()
        if replaced.wait(5):
            results.append(owner.commit(reservation))

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert reserved.wait(5)
        owner.accept(ChunkPlan(chunk(), 2, 5, 0), PlanSource("new"))
        before = owner.snapshot()
    finally:
        replaced.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results == [False]
    assert owner.snapshot() == before
