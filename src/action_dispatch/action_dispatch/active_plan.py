"""Local accepted-plan ownership. Lock order: node, owner, smoother.

Sources identify local acceptance, not remote policy-cache acknowledgments.
Refills allocate storage; ticks copy at most one action, never the whole plan.
"""

from collections import deque
from dataclasses import dataclass
from threading import RLock

import numpy as np
import torch

from .action_blending import BlendedAction
from .chunk_planning import validate_chunk_plan


@dataclass(frozen=True, slots=True)
class PlanSource:
    request_id: str
    request_generation: int | None = None
    session_id: str | None = None
    session_generation: int | None = None


@dataclass(frozen=True, slots=True)
class PlanSnapshot:
    revision: int
    remaining: int
    watermark: int
    source: PlanSource | None
    next_position: int | None


@dataclass(frozen=True, slots=True)
class PlanReservation:
    revision: int
    consumed: int
    source: PlanSource | None


class ActivePlan:
    def __init__(self, *, capacity: int, watermark: int, overflow: str = "bounded", smoother=None):
        if capacity <= 0 or overflow not in ("bounded", "fail_closed"):
            raise ValueError("invalid active plan capacity or overflow policy")
        self._lock = RLock()
        self._capacity = capacity
        self._default_watermark = watermark
        self._overflow = overflow
        self._smoother = smoother
        self._queue = deque()
        self._revision = 0
        self._consumed = 0
        self._source = None
        self._position = None
        self._watermark = watermark
        self._reserved_tail = None
        self._reservation = None

    def snapshot(self):
        with self._lock:
            remaining = self._smoother.plan_length if self._smoother is not None else len(self._queue)
            return PlanSnapshot(self._revision, remaining, self._watermark, self._source, self._position)

    def accept(self, candidate, source: PlanSource, *, action_dimension=None, tensor_actions=None):
        with self._lock:
            validate_chunk_plan(candidate, action_dimension=action_dimension)
            start, stop = candidate.start, candidate.stop
            watermark = candidate.replenishment_watermark
            if watermark is None:
                watermark = self._default_watermark
            if self._smoother is None:
                if stop - start > self._capacity:
                    if self._overflow == "fail_closed":
                        raise ValueError(
                            f"scheduled action chunk has {stop - start} remaining steps, "
                            f"queue capacity is {self._capacity}"
                        )
                    start = stop - self._capacity
                prepared = deque(candidate.actions[start:stop].copy())
                position = start
            else:
                actions = candidate.actions if tensor_actions is None else tensor_actions
                if actions.shape != candidate.actions.shape:
                    raise ValueError("tensor actions must match the candidate shape")
                prepared = self._smoother.prepare(actions[start:stop])
                position = None if self._smoother.is_enabled else start
            # Every validation/allocation precedes the storage and metadata swap.
            if self._smoother is None:
                self._queue = prepared
            else:
                self._smoother.commit(prepared)
            self._source = source
            self._position = position
            self._watermark = watermark
            self._revision += 1
            self._consumed = 0
            self._reserved_tail = None
            self._reservation = None
            return self.snapshot()

    def clear(self):
        with self._lock:
            self._queue.clear()
            if self._smoother is not None:
                self._smoother.reset()
            self._source = None
            self._position = None
            self._watermark = self._default_watermark
            self._revision += 1
            self._consumed = 0
            self._reserved_tail = None
            self._reservation = None

    def set_smoothing_enabled(self, enabled):
        """Legacy toggles retain the same store and its accepted metadata."""
        with self._lock:
            if self._smoother is not None:
                self._smoother.set_enabled(enabled)

    def reserve(self):
        with self._lock:
            if not self.snapshot().remaining:
                return None
            value = self._smoother.peek_next_action() if self._smoother is not None else self._queue[0]
            action = (
                value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.array(value, copy=True)
            )
            if self._smoother is not None:
                # Prepare tensor views before the episode commits. Consumption
                # then swaps references only, with no indexing/device conversion.
                store = self._smoother._smoother
                self._reserved_tail = (store._smoothed_actions[1:], store._action_counts[1:])
            if self._reservation is None:
                self._reservation = PlanReservation(self._revision, self._consumed, self._source)
            self._reserved_consumed = self._consumed + 1
            self._reserved_position = None if self._position is None else self._position + 1
            return self._reservation, action

    def is_current(self, reservation):
        with self._lock:
            return (
                reservation is not None
                and reservation is self._reservation
                and reservation.revision == self._revision
                and reservation.consumed == self._consumed
                and reservation.source == self._source
                and (self._smoother.plan_length if self._smoother is not None else len(self._queue)) > 0
            )

    def commit(self, reservation):
        with self._lock:
            if not self.is_current(reservation):
                return False
            self.commit_reserved()
            return True

    def commit_reserved(self):
        """Consume after is_current under the same node lock, without validation.

        Benchmark calls this only after its episode accepts the step. All tensor
        views and next counters were prepared by reserve before that transaction.
        """
        with self._lock:
            if self._smoother is not None:
                self._smoother.commit(self._reserved_tail)
            else:
                self._queue.popleft()
            self._consumed = self._reserved_consumed
            self._position = self._reserved_position
            self._reserved_tail = None
            self._reservation = None

    def take_action(self, *, last_action=None):
        with self._lock:
            reserved = self.reserve()
            if reserved is None:
                return BlendedAction(last_action, "hold" if last_action is not None else "empty")
            reservation, action = reserved
            self.commit(reservation)
            return BlendedAction(action, "smoother" if self._smoother is not None else "queue")
