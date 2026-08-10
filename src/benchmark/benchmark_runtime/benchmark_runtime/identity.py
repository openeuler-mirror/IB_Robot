"""Environment-owned identity, exactly-once step barrier and poison state.

Benchmark runtime scope: this module owns the runtime identity that the generic
``benchmark_environment_node`` uses to guarantee exactly-once semantics for
every benchmark adapter (LIBERO today, Meta-World / future benchmarks later).
It is intentionally pure-Python:

- imports only the standard library and NumPy;
- never imports ``rclpy``, ROS messages, ``benchmark_libero`` or LIBERO/MuJoCo;
- never allocates ROS timestamps (the node asks its ROS clock and passes the
  resulting stamp in);
- never creates threads, services, executors or registries.

The model encodes the LIBERO runtime transaction rules:

- ``episode_id`` starts at 0 (no episode yet) and increments monotonically by 1
  on each successful reset. The first successful reset produces
  ``episode_id = 1``.
- ``expected_step_id`` is 0 immediately after a successful reset. A successful
  step advances it by exactly 1.
- A failed reset does NOT consume a public episode ID and must invalidate the
  previous episode (the previous identity must not continue).
- A successful reset invalidates the previous episode and starts a new
  identity. The old identity never survives any reset, including a reset that
  itself fails afterwards.
- Stale, duplicate, future, wrong-episode or step-before-reset requests fail
  BEFORE the adapter is called and do not advance identity.
- If a native ``adapter.step()`` was entered and then raised or returned an
  invalid result, the simulator mutation cannot be disproved. The episode is
  marked ``poisoned``/``reset_required``: no new public step ID is committed,
  the same native step is never retried, and all later Step requests are
  rejected until a successful Reset starts a new episode.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepIdentity:
    """Snapshot of the currently committed public identity.

    ``episode_id`` is 0 when no episode is active (before the first successful
    reset, after a failed reset, or while the current episode is poisoned).
    ``expected_step_id`` is 0 immediately after a successful reset.
    ``poisoned`` is True after a native step entry failure; only a successful
    reset clears it.
    """

    episode_id: int
    expected_step_id: int
    poisoned: bool


class EnvironmentRuntime:
    """Pure-Python identity / exactly-once / poison state for the env node.

    All mutating methods take the internal lock. Read-only snapshot
    :meth:`snapshot` is also lock-protected. The lock is an ``RLock`` so the
    node may call public methods from sections already holding the lock.

    The runtime never sees ROS messages, NumPy arrays or action payloads. It
    only authorizes or rejects identity transitions.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._episode_id: int = 0
        self._expected_step_id: int = 0
        self._poisoned: bool = False
        self._next_episode_id: int = 1
        self._reset_in_progress: bool = False
        self._step_in_progress: bool = False
        self._adapter_step_count: int = 0

    @property
    def lock(self) -> threading.RLock:
        """Return the runtime lock. The node may lock it to make a compound
        operation atomic with respect to concurrent reset/step callbacks."""
        return self._lock

    # ------------------------------------------------------------------ #
    # Read-only snapshots
    # ------------------------------------------------------------------ #

    def snapshot(self) -> StepIdentity:
        with self._lock:
            return StepIdentity(
                episode_id=self._episode_id,
                expected_step_id=self._expected_step_id,
                poisoned=self._poisoned,
            )

    @property
    def episode_id(self) -> int:
        with self._lock:
            return self._episode_id

    @property
    def expected_step_id(self) -> int:
        with self._lock:
            return self._expected_step_id

    @property
    def is_poisoned(self) -> bool:
        with self._lock:
            return self._poisoned

    @property
    def adapter_step_count(self) -> int:
        """Number of successful native adapter.step() calls committed.

        Exposed for tests that need to prove exactly-once. The runtime does
        not use this value internally.
        """
        with self._lock:
            return self._adapter_step_count

    # ------------------------------------------------------------------ #
    # Reset transaction
    # ------------------------------------------------------------------ #

    def begin_reset(self) -> None:
        """Begin a reset transaction. Always invalidates the previous episode.

        Even if this reset later fails, the previous episode identity must NOT
        continue. Therefore we poison and clear the public episode id here.
        """
        with self._lock:
            if self._reset_in_progress:
                raise RuntimeError("reset already in progress")
            if self._step_in_progress:
                raise RuntimeError("cannot begin reset while a step is in progress")
            self._reset_in_progress = True
            # Invalidate the previous identity immediately. A failed reset
            # must NOT let the old identity continue.
            self._episode_id = 0
            self._expected_step_id = 0
            self._poisoned = True

    def commit_reset_success(self) -> tuple[int, int]:
        """Commit a successful reset. Returns the new ``(episode_id, step_id)``.

        Allocates the next monotonic episode ID, sets ``expected_step_id = 0``
        and clears the poison flag. The previous identity (already
        invalidated by ``begin_reset``) cannot survive.
        """
        with self._lock:
            if not self._reset_in_progress:
                raise RuntimeError("commit_reset_success without begin_reset")
            new_episode = self._next_episode_id
            self._next_episode_id += 1
            self._episode_id = new_episode
            self._expected_step_id = 0
            self._poisoned = False
            self._reset_in_progress = False
            return (self._episode_id, 0)

    def commit_reset_failure(self) -> None:
        """Commit a failed reset. Old episode stays invalidated; poison stays.

        Episode ID remains 0 (no current episode). The next successful reset
        will allocate the next monotonic ID. The previous identity cannot
        survive a failed reset.
        """
        with self._lock:
            if not self._reset_in_progress:
                raise RuntimeError("commit_reset_failure without begin_reset")
            self._episode_id = 0
            self._expected_step_id = 0
            self._poisoned = True
            self._reset_in_progress = False

    # ------------------------------------------------------------------ #
    # Step transaction
    # ------------------------------------------------------------------ #

    def begin_step(self, episode_id: int, expected_step_id: int) -> tuple[bool, str]:
        """Validate a step request before calling the adapter.

        Returns ``(ok, message)``. On failure, the message is a stable
        diagnostic string suitable for the StepBenchmark response. On
        success, marks a step-in-progress so a concurrent reset cannot
        invalidate the identity mid-step.

        The adapter is NOT called from inside this method. The node calls
        :meth:`begin_step`, then the adapter, then either
        :meth:`commit_step_success` or :meth:`poison_after_native_failure`.
        """
        with self._lock:
            if self._reset_in_progress:
                return (False, "reset in progress")
            if self._step_in_progress:
                return (False, "another step is in progress")
            if self._poisoned:
                return (False, "episode poisoned; reset_required")
            if self._episode_id == 0:
                return (False, "no active episode; reset first")
            if episode_id != self._episode_id:
                return (
                    False,
                    f"episode_id mismatch: got {episode_id}, expected {self._episode_id}",
                )
            if expected_step_id != self._expected_step_id:
                return (
                    False,
                    f"expected_step_id mismatch: got {expected_step_id}, expected {self._expected_step_id}",
                )
            self._step_in_progress = True
            return (True, "")

    def commit_step_success(self) -> int:
        """Commit a successful step. Returns the committed ``step_id``.

        Advances ``expected_step_id`` by exactly 1 and bumps the adapter step
        counter. Does NOT clear the poison flag (a successful step does not
        un-poison an episode; only a successful reset does — but a poisoned
        episode cannot reach here because begin_step rejects it).
        """
        with self._lock:
            if not self._step_in_progress:
                raise RuntimeError("commit_step_success without begin_step")
            committed = self._expected_step_id
            self._expected_step_id += 1
            self._adapter_step_count += 1
            self._step_in_progress = False
            return committed

    def poison_after_native_failure(self) -> None:
        """Poison the current episode after a native step entry failure.

        Called when ``adapter.step()`` raised or returned an invalid result
        AFTER the native environment was entered. The public committed step ID
        is NOT advanced, the same native step is NOT retried, and all
        subsequent Step requests are rejected until a successful Reset starts
        a new episode.
        """
        with self._lock:
            self._poisoned = True
            # Do NOT advance expected_step_id.
            self._step_in_progress = False

    def abort_step(self) -> None:
        """Abort a step that has not yet entered the native adapter.

        Used when action decode/validation fails AFTER begin_step authorized
        the request, but BEFORE the adapter was called. Identity is unchanged
        (no advance, no poison); the step slot is simply released so the next
        request can proceed.
        """
        with self._lock:
            self._step_in_progress = False
