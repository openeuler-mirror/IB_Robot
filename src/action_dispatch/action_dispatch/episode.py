"""Pure-Python episode state machine for goal-gated policy episodes.

This module owns ONLY the episode phase, preparation token, environment
identity, expected step, committed count, monotonic deadlines, last committed
step snapshot, and termination reason selection.

It must NOT import rclpy, LIBERO, benchmark runtime, or report code. It does
NOT own ROS handles, queue/smoother, inference client, executor, or
ActionServer callbacks. The dispatcher orchestrates those and calls into this
state machine to transition phases.

Identity boundary (permanent):

    production episode_id/expected_step_id are environment-reset-owned;
    this module never invents them.

    preparation_id is a dispatcher-local barrier token (monotonic from 1);
    it is NOT an environment identity.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum


class EpisodePhase(str, Enum):
    """Lifecycle phases of a single policy episode.

    IDLE              - no preparation active; gate closed.
    PREPARING         - strict barrier in progress (policy reset etc.).
    PREPARED          - barrier succeeded; valid unconsumed token issued; gate closed.
    ACCEPTED_PENDING  - goal accepted (token consumed) but handle not yet bound;
                        gate STILL CLOSED; no inference/step; startup deadline NOT running.
    STARTING          - handle bound; gate open; first inference pending; startup deadline running.
    RUNNING           - first inference succeeded; actions being committed.
    TERMINAL          - episode ended with a normal reason (success/limits/cancel).
    FAULTED           - episode ended with a fault reason (inference/execution/identity).
    """

    IDLE = "idle"
    PREPARING = "preparing"
    PREPARED = "prepared"
    ACCEPTED_PENDING = "accepted_pending"
    STARTING = "starting"
    RUNNING = "running"
    TERMINAL = "terminal"
    FAULTED = "faulted"


# Standard termination reasons (extensible string; future benchmarks may add
# new values but must not change the existing semantics).
TERMINATION_TASK_SUCCESS = "task_success"
TERMINATION_TERMINATED = "terminated"
TERMINATION_TRUNCATED = "truncated"
TERMINATION_MAX_ACTIONS = "max_actions"
TERMINATION_MAX_DURATION = "max_duration"
TERMINATION_CANCELED = "canceled"
TERMINATION_INFERENCE_FAILED = "inference_failed"
TERMINATION_INFERENCE_TIMEOUT = "inference_timeout"
TERMINATION_EXECUTION_REJECTED = "execution_rejected"
TERMINATION_EXECUTION_FAILED = "execution_failed"
TERMINATION_EXECUTION_UNCERTAIN = "execution_uncertain"
TERMINATION_IDENTITY_MISMATCH = "identity_mismatch"
TERMINATION_OBSERVATION_STARTUP_TIMEOUT = "observation_startup_timeout"
TERMINATION_EXTERNAL_RESET = "external_reset"

# Reasons that correspond to a ROS action SUCCEEDED terminal status.
_SUCCEEDED_REASONS = frozenset(
    {
        TERMINATION_TASK_SUCCESS,
        TERMINATION_TERMINATED,
        TERMINATION_TRUNCATED,
        TERMINATION_MAX_ACTIONS,
        TERMINATION_MAX_DURATION,
    }
)

# Terminal priority order (highest first). When multiple conditions are true
# on the same committed step, the first match in this list is selected as the
# termination_reason. All flags are still preserved verbatim in the snapshot.
_TERMINAL_PRIORITY = (
    TERMINATION_TASK_SUCCESS,
    TERMINATION_TERMINATED,
    TERMINATION_TRUNCATED,
    TERMINATION_MAX_ACTIONS,
    TERMINATION_MAX_DURATION,
)


def _is_finite_positive(value: float) -> bool:
    """Return True if value is a positive, finite number."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, int | float):
        return False
    if math.isinf(float(value)) or math.isnan(float(value)):
        return False
    return float(value) > 0.0


@dataclass(frozen=True, slots=True)
class EpisodeGoalSpec:
    """Validated RunPolicy goal parameters.

    All fields must be explicitly provided by the caller (environment /
    evaluator). The state machine never fills in defaults.

    production episode_id/expected_step_id are environment-reset-owned;
    this dataclass carries them but never invents them.
    """

    preparation_id: int
    episode_id: int
    initial_step_id: int
    initial_observation_timestamp_ns: int
    max_actions: int
    max_duration_sec: float
    startup_timeout_sec: float
    prompt: str


@dataclass(frozen=True, slots=True)
class StepCompletionData:
    """Step-level data extracted from an ExecutionCompletion for the state machine.

    The dispatcher builds this from the executor's drained completion. Raw
    JSON strings are preserved byte-for-byte; the state machine never parses,
    merges or aggregates them.
    """

    episode_id: int | None
    step_id: int | None
    observation_timestamp_ns: int | None
    has_reward: bool
    reward: float
    terminated: bool
    truncated: bool
    has_success: bool
    success: bool
    standard_metrics_json: str
    native_metrics_json: str
    info_json: str
    round_trip_latency_ms: float
    message: str


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Final snapshot for the RunPolicy result.

    ``has_final_step`` is False when no step was committed; in that case
    ``final_step_id`` must be ignored, presence flags are False, JSON fields
    are ``"{}"``, and latency is ``0.0``.
    """

    success: bool
    has_success: bool
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
class StepCommitOutcome:
    """Result of try_commit_step.

    - ``applied`` is True only when the step was actually committed (count
      incremented, snapshot stored, expected_step advanced). Feedback may
      only be published when ``applied`` is True.
    - ``terminal_reason`` is the termination reason if the episode should end
      after this commit, or None.
    - ``identity_mismatch`` is True if the step's identity didn't match.
    """

    applied: bool
    terminal_reason: str | None
    identity_mismatch: bool


@dataclass(frozen=True, slots=True)
class CloseResult:
    """Result of try_close_*_with_result.

    - ``applied`` is True if the close succeeded (generation matched, phase
      was active).
    - ``result`` is the immutable EpisodeResult snapshot frozen at close
      time, or None if the close was not applied.
    """

    applied: bool
    result: EpisodeResult | None


class EpisodeStateMachine:
    """Pure-Python episode state machine.

    The dispatcher calls into this object to transition phases. The state
    machine validates goal specs, manages the preparation token, tracks
    environment identity / expected step, counts committed actions, evaluates
    terminal conditions, and stores the last committed step snapshot.

    It does NOT perform I/O, contact ROS services, publish messages, write
    files, or aggregate metrics. All of those are the dispatcher's job.
    """

    __slots__ = (
        "_phase",
        "_token_counter",
        "_current_token",
        "_episode_id",
        "_expected_step_id",
        "_committed_count",
        "_startup_deadline_ns",
        "_episode_deadline_ns",
        "_max_actions",
        "_max_duration_sec",
        "_startup_timeout_sec",
        "_initial_timestamp_ns",
        "_current_timestamp_ns",
        "_is_first_inference",
        "_last_snapshot",
        "_termination_reason",
        "_lock",
        "_prompt",
        "_goal_generation",
    )

    def __init__(self) -> None:
        self._phase: EpisodePhase = EpisodePhase.IDLE
        # Monotonic token counter. 0 means "no valid unconsumed token".
        self._token_counter: int = 0
        self._current_token: int = 0
        # Environment identity (set on goal accept, cleared on terminal).
        self._episode_id: int | None = None
        self._expected_step_id: int | None = None
        # Committed action count (only matching COMMIT steps).
        self._committed_count: int = 0
        # Monotonic deadlines (nanoseconds).
        self._startup_deadline_ns: int = 0
        self._episode_deadline_ns: int = 0
        # Limits (stored for terminal evaluation).
        self._max_actions: int = 0
        self._max_duration_sec: float = 0.0
        self._startup_timeout_sec: float = 0.0
        # Timestamps for inference.
        self._initial_timestamp_ns: int | None = None
        self._current_timestamp_ns: int | None = None
        self._is_first_inference: bool = True
        # Last committed step snapshot (for result/feedback).
        self._last_snapshot: StepCompletionData | None = None
        # Termination reason (set on terminal/faulted).
        self._termination_reason: str | None = None
        # Thread-safety: protects all state mutations and atomic check-and-act
        # operations (try_accept_goal, try_begin_preparing). Read-only
        # properties do NOT acquire the lock; they may see slightly stale
        # state, which is acceptable for the control loop's tick-level decisions.
        self._lock: threading.RLock = threading.RLock()
        # benchmark episode controller correctness handling: store goal prompt for DispatchInfer.
        self._prompt: str | None = None
        # benchmark episode controller correctness handling: goal generation counter for cross-episode isolation.
        # Increments on each try_accept_goal. The execute callback captures its
        # generation and all terminal/cancel/abort operations check it.
        self._goal_generation: int = 0

    # ------------------------------------------------------------------
    # Phase accessors
    # ------------------------------------------------------------------

    @property
    def phase(self) -> EpisodePhase:
        return self._phase

    @property
    def is_gate_open(self) -> bool:
        """True when the episode gate allows inference/submit (STARTING or RUNNING).

        ACCEPTED_PENDING is NOT gate-open: the goal handle is not yet bound,
        so no inference/step may proceed.
        """
        return self._phase in (EpisodePhase.STARTING, EpisodePhase.RUNNING)

    @property
    def is_active_goal(self) -> bool:
        """True when a RunPolicy goal is active or pending (ACCEPTED_PENDING,
        STARTING, or RUNNING). PreparePolicyEpisode is rejected in all three."""
        return self._phase in (
            EpisodePhase.ACCEPTED_PENDING,
            EpisodePhase.STARTING,
            EpisodePhase.RUNNING,
        )

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    @property
    def committed_count(self) -> int:
        return self._committed_count

    @property
    def episode_id(self) -> int | None:
        return self._episode_id

    @property
    def expected_step_id(self) -> int | None:
        return self._expected_step_id

    @property
    def max_actions(self) -> int:
        return self._max_actions

    @property
    def startup_timeout_sec(self) -> float:
        """Public read-only access to the startup timeout (seconds)."""
        return self._startup_timeout_sec

    @property
    def readiness_deadline_ns(self) -> int:
        """Monotonic deadline (ns) for the STARTING phase startup timeout."""
        return self._startup_deadline_ns

    @property
    def goal_generation(self) -> int:
        """Current goal generation counter (increments on each accepted goal)."""
        return self._goal_generation

    def get_prompt(self) -> str | None:
        """Return the current RunPolicy goal's prompt, or None if no active goal.

        Used by the dispatcher to pass the goal's prompt to DispatchInfer
        instead of the legacy ``inference_prompt`` parameter.
        """
        with self._lock:
            if self._phase in (EpisodePhase.STARTING, EpisodePhase.RUNNING):
                return self._prompt
            return None

    def check_startup_deadline(self, monotonic_now_ns: int) -> str | None:
        """Check if the startup timeout deadline has been reached.

        Used by the control loop on every tick during STARTING to catch
        server unavailable, slow inference, and repeated observation_not_ready.
        Returns TERMINATION_OBSERVATION_STARTUP_TIMEOUT if exceeded, or None.
        """
        if self._phase is EpisodePhase.STARTING and not self.is_within_startup_deadline(monotonic_now_ns):
            return TERMINATION_OBSERVATION_STARTUP_TIMEOUT
        return None

    # ------------------------------------------------------------------
    # Preparation token lifecycle
    # ------------------------------------------------------------------

    def can_prepare(self) -> bool:
        """True if a new PreparePolicyEpisode may begin now.

        Rejected when a goal is active or pending (ACCEPTED_PENDING, STARTING,
        or RUNNING) or a prepare is already in progress (PREPARING). Allowed
        in IDLE, PREPARED (reprepare invalidates old token), TERMINAL, and
        FAULTED.
        """
        return self._phase in (
            EpisodePhase.IDLE,
            EpisodePhase.PREPARED,
            EpisodePhase.TERMINAL,
            EpisodePhase.FAULTED,
        )

    @property
    def current_token(self) -> int:
        """The current valid unconsumed token (>= 1), or 0 if none."""
        return self._current_token

    def begin_preparing(self) -> None:
        """Transition to PREPARING and invalidate any old unconsumed token.

        Thread-safe (acquires RLock).
        """
        with self._lock:
            if not self.can_prepare():
                raise RuntimeError(f"cannot begin preparing from phase {self._phase}")
            self._phase = EpisodePhase.PREPARING
            self._current_token = 0

    def complete_prepare(self) -> int:
        """Transition PREPARING -> PREPARED and issue a new token.

        Returns the new preparation_id (>= 1). The token is unconsumed until
        a RunPolicy goal atomically consumes it via ``try_accept_goal``.
        """
        with self._lock:
            if self._phase is not EpisodePhase.PREPARING:
                raise RuntimeError(f"complete_prepare requires PREPARING, got {self._phase}")
            self._token_counter += 1
            self._current_token = self._token_counter
            self._phase = EpisodePhase.PREPARED
            return self._current_token

    def fail_prepare(self) -> None:
        """Transition PREPARING -> IDLE on policy reset failure.

        Gate remains closed; no token; ready for another prepare attempt.
        """
        with self._lock:
            if self._phase is not EpisodePhase.PREPARING:
                raise RuntimeError(f"fail_prepare requires PREPARING, got {self._phase}")
            self._phase = EpisodePhase.IDLE
            self._current_token = 0

    def try_begin_preparing(self) -> bool:
        """Atomically check can_prepare and transition to PREPARING.

        Returns True if the transition succeeded, False if the current phase
        does not allow preparation (e.g., active goal). This is the
        thread-safe entry point for the prepare callback; it prevents the
        check-then-act race where a goal callback consumes the token between
        can_prepare() and begin_preparing().
        """
        with self._lock:
            if not self.can_prepare():
                return False
            self._phase = EpisodePhase.PREPARING
            self._current_token = 0
            return True

    # ------------------------------------------------------------------
    # Goal validation and acceptance
    # ------------------------------------------------------------------

    def validate_goal(self, spec: EpisodeGoalSpec) -> str | None:
        """Validate a RunPolicy goal spec.

        Returns None if valid, or a diagnostic error string if invalid. Does
        NOT consume the token or transition phase.
        """
        if self._phase is not EpisodePhase.PREPARED:
            return f"not prepared (phase={self._phase.value})"
        if self._current_token == 0:
            return "no valid preparation token"
        if spec.preparation_id != self._current_token:
            return f"preparation_id mismatch: goal={spec.preparation_id} current={self._current_token}"
        if spec.preparation_id <= 0:
            return "preparation_id must be positive"
        if not isinstance(spec.episode_id, int) or isinstance(spec.episode_id, bool):
            return "episode_id must be an integer"
        if spec.episode_id < 0:
            return "episode_id must be non-negative"
        if spec.initial_step_id != 0:
            return f"initial_step_id must be 0, got {spec.initial_step_id}"
        if not isinstance(spec.initial_observation_timestamp_ns, int) or isinstance(
            spec.initial_observation_timestamp_ns, bool
        ):
            return "initial_observation_timestamp_ns must be an integer"
        if spec.initial_observation_timestamp_ns <= 0:
            return "initial_observation_timestamp_ns must be positive"
        if not isinstance(spec.max_actions, int) or isinstance(spec.max_actions, bool):
            return "max_actions must be an integer"
        if spec.max_actions <= 0:
            return "max_actions must be positive"
        if not _is_finite_positive(spec.max_duration_sec):
            return "max_duration_sec must be positive and finite"
        if not _is_finite_positive(spec.startup_timeout_sec):
            return "startup_timeout_sec must be positive and finite"
        if not isinstance(spec.prompt, str) or not spec.prompt:
            return "prompt must be a non-empty string"
        return None

    def accept_goal(self, spec: EpisodeGoalSpec, monotonic_now_ns: int) -> None:
        """Atomically consume the token and enter STARTING.

        Thread-safe (acquires RLock). Legacy API; the dispatcher uses
        ``try_accept_goal`` + ``bind_goal`` for the two-phase model.
        """
        with self._lock:
            error = self.validate_goal(spec)
            if error is not None:
                raise RuntimeError(f"cannot accept goal: {error}")
            self._phase = EpisodePhase.STARTING
            self._current_token = 0  # consumed
            self._episode_id = spec.episode_id
            self._expected_step_id = spec.initial_step_id  # must be 0
            self._committed_count = 0
            self._max_actions = spec.max_actions
            self._max_duration_sec = float(spec.max_duration_sec)
            self._startup_timeout_sec = float(spec.startup_timeout_sec)
            self._initial_timestamp_ns = spec.initial_observation_timestamp_ns
            self._current_timestamp_ns = spec.initial_observation_timestamp_ns
            self._is_first_inference = True
            self._last_snapshot = None
            self._termination_reason = None
            self._prompt = spec.prompt
            self._goal_generation += 1
            self._startup_deadline_ns = monotonic_now_ns + int(self._startup_timeout_sec * 1_000_000_000)
            self._episode_deadline_ns = monotonic_now_ns + int(self._max_duration_sec * 1_000_000_000)

    def try_accept_goal(self, spec: EpisodeGoalSpec, monotonic_now_ns: int) -> str | None:
        """Atomically validate and accept a RunPolicy goal under lock.

        Sets phase to ACCEPTED_PENDING (NOT STARTING). The gate remains
        closed until ``bind_goal`` is called from handle_accepted_callback.
        This prevents the control loop from doing work before the goal handle
        is bound.

        Returns None if accepted (token consumed, phase=ACCEPTED_PENDING,
        identity stored, generation incremented). Returns a diagnostic error
        string if rejected.
        """
        with self._lock:
            error = self.validate_goal(spec)
            if error is not None:
                return error
            self._phase = EpisodePhase.ACCEPTED_PENDING
            self._current_token = 0  # consumed
            self._episode_id = spec.episode_id
            self._expected_step_id = spec.initial_step_id  # must be 0
            self._committed_count = 0
            self._max_actions = spec.max_actions
            self._max_duration_sec = float(spec.max_duration_sec)
            self._startup_timeout_sec = float(spec.startup_timeout_sec)
            self._initial_timestamp_ns = spec.initial_observation_timestamp_ns
            self._current_timestamp_ns = spec.initial_observation_timestamp_ns
            self._is_first_inference = True
            self._last_snapshot = None
            self._termination_reason = None
            self._prompt = spec.prompt
            self._goal_generation += 1
            # Deadlines are NOT set here; bind_goal sets them when the handle
            # is bound and the gate opens.
            return None

    def bind_goal(self, monotonic_now_ns: int) -> None:
        """Transition ACCEPTED_PENDING -> STARTING and set deadlines.

        Called from handle_accepted_callback after the goal handle is bound.
        The startup and episode deadlines start ticking from this moment.
        """
        with self._lock:
            if self._phase is not EpisodePhase.ACCEPTED_PENDING:
                raise RuntimeError(f"bind_goal requires ACCEPTED_PENDING, got {self._phase}")
            self._phase = EpisodePhase.STARTING
            self._startup_deadline_ns = monotonic_now_ns + int(self._startup_timeout_sec * 1_000_000_000)
            self._episode_deadline_ns = monotonic_now_ns + int(self._max_duration_sec * 1_000_000_000)

    def try_bind_goal_for_handle(self, generation: int, goal_uuid: bytes, monotonic_now_ns: int) -> bool:
        """Atomically bind a goal handle: ACCEPTED_PENDING -> STARTING.

        Checks generation match and phase before transitioning. Sets
        deadlines from now. Returns True if bound, False if generation
        mismatch or wrong phase (superseded).
        """
        with self._lock:
            if generation != self._goal_generation:
                return False
            if self._phase is not EpisodePhase.ACCEPTED_PENDING:
                return False
            self._phase = EpisodePhase.STARTING
            self._startup_deadline_ns = monotonic_now_ns + int(self._startup_timeout_sec * 1_000_000_000)
            self._episode_deadline_ns = monotonic_now_ns + int(self._max_duration_sec * 1_000_000_000)
            return True

    # ------------------------------------------------------------------
    # Running / inference
    # ------------------------------------------------------------------

    def on_inference_success(self) -> None:
        """Mark that an inference result was received successfully.

        On the first successful inference: transitions STARTING -> RUNNING and
        clears ``is_first_inference``. Subsequent calls are no-ops.
        """
        with self._lock:
            if self._phase is EpisodePhase.STARTING:
                self._phase = EpisodePhase.RUNNING
            self._is_first_inference = False

    @property
    def is_first_inference(self) -> bool:
        return self._is_first_inference

    def get_inference_timestamp(self) -> int | None:
        """Return the timestamp to use for the next inference request.

        First inference: the goal's initial observation timestamp.
        Subsequent: the latest committed step's response timestamp.
        Returns None if no identity is set.
        """
        if self._phase not in (EpisodePhase.STARTING, EpisodePhase.RUNNING):
            return None
        if self._is_first_inference:
            return self._initial_timestamp_ns
        return self._current_timestamp_ns

    def is_within_startup_deadline(self, monotonic_now_ns: int) -> bool:
        """True if the startup timeout has not yet been reached."""
        return monotonic_now_ns < self._startup_deadline_ns

    def is_within_episode_deadline(self, monotonic_now_ns: int) -> bool:
        """True if the max_duration deadline has not yet been reached."""
        return monotonic_now_ns < self._episode_deadline_ns

    # ------------------------------------------------------------------
    # Step commit and terminal evaluation
    # ------------------------------------------------------------------

    def validate_step_identity(
        self,
        episode_id: int | None,
        step_id: int | None,
    ) -> str | None:
        """Validate that a completion's identity matches the current episode.

        Returns None if valid, or a diagnostic error string.
        """
        if self._episode_id is not None and (episode_id is None or episode_id != self._episode_id):
            return f"episode_id mismatch: completion={episode_id} expected={self._episode_id}"
        if self._expected_step_id is not None and (step_id is None or step_id != self._expected_step_id):
            return f"step_id mismatch: completion={step_id} expected={self._expected_step_id}"
        return None

    def commit_step(
        self,
        data: StepCompletionData,
        monotonic_now_ns: int,
    ) -> str | None:
        """Commit a matching step, advance expected step, evaluate terminal.

        Thread-safe: acquires the lock to prevent concurrent commits from
        polluting the state. The caller MUST have already confirmed via
        ``validate_step_identity`` that the identity matches.
        """
        with self._lock:
            self._last_snapshot = data
            if data.observation_timestamp_ns is not None and data.observation_timestamp_ns > 0:
                self._current_timestamp_ns = data.observation_timestamp_ns
            if data.step_id is not None:
                self._expected_step_id = data.step_id + 1
            self._committed_count += 1
            return self._evaluate_terminal(data, monotonic_now_ns)

    def _evaluate_terminal(
        self,
        data: StepCompletionData,
        monotonic_now_ns: int,
    ) -> str | None:
        """Evaluate terminal priority. Returns reason or None."""
        # Check each condition in priority order.
        if data.has_success and data.success:
            return TERMINATION_TASK_SUCCESS
        if data.terminated:
            return TERMINATION_TERMINATED
        if data.truncated:
            return TERMINATION_TRUNCATED
        if self._committed_count >= self._max_actions:
            return TERMINATION_MAX_ACTIONS
        if not self.is_within_episode_deadline(monotonic_now_ns):
            return TERMINATION_MAX_DURATION
        return None

    def check_episode_deadline(self, monotonic_now_ns: int) -> str | None:
        """Check if the max_duration deadline has been reached.

        Used by the control loop before starting new inference/step work.
        Returns TERMINATION_MAX_DURATION if the deadline has been reached,
        or None otherwise.
        """
        if not self.is_within_episode_deadline(monotonic_now_ns):
            return TERMINATION_MAX_DURATION
        return None

    def get_last_snapshot(self) -> StepCompletionData | None:
        return self._last_snapshot

    # ------------------------------------------------------------------
    # Terminal / cancel / abort
    # ------------------------------------------------------------------

    def close_terminal(self, reason: str) -> None:
        """Close the episode gate with a normal terminal reason.

        Thread-safe. Clears identity/token/prompt; the episode is over.
        A new prepare is required to start another episode.
        """
        with self._lock:
            self._phase = EpisodePhase.TERMINAL
            self._termination_reason = reason
            self._episode_id = None
            self._expected_step_id = None
            self._current_token = 0
            self._prompt = None

    def close_faulted(self, reason: str) -> None:
        """Close the episode gate with a fault reason (ABORTED).

        Thread-safe.
        """
        with self._lock:
            self._phase = EpisodePhase.FAULTED
            self._termination_reason = reason
            self._episode_id = None
            self._expected_step_id = None
            self._current_token = 0
            self._prompt = None

    def cancel(self) -> None:
        """Cancel the active episode. Thread-safe."""
        with self._lock:
            self._close_terminal_unlocked(TERMINATION_CANCELED)

    def external_reset(self) -> None:
        """External legacy reset during active goal -> ABORTED external_reset. Thread-safe."""
        with self._lock:
            self._close_faulted_unlocked(TERMINATION_EXTERNAL_RESET)

    def _close_terminal_unlocked(self, reason: str) -> None:
        """Close terminal without acquiring lock (caller holds lock)."""
        self._phase = EpisodePhase.TERMINAL
        self._termination_reason = reason
        self._episode_id = None
        self._expected_step_id = None
        self._current_token = 0
        self._prompt = None

    def _close_faulted_unlocked(self, reason: str) -> None:
        """Close faulted without acquiring lock (caller holds lock)."""
        self._phase = EpisodePhase.FAULTED
        self._termination_reason = reason
        self._episode_id = None
        self._expected_step_id = None
        self._current_token = 0
        self._prompt = None

    # ------------------------------------------------------------------
    # Generation-aware atomic APIs (Per-goal concurrency handling)
    # ------------------------------------------------------------------

    def try_on_inference_success(self, generation: int) -> bool:
        """Atomically mark inference success if generation matches.

        Returns True if the transition was applied, False if the generation
        does not match the current goal generation (stale callback).
        """
        with self._lock:
            if generation != self._goal_generation:
                return False
            if self._phase is EpisodePhase.STARTING:
                self._phase = EpisodePhase.RUNNING
            self._is_first_inference = False
            return True

    def try_commit_step(
        self,
        generation: int,
        data: StepCompletionData,
        monotonic_now_ns: int,
    ) -> StepCommitOutcome:
        """Atomically validate identity + commit step + evaluate terminal.

        All in one critical section under lock. Returns StepCommitOutcome:
        - applied=True, terminal_reason=None or str: commit succeeded
        - applied=False: stale/gate-closed — no pop, no feedback, no count
        - applied=False, identity_mismatch=True: ABORT identity_mismatch, no feedback
        """
        with self._lock:
            if generation != self._goal_generation:
                return StepCommitOutcome(False, None, False)  # stale; drop
            if self._phase not in (EpisodePhase.STARTING, EpisodePhase.RUNNING):
                return StepCommitOutcome(False, None, False)  # gate closed; drop
            identity_err = self._validate_step_identity_unlocked(data.episode_id, data.step_id)
            if identity_err is not None:
                return StepCommitOutcome(False, None, True)  # identity mismatch
            self._last_snapshot = data
            if data.observation_timestamp_ns is not None and data.observation_timestamp_ns > 0:
                self._current_timestamp_ns = data.observation_timestamp_ns
            if data.step_id is not None:
                self._expected_step_id = data.step_id + 1
            self._committed_count += 1
            reason = self._evaluate_terminal(data, monotonic_now_ns)
            return StepCommitOutcome(True, reason, False)

    def try_check_startup_deadline(self, generation: int, monotonic_now_ns: int) -> str | None:
        """Check startup deadline for the matching generation."""
        with self._lock:
            if generation != self._goal_generation:
                return None
            if self._phase is EpisodePhase.STARTING and not self.is_within_startup_deadline(monotonic_now_ns):
                return TERMINATION_OBSERVATION_STARTUP_TIMEOUT
            return None

    def try_check_episode_deadline(self, generation: int, monotonic_now_ns: int) -> str | None:
        """Check max_duration deadline for the matching generation."""
        with self._lock:
            if generation != self._goal_generation:
                return None
            if not self.is_within_episode_deadline(monotonic_now_ns):
                return TERMINATION_MAX_DURATION
            return None

    def try_close_terminal(self, generation: int, reason: str) -> bool:
        """Atomically close terminal if generation matches and phase is active.

        Returns True if closed, False if generation mismatch or already
        terminal/faulted (double-close prevention).
        """
        with self._lock:
            if generation != self._goal_generation:
                return False
            if self._phase not in (
                EpisodePhase.ACCEPTED_PENDING,
                EpisodePhase.STARTING,
                EpisodePhase.RUNNING,
            ):
                return False  # already terminal/faulted
            self._phase = EpisodePhase.TERMINAL
            self._termination_reason = reason
            self._episode_id = None
            self._expected_step_id = None
            self._current_token = 0
            self._prompt = None
            return True

    def try_close_faulted(self, generation: int, reason: str) -> bool:
        """Atomically close faulted if generation matches and phase is active.

        Returns True if closed, False if generation mismatch or already
        terminal/faulted.
        """
        with self._lock:
            if generation != self._goal_generation:
                return False
            if self._phase not in (
                EpisodePhase.ACCEPTED_PENDING,
                EpisodePhase.STARTING,
                EpisodePhase.RUNNING,
            ):
                return False
            self._phase = EpisodePhase.FAULTED
            self._termination_reason = reason
            self._episode_id = None
            self._expected_step_id = None
            self._current_token = 0
            self._prompt = None
            return True

    def try_cancel(self, generation: int) -> bool:
        """Atomically cancel if generation matches. Returns True/False."""
        return self.try_close_terminal(generation, TERMINATION_CANCELED)

    def try_external_reset(self, generation: int) -> bool:
        """Atomically external reset if generation matches. Returns True/False."""
        return self.try_close_faulted(generation, TERMINATION_EXTERNAL_RESET)

    # ------------------------------------------------------------------
    # Atomic close-and-snapshot APIs (Write-once result handling correctness handling)
    # ------------------------------------------------------------------

    def try_close_terminal_with_result(self, generation: int, reason: str) -> CloseResult:
        """Atomically close terminal + snapshot result under one RLock critical section.

        Prevents the window where a new prepare/goal could overwrite the
        state machine's last_snapshot before the old goal's result is frozen.
        """
        with self._lock:
            if generation != self._goal_generation:
                return CloseResult(False, None)
            if self._phase not in (
                EpisodePhase.ACCEPTED_PENDING,
                EpisodePhase.STARTING,
                EpisodePhase.RUNNING,
            ):
                return CloseResult(False, None)
            self._phase = EpisodePhase.TERMINAL
            self._termination_reason = reason
            result = self._build_result_unlocked()
            self._episode_id = None
            self._expected_step_id = None
            self._current_token = 0
            self._prompt = None
            return CloseResult(True, result)

    def try_close_faulted_with_result(self, generation: int, reason: str) -> CloseResult:
        """Atomically close faulted + snapshot result under one RLock critical section."""
        with self._lock:
            if generation != self._goal_generation:
                return CloseResult(False, None)
            if self._phase not in (
                EpisodePhase.ACCEPTED_PENDING,
                EpisodePhase.STARTING,
                EpisodePhase.RUNNING,
            ):
                return CloseResult(False, None)
            self._phase = EpisodePhase.FAULTED
            self._termination_reason = reason
            result = self._build_result_unlocked()
            self._episode_id = None
            self._expected_step_id = None
            self._current_token = 0
            self._prompt = None
            return CloseResult(True, result)

    def try_cancel_with_result(self, generation: int) -> CloseResult:
        """Atomically cancel + snapshot result."""
        return self.try_close_terminal_with_result(generation, TERMINATION_CANCELED)

    def _validate_step_identity_unlocked(
        self,
        episode_id: int | None,
        step_id: int | None,
    ) -> str | None:
        """Identity check without acquiring lock (caller holds lock)."""
        if self._episode_id is not None and (episode_id is None or episode_id != self._episode_id):
            return f"episode_id mismatch: completion={episode_id} expected={self._episode_id}"
        if self._expected_step_id is not None and (step_id is None or step_id != self._expected_step_id):
            return f"step_id mismatch: completion={step_id} expected={self._expected_step_id}"
        return None

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def build_result(self) -> EpisodeResult:
        """Build the final RunPolicy result from the current state.

        Thread-safe. Must be called after the episode has reached TERMINAL
        or FAULTED. If no step was committed, presence flags are False,
        JSON is ``"{}"``, and latency is ``0.0``.
        """
        with self._lock:
            return self._build_result_unlocked()

    def _build_result_unlocked(self) -> EpisodeResult:
        """Build result without acquiring lock (caller holds lock).

        Write-once result handling: used by try_close_*_with_result to snapshot the result in the
        same critical section as the state transition, preventing a new
        prepare/goal from overwriting the snapshot before the old goal's
        result is frozen.
        """
        reason = self._termination_reason or ""
        snap = self._last_snapshot
        if snap is not None:
            return EpisodeResult(
                success=snap.success,
                has_success=snap.has_success,
                termination_reason=reason,
                episode_id=snap.episode_id if snap.episode_id is not None else 0,
                has_final_step=True,
                final_step_id=snap.step_id if snap.step_id is not None else 0,
                published_actions=self._committed_count,
                has_reward=snap.has_reward,
                reward=snap.reward,
                terminated=snap.terminated,
                truncated=snap.truncated,
                standard_metrics_json=snap.standard_metrics_json,
                native_metrics_json=snap.native_metrics_json,
                info_json=snap.info_json,
                round_trip_latency_ms=snap.round_trip_latency_ms,
            )
        return EpisodeResult(
            success=False,
            has_success=False,
            termination_reason=reason,
            episode_id=0,
            has_final_step=False,
            final_step_id=0,
            published_actions=self._committed_count,
            has_reward=False,
            reward=0.0,
            terminated=False,
            truncated=False,
            standard_metrics_json="{}",
            native_metrics_json="{}",
            info_json="{}",
            round_trip_latency_ms=0.0,
        )

    @property
    def is_succeeded_reason(self) -> bool:
        """True if the termination reason maps to ROS action SUCCEEDED."""
        return self._termination_reason in _SUCCEEDED_REASONS

    @property
    def is_canceled_reason(self) -> bool:
        return self._termination_reason == TERMINATION_CANCELED
