"""Dependency-light mock executor for the internal HRI action."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

ARM_SIDES = ("left", "right", "auto")
MAX_IMITATION_DURATION_SEC = 20.0
CANCEL_CLEANUP_TIMEOUT = "CANCEL_CLEANUP_TIMEOUT"


class PrimitiveStateUnknown(RuntimeError):
    """Raised when a primitive may still be executing and recovery is unsafe."""


@dataclass(frozen=True)
class AnimationPlan:
    animation_id: str
    waypoints: tuple[tuple[float, ...], ...]
    duration_sec: float = MAX_IMITATION_DURATION_SEC


@dataclass(frozen=True)
class MockGoal:
    arm_side: str
    imitation_duration_sec: float
    timeout_sec: float


@dataclass
class MockStatus:
    warmup_ready: bool = True
    warmup_state: str = "READY"
    warmup_attempts: int = 0
    operation_state: str = "IDLE"
    pose_state: str = "HOME"
    active_animation_id: str = ""
    completed_phases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MockResult:
    success: bool
    error_code: str
    message: str
    animation_id: str
    requested_duration_sec: float
    actual_duration_sec: float
    completed_phases: tuple[str, ...]


class AnimationPlayer(Protocol):
    def play(
        self,
        plan: AnimationPlan,
        duration_sec: float,
        *,
        feedback: Callable[[str, float, str], None],
        is_cancel_requested: Callable[[], bool],
        deadline: float,
    ) -> str:
        """Return COMPLETED, CANCELED, TIMEOUT, or FAILED."""


class MockAnimationPlayer:
    """Time-based mock player used by the internal HRI Action node."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep

    def play(
        self,
        plan: AnimationPlan,
        duration_sec: float,
        *,
        feedback: Callable[[str, float, str], None],
        is_cancel_requested: Callable[[], bool],
        deadline: float,
    ) -> str:
        started = self._clock()
        while True:
            if is_cancel_requested():
                return "CANCELED"
            now = self._clock()
            elapsed = now - started
            if elapsed >= duration_sec:
                feedback("mock_playback", 1.0, f"{plan.animation_id} completed")
                return "COMPLETED"
            if now >= deadline:
                return "TIMEOUT"
            progress = min(1.0, max(0.0, elapsed / duration_sec))
            feedback("mock_playback", progress, f"playing {plan.animation_id}")
            self._sleep(min(0.05, duration_sec - elapsed, max(0.0, deadline - now)))


class MotionRecorder(Protocol):
    def record(
        self,
        duration_sec: float,
        *,
        feedback: Callable[[str, float, str], None],
        is_cancel_requested: Callable[[], bool],
        deadline: float,
    ) -> str:
        """Return COMPLETED, CANCELED, TIMEOUT, or FAILED."""


class MockMotionRecorder:
    """Hold the ``start`` phase open for exactly the requested capture window.

    The window is defined by elapsed time rather than by a frame count because
    neither the camera nor the pose estimator delivers a steady frame rate,
    while the wall-clock length of the window is accurate. Downstream builds the
    imitation animation from that length, so it has to be the number the caller
    asked for.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep

    def record(
        self,
        duration_sec: float,
        *,
        feedback: Callable[[str, float, str], None],
        is_cancel_requested: Callable[[], bool],
        deadline: float,
    ) -> str:
        started = self._clock()
        while True:
            if is_cancel_requested():
                return "CANCELED"
            now = self._clock()
            elapsed = now - started
            if elapsed >= duration_sec:
                feedback("start", 1.0, f"captured {duration_sec:.1f}s of human motion")
                return "COMPLETED"
            if now >= deadline:
                return "TIMEOUT"
            progress = min(1.0, max(0.0, elapsed / duration_sec))
            feedback("start", progress, f"capturing human motion {elapsed:.1f}/{duration_sec:.1f}s")
            self._sleep(min(0.05, duration_sec - elapsed, max(0.0, deadline - now)))


def _normalized_motion_config(
    joint_names: Sequence[str],
    reset_positions: Mapping[str, float],
    joint_limits: Mapping[str, Mapping[str, float] | Sequence[float]],
) -> tuple[tuple[str, ...], dict[str, float], dict[str, tuple[float, float]]]:
    names = tuple(str(name).strip() for name in joint_names)
    if len(names) < 2 or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("mock motion requires at least two unique arm joint names")

    reset: dict[str, float] = {}
    limits: dict[str, tuple[float, float]] = {}
    for name in names:
        if name not in reset_positions or name not in joint_limits:
            raise ValueError(f"mock motion configuration is missing joint {name}")
        position = float(reset_positions[name])
        raw_limits = joint_limits[name]
        if isinstance(raw_limits, Mapping):
            lower = float(raw_limits["min"])
            upper = float(raw_limits["max"])
        else:
            lower, upper = (float(value) for value in raw_limits)
        if not all(math.isfinite(value) for value in (position, lower, upper)) or lower >= upper:
            raise ValueError(f"mock motion configuration for joint {name} is invalid")
        if not lower <= position <= upper:
            raise ValueError(f"reset position is outside joint {name} limits")
        reset[name] = position
        limits[name] = (lower, upper)
    return names, reset, limits


def _waypoints(
    joint_names: tuple[str, ...],
    joint_limits: Mapping[str, tuple[float, float]],
    *points: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    for point in points:
        if len(point) != len(joint_names) or any(not math.isfinite(value) for value in point):
            raise ValueError("animation waypoint must contain all arm joint values")
        for joint_name, value in zip(joint_names, point, strict=True):
            lower, upper = joint_limits[joint_name]
            if not lower <= value <= upper:
                raise ValueError(f"animation waypoint is outside joint {joint_name} limits")
    return tuple(points)


def build_mock_animations(
    joint_names: Sequence[str],
    reset_positions: Mapping[str, float],
    joint_limits: Mapping[str, Mapping[str, float] | Sequence[float]],
) -> dict[str, AnimationPlan]:
    """Build three distinct mock animations from the robot configuration SSOT."""
    names, reset, limits = _normalized_motion_config(joint_names, reset_positions, joint_limits)
    first_joint, second_joint = names[:2]
    home = tuple(reset[name] for name in names)

    def offset(**values: float) -> tuple[float, ...]:
        target = dict(reset)
        for name, delta in values.items():
            target[name] += delta
        return tuple(target[name] for name in names)

    left = offset(**{first_joint: -0.707, second_joint: 0.707})
    right = offset(**{first_joint: 0.707, second_joint: 0.707})
    auto_left = offset(**{first_joint: -0.5, second_joint: 0.5854})
    auto_right = offset(**{first_joint: 0.5, second_joint: 0.5854})
    return {
        "mock_left_v1": AnimationPlan("mock_left_v1", _waypoints(names, limits, home, left, home, right, home)),
        "mock_right_v1": AnimationPlan("mock_right_v1", _waypoints(names, limits, home, right, home, left, home)),
        "mock_auto_v1": AnimationPlan(
            "mock_auto_v1", _waypoints(names, limits, home, auto_right, auto_left, auto_right, home)
        ),
    }


class MockExecutor:
    """Execute one internal HRI mock goal at a time."""

    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        reset_positions: Mapping[str, float],
        joint_limits: Mapping[str, Mapping[str, float] | Sequence[float]],
        warmup_ready: bool = True,
        player: AnimationPlayer | None = None,
        recorder: MotionRecorder | None = None,
        prepare: Callable[[], bool] | None = None,
        recover_safe_pose: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._joint_names, self._reset_positions, self._joint_limits = _normalized_motion_config(
            joint_names, reset_positions, joint_limits
        )
        self.status = MockStatus(
            warmup_ready=warmup_ready,
            warmup_state="READY" if warmup_ready else "NOT_READY",
        )
        self._player = player or MockAnimationPlayer(clock=clock)
        self._recorder = recorder or MockMotionRecorder(clock=clock)
        self._prepare = prepare or (lambda: True)
        self._recover_safe_pose = recover_safe_pose or (lambda: True)
        self._clock = clock
        self._lock = threading.Lock()
        self._active = False
        self._animations = build_mock_animations(self._joint_names, self._reset_positions, self._joint_limits)

    @property
    def animations(self) -> dict[str, AnimationPlan]:
        return dict(self._animations)

    def warmup(self, initialize: Callable[[], bool] | None = None) -> bool:
        """Attempt the one startup initialization used by the mock runtime."""
        initializer = initialize or (lambda: True)
        with self._lock:
            if self.status.warmup_state == "READY":
                return True
            self.status.warmup_attempts += 1
            self.status.warmup_state = "WARMING_UP"
        try:
            ready = bool(initializer())
        except Exception:
            ready = False
        with self._lock:
            self.status.warmup_ready = ready
            self.status.warmup_state = "READY" if ready else "FAILED"
        return ready

    def set_warmup_ready(self, ready: bool) -> None:
        with self._lock:
            self.status.warmup_ready = bool(ready)
            self.status.warmup_state = "READY" if ready else "FAILED"

    def can_accept(self, goal: MockGoal) -> tuple[bool, str]:
        if goal.arm_side not in ARM_SIDES:
            return False, "arm_side must be left, right, or auto"
        if not math.isfinite(goal.imitation_duration_sec) or goal.imitation_duration_sec <= 0.0:
            return False, "imitation_duration_sec must be finite and positive"
        if not math.isfinite(goal.timeout_sec) or goal.timeout_sec <= 0.0:
            return False, "timeout_sec must be finite and positive"
        with self._lock:
            if self._active or self.status.operation_state != "IDLE":
                return False, "imitate_human_motion is busy"
            if not self.status.warmup_ready:
                return False, "warmup is not ready"
            if self.status.pose_state not in {"HOME", "NOT_READY"}:
                return False, "safe pose is not confirmed"
        return True, ""

    def _result(
        self,
        *,
        success: bool,
        error_code: str,
        message: str,
        plan: AnimationPlan,
        requested_duration: float,
        actual_duration: float,
        phases: list[str],
    ) -> MockResult:
        return MockResult(
            success,
            error_code,
            message,
            plan.animation_id,
            requested_duration,
            actual_duration,
            tuple(phases),
        )

    def execute(
        self,
        goal: MockGoal,
        *,
        feedback: Callable[[str, float, str], None] | None = None,
        is_cancel_requested: Callable[[], bool] | None = None,
        player: AnimationPlayer | None = None,
        recorder: MotionRecorder | None = None,
        prepare: Callable[[], bool] | None = None,
        recover_safe_pose: Callable[[], bool] | None = None,
    ) -> MockResult:
        accepted, reason = self.can_accept(goal)
        if not accepted:
            if "warmup" in reason:
                error_code = "WARMUP_NOT_READY"
            elif "busy" in reason:
                error_code = "BUSY"
            elif "safe pose" in reason:
                error_code = "RESET_NOT_CONFIRMED"
            else:
                error_code = "INVALID_GOAL"
            return MockResult(False, error_code, reason, "", goal.imitation_duration_sec, 0.0, ())

        feedback = feedback or (lambda _phase, _progress, _detail: None)
        is_cancel_requested = is_cancel_requested or (lambda: False)
        active_player = player or self._player
        active_recorder = recorder or self._recorder
        active_prepare = prepare or self._prepare
        active_recover = recover_safe_pose or self._recover_safe_pose
        phases: list[str] = []
        plan = self._animations[f"mock_{goal.arm_side}_v1"]
        actual_duration = min(goal.imitation_duration_sec, MAX_IMITATION_DURATION_SEC)
        deadline = self._clock() + goal.timeout_sec
        result: MockResult | None = None
        reset_ok = False
        playback_started_at: float | None = None

        def playback_duration() -> float:
            if playback_started_at is None:
                return 0.0
            return min(actual_duration, max(0.0, self._clock() - playback_started_at))

        def phase(name: str, detail: str) -> None:
            if not phases or phases[-1] != name:
                phases.append(name)
            with self._lock:
                self.status.completed_phases = list(phases)
                self.status.operation_state = name.upper()
                self.status.active_animation_id = plan.animation_id
            feedback(name, 0.0, detail)

        with self._lock:
            if self._active or self.status.operation_state != "IDLE":
                return MockResult(
                    False,
                    "BUSY",
                    "imitate_human_motion is busy",
                    plan.animation_id,
                    goal.imitation_duration_sec,
                    0.0,
                    (),
                )
            if not self.status.warmup_ready:
                return MockResult(
                    False,
                    "WARMUP_NOT_READY",
                    "warmup is not ready",
                    plan.animation_id,
                    goal.imitation_duration_sec,
                    0.0,
                    (),
                )
            if self.status.pose_state not in {"HOME", "NOT_READY"}:
                return MockResult(
                    False,
                    "RESET_NOT_CONFIRMED",
                    "safe pose is not confirmed",
                    plan.animation_id,
                    goal.imitation_duration_sec,
                    0.0,
                    (),
                )
            self._active = True
            self.status.operation_state = "PREPARING"
            self.status.pose_state = "UNKNOWN"
            self.status.active_animation_id = plan.animation_id

        try:
            phase("prepare", "Preparing for human-motion imitation")
            if is_cancel_requested():
                result = self._result(
                    success=False,
                    error_code="CANCELED",
                    message="imitation cancelled during prepare",
                    plan=plan,
                    requested_duration=goal.imitation_duration_sec,
                    actual_duration=0.0,
                    phases=phases,
                )
            elif self._clock() >= deadline:
                result = self._result(
                    success=False,
                    error_code="SKILL_TIMEOUT",
                    message="imitation timeout during prepare",
                    plan=plan,
                    requested_duration=goal.imitation_duration_sec,
                    actual_duration=0.0,
                    phases=phases,
                )
            else:
                prepared = bool(active_prepare())
                if is_cancel_requested():
                    result = self._result(
                        success=False,
                        error_code="CANCELED",
                        message="imitation cancelled during prepare",
                        plan=plan,
                        requested_duration=goal.imitation_duration_sec,
                        actual_duration=0.0,
                        phases=phases,
                    )
                elif self._clock() >= deadline:
                    result = self._result(
                        success=False,
                        error_code="SKILL_TIMEOUT",
                        message="imitation timeout during prepare",
                        plan=plan,
                        requested_duration=goal.imitation_duration_sec,
                        actual_duration=0.0,
                        phases=phases,
                    )
                elif not prepared:
                    result = self._result(
                        success=False,
                        error_code="PREPARE_FAILED",
                        message="prepare failed",
                        plan=plan,
                        requested_duration=goal.imitation_duration_sec,
                        actual_duration=0.0,
                        phases=phases,
                    )
                else:
                    # The arm is parked at the imitation start pose and stays
                    # there for the whole window: this is the data-collection
                    # phase, and any arm motion during it would both drag the
                    # wrist camera off the person and pollute what is being
                    # recorded. ``imitation_duration_sec`` measures exactly this
                    # window -- not prepare, not playback, not reset -- because
                    # the animation is built from its wall-clock length rather
                    # than from a frame count the camera cannot guarantee.
                    phase("start", f"Capturing human motion for {actual_duration:.1f}s")
                    capture_started_at = self._clock()
                    capture_outcome = active_recorder.record(
                        actual_duration,
                        feedback=feedback,
                        is_cancel_requested=is_cancel_requested,
                        deadline=deadline,
                    )
                    captured_duration = max(0.0, self._clock() - capture_started_at)
                    capture_error, capture_message = {
                        "CANCELED": ("CANCELED", "imitation cancelled during capture"),
                        "TIMEOUT": ("SKILL_TIMEOUT", "imitation timeout during capture"),
                        "COMPLETED": ("", ""),
                    }.get(capture_outcome, ("CAPTURE_FAILED", "human-motion capture failed"))
                    if capture_error:
                        result = self._result(
                            success=False,
                            error_code=capture_error,
                            message=capture_message,
                            plan=plan,
                            requested_duration=goal.imitation_duration_sec,
                            actual_duration=0.0,
                            phases=phases,
                        )
                    else:
                        # Stand-in for the animation that will eventually be
                        # solved from the captured PEAR output; until that
                        # mapping exists a preset plan is played back instead,
                        # over the same span that was just recorded.
                        phase("mock_playback", f"Executing {plan.animation_id}")
                        playback_started_at = self._clock()
                        outcome = active_player.play(
                            plan,
                            actual_duration,
                            feedback=feedback,
                            is_cancel_requested=is_cancel_requested,
                            deadline=deadline,
                        )
                        error_code, message = {
                            "CANCELED": ("CANCELED", "imitation cancelled during playback"),
                            "TIMEOUT": ("SKILL_TIMEOUT", "imitation timeout during playback"),
                            "UNKNOWN": (CANCEL_CLEANUP_TIMEOUT, "primitive execution state is unknown"),
                            "COMPLETED": ("", f"Mock imitation completed; captured {captured_duration:.2f}s"),
                        }.get(outcome, ("MOCK_PLAYBACK_FAILED", "mock playback failed"))
                        result = self._result(
                            success=not error_code,
                            error_code=error_code,
                            message=message,
                            plan=plan,
                            requested_duration=goal.imitation_duration_sec,
                            actual_duration=playback_duration(),
                            phases=phases,
                        )
        except PrimitiveStateUnknown as exc:
            result = self._result(
                success=False,
                error_code=CANCEL_CLEANUP_TIMEOUT,
                message=str(exc),
                plan=plan,
                requested_duration=goal.imitation_duration_sec,
                actual_duration=playback_duration(),
                phases=phases,
            )
        except Exception as exc:
            result = self._result(
                success=False,
                error_code="MOCK_PLAYBACK_FAILED",
                message=str(exc),
                plan=plan,
                requested_duration=goal.imitation_duration_sec,
                actual_duration=playback_duration(),
                phases=phases,
            )
        finally:
            recovery_safe = result is None or result.error_code != CANCEL_CLEANUP_TIMEOUT
            recovery_unknown = False
            if recovery_safe:
                phase("reset", "Returning to safe pose")
                try:
                    reset_ok = bool(active_recover())
                except PrimitiveStateUnknown:
                    recovery_unknown = True
                    reset_ok = False
                except Exception:
                    reset_ok = False
            if result is None:
                result = self._result(
                    success=False,
                    error_code="MOCK_PLAYBACK_FAILED",
                    message="mock executor returned no result",
                    plan=plan,
                    requested_duration=goal.imitation_duration_sec,
                    actual_duration=playback_duration(),
                    phases=phases,
                )
            if recovery_unknown:
                result = self._result(
                    success=False,
                    error_code=CANCEL_CLEANUP_TIMEOUT,
                    message="reset primitive execution state is unknown",
                    plan=plan,
                    requested_duration=goal.imitation_duration_sec,
                    actual_duration=result.actual_duration_sec,
                    phases=phases,
                )
            elif recovery_safe and not reset_ok:
                result = self._result(
                    success=False,
                    error_code="RESET_FAILED",
                    message="recover_safe_pose failed",
                    plan=plan,
                    requested_duration=goal.imitation_duration_sec,
                    actual_duration=result.actual_duration_sec,
                    phases=phases,
                )
            if result.completed_phases != tuple(phases):
                result = self._result(
                    success=result.success and (reset_ok or not recovery_safe) and not recovery_unknown,
                    error_code=(
                        CANCEL_CLEANUP_TIMEOUT
                        if recovery_unknown
                        else result.error_code
                        if reset_ok or not recovery_safe
                        else "RESET_FAILED"
                    ),
                    message=(
                        "reset primitive execution state is unknown"
                        if recovery_unknown
                        else result.message
                        if reset_ok or not recovery_safe
                        else "recover_safe_pose failed"
                    ),
                    plan=plan,
                    requested_duration=goal.imitation_duration_sec,
                    actual_duration=result.actual_duration_sec,
                    phases=phases,
                )
            with self._lock:
                self._active = False
                self.status.operation_state = "IDLE"
                self.status.pose_state = "NOT_READY" if recovery_safe and reset_ok else "UNKNOWN"
                self.status.active_animation_id = ""
                self.status.completed_phases = list(phases)
        return result
