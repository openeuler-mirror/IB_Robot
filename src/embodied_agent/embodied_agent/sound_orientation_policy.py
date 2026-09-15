"""Pure decision policy for keyword and periodic sound orientation.

This module deliberately has no ROS dependency. The ROS node should adapt
SpeechDirection, Gateway status, and SkillCommand results to this policy.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

from embodied_common.text_normalization import normalize_trigger_text

_MAX_CONSUMED_DIRECTION_KEYS = 256
_MAX_PERIODIC_SEGMENTS = 256


class OrientationState(str, Enum):
    """Lifecycle states owned by the sound-orientation coordinator."""

    IDLE_LISTENING = "idle_listening"
    WAITING_FOR_DIRECTION = "waiting_for_direction"
    DISPATCHING = "dispatching"
    TURNING = "turning"
    COOLDOWN = "cooldown"
    FAULT_UNKNOWN = "fault_unknown"


class SessionState(str, Enum):
    """Session-level switch for periodic sound following.

    INACTIVE is the safe default: directions are cached but no nav_turn is
    dispatched.  ACTIVE schedules periodic turns.  SHUTTING_DOWN stops new
    dispatches while an in-flight nav_turn converges to a definite terminal.
    """

    INACTIVE = "inactive"
    ACTIVE = "active"
    SHUTTING_DOWN = "shutting_down"


class DecisionKind(str, Enum):
    """Result of processing one policy input."""

    IGNORED = "ignored"
    WAITING = "waiting"
    DISPATCH = "dispatch"
    DROPPED = "dropped"
    STATE_CHANGED = "state_changed"


@dataclass(frozen=True)
class OrientationPolicyConfig:
    """Validated policy values supplied by robot_config."""

    trigger_phrases: tuple[str, ...] = ("转向我",)
    direction_frame: str = "base_link"
    deadband_deg: float = 15.0
    max_direction_age_sec: float = 1.3
    direction_wait_sec: float = 0.5
    cooldown_sec: float = 1.5
    max_turn_deg: float = 180.0
    mode: str = "keyword"
    periodic_interval_sec: float = 10.0
    default_active: bool = False

    def __post_init__(self) -> None:
        phrases_list = []
        for value in self.trigger_phrases:
            normalized = normalize_trigger_text(value)
            if normalized:
                phrases_list.append(normalized)
        phrases = tuple(phrases_list)
        if not phrases:
            raise ValueError("trigger_phrases must contain a non-empty phrase")
        if not self.direction_frame.strip():
            raise ValueError("direction_frame must be non-empty")
        for name in (
            "deadband_deg",
            "max_direction_age_sec",
            "direction_wait_sec",
            "cooldown_sec",
            "max_turn_deg",
            "periodic_interval_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.deadband_deg >= 180.0:
            raise ValueError("deadband_deg must be less than 180 degrees")
        if self.max_direction_age_sec <= 0.0:
            raise ValueError("max_direction_age_sec must be positive")
        if self.max_turn_deg <= 0.0:
            raise ValueError("max_turn_deg must be positive")
        if self.periodic_interval_sec <= 0.0:
            raise ValueError("periodic_interval_sec must be positive")
        if self.mode not in {"keyword", "periodic"}:
            raise ValueError("mode must be keyword or periodic")
        object.__setattr__(self, "trigger_phrases", phrases)
        object.__setattr__(self, "direction_frame", self.direction_frame.strip())


@dataclass(frozen=True)
class DirectionSample:
    """A normalized view of one SpeechDirection message."""

    seq_id: int
    stamp_sec: float
    azimuth_rad: float
    frame_id: str = "base_link"
    segment_id: int = 0
    direction_type: str = ""

    @property
    def event_key(self) -> tuple[int, float]:
        """Return the stable identity used for consumer-side de-duplication."""

        return self.seq_id, self.stamp_sec


@dataclass(frozen=True)
class GatewaySnapshot:
    """Read-only fields needed for early admission filtering."""

    control_plane_ready: bool = True
    motion_authorized: bool = True
    busy: bool = False
    capability_ready: bool = True
    active_control_mode: str = "base_navigation"
    required_control_mode: str = "base_navigation"
    control_mode_switching_enabled: bool = True

    def admission_reason(self) -> str:
        """Return the first reason this snapshot cannot accept nav_turn."""

        if not self.control_plane_ready:
            return "CONTROL_PLANE_NOT_READY"
        if not self.motion_authorized:
            return "MOTION_NOT_AUTHORIZED"
        if self.busy:
            return "SKILL_BUSY"
        if not self.capability_ready:
            return "CAPABILITY_NOT_READY"
        if not self.control_mode_switching_enabled and self.active_control_mode != self.required_control_mode:
            return "CONTROL_MODE_MISMATCH"
        return ""


@dataclass(frozen=True)
class TurnRequest:
    """Action-independent nav_turn request produced by the policy."""

    direction: str
    degree: float
    direction_event_key: tuple[int, float]
    trigger_text: str


@dataclass(frozen=True)
class PolicyDecision:
    """Observable result of one policy transition."""

    kind: DecisionKind
    reason: str = ""
    request: TurnRequest | None = None


class SoundOrientationPolicy:
    """Finite state machine for keyword and session-gated orientation."""

    def __init__(self, config: OrientationPolicyConfig | None = None) -> None:
        self.config = config or OrientationPolicyConfig()
        self.state = OrientationState.IDLE_LISTENING
        self._session_state = (
            SessionState.ACTIVE
            if self.config.mode == "periodic" and self.config.default_active
            else SessionState.INACTIVE
        )
        self._latest_direction: DirectionSample | None = None
        self._consumed_direction_keys: set[tuple[int, float]] = set()
        self._consumed_direction_order: deque[tuple[int, float]] = deque()
        self._pending_trigger: tuple[str, float] | None = None
        self._cooldown_until = 0.0
        self._active_request: TurnRequest | None = None
        self._periodic_next_tick = 0.0
        self._periodic_latest_by_segment: dict[int, DirectionSample] = {}
        self._periodic_consumed_segments: set[int] = set()
        self._periodic_consumed_order: deque[int] = deque()

    @property
    def session_state(self) -> SessionState:
        return self._session_state

    @property
    def active_request(self) -> TurnRequest | None:
        """Return the request currently owned by this coordinator."""

        return self._active_request

    def handle_direction(
        self,
        sample: DirectionSample,
        *,
        now_sec: float,
        gateway: GatewaySnapshot | None = None,
    ) -> PolicyDecision:
        """Cache a direction or complete a pending fixed-trigger request."""

        if not self._valid_direction(sample, now_sec):
            return PolicyDecision(DecisionKind.IGNORED, "INVALID_OR_STALE_DIRECTION")
        if sample.event_key in self._consumed_direction_keys:
            return PolicyDecision(DecisionKind.IGNORED, "DUPLICATE_DIRECTION")
        if self.state == OrientationState.FAULT_UNKNOWN:
            return PolicyDecision(DecisionKind.IGNORED, "FAULT_UNKNOWN")

        if self.config.mode == "periodic":
            segment_id = sample.segment_id or sample.seq_id
            if segment_id in self._periodic_consumed_segments:
                return PolicyDecision(DecisionKind.IGNORED, "DUPLICATE_SEGMENT")
            previous = self._periodic_latest_by_segment.get(segment_id)
            if previous is None or self._direction_rank(sample) >= self._direction_rank(previous):
                self._periodic_latest_by_segment[segment_id] = sample
            self._prune_periodic_segments(now_sec)
            return PolicyDecision(DecisionKind.IGNORED, "DIRECTION_CACHED")

        if self.state in {OrientationState.TURNING, OrientationState.DISPATCHING, OrientationState.COOLDOWN}:
            return PolicyDecision(DecisionKind.IGNORED, "NOT_LISTENING")

        self._latest_direction = sample
        if self.state != OrientationState.WAITING_FOR_DIRECTION or self._pending_trigger is None:
            return PolicyDecision(DecisionKind.IGNORED, "DIRECTION_CACHED")

        trigger_text, deadline = self._pending_trigger
        if now_sec > deadline:
            self._pending_trigger = None
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.DROPPED, "DIRECTION_WAIT_TIMEOUT")
        if gateway is None:
            return PolicyDecision(DecisionKind.WAITING, "WAITING_FOR_GATEWAY")
        snapshot = gateway
        reason = snapshot.admission_reason()
        if reason:
            self._pending_trigger = None
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.DROPPED, reason)
        return self._prepare_dispatch(trigger_text, sample)

    def handle_text(
        self,
        text: str,
        *,
        now_sec: float,
        gateway: GatewaySnapshot | None = None,
    ) -> PolicyDecision:
        """Process one ASR final text using exact fixed-phrase matching."""

        if self.config.mode == "periodic":
            return PolicyDecision(DecisionKind.IGNORED, "PERIODIC_MODE_IGNORES_ASR")

        normalized = normalize_trigger_text(text)
        if normalized not in self.config.trigger_phrases:
            return PolicyDecision(DecisionKind.IGNORED, "NON_EXACT_TRIGGER")
        if self.state == OrientationState.FAULT_UNKNOWN:
            return PolicyDecision(DecisionKind.DROPPED, "FAULT_UNKNOWN")
        if self.state in {OrientationState.TURNING, OrientationState.DISPATCHING}:
            return PolicyDecision(DecisionKind.DROPPED, "NOT_IDLE")
        if self.state == OrientationState.WAITING_FOR_DIRECTION:
            return PolicyDecision(DecisionKind.IGNORED, "TRIGGER_PENDING")
        if self.state == OrientationState.COOLDOWN:
            return PolicyDecision(DecisionKind.DROPPED, "COOLDOWN")

        direction = self._fresh_direction(now_sec)
        if direction is None:
            self._pending_trigger = (normalized, now_sec + self.config.direction_wait_sec)
            self.state = OrientationState.WAITING_FOR_DIRECTION
            return PolicyDecision(DecisionKind.WAITING, "WAITING_FOR_DIRECTION")
        if gateway is None:
            self._pending_trigger = (normalized, now_sec + self.config.direction_wait_sec)
            self.state = OrientationState.WAITING_FOR_DIRECTION
            return PolicyDecision(DecisionKind.WAITING, "WAITING_FOR_GATEWAY")
        reason = gateway.admission_reason()
        if reason:
            return PolicyDecision(DecisionKind.DROPPED, reason)
        return self._prepare_dispatch(normalized, direction)

    def try_dispatch(self, *, now_sec: float, gateway: GatewaySnapshot) -> PolicyDecision:
        """Attempt a pending trigger after an authoritative status read."""

        if self.state != OrientationState.WAITING_FOR_DIRECTION or self._pending_trigger is None:
            return PolicyDecision(DecisionKind.IGNORED, "NO_PENDING_TRIGGER")
        _trigger_text, deadline = self._pending_trigger
        if now_sec > deadline:
            self._pending_trigger = None
            self._latest_direction = None
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.DROPPED, "DIRECTION_WAIT_TIMEOUT")
        reason = gateway.admission_reason()
        if reason:
            self._pending_trigger = None
            self._latest_direction = None
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.DROPPED, reason)
        direction = self._fresh_direction(now_sec)
        if direction is None:
            return PolicyDecision(DecisionKind.WAITING, "WAITING_FOR_DIRECTION")
        return self._prepare_dispatch(_trigger_text, direction)

    def mark_action_submitted(self) -> None:
        """Move a prepared request into the in-flight state after Action submit."""

        if self.state != OrientationState.DISPATCHING or self._active_request is None:
            raise RuntimeError("no dispatch is pending")
        self.state = OrientationState.TURNING

    def complete_action(self, *, now_sec: float, terminal_known: bool) -> PolicyDecision:
        """Handle a definite or unknown Action terminal result.

        A known failure is terminal and is not retried. An unknown result is a
        fail-closed condition because the physical stop state is uncertain.
        """

        if self.state not in {OrientationState.DISPATCHING, OrientationState.TURNING}:
            return PolicyDecision(DecisionKind.IGNORED, "NO_ACTIVE_ACTION")
        self._pending_trigger = None
        self._latest_direction = None
        if not terminal_known:
            self.state = OrientationState.FAULT_UNKNOWN
            if self.session_state is SessionState.SHUTTING_DOWN:
                self._session_state = SessionState.INACTIVE
            return PolicyDecision(DecisionKind.STATE_CHANGED, "FAULT_UNKNOWN")
        self._active_request = None
        self._cooldown_until = now_sec + self.config.cooldown_sec
        self.state = OrientationState.COOLDOWN
        if self.session_state is SessionState.SHUTTING_DOWN:
            self._session_state = SessionState.INACTIVE
            return PolicyDecision(DecisionKind.STATE_CHANGED, "SOUND_FOLLOWING_DEACTIVATED")
        return PolicyDecision(DecisionKind.STATE_CHANGED, "ACTION_TERMINAL")

    def tick(self, *, now_sec: float) -> PolicyDecision:
        """Advance wait/cooldown timers without generating a motion request."""

        # The wait-timeout branch is keyword-only: periodic mode never enters
        # WAITING_FOR_DIRECTION or records a pending trigger, so it falls
        # through to the shared cooldown transition below.
        if (
            self.state == OrientationState.WAITING_FOR_DIRECTION
            and self._pending_trigger is not None
            and now_sec >= self._pending_trigger[1]
        ):
            self._pending_trigger = None
            self._latest_direction = None
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.DROPPED, "DIRECTION_WAIT_TIMEOUT")
        if self.state == OrientationState.COOLDOWN and now_sec >= self._cooldown_until:
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.STATE_CHANGED, "COOLDOWN_COMPLETE")
        return PolicyDecision(DecisionKind.IGNORED, "NO_TRANSITION")

    def periodic_tick(self, *, now_sec: float, gateway: GatewaySnapshot | None = None) -> PolicyDecision:
        """Dispatch at most one fresh, previously unused voice segment per interval."""
        if self.config.mode != "periodic":
            return PolicyDecision(DecisionKind.IGNORED, "NOT_PERIODIC_MODE")
        self._prune_periodic_segments(now_sec)
        if self.session_state is not SessionState.ACTIVE:
            return PolicyDecision(DecisionKind.IGNORED, "SOUND_FOLLOWING_INACTIVE")
        if self.state == OrientationState.COOLDOWN and now_sec >= self._cooldown_until:
            self.state = OrientationState.IDLE_LISTENING
        if self.state == OrientationState.COOLDOWN:
            return PolicyDecision(DecisionKind.IGNORED, "COOLDOWN")
        if self.state in {
            OrientationState.TURNING,
            OrientationState.DISPATCHING,
            OrientationState.FAULT_UNKNOWN,
        }:
            return PolicyDecision(DecisionKind.IGNORED, "NOT_LISTENING")
        if now_sec < self._periodic_next_tick:
            return PolicyDecision(DecisionKind.IGNORED, "PERIODIC_WAIT")
        candidates = [
            sample
            for segment_id, sample in self._periodic_latest_by_segment.items()
            if segment_id not in self._periodic_consumed_segments
            and 0.0 <= now_sec - sample.stamp_sec <= self.config.max_direction_age_sec
        ]
        if not candidates:
            self._periodic_next_tick = now_sec + self.config.periodic_interval_sec
            return PolicyDecision(DecisionKind.IGNORED, "NO_FRESH_DIRECTION")
        sample = max(candidates, key=lambda item: item.stamp_sec)
        if gateway is None:
            return PolicyDecision(DecisionKind.WAITING, "WAITING_FOR_GATEWAY")
        self._periodic_next_tick = now_sec + self.config.periodic_interval_sec
        reason = gateway.admission_reason()
        if reason:
            self._remember_consumed_segment(sample.segment_id or sample.seq_id)
            return PolicyDecision(DecisionKind.DROPPED, reason)
        decision = self._prepare_dispatch("periodic_sound_orientation", sample)
        if decision.kind is DecisionKind.DISPATCH:
            self._remember_consumed_segment(sample.segment_id or sample.seq_id)
        return decision

    def reset_fault(self) -> None:
        """Clear the fail-closed state after an external recovery decision."""

        if self.state == OrientationState.FAULT_UNKNOWN:
            self._active_request = None
        self._pending_trigger = None
        self._latest_direction = None
        if self.state == OrientationState.FAULT_UNKNOWN:
            self.state = OrientationState.IDLE_LISTENING

    def activate_following(self) -> PolicyDecision:
        """Enable periodic following and discard directions collected before activation."""

        if self.config.mode != "periodic":
            return PolicyDecision(DecisionKind.DROPPED, "NOT_PERIODIC_MODE")
        if self.state == OrientationState.FAULT_UNKNOWN:
            return PolicyDecision(DecisionKind.DROPPED, "FAULT_UNKNOWN")
        if self.session_state is SessionState.SHUTTING_DOWN:
            return PolicyDecision(DecisionKind.DROPPED, "SOUND_FOLLOWING_SHUTTING_DOWN")
        if self.session_state is SessionState.ACTIVE:
            return PolicyDecision(DecisionKind.IGNORED, "ALREADY_ACTIVE")
        self._periodic_latest_by_segment.clear()
        self._periodic_consumed_segments.clear()
        self._periodic_consumed_order.clear()
        self._periodic_next_tick = 0.0
        self._session_state = SessionState.ACTIVE
        return PolicyDecision(DecisionKind.STATE_CHANGED, "SOUND_FOLLOWING_ACTIVATED")

    def deactivate_following(self) -> PolicyDecision:
        """Stop new periodic dispatches and let any in-flight turn converge."""

        if self.config.mode != "periodic":
            return PolicyDecision(DecisionKind.DROPPED, "NOT_PERIODIC_MODE")
        if self.session_state is SessionState.INACTIVE:
            return PolicyDecision(DecisionKind.IGNORED, "ALREADY_INACTIVE")
        if self.state in {OrientationState.DISPATCHING, OrientationState.TURNING}:
            self._session_state = SessionState.SHUTTING_DOWN
            return PolicyDecision(DecisionKind.STATE_CHANGED, "SOUND_FOLLOWING_SHUTTING_DOWN")
        self._periodic_latest_by_segment.clear()
        self._periodic_consumed_segments.clear()
        self._periodic_consumed_order.clear()
        self._session_state = SessionState.INACTIVE
        return PolicyDecision(DecisionKind.STATE_CHANGED, "SOUND_FOLLOWING_DEACTIVATED")

    @staticmethod
    def _direction_rank(sample: DirectionSample) -> int:
        return {"voice_begin": 1, "mid_long_seg": 2, "seg_end": 3}.get(sample.direction_type, 0)

    def _prune_periodic_segments(self, now_sec: float) -> None:
        expired = [
            segment_id
            for segment_id, sample in self._periodic_latest_by_segment.items()
            if now_sec - sample.stamp_sec > self.config.max_direction_age_sec
        ]
        for segment_id in expired:
            self._periodic_latest_by_segment.pop(segment_id, None)
        while len(self._periodic_latest_by_segment) > _MAX_PERIODIC_SEGMENTS:
            oldest = min(
                self._periodic_latest_by_segment, key=lambda key: self._periodic_latest_by_segment[key].stamp_sec
            )
            del self._periodic_latest_by_segment[oldest]

    def _remember_consumed_segment(self, segment_id: int) -> None:
        # Keep de-duplication beyond sample expiry: a long utterance may emit its
        # seg_end much later than the voice_begin that already caused a turn.
        if segment_id not in self._periodic_consumed_segments:
            self._periodic_consumed_segments.add(segment_id)
            self._periodic_consumed_order.append(segment_id)
        while len(self._periodic_consumed_order) > _MAX_PERIODIC_SEGMENTS:
            self._periodic_consumed_segments.discard(self._periodic_consumed_order.popleft())

    def drop_pending(self) -> None:
        """Drop a pending trigger without declaring a physical action unknown."""

        if self.state == OrientationState.WAITING_FOR_DIRECTION:
            self._pending_trigger = None
            self._latest_direction = None
            self.state = OrientationState.IDLE_LISTENING

    def _prepare_dispatch(self, trigger_text: str, sample: DirectionSample) -> PolicyDecision:
        magnitude_deg = math.degrees(abs(sample.azimuth_rad))
        if magnitude_deg < self.config.deadband_deg:
            self._pending_trigger = None
            self._latest_direction = None
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.DROPPED, "WITHIN_DEADBAND")
        if magnitude_deg > self.config.max_turn_deg:
            self._pending_trigger = None
            self._latest_direction = None
            self.state = OrientationState.IDLE_LISTENING
            return PolicyDecision(DecisionKind.DROPPED, "TURN_EXCEEDS_LIMIT")

        request = TurnRequest(
            direction="left" if sample.azimuth_rad > 0.0 else "right",
            degree=magnitude_deg,
            direction_event_key=sample.event_key,
            trigger_text=trigger_text,
        )
        self._remember_consumed_direction(sample.event_key)
        self._pending_trigger = None
        self._active_request = request
        self.state = OrientationState.DISPATCHING
        return PolicyDecision(DecisionKind.DISPATCH, "READY_TO_DISPATCH", request)

    def _fresh_direction(self, now_sec: float) -> DirectionSample | None:
        sample = self._latest_direction
        if sample is None:
            return None
        age = now_sec - sample.stamp_sec
        if age < 0.0 or age > self.config.max_direction_age_sec:
            return None
        if sample.event_key in self._consumed_direction_keys:
            return None
        return sample

    def _valid_direction(self, sample: DirectionSample, now_sec: float) -> bool:
        return (
            sample.frame_id == self.config.direction_frame
            and math.isfinite(sample.azimuth_rad)
            and math.isfinite(sample.stamp_sec)
            and 0.0 <= now_sec - sample.stamp_sec <= self.config.max_direction_age_sec
            and abs(sample.azimuth_rad) <= math.pi
        )

    def _remember_consumed_direction(self, event_key: tuple[int, float]) -> None:
        if event_key in self._consumed_direction_keys:
            return
        self._consumed_direction_keys.add(event_key)
        self._consumed_direction_order.append(event_key)
        while len(self._consumed_direction_order) > _MAX_CONSUMED_DIRECTION_KEYS:
            self._consumed_direction_keys.discard(self._consumed_direction_order.popleft())
