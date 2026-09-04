"""Pure decision policy for fixed-trigger sound orientation.

This module deliberately has no ROS dependency. The ROS node should adapt
SpeechDirection, Gateway status, and SkillCommand results to this policy.
"""

from __future__ import annotations

import math
import unicodedata
from collections import deque
from dataclasses import dataclass
from enum import Enum

_MAX_CONSUMED_DIRECTION_KEYS = 256


class OrientationState(str, Enum):
    """Lifecycle states owned by the sound-orientation coordinator."""

    IDLE_LISTENING = "idle_listening"
    WAITING_FOR_DIRECTION = "waiting_for_direction"
    DISPATCHING = "dispatching"
    TURNING = "turning"
    COOLDOWN = "cooldown"
    FAULT_UNKNOWN = "fault_unknown"


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

    def __post_init__(self) -> None:
        phrases = tuple(_normalize_text(value) for value in self.trigger_phrases if _normalize_text(value))
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
        object.__setattr__(self, "trigger_phrases", phrases)
        object.__setattr__(self, "direction_frame", self.direction_frame.strip())


@dataclass(frozen=True)
class DirectionSample:
    """A normalized view of one SpeechDirection message."""

    seq_id: int
    stamp_sec: float
    azimuth_rad: float
    frame_id: str = "base_link"

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
    """Finite state machine for fixed-trigger, one-shot orientation."""

    def __init__(self, config: OrientationPolicyConfig | None = None) -> None:
        self.config = config or OrientationPolicyConfig()
        self.state = OrientationState.IDLE_LISTENING
        self._latest_direction: DirectionSample | None = None
        self._consumed_direction_keys: set[tuple[int, float]] = set()
        self._consumed_direction_order: deque[tuple[int, float]] = deque()
        self._pending_trigger: tuple[str, float] | None = None
        self._cooldown_until = 0.0
        self._active_request: TurnRequest | None = None

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
        if self.state in {OrientationState.TURNING, OrientationState.DISPATCHING, OrientationState.COOLDOWN}:
            return PolicyDecision(DecisionKind.IGNORED, "NOT_LISTENING")
        if self.state == OrientationState.FAULT_UNKNOWN:
            return PolicyDecision(DecisionKind.IGNORED, "FAULT_UNKNOWN")

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

        normalized = _normalize_text(text)
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
            return PolicyDecision(DecisionKind.STATE_CHANGED, "FAULT_UNKNOWN")
        self._active_request = None
        self._cooldown_until = now_sec + self.config.cooldown_sec
        self.state = OrientationState.COOLDOWN
        return PolicyDecision(DecisionKind.STATE_CHANGED, "ACTION_TERMINAL")

    def tick(self, *, now_sec: float) -> PolicyDecision:
        """Advance wait/cooldown timers without generating a motion request."""

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

    def reset_fault(self) -> None:
        """Clear the fail-closed state after an external recovery decision."""

        if self.state == OrientationState.FAULT_UNKNOWN:
            self._active_request = None
            self._pending_trigger = None
            self._latest_direction = None
            self.state = OrientationState.IDLE_LISTENING

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


def _normalize_text(text: str) -> str:
    """Normalize ASR text for exact phrase matching, not intent parsing."""

    chars = [char.casefold() for char in text if not char.isspace()]
    while chars and unicodedata.category(chars[0]).startswith("P"):
        chars.pop(0)
    while chars and unicodedata.category(chars[-1]).startswith("P"):
        chars.pop()
    return "".join(chars)
