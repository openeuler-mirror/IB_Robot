"""Runtime mode model: named controller activation sets with validated transitions.

Implements the robot-runtime-contract "Mode service with validated
transitions" requirement. A mode maps to the set of controllers (or, for a
vendor runtime, the control tier) that must be active. Exactly one mode is
active at a time; enforcement of "one command source per joint group" is
delegated to the execution stack (controller_manager activation), not
re-implemented here.

Streaming commands observed on a channel while it is not active are counted
per channel so the runtime status can surface them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModeSpec:
    """One declared mode."""

    name: str
    controllers: tuple[str, ...] = ()
    #: Whether trajectory execution (motion.move_to_*) is permitted in this mode.
    allows_trajectory: bool = False
    #: Whether streaming position commands are accepted in this mode.
    allows_stream: bool = False
    #: Whether base velocity commands are accepted in this mode.
    allows_base: bool = False


@dataclass
class ModeModelConfig:
    modes: dict[str, ModeSpec] = field(default_factory=dict)
    transitions: dict[str, set[str]] = field(default_factory=dict)
    initial_mode: str = "idle"

    @classmethod
    def from_profile(cls, profile: dict) -> ModeModelConfig:
        """Build from the ``modes`` section of a runtime profile.

        Expected shape::

            modes:
              initial: idle
              idle: {controllers: [], transitions: [stream, trajectory]}
              stream: {controllers: [arm_position_controller], allows_stream: true,
                       transitions: [idle, trajectory]}
              trajectory: {controllers: [arm_trajectory_controller],
                           allows_trajectory: true, transitions: [idle, stream]}
        """
        section = dict(profile.get("modes") or {})
        initial = str(section.pop("initial", "idle"))
        modes: dict[str, ModeSpec] = {}
        transitions: dict[str, set[str]] = {}
        for name, raw in section.items():
            raw = raw or {}
            modes[str(name)] = ModeSpec(
                name=str(name),
                controllers=tuple(str(c) for c in raw.get("controllers", [])),
                allows_trajectory=bool(raw.get("allows_trajectory", False)),
                allows_stream=bool(raw.get("allows_stream", False)),
                allows_base=bool(raw.get("allows_base", False)),
            )
            transitions[str(name)] = {str(t) for t in raw.get("transitions", [])}
        if not modes:
            raise ValueError("runtime profile 'modes' must declare at least one mode")
        if initial not in modes:
            raise ValueError(f"runtime profile modes.initial {initial!r} is not a declared mode")
        for name, targets in transitions.items():
            unknown = sorted(t for t in targets if t not in modes)
            if unknown:
                raise ValueError(f"mode {name!r} declares transitions to undeclared modes: {unknown}")
        return cls(modes=modes, transitions=transitions, initial_mode=initial)


@dataclass
class ModeDecision:
    allowed: bool
    mode: str
    reason: str = ""


class ModeModel:
    """Thread-safe current-mode holder with transition validation."""

    def __init__(self, config: ModeModelConfig):
        self._lock = threading.Lock()
        self._config = config
        self._mode = config.initial_mode
        self._rejections: dict[str, int] = {}

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def spec(self, name: str | None = None) -> ModeSpec:
        with self._lock:
            return self._config.modes[name or self._mode]

    def declared_modes(self) -> list[str]:
        return sorted(self._config.modes)

    def valid_transitions(self) -> set[str]:
        with self._lock:
            return set(self._config.transitions.get(self._mode, set()))

    def can_switch(self, target: str) -> ModeDecision:
        with self._lock:
            if target == self._mode:
                return ModeDecision(True, self._mode, "already active")
            if target not in self._config.modes:
                return ModeDecision(
                    False, self._mode, f"mode {target!r} is not declared; declared: {sorted(self._config.modes)}"
                )
            allowed = self._config.transitions.get(self._mode, set())
            if target not in allowed:
                return ModeDecision(
                    False, self._mode, f"transition {self._mode!r} -> {target!r} not allowed; valid: {sorted(allowed)}"
                )
            return ModeDecision(True, target)

    def commit(self, target: str) -> None:
        """Record a completed switch (call after the execution stack confirmed it)."""
        with self._lock:
            self._mode = target

    def note_rejected(self, channel: str) -> int:
        """Count a streaming command observed while ``channel`` was not active."""
        with self._lock:
            self._rejections[channel] = self._rejections.get(channel, 0) + 1
            return self._rejections[channel]

    def rejections(self) -> dict[str, int]:
        with self._lock:
            return dict(self._rejections)
