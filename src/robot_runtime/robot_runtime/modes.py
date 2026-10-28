"""Runtime mode model: named controller activation sets with validated transitions.

Implements the robot-runtime-contract "Mode service with validated
transitions" requirement. A mode maps to the set of controllers (or, for a
vendor runtime, the control tier) that must be active. Exactly one mode is
active at a time; enforcement of "one command source per joint group" is
delegated to the execution stack (controller_manager activation), not
re-implemented here.

Streaming commands observed on a channel while it is not active are counted
per channel so the runtime status can surface them.

A mode may additionally declare ``controllers_by_platform`` (D3.1): a
per-simulation-platform controller set that REPLACES ``controllers`` when the
runtime selects that platform. This is the explicit "the simulator needs a
different controller combination" declaration (e.g. MuJoCo never releases a
velocity actuator's last target, so the base controller stays active in
``idle`` there and the base node is the gate). Keys are limited to physics
platforms — a hardware/SDK execution has no platform and no override.
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
    #: Per-physics-platform controller overrides (D3.1); the effective set for
    #: a platform replaces ``controllers`` entirely (an empty list included).
    controllers_by_platform: dict[str, tuple[str, ...]] = field(default_factory=dict)


def resolve_mode_controllers(
    controllers: tuple[str, ...],
    by_platform: dict[str, tuple[str, ...]] | None,
    sim_platform: str | None,
    *,
    mode_name: str = "<mode>",
) -> tuple[str, ...]:
    """One mode's effective controller set on ``sim_platform`` (D3.1).

    The override replaces ``controllers`` completely when the table declares
    the selected platform — an empty override list stays empty (no
    ``override or controllers`` resurrection) — and ``controllers`` is kept
    for every other platform, for no platform at all (hardware), and when the
    mode declares no table. The table's keys must be physics platforms and
    its values lists of names; anything else is a profile error naming the
    mode and the key.
    """
    if not by_platform:
        return controllers
    from robot_runtime.profile import PHYSICS_SIM_PLATFORMS

    for platform, override in by_platform.items():
        if platform not in PHYSICS_SIM_PLATFORMS:
            raise ValueError(
                f"mode {mode_name!r} controllers_by_platform key {platform!r} is not a physics platform "
                f"(supported: {list(PHYSICS_SIM_PLATFORMS)}); hardware and SDK executions take 'controllers'"
            )
        if not isinstance(override, list | tuple):
            raise ValueError(
                f"mode {mode_name!r} controllers_by_platform[{platform!r}] must be a list of controller names"
            )
    if sim_platform is not None and sim_platform in by_platform:
        return tuple(str(controller) for controller in by_platform[sim_platform])
    return controllers


def mode_controller_table(profile: dict, sim_platform: str | None = None) -> dict[str, tuple[str, ...]]:
    """Every mode's effective controller set on ``sim_platform`` (D3.1).

    The single resolver every consumer of mode controllers goes through (the
    spawner list, the facade's mode switch, its stop-path command-controller
    set, its health check): validating the whole ``controllers_by_platform``
    table even for an unselected platform, so a bad key fails at load, not the
    first time its platform is selected.
    """
    table: dict[str, tuple[str, ...]] = {}
    for name, raw in (profile.get("modes") or {}).items():
        if name == "initial" or raw is None:
            continue
        declared = tuple(str(controller) for controller in (raw or {}).get("controllers", []))
        by_platform = (raw or {}).get("controllers_by_platform") or {}
        if not isinstance(by_platform, dict):
            raise ValueError(f"mode {name!r} controllers_by_platform must be a mapping of platform to controller list")
        table[str(name)] = resolve_mode_controllers(declared, by_platform, sim_platform, mode_name=str(name))
    return table


@dataclass
class ModeModelConfig:
    modes: dict[str, ModeSpec] = field(default_factory=dict)
    transitions: dict[str, set[str]] = field(default_factory=dict)
    initial_mode: str = "idle"

    @classmethod
    def from_profile(cls, profile: dict, sim_platform: str | None = None) -> ModeModelConfig:
        """Build from the ``modes`` section of a runtime profile.

        Expected shape::

            modes:
              initial: idle
              idle: {controllers: [], transitions: [stream, trajectory]}
              stream: {controllers: [arm_position_controller], allows_stream: true,
                       transitions: [idle, trajectory]}
              trajectory: {controllers: [arm_trajectory_controller],
                           allows_trajectory: true, transitions: [idle, stream]}

        ``sim_platform`` selects the per-platform controller overrides
        (``controllers_by_platform``, D3.1): every :class:`ModeSpec` then
        carries the effective set for that platform, so all consumers (mode
        switch, readiness check, the stop-path command-controller set) see
        one consistent table.
        """
        section = dict(profile.get("modes") or {})
        initial = str(section.pop("initial", "idle"))
        modes: dict[str, ModeSpec] = {}
        transitions: dict[str, set[str]] = {}
        for name, raw in section.items():
            raw = raw or {}
            by_platform_raw = raw.get("controllers_by_platform") or {}
            if not isinstance(by_platform_raw, dict):
                raise ValueError(
                    f"mode {name!r} controllers_by_platform must be a mapping of platform to controller list"
                )
            by_platform = {
                str(platform): tuple(str(controller) for controller in override)
                for platform, override in by_platform_raw.items()
            }
            modes[str(name)] = ModeSpec(
                name=str(name),
                controllers=resolve_mode_controllers(
                    tuple(str(c) for c in raw.get("controllers", [])),
                    by_platform_raw,
                    sim_platform,
                    mode_name=str(name),
                ),
                controllers_by_platform=by_platform,
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
