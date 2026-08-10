"""Independent scheduler registry.

Mirrors the executor registry pattern: exact string matching, no aliasing,
no case-folding, no fallback, no dynamic import. The production registry
contains exactly ``continuous`` and ``wait_for_feedback``.

completion-aware executor contract boundary:
- no Python entry-point discovery, no directory scanning;
- no benchmark runtime or sim backend adapter imports;
- the registry does not select executors or vice versa.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import DispatchScheduler
from .continuous import ContinuousScheduler
from .step_barrier import StepBarrierScheduler

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    pass


class SchedulerRegistryError(Exception):
    """Base class for scheduler registry failures."""


class SchedulerAlreadyRegisteredError(SchedulerRegistryError):
    """Raised when a mode is registered more than once."""


class SchedulerNotFoundError(SchedulerRegistryError):
    """Raised when a requested mode is not registered."""


class SchedulerTypeMismatchError(SchedulerRegistryError):
    """Raised when a factory returns a value that is not the requested scheduler."""


# Module-level production registry. A plain dict keeps semantics explicit and
# avoids any implicit ordering, aliasing or fallback behaviour.
_REGISTRY: dict[str, Callable[[dict], DispatchScheduler]] = {}


def _validate_scheduler_mode(scheduler_mode: object) -> str:
    """Return the validated mode string unchanged, or raise.

    Type matching is exact: no lower/casefold, no alias, no fallback. Non-string,
    empty string and pure whitespace values fail immediately. The returned key is
    the original string, not a stripped or normalised copy.
    """
    if not isinstance(scheduler_mode, str):
        raise SchedulerRegistryError(f"scheduler mode must be a non-empty string, got {type(scheduler_mode).__name__}")
    stripped = scheduler_mode.strip()
    if not stripped:
        raise SchedulerRegistryError("scheduler mode must not be empty or whitespace")
    return scheduler_mode


def register_scheduler(
    scheduler_mode: str,
    factory: Callable[[dict], DispatchScheduler],
) -> None:
    """Register a factory for a scheduler mode.

    Args:
        scheduler_mode: Exact SSOT scheduler mode string. Must be a non-empty
            string. Matching is exact; no aliasing, case-folding or fallback.
        factory: Callable ``(config) -> DispatchScheduler``.

    Raises:
        SchedulerRegistryError: If mode is non-string, empty or whitespace.
        SchedulerAlreadyRegisteredError: If mode is already registered.
    """
    key = _validate_scheduler_mode(scheduler_mode)
    if key in _REGISTRY:
        raise SchedulerAlreadyRegisteredError(f"scheduler mode already registered: {key!r}")
    _REGISTRY[key] = factory


def get_scheduler_factory(scheduler_mode: str) -> Callable[[dict], DispatchScheduler]:
    """Return the registered factory for a mode.

    Raises:
        SchedulerNotFoundError: If mode is not registered. The error message
            contains both the requested mode and the available registered modes.
    """
    key = _validate_scheduler_mode(scheduler_mode)
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        available = sorted(_REGISTRY.keys())
        raise SchedulerNotFoundError(f"unknown scheduler mode {key!r}; available modes: {available}") from exc


def create_scheduler(scheduler_mode: str, config: dict) -> DispatchScheduler:
    """Create a scheduler instance via the registered factory.

    The factory return value must be a ``DispatchScheduler`` instance whose
    ``scheduler_mode`` exactly equals the requested mode. Both conditions are
    verified to prevent silent contract drift.

    Raises:
        SchedulerNotFoundError: If mode is not registered.
        SchedulerTypeMismatchError: If the factory returns a non-DispatchScheduler
            or a scheduler whose ``scheduler_mode`` differs from the request.
    """
    factory = get_scheduler_factory(scheduler_mode)
    scheduler = factory(config)
    if not isinstance(scheduler, DispatchScheduler):
        raise SchedulerTypeMismatchError(
            f"factory for {scheduler_mode!r} returned {type(scheduler).__name__} which is not a DispatchScheduler"
        )
    actual_mode = scheduler.scheduler_mode
    if actual_mode != scheduler_mode:
        raise SchedulerTypeMismatchError(f"factory for {scheduler_mode!r} returned scheduler with mode {actual_mode!r}")
    return scheduler


def registered_scheduler_modes() -> tuple[str, ...]:
    """Return the registered scheduler modes as an exact tuple (no ordering guarantee)."""
    return tuple(_REGISTRY.keys())


# Deterministic production registration of the two completion-aware executor contract scheduler modes.


def _create_continuous_scheduler(config: dict) -> DispatchScheduler:
    return ContinuousScheduler(watermark=int(config["watermark"]))


def _create_step_barrier_scheduler(config: dict) -> DispatchScheduler:
    return StepBarrierScheduler(
        watermark=int(config["watermark"]),
        execution_timeout_sec=float(config["execution_timeout_sec"]),
    )


register_scheduler("continuous", _create_continuous_scheduler)
register_scheduler("wait_for_feedback", _create_step_barrier_scheduler)
