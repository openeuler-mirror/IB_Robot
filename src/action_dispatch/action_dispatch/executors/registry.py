"""Independent executor registry.

This registry is intentionally separate from the benchmark adapter and sim
backend adapter registries. It only knows how to map an SSOT
``executor.type`` string to a factory that constructs an ``ActionExecutor``.

executor registry contract boundary:
- production registry contains exactly ``topic``;
- no Python entry-point discovery, no directory scanning, no dynamic import;
- no aliasing, no case-folding, no fallback;
- no benchmark runtime or sim backend adapter imports.

production benchmark wiring boundary:
- production registry contains exactly ``topic`` and ``benchmark``;
- ``benchmark`` factory only constructs ``BenchmarkStepExecutor`` via a sibling
  import from ``.benchmark``; the factory does NOT call ``initialize()`` and
  does NOT send any request;
- no LIBERO/MuJoCo/adapter/runtime/sim backend imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import ActionExecutor
from .benchmark import BenchmarkStepExecutor
from .topic import TopicExecutor

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from rclpy.node import Node


class ExecutorRegistryError(Exception):
    """Base class for executor registry failures."""


class ExecutorAlreadyRegisteredError(ExecutorRegistryError):
    """Raised when a type is registered more than once."""


class ExecutorNotFoundError(ExecutorRegistryError):
    """Raised when a requested type is not registered."""


class ExecutorTypeMismatchError(ExecutorRegistryError):
    """Raised when a factory returns a value that is not the requested executor."""


# Module-level production registry. A plain dict keeps semantics explicit and
# avoids any implicit ordering, aliasing or fallback behaviour.
_REGISTRY: dict[str, Callable[[Node, dict], ActionExecutor]] = {}


def _validate_executor_type(executor_type: object) -> str:
    """Return the validated type string unchanged, or raise.

    Type matching is exact: no lower/casefold, no alias, no fallback. Non-string,
    empty string and pure whitespace values fail immediately. The returned key is
    the original string, not a stripped or normalised copy.
    """
    if not isinstance(executor_type, str):
        raise ExecutorRegistryError(f"executor type must be a non-empty string, got {type(executor_type).__name__}")
    stripped = executor_type.strip()
    if not stripped:
        raise ExecutorRegistryError("executor type must not be empty or whitespace")
    # Return the original string unchanged (no lower/casefold/strip on the key).
    return executor_type


def register_executor(
    executor_type: str,
    factory: Callable[[Node, dict], ActionExecutor],
) -> None:
    """Register a factory for an executor type.

    Args:
        executor_type: Exact SSOT executor type string. Must be a non-empty
            string. Matching is exact; no aliasing, case-folding or fallback.
        factory: Callable ``(node, config) -> ActionExecutor``.

    Raises:
        ExecutorRegistryError: If type is non-string, empty or whitespace.
        ExecutorAlreadyRegisteredError: If type is already registered.
    """
    key = _validate_executor_type(executor_type)
    if key in _REGISTRY:
        raise ExecutorAlreadyRegisteredError(f"executor type already registered: {key!r}")
    _REGISTRY[key] = factory


def get_executor_factory(executor_type: str) -> Callable[[Node, dict], ActionExecutor]:
    """Return the registered factory for a type.

    Raises:
        ExecutorNotFoundError: If type is not registered. The error message
            contains both the requested type and the available registered types.
    """
    key = _validate_executor_type(executor_type)
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        available = sorted(_REGISTRY.keys())
        raise ExecutorNotFoundError(f"unknown executor type {key!r}; available types: {available}") from exc


def create_executor(executor_type: str, node: Node, config: dict) -> ActionExecutor:
    """Create an executor instance via the registered factory.

    The factory return value must be an ``ActionExecutor`` instance whose
    ``executor_type`` exactly equals the requested type. Both conditions are
    verified to prevent silent contract drift.

    Raises:
        ExecutorNotFoundError: If type is not registered.
        ExecutorTypeMismatchError: If the factory returns a non-ActionExecutor
            or an executor whose ``executor_type`` differs from the request.
    """
    factory = get_executor_factory(executor_type)
    executor = factory(node, config)
    if not isinstance(executor, ActionExecutor):
        raise ExecutorTypeMismatchError(
            f"factory for {executor_type!r} returned {type(executor).__name__} which is not an ActionExecutor"
        )
    actual_type = executor.executor_type
    if actual_type != executor_type:
        raise ExecutorTypeMismatchError(f"factory for {executor_type!r} returned executor with type {actual_type!r}")
    return executor


def registered_executor_types() -> tuple[str, ...]:
    """Return the registered executor types as an exact tuple (no ordering guarantee)."""
    return tuple(_REGISTRY.keys())


# Deterministic production registration of the only executor registry contract executor.


def _create_topic_executor(node: Node, config: dict) -> ActionExecutor:
    return TopicExecutor(node, config)


def _create_benchmark_executor(node: Node, config: dict) -> ActionExecutor:
    return BenchmarkStepExecutor(node, config)


register_executor("topic", _create_topic_executor)
register_executor("benchmark", _create_benchmark_executor)
