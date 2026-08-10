"""Production plugin descriptor and entry-point registry for benchmark adapters.

benchmark plugin contract scope: defines the :class:`BenchmarkPlugin` descriptor, the
``BENCHMARK_ADAPTER_ENTRY_POINT_GROUP`` and ``BENCHMARK_PLUGIN_API_VERSION``
constants, and discovery/load API. It does NOT create environment instances,
native reporters, MuJoCo contexts or GPU resources during discovery or load.

The registry is strictly separate from ``SimBackendAdapter``. It only
enumerates the fixed Python entry-point group ``ibrobot.benchmark_adapters``.
No fallback, no case-insensitive guessing, no second dynamic-import mechanism.

Tests monkeypatch ``importlib.metadata.entry_points`` with synthetic
entry-point objects. No ``benchmark_fake`` package, test entry-point
registration or fixture installation into the product is allowed.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark_runtime.adapter import BenchmarkAdapter
    from benchmark_runtime.models import BenchmarkEnvironmentConfig
    from benchmark_runtime.native_report import NativeReportExporter


BENCHMARK_PLUGIN_API_VERSION = 1
BENCHMARK_ADAPTER_ENTRY_POINT_GROUP = "ibrobot.benchmark_adapters"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class BenchmarkRegistryError(Exception):
    """Base class for all benchmark registry errors."""


class BenchmarkPluginNotFoundError(BenchmarkRegistryError):
    """Raised when a requested plugin name is not registered."""


class BenchmarkPluginDuplicateError(BenchmarkRegistryError):
    """Raised when two entry points share the same name."""


class BenchmarkPluginLoadError(BenchmarkRegistryError):
    """Raised when an entry point cannot be loaded or its factory fails."""


class BenchmarkPluginDescriptorError(BenchmarkRegistryError):
    """Raised when a loaded plugin descriptor fails contract validation."""


# --------------------------------------------------------------------------- #
# Plugin descriptor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BenchmarkPlugin:
    """Stable plugin descriptor returned by a ``create_plugin()`` factory.

    ``validate_environment_config`` parses a raw SSOT mapping into a
    :class:`BenchmarkEnvironmentConfig`. ``create_adapter`` and
    ``create_native_reporter`` are zero-argument factories called only by the
    environment node, never during discovery.

    Entry-point targets must be a ``create_plugin() -> BenchmarkPlugin``
    factory, not an adapter instance. Discovery/load must not call
    ``create_adapter``, ``create_native_reporter`` or the config validator.
    """

    name: str
    api_version: int
    validate_environment_config: Callable[[Mapping[str, Any]], BenchmarkEnvironmentConfig]
    create_adapter: Callable[[], BenchmarkAdapter]
    create_native_reporter: Callable[[], NativeReportExporter]


# --------------------------------------------------------------------------- #
# Discovery & load
# --------------------------------------------------------------------------- #


def _iter_group_entry_points(group: str) -> list[Any]:
    """Return raw entry points registered under *group*.

    Tests may monkeypatch ``importlib.metadata.entry_points`` or this
    function to supply synthetic entry-point objects.
    """
    eps = importlib.metadata.entry_points()
    select_fn = getattr(eps, "select", None)
    if select_fn is not None:
        return list(select_fn(group=group))
    # Fallback for dict-style API (Python < 3.10).
    return list(eps.get(group, []))  # type: ignore[union-attr]


def discover_plugins() -> list[str]:
    """Enumerate available benchmark plugin names without loading factories.

    Returns a sorted list of entry-point names registered under
    :data:`BENCHMARK_ADAPTER_ENTRY_POINT_GROUP`. Duplicate names are a hard
    error. This function does NOT call ``create_plugin``, ``create_adapter``,
    ``create_native_reporter`` or the config validator.
    """
    eps = _iter_group_entry_points(BENCHMARK_ADAPTER_ENTRY_POINT_GROUP)
    seen: set[str] = set()
    for ep in eps:
        name = ep.name
        if name in seen:
            raise BenchmarkPluginDuplicateError(
                f"Duplicate benchmark adapter entry point name '{name}' in group "
                f"'{BENCHMARK_ADAPTER_ENTRY_POINT_GROUP}'"
            )
        seen.add(name)
    return sorted(seen)


def load_plugin(name: str) -> BenchmarkPlugin:
    """Load and validate a benchmark plugin by exact name match.

    Name matching is exact — no case folding, no fallback. The entry-point
    factory (``create_plugin``) is called and the returned descriptor is
    validated for type, name consistency, API version and callable presence.
    This function does NOT call ``create_adapter``,
    ``create_native_reporter`` or the config validator.
    """
    if not isinstance(name, str) or not name.strip():
        raise BenchmarkRegistryError("Plugin name must be a non-empty string")

    eps = _iter_group_entry_points(BENCHMARK_ADAPTER_ENTRY_POINT_GROUP)
    matches = [ep for ep in eps if ep.name == name]
    if not matches:
        discovered = sorted(ep.name for ep in eps)
        raise BenchmarkPluginNotFoundError(
            f"Benchmark adapter '{name}' not found in group "
            f"'{BENCHMARK_ADAPTER_ENTRY_POINT_GROUP}'. Discovered names: {discovered}"
        )
    if len(matches) > 1:
        raise BenchmarkPluginDuplicateError(
            f"Multiple entry points named '{name}' in group '{BENCHMARK_ADAPTER_ENTRY_POINT_GROUP}'"
        )

    ep = matches[0]
    try:
        factory = ep.load()
    except Exception as exc:
        raise BenchmarkPluginLoadError(
            f"Failed to load benchmark adapter entry point name='{ep.name}' value='{ep.value}': {exc!r}"
        ) from exc

    if not callable(factory):
        raise BenchmarkPluginDescriptorError(
            f"Entry point name='{ep.name}' value='{ep.value}' did not return a callable: {factory!r}"
        )

    try:
        descriptor = factory()
    except Exception as exc:
        raise BenchmarkPluginLoadError(
            f"Entry point name='{ep.name}' value='{ep.value}' factory call failed: {exc!r}"
        ) from exc

    if not isinstance(descriptor, BenchmarkPlugin):
        raise BenchmarkPluginDescriptorError(
            f"Entry point name='{ep.name}' returned {type(descriptor).__name__}, expected BenchmarkPlugin"
        )

    if descriptor.name != ep.name:
        raise BenchmarkPluginDescriptorError(
            f"Plugin descriptor name '{descriptor.name}' does not match entry point name '{ep.name}'"
        )

    if descriptor.api_version != BENCHMARK_PLUGIN_API_VERSION:
        raise BenchmarkPluginDescriptorError(
            f"Plugin '{descriptor.name}' API version {descriptor.api_version} does not match "
            f"expected {BENCHMARK_PLUGIN_API_VERSION}"
        )

    for attr in ("validate_environment_config", "create_adapter", "create_native_reporter"):
        if not callable(getattr(descriptor, attr, None)):
            raise BenchmarkPluginDescriptorError(f"Plugin '{descriptor.name}' is missing callable attribute '{attr}'")

    return descriptor
