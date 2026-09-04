"""Pure-Python benchmark endpoint resolver (benchmark endpoint resolver).

Given the normalized robot SSOT ``benchmark`` mapping this module derives the
deterministic ROS namespace and ``reset`` / ``step`` / ``status`` names that
every benchmark instance (LIBERO today, Meta-World / future benchmarks later)
must use. The resolver is intentionally pure-Python:

- imports only the standard library;
- never imports ``rclpy``, ``launch`` or ROS messages;
- never imports ``benchmark_runtime`` or LIBERO / MuJoCo;
- does not read files, environment variables or the ROS graph;
- produces no network or process side-effects.

It does not create ROS clients, services, executors, registries, suite/task
parsers, episode/evaluator/report components or any production launch wiring.
Those are explicitly out of benchmark endpoint resolver scope and belong to benchmark step transport and later Work
Packages.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class BenchmarkEndpointError(ValueError):
    """Raised when benchmark endpoint identity cannot be resolved."""


@dataclass(frozen=True, slots=True)
class BenchmarkEndpoints:
    """Endpoint identity for a single benchmark instance.

    The model only describes ROS namespace/service/topic identity. It never
    carries suite, task, checkpoint, episode, step, lane, reward or success
    information -- those belong to evaluator/RunPlanner layers in later Work
    Packages.
    """

    benchmark_type: str
    adapter_name: str
    instance_id: str
    namespace: str
    reset_service: str
    step_service: str


_INSTANCE_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_FULL_PATH_OVERRIDE_KEYS = frozenset(("namespace", "reset_service", "step_service", "status_topic"))


def _is_non_empty_exact_string(value: Any) -> bool:
    """Return True only for a non-empty string without surrounding whitespace.

    Whitespace-padded / empty values are rejected so the resolver does not
    silently ``.strip()`` caller input. Bool/list/int are rejected too.
    """
    if not isinstance(value, str):
        return False
    if not value:
        return False
    return value.strip() == value


def _require_benchmark_section(robot_config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "benchmark" not in robot_config:
        raise BenchmarkEndpointError(
            "robot config is missing the 'benchmark' section; "
            "benchmark endpoint identity requires benchmark.type and benchmark.adapter"
        )
    benchmark = robot_config["benchmark"]
    if not isinstance(benchmark, Mapping):
        raise BenchmarkEndpointError("benchmark section must be a mapping when present")
    return benchmark


def _require_type(benchmark: Mapping[str, Any]) -> str:
    if "type" not in benchmark:
        raise BenchmarkEndpointError("benchmark.type is required and must be an exact non-empty string")
    value = benchmark["type"]
    if not _is_non_empty_exact_string(value):
        raise BenchmarkEndpointError(
            "benchmark.type must be an exact non-empty string (no surrounding whitespace, no bool/number/list)"
        )
    return value


def _require_adapter(benchmark: Mapping[str, Any]) -> str:
    if "adapter" not in benchmark:
        raise BenchmarkEndpointError("benchmark.adapter is required and must be an exact non-empty string")
    value = benchmark["adapter"]
    if not _is_non_empty_exact_string(value):
        raise BenchmarkEndpointError(
            "benchmark.adapter must be an exact non-empty string (no surrounding whitespace, no bool/number/list)"
        )
    return value


def _resolve_instance_id(benchmark: Mapping[str, Any], benchmark_type: str) -> str:
    if "instance_id" not in benchmark:
        candidate = benchmark_type
    else:
        candidate = benchmark["instance_id"]
        if not _is_non_empty_exact_string(candidate):
            raise BenchmarkEndpointError(
                "benchmark.instance_id must be an exact non-empty string "
                "(no surrounding whitespace, no bool/number/list)"
            )
    if not _INSTANCE_ID_PATTERN.match(candidate):
        raise BenchmarkEndpointError(
            f"benchmark.instance_id token {candidate!r} is not a legal ROS "
            f"identifier (expected pattern {_INSTANCE_ID_PATTERN.pattern}); "
            "configure benchmark.instance_id explicitly when benchmark.type is "
            "not a legal token; resolver will not rewrite, lowercase or alias it"
        )
    return candidate


def _reject_full_path_overrides(benchmark: Mapping[str, Any]) -> None:
    overlaps = _FULL_PATH_OVERRIDE_KEYS.intersection(benchmark.keys())
    if overlaps:
        joined = ", ".join(sorted(overlaps))
        raise BenchmarkEndpointError(
            "benchmark section must not provide full ROS path keys; "
            f"found: {joined}. "
            "Endpoints are derived from benchmark.type / benchmark.instance_id."
        )


def resolve_benchmark_endpoints(
    robot_config: Mapping[str, Any],
) -> BenchmarkEndpoints:
    """Resolve the deterministic benchmark ROS endpoint identity.

    Args:
        robot_config: The normalized robot SSOT mapping (the ``robot`` section
            after the loader has unpacked the outer ``{"robot": {...}}``
            wrapper). Must contain a top-level ``benchmark`` mapping with
            non-empty ``type`` and ``adapter`` strings.

    Returns:
        A frozen ``BenchmarkEndpoints`` with namespace, reset service and
        step service derived from ``instance_id``.

    Raises:
        BenchmarkEndpointError: For any malformed configuration. The resolver
            never returns ``None`` or partial results and never silently
            falls back.
    """
    if not isinstance(robot_config, Mapping):
        raise BenchmarkEndpointError("robot_config must be a mapping (the normalized robot SSOT)")

    benchmark = _require_benchmark_section(robot_config)
    _reject_full_path_overrides(benchmark)

    benchmark_type = _require_type(benchmark)
    adapter_name = _require_adapter(benchmark)
    instance_id = _resolve_instance_id(benchmark, benchmark_type)

    namespace = f"/benchmark/{instance_id}"
    return BenchmarkEndpoints(
        benchmark_type=benchmark_type,
        adapter_name=adapter_name,
        instance_id=instance_id,
        namespace=namespace,
        reset_service=f"{namespace}/reset",
        step_service=f"{namespace}/step",
    )
