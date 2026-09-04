"""Runtime target resolver for the IB-Robot launch pipeline.

The system can no longer be described by a single ``use_sim`` boolean. Four
orthogonal dimensions exist (runtime target, embodiment, world provider,
control mode); this module resolves only the runtime target dimension.

Resolution priority (highest first):

1. non-empty ``runtime_target`` launch override;
2. SSOT ``runtime.target``;
3. historical ``use_sim=true`` -> simulation;
4. default hardware.

Consistency rules (fail-fast):

- target benchmark/simulation requires ``use_sim=true``;
- explicit target hardware requires ``use_sim=false``;
- unknown target string is rejected;
- target benchmark requires ``benchmark.type`` and ``benchmark.adapter``.

``benchmark + use_sim=true`` must resolve to ``RuntimeTarget.BENCHMARK`` and
must never be re-resolved to simulation. The benchmark target must continue to
skip ``SimBackendAdapter``; this module never touches that registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any


class RuntimeTarget(str, Enum):
    """The runtime that carries the robot embodiment."""

    HARDWARE = "hardware"
    SIMULATION = "simulation"
    BENCHMARK = "benchmark"

    def __str__(self) -> str:
        return self.value


_VALID_TARGETS = frozenset(target.value for target in RuntimeTarget)


class RuntimeTargetError(ValueError):
    """Raised when the runtime target cannot be resolved consistently."""


def _normalize_target(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RuntimeTargetError(f"runtime target must be a string, got {type(value).__name__}: {value!r}")
    return value.strip().lower()


def _resolve_explicit_target(robot_config: Mapping[str, Any], override: str) -> str:
    """Return the explicit target string from override or SSOT.

    Returns an empty string when neither the launch override nor the SSOT
    ``runtime.target`` provides a value, so the caller falls back to the
    historical ``use_sim`` -> simulation rule or the default hardware.
    """

    if override:
        return override
    runtime_section = robot_config.get("runtime", {})
    if not isinstance(runtime_section, Mapping):
        raise RuntimeTargetError("runtime section must be a mapping when present")
    ssot_target = _normalize_target(runtime_section.get("target", "")) if runtime_section else ""
    if ssot_target:
        return ssot_target
    return ""


def _validate_benchmark_fields(robot_config: Mapping[str, Any]) -> None:
    benchmark = robot_config.get("benchmark", {})
    if not isinstance(benchmark, Mapping):
        raise RuntimeTargetError("benchmark section must be a mapping when target=benchmark")
    benchmark_type = benchmark.get("type")
    adapter = benchmark.get("adapter")
    if not isinstance(benchmark_type, str) or not benchmark_type.strip():
        raise RuntimeTargetError("runtime.target=benchmark requires a non-empty benchmark.type")
    if not isinstance(adapter, str) or not adapter.strip():
        raise RuntimeTargetError("runtime.target=benchmark requires a non-empty benchmark.adapter")


def resolve_runtime_target(
    robot_config: Mapping[str, Any],
    runtime_target_override: str,
    use_sim: bool | None,
) -> RuntimeTarget:
    """Resolve the runtime target from SSOT, launch override and use_sim.

    Args:
        robot_config: The loaded robot SSOT mapping (the ``robot`` section). It
            may carry ``runtime.target`` and ``benchmark.*`` fields.
        runtime_target_override: The ``runtime_target`` launch argument. An
            empty string means "no override; use SSOT or historical fallback".
        use_sim: Whether the current embodiment is virtual (``true``) or real
            (``false``). ``None`` means the launch argument was omitted; in
            that case the resolved target determines the embodiment. It is not
            the runtime target; it only constrains consistency when explicit.
    """

    override = _normalize_target(runtime_target_override)
    target_str = _resolve_explicit_target(robot_config, override)
    explicit_target = bool(target_str)
    if not target_str:
        target_str = RuntimeTarget.SIMULATION.value if use_sim is True else RuntimeTarget.HARDWARE.value

    if target_str not in _VALID_TARGETS:
        raise RuntimeTargetError(f"Unknown runtime target: {target_str!r}")

    target = RuntimeTarget(target_str)

    # Consistency between an explicit target and use_sim. When the target is
    # inferred from use_sim (or use_sim was omitted), there is no conflict to
    # validate; the resolved target is the single source of truth for launch.
    if explicit_target and use_sim is not None:
        if target is RuntimeTarget.HARDWARE and use_sim:
            raise RuntimeTargetError("runtime target 'hardware' requires use_sim=false but use_sim=true")
        if target is not RuntimeTarget.HARDWARE and not use_sim:
            source = "override" if override else "SSOT"
            raise RuntimeTargetError(
                f"runtime target {target.value!r} (from {source}) requires use_sim=true but use_sim=false"
            )

    if target is RuntimeTarget.BENCHMARK:
        _validate_benchmark_fields(robot_config)

    return target
