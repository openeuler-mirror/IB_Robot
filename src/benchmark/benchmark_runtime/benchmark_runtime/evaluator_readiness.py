"""Pure one-deadline readiness barrier for the benchmark evaluator."""

from __future__ import annotations

import math
from collections.abc import Iterable

from benchmark_runtime.plan import BenchmarkPlan


class ReadinessError(RuntimeError):
    """Raised when readiness is invalid or the single startup deadline expires."""


def parse_health_values(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in items:
        if key in values:
            raise ReadinessError(f"duplicate health key {key!r}")
        values[key] = value
    return values


class ReadinessBarrier:
    """Accumulate contract evidence under one injected monotonic deadline."""

    def __init__(self, start_monotonic: float, timeout_sec: float) -> None:
        if not math.isfinite(start_monotonic):
            raise ReadinessError("start_monotonic must be finite")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ReadinessError("timeout_sec must be finite and positive")
        self._timeout_sec = float(timeout_sec)
        self._deadline = start_monotonic + self._timeout_sec
        self._plan = False
        self._contract = False
        self._health = False
        self._services = {"reset": False, "finalize": False, "prepare": False}
        self._action = False

    @property
    def deadline(self) -> float:
        return self._deadline

    @property
    def plan_ready(self) -> bool:
        return self._plan

    @property
    def contract_ready(self) -> bool:
        return self._contract

    @property
    def health_ready(self) -> bool:
        return self._health

    @property
    def services_ready(self) -> bool:
        return all(self._services.values())

    @property
    def action_ready(self) -> bool:
        return self._action

    @property
    def ready(self) -> bool:
        return self._plan and self._contract and self._health and all(self._services.values()) and self._action

    def remaining(self, now: float) -> float:
        remaining = self._deadline - now
        if remaining <= 0:
            raise ReadinessError("startup deadline expired")
        return remaining

    def set_plan(self, plan: BenchmarkPlan, *, allow_artifacts: bool = False) -> None:
        if plan.timeouts.startup_timeout_sec != self._timeout_sec:
            raise ReadinessError(
                "returned plan startup_timeout_sec does not match configured startup_timeout_sec: "
                f"{plan.timeouts.startup_timeout_sec} != {self._timeout_sec}"
            )
        if not allow_artifacts and any(
            (
                plan.artifacts.save_sim_states,
                plan.artifacts.video_enabled,
                plan.artifacts.write_native,
                plan.artifacts.write_canonical,
            )
        ):
            raise ReadinessError("artifact-enabled plans require output readiness")
        self._plan = True

    def set_contract_compatible(self) -> None:
        self._contract = True

    def set_health(self, *, level: int, values: dict[str, str]) -> None:
        if level != 0 or values.get("state") != "ready" or values.get("backend_state") != "ready":
            raise ReadinessError("inference health must be exactly OK with state=ready and backend_state=ready")
        self._health = True

    def set_service_available(self, name: str, available: bool) -> None:
        if name not in self._services:
            raise ReadinessError(f"unknown readiness service {name!r}")
        self._services[name] = bool(available)

    def set_action_available(self, available: bool) -> None:
        self._action = bool(available)
