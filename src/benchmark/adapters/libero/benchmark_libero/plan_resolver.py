"""Pure LIBERO suite resolver with an injectable provider seam.

This module does not import LIBERO, MuJoCo, Torch, LeRobot, or rclpy. Native
provider access is isolated in :mod:`benchmark_libero.plan_provider`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from benchmark_runtime.plan import (
    BenchmarkArtifactFlags,
    BenchmarkPlan,
    BenchmarkPlanTask,
    BenchmarkTimeouts,
    InitStatePolicy,
)

_TEN_TASK_SUITES = frozenset({"libero_spatial", "libero_object", "libero_goal", "libero_10"})
_LARGE_SUITES = frozenset({"libero_90", "libero_100"})
_VALID_SUITES = _TEN_TASK_SUITES | _LARGE_SUITES
_SELECTION = "episode_index_modulo_available_native_init_state_count"


@dataclass(frozen=True, slots=True)
class ProviderTask:
    """Provider-owned task metadata before suite-local IDs are assigned."""

    name: str
    prompt: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    init_state_count: int

    def __post_init__(self) -> None:
        for field_name in ("name", "prompt", "problem_folder", "bddl_file", "init_states_file"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.init_state_count, int)
            or isinstance(self.init_state_count, bool)
            or self.init_state_count <= 0
        ):
            raise ValueError("init_state_count must be a positive int (bool rejected)")


class LiberoPlanProvider(Protocol):
    """Minimal provider seam used by the pure resolver."""

    def resolve_suite(self, suite: str, task_order_index: int) -> tuple[ProviderTask, ...]: ...


@dataclass(frozen=True, slots=True)
class LiberoPlanRequest:
    """Already schema-validated evaluation values needed to resolve a plan."""

    suite: str
    task_order_index: int
    tasks: tuple[int, ...] | None
    episodes_per_task: int
    seed: int
    max_steps: int
    startup_timeout_sec: float
    max_duration_sec: float
    save_sim_states: bool
    video_enabled: bool
    write_native: bool
    write_canonical: bool
    lane_count: int = 1

    def __post_init__(self) -> None:
        if self.suite not in _VALID_SUITES:
            raise ValueError(f"suite must be one of {sorted(_VALID_SUITES)}")
        if (
            not isinstance(self.task_order_index, int)
            or isinstance(self.task_order_index, bool)
            or self.task_order_index < 0
        ):
            raise ValueError("task_order_index must be a non-negative int (bool rejected)")
        if self.tasks is not None:
            object.__setattr__(self, "tasks", tuple(self.tasks))
            for task_id in self.tasks:
                if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
                    raise ValueError("tasks entries must be non-negative ints (bool rejected)")
        for field_name in ("episodes_per_task", "max_steps"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive int (bool rejected)")
        if not isinstance(self.lane_count, int) or isinstance(self.lane_count, bool) or self.lane_count < 1:
            raise ValueError("lane_count must be a positive int (bool rejected)")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative int (bool rejected)")
        for field_name in ("startup_timeout_sec", "max_duration_sec"):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be a finite positive number")
        for field_name in ("save_sim_states", "video_enabled", "write_native", "write_canonical"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")


class LiberoPlanResolver:
    """Resolve suite/task order through a provider without owning its runtime."""

    def __init__(self, provider: LiberoPlanProvider) -> None:
        self._provider = provider

    def resolve(self, request: LiberoPlanRequest) -> BenchmarkPlan:
        requested_order = request.task_order_index
        warnings: tuple[str, ...] = ()
        if request.suite in _TEN_TASK_SUITES:
            if requested_order > 20:
                raise ValueError(f"suite {request.suite!r} task_order_index must be in 0..20")
            effective_order = requested_order
            provider_tasks = self._provider.resolve_suite(request.suite, effective_order)
        else:
            effective_order = 0
            if requested_order != 0:
                warnings = (
                    f"suite {request.suite!r} requested task_order_index={requested_order} normalized to effective 0",
                )
            if request.suite == "libero_90":
                provider_tasks = self._provider.resolve_suite("libero_90", 0)
            else:
                provider_tasks = self._provider.resolve_suite("libero_90", 0) + self._provider.resolve_suite(
                    "libero_10", 0
                )

        expected_count = 90 if request.suite == "libero_90" else 100 if request.suite == "libero_100" else 10
        if len(provider_tasks) != expected_count:
            raise ValueError(f"suite {request.suite!r} resolved {len(provider_tasks)} tasks; expected {expected_count}")
        names = [task.name for task in provider_tasks]
        if len(set(names)) != len(names):
            raise ValueError(f"suite {request.suite!r} resolved duplicate task names")

        tasks = tuple(
            BenchmarkPlanTask(
                task_id=task_id,
                name=task.name,
                prompt=task.prompt,
                init_state_count=task.init_state_count,
                metadata={
                    "problem_folder": task.problem_folder,
                    "bddl_file": task.bddl_file,
                    "init_states_file": task.init_states_file,
                },
            )
            for task_id, task in enumerate(provider_tasks)
        )
        selected = tuple(range(len(tasks))) if request.tasks is None else request.tasks
        if not selected:
            raise ValueError("tasks must be omitted or contain at least one task ID")
        if len(set(selected)) != len(selected):
            raise ValueError("tasks contains a duplicate task ID")
        invalid = [task_id for task_id in selected if task_id >= len(tasks)]
        if invalid:
            raise ValueError(f"task ID out of range for suite {request.suite!r}: {invalid}")

        return BenchmarkPlan(
            benchmark_type="libero",
            adapter="libero",
            suite=request.suite,
            requested_task_order_index=requested_order,
            effective_task_order_index=effective_order,
            tasks=tasks,
            selected_task_ids=selected,
            episodes_per_task=request.episodes_per_task,
            planned_episodes=len(selected) * request.episodes_per_task,
            seed=request.seed,
            init_state_policy=InitStatePolicy(use_init_state_id=True, selection=_SELECTION),
            max_steps=request.max_steps,
            timeouts=BenchmarkTimeouts(
                startup_timeout_sec=request.startup_timeout_sec,
                max_duration_sec=request.max_duration_sec,
            ),
            artifacts=BenchmarkArtifactFlags(
                save_sim_states=request.save_sim_states,
                video_enabled=request.video_enabled,
                write_native=request.write_native,
                write_canonical=request.write_canonical,
            ),
            warnings=warnings,
            lane_count=request.lane_count,
        )
