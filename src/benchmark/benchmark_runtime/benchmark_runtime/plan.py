"""Immutable provider-agnostic benchmark plan models and strict JSON wire format."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from benchmark_runtime._strict_json import (
    StrictJSONError,
    dumps_strict,
    freeze_json,
    loads_strict,
    require_exact_keys,
    require_json_array,
    require_json_object,
    thaw_json,
)


class BenchmarkPlanJSONError(ValueError):
    """Raised when a benchmark plan JSON snapshot is malformed."""


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative int (bool rejected)")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive int (bool rejected)")
    return value


def _positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return result


def _exact_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _freeze_metadata(value: Mapping[str, Any] | None, field_name: str) -> MappingProxyType:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        frozen = freeze_json(value)
    except StrictJSONError as exc:
        raise ValueError(f"{field_name} must contain strict JSON values: {exc}") from exc
    if not isinstance(frozen, MappingProxyType):  # pragma: no cover - freeze_json invariant
        raise ValueError(f"{field_name} must be a JSON object")
    return frozen


@dataclass(frozen=True, slots=True)
class BenchmarkPlanTask:
    """Provider-resolved task snapshot with opaque provider-owned metadata.

    Only the stable task identity and prompt are generic. Native resource
    locators and provider-specific fields live in ``metadata`` and are
    interpreted exclusively by the owning adapter.
    """

    task_id: int
    name: str
    prompt: str
    init_state_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_negative_int(self.task_id, "task_id")
        _non_empty_string(self.name, "name")
        _non_empty_string(self.prompt, "prompt")
        if self.init_state_count is not None:
            _positive_int(self.init_state_count, "init_state_count")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class InitStatePolicy:
    """Resolved provider-independent initial-state selection policy."""

    use_init_state_id: bool
    selection: str

    def __post_init__(self) -> None:
        _exact_bool(self.use_init_state_id, "use_init_state_id")
        _non_empty_string(self.selection, "selection")


@dataclass(frozen=True, slots=True)
class BenchmarkTimeouts:
    startup_timeout_sec: float
    max_duration_sec: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "startup_timeout_sec", _positive_float(self.startup_timeout_sec, "startup_timeout_sec")
        )
        object.__setattr__(self, "max_duration_sec", _positive_float(self.max_duration_sec, "max_duration_sec"))


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactFlags:
    """Resolved artifact flags included in the plan snapshot for startup gating."""

    save_sim_states: bool
    video_enabled: bool
    write_native: bool
    write_canonical: bool

    def __post_init__(self) -> None:
        _exact_bool(self.save_sim_states, "save_sim_states")
        _exact_bool(self.video_enabled, "video_enabled")
        _exact_bool(self.write_native, "write_native")
        _exact_bool(self.write_canonical, "write_canonical")


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    """Complete immutable plan returned by ``GetBenchmarkPlan``."""

    benchmark_type: str
    adapter: str
    suite: str
    requested_task_order_index: int
    effective_task_order_index: int
    tasks: tuple[BenchmarkPlanTask, ...]
    selected_task_ids: tuple[int, ...]
    episodes_per_task: int
    planned_episodes: int
    seed: int
    init_state_policy: InitStatePolicy
    max_steps: int
    timeouts: BenchmarkTimeouts
    artifacts: BenchmarkArtifactFlags
    warnings: tuple[str, ...] = ()
    lane_count: int = 1

    def __post_init__(self) -> None:
        _non_empty_string(self.benchmark_type, "benchmark_type")
        _non_empty_string(self.adapter, "adapter")
        _non_empty_string(self.suite, "suite")
        _non_negative_int(self.requested_task_order_index, "requested_task_order_index")
        _non_negative_int(self.effective_task_order_index, "effective_task_order_index")
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "selected_task_ids", tuple(self.selected_task_ids))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if not self.tasks:
            raise ValueError("tasks must contain at least one resolved task")
        expected_ids = tuple(range(len(self.tasks)))
        actual_ids = tuple(task.task_id for task in self.tasks)
        if actual_ids != expected_ids:
            raise ValueError(f"task IDs must be contiguous suite-local IDs {expected_ids}; got {actual_ids}")
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("resolved task names must be unique")
        for task_id in self.selected_task_ids:
            _non_negative_int(task_id, "selected_task_ids entry")
        if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
            raise ValueError("selected_task_ids contains a duplicate task ID")
        if not self.selected_task_ids:
            raise ValueError("selected_task_ids must not be empty")
        invalid = [task_id for task_id in self.selected_task_ids if task_id >= len(self.tasks)]
        if invalid:
            raise ValueError(f"selected task IDs out of range for {len(self.tasks)} tasks: {invalid}")
        _positive_int(self.episodes_per_task, "episodes_per_task")
        _positive_int(self.planned_episodes, "planned_episodes")
        expected_episodes = len(self.selected_task_ids) * self.episodes_per_task
        if self.planned_episodes != expected_episodes:
            raise ValueError(
                f"planned_episodes must equal selected task count * episodes_per_task ({expected_episodes})"
            )
        _non_negative_int(self.seed, "seed")
        _positive_int(self.max_steps, "max_steps")
        _positive_int(self.lane_count, "lane_count")
        if not isinstance(self.init_state_policy, InitStatePolicy):
            raise ValueError("init_state_policy must be an InitStatePolicy")
        if self.init_state_policy.use_init_state_id:
            missing_counts = [task.task_id for task in self.tasks if task.init_state_count is None]
            if missing_counts:
                raise ValueError(f"init_state_count is required when initial-state IDs are enabled: {missing_counts}")
        if not isinstance(self.timeouts, BenchmarkTimeouts):
            raise ValueError("timeouts must be BenchmarkTimeouts")
        if not isinstance(self.artifacts, BenchmarkArtifactFlags):
            raise ValueError("artifacts must be BenchmarkArtifactFlags")
        for warning in self.warnings:
            _non_empty_string(warning, "warnings entry")


def _task_to_dict(task: BenchmarkPlanTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "name": task.name,
        "prompt": task.prompt,
        "init_state_count": task.init_state_count,
        "metadata": thaw_json(task.metadata),
    }


def _plan_to_dict(plan: BenchmarkPlan) -> dict[str, Any]:
    return {
        "benchmark_type": plan.benchmark_type,
        "adapter": plan.adapter,
        "suite": plan.suite,
        "requested_task_order_index": plan.requested_task_order_index,
        "effective_task_order_index": plan.effective_task_order_index,
        "tasks": [_task_to_dict(task) for task in plan.tasks],
        "selected_task_ids": list(plan.selected_task_ids),
        "episodes_per_task": plan.episodes_per_task,
        "planned_episodes": plan.planned_episodes,
        "seed": plan.seed,
        "init_state_policy": {
            "use_init_state_id": plan.init_state_policy.use_init_state_id,
            "selection": plan.init_state_policy.selection,
        },
        "max_steps": plan.max_steps,
        "timeouts": {
            "startup_timeout_sec": plan.timeouts.startup_timeout_sec,
            "max_duration_sec": plan.timeouts.max_duration_sec,
        },
        "artifacts": {
            "save_sim_states": plan.artifacts.save_sim_states,
            "video_enabled": plan.artifacts.video_enabled,
            "write_native": plan.artifacts.write_native,
            "write_canonical": plan.artifacts.write_canonical,
        },
        "warnings": list(plan.warnings),
        "lane_count": plan.lane_count,
    }


def benchmark_plan_to_json(plan: BenchmarkPlan) -> str:
    if not isinstance(plan, BenchmarkPlan):
        raise BenchmarkPlanJSONError(f"plan must be BenchmarkPlan, got {type(plan).__name__}")
    try:
        return dumps_strict(_plan_to_dict(plan))
    except StrictJSONError as exc:
        raise BenchmarkPlanJSONError(str(exc)) from exc


def _task_from_wire(raw_task: Any, index: int) -> BenchmarkPlanTask:
    task = require_json_object(raw_task, f"benchmark plan tasks[{index}]")
    required = {"task_id", "name", "prompt", "init_state_count"}
    missing = sorted(required - set(task))
    if missing:
        raise StrictJSONError(f"benchmark plan tasks[{index}] missing keys: {missing}")

    metadata_value = task.get("metadata")
    provider_fields = {key: value for key, value in task.items() if key not in required | {"metadata"}}
    if metadata_value is not None and provider_fields:
        raise StrictJSONError(
            f"benchmark plan tasks[{index}] cannot mix metadata with flattened provider fields: "
            f"{sorted(provider_fields)}"
        )
    metadata = (
        provider_fields
        if metadata_value is None
        else require_json_object(metadata_value, f"benchmark plan tasks[{index}].metadata")
    )
    return BenchmarkPlanTask(
        task_id=task["task_id"],
        name=task["name"],
        prompt=task["prompt"],
        init_state_count=task["init_state_count"],
        metadata=metadata,
    )


def benchmark_plan_from_json(payload: str) -> BenchmarkPlan:
    """Parse the plan schema and accept legacy flattened provider metadata."""
    try:
        root = require_json_object(loads_strict(payload), "benchmark plan")
        require_exact_keys(
            root,
            {
                "benchmark_type",
                "adapter",
                "suite",
                "requested_task_order_index",
                "effective_task_order_index",
                "tasks",
                "selected_task_ids",
                "episodes_per_task",
                "planned_episodes",
                "seed",
                "init_state_policy",
                "max_steps",
                "timeouts",
                "artifacts",
                "warnings",
                "lane_count",
            },
            "benchmark plan",
        )
        tasks_raw = require_json_array(root["tasks"], "benchmark plan tasks")
        tasks = tuple(_task_from_wire(raw_task, index) for index, raw_task in enumerate(tasks_raw))
        selected = require_json_array(root["selected_task_ids"], "selected_task_ids")
        warnings = require_json_array(root["warnings"], "warnings")
        init_policy = require_json_object(root["init_state_policy"], "init_state_policy")
        require_exact_keys(init_policy, {"use_init_state_id", "selection"}, "init_state_policy")
        timeouts = require_json_object(root["timeouts"], "timeouts")
        require_exact_keys(timeouts, {"startup_timeout_sec", "max_duration_sec"}, "timeouts")
        artifacts = require_json_object(root["artifacts"], "artifacts")
        require_exact_keys(
            artifacts,
            {"save_sim_states", "video_enabled", "write_native", "write_canonical"},
            "artifacts",
        )
        return BenchmarkPlan(
            benchmark_type=root["benchmark_type"],
            adapter=root["adapter"],
            suite=root["suite"],
            requested_task_order_index=root["requested_task_order_index"],
            effective_task_order_index=root["effective_task_order_index"],
            tasks=tasks,
            selected_task_ids=tuple(selected),
            episodes_per_task=root["episodes_per_task"],
            planned_episodes=root["planned_episodes"],
            seed=root["seed"],
            init_state_policy=InitStatePolicy(**init_policy),
            max_steps=root["max_steps"],
            timeouts=BenchmarkTimeouts(**timeouts),
            artifacts=BenchmarkArtifactFlags(**artifacts),
            warnings=tuple(warnings),
            lane_count=root["lane_count"],
        )
    except (StrictJSONError, TypeError, ValueError) as exc:
        raise BenchmarkPlanJSONError(str(exc)) from exc
