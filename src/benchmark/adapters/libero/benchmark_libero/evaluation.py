"""Parse the LIBERO evaluation mapping into the pure R1 plan request."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark_libero.plan_resolver import LiberoPlanRequest


def resolve_plan_request(options: Mapping[str, Any]) -> LiberoPlanRequest | None:
    """Return a plan request when evaluation is enabled, else ``None``."""
    evaluation = options.get("evaluation")
    if evaluation is None:
        return None
    if not isinstance(evaluation, Mapping):
        raise ValueError("benchmark.evaluation must be a mapping")
    if not bool(evaluation.get("enabled", False)):
        return None

    tasks_raw = evaluation.get("tasks")
    tasks = None if tasks_raw is None else tuple(tasks_raw)
    video = evaluation.get("video", {})
    output = evaluation.get("output", {})
    timeouts = evaluation.get("timeouts", {})
    if not isinstance(video, Mapping):
        raise ValueError("benchmark.evaluation.video must be a mapping")
    if not isinstance(output, Mapping):
        raise ValueError("benchmark.evaluation.output must be a mapping")
    if not isinstance(timeouts, Mapping):
        raise ValueError("benchmark.evaluation.timeouts must be a mapping")

    return LiberoPlanRequest(
        suite=evaluation.get("suite"),
        task_order_index=evaluation.get("task_order_index", 0),
        tasks=tasks,
        episodes_per_task=evaluation.get("episodes_per_task", 10),
        seed=evaluation.get("seed", 10000),
        max_steps=evaluation.get("max_steps", 600),
        startup_timeout_sec=timeouts.get("startup_timeout_sec", 120.0),
        max_duration_sec=timeouts.get("max_duration_sec", 600.0),
        save_sim_states=evaluation.get("save_sim_states", True),
        video_enabled=video.get("enabled", True),
        write_native=output.get("write_native", True),
        write_canonical=output.get("write_canonical", True),
        lane_count=evaluation.get("lane_count", 1),
    )
