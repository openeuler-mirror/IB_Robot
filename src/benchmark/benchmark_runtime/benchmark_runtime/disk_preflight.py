"""Conservative, provider-agnostic disk-space preflight for benchmark artifacts."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from math import ceil, isfinite
from pathlib import Path
from typing import Any

from benchmark_runtime.canonical_models import DiskEstimate

TERMINAL_RESET_FRAME_ALLOWANCE = 1
STATE_BYTES_PER_SAMPLE = 1024 * 1024
REPORT_OVERHEAD_BYTES = 16 * 1024 * 1024
SAFETY_FACTOR = 1.5
MIN_FREE_RESERVE_BYTES = 1024 * 1024 * 1024


class DiskPreflightError(RuntimeError):
    """Raised when configured output storage cannot satisfy the estimate."""


def _read(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _bool_config(config: Any, key: str, default: bool = False) -> bool:
    value = _read(config, key, default=default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _planned_episodes(plan: Any) -> int:
    direct = _read(plan, "planned_episodes")
    if direct is not None:
        if not isinstance(direct, int) or isinstance(direct, bool) or direct < 0:
            raise ValueError("planned_episodes must be a non-negative int")
        return direct
    tasks = _read(plan, "selected_tasks", "tasks", "planned_tasks")
    episodes_per_task = _read(plan, "episodes_per_task", default=None)
    if tasks is None or episodes_per_task is None:
        raise ValueError("plan must provide planned_episodes or tasks plus episodes_per_task")
    if not isinstance(tasks, Sequence) or isinstance(tasks, str | bytes):
        raise ValueError("tasks must be a sequence")
    if not isinstance(episodes_per_task, int) or isinstance(episodes_per_task, bool) or episodes_per_task < 0:
        raise ValueError("episodes_per_task must be a non-negative int")
    return len(tasks) * episodes_per_task


def _image_dimensions(image_size: Sequence[int] | None) -> tuple[int, int]:
    if image_size is None:
        return 0, 0
    if len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width = _positive_int(image_size[0], "image width")
    height = _positive_int(image_size[1], "image height")
    return width, height


def estimate_disk_usage(
    plan: Any,
    artifact_config: Any = None,
    image_size: Sequence[int] | None = None,
    *,
    max_steps: int | None = None,
    state_bytes_per_sample: int = STATE_BYTES_PER_SAMPLE,
    report_overhead_bytes: int = REPORT_OVERHEAD_BYTES,
    safety_factor: float = SAFETY_FACTOR,
    reserve_bytes: int = MIN_FREE_RESERVE_BYTES,
) -> DiskEstimate:
    """Estimate required bytes using the configured-size upper bound.

    ``plan`` accepts either a mapping/object with ``planned_episodes`` or one
    with ``tasks``/``selected_tasks`` and ``episodes_per_task``.  The artifact
    config accepts either ``video``/``save_sim_states`` or the corresponding
    flat keys.  Video bytes are intentionally raw RGB bytes, not a prediction
    of compressed MP4 size.
    """
    planned = _planned_episodes(plan)
    steps = max_steps if max_steps is not None else _read(plan, "max_steps", "max_actions", default=None)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("max_steps must be a non-negative int")
    if (
        not isinstance(state_bytes_per_sample, int)
        or isinstance(state_bytes_per_sample, bool)
        or state_bytes_per_sample < 0
    ):
        raise ValueError("state_bytes_per_sample must be a non-negative int")
    if (
        not isinstance(report_overhead_bytes, int)
        or isinstance(report_overhead_bytes, bool)
        or report_overhead_bytes < 0
    ):
        raise ValueError("report_overhead_bytes must be a non-negative int")
    if not isinstance(reserve_bytes, int) or isinstance(reserve_bytes, bool) or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a non-negative int")
    if not isinstance(safety_factor, int | float) or isinstance(safety_factor, bool):
        raise ValueError("safety_factor must be a finite number")
    if not isfinite(float(safety_factor)) or float(safety_factor) < 1.0:
        raise ValueError("safety_factor must be finite and at least 1")

    video_config = _read(artifact_config, "video", default={})
    video_enabled = _bool_config(artifact_config, "video_enabled", default=False)
    if isinstance(video_config, Mapping) or hasattr(video_config, "enabled"):
        video_enabled = _bool_config(video_config, "enabled", default=video_enabled)
    save_states = _read(artifact_config, "save_sim_states", "save_states", default=False)
    if not isinstance(save_states, bool):
        raise ValueError("save_sim_states must be a boolean")

    width, height = _image_dimensions(image_size)
    max_frames = planned * (steps + TERMINAL_RESET_FRAME_ALLOWANCE)
    video_bytes = width * height * 3 * max_frames if video_enabled else 0
    state_bytes = state_bytes_per_sample * planned * (steps + 1) if save_states else 0
    estimated = video_bytes + state_bytes + report_overhead_bytes
    required = ceil(estimated * float(safety_factor)) + reserve_bytes
    return DiskEstimate(
        planned_episodes=planned,
        max_frames=max_frames,
        video_bytes=video_bytes,
        state_bytes=state_bytes,
        report_overhead_bytes=report_overhead_bytes,
        safety_factor=float(safety_factor),
        estimated_bytes=estimated,
        reserve_bytes=reserve_bytes,
        required_bytes=required,
    )


def check_disk_preflight(
    output_root: str | Path,
    estimate: DiskEstimate,
    *,
    disk_usage_fn: Callable[[str | Path], Any] = shutil.disk_usage,
) -> DiskEstimate:
    """Check free space and return the estimate annotated with the result."""
    if not isinstance(estimate, DiskEstimate):
        raise TypeError("estimate must be DiskEstimate")
    usage = disk_usage_fn(output_root)
    free = getattr(usage, "free", None)
    if free is None and isinstance(usage, tuple) and len(usage) >= 3:
        free = usage[2]
    if not isinstance(free, int) or isinstance(free, bool) or free < 0:
        raise ValueError("disk_usage_fn must return free bytes")
    return replace(estimate, free_bytes=free, passed=free >= estimate.required_bytes)


def ensure_disk_preflight(
    output_root: str | Path,
    estimate: DiskEstimate,
    *,
    disk_usage_fn: Callable[[str | Path], Any] = shutil.disk_usage,
) -> DiskEstimate:
    """Check space and raise before the first reset when the check fails."""
    checked = check_disk_preflight(output_root, estimate, disk_usage_fn=disk_usage_fn)
    if not checked.passed:
        raise DiskPreflightError(
            f"insufficient disk space at {output_root}: free={checked.free_bytes} required={checked.required_bytes}"
        )
    return checked
