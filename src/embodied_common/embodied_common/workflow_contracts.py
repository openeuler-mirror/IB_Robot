"""Canonical typed Workflow contracts shared by planners and executors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from embodied_common.canon import sha256_text, to_canonical_json

_WORKFLOW_STEP_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "skill_name",
        "target_name",
        "container_name",
        "place_name",
        "motion_direction",
        "motion_distance",
        "arm_side",
        "imitation_duration_sec",
        "timeout_sec",
    }
)
_WORKFLOW_STEP_NAVIGATION_FIELDS = frozenset(
    {"direction", "distance", "degree", "has_x", "x", "has_y", "y", "has_yaw", "yaw"}
)

WORKFLOW_STEP_COMMON_FIELDS = _WORKFLOW_STEP_COMMON_FIELDS
WORKFLOW_STEP_NAVIGATION_FIELDS = _WORKFLOW_STEP_NAVIGATION_FIELDS


def validate_workflow_steps(steps: Sequence[Any], *, max_steps: int = 16) -> None:
    """Validate raw workflow mappings before normalization can coerce values."""
    if not isinstance(steps, Sequence) or isinstance(steps, str | bytes):
        raise TypeError("steps must be a sequence")
    if len(steps) > max_steps:
        raise ValueError(f"steps cannot contain more than {max_steps} entries")
    text_fields = {
        "skill_name",
        "target_name",
        "container_name",
        "place_name",
        "motion_direction",
        "arm_side",
        "direction",
    }
    number_fields = {"motion_distance", "imitation_duration_sec", "timeout_sec", "distance", "degree", "x", "y", "yaw"}
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            if not hasattr(step, "schema_version") or not hasattr(step, "skill_name"):
                raise TypeError(f"steps[{index}] must be a mapping or typed WorkflowStep")
            continue
        schema_version = step.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version not in {1, 2}:
            raise TypeError(f"steps[{index}].schema_version must be integer 1 or 2")
        allowed_fields = _WORKFLOW_STEP_COMMON_FIELDS | (
            _WORKFLOW_STEP_NAVIGATION_FIELDS if schema_version == 2 else set()
        )
        unknown_fields = sorted(set(step) - allowed_fields)
        if unknown_fields:
            raise ValueError(f"steps[{index}] contains unknown fields: {', '.join(unknown_fields)}")
        for field_name in text_fields & set(step):
            if not isinstance(step[field_name], str):
                raise TypeError(f"steps[{index}].{field_name} must be a string")
        for field_name in number_fields & set(step):
            value = step[field_name]
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise TypeError(f"steps[{index}].{field_name} must be a finite number")
        for field_name in {"has_x", "has_y", "has_yaw"} & set(step):
            if not isinstance(step[field_name], bool):
                raise TypeError(f"steps[{index}].{field_name} must be a boolean")


@dataclass(frozen=True)
class CanonicalWorkflowStep:
    schema_version: int
    skill_name: str
    target_name: str = ""
    container_name: str = ""
    place_name: str = ""
    motion_direction: str = ""
    motion_distance: float = 0.0
    arm_side: str = ""
    imitation_duration_sec: float = 0.0
    timeout_sec: float = 0.0
    direction: str = ""
    distance: float = 0.0
    degree: float = 0.0
    x: float | None = None
    y: float | None = None
    yaw: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError("WorkflowStep.schema_version must be 1 or 2")
        if not isinstance(self.skill_name, str):
            raise TypeError("WorkflowStep.skill_name must be a string")
        if not self.skill_name.strip():
            raise ValueError("WorkflowStep.skill_name must be non-empty")
        for field_name in ("target_name", "container_name", "place_name", "motion_direction", "arm_side", "direction"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"WorkflowStep.{field_name} must be a string")
        object.__setattr__(self, "skill_name", self.skill_name.strip())
        for field_name in ("target_name", "container_name", "place_name"):
            object.__setattr__(self, field_name, getattr(self, field_name).strip())
        for field_name in ("motion_direction", "arm_side", "direction"):
            object.__setattr__(self, field_name, getattr(self, field_name).strip().lower())
        if self.arm_side and self.arm_side not in {"left", "right", "auto"}:
            raise ValueError("WorkflowStep.arm_side must be left, right, or auto")
        for field_name in ("motion_distance", "imitation_duration_sec", "timeout_sec", "distance", "degree"):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"WorkflowStep.{field_name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"WorkflowStep.{field_name} must be non-negative")
        for field_name in ("x", "y", "yaw"):
            value = getattr(self, field_name)
            if value is not None:
                _finite_float(value)
        if self.schema_version == 1 and (
            self.direction.strip()
            or float(self.distance) != 0.0
            or float(self.degree) != 0.0
            or self.x is not None
            or self.y is not None
            or self.yaw is not None
        ):
            raise ValueError("navigation parameters require WorkflowStep schema_version 2")

    def to_dict(self) -> dict[str, Any]:
        normalized_arm_side = self.arm_side.strip().lower()
        normalized_imitation_duration = _finite_float(self.imitation_duration_sec)
        result = {
            "schema_version": self.schema_version,
            "skill_name": self.skill_name.strip(),
            "target_name": self.target_name.strip(),
            "container_name": self.container_name.strip(),
            "place_name": self.place_name.strip(),
            "motion_direction": self.motion_direction.strip().lower(),
            "motion_distance": _finite_float(self.motion_distance),
            "timeout_sec": _finite_float(self.timeout_sec),
        }
        if normalized_arm_side or normalized_imitation_duration:
            result.update(
                {
                    "arm_side": normalized_arm_side,
                    "imitation_duration_sec": normalized_imitation_duration,
                }
            )
        if self.schema_version == 2:
            result.update(
                {
                    "direction": self.direction.strip().lower(),
                    "distance": _finite_float(self.distance),
                    "degree": _finite_float(self.degree),
                    "has_x": self.x is not None,
                    "x": 0.0 if self.x is None else _finite_float(self.x),
                    "has_y": self.y is not None,
                    "y": 0.0 if self.y is None else _finite_float(self.y),
                    "has_yaw": self.yaw is not None,
                    "yaw": 0.0 if self.yaw is None else _finite_float(self.yaw),
                }
            )
        return result


def normalize_workflow_step(step: Any) -> CanonicalWorkflowStep:
    if isinstance(step, CanonicalWorkflowStep):
        return step
    schema_version = 0
    if isinstance(step, Mapping):
        values = step
        try:
            schema_version = int(values.get("schema_version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("WorkflowStep.schema_version must be an integer") from exc
        allowed_fields = _WORKFLOW_STEP_COMMON_FIELDS
        if schema_version == 2:
            allowed_fields |= _WORKFLOW_STEP_NAVIGATION_FIELDS
        unknown_fields = set(values) - allowed_fields
        if unknown_fields:
            names = ", ".join(sorted(str(name) for name in unknown_fields))
            raise ValueError(f"WorkflowStep contains unknown fields: {names}")
    else:
        values = {
            field: getattr(step, field, default)
            for field, default in (
                ("schema_version", 0),
                ("skill_name", ""),
                ("target_name", ""),
                ("container_name", ""),
                ("place_name", ""),
                ("motion_direction", ""),
                ("motion_distance", 0.0),
                ("arm_side", ""),
                ("imitation_duration_sec", 0.0),
                ("timeout_sec", 0.0),
                ("direction", ""),
                ("distance", 0.0),
                ("degree", 0.0),
                ("has_x", None),
                ("x", None),
                ("has_y", None),
                ("y", None),
                ("has_yaw", None),
                ("yaw", None),
            )
        }
        schema_version = int(values.get("schema_version", 0))
        if schema_version == 1 and (
            str(values.get("direction", "")).strip()
            or _finite_float(values.get("distance", 0.0)) != 0.0
            or _finite_float(values.get("degree", 0.0)) != 0.0
            or bool(values.get("has_x", False))
            or bool(values.get("has_y", False))
            or bool(values.get("has_yaw", False))
            or _finite_float(values.get("x", 0.0) or 0.0) != 0.0
            or _finite_float(values.get("y", 0.0) or 0.0) != 0.0
            or _finite_float(values.get("yaw", 0.0) or 0.0) != 0.0
        ):
            raise ValueError("navigation parameters require WorkflowStep schema_version 2")
    return CanonicalWorkflowStep(
        schema_version=schema_version,
        skill_name=str(values.get("skill_name", "")),
        target_name=str(values.get("target_name", "")),
        container_name=str(values.get("container_name", "")),
        place_name=str(values.get("place_name", "")),
        motion_direction=str(values.get("motion_direction", "")),
        motion_distance=_finite_float(values.get("motion_distance", 0.0)),
        arm_side=str(values.get("arm_side", "")),
        imitation_duration_sec=_finite_float(values.get("imitation_duration_sec", 0.0)),
        timeout_sec=_finite_float(values.get("timeout_sec", 0.0)),
        direction=str(values.get("direction", "")) if schema_version == 2 else "",
        distance=_finite_float(values.get("distance", 0.0)) if schema_version == 2 else 0.0,
        degree=_finite_float(values.get("degree", 0.0)) if schema_version == 2 else 0.0,
        x=_optional_workflow_coordinate(values, "x") if schema_version == 2 else None,
        y=_optional_workflow_coordinate(values, "y") if schema_version == 2 else None,
        yaw=_optional_workflow_coordinate(values, "yaw") if schema_version == 2 else None,
    )


def normalize_workflow_steps(steps: Sequence[Any], *, max_steps: int = 16) -> tuple[CanonicalWorkflowStep, ...]:
    if not steps:
        raise ValueError("workflow_steps must not be empty")
    if len(steps) > max_steps:
        raise ValueError(f"workflow_steps exceeds maximum of {max_steps}")
    return tuple(normalize_workflow_step(step) for step in steps)


def workflow_digest_preimage(
    *,
    root_task_id: str,
    task_budget: Any,
    expected_registry_epoch: str,
    expected_registry_generation: int,
    expected_registry_digest: str,
    workflow_steps: Sequence[Any],
) -> dict[str, Any]:
    if not root_task_id.strip():
        raise ValueError("root_task_id must be non-empty")
    if not expected_registry_epoch or not expected_registry_digest:
        raise ValueError("expected registry identity must be complete")
    if expected_registry_generation <= 0:
        raise ValueError("expected_registry_generation must be positive")
    steps = normalize_workflow_steps(workflow_steps)
    return {
        "schema_version": 1,
        "root_task_id": root_task_id,
        "task_budget": _task_budget_dict(task_budget),
        "expected_registry_epoch": expected_registry_epoch,
        "expected_registry_generation": int(expected_registry_generation),
        "expected_registry_digest": expected_registry_digest,
        "workflow_steps": [step.to_dict() for step in steps],
    }


def compute_workflow_digest(**kwargs: Any) -> str:
    payload = workflow_digest_preimage(**kwargs)
    return sha256_text(to_canonical_json(payload))


__all__ = [
    "CanonicalWorkflowStep",
    "WORKFLOW_STEP_COMMON_FIELDS",
    "WORKFLOW_STEP_NAVIGATION_FIELDS",
    "compute_workflow_digest",
    "normalize_workflow_step",
    "normalize_workflow_steps",
    "validate_workflow_steps",
    "workflow_digest_preimage",
]


def _task_budget_dict(task_budget: Any) -> dict[str, Any]:
    if is_dataclass(task_budget):
        task_budget = asdict(task_budget)
    started_at = _value(task_budget, "started_at")
    deadline = _value(task_budget, "deadline")
    schema_version = int(_value(task_budget, "schema_version"))
    if schema_version != 1:
        raise ValueError("TaskBudget.schema_version must be 1")
    return {
        "started_at": _time_dict(started_at),
        "deadline": _time_dict(deadline),
    }


def _time_dict(value: Any) -> dict[str, int]:
    sec = int(_value(value, "sec"))
    nanosec = int(_value(value, "nanosec"))
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("invalid builtin time")
    return {"sec": sec, "nanosec": nanosec}


def _value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value[key]
    return getattr(value, key)


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("number must be finite")
    return 0.0 if result == 0.0 else result


def _optional_workflow_coordinate(values: Mapping[str, Any], field_name: str) -> float | None:
    presence_name = f"has_{field_name}"
    presence = values.get(presence_name)
    if presence is None:
        return _finite_float(values[field_name]) if field_name in values and values[field_name] is not None else None
    if not isinstance(presence, bool):
        raise ValueError(f"WorkflowStep.{presence_name} must be a boolean")
    value = values.get(field_name, 0.0)
    normalized = _finite_float(value)
    if not presence:
        if normalized != 0.0:
            raise ValueError(f"WorkflowStep.{field_name} must be zero when {presence_name} is false")
        return None
    return normalized
