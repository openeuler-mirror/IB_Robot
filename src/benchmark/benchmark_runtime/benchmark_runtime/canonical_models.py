"""Pure data models for durable benchmark evaluation reports.

The canonical report layer is deliberately provider agnostic.  These models
contain identifiers, scalar results, and references to artifacts; they never
own raw video frames or simulator state arrays.  Values are validated at the
boundary so a report cannot contain ambiguous JSON (duplicate keys,
non-finite numbers, or provider-specific objects).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, ClassVar

_VALID_ARTIFACT_STATUSES = frozenset({"available", "disabled", "failed", "missing", "pending"})
_VALID_ARTIFACT_SCOPES = frozenset({"episode", "task", "run"})
_VALID_RUN_STATUSES = frozenset({"running", "complete", "partial", "failed"})


def _string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _int(value: Any, name: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int (not bool)")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _number(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number" + (" or None" if allow_none else ""))
    return float(value)


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise ValueError(f"{name} keys must be strings")
    return result


def _sequence(value: Sequence[Any] | None, name: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    """Status of one native or canonical artifact reference."""

    ref: str
    kind: str
    scope: str
    required: bool = True
    status: str = "available"
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    VALID_STATUSES: ClassVar[frozenset[str]] = _VALID_ARTIFACT_STATUSES
    VALID_SCOPES: ClassVar[frozenset[str]] = _VALID_ARTIFACT_SCOPES

    def __post_init__(self) -> None:
        _string(self.ref, "ref")
        _string(self.kind, "kind")
        if self.scope not in self.VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(self.VALID_SCOPES)}")
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean")
        if self.status not in self.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(self.VALID_STATUSES)}")
        if self.error is not None:
            _string(self.error, "error")
        if self.status in {"failed", "missing"} and not self.error:
            raise ValueError("failed or missing artifacts require an error")
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def failed(self) -> bool:
        """Whether this artifact is a required artifact failure."""
        return self.required and self.status in {"failed", "missing"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "scope": self.scope,
            "required": self.required,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class CanonicalEpisodeRecord:
    """One durable, finalized episode fact.

    ``native_result_available`` controls whether the episode participates in
    native success-rate denominators.  An infrastructure failure without a
    native result therefore cannot be turned into ``is_success=false``.
    Artifact failures are carried separately and never change this field.
    """

    run_id: str
    suite: str
    task_id: int
    task_name: str
    episode_index: int
    seed: int
    native_result_available: bool
    is_success: bool | None
    executed_steps: int = 0
    duration_s: float = 0.0
    requested_init_state_id: int | None = None
    actual_init_state_id: int | None = None
    episode_id: int | None = None
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = ""
    infrastructure_error: str | None = None
    artifacts: Sequence[ArtifactStatus] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _string(self.run_id, "run_id")
        _string(self.suite, "suite")
        _int(self.task_id, "task_id")
        _string(self.task_name, "task_name")
        _int(self.episode_index, "episode_index")
        _int(self.seed, "seed")
        if not isinstance(self.native_result_available, bool):
            raise ValueError("native_result_available must be a boolean")
        if self.is_success is not None and not isinstance(self.is_success, bool):
            raise ValueError("is_success must be a boolean or None")
        if self.native_result_available and self.is_success is None:
            raise ValueError("native result episodes require is_success")
        _int(self.executed_steps, "executed_steps")
        _number(self.duration_s, "duration_s")
        for value, name in (
            (self.requested_init_state_id, "requested_init_state_id"),
            (self.actual_init_state_id, "actual_init_state_id"),
            (self.episode_id, "episode_id"),
        ):
            if value is not None:
                _int(value, name)
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminated and truncated must be booleans")
        _string(self.termination_reason, "termination_reason", allow_empty=True)
        if self.infrastructure_error is not None:
            _string(self.infrastructure_error, "infrastructure_error")
        artifacts = _sequence(self.artifacts, "artifacts")
        if any(not isinstance(item, ArtifactStatus) for item in artifacts):
            raise ValueError("artifacts must contain ArtifactStatus values")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def native_model_success(self) -> bool | None:
        """Alias used by reducers and report consumers."""
        return self.is_success if self.native_result_available else None

    @property
    def artifact_failures(self) -> tuple[ArtifactStatus, ...]:
        return tuple(artifact for artifact in self.artifacts if artifact.failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "suite": self.suite,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "requested_init_state_id": self.requested_init_state_id,
            "actual_init_state_id": self.actual_init_state_id,
            "executed_steps": self.executed_steps,
            "duration_s": self.duration_s,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "native_result_available": self.native_result_available,
            "is_success": self.is_success,
            "infrastructure_error": self.infrastructure_error,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class CanonicalCounters:
    """Frozen reducer counters for one run."""

    planned_episodes: int
    started_episodes: int
    completed_native_episodes: int
    successful_native_episodes: int
    native_failed_episodes: int
    infrastructure_error_episodes: int
    artifact_failure_episodes: int

    def __post_init__(self) -> None:
        for name in (
            "planned_episodes",
            "started_episodes",
            "completed_native_episodes",
            "successful_native_episodes",
            "native_failed_episodes",
            "infrastructure_error_episodes",
            "artifact_failure_episodes",
        ):
            _int(getattr(self, name), name)
        if self.completed_native_episodes != self.successful_native_episodes + self.native_failed_episodes:
            raise ValueError("native counters do not reconcile")
        if self.started_episodes < self.completed_native_episodes:
            raise ValueError("started_episodes cannot be less than completed_native_episodes")

    @property
    def native_model_success_rate(self) -> float | None:
        if self.completed_native_episodes == 0:
            return None
        return self.successful_native_episodes / self.completed_native_episodes

    @property
    def planned_success_rate(self) -> float | None:
        if self.planned_episodes == 0:
            return None
        return self.successful_native_episodes / self.planned_episodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned_episodes": self.planned_episodes,
            "started_episodes": self.started_episodes,
            "completed_native_episodes": self.completed_native_episodes,
            "successful_native_episodes": self.successful_native_episodes,
            "native_failed_episodes": self.native_failed_episodes,
            "infrastructure_error_episodes": self.infrastructure_error_episodes,
            "artifact_failure_episodes": self.artifact_failure_episodes,
            "native_model_success_rate": self.native_model_success_rate,
            "planned_success_rate": self.planned_success_rate,
        }


@dataclass(frozen=True, slots=True)
class CanonicalTaskSummary:
    """Reducer output for one task."""

    task_id: int
    task_name: str
    planned_episodes: int
    records: int
    completed_native_episodes: int
    successful_native_episodes: int
    artifact_failure_episodes: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "planned_episodes": self.planned_episodes,
            "records": self.records,
            "completed_native_episodes": self.completed_native_episodes,
            "successful_native_episodes": self.successful_native_episodes,
            "artifact_failure_episodes": self.artifact_failure_episodes,
            "native_model_success_rate": (
                self.successful_native_episodes / self.completed_native_episodes
                if self.completed_native_episodes
                else None
            ),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class CanonicalRunSummary:
    """Canonical summary payload written to ``summary.json``."""

    run_id: str
    suite: str
    run_status: str
    counters: CanonicalCounters
    tasks: Sequence[CanonicalTaskSummary] = ()
    artifact_failures: Sequence[ArtifactStatus] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    VALID_STATUSES: ClassVar[frozenset[str]] = _VALID_RUN_STATUSES

    def __post_init__(self) -> None:
        _string(self.run_id, "run_id")
        _string(self.suite, "suite")
        if self.run_status not in self.VALID_STATUSES:
            raise ValueError(f"run_status must be one of {sorted(self.VALID_STATUSES)}")
        if not isinstance(self.counters, CanonicalCounters):
            raise ValueError("counters must be CanonicalCounters")
        tasks = _sequence(self.tasks, "tasks")
        failures = _sequence(self.artifact_failures, "artifact_failures")
        if any(not isinstance(item, CanonicalTaskSummary) for item in tasks):
            raise ValueError("tasks must contain CanonicalTaskSummary values")
        if any(not isinstance(item, ArtifactStatus) for item in failures):
            raise ValueError("artifact_failures must contain ArtifactStatus values")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "artifact_failures", failures)
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        counters = self.counters.to_dict()
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "suite": self.suite,
            "run_status": self.run_status,
            "status": self.run_status,
            "counters": counters,
            "rates": {
                "native_model_success_rate": counters["native_model_success_rate"],
                "planned_success_rate": counters["planned_success_rate"],
            },
            "tasks": [task.to_dict() for task in self.tasks],
            "artifact_failures": [artifact.to_dict() for artifact in self.artifact_failures],
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class DiskEstimate:
    """Conservative byte estimate and result of a disk-space check."""

    planned_episodes: int
    max_frames: int
    video_bytes: int
    state_bytes: int
    report_overhead_bytes: int
    safety_factor: float
    estimated_bytes: int
    reserve_bytes: int
    required_bytes: int
    free_bytes: int | None = None
    passed: bool | None = None

    def __post_init__(self) -> None:
        for name in (
            "planned_episodes",
            "max_frames",
            "video_bytes",
            "state_bytes",
            "report_overhead_bytes",
            "estimated_bytes",
            "reserve_bytes",
            "required_bytes",
        ):
            _int(getattr(self, name), name)
        if not isinstance(self.safety_factor, int | float) or isinstance(self.safety_factor, bool):
            raise ValueError("safety_factor must be a number")
        if float(self.safety_factor) < 1.0 or not isfinite(float(self.safety_factor)):
            raise ValueError("safety_factor must be finite and at least 1")
        if self.free_bytes is not None:
            _int(self.free_bytes, "free_bytes")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned_episodes": self.planned_episodes,
            "max_frames": self.max_frames,
            "video_bytes": self.video_bytes,
            "state_bytes": self.state_bytes,
            "report_overhead_bytes": self.report_overhead_bytes,
            "safety_factor": float(self.safety_factor),
            "estimated_bytes": self.estimated_bytes,
            "reserve_bytes": self.reserve_bytes,
            "required_bytes": self.required_bytes,
            "free_bytes": self.free_bytes,
            "passed": self.passed,
        }
