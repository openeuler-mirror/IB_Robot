"""Frozen runtime data models for benchmark evaluation.

benchmark plugin contract scope: this module defines all public cross-module dataclasses used by the
benchmark runtime, adapter protocol, native report exporter and registry. It
does NOT import ``adapter``, ``registry`` or ``native_report`` (no reverse
dependency), does NOT import ``rclpy``, and does NOT implement reducers, file
IO, ROS serialization or the evaluator state machine.

All public dataclasses are ``@dataclass(frozen=True, slots=True)``. Mapping
fields are stored as read-only ``MappingProxyType`` views and list/tuple
sequence fields as ``tuple``. Numpy observation/action arrays are stored
as-is: models must not copy them or silently change dtype/shape at
construction.

Invalid values (negative IDs, empty names, unknown metric scope/reducer,
unknown artifact kind) fail-fast with ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from benchmark_runtime.io_descriptor import ObservationBatch

# --------------------------------------------------------------------------- #
# Frozen helpers
# --------------------------------------------------------------------------- #


def _freeze_mapping(value: Mapping[str, Any] | None) -> MappingProxyType:
    """Return a read-only MappingProxyType over a shallow copy of *value*."""
    if value is None:
        return MappingProxyType({})
    if isinstance(value, MappingProxyType):
        return value
    return MappingProxyType(dict(value))


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int (not bool)")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return value


def _require_optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, field_name)


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int (not bool)")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


# --------------------------------------------------------------------------- #
# Capabilities & environment config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BenchmarkCapabilities:
    """Adapter-declared capabilities.

    Each flag states whether the adapter supports the corresponding feature.
    Unknown features are preserved through ``options`` on
    :class:`BenchmarkEnvironmentConfig`, not here.
    """

    supports_render: bool
    supports_init_state: bool
    supports_native_artifact: bool
    supports_reward: bool
    supports_success: bool
    max_lane_count: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "supports_render",
            "supports_init_state",
            "supports_native_artifact",
            "supports_reward",
            "supports_success",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")
        if not isinstance(self.max_lane_count, int) or isinstance(self.max_lane_count, bool) or self.max_lane_count < 1:
            raise ValueError("max_lane_count must be a positive int")


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironmentConfig:
    """Environment configuration resolved from SSOT.

    ``benchmark_type`` and ``adapter_name`` mirror the SSOT ``benchmark.type``
    and ``benchmark.adapter`` keys. ``options`` preserves any unknown
    configuration without dropping keys.
    """

    benchmark_type: str
    adapter_name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.benchmark_type, "benchmark_type")
        _require_non_empty_str(self.adapter_name, "adapter_name")
        object.__setattr__(self, "options", _freeze_mapping(self.options))


# --------------------------------------------------------------------------- #
# Task & reset request/result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """A single task within a benchmark suite.

    ``task_id`` is the stable suite-local task identifier (non-negative).
    ``metadata`` preserves unknown task-level keys.
    """

    suite: str
    task_id: int
    name: str
    prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.suite, "suite")
        _require_non_negative_int(self.task_id, "task_id")
        _require_non_empty_str(self.name, "name")
        _require_non_empty_str(self.prompt, "prompt")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ResetRequest:
    """Request payload for ``BenchmarkAdapter.reset``.

    When ``use_init_state_id`` is ``False`` the adapter must ignore
    ``init_state_id``. ``seed`` and ``task_id`` must be non-negative.
    """

    suite: str
    task_id: int
    seed: int
    use_init_state_id: bool
    init_state_id: int = 0

    def __post_init__(self) -> None:
        _require_non_empty_str(self.suite, "suite")
        _require_non_negative_int(self.task_id, "task_id")
        _require_non_negative_int(self.seed, "seed")
        _require_non_negative_int(self.init_state_id, "init_state_id")


@dataclass(frozen=True, slots=True)
class ResetResult:
    """Result of a successful ``BenchmarkAdapter.reset``.

    ``observations`` maps observation keys to numpy arrays (not copied).
    ``task_prompt`` is the task language prompt preserved verbatim.
    ``metadata`` preserves native keys. The observation timestamp and the
    ``episode_id``/``step_id`` identity are owned by the environment node
    (ROS layer) and are NOT part of this Python model: adapters must not
    generate them. The generic environment runtime assigns and validates
    them to guarantee exactly-once semantics across LIBERO, Meta-World and
    any future adapter without per-adapter protocol state. The ROS service
    ``ResetBenchmark.srv`` keeps ``episode_id``/``step_id`` in its response
    and the environment node fills them from its own runtime state.
    """

    observations: ObservationBatch | Mapping[str, np.ndarray]
    task_prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_prompt, str):
            raise ValueError("task_prompt must be a string")
        if not isinstance(self.observations, ObservationBatch):
            object.__setattr__(self, "observations", _freeze_mapping(self.observations))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


# --------------------------------------------------------------------------- #
# Step result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of a successful ``BenchmarkAdapter.step``.

    ``observations`` maps observation keys to numpy arrays (not copied).
    ``reward`` is ``None`` when the benchmark has no reward signal — it must
    NOT be replaced by ``0.0``. ``is_success`` is ``None`` when task success
    is unknown — it must NOT be replaced by ``False``. ``terminated`` and
    ``truncated`` follow Gymnasium semantics and must not be merged.
    ``standard_metrics``, ``native_metrics`` and ``info`` are JSON-object
    mappings that preserve unknown keys.
    """

    observations: ObservationBatch | Mapping[str, np.ndarray]
    terminated: bool
    truncated: bool
    reward: float | None = None
    is_success: bool | None = None
    standard_metrics: Mapping[str, Any] = field(default_factory=dict)
    native_metrics: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reward is not None and (not isinstance(self.reward, int | float) or isinstance(self.reward, bool)):
            raise ValueError("reward must be None or a number")
        if not isinstance(self.observations, ObservationBatch):
            object.__setattr__(self, "observations", _freeze_mapping(self.observations))
        object.__setattr__(self, "standard_metrics", _freeze_mapping(self.standard_metrics))
        object.__setattr__(self, "native_metrics", _freeze_mapping(self.native_metrics))
        object.__setattr__(self, "info", _freeze_mapping(self.info))


# --------------------------------------------------------------------------- #
# Metrics & artifacts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """A single scalar metric value with aggregation metadata.

    ``scope`` is one of: ``step``, ``episode``, ``task``, ``suite``, ``run``.
    ``reducer`` is one of: ``mean``, ``sum``, ``max``, ``min``, ``rate``,
    ``last``, ``none``. ``higher_is_better`` may be ``None`` when direction
    is unknown.
    """

    name: str
    value: float | int | bool
    unit: str
    higher_is_better: bool | None
    reducer: str
    scope: str

    VALID_SCOPES: ClassVar[frozenset[str]] = frozenset({"step", "episode", "task", "suite", "run"})
    VALID_REDUCERS: ClassVar[frozenset[str]] = frozenset({"mean", "sum", "max", "min", "rate", "last", "none"})

    def __post_init__(self) -> None:
        _require_non_empty_str(self.name, "name")
        if not isinstance(self.unit, str):
            raise ValueError("unit must be a string")
        if self.scope not in self.VALID_SCOPES:
            raise ValueError(f"invalid metric scope '{self.scope}'; expected one of {sorted(self.VALID_SCOPES)}")
        if self.reducer not in self.VALID_REDUCERS:
            raise ValueError(f"invalid metric reducer '{self.reducer}'; expected one of {sorted(self.VALID_REDUCERS)}")


@dataclass(frozen=True, slots=True)
class NativeArtifact:
    """A native artifact produced by a benchmark run.

    ``kind`` is one of: ``video``, ``image``, ``trajectory``, ``log``,
    ``dataset``, ``native_report``, ``other``. ``scope`` is one of:
    ``step``, ``episode``, ``task``, ``suite``, ``run``. The optional
    ``suite``/``task_id``/``episode_id`` references may be ``None``. ``metadata``
    preserves unknown keys.
    """

    name: str
    kind: str
    path: str
    mime_type: str
    scope: str
    suite: str | None = None
    task_id: int | None = None
    episode_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    VALID_KINDS: ClassVar[frozenset[str]] = frozenset(
        {"video", "image", "trajectory", "log", "dataset", "native_report", "other"}
    )
    VALID_SCOPES: ClassVar[frozenset[str]] = frozenset({"step", "episode", "task", "suite", "run"})

    def __post_init__(self) -> None:
        _require_non_empty_str(self.name, "name")
        _require_non_empty_str(self.kind, "kind")
        _require_non_empty_str(self.path, "path")
        _require_non_empty_str(self.mime_type, "mime_type")
        if self.kind not in self.VALID_KINDS:
            raise ValueError(f"invalid artifact kind '{self.kind}'; expected one of {sorted(self.VALID_KINDS)}")
        if self.scope not in self.VALID_SCOPES:
            raise ValueError(f"invalid artifact scope '{self.scope}'; expected one of {sorted(self.VALID_SCOPES)}")
        self.__class__._validate_optional_ref(self.suite, "suite")
        _require_optional_non_negative_int(self.task_id, "task_id")
        _require_optional_non_negative_int(self.episode_id, "episode_id")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @staticmethod
    def _validate_optional_ref(value: Any, field_name: str) -> None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field_name} must be None or a non-empty string")


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StepEvent:
    """A single step event emitted during a benchmark run.

    Carries the full identity (benchmark type, suite, task, episode, step),
    the environment-owned ``obs_timestamp_ns`` (positive nanoseconds), and the
    :class:`StepResult` produced by the adapter. The environment node is the
    only timestamp owner; the adapter never generates timestamps.
    """

    benchmark_type: str
    suite: str
    task_id: int
    episode_id: int
    step_id: int
    obs_timestamp_ns: int
    result: StepResult

    def __post_init__(self) -> None:
        _require_non_empty_str(self.benchmark_type, "benchmark_type")
        _require_non_empty_str(self.suite, "suite")
        _require_non_negative_int(self.task_id, "task_id")
        _require_non_negative_int(self.episode_id, "episode_id")
        _require_non_negative_int(self.step_id, "step_id")
        _require_positive_int(self.obs_timestamp_ns, "obs_timestamp_ns")


# --------------------------------------------------------------------------- #
# Episode result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Canonical episode result (V2 section 11.2 standard fields).

    ``success`` is ``None`` when task success is unknown (must NOT be
    replaced by ``False``). ``reward_sum`` and ``reward_max`` are ``None``
    when the benchmark has no reward. ``init_state_id`` is ``None`` when no
    explicit init state was used. ``task_id``, ``episode_index`` and ``seed``
    must be non-negative.
    """

    benchmark_type: str
    suite: str
    task_id: int
    task_name: str
    episode_index: int
    seed: int
    init_state_id: int | None
    success: bool | None
    reward_sum: float | None
    reward_max: float | None
    steps: int
    duration_s: float
    termination_reason: str
    error_category: str

    def __post_init__(self) -> None:
        _require_non_empty_str(self.benchmark_type, "benchmark_type")
        _require_non_empty_str(self.suite, "suite")
        _require_non_negative_int(self.task_id, "task_id")
        _require_non_empty_str(self.task_name, "task_name")
        _require_non_negative_int(self.episode_index, "episode_index")
        _require_non_negative_int(self.seed, "seed")
        _require_optional_non_negative_int(self.init_state_id, "init_state_id")
        _require_non_negative_int(self.steps, "steps")
        if not isinstance(self.duration_s, int | float) or isinstance(self.duration_s, bool):
            raise ValueError("duration_s must be a number")
        if float(self.duration_s) < 0.0:
            raise ValueError(f"duration_s must be non-negative, got {self.duration_s}")


# --------------------------------------------------------------------------- #
# Run manifest & summary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Created at run start; carries identity, output reference and plan metadata.

    ``run_id`` and ``benchmark_type`` must be non-empty. ``metadata`` and
    ``metrics`` preserve unknown keys for extensibility (git commit, versions,
    task/seed plan, etc.).
    """

    run_id: str
    benchmark_type: str
    status: str
    output_ref: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.run_id, "run_id")
        _require_non_empty_str(self.benchmark_type, "benchmark_type")
        _require_non_empty_str(self.status, "status")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "metrics", _freeze_mapping(self.metrics))


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Created at run end; carries final status and extensible metrics.

    ``run_id`` and ``benchmark_type`` must be non-empty. ``metadata`` and
    ``metrics`` preserve unknown keys (final aggregated results, etc.).
    """

    run_id: str
    benchmark_type: str
    status: str
    output_ref: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.run_id, "run_id")
        _require_non_empty_str(self.benchmark_type, "benchmark_type")
        _require_non_empty_str(self.status, "status")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "metrics", _freeze_mapping(self.metrics))
