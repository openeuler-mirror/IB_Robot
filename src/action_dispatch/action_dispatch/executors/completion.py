"""Frozen pure-Python execution control envelope.

This module defines the immutable data model for asynchronous action execution
completion. It is deliberately free of any rclpy, ROS message, benchmark
runtime, sim backend or concrete executor imports so it can be loaded and
unit-tested in non-ROS contexts.

completion-aware executor contract boundary:
- ``CompletionStatus`` only captures whether the transport/control call
  returned normally. ``completed`` does NOT mean benchmark task success;
  goal/terminal/reward/success judgements belong to a later Work Package.
- No reward, is_success, terminated, truncated, suite, task, report or LIBERO
  fields are defined here.
- IDs (correlation_id, episode_id, step_id) are only carried; this module
  never creates environment episode/step identities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class CompletionStatus(str, Enum):
    """Outcome of a single action submission.

    ``COMPLETED`` means the call returned normally at the transport level; it
    does not imply benchmark task success. ``FAILED`` means the executor or
    environment reported a definitive failure. ``UNCERTAIN`` means the outcome
    could not be determined (e.g. timeout) and must be treated as fail-closed.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


def _validate_correlation_id(value: object) -> str:
    """Return a non-empty string correlation id, or raise."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"correlation_id must be a non-empty string, got {value!r}")
    return value


def _validate_optional_id(name: str, value: object) -> int | None:
    """Validate that an optional id is a non-negative int (bool rejected)."""
    if value is None:
        return None
    # bool is a subclass of int; reject it explicitly so True/False do not
    # masquerade as 1/0.
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative int or None, got bool {value!r}")
    if not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative int or None, got {type(value).__name__} {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _validate_optional_timestamp_ns(name: str, value: object) -> int | None:
    """Validate that an optional observation timestamp is a positive int (bool rejected)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive int (nanoseconds) or None, got bool {value!r}")
    if not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int (nanoseconds) or None, got {type(value).__name__} {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int (nanoseconds), got {value}")
    return value


def _freeze_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Freeze a mapping into a read-only MappingProxyType, preserving unknown keys."""
    if value is None:
        return MappingProxyType({})
    if isinstance(value, Mapping):
        # Copy the underlying dict so the caller's mapping cannot be mutated
        # after the frozen wrapper is created. Unknown keys are preserved.
        return MappingProxyType(dict(value))
    raise ValueError(f"{name} must be a Mapping or None, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Immutable context attached to a single action submission.

    The dispatcher creates a fresh ``ExecutionContext`` (with a new
    ``correlation_id``) for every final action submission. Optional
    ``episode_id``/``expected_step_id`` are only carried when a future
    generic environment runtime provides them; completion-aware executor contract does not generate them.
    """

    correlation_id: str
    episode_id: int | None = None
    expected_step_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "correlation_id", _validate_correlation_id(self.correlation_id))
        object.__setattr__(self, "episode_id", _validate_optional_id("episode_id", self.episode_id))
        object.__setattr__(self, "expected_step_id", _validate_optional_id("expected_step_id", self.expected_step_id))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class ExecutionCompletion:
    """Immutable completion record for a single action submission.

    ``status=completed`` only means the call returned normally; it does not
    mean benchmark task success. Reward, terminal, success and metrics are
    intentionally not present here.

    The optional ``observation_timestamp_ns`` is a positive integer in
    nanoseconds; for matching ``COMPLETED`` under ``wait_for_feedback`` it is
    required and is used as the timestamp for the next inference request.
    """

    correlation_id: str
    status: CompletionStatus
    observation_timestamp_ns: int | None = None
    episode_id: int | None = None
    step_id: int | None = None
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "correlation_id", _validate_correlation_id(self.correlation_id))
        if not isinstance(self.status, CompletionStatus):
            raise ValueError(
                f"status must be a CompletionStatus member, got {type(self.status).__name__} {self.status!r}"
            )
        object.__setattr__(
            self,
            "observation_timestamp_ns",
            _validate_optional_timestamp_ns("observation_timestamp_ns", self.observation_timestamp_ns),
        )
        object.__setattr__(self, "episode_id", _validate_optional_id("episode_id", self.episode_id))
        object.__setattr__(self, "step_id", _validate_optional_id("step_id", self.step_id))
        if not isinstance(self.message, str):
            raise ValueError(f"message must be a string, got {type(self.message).__name__}")
        object.__setattr__(self, "details", _freeze_mapping(self.details, "details"))


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Immutable receipt returned by ``ActionExecutor.submit``.

    If ``accepted`` is True the executor has taken ownership of the action and
    will eventually deliver a completion (either immediately via
    ``immediate_completion`` or later via ``drain_completions``). If
    ``accepted`` is False the submission was rejected; the dispatcher must
    fail-closed and must not retry the same action.

    A rejected receipt must not carry an immediate ``completed`` completion.
    The receipt correlation id must match any immediate completion correlation.
    """

    correlation_id: str
    accepted: bool
    message: str = ""
    immediate_completion: ExecutionCompletion | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "correlation_id", _validate_correlation_id(self.correlation_id))
        if not isinstance(self.accepted, bool):
            raise ValueError(f"accepted must be a bool, got {type(self.accepted).__name__}")
        if not isinstance(self.message, str):
            raise ValueError(f"message must be a string, got {type(self.message).__name__}")
        if self.immediate_completion is not None:
            if not isinstance(self.immediate_completion, ExecutionCompletion):
                raise ValueError(
                    "immediate_completion must be an ExecutionCompletion or None, "
                    f"got {type(self.immediate_completion).__name__}"
                )
            if self.immediate_completion.correlation_id != self.correlation_id:
                raise ValueError(
                    "immediate_completion correlation_id must match receipt correlation_id: "
                    f"{self.immediate_completion.correlation_id!r} != {self.correlation_id!r}"
                )
            if not self.accepted and self.immediate_completion.status is CompletionStatus.COMPLETED:
                raise ValueError("rejected receipt must not carry an immediate completed completion")
