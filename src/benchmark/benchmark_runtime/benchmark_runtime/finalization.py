"""Immutable generic finalization payload models and strict ROS JSON fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from benchmark_runtime._strict_json import (
    StrictJSONError,
    dumps_strict,
    freeze_json,
    loads_strict,
    require_json_array,
    require_json_object,
)

_VALID_SCOPES = frozenset({"episode", "task", "run"})


class FinalizationJSONError(ValueError):
    """Raised when a FinalizeBenchmarkScope JSON field is malformed."""


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be None or a non-negative int (bool rejected)")
    return value


@dataclass(frozen=True, slots=True)
class FinalizationIdentity:
    """Generic scope identity; scope-specific requirements live on the payload."""

    run_id: str
    suite: str | None = None
    task_id: int | None = None
    episode_id: int | None = None
    episode_index: int | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.run_id, "run_id")
        if self.suite is not None:
            _non_empty_string(self.suite, "suite")
        _optional_non_negative_int(self.task_id, "task_id")
        _optional_non_negative_int(self.episode_id, "episode_id")
        _optional_non_negative_int(self.episode_index, "episode_index")


@dataclass(frozen=True, slots=True)
class FinalizationPayload:
    """Semantic request carried by ``FinalizeBenchmarkScope.srv``."""

    scope: str
    identity: FinalizationIdentity
    result: Mapping[str, Any] = field(default_factory=dict)
    termination_reason: str = ""
    error_category: str = ""
    partial: bool = False
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_VALID_SCOPES)}")
        if not isinstance(self.identity, FinalizationIdentity):
            raise ValueError("identity must be FinalizationIdentity")
        if not isinstance(self.result, Mapping):
            raise ValueError("result must be a JSON object mapping")
        try:
            frozen_result = freeze_json(self.result)
        except StrictJSONError as exc:
            raise ValueError(f"result is not strict JSON: {exc}") from exc
        object.__setattr__(self, "result", frozen_result)
        if not isinstance(self.termination_reason, str):
            raise ValueError("termination_reason must be a string")
        if not isinstance(self.error_category, str):
            raise ValueError("error_category must be a string")
        if not isinstance(self.partial, bool):
            raise ValueError("partial must be a boolean")
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        for ref in self.artifact_refs:
            _non_empty_string(ref, "artifact reference")
        if self.scope in {"episode", "task"}:
            if self.identity.suite is None:
                raise ValueError(f"suite is required for {self.scope} finalization")
            if self.identity.task_id is None:
                raise ValueError(f"task_id is required for {self.scope} finalization")
        if self.scope == "episode":
            if self.identity.episode_id is None:
                raise ValueError("episode_id is required for episode finalization")
            if self.identity.episode_index is None:
                raise ValueError("episode_index is required for episode finalization")


@dataclass(frozen=True, slots=True)
class FinalizationWirePayload:
    """Exact service request fields after deterministic JSON serialization."""

    scope: str
    identity_json: str
    result_json: str
    termination_reason: str
    error_category: str
    partial: bool
    artifact_refs_json: str


def _identity_to_dict(identity: FinalizationIdentity) -> dict[str, Any]:
    result: dict[str, Any] = {"run_id": identity.run_id}
    if identity.suite is not None:
        result["suite"] = identity.suite
    if identity.task_id is not None:
        result["task_id"] = identity.task_id
    if identity.episode_id is not None:
        result["episode_id"] = identity.episode_id
    if identity.episode_index is not None:
        result["episode_index"] = identity.episode_index
    return result


def finalization_payload_to_wire(payload: FinalizationPayload) -> FinalizationWirePayload:
    if not isinstance(payload, FinalizationPayload):
        raise FinalizationJSONError(f"payload must be FinalizationPayload, got {type(payload).__name__}")
    try:
        return FinalizationWirePayload(
            scope=payload.scope,
            identity_json=dumps_strict(_identity_to_dict(payload.identity)),
            result_json=dumps_strict(payload.result),
            termination_reason=payload.termination_reason,
            error_category=payload.error_category,
            partial=payload.partial,
            artifact_refs_json=dumps_strict(payload.artifact_refs),
        )
    except StrictJSONError as exc:
        raise FinalizationJSONError(str(exc)) from exc


def finalization_payload_from_wire(
    *,
    scope: str,
    identity_json: str,
    result_json: str,
    termination_reason: str,
    error_category: str,
    partial: bool,
    artifact_refs_json: str,
) -> FinalizationPayload:
    """Parse the exact generic service fields and reject ambiguous JSON."""
    try:
        identity_raw = require_json_object(loads_strict(identity_json), "identity_json")
        allowed_identity_keys = {"run_id", "suite", "task_id", "episode_id", "episode_index"}
        unknown = sorted(set(identity_raw) - allowed_identity_keys)
        if unknown:
            raise StrictJSONError(f"identity_json has unknown keys: {unknown}")
        if "run_id" not in identity_raw:
            raise StrictJSONError("identity_json missing keys: ['run_id']")
        result = require_json_object(loads_strict(result_json), "result_json")
        refs = require_json_array(loads_strict(artifact_refs_json), "artifact_refs_json")
        for ref in refs:
            if not isinstance(ref, str) or not ref.strip():
                raise StrictJSONError("each artifact reference must be a non-empty string")
        return FinalizationPayload(
            scope=scope,
            identity=FinalizationIdentity(**identity_raw),
            result=result,
            termination_reason=termination_reason,
            error_category=error_category,
            partial=partial,
            artifact_refs=tuple(refs),
        )
    except (StrictJSONError, TypeError, ValueError) as exc:
        raise FinalizationJSONError(str(exc)) from exc
