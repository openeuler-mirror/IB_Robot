"""Strict deterministic JSON helpers for generic benchmark contracts.

This pure module deliberately has no ROS or provider dependencies. It rejects
non-standard constants and duplicate object keys so wire payloads have one
unambiguous interpretation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


class StrictJSONError(ValueError):
    """Raised when a generic benchmark JSON payload is not strict JSON."""


def freeze_json(value: Any) -> Any:
    """Return an immutable JSON-compatible value, rejecting other types."""
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJSONError(f"JSON object key must be a string, got {type(key).__name__}")
            frozen[key] = freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not __import__("math").isfinite(value):
            raise StrictJSONError("JSON numbers must be finite")
        return value
    raise StrictJSONError(f"unsupported JSON value type {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Convert immutable JSON-compatible values back to plain containers."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [thaw_json(item) for item in value]
    return value


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-standard JSON constant {value!r} is forbidden")


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate key {key!r} in JSON object")
        result[key] = value
    return result


def loads_strict(payload: str) -> Any:
    """Parse strict JSON with duplicate-key and non-finite rejection."""
    if not isinstance(payload, str):
        raise StrictJSONError(f"JSON payload must be a string, got {type(payload).__name__}")
    try:
        return json.loads(payload, object_pairs_hook=_object_from_pairs, parse_constant=_reject_constant)
    except StrictJSONError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise StrictJSONError(f"invalid JSON: {exc}") from exc


def dumps_strict(value: Any, *, indent: int | None = None) -> str:
    """Serialize a JSON-compatible value deterministically.

    ``indent=None`` preserves the compact wire/JSONL representation. A
    positive indent produces human-readable strict JSON without changing the
    parsed payload.
    """
    plain = thaw_json(freeze_json(value))
    separators = (",", ":") if indent is None else None
    return json.dumps(plain, sort_keys=True, separators=separators, indent=indent, allow_nan=False)


def require_exact_keys(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    """Reject missing or unknown object keys."""
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise StrictJSONError(f"{context} missing keys: {missing}")
    if unknown:
        raise StrictJSONError(f"{context} has unknown keys: {unknown}")


def require_json_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictJSONError(f"{context} must be a JSON object")
    return value


def require_json_array(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        raise StrictJSONError(f"{context} must be a JSON array")
    return value
