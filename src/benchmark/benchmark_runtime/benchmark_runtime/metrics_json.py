"""Deterministic JSON serialization helper for benchmark responses.

Benchmark runtime requires that all StepBenchmark JSON response fields
(``standard_metrics_json``, ``native_metrics_json``, ``info_json``) are
deterministic, valid JSON objects — including when they contain NumPy scalar
or array values from native adapter ``info``. This helper serializes a
mapping to a JSON object string with stable key order and NumPy-safe value
conversion.

This module does NOT import ``rclpy`` or any concrete benchmark.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np


class JSONSerializationError(ValueError):
    """Raised when a value cannot be deterministically serialized to JSON."""


def _convert(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _convert(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_convert(v) for v in value]
    if isinstance(value, np.ndarray):
        return _convert(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        if not np.isfinite(v):  # noqa: UP046 - guard before JSON NaN injection
            raise JSONSerializationError(f"non-finite float in metrics/info: {v}")
        return v
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise JSONSerializationError(f"non-finite float in metrics/info: {value}")
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return None
    raise JSONSerializationError(f"unsupported value type {type(value).__name__} in metrics/info")


def to_metrics_json(payload: Mapping[str, Any] | None) -> str:
    """Serialize a metrics/info mapping to a deterministic JSON object string.

    Always returns a JSON object (``{...}``), even for ``None`` or empty
    input (``"{}"``). Non-finite floats and unsupported types raise
    :class:`JSONSerializationError`.
    """
    if payload is None:
        return "{}"
    if not isinstance(payload, Mapping):
        raise JSONSerializationError(f"metrics/info payload must be a mapping, got {type(payload).__name__}")
    converted = _convert(payload)
    if not isinstance(converted, dict):
        raise JSONSerializationError("converted payload is not a dict")
    return json.dumps(converted, sort_keys=True, separators=(",", ":"))
