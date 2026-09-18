"""JSON-compatible serialization helpers for UI and other adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

JS_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_ABSOLUTE_NS_FIELDS = {
    "base_ns",
    "end_ns",
    "observed_end_ns",
    "receive_ns",
    "send_ns",
    "start_ns",
    "timestamp_ns",
}


def to_js_safe(value: Any, *, field_name: str = "") -> Any:
    """Return a JSON-compatible value that cannot lose integer precision in JS."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): to_js_safe(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_js_safe(item, field_name=field_name) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        if field_name in _ABSOLUTE_NS_FIELDS or abs(value) > JS_MAX_SAFE_INTEGER:
            return str(value)
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value
