"""Downstream scheduled-result identity validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def result_identity_error(
    action: str,
    result: object,
    expected_fields: Mapping[str, Any],
    *,
    require_higher_drained_generation: bool = False,
) -> str:
    """Return an error code when a terminal result does not match its request.

    The caller treats every mismatch as UNKNOWN because a terminal belonging to
    another request cannot prove the side-effect state of the expected request.
    """
    for field, expected in expected_fields.items():
        if getattr(result, field, None) != expected:
            return f"{action}_{field}_mismatch"
    if require_higher_drained_generation:
        closed = int(getattr(result, "closed_pipeline_generation", getattr(result, "closed_session_generation", 0)))
        drained = int(getattr(result, "drained_generation", 0))
        if closed <= 0 or drained <= closed:
            return f"{action}_drained_generation_invalid"
    return ""


__all__ = ["result_identity_error"]
