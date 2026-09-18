"""Small dependency-free statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .model import NumericSummary

DEFAULT_METRIC_PRIORITY = ("total_ms", "inference_ms")


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= ratio <= 1:
        raise ValueError("percentile ratio must be between zero and one")
    index = max(0, min(len(sorted_values) - 1, math.ceil(ratio * len(sorted_values)) - 1))
    return sorted_values[index]


def summarize(values: Iterable[float]) -> NumericSummary:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return NumericSummary(count=0)
    return NumericSummary(
        count=len(ordered),
        minimum=ordered[0],
        p50=percentile(ordered, 0.50),
        p95=percentile(ordered, 0.95),
        p99=percentile(ordered, 0.99),
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
    )
