"""UI-neutral comparison of two offline trace analysis results."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Literal, TextIO

from ._span_hierarchy import build_span_hierarchy
from .analysis import AnalysisResult
from .model import NumericSummary
from .serialization import to_js_safe
from .statistics import DEFAULT_METRIC_PRIORITY, summarize

ComparisonStatistic = Literal["minimum", "p50", "p95", "p99", "maximum", "mean"]
COMPARISON_STATISTICS = ("minimum", "p50", "p95", "p99", "maximum", "mean")
DEFAULT_MAX_FLAME_NODES = 10_000


@dataclass(frozen=True, slots=True)
class StatisticDelta:
    absolute_delta_ms: float | None
    percent_delta: float | None

    @property
    def absolute_ms(self) -> float | None:
        return self.absolute_delta_ms

    @property
    def relative_percent(self) -> float | None:
        return self.percent_delta


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric: str
    baseline: NumericSummary
    candidate: NumericSummary
    baseline_count: int
    candidate_count: int
    deltas: dict[str, StatisticDelta]
    regression: bool


@dataclass(frozen=True, slots=True)
class ComparisonHistogramBucket:
    index: int
    start: float
    end: float
    baseline_count: int
    candidate_count: int


@dataclass(frozen=True, slots=True)
class ComparisonHistogram:
    metric: str | None
    unit: Literal["ms"]
    minimum: float | None
    maximum: float | None
    bin_count: int
    baseline_count: int
    candidate_count: int
    buckets: list[ComparisonHistogramBucket] = field(default_factory=list)


FlameDiffStatus = Literal["improved", "unchanged", "regressed", "added", "removed"]


@dataclass(frozen=True, slots=True)
class FlameDiffEntry:
    path_id: str
    parent_path_id: str
    frame: tuple[str, str, str]
    depth: int
    baseline_value_ns: int
    candidate_value_ns: int
    baseline_self_value_ns: int
    candidate_self_value_ns: int
    absolute_delta_ns: int
    percent_delta: float | None
    self_absolute_delta_ns: int
    self_percent_delta: float | None
    status: FlameDiffStatus


@dataclass(frozen=True, slots=True)
class TraceComparisonProjection:
    statistic: ComparisonStatistic
    relative_threshold_percent: float
    absolute_threshold_ms: float
    metrics: list[MetricComparison]
    histogram: ComparisonHistogram
    flame_diff: list[FlameDiffEntry]
    coverage_warnings: list[str]
    count_warnings: list[str]
    comparable: bool
    blocking_reasons: list[str]
    has_regression: bool

    def to_dict(self) -> dict[str, Any]:
        return to_js_safe(asdict(self))


def _is_metric_name(name: Any) -> bool:
    if not isinstance(name, str) or not name.endswith("_ms") or name.endswith("_source"):
        return False
    return not {"id", "ids"}.intersection(name.removesuffix("_ms").split("_"))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_values(result: AnalysisResult) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for row in result.request_rows:
        for name, value in row.items():
            if not _is_metric_name(name):
                continue
            number = _finite_number(value)
            if number is not None:
                values.setdefault(name, []).append(number)
    return values


def _percent_delta(baseline: float | int, candidate: float | int) -> float | None:
    if baseline == 0:
        return 0.0 if candidate == 0 else None
    return (candidate - baseline) / abs(baseline) * 100.0


def _statistic_delta(baseline: float | None, candidate: float | None) -> StatisticDelta:
    if baseline is None or candidate is None:
        return StatisticDelta(None, None)
    return StatisticDelta(candidate - baseline, _percent_delta(baseline, candidate))


def _default_metric(metrics: set[str]) -> str | None:
    priority = {metric: index for index, metric in enumerate(DEFAULT_METRIC_PRIORITY)}
    return min(metrics, key=lambda metric: (priority.get(metric, len(priority)), metric)) if metrics else None


def _histogram(
    metric: str | None,
    baseline_values: list[float],
    candidate_values: list[float],
    bins: int,
) -> ComparisonHistogram:
    all_values = baseline_values + candidate_values
    if not all_values:
        return ComparisonHistogram(metric, "ms", None, None, 0, 0, 0)
    minimum = min(all_values)
    maximum = max(all_values)
    bin_count = min(bins, len(all_values))
    value_range = maximum - minimum
    if minimum == maximum or not math.isfinite(value_range):
        bin_count = 1
    width = value_range / bin_count if bin_count > 1 else 0.0
    if bin_count > 1 and width == 0.0:
        bin_count = 1

    baseline_counts = [0] * bin_count
    candidate_counts = [0] * bin_count

    def add(values: list[float], counts: list[int]) -> None:
        for value in values:
            index = bin_count - 1 if value == maximum or bin_count == 1 else int((value - minimum) / width)
            counts[min(index, bin_count - 1)] += 1

    add(baseline_values, baseline_counts)
    add(candidate_values, candidate_counts)
    buckets = [
        ComparisonHistogramBucket(
            index=index,
            start=minimum if index == 0 else minimum + width * index,
            end=maximum if index == bin_count - 1 else minimum + width * (index + 1),
            baseline_count=baseline_counts[index],
            candidate_count=candidate_counts[index],
        )
        for index in range(bin_count)
    ]
    return ComparisonHistogram(
        metric,
        "ms",
        minimum,
        maximum,
        bin_count,
        len(baseline_values),
        len(candidate_values),
        buckets,
    )


@dataclass(slots=True)
class _CompactProfileNode:
    path_id: str
    parent_path_id: str
    frame: tuple[str, str, str]
    depth: int
    self_value_ns: int = 0
    value_ns: int = 0


def _compact_path_id(parent_path_id: str, frame: tuple[str, str, str]) -> str:
    payload = json.dumps((parent_path_id, frame), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _profile_nodes(result: AnalysisResult, max_nodes: int) -> tuple[dict[str, _CompactProfileNode], bool]:
    hierarchy = build_span_hierarchy(result.spans)
    nodes_by_id = hierarchy.by_id
    ordered_nodes = []
    stack = list(reversed(hierarchy.roots))
    while stack:
        occurrence_id = stack.pop()
        node = nodes_by_id[occurrence_id]
        ordered_nodes.append(node)
        stack.extend(reversed(node.child_ids))
    entries: dict[str, _CompactProfileNode] = {}
    entry_by_occurrence: dict[str, str] = {}
    truncated = False
    for node in ordered_nodes:
        parent_path_id = entry_by_occurrence.get(node.parent_id, "")
        if node.parent_id and not parent_path_id:
            entry_by_occurrence[node.occurrence_id] = ""
            truncated = True
            continue
        frame = (node.span.component_id, node.span.name, node.span.origin)
        path_id = _compact_path_id(parent_path_id, frame)
        entry = entries.get(path_id)
        if entry is None:
            if len(entries) >= max_nodes:
                entry_by_occurrence[node.occurrence_id] = ""
                truncated = True
                continue
            entry = _CompactProfileNode(
                path_id,
                parent_path_id,
                frame,
                entries[parent_path_id].depth + 1 if parent_path_id else 0,
            )
            entries[path_id] = entry
        elif entry.parent_path_id != parent_path_id or entry.frame != frame:
            raise RuntimeError("compact flame path identity collision")
        entry_by_occurrence[node.occurrence_id] = path_id
        if node.valid_for_weight and node.uncovered_wall_ns:
            entry.self_value_ns += node.uncovered_wall_ns

    for entry in sorted(entries.values(), key=lambda item: item.depth, reverse=True):
        entry.value_ns += entry.self_value_ns
        if entry.parent_path_id:
            entries[entry.parent_path_id].value_ns += entry.value_ns
    return entries, truncated


def _flame_diff(
    baseline: AnalysisResult,
    candidate: AnalysisResult,
    max_nodes: int,
) -> tuple[list[FlameDiffEntry], bool]:
    baseline_nodes, baseline_truncated = _profile_nodes(baseline, max_nodes)
    candidate_nodes, candidate_truncated = _profile_nodes(candidate, max_nodes)
    entries = []
    path_ids = baseline_nodes.keys() | candidate_nodes.keys()
    ordered_path_ids = sorted(
        path_ids,
        key=lambda item: (
            (baseline_nodes.get(item) or candidate_nodes[item]).depth,
            (baseline_nodes.get(item) or candidate_nodes[item]).parent_path_id,
            (baseline_nodes.get(item) or candidate_nodes[item]).frame,
            item,
        ),
    )
    union_truncated = len(ordered_path_ids) > max_nodes
    for path_id in ordered_path_ids[:max_nodes]:
        baseline_node = baseline_nodes.get(path_id)
        candidate_node = candidate_nodes.get(path_id)
        template = baseline_node or candidate_node
        assert template is not None
        baseline_value = baseline_node.value_ns if baseline_node is not None else 0
        baseline_self = baseline_node.self_value_ns if baseline_node is not None else 0
        candidate_value = candidate_node.value_ns if candidate_node is not None else 0
        candidate_self = candidate_node.self_value_ns if candidate_node is not None else 0
        if baseline_node is None:
            status: FlameDiffStatus = "added"
        elif candidate_node is None:
            status = "removed"
        elif candidate_value < baseline_value:
            status = "improved"
        elif candidate_value > baseline_value:
            status = "regressed"
        else:
            status = "unchanged"
        entries.append(
            FlameDiffEntry(
                path_id=path_id,
                parent_path_id=template.parent_path_id,
                frame=template.frame,
                depth=template.depth,
                baseline_value_ns=baseline_value,
                candidate_value_ns=candidate_value,
                baseline_self_value_ns=baseline_self,
                candidate_self_value_ns=candidate_self,
                absolute_delta_ns=candidate_value - baseline_value,
                percent_delta=_percent_delta(baseline_value, candidate_value),
                self_absolute_delta_ns=candidate_self - baseline_self,
                self_percent_delta=_percent_delta(baseline_self, candidate_self),
                status=status,
            )
        )
    return entries, baseline_truncated or candidate_truncated or union_truncated


class TraceComparisonService:
    def compare(
        self,
        baseline: AnalysisResult,
        candidate: AnalysisResult,
        *,
        statistic: ComparisonStatistic = "p95",
        relative_percent: float | None = None,
        absolute_ms: float | None = None,
        metric: str | None = None,
        bins: int = 30,
        max_flame_nodes: int = DEFAULT_MAX_FLAME_NODES,
    ) -> TraceComparisonProjection:
        if statistic not in COMPARISON_STATISTICS:
            raise ValueError(f"unsupported comparison statistic: {statistic}")
        relative_threshold = 0.0 if relative_percent is None else float(relative_percent)
        absolute_threshold = 0.0 if absolute_ms is None else float(absolute_ms)
        if not math.isfinite(relative_threshold) or relative_threshold < 0:
            raise ValueError("relative threshold must be a finite non-negative number")
        if not math.isfinite(absolute_threshold) or absolute_threshold < 0:
            raise ValueError("absolute threshold must be a finite non-negative number")
        if not 1 <= bins <= 100:
            raise ValueError("bins must be between 1 and 100")
        if max_flame_nodes < 1:
            raise ValueError("max_flame_nodes must be positive")

        baseline_values = _metric_values(baseline)
        candidate_values = _metric_values(candidate)
        metric_names = baseline_values.keys() | candidate_values.keys()
        shared_metrics = baseline_values.keys() & candidate_values.keys()
        selected_metric = metric or _default_metric(shared_metrics or metric_names)
        if selected_metric is not None:
            metric_names.add(selected_metric)

        metric_comparisons = []
        coverage_warnings = []
        count_warnings = []
        blocking_reasons = []
        if not baseline.request_rows:
            blocking_reasons.append("baseline has no request rows")
        if not candidate.request_rows:
            blocking_reasons.append("candidate has no request rows")
        if len(baseline.request_rows) != len(candidate.request_rows):
            count_warnings.append(
                f"request count differs: baseline={len(baseline.request_rows)}, candidate={len(candidate.request_rows)}"
            )
        if baseline.coverage != candidate.coverage:
            coverage_warnings.append("analysis coverage metadata differs between baseline and candidate")
        if not metric_names:
            blocking_reasons.append("no comparable latency metrics are available")

        for name in sorted(metric_names):
            baseline_summary = summarize(baseline_values.get(name, []))
            candidate_summary = summarize(candidate_values.get(name, []))
            if baseline_summary.count == 0:
                coverage_warnings.append(f"metric {name} is missing from the baseline")
                if name == selected_metric:
                    blocking_reasons.append(f"required metric {name} is missing from the baseline")
            if candidate_summary.count == 0:
                coverage_warnings.append(f"metric {name} is missing from the candidate")
                if name == selected_metric:
                    blocking_reasons.append(f"required metric {name} is missing from the candidate")
            if baseline_summary.count and candidate_summary.count and baseline_summary.count != candidate_summary.count:
                count_warnings.append(
                    f"metric {name} count differs: baseline={baseline_summary.count}, candidate={candidate_summary.count}"
                )
            deltas = {
                field_name: _statistic_delta(
                    getattr(baseline_summary, field_name),
                    getattr(candidate_summary, field_name),
                )
                for field_name in COMPARISON_STATISTICS
            }
            selected_delta = deltas[statistic]
            baseline_statistic = getattr(baseline_summary, statistic)
            candidate_statistic = getattr(candidate_summary, statistic)
            relative_exceeded = bool(
                baseline_statistic is not None
                and candidate_statistic is not None
                and (
                    (baseline_statistic == 0 and candidate_statistic > 0)
                    or (selected_delta.percent_delta is not None and selected_delta.percent_delta > relative_threshold)
                )
            )
            regression = bool(
                selected_delta.absolute_delta_ms is not None
                and selected_delta.absolute_delta_ms > absolute_threshold
                and relative_exceeded
            )
            metric_comparisons.append(
                MetricComparison(
                    name,
                    baseline_summary,
                    candidate_summary,
                    baseline_summary.count,
                    candidate_summary.count,
                    deltas,
                    regression,
                )
            )

        histogram = _histogram(
            selected_metric,
            baseline_values.get(selected_metric, []) if selected_metric else [],
            candidate_values.get(selected_metric, []) if selected_metric else [],
            bins,
        )
        flame_diff, flame_truncated = _flame_diff(baseline, candidate, max_flame_nodes)
        if flame_truncated:
            coverage_warnings.append(
                f"flame profile exceeds the {max_flame_nodes}-node comparison limit; diff is truncated"
            )
        return TraceComparisonProjection(
            statistic=statistic,
            relative_threshold_percent=relative_threshold,
            absolute_threshold_ms=absolute_threshold,
            metrics=metric_comparisons,
            histogram=histogram,
            flame_diff=flame_diff,
            coverage_warnings=coverage_warnings,
            count_warnings=count_warnings,
            comparable=not blocking_reasons,
            blocking_reasons=blocking_reasons,
            has_regression=any(item.regression for item in metric_comparisons),
        )


def render_trace_comparison(comparison: TraceComparisonProjection, stream: TextIO) -> None:
    stream.write(
        f"Trace Comparison ({comparison.statistic}, >{comparison.relative_threshold_percent:g}% and "
        f">{comparison.absolute_threshold_ms:g}ms)\n"
    )
    stream.write(f"Comparable: {'yes' if comparison.comparable else 'no'}\n")
    for reason in comparison.blocking_reasons:
        stream.write(f"Blocking reason: {reason}\n")
    for metric in comparison.metrics:
        baseline_value = getattr(metric.baseline, comparison.statistic)
        candidate_value = getattr(metric.candidate, comparison.statistic)
        delta = metric.deltas[comparison.statistic]
        status = "REGRESSION" if metric.regression else "ok"
        baseline_text = "n/a" if baseline_value is None else f"{baseline_value:.3f}ms"
        candidate_text = "n/a" if candidate_value is None else f"{candidate_value:.3f}ms"
        delta_text = "n/a" if delta.absolute_delta_ms is None else f"{delta.absolute_delta_ms:+.3f}ms"
        if delta.percent_delta is not None:
            percent_text = f"{delta.percent_delta:+.2f}%"
        elif baseline_value == 0 and candidate_value is not None and candidate_value > 0:
            percent_text = "unbounded"
        else:
            percent_text = "n/a"
        stream.write(
            f"{metric.metric:<32} {baseline_text:>12} -> {candidate_text:<12} "
            f"{delta_text:>12} {percent_text:>10}  {status}\n"
        )
    for warning in comparison.coverage_warnings:
        stream.write(f"Coverage warning: {warning}\n")
    for warning in comparison.count_warnings:
        stream.write(f"Count warning: {warning}\n")
    histogram = comparison.histogram
    if histogram.metric is None or histogram.minimum is None or histogram.maximum is None:
        stream.write("Histogram: unavailable\n")
    else:
        stream.write(
            f"Histogram: {histogram.metric}, bins={histogram.bin_count}, "
            f"range={histogram.minimum:g}..{histogram.maximum:g}ms, "
            f"baseline={histogram.baseline_count}, candidate={histogram.candidate_count}\n"
        )
    status_counts = Counter(entry.status for entry in comparison.flame_diff)
    stream.write(
        "Flame diff: "
        f"entries={len(comparison.flame_diff)}, "
        + ", ".join(
            f"{status}={status_counts[status]}" for status in ("added", "removed", "improved", "regressed", "unchanged")
        )
        + "\n"
    )
