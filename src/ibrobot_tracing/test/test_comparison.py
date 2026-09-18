import io
import json
import math

import pytest

from ibrobot_tracing.analysis import AnalysisResult
from ibrobot_tracing.comparison import TraceComparisonService, render_trace_comparison
from ibrobot_tracing.model import SpanRecord, TraceDataset


def result(rows=None, spans=None, coverage=None):
    return AnalysisResult(
        TraceDataset(),
        request_rows=list(rows or []),
        spans=list(spans or []),
        coverage=dict(coverage or {}),
    )


def span(name, span_id, duration_ns, *, parent="", component="component", request="request"):
    return SpanRecord(
        name=name,
        trace_id=request,
        span_id=span_id,
        parent_span_id=parent,
        component_id=component,
        start_ns=0,
        end_ns=duration_ns,
        status="ok",
        origin="user",
        fields={"duration_ns": duration_ns},
    )


def metrics_by_name(comparison):
    return {metric.metric: metric for metric in comparison.metrics}


def flame_paths(comparison):
    entries = {entry.path_id: entry for entry in comparison.flame_diff}
    paths = {}
    for entry in comparison.flame_diff:
        path = []
        current = entry
        while current is not None:
            path.append(current.frame)
            current = entries.get(current.parent_path_id)
        paths[tuple(reversed(path))] = entry
    return paths


def test_metric_union_summary_delta_and_zero_baseline_are_finite_safe():
    comparison = TraceComparisonService().compare(
        result([{"request_id": "a", "latency_ms": 0, "baseline_only_ms": 4, "bad_ms": math.inf}]),
        result([{"request_id": "b", "latency_ms": 10, "candidate_only_ms": 8, "bad_ms": math.nan}]),
    )
    metrics = metrics_by_name(comparison)

    assert set(metrics) == {"baseline_only_ms", "candidate_only_ms", "latency_ms"}
    assert metrics["latency_ms"].baseline.count == metrics["latency_ms"].baseline_count == 1
    assert metrics["latency_ms"].candidate.count == metrics["latency_ms"].candidate_count == 1
    assert metrics["latency_ms"].deltas["p95"].absolute_delta_ms == 10
    assert metrics["latency_ms"].deltas["p95"].percent_delta is None
    assert metrics["latency_ms"].regression
    assert comparison.comparable
    assert not comparison.blocking_reasons
    assert comparison.coverage_warnings
    assert comparison.histogram.metric == "latency_ms"
    assert metrics["baseline_only_ms"].deltas["p95"].absolute_delta_ms is None


def test_zero_baseline_positive_candidate_is_an_unbounded_relative_regression():
    comparison = TraceComparisonService().compare(
        result([{"latency_ms": 0}]),
        result([{"latency_ms": 10}]),
        relative_percent=1_000_000,
        absolute_ms=1,
    )
    metric = metrics_by_name(comparison)["latency_ms"]

    assert comparison.comparable
    assert comparison.has_regression
    assert metric.regression
    assert metric.deltas["p95"].percent_delta is None
    assert comparison.to_dict()["metrics"][0]["deltas"]["p95"]["percent_delta"] is None


def test_regression_requires_relative_and_absolute_thresholds_to_be_exceeded():
    service = TraceComparisonService()
    baseline = result([{"request_id": "base", "latency_ms": 100}])

    relative_not_exceeded = service.compare(
        baseline,
        result([{"request_id": "candidate", "latency_ms": 105}]),
        relative_percent=10,
        absolute_ms=2,
    )
    absolute_not_exceeded = service.compare(
        baseline,
        result([{"request_id": "candidate", "latency_ms": 120}]),
        relative_percent=10,
        absolute_ms=25,
    )
    both_exceeded = service.compare(
        baseline,
        result([{"request_id": "candidate", "latency_ms": 120}]),
        relative_percent=10,
        absolute_ms=5,
    )

    assert not relative_not_exceeded.has_regression
    assert not absolute_not_exceeded.has_regression
    assert both_exceeded.has_regression


def test_histogram_uses_shared_bounds_and_conserves_both_sample_sets():
    comparison = TraceComparisonService().compare(
        result([{"request_id": "a", "latency_ms": 0}, {"request_id": "b", "latency_ms": 10}]),
        result([{"request_id": "c", "latency_ms": 5}, {"request_id": "d", "latency_ms": 20}]),
        metric="latency_ms",
        bins=2,
    )
    histogram = comparison.histogram

    assert histogram.minimum == histogram.buckets[0].start == 0
    assert histogram.maximum == histogram.buckets[-1].end == 20
    assert sum(bucket.baseline_count for bucket in histogram.buckets) == histogram.baseline_count == 2
    assert sum(bucket.candidate_count for bucket in histogram.buckets) == histogram.candidate_count == 2


def test_flame_diff_uses_full_path_identity_and_reports_add_remove_and_delta():
    baseline = result(
        spans=[
            span("root-a", "root-a", 100, component="a"),
            span("work", "old", 40, parent="root-a", component="leaf"),
            span("root-b", "root-b", 60, component="b"),
            span("work", "same-name", 20, parent="root-b", component="leaf"),
        ]
    )
    candidate = result(
        spans=[
            span("root-a", "root-a", 120, component="a"),
            span("replacement", "new", 50, parent="root-a", component="leaf"),
            span("root-b", "root-b", 40, component="b"),
            span("work", "same-name", 10, parent="root-b", component="leaf"),
        ]
    )

    comparison = TraceComparisonService().compare(baseline, candidate)
    paths = flame_paths(comparison)
    old_path = (("a", "root-a", "user"), ("leaf", "work", "user"))
    new_path = (("a", "root-a", "user"), ("leaf", "replacement", "user"))
    retained_path = (("b", "root-b", "user"), ("leaf", "work", "user"))

    assert paths[old_path].status == "removed"
    assert paths[new_path].status == "added"
    assert paths[retained_path].status == "improved"
    assert paths[retained_path].absolute_delta_ns == -10
    assert paths[retained_path].percent_delta == -50
    assert len([entry for entry in comparison.flame_diff if entry.frame[1] == "work"]) == 2


def test_missing_selected_metrics_block_but_coverage_differences_only_warn():
    no_candidate_rows = TraceComparisonService().compare(
        result([{"latency_ms": 10}]),
        result(),
    )
    assert not no_candidate_rows.comparable
    assert "candidate has no request rows" in no_candidate_rows.blocking_reasons
    assert "required metric latency_ms is missing from the candidate" in no_candidate_rows.blocking_reasons

    missing_metric = TraceComparisonService().compare(
        result([{"latency_ms": 10}]),
        result([{"other_ms": 10}]),
    )
    assert not missing_metric.comparable
    assert any("required metric" in reason for reason in missing_metric.blocking_reasons)

    incompatible = TraceComparisonService().compare(
        result([{"latency_ms": 10}], coverage={"mode": "a"}),
        result([{"latency_ms": 10}], coverage={"mode": "b"}),
    )
    assert incompatible.comparable
    assert not incompatible.blocking_reasons
    assert "analysis coverage metadata differs between baseline and candidate" in incompatible.coverage_warnings
    assert metrics_by_name(incompatible)["latency_ms"].deltas["p95"].absolute_delta_ms == 0


@pytest.mark.parametrize("selected", [None, "inference_ms", "total_ms", "missing_ms"])
def test_selected_metric_availability_does_not_erase_other_comparisons(selected):
    comparison = TraceComparisonService().compare(
        result([{"total_ms": 20, "inference_ms": 10}], coverage={"observed_metrics": 2}),
        result([{"inference_ms": 12}], coverage={"observed_metrics": 1}),
        metric=selected,
    )
    metrics = metrics_by_name(comparison)

    assert metrics["inference_ms"].deltas["p95"].absolute_delta_ms == 2
    assert metrics["inference_ms"].regression
    assert metrics["total_ms"].baseline_count == 1
    assert metrics["total_ms"].candidate_count == 0
    assert metrics["total_ms"].deltas["p95"].absolute_delta_ms is None
    assert comparison.histogram.metric == (selected or "inference_ms")
    assert comparison.comparable == (selected in (None, "inference_ms"))
    assert all(f"required metric {selected}" in reason for reason in comparison.blocking_reasons)
    if selected == "missing_ms":
        assert len(comparison.blocking_reasons) == 2
        assert metrics[selected].baseline_count == metrics[selected].candidate_count == 0


def test_flame_profile_is_bounded_and_uses_compact_parent_linked_paths():
    spans = []
    parent = ""
    for index in range(200):
        span_id = f"span-{index}"
        spans.append(span(f"level-{index}", span_id, 1_000 - index, parent=parent))
        parent = span_id
    baseline = result([{"latency_ms": 10}], spans=spans)
    candidate = result([{"latency_ms": 10}], spans=spans)

    comparison = TraceComparisonService().compare(baseline, candidate, max_flame_nodes=50)

    assert comparison.comparable
    assert len(comparison.flame_diff) == 50
    assert not comparison.blocking_reasons
    assert any("50-node" in warning for warning in comparison.coverage_warnings)
    assert metrics_by_name(comparison)["latency_ms"].deltas["p95"].absolute_delta_ms == 0
    assert all(entry.path_id and isinstance(entry.frame, tuple) for entry in comparison.flame_diff)
    assert len(json.dumps(comparison.to_dict())) < 100_000

    disjoint = TraceComparisonService().compare(
        result([{"latency_ms": 10}], spans=[span(f"baseline-{index}", f"b-{index}", 10) for index in range(30)]),
        result([{"latency_ms": 10}], spans=[span(f"candidate-{index}", f"c-{index}", 10) for index in range(30)]),
        max_flame_nodes=50,
    )
    assert disjoint.comparable
    assert len(disjoint.flame_diff) == 50


def test_text_render_includes_comparability_histogram_and_flame_status_counts():
    comparison = TraceComparisonService().compare(
        result([{"latency_ms": 10}], spans=[span("work", "base", 10)]),
        result([{"latency_ms": 20}], spans=[span("work", "candidate", 20)]),
    )
    stream = io.StringIO()

    render_trace_comparison(comparison, stream)

    rendered = stream.getvalue()
    assert "Comparable: yes" in rendered
    assert "Histogram: latency_ms" in rendered
    assert "Flame diff: entries=" in rendered
    assert "regressed=" in rendered


@pytest.mark.parametrize(
    "arguments",
    [
        {"statistic": "median"},
        {"relative_percent": -1},
        {"absolute_ms": -1},
        {"bins": 0},
        {"max_flame_nodes": 0},
    ],
)
def test_invalid_comparison_options_are_rejected(arguments):
    with pytest.raises(ValueError):
        TraceComparisonService().compare(result(), result(), **arguments)
