import pytest
from pydantic import TypeAdapter, ValidationError

from ibrobot_tracing_web.schemas import (
    AggregateSpanProfileResponse,
    ComparisonJobCreate,
    ComparisonStatisticDelta,
    CriticalPathResponse,
    FlameDiffEntryResponse,
    GraphEdge,
    GraphNode,
    LatencyDistributionResponse,
    RequestSpanProfileResponse,
    SpanProfileResponse,
    TracepointDefinition,
)


def test_graph_and_tracepoint_schema_additions_keep_safe_defaults_and_forbid_extras():
    node_fields = {
        "id": "worker",
        "component_id": "worker",
        "name": "Worker",
        "kind": "module",
        "parent_id": "",
        "child_ids": [],
        "depth": 0,
        "provenance": "observed",
        "trace_name": "",
        "metrics": {},
        "metric_value_ms": None,
    }
    edge_fields = {
        "id": "contains",
        "edge_id": "contains",
        "source_id": "worker",
        "target_id": "tracepoint",
        "name": "Contains",
        "kind": "contains",
        "provenance": "derived",
        "data_type": "",
        "contract_key": "",
        "metrics": {},
        "metric_value_ms": None,
    }

    assert GraphNode(**node_fields).description == ""
    assert GraphNode(**node_fields, description="Runs work.").description == "Runs work."
    assert GraphEdge(**edge_fields).directed
    assert not GraphEdge(**edge_fields, directed=False).directed
    assert (
        TracepointDefinition(
            id="tracepoint:event:id", kind="event", component_id="worker", name="ready", origin="user"
        ).description
        == ""
    )

    with pytest.raises(ValidationError):
        GraphNode(**node_fields, filesystem_path="/private/traces/robot.log")


def _totals(**extra):
    return {
        "request_count": 1,
        "occurrence_count": 1,
        "error_count": 0,
        "incomplete_count": 0,
        "invalid_count": 0,
        "excluded_weight_count": 0,
        "sampled_uncovered_wall_ns": 10,
        "uncovered_wall_ns": 10,
        "selected_wall_union_ns": 10,
        "concurrency_factor": 1.0,
        "diagnostic_counts": {},
        **extra,
    }


def _common_profile(**extra):
    return {
        "measurement": "instrumented_wall",
        "source": "structured span instrumentation",
        "coverage": "selected instrumented spans",
        "not_cpu": True,
        "roots": ["node"],
        "diagnostics": [{"code": "orphan", "occurrence_ids": ["node"], "message": "orphan"}],
        "total_nodes": 1,
        "returned_nodes": 1,
        "truncated": False,
        "truncation_reason": "",
        **extra,
    }


def test_span_profile_schema_is_strict_discriminated_and_complete():
    request_data = _common_profile(
        mode="request",
        request_id="request",
        start_ns="9007199254741000",
        end_ns="9007199254741010",
        duration_ns=10,
        nodes=[
            {
                "id": "node",
                "occurrence_id": "node",
                "span_id": "span",
                "request_id": "request",
                "parent_id": "",
                "parent_span_id": "",
                "child_ids": [],
                "depth": 0,
                "track": 0,
                "component_id": "worker",
                "name": "work",
                "origin": "user",
                "status": "ok",
                "start_ns": "9007199254741000",
                "end_ns": "9007199254741010",
                "observed_end_ns": "9007199254741010",
                "start_offset_ns": 0,
                "duration_ns": 10,
                "duration_source": "observed",
                "uncovered_wall_ns": 10,
                "diagnostics": [],
                "fields": {},
            }
        ],
        totals=_totals(),
    )
    aggregate_data = _common_profile(
        mode="aggregate",
        nodes=[
            {
                "id": "node",
                "parent_id": "",
                "child_ids": [],
                "depth": 0,
                "component_id": "worker",
                "name": "work",
                "origin": "user",
                "path": [{"component_id": "worker", "name": "work", "origin": "user"}],
                "self_value_ns": 10,
                "value_ns": 10,
                "occurrence_count": 1,
                "request_count": 1,
                "error_count": 0,
                "incomplete_count": 0,
                "invalid_count": 0,
                "percentage": 100.0,
            }
        ],
        totals=_totals(value_ns=10),
    )
    adapter = TypeAdapter(SpanProfileResponse)

    assert isinstance(adapter.validate_python(request_data), RequestSpanProfileResponse)
    assert isinstance(adapter.validate_python(aggregate_data), AggregateSpanProfileResponse)

    with pytest.raises(ValidationError):
        RequestSpanProfileResponse.model_validate(request_data | {"filesystem_path": "/private/trace"})
    with pytest.raises(ValidationError):
        RequestSpanProfileResponse.model_validate(request_data | {"start_ns": 9_007_199_254_741_000})
    with pytest.raises(ValidationError):
        RequestSpanProfileResponse.model_validate(request_data | {"total_nodes": "1"})
    with pytest.raises(ValidationError):
        RequestSpanProfileResponse.model_validate(request_data | {"duration_ns": 9_007_199_254_740_992})
    with pytest.raises(ValidationError):
        RequestSpanProfileResponse.model_validate(request_data | {"start_ns": "not-a-timestamp"})
    with pytest.raises(ValidationError):
        adapter.validate_python(aggregate_data | {"mode": "unknown"})


def _critical_path(**extra):
    return {
        "method": "deepest_active_wall_partition",
        "measurement": "instrumented_wall",
        "not_cpu": True,
        "request_id": "request",
        "component_id": "",
        "include_flows": True,
        "boundary_source": "dispatch_request_to_first_action_execute",
        "boundary_start_source": "event:dispatch_request",
        "boundary_end_source": "event:first_action_execute",
        "start_ns": "9007199254741000",
        "end_ns": "9007199254741010",
        "duration_ns": 10,
        "segments": [
            {
                "id": "segment",
                "index": 0,
                "kind": "span",
                "source_id": "span",
                "label": "work",
                "component_id": "worker",
                "edge_id": "",
                "origin": "user",
                "status": "ok",
                "start_ns": "9007199254741000",
                "end_ns": "9007199254741010",
                "offset_ns": 0,
                "duration_ns": 10,
                "percentage": 100.0,
                "diagnostics": [],
                "fields": {},
            }
        ],
        "totals": {
            "partition_ns": 10,
            "attributed_ns": 10,
            "unattributed_ns": 0,
            "coverage_percent": 100.0,
            "by_kind_ns": {"flow": 0, "span": 10, "unattributed": 0},
            "by_component_ns": {"worker": 10},
            "by_edge_ns": {},
            "record_counts": {"valid_spans": 1},
            "diagnostic_counts": {},
        },
        "bottlenecks": [
            {
                "rank": 1,
                "kind": "span",
                "source_id": "span",
                "label": "work",
                "component_id": "worker",
                "edge_id": "",
                "origin": "user",
                "status": "ok",
                "duration_ns": 10,
                "percentage": 100.0,
                "segment_count": 1,
            }
        ],
        "diagnostics": [],
        "total_segments": 1,
        "returned_segments": 1,
        "returned_duration_ns": 10,
        "omitted_segments": 0,
        "omitted_duration_ns": 0,
        "truncated": False,
        "truncation_reason": "",
        **extra,
    }


def test_critical_path_schema_is_strict_bounded_and_js_safe():
    assert CriticalPathResponse.model_validate(_critical_path()).duration_ns == 10
    assert (
        CriticalPathResponse.model_validate(_critical_path(duration_ns="9007199254740992")).duration_ns
        == "9007199254740992"
    )

    for invalid in (
        _critical_path(filesystem_path="/private/trace"),
        _critical_path(returned_segments="1"),
        _critical_path(start_ns=9_007_199_254_741_000),
        _critical_path(duration_ns=9_007_199_254_740_992),
        _critical_path(duration_ns=-1),
        _critical_path(duration_ns="-1"),
    ):
        with pytest.raises(ValidationError):
            CriticalPathResponse.model_validate(invalid)

    for percentage in (-0.1, 100.1, float("inf"), float("nan")):
        invalid = _critical_path()
        invalid["segments"][0]["percentage"] = percentage
        with pytest.raises(ValidationError):
            CriticalPathResponse.model_validate(invalid)

    invalid_totals = _critical_path()
    invalid_totals["totals"]["coverage_percent"] = float("nan")
    with pytest.raises(ValidationError):
        CriticalPathResponse.model_validate(invalid_totals)


def _distribution(**extra):
    return {
        "metric": "inference_ms",
        "unit": "ms",
        "available_metrics": ["inference_ms"],
        "summary": {
            "count": 1,
            "minimum": 2.0,
            "p50": 2.0,
            "p95": 2.0,
            "p99": 2.0,
            "maximum": 2.0,
            "mean": 2.0,
        },
        "sample_count": 1,
        "invalid_count": 0,
        "minimum": 2.0,
        "maximum": 2.0,
        "bin_count": 1,
        "buckets": [
            {
                "index": 0,
                "start": 2.0,
                "end": 2.0,
                "count": 1,
                "request_ids": ["request"],
                "returned": 1,
                "truncated": False,
            }
        ],
        "outliers": [
            {
                "request_id": "request",
                "value": 2.0,
                "rank": 1,
                "percentile": 100.0,
                "p95_tail": True,
            }
        ],
        **extra,
    }


def test_distribution_schema_is_strict_finite_and_forbids_paths():
    assert LatencyDistributionResponse.model_validate(_distribution()).unit == "ms"

    with pytest.raises(ValidationError):
        LatencyDistributionResponse.model_validate(_distribution(sample_count="1"))
    with pytest.raises(ValidationError):
        LatencyDistributionResponse.model_validate(_distribution(filesystem_path="/private/trace"))
    with pytest.raises(ValidationError):
        LatencyDistributionResponse.model_validate(_distribution(maximum=float("inf")))


def test_comparison_request_schema_is_strict_bounded_and_path_free():
    valid = {
        "baseline_source_id": "b" * 32,
        "baseline_source_version": "c" * 64,
        "candidate_analysis_id": "a" * 32,
        "statistic": "p95",
        "relative_threshold_percent": 5.0,
        "absolute_threshold_ms": 1.0,
        "metric": "total_ms",
        "bins": 30,
    }
    assert ComparisonJobCreate.model_validate(valid).bins == 30

    for invalid in (
        valid | {"bins": 4},
        valid | {"bins": 101},
        valid | {"relative_threshold_percent": float("inf")},
        valid | {"absolute_threshold_ms": -1.0},
        valid | {"statistic": "median"},
        valid | {"baseline_source_id": "/private/baseline"},
        valid | {"baseline_source_version": "stale"},
        valid | {"candidate_analysis_id": "../analysis"},
        valid | {"source_path": "/private/trace"},
    ):
        with pytest.raises(ValidationError):
            ComparisonJobCreate.model_validate(invalid)


def test_comparison_schemas_forbid_paths_extras_and_unsafe_ns():
    with pytest.raises(ValidationError):
        ComparisonStatisticDelta.model_validate(
            {"absolute_delta_ms": 1.0, "percent_delta": 2.0, "path": "/private/trace"}
        )

    flame = {
        "path_id": "path",
        "parent_path_id": "",
        "frame": ["component", "work", "user"],
        "depth": 0,
        "baseline_value_ns": "9007199254740992",
        "candidate_value_ns": 10,
        "baseline_self_value_ns": 10,
        "candidate_self_value_ns": 10,
        "absolute_delta_ns": 0,
        "percent_delta": 0.0,
        "self_absolute_delta_ns": 0,
        "self_percent_delta": 0.0,
        "status": "unchanged",
    }
    assert FlameDiffEntryResponse.model_validate(flame).baseline_value_ns == "9007199254740992"
    with pytest.raises(ValidationError):
        FlameDiffEntryResponse.model_validate(flame | {"baseline_value_ns": 9_007_199_254_740_992})
    with pytest.raises(ValidationError):
        FlameDiffEntryResponse.model_validate(flame | {"status": "unknown"})
