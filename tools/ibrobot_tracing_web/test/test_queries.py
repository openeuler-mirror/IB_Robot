import pytest

from ibrobot_tracing.analysis import AnalysisResult, AnalysisService
from ibrobot_tracing.model import EventOrigin, FlowRecord, SpanRecord, TraceDataset, TraceEvent, TracepointDefinition
from ibrobot_tracing_web.queries import ResultQueries
from ibrobot_tracing_web.schemas import CriticalPathResponse, GraphResponse, LatencyDistributionResponse


def _event(timestamp_ns, name, **fields):
    return TraceEvent(timestamp_ns, name, fields, schema_version=1)


def _span(
    name,
    span_id,
    start_ns,
    end_ns,
    *,
    request_id="request",
    parent_span_id="",
    component_id="component",
    origin="user",
    status="ok",
    fields=None,
):
    return SpanRecord(
        name,
        request_id,
        span_id,
        parent_span_id,
        component_id,
        start_ns,
        end_ns,
        status,
        origin,
        fields or {},
    )


def test_current_query_and_projection_apis_are_used_with_js_safe_records():
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[
                _event(
                    1_000,
                    "span_begin",
                    trace_id="req",
                    span_id="work",
                    span_name="work",
                    component_id="worker",
                    origin="user",
                ),
                _event(1_100, "mark", trace_id="req", component_id="worker", origin="user"),
                _event(
                    2_000,
                    "span_end",
                    trace_id="req",
                    span_id="work",
                    span_name="work",
                    component_id="worker",
                    origin="user",
                ),
            ]
        )
    )
    queries = ResultQueries(result)

    events = queries.query_events(offset=0, limit=1, request_id="req")
    assert events["total"] == 3
    assert events["next_offset"] == 1
    assert events["items"][0]["id"].startswith("event:")
    assert events["items"][0]["timestamp_ns"] == "1000"

    timeline = queries.timeline_projection(
        request_id="req",
        component_id="",
        start_ns=None,
        end_ns=None,
        include_events=True,
        include_spans=True,
        include_flows=True,
    )
    assert {item["kind"] for item in timeline["items"]} == {"event", "span"}
    assert timeline["start_ns"] == "1000"

    tree = queries.call_tree_projection(request_id="req", component_id="", start_ns=None, end_ns=None)
    assert tree["nodes"][0]["span_id"] == "work"
    assert tree["root_ids"] == [tree["nodes"][0]["id"]]

    graph = queries.graph_projection(request_id="req", metric="p95", view="tracepoints")
    assert graph["topology_source"] == "observed"
    assert any(node["component_id"] == "worker" for node in graph["nodes"])
    assert GraphResponse.model_validate(graph).view == "tracepoints"


def test_warnings_do_not_expose_source_paths():
    source = "/private/traces/robot.log"
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[TraceEvent(1, "mark", origin=EventOrigin(path=source))],
            metadata={"source": source},
            warnings=[f"Could not parse trace record {source}:1"],
        )
    )

    warnings = ResultQueries(result).warnings()

    assert warnings == ["Could not parse trace record <trace-source>:1"]
    assert source not in warnings[0]


@pytest.mark.parametrize("total", [0, 2, 100, 101, 10_000])
def test_span_summary_preview_is_bounded_without_changing_core_groups(total):
    groups = [
        {"component_id": f"worker-{index}", "name": "work", "origin": "built-in", "status": "error", "count": 2}
        for index in range(total)
    ]
    result = AnalysisResult(TraceDataset(), span_summary=groups)

    preview = ResultQueries(result).span_summary()

    assert preview == {
        "span_summary": groups[:100],
        "span_summary_total": total,
        "span_summary_limit": 100,
        "span_summary_truncated": total > 100,
    }
    assert len(result.span_summary) == total
    assert preview["span_summary"] is not groups


def test_request_status_and_all_complete_occurrences_survive_web_serialization():
    events = []
    for index, (request_id, origin, component, status) in enumerate(
        [
            ("retry", "built-in", "policy.inference", "error"),
            ("retry", "built-in", "policy.inference", "ok"),
            ("retry", "built-in", "policy.inference", "ok"),
            ("failed", "built-in", "policy.inference", "error"),
            ("open", "built-in", "policy.inference", "incomplete"),
            ("custom", "user", "policy.inference", "ok"),
            ("other", "built-in", "other", "ok"),
        ]
    ):
        fields = dict(
            trace_id=request_id, origin=origin, component_id=component, span_name="model_call", span_id=str(index)
        )
        events.append(_event(index * 10_000_000, "span_begin", **fields))
        if status != "incomplete":
            events.append(_event(index * 10_000_000 + 1_000_000, "span_end", status=status, **fields))
    events.append(_event(100_000_000, "mark", trace_id="missing"))
    queries = ResultQueries(AnalysisService().analyze_dataset(TraceDataset(events=events)))

    rows = {row["request_id"]: row for row in queries.query_requests()["items"]}
    for request_id, status in [("retry", "ambiguous,error"), ("failed", "error"), ("open", "incomplete")]:
        assert rows[request_id]["inference_ms_status"] == status
        assert "inference_ms" not in rows[request_id]
    assert "inference_ms_status" not in rows["missing"]
    assert "inference_ms" not in rows["missing"]
    groups = queries.span_summary()["span_summary"]
    assert {(item["component_id"], item["name"], item["origin"], item["status"], item["count"]) for item in groups} == {
        ("policy.inference", "model_call", "built-in", "error", 2),
        ("policy.inference", "model_call", "built-in", "ok", 2),
        ("policy.inference", "model_call", "user", "ok", 1),
        ("other", "model_call", "built-in", "ok", 1),
    }


def test_tracepoint_queries_are_typed_filtered_paginated_and_path_safe():
    source = "/private/traces/robot.log"
    definitions = [
        TracepointDefinition("event", "worker", "ready", "user", "Ready for work."),
        TracepointDefinition("event", "worker", "waiting", "built-in"),
        TracepointDefinition("span", "worker", "work", "user", "Runs one item."),
    ]
    queries = ResultQueries(AnalysisResult(TraceDataset(metadata={"source": source}), definitions=definitions))

    first = queries.query_tracepoints(kind="event", component_id="worker", offset=0, limit=1)
    filtered = queries.query_tracepoints(
        kind="span",
        component_id="worker",
        name="work",
        origin="user",
        offset=0,
        limit=100,
    )

    assert first["total"] == 2
    assert first["next_offset"] == 1
    assert first["items"][0] == {
        "id": first["items"][0]["id"],
        "kind": "event",
        "component_id": "worker",
        "name": "ready",
        "origin": "user",
        "description": "Ready for work.",
    }
    assert first["items"][0]["id"].startswith("tracepoint:")
    assert filtered["total"] == 1
    assert filtered["items"][0]["name"] == "work"
    assert source not in str(first)


def test_span_profile_adapter_supports_both_modes_filters_truncation_and_path_safety():
    source = "/private/traces/robot.log"
    base_ns = 9_007_199_254_741_000
    result = AnalysisResult(
        TraceDataset(
            events=[TraceEvent(base_ns, "marker", origin=EventOrigin(path=source))],
            metadata={"source": source},
        ),
        spans=[
            _span("root", "root", base_ns, base_ns + 100, component_id="outer", fields={"path": source}),
            _span(
                "child",
                "child",
                base_ns + 10,
                base_ns + 40,
                parent_span_id="root",
                component_id="inner",
                status="error",
                fields={"debug_path": source},
            ),
            _span(
                "other",
                "other",
                base_ns + 200,
                base_ns + 250,
                request_id="other-request",
                component_id="other",
                origin="built-in",
            ),
        ],
    )
    queries = ResultQueries(result)

    request_profile = queries.span_profile_projection(
        mode="request",
        request_id="request",
        component_id="",
        start_ns=None,
        end_ns=None,
        origin="",
        status="",
        max_nodes=10_000,
    )
    filtered = queries.span_profile_projection(
        mode="request",
        request_id="request",
        component_id="inner",
        start_ns=base_ns + 20,
        end_ns=base_ns + 35,
        origin="user",
        status="error",
        max_nodes=10_000,
    )
    aggregate = queries.span_profile_projection(
        mode="aggregate",
        request_id="",
        component_id="",
        start_ns=None,
        end_ns=None,
        origin="user",
        status="",
        max_nodes=1,
    )

    assert request_profile["mode"] == "request"
    assert request_profile["start_ns"] == str(base_ns)
    assert request_profile["end_ns"] == str(base_ns + 100)
    assert all(isinstance(node["start_ns"], str) for node in request_profile["nodes"])
    assert [node["span_id"] for node in filtered["nodes"]] == ["child"]
    assert filtered["nodes"][0]["status"] == "error"
    assert aggregate["mode"] == "aggregate"
    assert aggregate["total_nodes"] == 2
    assert aggregate["returned_nodes"] == 1
    assert aggregate["truncated"]
    assert aggregate["truncation_reason"] == "max_nodes"
    assert source not in str(request_profile)
    assert source not in str(filtered)


def test_critical_path_adapter_is_exact_js_safe_and_path_free():
    source = "/private/traces/robot.log"
    base_ns = 9_007_199_254_741_000
    result = AnalysisResult(
        TraceDataset(
            events=[
                _event(base_ns, "dispatch_request", trace_id="request"),
                _event(base_ns + 100, "first_action_execute", trace_id="request"),
            ],
            metadata={"source": source},
        ),
        spans=[
            _span("root", "root", base_ns, base_ns + 100, fields={"debug_path": source}),
            _span(
                "child",
                "child",
                base_ns + 20,
                base_ns + 80,
                parent_span_id="root",
                component_id="child",
                fields={"source_path": source},
            ),
        ],
        flows=[FlowRecord("edge", "transfer", "request", base_ns + 30, base_ns + 40, "complete")],
    )

    document = ResultQueries(result).critical_path_projection(
        request_id="request",
        component_id="",
        start_ns=None,
        end_ns=None,
        include_flows=True,
        max_segments=1_000,
    )
    response = CriticalPathResponse.model_validate(document)

    assert response.start_ns == str(base_ns)
    assert response.end_ns == str(base_ns + 100)
    assert [(item.kind, item.label, item.duration_ns) for item in response.segments] == [
        ("span", "root", 20),
        ("span", "child", 10),
        ("flow", "edge", 10),
        ("span", "child", 40),
        ("span", "root", 20),
    ]
    assert response.totals.partition_ns == 100
    assert source not in str(document)


def test_latency_distribution_adapter_is_thin_js_safe_and_path_free():
    source = "/private/traces/robot.log"
    result = AnalysisResult(
        TraceDataset(metadata={"source": source}),
        request_rows=[
            {"request_id": "fast", "inference_ms": -1.0},
            {"request_id": "slow", "inference_ms": 9.0},
            {"request_id": "invalid", "inference_ms": float("nan")},
        ],
    )

    document = ResultQueries(result).latency_distribution_projection(
        metric=None,
        bins=5,
        outlier_limit=2,
        bucket_request_limit=1,
    )
    response = LatencyDistributionResponse.model_validate(document)

    assert response.metric == "inference_ms"
    assert response.sample_count == 2
    assert response.invalid_count == 1
    assert sum(bucket.count for bucket in response.buckets) == 2
    assert response.outliers[0].request_id == "slow"
    assert source not in str(document)
