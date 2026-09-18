import json
from pathlib import Path

import pytest

import ibrobot_tracing.critical_path as critical_path_module
from ibrobot_tracing import (
    CriticalPathProjection as PublicCriticalPathProjection,
)
from ibrobot_tracing import (
    CriticalPathQuery as PublicCriticalPathQuery,
)
from ibrobot_tracing import cli
from ibrobot_tracing import project_critical_path as public_project_critical_path
from ibrobot_tracing.analysis import AnalysisResult
from ibrobot_tracing.critical_path import CriticalPathProjection, CriticalPathQuery, project_critical_path
from ibrobot_tracing.model import (
    Component,
    DataFlowEdge,
    EventOrigin,
    FlowRecord,
    SpanRecord,
    TraceDataset,
    TraceEvent,
    TraceTopology,
)
from ibrobot_tracing.query import stable_span_id


def event(timestamp_ns, name, request_id="request", component_id="", **fields):
    return TraceEvent(
        timestamp_ns,
        name,
        {"trace_id": request_id, "component_id": component_id, **fields},
        schema_version=1,
    )


def span(
    name,
    span_id,
    start_ns,
    end_ns,
    *,
    parent_span_id="",
    component_id="component",
    status="ok",
    fields=None,
):
    return SpanRecord(
        name,
        "request",
        span_id,
        parent_span_id,
        component_id,
        start_ns,
        end_ns,
        status,
        "user",
        fields or {},
    )


def result(*, spans=(), flows=(), events=(), topology=None):
    return AnalysisResult(TraceDataset(events=list(events)), spans=list(spans), flows=list(flows), topology=topology)


def diagnostic_codes(projection):
    return {diagnostic.code for diagnostic in projection.diagnostics}


def test_exact_partition_nested_spans_flow_priority_and_ranked_owners():
    projection = project_critical_path(
        result(
            events=[event(0, "dispatch_request"), event(100, "first_action_execute")],
            spans=[
                span("root", "root", 0, 100),
                span("child", "child", 20, 80, parent_span_id="root", component_id="child"),
            ],
            flows=[FlowRecord("edge", "transfer", "request", 30, 40, "complete")],
        ),
        CriticalPathQuery("request"),
    )

    assert projection.method == "deepest_active_wall_partition"
    assert projection.measurement == "instrumented_wall"
    assert projection.not_cpu
    assert projection.boundary_source == "dispatch_request_to_first_action_execute"
    assert [(item.kind, item.label, item.start_ns, item.end_ns) for item in projection.segments] == [
        ("span", "root", 0, 20),
        ("span", "child", 20, 30),
        ("flow", "edge", 30, 40),
        ("span", "child", 40, 80),
        ("span", "root", 80, 100),
    ]
    assert sum(item.duration_ns for item in projection.segments) == projection.duration_ns == 100
    assert projection.totals.partition_ns == 100
    assert projection.totals.by_kind_ns == {"flow": 10, "span": 90, "unattributed": 0}
    assert projection.totals.by_component_ns == {"child": 50, "component": 40}
    assert [item.label for item in projection.bottlenecks] == ["child", "root", "edge"]
    assert projection.bottlenecks[0].segment_count == 2


def test_gaps_merge_and_deterministic_shortest_interval_tie_break():
    projection = project_critical_path(
        result(
            events=[event(0, "dispatch_request"), event(30, "first_action_execute")],
            spans=[span("long", "long", 0, 30), span("short", "short", 10, 20)],
        ),
        CriticalPathQuery("request", start_ns=5, end_ns=25),
    )
    assert [(item.label, item.start_ns, item.end_ns) for item in projection.segments] == [
        ("long", 5, 10),
        ("short", 10, 20),
        ("long", 20, 25),
    ]

    gaps = project_critical_path(
        result(
            events=[event(0, "dispatch_request"), event(30, "first_action_execute")],
            spans=[span("work", "work", 10, 20)],
        ),
        CriticalPathQuery("request"),
    )
    assert [(item.kind, item.start_ns, item.end_ns) for item in gaps.segments] == [
        ("unattributed", 0, 10),
        ("span", 10, 20),
        ("unattributed", 20, 30),
    ]
    assert gaps.totals.attributed_ns == 10
    assert gaps.totals.unattributed_ns == 20
    assert gaps.totals.coverage_percent == pytest.approx(100 / 3)


def test_boundary_fallback_and_requested_window_are_explicit_and_clamped():
    projection = project_critical_path(
        result(events=[event(5, "mark", component_id="worker")], spans=[span("work", "work", 10, 30)]),
        CriticalPathQuery("request", start_ns=0, end_ns=100),
    )

    assert projection.boundary_source == "selected_record_bounds_fallback"
    assert projection.start_ns == 5
    assert projection.end_ns == 30
    assert {
        "boundary_start_fallback",
        "boundary_end_fallback",
        "window_start_clamped",
        "window_end_clamped",
    } <= diagnostic_codes(projection)


def test_zero_length_and_inverted_partial_fallback_boundaries_are_safe():
    zero = project_critical_path(
        result(
            events=[event(10, "dispatch_request"), event(10, "first_action_execute")],
            spans=[span("work", "work", 0, 20)],
        ),
        CriticalPathQuery("request"),
    )
    assert zero.start_ns == zero.end_ns == 10
    assert zero.duration_ns == 0
    assert zero.segments == []
    assert zero.totals.partition_ns == 0

    partial = project_critical_path(
        result(
            events=[event(100, "dispatch_request", component_id="dispatcher")],
            spans=[span("old", "old", 10, 20, component_id="worker")],
        ),
        CriticalPathQuery("request", component_id="worker"),
    )
    assert partial.start_ns == partial.end_ns == 100
    assert partial.duration_ns == 0
    assert partial.segments == []
    assert "boundary_end_before_start" in diagnostic_codes(partial)


def test_component_filter_keeps_incident_flows_and_turns_other_work_into_gaps():
    topology = TraceTopology(
        components=[Component("selected", "Selected", "module"), Component("other", "Other", "module")],
        edges=[DataFlowEdge("edge", "selected", "other", "Transfer", "flow")],
    )
    projection = project_critical_path(
        result(
            events=[event(0, "dispatch_request"), event(100, "first_action_execute")],
            spans=[
                span("selected", "selected", 20, 40, component_id="selected"),
                span("other", "other", 0, 100, component_id="other"),
            ],
            flows=[FlowRecord("edge", "transfer", "request", 50, 60, "complete")],
            topology=topology,
        ),
        CriticalPathQuery("request", component_id="selected"),
    )

    assert [(item.kind, item.start_ns, item.end_ns) for item in projection.segments] == [
        ("unattributed", 0, 20),
        ("span", 20, 40),
        ("unattributed", 40, 50),
        ("flow", 50, 60),
        ("unattributed", 60, 100),
    ]
    assert projection.totals.by_component_ns == {"selected": 20}
    assert projection.totals.by_edge_ns == {"edge": 10}


def test_invalid_spans_and_non_allowlisted_or_unsynchronized_flows_are_excluded():
    events = [
        event(0, "dispatch_request"),
        event(100, "first_action_execute"),
        TraceEvent(
            20,
            "flow_send",
            {"trace_id": "request", "edge_id": "cross", "flow_id": "cross"},
            origin=EventOrigin(host="one"),
            clock="realtime",
        ),
    ]
    events.append(
        TraceEvent(
            30,
            "flow_receive",
            {"trace_id": "request", "edge_id": "cross", "flow_id": "cross"},
            origin=EventOrigin(host="two"),
            clock="realtime",
        )
    )
    projection = project_critical_path(
        result(
            events=events,
            spans=[
                span("incomplete", "incomplete", 10, None, status="incomplete"),
                span("negative", "negative", 50, 40),
            ],
            flows=[
                FlowRecord("missing", "missing", "request", 10, None, "incomplete"),
                FlowRecord("negative", "negative", "request", 40, 30, "negative"),
                FlowRecord("future", "future", "request", 50, 60, "clock_unknown"),
                FlowRecord("cross", "cross", "request", 20, 30, "complete"),
            ],
        ),
        CriticalPathQuery("request"),
    )

    assert {item.kind for item in projection.segments} == {"unattributed"}
    assert {
        "incomplete",
        "negative_duration",
        "flow_incomplete",
        "flow_negative_duration",
        "flow_status_excluded",
        "flow_cross_host_unsynchronized",
    } <= diagnostic_codes(projection)
    assert projection.totals.record_counts["excluded_spans"] == 2
    assert projection.totals.record_counts["excluded_flows"] == 4


def test_flow_clock_exclusions_are_explicit_and_missing_raw_endpoints_are_safe():
    events = [event(0, "dispatch_request"), event(100, "first_action_execute")]
    for timestamp_ns, name, edge_id, flow_id, clock in (
        (10, "flow_send", "unknown", "unknown", "monotonic"),
        (20, "flow_receive", "unknown", "unknown", "monotonic"),
        (30, "flow_send", "mismatch", "mismatch", "realtime"),
        (40, "flow_receive", "mismatch", "mismatch", "monotonic"),
    ):
        events.append(
            TraceEvent(
                timestamp_ns,
                name,
                {"trace_id": "request", "edge_id": edge_id, "flow_id": flow_id},
                clock=clock,
            )
        )
    projection = project_critical_path(
        result(
            events=events,
            flows=[
                FlowRecord("unknown", "unknown", "request", 10, 20, "complete"),
                FlowRecord("mismatch", "mismatch", "request", 30, 40, "complete"),
                FlowRecord("metadata-absent", "safe", "request", 60, 70, "complete"),
            ],
        ),
        CriticalPathQuery("request"),
    )

    assert [(item.kind, item.edge_id, item.start_ns, item.end_ns) for item in projection.segments] == [
        ("unattributed", "", 0, 60),
        ("flow", "metadata-absent", 60, 70),
        ("unattributed", "", 70, 100),
    ]
    assert {"flow_clock_unknown", "flow_clock_mismatch"} <= diagnostic_codes(projection)


def test_flow_and_span_active_identities_cannot_collide(monkeypatch):
    work = span("work", "work", 0, 30)
    monkeypatch.setattr(critical_path_module, "stable_flow_id", lambda _flow: stable_span_id(work))

    projection = project_critical_path(
        result(
            events=[event(0, "dispatch_request"), event(30, "first_action_execute")],
            spans=[work],
            flows=[FlowRecord("edge", "flow", "request", 10, 20, "complete")],
        ),
        CriticalPathQuery("request"),
    )

    assert [(item.kind, item.start_ns, item.end_ns) for item in projection.segments] == [
        ("span", 0, 10),
        ("flow", 10, 20),
        ("span", 20, 30),
    ]


def test_unknown_span_diagnostic_has_deterministic_safe_order(monkeypatch):
    build_hierarchy = critical_path_module.build_span_hierarchy

    def hierarchy_with_future_diagnostic(spans):
        hierarchy = build_hierarchy(spans)
        hierarchy.nodes[0].diagnostics.add("future_diagnostic")
        return hierarchy

    monkeypatch.setattr(critical_path_module, "build_span_hierarchy", hierarchy_with_future_diagnostic)
    projection = project_critical_path(
        result(
            events=[event(0, "dispatch_request"), event(10, "first_action_execute")],
            spans=[span("work", "work", 0, 10)],
        ),
        CriticalPathQuery("request"),
    )

    assert [item.code for item in projection.diagnostics] == ["future_diagnostic"]
    assert projection.segments[0].diagnostics == ["future_diagnostic"]


def test_truncation_is_a_deterministic_prefix_with_full_partition_loss_metadata():
    analysis = result(
        events=[event(0, "dispatch_request"), event(30, "first_action_execute")],
        spans=[span("work", "work", 10, 20)],
    )
    first = project_critical_path(analysis, CriticalPathQuery("request", max_segments=1))
    second = project_critical_path(analysis, CriticalPathQuery("request", max_segments=1))

    assert first.to_dict() == second.to_dict()
    assert first.total_segments == 3
    assert first.returned_segments == 1
    assert first.returned_duration_ns == 10
    assert first.omitted_segments == 2
    assert first.omitted_duration_ns == 20
    assert first.totals.partition_ns == 30
    assert first.truncation_reason == "max_segments_chronological_prefix"


def test_empty_projection_validation_public_exports_and_cli(monkeypatch, capsys):
    empty = project_critical_path(result(), CriticalPathQuery("missing"))
    assert empty.start_ns is empty.end_ns is None
    assert empty.segments == []
    assert empty.totals.partition_ns == 0
    assert diagnostic_codes(empty) == {"boundary_unavailable"}

    with pytest.raises(ValueError, match="requires request_id"):
        CriticalPathQuery("")
    with pytest.raises(ValueError, match="max_segments"):
        CriticalPathQuery("request", max_segments=10_001)
    assert PublicCriticalPathQuery is CriticalPathQuery
    assert PublicCriticalPathProjection is CriticalPathProjection
    assert public_project_critical_path is project_critical_path

    analysis = result(
        events=[event(0, "dispatch_request"), event(10, "first_action_execute")],
        spans=[span("work", "work", 0, 10)],
    )
    monkeypatch.setattr(cli, "_load", lambda _args: analysis)
    assert cli.main(["critical-path", str(Path("trace")), "--request-id", "request", "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["method"] == "deepest_active_wall_partition"
    assert document["not_cpu"] is True
