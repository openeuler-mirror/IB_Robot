from copy import deepcopy

import pytest

from ibrobot_tracing.analysis import AnalysisResult, AnalysisService
from ibrobot_tracing.model import (
    Component,
    DataFlowEdge,
    FlowRecord,
    SpanRecord,
    TraceDataset,
    TraceEvent,
    TracepointDefinition,
    TraceTopology,
)
from ibrobot_tracing.projection import (
    CallTreeQuery,
    GraphQuery,
    TimelineQuery,
    project_call_tree,
    project_graph,
    project_timeline,
)
from ibrobot_tracing.query import (
    EventQuery,
    FlowQuery,
    QueryService,
    RequestQuery,
    stable_component_id,
    stable_event_id,
    stable_flow_id,
    stable_request_id,
    stable_span_id,
)
from ibrobot_tracing.serialization import JS_MAX_SAFE_INTEGER, to_js_safe


def event(timestamp_ns, name, request_id="", sequence=0, **fields):
    if request_id:
        fields = {"trace_id": request_id, **fields}
    return TraceEvent(timestamp_ns, name, fields, sequence=sequence)


def span(name, span_id, start_ns, end_ns, *, parent_span_id="", trace_id="req", component_id="component"):
    return SpanRecord(
        name,
        trace_id,
        span_id,
        parent_span_id,
        component_id,
        start_ns,
        end_ns,
        "ok",
        "user",
    )


def test_flow_id_does_not_create_request_and_ids_are_deterministic():
    dataset = TraceDataset(
        events=[
            event(10, "flow_send", edge_id="edge", flow_id="transfer", component_id="source"),
            event(20, "flow_receive", edge_id="edge", flow_id="transfer", component_id="target"),
        ]
    )

    result = AnalysisService().analyze_dataset(dataset)

    assert result.request_rows == []
    assert result.flows[0].trace_id == ""
    assert stable_flow_id(result.flows[0]) == stable_flow_id(result.flows[0])
    assert stable_event_id(result.dataset.events[0]) != stable_event_id(result.dataset.events[1])
    assert stable_request_id("request") == stable_request_id({"request_id": "request"})
    assert stable_component_id("component") == "component"


def test_analysis_does_not_mutate_caller_dataset_or_topology():
    dataset = TraceDataset(
        events=[
            event(0, "span_begin", "req", span_id="child", span_name="resize"),
            event(10, "span_end", "req", span_id="child", span_name="resize"),
        ]
    )
    topology = TraceTopology(
        components=[Component("policy", "Policy", "ros_node")],
        logger_to_component={"": "policy"},
    )
    original_dataset = deepcopy(dataset)
    original_topology = deepcopy(topology)

    result = AnalysisService().analyze_dataset(dataset, topology=topology)

    assert dataset == original_dataset
    assert topology == original_topology
    assert result.dataset is not dataset
    assert result.topology is not topology
    assert result.dataset.events[0].component_id == "policy"
    assert any(component.kind == "operation" for component in result.topology.components)


def test_allowlisted_queries_filter_and_paginate_safely():
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[
                event(3, "mark", "req-2", sequence=2, component_id="two"),
                event(1, "mark", "req-1", sequence=0, component_id="one"),
                event(2, "other", "req-1", sequence=1, component_id="one"),
            ]
        )
    )
    service = QueryService(result)

    first = service.events(EventQuery(request_id="req-1", offset=0, limit=1))
    second = service.events(EventQuery(request_id="req-1", offset=first.next_offset, limit=1))

    assert [item.timestamp_ns for item in first.items] == [1]
    assert [item.timestamp_ns for item in second.items] == [2]
    assert first.total == 2
    assert second.next_offset is None
    with pytest.raises(ValueError, match="offset"):
        EventQuery(offset=-1)
    with pytest.raises(ValueError, match="limit"):
        EventQuery(limit=1001)
    with pytest.raises(ValueError, match="sort field"):
        RequestQuery(sort_by="__dict__")


def test_js_safe_serialization_strings_absolute_ns_and_unsafe_integers():
    value = to_js_safe(
        {
            "timestamp_ns": 10,
            "start_offset_ns": 10,
            "counter": JS_MAX_SAFE_INTEGER + 1,
            "nested": {"end_ns": 20},
        }
    )

    assert value == {
        "timestamp_ns": "10",
        "start_offset_ns": 10,
        "counter": str(JS_MAX_SAFE_INTEGER + 1),
        "nested": {"end_ns": "20"},
    }
    assert TraceEvent(10, "mark").to_dict()["timestamp_ns"] == 10


def test_timeline_projects_lanes_events_spans_flows_and_offsets():
    topology = TraceTopology(
        components=[Component("source", "Source", "module"), Component("target", "Target", "module")],
        edges=[DataFlowEdge("edge", "source", "target", "Transfer", "flow")],
    )
    dataset = TraceDataset(
        events=[
            event(1_000, "span_begin", "req", span_id="work", span_name="work", component_id="source"),
            event(1_100, "mark", "req", component_id="source"),
            event(1_200, "flow_send", "req", edge_id="edge", flow_id="transfer", component_id="source"),
            event(1_300, "flow_receive", "req", edge_id="edge", flow_id="transfer", component_id="target"),
            event(1_400, "span_end", "req", span_id="work", span_name="work", component_id="source"),
        ]
    )
    result = AnalysisService().analyze_dataset(dataset, topology=topology)

    timeline = project_timeline(result, TimelineQuery(request_id="req"))
    document = timeline.to_dict()

    assert {item.kind for item in timeline.items} == {"event", "span", "flow"}
    assert {lane.kind for lane in timeline.lanes} == {"component", "flow"}
    assert timeline.start_ns == 1_000
    assert min(item.start_offset_ns for item in timeline.items) == 0
    assert document["start_ns"] == "1000"
    assert isinstance(document["items"][0]["start_offset_ns"], int)
    assert len({item.id for item in timeline.items}) == len(timeline.items)


@pytest.mark.parametrize("with_manifest", [False, True])
@pytest.mark.parametrize(
    "component_id, request_id, start_ns, end_ns, expected",
    [
        ("source", "req", 15, 15, ["transfer"]),
        ("target", "req", 15, 15, ["transfer"]),
        ("unrelated", "req", 15, 15, []),
        ("source", "missing", 15, 15, []),
        ("source", "req", 20, 20, ["transfer"]),
        ("source", "req", 21, 29, []),
        ("source", "req", 0, 9, []),
    ],
)
def test_timeline_component_flows_reuse_query_filters(
    with_manifest, component_id, request_id, start_ns, end_ns, expected
):
    topology = (
        TraceTopology(
            components=[Component("source", "Source", "module"), Component("target", "Target", "module")],
            edges=[DataFlowEdge("edge", "source", "target", "Transfer", "flow")],
        )
        if with_manifest
        else None
    )
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[
                event(timestamp, name, request, edge_id="edge", flow_id=flow_id, component_id=component)
                for request, flow_id, send, receive in (
                    ("req", "transfer", 10, 20),
                    ("other-req", "other-transfer", 10, 20),
                    ("req", "later", 30, 40),
                )
                for timestamp, name, component in ((send, "flow_send", "source"), (receive, "flow_receive", "target"))
            ]
        ),
        topology=topology,
    )
    filters = dict(component_id=component_id, request_id=request_id, start_ns=start_ns, end_ns=end_ns)
    flows = QueryService(result).flows(FlowQuery(**filters)).items

    timeline = project_timeline(result, TimelineQuery(**filters, include_events=False, include_spans=False))

    assert [flow.flow_id for flow in flows] == expected
    assert [item.id for item in timeline.items] == [stable_flow_id(flow) for flow in flows]
    assert all(item.kind == "flow" for item in timeline.items)


def test_flat_call_tree_handles_orphans_cycles_and_overlapping_child_self_time():
    spans = [
        span("root", "root", 0, 10_000_000),
        span("left", "left", 2_000_000, 6_000_000, parent_span_id="root"),
        span("right", "right", 4_000_000, 8_000_000, parent_span_id="root"),
        span("orphan", "orphan", 11_000_000, 12_000_000, parent_span_id="missing"),
        span("cycle-a", "a", 13_000_000, 14_000_000, parent_span_id="b"),
        span("cycle-b", "b", 13_000_000, 14_000_000, parent_span_id="a"),
    ]
    result = AnalysisResult(TraceDataset(), spans=spans)

    tree = project_call_tree(result, CallTreeQuery(request_id="req"))
    by_span_id = {node.span_id: node for node in tree.nodes}

    assert by_span_id["root"].self_duration_ns == 4_000_000
    assert by_span_id["orphan"].orphan
    assert by_span_id["orphan"].id in tree.root_ids
    assert by_span_id["a"].cycle and by_span_id["b"].cycle
    assert by_span_id["a"].parent_id == by_span_id["b"].parent_id == ""
    assert stable_span_id(spans[0]) == by_span_id["root"].id


def test_graph_has_all_metrics_request_filter_views_and_observed_fallback():
    topology = TraceTopology(
        components=[Component("worker", "Worker", "module")],
        edges=[DataFlowEdge("edge", "worker", "worker", "Loop", "flow")],
    )
    spans = [
        span("work", "one", 0, 1_000_000, trace_id="req-1", component_id="worker"),
        span("work", "two", 0, 3_000_000, trace_id="req-2", component_id="worker"),
    ]
    result = AnalysisResult(
        TraceDataset(),
        spans=spans,
        flows=[FlowRecord("edge", "one", "req-1", 0, 2_000_000, "complete")],
        topology=topology,
    )

    components = project_graph(result, GraphQuery(metric="mean", view="components"))
    tracepoints = project_graph(result, GraphQuery(request_id="req-2", metric="p99", view="tracepoints"))
    worker = next(node for node in components.nodes if node.component_id == "worker")
    operation = next(node for node in tracepoints.nodes if node.kind == "operation")

    assert set(worker.metrics["processing_ms"]) == {"count", "minimum", "p50", "p95", "p99", "maximum", "mean"}
    assert worker.metric_value_ms == 2.0
    assert operation.metric_value_ms == 3.0
    assert all(node.kind != "operation" for node in components.nodes)
    assert QueryService(result).flows(FlowQuery(component_id="worker")).items == result.flows

    observed_dataset = TraceDataset(
        events=[
            event(0, "flow_send", "req", edge_id="observed", flow_id="one", component_id="source"),
            event(1_000_000, "flow_receive", "req", edge_id="observed", flow_id="one", component_id="target"),
        ]
    )
    observed = project_graph(AnalysisService().analyze_dataset(observed_dataset))

    assert observed.topology_source == "observed"
    assert {node.component_id for node in observed.nodes} == {"source", "target"}
    assert [(edge.source_id, edge.target_id) for edge in observed.edges] == [("source", "target")]


def test_graph_nodes_collapse_hierarchy_and_aggregate_bidirectional_metrics():
    topology = TraceTopology(
        components=[
            Component("source", "Camera", "data_source"),
            Component("node_a", "Node A", "ros_node"),
            Component("node_a.container", "Container", "module", "node_a"),
            Component("node_a.input", "Input", "module", "node_a.container"),
            Component("node_a.output", "Output", "module", "node_a.container"),
            Component("node_b", "Node B", "ros_node"),
            Component("node_b.input", "Input", "module", "node_b"),
            Component("node_b.operation.receive", "Receive", "operation", "node_b.input", trace_name="receive"),
            Component("sink", "Controller", "data_sink"),
        ],
        edges=[
            DataFlowEdge("source_in", "source", "node_a.input", "Image", "topic"),
            DataFlowEdge("internal", "node_a.input", "node_a.output", "Internal", "flow"),
            DataFlowEdge("a_to_b_1", "node_a.output", "node_b.input", "Request", "flow"),
            DataFlowEdge("a_to_b_2", "node_a.output", "node_b.operation.receive", "Retry", "flow"),
            DataFlowEdge("b_to_a", "node_b.input", "node_a.input", "Response", "flow"),
            DataFlowEdge("to_sink", "node_b.input", "sink", "Command", "topic"),
        ],
    )
    result = AnalysisResult(
        TraceDataset(),
        spans=[
            span("input", "a-in", 0, 1_000_000, component_id="node_a.input"),
            span("output", "a-out", 0, 3_000_000, component_id="node_a.output"),
            span("input", "b-in", 0, 5_000_000, component_id="node_b.input"),
        ],
        flows=[
            FlowRecord("internal", "internal", "req", 0, 1_000_000, "complete"),
            FlowRecord("a_to_b_1", "one", "req", 0, 2_000_000, "complete"),
            FlowRecord("a_to_b_2", "two", "req", 0, 4_000_000, "complete"),
            FlowRecord("b_to_a", "back", "req", 0, 6_000_000, "complete"),
        ],
        topology=topology,
    )

    graph = project_graph(result, GraphQuery(metric="mean", view="nodes"))
    by_id = {node.id: node for node in graph.nodes}
    data_edges = [edge for edge in graph.edges if edge.kind != "contains"]
    by_direction = {(edge.source_id, edge.target_id): edge for edge in data_edges}

    assert set(by_id) == {"source", "node_a", "node_b", "sink"}
    assert by_id["node_a"].metrics["processing_ms"]["count"] == 2
    assert by_id["node_a"].metric_value_ms == 2.0
    assert ("node_a", "node_a") not in by_direction
    assert by_direction[("node_a", "node_b")].metrics["latency_ms"]["count"] == 2
    assert by_direction[("node_a", "node_b")].metric_value_ms == 3.0
    assert by_direction[("node_b", "node_a")].metric_value_ms == 6.0
    assert by_direction[("node_a", "node_b")].id.startswith("derived:nodes:")
    assert by_direction[("node_a", "node_b")].provenance == "derived"


def test_observed_graph_without_manifest_or_flows_keeps_components_in_all_coarse_views():
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[
                event(1, "mark", "req", component_id="worker", origin="user"),
                event(2, "child_mark", "req", component_id="worker.child", origin="user"),
            ]
        )
    )

    components = project_graph(result, GraphQuery(view="components"))
    nodes = project_graph(result, GraphQuery(view="nodes"))

    assert [node.component_id for node in components.nodes] == ["worker.child"]
    assert [node.component_id for node in nodes.nodes] == ["worker"]


def test_request_scoped_graph_uses_first_actual_value_instead_of_percentile():
    result = AnalysisResult(
        TraceDataset(),
        spans=[
            span("first", "one", 0, 1_000_000, trace_id="req", component_id="worker"),
            span("second", "two", 0, 9_000_000, trace_id="req", component_id="worker"),
        ],
        topology=TraceTopology(components=[Component("worker", "Worker", "module")]),
    )

    graph = project_graph(result, GraphQuery(request_id="req", metric="p95", view="components"))

    assert graph.nodes[0].metrics["processing_ms"]["maximum"] == 9.0
    assert graph.nodes[0].metric_value_ms == 1.0


def test_component_view_hides_pure_containers_but_keeps_connected_root_node():
    topology = TraceTopology(
        components=[
            Component("policy", "Policy", "ros_node"),
            Component("policy.container", "Container", "module", "policy"),
            Component("policy.preprocess", "Preprocess", "module", "policy.container"),
            Component("cloud_inference", "Cloud", "ros_node"),
            Component("source", "Camera", "data_source"),
        ],
        edges=[
            DataFlowEdge("source_to_pre", "source", "policy.preprocess", "Image", "topic"),
            DataFlowEdge("pre_to_cloud", "policy.preprocess", "cloud_inference", "Inference", "flow"),
        ],
    )

    graph = project_graph(AnalysisResult(TraceDataset(), topology=topology), GraphQuery(view="components"))
    by_id = {node.id: node for node in graph.nodes}

    assert set(by_id) == {"policy.preprocess", "cloud_inference", "source"}
    assert by_id["policy.preprocess"].parent_id == "policy.container"
    assert "policy" not in by_id
    assert "policy.container" not in by_id


def test_tracepoint_view_contains_hierarchy_descriptions_and_distinct_event_span_ids():
    topology = TraceTopology(
        components=[
            Component("worker", "Worker", "ros_node"),
            Component("worker.module", "Worker Module", "module", "worker"),
        ]
    )
    definitions = [
        TracepointDefinition("span", "worker.module", "work", "user", "Runs one work item."),
        TracepointDefinition("event", "worker.module", "work", "built-in", "Reports completed work."),
    ]
    result = AnalysisResult(
        TraceDataset(
            events=[
                TraceEvent(
                    2,
                    "work",
                    {"component_id": "worker.module", "origin": "built-in"},
                    schema_version=1,
                )
            ]
        ),
        spans=[span("work", "work", 0, 1_000_000, component_id="worker.module")],
        topology=topology,
        definitions=definitions,
    )

    graph = project_graph(result, GraphQuery(view="tracepoints"))
    tracepoints = [node for node in graph.nodes if node.kind in {"operation", "instant_event"}]
    contains = [edge for edge in graph.edges if edge.kind == "contains"]
    by_kind = {node.kind: node for node in tracepoints}

    assert set(by_kind) == {"operation", "instant_event"}
    assert by_kind["operation"].id != by_kind["instant_event"].id
    assert by_kind["operation"].description == "Runs one work item."
    assert by_kind["instant_event"].description == "Reports completed work."
    assert by_kind["instant_event"].provenance == "observed"
    assert all(not edge.directed for edge in contains)
    assert {(edge.source_id, edge.target_id) for edge in contains} >= {
        ("worker", "worker.module"),
        ("worker.module", by_kind["operation"].id),
        ("worker.module", by_kind["instant_event"].id),
    }
    with pytest.raises(ValueError, match="unsupported graph view"):
        GraphQuery(view="systems")
