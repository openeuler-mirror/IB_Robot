"""UI-neutral timeline, call-tree, distribution, and graph projections."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from numbers import Real
from typing import Any, Literal
from urllib.parse import quote

from ._span_hierarchy import (
    DIAGNOSTIC_ORDER,
    SpanHierarchyNode,
    assign_tracks,
    build_span_hierarchy,
    interval_union_ns,
)
from .analysis import AnalysisResult
from .model import Component, DataFlowEdge, FlowRecord, NumericSummary, SpanRecord, TraceEvent, TraceTopology
from .query import (
    MAX_PAGE_SIZE,
    EventQuery,
    FlowQuery,
    QueryService,
    SpanQuery,
    stable_event_id,
    stable_flow_id,
    stable_span_id,
)
from .serialization import to_js_safe
from .statistics import DEFAULT_METRIC_PRIORITY, summarize
from .topology import normalize_topology

GraphMetric = Literal["p50", "p95", "p99", "minimum", "maximum", "mean"]
GraphView = Literal["nodes", "components", "tracepoints"]
GRAPH_METRICS = frozenset({"p50", "p95", "p99", "minimum", "maximum", "mean"})
GRAPH_VIEWS = frozenset({"nodes", "components", "tracepoints"})
MIN_DISTRIBUTION_BINS = 5
MAX_DISTRIBUTION_BINS = 100
DEFAULT_DISTRIBUTION_BINS = 30
MIN_DISTRIBUTION_OUTLIERS = 1
MAX_DISTRIBUTION_OUTLIERS = 100
DEFAULT_DISTRIBUTION_OUTLIERS = 20
MIN_BUCKET_REQUEST_LIMIT = 1
MAX_BUCKET_REQUEST_LIMIT = 1_000
DEFAULT_BUCKET_REQUEST_LIMIT = 100


class Projection:
    def to_dict(self) -> dict[str, Any]:
        return to_js_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class TimelineQuery:
    request_id: str = ""
    component_id: str = ""
    start_ns: int | None = None
    end_ns: int | None = None
    include_events: bool = True
    include_spans: bool = True
    include_flows: bool = True


@dataclass(frozen=True, slots=True)
class TimelineLane(Projection):
    id: str
    label: str
    kind: str
    component_id: str = ""
    edge_id: str = ""


@dataclass(frozen=True, slots=True)
class TimelineItem(Projection):
    id: str
    lane_id: str
    kind: str
    label: str
    start_ns: int
    end_ns: int
    start_offset_ns: int
    end_offset_ns: int
    duration_ns: int
    request_id: str = ""
    component_id: str = ""
    status: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimelineProjection(Projection):
    start_ns: int | None
    end_ns: int | None
    duration_ns: int
    lanes: list[TimelineLane]
    items: list[TimelineItem]


@dataclass(frozen=True, slots=True)
class CallTreeQuery:
    request_id: str = ""
    component_id: str = ""
    start_ns: int | None = None
    end_ns: int | None = None


@dataclass(frozen=True, slots=True)
class CallTreeNode(Projection):
    id: str
    span_id: str
    request_id: str
    parent_id: str
    parent_span_id: str
    child_ids: list[str]
    name: str
    component_id: str
    origin: str
    status: str
    start_ns: int
    end_ns: int | None
    start_offset_ns: int
    duration_ns: int | None
    self_duration_ns: int | None
    orphan: bool = False
    cycle: bool = False
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CallTreeProjection(Projection):
    start_ns: int | None
    root_ids: list[str]
    nodes: list[CallTreeNode]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LatencyDistributionQuery:
    metric: str | None = None
    bins: int = DEFAULT_DISTRIBUTION_BINS
    outlier_limit: int = DEFAULT_DISTRIBUTION_OUTLIERS
    bucket_request_limit: int = DEFAULT_BUCKET_REQUEST_LIMIT

    def __post_init__(self) -> None:
        if self.metric is not None and not _is_latency_metric_name(self.metric):
            raise ValueError(f"unsupported latency metric: {self.metric}")
        if not MIN_DISTRIBUTION_BINS <= self.bins <= MAX_DISTRIBUTION_BINS:
            raise ValueError(f"bins must be between {MIN_DISTRIBUTION_BINS} and {MAX_DISTRIBUTION_BINS}")
        if not MIN_DISTRIBUTION_OUTLIERS <= self.outlier_limit <= MAX_DISTRIBUTION_OUTLIERS:
            raise ValueError(
                f"outlier_limit must be between {MIN_DISTRIBUTION_OUTLIERS} and {MAX_DISTRIBUTION_OUTLIERS}"
            )
        if not MIN_BUCKET_REQUEST_LIMIT <= self.bucket_request_limit <= MAX_BUCKET_REQUEST_LIMIT:
            raise ValueError(
                f"bucket_request_limit must be between {MIN_BUCKET_REQUEST_LIMIT} and {MAX_BUCKET_REQUEST_LIMIT}"
            )


@dataclass(frozen=True, slots=True)
class LatencyDistributionBucket(Projection):
    index: int
    start: float
    end: float
    count: int
    request_ids: list[str]
    returned: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class LatencyDistributionOutlier(Projection):
    request_id: str
    value: float
    rank: int
    percentile: float
    p95_tail: bool


@dataclass(frozen=True, slots=True)
class LatencyDistributionProjection(Projection):
    metric: str | None
    unit: Literal["ms"]
    available_metrics: list[str]
    summary: NumericSummary
    sample_count: int
    invalid_count: int
    minimum: float | None
    maximum: float | None
    bin_count: int
    buckets: list[LatencyDistributionBucket]
    outliers: list[LatencyDistributionOutlier]


SpanProfileMode = Literal["request", "aggregate"]


@dataclass(frozen=True, slots=True)
class SpanProfileQuery:
    mode: SpanProfileMode = "request"
    request_id: str = ""
    component_id: str = ""
    start_ns: int | None = None
    end_ns: int | None = None
    origin: str = ""
    status: str = ""
    max_nodes: int = 10_000

    def __post_init__(self) -> None:
        if self.mode not in {"request", "aggregate"}:
            raise ValueError(f"unsupported span profile mode: {self.mode}")
        if self.mode == "request" and not self.request_id:
            raise ValueError("request mode requires request_id")
        if self.start_ns is not None and self.end_ns is not None and self.start_ns > self.end_ns:
            raise ValueError("start_ns must not be greater than end_ns")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")


@dataclass(frozen=True, slots=True)
class SpanProfileDiagnostic(Projection):
    code: str
    occurrence_ids: list[str]
    message: str


@dataclass(frozen=True, slots=True)
class RequestSpanProfileNode(Projection):
    id: str
    occurrence_id: str
    span_id: str
    request_id: str
    parent_id: str
    parent_span_id: str
    child_ids: list[str]
    depth: int
    track: int
    component_id: str
    name: str
    origin: str
    status: str
    start_ns: int
    end_ns: int | None
    observed_end_ns: int | None
    start_offset_ns: int
    duration_ns: int | None
    duration_source: str
    uncovered_wall_ns: int | None
    diagnostics: list[str]
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AggregateSpanProfileNode(Projection):
    id: str
    parent_id: str
    child_ids: list[str]
    depth: int
    component_id: str
    name: str
    origin: str
    path: list[dict[str, str]]
    self_value_ns: int
    value_ns: int
    occurrence_count: int
    request_count: int
    error_count: int
    incomplete_count: int
    invalid_count: int
    percentage: float


@dataclass(frozen=True, slots=True)
class RequestSpanProfileProjection(Projection):
    mode: Literal["request"]
    measurement: Literal["instrumented_wall"]
    source: str
    coverage: str
    not_cpu: bool
    request_id: str
    start_ns: int | None
    end_ns: int | None
    duration_ns: int
    roots: list[str]
    nodes: list[RequestSpanProfileNode]
    totals: dict[str, Any]
    diagnostics: list[SpanProfileDiagnostic]
    total_nodes: int
    returned_nodes: int
    truncated: bool
    truncation_reason: str


@dataclass(frozen=True, slots=True)
class AggregateSpanProfileProjection(Projection):
    mode: Literal["aggregate"]
    measurement: Literal["instrumented_wall"]
    source: str
    coverage: str
    not_cpu: bool
    roots: list[str]
    nodes: list[AggregateSpanProfileNode]
    totals: dict[str, Any]
    diagnostics: list[SpanProfileDiagnostic]
    total_nodes: int
    returned_nodes: int
    truncated: bool
    truncation_reason: str


SpanProfileProjection = RequestSpanProfileProjection | AggregateSpanProfileProjection


@dataclass(frozen=True, slots=True)
class GraphQuery:
    request_id: str = ""
    metric: GraphMetric = "p95"
    view: GraphView = "components"

    def __post_init__(self) -> None:
        if self.metric not in GRAPH_METRICS:
            raise ValueError(f"unsupported graph metric: {self.metric}")
        if self.view not in GRAPH_VIEWS:
            raise ValueError(f"unsupported graph view: {self.view}")


@dataclass(frozen=True, slots=True)
class GraphNode(Projection):
    id: str
    component_id: str
    name: str
    kind: str
    parent_id: str
    child_ids: list[str]
    depth: int
    provenance: str
    trace_name: str
    metrics: dict[str, Any]
    metric_value_ms: float | None
    description: str = ""


@dataclass(frozen=True, slots=True)
class GraphEdge(Projection):
    id: str
    edge_id: str
    source_id: str
    target_id: str
    name: str
    kind: str
    provenance: str
    data_type: str
    contract_key: str
    metrics: dict[str, Any]
    metric_value_ms: float | None
    directed: bool = True


@dataclass(frozen=True, slots=True)
class GraphProjection(Projection):
    view: GraphView
    metric: GraphMetric
    request_id: str
    topology_source: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def _is_latency_metric_name(name: Any) -> bool:
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


def _ordered_distribution_metrics(metrics: set[str]) -> list[str]:
    priority = {metric: index for index, metric in enumerate(DEFAULT_METRIC_PRIORITY)}
    return sorted(metrics, key=lambda metric: (priority.get(metric, len(priority)), metric))


def project_latency_distribution(
    result: AnalysisResult,
    query: LatencyDistributionQuery = LatencyDistributionQuery(),
) -> LatencyDistributionProjection:
    known_metrics = {key for row in result.request_rows for key in row if _is_latency_metric_name(key)}
    available_metrics = _ordered_distribution_metrics(
        {
            metric
            for metric in known_metrics
            if any(_finite_number(row.get(metric)) is not None for row in result.request_rows)
        }
    )
    metric = query.metric or (available_metrics[0] if available_metrics else None)
    if metric is not None and metric not in known_metrics:
        raise ValueError(f"latency metric is not available: {metric}")
    if metric is None:
        return LatencyDistributionProjection(
            metric=None,
            unit="ms",
            available_metrics=available_metrics,
            summary=NumericSummary(count=0),
            sample_count=0,
            invalid_count=0,
            minimum=None,
            maximum=None,
            bin_count=0,
            buckets=[],
            outliers=[],
        )

    samples: list[tuple[float, str, int]] = []
    invalid_count = 0
    for index, row in enumerate(result.request_rows):
        value = _finite_number(row.get(metric))
        if value is None:
            invalid_count += 1
            continue
        samples.append((value, str(row.get("request_id", "")), index))
    samples.sort(key=lambda sample: (sample[0], sample[1], sample[2]))
    values = [sample[0] for sample in samples]
    summary = summarize(values)
    if not samples:
        return LatencyDistributionProjection(
            metric=metric,
            unit="ms",
            available_metrics=available_metrics,
            summary=summary,
            sample_count=0,
            invalid_count=invalid_count,
            minimum=None,
            maximum=None,
            bin_count=0,
            buckets=[],
            outliers=[],
        )

    minimum = values[0]
    maximum = values[-1]
    bin_count = min(query.bins, len(samples))
    value_range = maximum - minimum
    if minimum == maximum or not math.isfinite(value_range):
        bin_count = 1
    width = value_range / bin_count if bin_count > 1 else 0.0
    if bin_count > 1 and width == 0.0:
        bin_count = 1

    bucket_samples: list[list[tuple[float, str, int]]] = [[] for _ in range(bin_count)]
    for sample in samples:
        value = sample[0]
        index = bin_count - 1 if value == maximum or bin_count == 1 else int((value - minimum) / width)
        bucket_samples[min(index, bin_count - 1)].append(sample)

    buckets = []
    for index, items in enumerate(bucket_samples):
        start = minimum if index == 0 else minimum + width * index
        end = maximum if index == bin_count - 1 else minimum + width * (index + 1)
        request_ids = [sample[1] for sample in items[: query.bucket_request_limit]]
        buckets.append(
            LatencyDistributionBucket(
                index=index,
                start=start,
                end=end,
                count=len(items),
                request_ids=request_ids,
                returned=len(request_ids),
                truncated=len(items) > len(request_ids),
            )
        )

    p95 = summary.p95
    descending = sorted(samples, key=lambda sample: (-sample[0], sample[1], sample[2]))
    outliers = [
        LatencyDistributionOutlier(
            request_id=request_id,
            value=value,
            rank=rank,
            percentile=round(bisect_right(values, value) * 100 / len(values), 6),
            p95_tail=p95 is not None and value >= p95,
        )
        for rank, (value, request_id, _index) in enumerate(descending[: query.outlier_limit], start=1)
    ]
    return LatencyDistributionProjection(
        metric=metric,
        unit="ms",
        available_metrics=available_metrics,
        summary=summary,
        sample_count=len(samples),
        invalid_count=invalid_count,
        minimum=minimum,
        maximum=maximum,
        bin_count=bin_count,
        buckets=buckets,
        outliers=outliers,
    )


def _all_pages(fetch: Any, query: Any) -> list[Any]:
    items = []
    current = replace(query, offset=0, limit=MAX_PAGE_SIZE)
    while True:
        page = fetch(current)
        items.extend(page.items)
        if page.next_offset is None:
            return items
        current = replace(current, offset=page.next_offset)


def _event_component(event: TraceEvent) -> str:
    return event.component_id or event.origin.provider


def _span_duration_ns(span: SpanRecord) -> int | None:
    duration = span.fields.get("duration_ns")
    try:
        if duration is not None:
            return max(0, int(duration))
    except (TypeError, ValueError):
        pass
    if span.end_ns is None:
        return None
    return max(0, span.end_ns - span.start_ns)


def _flow_bounds(flow: FlowRecord) -> tuple[int, int] | None:
    points = [point for point in (flow.send_ns, flow.receive_ns) if point is not None]
    return (min(points), max(points)) if points else None


def _timeline_from_records(
    events: list[TraceEvent],
    spans: list[SpanRecord],
    flows: list[FlowRecord],
    topology: TraceTopology | None,
) -> TimelineProjection:
    points = [event.timestamp_ns for event in events]
    points.extend(span.start_ns for span in spans)
    points.extend(span.end_ns for span in spans if span.end_ns is not None)
    for flow in flows:
        points.extend(point for point in (flow.send_ns, flow.receive_ns) if point is not None)
    if not points:
        return TimelineProjection(None, None, 0, [], [])

    start_ns = min(points)
    end_ns = max(points)
    component_names = (
        {component.component_id: component.name for component in topology.components} if topology is not None else {}
    )
    lanes: dict[str, TimelineLane] = {}
    items: list[TimelineItem] = []

    def component_lane(component_id: str) -> str:
        identity = component_id or "unassigned"
        lane_id = f"lane:component:{identity}"
        lanes.setdefault(
            lane_id,
            TimelineLane(
                lane_id, component_names.get(component_id, component_id or "Unassigned"), "component", component_id
            ),
        )
        return lane_id

    for event in events:
        component_id = _event_component(event)
        item_start = event.timestamp_ns
        items.append(
            TimelineItem(
                stable_event_id(event),
                component_lane(component_id),
                "event",
                event.name,
                item_start,
                item_start,
                item_start - start_ns,
                item_start - start_ns,
                0,
                event.request_id,
                component_id,
                fields=dict(event.fields),
            )
        )
    for span in spans:
        item_end = span.end_ns if span.end_ns is not None else span.start_ns
        duration_ns = _span_duration_ns(span)
        items.append(
            TimelineItem(
                stable_span_id(span),
                component_lane(span.component_id),
                "span",
                span.name,
                span.start_ns,
                item_end,
                span.start_ns - start_ns,
                item_end - start_ns,
                duration_ns or 0,
                span.trace_id,
                span.component_id,
                span.status,
                dict(span.fields),
            )
        )
    for flow in flows:
        bounds = _flow_bounds(flow)
        if bounds is None:
            continue
        item_start, item_end = bounds
        lane_id = f"lane:flow:{flow.edge_id}"
        lanes.setdefault(lane_id, TimelineLane(lane_id, flow.edge_id, "flow", edge_id=flow.edge_id))
        items.append(
            TimelineItem(
                stable_flow_id(flow),
                lane_id,
                "flow",
                flow.edge_id,
                item_start,
                item_end,
                item_start - start_ns,
                item_end - start_ns,
                item_end - item_start,
                flow.trace_id,
                status=flow.status,
                fields={"edge_id": flow.edge_id, "flow_id": flow.flow_id},
            )
        )
    items.sort(key=lambda item: (item.start_ns, {"span": 0, "event": 1, "flow": 2}[item.kind]))
    ordered_lanes = sorted(lanes.values(), key=lambda lane: (lane.kind, lane.id))
    return TimelineProjection(start_ns, end_ns, end_ns - start_ns, ordered_lanes, items)


def project_timeline(result: AnalysisResult, query: TimelineQuery = TimelineQuery()) -> TimelineProjection:
    service = QueryService(result)
    events = (
        _all_pages(
            service.events,
            EventQuery(
                request_id=query.request_id,
                component_id=query.component_id,
                start_ns=query.start_ns,
                end_ns=query.end_ns,
            ),
        )
        if query.include_events
        else []
    )
    spans = (
        _all_pages(
            service.spans,
            SpanQuery(
                request_id=query.request_id,
                component_id=query.component_id,
                start_ns=query.start_ns,
                end_ns=query.end_ns,
            ),
        )
        if query.include_spans
        else []
    )
    flows = (
        _all_pages(
            service.flows,
            FlowQuery(
                request_id=query.request_id,
                component_id=query.component_id,
                start_ns=query.start_ns,
                end_ns=query.end_ns,
            ),
        )
        if query.include_flows
        else []
    )
    return _timeline_from_records(events, spans, flows, result.topology)


def project_event_timeline(events: list[TraceEvent]) -> TimelineProjection:
    """Compatibility helper for callers that only have raw events."""
    return _timeline_from_records(events, [], [], None)


def _call_tree_from_spans(spans: list[SpanRecord]) -> CallTreeProjection:
    hierarchy = build_span_hierarchy(spans)
    if not hierarchy.nodes:
        return CallTreeProjection(None, [], [])
    base_ns = min(node.span.start_ns for node in hierarchy.nodes)
    nodes = [
        CallTreeNode(
            id=node.occurrence_id,
            span_id=node.span.span_id,
            request_id=node.span.trace_id,
            parent_id=node.parent_id,
            parent_span_id=node.span.parent_span_id,
            child_ids=list(node.child_ids),
            name=node.span.name,
            component_id=node.span.component_id,
            origin=node.span.origin,
            status=node.span.status,
            start_ns=node.span.start_ns,
            end_ns=node.span.end_ns,
            start_offset_ns=node.span.start_ns - base_ns,
            duration_ns=node.compatibility_duration_ns,
            self_duration_ns=node.compatibility_uncovered_ns,
            orphan="orphan" in node.diagnostics,
            cycle="cycle" in node.diagnostics,
            fields=dict(node.span.fields),
        )
        for node in hierarchy.nodes
    ]
    warnings = []
    orphan_count = sum("orphan" in node.diagnostics for node in hierarchy.nodes)
    cycle_count = sum("cycle" in node.diagnostics for node in hierarchy.nodes)
    if orphan_count:
        warnings.append(f"{orphan_count} orphan span(s)")
    if cycle_count:
        warnings.append(f"{cycle_count} cyclic span(s)")
    return CallTreeProjection(base_ns, hierarchy.roots, nodes, warnings)


def project_call_tree(result: AnalysisResult, query: CallTreeQuery = CallTreeQuery()) -> CallTreeProjection:
    spans = _all_pages(
        QueryService(result).spans,
        SpanQuery(
            request_id=query.request_id,
            component_id=query.component_id,
            start_ns=query.start_ns,
            end_ns=query.end_ns,
        ),
    )
    return _call_tree_from_spans(spans)


def project_span_call_tree(spans: list[SpanRecord]) -> CallTreeProjection:
    """Compatibility helper for callers that only have span records."""
    return _call_tree_from_spans(spans)


_SPAN_PROFILE_SOURCE = "structured span instrumentation; monotonic duration_ns is preferred for geometry"
_SPAN_PROFILE_COVERAGE = "selected instrumented spans only; uninstrumented work is not represented"


@dataclass(slots=True)
class _VisibleSpanHierarchy:
    by_id: dict[str, SpanHierarchyNode]
    selected_ids: set[str]
    parent_by_id: dict[str, str]
    children: dict[str, list[str]]
    roots: list[str]
    depth: dict[str, int]
    tracks: dict[str, int]
    node_diagnostics: dict[str, set[str]]
    diagnostics: list[SpanProfileDiagnostic]
    order: list[str]


def _profile_matches(node: SpanHierarchyNode, query: SpanProfileQuery) -> bool:
    span = node.span
    return (
        (not query.request_id or span.trace_id == query.request_id)
        and (not query.component_id or span.component_id == query.component_id)
        and (not query.origin or span.origin == query.origin)
        and (not query.status or span.status == query.status)
        and (query.start_ns is None or node.end_ns is None or node.end_ns >= query.start_ns)
        and (query.end_ns is None or span.start_ns <= query.end_ns)
    )


def _visible_span_hierarchy(result: AnalysisResult, query: SpanProfileQuery) -> _VisibleSpanHierarchy:
    hierarchy_spans = (
        [span for span in result.spans if span.trace_id == query.request_id] if query.request_id else result.spans
    )
    hierarchy = build_span_hierarchy(hierarchy_spans)
    by_id = hierarchy.by_id
    selected_ids = {node.occurrence_id for node in hierarchy.nodes if _profile_matches(node, query)}
    parent_by_id = {
        occurrence_id: by_id[occurrence_id].parent_id
        for occurrence_id in selected_ids
        if by_id[occurrence_id].parent_id in selected_ids
    }
    children: dict[str, list[str]] = defaultdict(list)
    for occurrence_id, parent_id in parent_by_id.items():
        children[parent_id].append(occurrence_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda occurrence_id: (by_id[occurrence_id].span.start_ns, occurrence_id))
    roots = [occurrence_id for occurrence_id in selected_ids if occurrence_id not in parent_by_id]
    roots.sort(key=lambda occurrence_id: (by_id[occurrence_id].span.start_ns, occurrence_id))

    depth: dict[str, int] = {}
    order = []
    stack = [(root_id, 0) for root_id in reversed(roots)]
    while stack:
        occurrence_id, value = stack.pop()
        depth[occurrence_id] = value
        order.append(occurrence_id)
        stack.extend((child_id, value + 1) for child_id in reversed(children.get(occurrence_id, [])))

    roots_by_request: dict[str, list[str]] = defaultdict(list)
    for root_id in roots:
        roots_by_request[by_id[root_id].span.trace_id].append(root_id)
    tracks: dict[str, int] = {}
    for sibling_ids in [*roots_by_request.values(), *(children.values())]:
        tracks.update(assign_tracks(sibling_ids, by_id))

    node_diagnostics = {occurrence_id: set(by_id[occurrence_id].diagnostics) for occurrence_id in selected_ids}
    diagnostics = [
        SpanProfileDiagnostic(diagnostic.code, list(diagnostic.occurrence_ids), diagnostic.message)
        for diagnostic in hierarchy.diagnostics
        if selected_ids.intersection(diagnostic.member_ids)
    ]
    for occurrence_id in sorted(selected_ids):
        node = by_id[occurrence_id]
        if node.candidate_parent_id and node.candidate_parent_id not in selected_ids:
            node_diagnostics[occurrence_id].add("filtered_parent")
            parent = by_id[node.candidate_parent_id]
            diagnostics.append(
                SpanProfileDiagnostic(
                    "filtered_parent",
                    [occurrence_id, node.candidate_parent_id],
                    f"parent span was removed by profile filters: {node.span.trace_id}/{node.span.span_id}, "
                    f"{parent.span.trace_id}/{parent.span.span_id}",
                )
            )
    diagnostic_rank = {code: index for index, code in enumerate(DIAGNOSTIC_ORDER)}
    diagnostics.sort(
        key=lambda diagnostic: (
            diagnostic_rank.get(diagnostic.code, len(diagnostic_rank)),
            diagnostic.occurrence_ids,
        )
    )
    return _VisibleSpanHierarchy(
        by_id,
        selected_ids,
        parent_by_id,
        children,
        roots,
        depth,
        tracks,
        node_diagnostics,
        diagnostics,
        order,
    )


def _retained_occurrences(visible: _VisibleSpanHierarchy, max_nodes: int) -> set[str]:
    priority = sorted(
        visible.selected_ids,
        key=lambda occurrence_id: (
            visible.depth[occurrence_id],
            visible.by_id[occurrence_id].span.start_ns,
            occurrence_id,
        ),
    )
    return set(priority[:max_nodes])


def _profile_totals(visible: _VisibleSpanHierarchy) -> dict[str, Any]:
    nodes = [visible.by_id[occurrence_id] for occurrence_id in visible.selected_ids]
    sampled_wall_ns = sum(node.uncovered_wall_ns or 0 for node in nodes if node.valid_for_weight)
    nodes_by_request: dict[str, list[SpanHierarchyNode]] = defaultdict(list)
    for node in nodes:
        nodes_by_request[node.span.trace_id].append(node)
    wall_union_ns = sum(interval_union_ns(request_nodes) for request_nodes in nodes_by_request.values())
    diagnostic_counts = Counter(
        diagnostic for occurrence_id in visible.selected_ids for diagnostic in visible.node_diagnostics[occurrence_id]
    )
    return {
        "request_count": len(nodes_by_request),
        "occurrence_count": len(nodes),
        "error_count": sum(node.span.status == "error" for node in nodes),
        "incomplete_count": sum(node.incomplete for node in nodes),
        "invalid_count": sum(node.negative_duration for node in nodes),
        "excluded_weight_count": sum(not node.valid_for_weight for node in nodes),
        "sampled_uncovered_wall_ns": sampled_wall_ns,
        "uncovered_wall_ns": sampled_wall_ns,
        "selected_wall_union_ns": wall_union_ns,
        "concurrency_factor": sampled_wall_ns / wall_union_ns if wall_union_ns else 0.0,
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
    }


def _project_request_span_profile(
    visible: _VisibleSpanHierarchy,
    query: SpanProfileQuery,
) -> RequestSpanProfileProjection:
    retained = _retained_occurrences(visible, query.max_nodes)
    ordered_ids = [occurrence_id for occurrence_id in visible.order if occurrence_id in retained]
    starts = [visible.by_id[occurrence_id].span.start_ns for occurrence_id in visible.selected_ids]
    ends = [
        visible.by_id[occurrence_id].end_ns
        if visible.by_id[occurrence_id].end_ns is not None
        else visible.by_id[occurrence_id].span.start_ns
        for occurrence_id in visible.selected_ids
    ]
    start_ns = min(starts) if starts else None
    end_ns = max(ends) if ends else None
    nodes = []
    for occurrence_id in ordered_ids:
        node = visible.by_id[occurrence_id]
        span = node.span
        nodes.append(
            RequestSpanProfileNode(
                id=occurrence_id,
                occurrence_id=occurrence_id,
                span_id=span.span_id,
                request_id=span.trace_id,
                parent_id=visible.parent_by_id.get(occurrence_id, ""),
                parent_span_id=span.parent_span_id,
                child_ids=[child_id for child_id in visible.children.get(occurrence_id, []) if child_id in retained],
                depth=visible.depth[occurrence_id],
                track=visible.tracks.get(occurrence_id, 0),
                component_id=span.component_id,
                name=span.name,
                origin=span.origin,
                status=span.status,
                start_ns=span.start_ns,
                end_ns=node.end_ns,
                observed_end_ns=node.observed_end_ns,
                start_offset_ns=span.start_ns - start_ns if start_ns is not None else 0,
                duration_ns=node.duration_ns,
                duration_source=node.duration_source,
                uncovered_wall_ns=node.uncovered_wall_ns,
                diagnostics=sorted(
                    visible.node_diagnostics[occurrence_id],
                    key=lambda code: DIAGNOSTIC_ORDER.index(code),
                ),
                fields=dict(span.fields),
            )
        )
    total_nodes = len(visible.selected_ids)
    truncated = len(retained) < total_nodes
    return RequestSpanProfileProjection(
        mode="request",
        measurement="instrumented_wall",
        source=_SPAN_PROFILE_SOURCE,
        coverage=_SPAN_PROFILE_COVERAGE,
        not_cpu=True,
        request_id=query.request_id,
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ns=(end_ns - start_ns) if start_ns is not None and end_ns is not None else 0,
        roots=[root_id for root_id in visible.roots if root_id in retained],
        nodes=nodes,
        totals=_profile_totals(visible),
        diagnostics=visible.diagnostics,
        total_nodes=total_nodes,
        returned_nodes=len(nodes),
        truncated=truncated,
        truncation_reason="max_nodes" if truncated else "",
    )


@dataclass(slots=True)
class _AggregateEntry:
    id: str
    identity: tuple[str, str, str]
    parent_id: str
    depth: int
    child_ids: set[str] = field(default_factory=set)
    self_value_ns: int = 0
    value_ns: int = 0
    occurrence_count: int = 0
    request_ids: set[str] = field(default_factory=set)
    error_count: int = 0
    incomplete_count: int = 0
    invalid_count: int = 0


def _aggregate_id(parent_id: str, identity: tuple[str, str, str]) -> str:
    payload = json.dumps((parent_id, identity), ensure_ascii=True, separators=(",", ":"))
    return f"span-profile:{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def _project_aggregate_span_profile(
    visible: _VisibleSpanHierarchy,
    query: SpanProfileQuery,
) -> AggregateSpanProfileProjection:
    entries: dict[str, _AggregateEntry] = {}
    entries_by_key: dict[tuple[str, tuple[str, str, str]], _AggregateEntry] = {}
    entry_by_occurrence: dict[str, str] = {}
    for occurrence_id in visible.order:
        node = visible.by_id[occurrence_id]
        identity = (node.span.component_id, node.span.name, node.span.origin)
        parent_id = entry_by_occurrence.get(visible.parent_by_id.get(occurrence_id, ""), "")
        key = (parent_id, identity)
        entry = entries_by_key.get(key)
        if entry is None:
            entry = _AggregateEntry(
                _aggregate_id(parent_id, identity),
                identity,
                parent_id,
                entries[parent_id].depth + 1 if parent_id else 0,
            )
            entries[entry.id] = entry
            entries_by_key[key] = entry
            if parent_id:
                entries[parent_id].child_ids.add(entry.id)
        entry_by_occurrence[occurrence_id] = entry.id
        entry.occurrence_count += 1
        entry.request_ids.add(node.span.trace_id)
        entry.error_count += node.span.status == "error"
        entry.incomplete_count += node.incomplete
        entry.invalid_count += node.negative_duration
        if node.valid_for_weight and node.uncovered_wall_ns:
            entry.self_value_ns += node.uncovered_wall_ns

    root_ids = [entry_id for entry_id, entry in entries.items() if not entry.parent_id]

    stack = [(root_id, False) for root_id in reversed(root_ids)]
    while stack:
        entry_id, expanded = stack.pop()
        entry = entries[entry_id]
        if expanded:
            entry.value_ns = entry.self_value_ns + sum(entries[child_id].value_ns for child_id in entry.child_ids)
            continue
        stack.append((entry_id, True))
        stack.extend((child_id, False) for child_id in sorted(entry.child_ids, reverse=True))
    total_value_ns = sum(entries[root_id].value_ns for root_id in root_ids)

    def entry_order(entry_id: str) -> tuple[Any, ...]:
        entry = entries[entry_id]
        return (-entry.value_ns, entry.identity, entry.id)

    lexical_rank: dict[str, int] = {}
    stack = list(reversed(sorted(root_ids, key=lambda entry_id: (entries[entry_id].identity, entries[entry_id].id))))
    while stack:
        entry_id = stack.pop()
        lexical_rank[entry_id] = len(lexical_rank)
        child_ids = sorted(
            entries[entry_id].child_ids,
            key=lambda child_id: (entries[child_id].identity, entries[child_id].id),
        )
        stack.extend(reversed(child_ids))

    retained_ids = set(
        sorted(
            entries,
            key=lambda entry_id: (
                entries[entry_id].depth,
                -entries[entry_id].value_ns,
                lexical_rank[entry_id],
            ),
        )[: query.max_nodes]
    )
    ordered_ids = []
    stack = list(reversed(sorted(root_ids, key=entry_order)))
    while stack:
        entry_id = stack.pop()
        if entry_id not in retained_ids:
            continue
        ordered_ids.append(entry_id)
        child_ids = sorted(entries[entry_id].child_ids, key=entry_order)
        stack.extend(reversed(child_ids))

    def entry_path(entry_id: str) -> list[dict[str, str]]:
        path = []
        while entry_id:
            entry = entries[entry_id]
            component_id, name, origin = entry.identity
            path.append({"component_id": component_id, "name": name, "origin": origin})
            entry_id = entry.parent_id
        path.reverse()
        return path

    nodes = []
    for entry_id in ordered_ids:
        entry = entries[entry_id]
        component_id, name, origin = entry.identity
        nodes.append(
            AggregateSpanProfileNode(
                id=entry.id,
                parent_id=entry.parent_id,
                child_ids=[
                    child_id for child_id in sorted(entry.child_ids, key=entry_order) if child_id in retained_ids
                ],
                depth=entry.depth,
                component_id=component_id,
                name=name,
                origin=origin,
                path=entry_path(entry_id),
                self_value_ns=entry.self_value_ns,
                value_ns=entry.value_ns,
                occurrence_count=entry.occurrence_count,
                request_count=len(entry.request_ids),
                error_count=entry.error_count,
                incomplete_count=entry.incomplete_count,
                invalid_count=entry.invalid_count,
                percentage=(entry.value_ns / total_value_ns * 100) if total_value_ns else 0.0,
            )
        )
    totals = _profile_totals(visible)
    totals["value_ns"] = total_value_ns
    total_nodes = len(entries)
    truncated = len(retained_ids) < total_nodes
    return AggregateSpanProfileProjection(
        mode="aggregate",
        measurement="instrumented_wall",
        source=_SPAN_PROFILE_SOURCE,
        coverage=_SPAN_PROFILE_COVERAGE,
        not_cpu=True,
        roots=[root_id for root_id in sorted(root_ids, key=entry_order) if root_id in retained_ids],
        nodes=nodes,
        totals=totals,
        diagnostics=visible.diagnostics,
        total_nodes=total_nodes,
        returned_nodes=len(nodes),
        truncated=truncated,
        truncation_reason="max_nodes" if truncated else "",
    )


def project_span_profile(result: AnalysisResult, query: SpanProfileQuery) -> SpanProfileProjection:
    """Project structured spans as request intervals or aggregate full-path wall weights."""
    visible = _visible_span_hierarchy(result, query)
    if query.mode == "request":
        return _project_request_span_profile(visible, query)
    return _project_aggregate_span_profile(visible, query)


def span_profile_speedscope(profile: AggregateSpanProfileProjection) -> dict[str, Any]:
    """Convert a complete aggregate profile to weighted sampled speedscope JSON."""
    if profile.truncated:
        raise ValueError("cannot export a truncated span profile to speedscope")
    frame_keys = sorted(
        {
            (frame["component_id"], frame["name"], frame["origin"])
            for node in profile.nodes
            if node.self_value_ns > 0
            for frame in node.path
        }
    )
    frame_indexes = {key: index for index, key in enumerate(frame_keys)}
    samples = []
    for node in sorted(
        (item for item in profile.nodes if item.self_value_ns > 0),
        key=lambda item: tuple((part["component_id"], part["name"], part["origin"]) for part in item.path),
    ):
        samples.append(
            (
                [frame_indexes[(frame["component_id"], frame["name"], frame["origin"])] for frame in node.path],
                node.self_value_ns,
            )
        )
    total_weight = sum(weight for _, weight in samples)
    return {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "name": "IB-Robot Instrumented Span Wall Time (not CPU)",
        "activeProfileIndex": 0,
        "exporter": "ibrobot_tracing",
        "shared": {
            "frames": [
                {"name": f"{name} [{origin}] ({component_id or '-'})"} for component_id, name, origin in frame_keys
            ]
        },
        "profiles": [
            {
                "type": "sampled",
                "name": "Instrumented Span Wall Time (not CPU)",
                "unit": "nanoseconds",
                "startValue": 0,
                "endValue": total_weight,
                "samples": [sample for sample, _ in samples],
                "weights": [weight for _, weight in samples],
            }
        ],
    }


def _observed_topology(result: AnalysisResult) -> TraceTopology:
    component_ids = {span.component_id for span in result.spans if span.component_id} | {
        _event_component(event) for event in result.dataset.events if _event_component(event)
    }
    components = []
    for component_id in sorted(component_ids):
        parent_id = ""
        parts = component_id.split(".")
        for index in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:index])
            if candidate in component_ids:
                parent_id = candidate
                break
        components.append(
            Component(
                component_id,
                component_id.replace("_", " ").replace(".", " / ").title(),
                "observed_component",
                parent_id,
                provenance="observed",
            )
        )

    event_by_flow: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for event in result.dataset.events:
        if event.name not in {"flow_send", "flow_receive"}:
            continue
        edge_id = str(event.field("edge_id", ""))
        flow_id = str(event.field("flow_id", ""))
        if edge_id and flow_id:
            event_by_flow[(event.request_id, edge_id, flow_id)][event.name] = _event_component(event)
    observed_edges: dict[str, DataFlowEdge] = {}
    for flow in result.flows:
        endpoints = event_by_flow.get((flow.trace_id, flow.edge_id, flow.flow_id), {})
        source_id = endpoints.get("flow_send", "")
        target_id = endpoints.get("flow_receive", "")
        if source_id and target_id:
            observed_edges.setdefault(
                flow.edge_id,
                DataFlowEdge(
                    flow.edge_id,
                    source_id,
                    target_id,
                    flow.edge_id,
                    "observed_flow",
                    provenance="observed",
                ),
            )
    return TraceTopology(components=components, edges=list(observed_edges.values()))


_STRUCTURAL_EVENTS = {"span_begin", "span_end", "flow_send", "flow_receive"}


def _tracepoint_node_id(kind: str, component_id: str, trace_name: str, origin: str) -> str:
    identity = ":".join(quote(value, safe="") for value in (component_id, trace_name, origin))
    return f"tracepoint:{kind}:{identity}"


def _tracepoint_components(
    result: AnalysisResult,
    topology: TraceTopology,
) -> tuple[list[Component], dict[str, str], dict[str, str]]:
    base_ids = {
        component.component_id
        for component in topology.components
        if component.kind not in {"operation", "instant_event"}
    }
    definitions = {definition.identity: definition.description for definition in result.definitions}
    tracepoints: dict[tuple[str, str, str, str], Component] = {}
    legacy_endpoint_ids: dict[str, list[tuple[str, str]]] = defaultdict(list)
    span_origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    event_origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    for span in result.spans:
        span_origins[(span.component_id, span.name)].add(span.origin)
    for event in result.dataset.events:
        if event.name not in _STRUCTURAL_EVENTS:
            event_origins[(event.component_id, event.name)].add(str(event.field("origin", "built-in")))

    def add(
        kind: str,
        parent_id: str,
        trace_name: str,
        origin: str,
        provenance: str,
        *,
        name: str = "",
        legacy_id: str = "",
    ) -> None:
        if not parent_id or parent_id not in base_ids or not trace_name:
            return
        identity = (kind, parent_id, trace_name, origin)
        component_id = _tracepoint_node_id(kind, parent_id, trace_name, origin)
        existing = tracepoints.get(identity)
        if existing is None or (existing.provenance == "observed" and provenance == "declared"):
            tracepoints[identity] = Component(
                component_id,
                name or trace_name.replace("_", " ").title(),
                "operation" if kind == "span" else "instant_event",
                parent_id,
                provenance=provenance,
                trace_name=trace_name,
            )
        if legacy_id:
            legacy_endpoint_ids[legacy_id].append((origin, component_id))

    for component in topology.components:
        if component.kind not in {"operation", "instant_event"}:
            continue
        kind = "span" if component.kind == "operation" else "event"
        origin = "built-in"
        if component.provenance == "observed":
            key = (component.parent_id, component.trace_name)
            observed_origins = set(span_origins[key] if kind == "span" else event_origins[key])
            if observed_origins:
                origin = sorted(observed_origins, key=lambda value: (value != "built-in", value != "user", value))[0]
        add(
            kind,
            component.parent_id,
            component.trace_name,
            origin,
            component.provenance,
            name=component.name,
            legacy_id=component.component_id,
        )

    for span in result.spans:
        add("span", span.component_id, span.name, span.origin, "observed")
    for event in result.dataset.events:
        if event.schema_version <= 0 or event.name in _STRUCTURAL_EVENTS:
            continue
        add(
            "event",
            event.component_id,
            event.name,
            str(event.field("origin", "built-in")),
            "observed",
        )

    endpoint_map = {
        legacy_id: sorted(candidates, key=lambda item: (item[0] != "built-in", item[0] != "user", item))[0][1]
        for legacy_id, candidates in legacy_endpoint_ids.items()
    }
    descriptions = {
        component.component_id: definitions.get(identity, "") for identity, component in tracepoints.items()
    }
    return list(tracepoints.values()), endpoint_map, descriptions


def _ordered_components(components: list[Component]) -> tuple[list[Component], dict[str, str], dict[str, list[str]]]:
    by_id = {component.component_id: component for component in components}
    parent_by_id = {
        component.component_id: component.parent_id if component.parent_id in by_id else "" for component in components
    }
    for component_id in by_id:
        seen = {component_id}
        parent_id = parent_by_id[component_id]
        while parent_id:
            if parent_id in seen:
                parent_by_id[component_id] = ""
                break
            seen.add(parent_id)
            parent_id = parent_by_id.get(parent_id, "")
    children: dict[str, list[str]] = defaultdict(list)
    for component in components:
        children[parent_by_id[component.component_id]].append(component.component_id)
    ordered = []
    visited = set()

    def visit(component_id: str) -> None:
        if component_id in visited:
            return
        visited.add(component_id)
        ordered.append(by_id[component_id])
        for child_id in children.get(component_id, []):
            visit(child_id)

    for root_id in children.get("", []):
        visit(root_id)
    for component in components:
        visit(component.component_id)
    return ordered, parent_by_id, children


def _depth(component_id: str, parent_by_id: dict[str, str]) -> int:
    value = 0
    parent_id = parent_by_id.get(component_id, "")
    while parent_id:
        value += 1
        parent_id = parent_by_id.get(parent_id, "")
    return value


def _nearest_visible(component_id: str, visible: set[str], parent_by_id: dict[str, str]) -> str:
    seen = set()
    while component_id and component_id not in seen:
        if component_id in visible:
            return component_id
        seen.add(component_id)
        component_id = parent_by_id.get(component_id, "")
    return ""


def _metric(
    values: list[float], metric: GraphMetric, request_scoped: bool = False
) -> tuple[dict[str, Any], float | None]:
    stats = summarize(values)
    return stats.to_dict(), values[0] if request_scoped and values else getattr(stats, metric)


def _project_data_edges(
    result: AnalysisResult,
    topology: TraceTopology,
    query: GraphQuery,
    endpoint: Any,
) -> list[GraphEdge]:
    grouped: dict[tuple[str, ...], list[DataFlowEdge]] = defaultdict(list)
    for edge in topology.edges:
        source_id = endpoint(edge.source_id)
        target_id = endpoint(edge.target_id)
        if not source_id or not target_id or (query.view == "nodes" and source_id == target_id):
            continue
        key = (source_id, target_id) if query.view == "nodes" else (source_id, target_id, edge.edge_id)
        grouped[key].append(edge)

    projected = []
    for key, original_edges in grouped.items():
        source_id, target_id = key[:2]
        original_ids = {edge.edge_id for edge in original_edges}
        values = [
            flow.duration_ms
            for flow in result.flows
            if flow.edge_id in original_ids
            and flow.duration_ms is not None
            and (not query.request_id or flow.trace_id == query.request_id)
        ]
        metrics, metric_value = _metric(values, query.metric, bool(query.request_id))
        first = original_edges[0]
        remapped = first.source_id != source_id or first.target_id != target_id
        derived = query.view == "nodes" or remapped or len(original_edges) > 1
        if derived:
            identity = "" if query.view == "nodes" else f"{quote(first.edge_id, safe='')}:"
            edge_id = f"derived:{query.view}:{identity}{quote(source_id, safe='')}->{quote(target_id, safe='')}"
            names = list(dict.fromkeys(edge.name for edge in original_edges))
            kinds = {edge.kind for edge in original_edges}
            data_types = list(dict.fromkeys(edge.data_type for edge in original_edges if edge.data_type))
            contract_keys = list(dict.fromkeys(edge.contract_key for edge in original_edges if edge.contract_key))
            name = " / ".join(names)
            kind = next(iter(kinds)) if len(kinds) == 1 else "data_flow"
            provenance = "derived"
            data_type = ", ".join(data_types)
            contract_key = ", ".join(contract_keys)
        else:
            edge_id = first.edge_id
            name = first.name
            kind = first.kind
            provenance = first.provenance
            data_type = first.data_type
            contract_key = first.contract_key
        projected.append(
            GraphEdge(
                edge_id,
                edge_id,
                source_id,
                target_id,
                name,
                kind,
                provenance,
                data_type,
                contract_key,
                {"latency_ms": metrics},
                metric_value,
            )
        )
    return projected


def project_graph(result: AnalysisResult, query: GraphQuery = GraphQuery()) -> GraphProjection:
    observed = result.topology is None
    topology = normalize_topology(result.topology or _observed_topology(result))
    source = "observed" if observed and topology.components else ("none" if observed else "declared")
    base_components = [
        component for component in topology.components if component.kind not in {"operation", "instant_event"}
    ]
    _, base_parent_by_id, _ = _ordered_components(base_components)
    _, topology_parent_by_id, topology_children = _ordered_components(topology.components)
    degree: dict[str, int] = defaultdict(int)
    for edge in topology.edges:
        degree[edge.source_id] += 1
        degree[edge.target_id] += 1
    filtered_spans = [span for span in result.spans if not query.request_id or span.trace_id == query.request_id]
    descriptions: dict[str, str] = {}
    legacy_endpoint_map: dict[str, str] = {}

    if query.view == "nodes":
        visible = {
            component.component_id
            for component in base_components
            if component.kind == "ros_node"
            or (component.kind == "observed_component" and not base_parent_by_id[component.component_id])
            or (component.kind in {"data_source", "data_sink"} and not base_parent_by_id[component.component_id])
        }

        def endpoint(component_id: str) -> str:
            return _nearest_visible(component_id, visible, topology_parent_by_id)

        ordered = [component for component in base_components if component.component_id in visible]
        parent_by_id = {component.component_id: "" for component in ordered}
        children: dict[str, list[str]] = defaultdict(list)
        span_values: dict[str, list[float]] = defaultdict(list)
        for span in filtered_spans:
            node_id = endpoint(span.component_id)
            if node_id and span.duration_ms is not None:
                span_values[node_id].append(span.duration_ms)
    elif query.view == "components":
        visible = {
            component.component_id
            for component in base_components
            if not (topology_children.get(component.component_id) and degree[component.component_id] == 0)
            and (component.kind in {"module", "observed_component"} or degree[component.component_id] > 0)
        }

        def endpoint(component_id: str) -> str:
            return _nearest_visible(component_id, visible, topology_parent_by_id)

        ordered = [component for component in base_components if component.component_id in visible]
        parent_by_id = {component.component_id: base_parent_by_id[component.component_id] for component in ordered}
        children = defaultdict(list)
        for component in ordered:
            children[parent_by_id[component.component_id]].append(component.component_id)
        span_values = defaultdict(list)
        for span in filtered_spans:
            if span.component_id in visible and span.duration_ms is not None:
                span_values[span.component_id].append(span.duration_ms)
    else:
        tracepoint_components, legacy_endpoint_map, descriptions = _tracepoint_components(result, topology)
        ordered, parent_by_id, children = _ordered_components(base_components + tracepoint_components)
        visible = {component.component_id for component in ordered}

        def endpoint(component_id: str) -> str:
            return legacy_endpoint_map.get(component_id, component_id if component_id in visible else "")

        span_values = defaultdict(list)
        for span in filtered_spans:
            if span.duration_ms is None:
                continue
            if span.component_id in visible:
                span_values[span.component_id].append(span.duration_ms)
            tracepoint_id = _tracepoint_node_id("span", span.component_id, span.name, span.origin)
            if tracepoint_id in visible:
                span_values[tracepoint_id].append(span.duration_ms)

    included = {component.component_id for component in ordered}
    nodes = []
    for component in ordered:
        metrics, metric_value = _metric(
            span_values[component.component_id],
            query.metric,
            bool(query.request_id),
        )
        parent_id = parent_by_id.get(component.component_id, "")
        nodes.append(
            GraphNode(
                component.component_id,
                component.component_id,
                component.name,
                component.kind,
                parent_id,
                [child_id for child_id in children.get(component.component_id, []) if child_id in included],
                _depth(component.component_id, parent_by_id),
                component.provenance,
                component.trace_name,
                {"processing_ms": metrics},
                metric_value,
                descriptions.get(component.component_id, ""),
            )
        )

    edges = _project_data_edges(result, topology, query, endpoint)
    if query.view == "tracepoints":
        for component in ordered:
            parent_id = parent_by_id.get(component.component_id, "")
            if not parent_id:
                continue
            edge_id = f"contains:{quote(parent_id, safe='')}->{quote(component.component_id, safe='')}"
            edges.append(
                GraphEdge(
                    edge_id,
                    edge_id,
                    parent_id,
                    component.component_id,
                    "Contains",
                    "contains",
                    "derived",
                    "",
                    "",
                    {"latency_ms": summarize([]).to_dict()},
                    None,
                    False,
                )
            )
    return GraphProjection(query.view, query.metric, query.request_id, source, nodes, edges)
