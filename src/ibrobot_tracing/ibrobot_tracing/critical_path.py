"""Request-scoped instrumented wall-time attribution partitions."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ._span_hierarchy import DIAGNOSTIC_ORDER, build_span_hierarchy
from .model import DataFlowEdge, FlowRecord, TraceEvent
from .query import stable_event_id, stable_flow_id
from .serialization import to_js_safe

if TYPE_CHECKING:
    from .analysis import AnalysisResult

DEFAULT_MAX_CRITICAL_PATH_SEGMENTS = 1_000
MAX_CRITICAL_PATH_SEGMENTS = 10_000
_FLOW_STATUS_ALLOWLIST = frozenset({"complete"})
_FLOW_CLOCK_ALLOWLIST = frozenset({"realtime"})
_DIAGNOSTIC_ID_LIMIT = 20
_DIAGNOSTIC_RANK = {code: index for index, code in enumerate(DIAGNOSTIC_ORDER)}

__all__ = [
    "DEFAULT_MAX_CRITICAL_PATH_SEGMENTS",
    "MAX_CRITICAL_PATH_SEGMENTS",
    "CriticalPathBottleneck",
    "CriticalPathDiagnostic",
    "CriticalPathProjection",
    "CriticalPathQuery",
    "CriticalPathSegment",
    "CriticalPathTotals",
    "project_critical_path",
]


class CriticalPathProjectionBase:
    def to_dict(self) -> dict[str, Any]:
        return to_js_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class CriticalPathQuery:
    request_id: str
    component_id: str = ""
    start_ns: int | None = None
    end_ns: int | None = None
    include_flows: bool = True
    max_segments: int = DEFAULT_MAX_CRITICAL_PATH_SEGMENTS

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("critical path requires request_id")
        if self.start_ns is not None and self.end_ns is not None and self.start_ns > self.end_ns:
            raise ValueError("start_ns must not be greater than end_ns")
        if not 1 <= self.max_segments <= MAX_CRITICAL_PATH_SEGMENTS:
            raise ValueError(f"max_segments must be between 1 and {MAX_CRITICAL_PATH_SEGMENTS}")


@dataclass(frozen=True, slots=True)
class CriticalPathDiagnostic(CriticalPathProjectionBase):
    code: str
    count: int
    source_ids: list[str]
    message: str


@dataclass(frozen=True, slots=True)
class CriticalPathSegment(CriticalPathProjectionBase):
    id: str
    index: int
    kind: Literal["span", "flow", "unattributed"]
    source_id: str
    label: str
    component_id: str
    edge_id: str
    origin: str
    status: str
    start_ns: int
    end_ns: int
    offset_ns: int
    duration_ns: int
    percentage: float
    diagnostics: list[str]
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CriticalPathBottleneck(CriticalPathProjectionBase):
    rank: int
    kind: Literal["span", "flow"]
    source_id: str
    label: str
    component_id: str
    edge_id: str
    origin: str
    status: str
    duration_ns: int
    percentage: float
    segment_count: int


@dataclass(frozen=True, slots=True)
class CriticalPathTotals(CriticalPathProjectionBase):
    partition_ns: int
    attributed_ns: int
    unattributed_ns: int
    coverage_percent: float
    by_kind_ns: dict[str, int]
    by_component_ns: dict[str, int]
    by_edge_ns: dict[str, int]
    record_counts: dict[str, int]
    diagnostic_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CriticalPathProjection(CriticalPathProjectionBase):
    method: Literal["deepest_active_wall_partition"]
    measurement: Literal["instrumented_wall"]
    not_cpu: bool
    request_id: str
    component_id: str
    include_flows: bool
    boundary_source: str
    boundary_start_source: str
    boundary_end_source: str
    start_ns: int | None
    end_ns: int | None
    duration_ns: int
    segments: list[CriticalPathSegment]
    totals: CriticalPathTotals
    bottlenecks: list[CriticalPathBottleneck]
    diagnostics: list[CriticalPathDiagnostic]
    total_segments: int
    returned_segments: int
    returned_duration_ns: int
    omitted_segments: int
    omitted_duration_ns: int
    truncated: bool
    truncation_reason: str


@dataclass(slots=True)
class _DiagnosticAccumulator:
    ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    counts: Counter[str] = field(default_factory=Counter)
    messages: dict[str, str] = field(default_factory=dict)

    def add(self, code: str, source_id: str, message: str) -> None:
        self.counts[code] += 1
        if source_id:
            self.ids[code].add(source_id)
        self.messages.setdefault(code, message)

    def project(self) -> list[CriticalPathDiagnostic]:
        return [
            CriticalPathDiagnostic(
                code,
                self.counts[code],
                sorted(self.ids[code])[:_DIAGNOSTIC_ID_LIMIT],
                self.messages[code],
            )
            for code in sorted(
                self.counts,
                key=lambda code: (_DIAGNOSTIC_RANK.get(code, len(_DIAGNOSTIC_RANK)), code),
            )
        ]


@dataclass(frozen=True, slots=True)
class _Owner:
    kind: Literal["span", "flow"]
    source_id: str
    label: str
    component_id: str
    edge_id: str
    origin: str
    status: str
    start_ns: int
    end_ns: int
    priority: tuple[Any, ...]
    diagnostics: tuple[str, ...]
    fields: dict[str, Any]


def _segment_id(
    request_id: str,
    kind: str,
    source_id: str,
    start_ns: int,
    end_ns: int,
) -> str:
    payload = json.dumps((request_id, kind, source_id, start_ns, end_ns), separators=(",", ":"), ensure_ascii=True)
    return f"critical-segment:{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def _flow_events(events: list[TraceEvent]) -> dict[tuple[str, str], dict[str, TraceEvent]]:
    pairs: dict[tuple[str, str], dict[str, TraceEvent]] = defaultdict(dict)
    for event in sorted(events, key=lambda item: (item.timestamp_ns, item.sequence, stable_event_id(item))):
        if event.name not in {"flow_send", "flow_receive"}:
            continue
        edge_id = str(event.field("edge_id", ""))
        flow_id = str(event.field("flow_id", ""))
        if edge_id and flow_id:
            pairs[(edge_id, flow_id)][event.name] = event
    return pairs


def _flow_exclusion(
    flow: FlowRecord,
    pair: dict[str, TraceEvent],
) -> tuple[str, str] | None:
    if flow.send_ns is None or flow.receive_ns is None:
        return "flow_incomplete", "flow is missing a send or receive endpoint"
    if flow.receive_ns < flow.send_ns:
        return "flow_negative_duration", "flow receive precedes flow send"
    if flow.status not in _FLOW_STATUS_ALLOWLIST:
        return "flow_status_excluded", "flow status is not in the complete-only attribution allowlist"
    send = pair.get("flow_send")
    receive = pair.get("flow_receive")
    if send is None or receive is None:
        return None
    if send.origin.host and receive.origin.host and send.origin.host != receive.origin.host:
        return "flow_cross_host_unsynchronized", "cross-host flow has no shared-clock guarantee"
    if send.clock != receive.clock:
        return "flow_clock_mismatch", "flow endpoints use different clocks"
    if send.clock not in _FLOW_CLOCK_ALLOWLIST:
        return "flow_clock_unknown", "flow clock is not in the wall-clock attribution allowlist"
    return None


def _incident_edges(result: AnalysisResult, events: list[TraceEvent], component_id: str) -> set[str]:
    if not component_id:
        return set()
    incident = {
        edge.edge_id
        for edge in (result.topology.edges if result.topology is not None else [])
        if component_id in {edge.source_id, edge.target_id}
    }
    incident.update(
        str(event.field("edge_id", ""))
        for event in events
        if event.name in {"flow_send", "flow_receive"} and event.component_id == component_id and event.field("edge_id")
    )
    return incident


def _edge_fields(edge: DataFlowEdge | None, pair: dict[str, TraceEvent], flow_id: str) -> dict[str, Any]:
    send = pair.get("flow_send")
    receive = pair.get("flow_receive")
    fields: dict[str, Any] = {
        "flow_id": flow_id,
        "source_component_id": edge.source_id if edge is not None else (send.component_id if send else ""),
        "target_component_id": edge.target_id if edge is not None else (receive.component_id if receive else ""),
    }
    if edge is not None:
        fields.update(
            {
                "edge_name": edge.name,
                "edge_kind": edge.kind,
                "data_type": edge.data_type,
                "contract_key": edge.contract_key,
                "provenance": edge.provenance,
            }
        )
    return fields


def _boundary(
    events: list[TraceEvent],
    fallback_points: list[int],
    query: CriticalPathQuery,
    diagnostics: _DiagnosticAccumulator,
) -> tuple[str, str, str, int | None, int | None]:
    dispatches = sorted(
        (event for event in events if event.name == "dispatch_request"),
        key=lambda event: (event.timestamp_ns, event.sequence, stable_event_id(event)),
    )
    dispatch = dispatches[0] if dispatches else None
    executions = sorted(
        (
            event
            for event in events
            if event.name == "first_action_execute"
            and (dispatch is None or event.timestamp_ns >= dispatch.timestamp_ns)
        ),
        key=lambda event: (event.timestamp_ns, event.sequence, stable_event_id(event)),
    )
    execution = executions[0] if executions else None
    fallback_start = min(fallback_points) if fallback_points else None
    fallback_end = max(fallback_points) if fallback_points else None
    if dispatch is not None:
        start_ns = dispatch.timestamp_ns
        start_source = "event:dispatch_request"
    elif fallback_start is not None:
        start_ns = fallback_start
        start_source = "fallback:selected_records"
    elif execution is not None:
        start_ns = execution.timestamp_ns
        start_source = "fallback:preferred_anchor"
    else:
        start_ns = None
        start_source = "fallback:selected_records"
    if execution is not None:
        end_ns = execution.timestamp_ns
        end_source = "event:first_action_execute"
    elif fallback_end is not None:
        end_ns = fallback_end
        end_source = "fallback:selected_records"
    elif dispatch is not None:
        end_ns = dispatch.timestamp_ns
        end_source = "fallback:preferred_anchor"
    else:
        end_ns = None
        end_source = "fallback:selected_records"
    if start_ns is None or end_ns is None:
        diagnostics.add("boundary_unavailable", "", "no preferred boundary or selected record bounds are available")
        return "unavailable", start_source, end_source, None, None
    if dispatch is None:
        diagnostics.add(
            "boundary_start_fallback", "", "dispatch_request was unavailable; selected record bounds were used"
        )
    if execution is None:
        diagnostics.add(
            "boundary_end_fallback",
            "",
            "first_action_execute was unavailable; selected record bounds were used",
        )
    if end_ns < start_ns:
        if dispatch is not None and execution is None:
            end_ns = start_ns
            end_source = "fallback:preferred_anchor"
            diagnostics.add(
                "boundary_end_before_start",
                "",
                "selected record end preceded dispatch_request; the fallback boundary was collapsed to zero length",
            )
        else:
            start_ns = end_ns
            start_source = "fallback:preferred_anchor"
            diagnostics.add(
                "boundary_start_after_end",
                "",
                "selected record start followed first_action_execute; the fallback boundary was collapsed to zero length",
            )
    if dispatch is not None and execution is not None:
        source = "dispatch_request_to_first_action_execute"
    elif dispatch is not None or execution is not None:
        source = "partial_boundary_fallback"
    else:
        source = "selected_record_bounds_fallback"

    natural_start, natural_end = start_ns, end_ns
    requested_start = query.start_ns if query.start_ns is not None else natural_start
    requested_end = query.end_ns if query.end_ns is not None else natural_end
    start_ns = min(max(requested_start, natural_start), natural_end)
    end_ns = min(max(requested_end, natural_start), natural_end)
    if query.start_ns is not None and start_ns != query.start_ns:
        diagnostics.add("window_start_clamped", "", "requested start_ns was clamped to the request boundary")
    if query.end_ns is not None and end_ns != query.end_ns:
        diagnostics.add("window_end_clamped", "", "requested end_ns was clamped to the request boundary")
    return source, start_source, end_source, start_ns, end_ns


def _empty_totals(record_counts: dict[str, int], diagnostics: _DiagnosticAccumulator) -> CriticalPathTotals:
    return CriticalPathTotals(
        partition_ns=0,
        attributed_ns=0,
        unattributed_ns=0,
        coverage_percent=0.0,
        by_kind_ns={"flow": 0, "span": 0, "unattributed": 0},
        by_component_ns={},
        by_edge_ns={},
        record_counts=dict(sorted(record_counts.items())),
        diagnostic_counts=dict(sorted(diagnostics.counts.items())),
    )


def _sweep(owners: list[_Owner], start_ns: int, end_ns: int) -> list[tuple[_Owner | None, int, int]]:
    starts: dict[int, list[_Owner]] = defaultdict(list)
    ends: dict[int, list[_Owner]] = defaultdict(list)
    boundaries = {start_ns, end_ns}
    for owner in owners:
        clipped_start = max(start_ns, owner.start_ns)
        clipped_end = min(end_ns, owner.end_ns)
        if clipped_end <= clipped_start:
            continue
        starts[clipped_start].append(owner)
        ends[clipped_end].append(owner)
        boundaries.update((clipped_start, clipped_end))

    span_heap: list[tuple[tuple[Any, ...], tuple[str, str], _Owner]] = []
    flow_heap: list[tuple[tuple[Any, ...], tuple[str, str], _Owner]] = []
    active: set[tuple[str, str]] = set()
    intervals = []
    ordered_boundaries = sorted(boundaries)
    for index, point in enumerate(ordered_boundaries[:-1]):
        for owner in ends.get(point, []):
            active.discard((owner.kind, owner.source_id))
        for owner in starts.get(point, []):
            identity = (owner.kind, owner.source_id)
            active.add(identity)
            heap = flow_heap if owner.kind == "flow" else span_heap
            heapq.heappush(heap, (owner.priority, identity, owner))
        for heap in (flow_heap, span_heap):
            while heap and heap[0][1] not in active:
                heapq.heappop(heap)
        interval_end = ordered_boundaries[index + 1]
        if interval_end <= point:
            continue
        owner = flow_heap[0][2] if flow_heap else (span_heap[0][2] if span_heap else None)
        if intervals and intervals[-1][0] == owner and intervals[-1][2] == point:
            intervals[-1] = (owner, intervals[-1][1], interval_end)
        else:
            intervals.append((owner, point, interval_end))
    return intervals


def _segments(
    intervals: list[tuple[_Owner | None, int, int]],
    request_id: str,
    start_ns: int,
    duration_ns: int,
) -> list[CriticalPathSegment]:
    segments = []
    for index, (owner, interval_start, interval_end) in enumerate(intervals):
        segment_duration = interval_end - interval_start
        kind: Literal["span", "flow", "unattributed"] = owner.kind if owner is not None else "unattributed"
        source_id = owner.source_id if owner is not None else ""
        segments.append(
            CriticalPathSegment(
                id=_segment_id(request_id, kind, source_id, interval_start, interval_end),
                index=index,
                kind=kind,
                source_id=source_id,
                label=owner.label if owner is not None else "Unattributed",
                component_id=owner.component_id if owner is not None else "",
                edge_id=owner.edge_id if owner is not None else "",
                origin=owner.origin if owner is not None else "",
                status=owner.status if owner is not None else "unattributed",
                start_ns=interval_start,
                end_ns=interval_end,
                offset_ns=interval_start - start_ns,
                duration_ns=segment_duration,
                percentage=(segment_duration / duration_ns * 100) if duration_ns else 0.0,
                diagnostics=list(owner.diagnostics) if owner is not None else ["no_active_instrumented_record"],
                fields=dict(owner.fields) if owner is not None else {},
            )
        )
    return segments


def _totals(
    segments: list[CriticalPathSegment],
    duration_ns: int,
    record_counts: dict[str, int],
    diagnostics: _DiagnosticAccumulator,
) -> CriticalPathTotals:
    by_kind = Counter({"flow": 0, "span": 0, "unattributed": 0})
    by_component: Counter[str] = Counter()
    by_edge: Counter[str] = Counter()
    for segment in segments:
        by_kind[segment.kind] += segment.duration_ns
        if segment.kind == "span" and segment.component_id:
            by_component[segment.component_id] += segment.duration_ns
        if segment.kind == "flow" and segment.edge_id:
            by_edge[segment.edge_id] += segment.duration_ns
    unattributed_ns = by_kind["unattributed"]
    attributed_ns = duration_ns - unattributed_ns
    return CriticalPathTotals(
        partition_ns=sum(segment.duration_ns for segment in segments),
        attributed_ns=attributed_ns,
        unattributed_ns=unattributed_ns,
        coverage_percent=(attributed_ns / duration_ns * 100) if duration_ns else 0.0,
        by_kind_ns=dict(sorted(by_kind.items())),
        by_component_ns=dict(sorted(by_component.items())),
        by_edge_ns=dict(sorted(by_edge.items())),
        record_counts=dict(sorted(record_counts.items())),
        diagnostic_counts=dict(sorted(diagnostics.counts.items())),
    )


def _bottlenecks(segments: list[CriticalPathSegment], duration_ns: int) -> list[CriticalPathBottleneck]:
    grouped: dict[tuple[str, str], list[CriticalPathSegment]] = defaultdict(list)
    for segment in segments:
        if segment.kind != "unattributed":
            grouped[(segment.kind, segment.source_id)].append(segment)
    ranked = sorted(
        grouped.values(),
        key=lambda items: (-sum(item.duration_ns for item in items), items[0].kind, items[0].source_id),
    )
    return [
        CriticalPathBottleneck(
            rank=rank,
            kind=items[0].kind,  # type: ignore[arg-type]
            source_id=items[0].source_id,
            label=items[0].label,
            component_id=items[0].component_id,
            edge_id=items[0].edge_id,
            origin=items[0].origin,
            status=items[0].status,
            duration_ns=sum(item.duration_ns for item in items),
            percentage=(sum(item.duration_ns for item in items) / duration_ns * 100) if duration_ns else 0.0,
            segment_count=len(items),
        )
        for rank, items in enumerate(ranked, start=1)
    ]


def project_critical_path(result: AnalysisResult, query: CriticalPathQuery) -> CriticalPathProjection:
    """Partition one request's wall interval; this is neither CPU time nor a scheduling DAG."""
    diagnostics = _DiagnosticAccumulator()
    request_events = [event for event in result.dataset.events if event.request_id == query.request_id]
    request_spans = [span for span in result.spans if span.trace_id == query.request_id]
    request_flows = [flow for flow in result.flows if flow.trace_id == query.request_id]
    hierarchy = build_span_hierarchy(request_spans)
    selected_nodes = [
        node for node in hierarchy.nodes if not query.component_id or node.span.component_id == query.component_id
    ]
    valid_nodes = [
        node
        for node in selected_nodes
        if node.valid_for_weight and node.end_ns is not None and node.end_ns > node.span.start_ns
    ]
    diagnostic_messages = {
        "orphan": "span parent was not observed",
        "cycle": "span parent relationship is cyclic",
        "duplicate_span_id": "span_id occurs more than once in the request",
        "incomplete": "span has no complete end interval and was excluded",
        "negative_duration": "span duration is negative and was excluded",
        "child_outside_parent": "child span interval is outside its parent",
        "overlapping_siblings": "sibling spans overlap",
    }
    for node in selected_nodes:
        for code in sorted(
            node.diagnostics,
            key=lambda item: (_DIAGNOSTIC_RANK.get(item, len(_DIAGNOSTIC_RANK)), item),
        ):
            diagnostics.add(code, node.occurrence_id, diagnostic_messages.get(code, code.replace("_", " ")))

    edge_by_id = {edge.edge_id: edge for edge in (result.topology.edges if result.topology is not None else [])}
    event_pairs = _flow_events(request_events)
    incident_edges = _incident_edges(result, request_events, query.component_id)
    selected_flows = [flow for flow in request_flows if not query.component_id or flow.edge_id in incident_edges]
    valid_flows: list[tuple[FlowRecord, dict[str, TraceEvent]]] = []
    if query.include_flows:
        for flow in selected_flows:
            pair = event_pairs.get((flow.edge_id, flow.flow_id), {})
            exclusion = _flow_exclusion(flow, pair)
            if exclusion is not None:
                diagnostics.add(exclusion[0], stable_flow_id(flow), exclusion[1])
                continue
            valid_flows.append((flow, pair))

    selected_events = [
        event for event in request_events if not query.component_id or event.component_id == query.component_id
    ]
    fallback_points = [event.timestamp_ns for event in selected_events]
    for node in valid_nodes:
        fallback_points.extend((node.span.start_ns, node.end_ns))
    for flow, _pair in valid_flows:
        assert flow.send_ns is not None and flow.receive_ns is not None
        fallback_points.extend((flow.send_ns, flow.receive_ns))
    boundary_source, start_source, end_source, start_ns, end_ns = _boundary(
        request_events,
        fallback_points,
        query,
        diagnostics,
    )
    record_counts = {
        "events": len(request_events),
        "selected_events": len(selected_events),
        "spans": len(request_spans),
        "selected_spans": len(selected_nodes),
        "valid_spans": len(valid_nodes),
        "excluded_spans": len(selected_nodes) - len(valid_nodes),
        "flows": len(request_flows),
        "selected_flows": len(selected_flows) if query.include_flows else 0,
        "valid_flows": len(valid_flows),
        "excluded_flows": (len(selected_flows) - len(valid_flows)) if query.include_flows else 0,
    }
    if start_ns is None or end_ns is None:
        return CriticalPathProjection(
            method="deepest_active_wall_partition",
            measurement="instrumented_wall",
            not_cpu=True,
            request_id=query.request_id,
            component_id=query.component_id,
            include_flows=query.include_flows,
            boundary_source=boundary_source,
            boundary_start_source=start_source,
            boundary_end_source=end_source,
            start_ns=None,
            end_ns=None,
            duration_ns=0,
            segments=[],
            totals=_empty_totals(record_counts, diagnostics),
            bottlenecks=[],
            diagnostics=diagnostics.project(),
            total_segments=0,
            returned_segments=0,
            returned_duration_ns=0,
            omitted_segments=0,
            omitted_duration_ns=0,
            truncated=False,
            truncation_reason="",
        )

    owners = [
        _Owner(
            kind="span",
            source_id=node.occurrence_id,
            label=node.span.name,
            component_id=node.span.component_id,
            edge_id="",
            origin=node.span.origin,
            status=node.span.status,
            start_ns=node.span.start_ns,
            end_ns=node.end_ns,
            priority=(-node.depth, node.duration_ns or 0, node.span.start_ns, node.occurrence_id),
            diagnostics=tuple(
                sorted(
                    node.diagnostics,
                    key=lambda item: (_DIAGNOSTIC_RANK.get(item, len(_DIAGNOSTIC_RANK)), item),
                )
            ),
            fields=dict(node.span.fields),
        )
        for node in valid_nodes
    ]
    flow_occurrences: Counter[str] = Counter()
    for flow, pair in sorted(
        valid_flows,
        key=lambda item: (item[0].send_ns, item[0].receive_ns, stable_flow_id(item[0])),
    ):
        assert flow.send_ns is not None and flow.receive_ns is not None
        base_id = stable_flow_id(flow)
        flow_occurrences[base_id] += 1
        source_id = base_id if flow_occurrences[base_id] == 1 else f"{base_id}:{flow_occurrences[base_id]}"
        edge = edge_by_id.get(flow.edge_id)
        owners.append(
            _Owner(
                kind="flow",
                source_id=source_id,
                label=edge.name if edge is not None else flow.edge_id,
                component_id="",
                edge_id=flow.edge_id,
                origin="",
                status=flow.status,
                start_ns=flow.send_ns,
                end_ns=flow.receive_ns,
                priority=(flow.receive_ns - flow.send_ns, flow.send_ns, source_id),
                diagnostics=(),
                fields=_edge_fields(edge, pair, flow.flow_id),
            )
        )

    duration_ns = end_ns - start_ns
    full_segments = _segments(_sweep(owners, start_ns, end_ns), query.request_id, start_ns, duration_ns)
    totals = _totals(full_segments, duration_ns, record_counts, diagnostics)
    bottlenecks = _bottlenecks(full_segments, duration_ns)
    returned = full_segments[: query.max_segments]
    truncated = len(returned) < len(full_segments)
    omitted_duration_ns = sum(segment.duration_ns for segment in full_segments[len(returned) :])
    if truncated:
        diagnostics.add(
            "segments_truncated",
            "",
            "segments were deterministically truncated to a chronological prefix; totals describe the full partition",
        )
        totals = _totals(full_segments, duration_ns, record_counts, diagnostics)
    return CriticalPathProjection(
        method="deepest_active_wall_partition",
        measurement="instrumented_wall",
        not_cpu=True,
        request_id=query.request_id,
        component_id=query.component_id,
        include_flows=query.include_flows,
        boundary_source=boundary_source,
        boundary_start_source=start_source,
        boundary_end_source=end_source,
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ns=duration_ns,
        segments=returned,
        totals=totals,
        bottlenecks=bottlenecks,
        diagnostics=diagnostics.project(),
        total_segments=len(full_segments),
        returned_segments=len(returned),
        returned_duration_ns=sum(segment.duration_ns for segment in returned),
        omitted_segments=len(full_segments) - len(returned),
        omitted_duration_ns=omitted_duration_ns,
        truncated=truncated,
        truncation_reason="max_segments_chronological_prefix" if truncated else "",
    )
