"""Allowlisted, paginated queries over an analysis result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .model import Component, FlowRecord, SpanRecord, TraceEvent, TracepointDefinition

if TYPE_CHECKING:
    from .analysis import AnalysisResult

T = TypeVar("T")
MAX_PAGE_SIZE = 1_000
REQUEST_SORT_FIELDS = frozenset(
    {
        "action_chunk_publish_ms",
        "cloud_roundtrip_ms",
        "dispatch_decode_ms",
        "dispatch_to_infer_ms",
        "execute_publish_ms",
        "inference_ms",
        "obs_frame_ms",
        "policy_total_reported_ms",
        "postprocess_ms",
        "preprocess_ms",
        "queue_refill_ms",
        "refill_to_execute_ms",
        "request_id",
        "total_ms",
    }
)


def _validate_page(offset: int, limit: int) -> None:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")


@dataclass(frozen=True, slots=True)
class QueryPage(Generic[T]):
    items: list[T]
    total: int
    offset: int
    limit: int
    next_offset: int | None


@dataclass(frozen=True, slots=True)
class EventQuery:
    request_id: str = ""
    component_id: str = ""
    event_name: str = ""
    provider: str = ""
    origin_kind: str = ""
    start_ns: int | None = None
    end_ns: int | None = None
    offset: int = 0
    limit: int = 100

    def __post_init__(self) -> None:
        _validate_page(self.offset, self.limit)


@dataclass(frozen=True, slots=True)
class SpanQuery:
    request_id: str = ""
    component_id: str = ""
    name: str = ""
    status: str = ""
    origin: str = ""
    start_ns: int | None = None
    end_ns: int | None = None
    offset: int = 0
    limit: int = 100

    def __post_init__(self) -> None:
        _validate_page(self.offset, self.limit)


@dataclass(frozen=True, slots=True)
class FlowQuery:
    request_id: str = ""
    component_id: str = ""
    edge_id: str = ""
    status: str = ""
    start_ns: int | None = None
    end_ns: int | None = None
    offset: int = 0
    limit: int = 100

    def __post_init__(self) -> None:
        _validate_page(self.offset, self.limit)


@dataclass(frozen=True, slots=True)
class RequestQuery:
    request_id: str = ""
    sort_by: str = "total_ms"
    descending: bool = True
    offset: int = 0
    limit: int = 100

    def __post_init__(self) -> None:
        _validate_page(self.offset, self.limit)
        if self.sort_by not in REQUEST_SORT_FIELDS:
            raise ValueError(f"unsupported request sort field: {self.sort_by}")


@dataclass(frozen=True, slots=True)
class ComponentQuery:
    component_id: str = ""
    parent_id: str = ""
    kind: str = ""
    provenance: str = ""
    offset: int = 0
    limit: int = 100

    def __post_init__(self) -> None:
        _validate_page(self.offset, self.limit)


@dataclass(frozen=True, slots=True)
class TracepointQuery:
    kind: str = ""
    component_id: str = ""
    name: str = ""
    origin: str = ""
    offset: int = 0
    limit: int = 100

    def __post_init__(self) -> None:
        _validate_page(self.offset, self.limit)
        if self.kind and self.kind not in {"event", "span"}:
            raise ValueError(f"unsupported tracepoint kind: {self.kind}")


def _page(items: list[T], offset: int, limit: int) -> QueryPage[T]:
    total = len(items)
    page_items = items[offset : offset + limit]
    next_offset = offset + len(page_items)
    return QueryPage(page_items, total, offset, limit, next_offset if next_offset < total else None)


def _digest(kind: str, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)
    return f"{kind}:{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def stable_event_id(event: TraceEvent) -> str:
    return _digest("event", asdict(event))


def stable_span_id(span: SpanRecord) -> str:
    return _digest("span", (span.trace_id, span.span_id))


def stable_flow_id(flow: FlowRecord) -> str:
    return _digest("flow", (flow.trace_id, flow.edge_id, flow.flow_id))


def stable_request_id(request: str | dict[str, Any]) -> str:
    request_id = request if isinstance(request, str) else str(request.get("request_id", ""))
    return _digest("request", request_id)


def stable_component_id(component: Component | str) -> str:
    return component if isinstance(component, str) else component.component_id


def stable_tracepoint_id(definition: TracepointDefinition) -> str:
    return _digest("tracepoint", definition.identity)


class QueryService:
    """Execute explicit queries without exposing arbitrary field expressions."""

    def __init__(self, result: AnalysisResult):
        self.result = result

    def events(self, query: EventQuery = EventQuery()) -> QueryPage[TraceEvent]:
        items = [
            event
            for event in self.result.dataset.events
            if (not query.request_id or event.request_id == query.request_id)
            and (not query.component_id or event.component_id == query.component_id)
            and (not query.event_name or event.name == query.event_name)
            and (not query.provider or event.origin.provider == query.provider)
            and (not query.origin_kind or str(event.fields.get("origin", "built-in")) == query.origin_kind)
            and (query.start_ns is None or event.timestamp_ns >= query.start_ns)
            and (query.end_ns is None or event.timestamp_ns <= query.end_ns)
        ]
        items.sort(key=lambda event: (event.timestamp_ns, event.sequence, stable_event_id(event)))
        return _page(items, query.offset, query.limit)

    def spans(self, query: SpanQuery = SpanQuery()) -> QueryPage[SpanRecord]:
        items = [
            span
            for span in self.result.spans
            if (not query.request_id or span.trace_id == query.request_id)
            and (not query.component_id or span.component_id == query.component_id)
            and (not query.name or span.name == query.name)
            and (not query.status or span.status == query.status)
            and (not query.origin or span.origin == query.origin)
            and (query.start_ns is None or span.end_ns is None or span.end_ns >= query.start_ns)
            and (query.end_ns is None or span.start_ns <= query.end_ns)
        ]
        items.sort(key=lambda span: (span.start_ns, stable_span_id(span)))
        return _page(items, query.offset, query.limit)

    def flows(self, query: FlowQuery = FlowQuery()) -> QueryPage[FlowRecord]:
        component_edges = {
            edge.edge_id
            for edge in (self.result.topology.edges if self.result.topology is not None else [])
            if edge.source_id == query.component_id or edge.target_id == query.component_id
        }
        component_edges.update(
            str(event.field("edge_id", ""))
            for event in self.result.dataset.events
            if event.name in {"flow_send", "flow_receive"}
            and event.component_id == query.component_id
            and event.field("edge_id")
        )

        def overlaps(flow: FlowRecord) -> bool:
            points = [point for point in (flow.send_ns, flow.receive_ns) if point is not None]
            if not points:
                return query.start_ns is None and query.end_ns is None
            return (query.start_ns is None or max(points) >= query.start_ns) and (
                query.end_ns is None or min(points) <= query.end_ns
            )

        items = [
            flow
            for flow in self.result.flows
            if (not query.request_id or flow.trace_id == query.request_id)
            and (not query.component_id or flow.edge_id in component_edges)
            and (not query.edge_id or flow.edge_id == query.edge_id)
            and (not query.status or flow.status == query.status)
            and overlaps(flow)
        ]
        items.sort(
            key=lambda flow: (
                min(point for point in (flow.send_ns, flow.receive_ns) if point is not None)
                if flow.send_ns is not None or flow.receive_ns is not None
                else -1,
                stable_flow_id(flow),
            )
        )
        return _page(items, query.offset, query.limit)

    def requests(self, query: RequestQuery = RequestQuery()) -> QueryPage[dict[str, Any]]:
        items = [
            dict(row)
            for row in self.result.request_rows
            if not query.request_id or row.get("request_id") == query.request_id
        ]

        def sort_value(row: dict[str, Any]) -> str | float:
            if query.sort_by == "request_id":
                return str(row.get("request_id", ""))
            try:
                return float(row.get(query.sort_by, -1))
            except (TypeError, ValueError):
                return -1.0

        items.sort(key=lambda row: (sort_value(row), str(row.get("request_id", ""))), reverse=query.descending)
        return _page(items, query.offset, query.limit)

    def components(self, query: ComponentQuery = ComponentQuery()) -> QueryPage[Component]:
        components = self.result.topology.components if self.result.topology is not None else []
        items = [
            component
            for component in components
            if (not query.component_id or component.component_id == query.component_id)
            and (not query.parent_id or component.parent_id == query.parent_id)
            and (not query.kind or component.kind == query.kind)
            and (not query.provenance or component.provenance == query.provenance)
        ]
        return _page(items, query.offset, query.limit)

    def tracepoints(self, query: TracepointQuery = TracepointQuery()) -> QueryPage[TracepointDefinition]:
        items = [
            definition
            for definition in self.result.definitions
            if (not query.kind or definition.kind == query.kind)
            and (not query.component_id or definition.component_id == query.component_id)
            and (not query.name or definition.name == query.name)
            and (not query.origin or definition.origin == query.origin)
        ]
        items.sort(key=lambda definition: (definition.identity, stable_tracepoint_id(definition)))
        return _page(items, query.offset, query.limit)
