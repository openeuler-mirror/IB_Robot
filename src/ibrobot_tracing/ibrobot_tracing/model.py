"""UI-neutral trace, topology, and performance result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EventOrigin:
    source: str = "unknown"
    path: str = ""
    provider: str = ""
    host: str = ""
    process_id: int | None = None
    thread_id: int | None = None
    node: str = ""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    timestamp_ns: int
    name: str
    fields: dict[str, Any] = field(default_factory=dict)
    origin: EventOrigin = field(default_factory=EventOrigin)
    clock: str = "realtime"
    schema_version: int = 0
    sequence: int = 0

    def field(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)

    @property
    def request_id(self) -> str:
        for key in ("trace_id", "request_id", "inference_id"):
            value = self.fields.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    @property
    def component_id(self) -> str:
        return str(self.fields.get("component_id", ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TracepointDefinition:
    kind: str
    component_id: str
    name: str
    origin: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"event", "span"}:
            raise ValueError(f"Unsupported tracepoint kind: {self.kind}")
        if not self.name:
            raise ValueError("Tracepoint name must not be empty")

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.kind, self.component_id, self.name, self.origin)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class TraceDataset:
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    definitions: list[TracepointDefinition] = field(default_factory=list)

    def sort(self) -> None:
        self.events.sort(key=lambda event: (event.timestamp_ns, event.sequence))

    def select(
        self,
        *,
        request_id: str = "",
        component_id: str = "",
        event_name: str = "",
        provider: str = "",
        origin_kind: str = "",
    ) -> list[TraceEvent]:
        return [
            event
            for event in self.events
            if (not request_id or event.request_id == request_id)
            and (not component_id or event.component_id == component_id)
            and (not event_name or event.name == event_name)
            and (not provider or event.origin.provider == provider)
            and (not origin_kind or str(event.fields.get("origin", "built-in")) == origin_kind)
        ]


@dataclass(frozen=True, slots=True)
class NumericSummary:
    count: int
    minimum: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    maximum: float | None = None
    mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Component:
    component_id: str
    name: str
    kind: str
    parent_id: str = ""
    package: str = ""
    executable: str = ""
    node: str = ""
    provenance: str = "declared"
    trace_name: str = ""


@dataclass(frozen=True, slots=True)
class DataFlowEdge:
    edge_id: str
    source_id: str
    target_id: str
    name: str
    kind: str
    data_type: str = ""
    contract_key: str = ""
    provenance: str = "declared"


@dataclass(slots=True)
class TraceTopology:
    robot_name: str = ""
    control_mode: str = ""
    execution_mode: str = "monolithic"
    components: list[Component] = field(default_factory=list)
    edges: list[DataFlowEdge] = field(default_factory=list)
    logger_to_component: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    definitions: list[TracepointDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "robot_name": self.robot_name,
            "control_mode": self.control_mode,
            "execution_mode": self.execution_mode,
            "components": [asdict(component) for component in self.components],
            "edges": [asdict(edge) for edge in self.edges],
            "logger_to_component": self.logger_to_component,
            "metadata": self.metadata,
            "tracepoints": [definition.to_dict() for definition in self.definitions],
        }


@dataclass(frozen=True, slots=True)
class SpanRecord:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str
    component_id: str
    start_ns: int
    end_ns: int | None
    status: str
    origin: str
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        duration_ns = self.fields.get("duration_ns")
        if duration_ns is not None:
            try:
                return float(duration_ns) / 1_000_000
            except (TypeError, ValueError):
                pass
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000


@dataclass(frozen=True, slots=True)
class FlowRecord:
    edge_id: str
    flow_id: str
    trace_id: str
    send_ns: int | None
    receive_ns: int | None
    status: str

    @property
    def duration_ms(self) -> float | None:
        if self.send_ns is None or self.receive_ns is None:
            return None
        return (self.receive_ns - self.send_ns) / 1_000_000
