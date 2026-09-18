"""Reusable offline performance analysis service."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .definitions import builtin_tracepoint_definitions, resolve_tracepoint_definitions
from .model import (
    FlowRecord,
    NumericSummary,
    SpanRecord,
    TraceDataset,
    TraceEvent,
    TracepointDefinition,
    TraceTopology,
)
from .parsing import parse_trace
from .statistics import summarize
from .topology import TopologyMetadataError, enrich_topology_with_observed_operations, read_topology_manifest

_REQUEST_METRICS = {
    "obs_frame_ms",
    "dispatch_to_infer_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "queue_refill_ms",
    "refill_to_execute_ms",
    "total_ms",
}
_CLOUD_STAGE = "cloud_roundtrip_ms"
_SPAN_METRICS = {
    "preprocess_ms": "preprocess",
    "inference_ms": "model_call",
    "postprocess_ms": "postprocess",
    "action_chunk_publish_ms": "action_chunk_publish",
    "dispatch_decode_ms": "dispatch_decode",
    "execute_publish_ms": "first_action_execute",
    "cloud_roundtrip_ms": "cloud_roundtrip",
}
_SUMMARY_COMPONENTS = {
    ("span", "preprocess"): ("policy.preprocess",),
    ("span", "model_call"): ("policy.inference", "cloud_inference"),
    ("span", "postprocess"): ("policy.postprocess",),
    ("span", "action_chunk_publish"): ("policy.postprocess",),
    ("span", "dispatch_decode"): ("action_dispatcher.decode",),
    ("span", "first_action_execute"): ("action_dispatcher.execute",),
    ("span", "cloud_roundtrip"): ("policy",),
    ("span", "queue_refill"): ("action_dispatcher.queue",),
    ("event", "dispatch_request"): ("action_dispatcher.request",),
    ("event", "obs_frame"): ("policy.observation",),
    ("event", "queue_refill"): ("action_dispatcher.queue",),
    ("event", "first_action_execute"): ("action_dispatcher.execute",),
}
_SUMMARY_IDENTITIES = {
    definition.identity
    for definition in builtin_tracepoint_definitions()
    if definition.origin == "built-in"
    and definition.component_id in _SUMMARY_COMPONENTS.get((definition.kind, definition.name), ())
}


class _Warnings(list):
    """Bound detailed diagnostics independently of the input event limit."""

    dropped = 0

    def append(self, message):
        if len(self) < 1000:
            super().append(message)
        else:
            self.dropped += 1

    def extend(self, messages):
        for message in messages:
            self.append(message)

    def finish(self):
        return list(self) + ([f"{self.dropped} additional analysis diagnostics suppressed"] if self.dropped else [])


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    source: Path
    source_kind: str = "auto"
    topology_path: Path | None = None
    max_events: int | None = None


@dataclass(slots=True)
class AnalysisResult:
    dataset: TraceDataset
    request_rows: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    stage_summary: dict[str, NumericSummary] = field(default_factory=dict)
    custom_span_summary: list[dict[str, Any]] = field(default_factory=list)
    custom_mark_summary: list[dict[str, Any]] = field(default_factory=list)
    spans: list[SpanRecord] = field(default_factory=list)
    flows: list[FlowRecord] = field(default_factory=list)
    topology: TraceTopology | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    definitions: list[TracepointDefinition] = field(default_factory=list)
    span_summary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, compatibility: bool = False) -> dict[str, Any]:
        requests = []
        for row in self.request_rows:
            item = dict(row)
            if compatibility:
                item = {key: value for key, value in item.items() if not key.endswith(("_source", "_status"))}
            if "cloud_roundtrip_ms" in item:
                item["network_ms"] = item["cloud_roundtrip_ms"]
                if compatibility:
                    item.pop("cloud_roundtrip_ms", None)
            requests.append(item)
        if compatibility:
            return {"requests": requests, "observations": self.observations}
        return {
            "schema_version": 1,
            "metadata": self.dataset.metadata,
            "requests": requests,
            "observations": self.observations,
            "stages": {key: value.to_dict() for key, value in self.stage_summary.items()},
            "span_summary": self.span_summary,
            "custom_span_summary": self.custom_span_summary,
            "custom_mark_summary": self.custom_mark_summary,
            "spans": [asdict(span) | {"duration_ms": span.duration_ms} for span in self.spans],
            "flows": [asdict(flow) | {"duration_ms": flow.duration_ms} for flow in self.flows],
            "tracepoints": [definition.to_dict() for definition in self.definitions],
            "topology": self.topology.to_dict() if self.topology else None,
            "coverage": self.coverage,
            "warnings": self.warnings,
        }

    def requests(self, *, sort_by: str = "total_ms", limit: int | None = None) -> list[dict[str, Any]]:
        rows = sorted(self.request_rows, key=lambda row: float(row.get(sort_by, -1)), reverse=True)
        return rows[:limit] if limit is not None else rows

    def timeline(self, *, request_id: str = "", component_id: str = "") -> list[TraceEvent]:
        return self.dataset.select(request_id=request_id, component_id=component_id)

    def component_spans(self, component_id: str, *, request_id: str = "") -> list[SpanRecord]:
        return [
            span
            for span in self.spans
            if span.component_id == component_id and (not request_id or span.trace_id == request_id)
        ]


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _bind_components(dataset: TraceDataset, topology: TraceTopology | None) -> None:
    if topology is None:
        return
    bound = []
    for event in dataset.events:
        if event.component_id:
            bound.append(event)
            continue
        component_id = topology.logger_to_component.get(event.origin.provider, "")
        if not component_id:
            bound.append(event)
            continue
        fields = dict(event.fields)
        fields["component_id"] = component_id
        bound.append(replace(event, fields=fields))
    dataset.events = bound
    definitions = []
    for definition in dataset.definitions:
        if definition.component_id:
            definitions.append(definition)
            continue
        if definition.kind == "event":
            candidates = {
                event.component_id
                for event in dataset.events
                if event.name == definition.name
                and str(event.field("origin", "built-in")) == definition.origin
                and event.component_id
            }
        else:
            candidates = {
                event.component_id
                for event in dataset.events
                if event.name in {"span_begin", "span_end"}
                and str(event.field("span_name", "")) == definition.name
                and str(event.field("origin", "built-in")) == definition.origin
                and event.component_id
            }
        definitions.append(replace(definition, component_id=candidates.pop()) if len(candidates) == 1 else definition)
    dataset.definitions = definitions


def _requests(dataset: TraceDataset, spans: list[SpanRecord], warnings: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TraceEvent]] = {}
    for event in dataset.events:
        if event.request_id:
            grouped.setdefault(event.request_id, []).append(event)
    rows = []
    spans_by_request: dict[str, list[SpanRecord]] = {}
    for span in spans:
        spans_by_request.setdefault(span.trace_id, []).append(span)
    for request_id, events in grouped.items():
        row: dict[str, Any] = {"request_id": request_id}
        spans_by_name: dict[str, list[SpanRecord]] = {}
        for span in spans_by_request.get(request_id, []):
            if TracepointDefinition("span", span.component_id, span.name, span.origin).identity in _SUMMARY_IDENTITIES:
                spans_by_name.setdefault(span.name, []).append(span)
        structured_events: dict[str, list[TraceEvent]] = {}
        for event in events:
            if event.schema_version > 0 and (
                TracepointDefinition(
                    "event", event.component_id, event.name, str(event.field("origin", "built-in"))
                ).identity
                in _SUMMARY_IDENTITIES
            ):
                structured_events.setdefault(event.name, []).append(event)

        def unique(metric: str, groups, source: str, *, row=row):
            if f"{metric}_source" in row:
                return None
            statuses = set()
            for group in groups:
                for record in group:
                    if isinstance(record, SpanRecord):
                        status = record.status if record.end_ns is not None else "incomplete"
                        duration = record.duration_ms
                        if status == "ok" and (duration is None or not math.isfinite(duration) or duration < 0):
                            status = "invalid"
                    else:
                        status = str(record.field("status", "ok"))
                        if record.field("success") is False:
                            status = "error"
                        if (
                            status == "ok"
                            and record.name == "first_action_execute"
                            and metric in {"refill_to_execute_ms", "total_ms"}
                        ):
                            publish_end = record.field("publish_end_ns")
                            if publish_end is None:
                                status = "incomplete"
                            elif type(publish_end) is not int or publish_end < 0:
                                status = "invalid"
                    if status != "ok":
                        statuses.add(status or "unknown")
            # Count before filtering status: an error followed by success is still two occurrences.
            if any(len(group) > 1 for group in groups):
                statuses.add("ambiguous")
            status = ",".join(sorted(statuses))
            if status:
                row[f"{metric}_status"] = status
                row[f"{metric}_source"] = source
                warnings.append(f"request {row['request_id']} metric {metric}: {status}; excluded from request summary")
                return None
            return [group[0] for group in groups] if all(groups) else None

        for metric, span_name in _SPAN_METRICS.items():
            candidates = spans_by_name.get(span_name, [])
            if span_name == "first_action_execute" and not candidates:
                candidates = structured_events.get(span_name, [])
            selected = unique(metric, [candidates], "structured")
            if selected:
                record = selected[0]
                raw_value = record.duration_ms if isinstance(record, SpanRecord) else record.field("publish_ms")
                value = _number(raw_value)
                if value is not None and value >= 0:
                    row[metric] = round(value, 2)
                else:
                    status = "incomplete" if raw_value is None else "invalid"
                    row[f"{metric}_status"] = status
                    warnings.append(
                        f"request {request_id} metric {metric}: {status} publish_ms; excluded from request summary"
                    )
                row[f"{metric}_source"] = "structured"

        dispatch = structured_events.get("dispatch_request", [])
        frame = structured_events.get("obs_frame", [])
        model = spans_by_name.get("model_call", [])
        # Existing spans remain authoritative, including failed or repeated occurrences.
        refill = spans_by_name.get("queue_refill", []) or structured_events.get("queue_refill", [])
        execute = spans_by_name.get("first_action_execute", []) or structured_events.get("first_action_execute", [])
        for metric, starts, start_attr, ends, end_attr in (
            ("obs_frame_ms", dispatch, "timestamp_ns", frame, "timestamp_ns"),
            ("dispatch_to_infer_ms", dispatch, "timestamp_ns", model, "start_ns"),
            ("queue_refill_ms", dispatch, "timestamp_ns", refill, "end_ns"),
            ("refill_to_execute_ms", refill, "end_ns", execute, "end_ns"),
            ("total_ms", dispatch, "timestamp_ns", execute, "end_ns"),
        ):
            selected = unique(metric, [starts, ends], "structured")
            if selected:
                points = []
                for record, attr in ((selected[0], start_attr), (selected[1], end_attr)):
                    if isinstance(record, TraceEvent) and attr == "end_ns":
                        # Event timestamps differ between dispatchers; never infer publish end.
                        point = (
                            record.field("publish_end_ns")
                            if record.name == "first_action_execute"
                            else record.timestamp_ns
                        )
                    else:
                        point = getattr(record, attr)
                    points.append(point)
                if all(type(point) is int for point in points) and points[1] >= points[0]:
                    row[metric] = round((points[1] - points[0]) / 1_000_000, 2)
                else:
                    status = "incomplete" if any(point is None for point in points) else "invalid"
                    row[f"{metric}_status"] = status
                    warnings.append(
                        f"request {request_id} metric {metric}: {status} boundary; excluded from request summary"
                    )
                row[f"{metric}_source"] = "structured"
        for metric, event_name, field_name, component in (
            ("action_chunk_publish_ms", "action_chunk_publish", "publish_ms", "policy.postprocess"),
            ("dispatch_decode_ms", "dispatch_decode", "decode_ms", "action_dispatcher.decode"),
            ("policy_total_reported_ms", "dispatch_result", "policy_total_ms", "policy"),
        ):
            candidates = [
                event
                for event in events
                if event.name == event_name
                and event.schema_version > 0
                and event.component_id == component
                and event.field("origin", "built-in") == "built-in"
            ]
            selected = unique(metric, [candidates], "reported")
            if selected:
                raw_value = selected[0].field(field_name)
                value = _number(raw_value)
                row[f"{metric}_source"] = "reported"
                if value is not None and value >= 0:
                    row[metric] = round(value, 2)
                else:
                    status = "incomplete" if raw_value is None else "invalid"
                    row[f"{metric}_status"] = status
                    warnings.append(
                        f"request {request_id} metric {metric}: {status} {field_name}; excluded from request summary"
                    )
        rows.append(row)
    return rows


def _observations(dataset: TraceDataset) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for event in dataset.events:
        if event.schema_version <= 0:
            continue
        key = str(event.field("key", ""))
        if not key:
            continue
        bucket = result.setdefault(key, {"transport_ms": [], "age_ms": []})
        if event.name == "obs_receive":
            value = _number(event.field("transport_ms"))
            if value is not None:
                bucket["transport_ms"].append(value)
        elif event.name == "obs_sample" and str(event.field("ready", "0")).lower() in {"1", "true"}:
            value = _number(event.field("age_ms"))
            if value is not None:
                bucket["age_ms"].append(value)
    return result


def _spans(dataset: TraceDataset, warnings: list[str]) -> list[SpanRecord]:
    begins: dict[tuple[str, str], TraceEvent] = {}
    spans: list[SpanRecord] = []
    for event in dataset.events:
        span_id = str(event.field("span_id", ""))
        key = (event.request_id, span_id)
        if event.name == "span_begin" and span_id:
            if key in begins:
                warnings.append(f"duplicate span_begin: {event.request_id}/{span_id}")
            begins[key] = event
        elif event.name == "span_end" and span_id:
            begin = begins.pop(key, None)
            if begin is None:
                warnings.append(f"span_end without span_begin: {event.request_id}/{span_id}")
                continue
            spans.append(
                SpanRecord(
                    name=str(event.field("span_name", begin.field("span_name", "span"))),
                    trace_id=event.request_id or begin.request_id,
                    span_id=span_id,
                    parent_span_id=str(event.field("parent_span_id", begin.field("parent_span_id", ""))),
                    component_id=str(event.field("component_id", begin.field("component_id", ""))),
                    start_ns=begin.timestamp_ns,
                    end_ns=event.timestamp_ns,
                    status=str(event.field("status", "ok")),
                    origin=str(event.field("origin", begin.field("origin", "user"))),
                    fields=dict(event.fields),
                )
            )
    for (trace_id, span_id), begin in begins.items():
        warnings.append(f"incomplete span: {trace_id}/{span_id}")
        spans.append(
            SpanRecord(
                name=str(begin.field("span_name", "span")),
                trace_id=begin.request_id,
                span_id=span_id,
                parent_span_id=str(begin.field("parent_span_id", "")),
                component_id=begin.component_id,
                start_ns=begin.timestamp_ns,
                end_ns=None,
                status="incomplete",
                origin=str(begin.field("origin", "user")),
                fields=dict(begin.fields),
            )
        )
    return spans


def _flows(dataset: TraceDataset, warnings: list[str]) -> list[FlowRecord]:
    grouped: dict[tuple[str, str, str], dict[str, TraceEvent]] = {}
    for event in dataset.events:
        if event.name not in {"flow_send", "flow_receive"}:
            continue
        edge_id = str(event.field("edge_id", ""))
        flow_id = str(event.field("flow_id", ""))
        if edge_id and flow_id:
            key = (event.request_id, edge_id, flow_id)
            pair = grouped.setdefault(key, {})
            if event.name in pair:
                warnings.append(f"duplicate {event.name}: {event.request_id}/{edge_id}/{flow_id}")
            pair[event.name] = event
    records = []
    for (trace_id, edge_id, flow_id), pair in grouped.items():
        send = pair.get("flow_send")
        receive = pair.get("flow_receive")
        status = "complete" if send and receive else "incomplete"
        if send and receive and receive.timestamp_ns < send.timestamp_ns:
            status = "negative"
            warnings.append(f"negative flow duration: {edge_id}/{flow_id}")
        records.append(
            FlowRecord(
                edge_id=edge_id,
                flow_id=flow_id,
                trace_id=trace_id,
                send_ns=send.timestamp_ns if send else None,
                receive_ns=receive.timestamp_ns if receive else None,
                status=status,
            )
        )
    return records


def _custom_span_summary(spans: list[SpanRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for span in spans:
        if span.origin != "user" or span.duration_ms is None:
            continue
        grouped.setdefault((span.component_id, span.name), []).append(span.duration_ms)
    return [
        {"component_id": component_id, "name": name, **summarize(values).to_dict()}
        for (component_id, name), values in sorted(grouped.items())
    ]


def _span_summary(spans: list[SpanRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for span in spans:
        duration = span.duration_ms
        if span.end_ns is None or duration is None or not math.isfinite(duration) or duration < 0:
            continue
        grouped.setdefault((span.component_id, span.name, span.origin, span.status), []).append(duration)
    return [
        {"component_id": component, "name": name, "origin": origin, "status": status, **summarize(values).to_dict()}
        for (component, name, origin, status), values in sorted(grouped.items())
    ]


def _custom_mark_summary(dataset: TraceDataset) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for event in dataset.events:
        if (
            event.schema_version <= 0
            or event.field("origin", "built-in") != "user"
            or event.name in {"span_begin", "span_end", "flow_send", "flow_receive"}
        ):
            continue
        key = (event.component_id, event.name)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"component_id": component_id, "name": name, "count": count}
        for (component_id, name), count in sorted(counts.items())
    ]


class AnalysisService:
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        dataset = parse_trace(request.source, source_kind=request.source_kind, max_events=request.max_events)
        topology_path = request.topology_path
        if topology_path is None and request.source.is_dir():
            candidate = request.source / "ibrobot-topology.json"
            topology_path = candidate if candidate.exists() else None
        topology = None
        if topology_path is not None:
            try:
                topology = read_topology_manifest(topology_path)
            except TopologyMetadataError as exc:
                dataset.warnings.append(f"Ignoring topology manifest: {exc}; using observed topology")
        return self.analyze_dataset(dataset, topology=topology)

    def analyze_dataset(self, dataset: TraceDataset, *, topology: TraceTopology | None = None) -> AnalysisResult:
        dataset = deepcopy(dataset)
        _bind_components(dataset, topology)
        warnings = _Warnings()
        warnings.extend(dataset.warnings)
        spans = _spans(dataset, warnings)
        rows = _requests(dataset, spans, warnings)
        topology = enrich_topology_with_observed_operations(topology, spans, dataset.events)
        definitions, definition_warnings = resolve_tracepoint_definitions(
            dataset.definitions,
            topology.definitions if topology is not None else [],
            dataset.events,
            spans,
        )
        warnings.extend(definition_warnings)
        flows = _flows(dataset, warnings)
        stage_names = {key for row in rows for key in row if key.endswith("_ms") and not key.endswith("_source")}
        stage_summary = {
            name: summarize(float(row[name]) for row in rows if name in row) for name in sorted(stage_names)
        }
        expected_metrics = set(_REQUEST_METRICS)
        if topology is not None and topology.execution_mode == "distributed":
            expected_metrics.add(_CLOUD_STAGE)
        observed_metrics = {metric for metric in expected_metrics if any(metric in row for row in rows)}
        coverage = {
            "boundary": "Dispatch Request -> First Action Execute",
            "expected_metrics": len(expected_metrics),
            "observed_metrics": len(observed_metrics),
            "missing_events": sorted(expected_metrics - observed_metrics),
            "todos": [
                "Extend coverage from sensor capture to hardware write/physical actuation",
                "Add C++ and native UST custom instrumentation",
            ],
        }
        return AnalysisResult(
            dataset=dataset,
            request_rows=rows,
            observations=_observations(dataset),
            stage_summary=stage_summary,
            custom_span_summary=_custom_span_summary(spans),
            custom_mark_summary=_custom_mark_summary(dataset),
            spans=spans,
            flows=flows,
            topology=topology,
            coverage=coverage,
            warnings=warnings.finish(),
            definitions=definitions,
            span_summary=_span_summary(spans),
        )
