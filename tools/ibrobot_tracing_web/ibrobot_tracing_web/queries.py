"""Thin web serialization adapter over the UI-neutral tracing query layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ibrobot_tracing.critical_path import CriticalPathQuery, project_critical_path
from ibrobot_tracing.projection import (
    CallTreeQuery,
    GraphQuery,
    LatencyDistributionQuery,
    SpanProfileQuery,
    TimelineQuery,
    project_call_tree,
    project_graph,
    project_latency_distribution,
    project_span_profile,
    project_timeline,
)
from ibrobot_tracing.query import (
    ComponentQuery,
    EventQuery,
    FlowQuery,
    QueryService,
    RequestQuery,
    SpanQuery,
    TracepointQuery,
    stable_event_id,
    stable_flow_id,
    stable_span_id,
    stable_tracepoint_id,
)
from ibrobot_tracing.serialization import to_js_safe


def _page(page: Any, serializer) -> dict[str, Any]:
    return {
        "items": [serializer(item) for item in page.items],
        "total": page.total,
        "offset": page.offset,
        "limit": page.limit,
        "next_offset": page.next_offset,
    }


def _redact_paths(message: str, paths: set[str]) -> str:
    for path in sorted(paths, key=len, reverse=True):
        message = message.replace(path, "<trace-source>")
    return message


def _redact_path_values(value: Any, paths: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {_redact_paths(str(key), paths): _redact_path_values(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_path_values(item, paths) for item in value]
    if isinstance(value, str):
        return _redact_paths(value, paths)
    return value


class ResultQueries:
    """Expose one AnalysisResult through stable, path-safe web DTOs."""

    def __init__(self, result: Any):
        self.result = result
        self.query = QueryService(result)

    @staticmethod
    def record_dict(value: Any) -> dict[str, Any]:
        return to_js_safe(asdict(value))

    @staticmethod
    def event_dict(event: Any) -> dict[str, Any]:
        item = asdict(event)
        item["origin"].pop("path", None)
        item["id"] = stable_event_id(event)
        item["request_id"] = event.request_id
        item["component_id"] = event.component_id
        return to_js_safe(item)

    @staticmethod
    def span_dict(span: Any) -> dict[str, Any]:
        return to_js_safe(asdict(span) | {"id": stable_span_id(span), "duration_ms": span.duration_ms})

    @staticmethod
    def flow_dict(flow: Any) -> dict[str, Any]:
        return to_js_safe(asdict(flow) | {"id": stable_flow_id(flow), "duration_ms": flow.duration_ms})

    @staticmethod
    def tracepoint_dict(definition: Any) -> dict[str, Any]:
        return to_js_safe(asdict(definition) | {"id": stable_tracepoint_id(definition)})

    def metadata(self) -> dict[str, Any]:
        return to_js_safe(
            {
                key: value
                for key, value in self.result.dataset.metadata.items()
                if key != "source" and not key.endswith("_path")
            }
        )

    def _source_paths(self) -> set[str]:
        return {
            value
            for value in [
                str(self.result.dataset.metadata.get("source", "")),
                *(event.origin.path for event in self.result.dataset.events),
            ]
            if value and Path(value).is_absolute()
        }

    def request_rows(self) -> list[dict[str, Any]]:
        return self.result.request_rows

    def events(self) -> list[Any]:
        return self.result.dataset.events

    def spans(self) -> list[Any]:
        return self.result.spans

    def flows(self) -> list[Any]:
        return self.result.flows

    def warnings(self, *, limit: int | None = None) -> list[str]:
        paths = self._source_paths()
        return [_redact_paths(warning, paths) for warning in self.result.warnings[:limit]]

    def stage_summary(self) -> dict[str, dict[str, Any]]:
        return {name: summary.to_dict() for name, summary in self.result.stage_summary.items()}

    def observations(self) -> dict[str, dict[str, list[float]]]:
        return to_js_safe(self.result.observations)

    def span_summary(self) -> dict[str, Any]:
        limit = 100
        rows = self.result.span_summary
        return {
            "span_summary": to_js_safe(rows[:limit]),
            "span_summary_total": len(rows),
            "span_summary_limit": limit,
            "span_summary_truncated": len(rows) > limit,
        }

    def custom_span_summary(self) -> list[dict[str, Any]]:
        return self.result.custom_span_summary

    def custom_mark_summary(self) -> list[dict[str, Any]]:
        return self.result.custom_mark_summary

    def coverage(self) -> dict[str, Any]:
        return self.result.coverage

    def query_requests(self, **filters: Any) -> dict[str, Any]:
        page = self.query.requests(RequestQuery(**filters))
        return _page(page, lambda row: to_js_safe(dict(row)))

    def query_events(self, **filters: Any) -> dict[str, Any]:
        return _page(self.query.events(EventQuery(**filters)), self.event_dict)

    def query_spans(self, **filters: Any) -> dict[str, Any]:
        return _page(self.query.spans(SpanQuery(**filters)), self.span_dict)

    def query_flows(self, **filters: Any) -> dict[str, Any]:
        return _page(self.query.flows(FlowQuery(**filters)), self.flow_dict)

    def query_components(self, **filters: Any) -> dict[str, Any]:
        return _page(self.query.components(ComponentQuery(**filters)), self.record_dict)

    def query_tracepoints(self, **filters: Any) -> dict[str, Any]:
        return _page(self.query.tracepoints(TracepointQuery(**filters)), self.tracepoint_dict)

    def timeline_projection(self, **filters: Any) -> dict[str, Any]:
        return project_timeline(self.result, TimelineQuery(**filters)).to_dict()

    def call_tree_projection(self, **filters: Any) -> dict[str, Any]:
        return project_call_tree(self.result, CallTreeQuery(**filters)).to_dict()

    def latency_distribution_projection(self, **filters: Any) -> dict[str, Any]:
        return project_latency_distribution(self.result, LatencyDistributionQuery(**filters)).to_dict()

    def span_profile_projection(self, **filters: Any) -> dict[str, Any]:
        profile = project_span_profile(self.result, SpanProfileQuery(**filters)).to_dict()
        return _redact_path_values(profile, self._source_paths())

    def critical_path_projection(self, **filters: Any) -> dict[str, Any]:
        projection = project_critical_path(self.result, CriticalPathQuery(**filters)).to_dict()
        return _redact_path_values(projection, self._source_paths())

    def graph_projection(self, **filters: Any) -> dict[str, Any]:
        return project_graph(self.result, GraphQuery(**filters)).to_dict()
