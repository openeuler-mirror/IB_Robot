"""Text and JSON renderers for analysis results."""

from __future__ import annotations

import json
from typing import TextIO

from .analysis import AnalysisResult
from .critical_path import CriticalPathProjection
from .model import SpanRecord, TraceEvent
from .projection import (
    CallTreeProjection,
    GraphProjection,
    GraphQuery,
    LatencyDistributionProjection,
    RequestSpanProfileProjection,
    SpanProfileProjection,
    TimelineProjection,
    project_event_timeline,
    project_graph,
    project_span_call_tree,
)

_STAGE_LABELS = {
    "obs_frame_ms": "Obs sampling",
    "dispatch_to_infer_ms": "Dispatch->Infer",
    "preprocess_ms": "Preprocess",
    "inference_ms": "Model call",
    "postprocess_ms": "Postprocess",
    "action_chunk_publish_ms": "Chunk publish",
    "dispatch_decode_ms": "Dispatch decode",
    "queue_refill_ms": "Dispatch->Refill",
    "refill_to_execute_ms": "Refill->Execute",
    "execute_publish_ms": "Execute publish",
    "cloud_roundtrip_ms": "Cloud round trip",
    "total_ms": "Dispatch->Execute",
}
_SUMMARY_WIDTH = 88


def render_json(result: AnalysisResult, stream: TextIO, *, compatibility: bool = False) -> None:
    json.dump(result.to_dict(compatibility=compatibility), stream, indent=2)
    stream.write("\n")


def render_summary(result: AnalysisResult, stream: TextIO) -> None:
    stream.write("=" * _SUMMARY_WIDTH + "\n")
    stream.write("IB-Robot Performance Report\n")
    stream.write(f"Requests: {len(result.request_rows)}\n")
    stream.write(f"Boundary: {result.coverage.get('boundary', '-')}\n")
    stream.write("=" * _SUMMARY_WIDTH + "\n")
    stream.write(f"{'Stage':<26} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9} {'mean':>9} {'n':>5}\n")
    stream.write("-" * _SUMMARY_WIDTH + "\n")
    for name in _STAGE_LABELS:
        stats = result.stage_summary.get(name)
        if not stats or not stats.count:
            continue
        prefix = ">>>" if name == "total_ms" else "   "
        stream.write(
            f"{prefix}{_STAGE_LABELS[name]:<23} {stats.p50:>8.1f}ms {stats.p95:>8.1f}ms "
            f"{stats.p99:>8.1f}ms {stats.maximum:>8.1f}ms {stats.mean:>8.1f}ms {stats.count:>4}\n"
        )
    stream.write("=" * _SUMMARY_WIDTH + "\n")
    if result.custom_span_summary:
        stream.write("Custom Span Summary\n")
        stream.write("-" * _SUMMARY_WIDTH + "\n")
        stream.write(f"{'Span':<22} {'Component':<28} {'p50':>8} {'p95':>8} {'max':>8} {'mean':>8} {'n':>5}\n")
        for item in result.custom_span_summary:
            stream.write(
                f"{item['name']:<22} {item['component_id'] or '-':<28} "
                f"{item['p50']:>7.1f}ms {item['p95']:>7.1f}ms {item['maximum']:>7.1f}ms "
                f"{item['mean']:>7.1f}ms {item['count']:>4}\n"
            )
        stream.write("=" * _SUMMARY_WIDTH + "\n")
    if result.custom_mark_summary:
        stream.write("Custom Marks\n")
        stream.write("-" * _SUMMARY_WIDTH + "\n")
        for item in result.custom_mark_summary:
            stream.write(f"{item['name']:<30} {item['component_id'] or '-':<40} count={item['count']}\n")
        stream.write("=" * _SUMMARY_WIDTH + "\n")
    if result.spans:
        stream.write(f"Structured spans: {len(result.spans)}\n")
    if result.flows:
        stream.write(f"Correlated flows: {len(result.flows)}\n")
    missing = result.coverage.get("missing_events", [])
    if missing:
        stream.write("Missing coverage: " + ", ".join(missing) + "\n")


def render_latency_distribution(distribution: LatencyDistributionProjection, stream: TextIO) -> None:
    metric = distribution.metric or "none"
    stream.write(f"Latency Distribution: {metric} ({distribution.unit})\n")
    stream.write(
        f"Samples: {distribution.sample_count}  Invalid: {distribution.invalid_count}  Bins: {distribution.bin_count}\n"
    )
    if not distribution.buckets:
        available = ", ".join(distribution.available_metrics) or "none"
        stream.write(f"No finite samples. Available metrics: {available}\n")
        return
    summary = distribution.summary
    stream.write(
        f"Min: {summary.minimum:.3f}  p50: {summary.p50:.3f}  p95: {summary.p95:.3f}  "
        f"p99: {summary.p99:.3f}  Max: {summary.maximum:.3f}  Mean: {summary.mean:.3f}\n"
    )
    stream.write("\nHistogram\n")
    largest = max(bucket.count for bucket in distribution.buckets)
    for bucket in distribution.buckets:
        bar_size = round(bucket.count * 40 / largest) if largest else 0
        request_suffix = ""
        if bucket.request_ids:
            request_suffix = "  " + ", ".join(bucket.request_ids)
            if bucket.truncated:
                request_suffix += ", ..."
        stream.write(
            f"{bucket.index:>3} [{bucket.start:>10.3f}, {bucket.end:>10.3f}] "
            f"{bucket.count:>6} {'#' * bar_size}{request_suffix}\n"
        )
    stream.write("\nHighest values\n")
    for outlier in distribution.outliers:
        marker = " p95-tail" if outlier.p95_tail else ""
        stream.write(
            f"{outlier.rank:>3}. {outlier.request_id or '-'}  {outlier.value:.3f}{distribution.unit}  "
            f"p{outlier.percentile:g}{marker}\n"
        )


def render_legacy_summary(result: AnalysisResult, stream: TextIO) -> None:
    labels = dict(_STAGE_LABELS)
    labels["inference_ms"] = "Inference"
    labels["cloud_roundtrip_ms"] = "Network (edge<->cloud)"
    stream.write("=" * 70 + "\n")
    stream.write("IB-Robot Inference Chain Latency Report\n")
    stream.write(f"Requests: {len(result.request_rows)}\n")
    stream.write("=" * 70 + "\n")
    stream.write(f"{'Stage':<25} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'n':>5}\n")
    stream.write("-" * 70 + "\n")
    for name in _STAGE_LABELS:
        stats = result.stage_summary.get(name)
        if not stats or not stats.count:
            continue
        prefix = ">>>" if name == "total_ms" else "   "
        stream.write(
            f"{prefix}{labels[name]:<22} {stats.p50:>7.1f}ms {stats.p95:>7.1f}ms "
            f"{stats.p99:>7.1f}ms {stats.maximum:>7.1f}ms {stats.count:>4}\n"
        )
    stream.write("=" * 70 + "\n")
    if not result.observations:
        return
    stream.write("Observation Ingress / Sampling Summary\n")
    stream.write("=" * 70 + "\n")
    stream.write(f"{'Observation':<28} {'recv_p50':>9} {'recv_p95':>9} {'age_p50':>9} {'age_p95':>9}\n")
    stream.write("-" * 70 + "\n")
    from .statistics import summarize

    for key in sorted(result.observations):
        transport = summarize(result.observations[key]["transport_ms"])
        age = summarize(result.observations[key]["age_ms"])

        def value(number: float | None) -> str:
            return f"{number:.1f}ms" if number is not None else "-"

        stream.write(
            f"{key:<28} {value(transport.p50):>9} {value(transport.p95):>9} {value(age.p50):>9} {value(age.p95):>9}\n"
        )
    stream.write("=" * 70 + "\n")


def render_timeline(events: list[TraceEvent] | TimelineProjection, stream: TextIO) -> None:
    timeline = events if isinstance(events, TimelineProjection) else project_event_timeline(events)
    event_items = [item for item in timeline.items if item.kind == "event"]
    if not event_items:
        stream.write("No matching events.\n")
        return
    base = event_items[0].start_ns
    previous = base
    stream.write(f"{'Relative':>12} {'Delta':>12} {'Component':<30} Event\n")
    stream.write("-" * 90 + "\n")
    for item in event_items:
        relative = (item.start_ns - base) / 1_000_000
        delta = (item.start_ns - previous) / 1_000_000
        previous = item.start_ns
        component = item.component_id or "-"
        details = " ".join(
            f"{key}={value}"
            for key, value in item.fields.items()
            if key not in {"component_id", "trace_id", "request_id", "span_id", "parent_span_id"}
        )
        stream.write(f"{relative:>10.3f}ms {delta:>10.3f}ms {component:<30} {item.label} {details}\n")


def render_call_tree(spans: list[SpanRecord] | CallTreeProjection, stream: TextIO) -> None:
    tree = spans if isinstance(spans, CallTreeProjection) else project_span_call_tree(spans)
    if not tree.nodes:
        stream.write("No structured spans found. Add span() instrumentation and capture a new trace.\n")
        return
    by_id = {node.id: node for node in tree.nodes}
    stack = [(root_id, 0) for root_id in reversed(tree.root_ids)]
    while stack:
        node_id, depth = stack.pop()
        item = by_id[node_id]
        duration = f"{item.duration_ns / 1_000_000:.3f}ms" if item.duration_ns is not None else "incomplete"
        stream.write(f"{'  ' * depth}{item.name} [{item.origin}] {duration} ({item.component_id or '-'})\n")
        stack.extend((child_id, depth + 1) for child_id in reversed(item.child_ids))


def render_span_profile(profile: SpanProfileProjection, stream: TextIO) -> None:
    stream.write("Instrumented Span Wall Time (not CPU)\n")
    stream.write(
        f"Source: {profile.source}\nCoverage: {profile.coverage}\nNodes: {profile.returned_nodes}/{profile.total_nodes}"
    )
    if profile.truncated:
        stream.write(f" (truncated: {profile.truncation_reason})")
    stream.write("\n")
    if not profile.nodes:
        stream.write("No matching structured spans.\n")
        return
    by_id = {node.id: node for node in profile.nodes}

    stack = list(reversed(profile.roots))
    while stack:
        node_id = stack.pop()
        node = by_id[node_id]
        if isinstance(profile, RequestSpanProfileProjection):
            duration = f"{node.duration_ns / 1_000_000:.3f}ms" if node.duration_ns is not None else "unavailable"
            uncovered = (
                f"{node.uncovered_wall_ns / 1_000_000:.3f}ms" if node.uncovered_wall_ns is not None else "unavailable"
            )
            stream.write(
                f"{'  ' * node.depth}{node.name} [{node.origin}] wall={duration} "
                f"uncovered={uncovered} track={node.track} ({node.component_id or '-'})\n"
            )
        else:
            stream.write(
                f"{'  ' * node.depth}{node.name} [{node.origin}] "
                f"wall={node.value_ns / 1_000_000:.3f}ms uncovered={node.self_value_ns / 1_000_000:.3f}ms "
                f"{node.percentage:.2f}% n={node.occurrence_count} ({node.component_id or '-'})\n"
            )
        stack.extend(reversed(node.child_ids))


def render_critical_path(projection: CriticalPathProjection, stream: TextIO) -> None:
    stream.write("Request Critical Wall Path (instrumented wall attribution, not CPU or a scheduling DAG)\n")
    stream.write(
        f"Request: {projection.request_id}  Boundary: {projection.boundary_source}  "
        f"Duration: {projection.duration_ns / 1_000_000:.3f}ms\n"
    )
    stream.write(
        f"Coverage: {projection.totals.coverage_percent:.2f}%  "
        f"Segments: {projection.returned_segments}/{projection.total_segments}"
    )
    if projection.truncated:
        stream.write(
            f" (truncated: {projection.truncation_reason}, omitted "
            f"{projection.omitted_segments} segment(s) / {projection.omitted_duration_ns / 1_000_000:.3f}ms)"
        )
    stream.write("\n")
    if not projection.segments:
        stream.write("No request boundary or attributable wall intervals were found.\n")
    else:
        stream.write(f"{'Offset':>12} {'Duration':>12} {'Kind':<13} {'Owner':<32} Component/Edge\n")
        stream.write("-" * 96 + "\n")
        for segment in projection.segments:
            owner = segment.label or "-"
            location = segment.component_id or segment.edge_id or "-"
            stream.write(
                f"{segment.offset_ns / 1_000_000:>10.3f}ms "
                f"{segment.duration_ns / 1_000_000:>10.3f}ms "
                f"{segment.kind:<13} {owner:<32} {location}\n"
            )
    if projection.diagnostics:
        stream.write("Diagnostics:\n")
        for diagnostic in projection.diagnostics:
            stream.write(f"  {diagnostic.code} ({diagnostic.count}): {diagnostic.message}\n")


def render_graph(
    result: AnalysisResult | GraphProjection,
    stream: TextIO,
    *,
    request_id: str = "",
    metric: str = "p95",
    view: str = "components",
) -> None:
    graph = (
        result
        if isinstance(result, GraphProjection)
        else project_graph(result, GraphQuery(request_id=request_id, metric=metric, view=view))
    )
    if graph.topology_source == "none":
        stream.write("No topology manifest is available.\n")
        return
    for node in graph.nodes:
        indent = "  " * node.depth
        suffix = f"  processing={node.metric_value_ms:.3f}ms" if node.metric_value_ms is not None else ""
        marker = "*" if node.provenance == "observed" else ""
        stream.write(f"{indent}[{node.name}{marker}] {node.component_id}{suffix}\n")
    stream.write("\nFlows:\n")
    for edge in graph.edges:
        suffix = f" {edge.metric_value_ms:.3f}ms" if edge.metric_value_ms is not None else ""
        connector = f"--{edge.name}{suffix}--" if not edge.directed else f"--{edge.name}{suffix}-->"
        stream.write(f"  {edge.source_id} {connector} {edge.target_id}\n")
