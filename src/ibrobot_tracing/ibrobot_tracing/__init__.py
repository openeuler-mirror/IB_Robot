"""Public APIs for IB-Robot tracing and offline performance analysis."""

from .instrumentation import (
    TraceEmitter,
    create_trace_logger,
    end_span,
    flow_receive,
    flow_send,
    get_trace_emitter,
    mark,
    span,
    start_span,
    trace_context,
)

_LAZY_EXPORTS = {
    **dict.fromkeys(("AnalysisRequest", "AnalysisResult", "AnalysisService"), "analysis"),
    **dict.fromkeys(
        (
            "ComparisonHistogram",
            "FlameDiffEntry",
            "MetricComparison",
            "StatisticDelta",
            "TraceComparisonProjection",
            "TraceComparisonService",
        ),
        "comparison",
    ),
    **dict.fromkeys(("CriticalPathProjection", "CriticalPathQuery", "project_critical_path"), "critical_path"),
    **dict.fromkeys(("BUILTIN_TRACEPOINT_DESCRIPTIONS", "BUILTIN_TRACEPOINT_REGISTRY"), "definitions"),
    **dict.fromkeys(("EventOrigin", "TraceDataset", "TraceEvent", "TracepointDefinition", "TraceTopology"), "model"),
    **dict.fromkeys(
        (
            "AggregateSpanProfileProjection",
            "CallTreeQuery",
            "GraphQuery",
            "LatencyDistributionProjection",
            "LatencyDistributionQuery",
            "RequestSpanProfileProjection",
            "SpanProfileProjection",
            "SpanProfileQuery",
            "TimelineQuery",
            "project_call_tree",
            "project_graph",
            "project_latency_distribution",
            "project_span_profile",
            "project_timeline",
            "span_profile_speedscope",
        ),
        "projection",
    ),
    **dict.fromkeys(
        (
            "ComponentQuery",
            "EventQuery",
            "FlowQuery",
            "QueryService",
            "RequestQuery",
            "SpanQuery",
            "TracepointQuery",
            "stable_component_id",
            "stable_event_id",
            "stable_flow_id",
            "stable_request_id",
            "stable_span_id",
            "stable_tracepoint_id",
        ),
        "query",
    ),
    "bind_robot_topology": "topology",
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisService",
    "AggregateSpanProfileProjection",
    "BUILTIN_TRACEPOINT_DESCRIPTIONS",
    "BUILTIN_TRACEPOINT_REGISTRY",
    "CallTreeQuery",
    "CriticalPathProjection",
    "CriticalPathQuery",
    "ComparisonHistogram",
    "ComponentQuery",
    "EventQuery",
    "EventOrigin",
    "FlameDiffEntry",
    "FlowQuery",
    "GraphQuery",
    "LatencyDistributionProjection",
    "LatencyDistributionQuery",
    "MetricComparison",
    "QueryService",
    "RequestQuery",
    "RequestSpanProfileProjection",
    "SpanProfileProjection",
    "SpanProfileQuery",
    "SpanQuery",
    "StatisticDelta",
    "TimelineQuery",
    "TraceDataset",
    "TraceComparisonProjection",
    "TraceComparisonService",
    "TraceEmitter",
    "TraceEvent",
    "TracepointDefinition",
    "TracepointQuery",
    "TraceTopology",
    "bind_robot_topology",
    "create_trace_logger",
    "end_span",
    "flow_receive",
    "flow_send",
    "get_trace_emitter",
    "mark",
    "project_call_tree",
    "project_critical_path",
    "project_graph",
    "project_latency_distribution",
    "project_span_profile",
    "project_timeline",
    "span",
    "start_span",
    "stable_component_id",
    "stable_event_id",
    "stable_flow_id",
    "stable_request_id",
    "stable_span_id",
    "stable_tracepoint_id",
    "span_profile_speedscope",
    "trace_context",
]
