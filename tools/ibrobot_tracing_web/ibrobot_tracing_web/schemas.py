"""Pydantic request and response schemas for API version 1."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrictApiModel(ApiModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    worker_running: bool
    source_count: int
    analysis_count: int
    queued_jobs: int


class CapabilitiesResponse(ApiModel):
    api_version: str = "v1"
    authentication: Literal["none"] = "none"
    tls_enabled: bool
    cors_enabled: bool
    baseline_compare: bool
    source_kinds: list[str]
    analysis_views: list[str]
    limits: dict[str, int]


class Source(ApiModel):
    id: str
    name: str
    kind: Literal["ctf", "log"]
    version: str
    modified_at: datetime
    size_bytes: int
    file_count: int


class SourceListResponse(ApiModel):
    generation: int
    refreshed_at: datetime
    items: list[Source]
    warnings: list[str]


class AnalysisJobCreate(ApiModel):
    source_id: str = Field(min_length=1, max_length=64)


class AnalysisJob(ApiModel):
    id: str
    source_id: str
    source_version: str
    status: Literal["queued", "running", "completed", "failed", "cancelled", "expired"]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    analysis_id: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    queue_position: int | None = None
    deduplicated: bool = False


class Analysis(ApiModel):
    id: str
    source_id: str
    source_name: str
    source_kind: Literal["ctf", "log"]
    source_version: str
    created_at: datetime
    event_count: int
    request_count: int
    span_count: int
    flow_count: int
    warning_count: int


class AnalysisListResponse(ApiModel):
    items: list[Analysis]


class NumericSummary(ApiModel):
    count: int
    minimum: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    maximum: float | None = None
    mean: float | None = None


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class LatencyDistributionSummary(StrictApiModel):
    count: NonNegativeInt
    minimum: FiniteFloat | None = None
    p50: FiniteFloat | None = None
    p95: FiniteFloat | None = None
    p99: FiniteFloat | None = None
    maximum: FiniteFloat | None = None
    mean: FiniteFloat | None = None


class LatencyDistributionBucket(StrictApiModel):
    index: NonNegativeInt
    start: FiniteFloat
    end: FiniteFloat
    count: NonNegativeInt
    request_ids: list[str]
    returned: NonNegativeInt
    truncated: bool


class LatencyDistributionOutlier(StrictApiModel):
    request_id: str
    value: FiniteFloat
    rank: Annotated[int, Field(ge=1)]
    percentile: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    p95_tail: bool


class LatencyDistributionResponse(StrictApiModel):
    metric: str | None
    unit: Literal["ms"]
    available_metrics: list[str]
    summary: LatencyDistributionSummary
    sample_count: NonNegativeInt
    invalid_count: NonNegativeInt
    minimum: FiniteFloat | None
    maximum: FiniteFloat | None
    bin_count: NonNegativeInt
    buckets: list[LatencyDistributionBucket]
    outliers: list[LatencyDistributionOutlier]


class SummaryResponse(ApiModel):
    analysis: Analysis
    metadata: dict[str, Any]
    stages: dict[str, NumericSummary]
    observations: dict[str, dict[str, list[float]]]
    span_summary: list[dict[str, Any]] = Field(
        max_length=100,
        description="First 100 component/name/origin/status groups of complete span occurrences, not request success",
    )
    span_summary_total: int = Field(ge=0, description="Total groups before truncation, not occurrence count")
    span_summary_limit: Literal[100]
    span_summary_truncated: bool
    custom_span_summary: list[dict[str, Any]]
    custom_mark_summary: list[dict[str, Any]]
    coverage: dict[str, Any]


class Page(ApiModel):
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
    items: list[dict[str, Any]]


class EventOrigin(ApiModel):
    source: str = "unknown"
    provider: str = ""
    host: str = ""
    process_id: int | None = None
    thread_id: int | None = None
    node: str = ""


class TraceEvent(ApiModel):
    id: str
    timestamp_ns: str | int
    name: str
    fields: dict[str, Any]
    origin: EventOrigin
    clock: str
    schema_version: int
    sequence: int
    request_id: str
    component_id: str


class EventPage(ApiModel):
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
    items: list[TraceEvent]


class Span(ApiModel):
    id: str
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str
    component_id: str
    start_ns: str | int
    end_ns: str | int | None
    status: str
    origin: str
    fields: dict[str, Any]
    duration_ms: float | None


class SpanPage(ApiModel):
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
    items: list[Span]


class Flow(ApiModel):
    id: str
    edge_id: str
    flow_id: str
    trace_id: str
    send_ns: str | int | None
    receive_ns: str | int | None
    status: str
    duration_ms: float | None


class FlowPage(ApiModel):
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
    items: list[Flow]


class Component(ApiModel):
    component_id: str
    name: str
    kind: str
    parent_id: str = ""
    package: str = ""
    executable: str = ""
    node: str = ""
    provenance: str = "declared"
    trace_name: str = ""


class ComponentListResponse(ApiModel):
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
    items: list[Component]


class TracepointDefinition(ApiModel):
    id: str
    kind: Literal["event", "span"]
    component_id: str
    name: str
    origin: str
    description: str = ""


class TracepointPage(ApiModel):
    total: int
    offset: int
    limit: int
    next_offset: int | None = None
    items: list[TracepointDefinition]


class TimelineLane(ApiModel):
    id: str
    label: str
    kind: str
    component_id: str = ""
    edge_id: str = ""


class TimelineItem(ApiModel):
    id: str
    lane_id: str
    kind: str
    label: str
    start_ns: str | int
    end_ns: str | int
    start_offset_ns: int
    end_offset_ns: int
    duration_ns: int
    request_id: str = ""
    component_id: str = ""
    status: str = ""
    fields: dict[str, Any]


class TimelineResponse(ApiModel):
    start_ns: str | int | None
    end_ns: str | int | None
    duration_ns: int
    lanes: list[TimelineLane]
    items: list[TimelineItem]


class CallTreeNode(ApiModel):
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
    start_ns: str | int
    end_ns: str | int | None
    start_offset_ns: int
    duration_ns: int | None
    self_duration_ns: int | None
    orphan: bool = False
    cycle: bool = False
    fields: dict[str, Any]


class CallTreeResponse(ApiModel):
    start_ns: str | int | None
    root_ids: list[str]
    nodes: list[CallTreeNode]
    warnings: list[str]


JsSafeInteger = (
    Annotated[int, Field(ge=-9_007_199_254_740_991, le=9_007_199_254_740_991)]
    | Annotated[str, Field(pattern=r"^-?\d+$")]
)
NonNegativeJsSafeInteger = (
    Annotated[int, Field(ge=0, le=9_007_199_254_740_991)] | Annotated[str, Field(pattern=r"^\d+$")]
)
JsSafeTimestamp = Annotated[str, Field(pattern=r"^-?\d+$")]


class SpanProfileDiagnostic(StrictApiModel):
    code: str
    occurrence_ids: list[str]
    message: str


class SpanProfileTotals(StrictApiModel):
    request_count: int
    occurrence_count: int
    error_count: int
    incomplete_count: int
    invalid_count: int
    excluded_weight_count: int
    sampled_uncovered_wall_ns: JsSafeInteger
    uncovered_wall_ns: JsSafeInteger
    selected_wall_union_ns: JsSafeInteger
    concurrency_factor: float
    diagnostic_counts: dict[str, int]


class AggregateSpanProfileTotals(SpanProfileTotals):
    value_ns: JsSafeInteger


class RequestSpanProfileNode(StrictApiModel):
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
    start_ns: JsSafeTimestamp
    end_ns: JsSafeTimestamp | None
    observed_end_ns: JsSafeTimestamp | None
    start_offset_ns: JsSafeInteger
    duration_ns: JsSafeInteger | None
    duration_source: str
    uncovered_wall_ns: JsSafeInteger | None
    diagnostics: list[str]
    fields: dict[str, Any]


class AggregateSpanProfilePathItem(StrictApiModel):
    component_id: str
    name: str
    origin: str


class AggregateSpanProfileNode(StrictApiModel):
    id: str
    parent_id: str
    child_ids: list[str]
    depth: int
    component_id: str
    name: str
    origin: str
    path: list[AggregateSpanProfilePathItem]
    self_value_ns: JsSafeInteger
    value_ns: JsSafeInteger
    occurrence_count: int
    request_count: int
    error_count: int
    incomplete_count: int
    invalid_count: int
    percentage: float


class SpanProfileResponseBase(StrictApiModel):
    measurement: Literal["instrumented_wall"]
    source: str
    coverage: str
    not_cpu: bool
    roots: list[str]
    diagnostics: list[SpanProfileDiagnostic]
    total_nodes: int
    returned_nodes: int
    truncated: bool
    truncation_reason: str


class RequestSpanProfileResponse(SpanProfileResponseBase):
    mode: Literal["request"]
    request_id: str
    start_ns: JsSafeTimestamp | None
    end_ns: JsSafeTimestamp | None
    duration_ns: JsSafeInteger
    nodes: list[RequestSpanProfileNode]
    totals: SpanProfileTotals


class AggregateSpanProfileResponse(SpanProfileResponseBase):
    mode: Literal["aggregate"]
    nodes: list[AggregateSpanProfileNode]
    totals: AggregateSpanProfileTotals


SpanProfileResponse = Annotated[
    RequestSpanProfileResponse | AggregateSpanProfileResponse,
    Field(discriminator="mode"),
]


class CriticalPathDiagnostic(StrictApiModel):
    code: str
    count: NonNegativeInt
    source_ids: list[str]
    message: str


class CriticalPathSegment(StrictApiModel):
    id: str
    index: NonNegativeInt
    kind: Literal["span", "flow", "unattributed"]
    source_id: str
    label: str
    component_id: str
    edge_id: str
    origin: str
    status: str
    start_ns: JsSafeTimestamp
    end_ns: JsSafeTimestamp
    offset_ns: NonNegativeJsSafeInteger
    duration_ns: NonNegativeJsSafeInteger
    percentage: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    diagnostics: list[str]
    fields: dict[str, Any]


class CriticalPathBottleneck(StrictApiModel):
    rank: Annotated[int, Field(ge=1)]
    kind: Literal["span", "flow"]
    source_id: str
    label: str
    component_id: str
    edge_id: str
    origin: str
    status: str
    duration_ns: NonNegativeJsSafeInteger
    percentage: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    segment_count: Annotated[int, Field(ge=1)]


class CriticalPathTotals(StrictApiModel):
    partition_ns: NonNegativeJsSafeInteger
    attributed_ns: NonNegativeJsSafeInteger
    unattributed_ns: NonNegativeJsSafeInteger
    coverage_percent: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    by_kind_ns: dict[str, NonNegativeJsSafeInteger]
    by_component_ns: dict[str, NonNegativeJsSafeInteger]
    by_edge_ns: dict[str, NonNegativeJsSafeInteger]
    record_counts: dict[str, NonNegativeInt]
    diagnostic_counts: dict[str, NonNegativeInt]


class CriticalPathResponse(StrictApiModel):
    method: Literal["deepest_active_wall_partition"]
    measurement: Literal["instrumented_wall"]
    not_cpu: bool
    request_id: str
    component_id: str
    include_flows: bool
    boundary_source: str
    boundary_start_source: str
    boundary_end_source: str
    start_ns: JsSafeTimestamp | None
    end_ns: JsSafeTimestamp | None
    duration_ns: NonNegativeJsSafeInteger
    segments: list[CriticalPathSegment]
    totals: CriticalPathTotals
    bottlenecks: list[CriticalPathBottleneck]
    diagnostics: list[CriticalPathDiagnostic]
    total_segments: NonNegativeInt
    returned_segments: NonNegativeInt
    returned_duration_ns: NonNegativeJsSafeInteger
    omitted_segments: NonNegativeInt
    omitted_duration_ns: NonNegativeJsSafeInteger
    truncated: bool
    truncation_reason: str


class DataFlowEdge(ApiModel):
    edge_id: str
    source_id: str
    target_id: str
    name: str
    kind: str
    data_type: str = ""
    contract_key: str = ""
    provenance: str = "declared"


class GraphNode(ApiModel):
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


class GraphEdge(ApiModel):
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


class GraphResponse(ApiModel):
    metric: str
    view: Literal["nodes", "components", "tracepoints"]
    request_id: str
    topology_source: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class WarningListResponse(ApiModel):
    items: list[str]
    total: int = 0
    limit: int = 100
    truncated: bool = False


SourceIdentifier = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
SourceVersion = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
AnalysisIdentifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
ComparisonIdentifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
ComparisonMetricName = Annotated[
    str,
    Field(min_length=1, max_length=256, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*_ms$"),
]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
ComparisonStatistic = Literal["minimum", "p50", "p95", "p99", "maximum", "mean"]


class ComparisonJobCreate(StrictApiModel):
    baseline_source_id: SourceIdentifier
    baseline_source_version: SourceVersion
    candidate_analysis_id: AnalysisIdentifier
    statistic: ComparisonStatistic
    relative_threshold_percent: NonNegativeFiniteFloat
    absolute_threshold_ms: NonNegativeFiniteFloat
    bins: Annotated[int, Field(ge=5, le=100)]
    metric: ComparisonMetricName | None = None


class ComparisonJob(StrictApiModel):
    id: ComparisonIdentifier
    baseline_source_id: SourceIdentifier
    baseline_source_version: SourceVersion
    candidate_analysis_id: AnalysisIdentifier
    statistic: ComparisonStatistic
    relative_threshold_percent: NonNegativeFiniteFloat
    absolute_threshold_ms: NonNegativeFiniteFloat
    bins: Annotated[int, Field(ge=5, le=100)]
    metric: ComparisonMetricName | None = None
    status: Literal["queued", "running", "completed", "failed", "expired"]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    comparison_id: ComparisonIdentifier | None = None
    error: str | None = None
    queue_position: Annotated[int, Field(ge=1)] | None = None
    deduplicated: bool = False


class ComparisonNumericSummary(StrictApiModel):
    count: NonNegativeInt
    minimum: FiniteFloat | None = None
    p50: FiniteFloat | None = None
    p95: FiniteFloat | None = None
    p99: FiniteFloat | None = None
    maximum: FiniteFloat | None = None
    mean: FiniteFloat | None = None


class ComparisonStatisticDelta(StrictApiModel):
    absolute_delta_ms: FiniteFloat | None
    percent_delta: FiniteFloat | None


class ComparisonDeltas(StrictApiModel):
    minimum: ComparisonStatisticDelta
    p50: ComparisonStatisticDelta
    p95: ComparisonStatisticDelta
    p99: ComparisonStatisticDelta
    maximum: ComparisonStatisticDelta
    mean: ComparisonStatisticDelta


class ComparisonMetric(StrictApiModel):
    metric: str
    baseline: ComparisonNumericSummary
    candidate: ComparisonNumericSummary
    baseline_count: NonNegativeInt
    candidate_count: NonNegativeInt
    deltas: ComparisonDeltas
    regression: bool


class SharedHistogramBucket(StrictApiModel):
    index: NonNegativeInt
    start: FiniteFloat
    end: FiniteFloat
    baseline_count: NonNegativeInt
    candidate_count: NonNegativeInt


class SharedHistogram(StrictApiModel):
    metric: str | None
    unit: Literal["ms"]
    minimum: FiniteFloat | None
    maximum: FiniteFloat | None
    bin_count: NonNegativeInt
    baseline_count: NonNegativeInt
    candidate_count: NonNegativeInt
    buckets: list[SharedHistogramBucket]


class FlameDiffEntryResponse(StrictApiModel):
    path_id: str
    parent_path_id: str
    frame: Annotated[list[str], Field(min_length=3, max_length=3)]
    depth: NonNegativeInt
    baseline_value_ns: JsSafeInteger
    candidate_value_ns: JsSafeInteger
    baseline_self_value_ns: JsSafeInteger
    candidate_self_value_ns: JsSafeInteger
    absolute_delta_ns: JsSafeInteger
    percent_delta: FiniteFloat | None
    self_absolute_delta_ns: JsSafeInteger
    self_percent_delta: FiniteFloat | None
    status: Literal["improved", "unchanged", "regressed", "added", "removed"]


class ComparisonResultResponse(StrictApiModel):
    id: ComparisonIdentifier
    job_id: ComparisonIdentifier
    baseline_source_id: SourceIdentifier
    baseline_source_version: SourceVersion
    candidate_analysis_id: AnalysisIdentifier
    created_at: datetime
    statistic: ComparisonStatistic
    relative_threshold_percent: NonNegativeFiniteFloat
    absolute_threshold_ms: NonNegativeFiniteFloat
    metrics: list[ComparisonMetric]
    histogram: SharedHistogram
    flame_diff: list[FlameDiffEntryResponse]
    coverage_warnings: list[str]
    count_warnings: list[str]
    comparable: bool = Field(description="Selected metric data is sufficient for comparison; not business success")
    blocking_reasons: list[str]
    has_regression: bool


class ComparisonSummary(StrictApiModel):
    id: ComparisonIdentifier
    job_id: ComparisonIdentifier
    baseline_source_id: SourceIdentifier
    baseline_source_version: SourceVersion
    candidate_analysis_id: AnalysisIdentifier
    created_at: datetime
    statistic: ComparisonStatistic
    metric: str | None
    comparable: bool = Field(description="Selected metric data is sufficient for comparison; not business success")
    has_regression: bool
    warning_count: NonNegativeInt


class ComparisonListResponse(StrictApiModel):
    items: list[ComparisonSummary]
