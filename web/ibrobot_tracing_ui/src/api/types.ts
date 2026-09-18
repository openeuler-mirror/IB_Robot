export type LoadState = "idle" | "loading" | "ready" | "error";
export type DetailTab = "summary" | "requests" | "distribution" | "comparison" | "timeline" | "span-profile" | "critical-path" | "events" | "spans" | "flows" | "warnings";
export type MetricName = "p50" | "p95" | "p99" | "minimum" | "maximum" | "mean";
export type ComparisonStatistic = MetricName;
export type GraphView = "nodes" | "components" | "tracepoints";
export type SpanProfileMode = "request" | "aggregate";
export type SpanAnalysisMode = "call-tree" | SpanProfileMode;
export type JsSafeInteger = number | string;

export interface CapabilitiesResponse {
  api_version: string;
  authentication: "none";
  tls_enabled: boolean;
  cors_enabled: boolean;
  baseline_compare: boolean;
  source_kinds: string[];
  analysis_views: string[];
  limits: Record<string, number>;
}

export interface ApiList<T> {
  items: T[];
  total: number;
  offset?: number;
  limit?: number;
  next_offset?: number | null;
}

export interface TraceSource {
  id: string;
  name: string;
  kind: "ctf" | "log";
  version: string;
  size_bytes: number;
  file_count: number;
  modified_at: string;
}

export interface SourceListResponse {
  generation: number;
  refreshed_at: string;
  items: TraceSource[];
  warnings: string[];
}

export interface LoadJob {
  id: string;
  source_id: string;
  source_version: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "expired";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  analysis_id: string | null;
  error: string | null;
  cancel_requested: boolean;
  queue_position: number | null;
  deduplicated: boolean;
}

export interface ComparisonJobCreate {
  baseline_source_id: string;
  baseline_source_version: string;
  candidate_analysis_id: string;
  statistic: ComparisonStatistic;
  relative_threshold_percent: number;
  absolute_threshold_ms: number;
  bins: number;
  metric: string | null;
}

export interface ComparisonJob {
  id: string;
  baseline_source_id: string;
  baseline_source_version: string;
  candidate_analysis_id: string;
  statistic: ComparisonStatistic;
  relative_threshold_percent: number;
  absolute_threshold_ms: number;
  bins: number;
  metric: string | null;
  status: "queued" | "running" | "completed" | "failed" | "expired";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  comparison_id: string | null;
  error: string | null;
  queue_position: number | null;
  deduplicated: boolean;
}

export interface NumericSummary {
  count: number;
  minimum: number | null;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  maximum: number | null;
  mean: number | null;
}

export interface ComparisonStatisticDelta {
  absolute_delta_ms: number | null;
  percent_delta: number | null;
}

export interface ComparisonMetricResult {
  metric: string;
  baseline: NumericSummary;
  candidate: NumericSummary;
  baseline_count: number;
  candidate_count: number;
  deltas: Record<ComparisonStatistic, ComparisonStatisticDelta>;
  regression: boolean;
}

export interface SharedHistogramBucket {
  index: number;
  start: number;
  end: number;
  baseline_count: number;
  candidate_count: number;
}

export interface SharedHistogram {
  metric: string | null;
  unit: "ms";
  minimum: number | null;
  maximum: number | null;
  bin_count: number;
  baseline_count: number;
  candidate_count: number;
  buckets: SharedHistogramBucket[];
}

export type FlameDiffStatus = "improved" | "unchanged" | "regressed" | "added" | "removed";

export interface FlameDiffEntry {
  path_id: string;
  parent_path_id: string;
  frame: [string, string, string];
  depth: number;
  baseline_value_ns: JsSafeInteger;
  candidate_value_ns: JsSafeInteger;
  baseline_self_value_ns: JsSafeInteger;
  candidate_self_value_ns: JsSafeInteger;
  absolute_delta_ns: JsSafeInteger;
  percent_delta: number | null;
  self_absolute_delta_ns: JsSafeInteger;
  self_percent_delta: number | null;
  status: FlameDiffStatus;
}

export interface ComparisonResult {
  id: string;
  job_id: string;
  baseline_source_id: string;
  baseline_source_version: string;
  candidate_analysis_id: string;
  created_at: string;
  statistic: ComparisonStatistic;
  relative_threshold_percent: number;
  absolute_threshold_ms: number;
  metrics: ComparisonMetricResult[];
  histogram: SharedHistogram;
  flame_diff: FlameDiffEntry[];
  coverage_warnings: string[];
  count_warnings: string[];
  /** Whether the selected metric has sufficient data, not whether business execution succeeded. */
  comparable: boolean;
  blocking_reasons: string[];
  has_regression: boolean;
}

export interface ComparisonSummary {
  id: string;
  job_id: string;
  baseline_source_id: string;
  baseline_source_version: string;
  candidate_analysis_id: string;
  created_at: string;
  statistic: ComparisonStatistic;
  metric: string | null;
  /** Whether the selected metric has sufficient data, not whether business execution succeeded. */
  comparable: boolean;
  has_regression: boolean;
  warning_count: number;
}

export interface CustomSpanSummary extends NumericSummary {
  component_id: string;
  name: string;
}

export interface SpanSummary extends CustomSpanSummary {
  origin: string;
  status: string;
}

export interface CustomMarkSummary {
  component_id: string;
  name: string;
  count: number;
}

export interface AnalysisSummary {
  analysis: {
    id: string;
    source_id: string;
    source_name: string;
    source_kind: "ctf" | "log";
    source_version: string;
    created_at: string;
    event_count: number;
    request_count: number;
    span_count: number;
    flow_count: number;
    warning_count: number;
  };
  metadata: Record<string, unknown>;
  stages: Record<string, NumericSummary>;
  observations: Record<string, Record<string, number[]>>;
  span_summary: SpanSummary[];
  span_summary_total: number;
  span_summary_limit: number;
  span_summary_truncated: boolean;
  custom_span_summary: CustomSpanSummary[];
  custom_mark_summary: CustomMarkSummary[];
  coverage: {
    boundary?: string;
    expected_metrics?: number;
    observed_metrics?: number;
    missing_events?: string[];
    todos?: string[];
  };
}

export interface RequestRecord extends Record<string, string | number | null | undefined> {
  request_id: string;
  [key: `${string}_ms_status`]: string | undefined;
}

export interface LatencyDistributionBucket {
  index: number;
  start: number;
  end: number;
  count: number;
  request_ids: string[];
  returned: number;
  truncated: boolean;
}

export interface LatencyDistributionOutlier {
  request_id: string;
  value: number;
  rank: number;
  percentile: number;
  p95_tail: boolean;
}

export interface LatencyDistributionResponse {
  metric: string | null;
  unit: "ms";
  available_metrics: string[];
  summary: NumericSummary;
  sample_count: number;
  invalid_count: number;
  minimum: number | null;
  maximum: number | null;
  bin_count: number;
  buckets: LatencyDistributionBucket[];
  outliers: LatencyDistributionOutlier[];
}

export interface LatencyDistributionFilters {
  metric?: string;
  bins?: number;
  outlierLimit?: number;
  bucketRequestLimit?: number;
}

export interface EventOrigin {
  source: string;
  provider: string;
  host: string;
  process_id: number | null;
  thread_id: number | null;
  node: string;
}

export interface TraceEvent {
  id: string;
  timestamp_ns: string | number;
  name: string;
  fields: Record<string, unknown>;
  origin: EventOrigin;
  clock: string;
  schema_version: number;
  sequence: number;
  request_id: string;
  component_id: string;
}

export interface TracepointDefinition {
  id: string;
  kind: "event" | "span";
  component_id: string;
  name: string;
  origin: string;
  description: string;
}

export interface SpanRecord {
  id: string;
  name: string;
  trace_id: string;
  span_id: string;
  parent_span_id: string;
  component_id: string;
  start_ns: string | number;
  end_ns: string | number | null;
  duration_ms: number | null;
  status: string;
  origin: string;
  fields: Record<string, unknown>;
}

export interface FlowRecord {
  id: string;
  edge_id: string;
  flow_id: string;
  trace_id: string;
  send_ns: string | number | null;
  receive_ns: string | number | null;
  duration_ms: number | null;
  status: string;
}

export interface GraphNode {
  id: string;
  component_id: string;
  name: string;
  kind: string;
  parent_id: string;
  child_ids: string[];
  depth: number;
  provenance: string;
  trace_name: string;
  metrics: Record<string, unknown>;
  metric_value_ms: number | null;
  description: string;
}

export interface GraphEdge {
  id: string;
  edge_id: string;
  source_id: string;
  target_id: string;
  name: string;
  kind: string;
  data_type?: string;
  contract_key?: string;
  provenance: string;
  metrics: Record<string, unknown>;
  metric_value_ms: number | null;
  directed: boolean;
}

export interface TraceTopology {
  metric: MetricName;
  view: GraphView;
  request_id: string;
  topology_source: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface TimelineLane {
  id: string;
  label: string;
  kind: string;
  component_id: string;
  edge_id: string;
}

export interface TimelineItem {
  id: string;
  lane_id: string;
  kind: "event" | "span" | "flow";
  label: string;
  start_ns: string | number;
  end_ns: string | number;
  start_offset_ns: number;
  end_offset_ns: number;
  duration_ns: number;
  request_id: string;
  component_id: string;
  status: string;
  fields: Record<string, unknown>;
}

export interface TimelineProjection {
  start_ns: string | number | null;
  end_ns: string | number | null;
  duration_ns: number;
  lanes: TimelineLane[];
  items: TimelineItem[];
}

export interface CallTreeNode {
  id: string;
  span_id: string;
  request_id: string;
  parent_id: string;
  parent_span_id: string;
  child_ids: string[];
  name: string;
  component_id: string;
  origin: string;
  status: string;
  start_ns: string | number;
  end_ns: string | number | null;
  start_offset_ns: number;
  duration_ns: number | null;
  self_duration_ns: number | null;
  orphan: boolean;
  cycle: boolean;
  fields: Record<string, unknown>;
}

export interface CallTreeProjection {
  start_ns: string | number | null;
  root_ids: string[];
  nodes: CallTreeNode[];
  warnings: string[];
}

export interface SpanProfileDiagnostic {
  code: string;
  occurrence_ids: string[];
  message: string;
}

export interface SpanProfileTotals {
  request_count: number;
  occurrence_count: number;
  error_count: number;
  incomplete_count: number;
  invalid_count: number;
  excluded_weight_count: number;
  sampled_uncovered_wall_ns: JsSafeInteger;
  uncovered_wall_ns: JsSafeInteger;
  selected_wall_union_ns: JsSafeInteger;
  concurrency_factor: number;
  diagnostic_counts: Record<string, number>;
}

export interface RequestSpanProfileNode {
  id: string;
  occurrence_id: string;
  span_id: string;
  request_id: string;
  parent_id: string;
  parent_span_id: string;
  child_ids: string[];
  depth: number;
  track: number;
  component_id: string;
  name: string;
  origin: string;
  status: string;
  start_ns: string;
  end_ns: string | null;
  observed_end_ns: string | null;
  start_offset_ns: JsSafeInteger;
  duration_ns: JsSafeInteger | null;
  duration_source: string;
  uncovered_wall_ns: JsSafeInteger | null;
  diagnostics: string[];
  fields: Record<string, unknown>;
}

export interface AggregateSpanProfilePathItem {
  component_id: string;
  name: string;
  origin: string;
}

export interface AggregateSpanProfileNode {
  id: string;
  parent_id: string;
  child_ids: string[];
  depth: number;
  component_id: string;
  name: string;
  origin: string;
  path: AggregateSpanProfilePathItem[];
  self_value_ns: JsSafeInteger;
  value_ns: JsSafeInteger;
  occurrence_count: number;
  request_count: number;
  error_count: number;
  incomplete_count: number;
  invalid_count: number;
  percentage: number;
}

interface SpanProfileResponseBase {
  measurement: "instrumented_wall";
  source: string;
  coverage: string;
  not_cpu: boolean;
  roots: string[];
  diagnostics: SpanProfileDiagnostic[];
  total_nodes: number;
  returned_nodes: number;
  truncated: boolean;
  truncation_reason: string;
}

export interface RequestSpanProfileResponse extends SpanProfileResponseBase {
  mode: "request";
  request_id: string;
  start_ns: string | null;
  end_ns: string | null;
  duration_ns: JsSafeInteger;
  nodes: RequestSpanProfileNode[];
  totals: SpanProfileTotals;
}

export interface AggregateSpanProfileResponse extends SpanProfileResponseBase {
  mode: "aggregate";
  nodes: AggregateSpanProfileNode[];
  totals: SpanProfileTotals & { value_ns: JsSafeInteger };
}

export type SpanProfileResponse = RequestSpanProfileResponse | AggregateSpanProfileResponse;

interface SpanProfileFiltersBase {
  componentId?: string;
  startNs?: JsSafeInteger;
  endNs?: JsSafeInteger;
  origin?: string;
  status?: string;
  maxNodes?: number;
}

export type SpanProfileFilters =
  | (SpanProfileFiltersBase & { mode: "request"; requestId: string })
  | (SpanProfileFiltersBase & { mode: "aggregate"; requestId?: never });

export interface CriticalPathDiagnostic {
  code: string;
  count: number;
  source_ids: string[];
  message: string;
}

export interface CriticalPathSegment {
  id: string;
  index: number;
  kind: "span" | "flow" | "unattributed";
  source_id: string;
  label: string;
  component_id: string;
  edge_id: string;
  origin: string;
  status: string;
  start_ns: JsSafeInteger;
  end_ns: JsSafeInteger;
  offset_ns: JsSafeInteger;
  duration_ns: JsSafeInteger;
  percentage: number;
  diagnostics: string[];
  fields: Record<string, unknown>;
}

export interface CriticalPathBottleneck {
  rank: number;
  kind: "span" | "flow";
  source_id: string;
  label: string;
  component_id: string;
  edge_id: string;
  origin: string;
  status: string;
  duration_ns: JsSafeInteger;
  percentage: number;
  segment_count: number;
}

export interface CriticalPathTotals {
  partition_ns: JsSafeInteger;
  attributed_ns: JsSafeInteger;
  unattributed_ns: JsSafeInteger;
  coverage_percent: number;
  by_kind_ns: Record<string, JsSafeInteger>;
  by_component_ns: Record<string, JsSafeInteger>;
  by_edge_ns: Record<string, JsSafeInteger>;
  record_counts: Record<string, number>;
  diagnostic_counts: Record<string, number>;
}

export interface CriticalPathResponse {
  method: "deepest_active_wall_partition";
  measurement: "instrumented_wall";
  not_cpu: boolean;
  request_id: string;
  component_id: string;
  include_flows: boolean;
  boundary_source: string;
  boundary_start_source: string;
  boundary_end_source: string;
  start_ns: JsSafeInteger | null;
  end_ns: JsSafeInteger | null;
  duration_ns: JsSafeInteger;
  segments: CriticalPathSegment[];
  totals: CriticalPathTotals;
  bottlenecks: CriticalPathBottleneck[];
  diagnostics: CriticalPathDiagnostic[];
  total_segments: number;
  returned_segments: number;
  returned_duration_ns: JsSafeInteger;
  omitted_segments: number;
  omitted_duration_ns: JsSafeInteger;
  truncated: boolean;
  truncation_reason: string;
}

export interface CriticalPathFilters {
  requestId: string;
  componentId?: string;
  startNs?: JsSafeInteger;
  endNs?: JsSafeInteger;
  includeFlows?: boolean;
  maxSegments?: number;
}

export type SelectionKind = "request" | "event" | "span" | "flow" | "critical-path" | "component" | "edge" | "warning" | "comparison";

export interface InspectorSelection {
  kind: SelectionKind;
  title: string;
  subtitle?: string;
  data: Record<string, unknown>;
}
