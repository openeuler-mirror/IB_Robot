import type {
  ComparisonStatistic,
  DetailTab,
  GraphView,
  MetricName,
  SpanAnalysisMode,
} from "../api/types";

export const DETAIL_TABS: DetailTab[] = [
  "summary",
  "requests",
  "distribution",
  "comparison",
  "timeline",
  "span-profile",
  "critical-path",
  "events",
  "spans",
  "flows",
  "warnings",
];

const METRICS: MetricName[] = ["p50", "p95", "p99", "minimum", "maximum", "mean"];
const GRAPH_VIEWS: GraphView[] = ["nodes", "components", "tracepoints"];
const SPAN_ANALYSIS_MODES: SpanAnalysisMode[] = ["call-tree", "request", "aggregate"];
const COMPARISON_METRIC = /^[A-Za-z_][A-Za-z0-9_.-]*_ms$/;

export interface ProfilerRouteState {
  activeTab: DetailTab;
  requestId: string;
  componentId: string;
  distributionMetric: string;
  metric: MetricName;
  graphView: GraphView;
  spanAnalysisMode: SpanAnalysisMode;
  searchQuery: string;
  selectedBaselineSourceId: string;
  comparisonStatistic: ComparisonStatistic;
  comparisonRelativeThreshold: number;
  comparisonAbsoluteThresholdMs: number;
  comparisonMetric: string;
}

function value(input: unknown): string {
  return Array.isArray(input) ? String(input[0] ?? "") : String(input ?? "");
}

function nonNegativeNumber(input: unknown, fallback: number): number {
  const text = value(input).trim();
  if (!text) return fallback;
  const number = Number(text);
  return Number.isFinite(number) && number >= 0 ? number : fallback;
}

export function parseProfilerRoute(query: Record<string, unknown>): ProfilerRouteState {
  const requestedTab = value(query.tab);
  const tab = (requestedTab === "call-tree" ? "span-profile" : requestedTab) as DetailTab;
  const metric = value(query.metric) as MetricName;
  const graphView = value(query.view) as GraphView;
  const requestedSpanMode = (requestedTab === "call-tree" ? "call-tree" : value(query.mode)) as SpanAnalysisMode;
  const statistic = value(query.statistic) as ComparisonStatistic;
  const comparisonMetric = value(query.comparison_metric).trim();
  return {
    activeTab: DETAIL_TABS.includes(tab) ? tab : "summary",
    requestId: value(query.request),
    componentId: value(query.component),
    distributionMetric: value(query.distribution_metric),
    metric: METRICS.includes(metric) ? metric : "p95",
    graphView: GRAPH_VIEWS.includes(graphView) ? graphView : "components",
    spanAnalysisMode: SPAN_ANALYSIS_MODES.includes(requestedSpanMode) ? requestedSpanMode : "request",
    searchQuery: value(query.q),
    selectedBaselineSourceId: value(query.baseline_source),
    comparisonStatistic: METRICS.includes(statistic) ? statistic : "p95",
    comparisonRelativeThreshold: nonNegativeNumber(query.relative_threshold, 10),
    comparisonAbsoluteThresholdMs: nonNegativeNumber(query.absolute_threshold_ms, 1),
    comparisonMetric: COMPARISON_METRIC.test(comparisonMetric) ? comparisonMetric : "",
  };
}

export function buildProfilerQuery(state: ProfilerRouteState): Record<string, string> {
  const query: Record<string, string> = {};
  if (state.activeTab !== "summary") query.tab = state.activeTab;
  if (state.requestId) query.request = state.requestId;
  if (state.componentId) query.component = state.componentId;
  if (state.distributionMetric) query.distribution_metric = state.distributionMetric;
  if (state.metric !== "p95") query.metric = state.metric;
  if (state.graphView !== "components") query.view = state.graphView;
  if (state.spanAnalysisMode !== "request") query.mode = state.spanAnalysisMode;
  if (state.searchQuery) query.q = state.searchQuery;
  if (state.selectedBaselineSourceId) query.baseline_source = state.selectedBaselineSourceId;
  if (state.comparisonStatistic !== "p95") query.statistic = state.comparisonStatistic;
  if (state.comparisonRelativeThreshold !== 10) query.relative_threshold = String(state.comparisonRelativeThreshold);
  if (state.comparisonAbsoluteThresholdMs !== 1) query.absolute_threshold_ms = String(state.comparisonAbsoluteThresholdMs);
  if (state.comparisonMetric) query.comparison_metric = state.comparisonMetric;
  return query;
}
