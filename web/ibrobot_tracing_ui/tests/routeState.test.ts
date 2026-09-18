import { describe, expect, it } from "vitest";
import { buildProfilerQuery, DETAIL_TABS, parseProfilerRoute } from "../src/utils/routeState";

describe("comparison route state", () => {
  it("keeps Timeline, Span analysis, and Critical Path adjacent", () => {
    const state = parseProfilerRoute({ tab: "critical-path", request: "request 1", component: "worker" });

    expect(DETAIL_TABS.indexOf("critical-path")).toBe(DETAIL_TABS.indexOf("span-profile") + 1);
    expect(state).toMatchObject({ activeTab: "critical-path", requestId: "request 1", componentId: "worker" });
    expect(buildProfilerQuery(state)).toMatchObject({ tab: "critical-path", request: "request 1", component: "worker" });
  });

  it("maps the legacy Call Tree tab into the grouped Span analysis entry", () => {
    const state = parseProfilerRoute({ tab: "call-tree", request: "request-1", orientation: "flame" });

    expect(state).toMatchObject({ activeTab: "span-profile", spanAnalysisMode: "call-tree" });
    expect(buildProfilerQuery(state)).toEqual({ tab: "span-profile", request: "request-1", mode: "call-tree" });
  });

  it("restores comparison controls and metric from a URL query", () => {
    const state = parseProfilerRoute({
      tab: "comparison",
      baseline_source: "a".repeat(32),
      statistic: "p99",
      relative_threshold: "12.5",
      absolute_threshold_ms: "0.75",
      comparison_metric: "inference_ms",
      orientation: "flame",
    });

    expect(state).toMatchObject({
      activeTab: "comparison",
      selectedBaselineSourceId: "a".repeat(32),
      comparisonStatistic: "p99",
      comparisonRelativeThreshold: 12.5,
      comparisonAbsoluteThresholdMs: 0.75,
      comparisonMetric: "inference_ms",
      spanAnalysisMode: "request",
    });
    expect(buildProfilerQuery(state)).toMatchObject({
      tab: "comparison",
      baseline_source: "a".repeat(32),
      statistic: "p99",
      relative_threshold: "12.5",
      absolute_threshold_ms: "0.75",
      comparison_metric: "inference_ms",
    });
  });

  it("uses safe defaults for invalid thresholds, statistics, and metric names", () => {
    const state = parseProfilerRoute({
      statistic: "median",
      relative_threshold: "-2",
      absolute_threshold_ms: "NaN",
      comparison_metric: "/tmp/trace",
    });

    expect(state.comparisonStatistic).toBe("p95");
    expect(state.comparisonRelativeThreshold).toBe(10);
    expect(state.comparisonAbsoluteThresholdMs).toBe(1);
    expect(state.comparisonMetric).toBe("");
    expect(buildProfilerQuery(state)).toEqual({});
  });
});
