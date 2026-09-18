import { createPinia, setActivePinia } from "pinia";
import { nextTick, watch } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { tracingApi } from "../src/api/client";
import type { AnalysisSummary, ApiList, CriticalPathResponse, LatencyDistributionResponse, SpanProfileResponse, TracepointDefinition, TraceTopology } from "../src/api/types";
import { useProfilerStore } from "../src/stores/profiler";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function summary(id: string): AnalysisSummary {
  return {
    analysis: { id, source_id: `source-${id}` },
  } as AnalysisSummary;
}

function topology(id: string) {
  return { metric: "p95", view: "components", request_id: "", topology_source: "declared", nodes: [], edges: [], id } as any;
}

function tracepoint(name: string): TracepointDefinition {
  return {
    id: `tracepoint:event:worker:${name}:user`,
    kind: "event",
    component_id: "worker",
    name,
    origin: "user",
    description: `${name} description`,
  };
}

function spanProfile(mode: "request" | "aggregate", name: string = mode): SpanProfileResponse {
  const base = {
    measurement: "instrumented_wall" as const,
    source: "paired_spans",
    coverage: "selected",
    not_cpu: true as const,
    roots: [name],
    diagnostics: [],
    total_nodes: 1,
    returned_nodes: 1,
    truncated: false,
    truncation_reason: "",
    totals: {
      request_count: 1,
      occurrence_count: 1,
      error_count: 0,
      incomplete_count: 0,
      invalid_count: 0,
      excluded_weight_count: 0,
      sampled_uncovered_wall_ns: 10,
      uncovered_wall_ns: 10,
      selected_wall_union_ns: 10,
      concurrency_factor: 1,
      diagnostic_counts: {},
    },
  };
  if (mode === "request") {
    return {
      ...base,
      mode,
      request_id: "request",
      start_ns: "0",
      end_ns: "10",
      duration_ns: 10,
      nodes: [{
        id: name,
        occurrence_id: name,
        span_id: name,
        request_id: "request",
        parent_id: "",
        parent_span_id: "",
        child_ids: [],
        depth: 0,
        track: 0,
        component_id: "worker",
        name,
        origin: "user",
        status: "ok",
        start_ns: "0",
        end_ns: "10",
        observed_end_ns: "10",
        start_offset_ns: 0,
        duration_ns: 10,
        duration_source: "observed",
        uncovered_wall_ns: 10,
        diagnostics: [],
        fields: {},
      }],
    };
  }
  return {
    ...base,
    mode,
    totals: { ...base.totals, value_ns: 10 },
    nodes: [{
      id: name,
      parent_id: "",
      child_ids: [],
      depth: 0,
      component_id: "worker",
      name,
      origin: "user",
      path: [{ component_id: "worker", name, origin: "user" }],
      self_value_ns: 10,
      value_ns: 10,
      occurrence_count: 1,
      request_count: 1,
      error_count: 0,
      incomplete_count: 0,
      invalid_count: 0,
      percentage: 100,
    }],
  };
}

function latencyDistribution(metric: string, value: number): LatencyDistributionResponse {
  return {
    metric,
    unit: "ms",
    available_metrics: ["inference_ms", "total_ms"],
    summary: { count: 1, minimum: value, p50: value, p95: value, p99: value, maximum: value, mean: value },
    sample_count: 1,
    invalid_count: 0,
    minimum: value,
    maximum: value,
    bin_count: 1,
    buckets: [{ index: 0, start: value, end: value, count: 1, request_ids: [metric], returned: 1, truncated: false }],
    outliers: [{ request_id: metric, value, rank: 1, percentile: 100, p95_tail: true }],
  };
}

function criticalPath(requestId: string, componentId: string, label: string): CriticalPathResponse {
  return {
    method: "deepest_active_wall_partition",
    measurement: "instrumented_wall",
    not_cpu: true,
    request_id: requestId,
    component_id: componentId,
    include_flows: true,
    boundary_source: "dispatch_request_to_first_action_execute",
    boundary_start_source: "event:dispatch_request",
    boundary_end_source: "event:first_action_execute",
    start_ns: "9007199254741000",
    end_ns: "9007199254741010",
    duration_ns: 10,
    segments: [{
      id: label,
      index: 0,
      kind: "span",
      source_id: label,
      label,
      component_id: componentId,
      edge_id: "",
      origin: "user",
      status: "ok",
      start_ns: "9007199254741000",
      end_ns: "9007199254741010",
      offset_ns: 0,
      duration_ns: 10,
      percentage: 100,
      diagnostics: [],
      fields: {},
    }],
    totals: {
      partition_ns: 10,
      attributed_ns: 10,
      unattributed_ns: 0,
      coverage_percent: 100,
      by_kind_ns: { span: 10, flow: 0, unattributed: 0 },
      by_component_ns: { [componentId]: 10 },
      by_edge_ns: {},
      record_counts: { valid_spans: 1 },
      diagnostic_counts: {},
    },
    bottlenecks: [{
      rank: 1,
      kind: "span",
      source_id: label,
      label,
      component_id: componentId,
      edge_id: "",
      origin: "user",
      status: "ok",
      duration_ns: 10,
      percentage: 100,
      segment_count: 1,
    }],
    diagnostics: [],
    total_segments: 1,
    returned_segments: 1,
    returned_duration_ns: 10,
    omitted_segments: 0,
    omitted_duration_ns: 0,
    truncated: false,
    truncation_reason: "",
  };
}

describe("profiler store", () => {
  beforeEach(() => setActivePinia(createPinia()));
  afterEach(() => vi.restoreAllMocks());

  it("ignores a stale tab response after the scope changes", async () => {
    const first = deferred<any>();
    const second = deferred<any>();
    vi.spyOn(tracingApi, "events").mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.activeTab = "events";
    store.requestId = "old";

    const oldRequest = store.fetchTab("events", true);
    store.requestId = "new";
    const newRequest = store.refreshScopedData();
    second.resolve({ items: [{ id: "new" }], total: 1 });
    await newRequest;
    first.resolve({ items: [{ id: "old" }], total: 1 });
    await oldRequest;

    expect(store.events).toEqual([{ id: "new" }]);
    expect(store.tabState.events).toBe("ready");
  });

  it("stores a backend-selected default distribution metric and reuses its cache", async () => {
    const api = vi.spyOn(tracingApi, "distribution").mockResolvedValue(latencyDistribution("inference_ms", 8));
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.activeTab = "distribution";

    await store.fetchTab("distribution");
    await store.fetchTab("distribution");
    await store.setDistributionMetric("");

    expect(api).toHaveBeenCalledTimes(1);
    expect(api).toHaveBeenCalledWith("analysis", { metric: undefined }, expect.any(AbortSignal));
    expect(store.distributionMetric).toBe("inference_ms");
    expect(store.distribution?.metric).toBe("inference_ms");
    expect(store.tabState.distribution).toBe("ready");
  });

  it("aborts and ignores a stale distribution response after a metric change", async () => {
    const oldDistribution = deferred<LatencyDistributionResponse>();
    const newDistribution = deferred<LatencyDistributionResponse>();
    const api = vi.spyOn(tracingApi, "distribution")
      .mockReturnValueOnce(oldDistribution.promise)
      .mockReturnValueOnce(newDistribution.promise);
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.activeTab = "distribution";
    store.distributionMetric = "total_ms";

    const stale = store.fetchTab("distribution", true);
    const oldSignal = api.mock.calls[0][2];
    const changed = store.setDistributionMetric("inference_ms");
    newDistribution.resolve(latencyDistribution("inference_ms", 4));
    await changed;
    oldDistribution.resolve(latencyDistribution("total_ms", 99));
    await stale;

    expect(oldSignal?.aborted).toBe(true);
    expect(store.distributionMetric).toBe("inference_ms");
    expect(store.distribution?.maximum).toBe(4);
    expect(store.tabState.distribution).toBe("ready");
  });

  it("sets an exact request scope and safely opens every request detail view", async () => {
    const timeline = vi.spyOn(tracingApi, "timeline").mockResolvedValue({
      start_ns: null,
      end_ns: null,
      duration_ns: 0,
      lanes: [],
      items: [],
    });
    const callTree = vi.spyOn(tracingApi, "callTree").mockResolvedValue({ start_ns: null, root_ids: [], nodes: [], warnings: [] });
    const profile = vi.spyOn(tracingApi, "spanProfile").mockResolvedValue(spanProfile("request"));
    const path = vi.spyOn(tracingApi, "criticalPath").mockResolvedValue(criticalPath("slow", "", "path"));
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.activeTab = "distribution";
    store.requestId = "slow";
    store.componentId = "worker";
    store.spanAnalysisMode = "aggregate";

    await store.navigateToRequest("slow", "timeline");
    expect(store.requestId).toBe("slow");
    expect(store.componentId).toBe("");
    expect(store.activeTab).toBe("timeline");
    expect(timeline).toHaveBeenCalledWith("analysis", { requestId: "slow", componentId: "" }, expect.any(AbortSignal));

    store.componentId = "worker";
    await store.navigateToRequest("slow", "span-profile");
    expect(store.requestId).toBe("slow");
    expect(store.componentId).toBe("");
    expect(store.spanAnalysisMode).toBe("request");
    expect(store.activeTab).toBe("span-profile");
    expect(profile).toHaveBeenCalledWith(
      "analysis",
      { mode: "request", requestId: "slow", componentId: "", maxNodes: 3_000 },
      expect.any(AbortSignal),
    );

    await store.setSpanAnalysisMode("call-tree");
    expect(store.spanAnalysisMode).toBe("call-tree");
    expect(callTree).toHaveBeenCalledWith("analysis", { requestId: "slow", componentId: "" }, expect.any(AbortSignal));

    store.componentId = "worker";
    await store.navigateToRequest("slow", "critical-path");
    expect(store.requestId).toBe("slow");
    expect(store.componentId).toBe("");
    expect(store.activeTab).toBe("critical-path");
    expect(path).toHaveBeenCalledWith(
      "analysis",
      { requestId: "slow", componentId: "", includeFlows: true, maxSegments: 1_000 },
      expect.any(AbortSignal),
    );
  });

  it("does not request a critical path without both a ready analysis and request", async () => {
    const api = vi.spyOn(tracingApi, "criticalPath");
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.activeTab = "critical-path";
    store.requestId = "request";

    await store.fetchTab("critical-path", true);
    store.analysisState = "ready";
    store.requestId = "";
    await store.fetchTab("critical-path", true);

    expect(api).not.toHaveBeenCalled();
    expect(store.criticalPath).toBeNull();
    expect(store.tabState["critical-path"]).toBe("ready");
  });

  it("aborts a stale critical path and applies the latest request and component scope", async () => {
    const oldPath = deferred<CriticalPathResponse>();
    const newPath = deferred<CriticalPathResponse>();
    const api = vi.spyOn(tracingApi, "criticalPath")
      .mockReturnValueOnce(oldPath.promise)
      .mockReturnValueOnce(newPath.promise);
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.activeTab = "critical-path";
    store.requestId = "old-request";
    store.componentId = "old-component";

    const stale = store.fetchTab("critical-path", true);
    const oldSignal = api.mock.calls[0][2];
    store.requestId = "new-request";
    store.componentId = "new-component";
    const current = store.refreshScopedData();
    newPath.resolve(criticalPath("new-request", "new-component", "new"));
    await current;
    oldPath.resolve(criticalPath("old-request", "old-component", "old"));
    await stale;

    expect(oldSignal?.aborted).toBe(true);
    expect(api.mock.calls[1][1]).toEqual({
      requestId: "new-request",
      componentId: "new-component",
      includeFlows: true,
      maxSegments: 1_000,
    });
    expect(store.criticalPath?.request_id).toBe("new-request");
    expect(store.criticalPath?.component_id).toBe("new-component");
    expect(store.criticalPath?.segments[0].label).toBe("new");
    expect(store.tabState["critical-path"]).toBe("ready");
  });

  it("shows request-mode prompt state without issuing an invalid profile call", async () => {
    const api = vi.spyOn(tracingApi, "spanProfile");
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.activeTab = "span-profile";

    await store.fetchTab("span-profile", true);

    expect(api).not.toHaveBeenCalled();
    expect(store.spanProfile).toBeNull();
    expect(store.tabState["span-profile"]).toBe("ready");
  });

  it("loads a selected request profile with the scoped cache inputs", async () => {
    const api = vi.spyOn(tracingApi, "spanProfile").mockResolvedValue(spanProfile("request"));
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.requestId = "request";
    store.componentId = "worker";

    await store.fetchTab("span-profile", true);

    expect(api).toHaveBeenCalledWith(
      "analysis",
      { mode: "request", requestId: "request", componentId: "worker", maxNodes: 3_000 },
      expect.any(AbortSignal),
    );
    expect(store.spanProfile?.mode).toBe("request");
  });

  it("loads aggregate mode without requiring or forwarding a request", async () => {
    const api = vi.spyOn(tracingApi, "spanProfile").mockResolvedValue(spanProfile("aggregate"));
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.spanAnalysisMode = "aggregate";

    await store.fetchTab("span-profile", true);

    expect(api).toHaveBeenCalledWith(
      "analysis",
      { mode: "aggregate", componentId: "", maxNodes: 3_000 },
      expect.any(AbortSignal),
    );
    expect(store.spanProfile?.mode).toBe("aggregate");
  });

  it("aborts and ignores a stale request profile when mode changes", async () => {
    const oldProfile = deferred<SpanProfileResponse>();
    const aggregateProfile = deferred<SpanProfileResponse>();
    const api = vi.spyOn(tracingApi, "spanProfile")
      .mockReturnValueOnce(oldProfile.promise)
      .mockReturnValueOnce(aggregateProfile.promise);
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.activeTab = "span-profile";
    store.requestId = "request";

    const stale = store.fetchTab("span-profile", true);
    const oldSignal = api.mock.calls[0][2];
    const changed = store.setSpanAnalysisMode("aggregate");
    aggregateProfile.resolve(spanProfile("aggregate", "new"));
    await changed;
    oldProfile.resolve(spanProfile("request", "old"));
    await stale;

    expect(oldSignal?.aborted).toBe(true);
    expect(store.spanProfile?.mode).toBe("aggregate");
    expect(store.spanProfile?.nodes[0].name).toBe("new");
    expect(store.tabState["span-profile"]).toBe("ready");
  });

  it("clears a span inspector selection when profile mode changes", async () => {
    const store = useProfilerStore();
    store.selection = { kind: "span", title: "old", data: { occurrence_id: "old" } };

    await store.setSpanAnalysisMode("aggregate");

    expect(store.selection).toBeNull();
  });

  it("scopes an operation node to its owning component", () => {
    const store = useProfilerStore();
    store.topology = {
      nodes: [
        {
          component_id: "worker.operation.work",
          parent_id: "worker",
          kind: "operation",
          name: "Work",
        },
      ],
      edges: [],
    } as any;

    store.selectComponent("worker.operation.work");

    expect(store.componentId).toBe("worker");
    expect(store.selection?.title).toBe("Work");
  });

  it("loads every tracepoint page and clears the catalog on analysis change", async () => {
    const secondCatalog = deferred<ApiList<TracepointDefinition>>();
    vi.spyOn(tracingApi, "summary").mockImplementation(async (id) => summary(id));
    vi.spyOn(tracingApi, "requests").mockResolvedValue({ items: [], total: 0 });
    vi.spyOn(tracingApi, "topology").mockImplementation(async (id) => topology(id));
    const catalog = vi.spyOn(tracingApi, "tracepoints").mockImplementation(async (id, pagination) => {
      if (id === "second") return secondCatalog.promise;
      if (pagination?.offset === 0) {
        return { items: [tracepoint("ready")], total: 2, offset: 0, limit: 1000, next_offset: 1 };
      }
      return { items: [tracepoint("work")], total: 2, offset: 1, limit: 1000, next_offset: null };
    });
    const store = useProfilerStore();

    await store.openAnalysis("first");
    expect(store.tracepoints.map((item) => item.name)).toEqual(["ready", "work"]);
    expect(catalog.mock.calls.slice(0, 2).map((call) => call[1]?.offset)).toEqual([0, 1]);

    const pending = store.openAnalysis("second");
    expect(store.tracepoints).toEqual([]);
    secondCatalog.resolve({ items: [tracepoint("second")], total: 1, offset: 0, limit: 1000, next_offset: null });
    await pending;

    expect(store.tracepoints.map((item) => item.name)).toEqual(["second"]);
    expect(catalog).toHaveBeenCalledTimes(3);
  });

  it("does not apply a stale analysis catalog", async () => {
    const oldSummary = deferred<AnalysisSummary>();
    let oldSignal: AbortSignal | undefined;
    vi.spyOn(tracingApi, "summary").mockImplementation((id, signal) => {
      if (id === "old") {
        oldSignal = signal;
        return oldSummary.promise;
      }
      return Promise.resolve(summary(id));
    });
    vi.spyOn(tracingApi, "requests").mockResolvedValue({ items: [], total: 0 });
    vi.spyOn(tracingApi, "topology").mockImplementation(async (id) => topology(id));
    vi.spyOn(tracingApi, "tracepoints").mockImplementation(async (id) => ({
      items: [tracepoint(id)],
      total: 1,
      offset: 0,
      limit: 1000,
      next_offset: null,
    }));
    const store = useProfilerStore();

    const stale = store.openAnalysis("old");
    await store.openAnalysis("new");
    oldSummary.resolve(summary("old"));
    await stale;

    expect(oldSignal?.aborted).toBe(true);
    expect(store.analysisId).toBe("new");
    expect(store.tracepoints.map((item) => item.name)).toEqual(["new"]);
  });

  it.each(["metric", "graphView", "requestId", "all"])("requeries initial topology with current parameters after %s changes while loading", async (change) => {
    const initialTopology = deferred<TraceTopology>();
    const currentTopology = deferred<TraceTopology>();
    vi.spyOn(tracingApi, "summary").mockResolvedValue(summary("analysis"));
    vi.spyOn(tracingApi, "requests").mockResolvedValue({ items: [], total: 0 });
    vi.spyOn(tracingApi, "tracepoints").mockResolvedValue({ items: [], total: 0 });
    const api = vi.spyOn(tracingApi, "topology")
      .mockReturnValueOnce(initialTopology.promise)
      .mockReturnValueOnce(currentTopology.promise);
    const store = useProfilerStore();
    const applied: TraceTopology[] = [];
    const stopApplied = watch(() => store.topology, (value) => {
      if (value) applied.push(value);
    }, { flush: "sync" });
    const stopRefresh = watch(
      [() => store.metric, () => store.graphView, () => store.requestId],
      () => store.refreshTopology(),
    );

    try {
      const loading = store.openAnalysis("analysis");
      if (change === "metric" || change === "all") store.metric = "mean";
      if (change === "graphView" || change === "all") store.graphView = "tracepoints";
      if (change === "requestId" || change === "all") store.requestId = "request-2";
      await nextTick();

      expect(store.analysisState).toBe("loading");
      expect(api).toHaveBeenCalledTimes(1);
      const expected = {
        ...topology("analysis"),
        metric: store.metric,
        view: store.graphView,
        request_id: store.requestId,
      };
      initialTopology.resolve(topology("analysis"));
      currentTopology.resolve(expected);
      await loading;
      await nextTick();

      expect(api).toHaveBeenCalledTimes(2);
      expect(api.mock.calls[1][1]).toEqual({ metric: store.metric, view: store.graphView, requestId: store.requestId });
      expect(applied).toEqual([expected]);
      expect(store.topology).toEqual(expected);
      expect(store.analysisState).toBe("ready");
    } finally {
      stopRefresh();
      stopApplied();
    }
  });

  it("does not duplicate the initial topology request when its parameters are still current", async () => {
    const initialTopology = deferred<TraceTopology>();
    vi.spyOn(tracingApi, "summary").mockResolvedValue(summary("analysis"));
    vi.spyOn(tracingApi, "requests").mockResolvedValue({ items: [], total: 0 });
    vi.spyOn(tracingApi, "tracepoints").mockResolvedValue({ items: [], total: 0 });
    const api = vi.spyOn(tracingApi, "topology").mockReturnValue(initialTopology.promise);
    const store = useProfilerStore();
    const stopRefresh = watch(
      [() => store.metric, () => store.graphView, () => store.requestId],
      () => store.refreshTopology(),
    );

    try {
      const loading = store.openAnalysis("analysis");
      store.metric = "mean";
      await nextTick();
      store.metric = "p95";
      await nextTick();
      initialTopology.resolve(topology("analysis"));
      await loading;
      await nextTick();

      expect(api).toHaveBeenCalledTimes(1);
      expect(store.topology?.metric).toBe("p95");
      expect(store.analysisState).toBe("ready");
    } finally {
      stopRefresh();
    }
  });

  it("aborts a superseded recovery request without applying stale topology or duplicating calls", async () => {
    const initialTopology = deferred<TraceTopology>();
    const recoveryTopology = deferred<TraceTopology>();
    const latestTopology = { ...topology("analysis"), metric: "p99", view: "nodes", request_id: "latest" };
    vi.spyOn(tracingApi, "summary").mockResolvedValue(summary("analysis"));
    vi.spyOn(tracingApi, "requests").mockResolvedValue({ items: [], total: 0 });
    vi.spyOn(tracingApi, "tracepoints").mockResolvedValue({ items: [], total: 0 });
    const api = vi.spyOn(tracingApi, "topology")
      .mockReturnValueOnce(initialTopology.promise)
      .mockReturnValueOnce(recoveryTopology.promise)
      .mockResolvedValueOnce(latestTopology);
    const store = useProfilerStore();
    const applied: TraceTopology[] = [];
    const stopApplied = watch(() => store.topology, (value) => {
      if (value) applied.push(value);
    }, { flush: "sync" });
    const stopRefresh = watch(
      [() => store.metric, () => store.graphView, () => store.requestId],
      () => store.refreshTopology(),
    );

    try {
      const loading = store.openAnalysis("analysis");
      store.metric = "mean";
      await nextTick();
      initialTopology.resolve(topology("analysis"));
      await vi.waitFor(() => expect(api).toHaveBeenCalledTimes(2));
      expect(store.analysisState).toBe("ready");
      expect(store.topology).toBeNull();

      store.metric = "p99";
      store.graphView = "nodes";
      store.requestId = "latest";
      await nextTick();
      expect(api.mock.calls[1][2]?.aborted).toBe(true);
      recoveryTopology.resolve({ ...topology("analysis"), metric: "mean" });
      await loading;
      await nextTick();

      expect(api).toHaveBeenCalledTimes(3);
      expect(api.mock.calls[2][1]).toEqual({ metric: "p99", view: "nodes", requestId: "latest" });
      expect(applied).toEqual([latestTopology]);
      expect(store.topology).toEqual(latestTopology);
      expect(store.analysisState).toBe("ready");
      expect(store.analysisError).toBe("");
    } finally {
      stopRefresh();
      stopApplied();
    }
  });

  it("refreshes an open component inspector with the latest topology data", async () => {
    vi.spyOn(tracingApi, "topology").mockResolvedValue({
      metric: "mean",
      view: "components",
      request_id: "",
      topology_source: "declared",
      nodes: [
        {
          id: "worker",
          component_id: "worker",
          name: "Worker",
          kind: "module",
          parent_id: "",
          child_ids: [],
          depth: 0,
          provenance: "declared",
          trace_name: "",
          metrics: { processing_ms: { mean: 8 } },
          metric_value_ms: 8,
          description: "",
        },
      ],
      edges: [],
    });
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.metric = "mean";
    store.selection = {
      kind: "component",
      title: "Worker",
      subtitle: "worker",
      data: { component_id: "worker", metric_value_ms: 1 },
    };

    await store.refreshTopology();

    expect(store.selection?.data.metric_value_ms).toBe(8);
  });

  it("refreshes an open edge inspector with the latest topology data", async () => {
    vi.spyOn(tracingApi, "topology").mockResolvedValue({
      metric: "p95",
      view: "components",
      request_id: "request",
      topology_source: "declared",
      nodes: [],
      edges: [
        {
          id: "edge",
          edge_id: "edge",
          source_id: "source",
          target_id: "target",
          name: "Transfer",
          kind: "flow",
          provenance: "declared",
          metrics: { latency_ms: { p95: 9 } },
          metric_value_ms: 9,
          directed: true,
        },
      ],
    });
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.analysisState = "ready";
    store.requestId = "request";
    store.selection = {
      kind: "edge",
      title: "Transfer",
      subtitle: "edge",
      data: { edge_id: "edge", metric_value_ms: 1 },
    };

    await store.refreshTopology();

    expect(store.selection?.data.metric_value_ms).toBe(9);
  });
});
