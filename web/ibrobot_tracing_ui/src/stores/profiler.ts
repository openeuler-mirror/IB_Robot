import { computed, reactive, ref, watch } from "vue";
import { defineStore } from "pinia";
import { ApiError, pollComparisonJob, pollJob, tracingApi } from "../api/client";
import type {
  AnalysisSummary,
  CallTreeProjection,
  CapabilitiesResponse,
  ComparisonJob,
  ComparisonResult,
  ComparisonStatistic,
  CriticalPathResponse,
  DetailTab,
  FlowRecord,
  GraphView,
  InspectorSelection,
  LatencyDistributionResponse,
  LoadJob,
  LoadState,
  MetricName,
  RequestRecord,
  SpanRecord,
  SpanAnalysisMode,
  SpanProfileResponse,
  TimelineProjection,
  TraceEvent,
  TracepointDefinition,
  TraceSource,
  TraceTopology,
} from "../api/types";
import { resolveTracepointDescription } from "../utils/tracepoints";

const DATA_TABS = new Set<DetailTab>(["distribution", "timeline", "span-profile", "critical-path", "events", "spans", "flows", "warnings"]);
const SCOPED_TABS: DetailTab[] = ["timeline", "span-profile", "critical-path", "events", "spans", "flows"];
const SPAN_PROFILE_SVG_NODE_LIMIT = 3_000;
const COMPARISON_METRIC_PATTERN = /^[A-Za-z_][A-Za-z0-9_.-]*_ms$/;
type RequestDetailTab = "timeline" | "span-profile" | "critical-path";

function messageFrom(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "发生未知错误";
}

export const useProfilerStore = defineStore("profiler", () => {
  const sources = ref<TraceSource[]>([]);
  const capabilities = ref<CapabilitiesResponse | null>(null);
  const sourcesState = ref<LoadState>("idle");
  const sourcesError = ref("");
  const sourceId = ref("");
  const analysisId = ref("");
  const analysisState = ref<LoadState>("idle");
  const analysisError = ref("");
  const loadJob = ref<LoadJob | null>(null);

  const selectedBaselineSourceId = ref("");
  const comparisonStatistic = ref<ComparisonStatistic>("p95");
  const comparisonRelativeThreshold = ref(10);
  const comparisonAbsoluteThresholdMs = ref(1);
  const comparisonMetric = ref("");
  const comparisonJob = ref<ComparisonJob | null>(null);
  const comparisonResult = ref<ComparisonResult | null>(null);
  const comparisonState = ref<LoadState>("idle");
  const comparisonError = ref("");

  const summary = ref<AnalysisSummary | null>(null);
  const requests = ref<RequestRecord[]>([]);
  const distribution = ref<LatencyDistributionResponse | null>(null);
  const topology = ref<TraceTopology | null>(null);
  const timeline = ref<TimelineProjection | null>(null);
  const callTree = ref<CallTreeProjection | null>(null);
  const spanProfile = ref<SpanProfileResponse | null>(null);
  const criticalPath = ref<CriticalPathResponse | null>(null);
  const events = ref<TraceEvent[]>([]);
  const tracepoints = ref<TracepointDefinition[]>([]);
  const spans = ref<SpanRecord[]>([]);
  const flows = ref<FlowRecord[]>([]);
  const warnings = ref<string[]>([]);
  const warningTotal = ref(0);
  const warningsTruncated = ref(false);

  const activeTab = ref<DetailTab>("summary");
  const requestId = ref("");
  const componentId = ref("");
  const distributionMetric = ref("");
  const metric = ref<MetricName>("p95");
  const graphView = ref<GraphView>("components");
  const spanAnalysisMode = ref<SpanAnalysisMode>("request");
  const searchQuery = ref("");
  const motionPaused = ref(false);
  const selection = ref<InspectorSelection | null>(null);
  const tabState = reactive<Record<DetailTab, LoadState>>({
    summary: "idle",
    requests: "idle",
    distribution: "idle",
    comparison: "idle",
    timeline: "idle",
    "span-profile": "idle",
    "critical-path": "idle",
    events: "idle",
    spans: "idle",
    flows: "idle",
    warnings: "idle",
  });
  const tabError = reactive<Record<DetailTab, string>>({
    summary: "",
    requests: "",
    distribution: "",
    comparison: "",
    timeline: "",
    "span-profile": "",
    "critical-path": "",
    events: "",
    spans: "",
    flows: "",
    warnings: "",
  });
  const tabKey = reactive<Record<DetailTab, string>>({
    summary: "",
    requests: "",
    distribution: "",
    comparison: "",
    timeline: "",
    "span-profile": "",
    "critical-path": "",
    events: "",
    spans: "",
    flows: "",
    warnings: "",
  });

  let loadController: AbortController | null = null;
  let topologyController: AbortController | null = null;
  let comparisonController: AbortController | null = null;
  let comparisonGeneration = 0;
  const tabControllers = new Map<DetailTab, AbortController>();
  const distributionCache = new Map<string, LatencyDistributionResponse>();

  const selectedSource = computed(() => sources.value.find((source) => source.id === sourceId.value) ?? null);
  const baselineSources = computed(() => sources.value.filter((source) => source.id !== summary.value?.analysis.source_id));
  const selectedBaselineSource = computed(() => baselineSources.value.find((source) => source.id === selectedBaselineSourceId.value) ?? null);
  const components = computed(() => topology.value?.nodes ?? []);
  const selectedRequest = computed(() => requests.value.find((request) => request.request_id === requestId.value) ?? null);
  const selectionDescription = computed(() => resolveTracepointDescription(selection.value, tracepoints.value));
  const topologyQueryKey = computed(() => JSON.stringify([analysisId.value, requestId.value, metric.value, graphView.value]));

  // Invalidate before a replacement comparison can start in the same tick.
  watch([selectedBaselineSourceId, () => selectedBaselineSource.value?.version], clearComparison, { flush: "sync" });

  async function loadSources(): Promise<void> {
    sourcesState.value = "loading";
    sourcesError.value = "";
    const sourceLoad = tracingApi.sources().then((response) => {
      sources.value = response.items;
      if (!sourceId.value && response.items.length) sourceId.value = response.items[0].id;
      repairBaselineSourceSelection();
      sourcesError.value = response.warnings.join("；");
      sourcesState.value = "ready";
    }).catch((error: unknown) => {
      sourcesState.value = "error";
      sourcesError.value = messageFrom(error);
    });
    const capabilityLoad = tracingApi.capabilities().then((response) => {
      capabilities.value = response;
    }).catch(() => undefined);
    await Promise.all([sourceLoad, capabilityLoad]);
  }

  function repairBaselineSourceSelection(): void {
    if (!baselineSources.value.some((source) => source.id === selectedBaselineSourceId.value)) {
      selectedBaselineSourceId.value = baselineSources.value[0]?.id ?? "";
    }
  }

  async function refreshSources(): Promise<void> {
    sourcesState.value = "loading";
    sourcesError.value = "";
    try {
      const response = await tracingApi.refreshSources();
      sources.value = response.items;
      if (!response.items.some((source) => source.id === sourceId.value)) sourceId.value = response.items[0]?.id ?? "";
      repairBaselineSourceSelection();
      sourcesError.value = response.warnings.join("；");
      sourcesState.value = "ready";
    } catch (error) {
      sourcesState.value = "error";
      sourcesError.value = messageFrom(error);
    }
  }

  function clearAnalysis(): void {
    clearComparison();
    summary.value = null;
    requests.value = [];
    distribution.value = null;
    topology.value = null;
    timeline.value = null;
    callTree.value = null;
    spanProfile.value = null;
    criticalPath.value = null;
    events.value = [];
    tracepoints.value = [];
    spans.value = [];
    flows.value = [];
    warnings.value = [];
    warningTotal.value = 0;
    warningsTruncated.value = false;
    distributionCache.clear();
    selection.value = null;
    Object.keys(tabState).forEach((key) => {
      tabState[key as DetailTab] = "idle";
      tabError[key as DetailTab] = "";
      tabKey[key as DetailTab] = "";
    });
  }

  function clearComparison(): void {
    comparisonGeneration += 1;
    comparisonController?.abort();
    comparisonController = null;
    comparisonJob.value = null;
    comparisonResult.value = null;
    comparisonState.value = "idle";
    comparisonError.value = "";
    if (selection.value?.kind === "comparison") selection.value = null;
  }

  async function compareWithSourceBaseline(): Promise<void> {
    const baseline = selectedBaselineSource.value;
    const candidateAnalysisId = analysisId.value;
    const validationError = !candidateAnalysisId || analysisState.value !== "ready"
      ? "请先载入候选分析"
        : !baseline
          ? "请先选择另一份追踪作为基线"
          : "";
    const relativeThreshold = Number(comparisonRelativeThreshold.value);
    const absoluteThreshold = Number(comparisonAbsoluteThresholdMs.value);
    const selectedMetric = comparisonMetric.value.trim();
    const thresholdError = !Number.isFinite(relativeThreshold) || relativeThreshold < 0
      ? "相对阈值必须是非负有限数值"
      : !Number.isFinite(absoluteThreshold) || absoluteThreshold < 0
        ? "绝对阈值必须是非负有限数值"
        : selectedMetric && !COMPARISON_METRIC_PATTERN.test(selectedMetric)
          ? "比较指标名称无效"
        : "";
    if (validationError || thresholdError) {
      comparisonState.value = "error";
      comparisonError.value = validationError || thresholdError;
      return;
    }
    if (!baseline) return;

    comparisonController?.abort();
    const controller = new AbortController();
    comparisonController = controller;
    const generation = ++comparisonGeneration;
    const current = () => (
      comparisonController === controller
      && !controller.signal.aborted
      && comparisonGeneration === generation
      && analysisId.value === candidateAnalysisId
      && selectedBaselineSourceId.value === baseline.id
      && selectedBaselineSource.value?.version === baseline.version
    );
    comparisonJob.value = null;
    comparisonResult.value = null;
    comparisonState.value = "loading";
    comparisonError.value = "";
    try {
      const initial = await tracingApi.createComparison({
        baseline_source_id: baseline.id,
        baseline_source_version: baseline.version,
        candidate_analysis_id: candidateAnalysisId,
        statistic: comparisonStatistic.value,
        relative_threshold_percent: relativeThreshold,
        absolute_threshold_ms: absoluteThreshold,
        bins: 30,
        metric: selectedMetric || null,
      }, controller.signal);
      if (!current()) return;
      const completed = await pollComparisonJob(initial, (job) => {
        if (current()) comparisonJob.value = job;
      }, controller.signal);
      if (!current()) return;
      if (completed.status !== "completed") throw new Error(completed.error || "追踪比较失败");
      if (!completed.comparison_id) throw new Error("比较任务未返回结果标识");
      const result = await tracingApi.comparison(completed.comparison_id, controller.signal);
      if (!current()) return;
      comparisonResult.value = result;
      if (result.histogram.metric && COMPARISON_METRIC_PATTERN.test(result.histogram.metric)) {
        comparisonMetric.value = result.histogram.metric;
      }
      comparisonState.value = "ready";
    } catch (error) {
      if (controller.signal.aborted || comparisonController !== controller || comparisonGeneration !== generation) return;
      comparisonState.value = "error";
      comparisonError.value = messageFrom(error);
    }
  }

  async function deleteComparisonResult(): Promise<void> {
    const resultId = comparisonResult.value?.id;
    clearComparison();
    if (!resultId) return;
    try {
      await tracingApi.deleteComparison(resultId);
    } catch (error) {
      comparisonState.value = "error";
      comparisonError.value = messageFrom(error);
    }
  }

  function queryKey(tab: DetailTab): string {
    if (tab === "warnings") return `${analysisId.value}:${tab}`;
    if (tab === "distribution") return `${analysisId.value}:${tab}:${distributionMetric.value}`;
    if (tab === "span-profile") {
      const requestScope = spanAnalysisMode.value === "aggregate" ? "" : requestId.value;
      return `${analysisId.value}:${tab}:${spanAnalysisMode.value}:${requestScope}:${componentId.value}`;
    }
    return `${analysisId.value}:${tab}:${requestId.value}:${componentId.value}`;
  }

  function abortTabRequests(): void {
    tabControllers.forEach((controller) => controller.abort());
    tabControllers.clear();
  }

  async function loadTracepointCatalog(id: string, signal: AbortSignal): Promise<TracepointDefinition[]> {
    const items: TracepointDefinition[] = [];
    let offset = 0;
    while (!signal.aborted) {
      const response = await tracingApi.tracepoints(id, { offset, limit: 1000 }, signal);
      if (signal.aborted) return items;
      items.push(...response.items);
      if (response.next_offset === null || response.next_offset === undefined) return items;
      offset = response.next_offset;
    }
    return items;
  }

  async function openAnalysis(id: string): Promise<void> {
    if (!id) return;
    loadController?.abort();
    topologyController?.abort();
    abortTabRequests();
    loadController = new AbortController();
    const { signal } = loadController;
    clearAnalysis();
    analysisId.value = id;
    analysisState.value = "loading";
    analysisError.value = "";
    tabState.summary = "loading";
    tabState.requests = "loading";
    const topologyKey = topologyQueryKey.value;
    try {
      const [summaryResponse, requestResponse, topologyResponse, tracepointResponse] = await Promise.all([
        tracingApi.summary(id, signal),
        tracingApi.requests(id, signal),
        tracingApi.topology(id, { metric: metric.value, view: graphView.value, requestId: requestId.value }, signal),
        loadTracepointCatalog(id, signal),
      ]);
      if (signal.aborted || analysisId.value !== id) return;
      summary.value = summaryResponse;
      requests.value = requestResponse.items;
      if (topologyKey === topologyQueryKey.value) topology.value = topologyResponse;
      tracepoints.value = tracepointResponse;
      sourceId.value = summaryResponse.analysis.source_id;
      repairBaselineSourceSelection();
      tabState.summary = "ready";
      tabState.requests = "ready";
      analysisState.value = "ready";
      // Parameter watchers cannot refresh topology until the analysis is ready.
      if (topologyKey !== topologyQueryKey.value) {
        await refreshTopology();
        if (signal.aborted || analysisId.value !== id) return;
      }
      await fetchTab(activeTab.value);
    } catch (error) {
      if (signal.aborted) return;
      const message = messageFrom(error);
      analysisState.value = "error";
      analysisError.value = message;
      tabState.summary = "error";
      tabState.requests = "error";
      tabError.summary = message;
      tabError.requests = message;
    }
  }

  async function analyzeSelectedSource(): Promise<string> {
    const source = selectedSource.value;
    if (!source) throw new Error("请先选择追踪源");
    loadController?.abort();
    loadController = new AbortController();
    analysisState.value = "loading";
    analysisError.value = "";
    loadJob.value = null;
    try {
      const initial = await tracingApi.createAnalysis(source.id);
      const completed = await pollJob(initial, (job) => (loadJob.value = job), loadController.signal);
      if (completed.status === "failed" || completed.status === "cancelled" || completed.status === "expired") {
        throw new Error(completed.error || "追踪分析失败");
      }
      if (!completed.analysis_id) throw new Error("分析任务未返回分析标识");
      await openAnalysis(completed.analysis_id);
      return completed.analysis_id;
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      analysisState.value = "error";
      analysisError.value = messageFrom(error);
      throw error;
    }
  }

  async function refreshTopology(): Promise<void> {
    if (!analysisId.value || analysisState.value !== "ready") return;
    topologyController?.abort();
    topologyController = new AbortController();
    const controller = topologyController;
    const { signal } = controller;
    const id = analysisId.value;
    const topologyKey = topologyQueryKey.value;
    try {
      const response = await tracingApi.topology(
        id,
        { metric: metric.value, view: graphView.value, requestId: requestId.value },
        signal,
      );
      if (!signal.aborted && topologyController === controller && topologyKey === topologyQueryKey.value) {
        topology.value = response;
        if (selection.value?.kind === "component") {
          const selectedId = String(selection.value.data.component_id ?? "");
          const component = response.nodes.find((item) => item.component_id === selectedId);
          selection.value = component
            ? { kind: "component", title: component.name, subtitle: component.component_id, data: { ...component } }
            : null;
        } else if (selection.value?.kind === "edge") {
          const selectedId = String(selection.value.data.edge_id ?? "");
          const edge = response.edges.find((item) => item.edge_id === selectedId);
          selection.value = edge
            ? { kind: "edge", title: edge.name, subtitle: edge.edge_id, data: { ...edge } }
            : null;
        }
      }
    } catch (error) {
      if (!signal.aborted && topologyController === controller && topologyKey === topologyQueryKey.value) {
        analysisError.value = messageFrom(error);
      }
    }
  }

  async function fetchDistribution(force = false): Promise<void> {
    if (!analysisId.value) return;
    const tab: DetailTab = "distribution";
    const id = analysisId.value;
    const requestedMetric = distributionMetric.value;
    const key = `${id}:${tab}:${requestedMetric}`;
    const cached = !force ? distributionCache.get(key) : undefined;
    if (cached) {
      distribution.value = cached;
      distributionMetric.value = cached.metric ?? "";
      tabKey[tab] = `${id}:${tab}:${distributionMetric.value}`;
      tabState[tab] = "ready";
      tabError[tab] = "";
      return;
    }
    if (!force && tabState[tab] === "ready" && tabKey[tab] === key) return;

    tabControllers.get(tab)?.abort();
    const controller = new AbortController();
    tabControllers.set(tab, controller);
    const { signal } = controller;
    tabState[tab] = "loading";
    tabError[tab] = "";
    try {
      const response = await tracingApi.distribution(
        id,
        { metric: requestedMetric || undefined },
        signal,
      );
      const current = !signal.aborted
        && tabControllers.get(tab) === controller
        && analysisId.value === id
        && distributionMetric.value === requestedMetric;
      if (!current) return;
      const resolvedMetric = response.metric ?? "";
      distributionCache.set(key, response);
      distributionCache.set(`${id}:${tab}:${resolvedMetric}`, response);
      distribution.value = response;
      distributionMetric.value = resolvedMetric;
      tabKey[tab] = `${id}:${tab}:${resolvedMetric}`;
      tabState[tab] = "ready";
    } catch (error) {
      if (signal.aborted || tabControllers.get(tab) !== controller) return;
      tabState[tab] = "error";
      tabError[tab] = messageFrom(error);
    }
  }

  async function fetchTab(tab: DetailTab, force = false): Promise<void> {
    if (!analysisId.value || !DATA_TABS.has(tab)) return;
    if (tab === "distribution") return fetchDistribution(force);
    if (tab === "critical-path" && analysisState.value !== "ready") return;
    const key = queryKey(tab);
    if ((tab === "span-profile" && spanAnalysisMode.value !== "aggregate" && !requestId.value) || (tab === "critical-path" && !requestId.value)) {
      tabControllers.get(tab)?.abort();
      tabControllers.delete(tab);
      if (tab === "span-profile") {
        callTree.value = null;
        spanProfile.value = null;
      }
      else criticalPath.value = null;
      tabState[tab] = "ready";
      tabError[tab] = "";
      tabKey[tab] = key;
      return;
    }
    if (!force && tabState[tab] === "ready" && tabKey[tab] === key) return;
    tabControllers.get(tab)?.abort();
    const controller = new AbortController();
    tabControllers.set(tab, controller);
    const { signal } = controller;
    const id = analysisId.value;
    tabState[tab] = "loading";
    tabError[tab] = "";
    const current = () => !signal.aborted && tabControllers.get(tab) === controller && queryKey(tab) === key;
    try {
      if (tab === "timeline") {
        const response = await tracingApi.timeline(
          id,
          { requestId: requestId.value, componentId: componentId.value },
          signal,
        );
        if (!current()) return;
        timeline.value = response;
      } else if (tab === "events") {
        const response = await tracingApi.events(
          id,
          { requestId: requestId.value, componentId: componentId.value },
          signal,
        );
        if (!current()) return;
        events.value = response.items;
      } else if (tab === "span-profile") {
        if (spanAnalysisMode.value === "call-tree") {
          const response = await tracingApi.callTree(
            id,
            { requestId: requestId.value, componentId: componentId.value },
            signal,
          );
          if (!current()) return;
          callTree.value = response;
          spanProfile.value = null;
        } else {
          const response = spanAnalysisMode.value === "request"
            ? await tracingApi.spanProfile(
              id,
              {
                mode: "request",
                requestId: requestId.value,
                componentId: componentId.value,
                maxNodes: SPAN_PROFILE_SVG_NODE_LIMIT,
              },
              signal,
            )
            : await tracingApi.spanProfile(
              id,
              {
                mode: "aggregate",
                componentId: componentId.value,
                maxNodes: SPAN_PROFILE_SVG_NODE_LIMIT,
              },
              signal,
            );
          if (!current()) return;
          spanProfile.value = response;
          callTree.value = null;
        }
      } else if (tab === "critical-path") {
        const response = await tracingApi.criticalPath(
          id,
          {
            requestId: requestId.value,
            componentId: componentId.value,
            includeFlows: true,
            maxSegments: 1_000,
          },
          signal,
        );
        if (!current()) return;
        criticalPath.value = response;
      } else if (tab === "spans") {
        const response = await tracingApi.spans(
          id,
          { requestId: requestId.value, componentId: componentId.value },
          signal,
        );
        if (!current()) return;
        spans.value = response.items;
      } else if (tab === "flows") {
        const response = await tracingApi.flows(
          id,
          { requestId: requestId.value, componentId: componentId.value },
          signal,
        );
        if (!current()) return;
        flows.value = response.items;
      } else if (tab === "warnings") {
        const response = await tracingApi.warnings(id, signal);
        if (!current()) return;
        warnings.value = response.items;
        warningTotal.value = response.total ?? response.items.length;
        warningsTruncated.value = response.truncated ?? false;
      }
      if (!current()) return;
      tabKey[tab] = key;
      tabState[tab] = "ready";
    } catch (error) {
      if (signal.aborted || tabControllers.get(tab) !== controller) return;
      tabState[tab] = "error";
      tabError[tab] = messageFrom(error);
    }
  }

  async function refreshScopedData(): Promise<void> {
    const selectedComponent = selection.value?.kind === "component" ? selection.value.data : null;
    const selectedRequestId = selection.value?.kind === "request" ? String(selection.value.data.request_id ?? "") : "";
    const keepSelection = selectedComponent
      ? [selectedComponent.component_id, selectedComponent.parent_id].includes(componentId.value)
      : Boolean(selectedRequestId && selectedRequestId === requestId.value);
    if (!keepSelection) selection.value = null;
    SCOPED_TABS.forEach((tab) => {
      tabControllers.get(tab)?.abort();
      tabControllers.delete(tab);
      tabState[tab] = "idle";
      tabError[tab] = "";
      tabKey[tab] = "";
    });
    timeline.value = null;
    callTree.value = null;
    spanProfile.value = null;
    criticalPath.value = null;
    events.value = [];
    spans.value = [];
    flows.value = [];
    if (SCOPED_TABS.includes(activeTab.value)) {
      await fetchTab(activeTab.value, true);
    }
  }

  async function setSpanAnalysisMode(mode: SpanAnalysisMode): Promise<void> {
    if (spanAnalysisMode.value === mode) return;
    spanAnalysisMode.value = mode;
    if (selection.value?.kind === "span") selection.value = null;
    tabControllers.get("span-profile")?.abort();
    tabControllers.delete("span-profile");
    callTree.value = null;
    spanProfile.value = null;
    tabState["span-profile"] = "idle";
    tabError["span-profile"] = "";
    tabKey["span-profile"] = "";
    if (activeTab.value === "span-profile" && analysisState.value === "ready") {
      await fetchTab("span-profile", true);
    }
  }

  async function setDistributionMetric(metricName: string): Promise<void> {
    const nextMetric = metricName.trim();
    if (distributionMetric.value === nextMetric) return;
    distributionMetric.value = nextMetric;
    tabControllers.get("distribution")?.abort();
    tabControllers.delete("distribution");
    tabState.distribution = "idle";
    tabError.distribution = "";
    tabKey.distribution = "";
    if (activeTab.value === "distribution" && analysisState.value === "ready") {
      await fetchDistribution();
    }
  }

  function clearCriticalPath(): void {
    tabControllers.get("critical-path")?.abort();
    tabControllers.delete("critical-path");
    criticalPath.value = null;
    tabState["critical-path"] = "idle";
    tabError["critical-path"] = "";
    tabKey["critical-path"] = "";
    if (selection.value?.kind === "critical-path") selection.value = null;
  }

  function selectRequest(id: string): void {
    requestId.value = requestId.value === id ? "" : id;
    const request = requests.value.find((item) => item.request_id === id);
    selection.value = request
      ? { kind: "request", title: `请求 ${id}`, subtitle: "请求延迟明细", data: { ...request } }
      : null;
  }

  async function navigateToRequest(id: string, tab: RequestDetailTab): Promise<void> {
    requestId.value = id;
    componentId.value = "";
    if (tab === "span-profile") spanAnalysisMode.value = "request";
    const request = requests.value.find((item) => item.request_id === id);
    selection.value = {
      kind: "request",
      title: `请求 ${id}`,
      subtitle: "请求延迟明细",
      data: request ? { ...request } : { request_id: id },
    };
    activeTab.value = tab;
    await refreshScopedData();
  }

  function selectComponent(id: string): void {
    const component = components.value.find((item) => item.component_id === id);
    const scopeId = component?.kind === "operation" || component?.kind === "instant_event" ? component.parent_id : id;
    componentId.value = componentId.value === scopeId ? "" : scopeId;
    selection.value = component
      ? { kind: "component", title: component.name, subtitle: component.component_id, data: { ...component } }
      : null;
  }

  return {
    sources,
    capabilities,
    baselineSources,
    selectedBaselineSourceId,
    selectedBaselineSource,
    comparisonStatistic,
    comparisonRelativeThreshold,
    comparisonAbsoluteThresholdMs,
    comparisonMetric,
    comparisonJob,
    comparisonResult,
    comparisonState,
    comparisonError,
    sourcesState,
    sourcesError,
    sourceId,
    selectedSource,
    analysisId,
    analysisState,
    analysisError,
    loadJob,
    summary,
    requests,
    distribution,
    topology,
    timeline,
    callTree,
    spanProfile,
    criticalPath,
    components,
    events,
    tracepoints,
    spans,
    flows,
    warnings,
    warningTotal,
    warningsTruncated,
    activeTab,
    requestId,
    selectedRequest,
    componentId,
    distributionMetric,
    metric,
    graphView,
    spanAnalysisMode,
    searchQuery,
    motionPaused,
    selection,
    selectionDescription,
    tabState,
    tabError,
    loadSources,
    refreshSources,
    openAnalysis,
    analyzeSelectedSource,
    refreshTopology,
    fetchTab,
    refreshScopedData,
    setSpanAnalysisMode,
    setDistributionMetric,
    clearCriticalPath,
    compareWithSourceBaseline,
    clearComparison,
    deleteComparisonResult,
    selectRequest,
    navigateToRequest,
    selectComponent,
  };
});
