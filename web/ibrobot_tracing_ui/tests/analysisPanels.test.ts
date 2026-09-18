import { createPinia, getActivePinia, setActivePinia } from "pinia";
import { createApp, nextTick, type App, type Component } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalysisSummary, ComparisonResult } from "../src/api/types";
import ComparisonPanel from "../src/components/tabs/ComparisonPanel.vue";
import RequestsPanel from "../src/components/tabs/RequestsPanel.vue";
import SummaryPanel from "../src/components/tabs/SummaryPanel.vue";
import WarningsPanel from "../src/components/tabs/WarningsPanel.vue";
import { useProfilerStore } from "../src/stores/profiler";

let app: App;
let host: HTMLDivElement;

function mount(component: Component): HTMLDivElement {
  host = document.createElement("div");
  document.body.appendChild(host);
  app = createApp(component).use(getActivePinia()!);
  app.mount(host);
  return host;
}

function summary(): AnalysisSummary {
  return {
    analysis: {
      id: "analysis", source_id: "candidate", source_name: "candidate", source_kind: "log",
      source_version: "v1", created_at: "2026-09-15T00:00:00Z", event_count: 10,
      request_count: 1, span_count: 6, flow_count: 0, warning_count: 1,
    },
    metadata: {}, stages: {}, observations: {}, coverage: { missing_events: ["inference_ms"] },
    span_summary: [], span_summary_total: 0, span_summary_limit: 100, span_summary_truncated: false,
    custom_span_summary: [], custom_mark_summary: [],
  };
}

describe("non-intrusive analysis panels", () => {
  beforeEach(() => setActivePinia(createPinia()));
  afterEach(() => {
    app?.unmount();
    host?.remove();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("makes a truncated diagnostic response visible instead of claiming completeness", () => {
    const store = useProfilerStore();
    store.tabState.warnings = "ready";
    store.warnings = ["first", "second"];
    store.warningTotal = 1234;
    store.warningsTruncated = true;
    mount(WarningsPanel);
    expect(host.textContent).toContain("共 1234 条诊断，仅展示前 2 条");
    expect(host.textContent).toContain("筛选仅作用于已展示记录");
  });

  it("distinguishes metric observation statuses from missing samples without offering a retry", async () => {
    const store = useProfilerStore();
    store.tabState.requests = "ready";
    store.requests = [
      { request_id: "retry", inference_ms_status: "ambiguous", inference_ms_source: "structured" },
      { request_id: "failed", inference_ms_status: "error" },
      { request_id: "open", inference_ms_status: "incomplete" },
      { request_id: "missing" },
      { request_id: "ok", inference_ms: 0 },
      { request_id: "invalid", inference_ms_status: "invalid" },
      { request_id: "mixed", inference_ms_status: "error,incomplete" },
      { request_id: "custom", inference_ms_status: "cancelled" },
    ];
    mount(RequestsPanel);
    const cell = (id: string) => host.querySelector(`[id="grid-row-${id}"] td:nth-child(3)`)?.textContent;
    expect(cell("retry")).toBe("歧义 (ambiguous)");
    expect(cell("failed")).toBe("错误记录 (error)");
    expect(cell("open")).toBe("记录不完整 (incomplete)");
    expect(cell("missing")).toBe("未采到");
    expect(cell("ok")).not.toBe("未采到");
    expect(cell("invalid")).toBe("无效数据 (invalid)");
    expect(cell("mixed")).toBe("错误记录 (error) / 记录不完整 (incomplete)");
    expect(cell("custom")).toBe("cancelled");
    expect(Array.from(host.querySelectorAll("button")).some((button) => button.textContent === "重试")).toBe(false);

    const navigate = vi.spyOn(store, "navigateToRequest").mockResolvedValue();
    const button = host.querySelector('[id="grid-row-retry"] button[aria-label*="Span 分析"]') as HTMLButtonElement;
    button.click();
    await nextTick();
    expect(navigate).toHaveBeenCalledWith("retry", "span-profile");
  });

  it("shows the bounded all-origin/status preview and opens the existing aggregate Span analysis", async () => {
    const store = useProfilerStore();
    store.tabState.summary = "ready";
    store.summary = summary();
    store.summary.span_summary = Array.from({ length: 100 }, (_, index) => ({
      component_id: "worker", name: `work-${Math.floor(index / 4)}`,
      origin: index % 4 < 2 ? "built-in" : "user", status: index % 2 ? "ok" : "error",
      count: 2, minimum: 1, p50: 1, p95: 2, p99: 2, maximum: 2, mean: 1.5,
    }));
    store.summary.span_summary_total = 250;
    store.summary.span_summary_truncated = true;
    mount(SummaryPanel);
    expect(host.textContent).toContain("请求链投影");
    expect(host.textContent).toContain("未形成投影数值");
    expect(host.textContent).toContain("不等于事件未采到");
    expect(host.textContent).toContain("100 / 250 组，上限 100");
    expect(host.textContent).toContain("已截断");
    expect(host.querySelectorAll("tr[id]")).toHaveLength(100);
    expect(new Set(Array.from(host.querySelectorAll("tr[id]"), (row) => row.id)).size).toBe(100);

    store.searchQuery = "error";
    await nextTick();
    expect(host.querySelectorAll("tr[id]")).toHaveLength(50);
    const fetch = vi.spyOn(store, "fetchTab").mockResolvedValue();
    store.componentId = "old-scope";
    const more = Array.from(host.querySelectorAll("button")).find((button) => button.textContent === "更多见Span分析");
    more!.click();
    await vi.waitFor(() => expect(store.activeTab).toBe("span-profile"));
    expect(store.spanAnalysisMode).toBe("aggregate");
    expect(store.componentId).toBe("");
    expect(store.searchQuery).toBe("");
    expect(fetch).toHaveBeenCalledWith("span-profile");
  });

  it.each([true, false])("labels comparable=%s as metric sufficiency rather than business success", (comparable) => {
    vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
    const store = useProfilerStore();
    store.summary = summary();
    store.sources = [{ id: "baseline", name: "baseline", kind: "log", version: "v1", modified_at: "", size_bytes: 0, file_count: 1 }];
    store.comparisonState = "ready";
    store.comparisonResult = {
      id: "comparison", job_id: "job", baseline_source_id: "baseline", baseline_source_version: "v1",
      candidate_analysis_id: "analysis", created_at: "2026-09-15T00:00:00Z", statistic: "p95",
      relative_threshold_percent: 10, absolute_threshold_ms: 1, metrics: [], flame_diff: [],
      histogram: { metric: "preprocess_ms", unit: "ms", minimum: null, maximum: null, bin_count: 0, baseline_count: 0, candidate_count: 0, buckets: [] },
      comparable, has_regression: false, coverage_warnings: [], count_warnings: [],
      blocking_reasons: comparable ? [] : ["required metric preprocess_ms is missing from the candidate"],
    } satisfies ComparisonResult;
    mount(ComparisonPanel);
    expect(host.textContent).toContain("comparable 仅表示所选指标数据足够进行比较，不等同于业务成功或可用性");
    expect(host.textContent).toContain(comparable ? "所选指标数据足够比较" : "所选指标数据不足以比较");
    expect(host.textContent).not.toContain("比较被阻止");
    expect(host.textContent).not.toContain("业务不可用");
  });
});
