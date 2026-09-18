import { createPinia, setActivePinia } from "pinia";
import { nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { tracingApi } from "../src/api/client";
import type { AnalysisSummary, ComparisonJob, ComparisonResult, TraceSource } from "../src/api/types";
import { useProfilerStore } from "../src/stores/profiler";

const CANDIDATE_ID = "c".repeat(32);
const BASELINE_ID = "a".repeat(32);
const OTHER_ID = "b".repeat(32);
const BASELINE_VERSION = "1".repeat(64);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function source(id: string, version = BASELINE_VERSION): TraceSource {
  return {
    id,
    name: `trace-${id[0]}`,
    kind: "ctf",
    version,
    modified_at: "2026-07-21T00:00:00Z",
    size_bytes: 100,
    file_count: 1,
  };
}

function summary(): AnalysisSummary {
  return {
    analysis: {
      id: "analysis-1",
      source_id: CANDIDATE_ID,
      source_name: "candidate",
      source_kind: "ctf",
      source_version: "2".repeat(64),
      created_at: "2026-07-21T00:00:00Z",
      event_count: 0,
      request_count: 0,
      span_count: 0,
      flow_count: 0,
      warning_count: 0,
    },
    metadata: {},
    stages: {},
    observations: {},
    span_summary: [],
    span_summary_total: 0,
    span_summary_limit: 100,
    span_summary_truncated: false,
    custom_span_summary: [],
    custom_mark_summary: [],
    coverage: {},
  };
}

function job(status: ComparisonJob["status"] = "completed", comparisonId: string | null = "comparison-1"): ComparisonJob {
  return {
    id: "comparison-job-1",
    baseline_source_id: BASELINE_ID,
    baseline_source_version: BASELINE_VERSION,
    candidate_analysis_id: "analysis-1",
    statistic: "p95",
    relative_threshold_percent: 10,
    absolute_threshold_ms: 1,
    bins: 30,
    metric: null,
    status,
    created_at: "2026-07-21T00:00:00Z",
    started_at: null,
    finished_at: null,
    comparison_id: comparisonId,
    error: status === "failed" ? "comparison exploded" : null,
    queue_position: null,
    deduplicated: false,
  };
}

function result(): ComparisonResult {
  return {
    id: "comparison-1",
    job_id: "comparison-job-1",
    baseline_source_id: BASELINE_ID,
    baseline_source_version: BASELINE_VERSION,
    candidate_analysis_id: "analysis-1",
    created_at: "2026-07-21T00:00:01Z",
    statistic: "p95",
    relative_threshold_percent: 10,
    absolute_threshold_ms: 1,
    metrics: [],
    histogram: {
      metric: "total_ms",
      unit: "ms",
      minimum: 1,
      maximum: 2,
      bin_count: 1,
      baseline_count: 1,
      candidate_count: 1,
      buckets: [{ index: 0, start: 1, end: 2, baseline_count: 1, candidate_count: 1 }],
    },
    flame_diff: [],
    coverage_warnings: [],
    count_warnings: [],
    comparable: true,
    blocking_reasons: [],
    has_regression: false,
  };
}

function prepareComparisonStore() {
  const store = useProfilerStore();
  store.analysisId = "analysis-1";
  store.analysisState = "ready";
  store.summary = summary();
  store.sources = [source(CANDIDATE_ID), source(BASELINE_ID), source(OTHER_ID, "3".repeat(64))];
  store.selectedBaselineSourceId = BASELINE_ID;
  return store;
}

describe("comparison store", () => {
  beforeEach(() => setActivePinia(createPinia()));
  afterEach(() => vi.restoreAllMocks());

  it("excludes the current analysis source from baseline choices", () => {
    const store = prepareComparisonStore();

    expect(store.baselineSources.map((item) => item.id)).toEqual([BASELINE_ID, OTHER_ID]);
    expect(store.selectedBaselineSource?.id).toBe(BASELINE_ID);
  });

  it("submits the selected trace ID and immutable catalog version without paths", async () => {
    const create = vi.spyOn(tracingApi, "createComparison").mockResolvedValue(job());
    vi.spyOn(tracingApi, "comparison").mockResolvedValue(result());
    const store = prepareComparisonStore();

    await store.compareWithSourceBaseline();

    expect(create).toHaveBeenCalledWith({
      baseline_source_id: BASELINE_ID,
      baseline_source_version: BASELINE_VERSION,
      candidate_analysis_id: "analysis-1",
      statistic: "p95",
      relative_threshold_percent: 10,
      absolute_threshold_ms: 1,
      bins: 30,
      metric: null,
    }, expect.any(AbortSignal));
    expect(create.mock.calls[0][0]).not.toHaveProperty("path");
    expect(store.comparisonState).toBe("ready");
    expect(store.comparisonMetric).toBe("total_ms");
  });

  it("clears and aborts a comparison when the selected baseline trace changes", async () => {
    const pending = deferred<ComparisonResult>();
    const create = vi.spyOn(tracingApi, "createComparison").mockResolvedValue(job());
    const fetchResult = vi.spyOn(tracingApi, "comparison").mockReturnValue(pending.promise);
    const store = prepareComparisonStore();

    const stale = store.compareWithSourceBaseline();
    await vi.waitFor(() => expect(fetchResult).toHaveBeenCalledTimes(1));
    store.selectedBaselineSourceId = OTHER_ID;
    pending.resolve(result());
    await stale;

    expect(store.comparisonState).toBe("idle");
    expect(create.mock.calls[0][1]?.aborted).toBe(true);
    expect(store.comparisonJob).toBeNull();
    expect(store.comparisonResult).toBeNull();
    expect(store.comparisonError).toBe("");
  });

  it.each(["version", "removed"])("invalidates an in-flight comparison after source refresh: %s", async (change) => {
    const pending = deferred<ComparisonJob>();
    const create = vi.spyOn(tracingApi, "createComparison").mockReturnValue(pending.promise);
    const fetchResult = vi.spyOn(tracingApi, "comparison");
    const refreshedSources = change === "version"
      ? [source(CANDIDATE_ID), source(BASELINE_ID, "4".repeat(64)), source(OTHER_ID)]
      : [source(CANDIDATE_ID), source(OTHER_ID)];
    vi.spyOn(tracingApi, "refreshSources").mockResolvedValue({
      items: refreshedSources,
      warnings: [],
      generation: 2,
      refreshed_at: "2026-07-21T00:00:02Z",
    });
    const store = prepareComparisonStore();

    const stale = store.compareWithSourceBaseline();
    await store.refreshSources();
    pending.resolve(job());
    await stale;

    expect(store.comparisonState).toBe("idle");
    expect(create.mock.calls[0][1]?.aborted).toBe(true);
    expect(store.selectedBaselineSourceId).toBe(change === "version" ? BASELINE_ID : OTHER_ID);
    expect(store.comparisonJob).toBeNull();
    expect(store.comparisonResult).toBeNull();
    expect(fetchResult).not.toHaveBeenCalled();
  });

  it.each(["selection", "version"])("does not clear a replacement comparison started in the same tick as baseline %s changes", async (change) => {
    const oldJob = deferred<ComparisonJob>();
    const newJob = deferred<ComparisonJob>();
    const create = vi.spyOn(tracingApi, "createComparison")
      .mockReturnValueOnce(oldJob.promise)
      .mockReturnValueOnce(newJob.promise);
    const store = prepareComparisonStore();

    const stale = store.compareWithSourceBaseline();
    if (change === "selection") store.selectedBaselineSourceId = OTHER_ID;
    else store.selectedBaselineSource!.version = "4".repeat(64);
    const newBaseline = store.selectedBaselineSource!;
    const newResult = {
      ...result(),
      id: "comparison-2",
      baseline_source_id: newBaseline.id,
      baseline_source_version: newBaseline.version,
    };
    const fetchResult = vi.spyOn(tracingApi, "comparison").mockResolvedValue(newResult);
    const current = store.compareWithSourceBaseline();
    await nextTick();
    oldJob.resolve(job());
    await stale;

    expect(store.comparisonState).toBe("loading");
    expect(create.mock.calls[0][1]?.aborted).toBe(true);
    expect(create.mock.calls[1][1]?.aborted).toBe(false);
    expect(create.mock.calls[0][0].baseline_source_version).toBe(BASELINE_VERSION);
    expect(create.mock.calls[1][0].baseline_source_version).toBe(newBaseline.version);
    expect(fetchResult).not.toHaveBeenCalled();

    newJob.resolve({
      ...job("completed", "comparison-2"),
      baseline_source_id: newBaseline.id,
      baseline_source_version: newBaseline.version,
    });
    await current;
    await nextTick();

    expect(store.comparisonState).toBe("ready");
    expect(store.comparisonResult).toEqual(newResult);
    expect(create).toHaveBeenCalledTimes(2);
    expect(fetchResult).toHaveBeenCalledTimes(1);
  });

  it.each(["resolve", "reject"])("ignores a stale comparison result that settles after its replacement: %s", async (outcome) => {
    const pending = deferred<ComparisonResult>();
    const create = vi.spyOn(tracingApi, "createComparison").mockResolvedValue(job());
    const newResult = { ...result(), id: "comparison-2", baseline_source_id: OTHER_ID };
    const fetchResult = vi.spyOn(tracingApi, "comparison")
      .mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce(newResult);
    const store = prepareComparisonStore();

    const stale = store.compareWithSourceBaseline();
    await vi.waitFor(() => expect(fetchResult).toHaveBeenCalledTimes(1));
    store.selectedBaselineSourceId = OTHER_ID;
    await store.compareWithSourceBaseline();
    if (outcome === "resolve") pending.resolve(result());
    else pending.reject(new Error("stale result failed"));
    await stale;
    await nextTick();

    expect(store.comparisonState).toBe("ready");
    expect(store.comparisonResult).toEqual(newResult);
    expect(store.comparisonError).toBe("");
    expect(create.mock.calls[1][1]?.aborted).toBe(false);
    expect(create).toHaveBeenCalledTimes(2);
  });

  it("aborts polling and ignores late progress after baseline invalidation", async () => {
    vi.useFakeTimers();
    try {
      const pending = deferred<ComparisonJob>();
      vi.spyOn(tracingApi, "createComparison").mockResolvedValue(job("queued", null));
      const poll = vi.spyOn(tracingApi, "comparisonJob").mockReturnValue(pending.promise);
      const fetchResult = vi.spyOn(tracingApi, "comparison");
      const store = prepareComparisonStore();

      const stale = store.compareWithSourceBaseline();
      await vi.advanceTimersByTimeAsync(700);
      expect(poll).toHaveBeenCalledTimes(1);
      store.selectedBaselineSourceId = OTHER_ID;
      expect(store.comparisonState).toBe("idle");
      expect(poll.mock.calls[0][1]?.aborted).toBe(true);
      pending.resolve(job("running", null));
      await stale;

      expect(store.comparisonJob).toBeNull();
      expect(store.comparisonState).toBe("idle");
      expect(store.comparisonError).toBe("");
      expect(fetchResult).not.toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("preserves an in-flight comparison across a refresh with unchanged baseline identity", async () => {
    const pending = deferred<ComparisonJob>();
    const create = vi.spyOn(tracingApi, "createComparison").mockReturnValue(pending.promise);
    vi.spyOn(tracingApi, "comparison").mockResolvedValue(result());
    vi.spyOn(tracingApi, "refreshSources").mockResolvedValue({
      items: [source(OTHER_ID, "4".repeat(64)), source(BASELINE_ID), source(CANDIDATE_ID)],
      warnings: [],
      generation: 2,
      refreshed_at: "2026-07-21T00:00:02Z",
    });
    const store = prepareComparisonStore();

    const current = store.compareWithSourceBaseline();
    await store.refreshSources();
    expect(store.comparisonState).toBe("loading");
    expect(create.mock.calls[0][1]?.aborted).toBe(false);
    pending.resolve(job());
    await current;
    store.selection = { kind: "comparison", title: "result", data: {} };
    store.selectedBaselineSource!.version = "5".repeat(64);

    expect(store.comparisonState).toBe("idle");
    expect(store.comparisonResult).toBeNull();
    expect(store.comparisonJob).toBeNull();
    expect(store.selection).toBeNull();
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("requires another trace and preserves the current analysis", async () => {
    const store = prepareComparisonStore();
    store.sources = [source(CANDIDATE_ID)];
    store.selectedBaselineSourceId = "";

    await store.compareWithSourceBaseline();

    expect(store.analysisState).toBe("ready");
    expect(store.comparisonState).toBe("error");
    expect(store.comparisonError).toContain("另一份追踪");
  });

  it("surfaces a failed comparison job without disturbing the analysis", async () => {
    vi.spyOn(tracingApi, "createComparison").mockResolvedValue(job("failed", null));
    const store = prepareComparisonStore();

    await store.compareWithSourceBaseline();

    expect(store.analysisState).toBe("ready");
    expect(store.comparisonState).toBe("error");
    expect(store.comparisonError).toBe("comparison exploded");
  });
});
