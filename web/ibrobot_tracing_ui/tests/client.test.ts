import { afterEach, describe, expect, it, vi } from "vitest";
import { pollComparisonJob, pollJob, tracingApi } from "../src/api/client";
import type { ComparisonJob, LoadJob } from "../src/api/types";

afterEach(() => vi.restoreAllMocks());

describe("追踪接口客户端", () => {
  it("对服务端错误生成带状态码的异常", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ message: "追踪不存在" }), { status: 404 })));
    await expect(tracingApi.summary("missing")).rejects.toEqual(expect.objectContaining({ status: 404, message: "追踪不存在" }));
  });

  it("持续轮询直到分析完成", async () => {
    const base = {
      id: "job-1",
      source_id: "source-1",
      source_version: "version-1",
      created_at: "2026-07-15T00:00:00Z",
      started_at: null,
      finished_at: null,
      analysis_id: null,
      error: null,
      cancel_requested: false,
      queue_position: null,
      deduplicated: false,
    };
    const running: LoadJob = { ...base, status: "running" };
    const completed: LoadJob = { ...base, status: "completed", analysis_id: "analysis-1" };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(completed), { status: 200 })));
    const updates: string[] = [];
    const result = await pollJob(running, (job) => updates.push(job.status), undefined, 0);
    expect(result.analysis_id).toBe("analysis-1");
    expect(updates).toEqual(["running", "completed"]);
  });

  it("使用后端定义的任务与图投影端点", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "job-1" }), { status: 202 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ metric: "p95", view: "components", request_id: "", topology_source: "manifest", nodes: [], edges: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await tracingApi.createAnalysis("source-1");
    await tracingApi.topology("analysis-1", { metric: "p95", view: "components" });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/analysis-jobs");
    expect(fetchMock.mock.calls[0][1].body).toBe(JSON.stringify({ source_id: "source-1" }));
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/analyses/analysis-1/graph?metric=p95&view=components");
  });

  it("使用带分页参数的埋点目录端点", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ items: [], total: 0, offset: 1000, limit: 1000, next_offset: null }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await tracingApi.tracepoints("analysis/1", { offset: 1000, limit: 1000 });

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/analyses/analysis%2F1/tracepoints?offset=1000&limit=1000");
  });

  it("分布端点允许省略指标以使用后端默认值", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ metric: "inference_ms" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await tracingApi.distribution("analysis/1");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/analyses/analysis%2F1/distribution");
  });

  it("为分布端点构造严格且转义的筛选 URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ metric: "custom latency_ms" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await tracingApi.distribution("analysis/1", {
      metric: "custom latency_ms",
      bins: 18,
      outlierLimit: 7,
      bucketRequestLimit: 40,
    });

    expect(fetchMock.mock.calls[0][0]).toBe(
      "/api/v1/analyses/analysis%2F1/distribution?metric=custom+latency_ms&bins=18&outlier_limit=7&bucket_request_limit=40",
    );
  });

  it("为两种 Span 剖析模式构造严格 URL 并默认限制一万个节点", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ mode: "request" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await tracingApi.spanProfile("analysis/1", {
      mode: "request",
      requestId: "request 1",
      componentId: "worker",
      startNs: "9007199254741000",
      status: "error",
    });
    await tracingApi.spanProfile("analysis/1", { mode: "aggregate", origin: "user", maxNodes: 250 });

    expect(fetchMock.mock.calls[0][0]).toBe(
      "/api/v1/analyses/analysis%2F1/span-profile?mode=request&request_id=request+1&component_id=worker&start_ns=9007199254741000&status=error&max_nodes=10000",
    );
    expect(fetchMock.mock.calls[1][0]).toBe(
      "/api/v1/analyses/analysis%2F1/span-profile?mode=aggregate&origin=user&max_nodes=250",
    );
  });

  it("为关键路径构造转义查询且不发送任何文件路径", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ method: "deepest_active_wall_partition" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await tracingApi.criticalPath("analysis/1", {
      requestId: "request / 1",
      componentId: "worker/arm",
    });

    expect(fetchMock.mock.calls[0][0]).toBe(
      "/api/v1/analyses/analysis%2F1/critical-path?request_id=request+%2F+1&component_id=worker%2Farm&include_flows=true&max_segments=1000",
    );
    expect(fetchMock.mock.calls[0][0]).not.toContain("path=");
  });

  it("创建比较时只发送基线 trace 和当前分析标识，不发送路径", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: "job-1" }), { status: 202 }));
    vi.stubGlobal("fetch", fetchMock);
    const payload = {
      baseline_source_id: "a".repeat(32),
      baseline_source_version: "b".repeat(64),
      candidate_analysis_id: "analysis-1",
      statistic: "p95" as const,
      relative_threshold_percent: 10,
      absolute_threshold_ms: 1,
      bins: 30,
      metric: "total_ms",
    };

    await tracingApi.createComparison(payload);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/comparison-jobs");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual(payload);
    expect(fetchMock.mock.calls[0][1].body).not.toContain("path");
  });

  it("持续轮询比较任务并读取及删除编码后的结果", async () => {
    const base: ComparisonJob = {
      id: "job/1",
      baseline_source_id: "a".repeat(32),
      baseline_source_version: "b".repeat(64),
      candidate_analysis_id: "analysis-1",
      statistic: "p95",
      relative_threshold_percent: 10,
      absolute_threshold_ms: 1,
      bins: 30,
      metric: null,
      status: "running",
      created_at: "2026-07-21T00:00:00Z",
      started_at: null,
      finished_at: null,
      comparison_id: null,
      error: null,
      queue_position: null,
      deduplicated: false,
    };
    const completed: ComparisonJob = { ...base, status: "completed", comparison_id: "result/1" };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(completed), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "result/1" }), { status: 200 }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const updates: string[] = [];

    const result = await pollComparisonJob(base, (job) => updates.push(job.status), undefined, 0);
    await tracingApi.comparison(result.comparison_id!);
    await tracingApi.deleteComparison(result.comparison_id!);

    expect(updates).toEqual(["running", "completed"]);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/comparison-jobs/job%2F1");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/comparisons/result%2F1");
    expect(fetchMock.mock.calls[2][0]).toBe("/api/v1/comparisons/result%2F1");
    expect(fetchMock.mock.calls[2][1].method).toBe("DELETE");
  });
});
