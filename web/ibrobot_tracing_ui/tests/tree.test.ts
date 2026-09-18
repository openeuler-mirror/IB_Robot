import { describe, expect, it } from "vitest";
import type { SpanRecord } from "../src/api/types";
import { buildSpanForest, flattenSpanForest } from "../src/utils/tree";

function span(id: string, parent: string, start: number): SpanRecord {
  return {
    id: `span:${id}`,
    name: id,
    trace_id: "请求-1",
    span_id: id,
    parent_span_id: parent,
    component_id: "policy",
    start_ns: start,
    end_ns: start + 10,
    duration_ms: 0.01,
    status: "ok",
    origin: "built-in",
    fields: {},
  };
}

describe("调用树构建", () => {
  const spans = [span("子-2", "根", 30), span("孤立", "缺失父项", 5), span("根", "", 10), span("子-1", "根", 20)];

  it("按父标识分组，并按起始时间排序", () => {
    const forest = buildSpanForest(spans);
    expect(forest.map((node) => node.span.span_id)).toEqual(["孤立", "根"]);
    expect(forest[1].children.map((node) => node.span.span_id)).toEqual(["子-1", "子-2"]);
  });

  it("折叠节点时跳过其后代", () => {
    const forest = buildSpanForest(spans);
    expect(flattenSpanForest(forest, new Set()).map((node) => node.span.span_id)).toEqual(["孤立", "根", "子-1", "子-2"]);
    expect(flattenSpanForest(forest, new Set(["根"])).map((node) => node.span.span_id)).toEqual(["孤立", "根"]);
  });
});
