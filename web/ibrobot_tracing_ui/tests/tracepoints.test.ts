import { describe, expect, it } from "vitest";
import type { InspectorSelection, TracepointDefinition } from "../src/api/types";
import { resolveTracepointDescription } from "../src/utils/tracepoints";

const definitions: TracepointDefinition[] = [
  { id: "event-built-in", kind: "event", component_id: "camera", name: "frame", origin: "built-in", description: "相机帧到达。" },
  { id: "event-user", kind: "event", component_id: "worker", name: "ready", origin: "user", description: "工作器就绪。" },
  { id: "span-user", kind: "span", component_id: "worker", name: "infer", origin: "user", description: "执行一次推理。" },
];

function selection(kind: InspectorSelection["kind"], data: Record<string, unknown>): InspectorSelection {
  return { kind, title: "selection", data };
}

describe("埋点说明解析", () => {
  it("直接使用图节点说明", () => {
    expect(resolveTracepointDescription(selection("component", { description: "图节点说明。" }), definitions)).toBe("图节点说明。");
  });

  it("按事件、Span 和时间线项目的完整身份匹配", () => {
    const event = selection("event", { component_id: "camera", name: "frame", fields: {} });
    const span = selection("span", { component_id: "worker", name: "infer", origin: "user", fields: { origin: "built-in" } });
    const timelineEvent = selection("event", { kind: "event", component_id: "worker", label: "ready", fields: { origin: "user" } });
    const timelineSpan = selection("span", { kind: "span", component_id: "worker", label: "infer", fields: { origin: "user" } });

    expect(resolveTracepointDescription(event, definitions)).toBe("相机帧到达。");
    expect(resolveTracepointDescription(span, definitions)).toBe("执行一次推理。");
    expect(resolveTracepointDescription(timelineEvent, definitions)).toBe("工作器就绪。");
    expect(resolveTracepointDescription(timelineSpan, definitions)).toBe("执行一次推理。");
  });

  it("身份不完整或不匹配时省略说明", () => {
    expect(resolveTracepointDescription(selection("event", { component_id: "worker", name: "missing", fields: {} }), definitions)).toBe("");
    expect(resolveTracepointDescription(selection("flow", { kind: "flow", label: "edge" }), definitions)).toBe("");
  });
});
