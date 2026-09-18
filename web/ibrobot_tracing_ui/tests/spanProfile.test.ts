import { describe, expect, it } from "vitest";
import type { AggregateSpanProfileNode, RequestSpanProfileNode } from "../src/api/types";
import {
  aggregateBreadcrumb,
  layoutAggregateProfile,
  layoutRequestProfile,
  profileNodeKeyAction,
  profileNodeMatches,
} from "../src/utils/spanProfile";

function requestNode(
  id: string,
  start: number,
  duration: number,
  depth = 0,
  parentId = "",
): RequestSpanProfileNode {
  return {
    id,
    occurrence_id: id,
    span_id: id,
    request_id: "request",
    parent_id: parentId,
    parent_span_id: parentId,
    child_ids: [],
    depth,
    track: 0,
    component_id: `component-${id}`,
    name: id,
    origin: "user",
    status: "ok",
    start_ns: String(start),
    end_ns: String(start + duration),
    observed_end_ns: String(start + duration),
    start_offset_ns: start,
    duration_ns: duration,
    duration_source: "observed",
    uncovered_wall_ns: duration,
    diagnostics: [],
    fields: {},
  };
}

function aggregateNode(
  id: string,
  value: number,
  selfValue: number,
  parentId = "",
  childIds: string[] = [],
): AggregateSpanProfileNode {
  return {
    id,
    parent_id: parentId,
    child_ids: childIds,
    depth: parentId ? 1 : 0,
    component_id: `component-${id}`,
    name: id,
    origin: "user",
    path: [{ component_id: `component-${id}`, name: id, origin: "user" }],
    self_value_ns: selfValue,
    value_ns: value,
    occurrence_count: 1,
    request_count: 1,
    error_count: 0,
    incomplete_count: 0,
    invalid_count: 0,
    percentage: value,
  };
}

describe("Span 剖析几何", () => {
  it("在每个逻辑深度全局打包重叠 Span，包括多个根", () => {
    const layout = layoutRequestProfile([
      requestNode("root-a", 0, 70),
      requestNode("root-b", 20, 50),
      requestNode("root-c", 75, 20),
      requestNode("child", 10, 20, 1, "root-a"),
    ], 100, 1000);
    const roots = layout.rects.filter((rect) => rect.depth === 0);
    const rootA = roots.find((rect) => rect.node.id === "root-a")!;
    const rootB = roots.find((rect) => rect.node.id === "root-b")!;
    const rootC = roots.find((rect) => rect.node.id === "root-c")!;

    expect(rootA.lane).not.toBe(rootB.lane);
    expect(rootC.lane).toBe(rootA.lane);
    expect(layout.depthBands).toEqual([
      expect.objectContaining({ depth: 0, laneCount: 2 }),
      expect.objectContaining({ depth: 1, laneCount: 1 }),
    ]);
    for (const lane of new Set(roots.map((rect) => rect.lane))) {
      const items = roots.filter((rect) => rect.lane === lane).sort((left, right) => left.x - right.x);
      items.slice(1).forEach((item, index) => expect(items[index].x + items[index].width).toBeLessThanOrEqual(item.x));
    }
  });

  it("将尾部零时长或未完成 Span 保持在绘图区内且至少三像素", () => {
    const complete = requestNode("complete", 0, 50);
    const incomplete = requestNode("incomplete", 100, 0);
    incomplete.status = "incomplete";
    incomplete.end_ns = null;
    incomplete.duration_ns = null;
    incomplete.uncovered_wall_ns = null;

    const layout = layoutRequestProfile([complete, incomplete], 100, 100);
    const trailing = layout.rects.find((rect) => rect.node.id === "incomplete")!;

    expect(trailing.width).toBe(3);
    expect(trailing.x).toBe(97);
    expect(trailing.x + trailing.width).toBeLessThanOrEqual(100);
  });

  it("使用堆为大量同点 Span 分配稳定通道", () => {
    const count = 1_500;
    const layout = layoutRequestProfile(
      Array.from({ length: count }, (_, index) => requestNode(`zero-${index}`, 0, 0)),
      0,
      600,
    );

    expect(layout.rects).toHaveLength(count);
    expect(layout.depthBands[0].laneCount).toBe(count);
    expect(new Set(layout.rects.map((rect) => rect.lane)).size).toBe(count);
  });

  it("按包含权重稳定分区根和子项且不重叠", () => {
    const nodes = [
      aggregateNode("root", 100, 40, "", ["small", "large"]),
      aggregateNode("small", 20, 20, "root"),
      aggregateNode("large", 40, 40, "root"),
      aggregateNode("second-root", 50, 50),
    ];
    const layout = layoutAggregateProfile(nodes, ["second-root", "root"], 900);
    const root = layout.rects.find((rect) => rect.node.id === "root")!;
    const secondRoot = layout.rects.find((rect) => rect.node.id === "second-root")!;
    const children = layout.rects.filter((rect) => rect.node.parent_id === "root").sort((left, right) => left.x - right.x);

    expect(root.x).toBe(0);
    expect(root.width).toBeCloseTo(600);
    expect(secondRoot.x).toBeCloseTo(600);
    expect(secondRoot.x + secondRoot.width).toBeCloseTo(900);
    expect(children[0].x).toBeGreaterThanOrEqual(root.x);
    expect(children[0].x + children[0].width).toBeLessThanOrEqual(children[1].x);
    expect(children[1].x + children[1].width).toBeLessThanOrEqual(root.x + root.width);
    expect(root.selfWidth).toBeCloseTo(240);
  });

  it("固定使用根在上、子项在下的冰柱方向", () => {
    const layout = layoutRequestProfile([requestNode("root", 0, 100), requestNode("child", 10, 20, 1)], 100, 500);
    const root = layout.rects.find((rect) => rect.node.id === "root")!;
    const child = layout.rects.find((rect) => rect.node.id === "child")!;

    expect(root.y).toBe(0);
    expect(child.y).toBeGreaterThan(root.y);
  });

  it("聚焦子树并生成可返回根的面包屑", () => {
    const nodes = [
      aggregateNode("root", 100, 50, "", ["child"]),
      aggregateNode("child", 50, 30, "root", ["leaf"]),
      aggregateNode("leaf", 20, 20, "child"),
      aggregateNode("other", 25, 25),
    ];
    const layout = layoutAggregateProfile(nodes, ["root", "other"], 800, "child");

    expect(layout.rects.map((rect) => rect.node.id)).toEqual(["child", "leaf"]);
    expect(layout.rects[0]).toEqual(expect.objectContaining({ x: 0, width: 800, depth: 0 }));
    expect(aggregateBreadcrumb(nodes, "leaf").map((node) => node.id)).toEqual(["root", "child", "leaf"]);
  });

  it("迭代布局超过一万层的聚合子树", () => {
    const count = 10_050;
    const nodes = Array.from({ length: count }, (_, index) => {
      const id = String(index);
      const node = aggregateNode(
        id,
        1,
        index === count - 1 ? 1 : 0,
        index ? String(index - 1) : "",
        index + 1 < count ? [String(index + 1)] : [],
      );
      node.depth = index;
      return node;
    });

    const layout = layoutAggregateProfile(nodes, ["0"], 600, "0");

    expect(layout.rects).toHaveLength(count);
    expect(layout.rects.at(-1)?.depth).toBe(count - 1);
  });

  it("将键盘输入映射为检查和聚焦动作", () => {
    expect(profileNodeKeyAction("Enter", false, false)).toBe("inspect");
    expect(profileNodeKeyAction(" ", false, true)).toBe("inspect");
    expect(profileNodeKeyAction("Enter", true, true)).toBe("focus");
    expect(profileNodeKeyAction("Enter", true, false)).toBe("inspect");
    expect(profileNodeKeyAction("Escape", false, true)).toBeNull();
  });

  it("搜索名称、组件、来源、状态、诊断和聚合路径", () => {
    const request = requestNode("work", 0, 10);
    request.status = "error";
    request.diagnostics = ["overlapping_siblings"];
    const aggregate = aggregateNode("leaf", 10, 10, "root");
    aggregate.path = [{ component_id: "camera", name: "capture", origin: "built-in" }];

    expect(profileNodeMatches(request, "work error")).toBe(true);
    expect(profileNodeMatches(request, "overlapping_siblings")).toBe(true);
    expect(profileNodeMatches(aggregate, "camera built-in")).toBe(true);
    expect(profileNodeMatches(aggregate, "missing")).toBe(false);
    expect(profileNodeMatches(aggregate, "")).toBe(true);
  });
});
