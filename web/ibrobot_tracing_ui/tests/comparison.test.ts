import { describe, expect, it } from "vitest";
import type { FlameDiffEntry } from "../src/api/types";
import {
  buildSharedHistogramGeometry,
  flameDiffBreadcrumb,
  flameDiffMatches,
  layoutFlameDiff,
} from "../src/utils/comparison";

function entry(
  id: string,
  parentId: string,
  baseline: number,
  candidate: number,
  status: FlameDiffEntry["status"] = "unchanged",
): FlameDiffEntry {
  return {
    path_id: id,
    parent_path_id: parentId,
    frame: [`component-${id}`, `span-${id}`, "user"],
    depth: parentId ? 1 : 0,
    baseline_value_ns: baseline,
    candidate_value_ns: candidate,
    baseline_self_value_ns: baseline / 2,
    candidate_self_value_ns: candidate / 2,
    absolute_delta_ns: candidate - baseline,
    percent_delta: baseline ? (candidate - baseline) / baseline * 100 : null,
    self_absolute_delta_ns: (candidate - baseline) / 2,
    self_percent_delta: baseline ? (candidate - baseline) / baseline * 100 : null,
    status,
  };
}

describe("comparison geometry", () => {
  it("overlays baseline and candidate counts against one shared maximum", () => {
    const geometry = buildSharedHistogramGeometry([
      { index: 0, baseline_count: 4, candidate_count: 2 },
      { index: 1, baseline_count: 1, candidate_count: 8 },
    ], 200, 100, 4);

    expect(geometry.maximumCount).toBe(8);
    expect(geometry.bars[0]).toMatchObject({
      x: 2,
      width: 96,
      baselineY: 50,
      baselineHeight: 50,
      candidateY: 75,
      candidateHeight: 25,
    });
    expect(geometry.bars[0].candidateX).toBeGreaterThan(geometry.bars[0].x);
    expect(geometry.bars[0].candidateWidth).toBeLessThan(geometry.bars[0].width);
    expect(geometry.bars[1].candidateHeight).toBe(100);
  });

  it("returns stable empty histogram geometry", () => {
    expect(buildSharedHistogramGeometry([], 300, 120)).toEqual({
      width: 300,
      height: 120,
      maximumCount: 0,
      bars: [],
    });
  });

  it("partitions full paths by max baseline or candidate weight", () => {
    const entries = [
      entry("root-a", "", 100, 40),
      entry("child-a", "root-a", 30, 60, "regressed"),
      entry("child-b", "root-a", 20, 10, "improved"),
      entry("root-b", "", 20, 100, "added"),
    ];
    const layout = layoutFlameDiff(entries, 800);
    const rootA = layout.rects.find((rect) => rect.entry.path_id === "root-a")!;
    const rootB = layout.rects.find((rect) => rect.entry.path_id === "root-b")!;
    const children = layout.rects.filter((rect) => rect.depth === 1).sort((left, right) => left.x - right.x);

    expect(rootA.width).toBeCloseTo(400);
    expect(rootB.width).toBeCloseTo(400);
    expect(children[0].x).toBeGreaterThanOrEqual(rootA.x);
    expect(children[1].x + children[1].width).toBeLessThanOrEqual(rootA.x + rootA.width);
    expect(children[0].width).toBeCloseTo(240);
    expect(children[1].width).toBeCloseTo(80);
  });

  it("localizes depth when focusing and keeps icicle orientation", () => {
    const entries = [
      entry("root", "", 100, 100),
      entry("child", "root", 60, 80),
      entry("leaf", "child", 20, 30),
    ];
    const focused = layoutFlameDiff(entries, 600, "child");

    expect(focused.rects.map((rect) => [rect.entry.path_id, rect.depth])).toEqual([["child", 0], ["leaf", 1]]);
    expect(focused.rects[0].width).toBe(600);
    expect(focused.rects[0].y).toBe(0);
    expect(focused.rects[1].y).toBeGreaterThan(focused.rects[0].y);
  });

  it("reconstructs breadcrumbs, tolerates parent cycles, and searches frame status", () => {
    const entries = [
      entry("root", "", 100, 100),
      entry("child", "root", 50, 70, "regressed"),
      entry("cycle-a", "cycle-b", 10, 10),
      entry("cycle-b", "cycle-a", 10, 10),
    ];

    expect(flameDiffBreadcrumb(entries, "child").map((item) => item.path_id)).toEqual(["root", "child"]);
    expect(layoutFlameDiff(entries, 500).rects).toHaveLength(entries.length);
    expect(flameDiffMatches(entries[1], "component-child regressed")).toBe(true);
    expect(flameDiffMatches(entries[1], "improved")).toBe(false);
  });
});
