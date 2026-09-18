import { describe, expect, it } from "vitest";
import type { CriticalPathSegment } from "../src/api/types";
import { buildCriticalPathGeometry } from "../src/utils/criticalPath";

function segment(id: string, index: number, offset: number, duration: number): CriticalPathSegment {
  return {
    id,
    index,
    kind: "span",
    source_id: id,
    label: id,
    component_id: "worker",
    edge_id: "",
    origin: "user",
    status: "ok",
    start_ns: offset,
    end_ns: offset + duration,
    offset_ns: offset,
    duration_ns: duration,
    percentage: duration,
    diagnostics: [],
    fields: {},
  };
}

describe("critical path geometry", () => {
  it("orders segments chronologically and keeps visual partitions non-overlapping", () => {
    const geometry = buildCriticalPathGeometry([
      segment("second", 1, 40, 40),
      segment("first", 0, 0, 40),
      segment("overlap", 2, 70, 20),
    ], 100, 90, 0, 1_000);

    expect(geometry.rects.map((rect) => rect.segment.id)).toEqual(["first", "second", "overlap"]);
    expect(geometry.rects[0]).toMatchObject({ x: 0, width: 400 });
    expect(geometry.rects[1]).toMatchObject({ x: 400, width: 400 });
    expect(geometry.rects[2]).toMatchObject({ x: 800, width: 100 });
    geometry.rects.slice(1).forEach((rect, index) => {
      expect(geometry.rects[index].x + geometry.rects[index].width).toBeLessThanOrEqual(rect.x);
    });
  });

  it("reserves an exact proportional omitted tail instead of stretching the returned prefix", () => {
    const geometry = buildCriticalPathGeometry([segment("prefix", 0, 0, 30)], 100, 30, 70, 1_000);

    expect(geometry.rects[0]).toMatchObject({ x: 0, width: 300 });
    expect(geometry.omitted).toEqual({ x: 300, width: 700 });
  });

  it("returns safe empty geometry for empty and zero-duration paths", () => {
    expect(buildCriticalPathGeometry([], 0, 0, 0, 800)).toEqual({
      width: 800,
      durationNs: 0,
      rects: [],
      omitted: null,
    });
  });
});
