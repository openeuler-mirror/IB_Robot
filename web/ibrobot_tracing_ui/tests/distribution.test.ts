import { describe, expect, it } from "vitest";
import { buildHistogramGeometry, valuePosition } from "../src/utils/distribution";

describe("distribution geometry", () => {
  it("returns no bars for an empty distribution", () => {
    expect(buildHistogramGeometry([], 300, 120)).toEqual({
      width: 300,
      height: 120,
      maximumCount: 0,
      bars: [],
    });
  });

  it("scales bars against the largest bucket", () => {
    const geometry = buildHistogramGeometry([
      { index: 0, count: 2 },
      { index: 1, count: 4 },
      { index: 2, count: 0 },
    ], 300, 100, 2);

    expect(geometry.maximumCount).toBe(4);
    expect(geometry.bars).toHaveLength(3);
    expect(geometry.bars[0]).toMatchObject({ x: 1, y: 50, width: 98, height: 50 });
    expect(geometry.bars[1]).toMatchObject({ x: 101, y: 0, width: 98, height: 100 });
    expect(geometry.bars[2]).toMatchObject({ y: 100, height: 0 });
  });

  it("centers markers when all values are equal", () => {
    expect(valuePosition(4.5, 4.5, 4.5, 240)).toBe(120);
    expect(buildHistogramGeometry([{ index: 0, count: 3 }], 240, 80).bars[0].height).toBe(80);
  });

  it("positions negative and positive values in the same domain", () => {
    expect(valuePosition(-10, -10, 10, 200)).toBe(0);
    expect(valuePosition(-5, -10, 10, 200)).toBe(50);
    expect(valuePosition(0, -10, 10, 200)).toBe(100);
    expect(valuePosition(10, -10, 10, 200)).toBe(200);
  });
});
