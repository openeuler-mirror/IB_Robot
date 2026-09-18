import { describe, expect, it } from "vitest";
import { formatDuration, metricLabel, shortId, stageLabel } from "../src/utils/format";

describe("追踪值格式化", () => {
  it("按数量级显示延迟并保留缺失值", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(0.0005)).toBe("0.5 μs");
    expect(formatDuration(12.345)).toBe("12.35 ms");
    expect(formatDuration(1250)).toBe("1.25 s");
  });

  it("本地化阶段名称并缩短长标识", () => {
    expect(stageLabel("inference_ms")).toBe("模型调用");
    expect(stageLabel("custom_ms")).toBe("custom_ms");
    expect(metricLabel("minimum")).toBe("min");
    expect(metricLabel("maximum")).toBe("max");
    expect(shortId("1234567890abcdef", 8)).toBe("12345678…");
  });
});
