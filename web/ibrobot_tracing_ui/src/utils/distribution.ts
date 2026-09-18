import type { LatencyDistributionBucket } from "../api/types";

export interface HistogramBar {
  index: number;
  x: number;
  y: number;
  width: number;
  height: number;
  count: number;
}

export interface HistogramGeometry {
  width: number;
  height: number;
  maximumCount: number;
  bars: HistogramBar[];
}

export function buildHistogramGeometry(
  buckets: readonly Pick<LatencyDistributionBucket, "index" | "count">[],
  width: number,
  height: number,
  gap = 2,
): HistogramGeometry {
  const safeWidth = Math.max(0, Number.isFinite(width) ? width : 0);
  const safeHeight = Math.max(0, Number.isFinite(height) ? height : 0);
  const maximumCount = Math.max(0, ...buckets.map((bucket) => bucket.count));
  if (!buckets.length || safeWidth === 0 || safeHeight === 0) {
    return { width: safeWidth, height: safeHeight, maximumCount, bars: [] };
  }

  const slotWidth = safeWidth / buckets.length;
  const safeGap = Math.min(Math.max(0, gap), slotWidth);
  const barWidth = Math.max(0, slotWidth - safeGap);
  const bars = buckets.map((bucket, position) => {
    const barHeight = maximumCount > 0 ? Math.max(0, bucket.count) * safeHeight / maximumCount : 0;
    return {
      index: bucket.index,
      x: position * slotWidth + safeGap / 2,
      y: safeHeight - barHeight,
      width: barWidth,
      height: barHeight,
      count: bucket.count,
    };
  });
  return { width: safeWidth, height: safeHeight, maximumCount, bars };
}

export function valuePosition(value: number, minimum: number, maximum: number, width: number): number {
  if (![value, minimum, maximum, width].every(Number.isFinite) || width <= 0) return 0;
  if (minimum === maximum) return width / 2;
  const ratio = (value - minimum) / (maximum - minimum);
  return Math.min(width, Math.max(0, ratio * width));
}
