import type { CriticalPathSegment, JsSafeInteger } from "../api/types";
import { numericNs } from "./spanProfile";

export interface CriticalPathRect {
  segment: CriticalPathSegment;
  x: number;
  width: number;
}

export interface CriticalPathOmittedRect {
  x: number;
  width: number;
}

export interface CriticalPathGeometry {
  width: number;
  durationNs: number;
  rects: CriticalPathRect[];
  omitted: CriticalPathOmittedRect | null;
}

function clamp(value: number, maximum: number): number {
  return Math.min(maximum, Math.max(0, value));
}

export function buildCriticalPathGeometry(
  segments: CriticalPathSegment[],
  durationNs: JsSafeInteger,
  returnedDurationNs: JsSafeInteger,
  omittedDurationNs: JsSafeInteger,
  width: number,
): CriticalPathGeometry {
  const chartWidth = Math.max(0, width);
  const duration = Math.max(0, numericNs(durationNs));
  if (!duration || !chartWidth) return { width: chartWidth, durationNs: duration, rects: [], omitted: null };

  let previousEnd = 0;
  const rects = [...segments]
    .sort((left, right) => numericNs(left.offset_ns) - numericNs(right.offset_ns) || left.index - right.index)
    .map((segment) => {
      const sourceStart = clamp(numericNs(segment.offset_ns), duration);
      const sourceEnd = clamp(sourceStart + Math.max(0, numericNs(segment.duration_ns)), duration);
      const start = Math.max(previousEnd, sourceStart);
      const end = Math.max(start, sourceEnd);
      previousEnd = end;
      return {
        segment,
        x: start / duration * chartWidth,
        width: (end - start) / duration * chartWidth,
      };
    });

  const omittedDuration = clamp(numericNs(omittedDurationNs), duration);
  const omittedStart = clamp(numericNs(returnedDurationNs), duration);
  const omittedEnd = clamp(omittedStart + omittedDuration, duration);
  const omitted = omittedDuration > 0
    ? { x: omittedStart / duration * chartWidth, width: (omittedEnd - omittedStart) / duration * chartWidth }
    : null;
  return { width: chartWidth, durationNs: duration, rects, omitted };
}
