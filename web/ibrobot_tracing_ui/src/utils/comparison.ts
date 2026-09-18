import type {
  FlameDiffEntry,
  SharedHistogramBucket,
} from "../api/types";
import { numericNs } from "./spanProfile";

export interface SharedHistogramBar {
  index: number;
  x: number;
  width: number;
  baselineY: number;
  baselineHeight: number;
  candidateX: number;
  candidateWidth: number;
  candidateY: number;
  candidateHeight: number;
}

export interface SharedHistogramGeometry {
  width: number;
  height: number;
  maximumCount: number;
  bars: SharedHistogramBar[];
}

export interface FlameDiffRect {
  entry: FlameDiffEntry;
  x: number;
  y: number;
  width: number;
  height: number;
  depth: number;
  baselineSelfWidth: number;
  candidateSelfWidth: number;
}

export interface FlameDiffLayout {
  rects: FlameDiffRect[];
  contentHeight: number;
  depthCount: number;
  rootIds: string[];
}

function safeCount(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function entryWeight(entry: FlameDiffEntry): number {
  return Math.max(0, numericNs(entry.baseline_value_ns), numericNs(entry.candidate_value_ns));
}

function entryOrder(left: FlameDiffEntry, right: FlameDiffEntry): number {
  return entryWeight(right) - entryWeight(left)
    || compareText(left.frame.join("\u0000"), right.frame.join("\u0000"))
    || compareText(left.path_id, right.path_id);
}

export function buildSharedHistogramGeometry(
  buckets: readonly Pick<SharedHistogramBucket, "index" | "baseline_count" | "candidate_count">[],
  width: number,
  height: number,
  gap = 3,
): SharedHistogramGeometry {
  const safeWidth = Number.isFinite(width) ? Math.max(0, width) : 0;
  const safeHeight = Number.isFinite(height) ? Math.max(0, height) : 0;
  const maximumCount = Math.max(
    0,
    ...buckets.flatMap((bucket) => [safeCount(bucket.baseline_count), safeCount(bucket.candidate_count)]),
  );
  if (!buckets.length || safeWidth === 0 || safeHeight === 0) {
    return { width: safeWidth, height: safeHeight, maximumCount, bars: [] };
  }

  const slotWidth = safeWidth / buckets.length;
  const safeGap = Math.min(Math.max(0, gap), slotWidth);
  const widthWithoutGap = Math.max(0, slotWidth - safeGap);
  const bars = buckets.map((bucket, position) => {
    const baselineHeight = maximumCount ? safeCount(bucket.baseline_count) / maximumCount * safeHeight : 0;
    const candidateHeight = maximumCount ? safeCount(bucket.candidate_count) / maximumCount * safeHeight : 0;
    const x = position * slotWidth + safeGap / 2;
    return {
      index: bucket.index,
      x,
      width: widthWithoutGap,
      baselineY: safeHeight - baselineHeight,
      baselineHeight,
      candidateX: x + widthWithoutGap * 0.2,
      candidateWidth: widthWithoutGap * 0.6,
      candidateY: safeHeight - candidateHeight,
      candidateHeight,
    };
  });
  return { width: safeWidth, height: safeHeight, maximumCount, bars };
}

function childMap(entries: FlameDiffEntry[]): Map<string, FlameDiffEntry[]> {
  const byId = new Map(entries.map((entry) => [entry.path_id, entry]));
  const children = new Map<string, FlameDiffEntry[]>();
  entries.forEach((entry) => {
    if (!entry.parent_path_id || entry.parent_path_id === entry.path_id || !byId.has(entry.parent_path_id)) return;
    const group = children.get(entry.parent_path_id) ?? [];
    group.push(entry);
    children.set(entry.parent_path_id, group);
  });
  children.forEach((group) => group.sort(entryOrder));
  return children;
}

function descendants(children: Map<string, FlameDiffEntry[]>, rootId: string): Set<string> {
  const result = new Set<string>();
  const stack = [rootId];
  while (stack.length) {
    const id = stack.pop()!;
    if (result.has(id)) continue;
    result.add(id);
    const childEntries = children.get(id) ?? [];
    for (let index = childEntries.length - 1; index >= 0; index -= 1) stack.push(childEntries[index].path_id);
  }
  return result;
}

function reconstructedRoots(entries: FlameDiffEntry[], children: Map<string, FlameDiffEntry[]>): FlameDiffEntry[] {
  const byId = new Map(entries.map((entry) => [entry.path_id, entry]));
  const roots = entries.filter((entry) => (
    !entry.parent_path_id
    || entry.parent_path_id === entry.path_id
    || !byId.has(entry.parent_path_id)
  )).sort(entryOrder);
  const covered = new Set<string>();
  roots.forEach((root) => descendants(children, root.path_id).forEach((id) => covered.add(id)));
  entries.sort(entryOrder).forEach((entry) => {
    if (covered.has(entry.path_id)) return;
    roots.push(entry);
    descendants(children, entry.path_id).forEach((id) => covered.add(id));
  });
  return roots;
}

export function layoutFlameDiff(
  sourceEntries: FlameDiffEntry[],
  width: number,
  focusId = "",
  rowHeight = 27,
): FlameDiffLayout {
  const safeWidth = Number.isFinite(width) ? Math.max(0, width) : 0;
  const entries = [...sourceEntries];
  const byId = new Map(entries.map((entry) => [entry.path_id, entry]));
  const children = childMap(entries);
  const focused = focusId && byId.has(focusId) ? byId.get(focusId)! : null;
  const visible = focused ? descendants(children, focused.path_id) : new Set(entries.map((entry) => entry.path_id));
  const roots = (focused ? [focused] : reconstructedRoots(entries, children))
    .filter((entry) => visible.has(entry.path_id))
    .sort(entryOrder);
  const rects: FlameDiffRect[] = [];
  const rootTotal = roots.reduce((sum, entry) => sum + entryWeight(entry), 0);
  let rootX = 0;
  roots.forEach((entry, index) => {
    const desiredWidth = rootTotal > 0 ? safeWidth * entryWeight(entry) / rootTotal : safeWidth / Math.max(1, roots.length);
    const rectWidth = index === roots.length - 1 ? safeWidth - rootX : Math.min(safeWidth - rootX, desiredWidth);
    const weight = entryWeight(entry);
    rects.push({
      entry,
      x: rootX,
      y: 0,
      width: Math.max(0, rectWidth),
      height: Math.max(0, rowHeight - 2),
      depth: 0,
      baselineSelfWidth: weight ? rectWidth * Math.max(0, numericNs(entry.baseline_self_value_ns)) / weight : 0,
      candidateSelfWidth: weight ? rectWidth * Math.max(0, numericNs(entry.candidate_self_value_ns)) / weight : 0,
    });
    rootX += Math.max(0, rectWidth);
  });

  const visited = new Set<string>();
  const stack = [...rects].reverse();
  while (stack.length) {
    const parent = stack.pop()!;
    if (visited.has(parent.entry.path_id)) continue;
    visited.add(parent.entry.path_id);
    const childEntries = (children.get(parent.entry.path_id) ?? []).filter((entry) => (
      visible.has(entry.path_id) && !visited.has(entry.path_id)
    ));
    const childTotal = childEntries.reduce((sum, entry) => sum + entryWeight(entry), 0);
    const denominator = Math.max(entryWeight(parent.entry), childTotal);
    let x = parent.x;
    const childRects: FlameDiffRect[] = [];
    childEntries.forEach((entry, index) => {
      const desiredWidth = denominator > 0
        ? parent.width * entryWeight(entry) / denominator
        : parent.width / Math.max(1, childEntries.length);
      const fillsParent = denominator === childTotal && index === childEntries.length - 1;
      const rectWidth = Math.max(0, Math.min(parent.x + parent.width - x, fillsParent ? parent.x + parent.width - x : desiredWidth));
      const weight = entryWeight(entry);
      const rect: FlameDiffRect = {
        entry,
        x,
        y: parent.y + rowHeight,
        width: rectWidth,
        height: Math.max(0, rowHeight - 2),
        depth: parent.depth + 1,
        baselineSelfWidth: weight ? rectWidth * Math.max(0, numericNs(entry.baseline_self_value_ns)) / weight : 0,
        candidateSelfWidth: weight ? rectWidth * Math.max(0, numericNs(entry.candidate_self_value_ns)) / weight : 0,
      };
      rects.push(rect);
      childRects.push(rect);
      x += rectWidth;
    });
    for (let index = childRects.length - 1; index >= 0; index -= 1) stack.push(childRects[index]);
  }

  const depthCount = rects.reduce((maximum, rect) => Math.max(maximum, rect.depth + 1), 0);
  return {
    rects: rects.sort((left, right) => left.depth - right.depth || left.x - right.x || compareText(left.entry.path_id, right.entry.path_id)),
    contentHeight: depthCount * rowHeight,
    depthCount,
    rootIds: roots.map((entry) => entry.path_id),
  };
}

export function flameDiffBreadcrumb(entries: FlameDiffEntry[], focusId: string): FlameDiffEntry[] {
  if (!focusId) return [];
  const byId = new Map(entries.map((entry) => [entry.path_id, entry]));
  const result: FlameDiffEntry[] = [];
  const visited = new Set<string>();
  let current = byId.get(focusId);
  while (current && !visited.has(current.path_id)) {
    result.unshift(current);
    visited.add(current.path_id);
    current = byId.get(current.parent_path_id);
  }
  return result;
}

export function flameDiffMatches(entry: FlameDiffEntry, query: string): boolean {
  const terms = query.trim().toLocaleLowerCase("zh-CN").split(/\s+/).filter(Boolean);
  if (!terms.length) return true;
  const haystack = `${entry.frame.join(" ")} ${entry.status}`.toLocaleLowerCase("zh-CN");
  return terms.every((term) => haystack.includes(term));
}
