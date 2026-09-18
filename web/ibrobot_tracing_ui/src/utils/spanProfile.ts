import type {
  AggregateSpanProfileNode,
  RequestSpanProfileNode,
} from "../api/types";

export interface ProfileRect<T> {
  node: T;
  x: number;
  y: number;
  width: number;
  height: number;
  depth: number;
  lane: number;
  selfWidth: number;
}

export interface ProfileDepthBand {
  depth: number;
  y: number;
  height: number;
  laneCount: number;
}

export interface ProfileLayout<T> {
  rects: ProfileRect<T>[];
  depthBands: ProfileDepthBand[];
  contentHeight: number;
}

const COMPONENT_COLORS = [
  "#3158b8",
  "#087f8c",
  "#6746b3",
  "#b45117",
  "#18774a",
  "#9b365a",
  "#4e708f",
  "#7b6414",
  "#36509b",
  "#00706e",
  "#74407e",
  "#995027",
];

export function numericNs(value: number | string | null | undefined): number {
  const parsed = typeof value === "number" ? value : Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

class MinHeap<T> {
  private readonly items: T[] = [];

  constructor(private readonly compare: (left: T, right: T) => number) {}

  get length(): number {
    return this.items.length;
  }

  peek(): T | undefined {
    return this.items[0];
  }

  push(value: T): void {
    this.items.push(value);
    let index = this.items.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.compare(this.items[parent], value) <= 0) break;
      this.items[index] = this.items[parent];
      index = parent;
    }
    this.items[index] = value;
  }

  pop(): T | undefined {
    const first = this.items[0];
    const last = this.items.pop();
    if (this.items.length && last !== undefined) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        const right = left + 1;
        if (left >= this.items.length) break;
        const child = right < this.items.length && this.compare(this.items[right], this.items[left]) < 0 ? right : left;
        if (this.compare(last, this.items[child]) <= 0) break;
        this.items[index] = this.items[child];
        index = child;
      }
      this.items[index] = last;
    }
    return first;
  }
}

export type ProfileNodeKeyAction = "inspect" | "focus" | null;

export function profileNodeKeyAction(key: string, shiftKey: boolean, aggregate: boolean): ProfileNodeKeyAction {
  if (aggregate && shiftKey && key === "Enter") return "focus";
  if (key === "Enter" || key === " ") return "inspect";
  return null;
}

export function componentColor(componentId: string): string {
  let hash = 2166136261;
  for (let index = 0; index < componentId.length; index += 1) {
    hash ^= componentId.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return COMPONENT_COLORS[(hash >>> 0) % COMPONENT_COLORS.length];
}

export function layoutRequestProfile(
  nodes: RequestSpanProfileNode[],
  durationNs: number | string,
  width: number,
  rowHeight = 25,
  depthGap = 5,
): ProfileLayout<RequestSpanProfileNode> {
  const totalDuration = Math.max(1, numericNs(durationNs));
  const byDepth = new Map<number, RequestSpanProfileNode[]>();
  nodes.forEach((node) => {
    const depth = Math.max(0, node.depth);
    const group = byDepth.get(depth) ?? [];
    group.push(node);
    byDepth.set(depth, group);
  });

  const placements: ProfileRect<RequestSpanProfileNode>[] = [];
  const laneCounts = new Map<number, number>();
  const minimumWidth = Math.min(3, Math.max(1, width));

  [...byDepth.entries()].sort(([left], [right]) => left - right).forEach(([depth, group]) => {
    const active = new MinHeap<{ end: number; lane: number }>((left, right) => left.end - right.end || left.lane - right.lane);
    const available = new MinHeap<number>((left, right) => left - right);
    let nextLane = 0;
    const sorted = [...group].sort((left, right) => {
      const start = numericNs(left.start_offset_ns) - numericNs(right.start_offset_ns);
      if (start) return start;
      const duration = numericNs(right.duration_ns) - numericNs(left.duration_ns);
      return duration || compareText(left.id, right.id);
    });

    sorted.forEach((node) => {
      const start = Math.max(0, numericNs(node.start_offset_ns));
      const duration = Math.max(0, numericNs(node.duration_ns));
      const desiredWidth = Math.min(width, Math.max(minimumWidth, (duration / totalDuration) * width));
      const x = Math.min(Math.max(0, width - desiredWidth), (start / totalDuration) * width);
      const rectWidth = Math.min(width - x, desiredWidth);
      while (active.length && (active.peek()?.end ?? Infinity) + 1 <= x) {
        const released = active.pop();
        if (released) available.push(released.lane);
      }
      const lane = available.pop() ?? nextLane++;
      active.push({ end: x + rectWidth, lane });
      const uncovered = Math.max(0, numericNs(node.uncovered_wall_ns));
      placements.push({
        node,
        x,
        y: 0,
        width: rectWidth,
        height: rowHeight - 2,
        depth,
        lane,
        selfWidth: duration > 0 ? rectWidth * Math.min(1, uncovered / duration) : 0,
      });
    });
    laneCounts.set(depth, Math.max(1, nextLane));
  });

  let y = 0;
  const depthBands: ProfileDepthBand[] = [];
  const depthY = new Map<number, number>();
  [...laneCounts.entries()].sort(([left], [right]) => left - right).forEach(([depth, laneCount]) => {
    const height = laneCount * rowHeight;
    depthY.set(depth, y);
    depthBands.push({ depth, y, height, laneCount });
    y += height + depthGap;
  });
  placements.forEach((rect) => {
    rect.y = (depthY.get(rect.depth) ?? 0) + rect.lane * rowHeight;
  });

  return {
    rects: placements.sort((left, right) => left.depth - right.depth || left.lane - right.lane || left.x - right.x || compareText(left.node.id, right.node.id)),
    depthBands,
    contentHeight: Math.max(0, y - (depthBands.length ? depthGap : 0)),
  };
}

function aggregateOrder(left: AggregateSpanProfileNode, right: AggregateSpanProfileNode): number {
  return numericNs(right.value_ns) - numericNs(left.value_ns)
    || compareText(left.component_id, right.component_id)
    || compareText(left.name, right.name)
    || compareText(left.id, right.id);
}

export function profileDescendantIds(nodes: AggregateSpanProfileNode[], focusId: string): Set<string> {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const result = new Set<string>();
  const stack = [focusId];
  while (stack.length) {
    const id = stack.pop()!;
    if (result.has(id) || !byId.has(id)) continue;
    result.add(id);
    const childIds = byId.get(id)?.child_ids ?? [];
    for (let index = childIds.length - 1; index >= 0; index -= 1) stack.push(childIds[index]);
  }
  return result;
}

export function aggregateBreadcrumb(
  nodes: AggregateSpanProfileNode[],
  focusId: string,
): AggregateSpanProfileNode[] {
  if (!focusId) return [];
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const path: AggregateSpanProfileNode[] = [];
  const visited = new Set<string>();
  let current = byId.get(focusId);
  while (current && !visited.has(current.id)) {
    path.unshift(current);
    visited.add(current.id);
    current = byId.get(current.parent_id);
  }
  return path;
}

export function layoutAggregateProfile(
  nodes: AggregateSpanProfileNode[],
  rootIds: string[],
  width: number,
  focusId = "",
  rowHeight = 27,
): ProfileLayout<AggregateSpanProfileNode> {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const focused = focusId && byId.has(focusId) ? focusId : "";
  const visible = focused ? profileDescendantIds(nodes, focused) : new Set(nodes.map((node) => node.id));
  const roots = (focused ? [focused] : rootIds).map((id) => byId.get(id)).filter((node): node is AggregateSpanProfileNode => Boolean(node && visible.has(node.id))).sort(aggregateOrder);
  const rects: ProfileRect<AggregateSpanProfileNode>[] = [];
  const visited = new Set<string>();

  const rootTotal = roots.reduce((sum, node) => sum + Math.max(0, numericNs(node.value_ns)), 0);
  let rootX = 0;
  roots.forEach((node, index) => {
    const rootWidth = rootTotal > 0
      ? width * (Math.max(0, numericNs(node.value_ns)) / rootTotal)
      : width / Math.max(1, roots.length);
    const boundedWidth = Math.max(0, index === roots.length - 1 ? width - rootX : Math.min(width - rootX, rootWidth));
    const value = Math.max(0, numericNs(node.value_ns));
    const selfValue = Math.max(0, numericNs(node.self_value_ns));
    const rect: ProfileRect<AggregateSpanProfileNode> = {
      node,
      x: rootX,
      y: 0,
      width: boundedWidth,
      height: rowHeight - 2,
      depth: 0,
      lane: 0,
      selfWidth: value > 0 ? boundedWidth * Math.min(1, selfValue / value) : 0,
    };
    rects.push(rect);
    rootX += boundedWidth;
  });

  const stack = [...rects].reverse();
  while (stack.length) {
    const parent = stack.pop()!;
    if (visited.has(parent.node.id)) continue;
    visited.add(parent.node.id);
    const children = parent.node.child_ids
      .map((id) => byId.get(id))
      .filter((node): node is AggregateSpanProfileNode => Boolean(node && visible.has(node.id)))
      .sort(aggregateOrder);
    const childTotal = children.reduce((sum, node) => sum + Math.max(0, numericNs(node.value_ns)), 0);
    const parentValue = Math.max(0, numericNs(parent.node.value_ns));
    const denominator = Math.max(parentValue, childTotal);
    let x = parent.x;
    const childRects: ProfileRect<AggregateSpanProfileNode>[] = [];
    children.forEach((node, index) => {
      const childWidth = denominator > 0
        ? parent.width * (Math.max(0, numericNs(node.value_ns)) / denominator)
        : parent.width / Math.max(1, children.length);
      const boundedWidth = Math.max(0, Math.min(parent.x + parent.width - x, index === children.length - 1 && denominator === childTotal ? parent.x + parent.width - x : childWidth));
      const value = Math.max(0, numericNs(node.value_ns));
      const selfValue = Math.max(0, numericNs(node.self_value_ns));
      const rect: ProfileRect<AggregateSpanProfileNode> = {
        node,
        x,
        y: parent.y + rowHeight,
        width: boundedWidth,
        height: rowHeight - 2,
        depth: parent.depth + 1,
        lane: 0,
        selfWidth: value > 0 ? boundedWidth * Math.min(1, selfValue / value) : 0,
      };
      rects.push(rect);
      childRects.push(rect);
      x += boundedWidth;
    });
    for (let index = childRects.length - 1; index >= 0; index -= 1) stack.push(childRects[index]);
  }

  const depthCount = rects.reduce((maximum, rect) => Math.max(maximum, rect.depth + 1), 0);
  return {
    rects: rects.sort((left, right) => left.depth - right.depth || left.x - right.x || compareText(left.node.id, right.node.id)),
    depthBands: Array.from({ length: depthCount }, (_, depth) => ({ depth, y: depth * rowHeight, height: rowHeight, laneCount: 1 })),
    contentHeight: depthCount * rowHeight,
  };
}

export function profileNodeMatches(
  node: RequestSpanProfileNode | AggregateSpanProfileNode,
  query: string,
): boolean {
  const terms = query.trim().toLocaleLowerCase("zh-CN").split(/\s+/).filter(Boolean);
  if (!terms.length) return true;
  const aggregatePath = "path" in node ? node.path.map((item) => `${item.component_id} ${item.name} ${item.origin}`).join(" ") : "";
  const requestDetails = "diagnostics" in node ? `${node.status} ${node.duration_source} ${node.diagnostics.join(" ")}` : "";
  const haystack = `${node.name} ${node.component_id} ${node.origin} ${aggregatePath} ${requestDetails}`.toLocaleLowerCase("zh-CN");
  return terms.every((term) => haystack.includes(term));
}
