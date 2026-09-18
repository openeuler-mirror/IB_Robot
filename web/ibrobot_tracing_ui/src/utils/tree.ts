import type { SpanRecord } from "../api/types";

export interface SpanTreeNode {
  span: SpanRecord;
  children: SpanTreeNode[];
}

export interface FlatSpanNode extends SpanTreeNode {
  depth: number;
  hasChildren: boolean;
}

export function buildSpanForest(spans: SpanRecord[]): SpanTreeNode[] {
  const nodes = new Map(spans.map((span) => [span.span_id, { span, children: [] as SpanTreeNode[] }]));
  const roots: SpanTreeNode[] = [];
  for (const node of nodes.values()) {
    const parent = nodes.get(node.span.parent_span_id);
    if (parent && parent !== node) parent.children.push(node);
    else roots.push(node);
  }
  const sort = (items: SpanTreeNode[]) => {
    items.sort((a, b) => Number(a.span.start_ns) - Number(b.span.start_ns));
    items.forEach((item) => sort(item.children));
  };
  sort(roots);
  return roots;
}

export function flattenSpanForest(forest: SpanTreeNode[], collapsed: ReadonlySet<string>): FlatSpanNode[] {
  const rows: FlatSpanNode[] = [];
  const visit = (items: SpanTreeNode[], depth: number) => {
    items.forEach((item) => {
      rows.push({ ...item, depth, hasChildren: item.children.length > 0 });
      if (!collapsed.has(item.span.span_id)) visit(item.children, depth + 1);
    });
  };
  visit(forest, 0);
  return rows;
}
